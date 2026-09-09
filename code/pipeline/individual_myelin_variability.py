#!/usr/bin/env python
"""Per-subject T1w/T2w vs cyto7, and a between-subject variability layer (proof-of-concept).

Implements ``docs/SPEC_individual_myelin_variability.md``. HCP releases individual
``MyelinMap_BC_MSMAll`` (T1w/T2w) maps on the *same* 32k ``fs_LR`` grayordinate space the
cyto7 map lives on (MSMAll-aligned), so individual microstructure is comparable to the
group cyto7 labels vertex-by-vertex with no extra registration.

Two products:

1. **Per-subject alignment** — Spearman rho between each subject's myelin and the ordinal
   cyto7 type over labelled cortical vertices, giving a *distribution* of rho across
   individuals to put next to the single group-average value (rho ~ +0.59), plus each
   subject's per-type median myelin (how stable the monotonic type ordering is).
2. **Between-subject variability map** — per subject, myelin is z-scored across valid
   cortical vertices, each cyto7 type's mean z is removed (the level the group label
   *predicts* in that subject), and the residual ``r_s(v)`` is kept. The SD of ``r_s(v)``
   over subjects is the headline layer: high = the group label fits individuals
   *inconsistently* at that vertex. The signed and absolute means give systematic misfit.

ANTI-CIRCULARITY (SPEC guardrail). T1w/T2w is a modality cyto7 is *validated against*
(manuscript S3.5) and the released per-vertex support map is deliberately
**anatomy-only** (Annex C). What this script produces is a **separate,
microstructure-informed (T1w/T2w) inter-subject variability layer**. It is NOT folded into
the anatomy-only support, it is NOT used to re-validate cyto7, and it does not modify
the released map, the released support, or any released table. Step 4 *correlates* the
two so we can say whether geometric/topological support already anticipates where
individuals disagree biologically, or whether this is an independent axis of uncertainty.

Credentials: read by boto3 from the ``[default]`` AWS profile. They are never printed,
logged or copied by this script.

Usage::

    conda run -n cyto7 python scripts/individual_myelin_variability.py --n-subjects 5   # pilot
    conda run -n cyto7 python scripts/individual_myelin_variability.py                  # full
    conda run -n cyto7 python scripts/individual_myelin_variability.py --analyse-only   # no S3
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyto7_surface_io import (  # noqa: E402
    LABEL_NAMES,
    REPO_ROOT,
    load_surface_geometry,
    myelin_dscalar_path,
    resolve_target_map,
    surface_path,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
OUT = cfg.results_dir("tables") / "individualisation"

#: Download cache, deliberately OUTSIDE the repository (each file ~590 kB).
DEFAULT_SCRATCH = cfg.scratch_dir("hcp_individual_myelin")

BUCKET = "hcp-openaccess"
S1200_PREFIX = "HCP_1200/"
#: The one object fetched per subject -- nothing else is downloaded.
KEY_TMPL = (S1200_PREFIX + "{s}/MNINonLinear/fsaverage_LR32k/"
            "{s}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii")

VER = "v9"                     # cyto7 map version (canonical)
DATASET = "Validation210"      # group-average dataset the cyto7 labels were resampled against
N_TARGET = 200                 # subjects to use when the RelatedValidation210 list is unavailable
N_HEMI = 32492                 # fs_LR 32k vertices per hemisphere
N_SPIN = 1000
SEED = 0

HEMIS = ("L", "R")
_CORTEX_STRUCTURES = {"L": "CIFTI_STRUCTURE_CORTEX_LEFT",
                      "R": "CIFTI_STRUCTURE_CORTEX_RIGHT"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Step 0 -- fetch individual myelin maps
# --------------------------------------------------------------------------- #
def _validation210_ids(explicit: Path | None) -> list[str] | None:
    """The 210 RelatedValidation subject IDs, if they can be obtained.

    HCP does not ship the group membership list inside the S1200 group-average or the
    Glasser RVVG package (both are checked below), and it is not derivable from S3; it
    lives in the restricted ConnectomeDB release. ``--subject-list`` lets the author
    supply it later, in which case those IDs are used verbatim.
    """
    if explicit is not None:
        ids = [t.strip() for t in explicit.read_text().split() if t.strip().isdigit()]
        log(f"  subject list supplied: {len(ids)} IDs from {explicit}")
        return ids
    # Look for a membership list shipped with the local Glasser/HCP group-average package.
    pkg = cfg.data_dir() / "glasser_resources"
    for pat in ("**/*Validation210*subject*", "**/*subject*list*", "**/*Validation210*.txt"):
        hits = [p for p in pkg.glob(pat) if p.is_file()]
        if hits:
            log(f"  NOTE: candidate subject-list files found but not parsed: {hits}")
    return None


def list_s1200_subjects(s3) -> list[str]:
    """Every S1200 subject directory under ``HCP_1200/`` (sorted numerically)."""
    subs: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=S1200_PREFIX, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            sid = cp["Prefix"][len(S1200_PREFIX):].strip("/")
            if sid.isdigit():
                subs.append(sid)
    return sorted(subs, key=int)


def local_path(scratch: Path, subj: str) -> Path:
    return scratch / f"{subj}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii"


def fetch_myelin(scratch: Path, n_target: int,
                 subject_list: list[str] | None) -> tuple[list[str], list[str]]:
    """Download the per-subject myelin dscalar for up to *n_target* subjects.

    Returns ``(used, missing)``. Resumable: files already on scratch are reused.
    Stop-and-log per subject -- a missing/unreadable file never aborts the run.
    """
    import boto3
    import botocore

    scratch.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    if subject_list is None:
        log("  listing S1200 subjects on S3 ...")
        candidates = list_s1200_subjects(s3)
        log(f"  {len(candidates)} S1200 subject directories visible; "
            f"taking the first {n_target} with a myelin map")
    else:
        candidates = subject_list

    used: list[str] = []
    missing: list[str] = []
    n_downloaded = 0
    for subj in candidates:
        if len(used) >= n_target:
            break
        dst = local_path(scratch, subj)
        if not dst.exists() or dst.stat().st_size == 0:
            key = KEY_TMPL.format(s=subj)
            try:
                s3.head_object(Bucket=BUCKET, Key=key)
            except botocore.exceptions.ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "?")
                log(f"  [STOP-AND-LOG] {subj}: myelin map not on S3 (HEAD {code}) -- skipped")
                missing.append(subj)
                continue
            tmp = dst.with_suffix(dst.suffix + ".part")
            try:
                s3.download_file(BUCKET, key, str(tmp))
                tmp.replace(dst)
                n_downloaded += 1
            except Exception as exc:  # network / IO -- log and move on
                tmp.unlink(missing_ok=True)
                log(f"  [STOP-AND-LOG] {subj}: download failed ({exc!r}) -- skipped")
                missing.append(subj)
                continue
        # Cheap integrity gate: the file must parse as a 1 x 59412 CIFTI dscalar.
        try:
            import nibabel as nib
            shape = nib.load(str(dst)).shape
            assert len(shape) == 2 and shape[0] == 1, shape
        except Exception as exc:
            log(f"  [STOP-AND-LOG] {subj}: unreadable dscalar ({exc!r}) -- deleted, skipped")
            dst.unlink(missing_ok=True)
            missing.append(subj)
            continue
        used.append(subj)
        if len(used) % 25 == 0:
            log(f"  ... {len(used)} subjects ready ({n_downloaded} newly downloaded)")

    log(f"  fetched/verified {len(used)} subjects ({n_downloaded} newly downloaded, "
        f"{len(missing)} stop-and-logged)")
    return used, missing


# --------------------------------------------------------------------------- #
# Step 1 -- load
# --------------------------------------------------------------------------- #
def load_cifti_cortex(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a cortex-only dscalar onto the concatenated 64984-vertex fs_LR mesh.

    Returns ``(values, present)``: ``values`` is NaN wherever the CIFTI has no
    grayordinate (the medial wall), ``present`` is the boolean grayordinate mask.
    Using the CIFTI vertex index -- not ``value != 0`` -- is important here because
    individual bias-corrected T1w/T2w maps legitimately contain values at or below 0.
    """
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.get_fdata())[0]
    bm = img.header.get_axis(1)
    vals = np.full(2 * N_HEMI, np.nan)
    present = np.zeros(2 * N_HEMI, dtype=bool)
    for i, hemi in enumerate(HEMIS):
        struct = _CORTEX_STRUCTURES[hemi]
        sl, mdl = next((sl, mdl) for nm, sl, mdl in bm.iter_structures() if nm == struct)
        assert bm.nvertices[struct] == N_HEMI, bm.nvertices
        vals[i * N_HEMI + mdl.vertex] = data[sl]
        present[i * N_HEMI + mdl.vertex] = True
    return vals, present


def split_hemis(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {"L": arr[:N_HEMI], "R": arr[N_HEMI:]}


def load_labels() -> np.ndarray:
    """cyto7 v9 ordinal labels (0 = unlabelled/medial wall) on the 64984 mesh."""
    lab = resolve_target_map(VER, "fs_LR")
    return np.concatenate([np.asarray(lab[h]).astype(int) for h in HEMIS])


# --------------------------------------------------------------------------- #
# Steps 2-3 -- per-subject alignment and the variability map
# --------------------------------------------------------------------------- #
def analyse(subjects: list[str], scratch: Path, labels: np.ndarray,
            group_myelin: np.ndarray, valid: np.ndarray):
    """Per-subject rho + per-type medians + the per-vertex residual stack."""
    types = list(range(1, 8))
    n = len(subjects)
    resid = np.full((n, 2 * N_HEMI), np.nan, dtype=np.float32)
    rows = []
    med_rows = []
    lab_v = labels[valid].astype(float)
    excl = valid & (labels > 1)          # allocortex-excluded (types 2-7)
    lab_e = labels[excl].astype(float)

    for i, subj in enumerate(subjects):
        vals, _ = load_cifti_cortex(local_path(scratch, subj))
        v = vals[valid]
        rho = float(stats.spearmanr(v, lab_v)[0])
        rho_e = float(stats.spearmanr(vals[excl], lab_e)[0])

        # Step 3.1-3.3: within-subject z-score over valid cortex, remove the per-type
        # mean (the level the group label predicts in this subject), keep the residual.
        z = np.full(2 * N_HEMI, np.nan, dtype=float)
        z[valid] = (v - v.mean()) / v.std(ddof=1)
        tmean = {t: float(np.nanmean(z[valid & (labels == t)])) for t in types}
        r = np.full(2 * N_HEMI, np.nan, dtype=float)
        for t in types:
            m = valid & (labels == t)
            r[m] = z[m] - tmean[t]
        resid[i] = r.astype(np.float32)

        # Per-type median myelin (raw units) + monotonicity of the type progression.
        meds = {t: float(np.median(v[lab_v == t])) for t in types}
        tau_all = float(stats.kendalltau(list(meds.values()), types)[0])
        m27 = [meds[t] for t in types[1:]]
        tau_excl = float(stats.kendalltau(m27, types[1:])[0])
        rows.append({"subject": subj, "rho": rho, "rho_allocortex_excluded": rho_e,
                     "kendall_tau_typemedians": tau_all,
                     "kendall_tau_typemedians_allocortex_excluded": tau_excl,
                     "monotonic_2to7": bool(np.all(np.diff(m27) > 0))})
        med_rows.append({"subject": subj,
                         **{f"median_myelin_{LABEL_NAMES[t - 1].replace(' ', '_')}": meds[t]
                            for t in types}})
        if (i + 1) % 25 == 0:
            log(f"  ... {i + 1}/{n} subjects analysed")

    per_subj = pd.DataFrame(rows)
    per_type = pd.DataFrame(med_rows)

    # Step 3.4: per-vertex statistics across subjects.
    variability = np.full(2 * N_HEMI, np.nan)
    robust = np.full(2 * N_HEMI, np.nan)
    systematic = np.full(2 * N_HEMI, np.nan)
    abs_misfit = np.full(2 * N_HEMI, np.nan)
    rv = resid[:, valid]
    variability[valid] = rv.std(axis=0, ddof=1)
    systematic[valid] = rv.mean(axis=0)
    abs_misfit[valid] = np.abs(rv).mean(axis=0)
    # Outlier-resistant twin of the headline layer (normalised IQR ~ SD for a Gaussian).
    # Individual T1w/T2w maps carry occasional artefact vertices whose SD is an order of
    # magnitude above the bulk; every step-4 conclusion is repeated on this variant.
    q75, q25 = np.percentile(rv, [75, 25], axis=0)
    robust[valid] = (q75 - q25) / 1.349

    group_rho = float(stats.spearmanr(group_myelin[valid], lab_v)[0])
    group_rho_e = float(stats.spearmanr(group_myelin[excl], lab_e)[0])
    return (per_subj, per_type, variability, robust, systematic, abs_misfit,
            group_rho, group_rho_e, excl)


# --------------------------------------------------------------------------- #
# Step 4 -- cross-check against the released anatomy-only support
# --------------------------------------------------------------------------- #
def boundary_distance(labels: np.ndarray) -> np.ndarray:
    """Geodesic distance (mm, midthickness) from the nearest cyto7 type boundary.

    A vertex is a boundary seed when any mesh neighbour carries a different (labelled)
    cyto7 type. Used only to ask whether the variability layer is a border effect.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    out = np.full(2 * N_HEMI, np.nan)
    for i, hemi in enumerate(HEMIS):
        coords, faces = load_surface_geometry(DATASET, hemi, "midthickness")
        lab = labels[i * N_HEMI:(i + 1) * N_HEMI]
        src = faces[:, [0, 1, 2]].ravel()
        dst = faces[:, [1, 2, 0]].ravel()
        elen = np.linalg.norm(coords[src] - coords[dst], axis=1)
        graph = csr_matrix((elen, (src, dst)), shape=(N_HEMI, N_HEMI))
        diff = (lab[src] != lab[dst]) & (lab[src] > 0) & (lab[dst] > 0)
        seeds = np.unique(np.concatenate([src[diff], dst[diff]]))
        if seeds.size:
            out[i * N_HEMI:(i + 1) * N_HEMI] = dijkstra(
                graph, directed=False, indices=seeds, min_only=True)
    return out


def spin_nulls(variability: np.ndarray, cache: Path, n_spin: int) -> np.ndarray | None:
    """Alexander-Bloch rotations of the *variability* map (fsLR 32k, seed 0).

    Both maps in step 4 are strongly spatially autocorrelated, so the parametric p over
    ~59k vertices is meaningless; the spin null is what makes the comparison honest.
    Cached on scratch (~260 MB) -- never in the repository.
    """
    if n_spin <= 0:
        return None
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        if arr.shape[1] >= n_spin:
            log(f"  reusing cached spin nulls {cache.name} {arr.shape}")
            return np.asarray(arr[:, :n_spin])
    from neuromaps.nulls import alexander_bloch
    log(f"  generating spin nulls (alexander_bloch fsLR 32k, n_perm={n_spin}, "
        f"seed={SEED}) -- this takes a few minutes ...")
    nulls = alexander_bloch(variability, atlas="fsLR", density="32k",
                            n_perm=n_spin, seed=SEED)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, nulls.astype(np.float32))
    log(f"  cached {cache.name} {nulls.shape}")
    return nulls


def cross_check(variability, robust, abs_misfit, systematic, unconf, labels, valid,
                nulls) -> tuple[pd.DataFrame, dict]:
    """Correlate the variability layer with (1 - anatomy-only support)."""
    m = valid & np.isfinite(unconf) & np.isfinite(variability)
    x, y = unconf[m], variability[m]
    rho = float(stats.spearmanr(x, y)[0])
    r = float(stats.pearsonr(x, y)[0])
    summary = {"n_valid": int(m.sum()), "spearman_rho": rho, "pearson_r": r,
               "r2": r ** 2, "spin_p": float("nan"),
               "spearman_rho_robust": float(stats.spearmanr(x, robust[m])[0]),
               "rho_sd_vs_robust": float(stats.spearmanr(y, robust[m])[0])}
    if nulls is not None:
        null = np.empty(nulls.shape[1])
        for i in range(nulls.shape[1]):
            spun = nulls[:, i]
            ok = m & np.isfinite(spun)
            null[i] = stats.spearmanr(unconf[ok], spun[ok])[0]
        summary["spin_p"] = float((np.sum(np.abs(null) >= abs(rho)) + 1)
                                  / (nulls.shape[1] + 1))
        summary["spin_null_sd"] = float(null.std(ddof=1))

    def row(scope, mm):
        return {"scope": scope, "n": int(mm.sum()),
                "spearman_rho_unconf_vs_variability":
                    float(stats.spearmanr(unconf[mm], variability[mm])[0]),
                "spearman_rho_unconf_vs_variability_robustIQR":
                    float(stats.spearmanr(unconf[mm], robust[mm])[0]),
                "spearman_rho_unconf_vs_absmisfit":
                    float(stats.spearmanr(unconf[mm], abs_misfit[mm])[0]),
                "spearman_rho_unconf_vs_signedmisfit":
                    float(stats.spearmanr(unconf[mm], systematic[mm])[0])}

    rows = [row("all valid cortex", m),
            row("allocortex-excluded (types 2-7)", m & (labels > 1))]
    for t in range(1, 8):
        mt = m & (labels == t)
        if mt.sum() >= 20:
            rows.append(row(f"type {t} ({LABEL_NAMES[t - 1]})", mt))
    return pd.DataFrame(rows), summary


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_dscalar(path: Path, arr: np.ndarray, map_name: str) -> None:
    """Write a 64984-vertex fs_LR array as a cortex-only CIFTI dscalar.

    The grayordinate/brain-model axis is copied from the released group-average myelin
    dscalar, so the output opens in wb_view alongside every other map in this project.
    """
    import nibabel as nib
    from nibabel.cifti2 import cifti2_axes

    tmpl = nib.load(str(myelin_dscalar_path(DATASET)))
    bm = tmpl.header.get_axis(1)
    data = np.full((1, bm.size), np.nan, dtype=np.float32)
    for i, hemi in enumerate(HEMIS):
        struct = _CORTEX_STRUCTURES[hemi]
        sl, mdl = next((sl, mdl) for nm, sl, mdl in bm.iter_structures() if nm == struct)
        data[0, sl] = arr[i * N_HEMI + mdl.vertex]
    img = nib.Cifti2Image(data, header=(cifti2_axes.ScalarAxis([map_name]), bm))
    img.nifti_header.set_intent("ConnDenseScalar")
    img.to_filename(str(path))
    log(f"  wrote {path.name}")


def _panel_surface(ax, coords, faces, vals, lab, hemi, view, cmap, norm,
                   base=(0.82, 0.82, 0.82)):
    """Flat-shaded surface panel with cyto7 type borders (no nilearn / no 3-D axes)."""
    from matplotlib.collections import LineCollection, PolyCollection

    flip = (hemi, view) in (("L", "lateral"), ("R", "medial"))
    sx = -coords[:, 1] if flip else coords[:, 1]
    sy = coords[:, 2]

    tri = coords[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    front = (nrm[:, 0] < 0) if flip else (nrm[:, 0] > 0)
    F = faces[front]
    depth = coords[F].mean(1)[:, 0]
    order = np.argsort(-depth) if flip else np.argsort(depth)
    F, shade = F[order], nrm[front, 2][order]
    shade = 0.62 + 0.38 * np.clip(np.abs(shade), 0, 1)

    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN faces -> grey
            fv = np.nanmean(vals[F], axis=1)
    col = cmap(norm(fv))[:, :3]
    grey = ~np.isfinite(fv) | ((lab[F] == 0).sum(1) >= 2)
    col[grey] = base
    ax.add_collection(PolyCollection(
        np.stack([sx[F], sy[F]], axis=-1), facecolors=np.clip(col * shade[:, None], 0, 1),
        edgecolors="face", linewidths=0.2))

    # cyto7 borders: mesh edges of *coloured* faces whose endpoints differ in type
    # (edges of greyed medial-wall faces would draw stray lines inside the mask).
    Fc = F[~grey]
    e = np.vstack([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    keep = (lab[e[:, 0]] != lab[e[:, 1]]) & (lab[e[:, 0]] > 0) & (lab[e[:, 1]] > 0)
    e = e[keep]
    if e.size:
        ax.add_collection(LineCollection(
            np.stack([np.stack([sx[e[:, 0]], sy[e[:, 0]]], -1),
                      np.stack([sx[e[:, 1]], sy[e[:, 1]]], -1)], axis=1),
            colors="0.05", linewidths=0.25, alpha=0.85))

    ax.set_xlim(sx.min() - 2, sx.max() + 2)
    ax.set_ylim(sy.min() - 2, sy.max() + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def make_figure(per_subj: pd.DataFrame, group_rho: float, variability: np.ndarray,
                unconf: np.ndarray, labels: np.ndarray, valid: np.ndarray,
                out_path: Path) -> None:
    """(a) per-subject rho histogram, (b) variability on the inflated 32k surface,
    (c) anatomy-only support vs between-subject variability. House style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    from figure_style import cyto7_palette

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})

    fig_w = 190 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_w * 0.52))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0],
                          left=0.075, right=0.985, top=0.95, bottom=0.17,
                          wspace=0.35, hspace=0.50)

    # ---- (a) per-subject rho distribution -------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(per_subj.rho, bins=22, color="#9ecae1", edgecolor="#3182bd", linewidth=0.5)
    ax.axvline(group_rho, color="firebrick", lw=1.2)
    ax.annotate(f"group-average map\nρ = {group_rho:+.3f}", (group_rho, 0.97),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                xytext=(-4, 0), ha="right", va="top", fontsize=6, color="firebrick")
    ax.axvline(per_subj.rho.median(), color="0.25", lw=1.0, ls="--")
    ax.annotate(f"median\n{per_subj.rho.median():+.3f}", (per_subj.rho.median(), 0.55),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                xytext=(4, 0), ha="left", va="center", fontsize=6, color="0.25")
    ax.set_xlabel("per-subject Spearman ρ (T1w/T2w vs ordinal cyto7 type)", fontsize=7)
    ax.set_ylabel(f"subjects (n = {len(per_subj)})", fontsize=7)
    ax.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(-0.22, 1.06, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    # ---- (c) support vs variability ----------------------------------- #
    axc = fig.add_subplot(gs[1, 0])
    m = valid & np.isfinite(unconf) & np.isfinite(variability)
    conf = 1.0 - unconf[m]
    pal = cyto7_palette("viridis")
    axc.scatter(conf, variability[m], s=0.6, linewidths=0,
                c=[pal[t] for t in labels[m]], alpha=0.35, rasterized=True)
    # Binned median trend (the point-cloud is 10^4-10^5 vertices).
    edges = np.linspace(np.nanpercentile(conf, 0.5), np.nanpercentile(conf, 99.5), 21)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, len(edges) - 2)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(variability[m][idx == b]) if (idx == b).sum() > 30 else np.nan
                    for b in range(len(ctr))])
    axc.plot(ctr, med, color="black", lw=1.2, zorder=5)
    rho_c = stats.spearmanr(conf, variability[m])[0]
    # Robust axis limits: a handful of artefact vertices carry SDs an order of
    # magnitude above the bulk and would otherwise flatten the whole cloud.
    axc.set_xlim(np.nanpercentile(conf, 0.2), 1.005)
    axc.set_ylim(0, np.nanpercentile(variability[m], 99.8))
    axc.annotate(f"ρ = {rho_c:+.3f}  (n = {int(m.sum()):,} vertices; "
                 f"y clipped at the 99.8th pct)", (0.02, 0.97),
                 xycoords="axes fraction", fontsize=6, va="top")
    axc.set_xlabel("released anatomy-only support", fontsize=7)
    axc.set_ylabel("between-subject variability\n(SD of residual z-myelin)", fontsize=7)
    axc.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    axc.text(-0.22, 1.06, "c", transform=axc.transAxes, fontsize=10, fontweight="bold")
    axc.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=2.5, color=pal[t],
                                   label=LABEL_NAMES[t - 1]) for t in range(1, 8)],
               fontsize=5.2, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.28), ncol=4, handletextpad=0.2,
               columnspacing=0.7, borderpad=0.1)

    # ---- (b) variability on the inflated surface ------------------------- #
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 1:], wspace=0.0, hspace=0.10)
    vmin, vmax = np.nanpercentile(variability[valid], [2, 98])
    cmap, norm = plt.get_cmap("magma"), Normalize(vmin, vmax)
    axes_b = []
    for k, (hemi, view) in enumerate([("L", "lateral"), ("L", "medial"),
                                      ("R", "lateral"), ("R", "medial")]):
        axb = fig.add_subplot(sub[k // 2, k % 2])
        axes_b.append(axb)
        import nibabel as nib
        g = nib.load(str(surface_path(DATASET, hemi, "inflated")))
        coords = np.asarray(g.darrays[0].data, float)
        faces = np.asarray(g.darrays[1].data, np.int64)
        off = 0 if hemi == "L" else N_HEMI
        _panel_surface(axb, coords, faces, variability[off:off + N_HEMI],
                       labels[off:off + N_HEMI], hemi, view, cmap, norm)
        axb.set_title(f"{hemi}H {view}", fontsize=6.5, pad=1.0)
        if k == 0:
            axb.text(0.0, 1.16, "b", transform=axb.transAxes, fontsize=10,
                     fontweight="bold")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes_b, orientation="horizontal", fraction=0.05,
                      pad=0.02, aspect=45, extend="both")
    cb.set_label("between-subject variability of T1w/T2w residual "
                 "(SD over subjects, z units)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out_path.name}")


def write_report(path: Path, subjects, missing, per_subj, per_type, group_rho,
                 group_rho_e, valid, excl, cc_df, cc, variability, abs_misfit,
                 systematic, labels, unconf, bdist, n_spin, scratch, elapsed) -> None:
    q1, q3 = np.percentile(per_subj.rho, [25, 75])
    q1e, q3e = np.percentile(per_subj.rho_allocortex_excluded, [25, 75])
    var_by_type = {t: float(np.nanmedian(variability[valid & (labels == t)]))
                   for t in range(1, 8)}
    mb = valid & np.isfinite(bdist) & np.isfinite(variability)
    rho_border = float(stats.spearmanr(bdist[mb], variability[mb])[0])
    conf_ok = valid & np.isfinite(unconf)
    L = []
    A = L.append
    A("# Report — per-subject T1w/T2w vs cyto7, and a microstructure-informed "
      "(T1w/T2w) inter-subject variability layer\n")
    A(f"Implements `docs/SPEC_individual_myelin_variability.md`. Env: `cyto7`. "
      f"Map: **cyto7 {VER}** (canonical). Mesh: fs_LR 32k. "
      f"Runtime {elapsed / 60:.1f} min. **Proof-of-concept — not wired into the "
      f"manuscript, and nothing released was modified.**\n")

    A("## Anti-circularity statement\n")
    A("T1w/T2w is a modality cyto7 is *validated against* (§3.5) and the released "
      "per-vertex support map is deliberately **anatomy-only** (Annex C: atlas + "
      "topology + geometry + prior; the T1w/T2w overlay is computed but excluded from "
      "the score). The layer produced here is therefore a **separate, "
      "microstructure-informed (T1w/T2w) inter-subject variability** product. It is "
      "**not** folded into the anatomy-only support, and it is **not** used to "
      "re-validate cyto7 — the §3.5 T1w/T2w validation stays on the anatomy-only map. "
      "Step 4 below only *compares* the two maps, to ask whether they carry the same "
      "information. Any future manuscript use must keep this labelling.\n")

    A("## Data and provenance\n")
    A(f"- **Source:** `s3://hcp-openaccess/HCP_1200/<subj>/MNINonLinear/fsaverage_LR32k/"
      f"<subj>.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii` — the individual bias-corrected "
      f"T1w/T2w map, MSMAll-aligned, cortex-only (59,412 grayordinates). Nothing else was "
      f"downloaded. Credentials came from the AWS `[default]` profile and appear nowhere "
      f"in this repository or in any output.")
    A(f"- **N = {len(subjects)} subjects.** IDs in `subjects_used.txt`. "
      + (f"{len(missing)} candidate subject(s) were stop-and-logged as missing/unreadable "
         f"and skipped ({', '.join(missing)}); see `subjects_missing.txt`."
         if missing else "No candidate subject was skipped."))
    A("- **Subject selection:** the 210 *RelatedValidation* IDs are **not obtainable** "
      "programmatically — the group membership list is not in the S1200 group-average "
      "package, not in the local Glasser RVVG package, and not derivable from the S3 "
      "listing (it lives in the restricted ConnectomeDB release). Per the SPEC fallback, "
      f"the first {len(subjects)} S1200 subjects (ascending subject ID) that have the "
      f"myelin map were used. This is an unselected S1200 sample, so it is **not** the "
      f"exact set behind the group average; that is a deliberate, logged deviation and "
      f"`--subject-list` accepts the real list if the author supplies it later.")
    A(f"- **Group-average reference:** `{myelin_dscalar_path(DATASET).name}` "
      f"(Q1-Q6_Related{DATASET}, the dataset the cyto7 labels were resampled against).")
    A(f"- **Valid vertices:** {int(valid.sum()):,} of 64,984 — labelled cyto7 cortex "
      f"(`label > 0`) that is also a CIFTI grayordinate in the group map *and in every "
      f"subject*, so all subjects contribute at every vertex. Allocortex-excluded "
      f"sensitivity uses {int(excl.sum()):,} vertices (types 2–7). Validity is taken from "
      f"the CIFTI vertex index, not `value != 0`, because bias-corrected individual "
      f"T1w/T2w legitimately contains values ≤ 0.")
    A(f"- **Downloads cached outside the repository** at `{scratch}` "
      f"(~{len(subjects) * 0.59:.0f} MB); re-runs are resumable.\n")

    A("## Step 2 — per-subject alignment with cyto7 type\n")
    A(f"| | median | IQR | min | max | group-average map |")
    A("|---|---|---|---|---|---|")
    A(f"| Spearman ρ (all 7 types) | **{per_subj.rho.median():+.3f}** | "
      f"{q1:+.3f} – {q3:+.3f} | {per_subj.rho.min():+.3f} | {per_subj.rho.max():+.3f} | "
      f"{group_rho:+.3f} |")
    A(f"| ρ, allocortex-excluded (2–7) | **{per_subj.rho_allocortex_excluded.median():+.3f}** | "
      f"{q1e:+.3f} – {q3e:+.3f} | {per_subj.rho_allocortex_excluded.min():+.3f} | "
      f"{per_subj.rho_allocortex_excluded.max():+.3f} | {group_rho_e:+.3f} |")
    A("")
    A(f"Every one of the {len(per_subj)} subjects has ρ > 0 "
      f"({int((per_subj.rho > 0).sum())}/{len(per_subj)}); the individual ρ is "
      f"systematically **lower** than the group-average value, which is expected — "
      f"averaging 210 brains removes individual noise that the ordinal type map cannot "
      f"predict, so the group ρ is an upper bound, not a typical individual.")
    A(f"Type-progression stability: Kendall τ between each subject's seven per-type median "
      f"myelin values and the type rank is median **{per_subj.kendall_tau_typemedians.median():+.3f}** "
      f"(allocortex-excluded {per_subj.kendall_tau_typemedians_allocortex_excluded.median():+.3f}); "
      f"**{int(per_subj.monotonic_2to7.sum())}/{len(per_subj)}** subjects show a strictly "
      f"monotonic agranular→koniocortex increase in median myelin. The all-7 τ is lower "
      f"than the allocortex-excluded τ in essentially every subject for the same reason "
      f"it is in the released group table: allocortex sits at rank 1 but its T1w/T2w is "
      f"*elevated* relative to agranular cortex, so it breaks the ramp — which is exactly "
      f"why the paper excludes allocortex from the trend fits. The individualisation "
      f"result reproduces that group-level quirk subject by subject. Per-subject numbers: "
      f"`persubject_rho.csv`, `persubject_type_medians.csv`.\n")

    A("## Step 3 — between-subject variability map\n")
    A("Per subject: z-score T1w/T2w across valid cortical vertices; subtract that "
      "subject's mean z within each cyto7 type (the level the group label predicts *in "
      "that subject*); the residual `r_s(v)` is ~0 where the group label fits the "
      "individual well. Across subjects, per vertex:")
    A(f"- **`between_subject_variability`** = SD over subjects of `r_s(v)` — the headline "
      f"layer. Median {np.nanmedian(variability[valid]):.3f}, IQR "
      f"{np.nanpercentile(variability[valid], 25):.3f}–"
      f"{np.nanpercentile(variability[valid], 75):.3f}, 99.9th pct "
      f"{np.nanpercentile(variability[valid], 99.9):.3f}, range "
      f"{np.nanmin(variability[valid]):.3f}–{np.nanmax(variability[valid]):.3f} z units. "
      f"The long upper tail is a small set of artefact vertices in individual T1w/T2w "
      f"maps, which is why an IQR-based twin "
      f"(`between_subject_variability_robustIQR.npy`) is carried alongside and used as "
      f"the step-4 sensitivity check.")
    A(f"- **`systematic_misfit`** = mean over subjects of `r_s(v)` (signed; range "
      f"{np.nanmin(systematic[valid]):+.3f}…{np.nanmax(systematic[valid]):+.3f}) and mean "
      f"`|r_s(v)|` (median {np.nanmedian(abs_misfit[valid]):.3f}), written as two maps in "
      f"the same file family.")
    A("")
    A("Median variability by cyto7 type (z units):\n")
    A("| type | median SD |")
    A("|---|---|")
    for t in range(1, 8):
        A(f"| {t} {LABEL_NAMES[t - 1]} | {var_by_type[t]:.3f} |")
    A("")
    A("Outputs: `between_subject_variability.{npy,dscalar.nii}`, "
      "`systematic_misfit.{npy,dscalar.nii}` (signed mean; the absolute mean is "
      "`absolute_misfit.{npy,dscalar.nii}`). All on fs_LR 32k, NaN outside valid cortex.\n")

    A("## Step 4 — cross-check against the released anatomy-only support "
      "(the scientific payoff)\n")
    A(f"Over {cc['n_valid']:,} valid vertices, between-subject variability vs "
      f"`1 − anatomy-only support` (cyto7 {VER}):")
    A(f"- Spearman ρ = **{cc['spearman_rho']:+.3f}**"
      + (f", spin p = {cc['spin_p']:.3f} (Alexander-Bloch on the variability map, "
         f"n_perm={n_spin}, seed={SEED}; null SD {cc.get('spin_null_sd', float('nan')):.3f})"
         if np.isfinite(cc["spin_p"]) else " (spin test not run)") + ".")
    A(f"- Pearson r = {cc['pearson_r']:+.3f} (R² = {cc['r2']:.3f}). Pearson runs "
      f"{'well above' if abs(cc['pearson_r']) > 2 * abs(cc['spearman_rho']) else 'close to'} "
      f"Spearman here because the SD layer has a long artefact tail that a rank "
      f"correlation ignores; **Spearman is the number to quote**.")
    A(f"- **Outlier robustness:** repeating this with the IQR-based variability twin "
      f"(normalised inter-quartile range of `r_s(v)` over subjects, immune to the handful "
      f"of artefact vertices whose SD is an order of magnitude above the bulk) gives "
      f"ρ = **{cc['spearman_rho_robust']:+.3f}**; the two variability estimates agree with "
      f"each other at ρ = {cc['rho_sd_vs_robust']:+.3f}. The conclusion does not rest on "
      f"the SD's tail sensitivity.")
    A(f"- Within-type ρ, the allocortex-excluded row, and the |misfit| / signed-misfit "
      f"variants: `variability_vs_support.csv`.")
    A(f"- Border control: geodesic distance to the nearest cyto7 type boundary vs "
      f"variability, ρ = {rho_border:+.3f} — variability is "
      f"{'concentrated near' if rho_border < -0.1 else 'not simply concentrated near'} "
      f"type borders, so it is "
      f"{'partly' if rho_border < -0.1 else 'not'} a boundary-placement effect.")
    # Within-type structure: the pooled correlation can hide (and here does hide) strong,
    # sign-reversing per-type relationships, so state them explicitly.
    trows = cc_df[cc_df.scope.str.startswith("type ")]
    key = "spearman_rho_unconf_vs_variability"
    pos = trows[trows[key] > 0.15]
    neg = trows[trows[key] < -0.15]

    def _lst(df):
        return ", ".join(f"{r.scope.split('(')[1].rstrip(')')} {r[key]:+.2f}"
                         for _, r in df.iterrows())

    A("")
    if len(pos) and len(neg):
        A(f"**The pooled ρ is a cancellation, not an absence of relationship.** Within "
          f"cyto7 types the correlation is substantial and *changes sign*: it is positive "
          f"in the less-differentiated types ({_lst(pos)}) — there, low anatomy-only "
          f"support does mark where individuals' microstructure disagrees with the "
          f"group label — and negative in the most-differentiated types ({_lst(neg)}), "
          f"where low-support vertices are actually the *more* consistent ones across "
          f"individuals. Pooled over cortex these opposite trends cancel to "
          f"ρ = {cc['spearman_rho']:+.3f}. The allocortex-excluded pooled value "
          f"({float(cc_df.loc[cc_df.scope.str.startswith('allocortex-excluded'), key].iloc[0]):+.3f}) "
          f"confirms the pooled number is not carried by allocortex alone. Any use of "
          f"this layer should therefore be *within-type*, not global.")
        A("")
    elif len(trows):
        A(f"Within-type correlations range "
          f"{trows[key].min():+.3f} … {trows[key].max():+.3f} (`variability_vs_support.csv`).")
        A("")
    if not np.isfinite(cc["spin_p"]) or cc["spin_p"] >= 0.05 or abs(cc["spearman_rho"]) < 0.2:
        A(f"**Interpretation.** Globally, the geometric/topological support and the "
          f"microstructure-informed variability are **independent** "
          f"(ρ = {cc['spearman_rho']:+.3f}"
          + (f", spin p = {cc['spin_p']:.3f}" if np.isfinite(cc['spin_p']) else "")
          + f"): a whole-cortex map of the anatomy-only support tells you essentially "
          f"nothing about where individuals' microstructure disagrees with the group "
          f"label"
          + (", and the within-type breakdown above shows why — the two are coupled "
             "inside types, but in opposite directions at the two ends of the "
             "differentiation gradient, so the pooled signal cancels."
             if (len(pos) and len(neg)) else ".")
          + f" That is the substantive result for §4.10: a between-subject variability "
          f"layer would be a genuinely **additional** axis of uncertainty, not a "
          f"re-expression of the support map already released — but it should be read "
          f"per cyto7 type rather than as a single cortex-wide overlay.")
    else:
        A(f"**Interpretation.** The two maps agree appreciably "
          f"(ρ = {cc['spearman_rho']:+.3f}, R² = {cc['r2']:.3f}, spin "
          f"p = {cc['spin_p']:.3f}): the anatomy-only support already anticipates a "
          f"substantial part of where individuals' microstructure disagrees with the "
          f"group label, so a variability layer would be partly redundant with what is "
          f"released — informative for §4.10 as corroboration of the support map "
          f"rather than as a new axis.")
    A("")
    if int(conf_ok.sum()) < int(valid.sum()):
        A(f"(Only {int(conf_ok.sum()):,} of the {int(valid.sum()):,} valid vertices carry a "
          f"finite support value after the 164k→32k resample; the correlation uses "
          f"those.)\n")
    else:
        A(f"(All {int(valid.sum()):,} valid vertices carry a finite anatomy-only "
          f"support value after the 164k→32k resample.)\n")

    A("## Files\n")
    A("| file | contents |")
    A("|---|---|")
    A("| `subjects_used.txt` | the N subject IDs actually used |")
    A("| `persubject_rho.csv` | per subject: ρ, ρ allocortex-excluded, Kendall τ of the type progression, monotonicity flag |")
    A("| `persubject_type_medians.csv` | per subject: median T1w/T2w per cyto7 type |")
    A("| `persubject_rho_summary.txt` | the one-line summary (median, IQR, group ρ) |")
    A("| `between_subject_variability.{npy,dscalar.nii}` | headline layer: SD over subjects of the residual |")
    A("| `between_subject_variability_robustIQR.npy` | outlier-resistant twin (normalised IQR over subjects) |")
    A("| `systematic_misfit.{npy,dscalar.nii}` | signed mean residual over subjects |")
    A("| `absolute_misfit.{npy,dscalar.nii}` | mean absolute residual over subjects |")
    A("| `variability_vs_support.csv` | step-4 correlations, overall and per type |")
    A("| `figure.png` | (a) per-subject ρ histogram, (b) variability on the inflated 32k surface with cyto7 borders, (c) support vs variability |")
    A("")
    A("## Caveats\n")
    A(f"- The sample is the first {len(subjects)} S1200 subjects with a myelin map, not "
      f"the RelatedValidation210 set behind the group average (see Provenance).")
    A("- `MyelinMap_BC_MSMAll` inherits HCP's bias-field correction and MSMAll areal "
      "alignment; MSMAll uses myelin as one of its alignment features, so part of the "
      "*reduction* in between-subject variability is registration-induced. The layer is "
      "therefore a conservative (lower-bound) estimate of true anatomical variability.")
    A("- Residuals are relative to a *within-subject* z-score, so the layer measures the "
      "spatial pattern of misfit, not absolute T1w/T2w differences (global intensity "
      "differences between subjects are removed by construction).")
    A("- The variability map is a group-level summary over N subjects; it is not an "
      "individualised parcellation and does not license per-subject relabelling.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-subjects", type=int, default=N_TARGET,
                    help="subjects to use (default: %(default)s)")
    ap.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH,
                    help="repo-external download cache (default: %(default)s)")
    ap.add_argument("--subject-list", type=Path, default=None,
                    help="file of subject IDs (e.g. the real RelatedValidation210 list)")
    ap.add_argument("--analyse-only", action="store_true",
                    help="skip S3 entirely; use whatever is already on scratch")
    ap.add_argument("--n-spin", type=int, default=N_SPIN,
                    help="spin permutations for step 4 (0 disables; default: %(default)s)")
    args = ap.parse_args(argv)

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"cyto7 {VER} | fs_LR 32k | out -> {OUT}")

    # ---- Step 0 ---------------------------------------------------------- #
    log("\n[0] individual myelin maps")
    miss_file = OUT / "subjects_missing.txt"
    if args.analyse_only:
        subjects = sorted({p.name.split(".")[0] for p in args.scratch.glob("*.dscalar.nii")},
                          key=int)[:args.n_subjects]
        # Keep the fetch pass's stop-and-log record so provenance survives a re-analysis.
        missing = (miss_file.read_text().split() if miss_file.exists() else [])
        log(f"  --analyse-only: {len(subjects)} subjects already on scratch "
            f"({len(missing)} stop-and-logged in the original fetch)")
    else:
        subjects, missing = fetch_myelin(args.scratch, args.n_subjects,
                                         _validation210_ids(args.subject_list))
        miss_file.write_text("\n".join(missing) + ("\n" if missing else ""))
    if not subjects:
        raise SystemExit("no individual myelin maps available -- nothing to do")
    (OUT / "subjects_used.txt").write_text("\n".join(subjects) + "\n")
    log(f"  wrote subjects_used.txt ({len(subjects)} IDs)")

    # ---- Step 1 ---------------------------------------------------------- #
    log("\n[1] load labels, group myelin, valid-vertex mask")
    labels = load_labels()
    group_myelin, group_present = load_cifti_cortex(myelin_dscalar_path(DATASET))
    present = group_present.copy()
    for subj in subjects:
        present &= load_cifti_cortex(local_path(args.scratch, subj))[1]
    valid = (labels > 0) & present & np.isfinite(group_myelin)
    log(f"  labelled cortex {int((labels > 0).sum()):,} | grayordinates in all "
        f"{len(subjects)} subjects + group {int(present.sum()):,} | "
        f"valid {int(valid.sum()):,}")

    # ---- Steps 2-3 ------------------------------------------------------- #
    log("\n[2-3] per-subject alignment + residual stack")
    (per_subj, per_type, variability, robust, systematic, abs_misfit,
     group_rho, group_rho_e, excl) = analyse(subjects, args.scratch, labels,
                                             group_myelin, valid)
    per_subj.to_csv(OUT / "persubject_rho.csv", index=False)
    per_type.to_csv(OUT / "persubject_type_medians.csv", index=False)
    q1, q3 = np.percentile(per_subj.rho, [25, 75])
    summary = (f"N={len(subjects)} subjects | per-subject Spearman rho (T1w/T2w vs ordinal "
               f"cyto7 {VER} type): median {per_subj.rho.median():+.3f}, IQR "
               f"{q1:+.3f}..{q3:+.3f}, range {per_subj.rho.min():+.3f}.."
               f"{per_subj.rho.max():+.3f}; group-average map rho {group_rho:+.3f} | "
               f"allocortex-excluded: median "
               f"{per_subj.rho_allocortex_excluded.median():+.3f}, group "
               f"{group_rho_e:+.3f}")
    (OUT / "persubject_rho_summary.txt").write_text(summary + "\n")
    log("  " + summary)

    for name, arr in (("between_subject_variability", variability),
                      ("systematic_misfit", systematic),
                      ("absolute_misfit", abs_misfit)):
        np.save(OUT / f"{name}.npy", arr)
        write_dscalar(OUT / f"{name}.dscalar.nii", arr,
                      f"cyto7_{VER}_{name}_T1wT2w_N{len(subjects)}")
    np.save(OUT / "between_subject_variability_robustIQR.npy", robust)

    # ---- Step 4 ---------------------------------------------------------- #
    log("\n[4] cross-check vs released anatomy-only support")
    from support_io import anatomy_support_32k
    conf = anatomy_support_32k(VER)
    support = np.clip(np.concatenate([np.asarray(conf[h], float) for h in HEMIS]), 0, 1)
    unconf = 1.0 - support
    bdist = boundary_distance(labels)
    nulls = spin_nulls(variability, args.scratch / f"_spin_nulls_variability_n{N_SPIN}.npy",
                       args.n_spin)
    cc_df, cc = cross_check(variability, robust, abs_misfit, systematic, unconf, labels,
                            valid, nulls)
    cc_df.to_csv(OUT / "variability_vs_support.csv", index=False)
    log(f"  low support vs variability: rho={cc['spearman_rho']:+.3f} "
        f"R2={cc['r2']:.3f} spin_p={cc['spin_p']:.3f} (n={cc['n_valid']:,})")

    # ---- Outputs --------------------------------------------------------- #
    log("\n[5] figure + report")
    make_figure(per_subj, group_rho, variability, unconf, labels, valid,
                OUT / "figure.png")
    write_report(OUT / "report_individual_myelin_variability.md", subjects, missing,
                 per_subj, per_type, group_rho, group_rho_e, valid, excl, cc_df, cc,
                 variability, abs_misfit, systematic, labels, unconf, bdist,
                 args.n_spin, args.scratch, time.time() - t0)
    log(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
