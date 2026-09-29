"""Per-vertex support map for the cyto7 map (Layer 1.5).

Implements ``docs/REFINE_allocortex.md`` Layer 1.5: a per-vertex support in
[0,1] for every labelled vertex. **The score combines four anatomy components;
a fifth, C_data, is computed and saved but deliberately kept out of it**, which
is what lets section 2.4 call the score independent of the modalities used to
validate the atlas. All five are saved separately, so the map is interpretable
and source-decomposed:

1. **C_atlas**  - concordance of the cyto7 type with independent parcellations.
   Isocortex: the von-Economo-derived type (``resources/voneconomo``), scored by
   ordinal closeness ``1 - |dOrdinal|/6``. Allocortex: agreement with a
   periallocortex-belt reference (FreeSurfer ex-vivo entorhinal+perirhinal, and
   Destrieux parahippocampal+temporal-pole) = fraction of sources that place the
   vertex in the belt. Where no relevant atlas applies, C_atlas does not
   contribute (per-vertex weight 0).
2. **C_topo**   - 1.0 where the topology rules (R1-R4, see audit_topology.py)
   hold; x0.5 within ``--topo-hops`` graph-hops of any violation.
3. **C_geom**   - geometric reliability: lower near label boundaries (within
   ``--boundary-verts`` hops) and near the medial wall (within
   ``--medial-buffer-mm`` geodesic mm). Boundaries/medial rim are intrinsically
   less certain (critical for the allocortex ring).
4. **C_data**   - range-restricted data concordance: agreement of local T1w/T2w
   myelin with the type's expected median, from
   ``figures/v9/structure_function/functional_summary_table_v9.csv``.
   ONLY computed for Eulaminate I/II/III outside sensorimotor cortex
   (precentral/postcentral/paracentral); elsewhere it does not contribute
   (myelin proportional to differentiation does not hold there, by design).
   **NOT part of the combined score** - see ``main()``, which combines only
   ``["atlas", "topo", "geom", "prior"]``. It ships as
   ``confidence_data_overlay.shape.gii`` and as the bottom row of the diagnostic
   ``support_map.png``, both labelled "NOT in score". Its ``--w-data`` weight is
   therefore never consumed (RR34 Part A).
5. **C_prior**  - per-class provenance prior (expert-review status): higher for
   the vetted isocortical types, lower for the allocortex/agranular/dysgranular
   zone the painter flagged as uncertain (deck slide 16). See ``PRIOR``.

Combine (weighted geometric mean, weights are CLI flags):
``support = (prod_i C_i^{w_i})^{1/sum w_i}`` over the **anatomy** components
active at each vertex. A categorical map (high>=0.66 / medium / low<0.33) is
also emitted.

Run::

    conda activate cyto7
    python scripts/build_support_map.py                 # default annot = v2 if present else v1
    python scripts/build_support_map.py --annot <template> --help
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from cyto7_surface_io import REPO_ROOT
from audit_topology import adjacency, audit_hemi, ordinal
from figure_style import CONFIDENCE  # centralized support colormap + 0-1 range
from make_presentation_figures import (
    HEMIS, LIGHT_BLUE, lighting_normals, lit_panel, load_labels, load_surface,
    resolve_annot_paths, version_label,
)

DERIVED_DIR: Path = cfg.atlas_dir("fsaverage")
REFINE_DIR: Path = cfg.figures_dir() / "refine"
VE_DIR: Path = cfg.data_dir() / "voneconomo"
ATLAS_DIR: Path = cfg.data_dir() / "refine_atlases"
CACHE: Path = cfg.data_dir() / "neuromaps_cache"

TYPE_CODES = [1, 2, 3, 4, 5, 6, 7]
TYPE_NAMES = ["Allocortex", "Agranular", "Dysgranular", "Eulaminate I",
              "Eulaminate II", "Eulaminate III", "Koniocortex"]
CODE_NAME = {c: TYPE_NAMES[c - 1] for c in TYPE_CODES}
MAXD = 6.0  # ordinal span Allocortex(0)..Koniocortex(6)
EPS = 0.02  # floor for geometric mean

#: C_prior per class (documented provenance prior). Lower = the painter flagged
#: this zone as least certain (allocortex/agranular/dysgranular); higher = the
#: isocortical types already vetted in expert review.
PRIOR = {1: 0.50, 2: 0.65, 3: 0.70, 4: 0.90, 5: 0.95, 6: 0.95, 7: 0.90}

#: Destrieux parcels forming the periallocortex belt reference.
DESTRIEUX_BELT = ("G_oc-temp_med-Parahip", "Pole_temporal", "G_subcallosal")
#: Desikan parcels excluded from C_data (sensorimotor).
SENSORIMOTOR = ("precentral", "postcentral", "paracentral")


# --------------------------------------------------------------------------- #
# Atlas / data loaders
# --------------------------------------------------------------------------- #


def voneconomo_code(hemi: str, n: int, lookup: pd.DataFrame) -> np.ndarray:
    """von-Economo-derived cyto7 code per vertex (0 where no isocortical type)."""
    labels, _ctab, names = nib.freesurfer.io.read_annot(str(VE_DIR / f"{hemi}.economo.annot"))
    names = np.asarray([x.decode() if isinstance(x, bytes) else x for x in names], dtype=object)
    acr = names[np.asarray(labels)]
    acr2code = {r.economo_acronym: int(r.cyto7_code) for r in lookup.itertuples()
                if pd.notna(r.cyto7_code) and str(r.cyto7_code) != ""}
    out = np.zeros(n, int)
    for a, c in acr2code.items():
        out[acr == a] = c
    return out


def label_mask(path: Path, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    m[nib.freesurfer.io.read_label(str(path))] = True
    return m


def destrieux_belt_mask(hemi: str, n: int) -> np.ndarray:
    labels, _c, names = nib.freesurfer.io.read_annot(str(ATLAS_DIR / f"{hemi}.aparc.a2009s.annot"))
    names = [x.decode() if isinstance(x, bytes) else x for x in names]
    idx = [i for i, nm in enumerate(names) if nm in DESTRIEUX_BELT]
    return np.isin(np.clip(labels, 0, len(names) - 1), idx)


def desikan_sensorimotor_mask(hemi: str, n: int) -> np.ndarray:
    labels, _c, names = nib.freesurfer.io.read_annot(str(VE_DIR / f"{hemi}.aparc.annot"))
    names = [x.decode() if isinstance(x, bytes) else x for x in names]
    idx = [i for i, nm in enumerate(names) if nm in SENSORIMOTOR]
    return np.isin(np.clip(labels, 0, len(names) - 1), idx)


def myelin_on_fsaverage(hemi: str, allow_missing: bool = False) -> np.ndarray | None:
    """The T1w/T2w map behind C_data, the overlay that is kept OUT of the score.

    RR34 B2: this used to return None whenever the cache was absent, so the data overlay
    silently became all-NaN and the run still reported success. Skipping it is legitimate,
    but it has to be asked for: pass --allow-missing-myelin (or allow_missing=True).
    """
    p = CACHE / f"myelin_fsaverage_164k_{hemi}.npy"
    if p.exists():
        return np.load(p)
    if allow_missing:
        print(f"  [{hemi}] {p.name} absent and --allow-missing-myelin was passed: "
              "the data overlay will be empty. The combined score is unaffected.")
        return None
    raise SystemExit(
        f"{p} is missing, so C_data (the data overlay) cannot be computed. It is not part "
        "of the combined score, so skipping it is harmless - but say so explicitly with "
        "--allow-missing-myelin rather than letting the run succeed with an empty overlay.")


def myelin_medians(hemi: str) -> dict[int, float]:
    # RR33: this used to read REFINE_DIR.parent / "functional_summary_table.csv", i.e.
    # figures/functional_summary_table.csv -- a pre-v9 summary above the version tree, now in
    # figures/_superseded_pre_v9/. Because rerun_all_v9.py built the support map *before* the
    # structure-function step that writes the v9 table, the support map could never see v9
    # myelin medians; it always consumed the 2026-07-06 file. That ordering is fixed in
    # rerun_all_v9.py, and this read now fails loudly instead of silently using a stale table.
    tbl = (REPO_ROOT / "figures" / "v9" / "structure_function"
           / "functional_summary_table_v9.csv")
    if not tbl.exists():
        raise SystemExit(
            f"{tbl} is missing. The support map needs the v9 per-type myelin medians, so "
            "summarise_functional_features.py must run before build_support_map.py. Do not "
            "substitute an untagged summary table: that is the defect RR33 removed.")
    t = pd.read_csv(tbl)
    t = t[(t.FeatureKey == "myelin") & (t.Hemi == hemi.upper())]
    lower2code = {v.lower(): k for k, v in CODE_NAME.items()}  # CSV type names are lowercase
    out: dict[int, float] = {}
    for _, row in t.iterrows():
        code = lower2code.get(str(row["Type"]).lower())
        if code is not None:
            out[code] = float(row["median"])
    return out


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #


def geodesic_from_medial(coords, faces, medial) -> np.ndarray:
    n = coords.shape[0]
    src = faces[:, [0, 1, 2]].ravel(); dst = faces[:, [1, 2, 0]].ravel()
    w = np.linalg.norm(coords[src] - coords[dst], axis=1)
    g = csr_matrix((w, (src, dst)), shape=(n, n)); g = g.maximum(g.T)
    seeds = np.where(medial)[0]
    if seeds.size == 0:
        return np.full(n, np.inf)
    return dijkstra(g, directed=False, indices=seeds, min_only=True)


def hops_to_boundary(lab, A, cap) -> np.ndarray:
    """Graph-hop distance of each labelled vertex to the nearest label boundary."""
    n = lab.shape[0]
    g = A.tocoo()
    bnd = np.zeros(n, bool)
    diff = lab[g.row] != lab[g.col]
    bnd[g.row[diff]] = True  # any vertex adjacent to a different label (incl. medial)
    dist = np.full(n, cap, float)
    dist[bnd] = 0
    frontier = bnd.copy()
    for d in range(1, cap + 1):
        nxt = (A @ frontier.astype(np.int8) > 0) & (dist > d - 0.5) & (~frontier)
        nxt = nxt & (dist >= d)
        upd = nxt & (dist > d)
        dist[upd] = d
        frontier = upd
        if not upd.any():
            break
    return dist


def compute_components(hemi, lab, coords, faces, A, lookup, args):
    """Return dict of the five [0,1] component arrays + their per-vertex weights."""
    n = lab.shape[0]
    labelled = lab >= 1
    ordc = ordinal(lab)

    # ---- C_atlas ----
    C_atlas = np.full(n, np.nan)
    w_atlas = np.zeros(n)
    ve = voneconomo_code(hemi, n, lookup)
    iso = labelled & (lab >= 2) & (ve >= 1)            # isocortex with a vE type
    C_atlas[iso] = 1.0 - np.abs(ordc[iso] - ordinal(ve)[iso]) / MAXD
    w_atlas[iso] = args.w_atlas
    # allocortex: belt consensus (FS ex-vivo + Destrieux)
    fs_belt = (label_mask(ATLAS_DIR / f"{hemi}.entorhinal_exvivo.label", n) |
               label_mask(ATLAS_DIR / f"{hemi}.perirhinal_exvivo.label", n))
    dx_belt = destrieux_belt_mask(hemi, n)
    belt_votes = fs_belt.astype(float) + dx_belt.astype(float)
    allo = lab == 1
    C_atlas[allo] = belt_votes[allo] / 2.0
    w_atlas[allo] = args.w_atlas

    # ---- C_topo ----
    # STRICT (exact adjacency): penalise only the vertices that are themselves on
    # a topology violation; no "0.5 within N hops" softening (GC request, v4).
    # With --topo-hops 0 (default) the neighbourhood growth is skipped entirely.
    _res, flags = audit_hemi(lab, faces, A, coords)
    viol = flags["skip"] | flags["ring_gap"] | flags["island_viol"]
    near = viol.copy()
    for _ in range(max(args.topo_hops, 0)):
        near = near | (A @ near.astype(np.int8) > 0)
    C_topo = np.where(near & labelled, 0.5, 1.0).astype(float)
    C_topo[~labelled] = np.nan

    # ---- C_geom ----
    medial = lab == 0
    gdist = geodesic_from_medial(coords, faces, medial)
    c_med = np.clip(0.5 + 0.5 * gdist / max(args.medial_buffer_mm, 1e-6), 0.5, 1.0)
    bhops = hops_to_boundary(lab, A, args.boundary_verts)
    c_bnd = np.clip(0.5 + 0.5 * bhops / max(args.boundary_verts, 1), 0.5, 1.0)
    C_geom = np.minimum(c_med, c_bnd)
    C_geom[~labelled] = np.nan

    # ---- C_data (range-restricted) ----
    C_data = np.full(n, np.nan)
    w_data = np.zeros(n)
    myelin = myelin_on_fsaverage(hemi, allow_missing=args.allow_missing_myelin)
    if myelin is not None:
        med = myelin_medians(hemi)
        gap = max(med.get(6, 1.37) - med.get(4, 1.25), 1e-3)  # EuIII - EuI spread
        sm = desikan_sensorimotor_mask(hemi, n)
        valid = labelled & np.isin(lab, [4, 5, 6]) & (~sm) & (myelin > 0)
        own = np.array([med.get(int(c), np.nan) for c in lab])
        C_data[valid] = np.clip(1.0 - np.abs(myelin[valid] - own[valid]) / gap, 0, 1)
        w_data[valid] = args.w_data

    # ---- C_prior ----
    C_prior = np.full(n, np.nan)
    for c in TYPE_CODES:
        C_prior[lab == c] = PRIOR[c]

    return {
        "atlas": (C_atlas, w_atlas),
        "topo": (C_topo, np.where(labelled, args.w_topo, 0.0)),
        "geom": (C_geom, np.where(labelled, args.w_geom, 0.0)),
        "data": (C_data, w_data),
        "prior": (C_prior, np.where(labelled, args.w_prior, 0.0)),
    }


def combine(components: dict, n: int) -> np.ndarray:
    lnsum = np.zeros(n); wsum = np.zeros(n)
    for C, w in components.values():
        active = (w > 0) & np.isfinite(C)
        Cc = np.clip(np.where(active, C, 1.0), EPS, 1.0)
        lnsum += np.where(active, w * np.log(Cc), 0.0)
        wsum += np.where(active, w, 0.0)
    out = np.full(n, np.nan)
    ok = wsum > 0
    out[ok] = np.exp(lnsum[ok] / wsum[ok])
    return out


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #


def write_shape_gii(path: Path, data: np.ndarray) -> None:
    arr = nib.gifti.GiftiDataArray(np.nan_to_num(data, nan=0.0).astype(np.float32),
                                   intent="NIFTI_INTENT_NONE")
    nib.gifti.GiftiImage(darrays=[arr]).to_filename(str(path))


CAT_CTAB = np.array([[200, 200, 200, 0, 0],      # 0 unknown/medial
                     [214, 39, 40, 0, 1],         # 1 low (red)
                     [255, 187, 39, 0, 2],        # 2 medium (amber)
                     [44, 160, 44, 0, 3]], int)   # 3 high (green)
CAT_NAMES = ["unknown", "low", "medium", "high"]


def categorical(conf: np.ndarray) -> np.ndarray:
    cat = np.zeros(conf.shape[0], np.int32)
    ok = np.isfinite(conf)
    cat[ok & (conf < 0.33)] = 1
    cat[ok & (conf >= 0.33) & (conf < 0.66)] = 2
    cat[ok & (conf >= 0.66)] = 3
    return cat


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def render_support(maps: dict, labs: dict, out_path: Path, dpi: int):
    """maps: {component_name: {hemi: array}}; first row = combined."""
    geom = {h: load_surface(h, "inflated") for h in HEMIS}
    vn = {h: lighting_normals(h) for h in HEMIS}
    order = ["combined", "atlas", "topo", "geom", "prior", "data_overlay"]
    row_label = {k: k for k in order}
    row_label["combined"] = "combined (anatomy)"
    row_label["data_overlay"] = "data overlay\n(NOT in score)"
    cmap = plt.get_cmap(CONFIDENCE.cmap)   # figure_style: viridis, 0-1
    panels = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
    nrow, ncol = len(order), len(panels)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.6),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    for r, name in enumerate(order):
        for c, (hemi, view) in enumerate(panels):
            ax = axes[r, c]
            coords, faces = geom[hemi]
            val = maps[name][hemi]
            rgb = np.tile(np.array(LIGHT_BLUE), (coords.shape[0], 1))
            ok = np.isfinite(val)
            rgb[ok] = cmap(np.clip(val[ok], 0, 1))[:, :3]
            lit_panel(ax, coords, faces, rgb, hemi, view, vn[hemi])
            if r == 0:
                ax.set_title(f"{hemi.upper()} {view}", fontsize=10)
            if c == 0:
                ax.text2D(-0.08, 0.5, row_label[name], transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=10, weight="bold")
    sm = cm.ScalarMappable(cmap=cmap, norm=CONFIDENCE.norm())
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
    cb.set_label(CONFIDENCE.label, fontsize=10)
    fig.suptitle("cyto7 per-vertex support: combined + components", fontsize=15, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out_path}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def default_annot() -> str:
    v2 = DERIVED_DIR / "pial.lh.cyto7.v2.annot"
    if v2.exists():
        return str(DERIVED_DIR / "pial.{hemi}.cyto7.v2.annot")
    return str(cfg.atlas_dir("provenance/as_painted") /
               "pial.{hemi}.cyto7.annot")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annot", default=default_annot())
    p.add_argument("--out-derived", type=Path, default=DERIVED_DIR)
    p.add_argument("--out-fig", type=Path, default=REFINE_DIR)
    p.add_argument("--w-atlas", type=float, default=1.0)
    p.add_argument("--w-topo", type=float, default=1.0)
    p.add_argument("--w-geom", type=float, default=1.0)
    p.add_argument("--w-data", type=float, default=0.5)
    p.add_argument("--w-prior", type=float, default=1.0)
    p.add_argument("--topo-hops", type=int, default=0,
                   help="Graph-hops to grow the topo penalty around each violation. "
                        "0 = STRICT exact adjacency (v4 default; GC request); 2 = old softening.")
    p.add_argument("--boundary-verts", type=int, default=3)
    p.add_argument("--medial-buffer-mm", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--allow-missing-myelin", action="store_true",
                   help="proceed with an empty data overlay if the T1w/T2w cache is absent; "
                        "without this the run stops rather than falling back silently (RR34)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    annot_paths = resolve_annot_paths(args.annot)
    version = version_label(annot_paths)
    print(f"Support map for cyto7 {version}: {annot_paths['lh']}")
    lookup = pd.read_csv(VE_DIR / "von_economo_cortical_types.csv")
    args.out_derived.mkdir(parents=True, exist_ok=True)
    args.out_fig.mkdir(parents=True, exist_ok=True)

    # ANATOMY-ONLY combined score: atlas + topology + geometry + prior. The
    # T1w/T2w "data" agreement is computed but kept OUT of the combined score
    # (it would be circular against the EXTEND structure-function analyses);
    # it is saved separately as confidence_data_overlay.shape.gii.
    anatomy_names = ["atlas", "topo", "geom", "prior"]
    maps = {k: {} for k in ["combined", *anatomy_names, "data_overlay"]}
    labs = {}
    summary_rows = []

    for hemi in HEMIS:
        lab = load_labels(annot_paths[hemi], hemi)
        coords, faces = load_surface(hemi, "pial")
        A = adjacency(faces, lab.shape[0])
        labs[hemi] = lab
        comps = compute_components(hemi, lab, coords, faces, A, lookup, args)
        anatomy = {k: comps[k] for k in anatomy_names}
        conf = combine(anatomy, lab.shape[0])
        maps["combined"][hemi] = conf
        for k in anatomy_names:
            maps[k][hemi] = comps[k][0]
        maps["data_overlay"][hemi] = comps["data"][0]

        # write shape.gii: combined + 4 anatomy components + separate data overlay
        write_shape_gii(args.out_derived / f"pial.{hemi}.cyto7.confidence.shape.gii", conf)
        for k in anatomy_names:
            write_shape_gii(args.out_derived / f"pial.{hemi}.cyto7.confidence_{k}.shape.gii",
                            comps[k][0])
        write_shape_gii(args.out_derived / f"pial.{hemi}.cyto7.confidence_data_overlay.shape.gii",
                        comps["data"][0])
        # categorical annot
        cat = categorical(conf)
        nib.freesurfer.io.write_annot(
            str(args.out_derived / f"pial.{hemi}.cyto7.confidence_categorical.annot"),
            cat, CAT_CTAB, CAT_NAMES, fill_ctab=True)
        print(f"  {hemi}: wrote anatomy-only support (combined+4 comps) + data_overlay + categorical")

        # summary per type
        for c in TYPE_CODES:
            m = lab == c
            v = conf[m]; v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            summary_rows.append({
                "hemi": hemi, "type": CODE_NAME[c], "n": int(v.size),
                "mean_support": round(float(v.mean()), 4),
                "median_support": round(float(np.median(v)), 4),
                "pct_low_<0.33": round(float(100 * np.mean(v < 0.33)), 1),
                "pct_high_>=0.66": round(float(100 * np.mean(v >= 0.66)), 1),
            })

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(args.out_fig / "support_summary.csv", index=False)
    print(f"  wrote {args.out_fig / 'support_summary.csv'}")
    if not args.no_figure:
        render_support(maps, labs, args.out_fig / "support_map.png", args.dpi)
    print("Done.")
    # console preview
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
