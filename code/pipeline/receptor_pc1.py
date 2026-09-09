"""Assemble the Hansen-2022 PET receptor/transporter PC1 on 32k fs_LR.

SPEC_external_validation.md Tier 1B (receptors). Fetches 19 canonical
receptor/transporter PET maps from neuromaps (one representative tracer per receptor,
the Hansen-2022 collection), resamples each MNI152 volume to fs_LR 32k
(``mni152_to_fslr``, nearest-registration-fusion, linear), z-scores each, and takes the
first principal component across the 19 maps as the "receptor PC1".

Each tracer is cached individually (``receptor_{source}_{desc}_fsLR32k_hemi-{H}.npy``) so
reruns skip the heavy volumetric fetch/resample; the PC1 result is cached as
``hansen_receptorpc1_fsLR32k_hemi-{H}.npy`` (loaded by ``external_validation.py``).

Sign convention: PC1 is sign-aligned to correlate positively with the across-tracer mean
density (a neutral anchor, independent of the cyto7 map), so the reported ρ vs cyto7 type is
not sign-rigged.

Run (heavy, needs network + Workbench):
    conda activate cyto7 && python scripts/receptor_pc1.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import os
from pathlib import Path

import numpy as np

from cyto7_surface_io import REPO_ROOT

CACHE = cfg.data_dir() / "neuromaps_cache"
WB_DEFAULT = cfg.workbench_dir()

#: Hansen-2022 canonical set: (receptor, [(source, desc) alternates]). One representative
#: tracer per receptor/transporter; alternates are tried in order if a download fails
#: (all verified in available_annotations()). Native surface versions are preferred at
#: fetch time (see _pick_space).
RECEPTORS = [
    ("5HT1a", [("beliveau2017", "cumi101")]),
    ("5HT1b", [("gallezot2010", "p943"), ("savli2012", "p943")]),
    ("5HT2a", [("beliveau2017", "cimbi36")]),
    ("5HT4", [("beliveau2017", "sb207145")]),
    ("5HT6", [("radnakrishnan2018", "gsk215083")]),
    ("5HTT", [("beliveau2017", "dasb"), ("savli2012", "dasb")]),
    ("A4B2", [("hillmer2016", "flubatine")]),
    ("CB1", [("normandin2015", "omar")]),
    ("D1", [("kaller2017", "sch23390")]),
    ("D2", [("smith2017", "flb457"), ("sandiego2015", "flb457"), ("alarkurtti2015", "raclopride")]),
    ("DAT", [("dukart2018", "fpcit")]),
    ("GABAa", [("norgaard2021", "flumazenil"), ("dukart2018", "flumazenil")]),
    ("H3", [("gallezot2017", "gsk189254")]),
    ("M1", [("naganawa2020", "lsn3172176")]),
    ("mGluR5", [("dubois2015", "abp688"), ("rosaneto", "abp688"), ("smart2019", "abp688")]),
    ("MU", [("kantonen2020", "carfentanil"), ("turtonen2020", "carfentanil")]),
    ("NAT", [("ding2010", "mrb"), ("hesse2017", "methylreboxetine")]),
    ("NMDA", [("galovic2021", "ge179")]),
    ("VAChT", [("aghourian2017", "feobv"), ("bedard2019", "feobv"), ("tuominen", "feobv")]),
]


def _wb_on_path():
    wb = os.environ.get("WORKBENCH_BIN") or WB_DEFAULT
    if wb and Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")


def _tracer_cache(source, desc, H):
    return CACHE / f"receptor_{source}_{desc}_fsLR32k_hemi-{H}.npy"


def _pc1_cache(H):
    return CACHE / f"hansen_receptorpc1_fsLR32k_hemi-{H}.npy"


def _pick_space(source, desc):
    """Choose the best (space, den/res) for a tracer: prefer a native SURFACE version
    (fsaverage > fsLR > civet) over MNI152 — neuromaps warns MNI152 cortical PET should
    only be used subcortically, and Hansen-2022 used surface versions where available."""
    from neuromaps import datasets
    cand = [a for a in datasets.available_annotations() if a[0] == source and a[1] == desc]
    for pref in ("fsaverage", "fsLR", "civet", "MNI152"):
        hit = sorted([a for a in cand if a[2] == pref], key=lambda a: a[3], reverse=True)
        if hit:
            return hit[0][2], hit[0][3]
    raise RuntimeError(f"no annotation for {source}/{desc}")


def _to_fslr32k(img, space, dr):
    """Resample a fetched annotation (surface tuple or MNI volume) to fs_LR 32k (L,R)."""
    from neuromaps import transforms
    if space == "fsaverage":
        return transforms.fsaverage_to_fslr(img, "32k", method="linear")
    if space == "civet":
        return transforms.civet_to_fslr(img, "32k", method="linear")
    if space == "fsLR":
        return img if dr == "32k" else transforms.fslr_to_fslr(img, "32k", method="linear")
    return transforms.mni152_to_fslr(img, "32k", method="linear")  # MNI152 volume


def _fetch_tracer(source, desc):
    """Fetch one tracer (best space), resample to fs_LR 32k, cache per hemi. Returns {H: arr}."""
    if _tracer_cache(source, desc, "L").exists() and _tracer_cache(source, desc, "R").exists():
        return {H: np.load(_tracer_cache(source, desc, H)) for H in ("L", "R")}
    _wb_on_path()
    from neuromaps import datasets
    space, dr = _pick_space(source, desc)
    print(f"  fetching {source}/{desc} ({space} {dr})...")
    kw = {"den": dr} if space in ("fsaverage", "fsLR", "civet") else {"res": dr}
    img = datasets.fetch_annotation(source=source, desc=desc, space=space, **kw)
    print(f"    resampling {space} -> fs_LR 32k...")
    lr = _to_fslr32k(img, space, dr)
    out = {}
    for H, gii in zip(("L", "R"), lr):
        arr = np.asarray(gii.agg_data() if hasattr(gii, "agg_data") else gii, float)
        assert arr.shape[0] == 32492, (source, desc, H, arr.shape)
        np.save(_tracer_cache(source, desc, H), arr)
        out[H] = arr
    return out


def build_receptor_pc1(force: bool = False) -> None:
    if not force and _pc1_cache("L").exists() and _pc1_cache("R").exists():
        print("  receptor PC1 already cached.")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"== assembling receptor PC1 from up to {len(RECEPTORS)} receptors ==")
    tracers, used, dropped = {}, [], []
    for rec, alts in RECEPTORS:
        got = None
        for source, desc in alts:
            try:
                got = _fetch_tracer(source, desc)
                used.append((rec, source, desc))
                break
            except Exception as exc:  # transient download / encoding / missing
                print(f"  WARN {rec}: {source}/{desc} failed ({type(exc).__name__}: {exc}); trying next.")
        if got is None:
            dropped.append(rec)
            print(f"  DROPPED {rec}: all sources failed.")
        else:
            tracers[rec] = got
    recs = [r for r, _a in RECEPTORS if r in tracers]
    print(f"  assembled {len(recs)}/{len(RECEPTORS)} receptors: {recs}")
    if dropped:
        print(f"  dropped (download failed): {dropped}")
    if len(recs) < 12:
        raise RuntimeError(f"only {len(recs)} receptors available; too few for a stable PC1.")

    # Stack into (vertices, receptors) matrices per hemi; complete-case cortex vertices.
    n = 32492
    mats = {H: np.column_stack([tracers[r][H] for r in recs]) for H in ("L", "R")}
    pooled = np.vstack([mats["L"], mats["R"]])                       # (2n, R)
    finite = np.isfinite(pooled).all(axis=1)                        # valid where all tracers finite
    # also drop all-zero (out-of-brain) rows
    finite &= ~np.all(pooled == 0, axis=1)
    X = pooled[finite]
    # z-score each tracer over valid vertices
    mu, sd = X.mean(0), X.std(0, ddof=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    # PC1 via SVD
    U, S, Vt = np.linalg.svd(Xz - Xz.mean(0), full_matrices=False)
    pc1_load = Vt[0]
    pc1 = Xz @ pc1_load
    # sign-align to the across-tracer mean density (neutral anchor)
    mean_density = Xz.mean(1)
    if np.corrcoef(pc1, mean_density)[0, 1] < 0:
        pc1 = -pc1; pc1_load = -pc1_load
    var_expl = float(S[0] ** 2 / np.sum(S ** 2))
    print(f"  PC1 variance explained: {var_expl:.1%}; n valid vertices: {int(finite.sum())}")
    print("  top |loadings|: " + ", ".join(
        f"{recs[i]} {pc1_load[i]:+.2f}"
        for i in np.argsort(-np.abs(pc1_load))[:6]))

    # scatter back to full 2n, NaN outside valid, split to hemis
    full = np.full(2 * n, np.nan)
    full[finite] = pc1
    for H, sl in (("L", slice(0, n)), ("R", slice(n, 2 * n))):
        np.save(_pc1_cache(H), full[sl])
        print(f"  cached {_pc1_cache(H).name}")


if __name__ == "__main__":
    build_receptor_pc1()
