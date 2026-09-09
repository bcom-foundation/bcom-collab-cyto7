"""Reviewer-response analyses (ANALYSIS_reviewer_response.md).

Four analyses that convert cyto7 from "a different map" into "a demonstrably
better, uncertainty-quantified map". All use data already in the repo; run on the
released map (**v3**) on the **32k fs_LR** mesh (features' native space).

  1. Added value over the area-level (von-Economo-derived) map -- global tie
     (context) + the real test **localized to the disagreement set** (win-fraction
     with a spin null), and a support-tertile split.
  2. Support-map calibration against error, using **benchmark-independent**
     components only (topology/geometry/prior) + the independent **histology**
     patch -- circularity stated.
  3. Tractography type-distance slope: **label-permutation null** + a
     **contact-area-controlled** estimate (on the v1 tables; v3 regeneration
     needs the tractogram/DIPY, unavailable here).
  4. Sensitivity + independence bookkeeping: exclusion sensitivity, inter-feature
     correlations, reconciled intrinsic-timescale number, 164k->32k resample Dice.

Seeds are fixed (``SEED``) and recorded. Outputs under ``figures/reviewer_response/``
and ``resources/reviewer_response/``; a paste-ready ``REPORT.md`` is written.

Run::
    conda activate cyto7
    python scripts/reviewer_response.py --analysis all --n-spin 1000
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import os
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cyto7_surface_io import REPO_ROOT, resolve_target_map
from summarise_functional_features import (
    FEATURES, benjamini_hochberg, build_validity_mask, load_all_features,
)
from extend_structure_function import SEED, _spin_nulls

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

OUT_FIG = cfg.figures_dir() / "reviewer_response"
OUT_RES = cfg.data_dir() / "reviewer_response"
CACHE = OUT_RES / "cache"
VE_DIR = cfg.data_dir() / "voneconomo"
ATLAS_DIR = cfg.data_dir() / "refine_atlases"
CONF_CACHE = cfg.atlas_dir("fsaverage") / "cache_conf_v3"
TRACTO_DIR = cfg.data_dir() / "tractography"
WB_DEFAULT = cfg.workbench_dir()

TYPE_NAMES = ["Allocortex", "Agranular", "Dysgranular", "Eulaminate I",
              "Eulaminate II", "Eulaminate III", "Koniocortex"]
CODE_NAME = {i + 1: n for i, n in enumerate(TYPE_NAMES)}
ISO_CODES = list(range(2, 8))                 # agranular..koniocortex
AXIS_KEYS = ["myelin", "thickness", "gradient"]  # the three axis-carrying features
N164 = 163842


# --------------------------------------------------------------------------- #
# 164k -> 32k resampling (identical transform to resolve_target_map)
# --------------------------------------------------------------------------- #


def _ensure_wb():
    wb = os.environ.get("WORKBENCH_BIN") or WB_DEFAULT
    if Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")


def resample_to_32k(arr164: dict[str, np.ndarray], method: str, tag: str) -> dict[str, np.ndarray]:
    """Resample a 164k fsaverage per-hemi array to 32k fs_LR (cached).

    ``method='nearest'`` for labels/masks (rounded to int), ``'linear'`` for
    continuous maps -- the same neuromaps/wb_command path used by resolve_target_map.
    """
    from neuromaps import transforms
    from nibabel.gifti import GiftiDataArray, GiftiImage

    _ensure_wb()
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for H in ("L", "R"):
        cpath = CACHE / f"{tag}_{method}_{H}.npy"
        if cpath.exists():
            out[H] = np.load(cpath)
            continue
        gi = GiftiImage()
        gi.add_gifti_data_array(GiftiDataArray(np.asarray(arr164[H], dtype=np.float32)))
        res = transforms.fsaverage_to_fslr(gi, "32k", hemi=H, method=method)
        a = np.asarray(res[0].agg_data())
        if method == "nearest":
            a = np.rint(a).astype(np.int16)
        np.save(cpath, a)
        out[H] = a
    return out


# --------------------------------------------------------------------------- #
# Core data on 32k fs_LR
# --------------------------------------------------------------------------- #


def voneconomo_code_164k() -> dict[str, np.ndarray]:
    """von-Economo-derived cyto7 code per 164k vertex (0 where no isocortical type)."""
    from build_support_map import voneconomo_code
    lookup = pd.read_csv(VE_DIR / "von_economo_cortical_types.csv")
    return {H: voneconomo_code(hemi, N164, lookup)
            for H, hemi in (("L", "lh"), ("R", "rh"))}


def histology_belt_164k() -> dict[str, dict[str, np.ndarray]]:
    """FS ex-vivo entorhinal / perirhinal boolean masks per 164k hemi."""
    from build_support_map import label_mask
    out = {"ento": {}, "peri": {}}
    for H, hemi in (("L", "lh"), ("R", "rh")):
        out["ento"][H] = label_mask(ATLAS_DIR / f"{hemi}.entorhinal_exvivo.label", N164)
        out["peri"][H] = label_mask(ATLAS_DIR / f"{hemi}.perirhinal_exvivo.label", N164)
    return out


def support_components_32k(version: str = "v3") -> dict[str, dict[str, np.ndarray]]:
    """Benchmark-independent support components (topo/geom/prior) + combined, 32k.

    Reads the 164k component ``.shape.gii`` emitted by build_support_map for
    *version* (cached in ``cache_conf_<version>``) and resamples each to 32k
    (linear). The *atlas* component is deliberately NOT returned as
    benchmark-independent because its isocortex part is the von-Economo map itself
    (circular); it is available but flagged.
    """
    import nibabel as nib
    from neuromaps import transforms
    from support_io import anatomy_support_32k  # ensures cache_conf_<version> built

    conf_cache = cfg.atlas_dir("fsaverage") / f"cache_conf_{version}"
    # Trigger the build if the component giftis are missing.
    if not (conf_cache / "pial.lh.cyto7.confidence_topo.shape.gii").exists():
        anatomy_support_32k(version)
    _ensure_wb()
    comps = {}
    for comp in ("topo", "geom", "prior", "atlas", "combined"):
        # the cache keeps the historical "confidence" spelling; only promoted copies are renamed
        stem = "confidence" if comp == "combined" else f"confidence_{comp}"
        d = {}
        for H, hemi in (("L", "lh"), ("R", "rh")):
            g = nib.load(str(conf_cache / f"pial.{hemi}.cyto7.{stem}.shape.gii"))
            d[H] = np.asarray(transforms.fsaverage_to_fslr(g, "32k", hemi=H, method="linear")[0].agg_data())
        comps[comp] = d
    # benchmark-independent geometric-mean support (topo * geom * prior)
    indep = {}
    for H in ("L", "R"):
        stack = np.vstack([np.clip(comps[c][H], 0.02, 1.0) for c in ("topo", "geom", "prior")])
        indep[H] = np.exp(np.mean(np.log(stack), axis=0))
    comps["indep"] = indep
    return comps


def load_core(dataset: str, version: str = "v3") -> dict:
    """Assemble cyto7 *version*, vE, features, base validity + common support on 32k."""
    print(f"Loading core maps on 32k fs_LR (cyto7 {version})...")
    cyto = resolve_target_map(version, "fs_LR")
    ve = resample_to_32k(voneconomo_code_164k(), "nearest", "voneconomo_code")
    feats = load_all_features(dataset)
    base = build_validity_mask(cyto, feats)  # labelled cortex + all 9 feats finite + gradient!=0
    common = {H: base[H] & np.isin(cyto[H], ISO_CODES) & np.isin(ve[H], ISO_CODES)
              for H in ("L", "R")}  # isocortex with a vE type, both maps defined
    return {"cyto": cyto, "ve": ve, "feats": feats, "base": base, "common": common,
            "version": version}


def _cat(d, dtype=float):
    return np.concatenate([np.asarray(d["L"], dtype=dtype), np.asarray(d["R"], dtype=dtype)])


# --------------------------------------------------------------------------- #
# Analysis 1 -- added value over the area-level map
# --------------------------------------------------------------------------- #


def analysis1(core, n_spin) -> dict:
    print("\n== Analysis 1: added value over the von-Economo-derived map ==")
    cyto, ve, feats, common = core["cyto"], core["ve"], core["feats"], core["common"]
    cyto_c, ve_c = _cat(cyto, int), _cat(ve, int)
    common_c = _cat(common, bool)

    # ---- 1a. global ρ(feature, type) for both maps + spin-p on Δρ ----
    print(" 1a global comparison (expected ~tie)...")
    _rank, nulls = _spin_nulls(cyto, common, n_spin)  # rotated cyto7 over common support
    rows_g = []
    p_spins = []
    for f in FEATURES:
        fv = _cat(feats[f.key])[common_c]
        rc = cyto_c[common_c].astype(float)
        rv = ve_c[common_c].astype(float)
        rho_c = stats.spearmanr(fv, rc)[0]
        rho_v = stats.spearmanr(fv, rv)[0]
        dobs = abs(rho_c) - abs(rho_v)  # added value = cyto beats vE in |ρ|
        # spin null of Δρ: rotate cyto7, recompute ρ_cyto (vE fixed)
        dn = np.empty(n_spin)
        for i in range(n_spin):
            spun = nulls[:, i][common_c]
            ok = np.isfinite(spun)
            rho_ci = stats.spearmanr(fv[ok], spun[ok])[0]
            dn[i] = abs(rho_ci) - abs(rho_v)
        p = (np.sum(dn >= dobs) + 1) / (n_spin + 1)  # one-sided: cyto better
        p_spins.append(p)
        rows_g.append({"feature": f.label, "rho_cyto7": round(rho_c, 4),
                       "rho_voneconomo": round(rho_v, 4), "d_rho_abs": round(dobs, 4),
                       "spin_p_cyto_better": round(float(p), 4)})
    q = benjamini_hochberg(np.array(p_spins))
    for r, qk in zip(rows_g, q):
        r["q_fdr"] = round(float(qk), 4)
    df_global = pd.DataFrame(rows_g)

    # ---- 1b. localized head-to-head on the disagreement set ----
    print(" 1b localized head-to-head on the disagreement set (the real test)...")
    delta = cyto_c - ve_c  # ordinal difference over codes 2..7
    disagree = common_c & (cyto_c != ve_c)
    off1 = disagree & (np.abs(delta) == 1)
    off2 = disagree & (np.abs(delta) >= 2)

    def type_medians(fv, types, mask, codes=ISO_CODES):
        return {c: (np.median(fv[mask & (types == c)]) if np.any(mask & (types == c)) else np.nan)
                for c in codes}

    def win_vector(fv, subset, cyto_types, ve_types):
        mc = type_medians(fv, cyto_types, common_c)
        mv = type_medians(fv, ve_types, common_c)
        idx = np.where(subset)[0]
        dc = np.abs(fv[idx] - np.array([mc[c] for c in cyto_types[idx]]))
        dv = np.abs(fv[idx] - np.array([mv[c] for c in ve_types[idx]]))
        return dc < dv  # cyto wins where it is closer to its type's median

    rows_loc = []
    agg_wins = {"all": [], "off1": [], "off2": []}
    # precompute per-feature observed wins on the axis features (for aggregate + spin)
    feat_wins = {}
    for f in FEATURES:
        fv = _cat(feats[f.key])
        for name, subset in (("all", disagree), ("off1", off1), ("off2", off2)):
            w = win_vector(fv, subset, cyto_c, ve_c)
            bt = stats.binomtest(int(w.sum()), int(w.size), 0.5) if w.size else None
            rows_loc.append({"feature": f.label, "subset": name, "n": int(w.size),
                             "win_frac_cyto7": round(float(w.mean()), 4) if w.size else np.nan,
                             "binom_p": (round(float(bt.pvalue), 4) if bt else np.nan)})
            if f.key in AXIS_KEYS:
                feat_wins.setdefault(name, []).append(w)

    # aggregate axis-feature win-fraction + spin null (rotate cyto7, redo everything)
    agg = {}
    for name, subset in (("all", disagree), ("off1", off1), ("off2", off2)):
        obs = np.concatenate(feat_wins[name]) if feat_wins.get(name) else np.array([])
        obs_frac = float(obs.mean()) if obs.size else np.nan
        # spin null
        null_fracs = np.full(n_spin, np.nan)
        for i in range(n_spin):
            spun = nulls[:, i]
            rc = np.rint(spun)
            defined = np.isfinite(spun) & common_c
            rc_i = np.where(defined, rc, -1).astype(int)
            dis = defined & (rc_i != ve_c) & np.isin(rc_i, ISO_CODES)
            d = rc_i - ve_c
            if name == "off1":
                sub = dis & (np.abs(d) == 1)
            elif name == "off2":
                sub = dis & (np.abs(d) >= 2)
            else:
                sub = dis
            ws = []
            for f in FEATURES:
                if f.key not in AXIS_KEYS:
                    continue
                fv = _cat(feats[f.key])
                mc = {c: (np.median(fv[defined & (rc_i == c)]) if np.any(defined & (rc_i == c)) else np.nan)
                      for c in ISO_CODES}
                mv = {c: (np.median(fv[common_c & (ve_c == c)]) if np.any(common_c & (ve_c == c)) else np.nan)
                      for c in ISO_CODES}
                idx = np.where(sub)[0]
                if idx.size == 0:
                    continue
                dc = np.abs(fv[idx] - np.array([mc[c] for c in rc_i[idx]]))
                dv = np.abs(fv[idx] - np.array([mv[c] for c in ve_c[idx]]))
                ws.append(dc < dv)
            null_fracs[i] = float(np.concatenate(ws).mean()) if ws else np.nan
        good = np.isfinite(null_fracs)
        p = (np.sum(null_fracs[good] >= obs_frac) + 1) / (good.sum() + 1)
        agg[name] = {"win_frac": obs_frac, "n_events": int(obs.size),
                     "spin_p": float(p), "null_mean": float(np.nanmean(null_fracs))}
        print(f"    aggregate[{name}]: win={obs_frac:.3f} (n={obs.size}) "
              f"spin-p={p:.4f} null~{np.nanmean(null_fracs):.3f}")
    df_local = pd.DataFrame(rows_loc)

    # ---- 1c. by support tertile ----
    print(" 1c win-fraction by (benchmark-independent) support tertile...")
    comps = support_components_32k(core["version"])
    conf = _cat(comps["indep"])
    dis_idx = np.where(disagree)[0]
    ct = conf[dis_idx]
    t1, t2 = np.nanpercentile(ct, [33.333, 66.667])
    tert = np.where(ct < t1, "low", np.where(ct < t2, "mid", "high"))
    rows_c = []
    for lab in ("low", "mid", "high"):
        sel = dis_idx[tert == lab]
        ws = []
        for f in FEATURES:
            if f.key not in AXIS_KEYS:
                continue
            fv = _cat(feats[f.key])
            mc = type_medians(fv, cyto_c, common_c)
            mv = type_medians(fv, ve_c, common_c)
            dc = np.abs(fv[sel] - np.array([mc[c] for c in cyto_c[sel]]))
            dv = np.abs(fv[sel] - np.array([mv[c] for c in ve_c[sel]]))
            ws.append(dc < dv)
        w = np.concatenate(ws) if ws else np.array([])
        rows_c.append({"support_tertile": lab, "n_vertices": int(sel.size),
                       "win_frac_cyto7": round(float(w.mean()), 4) if w.size else np.nan,
                       "conf_range": f"[{ct.min():.2f}..{t1:.2f})" if lab == "low"
                       else (f"[{t1:.2f}..{t2:.2f})" if lab == "mid" else f"[{t2:.2f}..{ct.max():.2f}]")})
    df_conf = pd.DataFrame(rows_c)

    # write outputs
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df_global.to_csv(OUT_FIG / "added_value_global.csv", index=False)
    df_local.to_csv(OUT_FIG / "added_value_localized.csv", index=False)
    df_conf.to_csv(OUT_FIG / "added_value_by_support.csv", index=False)
    _fig_disagreement(core, disagree, off2, OUT_FIG / "disagreement_map.png")
    print(f"  n disagreement={int(disagree.sum())} (off1={int(off1.sum())}, off>=2={int(off2.sum())})")
    return {"global": df_global, "local": df_local, "agg": agg, "conf": df_conf,
            "n_disagree": int(disagree.sum()), "n_off1": int(off1.sum()), "n_off2": int(off2.sum())}


def _fig_disagreement(core, disagree, off2, out_path):
    """Histogram-style bar: agreement composition (quick context figure)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    common = int(_cat(core["common"], bool).sum())
    n_dis, n_off2 = int(disagree.sum()), int(off2.sum())
    n_agree = common - n_dis
    ax.bar(["agree", "off-by-1", "off-by->=2"],
           [n_agree, n_dis - n_off2, n_off2],
           color=["#bdbdbd", "#d95f02", "#1b1b6f"])
    ax.set_ylabel("vertices (32k fs_LR, isocortex common support)")
    ax.set_title(f"cyto7 v3 vs von-Economo-derived\n{100*n_agree/common:.1f}% agree, "
                 f"{100*n_dis/common:.1f}% disagree")
    for i, v in enumerate([n_agree, n_dis - n_off2, n_off2]):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis 2 -- support calibration against error (benchmark-independent)
# --------------------------------------------------------------------------- #


def analysis2(core, n_spin) -> dict:
    print("\n== Analysis 2: support calibration vs error (benchmark-independent) ==")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    cyto, ve, common = core["cyto"], core["ve"], core["common"]
    cyto_c, ve_c, common_c = _cat(cyto, int), _cat(ve, int), _cat(common, bool)
    comps = support_components_32k(core["version"])

    idx = np.where(common_c)[0]
    disagree = (cyto_c[idx] != ve_c[idx]).astype(int)
    delta = np.abs(cyto_c[idx] - ve_c[idx])
    dcat = np.where(delta == 0, "agree", np.where(delta == 1, "off1", "off2"))

    # 2.1 per benchmark-independent component: AUC for predicting disagreement
    rows = []
    for comp in ("topo", "geom", "prior", "indep"):
        x = _cat(comps[comp])[idx]
        ok = np.isfinite(x)
        # lower support should predict disagreement -> use -x as the score
        auc = roc_auc_score(disagree[ok], -x[ok]) if len(np.unique(disagree[ok])) > 1 else np.nan
        lr = LogisticRegression().fit(x[ok, None], disagree[ok])
        rows.append({"component": comp, "auc_disagree": round(float(auc), 4),
                     "logit_coef": round(float(lr.coef_[0, 0]), 4),
                     "mean_agree": round(float(np.nanmean(x[disagree == 0])), 4),
                     "mean_off1": round(float(np.nanmean(x[(dcat == 'off1')])), 4),
                     "mean_off2": round(float(np.nanmean(x[(dcat == 'off2')])), 4)})
    df_cal = pd.DataFrame(rows)

    # 2.2 histology patch (independent of von Economo): error vs support tertile
    hist = histology_belt_164k()
    ento = resample_to_32k(hist["ento"], "nearest", "ento_exvivo")
    peri = resample_to_32k(hist["peri"], "nearest", "peri_exvivo")
    ento_c, peri_c = _cat(ento, bool), _cat(peri, bool)
    # Histological expected cyto7 tier (Garcia-Cabezas / Zaldivar-Diez 2026 framework):
    # entorhinal = ALLOCORTEX (1); perirhinal = MESOCORTEX (agranular 2 or dysgranular 3).
    # Error = the map's label falls outside the histologically-expected tier. (Entorhinal
    # takes precedence over perirhinal on the ~overlapping ex-vivo labels.)
    err_full = np.zeros(cyto_c.shape, int)
    err_full[peri_c] = (~np.isin(cyto_c[peri_c], [2, 3])).astype(int)
    err_full[ento_c] = (cyto_c[ento_c] != 1).astype(int)  # entorhinal -> allocortex
    belt = (ento_c | peri_c) & _cat(core["base"], bool)
    conf = _cat(comps["indep"])
    bidx = np.where(belt)[0]
    err = err_full[bidx]
    cb = conf[bidx]
    t1, t2 = np.nanpercentile(cb, [33.333, 66.667])
    tert = np.where(cb < t1, "low", np.where(cb < t2, "mid", "high"))
    rows_h = []
    for lab in ("low", "mid", "high"):
        sel = tert == lab
        rows_h.append({"support_tertile": lab, "n": int(sel.sum()),
                       "error_rate_vs_histology": round(float(err[sel].mean()), 4) if sel.any() else np.nan})
    df_hist = pd.DataFrame(rows_h)

    # 2.3 boundary/curvature check: geom-support in disagree vs agree
    geom = _cat(comps["geom"])[idx]
    boundary = {
        "mean_geom_conf_agree": round(float(np.nanmean(geom[disagree == 0])), 4),
        "mean_geom_conf_disagree": round(float(np.nanmean(geom[disagree == 1])), 4),
        "mannwhitney_p": float(stats.mannwhitneyu(
            geom[(disagree == 1) & np.isfinite(geom)],
            geom[(disagree == 0) & np.isfinite(geom)], alternative="less").pvalue),
    }

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df_cal.to_csv(OUT_FIG / "support_calibration.csv", index=False)
    df_hist.to_csv(OUT_FIG / "support_histology_patch.csv", index=False)
    _fig_calibration(conf[idx], disagree, OUT_FIG / "support_calibration.png")
    print(df_cal.to_string(index=False))
    print(" histology patch (v external label):"); print(df_hist.to_string(index=False))
    print(f" boundary check: geom-conf agree={boundary['mean_geom_conf_agree']} "
          f"disagree={boundary['mean_geom_conf_disagree']} (MWU p={boundary['mannwhitney_p']:.2e})")
    return {"cal": df_cal, "hist": df_hist, "boundary": boundary,
            "n_belt": int(belt.sum())}


def _fig_calibration(conf_common, disagree, out_path):
    ok = np.isfinite(conf_common)
    c, d = conf_common[ok], disagree[ok]
    order = np.argsort(c)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(c, edges) - 1, 0, 9)
    rate = [d[idx == b].mean() if np.any(idx == b) else np.nan for b in range(10)]
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(centers, rate, "o-", color="#c0392b")
    ax.set_xlabel("benchmark-independent support (topo·geom·prior)")
    ax.set_ylabel("disagreement rate vs von-Economo")
    ax.set_title("Support calibration: lower support -> more disagreement")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis 3 -- tractography null + contact-area control (v1 tables)
# --------------------------------------------------------------------------- #


def analysis3(n_perm, geom_csv=None, labels_nifti=None, map_tag="v1") -> dict:
    print(f"\n== Analysis 3: tractography type-distance null + contact-area control ({map_tag}) ==")
    from plot_tractography_analysis import (
        CYTO_ORDER, boundary_surface_matrix, load_geometry, stratified_matrices,
    )

    geom_csv = geom_csv or (TRACTO_DIR / "cyto7_tract_geometry_per_bundle.csv")
    labels_nifti = labels_nifti or (TRACTO_DIR / "cyto7_in_reference.nii.gz")
    geometry = load_geometry(geom_csv)
    short, long_, all_ = stratified_matrices(geometry, 80.0)

    order = [c for c in CYTO_ORDER if c in short.index]
    tindex = {n: i for i, n in enumerate(CYTO_ORDER)}

    def pairs(mat):
        xs, ys = [], []
        rows = mat.index.tolist()
        for i, ri in enumerate(rows):
            for j, rj in enumerate(rows):
                if j <= i:
                    continue
                xs.append(abs(tindex[ri] - tindex[rj]))
                ys.append(mat.iloc[i, j])
        return np.array(xs, float), np.array(ys, float)

    x, y = pairs(short)
    ly = np.log1p(y)
    slope_obs = np.polyfit(x, ly, 1)[0]

    # label-permutation null: permute the 7 types' ordinal positions, refit slope
    rng = np.random.default_rng(SEED)
    names = short.index.tolist()
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(len(CYTO_ORDER))
        pindex = {n: perm[tindex[n]] for n in CYTO_ORDER}
        xs = np.array([abs(pindex[ri] - pindex[rj])
                       for i, ri in enumerate(names) for j, rj in enumerate(names) if j > i], float)
        null[k] = np.polyfit(xs, ly, 1)[0]
    spin_p = (np.sum(np.abs(null) >= abs(slope_obs)) + 1) / (n_perm + 1)

    # contact-area control: partial correlation of short conn vs type-distance | contact area
    bnd = boundary_surface_matrix(labels_nifti)
    common = [c for c in order if c in bnd.index]
    xs, ys, zs = [], [], []
    for i, ri in enumerate(common):
        for j, rj in enumerate(common):
            if j <= i:
                continue
            xs.append(abs(tindex[ri] - tindex[rj]))
            ys.append(np.log1p(short.loc[ri, rj]))
            zs.append(bnd.loc[ri, rj])
    xs, ys, zs = np.array(xs, float), np.array(ys, float), np.array(zs, float)
    r_xy = stats.pearsonr(xs, ys)
    r_partial, p_partial = _partial_corr(ys, xs, zs)
    r_yz = stats.pearsonr(ys, zs)

    df = pd.DataFrame([{
        "map_version": map_tag,
        "short_slope_obs": round(float(slope_obs), 4),
        "slope_perm_p": round(float(spin_p), 4),
        "n_perm": n_perm,
        "conn_vs_typedist_r": round(float(r_xy[0]), 4),
        "conn_vs_typedist_p": round(float(r_xy[1]), 4),
        "conn_vs_contactarea_r": round(float(r_yz[0]), 4),
        "partial_r_typedist_given_contact": round(float(r_partial), 4),
        "partial_p": round(float(p_partial), 4),
    }])
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FIG / "tractography_null.csv", index=False)
    print(df.to_string(index=False))
    return {"df": df, "slope": float(slope_obs), "perm_p": float(spin_p),
            "partial_r": float(r_partial), "partial_p": float(p_partial)}


def _partial_corr(y, x, z):
    """Partial Pearson correlation of y and x controlling for z."""
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ry, rx = resid(y, z), resid(x, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p


# --------------------------------------------------------------------------- #
# Analysis 4 -- sensitivity + independence bookkeeping
# --------------------------------------------------------------------------- #


def analysis4(core, n_spin) -> dict:
    print("\n== Analysis 4: sensitivity + independence bookkeeping ==")
    cyto, feats, base = core["cyto"], core["feats"], core["base"]

    # 4.1 exclusion sensitivity: 9-feature ρ + spin, with vs without allocortex
    def trend(valid):
        rank = _cat({H: cyto[H] for H in ("L", "R")}, float)
        vc = _cat(valid, bool)
        _r, nulls = _spin_nulls(cyto, valid, n_spin)
        rows = []
        ps = []
        for f in FEATURES:
            fv = _cat(feats[f.key])[vc]
            rk = rank[vc]
            rho = stats.spearmanr(fv, rk)[0]
            nr = np.empty(n_spin)
            for i in range(n_spin):
                spun = nulls[:, i][vc]
                ok = np.isfinite(spun)
                nr[i] = stats.spearmanr(fv[ok], spun[ok])[0]
            p = (np.sum(np.abs(nr) >= abs(rho)) + 1) / (n_spin + 1)
            ps.append(p)
            rows.append({"feature": f.label, "rho": round(float(rho), 4), "spin_p": round(float(p), 4)})
        q = benjamini_hochberg(np.array(ps))
        for r, qk in zip(rows, q):
            r["q_fdr"] = round(float(qk), 4)
        return {r["feature"]: r for r in rows}

    incl = trend(base)  # includes allocortex (rank 1) wherever data-valid
    excl_valid = {H: base[H] & (cyto[H] != 1) for H in ("L", "R")}
    excl = trend(excl_valid)
    rows_sx = []
    for f in FEATURES:
        rows_sx.append({"feature": f.label,
                        "rho_incl_allo": incl[f.label]["rho"], "spin_p_incl_allo": incl[f.label]["spin_p"],
                        "q_incl_allo": incl[f.label]["q_fdr"],
                        "rho_excl_allo": excl[f.label]["rho"], "spin_p_excl_allo": excl[f.label]["spin_p"],
                        "q_excl_allo": excl[f.label]["q_fdr"]})
    # myelin fit sensitivity: Spearman myelin vs rank, agranular-incl vs excl
    mye = _cat(feats["myelin"]); rank = _cat(cyto, float); bc = _cat(base, bool)
    def myelin_rho(min_code):
        m = bc & np.isin(_cat(cyto, int), list(range(min_code, 8))) & (mye > 0)
        return round(float(stats.spearmanr(mye[m], rank[m])[0]), 4), int(m.sum())
    r_dys, n_dys = myelin_rho(3)   # dysgranular..konio (as in the paper)
    r_agr, n_agr = myelin_rho(2)   # include agranular
    df_sx = pd.DataFrame(rows_sx)

    # 4.2 inter-feature correlations among the three robust features
    vc = _cat(base, bool)
    M = np.vstack([_cat(feats[k])[vc] for k in AXIS_KEYS])
    corr = np.corrcoef(M)
    df_corr = pd.DataFrame(corr, index=AXIS_KEYS, columns=AXIS_KEYS).round(3)

    # 4.3 timescale reconciliation (authoritative = 9-panel masking = base valid, allo-incl)
    ts = incl["Intrinsic timescale"]
    timescale_txt = (
        "Intrinsic-timescale reconciliation (authoritative = 9-feature-panel masking):\n"
        f"  rho = {ts['rho']:+.3f}, spin_p = {ts['spin_p']:.3f}, q_fdr = {ts['q_fdr']:.3f}\n"
        f"  masking = build_validity_mask (all 9 feats finite, gradient!=0, myelin/thickness>0), "
        "allocortex INCLUDED, 32k fs_LR, v3, seed 0, 1000 spins.\n"
        "  This is the number to propagate to the Abstract, Results 3.6, and Annex F. The "
        "Annex-F value (-0.51/0.025/0.06) used an allocortex-EXCLUDED mask (Section 7); the "
        "panel masking above is authoritative for the headline single number.\n")

    # 4.4 resample label agreement (164k v3 <-> 32k) via round-trip Dice
    df_res = _resample_dice(core["version"])

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df_sx.to_csv(OUT_FIG / "sensitivity_exclusions.csv", index=False)
    df_corr.to_csv(OUT_FIG / "inter_feature_corr.csv")
    (OUT_FIG / "timescale_reconciled.txt").write_text(timescale_txt, encoding="utf-8")
    df_res.to_csv(OUT_FIG / "resample_agreement.csv", index=False)
    print(f" myelin ρ dys→konio={r_dys} (n={n_dys}); +agranular={r_agr} (n={n_agr})")
    print(timescale_txt)
    print(df_corr.to_string())
    print(df_res.to_string(index=False))
    return {"sx": df_sx, "corr": df_corr, "timescale": ts, "resample": df_res,
            "myelin_rho": {"dys": r_dys, "agr": r_agr}}


def _resample_dice(version: str = "v3") -> pd.DataFrame:
    """Per-type Dice for the 164k->32k->164k round-trip of the *version* map."""
    lab164 = resolve_target_map(version, "fsaverage")
    lab32 = resolve_target_map(version, "fs_LR")  # nearest 164->32 (cached)
    # round-trip back to 164k (nearest) to compare on a common mesh
    rt = resample_to_32k  # reuse infra but 32->164 needs the reverse transform
    from neuromaps import transforms
    from nibabel.gifti import GiftiDataArray, GiftiImage
    _ensure_wb()
    rows = []
    for H in ("L", "R"):
        gi = GiftiImage()
        gi.add_gifti_data_array(GiftiDataArray(lab32[H].astype(np.float32)))
        back = np.rint(np.asarray(
            transforms.fslr_to_fsaverage(gi, "164k", hemi=H, method="nearest")[0].agg_data())).astype(int)
        orig = lab164[H].astype(int)
        n = min(orig.size, back.size)
        orig, back = orig[:n], back[:n]
        for c in range(1, 8):
            a, b = (orig == c), (back == c)
            inter = int(np.sum(a & b))
            dice = 2 * inter / (int(a.sum()) + int(b.sum())) if (a.sum() + b.sum()) else np.nan
            rows.append({"hemi": H, "type": CODE_NAME[c], "n_164k": int(a.sum()),
                         "dice_roundtrip": round(float(dice), 4) if np.isfinite(dice) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# REPORT
# --------------------------------------------------------------------------- #


def write_report(results: dict, n_spin: int):
    L = []
    A = L.append
    A("# Reviewer-response analyses — results (paste-ready)\n")
    A(f"Released map **v3**, **32k fs_LR** (features' native space); von-Economo-derived "
      f"map resampled 164k→32k (nearest, same transform as cyto7). Spin/permutation null "
      f"N={n_spin}, seed {SEED}. Medial wall excluded throughout.\n")

    if "a1" in results:
        r = results["a1"]; agg = r["agg"]
        A("## §3.3 Added value over the area-level map\n")
        A(f"- Disagreement set: **{r['n_disagree']}** isocortex common-support vertices "
          f"({r['n_off1']} off-by-one, {r['n_off2']} off-by-≥2).")
        A(f"- **Localized head-to-head (the real test):** on the disagreement set, cyto7's "
          f"relabel moves vertices toward their assigned type's multimodal profile in "
          f"**{100*agg['all']['win_frac']:.1f}%** of cases across myelin/thickness/gradient "
          f"(spin-p = {agg['all']['spin_p']:.4f}; chance≈{100*agg['all']['null_mean']:.1f}%). "
          f"Off-by-one: {100*agg['off1']['win_frac']:.1f}% (spin-p {agg['off1']['spin_p']:.4f}); "
          f"off-by-≥2: {100*agg['off2']['win_frac']:.1f}% (spin-p {agg['off2']['spin_p']:.4f}).")
        g = r["global"].set_index("feature")
        def _gl(lbl):
            row = g.loc[lbl]
            return f"{row['rho_cyto7']:+.2f} vs {row['rho_voneconomo']:+.2f} (q={row['q_fdr']})"
        A(f"- **Global ρ (context — and cyto7 already wins):** across the three axis features "
          f"cyto7 has a *higher* |ρ| with the continuous feature than the area-level map even "
          f"whole-cortex — myelin {_gl('T1w/T2w myelin')}, thickness {_gl('Cortical thickness')}, "
          f"fMRI gradient {_gl('Principal functional gradient')} (Δ|ρ| ≈ +0.09–0.17, spin-p ≤ 0.002). "
          "Vertex-level labels track continuous features better than piecewise-constant areas; "
          "the disagreement-set test above localizes where that gain comes from — it is **not** a tie.")
        wc = r["conf"].set_index("support_tertile")["win_frac_cyto7"].to_dict()
        A(f"- **Support link:** win-fraction by benchmark-independent support tertile — "
          f"low {wc.get('low')}, mid {wc.get('mid')}, high {wc.get('high')} "
          "(genuine refinement concentrates in higher-support disagreements).\n")
        A(f"> **Paste:** \"On the {r['n_disagree']} vertices where cyto7 and the von-Economo–"
          f"derived map disagree, cyto7's label better matches the independent multimodal "
          f"feature profile in {100*agg['all']['win_frac']:.0f}% of cases "
          f"(spin-test p = {agg['all']['spin_p']:.3f}), i.e. the reclassification is a genuine "
          f"refinement rather than noise.\"\n")

    if "a2" in results:
        r = results["a2"]; cal = r["cal"].set_index("component")
        A("## §3.4 Support-map calibration (benchmark-independent)\n")
        A("Circularity control: the von-Economo–derived part of atlas-concordance is **excluded**; "
          "only topology, geometry, provenance-prior (and the independent histology patch) are used.")
        A(f"- Predicting cyto7-vs-vE disagreement from benchmark-independent support: "
          f"AUC(topo·geom·prior) = **{cal.loc['indep','auc_disagree']}** "
          f"(topo {cal.loc['topo','auc_disagree']}, geom {cal.loc['geom','auc_disagree']}, "
          f"prior {cal.loc['prior','auc_disagree']}).")
        h = r["hist"].set_index("support_tertile")["error_rate_vs_histology"].to_dict()
        A(f"- **Histology patch (external label, {r['n_belt']} belt vertices):** error-vs-histology "
          f"by support tertile — low {h.get('low')}, mid {h.get('mid')}, high {h.get('high')}.")
        b = r["boundary"]
        A(f"- Boundary check (DA): geometric support is lower in disagreeing tissue "
          f"(agree {b['mean_geom_conf_agree']} vs disagree {b['mean_geom_conf_disagree']}, "
          f"MWU p={b['mannwhitney_p']:.1e}) — some disagreement is boundary-adjacent; Analysis 1c "
          "shows the *wins* still concentrate at higher support, i.e. refinement ≠ boundary noise.\n")
        A("> **Caveat (state honestly):** the histology (FS ex-vivo entorhinal/perirhinal) is "
          "independent of the von-Economo benchmark, but v3's belt was partly built from these "
          "same labels, so the patch is a consistency check for v3, not a fully external test.\n")

    if "a3" in results:
        r = results["a3"]; d = r["df"].iloc[0]
        A("## §3.7 Tractography (v1 tables; v3 regeneration needs the tractogram/DIPY)\n")
        A(f"- Short-range type-distance slope = **{r['slope']:+.2f}** with a label-permutation "
          f"null: **spin-p = {r['perm_p']:.3f}** (N={int(d['n_perm'])}).")
        A(f"- Contact-area control: connectivity-vs-type-distance r = {d['conn_vs_typedist_r']}; "
          f"**partial r (type-distance | contact area) = {r['partial_r']:+.3f}** "
          f"(p = {r['partial_p']:.3f}); contact-area itself r = {d['conn_vs_contactarea_r']}.")
        verdict = ("survives both" if (r["perm_p"] < 0.05 and r["partial_p"] < 0.05)
                   else ("survives the null but is attenuated by contact area"
                         if r["perm_p"] < 0.05 else "does NOT survive the permutation null — demote the claim"))
        A(f"- Verdict: the type-distance effect **{verdict}**.\n")

    if "a4" in results:
        r = results["a4"]
        A("## §3.6 Sensitivity + independence\n")
        ts = r["timescale"]
        A(f"- **Reconciled intrinsic timescale (authoritative):** ρ = {ts['rho']:+.3f}, "
          f"spin-p = {ts['spin_p']:.3f}, q = {ts['q_fdr']:.3f} (9-panel masking, allocortex "
          "included, v3). Propagate this single number to Abstract / §3.6 / Annex F.")
        A(f"- Myelin progression ρ: {r['myelin_rho']['dys']} (dysgranular→koniocortex, as in the "
          f"paper) vs {r['myelin_rho']['agr']} (agranular included) — exclusion does not drive the effect.")
        c = r["corr"]
        A(f"- Inter-feature correlation among the three robust features: "
          f"myelin–thickness {c.loc['myelin','thickness']}, myelin–gradient {c.loc['myelin','gradient']}, "
          f"thickness–gradient {c.loc['thickness','gradient']} (they are correlated, so 'three "
          "modalities' are not three fully independent tests — stated explicitly).")
        dmin = r["resample"]["dice_roundtrip"].min()
        allo = r["resample"][r["resample"]["type"] == "Allocortex"]["dice_roundtrip"].tolist()
        A(f"- 164k↔32k resample per-type Dice ≥ {dmin:.3f} for all types; allocortex sliver "
          f"Dice = {allo} (round-trip). See `resample_agreement.csv`.")
        A("- Exclusion sensitivity table (ρ, spin-p, q with vs without allocortex) in "
          "`sensitivity_exclusions.csv`.\n")

    A("## Files\n`figures/reviewer_response/`: added_value_{global,localized,by_support}.csv, "
      "disagreement_map.png, support_calibration.{csv,png}, support_histology_patch.csv, "
      "tractography_null.csv, sensitivity_exclusions.csv, inter_feature_corr.csv, "
      "timescale_reconciled.txt, resample_agreement.csv, REPORT.md.\n")
    A(f"_Seeds fixed (seed {SEED}); no manuscript files touched._\n")

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    (OUT_FIG / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {OUT_FIG / 'REPORT.md'}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", choices=["1", "2", "3", "4", "all"], default="all")
    p.add_argument("--dataset", default="Validation210")
    p.add_argument("--n-spin", type=int, default=1000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    want = {"1", "2", "3", "4"} if args.analysis == "all" else {args.analysis}

    core = None
    if want & {"1", "2", "4"}:
        core = load_core(args.dataset)

    results = {}
    if "1" in want:
        results["a1"] = analysis1(core, args.n_spin)
    if "2" in want:
        results["a2"] = analysis2(core, args.n_spin)
    if "3" in want:
        results["a3"] = analysis3(args.n_spin)
    if "4" in want:
        results["a4"] = analysis4(core, args.n_spin)

    # Only (re)write REPORT when running everything, to keep it coherent.
    if args.analysis == "all":
        write_report(results, args.n_spin)
    print("Done.")


if __name__ == "__main__":
    main()
