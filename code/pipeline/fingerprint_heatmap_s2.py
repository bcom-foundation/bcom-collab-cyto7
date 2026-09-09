#!/usr/bin/env python
"""Supplementary Fig. S2 fingerprint heatmap (SPEC_meg_v2_figure_layout_fix, Part 2).

9-row z-scored per-type fingerprint: myelin, cortical thickness, functional
gradient, intrinsic timescale, and the five MEG band powers. The eight
non-timescale rows are the per-type medians from the released
`functional_summary_table_v9.csv` (unchanged); the **intrinsic-timescale row is
refreshed to the v2 source-reconstructed values** (ACF-area, from the persisted
32k group-mean map) and marked as surviving (*), so the row matches Table S3 /
Annex F (q=0.014) instead of the old group-average (non-surviving) timescale.
Band-power rows stay group-average (no *).

Same row order / colour map (RdBu_r) / filename / house style as before
(190 mm, 600 dpi, no title — title lives in the LaTeX caption). Released tables
are NOT modified.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

SF = REPO / "figures" / "v9" / "structure_function"
MAPS = SF / "meg_v2_maps"
LABCACHE = REPO / "resources" / "cyto7_derived" / "cache"
OUT = REPO / "figures" / "v9" / "manuscript" / "cyto7_fingerprint_heatmap.png"
STAGE = REPO / "manuscript" / "preprint" / "figures" / "cyto7_fingerprint_heatmap.png"

TYPES = ["Allocortex", "agranular", "dysgranular", "eulaminate I", "eulaminate II",
         "eulaminate III", "koniocortex"]
TLAB = ["Allocortex", "Agranular", "Dysgranular", "Eulaminate I", "Eulaminate II",
        "Eulaminate III", "Koniocortex"]
FEATS = [("myelin", "Myelin (T1w/T2w)"), ("thickness", "Cortical thickness"),
         ("gradient", "Functional gradient"), ("timescale", "Intrinsic timescale"),
         ("delta", "Delta power"), ("theta", "Theta power"), ("alpha", "Alpha power"),
         ("beta", "Beta power"), ("gamma", "Gamma power")]
TIMESCALE_V2_Q = 0.014   # single-member family q (Annex F / Table S3); survives


def v2_timescale_medians():
    """Per-type (7) hemisphere-averaged median of the v2 ACF-area timescale (ms)."""
    labs = {H: np.load(LABCACHE / f"v9_labels_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}
    d = {H: np.load(MAPS / f"int_area_ms_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}
    med = []
    for t in range(1, 8):
        per_hemi = [np.median(d[H][(labs[H] == t) & np.isfinite(d[H])]) for H in ("L", "R")]
        med.append(float(np.nanmean(per_hemi)))
    return np.array(med)


def main():
    t = pd.read_csv(SF / "functional_summary_table_v9.csv")
    ts_med = v2_timescale_medians()
    print("  v2 timescale per-type medians (ms):", np.round(ts_med, 1).tolist())

    Z = np.full((len(FEATS), 7), np.nan)
    surv = {}
    for i, (k, _) in enumerate(FEATS):
        if k == "timescale":
            med = ts_med
            surv[k] = TIMESCALE_V2_Q < 0.05           # v2 survives
        else:
            sub = t[t.FeatureKey == k]
            surv[k] = float(sub.iloc[0].p_spin_fdr) < 0.05
            med = np.array([np.mean(sub[sub.Type == ty]["median"].values)
                            if len(sub[sub.Type == ty]) else np.nan for ty in TYPES], float)
        Z[i] = (med - np.nanmean(med)) / np.nanstd(med)

    vmax = np.ceil(np.nanmax(np.abs(Z)) * 10) / 10
    w = 190 / 25.4
    fig, ax = plt.subplots(figsize=(w, w * 0.86)); fig.patch.set_facecolor("white")
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(7)); ax.set_xticklabels(TLAB, rotation=35, ha="right", fontsize=9)
    ylabels = [f"{nm}{' *' if surv[k] else ''}" for k, nm in FEATS]
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(ylabels, fontsize=9)
    for i in range(len(FEATS)):
        for j in range(7):
            v = Z[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.6 * vmax else "0.15")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("median z-score (per feature)", fontsize=9); cb.ax.tick_params(labelsize=7.5)
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(FEATS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=600, facecolor="white", bbox_inches="tight")
    print(f"saved {OUT}")
    if STAGE.parent.exists():
        import shutil
        shutil.copyfile(OUT, STAGE)
        print(f"staged {STAGE}")
    plt.close(fig)
    print("survivors (*):", [k for k, _ in FEATS if surv[k]])


if __name__ == "__main__":
    main()
