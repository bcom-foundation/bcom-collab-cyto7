"""Supplementary Figure S3 — external-validation gallery (SPEC_figures_round2 C).

Consolidates the independent (non-MRI) maps into one supplementary figure, feature-gallery
row style (surface, magma + box-by-type + spin ρ/p/q): **gene PC1**, **receptor PC1**, and the
**BigBrain profile skewness** (pre-registered index), plus a **BigBrain mean-profile-by-type**
panel showing the profiles differ in shape by type (the "signal present but underpowered" point).

ρ/p/q are READ from the tables (not recomputed): gene/receptor from `external_validation_table.csv`,
skewness from `bigbrain_profiles/profile_features_by_type.csv`. Reuses
`external_validation.render_gallery` for the three surface+box rows, then stacks the mean-profile
panel below. Does NOT rerun external_validation (keeps the Table S2 edit intact).

Output: `figures/v9/structure_function/supp_fig_S3_external_validation.png`
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from cyto7_surface_io import REPO_ROOT, resolve_target_map, LABEL_NAMES
import external_validation as ev
import bigbrain_profile_validation as bpv

SF = cfg.results_dir("tables") / "structure_function"
OUT = SF / "supp_fig_S3_external_validation.png"


def _skewness_map():
    prof = bpv.load_profiles()
    return {H: bpv.profile_features(prof[H])["skewness"] for H in ("L", "R")}


def _mean_profile_panel(path):
    """Standalone BigBrain mean intracortical profile by cyto7 type (line plot)."""
    prof = bpv.load_profiles()
    lab = resolve_target_map("v9", "fs_LR")
    fig, ax = plt.subplots(figsize=(6.2, 4.6)); fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("viridis")
    for t in range(1, 8):
        allc = np.concatenate([prof[H][:, lab[H] == t] for H in ("L", "R")], axis=1)
        allc = allc[:, allc.max(0) < 65535]
        if allc.shape[1] == 0:
            continue
        ax.plot(np.linspace(0, 100, 50), allc.mean(1), color=cmap((t - 1) / 6), lw=2,
                label=LABEL_NAMES[t - 1][:11])
    ax.set_xlabel("cortical depth % (pial→white)"); ax.set_ylabel("BigBrain staining intensity")
    ax.set_title("BigBrain mean intracortical profile by cyto7 type\n"
                 "(shapes differ by type — signal present; scalar indices underpowered, n=1)",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # features + read ρ/p/q from tables (no recompute)
    features = {
        "genepc1": {H: np.load(ev._cache("abagen", "genepc1", H)) for H in ("L", "R")},
        "receptorpc1": {H: np.load(ev._cache("hansen", "receptorpc1", H)) for H in ("L", "R")},
        "bb_skewness": _skewness_map(),
    }
    meta = {
        "genepc1": {"label": "Gene expression PC1 (AHBA)"},
        "receptorpc1": {"label": "Receptor PC1 (Hansen-2022 PET)"},
        "bb_skewness": {"label": "BigBrain profile skewness (pre-registered)"},
    }
    ext = pd.read_csv(SF / "external_validation_table.csv").set_index("FeatureKey")
    prof_tab = pd.read_csv(SF / "bigbrain_profiles" / "profile_features_by_type.csv").set_index("FeatureKey")
    rows = []
    for k, src in (("genepc1", ext.loc["genepc1"]), ("receptorpc1", ext.loc["receptorpc1"]),
                   ("bb_skewness", prof_tab.loc["skewness"])):
        rows.append({"FeatureKey": k, "spearman_rho": float(src.spearman_rho),
                     "p_spin": float(src.p_spin), "p_spin_fdr": float(src.p_spin_fdr)})
    stats_df = pd.DataFrame(rows)

    tmp = Path(tempfile.mkdtemp(prefix="s3_"))
    rows_png = tmp / "rows.png"
    prof_png = tmp / "profile.png"
    ev.render_gallery(features, meta, stats_df, rows_png, dpi=170)
    _mean_profile_panel(prof_png)

    # stack: 3 feature rows on top, mean-profile panel below
    top = mpimg.imread(str(rows_png)); bot = mpimg.imread(str(prof_png))
    fig = plt.figure(figsize=(11, 15)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[top.shape[0], bot.shape[0] * 1.25],
                          hspace=0.03, left=0.02, right=0.98, top=0.965, bottom=0.01)
    a0 = fig.add_subplot(gs[0]); a0.imshow(top); a0.axis("off")
    a1 = fig.add_subplot(gs[1]); a1.imshow(bot); a1.axis("off")
    fig.suptitle("Supplementary Fig S3 — external validation (gene / receptor / BigBrain), "
                 "cyto7 v9", fontsize=13, weight="bold", y=0.99)
    fig.savefig(str(OUT), dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
