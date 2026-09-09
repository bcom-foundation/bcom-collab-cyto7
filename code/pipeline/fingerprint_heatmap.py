"""Fig 5 — feature fingerprint heatmap (SPEC_figures_round2 B: + molecular rows).

11 rows: the 9 structure/MEG/fMRI features (per-type z-scored median from
`functional_summary_table_v9.csv`) plus **gene PC1** and **receptor PC1** (external-validation
maps). ρ/p/q are NOT recomputed — the FDR q for the asterisk is read from
`external_validation_table.csv`; the per-type medians for the two molecular rows are computed
descriptively from the cached fs_LR maps (cyto7 v9 labels). RdBu_r, * = FDR q<0.05 (spin test).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from cyto7_surface_io import resolve_target_map  # noqa: E402

SF = os.path.join(REPO, "figures", "v9", "structure_function")
OUT = os.path.join(REPO, "figures", "v9", "manuscript", "figure_5_fingerprint_heatmap.png")
CACHE = os.path.join(REPO, "resources", "neuromaps_cache")

TYPES = ["Allocortex", "agranular", "dysgranular", "eulaminate I", "eulaminate II",
         "eulaminate III", "koniocortex"]
TLAB = ["Allocortex", "Agranular", "Dysgranular", "Eulaminate I", "Eulaminate II",
        "Eulaminate III", "Koniocortex"]

# display order: structural + gradient, then the two molecular rows grouped, then MEG.
FEATS = [("myelin", "Myelin (T1w/T2w)"), ("thickness", "Cortical thickness"),
         ("gradient", "Functional gradient"),
         ("genepc1", "Gene expression PC1"), ("receptorpc1", "Receptor PC1"),
         ("timescale", "Intrinsic timescale"), ("delta", "Delta power"),
         ("theta", "Theta power"), ("alpha", "Alpha power"), ("beta", "Beta power"),
         ("gamma", "Gamma power")]
MOLECULAR = {"genepc1": "abagen_genepc1", "receptorpc1": "hansen_receptorpc1"}


def molecular_medians_and_q():
    """Per-type median (avg of LH,RH per-type medians) + FDR q for gene/receptor PC1."""
    ext = pd.read_csv(os.path.join(SF, "external_validation_table.csv")).set_index("FeatureKey")
    lab = resolve_target_map("v9", "fs_LR")
    out = {}
    for key, stem in MOLECULAR.items():
        maps = {H: np.load(os.path.join(CACHE, f"{stem}_fsLR32k_hemi-{H}.npy")) for H in ("L", "R")}
        per_hemi = []
        for H in ("L", "R"):
            m = (lab[H] > 0) & np.isfinite(maps[H])
            meds = [np.median(maps[H][m & (lab[H] == c)]) if np.any(m & (lab[H] == c)) else np.nan
                    for c in range(1, 8)]
            per_hemi.append(meds)
        med = np.nanmean(np.array(per_hemi, float), axis=0)      # avg LH/RH per-type median
        q = float(ext.loc[key, "p_spin_fdr"]) if key in ext.index else np.nan
        out[key] = (med, q)
    return out


def main():
    t = pd.read_csv(os.path.join(SF, "functional_summary_table_v9.csv"))
    mol = molecular_medians_and_q()

    Z = np.full((len(FEATS), 7), np.nan)
    qv = {}
    for i, (k, _) in enumerate(FEATS):
        if k in mol:
            med, qv[k] = mol[k]
        else:
            sub = t[t.FeatureKey == k]
            qv[k] = float(sub.iloc[0].p_spin_fdr)
            med = np.array([np.mean(sub[sub.Type == ty]["median"].values)
                            if len(sub[sub.Type == ty]) else np.nan for ty in TYPES], float)
        Z[i] = (med - np.nanmean(med)) / np.nanstd(med)

    vmax = np.ceil(np.nanmax(np.abs(Z)) * 10) / 10
    fig, ax = plt.subplots(figsize=(8.4, 8.4)); fig.patch.set_facecolor("white")
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(7)); ax.set_xticklabels(TLAB, rotation=35, ha="right", fontsize=10)
    ylabels = [f"{nm}{' *' if (np.isfinite(qv[k]) and qv[k] < 0.05) else ''}" for k, nm in FEATS]
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(ylabels, fontsize=10)
    # highlight the two molecular rows
    for i, (k, _) in enumerate(FEATS):
        if k in MOLECULAR:
            ax.get_yticklabels()[i].set_color("#5a2a83")
            ax.get_yticklabels()[i].set_fontweight("bold")
    for i in range(len(FEATS)):
        for j in range(7):
            v = Z[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 * vmax else "0.15")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("median z-score (per feature)", fontsize=10)
    ax.set_title("Feature fingerprint per cortical type\n"
                 "(z-scored median; * = FDR q<0.05, spin test; purple = molecular)",
                 fontsize=13, weight="bold")
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(FEATS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    print("saved", OUT)
    print("q<0.05:", [k for k, _ in FEATS if np.isfinite(qv[k]) and qv[k] < 0.05])


if __name__ == "__main__":
    main()
