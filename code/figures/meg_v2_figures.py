#!/usr/bin/env python
"""v2 MEG surface figures (SPEC_meg_v2_figures) — reuse feature_gallery's renderer.

Renders the v2 source-reconstructed group-mean maps on the 32k fs_LR inflated
surface (LH/RH lateral+medial) with cyto7 v9 borders, plus box-by-type panels
annotated with the v2 per-subject rho / spin-p / FDR-q (from
meg_dynamics_v2_summary.csv). Produces:

  * `meg_v2_dynamics_gallery.png`  — S1/S3 replacement: intrinsic timescale
    (survives), spectral centroid (survives), aperiodic-corrected beta (survives),
    aperiodic exponent (null).
  * `meg_v2_timescale_vs_exponent.png` — candidate main panel: the dissociation
    (timescale falls toward koniocortex; exponent flat/null).

House style: 190 mm width, 600 dpi, no title (title lives in the LaTeX caption).
Depends on scripts/meg_v2_build_maps.py having written figures/v9/structure_function/
meg_v2_maps/. Released tables are not touched.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg  # noqa: F401  (puts sibling code dirs on sys.path)

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from cyto7_surface_io import surface_path  # noqa: E402
from feature_gallery import precompute, VIEWS, NAMES  # reuse the renderer  # noqa: E402

MAPS = REPO / "figures" / "v9" / "structure_function" / "meg_v2_maps"
CACHE = REPO / "resources" / "cyto7_derived" / "cache"
OUTDIR = REPO / "figures" / "v9" / "manuscript"
SUMMARY = REPO / "figures" / "v9" / "structure_function" / "meg_dynamics_v2_summary.csv"
SUMMARY_SINGLE = REPO / "figures" / "v9" / "structure_function" / "meg_dynamics_v2_summary_fdr_singlemember.csv"

# (map key, summary-metric key, label, unit)
FEATS = {
    "int_area": ("int_area_ms", "int_area", "Intrinsic timescale", "ms"),
    "centroid": ("centroid", "centroid", "Spectral centroid", "Hz"),
    "osc_beta": ("osc_beta", "osc_beta", "Aperiodic-corrected beta power", "a.u."),
    "exponent": ("exponent", "exponent_fixed", "Aperiodic exponent", "a.u."),
}
# compact 2-line row-label (fits the left gutter without touching the surfaces)
GUTTER = {
    "int_area": "Intrinsic\ntimescale",
    "centroid": "Spectral\ncentroid",
    "osc_beta": "Aperiodic-corr.\nbeta power",
    "exponent": "Aperiodic\nexponent",
}


def load_map(mapkey):
    return {H: np.load(MAPS / f"{mapkey}_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}


def load_stats():
    """group ρ + spin p from the summary; FDR q from the SINGLE-MEMBER family
    (§2.6: one committed member per construct) so the panels show timescale/centroid
    q=0.014, slow-fast/β q=0.021, not the old over-split 0.028."""
    df = pd.read_csv(SUMMARY).set_index("metric")
    if SUMMARY_SINGLE.exists():
        q1 = pd.read_csv(SUMMARY_SINGLE).set_index("metric")["fdr_q_singlemember"]
        df["fdr_q"] = q1.reindex(df.index).combine_first(df["fdr_q"])
    return df


def render(feat_keys, out, dpi=600):
    lab = {H: np.load(CACHE / f"v9_labels_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}
    geo = {H: (lambda g: (np.asarray(g.darrays[0].data, float), np.asarray(g.darrays[1].data, int)))(
        nib.load(str(surface_path("Validation210", H, "inflated")))) for H in ("L", "R")}
    G = precompute(lab, geo)
    stats = load_stats()
    cmap = plt.get_cmap("magma")

    nrow = len(feat_keys)
    w_in = 190.0 / 25.4
    fig = plt.figure(figsize=(w_in, 1.72 * nrow))
    fig.patch.set_facecolor("white")
    # cols: [row-label gutter | 4 surfaces | colorbar | WIDE spacer | box-and-whisker]
    # the wide spacer keeps the colorbar's (right-side) tick numbers clear of the
    # box's (left-side) y-axis tick numbers; the gutter holds the 2-line row label.
    gs = fig.add_gridspec(nrow, 8, width_ratios=[0.38, 1, 1, 1, 1, 0.06, 0.55, 2.2],
                          wspace=0.04, hspace=1.05, left=0.005, right=0.997,
                          top=0.93, bottom=0.11)
    for r, fk in enumerate(feat_keys):
        mapkey, metric, name, unit = FEATS[fk]
        d = load_map(mapkey)
        allv = np.concatenate([d[H][lab[H] > 0] for H in ("L", "R")])
        allv = allv[np.isfinite(allv)]
        vmin, vmax = np.percentile(allv, 2), np.percentile(allv, 98)
        nm = Normalize(vmin, vmax)
        # --- dedicated left gutter for the metric name (never over the surfaces) ---
        axl = fig.add_subplot(gs[r, 0]); axl.axis("off")
        axl.text(0.5, 0.5, GUTTER[fk], transform=axl.transAxes,
                 rotation=90, va="center", ha="center", fontsize=8, weight="bold")
        for ci, (H, view) in enumerate(VIEWS):
            ax = fig.add_subplot(gs[r, ci + 1]); g = G[(H, view)]; val = d[H]
            fv = np.nanmean(val[g["F"]], axis=1); col = cmap(nm(fv))[:, :3]
            bad = g["medf"] | ~np.isfinite(fv)
            col[bad] = (0.68, 0.85, 0.90)
            col = np.clip(col * g["sh"][:, None], 0, 1)
            ax.add_collection(PolyCollection(g["verts"], facecolors=col, edgecolors="face", linewidths=0.3))
            ax.add_collection(LineCollection(g["segs"], colors="white", linewidths=0.3, alpha=0.6))
            x0, x1, y0, y1 = g["lim"]; ax.set_xlim(x0 - 2, x1 + 2); ax.set_ylim(y0 - 2, y1 + 2)
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                ax.set_title(f"{H}H {view}", fontsize=9)
        cax = fig.add_subplot(gs[r, 5]); cb = fig.colorbar(cm.ScalarMappable(norm=nm, cmap=cmap), cax=cax)
        # tick numbers on the RIGHT (into the wide spacer, away from the surfaces); the tiny
        # unit label alone on the LEFT (fits the small surface-to-colorbar gap).
        cb.ax.tick_params(labelsize=6.5)
        cax.set_title(unit, fontsize=7)  # unit above the colorbar -> clear of surfaces + box
        axb = fig.add_subplot(gs[r, 7])
        for t in range(1, 8):
            for H, off, fc, mc in (("L", -0.18, "0.35", "w"), ("R", 0.18, "0.78", "0.2")):
                vals = d[H][(lab[H] == t) & np.isfinite(d[H])]
                if vals.size:
                    axb.boxplot(vals, positions=[t + off], widths=0.32, showfliers=False, patch_artist=True,
                                boxprops=dict(facecolor=fc, edgecolor="0.3"), medianprops=dict(color=mc),
                                whiskerprops=dict(color=fc), capprops=dict(color=fc))
        s = stats.loc[metric]; sig = "*" if s.fdr_q < 0.05 else ""
        axb.set_xticks(range(1, 8)); axb.set_xticklabels([NAMES[t] for t in range(1, 8)], rotation=30, ha="right", fontsize=7.5)
        axb.set_xlim(0.4, 7.6); axb.grid(axis="y", ls=":", alpha=0.4)
        axb.tick_params(axis="y", labelsize=7)  # metric + unit named by the gutter label / colorbar
        # annotation as a single-line HEADER above the box axes -> never overlaps data or the edge
        axb.text(0.0, 1.03,
                 f"$\\rho$={s.group_mean_rho:+.2f}   $p$={s.spin_p:.3f}   $q$={s.fdr_q:.3f}{sig}   "
                 + ("survives" if sig else "n.s."),
                 transform=axb.transAxes, fontsize=7, fontweight="bold", va="bottom", ha="left", color="0.1")
        if r == 0:
            axb.legend(handles=[Patch(facecolor="0.35", label="LH"), Patch(facecolor="0.78", label="RH")],
                       loc="upper right", fontsize=7.5, frameon=False)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def main():
    # S1/S3 replacement gallery: the surviving dynamics gradients + the null exponent
    render(["int_area", "centroid", "osc_beta", "exponent"],
           OUTDIR / "meg_v2_dynamics_gallery.png")
    # Candidate main panel: the timescale (survives) vs exponent (null) dissociation
    render(["int_area", "exponent"], OUTDIR / "meg_v2_timescale_vs_exponent.png")


if __name__ == "__main__":
    main()
