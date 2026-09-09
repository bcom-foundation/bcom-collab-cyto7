"""Extra structure-function figures on v6 (SPEC_extra_figures_v6.md).

Reuses the MEG band features + per-type masking of ``summarise_functional_features``
(32k fs_LR, seed 0, allocortex included — the same single validity mask as the
fingerprint) and the shared ``figure_style`` palettes.

A. **Band composition per cyto7 type** — mean MEG band power (delta/theta/alpha/beta/
   gamma) per type, normalised to % of the 5-band total; stacked bars, allo -> konio.
   Descriptive/compositional: no single band survives the spin+FDR null (Annex F).
   -> ``band_composition_bars.png`` + ``band_composition.csv``.

B. **Allo / meso / iso grouping** — tiers allocortex=1, mesocortex=2-3, isocortex=4-7:
   1. a 3-tier surface map (medial + lateral, both hemispheres);
   2. the structure-function progression collapsed to the three tiers (mean +/- 95% CI
      for myelin, thickness, functional gradient, intrinsic timescale) + the band
      composition re-binned to the three tiers.
   -> ``tier_grouping_surface.png``, ``tier_progression.png`` + ``tier_progression.csv``.

Run::  conda activate cyto7 && python scripts/extra_figures_v6.py --map v6
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from cyto7_surface_io import REPO_ROOT, resolve_target_map
from figure_style import CYTO7_NAMES, TIER3, tier3_legend_handles
from make_presentation_figures import (
    HEMIS, LIGHT_BLUE, lighting_normals, lit_panel, load_surface, sulc_shading,
)
from summarise_functional_features import build_validity_mask, load_all_features

OUT_DIR = cfg.results_dir("tables") / "structure_function"

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_LABELS = {"delta": "Delta (0.5-4 Hz)", "theta": "Theta (4-8 Hz)",
               "alpha": "Alpha (8-12 Hz)", "beta": "Beta (12-30 Hz)",
               "gamma": "Gamma (30-60 Hz)"}
#: ordered low -> high frequency (viridis: low freq dark -> high freq bright).
BAND_COLORS = {b: tuple(mpl.colormaps["viridis"](x)[:3])
               for b, x in zip(BANDS, np.linspace(0.0, 0.92, len(BANDS)))}

#: tier grouping of the 7 cyto7 codes.
TIER_OF = {1: "allocortex", 2: "mesocortex", 3: "mesocortex",
           4: "isocortex", 5: "isocortex", 6: "isocortex", 7: "isocortex"}
TIER_ORDER = ["allocortex", "mesocortex", "isocortex"]
TIER_CODE = {"allocortex": 1, "mesocortex": 2, "isocortex": 3}

PROG_FEATURES = [("myelin", "Myelin (T1w/T2w)"), ("thickness", "Cortical thickness"),
                 ("gradient", "Functional gradient"), ("timescale", "Intrinsic timescale")]


# --------------------------------------------------------------------------- #
# Shared data assembly
# --------------------------------------------------------------------------- #


def _pooled(labels, feats, valid):
    """Pool valid vertices across both hemispheres: (labs, {feat: values})."""
    labs = np.concatenate([labels[h][valid[h]] for h in ("L", "R")]).astype(int)
    data = {k: np.concatenate([feats[k][h][valid[h]] for h in ("L", "R")])
            for k in feats}
    return labs, data


def _band_pct_by_group(labs, data, groups):
    """% of 5-band total power per group. *groups* = list of (name, code-set).

    Returns a DataFrame indexed by group name, columns = bands (percent)."""
    rows = {}
    for name, codes in groups:
        sel = np.isin(labs, list(codes))
        means = np.array([data[b][sel].mean() for b in BANDS])
        rows[name] = 100.0 * means / means.sum()
    return pd.DataFrame(rows, index=BANDS).T  # rows=group, cols=band


# --------------------------------------------------------------------------- #
# A. Band composition per type
# --------------------------------------------------------------------------- #


def figure_band_composition(labs, data, out_dir, dpi):
    groups = [(CYTO7_NAMES[c - 1], {c}) for c in range(1, 8)]
    pct = _band_pct_by_group(labs, data, groups)          # 7 types x 5 bands
    pct.to_csv(out_dir / "band_composition.csv")
    print(f"  wrote {out_dir / 'band_composition.csv'}")

    fig, ax = plt.subplots(figsize=(9.5, 6))
    fig.patch.set_facecolor("white")
    x = np.arange(len(pct.index))
    bottom = np.zeros(len(pct.index))
    for b in BANDS:
        ax.bar(x, pct[b].values, bottom=bottom, width=0.72, color=BAND_COLORS[b],
               edgecolor="white", linewidth=0.5, label=BAND_LABELS[b])
        bottom += pct[b].values
    ax.set_xticks(x)
    ax.set_xticklabels(pct.index, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("% of total MEG band power", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Frequency-band composition per cyto7 type (allocortex → koniocortex)",
                 fontsize=13, weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, fontsize=8.5,
              frameon=False)
    fig.text(0.5, -0.02,
             "Compositional, descriptive — individual bands do not survive the spin+FDR "
             "null (Annex F). 32k fs_LR, v6, allocortex included.",
             ha="center", va="top", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = out_dir / "band_composition_bars.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")
    return pct


# --------------------------------------------------------------------------- #
# B1. 3-tier surface map
# --------------------------------------------------------------------------- #


def figure_tier_surface(map_version, out_dir, dpi, sulc_strength=0.35):
    labels164 = resolve_target_map(map_version, "fsaverage")
    hkey = {"lh": "L", "rh": "R"}
    vnorm = {h: lighting_normals(h) for h in HEMIS}
    views = ["lateral", "medial"]
    fig, axes = plt.subplots(len(HEMIS), len(views), figsize=(len(views) * 3.4, len(HEMIS) * 3.4),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)
    for r, hemi in enumerate(HEMIS):
        lab = labels164[hkey[hemi]]
        coords, faces = load_surface(hemi, "inflated")
        rgb = np.tile(np.array(LIGHT_BLUE), (coords.shape[0], 1))
        for code, tier in TIER_OF.items():
            rgb[lab == code] = np.array(TIER3[tier])
        for c, view in enumerate(views):
            ax = axes[r, c]
            lit_panel(ax, coords, faces, rgb, hemi, view, vnorm[hemi])
            if r == 0:
                ax.set_title(view, fontsize=13)
            if c == 0:
                ax.text2D(-0.05, 0.5, hemi.upper(), transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=13, weight="bold")
    fig.legend(handles=tier3_legend_handles(), loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"cyto7 {map_version} — allocortex / mesocortex / isocortex tiers",
                 fontsize=15, y=0.98)
    fig.subplots_adjust(bottom=0.1, top=0.9)
    out = out_dir / "tier_grouping_surface.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# B2. Tier progression (mean +/- 95% CI) + 3-tier band composition
# --------------------------------------------------------------------------- #


def _tier_stats(labs, data):
    """Per-tier mean/95% CI for the progression features. Returns tidy DataFrame."""
    rows = []
    for tier in TIER_ORDER:
        codes = [c for c, t in TIER_OF.items() if t == tier]
        sel = np.isin(labs, codes)
        for key, name in PROG_FEATURES:
            v = data[key][sel]
            n = v.size
            mean = float(v.mean())
            # naive 95% CI (vertices are spatially autocorrelated -> optimistic;
            # descriptive only)
            ci = 1.96 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
            rows.append({"tier": tier, "feature": key, "feature_label": name,
                         "n": n, "mean": mean, "ci95": ci,
                         "ci_low": mean - ci, "ci_high": mean + ci})
    return pd.DataFrame(rows)


def figure_tier_progression(labs, data, out_dir, dpi):
    stats = _tier_stats(labs, data)
    tier_pct = _band_pct_by_group(
        labs, data, [(t, [c for c, tt in TIER_OF.items() if tt == t]) for t in TIER_ORDER])
    # combined CSV
    stats_out = stats.copy()
    stats.to_csv(out_dir / "tier_progression.csv", index=False)
    tier_pct.to_csv(out_dir / "tier_band_composition.csv")
    print(f"  wrote {out_dir / 'tier_progression.csv'} + tier_band_composition.csv")

    x = np.arange(len(TIER_ORDER))
    tier_colors = [TIER3[t] for t in TIER_ORDER]
    fig = plt.figure(figsize=(16, 4.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 5, wspace=0.42)
    for i, (key, name) in enumerate(PROG_FEATURES):
        ax = fig.add_subplot(gs[0, i])
        sub = stats[stats.feature == key].set_index("tier").loc[TIER_ORDER]
        ax.errorbar(x, sub["mean"], yerr=sub["ci95"], fmt="none", ecolor="0.4",
                    capsize=4, zorder=1)
        ax.scatter(x, sub["mean"], c=tier_colors, s=110, edgecolor="0.2", zorder=2)
        ax.plot(x, sub["mean"], color="0.5", lw=1.2, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(["allo", "meso", "iso"], fontsize=9)
        ax.set_title(name, fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.3)
    # 5th panel: 3-tier band composition (stacked)
    ax = fig.add_subplot(gs[0, 4])
    bottom = np.zeros(len(TIER_ORDER))
    for b in BANDS:
        ax.bar(x, tier_pct[b].values, bottom=bottom, width=0.7, color=BAND_COLORS[b],
               edgecolor="white", linewidth=0.5, label=BAND_LABELS[b])
        bottom += tier_pct[b].values
    ax.set_xticks(x)
    ax.set_xticklabels(["allo", "meso", "iso"], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% band power", fontsize=9)
    ax.set_title("Band composition", fontsize=11, weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=7,
              frameon=False)
    fig.suptitle("Structure-function progression collapsed to allo / meso / iso tiers "
                 "(mean ± 95% CI)", fontsize=14, weight="bold", y=1.02)
    fig.text(0.5, -0.08,
             "Descriptive; 95% CI is naive (cortical vertices are spatially "
             "autocorrelated, so intervals are optimistic). 32k fs_LR, v6, allocortex included.",
             ha="center", va="top", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = out_dir / "tier_progression.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")
    return stats_out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", default="v9")
    p.add_argument("--dataset", default="Validation210")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Target map: {args.map} (32k fs_LR); dataset {args.dataset}")
    labels = resolve_target_map(args.map, "fs_LR")
    feats = load_all_features(args.dataset)
    valid = build_validity_mask(labels, feats)
    labs, data = _pooled(labels, feats, valid)
    print(f"pooled valid vertices: {labs.size} "
          f"(per type: {[int((labs==c).sum()) for c in range(1,8)]})")

    print("== A. band composition per type ==")
    figure_band_composition(labs, data, args.out, args.dpi)
    print("== B1. 3-tier surface map ==")
    figure_tier_surface(args.map, args.out, args.dpi)
    print("== B2. tier progression ==")
    figure_tier_progression(labs, data, args.out, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
