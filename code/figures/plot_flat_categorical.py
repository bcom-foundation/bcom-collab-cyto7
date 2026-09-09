"""Flat categorical 7-type cyto7 map (SPEC_flat_and_3d_figures.md Part A).

Renders the categorical 7-type cyto7 map on the flat 32k fs_LR fsaverage surface,
both hemispheres, with the shared type key. Complements Figure 1. Colours come from
the central :mod:`figure_style` module; ``--palette`` selects gc / viridis / distinct
(``both`` renders the gc + viridis variants). Uses the released map (v6) via
``resolve_target_map``.

Run::  conda activate cyto7 && python scripts/plot_flat_categorical.py --map v6 --palette both
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from nilearn import plotting
from cyto7_surface_io import REPO_ROOT, resolve_target_map, surface_path
from figure_style import CYTO7_NAMES, CYTO7_CODES, cyto7_listed_cmap, cyto7_palette
import flat_display


def _render(labels, dataset, palette, out_base, dpi, map_tag):
    cmap = cyto7_listed_cmap(palette)
    pal = cyto7_palette(palette)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    for ax, (H, hemi) in zip(axes, (("L", "left"), ("R", "right"))):
        surf = str(surface_path(dataset, H, "flat"))
        lab = labels[H].astype(float)
        lab[labels[H] == 0] = np.nan                  # medial wall masked
        plotting.plot_surf_roi(surf, lab, hemi=hemi, view="dorsal", axes=ax,
                               cmap=cmap, vmin=1, vmax=7, colorbar=False)
        ax.set_title(f"{H}H flat", fontsize=15)
    handles = [Patch(facecolor=pal[c], label=CYTO7_NAMES[c - 1]) for c in CYTO7_CODES]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"cyto7 ({map_tag}, {palette} palette) — categorical 7-type map on the "
                 "flat fsaverage surface", fontsize=17, y=0.97)
    fig.subplots_adjust(bottom=0.08)
    out = out_base.with_name(f"{out_base.stem}_{palette}{out_base.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="v9")
    p.add_argument("--dataset", default="Validation210")
    p.add_argument("--palette", choices=["gc", "viridis", "distinct", "both"],
                   default="both", help="cyto7 palette (both = gc + viridis).")
    p.add_argument("--out", type=Path,
                   default=cfg.results_dir("tables") / "surface" / "cyto7_flat.png")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)

    # Display-only labels: identical to the released 32k resample except at the handful of
    # triple points the 164k->32k nearest-neighbour resample pinched out
    # (SPEC_flat_panel_median_recolor; see scripts/flat_display.py). NOTE: face colouring
    # here is nilearn's, and `plot_surf_roi` already defaults to avg_method="median", so the
    # majority-tie bug that affected the PolyCollection renderer never applied to this figure.
    labels = {H: flat_display.display_labels(H, version=args.map) for H in ("L", "R")}
    palettes = ["gc", "viridis"] if args.palette == "both" else [args.palette]
    for pal in palettes:
        _render(labels, args.dataset, pal, args.out, args.dpi, args.map)


if __name__ == "__main__":
    main()
