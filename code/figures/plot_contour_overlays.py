"""Reproduce the cytoarchitecture / myelin contour-overlay figures.

This script regenerates three figures that overlay the seven-class
cytoarchitectural map (the "cyto7" parcellation, hand-painted on the standard
fsaverage mesh and resampled to the 32k ``fs_LR`` mesh) as greyscale contour
lines on top of the Glasser et al. (2016) HCP T1w/T2w myelin map:

* ``contour_overlay_greyscale.png`` -- pial surface, lateral + medial views.
* ``contour_overlay_inflated.png``  -- inflated surface, lateral + medial views.
* ``contour_overlay_flat.png``      -- flattened surface, dorsal view.

All three figures share the same recipe: a vibrant HCP-style myelin map drawn
with :func:`nilearn.plotting.plot_surf_stat_map`, with the cyto7 class
boundaries drawn on top as greyscale contour lines via
:func:`nilearn.plotting.plot_surf_contours`. They differ only in the surface
geometry (pial / inflated / flat) they are rendered on.

The myelin scalar is read directly from the Glasser CIFTI dscalar file, so no
Connectome Workbench installation is required. CIFTI files only store the
~29.7k non-medial-wall vertices, so the values are scattered back onto the full
32,492-vertex surface (medial wall left at zero, which renders as the dark end
of the HCP colour scale, exactly as in the original figures).

Run from anywhere; paths are resolved relative to the repository root::

    conda activate cyto7
    python scripts/plot_contour_overlays.py

See ``--help`` for options (dataset choice, output directory, DPI, which
figures to render).
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed (works under WSL/CI)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from nilearn import plotting

from cyto7_surface_io import (
    LABEL_LEVELS,
    LABEL_NAMES,
    REPO_ROOT,
    load_cyto7_labels,
    load_myelin_on_surface,
    resolve_target_map,
    surface_path,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Colours now come from the central style module (single source of truth). The cyto7
# greyscale-by-type ramp (GC Fig. 8 convention) and the HCP myelin scale/range live in
# figure_style; kept under the historical names here for backward compatibility.
from figure_style import CYTO7_GC as _CYTO7_GC, HCP_CMAP, MYELIN_VMAX, MYELIN_VMIN

#: Greyscale shade per cyto7 class (darker = less differentiated), from figure_style.
GREY_COLORS: list[tuple[float, float, float]] = [_CYTO7_GC[c] for c in range(1, 8)]

#: Width of the cyto7 boundary contour lines.
CONTOUR_LINEWIDTH: float = 2.5

#: Colour-bar label and ticks for the shared T1w/T2w scale (see --colorbar).
COLORBAR_LABEL: str = "T1w/T2w ratio (intracortical myelin proxy)"
COLORBAR_TICKS: list[float] = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def _plot_hemi_view(
    surf_mesh: str,
    myelin: np.ndarray,
    labels: np.ndarray,
    hemi: str,
    view: str,
    ax: plt.Axes,
    title: str,
) -> None:
    """Render one hemisphere/view: myelin map plus cyto7 greyscale contours.

    Parameters
    ----------
    surf_mesh:
        Path to the surface GIFTI to render on.
    myelin:
        Per-vertex myelin (T1w/T2w) values (already on the full surface).
    labels:
        Per-vertex cyto7 class labels (0-7).
    hemi:
        ``"left"`` or ``"right"`` (nilearn convention).
    view:
        nilearn view name, e.g. ``"lateral"``, ``"medial"``, ``"dorsal"``.
    ax:
        A 3D matplotlib axis to draw into.
    title:
        Panel title.
    """
    myelin_clipped = np.clip(myelin, MYELIN_VMIN, MYELIN_VMAX)

    # Background: HCP-style myelin map.
    plotting.plot_surf_stat_map(
        surf_mesh,
        myelin_clipped,
        hemi=hemi,
        view=view,
        axes=ax,
        cmap=HCP_CMAP,
        vmax=MYELIN_VMAX,
        colorbar=False,
    )
    ax.set_title(title, color="white", pad=20, fontsize=16)

    # Foreground: cyto7 class boundaries as greyscale contours. Only draw the
    # classes actually present on this hemisphere so the greyscale colour map
    # lines up with the requested contour levels.
    present_levels = [lvl for lvl in LABEL_LEVELS if lvl in np.unique(labels)]
    if present_levels:
        present_greys = [GREY_COLORS[LABEL_LEVELS.index(lvl)] for lvl in present_levels]
        plotting.plot_surf_contours(
            surf_mesh,
            labels,
            levels=present_levels,
            axes=ax,
            cmap=ListedColormap(present_greys),
            linewidths=CONTOUR_LINEWIDTH,
        )


def _new_figure(rows: int, cols: int, figsize: tuple[float, float]):
    """Create a black-background figure of 3D axes for surface plotting."""
    fig, axes = plt.subplots(
        rows, cols, figsize=figsize, subplot_kw={"projection": "3d"}
    )
    fig.patch.set_facecolor("black")
    return fig, np.atleast_1d(axes)


def _add_myelin_colorbar(fig: plt.Figure) -> None:
    """Add one shared horizontal T1w/T2w colour bar at the bottom of *fig*.

    Built from a ``ScalarMappable`` over the *same* ``HCP_CMAP`` and clip window
    used to render the surfaces, so the bar matches the pixels. White text/ticks
    on the black background; ``extend='both'`` because myelin is clipped to
    ``[MYELIN_VMIN, MYELIN_VMAX]`` before plotting.
    """
    sm = ScalarMappable(norm=Normalize(MYELIN_VMIN, MYELIN_VMAX), cmap=HCP_CMAP)
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.02])  # reserved bottom strip
    cax.set_facecolor("black")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both",
                      ticks=COLORBAR_TICKS)
    cb.set_label(COLORBAR_LABEL, color="white", fontsize=14)
    cb.ax.xaxis.set_tick_params(color="white", labelcolor="white")
    plt.setp(cb.ax.get_xticklabels(), color="white")
    cb.outline.set_edgecolor("white")


def _add_contour_legend(fig: plt.Figure, labels: dict[str, np.ndarray]) -> None:
    """Add a compact legend mapping the greyscale contour shades to type names.

    Only types actually present on either hemisphere are shown.
    """
    present = [lvl for lvl in LABEL_LEVELS
              if lvl in np.unique(labels["L"]) or lvl in np.unique(labels["R"])]
    handles = [
        Line2D([0], [0], color=GREY_COLORS[LABEL_LEVELS.index(lvl)],
               lw=CONTOUR_LINEWIDTH, label=f"{lvl} {LABEL_NAMES[lvl - 1]}")
        for lvl in present
    ]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.10),
                     ncol=len(handles), frameon=False, fontsize=11,
                     labelcolor="white", handlelength=2.2, columnspacing=1.4)
    leg.set_title("cyto7 type boundaries", prop={"size": 11})
    if leg.get_title() is not None:
        leg.get_title().set_color("white")


def plot_pial_overlay(
    dataset: str,
    myelin: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    """Render the pial-surface overlay (``contour_overlay_greyscale.png``)."""
    print("Rendering pial-surface overlay (greyscale)...")
    fig, axes = _new_figure(2, 2, figsize=(18, 14))
    surf_l = str(surface_path(dataset, "L", "pial"))
    surf_r = str(surface_path(dataset, "R", "pial"))

    _plot_hemi_view(surf_l, myelin["L"], labels["L"], "left", "lateral", axes[0, 0], "LH Lateral")
    _plot_hemi_view(surf_l, myelin["L"], labels["L"], "left", "medial", axes[0, 1], "LH Medial")
    _plot_hemi_view(surf_r, myelin["R"], labels["R"], "right", "lateral", axes[1, 0], "RH Lateral")
    _plot_hemi_view(surf_r, myelin["R"], labels["R"], "right", "medial", axes[1, 1], "RH Medial")

    fig.suptitle(
        "Myelin Map with Greyscale Cytoarchitectural Boundaries",
        fontsize=22,
        y=0.95,
        color="white",
    )
    _save(fig, output_path, dpi)


def plot_inflated_overlay(
    dataset: str,
    myelin: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
    colorbar: bool = False,
) -> None:
    """Render the inflated-surface overlay (``contour_overlay_inflated.png``).

    With ``colorbar=True`` a single shared horizontal T1w/T2w colour bar and a
    greyscale-contour type legend are added at the bottom (panels are unchanged;
    per-panel bars stay off so only one shared scale is shown).
    """
    print("Rendering inflated-surface overlay...")
    fig, axes = _new_figure(2, 2, figsize=(18, 14))
    surf_l = str(surface_path(dataset, "L", "inflated"))
    surf_r = str(surface_path(dataset, "R", "inflated"))

    _plot_hemi_view(surf_l, myelin["L"], labels["L"], "left", "lateral", axes[0, 0], "LH Inflated Lateral")
    _plot_hemi_view(surf_l, myelin["L"], labels["L"], "left", "medial", axes[0, 1], "LH Inflated Medial")
    _plot_hemi_view(surf_r, myelin["R"], labels["R"], "right", "lateral", axes[1, 0], "RH Inflated Lateral")
    _plot_hemi_view(surf_r, myelin["R"], labels["R"], "right", "medial", axes[1, 1], "RH Inflated Medial")

    fig.suptitle(
        "Myelin Map & Cytoarchitecture (Inflated)", fontsize=22, y=0.95, color="white"
    )
    if colorbar:
        # Reserve bottom space so the bar/legend never overlap the panels.
        fig.subplots_adjust(bottom=0.16)
        _add_contour_legend(fig, labels)
        _add_myelin_colorbar(fig)
    _save(fig, output_path, dpi)


def plot_flat_overlay(
    dataset: str,
    myelin: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    """Render the flat-surface overlay (``contour_overlay_flat.png``).

    Flat maps are viewed from ``"dorsal"`` so the camera looks straight down at
    the 2D plane.
    """
    print("Rendering flat-surface overlay...")
    fig, axes = _new_figure(1, 2, figsize=(20, 10))
    surf_l = str(surface_path(dataset, "L", "flat"))
    surf_r = str(surface_path(dataset, "R", "flat"))

    _plot_hemi_view(surf_l, myelin["L"], labels["L"], "left", "dorsal", axes[0], "LH Flat Projection")
    _plot_hemi_view(surf_r, myelin["R"], labels["R"], "right", "dorsal", axes[1], "RH Flat Projection")

    fig.suptitle(
        "Myelin Map & Cytoarchitecture (Flat)", fontsize=22, y=0.95, color="white"
    )
    _save(fig, output_path, dpi)


def _save(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    """Save *fig* to *output_path* with a black background and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  saved {output_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

#: Maps the ``--figures`` choices to (filename, rendering function).
FIGURE_BUILDERS = {
    "greyscale": ("contour_overlay_greyscale.png", plot_pial_overlay),
    "inflated": ("contour_overlay_inflated.png", plot_inflated_overlay),
    "flat": ("contour_overlay_flat.png", plot_flat_overlay),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        choices=["Validation210", "Parcellation210"],
        default="Validation210",
        help="Which Glasser HCP group-average dataset to use for surfaces and "
        "the myelin map (default: %(default)s, matching the cyto7 labels).",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=list(FIGURE_BUILDERS) + ["all"],
        default=["all"],
        help="Which figures to render (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.figures_dir(),
        help="Directory to write the PNG files to (default: <repo>/figures).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output resolution (default: %(default)s)."
    )
    parser.add_argument(
        "--annot-version", default=None,
        help="cyto7 target-map version (v1|v3|...) or {hemi}-path; resolved 164k->32k nearest.",
    )
    parser.add_argument("--out-suffix", default="", help="Suffix appended to output filenames.")
    parser.add_argument(
        "--colorbar", action=argparse.BooleanOptionalAction, default=False,
        help="Render ONLY the inflated overlay with a shared T1w/T2w colour bar + "
        "type-contour legend, to '<stem><out-suffix>_colorbar.png' (existing figures untouched).",
    )
    parser.add_argument(
        "--map", default=None,
        help="cyto7 map for the colour-bar output (v1|v3|v3_clean|{hemi}-path). "
        "Defaults to v3_clean when --colorbar is set; falls back to --annot-version otherwise.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the data once and render the requested figures."""
    args = parse_args(argv)

    # Map selection: --map wins; else --annot-version; else bundled 32k (v1).
    # When --colorbar is set and neither is given, default to the released v3_clean.
    map_arg = args.map or args.annot_version
    if args.colorbar and map_arg is None:
        map_arg = "v3_clean"

    print(f"Dataset: {args.dataset}")
    print("Loading cyto7 labels and myelin map...")
    if map_arg:
        print(f"Target map: {map_arg} (resolved 164k->32k nearest)")
        labels = resolve_target_map(map_arg, "fs_LR")
        map_used = map_arg
    else:
        labels = {"L": load_cyto7_labels("lh"), "R": load_cyto7_labels("rh")}
        map_used = "bundled 32k fs_LR (v1)"
    myelin = load_myelin_on_surface(args.dataset)
    print(f"Map used: {map_used}; myelin display range [{MYELIN_VMIN}, {MYELIN_VMAX}] (clipped)")

    if args.colorbar:
        # Only the inflated overlay gets the shared bar; distinct output filename
        # so the existing contour_overlay_inflated*.png is never overwritten.
        stem = Path("contour_overlay_inflated.png")
        out_name = f"{stem.stem}{args.out_suffix}_colorbar{stem.suffix}"
        plot_inflated_overlay(args.dataset, myelin, labels,
                              args.output_dir / out_name, args.dpi, colorbar=True)
        print("Done.")
        return

    selected = list(FIGURE_BUILDERS) if "all" in args.figures else args.figures
    for key in selected:
        filename, builder = FIGURE_BUILDERS[key]
        stem = Path(filename)
        out_name = f"{stem.stem}{args.out_suffix}{stem.suffix}"
        builder(args.dataset, myelin, labels, args.output_dir / out_name, args.dpi)

    print("Done.")


if __name__ == "__main__":
    main()
