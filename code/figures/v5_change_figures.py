"""v4 -> v5 change-history figures for the Miguel report (SPEC_v5_map.md).

Three deliverables (per hemisphere unless noted):

* ``figures/v9/change_history/v4_to_v5_changes_{lh,rh}.png`` -- per-vertex change
  map coloured by **reason** (cingulate_gradient / ring_closure / topology_buffer /
  speck_cleanup / boundary_smooth), medial + lateral, inflated + pial + flat, with
  a per-reason count bar.
* ``figures/v9/change_history/cingulate_rings_v5_{lh,rh}.png`` -- the closed
  **agranular inner ring + dysgranular outer ring** on the medial surface
  (inflated + pial). Headline for Miguel.
* ``figures/v9/change_history/boundary_before_after_v4_v5.png`` -- a representative
  cortical patch rendered on v4 vs v5 to show the gentle boundary de-jaggedness.

Run::  conda activate cyto7 && python scripts/v5_change_figures.py
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from cyto7_surface_io import REPO_ROOT, load_surface_geometry
from make_presentation_figures import (
    HEMIS, LIGHT_BLUE, TYPE_GREY, TYPE_NAMES, lighting_normals, lit_panel,
    load_surface, sulc_shading,
)
from change_history_figures import resample_164k_to_32k_nearest
import nibabel as nib

DERIVED = cfg.atlas_dir("fsaverage")
SURF_DIR = cfg.data_dir() / "fsaverage_surfaces"
OUT_DIR = cfg.results_dir("tables") / "change_history"

TYPE_CODES = [1, 2, 3, 4, 5, 6, 7]

#: reason -> (code, colour, label). Code order fixes the flat resample mapping.
REASONS = {
    "cingulate_gradient": (1, (0.20, 0.30, 0.78), "cingulate gradient (agr→dys)"),
    "ring_closure":       (2, (0.15, 0.65, 0.22), "ring closure (→agr)"),
    "topology_buffer":    (3, (0.15, 0.15, 0.15), "topology (strict R1)"),
    "speck_cleanup":      (4, (0.96, 0.55, 0.10), "speck cleanup (<30 v)"),
    "boundary_smooth":    (5, (0.72, 0.25, 0.78), "boundary smoothing"),
}
REASON_RGB = {v[0]: v[1] for v in REASONS.values()}
C_UNCHANGED = (0.85, 0.85, 0.85)

#: distinct per-type palette for the boundary-patch zoom (greyscale is too faint).
DISTINCT = {1: (0.85, 0.10, 0.55), 2: (0.30, 0.15, 0.55), 3: (0.20, 0.45, 0.80),
            4: (0.13, 0.63, 0.60), 5: (0.45, 0.80, 0.42), 6: (0.85, 0.78, 0.20),
            7: (0.98, 0.90, 0.15)}


def _annot(path):
    lab, _c, _n = nib.freesurfer.io.read_annot(str(path))
    return np.asarray(lab)


def _load(hemi, ver):
    return _annot(DERIVED / f"pial.{hemi}.cyto7.{ver}.annot")


def _reason_field(hemi: str, n: int) -> np.ndarray:
    """Per-vertex reason code (0 = unchanged) from the changelog."""
    field = np.zeros(n, int)
    with open(DERIVED / "v4_to_v5_changelog.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["hemi"] != hemi:
                continue
            code = REASONS.get(row["reason"], (0,))[0]
            field[int(row["vertex"])] = code
    return field


def _reason_rgb(field: np.ndarray, v4: np.ndarray) -> np.ndarray:
    rgb = np.tile(np.array(C_UNCHANGED), (field.shape[0], 1))
    rgb[v4 == 0] = LIGHT_BLUE                      # medial wall / unpainted
    for code, col in REASON_RGB.items():
        rgb[field == code] = col
    return rgb


def _type_rgb(lab: np.ndarray, palette=None) -> np.ndarray:
    palette = palette or TYPE_GREY
    rgb = np.tile(np.array(LIGHT_BLUE), (lab.shape[0], 1))
    for c in TYPE_CODES:
        rgb[lab == c] = np.array(palette[c])
    return rgb


# --------------------------------------------------------------------------- #
# Figure 1 -- change map coloured by reason
# --------------------------------------------------------------------------- #


def figure_change_by_reason(hemi: str, dpi: int, dataset: str) -> None:
    v4, v5 = _load(hemi, "v4"), _load(hemi, "v5")
    n = v4.shape[0]
    field = _reason_field(hemi, n)
    vnorm = lighting_normals(hemi)
    geom = {s: load_surface(hemi, s) for s in ("inflated", "pial")}
    ss = sulc_shading(hemi, 0.35)
    rgb = _reason_rgb(field, v4)

    # flat 32k
    H = "L" if hemi == "lh" else "R"
    field32 = resample_164k_to_32k_nearest(field, H)
    v4_32 = resample_164k_to_32k_nearest(v4, H)
    fcoords, ffaces = load_surface_geometry(dataset, H, "flat")
    rgb32 = _reason_rgb(field32, v4_32)

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1.15],
                          hspace=0.04, wspace=0.03)
    for r, view in enumerate(("medial", "lateral")):
        for c, surf in enumerate(("inflated", "pial")):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            coords, faces = geom[surf]
            lit_panel(ax, coords, faces, rgb, hemi, view, vnorm,
                      sulc_shade=(ss if surf == "pial" else None))
            if r == 0:
                ax.set_title(surf, fontsize=12)
            if c == 0:
                ax.text2D(-0.05, 0.5, view, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=12, weight="bold")
    # flat (row 0, col 2)
    axf = fig.add_subplot(gs[0, 2])
    tri = np.asarray(fcoords)[:, :2][ffaces]
    axf.add_collection(PolyCollection(tri, facecolors=rgb32[ffaces].mean(1),
                                      edgecolors="none", antialiased=False))
    axf.set_xlim(fcoords[:, 0].min(), fcoords[:, 0].max())
    axf.set_ylim(fcoords[:, 1].min(), fcoords[:, 1].max())
    axf.set_aspect("equal"); axf.axis("off")
    axf.set_title("flat (32k fs_LR)", fontsize=12)
    # reason count bar (row 1, col 2)
    axb = fig.add_subplot(gs[1, 2])
    counts = {name: int((field == code).sum()) for name, (code, *_ ) in REASONS.items()}
    names = list(counts.keys())
    ypos = np.arange(len(names))
    axb.barh(ypos, [counts[k] for k in names],
             color=[REASONS[k][1] for k in names], edgecolor="0.3")
    axb.set_yticks(ypos)
    axb.set_yticklabels([REASONS[k][2] for k in names], fontsize=8)
    for y, k in zip(ypos, names):
        axb.text(counts[k], y, f" {counts[k]}", va="center", fontsize=8)
    axb.set_xlabel("vertices changed", fontsize=9)
    axb.invert_yaxis(); axb.margins(x=0.18)
    axb.set_title(f"{hemi.upper()} changes by reason (total {int((field>0).sum())})",
                  fontsize=10)

    handles = [Patch(facecolor=REASONS[k][1], label=REASONS[k][2]) for k in REASONS]
    handles.append(Patch(facecolor=C_UNCHANGED, label="unchanged"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"cyto7 v4 → v5 changes by reason — {hemi.upper()}", fontsize=15, y=0.99)
    out = OUT_DIR / f"v4_to_v5_changes_{hemi}.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# Figure 2 -- cingulate rings (headline)
# --------------------------------------------------------------------------- #

RING_COLORS = {1: (0.55, 0.20, 0.65),    # allocortex core (purple)
               2: (0.95, 0.70, 0.10),    # agranular inner ring (gold)
               3: (0.15, 0.45, 0.80)}    # dysgranular outer ring (blue)
RING_FADE = (0.86, 0.86, 0.86)


def _ring_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = np.tile(np.array(RING_FADE), (lab.shape[0], 1))
    rgb[lab == 0] = LIGHT_BLUE
    for c, col in RING_COLORS.items():
        rgb[lab == c] = col
    return rgb


def figure_cingulate_rings(hemi: str, dpi: int) -> None:
    v5 = _load(hemi, "v5")
    vnorm = lighting_normals(hemi)
    rgb = _ring_rgb(v5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    for ax, surf in zip(axes, ("inflated", "pial")):
        coords, faces = load_surface(hemi, surf)
        ss = sulc_shading(hemi, 0.35) if surf == "pial" else None
        lit_panel(ax, coords, faces, rgb, hemi, "medial", vnorm, sulc_shade=ss)
        ax.set_title(f"{hemi.upper()} medial — {surf}", fontsize=12)
    handles = [Patch(facecolor=RING_COLORS[1], label="allocortex core"),
               Patch(facecolor=RING_COLORS[2], label="agranular inner ring"),
               Patch(facecolor=RING_COLORS[3], label="dysgranular outer ring"),
               Patch(facecolor=RING_FADE, label="eulaminate / other")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"cyto7 v5 — cingulate concentric rings ({hemi.upper()}): "
                 "agranular inner + dysgranular outer around the medial wall",
                 fontsize=13, y=1.0)
    fig.text(0.5, 0.92, "dysgranular = single encircling outer ring; agranular = "
             "near-complete inner ring (interrupted at the retrosplenial isthmus, "
             "flagged for GC).", ha="center", va="top", fontsize=8, color="0.35")
    fig.subplots_adjust(bottom=0.1, top=0.86)
    out = OUT_DIR / f"cingulate_rings_v5_{hemi}.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# Figure 3 -- boundary de-jaggedness before/after on a representative patch
# --------------------------------------------------------------------------- #


def _smooth_centroid(hemi: str, coords: np.ndarray) -> np.ndarray:
    """Centroid of the lateral-facing boundary_smooth vertices (patch centre)."""
    vs = []
    with open(DERIVED / "v4_to_v5_changelog.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["hemi"] == hemi and row["reason"] == "boundary_smooth":
                vs.append(int(row["vertex"]))
    vs = np.array(vs)
    cxmean = coords[:, 0].mean()
    lateral = vs[coords[vs, 0] < cxmean] if hemi == "lh" else vs[coords[vs, 0] > cxmean]
    use = lateral if lateral.size else vs
    return coords[use].mean(0)


def figure_boundary_before_after(hemi: str, dpi: int, radius: float = 26.0) -> None:
    coords, faces = load_surface(hemi, "inflated")
    vnorm = lighting_normals(hemi)
    centre = _smooth_centroid(hemi, coords)
    d = np.linalg.norm(coords - centre, axis=1)
    inball = d <= radius
    face_in = inball[faces].all(1)
    fpatch = faces[face_in]
    verts = np.unique(fpatch)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    for ax, ver in zip(axes, ("v4", "v5")):
        lab = _load(hemi, ver)
        rgb = _type_rgb(lab, DISTINCT)
        lit_panel(ax, coords, fpatch, rgb, hemi, "lateral", vnorm)
        # zoom to patch bbox
        mn, mx = coords[verts].min(0), coords[verts].max(0)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
        ax.set_box_aspect(mx - mn)
        ax.set_title(ver, fontsize=13, weight="bold")
    handles = [Patch(facecolor=DISTINCT[c], label=TYPE_NAMES[c - 1]) for c in TYPE_CODES]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"cyto7 boundary de-jaggedness ({hemi.upper()} lateral patch): "
                 "v4 (left) vs v5 (right)", fontsize=13, y=1.0)
    fig.subplots_adjust(bottom=0.12, top=0.9)
    out = OUT_DIR / f"boundary_before_after_v4_v5_{hemi}.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Validation210")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--which", choices=["all", "reason", "rings", "boundary"],
                   default="all")
    args = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for hemi in HEMIS:
        print(f"== {hemi} ==")
        if args.which in ("all", "reason"):
            figure_change_by_reason(hemi, args.dpi, args.dataset)
        if args.which in ("all", "rings"):
            figure_cingulate_rings(hemi, args.dpi)
        if args.which in ("all", "boundary"):
            figure_boundary_before_after(hemi, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
