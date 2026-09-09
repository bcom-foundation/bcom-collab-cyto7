"""v8 -> v9 change figures (SPEC_adopt_v8.md Step 4).

``figures/v9/change_history/v8_to_v9_changes_{lh,rh}.png`` -- the hand-painted v8->v9
expert-consensus change map, coloured by **transition** (old->new type), medial + lateral
(inflated + pial) + flat, with a transition-count bar. Mirrors ``v6_change_figures.py``.

Run::  conda activate cyto7 && python scripts/v8_change_figures.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg  # noqa: F401  (puts sibling code dirs on sys.path)

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from cyto7_surface_io import load_surface_geometry
from make_presentation_figures import (
    HEMIS, LIGHT_BLUE, lighting_normals, lit_panel, load_surface, sulc_shading,
)
from change_history_figures import resample_164k_to_32k_nearest
from v5_change_figures import DERIVED, OUT_DIR, _load

C_UNCHANGED = (0.85, 0.85, 0.85)
C_OTHER = (0.45, 0.38, 0.30)

#: v8->v9 transition (old->new) -> (code, colour, label). Top transitions get a
#: distinct colour; everything else changed falls into "other change" (code 99).
TRANSITIONS = {
    ("agranular", "Allocortex"): (1, (0.60, 0.20, 0.65), "agranular → allocortex"),
}
TRANS_RGB = {v[0]: v[1] for v in TRANSITIONS.values()}
OTHER_CODE = 99


def _transition_field(hemi: str, n: int) -> np.ndarray:
    field = np.zeros(n, int)
    with open(DERIVED / "v8_to_v9_changelog.csv", newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("hemi,"):
                continue
            parts = line.strip().split(",")
            if len(parts) != 4 or parts[0] != hemi:
                continue
            _, vtx, old, new = parts
            code = TRANSITIONS.get((old, new), (OTHER_CODE,))[0]
            field[int(vtx)] = code
    return field


def _trans_rgb(field: np.ndarray, v8: np.ndarray) -> np.ndarray:
    rgb = np.tile(np.array(C_UNCHANGED), (field.shape[0], 1))
    rgb[v8 == 0] = LIGHT_BLUE
    for code, col in TRANS_RGB.items():
        rgb[field == code] = col
    rgb[field == OTHER_CODE] = C_OTHER
    return rgb


def figure_change(hemi: str, dpi: int, dataset: str) -> None:
    v8 = _load(hemi, "v8")
    n = v8.shape[0]
    field = _transition_field(hemi, n)
    vnorm = lighting_normals(hemi)
    geom = {s: load_surface(hemi, s) for s in ("inflated", "pial")}
    ss = sulc_shading(hemi, 0.35)
    rgb = _trans_rgb(field, v8)

    H = "L" if hemi == "lh" else "R"
    field32 = resample_164k_to_32k_nearest(field, H)
    v8_32 = resample_164k_to_32k_nearest(v8, H)
    fcoords, ffaces = load_surface_geometry(dataset, H, "flat")
    rgb32 = _trans_rgb(field32, v8_32)

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], hspace=0.04, wspace=0.03)
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
    axf = fig.add_subplot(gs[0, 2])
    tri = np.asarray(fcoords)[:, :2][ffaces]
    axf.add_collection(PolyCollection(tri, facecolors=rgb32[ffaces].mean(1),
                                      edgecolors="none", antialiased=False))
    axf.set_xlim(fcoords[:, 0].min(), fcoords[:, 0].max())
    axf.set_ylim(fcoords[:, 1].min(), fcoords[:, 1].max())
    axf.set_aspect("equal"); axf.axis("off"); axf.set_title("flat (32k fs_LR)", fontsize=12)

    axb = fig.add_subplot(gs[1, 2])
    counts = {lbl: int((field == code).sum()) for (code, _c, lbl) in TRANSITIONS.values()}
    counts["other change"] = int((field == OTHER_CODE).sum())
    names = list(counts.keys())
    ypos = np.arange(len(names))
    cols = [TRANSITIONS[k][1] for k in TRANSITIONS] + [C_OTHER]
    axb.barh(ypos, [counts[n_] for n_ in names], color=cols, edgecolor="0.3")
    axb.set_yticks(ypos); axb.set_yticklabels(names, fontsize=8)
    for y, n_ in zip(ypos, names):
        axb.text(counts[n_], y, f" {counts[n_]}", va="center", fontsize=8)
    axb.set_xlabel("vertices changed", fontsize=9); axb.invert_yaxis(); axb.margins(x=0.2)
    axb.set_title(f"{hemi.upper()} v8→v9 changes by transition (total {int((field>0).sum())})",
                  fontsize=10)

    handles = [Patch(facecolor=TRANSITIONS[k][1], label=TRANSITIONS[k][2]) for k in TRANSITIONS]
    handles += [Patch(facecolor=C_OTHER, label="other change"),
                Patch(facecolor=C_UNCHANGED, label="unchanged")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"cyto7 v8 → v9 L/R entorhinal even-up — {hemi.upper()} "
                 "(entorhinal/parahippocampal/temporal-pole/fusiform agranular → allocortex; LH only)",
                 fontsize=14, y=0.99)
    out = OUT_DIR / f"v8_to_v9_changes_{hemi}.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Validation210")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for hemi in HEMIS:
        print(f"== {hemi} ==")
        figure_change(hemi, args.dpi, args.dataset)
    print("Done.")


if __name__ == "__main__":
    main()
