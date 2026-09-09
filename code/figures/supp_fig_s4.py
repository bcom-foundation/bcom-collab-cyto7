"""Supplementary Fig S4 + Table S3 — parcellation division (SPEC_figures_round2 D).

Shows that standard anatomical parcels are cytoarchitecturally mixed. Uses the existing crossed
composition tables (`figures/v9/crossed/{desikan,voneconomo}_x_cyto7_composition.csv`).

* **Table S4** — `table_s4_majority_type_lookup.csv`: parcel → majority cyto7 type + dominant %
  + #types ≥10% (the majority-type-per-parcel lookup for whole-brain models, §4.7). The two
  composition CSVs are Table S3's detailed content.
* **Supp Fig S4** — Desikan and von-Economo parcellations on the inflated surface coloured by each
  parcel's **majority cyto7 type** (viridis type palette), beside the composition matrices.
  Caption stats computed from the tables.

Run::  conda activate cyto7 && python scripts/supp_fig_s4.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from cyto7_surface_io import REPO_ROOT
from figure_style import CYTO7_VIRIDIS, CYTO7_NAMES
from labeled_areas_figure import proj, frontmask, read_annot, SURF

CROSSED = cfg.results_dir("tables") / "crossed"
OUTFIG = CROSSED.parent / "manuscript" / "supp_fig_S4_parcellation_division.png"
OUTTAB = CROSSED / "table_s4_majority_type_lookup.csv"
PCT = ["Allo_pct", "Agr_pct", "Dys_pct", "EulI_pct", "EulII_pct", "EulIII_pct", "Kon_pct"]
COL_LABELS = ["Allo", "Agr", "Dys", "EulI", "EulII", "EulIII", "Kon"]
ANNOT = {"desikan": str(cfg.data_dir() / "voneconomo" / "{h}.aparc.annot"),
         "voneconomo": str(cfg.data_dir() / "voneconomo" / "{h}.economo.annot")}
PAL = {**CYTO7_VIRIDIS, 0: (0.93, 0.93, 0.93)}


def _majority(df):
    """parcel -> (majority type code 1..7, dominant fraction, n_types>=10%)."""
    out = {}
    for _, r in df.iterrows():
        pcts = np.array([r[c] for c in PCT], float)
        out[r["anatomical_parcel"]] = (int(np.argmax(pcts)) + 1, pcts.max() / 100.0,
                                       int((pcts >= 10).sum()))
    return out


def _majority_map(atlas, maj):
    """Per-vertex majority-type code per hemi (0 where parcel not in the table)."""
    m = {}
    for H, h in (("L", "lh"), ("R", "rh")):
        lab, names = read_annot(ANNOT[atlas].format(h=h))
        code = np.zeros(lab.shape, int)
        for i, nm in enumerate(names):
            if nm in maj:
                code[lab == i] = maj[nm][0]
        m[H] = code
    return m


def _surf_panel(ax, h, view, code):
    coords, faces = nib.freesurfer.read_geometry(SURF.format(h=h, geom="inflated"))
    tri = coords[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    fr = frontmask(h, view, n[:, 0]); F = faces[fr]
    sx, sy = proj(h, view, coords)
    fl = code[F]
    fmaj = np.array([np.bincount(r[r > 0], minlength=8).argmax() if (r > 0).any() else 0 for r in fl])
    sh = 0.62 + 0.38 * np.clip(np.abs(n[fr, 2]), 0, 1)
    cols = np.clip(np.array([PAL[int(t)] for t in fmaj]) * sh[:, None], 0, 1)
    ax.add_collection(PolyCollection(np.stack([sx[F], sy[F]], -1), facecolors=cols,
                                     edgecolors="face", linewidths=0.5))
    ax.set_xlim(sx.min() - 3, sx.max() + 3); ax.set_ylim(sy.min() - 3, sy.max() + 3)
    ax.set_aspect("equal"); ax.axis("off")


def _draw_matrix(ax, df, atlas_tag, fs=1.0, annot_min=8.0):
    """Native (vector) composition heatmap drawn straight into the gridspec cell:
    rows = parcels (allo->konio staircase), cols = 7 cyto7 types, cell = % of the parcel's
    labelled cortex. Replaces the old rasterized-PNG paste so parcel names stay sharp at the
    figure DPI. ``fs`` scales every font so the panel keeps the same look at any figure size.
    Returns the AxesImage (for a colorbar)."""
    parcels = df["anatomical_parcel"].tolist()
    pct = df[PCT].to_numpy(float)
    ssum = pct.sum(1)
    dom = pct.argmax(1)
    wmean = (pct * np.arange(1, 8)).sum(1) / np.where(ssum > 0, ssum, 1)
    order = sorted(range(len(parcels)), key=lambda i: (dom[i], wmean[i]))
    parcels = [parcels[i] for i in order]
    pct = pct[order]
    nrow = len(parcels)
    im = ax.imshow(pct, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(7)); ax.set_xticklabels(COL_LABELS, fontsize=11 * fs)
    ax.set_yticks(range(nrow)); ax.set_yticklabels(parcels, fontsize=8 * fs)
    ax.set_xlabel("cyto7 type", fontsize=11 * fs)
    ax.tick_params(length=2 * fs, width=0.6 * fs, pad=1.5 * fs)
    for i in range(nrow):
        for j in range(7):
            v = pct[i, j]
            if v >= annot_min:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7 * fs,
                        color="white" if v < 60 else "black")
    ax.set_title(f"{atlas_tag} × cyto7 (v9) — composition per parcel (rows: allo→konio)",
                 fontsize=10 * fs, weight="bold", pad=3 * fs)
    return im


def main():
    tabs = {a: pd.read_csv(CROSSED / f"{a}_x_cyto7_composition.csv")
            for a in ("desikan", "voneconomo")}
    maj = {a: _majority(tabs[a]) for a in tabs}

    # ---- Table S3 lookup + stats ----
    rows, stats = [], {}
    for a, m in maj.items():
        for parc, (code, frac, ntypes) in m.items():
            rows.append({"parcellation": a, "parcel": parc,
                         "majority_cyto7_type": CYTO7_NAMES[code - 1],
                         "majority_type_code": code, "dominant_fraction": round(frac, 3),
                         "n_types_ge10pct": ntypes})
        fr = np.array([v[1] for v in m.values()]); nt = np.array([v[2] for v in m.values()])
        stats[a] = (len(m), float(np.median(fr)), float(nt.mean()), int((fr >= 0.90).sum()))
    pd.DataFrame(rows).to_csv(OUTTAB, index=False)
    print("wrote", OUTTAB)
    cap = ("Anatomical parcels are cytoarchitecturally mixed. "
           f"Desikan (n={stats['desikan'][0]}): median dominant-type fraction "
           f"{stats['desikan'][1]:.2f}, mean {stats['desikan'][2]:.1f} types/parcel (≥10%), "
           f"{stats['desikan'][3]}/{stats['desikan'][0]} parcels ≥90% one type. "
           f"von-Economo (n={stats['voneconomo'][0]}): median {stats['voneconomo'][1]:.2f}, "
           f"mean {stats['voneconomo'][2]:.1f} types/parcel, "
           f"{stats['voneconomo'][3]}/{stats['voneconomo'][0]} ≥90%.")
    print(cap)

    # ---- Supp Fig S4 ----
    # Layout per atlas row: a 2x2 surface block (LH lateral/medial on top, RH below), a wide
    # spacer, then the native composition matrix + colorbar. The spacer guarantees the long
    # parcel names (matrix y-ticks) never overlap the surface block. No title / no caption
    # (moved to the LaTeX caption). Exact 190 mm width at 4:3, 600 dpi — fonts scaled by ``s``
    # so the panel keeps the same look at the smaller physical size (crisp at 600 dpi).
    W_MM = 190.0
    w_in = W_MM / 25.4                 # 7.480 in
    h_in = w_in * (15.0 / 20.0)        # keep the 4:3 aspect ratio -> 142.5 mm
    s = 0.42                            # font scale for the reduced canvas
    fig = plt.figure(figsize=(w_in, h_in)); fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(2, 4, width_ratios=[1.0, 0.24, 1.05, 0.035], hspace=0.16,
                             wspace=0.05, left=0.07, right=0.965, top=0.955, bottom=0.075)
    views = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
    for r, a in enumerate(("desikan", "voneconomo")):
        cmap_v = _majority_map(a, maj[a])
        lab_ax = fig.add_subplot(outer[r, 0]); lab_ax.axis("off")
        lab_ax.text(-0.035, 0.5, a, transform=lab_ax.transAxes, rotation=90, va="center",
                    ha="center", fontsize=15 * s, weight="bold")
        sb = outer[r, 0].subgridspec(2, 2, hspace=0.14, wspace=0.02)
        for k, (h, view) in enumerate(views):
            ax = fig.add_subplot(sb[k // 2, k % 2])
            _surf_panel(ax, h, view, cmap_v["L" if h == "lh" else "R"])
            ax.set_title(f"{h.upper()} {view}", fontsize=11 * s, weight="bold", pad=2)
        axm = fig.add_subplot(outer[r, 2])
        im = _draw_matrix(axm, tabs[a], a, fs=s)
        cax = fig.add_subplot(outer[r, 3])
        cb = fig.colorbar(im, cax=cax); cb.set_label("% of parcel cortex", fontsize=10 * s)
        cb.ax.tick_params(labelsize=9 * s, length=2 * s, width=0.6 * s, pad=1.5 * s)
        cb.outline.set_linewidth(0.5 * s)
    handles = [Patch(facecolor=CYTO7_VIRIDIS[c], label=CYTO7_NAMES[c - 1]) for c in range(1, 8)]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=11 * s, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handlelength=1.2, handleheight=1.0,
               columnspacing=1.2, handletextpad=0.4)
    OUTFIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTFIG), dpi=600, facecolor="white")   # exact figsize -> 190 mm @ 600 dpi
    plt.close(fig)
    print("saved", OUTFIG)


if __name__ == "__main__":
    main()
