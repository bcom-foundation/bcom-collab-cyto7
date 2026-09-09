"""Labeled cyto7 v9 reference figure (SPEC_labeled_areas_figure.md).

Miguel's request: label the key areas (V1/A1/S1/M1 + allo/meso) to make the map
understandable, across **all three geometries** (inflated, pial, flat) and **both
colorscales** (viridis, grayscale), in **labeled and unlabeled** variants. Flat map
included and labeled (his explicit ask).

Builds on ``labeled_areas_figure_proto.py`` (2D orthographic projection + back-face
culling + Desikan-centroid labels), extended to pial + flat and the full variant matrix.
Colours from ``figure_style`` (viridis + the new ``grayscale`` palette).

Output: ``figures/v9/surface/labeled_reference/`` (folder name kept figures/v9; v9 content):
  cyto7_v9_labeled_{inflated,pial,flat}_{viridis,grayscale}.png  (+ _nolabels)
  cyto7_v9_labeled_contactsheet_{viridis,grayscale}.png

Run::  conda activate cyto7 && python scripts/labeled_areas_figure.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg  # noqa: F401  (puts sibling code dirs on sys.path)

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from cyto7_surface_io import resolve_target_map, surface_path  # noqa: E402
from figure_style import CYTO7_NAMES, CYTO7_VIRIDIS, CYTO7_GRAYSCALE  # noqa: E402
import flat_display  # noqa: E402  (display-only flat-panel corrections)

VER = "v9"
DATASET = "Validation210"
SURF = str(REPO / "resources" / "fsaverage_surfaces" / "{h}.{geom}")
ANN = str(REPO / "resources" / "cyto7_derived" / f"pial.{{h}}.cyto7.{VER}.annot")
APARC = str(REPO / "resources" / "voneconomo" / "{h}.aparc.annot")
CACHE = REPO / "resources" / "cyto7_derived" / "cache"
OUT = REPO / "figures" / "v9" / "surface" / "labeled_reference"

LABEL_FX = [pe.withStroke(linewidth=2.4, foreground="black")]

#: palette dicts extended with a medial-wall colour at key 0.
PALETTES = {
    "viridis": {**CYTO7_VIRIDIS, 0: (0.93, 0.93, 0.93)},
    "grayscale": {**CYTO7_GRAYSCALE, 0: (1.0, 1.0, 1.0)},
}

#: von-Economo annot (for areas not in Desikan, e.g. TF).
ECON = str(REPO / "resources" / "voneconomo" / "{h}.economo.annot")

#: area -> (label, 3D view where front-facing, atlas source). Miguel's round-2 edits:
#: suffixes removed; pOFC/TF/TH added; hippocampus is OFF the fsaverage surface (annotated,
#: not faked). atlas ∈ {"desikan", "economo", "offsurface"}.
#: TH is NOT a separate von-Economo area; approximated by posterior Desikan `parahippocampal`
#: and labelled "TH" (stated in the report / manifest).
AREAS = {
    "precentral": ("M1", "lateral", "desikan"),
    "postcentral": ("S1", "lateral", "desikan"),
    "transversetemporal": ("A1", "lateral", "desikan"),
    "insula": ("Ins", "lateral", "desikan"),
    "pericalcarine": ("V1", "medial", "desikan"),
    "entorhinal": ("EC", "medial", "desikan"),
    "temporalpole": ("TP", "lateral", "desikan"),
    "isthmuscingulate": ("RSC", "medial", "desikan"),
    "posteriorcingulate": ("PCC", "medial", "desikan"),
    "caudalanteriorcingulate": ("ACC", "medial", "desikan"),
    "medialorbitofrontal": ("pOFC", "medial", "desikan"),   # posterior/medial OFC (agranular)
    "TF": ("TF", "lateral", "economo"),                     # von Economo TF (ventral temporal)
    "parahippocampal": ("TH", "medial", "desikan"),         # TH ≈ posterior parahippocampal
    "__hippocampus__": ("Hippocampus (off-surface)", "medial", "offsurface"),
}

#: parcel used to anchor the off-surface hippocampus annotation (adjacent on the sheet).
HIPPO_REF = "parahippocampal"

#: Anchor each label on the centroid of (parcel ∩ its expected cortical type) so the
#: text sits on the right colour even where the flat/pial projection stretches the parcel.
EXPECT = {
    "precentral": {5, 6}, "postcentral": {7}, "transversetemporal": {7},
    "pericalcarine": {7}, "entorhinal": {1}, "insula": {2, 3},
    "temporalpole": {2, 3}, "isthmuscingulate": {2, 3},
    "posteriorcingulate": {2, 3}, "caudalanteriorcingulate": {2, 3},
    "medialorbitofrontal": {1, 2}, "parahippocampal": {1, 2, 3}, "TF": {3, 4, 5},
}


def _anchor(mask, lab, parc, coords_or_pts):
    """Centroid of (parcel ∩ expected type), falling back to the whole parcel."""
    exp = EXPECT.get(parc)
    ma = mask & np.isin(lab, list(exp)) if exp else mask
    if ma.sum() < 5:
        ma = mask
    return coords_or_pts[ma].mean(0)


def _draw_labels(ax, arrays, cyto_lab, coords, project, view, min_verts):
    """Draw the area labels. *arrays* = {"desikan": (lab,names), "economo": (lab,names)};
    *project* maps a 3-D/flat centroid to (x, y) axis coords; *view* filters by front-face
    (None = flat, show all). Off-surface hippocampus is an italic annotation near the
    parahippocampal edge (clearly flagged, not a fabricated surface location)."""
    for key, (short, vw, atlas) in AREAS.items():
        if view is not None and vw != view:
            continue
        if atlas == "offsurface":
            ap, apn = arrays["desikan"]
            if HIPPO_REF not in apn:
                continue
            m = ap == apn.index(HIPPO_REF)
            if m.sum() < min_verts:
                continue
            x, y = project(_anchor(m, cyto_lab, HIPPO_REF, coords))
            ax.annotate(short, (x, y), xytext=(x, y - 9), fontsize=7, style="italic",
                        ha="center", va="center", color="white", zorder=11,
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.1)
                        ).set_path_effects(LABEL_FX)
            continue
        arr, names = arrays[atlas]
        if key not in names:
            continue
        m = arr == names.index(key)
        if m.sum() < min_verts:
            continue
        x, y = project(_anchor(m, cyto_lab, key, coords))
        ax.annotate(short, (x, y), fontsize=8.5, weight="bold", ha="center",
                    va="center", color="white", zorder=10).set_path_effects(LABEL_FX)


def read_annot(p):
    lab, _c, names = nib.freesurfer.io.read_annot(p)
    return np.asarray(lab), [n.decode() if isinstance(n, bytes) else n for n in names]


def proj(h, view, coords):
    y, z = coords[:, 1], coords[:, 2]
    if (h, view) in (("lh", "lateral"), ("rh", "medial")):
        return -y, z
    return y, z


def frontmask(h, view, nx):
    return nx < 0 if (h, view) in (("lh", "lateral"), ("rh", "medial")) else nx > 0


# --------------------------------------------------------------------------- #
# 3D geometries (inflated / pial)
# --------------------------------------------------------------------------- #


def panel_3d(ax, h, view, geom, pal, labels_on):
    coords, faces = nib.freesurfer.read_geometry(SURF.format(h=h, geom=geom))
    lab = read_annot(ANN.format(h=h))[0]
    ap, apn = read_annot(APARC.format(h=h))
    tri = coords[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    fr = frontmask(h, view, n[:, 0])
    F = faces[fr]
    sx, sy = proj(h, view, coords)
    verts = np.stack([sx[F], sy[F]], axis=-1)
    fl = lab[F]
    fmaj = np.array([np.bincount(r[r > 0], minlength=8).argmax() if (r > 0).any() else 0 for r in fl])
    sh = 0.62 + 0.38 * np.clip(np.abs(n[fr, 2]), 0, 1)   # depth cue (no sulc file for fsaverage)
    cols = np.clip(np.array([pal[int(t)] for t in fmaj]) * sh[:, None], 0, 1)
    # painter's algorithm: draw far->near along the view axis (x) so nearer folded
    # faces overwrite deeper ones. Without this the pial looks patchy/see-through
    # (deep sulcal + medial-wall faces bleed through); harmless on the smooth inflated.
    fx = coords[F].mean(axis=1)[:, 0]
    order = np.argsort(-fx) if (h, view) in (("lh", "lateral"), ("rh", "medial")) else np.argsort(fx)
    verts, cols = verts[order], cols[order]
    # edgecolors="face" + a small linewidth fills the sub-pixel seams between triangles,
    # so the folded pial reads as a solid opaque surface (no antialiased white bleed-through).
    ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors="face", linewidths=0.5))
    ax.set_xlim(sx.min() - 3, sx.max() + 3)
    ax.set_ylim(sy.min() - 3, sy.max() + 3)
    ax.set_aspect("equal"); ax.axis("off")
    if labels_on:
        ep, epn = read_annot(ECON.format(h=h))
        arrays = {"desikan": (ap, apn), "economo": (ep, epn)}

        def project(c):
            a, b = proj(h, view, c[None, :])
            return float(a[0]), float(b[0])
        _draw_labels(ax, arrays, lab, coords, project, view, 20)


def render_3d(geom, pal_name, labeled):
    pal = PALETTES[pal_name]
    cols_ = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
    fig = plt.figure(figsize=(20, 5.2)); fig.patch.set_facecolor("white")
    for ci, (h, view) in enumerate(cols_):
        ax = fig.add_subplot(1, 4, ci + 1)
        panel_3d(ax, h, view, geom, pal, labeled)
        ax.set_title(f"{h.upper()} {view}", fontsize=12, weight="bold")
    _legend_title(fig, pal, f"cyto7 {VER} — {geom} ({pal_name})"
                  + ("" if labeled else " — no labels"))
    return _save(fig, geom, pal_name, labeled)


# --------------------------------------------------------------------------- #
# Flat (32k fs_LR)
# --------------------------------------------------------------------------- #


def _annot_32k(H, kind):
    """Parcel codes (Desikan 'aparc' or von-Economo 'economo') resampled to fs_LR 32k
    (nearest), cached; returns (codes32k, names)."""
    src = APARC if kind == "aparc" else ECON
    cpath = CACHE / f"{kind}_fsLR32k_hemi-{H}.npy"
    hemi = "lh" if H == "L" else "rh"
    _lab, names = read_annot(src.format(h=hemi))
    if cpath.exists():
        return np.load(cpath), names
    import os
    from cyto7_surface_io import _WORKBENCH_DEFAULT
    wb = os.environ.get("WORKBENCH_BIN") or _WORKBENCH_DEFAULT
    if wb and Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")
    from neuromaps import transforms
    from nibabel.gifti import GiftiDataArray, GiftiImage
    gi = GiftiImage(); gi.add_gifti_data_array(GiftiDataArray(_lab.astype(np.float32)))
    res = transforms.fsaverage_to_fslr(gi, "32k", hemi=H, method="nearest")
    arr = np.rint(np.asarray(res[0].agg_data())).astype(np.int32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cpath, arr)
    return arr, names


def _aparc_32k(H):
    """Desikan parcel codes resampled to fs_LR 32k (nearest), cached; + names."""
    cpath = CACHE / f"aparc_fsLR32k_hemi-{H}.npy"
    hemi = "lh" if H == "L" else "rh"
    _lab, names = read_annot(APARC.format(h=hemi))
    if cpath.exists():
        return np.load(cpath), names
    import os
    from cyto7_surface_io import _WORKBENCH_DEFAULT
    wb = os.environ.get("WORKBENCH_BIN") or _WORKBENCH_DEFAULT
    if wb and Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")
    from neuromaps import transforms
    from nibabel.gifti import GiftiDataArray, GiftiImage
    gi = GiftiImage(); gi.add_gifti_data_array(GiftiDataArray(_lab.astype(np.float32)))
    res = transforms.fsaverage_to_fslr(gi, "32k", hemi=H, method="nearest")
    arr = np.rint(np.asarray(res[0].agg_data())).astype(np.int32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cpath, arr)
    return arr, names


def panel_flat(ax, H, pal, labels_on):
    """Flat panel (Fig. 1 panel e). Display-only corrections per
    SPEC_flat_panel_median_recolor: repaired display labels, **median** face colouring
    (not majority-with-lowest-tie, which manufactures |Δtype| ≥ 2 seams), and
    folded/degenerate faces dropped with the rest painted largest-first. The released
    164k annot and 32k label GIFTIs are not touched — see scripts/flat_display.py."""
    g = nib.load(str(surface_path(DATASET, H, "flat")))
    pts = np.asarray(g.darrays[0].data, float)
    faces = np.asarray(g.darrays[1].data, int)
    lab = flat_display.display_labels(H, faces, VER)
    idx = flat_display.ordered_faces(pts[:, :2], faces)
    fcode = flat_display.face_median(faces[idx], lab)
    verts = np.stack([pts[faces[idx]][:, :, 0], pts[faces[idx]][:, :, 1]], axis=-1)
    cols = np.array([pal[int(t)] for t in fcode])
    ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors="face", linewidths=0.5))
    ax.set_xlim(pts[:, 0].min() - 3, pts[:, 0].max() + 3)
    ax.set_ylim(pts[:, 1].min() - 3, pts[:, 1].max() + 3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{H}H flat", fontsize=12, weight="bold")
    if labels_on:
        arrays = {"desikan": _annot_32k(H, "aparc"), "economo": _annot_32k(H, "economo")}
        _draw_labels(ax, arrays, lab, pts, lambda c: (float(c[0]), float(c[1])), None, 10)


def render_flat(pal_name, labeled):
    pal = PALETTES[pal_name]
    fig = plt.figure(figsize=(13, 6)); fig.patch.set_facecolor("white")
    for ci, H in enumerate(("L", "R")):
        ax = fig.add_subplot(1, 2, ci + 1)
        panel_flat(ax, H, pal, labeled)
    _legend_title(fig, pal, f"cyto7 {VER} — flat 32k fs_LR ({pal_name})"
                  + ("" if labeled else " — no labels"))
    return _save(fig, "flat", pal_name, labeled)


# --------------------------------------------------------------------------- #
# shared legend / save / contact sheet
# --------------------------------------------------------------------------- #


def _legend_title(fig, pal, title):
    handles = [Patch(facecolor=pal[c], label=CYTO7_NAMES[c - 1]) for c in range(1, 8)]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=15, weight="bold", y=0.99)
    fig.subplots_adjust(top=0.9, bottom=0.12, wspace=0.02)


def _save(fig, geom, pal_name, labeled):
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if labeled else "_nolabels"
    p = OUT / f"cyto7_{VER}_labeled_{geom}_{pal_name}{suffix}.png"
    fig.savefig(str(p), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.name}")
    return p


def contact_sheet(pal_name, labeled_paths):
    """Stack the 3 labeled geometry PNGs into one contact sheet."""
    import matplotlib.image as mpimg
    imgs = [mpimg.imread(str(labeled_paths[g])) for g in ("inflated", "pial", "flat")]
    fig = plt.figure(figsize=(16, 14)); fig.patch.set_facecolor("white")
    for i, (g, img) in enumerate(zip(("inflated", "pial", "flat"), imgs)):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.imshow(img); ax.axis("off")
    fig.suptitle(f"cyto7 {VER} labeled reference — {pal_name} (inflated / pial / flat)",
                 fontsize=15, weight="bold", y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, hspace=0.03)
    p = OUT / f"cyto7_{VER}_labeled_contactsheet_{pal_name}.png"
    fig.savefig(str(p), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.name}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=["all", "flat"], default="all",
                   help="'flat' re-renders only the flat panels (inflated/pial PNGs on disk "
                        "are left byte-untouched) and rebuilds the contact sheets from the "
                        "existing ones. Used by SPEC_flat_panel_median_recolor, where only "
                        "the flat panel changes.")
    args = p.parse_args(argv)

    for pal_name in ("viridis", "grayscale"):
        print(f"== {pal_name} ==")
        labeled_paths = {}
        for geom in ("inflated", "pial"):
            if args.only == "flat":
                labeled_paths[geom] = OUT / f"cyto7_{VER}_labeled_{geom}_{pal_name}.png"
                print(f"  kept    {labeled_paths[geom].name} (--only flat)")
                continue
            labeled_paths[geom] = render_3d(geom, pal_name, True)
            render_3d(geom, pal_name, False)
        labeled_paths["flat"] = render_flat(pal_name, True)
        render_flat(pal_name, False)
        if all(p_.exists() for p_ in labeled_paths.values()):
            contact_sheet(pal_name, labeled_paths)
        else:
            missing = [p_.name for p_ in labeled_paths.values() if not p_.exists()]
            print(f"  [skip] contact sheet — missing {missing}")
    print("Done.")


if __name__ == "__main__":
    main()
