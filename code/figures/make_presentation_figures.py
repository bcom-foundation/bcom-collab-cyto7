"""Presentation panels for the cyto7 map (reproduces the 18-Jun-2026 deck).

Layer-1 figures of ``docs/REFINE_allocortex.md``. Rebuilds the hand-made deck
figures from any cyto7 annot version so the deck can be refreshed automatically
after an edit (v1, v2, ...). Two figures, each on the **pial** and **inflated**
164k fsaverage surfaces:

* **Figure A - "Isolated labels" montage** (deck slide 8). A grid with the
  7 cyto7 types as columns and (view x hemisphere) as rows; in each cell only
  that one type is highlighted on an otherwise light-blue brain.
* **Figure B - "Comparison with paper"** (deck slides 6/7). The cyto7 map in
  **greyscale-by-type** (Garcia-Cabezas Fig. 8 convention: darker = less
  differentiated -> lighter = more), one figure per hemisphere, with a discrete
  type legend and an optional reference-paper image composited on the left.

Colour scheme (matches the PowerPoint): cortical types are greyscale; the base
brain / medial wall is light blue.

Rendering: a custom lit renderer (not nilearn's flat plot_surf). Each triangle
is shaded by Lambertian "headlight" lighting computed from its normal versus the
camera direction (so every view is lit) -- this gives the 3-D effect. No sulcal
texture is baked onto the surface: the label colours are clean flat fills,
modulated only by the smooth geometric lighting (so on pial the real folds show
through the lighting, and on inflated the surface is a clean lit form).

Surfaces load from ``resources/fsaverage_surfaces/{lh,rh}.{pial,inflated}``
(FreeSurfer-native, 163,842 verts/hemi), with a nilearn fallback.

Run::

    conda activate cyto7
    python scripts/make_presentation_figures.py --annot <path> --surface pial
    python scripts/make_presentation_figures.py --help

Usually driven by ``scripts/refresh_figures.py`` (loops surfaces/hemispheres).
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend (works under WSL/CI), as the other scripts do

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nibabel.freesurfer import read_geometry

from cyto7_surface_io import REPO_ROOT

# Colours from the central style module (single source of truth). The cyto7
# categorical palettes (GC greyscale default + viridis alternate + distinct) and the
# medial-wall base colour all live in figure_style.
import figure_style as fs
from figure_style import cyto7_palette
# Back-compat: GREY_COLORS was historically re-exported here via plot_contour_overlays.
from plot_contour_overlays import GREY_COLORS

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

FSAVG_SURF_DIR: Path = cfg.data_dir() / "fsaverage_surfaces"
V1_ANNOT_DIR: Path = cfg.atlas_dir("provenance/as_painted")
DERIVED_DIR: Path = cfg.atlas_dir("fsaverage")
DEFAULT_OUT: Path = cfg.results_dir("tables") / "surface"

#: cyto7 integer codes (1..7) and their canonical names, ascending laminar
#: differentiation. Code 0 = unlabelled / medial wall.
TYPE_CODES: list[int] = [1, 2, 3, 4, 5, 6, 7]
TYPE_NAMES: list[str] = [
    "Allocortex", "Agranular", "Dysgranular", "Eulaminate I",
    "Eulaminate II", "Eulaminate III", "Koniocortex",
]

#: Default cyto7-by-type RGB (codes 1..7): the GC greyscale-by-type ramp from
#: figure_style (== the annot colortable: Allocortex darkest -> koniocortex lightest).
#: Kept as TYPE_GREY for the many scripts that import it as the default palette.
TYPE_GREY: dict[int, tuple] = dict(fs.CYTO7_GC)

#: Light-blue base brain (medial wall + non-highlighted cortex), from figure_style.
LIGHT_BLUE: tuple = fs.LIGHT_BLUE

#: Distinct qualitative palette for isolated-single-type montages, from figure_style.
ISOLATED_DISTINCT: dict[int, tuple] = dict(fs.CYTO7_DISTINCT)

HEMIS = ("lh", "rh")
HEMI_NILEARN = {"lh": "left", "rh": "right"}

#: Camera (elev, azim) per (hemi, view). Matches the standard FreeSurfer views.
CAMERA: dict[tuple, tuple] = {
    ("lh", "lateral"): (0, 180), ("rh", "lateral"): (0, 0),
    ("lh", "medial"): (0, 0), ("rh", "medial"): (0, 180),
    ("lh", "ventral"): (-90, 180), ("rh", "ventral"): (-90, 0),
    ("lh", "dorsal"): (90, 180), ("rh", "dorsal"): (90, 0),
}

#: Lighting model (geometric only -- no sulcal texture baked onto the surface).
LIGHT_AMBIENT = 0.6
LIGHT_DIFFUSE = 0.5
#: Lighting always uses the *inflated* surface's normals (fold-free), lightly
#: smoothed, applied to whichever geometry is rendered. This way the pial keeps
#: its folded silhouette but its labels get clean, smooth shading with no
#: sulci-colouring (raw pial normals would shade every sulcal wall dark). A few
#: smoothing passes remove the inflated surface's residual bumps.
NORMAL_SMOOTH_ITERS = 30


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_surface(hemi: str, geometry: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a 164k fsaverage surface as ``(coords, faces)``.

    Prefers the local FreeSurfer-native file; falls back to nilearn's bundled
    fsaverage if absent.
    """
    local = FSAVG_SURF_DIR / f"{hemi}.{geometry}"
    if local.exists():
        coords, faces = read_geometry(str(local))
        return np.asarray(coords, float), np.asarray(faces, np.int64)
    local_gii = FSAVG_SURF_DIR / f"{hemi}.{geometry}.surf.gii"
    if local_gii.exists():
        from nilearn import surface
        coords, faces = surface.load_surf_mesh(str(local_gii))
        return np.asarray(coords, float), np.asarray(faces, np.int64)
    from nilearn import datasets, surface
    fa = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    key = {"pial": "pial", "inflated": "infl", "white": "white"}[geometry]
    coords, faces = surface.load_surf_mesh(fa[f"{key}_{HEMI_NILEARN[hemi]}"])
    return np.asarray(coords, float), np.asarray(faces, np.int64)


def load_labels(annot_path: Path, hemi: str) -> np.ndarray:
    """Per-vertex cyto7 codes (0..7) for one hemisphere from an annot file."""
    path = Path(str(annot_path).format(hemi=hemi, h=hemi))
    labels, _ctab, _names = nib.freesurfer.io.read_annot(str(path))
    return np.asarray(labels)


def resolve_annot_paths(annot: str) -> dict[str, Path]:
    """Resolve a CLI ``--annot`` value to {lh,rh} annot paths."""
    if "{hemi}" in annot or "{h}" in annot:
        return {h: Path(annot.format(hemi=h, h=h)) for h in HEMIS}
    p = Path(annot)
    name = p.name
    out = {}
    for h in HEMIS:
        other = "rh" if h == "lh" else "lh"
        if f".{h}." in name:
            out[h] = p
        elif f".{other}." in name:
            out[h] = p.with_name(name.replace(f".{other}.", f".{h}."))
        else:
            raise SystemExit(f"Cannot infer hemisphere from annot name: {name}")
    return out


def version_label(annot_paths: dict[str, Path]) -> str:
    """A short version tag inferred from the annot path (v1 / v2 / custom)."""
    s = str(annot_paths["lh"])
    if ".v2." in s:
        return "v2"
    if "cyto7_annot_files_standard_fsaverage" in s or ".cyto7.annot" in Path(s).name:
        return "v1"
    for tag in ("v3", "v4", "v5", "v6", "v7", "v8", "v9"):
        if f".{tag}." in s:
            return tag
    return "custom"


# --------------------------------------------------------------------------- #
# Lit surface renderer
# --------------------------------------------------------------------------- #


def smoothed_vertex_normals(coords: np.ndarray, faces: np.ndarray,
                            iters: int = NORMAL_SMOOTH_ITERS) -> np.ndarray:
    """Per-vertex normals, smoothed over the mesh graph.

    Area-weighted vertex normals are oriented outward, then low-pass filtered by
    repeated neighbour averaging. Smoothing removes fold-scale variation (so the
    pial surface no longer shades dark in every sulcus) while keeping the gross
    lobar form that gives the 3-D effect.
    """
    from scipy.sparse import coo_matrix

    fn = np.cross(coords[faces[:, 1]] - coords[faces[:, 0]],
                  coords[faces[:, 2]] - coords[faces[:, 0]])  # area-weighted
    vn = np.zeros_like(coords)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    # Orient outward globally (FreeSurfer winding is consistent).
    if np.einsum("ij,ij->i", vn, coords - coords.mean(0)).sum() < 0:
        vn = -vn
    if iters > 0:
        n = coords.shape[0]
        e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.vstack([e, e[:, ::-1]])
        A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        deg = np.asarray(A.sum(1)).ravel()
        deg[deg == 0] = 1
        for _ in range(iters):
            vn = A @ vn / deg[:, None]
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return vn


def lighting_normals(hemi: str) -> np.ndarray:
    """Smooth, fold-free per-vertex normals for lighting, from the *inflated*
    surface. Pial and inflated share vertex indices/topology, so these normals
    light either geometry without introducing fold-scale shading."""
    coords, faces = load_surface(hemi, "inflated")
    return smoothed_vertex_normals(coords, faces, iters=NORMAL_SMOOTH_ITERS)


#: Floor for the pial sulcal-depth shading factor (sulcal walls never darker).
SULC_SHADE_FLOOR = 0.55


def sulc_shading(hemi: str, strength: float, floor: float = SULC_SHADE_FLOOR) -> np.ndarray:
    """Per-vertex multiplicative shading factor from FreeSurfer sulcal depth.

    Adds gyral/sulcal contrast on the **pial** surface (the geometric headlight
    alone leaves it flat because lighting uses fold-free inflated normals). Uses
    FreeSurfer fsaverage ``sulc`` (via nilearn; standard-fsaverage 164k, same
    vertex ordering as the local surfaces). Rank-normalised per hemisphere so the
    factor is ``s = 1 - strength * depth`` (deep sulci darker), clipped to
    ``[floor, 1]`` so walls never crush to black. The sign is auto-oriented so
    gyral crowns are lighter than sulcal fundi.
    """
    from nilearn import datasets, surface
    fa = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    sulc = np.asarray(surface.load_surf_data(fa[f"sulc_{HEMI_NILEARN[hemi]}"]), float)
    r = sulc.argsort().argsort().astype(float) / (sulc.size - 1)  # rank in [0,1]
    # FreeSurfer sulc: positive = sulci (deeper). Orient so 'depth' is high in
    # sulci regardless of the stored sign (mean curvature of concave fundi > 0).
    depth = r  # higher sulc rank -> deeper sulcus -> darker
    s = 1.0 - strength * depth
    return np.clip(s, floor, 1.0)


def lit_panel(ax, coords, faces, vertex_rgb, hemi, view, vnormals, sulc_shade=None):
    """Render a triangulated surface with geometric headlight lighting only.

    No sulcal texture is applied: each label is a clean flat colour, modulated
    only by Lambertian lighting from a *smoothed* normal field (so fine folds do
    not darken the labels) versus the camera direction.

    Parameters
    ----------
    vertex_rgb : (N,3) float in [0,1]
        Per-vertex base colour (type colour or light-blue base).
    vnormals : (N,3) float
        Smoothed per-vertex normals (see :func:`smoothed_vertex_normals`).
    """
    elev, azim = CAMERA[(hemi, view)]
    tris = coords[faces]  # (F,3,3)

    # Per-face normal from the smoothed vertex normals (low-frequency shading).
    n = vnormals[faces].mean(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12

    # Headlight: light direction = camera direction (so each view is lit).
    e, a = np.radians(elev), np.radians(azim)
    light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    lambert = np.clip(n @ light, 0.0, 1.0)
    brightness = LIGHT_AMBIENT + LIGHT_DIFFUSE * lambert  # (F,)

    base = vertex_rgb[faces].mean(1)  # (F,3) face base colour
    shaded = np.clip(base * brightness[:, None], 0, 1)
    # Optional pial sulcal-depth modulation (gyri lighter than sulci); inflated
    # passes sulc_shade=None so it is unchanged.
    if sulc_shade is not None:
        face_s = sulc_shade[faces].mean(1)  # (F,)
        shaded = np.clip(shaded * face_s[:, None], 0, 1)
    rgba = np.concatenate([shaded, np.ones((shaded.shape[0], 1))], axis=1)

    pc = Poly3DCollection(tris, facecolors=rgba, edgecolors="none",
                          linewidths=0, antialiased=False)
    pc.set_rasterized(True)
    ax.add_collection3d(pc)

    mins, maxs = coords.min(0), coords.max(0)
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _save(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out_path}")


# --------------------------------------------------------------------------- #
# Figure A - isolated labels montage
# --------------------------------------------------------------------------- #


def figure_a_isolated(annot_paths, surface, out_dir, dpi, views, isolated_cmap,
                      sulc_strength=0.0):
    """7 type columns x (view x hemi) rows; one type highlighted per cell.

    *isolated_cmap* selects the highlight palette: 'gc' (greyscale ramp),
    'viridis', 'distinct' (default high-contrast), or legacy 'greyscale' == 'gc'.
    """
    tag = "gc" if isolated_cmap == "greyscale" else isolated_cmap
    print(f"  Figure A (isolated labels), {surface} [{tag}] ...")
    palette = cyto7_palette(tag)
    geom = {h: load_surface(h, surface) for h in HEMIS}
    vnorm = {h: lighting_normals(h) for h in HEMIS}  # fold-free (inflated normals)
    sulc = ({h: sulc_shading(h, sulc_strength) for h in HEMIS}
            if surface == "pial" and sulc_strength > 0 else {h: None for h in HEMIS})
    labels = {h: load_labels(annot_paths[h], h) for h in HEMIS}
    rows = [(v, h) for v in views for h in HEMIS]
    ncol, nrow = len(TYPE_CODES), len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.1, nrow * 2.1),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)
    for r, (view, hemi) in enumerate(rows):
        coords, faces = geom[hemi]
        lab = labels[hemi]
        for c, code in enumerate(TYPE_CODES):
            ax = axes[r, c]
            vrgb = np.tile(np.array(LIGHT_BLUE), (coords.shape[0], 1))
            vrgb[lab == code] = palette[code]
            lit_panel(ax, coords, faces, vrgb, hemi, view, vnorm[hemi], sulc_shade=sulc[hemi])
            if r == 0:
                ax.set_title(TYPE_NAMES[c], fontsize=11, pad=2)
            if c == 0:
                ax.text2D(-0.05, 0.5, f"{hemi.upper()} {view}", transform=ax.transAxes,
                          rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle("Isolated cyto7 labels  (each type highlighted in turn)",
                 fontsize=16, y=0.997)
    fig.text(0.5, 0.005,
             f"cyto7 {version_label(annot_paths)} | {surface} surface (164k fsaverage) "
             f"| {tag} palette", ha="center", va="bottom", fontsize=8, color="0.4")
    _save(fig, out_dir / f"isolated_labels_{surface}_{tag}.png", dpi)


# --------------------------------------------------------------------------- #
# Figure B - comparison with paper (greyscale-by-type)
# --------------------------------------------------------------------------- #


def figure_b_comparison(annot_paths, surface, out_dir, dpi, views, reference_img,
                        sulc_strength=0.0, palette="gc"):
    """cyto7-by-type map, one figure per hemisphere, with legend (palette-selectable)."""
    print(f"  Figure B (comparison with paper), {surface} [{palette}] ...")
    pal = cyto7_palette(palette)
    grey = {c: np.array(pal[c]) for c in TYPE_CODES}
    for hemi in HEMIS:
        coords, faces = load_surface(hemi, surface)
        vnorm = lighting_normals(hemi)  # fold-free (inflated normals)
        ss = (sulc_shading(hemi, sulc_strength)
              if surface == "pial" and sulc_strength > 0 else None)
        lab = load_labels(annot_paths[hemi], hemi)
        vrgb = np.tile(np.array(LIGHT_BLUE), (coords.shape[0], 1))  # medial wall = blue
        for c in TYPE_CODES:
            vrgb[lab == c] = grey[c]

        ncol = 1 + len(views)
        fig = plt.figure(figsize=(ncol * 3.0, 3.4))
        fig.patch.set_facecolor("white")

        ax_ref = fig.add_subplot(1, ncol, 1)
        ax_ref.axis("off")
        if reference_img and Path(reference_img).exists():
            ax_ref.imshow(mpimg.imread(str(reference_img)))
            ax_ref.set_title("reference (paper)", fontsize=10)
        else:
            ax_ref.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                           ls="--", ec="0.6"))
            ax_ref.text(0.5, 0.5, "reference paper panel\n(--reference-img)",
                        ha="center", va="center", fontsize=9, color="0.5")

        for i, view in enumerate(views):
            ax = fig.add_subplot(1, ncol, 2 + i, projection="3d")
            lit_panel(ax, coords, faces, vrgb, hemi, view, vnorm, sulc_shade=ss)
            ax.set_title(f"{hemi.upper()} {view}", fontsize=11)

        handles = [Patch(facecolor=pal[c], edgecolor="0.4", label=TYPE_NAMES[c - 1])
                   for c in TYPE_CODES]
        fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.5,
                   frameon=False, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f"cyto7 by-type vs paper - {hemi.upper()} ({palette})",
                     fontsize=14, y=1.04)
        fig.text(0.5, 0.93,
                 f"cyto7 {version_label(annot_paths)} | {surface} surface (164k fsaverage)"
                 f" | {palette} palette", ha="center", va="top", fontsize=8, color="0.4")
        _save(fig, out_dir / f"comparison_paper_{hemi.upper()}_{surface}_{palette}.png", dpi)


# --------------------------------------------------------------------------- #
# Combined inflated + pial composite (manuscript Fig 1)
# --------------------------------------------------------------------------- #


def figure_combined_inflated_pial(annot_paths, out_dir, dpi, sulc_strength,
                                  views=("lateral", "medial"), palette="gc"):
    """Manuscript Fig 1: cyto7-by-type atlas on inflated (top) + pial (bottom).

    Both hemispheres x lateral+medial (4 columns), inflated row over pial row,
    one shared type colour key. Inflated shows the clean type layout; pial shows
    it on the realistic folded surface with sulcal-depth shading. *palette* selects
    the cyto7 colours ('gc' default greyscale ramp, or 'viridis'); the output
    filename carries a ``_<palette>`` tag.
    """
    print(f"  Figure 1 combined (inflated + pial) composite [{palette}] ...")
    pal = cyto7_palette(palette)
    grey = {c: np.array(pal[c]) for c in TYPE_CODES}
    cols = [(h, v) for h in HEMIS for v in views]  # LH lat, LH med, RH lat, RH med
    surfaces = ["inflated", "pial"]
    vnorm = {h: lighting_normals(h) for h in HEMIS}
    labels = {h: load_labels(annot_paths[h], h) for h in HEMIS}
    sulc = {h: sulc_shading(h, sulc_strength) for h in HEMIS}

    fig, axes = plt.subplots(len(surfaces), len(cols),
                             figsize=(len(cols) * 3.0, len(surfaces) * 3.3),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)
    for r, surf in enumerate(surfaces):
        geom = {h: load_surface(h, surf) for h in HEMIS}
        for c, (hemi, view) in enumerate(cols):
            ax = axes[r, c]
            coords, faces = geom[hemi]
            vrgb = np.tile(np.array(LIGHT_BLUE), (coords.shape[0], 1))
            lab = labels[hemi]
            for code in TYPE_CODES:
                vrgb[lab == code] = grey[code]
            ss = sulc[hemi] if surf == "pial" else None
            lit_panel(ax, coords, faces, vrgb, hemi, view, vnorm[hemi], sulc_shade=ss)
            if r == 0:
                ax.set_title(f"{hemi.upper()} {view}", fontsize=13)
            if c == 0:
                ax.text2D(-0.06, 0.5, surf, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=13, weight="bold")
    handles = [Patch(facecolor=pal[c], edgecolor="0.4", label=TYPE_NAMES[c - 1])
               for c in TYPE_CODES]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"cyto7 atlas ({version_label(annot_paths)}, {palette} palette) — "
                 "inflated (top) & pial (bottom)", fontsize=16, y=0.99)
    fig.subplots_adjust(bottom=0.09, top=0.93, wspace=0.02, hspace=0.02)
    _save(fig, out_dir / f"comparison_paper_inflated_pial_{palette}.png", dpi)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def default_annot() -> str:
    v2 = DERIVED_DIR / "pial.lh.cyto7.v2.annot"
    if v2.exists():
        return str(DERIVED_DIR / "pial.{hemi}.cyto7.v2.annot")
    return str(V1_ANNOT_DIR / "pial.{hemi}.cyto7.annot")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--annot", default=default_annot(),
                   help="Annot path or {hemi} template. Default: v2 if present, else v1.")
    p.add_argument("--surface", choices=["pial", "inflated", "flat"], default="pial")
    p.add_argument("--which", choices=["isolated", "comparison", "both", "combined"],
                   default="both",
                   help="'combined' builds the inflated+pial composite (manuscript Fig 1); "
                        "ignores --surface.")
    p.add_argument("--sulc-shading", type=float, default=0.35,
                   help="Strength of pial sulcal-depth shading (0 disables; inflated unaffected).")
    p.add_argument("--palette", choices=["gc", "viridis", "both"], default="gc",
                   help="cyto7 categorical palette for the combined/comparison figures: "
                        "'gc' (default greyscale-by-type ramp), 'viridis', or 'both' "
                        "(renders each variant, filenames tagged _gc / _viridis).")
    p.add_argument("--isolated-cmap", choices=["greyscale", "gc", "viridis", "distinct"],
                   default="distinct",
                   help="Figure A (isolated montage) highlight palette; 'distinct' default.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--reference-img", default=None,
                   help="Optional reference-paper image to composite in Figure B.")
    p.add_argument("--no-ventral", action="store_true",
                   help="Use only lateral + medial views (default also adds ventral).")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    annot_paths = resolve_annot_paths(args.annot)
    for h in HEMIS:
        if not annot_paths[h].exists():
            raise SystemExit(f"annot not found: {annot_paths[h]}")
    args.out.mkdir(parents=True, exist_ok=True)

    palettes = ["gc", "viridis"] if args.palette == "both" else [args.palette]

    if args.which == "combined":
        print(f"Annot: {annot_paths['lh']} ({version_label(annot_paths)}); "
              f"combined inflated+pial composite; palettes={palettes}; "
              f"sulc-shading={args.sulc_shading}")
        for pal in palettes:
            figure_combined_inflated_pial(annot_paths, args.out, args.dpi,
                                          args.sulc_shading, palette=pal)
        print("Done.")
        return

    if args.surface == "flat":
        raise SystemExit(
            "flat is only defined on the 32k fs_LR mesh, not the 164k annot; "
            "it is not produced by this script (see refresh_figures.py --flat)."
        )
    views = ["lateral", "medial"] + ([] if args.no_ventral else ["ventral"])
    print(f"Annot: {annot_paths['lh']} ({version_label(annot_paths)}); "
          f"surface={args.surface}; views={views}; sulc-shading={args.sulc_shading}")
    if args.which in ("isolated", "both"):
        figure_a_isolated(annot_paths, args.surface, args.out, args.dpi, views,
                          args.isolated_cmap, sulc_strength=args.sulc_shading)
    if args.which in ("comparison", "both"):
        for pal in palettes:
            figure_b_comparison(annot_paths, args.surface, args.out, args.dpi, views,
                                args.reference_img, sulc_strength=args.sulc_shading,
                                palette=pal)
    print("Done.")


if __name__ == "__main__":
    main()
