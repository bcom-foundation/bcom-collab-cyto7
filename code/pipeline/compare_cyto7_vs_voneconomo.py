"""Vertex-wise comparison of cyto7 vs von-Economo-derived cortical types.

Implements ``COMPARE_cyto7_vs_voneconomo.md``: it compares, vertex by vertex on
the standard ``fsaverage`` mesh, the hand-painted 7-class cytoarchitectural map
(**cyto7**) against the cortical-type map the field derives by re-labelling
von Economo-Koskinas areas (Scholtens et al., 2018) with Garcia-Cabezas /
Barbas cortical types (Garcia-Cabezas et al., 2020). It then surfaces ranked,
anatomically-labelled **candidate regions for reviewing the hand-painting**.

Framing (respected throughout): *disagreement is not error.* The von-Economo
map's boundaries are coarse areal edges; cyto7 is drawn at vertex resolution, so
much disagreement is legitimate refinement. Only large, interior,
high-ordinal-gap, independently-discordant patches are genuine review
candidates. The final call belongs to the human authors.

Inputs (already in the repo):
  * cyto7 annots, standard fsaverage:
      resources/cyto7_annot_files_standard_fsaverage/pial.{lh,rh}.cyto7.annot
  * von Economo areal annots on fsaverage (made by make_economo_annot_fsaverage.sh):
      resources/voneconomo/{lh,rh}.economo.annot
  * area -> type lookup (made by fetch_voneconomo_atlas.py):
      resources/voneconomo/von_economo_cortical_types.csv
  * Desikan aparc (for anatomical labels): resources/voneconomo/{lh,rh}.aparc.annot

Outputs (--output-dir, default figures/comparison/):
  confusion_matrix.{csv,png}, agreement_summary.{csv,txt},
  sidebyside_surface.png, difference_map.png, delta_ordinal_signed.png,
  cyto7_allocortex_coverage.png, review_candidates.{csv,png}, REPORT.md

Run::

    conda activate cyto7
    python scripts/compare_cyto7_vs_voneconomo.py            # headline: fsaverage
    python scripts/compare_cyto7_vs_voneconomo.py --mesh fsaverage5
    python scripts/compare_cyto7_vs_voneconomo.py --help
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

from cyto7_surface_io import REPO_ROOT, resolve_target_map
from figure_style import CYTO7_VIRIDIS, SIGNED_DELTA  # centralized palette + signed-Δ scale

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

VONECONOMO_DIR: Path = cfg.data_dir() / "voneconomo"
CYTO7_STD_DIR: Path = cfg.atlas_dir("provenance/as_painted")

#: The six isocortical types, ordered by laminar differentiation. The cyto7
#: integer code is index+2 (agranular=2 ... koniocortex=7); the Garcia-Cabezas
#: *ordinal* used for |Delta| is index 0..5 (agranular=0 ... koniocortex=5).
TYPE_NAMES: list[str] = [
    "agranular", "dysgranular", "eulaminate I",
    "eulaminate II", "eulaminate III", "koniocortex",
]
TYPE_CODES: list[int] = [2, 3, 4, 5, 6, 7]  # cyto7 codes for the six types
CODE_TO_ORDINAL: dict[int, int] = {c: i for i, c in enumerate(TYPE_CODES)}
CODE_TO_NAME: dict[int, str] = dict(zip(TYPE_CODES, TYPE_NAMES))

#: Discrete colours for the six types (dark = least differentiated -> bright =
#: koniocortex), shared by the cyto7 and von-Economo surface panels. Sourced from
#: figure_style.CYTO7_VIRIDIS (codes 2-7) so Fig 2 stays colour-coded (as before) but
#: from the single source of truth; allocortex (code 1) is excluded from this benchmark.
TYPE_COLORS: dict[int, tuple] = {c: CYTO7_VIRIDIS[c] for c in (2, 3, 4, 5, 6, 7)}
ALLOCORTEX_CODE: int = 1

#: Difference-map categories (Step 4b).
DIFF_AGREE = 0
DIFF_CYTO7_MORE = 1   # off-by-one, cyto7 more differentiated
DIFF_CYTO7_LESS = 2   # off-by-one, cyto7 less differentiated
DIFF_GAP2 = 3         # off-by >= 2 (either direction)
DIFF_COLORS = {
    DIFF_AGREE: (0.82, 0.82, 0.82),
    DIFF_CYTO7_MORE: (0.84, 0.19, 0.15),   # red  = cyto7 more differentiated
    DIFF_CYTO7_LESS: (0.13, 0.40, 0.67),   # blue = cyto7 less differentiated
    DIFF_GAP2: (0.0, 0.0, 0.0),            # black = large gap
}
DIFF_LABELS = {
    DIFF_AGREE: "agree",
    DIFF_CYTO7_MORE: "off-by-one (cyto7 more differentiated)",
    DIFF_CYTO7_LESS: "off-by-one (cyto7 less differentiated)",
    DIFF_GAP2: "off-by >=2",
}

HEMIS = ("lh", "rh")
HEMI_NILEARN = {"lh": "left", "rh": "right"}


# --------------------------------------------------------------------------- #
# Loading & alignment (Step 1)
# --------------------------------------------------------------------------- #


def _read_annot(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read a FreeSurfer .annot -> (per-vertex structure index, name list)."""
    import nibabel as nib

    labels, _ctab, names = nib.freesurfer.io.read_annot(str(path))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return np.asarray(labels), names


def load_cyto7(hemi: str) -> np.ndarray:
    """cyto7 per-vertex codes on standard fsaverage (0=medial,1=allo,2-7=types)."""
    labels, _names = _read_annot(CYTO7_STD_DIR / f"pial.{hemi}.cyto7.annot")
    return labels


def load_economo_type_vector(hemi: str, lookup: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map the von Economo areal annot to a per-vertex cyto7-coded type vector.

    Returns ``(ve_code, area_idx)`` where ``ve_code`` is the von-Economo-derived
    type as a cyto7 code (2-7), or 0 where the area has no isocortical 6-type
    (medial wall / corpus callosum / excluded limbic areas); ``area_idx`` is the
    raw von Economo area structure index per vertex (for boundary detection).
    """
    labels, names = _read_annot(VONECONOMO_DIR / f"{hemi}.economo.annot")
    acr_to_code = {
        r.economo_acronym: int(r.cyto7_code)
        for r in lookup.itertuples()
        if pd.notna(r.cyto7_code) and r.cyto7_code != ""
    }
    name_arr = np.asarray(names, dtype=object)
    acr_per_vertex = name_arr[labels]
    ve_code = np.zeros(labels.shape[0], dtype=int)
    for acr, code in acr_to_code.items():
        ve_code[acr_per_vertex == acr] = code
    return ve_code, labels


def common_support_mask(cyto7: np.ndarray, ve_code: np.ndarray) -> np.ndarray:
    """Vertices kept for agreement metrics: cyto7 in 2-7 AND vE has a 6-type."""
    return np.isin(cyto7, TYPE_CODES) & np.isin(ve_code, TYPE_CODES)


# --------------------------------------------------------------------------- #
# Confusion matrix & agreement statistics (Steps 2 & 3)
# --------------------------------------------------------------------------- #


def confusion_matrix(cyto7: np.ndarray, ve: np.ndarray) -> np.ndarray:
    """6x6 confusion counts; rows = cyto7 type, cols = von-Economo type."""
    mat = np.zeros((6, 6), dtype=int)
    for i, rc in enumerate(TYPE_CODES):
        for j, cc in enumerate(TYPE_CODES):
            mat[i, j] = int(np.sum((cyto7 == rc) & (ve == cc)))
    return mat


def agreement_stats(cyto7: np.ndarray, ve: np.ndarray, label: str) -> dict:
    """Descriptive agreement statistics over a set of common-support vertices."""
    n = cyto7.size
    ord_c = np.array([CODE_TO_ORDINAL[c] for c in cyto7])
    ord_v = np.array([CODE_TO_ORDINAL[c] for c in ve])
    delta = ord_c - ord_v  # signed: + => cyto7 more differentiated

    mat = confusion_matrix(cyto7, ve)
    diag = np.diag(mat).sum()
    stats = {
        "set": label,
        "n_vertices": n,
        "overall_agreement": diag / n if n else np.nan,
        "cohen_kappa": cohen_kappa_score(cyto7, ve, labels=TYPE_CODES),
        "weighted_kappa_quadratic": cohen_kappa_score(
            cyto7, ve, labels=TYPE_CODES, weights="quadratic"
        ),
        "adjusted_rand_index": adjusted_rand_score(cyto7, ve),
        "mean_abs_delta_ordinal": float(np.mean(np.abs(delta))),
        "mean_signed_delta_ordinal": float(np.mean(delta)),
    }
    # Per-type Dice / IoU + per-type ordinal deltas.
    for i, code in enumerate(TYPE_CODES):
        name = CODE_TO_NAME[code]
        row, col, tp = mat[i, :].sum(), mat[:, i].sum(), mat[i, i]
        dice = 2 * tp / (row + col) if (row + col) else np.nan
        iou = tp / (row + col - tp) if (row + col - tp) else np.nan
        in_c = cyto7 == code
        stats[f"dice__{name}"] = dice
        stats[f"iou__{name}"] = iou
        stats[f"mean_abs_delta__{name}"] = (
            float(np.mean(np.abs(delta[in_c]))) if in_c.any() else np.nan
        )
        stats[f"mean_signed_delta__{name}"] = (
            float(np.mean(delta[in_c])) if in_c.any() else np.nan
        )
    return stats


# --------------------------------------------------------------------------- #
# Mesh geometry helpers (for figures & clustering)
# --------------------------------------------------------------------------- #


def load_geometry(mesh: str) -> dict:
    """Fetch fsaverage(5) surfaces, per-vertex areas and sulc shading via nilearn."""
    from nilearn import datasets, surface

    fa = datasets.fetch_surf_fsaverage(mesh=mesh)
    geom = {}
    for hemi in HEMIS:
        side = HEMI_NILEARN[hemi]
        coords, faces = surface.load_surf_mesh(fa[f"pial_{side}"])
        geom[hemi] = {
            "infl": fa[f"infl_{side}"],
            "coords": coords,
            "faces": faces,
            "area": np.asarray(surface.load_surf_data(fa[f"area_{side}"]), dtype=float),
            "sulc": fa[f"sulc_{side}"],
        }
    return geom


def adjacency_graph(coords: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """Symmetric edge-length-weighted vertex adjacency from triangle faces."""
    n = coords.shape[0]
    src = faces[:, [0, 1, 2]].ravel()
    dst = faces[:, [1, 2, 0]].ravel()
    w = np.linalg.norm(coords[src] - coords[dst], axis=1)
    g = csr_matrix((w, (src, dst)), shape=(n, n))
    return g.maximum(g.T)  # symmetrise


def distance_to_area_boundary(graph: csr_matrix, area_idx: np.ndarray) -> np.ndarray:
    """Geodesic (graph) distance of every vertex to the nearest von Economo
    areal edge. Boundary vertices are those with a neighbour in a different
    (cortical) area; distance 0 there, growing into area interiors."""
    g = graph.tocoo()
    diff_area = area_idx[g.row] != area_idx[g.col]
    seeds = np.unique(np.concatenate([g.row[diff_area], g.col[diff_area]]))
    if seeds.size == 0:
        return np.full(area_idx.shape[0], np.inf)
    return dijkstra(graph, directed=False, indices=seeds, min_only=True)


# --------------------------------------------------------------------------- #
# Surface rendering (Step 4)
# --------------------------------------------------------------------------- #


def _surf_panel(ax, infl, data, hemi, view, sulc, cmap, vmin, vmax, bg_on_data=True):
    from nilearn import plotting

    plotting.plot_surf_roi(
        infl, roi_map=data, hemi=HEMI_NILEARN[hemi], view=view, axes=ax,
        cmap=cmap, vmin=vmin, vmax=vmax, bg_map=sulc, bg_on_data=bg_on_data,
        colorbar=False,
    )


def _discrete_cmap(codes: list[int], color_map: dict) -> ListedColormap:
    return ListedColormap([color_map[c] for c in codes])


def render_sidebyside(geom, cyto7, ve, out_path, dpi):
    """(a) cyto7 vs von-Economo-derived type, same 6-colour discrete scale."""
    print("  rendering sidebyside_surface.png ...")
    cmap = _discrete_cmap(TYPE_CODES, TYPE_COLORS)
    fig, axes = plt.subplots(4, 2, figsize=(13, 20),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    col_data = {0: cyto7, 1: ve}
    col_title = {0: "cyto7 (hand-painted)", 1: "von-Economo-derived (Garcia-Cabezas)"}
    rows = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
    for r, (hemi, view) in enumerate(rows):
        for c in range(2):
            d = col_data[c][hemi].astype(float).copy()
            d[~np.isin(col_data[c][hemi], TYPE_CODES)] = np.nan
            # center each integer code on its own colour bin (robust discrete
            # mapping; codes 2-7 -> the 6 distinct ListedColormap entries)
            _surf_panel(axes[r, c], geom[hemi]["infl"], d, hemi, view,
                        geom[hemi]["sulc"], cmap, vmin=1.5, vmax=7.5)
            if r == 0:
                axes[r, c].set_title(col_title[c], fontsize=15, pad=12)
            axes[r, c].text2D(0.02, 0.5, f"{hemi.upper()} {view}", transform=axes[r, c].transAxes,
                              fontsize=11, rotation=90, va="center")
    handles = [Patch(facecolor=TYPE_COLORS[c], label=CODE_TO_NAME[c]) for c in TYPE_CODES]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("cyto7 vs von-Economo-derived cortical type", fontsize=18, y=0.995)
    _save(fig, out_path, dpi)


def render_difference(geom, diff, out_path, dpi):
    """(b) categorical per-vertex disagreement map."""
    print("  rendering difference_map.png ...")
    codes = [DIFF_AGREE, DIFF_CYTO7_MORE, DIFF_CYTO7_LESS, DIFF_GAP2]
    cmap = _discrete_cmap(codes, DIFF_COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    panels = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
    for ax, (hemi, view) in zip(axes.ravel(), panels):
        d = diff[hemi].astype(float).copy()
        d[np.isnan(diff[hemi])] = np.nan
        _surf_panel(ax, geom[hemi]["infl"], d, hemi, view, geom[hemi]["sulc"],
                    cmap, vmin=-0.5, vmax=3.5)
        ax.set_title(f"{hemi.upper()} {view}", fontsize=13)
    handles = [Patch(facecolor=DIFF_COLORS[c], label=DIFF_LABELS[c]) for c in codes]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Per-vertex disagreement (cyto7 vs von-Economo-derived)",
                 fontsize=17, y=0.99)
    _save(fig, out_path, dpi)


def render_signed_delta(geom, sdelta, out_path, dpi):
    """(c) signed Delta-ordinal continuous diverging map."""
    print("  rendering delta_ordinal_signed.png ...")
    from nilearn import plotting

    fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    panels = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
    for ax, (hemi, view) in zip(axes.ravel(), panels):
        plotting.plot_surf_stat_map(
            geom[hemi]["infl"], sdelta[hemi], hemi=HEMI_NILEARN[hemi], view=view,
            axes=ax, cmap=SIGNED_DELTA.cmap, vmax=SIGNED_DELTA.vmax, bg_map=geom[hemi]["sulc"],
            bg_on_data=True, colorbar=(view == "medial"),
        )
        ax.set_title(f"{hemi.upper()} {view}", fontsize=13)
    fig.suptitle("Signed Delta-ordinal  (red = cyto7 more differentiated, "
                 "blue = less)", fontsize=16, y=0.99)
    _save(fig, out_path, dpi)


def render_allocortex(geom, cyto7, out_path, dpi):
    """(d) cyto7 allocortex (code 1) coverage that von-Economo maps lack."""
    print("  rendering cyto7_allocortex_coverage.png ...")
    cmap = ListedColormap([(0.85, 0.10, 0.55)])
    fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    panels = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
    for ax, (hemi, view) in zip(axes.ravel(), panels):
        d = np.where(cyto7[hemi] == ALLOCORTEX_CODE, 1.0, np.nan)
        _surf_panel(ax, geom[hemi]["infl"], d, hemi, view, geom[hemi]["sulc"],
                    cmap, vmin=1, vmax=1)
        ax.set_title(f"{hemi.upper()} {view}", fontsize=13)
    fig.legend(handles=[Patch(facecolor=(0.85, 0.10, 0.55), label="cyto7 allocortex (code 1)")],
               loc="lower center", fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("cyto7 allocortex coverage  (no von-Economo 6-type counterpart; "
                 "cyto7's added scope)", fontsize=15, y=0.99)
    _save(fig, out_path, dpi)


def render_figure2_composite(geom, sdelta, cm_counts, cm_rownorm, pooled, out_path, dpi):
    """Manuscript Fig 2: signed ordinal-difference surface (left) + confusion
    matrix heatmap (right), annotated with quadratic-weighted κ, Cohen's κ, ARI
    and % exact agreement. Reuses the signed-Δ rendering and the confusion counts.
    """
    print("  rendering figure2_vs_voneconomo.png ...")
    import seaborn as sns
    from nilearn import plotting

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(21, 10.5))
    fig.patch.set_facecolor("white")
    # Extra thin bottom row hosts a dedicated signed-Δ colour bar under the
    # surface panels, so it never overlaps the confusion-matrix row labels.
    gs = fig.add_gridspec(
        3, 3, width_ratios=[1.0, 1.0, 1.35], height_ratios=[1.0, 1.0, 0.07],
        wspace=0.22, hspace=0.10,
    )

    # (left 2x2) signed Δ-ordinal on the inflated surface (no per-panel colour bar)
    panels = [("lh", "lateral", 0, 0), ("lh", "medial", 0, 1),
              ("rh", "lateral", 1, 0), ("rh", "medial", 1, 1)]
    for hemi, view, r, c in panels:
        ax = fig.add_subplot(gs[r, c], projection="3d")
        plotting.plot_surf_stat_map(
            geom[hemi]["infl"], sdelta[hemi], hemi=HEMI_NILEARN[hemi], view=view,
            axes=ax, cmap=SIGNED_DELTA.cmap, vmax=SIGNED_DELTA.vmax, bg_map=geom[hemi]["sulc"],
            bg_on_data=True, colorbar=False,
        )
        ax.set_title(f"{hemi.upper()} {view}", fontsize=12)

    # dedicated horizontal colour bar for the signed Δ, beneath the surface block
    cax = fig.add_subplot(gs[2, 0:2])
    cb = fig.colorbar(ScalarMappable(norm=SIGNED_DELTA.norm(), cmap=SIGNED_DELTA.cmap),
                      cax=cax, orientation="horizontal")
    cb.set_label("signed Δ ordinal  (blue: cyto7 less differentiated · red: more)",
                 fontsize=11)

    # (right) confusion heatmap spans the two surface rows only; colour =
    # row-normalised, annotation = counts
    axh = fig.add_subplot(gs[0:2, 2])
    names = TYPE_NAMES
    sns.heatmap(cm_rownorm, ax=axh, annot=cm_counts, fmt="d", cmap="magma",
                xticklabels=names, yticklabels=names, square=True,
                cbar_kws={"label": "row-normalised (cyto7 -> vE)", "shrink": 0.6},
                linewidths=0.5, linecolor="white")
    for i in range(len(names)):
        axh.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="cyan", lw=2))
    axh.set_xlabel("von-Economo-derived type")
    axh.set_ylabel("cyto7 type")
    axh.set_xticklabels(axh.get_xticklabels(), rotation=40, ha="right")
    axh.set_yticklabels(axh.get_yticklabels(), rotation=0)
    axh.set_title(
        f"Confusion matrix (counts; colour = row-norm)\n"
        f"quadratic-weighted κ = {pooled['weighted_kappa_quadratic']:.3f}   "
        f"Cohen's κ = {pooled['cohen_kappa']:.3f}\n"
        f"ARI = {pooled['adjusted_rand_index']:.3f}   "
        f"exact agreement = {pooled['overall_agreement']:.1%}",
        fontsize=12,
    )
    fig.suptitle("cyto7 vs von-Economo-derived cortical type: signed Δordinal (left) "
                 "& confusion matrix (right)", fontsize=16, y=0.98)
    _save(fig, out_path, dpi)


def _save(fig, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out_path}")


# --------------------------------------------------------------------------- #
# Myelin cross-check (Step 6)
# --------------------------------------------------------------------------- #


#: Default Connectome Workbench bin dir (neuromaps fs_LR<->fsaverage resampling
#: calls ``wb_command``). Matches scripts/fetch_neuromaps_features.py. Override
#: with --workbench-bin or $WORKBENCH_BIN.
DEFAULT_WORKBENCH_BIN = (
    cfg.workbench_dir()
)


def _ensure_workbench_on_path(workbench_bin: str | None) -> None:
    import os

    candidate = workbench_bin or os.environ.get("WORKBENCH_BIN") or DEFAULT_WORKBENCH_BIN
    if candidate and Path(candidate).is_dir() and candidate not in os.environ["PATH"]:
        os.environ["PATH"] = candidate + os.pathsep + os.environ["PATH"]


def myelin_on_fsaverage(mesh: str, n_vert: dict, workbench_bin: str | None = None) -> dict | None:
    """Resample HCP T1w/T2w myelin (32k fs_LR) -> fsaverage via neuromaps.

    Returns ``{"lh": arr, "rh": arr}`` matching the comparison mesh, or ``None``
    if neuromaps resampling is unavailable. Cached under the cache dir.
    """
    cache = cfg.data_dir() / "neuromaps_cache"
    cache.mkdir(parents=True, exist_ok=True)
    target_density = "164k" if mesh == "fsaverage" else "10k"
    out = {}
    try:
        _ensure_workbench_on_path(workbench_bin)
        from neuromaps import transforms

        from cyto7_surface_io import load_myelin_on_surface
        myelin_fslr = load_myelin_on_surface("Validation210")  # {"L","R"} 32492
        import nibabel as nib
        from nibabel.gifti import GiftiDataArray, GiftiImage

        for hemi, side in (("lh", "L"), ("rh", "R")):
            cpath = cache / f"myelin_fsaverage_{target_density}_{hemi}.npy"
            if cpath.exists():
                out[hemi] = np.load(cpath)
                continue
            gi = GiftiImage()
            gi.add_gifti_data_array(
                GiftiDataArray(myelin_fslr[side].astype(np.float32))
            )
            res = transforms.fslr_to_fsaverage(
                gi, target_density=target_density, hemi={"L": "L", "R": "R"}[side],
                method="linear",
            )
            arr = np.asarray(res[0].agg_data(), dtype=float)
            if mesh == "fsaverage5":
                arr = arr[: n_vert[hemi]]
            np.save(cpath, arr)
            out[hemi] = arr
        return out
    except Exception as exc:  # pragma: no cover - optional step
        print(f"  [myelin] resampling unavailable ({exc!r}); skipping myelin tag.")
        return None


def myelin_type_medians(myelin: dict, cyto7: dict) -> dict:
    """Median myelin per cyto7 type (2-7), pooled across hemispheres."""
    med = {}
    allm = np.concatenate([myelin["lh"], myelin["rh"]])
    allc = np.concatenate([cyto7["lh"], cyto7["rh"]])
    valid = allm > 0  # 0 = medial wall / no data
    for code in TYPE_CODES:
        sel = valid & (allc == code)
        med[code] = float(np.median(allm[sel])) if sel.any() else np.nan
    return med


def myelin_tag(cluster_myelin: float, c_code: int, v_code: int, med: dict) -> str:
    """Tag a cluster by whether local myelin favours the cyto7 or vE type."""
    if not np.isfinite(cluster_myelin) or c_code not in med or v_code not in med:
        return "n/a"
    mc, mv = med[c_code], med[v_code]
    if not (np.isfinite(mc) and np.isfinite(mv)):
        return "n/a"
    margin = 0.15 * abs(mc - mv)
    closer = abs(cluster_myelin - mv) - abs(cluster_myelin - mc)  # + => closer to cyto7
    if closer > margin:
        return "cyto7 supported by myelin"
    if closer < -margin:
        return "von-Economo supported by myelin"
    return "ambiguous"


# --------------------------------------------------------------------------- #
# Review candidates (Step 5)
# --------------------------------------------------------------------------- #


def aparc_labels(hemi: str) -> tuple[np.ndarray, list[str]]:
    """Desikan-Killiany per-vertex parcel index + names for anatomical labels."""
    return _read_annot(VONECONOMO_DIR / f"{hemi}.aparc.annot")


def find_review_clusters(
    geom, cyto7, ve, mask, myelin, med, min_vertices, boundary_mm, systematic_mm2
) -> pd.DataFrame:
    """Connected components of disagreement, scored & anatomically labelled."""
    rows = []
    for hemi in HEMIS:
        coords, faces = geom[hemi]["coords"], geom[hemi]["faces"]
        area = geom[hemi]["area"]
        graph = adjacency_graph(coords, faces)
        ve_h, c_h, m_h = ve[hemi], cyto7[hemi], mask[hemi]
        area_idx = geom[hemi]["_area_idx"]
        bdist = distance_to_area_boundary(graph, area_idx)
        apx, aname = aparc_labels(hemi)
        apx = np.clip(apx, 0, len(aname) - 1)  # FreeSurfer marks unknown as -1
        myel = myelin[hemi] if myelin is not None else None

        disagree = m_h & (c_h != ve_h)
        if not disagree.any():
            continue
        # Connected components within the disagreement sub-graph.
        idx = np.where(disagree)[0]
        sub = graph[idx][:, idx]
        n_comp, comp = connected_components(sub, directed=False)
        for k in range(n_comp):
            v = idx[comp == k]
            if v.size < min_vertices:
                continue
            ords_c = np.array([CODE_TO_ORDINAL[c] for c in c_h[v]])
            ords_v = np.array([CODE_TO_ORDINAL[c] for c in ve_h[v]])
            sdelta = ords_c - ords_v
            mean_abs = float(np.mean(np.abs(sdelta)))
            mean_signed = float(np.mean(sdelta))
            area_mm2 = float(area[v].sum())
            interior_mm = float(np.median(bdist[v]))
            # dominant cyto7 / vE type and aparc parcel.
            c_mode = int(np.bincount(c_h[v]).argmax())
            v_mode = int(np.bincount(ve_h[v]).argmax())
            parcel_counts = np.bincount(apx[v], minlength=len(aname))
            parcel = aname[int(parcel_counts.argmax())]
            centroid = coords[v].mean(0)
            cvert = v[np.argmin(np.linalg.norm(coords[v] - centroid, axis=1))]
            cl_myel = float(np.median(myel[v][myel[v] > 0])) if (
                myel is not None and (myel[v] > 0).any()) else np.nan
            rows.append({
                "hemi": hemi,
                "dominant_aparc": parcel,
                "n_vertices": int(v.size),
                "area_mm2": round(area_mm2, 1),
                "cyto7_type": CODE_TO_NAME[c_mode],
                "voneconomo_type": CODE_TO_NAME.get(v_mode, "?"),
                "mean_abs_delta_ordinal": round(mean_abs, 3),
                "mean_signed_delta_ordinal": round(mean_signed, 3),
                "median_dist_to_areal_edge_mm": round(interior_mm, 2),
                "centroid_vertex": int(cvert),
                "_centroid": centroid,
                "cluster_median_myelin": round(cl_myel, 4) if np.isfinite(cl_myel) else "",
                "myelin_adjudication": myelin_tag(cl_myel, c_mode, v_mode, med),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Composite review-priority: larger, higher ordinal-gap, more interior =
    # higher priority. Each component min-max normalised across clusters.
    def _norm(x):
        x = np.asarray(x, dtype=float)
        rng = np.nanmax(x) - np.nanmin(x)
        return (x - np.nanmin(x)) / rng if rng > 0 else np.zeros_like(x)

    comp_size = _norm(np.log1p(df["area_mm2"]))
    comp_gap = _norm(df["mean_abs_delta_ordinal"])
    comp_interior = _norm(df["median_dist_to_areal_edge_mm"].replace(np.inf, np.nan))
    df["review_priority"] = (comp_size + comp_gap + 0.5 * comp_interior).round(4)

    # Refinement vs review category (§0/§5/Step 5). Review candidates are: a
    # large ordinal gap (|Delta|>=2), OR a large *systematic* off-by-one patch
    # whose independent myelin signal favours the von Economo type (the task's
    # "strongest review candidate"). Everything else is expected refinement.
    cat = []
    for r in df.itertuples():
        boundary_adjacent = r.median_dist_to_areal_edge_mm <= boundary_mm
        myelin_favours_ve = r.myelin_adjudication == "von-Economo supported by myelin"
        if r.mean_abs_delta_ordinal >= 2 and not boundary_adjacent:
            cat.append("review candidate (large gap, interior)")
        elif r.mean_abs_delta_ordinal >= 2:
            cat.append("review candidate (large gap, boundary-adjacent)")
        elif r.area_mm2 >= systematic_mm2 and myelin_favours_ve:
            cat.append("review candidate (systematic off-by-one, myelin favours von Economo)")
        elif boundary_adjacent:
            cat.append("expected refinement (off-by-one, boundary-adjacent)")
        else:
            cat.append("refinement (off-by-one, interior)")
    df["category"] = cat
    return df.sort_values("review_priority", ascending=False).reset_index(drop=True)


def render_review_clusters(geom, df, mask, cyto7, ve, out_path, dpi, top_n):
    """Outline the top review clusters on the inflated surface."""
    print("  rendering review_candidates.png ...")
    top = df.head(top_n)
    # Build a per-vertex highlight map: priority rank colour for top clusters.
    fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    panels = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
    # Recompute cluster membership for the top clusters by centroid flood is
    # overkill; instead re-derive disagreement and mark top clusters by index.
    highlight = {h: np.full(cyto7[h].shape[0], np.nan) for h in HEMIS}
    from scipy.sparse.csgraph import connected_components as _cc
    for hemi in HEMIS:
        coords, faces = geom[hemi]["coords"], geom[hemi]["faces"]
        graph = adjacency_graph(coords, faces)
        disagree = mask[hemi] & (cyto7[hemi] != ve[hemi])
        idx = np.where(disagree)[0]
        if idx.size == 0:
            continue
        _n, comp = _cc(graph[idx][:, idx], directed=False)
        # map centroid_vertex -> component id
        for rank, r in enumerate(top.itertuples(), start=1):
            if r.hemi != hemi:
                continue
            cv = r.centroid_vertex
            pos = np.where(idx == cv)[0]
            if pos.size:
                members = idx[comp == comp[pos[0]]]
                highlight[hemi][members] = rank
    cmap = plt.get_cmap("turbo_r")  # rank 1 (highest priority) -> warm/salient
    for ax, (hemi, view) in zip(axes.ravel(), panels):
        _surf_panel(ax, geom[hemi]["infl"], highlight[hemi], hemi, view,
                    geom[hemi]["sulc"], cmap, vmin=1, vmax=max(top_n, 2))
        ax.set_title(f"{hemi.upper()} {view}", fontsize=13)
    fig.suptitle(f"Top {len(top)} review clusters (colour = priority rank, "
                 "1 = highest)", fontsize=15, y=0.99)
    _save(fig, out_path, dpi)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def slice_mesh(arr: np.ndarray, mesh: str, n5: int) -> np.ndarray:
    """Downsample an fsaverage array to fsaverage5 via the nested-ico subset."""
    return arr[:n5] if mesh == "fsaverage5" else arr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mesh", choices=["fsaverage", "fsaverage5"], default="fsaverage",
                   help="Comparison mesh. fsaverage (164k) is the exact, headline "
                        "route; fsaverage5 (10k) is a nested-ico down-sample (label "
                        "subset, nearest-neighbour) for cross-checks. Default: %(default)s.")
    p.add_argument("--annot-version", default=None,
                   help="cyto7 target-map version (v1|v3|...) or {hemi}-templated path to a "
                        "164k annot. If set, cyto7 labels are loaded via resolve_target_map "
                        "instead of the bundled v1 standard-fsaverage annot. v1 reproduces the "
                        "default exactly (same file, same reader).")
    p.add_argument("--output-dir", type=Path,
                   default=cfg.results_dir("tables") / "vs_voneconomo")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--min-cluster-vertices", type=int, default=30,
                   help="Minimum disagreement-cluster size for the ranked table.")
    p.add_argument("--boundary-mm", type=float, default=3.0,
                   help="A cluster within this geodesic distance of a von Economo "
                        "areal edge is treated as boundary-adjacent (expected refinement).")
    p.add_argument("--systematic-mm2", type=float, default=500.0,
                   help="A large off-by-one cluster (>= this area) whose myelin favours "
                        "the von Economo type is flagged a systematic review candidate.")
    p.add_argument("--top-n", type=int, default=20, help="Clusters to outline in the figure.")
    p.add_argument("--no-myelin", action="store_true", help="Skip the Step 6 myelin cross-check.")
    p.add_argument("--workbench-bin", default=None,
                   help="Connectome Workbench bin dir (for neuromaps fs_LR->fsaverage "
                        "myelin resampling). Defaults to $WORKBENCH_BIN or the project default.")
    p.add_argument("--no-figures", action="store_true", help="Skip surface rendering (stats only).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    N5 = 10242

    print(f"Mesh: {args.mesh}")
    print(f"cyto7 source: {args.annot_version or 'v1 (bundled standard-fsaverage annot)'}")
    lookup = pd.read_csv(VONECONOMO_DIR / "von_economo_cortical_types.csv")

    # ---- Step 1: load & align ----
    # cyto7 labels at 164k fsaverage. With --annot-version they come from
    # resolve_target_map (v1 reproduces load_cyto7 exactly: same file, same
    # reader); fsaverage5 is the nested-ico subset handled by slice_mesh below.
    if args.annot_version:
        labels164 = resolve_target_map(args.annot_version, "fsaverage")
        cyto7_full = {"lh": labels164["L"], "rh": labels164["R"]}
    else:
        cyto7_full = {hemi: load_cyto7(hemi) for hemi in HEMIS}

    cyto7, ve, area_idx = {}, {}, {}
    for hemi in HEMIS:
        c = slice_mesh(cyto7_full[hemi], args.mesh, N5)
        v, ai = load_economo_type_vector(hemi, lookup)
        cyto7[hemi] = c
        ve[hemi] = slice_mesh(v, args.mesh, N5)
        area_idx[hemi] = slice_mesh(ai, args.mesh, N5)
    mask = {h: common_support_mask(cyto7[h], ve[h]) for h in HEMIS}
    n_vert = {h: cyto7[h].shape[0] for h in HEMIS}

    # Allocortex coverage (reported separately).
    allo = {h: int(np.sum(cyto7[h] == ALLOCORTEX_CODE)) for h in HEMIS}
    print(f"  common-support vertices: LH={mask['lh'].sum()}, RH={mask['rh'].sum()}")
    print(f"  cyto7 allocortex (excluded from agreement): LH={allo['lh']}, RH={allo['rh']}")

    # ---- Step 2: confusion matrix (pooled) ----
    c_all = np.concatenate([cyto7[h][mask[h]] for h in HEMIS])
    v_all = np.concatenate([ve[h][mask[h]] for h in HEMIS])
    mat = confusion_matrix(c_all, v_all)
    cm = pd.DataFrame(mat, index=[f"cyto7:{CODE_TO_NAME[c]}" for c in TYPE_CODES],
                      columns=[f"vE:{CODE_TO_NAME[c]}" for c in TYPE_CODES])
    cm.to_csv(out / "confusion_matrix.csv")
    cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    cm_norm.to_csv(out / "confusion_matrix_rownorm.csv")
    print(f"  wrote {out / 'confusion_matrix.csv'}")
    if not args.no_figures:
        _render_confusion(cm.values, cm_norm.values, out / "confusion_matrix.png", args.dpi)

    # ---- Step 3: agreement statistics ----
    stats_rows = [agreement_stats(c_all, v_all, "pooled")]
    for hemi in HEMIS:
        stats_rows.append(agreement_stats(cyto7[hemi][mask[hemi]],
                                          ve[hemi][mask[hemi]], hemi.upper()))
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out / "agreement_summary.csv", index=False)
    _write_agreement_txt(stats_df, allo, args.mesh, out / "agreement_summary.txt")
    print(f"  wrote {out / 'agreement_summary.csv'} / .txt")

    # ---- geometry, difference fields ----
    geom = load_geometry(args.mesh)
    for hemi in HEMIS:
        geom[hemi]["_area_idx"] = area_idx[hemi]
    diff, sdelta = {}, {}
    for hemi in HEMIS:
        d = np.full(n_vert[hemi], np.nan)
        s = np.zeros(n_vert[hemi])
        m = mask[hemi]
        oc = np.array([CODE_TO_ORDINAL.get(x, np.nan) for x in cyto7[hemi]])
        ov = np.array([CODE_TO_ORDINAL.get(x, np.nan) for x in ve[hemi]])
        delta = oc - ov
        s[m] = delta[m]
        cat = np.full(n_vert[hemi], np.nan)
        cat[m & (delta == 0)] = DIFF_AGREE
        cat[m & (delta == 1)] = DIFF_CYTO7_MORE
        cat[m & (delta == -1)] = DIFF_CYTO7_LESS
        cat[m & (np.abs(delta) >= 2)] = DIFF_GAP2
        diff[hemi] = cat
        sdelta[hemi] = np.where(m, s, np.nan)

    # ---- myelin (Step 6) ----
    myelin = None if args.no_myelin else myelin_on_fsaverage(args.mesh, n_vert, args.workbench_bin)
    med = myelin_type_medians(myelin, cyto7) if myelin is not None else {}

    # ---- Step 5: review candidates ----
    print("  finding review clusters ...")
    clusters = find_review_clusters(geom, cyto7, ve, mask, myelin, med,
                                    args.min_cluster_vertices, args.boundary_mm,
                                    args.systematic_mm2)
    if not clusters.empty:
        clusters.drop(columns=["_centroid"]).to_csv(out / "review_candidates.csv", index=False)
        print(f"  wrote {out / 'review_candidates.csv'}  ({len(clusters)} clusters >= "
              f"{args.min_cluster_vertices} verts)")
    else:
        print("  no disagreement clusters above the size threshold.")

    # ---- figures ----
    if not args.no_figures:
        render_sidebyside(geom, cyto7, ve, out / "sidebyside_surface.png", args.dpi)
        render_difference(geom, diff, out / "difference_map.png", args.dpi)
        render_signed_delta(geom, sdelta, out / "delta_ordinal_signed.png", args.dpi)
        pooled = stats_df[stats_df["set"] == "pooled"].iloc[0].to_dict()
        render_figure2_composite(geom, sdelta, mat, cm_norm.values, pooled,
                                 out / "figure2_vs_voneconomo.png", args.dpi)
        render_allocortex(geom, cyto7, out / "cyto7_allocortex_coverage.png", args.dpi)
        if not clusters.empty:
            render_review_clusters(geom, clusters, mask, cyto7, ve,
                                   out / "review_candidates.png", args.dpi, args.top_n)

    # ---- REPORT.md ----
    _write_report(out / "REPORT.md", args, stats_df, cm, clusters, allo, n_vert,
                  myelin is not None, lookup)
    print("Done.")


# --------------------------------------------------------------------------- #
# Confusion heatmap, agreement .txt, REPORT.md writers
# --------------------------------------------------------------------------- #


def _render_confusion(counts, rownorm, out_path: Path, dpi: int) -> None:
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    names = TYPE_NAMES
    for ax, data, title, fmt, cmap in (
        (axes[0], counts, "Counts", "d", "Greys"),
        (axes[1], rownorm, "Row-normalised (cyto7 -> vE)", ".2f", "magma"),
    ):
        sns.heatmap(data, ax=ax, annot=True, fmt=fmt, cmap=cmap, cbar=True,
                    xticklabels=names, yticklabels=names, square=True,
                    linewidths=0.5, linecolor="white")
        ax.set_xlabel("von-Economo-derived type")
        ax.set_ylabel("cyto7 type")
        ax.set_title(title)
        # mark the diagonal
        for i in range(len(names)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="cyan", lw=2))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.suptitle("Confusion matrix: cyto7 (rows) vs von-Economo-derived (cols)",
                 fontsize=15)
    _save(fig, out_path, dpi)


def _write_agreement_txt(df: pd.DataFrame, allo: dict, mesh: str, out_path: Path) -> None:
    lines = ["Agreement summary: cyto7 vs von-Economo-derived cortical type",
             f"Comparison mesh: {mesh}", ""]
    for r in df.itertuples():
        lines += [
            f"[{r.set}]  n={r.n_vertices}",
            f"  overall agreement        : {r.overall_agreement:.3f}",
            f"  Cohen's kappa            : {r.cohen_kappa:.3f}",
            f"  quadratic-weighted kappa : {r.weighted_kappa_quadratic:.3f}  (HEADLINE: ordinal types)",
            f"  adjusted Rand index      : {r.adjusted_rand_index:.3f}",
            f"  mean |Delta-ordinal|     : {r.mean_abs_delta_ordinal:.3f}",
            f"  mean signed Delta-ordinal: {r.mean_signed_delta_ordinal:+.3f}  (+ => cyto7 more differentiated)",
            "",
        ]
    lines += [
        f"cyto7 allocortex (code 1), excluded from agreement: LH={allo['lh']}, RH={allo['rh']} vertices",
        "",
        "CAVEAT: these are DESCRIPTIVE statistics. Neighbouring vertices are",
        "spatially autocorrelated, so no naive p-values are attached. For a",
        "significance test, reuse the spin-test machinery in",
        "scripts/summarise_functional_features.py.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (avoids a tabulate dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(x) for x in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _write_report(out_path, args, stats_df, cm, clusters, allo, n_vert, did_myelin, lookup):
    pooled = stats_df[stats_df["set"] == "pooled"].iloc[0]
    n_amb = int((lookup["ambiguous"] == True).sum() + (lookup["ambiguous"] == "TRUE").sum())  # noqa: E712
    excl = lookup[lookup["garcia_cabezas_type"].isna() | (lookup["garcia_cabezas_type"] == "")]
    lines = []
    A = lines.append
    A("# cyto7 vs von-Economo-derived cortical types — comparison report\n")
    A(f"*Mesh: **{args.mesh}**"
      f"{' (163,842 verts/hemi, exact match — no resampling)' if args.mesh=='fsaverage' else ' (10,242 verts/hemi, nested-ico nearest-neighbour down-sample)'}.*\n")
    A(f"*cyto7 source: **{args.annot_version or 'v1 (hand-painted, standard fsaverage)'}**.*\n")
    A("> **Disagreement is not error.** The von-Economo-derived map is *area-level* "
      "(piecewise-constant within von Economo areas); cyto7 is drawn at *vertex* "
      "resolution. Most disagreement is cyto7 legitimately refining a coarse areal "
      "scaffold. Only large, interior, high-ordinal-gap, independently-discordant "
      "patches are genuine review candidates — and the final call belongs to the "
      "human authors against histology and the García-Cabezas protocol.\n")

    A("## 1. Headline numbers (pooled, common support)\n")
    A(f"- Common-support vertices: **{n_vert and (cm.values.sum())}** "
      "(cyto7 ∈ types 2–7 **and** von Economo has an isocortical 6-type).")
    A(f"- Overall agreement: **{pooled.overall_agreement:.1%}**")
    A(f"- **Quadratic-weighted κ (headline): {pooled.weighted_kappa_quadratic:.3f}**")
    A(f"- Cohen's κ (unweighted): {pooled.cohen_kappa:.3f}")
    A(f"- Adjusted Rand Index: {pooled.adjusted_rand_index:.3f}")
    A(f"- Mean |Δordinal|: {pooled.mean_abs_delta_ordinal:.3f}; "
      f"mean signed Δordinal: {pooled.mean_signed_delta_ordinal:+.3f} "
      "(+ ⇒ cyto7 more differentiated than the von-Economo area on average).\n")
    A("Per-hemisphere values and per-type Dice/IoU/Δ are in "
      "`agreement_summary.csv` / `.txt`; the full confusion matrix is in "
      "`confusion_matrix.csv` and `confusion_matrix.png`.\n")

    A("## 2. Figures\n")
    A("- `sidebyside_surface.png` — cyto7 vs von-Economo-derived type, shared 6-colour scale.")
    A("- `difference_map.png` — per-vertex disagreement category (agree / off-by-one ± / off-by-≥2).")
    A("- `delta_ordinal_signed.png` — signed Δordinal (red = cyto7 more differentiated).")
    A("- `cyto7_allocortex_coverage.png` — cyto7 allocortex, the scope the von-Economo 6-type map lacks "
      f"(LH={allo['lh']}, RH={allo['rh']} vertices).")
    A("- `review_candidates.png` — top clusters outlined, coloured by priority rank.\n")

    A("## 3. Candidate regions for review\n")
    if clusters is None or clusters.empty:
        A("No disagreement clusters above the size threshold.\n")
    else:
        review = clusters[clusters["category"].str.startswith("review candidate")]
        refine = clusters[~clusters["category"].str.startswith("review candidate")]
        A(f"`review_candidates.csv` ranks **{len(clusters)}** disagreement clusters "
          f"(≥ {args.min_cluster_vertices} vertices) by a composite `review_priority` "
          "(normalised size + ordinal gap + 0.5·interiorness).\n")
        A(f"- **Genuine review candidates** (large ordinal gap |Δ|≥2, or a large "
          "*systematic* off-by-one patch whose independent T1w/T2w myelin signal "
          f"favours the von Economo type): **{len(review)}**. "
          "These are worth re-checking against histology / the protocol — *not* presumed errors.")
        A(f"- **Expected refinements** (off-by-one and/or boundary-adjacent): **{len(refine)}**. "
          "These are the anticipated consequence of vertex-level painting over coarse areal edges.\n")
        A("Top 10 by review priority:\n")
        cols = ["hemi", "dominant_aparc", "area_mm2", "cyto7_type", "voneconomo_type",
                "mean_signed_delta_ordinal", "median_dist_to_areal_edge_mm",
                "myelin_adjudication", "review_priority", "category"]
        top = clusters[cols].head(10)
        A(_df_to_md(top))
        A("")
        if did_myelin:
            A("The `myelin_adjudication` column uses the independent T1w/T2w myelin "
              "ordering (myelin rises with differentiation): clusters tagged "
              "*von-Economo supported by myelin* are the strongest review candidates; "
              "*cyto7 supported by myelin* are most likely legitimate refinements.\n")
        else:
            A("*(Myelin cross-check skipped; `myelin_adjudication` = n/a.)*\n")

    A("## 4. Caveats (must be read with any quoted number)\n")
    A(f"1. **Lookup is a verification checkpoint.** The von-Economo-area → type "
      f"assignment (`von_economo_cortical_types.csv`, **verified=FALSE**) was read "
      f"from García-Cabezas et al. (2020) Tables 4–7. **{n_amb} of 40** typed areas "
      "aggregate García-Cabezas sub-areas of differing type and are flagged "
      "`ambiguous`; the authors must confirm these before quoting.")
    A("2. **6 vs 7 types.** The von-Economo-derived map is isocortex-only. cyto7 "
      "allocortex (code 1) has no counterpart and is excluded from agreement, "
      "reported separately as cyto7's added coverage.")
    A("3. **Area-level vs vertex-level.** The comparison map's boundaries are von "
      "Economo areal edges; disagreement near those edges is largely expected "
      "refinement, not error.")
    A(f"4. **Mesh / resampling.** This run used **{args.mesh}**. von Economo labels "
      "were produced directly on fsaverage via `mris_ca_label` (exact); the "
      "fsaverage5 option uses a nested-icosahedron label subset (nearest-neighbour).")
    A("5. **Descriptive statistics.** No correction for spatial autocorrelation; "
      "no naive p-values.")
    A("6. **Excluded limbic areas.** "
      f"{', '.join(excl['economo_acronym'])} (periallocortex) carry no isocortical "
      "6-type and are excluded.")
    A("7. **Native 5-type scheme is not used here.** The micaopen `economo7` map is "
      "the native von Economo types (agranular/frontal/parietal/polar/granular + "
      "limbic/insular), which are *not* ordinally aligned with cyto7; the headline "
      "comparison uses the García-Cabezas 6-type assignment only.\n")

    A("## 5. Reproduce\n")
    A("```bash\n"
      "python scripts/fetch_voneconomo_atlas.py            # atlas + lookup CSV\n"
      "wsl.exe -d Ubuntu-20.04 bash scripts/make_economo_annot_fsaverage.sh  # annot on fsaverage\n"
      f"python scripts/compare_cyto7_vs_voneconomo.py --mesh {args.mesh}"
      f"{f' --annot-version {args.annot_version}' if args.annot_version else ''}\n"
      "```")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
