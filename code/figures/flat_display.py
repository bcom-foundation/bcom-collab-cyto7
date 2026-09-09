"""Shared display-layer helpers for the flat cyto7 panels (SPEC_flat_panel_median_recolor).

Single source of truth for three *display-only* corrections to the flat renders, so the
renderers and the audit (``flat_panel_cleanup.py``) cannot drift apart. **Nothing here
touches the released 164k annot or the released 32k label GIFTIs.**

1. :func:`face_median` — colour each triangle by the **median** of its vertex labels rather
   than the majority. The majority rule breaks ties toward the lowest code, so a face
   straddling types {4,5,6} is painted 4 while its neighbour {5,6,6} is painted 6,
   manufacturing a |Δtype| = 2 seam out of vertex labels that never skip. Median cannot:
   two faces sharing an edge share two vertices whose labels differ by at most 1 (once
   vertex-level skips are repaired), and the median of a triple containing those two always
   lies between them, so adjacent face colours differ by at most 1 *by construction*. The
   median is also the natural summary of an ordinal quantity, which is what a cortical type
   is.

2. :func:`display_labels` — the ordinality-respecting repair of the three triple points that
   the 164k→32k fs_LR nearest-neighbour resample pinches out. The released map is clean on
   its native mesh (R1 = 0 skip-edges, §3.2), but the resample drops the single vertex
   carrying a one-vertex-wide intervening band, leaving three genuine skip adjacencies in
   the **32k display array**. Each is repaired by re-labelling one vertex to the intervening
   type — a type already present among that vertex's own labelled mesh neighbours — and only
   if that removes the skip without creating a new one.

3. :func:`drop_degenerate` — a keep-mask and draw order for the flat projection: drop
   orientation-flipped and near-zero-area triangles (which the locally non-injective
   flattening duplicates into the wrong area) and paint the largest faces first, so a
   genuine thin band is never buried under a big folded neighbour.

The repaired display labels are cached under ``resources/cyto7_derived/cache/`` with
``_display`` in the name, next to the existing resample caches, to make it unmistakable that
they are a rendering artefact and not a map release.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from pathlib import Path

import numpy as np

from cyto7_surface_io import REPO_ROOT, resolve_target_map

CACHE = cfg.scratch_dir("resample_cache")


# --------------------------------------------------------------------------- #
# Face colouring
# --------------------------------------------------------------------------- #
def face_median(faces: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Lower-median cyto7 label per face (0 where no vertex is labelled).

    Lower median (index ``(n-1)//2`` of the sorted labelled values) so the two-labelled-
    vertex case resolves to the lower type instead of a non-existent half-step. The
    between-ness property that makes adjacent face colours differ by at most 1 holds for
    the lower median too.
    """
    fl = lab[faces]
    out = np.zeros(len(faces), dtype=int)
    for i in np.where((fl > 0).any(axis=1))[0]:
        r = np.sort(fl[i][fl[i] > 0])
        out[i] = r[(len(r) - 1) // 2]
    return out


def face_mode(faces: np.ndarray, code: np.ndarray) -> np.ndarray:
    """Modal (majority) code per face — for *categorical*, non-ordinal fields.

    Used by the change-history flat panel, whose codes (added / removed / topology) have no
    order, so a median would be meaningless. The point of using it there is only to avoid
    averaging RGB across a face, which invents colours that are in no palette entry.
    """
    fl = code[faces]
    out = np.zeros(len(faces), dtype=int)
    for i in range(len(faces)):
        r = fl[i]
        nz = r[r > 0]
        out[i] = np.bincount(nz).argmax() if nz.size else 0
    return out


# --------------------------------------------------------------------------- #
# Display-label repair
# --------------------------------------------------------------------------- #
def edge_list(faces: np.ndarray) -> np.ndarray:
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def vertex_skips(edges: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Edges whose two labelled endpoints differ by >= 2 (a 'type skip')."""
    a, b = lab[edges[:, 0]], lab[edges[:, 1]]
    return edges[(a > 0) & (b > 0) & (np.abs(a - b) >= 2)]


def repair_display_labels(edges: np.ndarray, lab: np.ndarray):
    """Insert the intervening type at type-N / type-N+k (k >= 2) display adjacencies.

    Returns ``(repaired_labels, records)``. A candidate re-label is accepted only if the
    intervening type already occurs among that vertex's labelled mesh neighbours (so no type
    is invented anywhere) and it removes the skip without creating a new one.
    """
    lab = lab.copy()
    nbrs: dict[int, np.ndarray] = {}

    def neighbours(v: int) -> np.ndarray:
        if v not in nbrs:
            n = np.unique(edges[(edges[:, 0] == v) | (edges[:, 1] == v)])
            nbrs[v] = n[n != v]
        return nbrs[v]

    def creates_skip(v: int, new: int) -> bool:
        nl = lab[neighbours(v)]
        nl = nl[nl > 0]
        return bool((np.abs(nl - new) >= 2).any())

    records = []
    for _ in range(4):                       # converges immediately in practice
        sk = vertex_skips(edges, lab)
        if not len(sk):
            break
        for v0, v1 in sk:
            t0, t1 = int(lab[v0]), int(lab[v1])
            if abs(t0 - t1) < 2:
                continue                     # already resolved by an earlier repair
            inter = min(t0, t1) + 1
            lo, hi = (v0, v1) if t0 < t1 else (v1, v0)
            for v in (lo, hi):
                nl = lab[neighbours(v)]
                if inter not in nl[nl > 0] or creates_skip(v, inter):
                    continue
                records.append({"vertex": int(v), "from_type": int(lab[v]),
                                "to_type": int(inter),
                                "partner": int(v1 if v == v0 else v0),
                                "partner_type": int(lab[v1 if v == v0 else v0])})
                lab[v] = inter
                break
    return lab, records


def display_labels(H: str, faces: np.ndarray | None = None, version: str = "v9",
                   use_cache: bool = True) -> np.ndarray:
    """Repaired 32k fs_LR cyto7 labels **for rendering only**.

    Identical to ``resolve_target_map(version, "fs_LR")[H]`` except at the handful of
    vertices the resample pinched out. Cached; pass ``use_cache=False`` to recompute.
    """
    cpath = CACHE / f"{version}_display_labels_fsLR32k_hemi-{H}.npy"
    if use_cache and cpath.exists():
        return np.load(cpath)
    lab = np.asarray(resolve_target_map(version, "fs_LR")[H]).astype(int)
    if faces is None:
        import nibabel as nib
        from cyto7_surface_io import surface_path
        g = nib.load(str(surface_path("Validation210", H, "flat")))
        faces = np.asarray(g.darrays[1].data, int)
    fixed, _ = repair_display_labels(edge_list(faces), lab)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cpath, fixed)
    return fixed


# --------------------------------------------------------------------------- #
# Flat-projection folding
# --------------------------------------------------------------------------- #
def drop_degenerate(pts2d: np.ndarray, faces: np.ndarray, tiny_frac: float = 0.01):
    """``(keep_mask, draw_order)`` for a flat PolyCollection.

    Drops orientation-flipped and near-degenerate triangles, and orders the remainder
    largest-area-first so small genuine bands end up on top rather than buried.
    """
    tri = pts2d[faces]
    ax_, ay = tri[:, 1, 0] - tri[:, 0, 0], tri[:, 1, 1] - tri[:, 0, 1]
    bx, by = tri[:, 2, 0] - tri[:, 0, 0], tri[:, 2, 1] - tri[:, 0, 1]
    signed = 0.5 * (ax_ * by - ay * bx)
    area = np.abs(signed)
    med = float(np.median(area[area > 0]))
    flipped = signed < 0 if (signed > 0).sum() > len(signed) / 2 else signed > 0
    keep = ~(flipped | (area < tiny_frac * med))
    return keep, np.argsort(-area)


def ordered_faces(pts2d: np.ndarray, faces: np.ndarray):
    """Faces filtered and reordered for drawing; returns the index array into *faces*."""
    keep, order = drop_degenerate(pts2d, faces)
    return order[keep[order]]
