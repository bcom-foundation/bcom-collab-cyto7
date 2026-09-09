"""Shared helpers for the review-response batch (SPEC_review_response_RUNNER.md).

Small utilities every RR task needs: the released-file integrity check, the
released spin-null loader, mesh-hop distance on the 32k fs_LR surface, and the
decision-log appender. Nothing here recomputes a published number; each task
imports what it needs and writes only under ``figures/v9/review_response/``.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import hashlib
from pathlib import Path

import numpy as np

from cyto7_surface_io import REPO_ROOT

OUT = cfg.results_dir("tables/review_response")
DECISIONS = OUT / "DECISIONS.md"
NULLS = cfg.scratch_dir("nulls") / "_spin_nulls_v9_fsLR32k_seed0_n1000.npy"

# Files that must never change during the batch (guardrail 1 of the index).
FROZEN = [
    cfg.atlas_dir("fsaverage") / "pial.lh.cyto7.v9.annot",
    cfg.atlas_dir("fsaverage") / "pial.rh.cyto7.v9.annot",
    cfg.atlas_dir("fs_LR_32k") / "pial.lh.cyto7.32k_fs_LR.label.gii",
    cfg.atlas_dir("fs_LR_32k") / "pial.rh.cyto7.32k_fs_LR.label.gii",
]

TYPE_NAMES = ["Allocortex", "Agranular", "Dysgranular", "Eulaminate I",
              "Eulaminate II", "Eulaminate III", "Koniocortex"]
SEED = 0
N_SPIN = 1000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes() -> dict[str, str]:
    return {p.name: sha256(p) for p in FROZEN if p.exists()}


def check_frozen(before: dict[str, str]) -> tuple[bool, dict]:
    """Return (all_unchanged, per-file dict of before/after)."""
    after = frozen_hashes()
    detail = {k: {"before": before.get(k), "after": after.get(k),
                  "unchanged": before.get(k) == after.get(k)} for k in sorted(set(before) | set(after))}
    return all(v["unchanged"] for v in detail.values()), detail


def load_nulls(n_spin: int = N_SPIN) -> np.ndarray:
    """The released (64984, 1000) Alexander-Bloch rotations of the v9 type-rank map."""
    arr = np.load(NULLS, mmap_mode="r")
    return np.asarray(arr[:, :n_spin])


def labels_32k(version: str = "v9") -> dict[str, np.ndarray]:
    from cyto7_surface_io import resolve_target_map
    lab = resolve_target_map(version, "fs_LR")
    return {"L": np.asarray(lab["L"]), "R": np.asarray(lab["R"])}


def cat(d, dtype=float) -> np.ndarray:
    return np.concatenate([np.asarray(d["L"], dtype=dtype), np.asarray(d["R"], dtype=dtype)])


# --------------------------------------------------------------------------- #
# 32k fs_LR mesh geometry / hop distances
# --------------------------------------------------------------------------- #


def fslr_geometry(surface: str = "midthickness") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(coords, faces) per hemisphere for the 32k fs_LR mesh, from neuromaps fetch."""
    import nibabel as nib
    from neuromaps.datasets import fetch_fslr
    surfs = fetch_fslr(density="32k")
    paths = surfs[surface]
    out = {}
    for H, p in zip(("L", "R"), paths):
        g = nib.load(str(p))
        out[H] = (np.asarray(g.darrays[0].data, float), np.asarray(g.darrays[1].data, int))
    return out


def fsaverage_geometry(surface: str = "white", density: str = "164k"):
    """(coords, faces) per hemisphere for the fsaverage mesh, from neuromaps fetch."""
    import nibabel as nib
    from neuromaps.datasets import fetch_fsaverage
    surfs = fetch_fsaverage(density=density)
    out = {}
    for H, p in zip(("L", "R"), surfs[surface]):
        g = nib.load(str(p))
        out[H] = (np.asarray(g.darrays[0].data, float), np.asarray(g.darrays[1].data, int))
    return out


def geodesic_graph(coords: np.ndarray, faces: np.ndarray):
    """Sparse mesh graph with Euclidean edge weights (for Dijkstra)."""
    from scipy.sparse import csr_matrix
    n = coords.shape[0]
    src = faces[:, [0, 1, 2]].ravel()
    dst = faces[:, [1, 2, 0]].ravel()
    w = np.linalg.norm(coords[src] - coords[dst], axis=1)
    g = csr_matrix((w, (src, dst)), shape=(n, n))
    return g.maximum(g.T)


def adjacency(faces: np.ndarray, n_vertices: int):
    """Sparse boolean vertex adjacency from a triangle list."""
    from scipy import sparse
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    e = np.vstack([e, e[:, ::-1]])
    data = np.ones(len(e), bool)
    A = sparse.coo_matrix((data, (e[:, 0], e[:, 1])), shape=(n_vertices, n_vertices)).tocsr()
    A.data[:] = True
    return A


def hop_distance(seed_mask: np.ndarray, A) -> np.ndarray:
    """Mesh-hop (BFS) distance from the seed set. 0 on the seeds, -1 unreachable."""
    from scipy import sparse
    n = seed_mask.size
    dist = np.full(n, -1, np.int32)
    frontier = np.asarray(seed_mask, bool).copy()
    dist[frontier] = 0
    d = 0
    while frontier.any():
        d += 1
        nxt = (A @ sparse.csr_matrix(frontier.reshape(1, -1).astype(np.int8)).T).toarray().ravel() > 0
        nxt = nxt & (dist < 0)
        if not nxt.any():
            break
        dist[nxt] = d
        frontier = nxt
    return dist


def dilate(mask: np.ndarray, A, hops: int) -> np.ndarray:
    """Grow a boolean vertex mask by *hops* mesh steps."""
    out = np.asarray(mask, bool).copy()
    for _ in range(int(hops)):
        grown = (A @ out.astype(np.int8)) > 0
        out = out | np.asarray(grown).ravel()
    return out


def vertex_areas(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """One-third-of-incident-triangle-area per vertex (mm^2)."""
    v0, v1, v2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    tri = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    va = np.zeros(len(coords))
    for k in range(3):
        np.add.at(va, faces[:, k], tri / 3.0)
    return va


# --------------------------------------------------------------------------- #
# Decision log
# --------------------------------------------------------------------------- #


def log_decision(task: str, title: str, choice: str, alternatives: str, why: str,
                 reversible: str = "yes", affects_number: str = "no") -> None:
    """Append one entry to figures/v9/review_response/DECISIONS.md (idempotent by title)."""
    OUT.mkdir(parents=True, exist_ok=True)
    head = f"### {task} - {title}"
    entry = (f"{head}\n"
             f"Choice: {choice}\n"
             f"Alternatives: {alternatives}\n"
             f"Why: {why}\n"
             f"Reversible: {reversible}\n"
             f"Affects a reported number: {affects_number}\n\n")
    existing = DECISIONS.read_text(encoding="utf-8") if DECISIONS.exists() else \
        "# Review-response batch: decision log\n\nOne entry per judgment call, per RUNNER section 3.\n\n"
    if head in existing:
        return
    DECISIONS.write_text(existing + entry, encoding="utf-8")
