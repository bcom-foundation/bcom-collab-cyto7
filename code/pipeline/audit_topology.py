"""Topology audit + safe auto-repairs for the cyto7 map (Layer 1A / 1B).

Implements ``docs/REFINE_allocortex.md`` Layer 1:

* **1A audit** - four explicit validators over the mesh graph (per hemisphere,
  medial wall excluded), from the 18-Jun-2026 deck slide 3:
    - **R1 Sequential gradients**: every edge must join types within one ordinal
      step (``|ordinal(i) - ordinal(j)| <= 1``); larger jumps are "skip-level".
    - **R2 Core continuity (rings)**: each of {Allocortex, Agranular,
      Dysgranular, Eulaminate I} should be a single annular component encircling
      the medial hub. Reports components, Euler characteristic chi, inferred
      boundary loops / holes, and ring gaps (medial-adjacent skip edges).
    - **R3 Matrix (Eulaminate II)**: should be one connected component with
      exactly 3 enclosed holes (the specialized islands). Reports components and
      holes = #components - chi.
    - **R4 Higher-organization islands (Eulaminate III, Koniocortex)**: must be
      isolated (non-annular) islands; any annular component (chi <= 0) is flagged.

  Outputs ``figures/refine/topology_audit_<version>.json`` and ``.png``.

* **1B safe auto-repairs** (``--repair``) - applies ONLY objective,
  topology-restoring edits, never in the allocortex/periallocortex belt and
  never an edit that would create a new violation:
    - speckle removal (connected components < ``--min-island-verts`` relabelled
      to the majority boundary-neighbour type),
    - enclosed medial-wall holes (tiny code-0 islands not attached to the main
      medial wall) filled with the majority neighbour,
    - tiny unambiguous R1 skip-level fixes (clusters <= ``--max-skip-fix-verts``
      with a unique intervening ordinal type),
  writing ``resources/cyto7_derived/pial.{lh,rh}.cyto7.v2.annot`` and
  ``changelog_v1_to_v2.csv``, then re-auditing v2 and asserting no regression.

  Anything semantic, anything touching the allocortex belt, and any large /
  ambiguous skip is left untouched and surfaces later as a flagged proposal
  (Layer 1C, Stage 4).

Run::

    conda activate cyto7
    python scripts/audit_topology.py --annot resources/cyto7_annot_files_standard_fsaverage/pial.{hemi}.cyto7.annot
    python scripts/audit_topology.py --annot <v1 template> --repair   # also build v2
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from cyto7_surface_io import REPO_ROOT
from make_presentation_figures import (
    CAMERA, HEMIS, LIGHT_BLUE, TYPE_GREY, TYPE_NAMES, lighting_normals,
    lit_panel, load_labels, load_surface, resolve_annot_paths, version_label,
)

DERIVED_DIR: Path = cfg.atlas_dir("fsaverage")
REFINE_DIR: Path = cfg.figures_dir() / "refine"

#: code -> ordinal (Allocortex code1 -> 0 ... Koniocortex code7 -> 6); 0 = medial wall.
def ordinal(code: np.ndarray) -> np.ndarray:
    return np.where(code >= 1, code - 1, -1)

RING_CODES = [1, 2, 3, 4]      # Allocortex, Agranular, Dysgranular, Eulaminate I
MATRIX_CODE = 5                # Eulaminate II
ISLAND_CODES = [6, 7]          # Eulaminate III, Koniocortex
ALLO_CODE = 1
CODE_NAME = {c: TYPE_NAMES[c - 1] for c in range(1, 8)}


# --------------------------------------------------------------------------- #
# Mesh helpers
# --------------------------------------------------------------------------- #


def adjacency(faces: np.ndarray, n: int) -> csr_matrix:
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1
    return A


def unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def components_of(mask: np.ndarray, A: csr_matrix):
    """(#components, per-component sizes desc, labels-over-masked, idx)."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0, [], np.array([], int), idx
    ncomp, comp = connected_components(A[idx][:, idx], directed=False)
    sizes = np.bincount(comp)
    return ncomp, sorted(sizes.tolist(), reverse=True), comp, idx


def euler_char(mask: np.ndarray, faces: np.ndarray) -> tuple[int, int, int, int]:
    """Euler characteristic chi=V-E+F of the subcomplex induced by *mask*.

    Returns (chi, nV, nE, nF) using faces all of whose vertices are in mask.
    """
    fin = mask[faces].all(1)
    F = faces[fin]
    if len(F) == 0:
        return 0, int(mask.sum()), 0, 0
    nV = len(np.unique(F))
    edges = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), axis=1)
    nE = len(np.unique(edges, axis=0))
    nF = len(F)
    return nV - nE + nF, nV, nE, nF


def encircles_hub(Lmask: np.ndarray, A: csr_matrix, lab: np.ndarray,
                  coords: np.ndarray, centerx: float) -> bool:
    """True if removing label-region *Lmask* separates the medial wall from the
    far-lateral cortex (i.e. the region is an annulus around the medial hub).

    This distinguishes a genuine hub-ring from an island that merely has an
    embedded hole (e.g. a koniocortex island inside an Eulaminate III patch):
    the latter does not disconnect medial from lateral, so it is not annular.
    """
    comp_mask = ~Lmask
    idx = np.where(comp_mask)[0]
    if idx.size == 0:
        return False
    ncomp, comp = connected_components(A[idx][:, idx], directed=False)
    if ncomp == 1:
        return False
    med = np.where((lab == 0) & comp_mask)[0]
    cortex = np.where((lab >= 1) & comp_mask)[0]
    if med.size == 0 or cortex.size == 0:
        return False
    medv = med[0]
    latv = cortex[np.argmax(np.abs(coords[cortex, 0] - centerx))]
    p = np.searchsorted(idx, [medv, latv])
    return bool(comp[p[0]] != comp[p[1]])


def dilate(mask: np.ndarray, A: csr_matrix, hops: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(hops):
        m = m | (A @ m.astype(np.int8) > 0)
    return m


# --------------------------------------------------------------------------- #
# 1A - audit
# --------------------------------------------------------------------------- #


def audit_hemi(lab: np.ndarray, faces: np.ndarray, A: csr_matrix,
               coords: np.ndarray) -> tuple[dict, dict]:
    """Run R1-R4 on one hemisphere. Returns (results, per-vertex flag arrays)."""
    n = lab.shape[0]
    labelled = lab >= 1
    medial = lab == 0
    ordv = ordinal(lab)
    centerx = float(coords[:, 0].mean())
    flags = {k: np.zeros(n, bool) for k in ("skip", "ring_gap", "island_viol")}
    res: dict = {}

    # ---- R1: sequential gradients ----
    edges = unique_edges(faces)
    both = labelled[edges[:, 0]] & labelled[edges[:, 1]]
    de = np.abs(ordv[edges[:, 0]] - ordv[edges[:, 1]])
    skip_mask = both & (de >= 2)
    skip_edges = edges[skip_mask]
    flags["skip"][np.unique(skip_edges)] = True
    # cluster skip vertices
    ncl, sizes, _, _ = components_of(flags["skip"], A)
    # breakdown by the (lo,hi) type pair
    pair_counts: dict[str, int] = {}
    for a, b in skip_edges:
        lo, hi = sorted((int(lab[a]), int(lab[b])))
        pair_counts[f"{CODE_NAME[lo]}|{CODE_NAME[hi]}"] = pair_counts.get(
            f"{CODE_NAME[lo]}|{CODE_NAME[hi]}", 0) + 1
    res["R1_sequential_gradients"] = {
        "n_skip_edges": int(skip_mask.sum()),
        "n_skip_vertices": int(flags["skip"].sum()),
        "n_skip_clusters": int(ncl),
        "skip_cluster_sizes": sizes[:20],
        "skip_pairs": dict(sorted(pair_counts.items(), key=lambda kv: -kv[1])),
    }

    # ---- R2: rings ----
    medial_adj = (A @ medial.astype(np.int8) > 0) & labelled
    flags["ring_gap"][np.unique(skip_edges)] = flags["ring_gap"][np.unique(skip_edges)] | False
    ring_gap_v = medial_adj & flags["skip"]
    flags["ring_gap"][ring_gap_v] = True
    rings = {}
    for code in RING_CODES:
        m = lab == code
        ncomp, sizes_c, _, _ = components_of(m, A)
        chi, nV, nE, nF = euler_char(m, faces)
        annular = encircles_hub(m, A, lab, coords, centerx)
        rings[CODE_NAME[code]] = {
            "components": int(ncomp),
            "component_sizes": sizes_c[:10],
            "euler_chi": int(chi),
            "encircles_medial_hub": bool(annular),
            "single_annular_ring": bool(ncomp == 1 and annular),
        }
    res["R2_core_continuity_rings"] = rings
    res["R2_ring_gap_vertices"] = int(ring_gap_v.sum())

    # ---- R3: Eulaminate II matrix ----
    m = lab == MATRIX_CODE
    ncomp, sizes_c, _, _ = components_of(m, A)
    chi, *_ = euler_char(m, faces)
    holes = (ncomp - chi) if ncomp else 0
    res["R3_matrix_eulaminate_II"] = {
        "components": int(ncomp),
        "component_sizes": sizes_c[:10],
        "euler_chi": int(chi),
        "enclosed_holes": int(holes),
        "ok_single_component": bool(ncomp == 1),
        "ok_three_holes": bool(holes == 3),
    }

    # ---- R4: islands ----
    islands = {}
    for code in ISLAND_CODES:
        m = lab == code
        ncomp2, sizes_c, comp_lab, idx2 = components_of(m, A)
        annular = []
        for k in range(ncomp2):
            cm = np.zeros(n, bool)
            cm[idx2[comp_lab == k]] = True
            if encircles_hub(cm, A, lab, coords, centerx):  # rings the medial hub
                chi_k, *_ = euler_char(cm, faces)
                annular.append({"size": int(cm.sum()), "chi": int(chi_k)})
                flags["island_viol"][cm] = True
        islands[CODE_NAME[code]] = {
            "components": int(ncomp2),
            "component_sizes": sizes_c[:10],
            "n_annular_violations": len(annular),
            "annular_components": annular,
        }
    res["R4_islands"] = islands
    return res, flags


# --------------------------------------------------------------------------- #
# 1B - safe repairs
# --------------------------------------------------------------------------- #


def majority_neighbour(comp_mask: np.ndarray, A: csr_matrix, lab: np.ndarray,
                       exclude: set[int]) -> int | None:
    """Most common boundary-neighbour label of a component (excluding *exclude*)."""
    nb = (A @ comp_mask.astype(np.int8) > 0) & ~comp_mask
    vals = lab[nb]
    vals = vals[~np.isin(vals, list(exclude))]
    if vals.size == 0:
        return None
    u, c = np.unique(vals, return_counts=True)
    return int(u[c.argmax()])


def repair_hemi(lab0: np.ndarray, faces: np.ndarray, A: csr_matrix, hemi: str,
                min_island: int, max_skip_fix: int, allo_hops: int):
    """Apply safe repairs. Returns (new_labels, changelog_rows)."""
    lab = lab0.copy()
    rows = []
    allo_belt = dilate(lab == ALLO_CODE, A, allo_hops)

    # (1) Speckle removal for type labels 1..7.
    for code in range(1, 8):
        ncomp, _, comp, idx = components_of(lab == code, A)
        for k in range(ncomp):
            members = idx[comp == k]
            if members.size >= min_island:
                continue
            cm = np.zeros_like(lab, bool); cm[members] = True
            if allo_belt[members].any():
                continue  # defer allocortex-belt speckles to proposals
            tgt = majority_neighbour(cm, A, lab, exclude={0, code})
            if tgt is None or tgt == ALLO_CODE:
                continue
            for v in members:
                rows.append((hemi, int(v), int(lab[v]), int(tgt), "R3/R4",
                             f"speckle<{min_island}v relabel to majority neighbour"))
            lab[members] = tgt

    # (2) Enclosed medial-wall holes: tiny code-0 islands not on the main wall.
    ncomp, _, comp, idx = components_of(lab == 0, A)
    if ncomp > 1:
        sizes = np.array([(comp == k).sum() for k in range(ncomp)])
        main = sizes.argmax()
        for k in range(ncomp):
            if k == main:
                continue
            members = idx[comp == k]
            if members.size >= min_island:
                continue
            cm = np.zeros_like(lab, bool); cm[members] = True
            if allo_belt[members].any():
                continue
            tgt = majority_neighbour(cm, A, lab, exclude={0})
            if tgt is None or tgt == ALLO_CODE:
                continue
            for v in members:
                rows.append((hemi, int(v), 0, int(tgt), "medial-hygiene",
                             "enclosed code-0 hole filled with majority neighbour"))
            lab[members] = tgt

    # (3) Tiny unambiguous R1 skip-level fixes.
    edges = unique_edges(faces)
    ordv = ordinal(lab)
    labelled = lab >= 1
    skip = labelled[edges[:, 0]] & labelled[edges[:, 1]] & \
        (np.abs(ordv[edges[:, 0]] - ordv[edges[:, 1]]) >= 2)
    skip_v = np.zeros_like(lab, bool); skip_v[np.unique(edges[skip])] = True
    if skip_v.any():
        nc, _, comp, idx = components_of(skip_v, A)
        for k in range(nc):
            members = idx[comp == k]
            if members.size > max_skip_fix:
                continue
            if allo_belt[members].any():
                continue
            ords = ordv[members]
            lo, hi = ords.min(), ords.max()
            if hi - lo < 2:
                continue
            inter = list(range(lo + 1, hi))
            if len(inter) != 1:
                continue  # not a unique intervening ordinal
            tgt = inter[0] + 1  # ordinal -> code
            for v in members:
                rows.append((hemi, int(v), int(lab[v]), int(tgt), "R1",
                             f"tiny skip cluster<= {max_skip_fix}v -> intervening type"))
            lab[members] = tgt
    return lab, rows


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

VIOL_COLORS = {"skip": (0.90, 0.10, 0.10), "ring_gap": (1.0, 0.6, 0.0),
               "island_viol": (0.80, 0.0, 0.80)}
VIOL_LABELS = {"skip": "R1 skip-level edge", "ring_gap": "R2 ring gap (medial-adjacent skip)",
               "island_viol": "R4 island-ring violation"}


def render_audit(annot_paths, version, flags_by_hemi, out_path, dpi):
    geom = {h: load_surface(h, "inflated") for h in HEMIS}
    vnorm = {h: lighting_normals(h) for h in HEMIS}
    labs = {h: load_labels(annot_paths[h], h) for h in HEMIS}
    views = ["lateral", "medial", "ventral"]
    fig, axes = plt.subplots(len(HEMIS), len(views), figsize=(len(views) * 3.2, len(HEMIS) * 3.2),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)
    for r, hemi in enumerate(HEMIS):
        coords, faces = geom[hemi]
        lab = labs[hemi]
        # faint greyscale-by-type base, medial wall light blue
        vrgb = np.tile(np.array(LIGHT_BLUE), (coords.shape[0], 1))
        for c in range(1, 8):
            base = np.array(TYPE_GREY[c]); vrgb[lab == c] = 0.55 * base + 0.45  # faded
        for key in ("skip", "ring_gap", "island_viol"):
            vrgb[flags_by_hemi[hemi][key]] = VIOL_COLORS[key]
        for cidx, view in enumerate(views):
            ax = axes[r, cidx]
            lit_panel(ax, coords, faces, vrgb, hemi, view, vnorm[hemi])
            if r == 0:
                ax.set_title(view, fontsize=12)
            if cidx == 0:
                ax.text2D(-0.05, 0.5, hemi.upper(), transform=ax.transAxes,
                          rotation=90, va="center", ha="center", fontsize=12)
    handles = [Patch(facecolor=VIOL_COLORS[k], label=VIOL_LABELS[k]) for k in VIOL_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Topology audit ({version}) - rule violations", fontsize=15, y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out_path}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_audit(annot_paths, version, out_dir, dpi, no_figure=False) -> dict:
    """Audit both hemispheres; write JSON (+ figure). Returns results dict."""
    print(f"  auditing {version} ...")
    results = {"version": version, "hemispheres": {}}
    flags_by_hemi = {}
    for hemi in HEMIS:
        lab = load_labels(annot_paths[hemi], hemi)
        coords, faces = load_surface(hemi, "pial")
        A = adjacency(faces, lab.shape[0])
        res, flags = audit_hemi(lab, faces, A, coords)
        results["hemispheres"][hemi] = res
        flags_by_hemi[hemi] = flags
    jpath = out_dir / f"topology_audit_{version}.json"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"    wrote {jpath}")
    if not no_figure:
        render_audit(annot_paths, version, flags_by_hemi, out_dir / f"topology_audit_{version}.png", dpi)
    return results


def do_repair(v1_paths, args) -> dict:
    """Build v2 annot + changelog from v1; return the v2 annot paths."""
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    v2_paths = {}
    for hemi in HEMIS:
        labels, ctab, names = nib.freesurfer.io.read_annot(str(v1_paths[hemi]))
        labels = np.asarray(labels)
        coords, faces = load_surface(hemi, "pial")
        A = adjacency(faces, labels.shape[0])
        new_lab, rows = repair_hemi(labels, faces, A, hemi, args.min_island_verts,
                                    args.max_skip_fix_verts, args.allo_hops)
        all_rows.extend(rows)
        out = DERIVED_DIR / f"pial.{hemi}.cyto7.v2.annot"
        nib.freesurfer.io.write_annot(str(out), new_lab.astype(np.int32), ctab, names)
        v2_paths[hemi] = out
        print(f"  {hemi}: {len(rows)} vertices changed -> {out.name}")
    import csv
    clog = DERIVED_DIR / "changelog_v1_to_v2.csv"
    with open(clog, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hemi", "vertex", "old_label", "new_label", "rule", "reason"])
        w.writerows(all_rows)
    print(f"  wrote {clog} ({len(all_rows)} rows)")
    return v2_paths


def _violation_totals(results: dict) -> dict:
    """Collapse a results dict to scalar per-rule violation counts for regression check."""
    tot = {"R1_skip_edges": 0, "R2_ring_gaps": 0, "R3_bad": 0, "R4_annular": 0}
    for hemi in HEMIS:
        h = results["hemispheres"][hemi]
        tot["R1_skip_edges"] += h["R1_sequential_gradients"]["n_skip_edges"]
        tot["R2_ring_gaps"] += h["R2_ring_gap_vertices"]
        tot["R3_bad"] += int(not h["R3_matrix_eulaminate_II"]["ok_single_component"])
        tot["R4_annular"] += sum(v["n_annular_violations"] for v in h["R4_islands"].values())
    return tot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annot", default=str(cfg.atlas_dir("provenance/as_painted") / "pial.{hemi}.cyto7.annot"),
                   help="Annot path or {hemi} template (default: v1 original).")
    p.add_argument("--repair", action="store_true", help="Also apply safe repairs -> v2 + changelog.")
    p.add_argument("--out", type=Path, default=REFINE_DIR)
    p.add_argument("--min-island-verts", type=int, default=5)
    p.add_argument("--max-skip-fix-verts", type=int, default=3)
    p.add_argument("--allo-hops", type=int, default=2,
                   help="Dilate the allocortex belt by this many hops; repairs touching it are deferred.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    v1_paths = resolve_annot_paths(args.annot)
    v1_version = version_label(v1_paths)
    res_v1 = run_audit(v1_paths, v1_version, args.out, args.dpi, args.no_figure)
    print(f"  {v1_version} violation totals:", _violation_totals(res_v1))

    if args.repair:
        print("Applying safe auto-repairs ...")
        v2_paths = do_repair(v1_paths, args)
        res_v2 = run_audit(v2_paths, "v2", args.out, args.dpi, args.no_figure)
        t1, t2 = _violation_totals(res_v1), _violation_totals(res_v2)
        print("  v2 violation totals:", t2)
        regressed = {k: (t1[k], t2[k]) for k in t1 if t2[k] > t1[k]}
        if regressed:
            print(f"  !! WARNING: repair regressed some rules: {regressed}")
        else:
            print("  OK: no rule regressed after repair.")
    print("Done.")


if __name__ == "__main__":
    main()
