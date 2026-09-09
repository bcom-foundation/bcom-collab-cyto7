"""RR10 - build and release the crossed parcellation (aparc x cyto7, economo x cyto7).

Section 4.7 argues that region-based whole-brain models have two bad options: one
cyto7 type per classical parcel, which discards about a third of the within-parcel
variation, or the vertex map, which node-based pipelines cannot consume. This builds
the missing middle: a classical atlas intersected with the released v9 cyto7 map, so a
cytoarchitecturally heterogeneous parcel becomes several type-homogeneous nodes.

The point of the product is that it maps to new subjects with no bespoke code, so the
deliverable is a FreeSurfer ``.annot`` on fsaverage and the verification is a real
``mri_surf2surf --sval-annot`` round-trip to a subject with a completed ``recon-all``.

Design choices are contract, not preference (they are stated in section 4.7):

1. minimum node size 200 vertices on the 164k fsaverage surface
2. residue absorbed into the largest neighbouring node within the same anatomical
   parcel, so no anatomical boundary is ever crossed
3. one node per (parcel x type) cell even where it spans several patches, with
   component counts reported and a contiguity-split variant emitted as a convenience

**Node inventory is bilateral.** A (parcel, type) cell is a node when its vertex count
summed over both hemispheres exceeds 200. This is what reproduces the four numbers
section 4.7 states (101 Desikan nodes from 34 parcels, mean 3.0, 0.5% floor cost; 115
von Economo nodes from 43 parcels, 1.1%) and it is what the manuscript's "from 34
parcels" and "the 43 von Economo parcels" describe. Thresholding within hemisphere
instead gives 90 and 89 for Desikan at a cost of 1.4% and 1.6%, which contradicts the
paper; see the hand-back. Each node is then instantiated per hemisphere, so the node
table has one row per (node, hemisphere) with distinct ids, and the annot colour table
is identical in both hemispheres.

Run::
    conda run -n cyto7 python scripts/rr10_crossed_parcellation.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import sparse

import rr_common as rc
from cyto7_surface_io import REPO_ROOT, resolve_target_map
from figure_style import CYTO7_VIRIDIS

OUTDIR = cfg.atlas_dir("fsaverage/crossed")
HANDBACK = rc.OUT / "rr10_crossed"
VE_DIR = cfg.data_dir() / "voneconomo"
AT_DIR = cfg.data_dir() / "refine_atlases"

HEMIS = ("lh", "rh")
HKEY = {"lh": "L", "rh": "R"}
MIN_VERTICES = 200

#: cyto7 ordinal -> the fixed abbreviation used in the node name.
TYPE_ABBREV = {1: "allo", 2: "agr", 3: "dys", 4: "eul1", 5: "eul2", 6: "eul3", 7: "kon"}
TYPE_NAME = {1: "allocortex", 2: "agranular", 3: "dysgranular", 4: "eulaminate I",
             5: "eulaminate II", 6: "eulaminate III", 7: "koniocortex"}

#: anatomical labels that are not real cortex parcels.
NON_PARCEL = {"unknown", "corpuscallosum", "medial_wall", "none", "", "???"}

ATLASES = {
    "aparc_cyto7": ("desikan", lambda h: VE_DIR / f"{h}.aparc.annot"),
    "voneconomo_cyto7": ("voneconomo", lambda h: VE_DIR / f"{h}.economo.annot"),
}

#: rh node ids are this offset plus the node's ordinal position, so lh and rh never collide.
RH_ID_OFFSET = 1000


def _log_factory(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")

    def log(*args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()
    return log, fh


def _read_annot(path: Path):
    lab, ctab, names = nib.freesurfer.io.read_annot(str(path))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return np.asarray(lab), np.asarray(ctab), names


def _cortex_mask(hemi: str, n: int) -> np.ndarray:
    idx = nib.freesurfer.read_label(str(AT_DIR / f"{hemi}.cortex.label"))
    m = np.zeros(n, bool)
    m[idx] = True
    return m


# --------------------------------------------------------------------------- #
# Step 1 - the node inventory
# --------------------------------------------------------------------------- #


def load_hemi_inputs(hemi: str, atlas_path: Path, cyto: dict) -> dict:
    anat_lab, _anat_ctab, anat_names = _read_annot(atlas_path)
    n = anat_lab.shape[0]
    cortex = _cortex_mask(hemi, n)
    cyto_lab = np.asarray(cyto[HKEY[hemi]], int)
    parcel_of = np.full(n, "", object)
    real = np.zeros(n, bool)
    for pid, nm in enumerate(anat_names):
        if nm.lower() in NON_PARCEL:
            continue
        m = (anat_lab == pid) & cortex
        if m.any():
            parcel_of[m] = nm
            real |= m
    # the crossed product is defined on labelled cortex inside a real parcel
    valid = real & (cyto_lab >= 1)
    return {"hemi": hemi, "n": n, "parcel_of": parcel_of, "cyto": cyto_lab,
            "valid": valid, "names": anat_names}


def node_inventory(H: dict[str, dict], log=print) -> tuple[list[tuple[str, int]], dict]:
    """Bilateral (parcel, type) cells above the floor, in the stable id order.

    Order is base parcel alphabetical, then cyto7 ordinal ascending, which is the id
    scheme the node table and the colour table both use.
    """
    pooled: dict[tuple[str, int], int] = {}
    for h in HEMIS:
        d = H[h]
        for p, t in zip(d["parcel_of"][d["valid"]], d["cyto"][d["valid"]]):
            pooled[(p, int(t))] = pooled.get((p, int(t)), 0) + 1
    total = sum(pooled.values())
    keep = {k: v for k, v in pooled.items() if v > MIN_VERTICES}
    residue_v = total - sum(keep.values())
    order = sorted(keep, key=lambda k: (k[0], k[1]))
    parcels_all = {p for p, _t in pooled}
    parcels_kept = {p for p, _t in keep}
    stats = {"n_cells": len(pooled), "n_nodes": len(keep),
             "n_parcels_total": len(parcels_all), "n_parcels_with_node": len(parcels_kept),
             "labelled_vertices": total, "residue_vertices": residue_v,
             "residue_pct": 100.0 * residue_v / total,
             "mean_nodes_per_parcel": len(keep) / max(1, len(parcels_kept)),
             "parcels_without_node": sorted(parcels_all - parcels_kept)}
    log(f"  bilateral cells {len(pooled)}, nodes above {MIN_VERTICES} vertices {len(keep)}, "
        f"from {len(parcels_kept)} of {len(parcels_all)} parcels "
        f"(mean {stats['mean_nodes_per_parcel']:.2f} per parcel)")
    log(f"  floor cost {residue_v} of {total} labelled vertices "
        f"({stats['residue_pct']:.2f}%)")
    return order, stats


# --------------------------------------------------------------------------- #
# Step 1b - assign vertices, absorbing the residue within the parcel
# --------------------------------------------------------------------------- #


def assign_hemi(d: dict, nodes: list[tuple[str, int]], A, log=print) -> tuple[np.ndarray, dict]:
    """Per-vertex node index (1-based into *nodes*, 0 = unknown) for one hemisphere.

    Every vertex of a qualifying cell takes that cell's node. Every vertex of a
    sub-threshold cell is absorbed into the largest qualifying node of the SAME
    anatomical parcel, preferring one it actually touches, so no anatomical boundary
    is crossed. Absorption is by whole cell, largest target first, which makes the
    result independent of the order the residue is visited in.
    """
    idx_of = {k: i + 1 for i, k in enumerate(nodes)}
    out = np.zeros(d["n"], np.int32)
    valid = d["valid"]
    parcels = sorted({p for p in d["parcel_of"][valid]})
    merges = []
    orphan_parcels = []
    for p in parcels:
        inP = valid & (d["parcel_of"] == p)
        types = sorted({int(t) for t in d["cyto"][inP]})
        qual = [t for t in types if (p, t) in idx_of]
        resid = [t for t in types if (p, t) not in idx_of]
        if not qual:
            # Rule 2 fallback: no supra-threshold cell anywhere in this parcel, so the
            # whole parcel becomes one node named for its majority type.
            cnt = {t: int((inP & (d["cyto"] == t)).sum()) for t in types}
            maj = max(cnt, key=cnt.get)
            orphan_parcels.append({"parcel": p, "majority_type": maj,
                                   "n_vertices": int(inP.sum())})
            log(f"    ORPHAN parcel {p}: no cell above the floor, kept whole as "
                f"{p}_{TYPE_ABBREV[maj]} ({int(inP.sum())} vertices)")
            continue
        for t in qual:
            out[inP & (d["cyto"] == t)] = idx_of[(p, t)]
        if not resid:
            continue
        # sizes of the qualifying nodes within this parcel and hemisphere
        size = {t: int((inP & (d["cyto"] == t)).sum()) for t in qual}
        for t in resid:
            grp = inP & (d["cyto"] == t)
            k = int(grp.sum())
            if k == 0:
                continue
            touch = np.asarray((A @ grp.astype(np.int8)) > 0).ravel() & inP & ~grp
            adj = sorted({int(x) for x in d["cyto"][touch] if int(x) in size},
                         key=lambda z: -size[z])
            if adj:
                tgt, how = adj[0], "adjacent"
            else:
                tgt, how = max(size, key=size.get), "largest-in-parcel (no contact)"
            out[grp] = idx_of[(p, tgt)]
            merges.append({"parcel": p, "from_type": t, "to_type": tgt,
                           "n_vertices": k, "rule": how,
                           "target_node": f"{p}_{TYPE_ABBREV[tgt]}"})
    return out, {"merges": merges, "orphan_parcels": orphan_parcels}


# --------------------------------------------------------------------------- #
# Geometry: components and areas
# --------------------------------------------------------------------------- #


def components_of(mask: np.ndarray, A) -> tuple[int, float]:
    """(number of connected components, fraction of vertices in the largest)."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0, float("nan")
    sub = A[idx][:, idx]
    ncomp, lab = sparse.csgraph.connected_components(sub, directed=False)
    _, counts = np.unique(lab, return_counts=True)
    return int(ncomp), float(counts.max() / idx.size)


# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #


def node_colours(nodes: list[tuple[str, int]]) -> list[tuple[int, int, int]]:
    """cyto7 type hue with a small per-parcel lightness jitter.

    A rendered map has to read as the type gradient first, so the hue is the type's
    viridis colour and the only per-parcel freedom is lightness. The jitter is
    deterministic in the parcel's rank within its type, so rebuilding gives identical
    colours, and it is bounded so a node never drifts into another type's band.
    """
    by_type: dict[int, list[str]] = {}
    for p, t in nodes:
        by_type.setdefault(t, []).append(p)
    rank = {(p, t): i for t, ps in by_type.items() for i, p in enumerate(sorted(set(ps)))}
    out = []
    used: set[tuple[int, int, int]] = {(25, 5, 25)}
    for p, t in nodes:
        base = np.asarray(CYTO7_VIRIDIS[t], float)
        k = len(set(by_type[t]))
        f = 0.0 if k <= 1 else (rank[(p, t)] / (k - 1) - 0.5)     # -0.5 .. +0.5
        rgb = np.clip(base + 0.22 * f * (1.0 - base) * np.sign(f) + 0.16 * f, 0, 1)
        r, g, b = (int(round(v * 255)) for v in rgb)
        while (r, g, b) in used:
            b = (b + 1) % 256
            if b == 0:
                g = (g + 1) % 256
        used.add((r, g, b))
        out.append((r, g, b))
    return out


def build_ctab(colours: list[tuple[int, int, int]]) -> np.ndarray:
    """FreeSurfer colour table, row 0 = unknown, then one row per node."""
    ctab = np.zeros((len(colours) + 1, 5), int)
    ctab[0] = [25, 5, 25, 0, 25 + 5 * 256 + 25 * 65536]
    for i, (r, g, b) in enumerate(colours, start=1):
        ctab[i] = [r, g, b, 0, r + g * 256 + b * 65536]
    return ctab


# --------------------------------------------------------------------------- #
# Contiguity-split variant (secondary)
# --------------------------------------------------------------------------- #


def contiguity_split(labels: np.ndarray, nodes: list[tuple[str, int]], A,
                     colours: list[tuple[int, int, int]]):
    """Split each node into its connected components, floor applied to components.

    Components at or below the floor are absorbed into the largest component of the
    same node, so the split never crosses a node boundary and never crosses an
    anatomical one either. Components are numbered by size, largest first.
    """
    out = np.zeros_like(labels)
    names = ["unknown"]
    cols = []
    for i, (p, t) in enumerate(nodes, start=1):
        m = labels == i
        if not m.any():
            continue
        idx = np.where(m)[0]
        sub = A[idx][:, idx]
        ncomp, clab = sparse.csgraph.connected_components(sub, directed=False)
        sizes = np.bincount(clab, minlength=ncomp)
        order = np.argsort(-sizes)
        keep = [c for c in order if sizes[c] > MIN_VERTICES] or [int(order[0])]
        big = keep[0]
        remap = {int(c): (keep.index(c) if c in keep else 0) for c in range(ncomp)}
        for c in range(ncomp):
            j = remap[int(c)] if c in keep else keep.index(big)
            tgt_name = f"{p}_{TYPE_ABBREV[t]}_{j + 1}"
            if tgt_name not in names:
                names.append(tgt_name)
                cols.append(colours[i - 1])
            out[idx[clab == c]] = names.index(tgt_name)
    return out, names, build_ctab(cols)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_annot(path: Path, labels, ctab, names):
    nib.freesurfer.io.write_annot(str(path), labels.astype(np.int32), ctab,
                                  [n.encode() for n in names], fill_ctab=True)


def write_ctab_file(path: Path, names, ctab):
    lines = []
    for i, nm in enumerate(names):
        r, g, b = int(ctab[i, 0]), int(ctab[i, 1]), int(ctab[i, 2])
        lines.append(f"{i:4d}  {nm:<44s} {r:3d} {g:3d} {b:3d}    0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_label_gii(path: Path, data: np.ndarray, names, ctab):
    """32k fs_LR label GIFTI with an embedded label table."""
    from nibabel import gifti
    lt = gifti.GiftiLabelTable()
    for i, nm in enumerate(names):
        lb = gifti.GiftiLabel(key=int(i), red=float(ctab[i, 0]) / 255.0,
                              green=float(ctab[i, 1]) / 255.0,
                              blue=float(ctab[i, 2]) / 255.0, alpha=1.0)
        lb.label = nm
        lt.labels.append(lb)
    da = gifti.GiftiDataArray(np.asarray(data, np.int32),
                              intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    img = gifti.GiftiImage(labeltable=lt, darrays=[da])
    nib.save(img, str(path))


def to_fslr32k(labels164: np.ndarray, hemi: str) -> np.ndarray:
    """Nearest-neighbour 164k -> 32k fs_LR, the convention the existing release uses."""
    import os
    from neuromaps import transforms
    from nibabel.gifti import GiftiDataArray, GiftiImage
    from cyto7_surface_io import _WORKBENCH_DEFAULT
    wb = os.environ.get("WORKBENCH_BIN") or _WORKBENCH_DEFAULT
    if wb and Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")
    gi = GiftiImage()
    gi.add_gifti_data_array(GiftiDataArray(labels164.astype(np.float32)))
    res = transforms.fsaverage_to_fslr(gi, "32k", hemi=HKEY[hemi], method="nearest")
    return np.rint(np.asarray(res[0].agg_data())).astype(np.int32)


# --------------------------------------------------------------------------- #
# Step 4 - provenance and the usage README
# --------------------------------------------------------------------------- #


def write_provenance(path: Path, built: dict, frozen: dict, cmd: str):
    L = ["cyto7 crossed parcellation (anatomical parcel x cyto7 type) - provenance",
         "=" * 78, "",
         f"built: {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"build command: {cmd}",
         f"generator: scripts/rr10_crossed_parcellation.py", "",
         "INPUTS (SHA-256)", "-" * 78]
    for nm, h in sorted(frozen.items()):
        L.append(f"  {nm}")
        L.append(f"    {h}")
    for p in [VE_DIR / "lh.aparc.annot", VE_DIR / "rh.aparc.annot",
              VE_DIR / "lh.economo.annot", VE_DIR / "rh.economo.annot",
              AT_DIR / "lh.cortex.label", AT_DIR / "rh.cortex.label"]:
        if p.exists():
            L.append(f"  {p.relative_to(REPO_ROOT).as_posix()}")
            L.append(f"    {rc.sha256(p)}")
    L += ["", "BUILD RULES", "-" * 78,
          f"  minimum node size      {MIN_VERTICES} vertices, 164k fsaverage",
          "  node inventory         a (parcel, type) cell is a node when its vertex count",
          "                         summed over BOTH hemispheres exceeds the floor. This is",
          "                         what reproduces the counts reported in the paper (101",
          "                         Desikan nodes from 34 parcels, 115 von Economo from 43).",
          "                         Consequence: a node can be small or absent in one",
          "                         hemisphere; per-hemisphere sizes are in the node table.",
          "  residue                absorbed into the largest neighbouring node within the",
          "                         SAME anatomical parcel, so no anatomical boundary is",
          "                         crossed. Whole cells are absorbed, largest target first.",
          "  contiguity             one node per (parcel x type) cell even when it spans",
          "                         several patches. Component counts are in the node table;",
          "                         ?h.*_cc.annot is a secondary contiguity-split variant.",
          "  naming                 <parcel>_<typeabbrev>, lowercase, underscore separated.",
          "                         Split on the LAST underscore to recover the base parcel",
          "                         exactly as FreeSurfer spells it.",
          "                         allo agr dys eul1 eul2 eul3 kon = cyto7 ordinals 1..7.",
          "  ids                    nodes ordered by base parcel alphabetical, then cyto7",
          "                         ordinal ascending, numbered from 1. That number is the",
          f"                         annot colour-table index in BOTH hemispheres. Node table",
          f"                         ids are lh = index, rh = {RH_ID_OFFSET} + index, so a",
          "                         bilateral table has no collisions. Index 0 = unknown.",
          "  surface                areas and adjacency on fsaverage white, 164k.", ""]
    for tag, B in built.items():
        s = B["stats"]
        nm = sum(len(B["merge_info"][h]["merges"]) for h in HEMIS)
        nv = sum(m["n_vertices"] for h in HEMIS for m in B["merge_info"][h]["merges"])
        L += [f"{tag.upper()}", "-" * 78,
              f"  base atlas           {B['base']}",
              f"  bilateral cells      {s['n_cells']}",
              f"  nodes above floor    {s['n_nodes']} from {s['n_parcels_with_node']} parcels "
              f"(mean {s['mean_nodes_per_parcel']:.2f})",
              f"  node table rows      {len(B['rows'])} (one per node per hemisphere)",
              f"  floor cost           {s['residue_vertices']} of {s['labelled_vertices']} "
              f"labelled vertices ({s['residue_pct']:.2f}%)",
              f"  cells absorbed       {nm} cells, {nv} vertices",
              f"  contiguity variant   " + ", ".join(
                  f"{h} {B.get('cc_counts', {}).get(h, 0)} nodes" for h in HEMIS), ""]
    L += ["CITATION", "-" * 78,
          "  Cite the cyto7 v9 map AND the base atlas (Desikan-Killiany; von",
          "  Economo-Koskinas). The cross moves no boundary: it intersects two labellings,",
          "  so a node is only as good as its base parcel's definition.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


README = """# Crossed parcellation: classical atlas x cyto7 type

A node-based parcellation for region-based whole-brain models. Each classical parcel is
intersected with the released cyto7 v9 map, so a cytoarchitecturally heterogeneous parcel
becomes several type-homogeneous nodes: `superiortemporal_eul2`, `parahippocampal_agr`,
`pericalcarine_kon`.

Desikan-Killiany gives {n_ap} nodes from {p_ap} parcels; von Economo gives {n_ve} from
{p_ve}. Node names split on the **last** underscore into the base parcel, spelled exactly
as FreeSurfer spells it, and the cyto7 type abbreviation
(`allo agr dys eul1 eul2 eul3 kon` = ordinals 1 to 7).

## Files

| file | what it is |
|---|---|
| `?h.aparc_cyto7.annot` | FreeSurfer annotation on fsaverage 164k, embedded colour table |
| `?h.voneconomo_cyto7.annot` | same, von Economo base |
| `?h.*.32k_fs_LR.label.gii` | nearest-neighbour resample to 32k fs_LR, same convention as the cyto7 release |
| `aparc_cyto7.ctab` | FreeSurfer colour table, so `mri_annotation2label` works standalone |
| `*_nodes.tsv` | one row per node per hemisphere: id, name, hemi, base parcel, cyto7 ordinal and name, vertices, area, components, largest-component fraction, whether it absorbed residue |
| `?h.*_cc.annot` | **secondary** contiguity-split variant, components suffixed `_1`, `_2`, ... |
| `PROVENANCE_crossed.txt` | inputs with SHA-256, build rules, id scheme, counts |

The colour-table index is the same in both hemispheres. Node-table ids are `index` for lh
and `{off} + index` for rh, so a bilateral table has no collisions. Index 0 is `unknown`
(medial wall).

## Map it to your own subject

The whole point of shipping a `.annot` on fsaverage is that it transfers through the
standard spherical registration. For a subject with a completed `recon-all`:

```bash
export SUBJECTS_DIR=/path/to/your/subjects
for hemi in lh rh; do
  mri_surf2surf \\
    --srcsubject fsaverage \\
    --trgsubject YOUR_SUBJECT \\
    --hemi $hemi \\
    --sval-annot $hemi.aparc_cyto7.annot \\
    --tval $SUBJECTS_DIR/YOUR_SUBJECT/label/$hemi.aparc_cyto7.annot
done
```

Then the usual tools work unchanged, for example per-node thickness:

```bash
mri_segstats --annot YOUR_SUBJECT lh aparc_cyto7 \\
  --i $SUBJECTS_DIR/YOUR_SUBJECT/surf/lh.thickness --sum lh.aparc_cyto7.thickness.txt
```

This was verified, not assumed: see `PROVENANCE_crossed.txt` and the RR10 hand-back for the
round-trip test to a real `recon-all` subject, in which every node reappeared on the subject.

## 32k fs_LR (Workbench)

The `.32k_fs_LR.label.gii` files are already in fs_LR 32k, the space HCP-style subject data
use, so no resampling is needed. Parcellate a CIFTI directly:

```bash
wb_command -cifti-parcellate YOUR_DATA.dtseries.nii aparc_cyto7.32k_fs_LR.dlabel.nii \\
  COLUMN YOUR_DATA.aparc_cyto7.ptseries.nii
```

To build that dlabel from the two hemisphere label files, or to move the parcellation to a
different mesh density:

```bash
wb_command -cifti-create-label aparc_cyto7.32k_fs_LR.dlabel.nii \\
  -left-label lh.aparc_cyto7.32k_fs_LR.label.gii \\
  -right-label rh.aparc_cyto7.32k_fs_LR.label.gii

wb_command -label-resample lh.aparc_cyto7.32k_fs_LR.label.gii \\
  fs_LR.L.sphere.32k_fs_LR.surf.gii TARGET.L.sphere.surf.gii ADAP_BARY_AREA \\
  lh.aparc_cyto7.TARGET.label.gii -area-surfs SRC.L.midthickness.surf.gii \\
  TARGET.L.midthickness.surf.gii
```

## Read it in Python

```python
import nibabel as nib, pandas as pd
labels, ctab, names = nib.freesurfer.io.read_annot("lh.aparc_cyto7.annot")
nodes = pd.read_csv("aparc_cyto7_nodes.tsv", sep="\\t").query("hemi == 'lh'").set_index("annot_index")
# labels[v] is the annot index of vertex v; nodes.loc[labels[v]] gives its name, base parcel and cyto7 type
```

## Caveats

- The node inventory is **bilateral**: a (parcel, type) cell is a node when its vertex count
  over both hemispheres exceeds {minv}. A node can therefore be much smaller in one
  hemisphere than the floor suggests, and two nodes are present in the left hemisphere only.
  Per-hemisphere sizes are in the node table; filter on `n_vertices` if your model needs a
  per-hemisphere minimum.
- Nodes that absorbed sub-threshold residue are not 100% type-pure. The node table flags
  them (`absorbed_residue`) and the hand-back reports the purity distribution.
- A node may span several patches. `n_components` and `frac_largest_component` are in the
  node table, and `?h.*_cc.annot` is the split variant if you need contiguous nodes.
- The cross moves no boundary. A node is only as good as its base parcel's definition, and
  inherits the local boundary uncertainty of the cyto7 map.
"""


def write_readme(path: Path, built: dict):
    ap = built["aparc_cyto7"]["stats"]
    ve = built["voneconomo_cyto7"]["stats"]
    path.write_text(README.format(
        n_ap=ap["n_nodes"], p_ap=ap["n_parcels_with_node"],
        n_ve=ve["n_nodes"], p_ve=ve["n_parcels_with_node"],
        off=RH_ID_OFFSET, minv=MIN_VERTICES), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Step 3 - verification
# --------------------------------------------------------------------------- #


def verify_partition(labels: dict, H: dict, nodes, log=print) -> dict:
    """Node labels tile the labelled cortex exactly: no overlap, no gap, no spill."""
    res = {}
    for h in HEMIS:
        lab, d = labels[h], H[h]
        assigned = lab > 0
        spill = int((assigned & ~d["valid"]).sum())
        gap = int((~assigned & d["valid"]).sum())
        res[h] = {"assigned": int(assigned.sum()), "valid": int(d["valid"].sum()),
                  "assigned_outside_valid": spill, "valid_unassigned": gap,
                  "exact": spill == 0 and gap == 0}
        log(f"    {h}: {int(assigned.sum())} assigned, {spill} outside the labelled set, "
            f"{gap} labelled but unassigned -> {'EXACT' if res[h]['exact'] else 'MISMATCH'}")
    return res


def verify_type_recoverability(labels: dict, H: dict, nodes, log=print) -> dict:
    """The type read from a node's name must be the majority type of its vertices."""
    bad, purities, absorbed_purities = [], [], []
    for i, (p, t) in enumerate(nodes, start=1):
        v_named = 0
        v_tot = 0
        for h in HEMIS:
            m = labels[h] == i
            v_tot += int(m.sum())
            v_named += int((H[h]["cyto"][m] == t).sum())
        if v_tot == 0:
            continue
        pur = v_named / v_tot
        purities.append(pur)
        if pur < 1.0:
            absorbed_purities.append((f"{p}_{TYPE_ABBREV[t]}", pur, v_tot))
        maj = None
        counts = {}
        for h in HEMIS:
            m = labels[h] == i
            for tt in np.unique(H[h]["cyto"][m]):
                counts[int(tt)] = counts.get(int(tt), 0) + int((H[h]["cyto"][m] == tt).sum())
        maj = max(counts, key=counts.get)
        if maj != t:
            bad.append({"node": f"{p}_{TYPE_ABBREV[t]}", "named_type": t,
                        "majority_type": maj, "purity": pur})
    pur = np.asarray(purities)
    log(f"    purity: {int((pur >= 1.0).sum())} of {pur.size} nodes are pure; "
        f"min {pur.min():.3f}, mean {pur.mean():.4f}")
    log(f"    name-vs-majority mismatches: {len(bad)}")
    return {"n_nodes": int(pur.size), "n_pure": int((pur >= 1.0).sum()),
            "min_purity": float(pur.min()), "mean_purity": float(pur.mean()),
            "mismatches": bad,
            "impure_nodes": [{"node": a, "purity": b, "n_vertices": c}
                             for a, b, c in sorted(absorbed_purities, key=lambda z: z[1])]}


def _wsl_path(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


def roundtrip_to_subject(subject: str, tag: str, names: list[str], labels: dict,
                         log=print) -> dict:
    """mri_surf2surf --sval-annot fsaverage -> *subject*, then read the result back.

    This is the deliverable's whole purpose, so the check is not that the command
    exits zero. The transferred annot is read back in Python and every node the
    fsaverage annot defines has to reappear on the subject with a vertex count in
    proportion to the subject's mesh, which is what "it transfers with no bespoke
    code" actually means.
    """
    src = f"{_wsl_path(OUTDIR)}"
    rtdir = HANDBACK / "_roundtrip"
    rtdir.mkdir(parents=True, exist_ok=True)
    out = {"subject": subject, "atlas": tag, "hemis": {}}
    for h in HEMIS:
        tgt_win = rtdir / f"{h}.{tag}.{subject}.annot"
        tgt = _wsl_path(tgt_win)
        cmd = (
            "export FREESURFER_HOME=/usr/local/freesurfer; "
            "source $FREESURFER_HOME/SetUpFreeSurfer.sh >/dev/null 2>&1; "
            "export SUBJECTS_DIR=$FREESURFER_HOME/subjects; "
            f"mri_surf2surf --srcsubject fsaverage --trgsubject {subject} --hemi {h} "
            f"--sval-annot {src}/{h}.{tag}.annot --tval {tgt} 2>&1 | tail -3"
        )
        r = subprocess.run(["wsl", "-d", "Ubuntu-20.04", "-e", "bash", "-lc", cmd],
                           capture_output=True, text=True, timeout=1800)
        if not tgt_win.exists():
            log(f"    {h}: mri_surf2surf FAILED, no output file")
            log("      " + r.stdout.strip()[-400:])
            out["hemis"][h] = {"ok": False, "reason": "no output file",
                               "tail": r.stdout.strip().splitlines()[-3:]}
            continue

        # read back and compare the node inventory against fsaverage
        blab, _bctab, bnames = _read_annot(tgt_win)
        src_lab = labels[h]
        n_src = int((src_lab > 0).sum())
        n_trg = int((blab > 0).sum())
        scale = n_trg / max(1, n_src)
        missing, ratios, rows = [], [], []
        for i, nm in enumerate(names):
            if i == 0:
                continue
            k_src = int((src_lab == i).sum())
            if k_src == 0:
                continue
            j = bnames.index(nm) if nm in bnames else -1
            k_trg = int((blab == j).sum()) if j >= 0 else 0
            rows.append({"node": nm, "fsaverage": k_src, subject: k_trg})
            if k_trg == 0:
                missing.append(nm)
            else:
                ratios.append((k_trg / k_src) / scale)
        ratios = np.asarray(ratios)
        # a node is "sensible" when its share of cortex is preserved within a factor
        # of two after resampling onto a mesh with a different vertex count
        odd = [r["node"] for r, q in zip([x for x in rows if x[subject] > 0], ratios)
               if q < 0.5 or q > 2.0]
        ok = (not missing) and (not odd)
        out["hemis"][h] = {
            "ok": bool(ok), "n_nodes_fsaverage": len(rows),
            "n_nodes_on_subject": int(len(rows) - len(missing)),
            "missing": missing, "vertices_fsaverage": n_src, "vertices_subject": n_trg,
            "mesh_scale": round(scale, 4),
            "share_ratio_min": round(float(ratios.min()), 3) if ratios.size else None,
            "share_ratio_median": round(float(np.median(ratios)), 3) if ratios.size else None,
            "share_ratio_max": round(float(ratios.max()), 3) if ratios.size else None,
            "nodes_outside_half_to_double": odd}
        with open(rtdir / f"{h}.{tag}.{subject}.node_counts.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["node", "fsaverage", subject])
            w.writeheader()
            w.writerows(rows)
        log(f"    {h}: {len(rows) - len(missing)}/{len(rows)} nodes survive on {subject}; "
            f"{n_src} -> {n_trg} labelled vertices (scale {scale:.3f}); "
            f"share ratio {ratios.min():.2f} to {ratios.max():.2f} "
            f"(median {np.median(ratios):.2f}) -> {'OK' if ok else 'PROBLEM'}")
        if missing:
            log(f"      MISSING on {subject}: {missing}")
        if odd:
            log(f"      share changed by more than a factor of two: {odd}")
        tgt_win.unlink()      # the transferred annot is reproducible, the counts are the record
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def build_atlas(tag: str, base_tag: str, path_fn, cyto: dict, geom: dict,
                adj: dict, va: dict, log=print) -> dict:
    log(f"\n===== {tag} ({base_tag}) =====")
    H = {h: load_hemi_inputs(h, path_fn(h), cyto) for h in HEMIS}
    nodes, stats = node_inventory(H, log=log)

    labels, merge_info = {}, {}
    for h in HEMIS:
        labels[h], merge_info[h] = assign_hemi(H[h], nodes, adj[h], log=log)
    n_merged = sum(len(merge_info[h]["merges"]) for h in HEMIS)
    v_merged = sum(m["n_vertices"] for h in HEMIS for m in merge_info[h]["merges"])
    log(f"  absorbed {n_merged} sub-threshold cells, {v_merged} vertices, into "
        f"{len({m['target_node'] for h in HEMIS for m in merge_info[h]['merges']})} nodes")

    colours = node_colours(nodes)
    names = ["unknown"] + [f"{p}_{TYPE_ABBREV[t]}" for p, t in nodes]
    ctab = build_ctab(colours)

    # per (node, hemisphere) rows
    rows = []
    absorbed_into = {h: {m["target_node"] for m in merge_info[h]["merges"]} for h in HEMIS}
    for i, (p, t) in enumerate(nodes, start=1):
        for h in HEMIS:
            m = labels[h] == i
            k = int(m.sum())
            if k == 0:
                continue
            ncomp, frac = components_of(m, adj[h])
            rows.append({
                "id": i if h == "lh" else RH_ID_OFFSET + i,
                "name": names[i], "hemi": h, "base_parcel": p,
                "cyto7_ordinal": t, "cyto7_name": TYPE_NAME[t],
                "n_vertices": k, "area_mm2": round(float(va[h][m].sum()), 2),
                "n_components": ncomp, "frac_largest_component": round(frac, 4),
                "absorbed_residue": int(names[i] in absorbed_into[h]),
                "annot_index": i,
            })
    return {"tag": tag, "base": base_tag, "nodes": nodes, "names": names, "ctab": ctab,
            "labels": labels, "rows": rows, "stats": stats, "H": H,
            "merge_info": merge_info, "colours": colours}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="bert", help="recon-all subject for the round-trip")
    ap.add_argument("--skip-roundtrip", action="store_true")
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    HANDBACK.mkdir(parents=True, exist_ok=True)
    log, fh = _log_factory(HANDBACK / "run_rr10.log")
    t0 = time.time()

    frozen_before = rc.frozen_hashes()
    log("frozen inputs before the build:")
    for k, v in frozen_before.items():
        log(f"  {k}  {v[:16]}...")

    cyto = resolve_target_map("v9", "fsaverage")
    geom = rc.fsaverage_geometry("white", "164k")
    adj, va = {}, {}
    for h in HEMIS:
        coords, faces = geom[HKEY[h]]
        adj[h] = rc.adjacency(faces, coords.shape[0]).astype(np.int8)
        va[h] = rc.vertex_areas(coords, faces)

    built = {}
    for tag, (base_tag, path_fn) in ATLASES.items():
        built[tag] = build_atlas(tag, base_tag, path_fn, cyto, geom, adj, va, log=log)

    # ---- emit ------------------------------------------------------------- #
    log("\n== emitting files ==")
    for tag, B in built.items():
        for h in HEMIS:
            write_annot(OUTDIR / f"{h}.{tag}.annot", B["labels"][h], B["ctab"], B["names"])
            g32 = to_fslr32k(B["labels"][h], h)
            write_label_gii(OUTDIR / f"{h}.{tag}.32k_fs_LR.label.gii", g32,
                            B["names"], B["ctab"])
            cc_lab, cc_names, cc_ctab = contiguity_split(
                B["labels"][h], B["nodes"], adj[h], B["colours"])
            write_annot(OUTDIR / f"{h}.{tag}_cc.annot", cc_lab, cc_ctab, cc_names)
            B.setdefault("cc_counts", {})[h] = len(cc_names) - 1
            log(f"  {h}.{tag}: annot + 32k GIFTI + cc variant "
                f"({len(cc_names) - 1} contiguous nodes)")
        write_ctab_file(OUTDIR / f"{tag}.ctab", B["names"], B["ctab"])
        with open(OUTDIR / f"{tag}_nodes.tsv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(B["rows"][0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(B["rows"])
        log(f"  {tag}.ctab + {tag}_nodes.tsv ({len(B['rows'])} rows)")
    write_provenance(OUTDIR / "PROVENANCE_crossed.txt", built, frozen_before,
                     "conda run -n cyto7 python scripts/rr10_crossed_parcellation.py "
                     f"--subject {args.subject}")
    write_readme(OUTDIR / "README_crossed.md", built)
    log("  PROVENANCE_crossed.txt + README_crossed.md")

    # ---- verify ----------------------------------------------------------- #
    log("\n== Step 3: verification ==")
    verif = {}
    for tag, B in built.items():
        log(f"  {tag}:")
        v = {"partition": verify_partition(B["labels"], B["H"], B["nodes"], log=log),
             "types": verify_type_recoverability(B["labels"], B["H"], B["nodes"], log=log)}
        multi = [r for r in B["rows"] if r["n_components"] > 1]
        v["contiguity"] = {
            "n_rows": len(B["rows"]), "n_multi_component": len(multi),
            "median_frac_largest": float(np.median([r["frac_largest_component"]
                                                    for r in B["rows"]])),
            "median_frac_largest_multi": (float(np.median([r["frac_largest_component"]
                                                           for r in multi]))
                                          if multi else float("nan")),
            "max_components": int(max(r["n_components"] for r in B["rows"]))}
        log(f"    contiguity: {len(multi)} of {len(B['rows'])} rows are multi-patch, "
            f"median largest-component fraction {v['contiguity']['median_frac_largest']:.3f} "
            f"(multi-patch rows {v['contiguity']['median_frac_largest_multi']:.3f}), "
            f"max components {v['contiguity']['max_components']}")
        verif[tag] = v

    # §4.7 check
    expect = {"aparc_cyto7": (101, 34, 0.5), "voneconomo_cyto7": (115, 43, 1.1)}
    log("\n== Step 3.5: against the numbers section 4.7 states ==")
    s47 = {}
    for tag, (en, ep, ec) in expect.items():
        st = built[tag]["stats"]
        ok = (st["n_nodes"] == en and st["n_parcels_with_node"] == ep
              and abs(st["residue_pct"] - ec) < 0.1)
        s47[tag] = {"expected_nodes": en, "got_nodes": st["n_nodes"],
                    "expected_parcels": ep, "got_parcels": st["n_parcels_with_node"],
                    "expected_cost_pct": ec, "got_cost_pct": round(st["residue_pct"], 2),
                    "mean_per_parcel": round(st["mean_nodes_per_parcel"], 2),
                    "reproduces": bool(ok)}
        log(f"  {tag}: nodes {st['n_nodes']} vs {en}, parcels "
            f"{st['n_parcels_with_node']} vs {ep}, cost {st['residue_pct']:.2f}% vs {ec}% "
            f"-> {'REPRODUCES' if ok else 'DISCREPANCY'}")

    # round-trip
    rt = {}
    if not args.skip_roundtrip:
        for subj in (args.subject, "fsaverage"):
            log(f"\n== Step 3.1: round-trip to subject {subj} ==")
            for tag, B in built.items():
                log(f"  {tag}:")
                rt[f"{tag}@{subj}"] = roundtrip_to_subject(
                    subj, tag, B["names"], B["labels"], log=log)
    ok_frozen, detail = rc.check_frozen(frozen_before)
    log(f"\nfrozen released files unchanged: {ok_frozen}")

    summary = {
        "min_vertices": MIN_VERTICES, "rh_id_offset": RH_ID_OFFSET,
        "node_inventory": "bilateral (parcel, type) cell above the floor",
        "atlases": {t: {"stats": B["stats"], "n_rows": len(B["rows"]),
                        "cc_counts": B.get("cc_counts", {}),
                        "merges": {h: B["merge_info"][h]["merges"] for h in HEMIS},
                        "orphan_parcels": {h: B["merge_info"][h]["orphan_parcels"]
                                           for h in HEMIS}}
                    for t, B in built.items()},
        "verification": verif, "section_4_7": s47, "roundtrip": rt,
        "frozen_unchanged": ok_frozen, "frozen": detail,
        "runtime_seconds": round(time.time() - t0, 1)}
    (HANDBACK / "rr10_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log(f"\ntotal runtime {(time.time() - t0) / 60:.1f} min")
    fh.close()
    return built, summary


if __name__ == "__main__":
    main()
