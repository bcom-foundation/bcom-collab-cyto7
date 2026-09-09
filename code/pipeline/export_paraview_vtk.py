"""Export the archicortex-companion surfaces as independent .vtp files for ParaView.

Companion to ``scripts/hippunfold_companion.py`` (see ``docs/SPEC_hippunfold_companion.md``).
Instead of a fixed 2-D render, this writes each surface as its own **VTK XML PolyData**
(``.vtp``) file so they can be loaded into ParaView, rotated freely, and toggled
layer-by-layer. Every file carries per-vertex label scalars (integer, for
threshold/selection + categorical colormaps) **and** a matching ``*_RGB`` uint8 array
(the project colours, for "Map Scalars off" exact display).

All surfaces are written in the **same co-registered space** (fsaverage tkrRAS): the
HippUnfold hippocampus/dentate are snapped onto the fsaverage ``aseg`` hippocampus with the
same per-hemisphere centroid translation used by the companion figure.

Files written to ``figures/v9/anatomy3d/paraview_vtk/``:

* ``fsaverage_pial_{LH,RH}.vtp`` — cortex; scalar ``cyto7_type`` (0=medial wall, 1..7 types)
  + ``cyto7_RGB`` (García-Cabezas greyscale).
* ``hippunfold_hipp_{LH,RH}.vtp`` — hippocampus; scalar ``subfield``
  (0=??, 1 Subiculum … 5 CA4) + ``subfield_RGB``.
* ``hippunfold_dentate_{LH,RH}.vtp`` — dentate gyrus; scalar ``structure`` (const 6)
  + ``structure_RGB``.
* ``README.md`` — label tables + ParaView loading tips.

Run::

    conda activate cyto7
    python scripts/export_paraview_vtk.py
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from visualize_3d_anatomy import Mesh, load_surface  # noqa: E402
from hippunfold_companion import (  # noqa: E402
    ANNOT_DIR, ANNOT_VERSION, TPL, SURF_TPL, SUBF_TPL,
    SUBFIELD_NAMES, SUBFIELD_COLORS, COL_DENTATE,
    load_gii_surface, load_gii_labels, load_hippunfold, align_to_aseg,
)
import figure_style as fs  # noqa: E402

OUT_DIR = cfg.results_dir("tables") / "anatomy3d" / "paraview_vtk"

DENTATE_CODE = 6  # single structure id for the dentate surface


def _pv_poly(mesh: Mesh):
    import pyvista as pv

    faces = np.column_stack(
        [np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces]).ravel()
    return pv.PolyData(np.asarray(mesh.verts, float), faces)


def _rgb_u8(rgb01: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(rgb01) * 255.0, 0, 255).astype(np.uint8)


def write_surface(mesh: Mesh, scalars: dict, out: Path) -> None:
    """Write a mesh + named point-data arrays to a .vtp file."""
    poly = _pv_poly(mesh)
    for name, arr in scalars.items():
        poly.point_data[name] = np.asarray(arr)
    out.parent.mkdir(parents=True, exist_ok=True)
    poly.save(str(out))  # .vtp -> XML PolyData
    print(f"  wrote {out.name}  ({poly.n_points} pts, arrays: {list(scalars)})")


def export_cortex() -> None:
    import nibabel as nib

    pal = fs.cyto7_palette("gc")
    for hemi in ("lh", "rh"):
        s = load_surface(hemi, "pial")
        lab, _c, _n = nib.freesurfer.io.read_annot(
            str(ANNOT_DIR / f"pial.{hemi}.cyto7.{ANNOT_VERSION}.annot"))
        codes = np.asarray(lab, dtype=np.int32)
        rgb = fs.cyto7_vertex_rgb(codes, palette="gc")  # 0 -> light blue medial wall
        write_surface(
            s, {"cyto7_type": codes, "cyto7_RGB": _rgb_u8(rgb)},
            OUT_DIR / f"fsaverage_pial_{hemi.upper()}.vtp")


def export_hippocampus(hipp: dict, dentate: dict, subf: dict) -> None:
    for H in ("L", "R"):
        # hippocampus subfields
        codes = np.asarray(subf[H], dtype=np.int32)
        rgb = np.array([SUBFIELD_COLORS.get(int(k), (0.6, 0.6, 0.6)) for k in codes])
        write_surface(
            hipp[H], {"subfield": codes, "subfield_RGB": _rgb_u8(rgb)},
            OUT_DIR / f"hippunfold_hipp_{H}H.vtp")
        # dentate gyrus (single structure)
        n = dentate[H].verts.shape[0]
        dcodes = np.full(n, DENTATE_CODE, dtype=np.int32)
        drgb = np.tile(_rgb_u8(np.array(COL_DENTATE)), (n, 1))
        write_surface(
            dentate[H], {"structure": dcodes, "structure_RGB": drgb},
            OUT_DIR / f"hippunfold_dentate_{H}H.vtp")


def write_readme(sanity: dict) -> None:
    sub_rows = "\n".join(
        f"| {k} | {SUBFIELD_NAMES[k]} |" for k in sorted(SUBFIELD_NAMES))
    cyto_rows = "\n".join(
        f"| {c} | {fs.CYTO7_NAMES[c - 1]} |" for c in fs.CYTO7_CODES)
    d = sanity["per_hemi"]
    lines = [
        "# ParaView VTK export — cyto7 v7 cortex + HippUnfold hippocampus",
        "",
        "Independent `.vtp` surfaces in one co-registered space (fsaverage tkrRAS, mm).",
        "Load them all in ParaView and toggle each in the Pipeline Browser; rotate freely.",
        "",
        "## Files",
        "",
        "| file | structure | integer scalar | RGB array |",
        "| --- | --- | --- | --- |",
        "| `fsaverage_pial_LH.vtp`, `_RH.vtp` | cortex | `cyto7_type` | `cyto7_RGB` |",
        "| `hippunfold_hipp_LH.vtp`, `_RH.vtp` | hippocampus | `subfield` | `subfield_RGB` |",
        "| `hippunfold_dentate_LH.vtp`, `_RH.vtp` | dentate gyrus | `structure` (=6) | `structure_RGB` |",
        "",
        "## `cyto7_type` (cortex; 0 = medial wall / unlabelled)",
        "",
        "| code | type |",
        "| --- | --- |",
        cyto_rows,
        "",
        "## `subfield` (hippocampus; 0 = unlabelled)",
        "",
        "| code | subfield |",
        "| --- | --- |",
        sub_rows,
        "",
        "## Colouring in ParaView",
        "",
        "- **Exact project colours:** colour by the `*_RGB` array and set the array's",
        "  interpretation to *Direct RGB* (Coloring → the RGB array; disable *Map Scalars*).",
        "- **Categorical selection:** colour by the integer scalar, choose a categorical",
        "  colormap, or use *Threshold* / *Clip by scalar* to isolate one label (e.g.",
        "  `cyto7_type == 1` for allocortex, or `subfield == 1` for subiculum).",
        "- Make the cortex semi-transparent (Properties → Opacity ~0.2) to see the",
        "  hippocampus nested inside — the allocortex arc closing through the archicortex.",
        "",
        "## Co-registration",
        "",
        "HippUnfold `tpl-MNI152NLin2009cAsym` (`den-1mm`) surfaces snapped onto the fsaverage",
        "`aseg` hippocampus (labels 17/53) by a per-hemisphere centroid translation "
        "(pure translation — no rotation/scale/reflection). Residual: "
        f"L {d['L']['delta_norm']:.1f} mm, R {d['R']['delta_norm']:.1f} mm; bbox overlap "
        f"L {d['L']['bbox_overlap_frac']:.2f}, R {d['R']['bbox_overlap_frac']:.2f}. "
        "See `docs/PROVENANCE_hippunfold.md`.",
        "",
        "*Companion of two surfaces in a common volume space — not one topologically-",
        "continuous sheet; the hippocampal surface is the MNI152 template, not our subjects.*",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote README.md")


def main() -> int:
    print("Loading HippUnfold surfaces...")
    hipp, dentate, subf = load_hippunfold()
    sanity = align_to_aseg(hipp, dentate)  # in-place centroid snap
    print("Exporting .vtp files to", OUT_DIR)
    export_cortex()
    export_hippocampus(hipp, dentate, subf)
    write_readme(sanity)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
