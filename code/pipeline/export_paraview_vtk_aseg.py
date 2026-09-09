"""Export the aseg-derived hippocampus surface as .vtp files for ParaView.

Sibling of ``scripts/export_paraview_vtk.py``. That script exports the **HippUnfold**
hippocampus (subfield-resolved surface); this one exports the coarse **aseg** hippocampus
(the whole hippocampal formation, reconstructed by marching cubes from ``aseg`` labels
17 L / 53 R) — the mesh used in the anatomy-3D allocortex figure
(``scripts/visualize_3d_anatomy.py``).

Same co-registered space (fsaverage tkrRAS, mm) as the other export, so it overlays the
HippUnfold surfaces in ParaView — a direct coarse-structure vs subfield-resolved comparison.

Files written to ``figures/v9/anatomy3d/paraview_vtk_aseg/``:

* ``aseg_hippocampus_LH.vtp`` — scalar ``structure`` (const 17) + ``structure_RGB``.
* ``aseg_hippocampus_RH.vtp`` — scalar ``structure`` (const 53) + ``structure_RGB``.
* ``README.md`` — notes + ParaView tips.

Run::

    conda activate cyto7
    python scripts/export_paraview_vtk_aseg.py
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

from visualize_3d_anatomy import (  # noqa: E402
    aseg_label_mesh, clip_mesh_x, HIPPO_LABELS, COL_HIPPO,
)
from export_paraview_vtk import _pv_poly, _rgb_u8, write_surface  # noqa: E402

OUT_DIR = cfg.results_dir("tables") / "anatomy3d" / "paraview_vtk_aseg"

_MIDLINE_X = 2.0
ASEG_HIPPO = {"LH": 17, "RH": 53}  # FreeSurfer aseg labels


def main() -> int:
    print("Reconstructing aseg hippocampus mesh (labels 17/53, tkrRAS)...")
    hippo = aseg_label_mesh(HIPPO_LABELS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The two hippocampi are separate components; split cleanly at the midline.
    sides = {
        "LH": clip_mesh_x(hippo, None, hi=_MIDLINE_X)[0],
        "RH": clip_mesh_x(hippo, None, lo=_MIDLINE_X)[0],
    }
    for hemi, mesh in sides.items():
        n = mesh.verts.shape[0]
        codes = np.full(n, ASEG_HIPPO[hemi], dtype=np.int32)
        rgb = np.tile(_rgb_u8(np.array(COL_HIPPO)), (n, 1))
        write_surface(mesh, {"structure": codes, "structure_RGB": rgb},
                      OUT_DIR / f"aseg_hippocampus_{hemi}.vtp")

    lines = [
        "# ParaView VTK export — aseg hippocampus (whole hippocampal formation)",
        "",
        "Coarse hippocampus surface reconstructed by marching cubes from the fsaverage",
        "`aseg` (labels **17 L / 53 R**), in fsaverage tkrRAS (mm). Same space as",
        "`../paraview_vtk/` (the HippUnfold subfield surfaces), so they overlay directly.",
        "",
        "## Files",
        "",
        "| file | structure | integer scalar | RGB array |",
        "| --- | --- | --- | --- |",
        "| `aseg_hippocampus_LH.vtp` | L hippocampal formation | `structure` (=17) | `structure_RGB` |",
        "| `aseg_hippocampus_RH.vtp` | R hippocampal formation | `structure` (=53) | `structure_RGB` |",
        "",
        "## Notes",
        "",
        "- This is the **whole hippocampal formation** (no subfields) — a template-average",
        "  shape, and the `aseg` label under-segments the uncal head. For subfield detail",
        "  use the HippUnfold surfaces in `../paraview_vtk/`.",
        "- Colour by `structure_RGB` as *Direct RGB* for the project red, or by the integer",
        "  `structure` scalar for a categorical map.",
        "- Load the aseg and HippUnfold hippocampi together to compare the coarse envelope",
        "  with the subfield-resolved surface (they are co-registered to ~1 mm).",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote README.md")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
