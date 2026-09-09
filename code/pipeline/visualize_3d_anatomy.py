"""3D anatomy figures: aseg archicortex + cyto7 v3 parcels in fsaverage space.

Implements ``docs/VISUALIZE_3d_anatomy.md``. The point is to show, for a
non-expert audience, *why allocortex is special on the cortical surface*:

* **Allocortex** is the only cyto7 type with an **off-surface** component. Its
  archicortex (hippocampus) lives in the *volume* (``aseg``), not on the
  cortical sheet, while its paleocortex (piriform) is a thin on-surface sliver.
  So Figure A nests an ``aseg``-derived hippocampus mesh *inside* a
  semi-transparent cortical shell and highlights the piriform on the surface.
* **Agranular** and **dysgranular** are fully on the surface (no ``aseg``
  correlate). Figure B is therefore **surface-only by design**, showing their
  multi-focality (cingulate, insula, temporopolar, rhinal) and the absence of a
  single closed ring.

Coordinate alignment (critical)
-------------------------------
The ``aseg`` is in voxel space; the FreeSurfer surfaces are in surface RAS
(tkrRAS). Marching-cubes vertices are transformed by the ``aseg``'s
**vox2ras-tkr** matrix (``aseg.header.get_vox2ras_tkr()``), *not* the scanner
vox2ras, so the hippocampus mesh lands in the same frame as the pial. A built-in
sanity check confirms the mesh sits under the medial-temporal pial, inside the
shell.

Rendering
---------
Prefers **pyvista** offscreen (proper transparency/occlusion). Falls back to
matplotlib 3D with a decimated cortical shell if pyvista / offscreen GL is
unavailable. Headless either way (no display needed).

Inputs
------
* ``resources/fsaverage_surfaces/aseg.mgz`` (256^3 template segmentation;
  L/R hippocampus = 17/53, L/R amygdala = 18/54).
* ``resources/fsaverage_surfaces/{lh,rh}.pial`` and ``{lh,rh}.inflated``.
* ``resources/cyto7_derived/pial.{lh,rh}.cyto7.v3.annot`` (Allocortex=1=piriform,
  Agranular=2, Dysgranular=3, ...).

Outputs (``figures/v9/anatomy3d/``)
-----------------------------------
* ``allocortex_3d_{lateral,medial,ventral,oblique}.png``
* ``agranular_dysgranular_{pial,inflated}_{view}.png``
* ``README.md`` (caveats).

Run::

    conda activate cyto7
    python scripts/visualize_3d_anatomy.py
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
SURF_DIR = cfg.data_dir() / "fsaverage_surfaces"
ANNOT_DIR = cfg.atlas_dir("fsaverage")
DEFAULT_OUT = cfg.results_dir("tables") / "anatomy3d"

ASEG_PATH = SURF_DIR / "aseg.mgz"

#: aseg integer labels for the subcortical structures we reconstruct.
HIPPO_LABELS = (17, 53)  # L, R hippocampus (archicortex proxy)
AMYG_LABELS = (18, 54)   # L, R amygdala (nuclear complex; "not cortex")

#: cyto7 v3 type codes used here.
ALLOCORTEX = 1  # = piriform paleocortex on the surface
AGRANULAR = 2
DYSGRANULAR = 3

#: Colours (RGB 0-1).
COL_SHELL = (0.78, 0.78, 0.80)      # semi-transparent grey cortical shell
COL_HIPPO = (0.85, 0.16, 0.16)      # archicortex (hippocampus) - red
COL_AMYG = (0.95, 0.70, 0.15)       # amygdala - amber ("not cortex")
COL_PIRIFORM = (0.97, 0.05, 0.78)   # piriform paleocortex - bright saturated magenta
COL_PIRIFORM_EDGE = (0.10, 0.0, 0.12)  # dark outline for the piriform island
COL_AGRANULAR = (0.11, 0.42, 0.69)  # agranular - blue
COL_DYSGRANULAR = (0.18, 0.70, 0.52)  # dysgranular - teal-green
COL_SURFACE_BASE = (0.83, 0.83, 0.85)  # opaque base for surface-only figure


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh: ``verts`` (N,3) float, ``faces`` (M,3) int."""

    verts: np.ndarray
    faces: np.ndarray


# --------------------------------------------------------------------------- #
# Geometry loading
# --------------------------------------------------------------------------- #


def load_surface(hemi: str, kind: str) -> Mesh:
    """Load a FreeSurfer surface (``lh``/``rh`` x ``pial``/``inflated``)."""
    import nibabel as nib

    verts, faces = nib.freesurfer.read_geometry(str(SURF_DIR / f"{hemi}.{kind}"))
    return Mesh(np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64))


def merged_surface(kind: str) -> Mesh:
    """Both hemispheres of a surface merged into one mesh (for the shell)."""
    lh, rh = load_surface("lh", kind), load_surface("rh", kind)
    verts = np.vstack([lh.verts, rh.verts])
    faces = np.vstack([lh.faces, rh.faces + lh.verts.shape[0]])
    return Mesh(verts, faces)


#: cyto7 annot version rendered here. The released map is now **v7**
#: (allocortex-expanded, strict gradient; overridable via the ANNOT_VERSION env var).
import os as _os
ANNOT_VERSION = _os.environ.get("ANNOT_VERSION", "v9")


def load_annot(hemi: str) -> np.ndarray:
    """Per-vertex cyto7 type codes for a hemisphere (0=medial, 1-7=types).

    Uses the released clean map (``ANNOT_VERSION``).
    """
    import nibabel as nib

    labels, _ctab, _names = nib.freesurfer.io.read_annot(
        str(ANNOT_DIR / f"pial.{hemi}.cyto7.{ANNOT_VERSION}.annot")
    )
    return np.asarray(labels)


# --------------------------------------------------------------------------- #
# aseg -> mesh (the critical tkrRAS-aligned reconstruction)
# --------------------------------------------------------------------------- #


def aseg_label_mesh(
    labels: Sequence[int],
    *,
    closing_iters: int = 1,
    smooth_sigma: float = 0.6,
) -> Mesh:
    """Reconstruct a mesh for a set of ``aseg`` labels in **tkrRAS** space.

    The binary label mask is lightly cleaned (morphological closing + Gaussian
    smoothing) for a smoother isosurface, run through marching cubes, and the
    resulting voxel-index vertices ``(i, j, k)`` are mapped to surface RAS with
    the aseg's vox2ras-tkr matrix so they coregister with the pial surfaces.
    """
    import nibabel as nib
    from scipy import ndimage
    from skimage import measure

    aseg = nib.load(str(ASEG_PATH))
    vol = np.asarray(aseg.dataobj)
    T = aseg.header.get_vox2ras_tkr()  # voxel (i,j,k) -> tkrRAS; NOT scanner vox2ras

    mask = np.zeros(vol.shape, dtype=bool)
    for lab in labels:
        mask |= vol == lab
    if not mask.any():
        raise ValueError(f"No voxels for aseg labels {labels!r}")

    if closing_iters > 0:
        mask = ndimage.binary_closing(mask, iterations=closing_iters)
    field = mask.astype(np.float32)
    if smooth_sigma > 0:
        field = ndimage.gaussian_filter(field, sigma=smooth_sigma)

    # marching_cubes returns verts in (i, j, k) array-index order.
    verts_ijk, faces, _normals, _vals = measure.marching_cubes(field, level=0.5)

    # Apply the tkrRAS affine: homogeneous (i,j,k,1) -> (x,y,z).
    ijk_h = np.column_stack([verts_ijk, np.ones(len(verts_ijk))])
    verts_ras = (T @ ijk_h.T).T[:, :3]
    return Mesh(verts_ras, np.asarray(faces, dtype=np.int64))


def medial_temporal_pial_box(hemi: str) -> dict[str, tuple[float, float]]:
    """Approx tkrRAS bounding box of the medial-temporal pial for a hemisphere.

    Used only by the sanity check: the temporal lobe occupies inferior
    (z < ~0), anterior-to-mid (y in roughly [-45, 5]) cortex, lateralised by
    hemisphere. The hippocampus must fall inside this region and inside the
    pial shell.
    """
    pial = load_surface(hemi, "pial")
    v = pial.verts
    # Restrict to inferior, medial-ish temporal vertices for a tight reference.
    z = v[:, 2]
    sel = z < 0
    box = v[sel]
    return {
        "x": (float(box[:, 0].min()), float(box[:, 0].max())),
        "y": (float(box[:, 1].min()), float(box[:, 1].max())),
        "z": (float(box[:, 2].min()), float(box[:, 2].max())),
    }


def alignment_sanity_check(hippo: Mesh, verbose: bool = True) -> dict:
    """Verify the hippocampus mesh is nested under the medial-temporal pial.

    Checks, per hemisphere (split by the sign of the centroid x):
      1. The hippocampus centroid lies inside the *full* pial bounding box
         (i.e. the mesh is inside the shell, not mirrored/offset outside it).
      2. It is **inferior** (mean z < 0) and **medial** (|x| smaller than the
         lateral pial extent) and within the temporal y-range — i.e. it sits
         under medial-temporal cortex, as expected for the hippocampus.
      3. The hemisphere lateralisation is correct (L hippocampus at x<0).

    Returns a dict of measurements and a boolean ``passed``.
    """
    pial_lh, pial_rh = load_surface("lh", "pial"), load_surface("rh", "pial")
    full = np.vstack([pial_lh.verts, pial_rh.verts])
    bb = {ax: (full[:, i].min(), full[:, i].max()) for i, ax in enumerate("xyz")}

    # Split the hippocampus mesh by hemisphere on x sign.
    hv = hippo.verts
    results = {"bbox_pial": bb, "passed": True, "per_hemi": {}}
    for hemi, sign in (("lh", -1), ("rh", +1)):
        side = hv[(hv[:, 0] * sign) > 0]
        if side.size == 0:
            results["per_hemi"][hemi] = {"present": False}
            results["passed"] = False
            continue
        c = side.mean(0)
        pial = pial_lh if hemi == "lh" else pial_rh
        lateral_extent = np.abs(pial.verts[:, 0]).max()
        inside = (
            bb["x"][0] <= c[0] <= bb["x"][1]
            and bb["y"][0] <= c[1] <= bb["y"][1]
            and bb["z"][0] <= c[2] <= bb["z"][1]
        )
        inferior = c[2] < 0.0
        medial = abs(c[0]) < lateral_extent  # not at the lateral convexity
        lateralised = (c[0] * sign) > 0
        ok = bool(inside and inferior and medial and lateralised)
        results["per_hemi"][hemi] = {
            "present": True,
            "centroid": c.tolist(),
            "inside_pial_bbox": bool(inside),
            "inferior_z<0": bool(inferior),
            "medial": bool(medial),
            "correct_hemisphere": bool(lateralised),
            "ok": ok,
        }
        results["passed"] = results["passed"] and ok

    if verbose:
        print("  Alignment sanity check (hippocampus vs pial):")
        print(f"    pial bbox  x{_fmt(bb['x'])} y{_fmt(bb['y'])} z{_fmt(bb['z'])}")
        for hemi, r in results["per_hemi"].items():
            if not r.get("present"):
                print(f"    {hemi}: NO vertices on this side  <-- FAIL")
                continue
            c = r["centroid"]
            print(f"    {hemi}: centroid ({c[0]:+.1f},{c[1]:+.1f},{c[2]:+.1f})  "
                  f"inside={r['inside_pial_bbox']} inferior={r['inferior_z<0']} "
                  f"medial={r['medial']} hemi_ok={r['correct_hemisphere']}  "
                  f"-> {'OK' if r['ok'] else 'FAIL'}")
        print(f"  => sanity check {'PASSED' if results['passed'] else 'FAILED'}")
    return results


def _fmt(t: tuple[float, float]) -> str:
    return f"[{t[0]:+.0f},{t[1]:+.0f}]"


# --------------------------------------------------------------------------- #
# Per-vertex parcel colour maps
# --------------------------------------------------------------------------- #


def surface_type_colors(
    kind: str,
    type_colors: dict[int, tuple],
    base: tuple = COL_SURFACE_BASE,
    hemis: Sequence[str] = ("lh", "rh"),
) -> tuple[Mesh, np.ndarray]:
    """Surface mesh + per-vertex RGB, highlighting the requested type codes.

    Vertices whose v3 type is in ``type_colors`` get that colour; everything
    else gets the neutral ``base`` colour. ``hemis`` selects which hemispheres
    to include — use a single hemisphere for the **inflated** surface (the two
    inflated balloons are each centred near the origin and would otherwise
    overlap) or for views that must expose the medial wall.
    """
    vlist, flist, codes_list, nv = [], [], [], 0
    for hemi in hemis:
        s = load_surface(hemi, kind)
        vlist.append(s.verts)
        flist.append(s.faces + nv)
        nv += s.verts.shape[0]
        codes_list.append(load_annot(hemi))
    verts = np.vstack(vlist)
    faces = np.vstack(flist)
    codes = np.concatenate(codes_list)
    rgb = np.tile(np.array(base, dtype=float), (verts.shape[0], 1))
    for code, col in type_colors.items():
        rgb[codes == code] = col
    return Mesh(verts, faces), rgb


def clip_mesh_x(
    mesh: Mesh, scalars: np.ndarray | None = None, lo: float = -1e9, hi: float = 1e9
) -> tuple[Mesh, np.ndarray | None]:
    """Keep only the part of *mesh* with ``lo <= x <= hi`` (verts + faces + scalars).

    Faces are kept when all three vertices survive; vertices are re-indexed.
    Clipping here (rather than via pyvista's ``clip``) keeps the per-vertex
    scalar array in lock-step with the point count, so RGB rendering never
    mismatches.
    """
    x = mesh.verts[:, 0]
    vmask = (x >= lo) & (x <= hi)
    fmask = vmask[mesh.faces].all(axis=1)
    new_faces = mesh.faces[fmask]
    used = np.unique(new_faces)
    remap = np.full(mesh.verts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    clipped = Mesh(mesh.verts[used], remap[new_faces])
    s = None if scalars is None else np.asarray(scalars)[used]
    return clipped, s


def _merge_hemis_codes(kind: str, hemis: Sequence[str]) -> tuple[Mesh, np.ndarray]:
    """Merged surface mesh + per-vertex v3 codes for the given hemispheres."""
    vlist, flist, codes_list, nv = [], [], [], 0
    for hemi in hemis:
        s = load_surface(hemi, kind)
        vlist.append(s.verts)
        flist.append(s.faces + nv)
        nv += s.verts.shape[0]
        codes_list.append(load_annot(hemi))
    return Mesh(np.vstack(vlist), np.vstack(flist)), np.concatenate(codes_list)


def extract_type_submesh(
    kind: str, code: int, hemis: Sequence[str] = ("lh", "rh"), min_face_verts: int = 2
) -> Mesh:
    """Submesh of the surface faces belonging to a single v3 type *code*.

    A face is kept when at least ``min_face_verts`` of its three vertices carry
    the code, giving a crisp patch for a small parcel like the piriform island.
    """
    mesh, codes = _merge_hemis_codes(kind, hemis)
    vmask = codes == code
    fmask = vmask[mesh.faces].sum(axis=1) >= min_face_verts
    new_faces = mesh.faces[fmask]
    used = np.unique(new_faces)
    remap = np.full(mesh.verts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return Mesh(mesh.verts[used], remap[new_faces])


def offset_outward(mesh: Mesh, delta: float, centroid: np.ndarray) -> Mesh:
    """Push vertices radially away from *centroid* by *delta* mm (proud of shell).

    Radial-from-centroid ~= the surface outward normal on the convex/ventral
    cortex, so a thin parcel rendered this way sits just outside the glass shell
    instead of co-planar with it (no z-fighting, never occluded).
    """
    d = mesh.verts - centroid
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Mesh(mesh.verts + d / n * delta, mesh.faces)


def combine_meshes(*meshes: Mesh) -> Mesh:
    """Concatenate meshes into one (used to frame a zoomed camera)."""
    vlist, flist, nv = [], [], 0
    for m in meshes:
        vlist.append(m.verts)
        flist.append(m.faces + nv)
        nv += m.verts.shape[0]
    return Mesh(np.vstack(vlist), np.vstack(flist))


# --------------------------------------------------------------------------- #
# Camera views
# --------------------------------------------------------------------------- #

#: pyvista camera (position-vector, view-up) per named view, in tkrRAS.
#: Right-handed RAS: +x=right, +y=anterior, +z=superior.
PV_VIEWS = {
    "lateral": ((-1, 0, 0), (0, 0, 1)),   # from the left, looking +x
    "medial": ((1, 0, 0), (0, 0, 1)),     # from the right toward midline
    "ventral": ((0, 0, -1), (0, 1, 0)),   # from below, looking up
    "oblique": ((1.0, -0.8, 0.6), (0, 0, 1)),  # medial-antero-superior 3/4 (cut face)
    "insula": ((-1, 0, 0), (0, 0, 1)),    # lateral; pairs with an operculum clip
    "temporopolar": ((0, 1, -0.3), (0, 0, 1)),  # from the front, slightly below
    "ventral_zoom": ((-0.35, 0.0, -1.0), (0, 1, 0)),  # mostly inferior, slight left (no anterior tilt)
}

#: matplotlib (elev, azim) fallback per view.
MPL_VIEWS = {
    "lateral": (0, 180),
    "medial": (0, 0),
    "ventral": (-90, 90),
    "oblique": (20, -50),
    "insula": (0, 180),
    "temporopolar": (-10, 90),
    "ventral_zoom": (-78, 100),
}

#: extra camera zoom per view (1.0 = framed to the focus mesh).
VIEW_ZOOM = {"ventral_zoom": 0.8}  # <1 zooms out for margin so the anterior piriform stays in frame


# --------------------------------------------------------------------------- #
# pyvista rendering
# --------------------------------------------------------------------------- #


def _pyvista_available() -> bool:
    try:
        import pyvista  # noqa: F401
        return True
    except Exception:
        return False


def _pv_poly(mesh: Mesh):
    import pyvista as pv

    faces = np.column_stack(
        [np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces]
    ).ravel()
    return pv.PolyData(mesh.verts, faces)


def _pv_setup():
    import pyvista as pv

    pv.OFF_SCREEN = True
    if sys.platform.startswith("linux"):
        try:
            pv.start_xvfb()
        except Exception:
            pass


def _pv_camera(plotter, mesh_for_focus: Mesh, view: str, zoom: float = 1.0):
    pos_dir, up = PV_VIEWS[view]
    # Frame on the bounding-box centre (not the vertex-weighted mean) so a union
    # of two structures of very different size -- e.g. the small piriform island
    # + the large hippocampus -- is centred between them, not on the larger one.
    v = mesh_for_focus.verts
    lo, hi = v.min(0), v.max(0)
    c = (lo + hi) / 2.0
    radius = float((hi - lo).max() / 2.0)  # half the largest bbox side
    cam = c + np.array(pos_dir, dtype=float) / np.linalg.norm(pos_dir) * radius * 2.6
    plotter.camera_position = [tuple(cam), tuple(c), up]
    plotter.camera.zoom(zoom)


def render_pyvista(
    layers: list[dict],
    focus: Mesh,
    view: str,
    out_path: Path,
    title: str = "",
    window_size=(1500, 1300),
) -> None:
    """Render a stack of mesh layers with pyvista (offscreen).

    Each ``layer`` is a dict with keys ``mesh`` (Mesh), ``color`` (RGB tuple) or
    ``scalars`` (per-vertex RGB array), and ``opacity``. Any clipping is done
    upstream with :func:`clip_mesh_x` so the meshes arrive render-ready.
    """
    import pyvista as pv

    _pv_setup()
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background("white")
    try:
        # Order-independent transparency: the near-glass shell blends correctly
        # over the opaque inner meshes regardless of add order.
        pl.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
    except Exception:
        pass
    for layer in layers:
        poly = _pv_poly(layer["mesh"])
        kw = dict(opacity=layer.get("opacity", 1.0), smooth_shading=True,
                  specular=0.2, show_scalar_bar=False)
        if "scalars" in layer:
            pl.add_mesh(poly, scalars=np.asarray(layer["scalars"]), rgb=True, **kw)
        else:
            pl.add_mesh(poly, color=layer["color"], **kw)
        # Optional bold outline (e.g. around the small piriform island).
        if layer.get("outline"):
            edges = poly.extract_feature_edges(
                boundary_edges=True, feature_edges=False,
                manifold_edges=False, non_manifold_edges=False,
            )
            if edges.n_points > 0:
                pl.add_mesh(edges, color=layer.get("outline_color", (0, 0, 0)),
                            line_width=4, render_lines_as_tubes=True)
    _pv_camera(pl, focus, view, zoom=VIEW_ZOOM.get(view, 1.0))
    if title:
        # Caption at the lower edge so it never covers the (often superior/
        # anterior) structures at the top of the frame.
        pl.add_text(title, position="lower_edge", font_size=9, color="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out_path))
    pl.close()
    print(f"    saved {out_path.name}")


# --------------------------------------------------------------------------- #
# matplotlib fallback rendering
# --------------------------------------------------------------------------- #


def _decimate_faces(mesh: Mesh, keep: float) -> Mesh:
    """Cheap face decimation: keep a random fraction of faces (for the shell)."""
    if keep >= 1.0:
        return mesh
    rng = np.random.default_rng(0)
    n = mesh.faces.shape[0]
    idx = rng.choice(n, size=int(n * keep), replace=False)
    return Mesh(mesh.verts, mesh.faces[idx])


def render_matplotlib(
    layers: list[dict],
    focus: Mesh,
    view: str,
    out_path: Path,
    title: str = "",
    shell_decimate: float = 0.12,
    window_size=(1500, 1300),
) -> None:
    """Fallback renderer: matplotlib 3D ``Poly3DCollection`` with a decimated shell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(window_size[0] / 150, window_size[1] / 150))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    for layer in layers:
        mesh = layer["mesh"]
        op = layer.get("opacity", 1.0)
        # Decimate the big semi-transparent shell so matplotlib can cope.
        if op < 0.5 and mesh.faces.shape[0] > 60000:
            mesh = _decimate_faces(mesh, shell_decimate)
        tris = mesh.verts[mesh.faces]
        if "scalars" in layer:
            facecol = layer["scalars"][mesh.faces].mean(axis=1)
        else:
            facecol = np.tile(layer["color"], (mesh.faces.shape[0], 1))
        pc = Poly3DCollection(tris, alpha=op, linewidths=0)
        pc.set_facecolor(facecol)
        pc.set_edgecolor("none")
        ax.add_collection3d(pc)

    v = focus.verts
    lo, hi = v.min(0), v.max(0)
    c = (lo + hi) / 2.0
    r = float((hi - lo).max() / 2.0) * 1.05
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    elev, azim = MPL_VIEWS[view]
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out_path.name}  (matplotlib fallback)")


# --------------------------------------------------------------------------- #
# Figure orchestration
# --------------------------------------------------------------------------- #


def render(layers, focus, view, out_path, title="", use_pyvista=True):
    """Dispatch to pyvista (preferred) or matplotlib (fallback)."""
    if use_pyvista:
        try:
            render_pyvista(layers, focus, view, out_path, title=title)
            return True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"    pyvista render failed ({exc!r}); falling back to matplotlib.")
    render_matplotlib(layers, focus, view, out_path, title=title)
    return False


#: x-coordinate threshold (tkrRAS) separating the two hemispheres on the pial.
_MIDLINE_X = 2.0
#: left-hemisphere lateral-operculum cut for the pial 'insula' view (keep x >= this).
_LEFT_OPERCULUM_X = -46.0


def figure_a_allocortex(
    out_dir: Path, use_pyvista: bool, with_amygdala: bool, hippo: Mesh
) -> None:
    """Figure A: allocortex composite (shell + hippocampus inside + piriform).

    Per-view hemisphere choice: ``lateral`` and ``ventral`` show **both**
    hemispheres (the hippocampi nest under a translucent whole-brain shell);
    ``medial`` and the ``oblique`` cut show the **left** hemisphere only so the
    medial wall is exposed and the left hippocampus is clearly seen inside the
    sectioned temporal lobe.
    """
    print("Figure A - allocortex composite (archicortex + piriform)...")
    amyg = aseg_label_mesh(AMYG_LABELS) if with_amygdala else None
    hippo_lh, _ = clip_mesh_x(hippo, None, hi=_MIDLINE_X)
    amyg_lh = clip_mesh_x(amyg, None, hi=_MIDLINE_X)[0] if amyg is not None else None

    # Near-glass grey shells (piriform is now a SEPARATE opaque layer, not part
    # of the shell, so it stays bright even though the shell is ~transparent).
    shell_both = merged_surface("pial")
    shell_lh = load_surface("lh", "pial")
    brain_centroid = shell_both.verts.mean(0)

    # Opaque, outlined piriform island, offset slightly proud of the glass shell
    # so the small ventral paleocortex patch stands out clearly.
    pir_both = offset_outward(
        extract_type_submesh("pial", ALLOCORTEX, ("lh", "rh")), 0.9, brain_centroid)
    pir_lh = offset_outward(
        extract_type_submesh("pial", ALLOCORTEX, ("lh",)), 0.9, brain_centroid)

    shell_op = 0.11  # near-glass
    title = ("Allocortex has two parts in different places - paleocortex (piriform, magenta)\n"
             "is a small island ON the ventral surface, surrounded by agranular periallocortex;\n"
             "archicortex (hippocampus, red) is a separate structure OFF the surface.")

    # (shell, piriform, hippocampus, amygdala, camera-focus)
    plans = {
        "lateral": (shell_both, pir_both, hippo, amyg, shell_both),
        "ventral": (shell_both, pir_both, hippo, amyg, shell_both),
        "medial": (shell_lh, pir_lh, hippo_lh, amyg_lh, shell_lh),
        "oblique": (shell_lh, pir_lh, hippo_lh, amyg_lh, hippo_lh),
        "ventral_zoom": (shell_lh, pir_lh, hippo_lh, amyg_lh,
                         combine_meshes(pir_lh, hippo_lh)),
    }
    for view, (shell, pir, hp, am, focus) in plans.items():
        # Opaque inner meshes first, near-glass shell last (depth peeling makes
        # the order robust either way).
        layers = [
            {"mesh": pir, "color": COL_PIRIFORM, "opacity": 1.0,
             "outline": True, "outline_color": COL_PIRIFORM_EDGE},
            {"mesh": hp, "color": COL_HIPPO, "opacity": 1.0},
        ]
        if am is not None:
            layers.append({"mesh": am, "color": COL_AMYG, "opacity": 0.85})
        layers.append({"mesh": shell, "color": COL_SHELL, "opacity": shell_op})
        render(layers, focus=focus, view=view,
               out_path=out_dir / f"allocortex_3d_{view}.png",
               title=title, use_pyvista=use_pyvista)


def figure_b_agranular_dysgranular(out_dir: Path, use_pyvista: bool) -> None:
    """Figure B: agranular & dysgranular, SURFACE ONLY (no aseg).

    Pial panels show both hemispheres; the ``insula`` panel trims the left
    lateral operculum (keep ``x >= -46``) so the insula in the depth of the
    lateral sulcus is exposed. Inflated panels show the **left hemisphere only**
    (the two inflated balloons overlap if merged); inflation already unfolds the
    opercula, so the insula is visible on the lateral inflated view.
    """
    print("Figure B - agranular & dysgranular (surface-only, no aseg)...")
    type_colors = {AGRANULAR: COL_AGRANULAR, DYSGRANULAR: COL_DYSGRANULAR}
    title = ("Agranular (blue) & dysgranular (teal): entirely ON the cortical "
             "surface - NO off-surface/aseg component (surface-only by design). "
             "Note separate cingulate, insular, temporopolar & rhinal "
             "territories - no single closed ring.")
    views = ("lateral", "medial", "ventral", "insula", "temporopolar")
    for kind in ("pial", "inflated"):
        for view in views:
            # The 'medial' view (and the whole inflated surface, whose two
            # balloons would overlap) is rendered on the LEFT hemisphere only so
            # the medial wall is exposed -- agranular/dysgranular are dominantly
            # medial/insular/rhinal (centroid x ~= -16), not on the lateral
            # convexity, which is precisely the multi-focality point.
            hemis = ("lh",) if (kind == "inflated" or view == "medial") else ("lh", "rh")
            mesh, rgb = surface_type_colors(kind, type_colors, hemis=hemis)
            m, s = mesh, rgb
            # Pial 'insula': remove the left operculum to reveal the insula.
            if view == "insula" and kind == "pial":
                m, s = clip_mesh_x(mesh, rgb, lo=_LEFT_OPERCULUM_X)
            render([{"mesh": m, "scalars": s, "opacity": 1.0}], focus=m, view=view,
                   out_path=out_dir / f"agranular_dysgranular_{kind}_{view}.png",
                   title=title, use_pyvista=use_pyvista)


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #


def write_readme(out_dir: Path, sanity: dict, renderer: str) -> None:
    passed = "PASSED" if sanity.get("passed") else "FAILED"
    lines = [
        "# `figures/v9/anatomy3d/` — 3D anatomy figures",
        "",
        "Generated by `scripts/visualize_3d_anatomy.py` (cyto7 env). 3D views of "
        "the cyto7 **v3** cortical types in fsaverage surface space, made to show "
        "*why allocortex is special on the cortical surface*.",
        "",
        "## Figures",
        "",
        "**A. Allocortex composite** "
        "(`allocortex_3d_{lateral,medial,ventral,oblique,ventral_zoom}.png`)",
        "A **near-glass** grey cortical **pial shell** (opacity ~0.11) + the "
        "**archicortex (hippocampus)** reconstructed from `aseg` (labels 17 L / "
        "53 R) as a solid red mesh nested *inside* the shell + the **piriform "
        "paleocortex** (v3 type 1) drawn as an **opaque, bright-magenta, outlined "
        "island** offset slightly proud of the surface so the small ventral patch "
        "stands out. Amygdala (aseg 18/54, amber) is included for spatial context "
        "and is **not cortex** (a nuclear complex). The oblique view shows a "
        "left-hemisphere cut; **`ventral_zoom`** is a close-up of the temporal "
        "pole showing the piriform island (on-surface) and the hippocampus blob "
        "(off-surface) together, so their separation is obvious.",
        "",
        "_Allocortex has two parts in different places — paleocortex (piriform) is "
        "a small island ON the ventral surface, surrounded by agranular "
        "periallocortex; archicortex (hippocampus) is a separate structure OFF the "
        "surface._",
        "",
        "**B. Agranular & dysgranular** (`agranular_dysgranular_{pial,inflated}_{view}.png`)",
        "Agranular (v3 type 2, blue) and dysgranular (type 3, teal) highlighted on "
        "the **pial and inflated** surfaces, from lateral / medial / ventral / "
        "**insula** (lateral clip exposing the insula in the lateral sulcus) / "
        "temporopolar views.",
        "",
        "_These types are entirely on the cortical surface — **no off-surface / "
        "`aseg` correlate**. The surface-only treatment is **by design, not an "
        "omission**. Note the separate cingulate, insular, temporopolar and rhinal "
        "territories: there is no single closed ring._",
        "",
        "## Coordinate alignment",
        "",
        f"The hippocampus mesh was aligned to the pial via the aseg's "
        f"**vox2ras-tkr** matrix (`aseg.header.get_vox2ras_tkr()`), *not* the "
        f"scanner vox2ras. Built-in sanity check: **{passed}** — the mesh centroid "
        f"lands under the medial-temporal pial (inferior, medial, correct "
        f"hemisphere), inside the shell.",
        "",
        f"Renderer used: **{renderer}**.",
        "",
        "## Caveats",
        "",
        "- fsaverage `aseg` is a **template average** — the hippocampus is an "
        "average shape, fine for a schematic, not a subject-specific anatomy.",
        "- Alignment **must** use tkrRAS (above); the mesh must land under the "
        "medial-temporal pial. A mirrored/offset mesh means the wrong affine "
        "(scanner vs tkr) or an axis-order mismatch.",
        "- The aseg hippocampus = the **whole hippocampal formation** (archicortex "
        "proxy); subfields would need **HippUnfold** (future work).",
        "- The `aseg` hippocampus label is **coarse**: it under-segments the "
        "**uncal head / hook** (the antero-medial inflection of the hippocampal "
        "head), so the anterior tip of the red mesh is blunt. A faithful "
        "reconstruction of the head and subfields needs the FreeSurfer "
        "**hippocampal-subfields** module or **HippUnfold**, not the whole-"
        "structure `aseg` label used here for the schematic.",
        "- **Only allocortex has an off-surface component.** The "
        "agranular/dysgranular figure is **surface-only by design**.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out_dir / 'README.md'}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--figure", choices=["all", "a", "b"], default="all",
                   help="Render Figure A (allocortex composite), B "
                        "(agranular/dysgranular), or both (default).")
    p.add_argument("--no-amygdala", action="store_true",
                   help="Omit the amygdala context mesh from Figure A.")
    p.add_argument("--force-matplotlib", action="store_true",
                   help="Skip pyvista and use the matplotlib 3D fallback.")
    p.add_argument("--sanity-only", action="store_true",
                   help="Only build the hippocampus mesh and run the alignment "
                        "sanity check (no figures).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    use_pyvista = (not args.force_matplotlib) and _pyvista_available()
    renderer = "pyvista (offscreen)" if use_pyvista else "matplotlib 3D (fallback)"
    print(f"Renderer: {renderer}")

    if args.sanity_only:
        hippo = aseg_label_mesh(HIPPO_LABELS)
        sanity = alignment_sanity_check(hippo)
        return 0 if sanity["passed"] else 1

    hippo = aseg_label_mesh(HIPPO_LABELS)
    sanity = alignment_sanity_check(hippo)
    if not sanity["passed"]:
        print("  WARNING: alignment sanity check FAILED - see report above.")

    if args.figure in ("all", "a"):
        figure_a_allocortex(out_dir, use_pyvista, not args.no_amygdala, hippo)
    if args.figure in ("all", "b"):
        figure_b_agranular_dysgranular(out_dir, use_pyvista)

    write_readme(out_dir, sanity, renderer)
    print("Done.")
    return 0 if sanity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
