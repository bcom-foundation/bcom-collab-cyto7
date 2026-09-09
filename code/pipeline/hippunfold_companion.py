"""Archicortex companion: cyto7 v7 cortex (fsaverage) + HippUnfold hippocampus.

Implements ``docs/SPEC_hippunfold_companion.md``. The point is García-Cabezas's
"closed circle": on the fsaverage cortical sheet **alone** the allocortex is an
*open arc* (it has no on-surface archicortical component — the hippocampus lives
off the sheet, in the volume). Adding a faithful **HippUnfold** hippocampal
surface, coloured by its subfields, shows the allocortex ring **closing through
the archicortex**.

Two surfaces, one common space (route "a" — no HippUnfold run):

* **Cortex** — fsaverage pial, coloured by **cyto7 v7** (García-Cabezas greyscale
  ramp from :mod:`figure_style`). Rendered as a near-glass shell so the medially
  tucked hippocampus shows through.
* **Hippocampus** — HippUnfold ``tpl-MNI152NLin2009cAsym`` midthickness surface
  (``den-1mm``), coloured by its **subfields** (subiculum -> CA1 -> CA2 -> CA3 ->
  CA4) plus the **dentate** gyrus surface. All of it is allocortex *sensu
  stricto* — a purple ramp continuing the near-black cortical allocortex.

Common space
------------
The cyto7 atlas is on **fsaverage** (surface RAS / tkrRAS); the HippUnfold
surfaces are in **MNI152NLin2009cAsym** world mm. For the medial temporal lobe
these two frames are already co-registered to ~1 mm (verified: HippUnfold hipp
centroids match the fsaverage ``aseg`` hippocampus (labels 17/53) centroids axis
by axis). We therefore anchor the HippUnfold hippocampus **exactly** onto the
``aseg`` hippocampus with a per-hemisphere **centroid translation** — the most
conservative affine (no rotation/scale/reflection, so no artefact risk) — and
apply the same translation to that hemisphere's dentate. A sanity check reports
the residual centroid + bounding-box agreement.

GIFTI gotcha
------------
HippUnfold ``.surf.gii`` files do **not** put the pointset first — the RH files
have the triangle array in ``darrays[0]``. We select the POINTSET (intent 1008)
and TRIANGLE (intent 1009) arrays by **intent code**, never by position.

Rendering
---------
pyvista offscreen (depth peeling for correct glass-over-solid transparency);
matplotlib 3D fallback via ``--force-matplotlib``. Individual views are rendered
then composed into one captioned, legended figure.

Output
------
``figures/v9/anatomy3d/allocortex_hippunfold_companion.png``

Run::

    conda activate cyto7
    python scripts/hippunfold_companion.py
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from visualize_3d_anatomy import (  # noqa: E402  (reuse the anatomy-3D machinery)
    Mesh,
    aseg_label_mesh,
    clip_mesh_x,
    combine_meshes,
    load_surface,
    merged_surface,
    offset_outward,
    render,
    HIPPO_LABELS,
)
import figure_style as fs  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

HU_DIR = cfg.data_dir() / "hippunfold_tpl"
TPL = HU_DIR / "tpl-MNI152NLin2009cAsym"
SURF_TPL = "tpl-MNI152NLin2009cAsym_hemi-{H}_space-T1w_den-1mm_label-{L}_midthickness.surf.gii"
SUBF_TPL = str(HU_DIR / "hemi-{H}_den-1mm_label-hipp_subfields.label.gii")

ANNOT_DIR = cfg.atlas_dir("fsaverage")
ANNOT_VERSION = "v7"
DEFAULT_OUT = cfg.results_dir("tables") / "anatomy3d" / "allocortex_hippunfold_companion.png"

_MIDLINE_X = 2.0

# GIFTI intent codes.
INTENT_POINTSET = 1008
INTENT_TRIANGLE = 1009

# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #

COL_SHELL = (0.80, 0.80, 0.82)   # faint grey glass context shell
COL_SHELL_OPACITY = 0.14         # very faint so the belt + hippocampus dominate

#: cortical limbic-belt codes rendered opaque + proud so the arc is legible.
BELT_CODES = (1, 2, 3)  # Allocortex, Agranular, Dysgranular

#: Hippocampal subfields (HippUnfold keys 1..5) as a purple ramp: transitional
#: subiculum (light, next to the cortical allocortex) -> CA4 (deep). Dentate is
#: the innermost granule layer -> bright magenta. All are allocortex sensu stricto.
SUBFIELD_NAMES = {1: "Subiculum", 2: "CA1", 3: "CA2", 4: "CA3", 5: "CA4"}
SUBFIELD_COLORS = {
    1: (0.78, 0.60, 0.88),
    2: (0.64, 0.38, 0.80),
    3: (0.52, 0.20, 0.70),
    4: (0.40, 0.09, 0.56),
    5: (0.28, 0.03, 0.42),
}
COL_DENTATE = (0.90, 0.15, 0.60)  # dentate gyrus (granule layer) — bright magenta


# --------------------------------------------------------------------------- #
# GIFTI loading (intent-aware)
# --------------------------------------------------------------------------- #


def load_gii_surface(path: str) -> Mesh:
    """Load a GIFTI surface, selecting arrays by **intent code** (order varies)."""
    import nibabel as nib

    g = nib.load(str(path))
    pts = tri = None
    for da in g.darrays:
        code = int(da.intent)
        if code == INTENT_POINTSET:
            pts = np.asarray(da.data, dtype=float)
        elif code == INTENT_TRIANGLE:
            tri = np.asarray(da.data, dtype=np.int64)
    if pts is None or tri is None:  # fallback: infer from dtype/shape
        arrs = [np.asarray(d.data) for d in g.darrays]
        floats = [a for a in arrs if a.dtype.kind == "f"]
        ints = [a for a in arrs if a.dtype.kind in "iu"]
        pts = floats[0].astype(float)
        tri = ints[0].astype(np.int64)
    return Mesh(pts, tri)


def load_gii_labels(path: str) -> np.ndarray:
    """Per-vertex integer labels from a GIFTI label file."""
    import nibabel as nib

    g = nib.load(str(path))
    return np.asarray(g.darrays[0].data, dtype=int)


# --------------------------------------------------------------------------- #
# Alignment: HippUnfold (MNI152NLin2009cAsym) -> fsaverage tkrRAS
# --------------------------------------------------------------------------- #


def hemi_centroid(verts: np.ndarray, sign: int) -> np.ndarray:
    """Centroid of the vertices on one hemisphere side (sign -1 = L, +1 = R)."""
    side = verts[(verts[:, 0] * sign) > 0]
    return side.mean(0)


def align_to_aseg(hipp: dict, dentate: dict, verbose: bool = True) -> dict:
    """Snap each hemisphere's HippUnfold hipp+dentate onto the aseg hippocampus.

    Returns the sanity dict. Mutates the meshes in ``hipp``/``dentate`` in place
    (adds the per-hemisphere centroid translation).
    """
    aseg = aseg_label_mesh(HIPPO_LABELS)
    av = aseg.verts
    report = {"per_hemi": {}, "passed": True}
    for hemi, sign in (("L", -1), ("R", +1)):
        c_as = hemi_centroid(av, sign)
        c_hu = hipp[hemi].verts.mean(0)
        delta = c_as - c_hu
        hipp[hemi] = Mesh(hipp[hemi].verts + delta, hipp[hemi].faces)
        dentate[hemi] = Mesh(dentate[hemi].verts + delta, dentate[hemi].faces)
        # residual after translation (centroid now exact; check bbox overlap).
        aside = av[(av[:, 0] * sign) > 0]
        hv = hipp[hemi].verts
        bb_as = np.column_stack([aside.min(0), aside.max(0)])
        bb_hu = np.column_stack([hv.min(0), hv.max(0)])
        # fraction of the aseg bbox span that the (translated) hipp bbox covers
        overlap = np.minimum(bb_as[:, 1], bb_hu[:, 1]) - np.maximum(bb_as[:, 0], bb_hu[:, 0])
        span = bb_as[:, 1] - bb_as[:, 0]
        frac = float(np.clip(overlap / span, 0, 1).mean())
        ok = frac > 0.6
        report["per_hemi"][hemi] = {
            "delta_mm": delta.tolist(), "delta_norm": float(np.linalg.norm(delta)),
            "bbox_overlap_frac": frac, "ok": ok,
            "hipp_centroid": c_as.tolist(),
        }
        report["passed"] = report["passed"] and ok

    if verbose:
        print("  Alignment (HippUnfold MNI152NLin2009cAsym -> fsaverage tkrRAS,")
        print("  per-hemisphere centroid snap onto aseg hippocampus 17/53):")
        for hemi, r in report["per_hemi"].items():
            d = r["delta_mm"]
            print(f"    {hemi}: shift ({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f}) "
                  f"|d|={r['delta_norm']:.1f}mm  bbox-overlap={r['bbox_overlap_frac']:.2f}  "
                  f"-> {'OK' if r['ok'] else 'CHECK'}")
        print(f"  => alignment sanity {'PASSED' if report['passed'] else 'NEEDS REVIEW'}")
    return report


# --------------------------------------------------------------------------- #
# Build coloured meshes
# --------------------------------------------------------------------------- #


def _cortex_mesh_codes(hemis):
    """fsaverage pial merged over *hemis* + per-vertex cyto7 v7 codes."""
    import nibabel as nib

    vlist, flist, codes, nv = [], [], [], 0
    for hemi in hemis:
        s = load_surface(hemi, "pial")
        vlist.append(s.verts)
        flist.append(s.faces + nv)
        nv += s.verts.shape[0]
        lab, _c, _n = nib.freesurfer.io.read_annot(
            str(ANNOT_DIR / f"pial.{hemi}.cyto7.{ANNOT_VERSION}.annot"))
        codes.append(np.asarray(lab))
    return Mesh(np.vstack(vlist), np.vstack(flist)), np.concatenate(codes)


def cortex_layers(hemis) -> tuple[list[dict], Mesh]:
    """Faint glass shell + opaque proud limbic belt (allo/agr/dys), GC colours.

    Returns the render layers and the belt mesh (used to frame the camera on the
    cortical arc rather than the whole hemisphere).
    """
    mesh, codes = _cortex_mesh_codes(hemis)
    rgb = fs.cyto7_vertex_rgb(codes, palette="gc")
    brain_centroid = mesh.verts.mean(0)

    # Belt submesh: faces with all three vertices in the limbic belt, offset
    # slightly proud of the glass shell so it never z-fights or is occluded.
    vmask = np.isin(codes, BELT_CODES)
    fmask = vmask[mesh.faces].all(axis=1)
    new_faces = mesh.faces[fmask]
    used = np.unique(new_faces)
    remap = np.full(mesh.verts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    belt = Mesh(mesh.verts[used], remap[new_faces])
    belt_rgb = rgb[used]
    belt = offset_outward(belt, 0.8, brain_centroid)

    layers = [
        {"mesh": belt, "scalars": belt_rgb, "opacity": 1.0},
        {"mesh": mesh, "color": COL_SHELL, "opacity": COL_SHELL_OPACITY},
    ]
    return layers, belt


def hipp_layers(hipp: dict, dentate: dict, subf: dict, hemis) -> list[dict]:
    """Solid hippocampus (subfield ramp) + dentate layers for the given hemis."""
    layers = []
    for hemi in hemis:
        rgb = np.array([SUBFIELD_COLORS.get(int(k), (0.5, 0.5, 0.5)) for k in subf[hemi]])
        layers.append({"mesh": hipp[hemi], "scalars": rgb, "opacity": 1.0})
        layers.append({"mesh": dentate[hemi], "color": COL_DENTATE, "opacity": 1.0})
    return layers


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def load_hippunfold() -> tuple[dict, dict, dict]:
    """Load hipp + dentate surfaces and hipp subfield labels for L and R."""
    hipp, dentate, subf = {}, {}, {}
    for H in ("L", "R"):
        hipp[H] = load_gii_surface(TPL / SURF_TPL.format(H=H, L="hipp"))
        dentate[H] = load_gii_surface(TPL / SURF_TPL.format(H=H, L="dentate"))
        subf[H] = load_gii_labels(SUBF_TPL.format(H=H))
    return hipp, dentate, subf


def compose(view_pngs: dict, out_path: Path, sanity: dict) -> None:
    """Compose the rendered views into one captioned, legended figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = [k for k in ("oblique", "medial", "ventral") if k in view_pngs]
    titles = {"oblique": "Left — ventromedial oblique",
              "medial": "Left — medial", "ventral": "Ventral (both hemispheres)"}
    fig = plt.figure(figsize=(6.2 * len(order), 6.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, len(order), left=0.01, right=0.80, top=0.90, bottom=0.14,
                          wspace=0.02)
    for i, v in enumerate(order):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(plt.imread(str(view_pngs[v])))
        ax.set_title(titles.get(v, v), fontsize=12, weight="bold")
        ax.set_axis_off()

    cyto_leg = fs.cyto7_legend_handles(palette="gc")
    sub_leg = [Patch(facecolor=SUBFIELD_COLORS[k], edgecolor="0.3", label=SUBFIELD_NAMES[k])
               for k in sorted(SUBFIELD_COLORS)]
    sub_leg.append(Patch(facecolor=COL_DENTATE, edgecolor="0.3", label="Dentate gyrus"))
    l1 = fig.legend(handles=cyto_leg, loc="upper left", bbox_to_anchor=(0.81, 0.88),
                    fontsize=8.5, frameon=False, title="cyto7 v7 cortex (GC)")
    l1.get_title().set_fontweight("bold")
    fig.add_artist(l1)
    l2 = fig.legend(handles=sub_leg, loc="upper left", bbox_to_anchor=(0.81, 0.50),
                    fontsize=8.5, frameon=False, title="HippUnfold subfields\n(allocortex s.s.)")
    l2.get_title().set_fontweight("bold")

    fig.suptitle("The allocortex ring closed as a faithful surface:\n"
                 "cyto7 v7 cortex (fsaverage) + HippUnfold hippocampal subfields "
                 "(tpl-MNI152NLin2009cAsym, den-1mm)",
                 fontsize=13.5, weight="bold", y=0.995)
    caption = (
        "On the fsaverage mesh alone the allocortex is an open arc (β1=0); adding the archicortical "
        "surface completes the concentric ring. Honest framing: this is a companion of two surfaces in a "
        "common volume space (affine/centroid alignment, ~1 mm residual), not one topologically-continuous "
        "sheet — the meshes are different frameworks (FreeSurfer vs HippUnfold/Workbench). The "
        "hippocampal surface is the MNI152 template, not our subjects; per-subject HippUnfold is the "
        "quantitative version. Subfield colours (subiculum→CA4, dentate) continue the near-black "
        "cortical allocortex; the cortical allocortex/entorhinal/subiculum flows into the hippocampus."
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=8.2, wrap=True,
             color="0.15")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force-matplotlib", action="store_true")
    args = p.parse_args(argv)

    from visualize_3d_anatomy import _pyvista_available
    use_pyvista = (not args.force_matplotlib) and _pyvista_available()
    print(f"Renderer: {'pyvista (offscreen)' if use_pyvista else 'matplotlib 3D (fallback)'}")

    print("Loading HippUnfold surfaces...")
    hipp, dentate, subf = load_hippunfold()
    for H in ("L", "R"):
        assert len(subf[H]) == hipp[H].verts.shape[0], (
            f"subfield/vertex mismatch hemi {H}: {len(subf[H])} vs {hipp[H].verts.shape[0]}")
    sanity = align_to_aseg(hipp, dentate)
    if not sanity["passed"]:
        print("  WARNING: alignment sanity NEEDS REVIEW (see above).")

    tmp = Path(tempfile.mkdtemp(prefix="hippcomp_"))
    view_pngs = {}

    # LH-only cortex + LH hippocampus for the medial / oblique money shots;
    # both hemispheres for the ventral view.
    cortex_lh, belt_lh = cortex_layers(("lh",))
    hipp_lh = hipp_layers(hipp, dentate, subf, ("L",))
    cortex_both, belt_both = cortex_layers(("lh", "rh"))
    hipp_both = hipp_layers(hipp, dentate, subf, ("L", "R"))

    # Frame on the cortical belt + hippocampus so the whole limbic arc (and its
    # closure through the archicortex) is in view, not a tight hippocampus crop.
    focus_lh = combine_meshes(belt_lh, hipp["L"], dentate["L"])
    focus_both = combine_meshes(belt_both, hipp["L"], dentate["L"], hipp["R"], dentate["R"])

    plans = [
        ("oblique", hipp_lh + cortex_lh, focus_lh),
        ("medial", hipp_lh + cortex_lh, focus_lh),
        ("ventral", hipp_both + cortex_both, focus_both),
    ]
    for view, layers, focus in plans:
        out = tmp / f"companion_{view}.png"
        render(layers, focus=focus, view=view, out_path=out, title="",
               use_pyvista=use_pyvista)
        view_pngs[view] = out

    compose(view_pngs, args.output, sanity)
    print("Done.")
    return 0 if sanity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
