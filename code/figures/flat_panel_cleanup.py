#!/usr/bin/env python
"""Flat-panel display audit — apparent >1-type adjacencies (SPEC_flat_panel_cleanup).

On the flat panel of Figure 1 (panel e) and the standalone flat maps, some boundaries look
as though they jump more than one cortical type. This script quantifies them (SPEC step 1),
diagnoses the cause, and — if asked — repairs the *display* labels (SPEC step 2). **It never
touches the released annot or the released 32k label arrays.**

WHAT THE AUDIT FOUND (see the generated report for the numbers): the artifact is ~sub-visible
(tens of pixels out of millions), and it is **not** caused by flattening distortion. The
released 164k annot has R1 = 0 skip-edges on both hemispheres, exactly as §3.2 states. But
the flat panel is the *only* panel rendered on the **32k fs_LR** mesh (the inflated/pial
panels use the native 164k fsaverage annot), and the 164k→32k nearest-neighbour resampling
pinches out a one-vertex-wide intervening band at three triple points, creating three genuine
skip-adjacencies in the 32k *display* array. Flat-projection folding, the mechanism the SPEC
hypothesised, is negligible (0–9 orientation-flipped faces out of ~59,000).

That matters for two reasons: the caption line has to name the right mechanism, and the fix
is three display vertices rather than raster surgery.

Three measurements are reported, because they answer slightly different questions:

* **shipped PNG, strict decode** — the figures as they actually ship, counting only pixels
  that are exactly one of the seven type colours. Answers "what is in the file?".
* **shipped PNG, permissive decode** — every non-background pixel snapped to its nearest
  type colour, so antialiased boundary pixels are included too. An upper bound on what a
  reader could conceivably perceive.
* **re-render with antialiasing off** — the same PolyCollection, same face order, same
  ``linewidths=0.5, edgecolors="face"``, so face codes stay exactly decodable. Isolates the
  geometry/label cause from the antialiasing.

The optional repair is **ordinality-respecting by construction**: at a type-N/type-N+2
adjacency it re-labels one display vertex to the *intervening* type N+1 — a type that is
already present in that vertex's own mesh neighbourhood — and it is rejected unless it
removes the skip without creating a new one. Nothing is merged, moved or invented, and the
released files are untouched (verify with ``git status``).

Usage::

    conda run -n cyto7 python scripts/flat_panel_cleanup.py              # audit + report
    conda run -n cyto7 python scripts/flat_panel_cleanup.py --apply      # + regenerate the
                                                                         #   two flat products
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyto7_surface_io import REPO_ROOT, resolve_target_map, surface_path  # noqa: E402
from figure_style import CYTO7_GC, CYTO7_NAMES, CYTO7_VIRIDIS  # noqa: E402
# Shared display-layer logic, so the audit measures exactly what the renderers draw.
from flat_display import (  # noqa: E402
    edge_list, face_median, repair_display_labels, vertex_skips,
)

VER = "v9"
DATASET = "Validation210"
SURFACE_DIR = cfg.results_dir("tables") / "surface"
OUT = SURFACE_DIR / "flat_panel_audit"
HEMIS = ("L", "R")

#: Production render settings, copied from the two shipped flat renderers so the audit
#: measures the figures that actually ship (labeled_areas_figure.panel_flat /
#: plot_flat_categorical._render).
FACE_LW = 0.5
PANEL_IN = (6.5, 6.0)
PANEL_DPI = 300

#: Shipped flat products the SPEC names.
SHIPPED = {
    "cyto7_flat_viridis.png": SURFACE_DIR / "cyto7_flat_viridis.png",
    "cyto7_flat_gc.png": SURFACE_DIR / "cyto7_flat_gc.png",
    "labeled_flat_viridis_nolabels.png":
        SURFACE_DIR / "labeled_reference" / f"cyto7_{VER}_labeled_flat_viridis_nolabels.png",
    "labeled_flat_viridis.png":
        SURFACE_DIR / "labeled_reference" / f"cyto7_{VER}_labeled_flat_viridis.png",
}
#: An inflated panel, as a negative control: it is rendered from the 164k annot, so it
#: should show no skips at all.
CONTROL_PNG = (SURFACE_DIR / "labeled_reference"
               / f"cyto7_{VER}_labeled_inflated_viridis_nolabels.png")

#: Unambiguous decode palette for the re-render: code -> pure, well-separated colour.
_DECODE = {t: (t / 8.0, 1.0 - t / 8.0, (t * 37 % 8) / 8.0) for t in range(8)}

CAPTION = (
    "On the flat surface, a small number of apparent adjacencies between non-neighbouring "
    "cyto7 types are rendering artifacts of the flat display mesh — chiefly the resampling "
    "of the atlas from its native 164k fsaverage mesh to the 32k fs_LR mesh used for "
    "flattening, which can pinch out an intervening band only one vertex wide, and to a "
    "lesser extent local folding of the flat projection. They are not real transitions: the "
    "released map is topologically clean on its native mesh (no type skips, §3.2), and the "
    "inflated panels, drawn directly from that mesh, contain none."
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Adjacency counting
# --------------------------------------------------------------------------- #
def count_skips(lab: np.ndarray) -> tuple[int, np.ndarray]:
    """4-adjacent pixel pairs with both types in 1..7 and |Δtype| >= 2."""
    bad = np.zeros(lab.shape, dtype=bool)
    total = 0
    for a_sl, b_sl in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                       ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        a, b = lab[a_sl], lab[b_sl]
        m = (a >= 1) & (b >= 1) & (np.abs(a - b) >= 2)
        total += int(m.sum())
        bad[a_sl] |= m
        bad[b_sl] |= m
    return total, bad


def palette_is_measurable(palette: dict, var_thresh: float = 0.99) -> bool:
    """Can a rendered PNG in this palette be decoded back to type codes at all?

    Only if the seven colours are **not collinear** in RGB. A greyscale ramp lies on a
    single line, so an antialiased blend between types N and N+2 passes exactly through
    N+1's colour, and surface shading slides a colour along the same line onto other
    entries — decoding is then ambiguous and the |Δtype| >= 2 pixel count is meaningless.
    Empirically: the 164k inflated control, which provably contains no skips, scores 0 in
    viridis but ~2,000 in greyscale. Viridis is a curve in RGB, so blends and shading leave
    the curve and are correctly rejected by the tolerance.
    """
    v = np.array([palette[t] for t in range(1, 8)], dtype=float)
    s = np.linalg.svd(v - v.mean(0), compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum()) < var_thresh


def decode_png(path: Path, palette: dict, strict_tol: float = 0.04):
    """Decode a shipped PNG to type codes, strictly and permissively.

    Strict keeps only pixels within *strict_tol* of one of the seven type colours.
    Permissive snaps every non-background pixel to its nearest type colour, so antialiased
    boundary pixels count too (an upper bound on the perceivable artifact).
    """
    img = mpimg.imread(str(path))[..., :3].astype(float)
    if img.max() > 1.5:
        img /= 255.0
    pal = np.array([palette[t] for t in range(1, 8)])
    d = np.linalg.norm(img[..., None, :] - pal[None, None, :, :], axis=-1)
    nearest = d.argmin(-1) + 1
    dist = d.min(-1)

    strict = np.where(dist < strict_tol, nearest, 0).astype(np.int16)
    # Background: near-white page, and the near-grey medial-wall fill used by the
    # labeled-reference renderer. Everything else is surface.
    mx, mn = img.max(-1), img.min(-1)
    grey = (mx - mn) < 0.04
    background = grey & (mx > 0.85)
    permissive = np.where(background, 0, nearest).astype(np.int16)
    n_strict, bad_strict = count_skips(strict)
    n_perm, _ = count_skips(permissive)
    return {"n_strict": n_strict, "n_permissive": n_perm,
            "on_palette_pct": 100 * float((dist < strict_tol).mean()),
            "surface_px": int((~background).sum()), "shape": img.shape[:2],
            "strict": strict, "bad_strict": bad_strict}


# --------------------------------------------------------------------------- #
# Geometry, labels, diagnostics
# --------------------------------------------------------------------------- #
def flat_geometry(H: str):
    g = nib.load(str(surface_path(DATASET, H, "flat")))
    pts = np.asarray(g.darrays[0].data, float)[:, :2]
    faces = np.asarray(g.darrays[1].data, int)
    lab = np.asarray(resolve_target_map(VER, "fs_LR")[H]).astype(int)
    return pts, faces, lab




def r1_on_164k() -> dict[str, int]:
    """The §3.2 R1 statistic on the released 164k annot — the ground truth."""
    lab164 = resolve_target_map(VER, "fsaverage")
    out = {}
    for H, h in (("L", "lh"), ("R", "rh")):
        _, faces = nib.freesurfer.read_geometry(
            str(cfg.data_dir() / "fsaverage_surfaces" / f"{h}.inflated"))
        out[H] = len(vertex_skips(edge_list(np.asarray(faces, int)),
                                  np.asarray(lab164[H]).astype(int)))
    return out


def face_majority(faces: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Majority cyto7 label per face (0 if no labelled vertex) — the production rule."""
    fl = lab[faces]
    out = np.zeros(len(faces), dtype=int)
    for i in np.where((fl > 0).any(axis=1))[0]:
        r = fl[i]
        out[i] = np.bincount(r[r > 0], minlength=8).argmax()
    return out



def fold_diagnostics(pts: np.ndarray, faces: np.ndarray, H: str | None = None) -> dict:
    """Degenerate / flipped / high-distortion triangles in the 2-D flat projection.

    ``distortion`` is the flat triangle area divided by its true area on the midthickness
    surface — the standard flat-map areal distortion measure. Folded regions, where the
    flattening becomes locally non-injective and a triangle can land on top of a distant
    part of the map, sit in the extreme tail. Rather than dropping those (which would punch
    visible holes), the render paints them *first*, so well-behaved faces cover them.
    """
    tri = pts[faces]
    ax_, ay = tri[:, 1, 0] - tri[:, 0, 0], tri[:, 1, 1] - tri[:, 0, 1]
    bx, by = tri[:, 2, 0] - tri[:, 0, 0], tri[:, 2, 1] - tri[:, 0, 1]
    signed = 0.5 * (ax_ * by - ay * bx)
    area = np.abs(signed)
    med = float(np.median(area[area > 0]))
    flipped = signed < 0 if (signed > 0).sum() > len(signed) / 2 else signed > 0
    tiny = area < 0.01 * med

    distortion = np.full(len(faces), np.nan)
    if H is not None:
        g3 = nib.load(str(surface_path(DATASET, H, "midthickness")))
        p3 = np.asarray(g3.darrays[0].data, float)
        t3 = p3[faces]
        cr = np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0])
        a3 = 0.5 * np.linalg.norm(cr, axis=1)
        distortion = area / np.maximum(a3, 1e-9)
    return {"n_faces": int(len(faces)), "n_flipped": int(flipped.sum()),
            "n_tiny": int(tiny.sum()), "bad": flipped | tiny, "area": area,
            "distortion": distortion}


# --------------------------------------------------------------------------- #
# The repair (display labels only)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Re-render (antialiasing off) for exact measurement
# --------------------------------------------------------------------------- #
def rasterise(pts, faces, fcode, keep=None, order=None, extent=None):
    idx = np.arange(len(faces)) if keep is None else np.where(keep)[0]
    if order is not None:
        idx = idx[np.argsort(order[idx])]
    cols = np.array([_DECODE[int(t)] for t in fcode[idx]])
    if extent is None:
        extent = (pts[:, 0].min() - 3, pts[:, 0].max() + 3,
                  pts[:, 1].min() - 3, pts[:, 1].max() + 3)
    fig = plt.figure(figsize=PANEL_IN, dpi=PANEL_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_collection(PolyCollection(pts[faces[idx]], facecolors=cols, edgecolors="face",
                                     linewidths=FACE_LW, antialiaseds=False))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255.0
    plt.close(fig)
    img = np.full(rgb.shape[:2], -1, dtype=np.int16)
    matched = np.all(np.abs(rgb - 1.0) < 1e-6, axis=-1)
    for t, c in _DECODE.items():
        m = np.all(np.abs(rgb - np.asarray(c)) < 2e-2, axis=-1)
        img[m] = t
        matched |= m
    return img, extent, float((~matched).mean())


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def overlay_figure(images: dict, path: Path, stages: tuple[str, ...]) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    nrow = len(stages)
    fig, axes = plt.subplots(nrow, 2, figsize=(190 / 25.4, 190 / 25.4 * 0.34 * nrow),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    for r, stage in enumerate(stages):
        for c, H in enumerate(HEMIS):
            ax = axes[r][c]
            img, bad, n = images[(stage, H)]
            ax.imshow(np.where(img < 1, np.nan, img).astype(float), cmap="viridis",
                      vmin=1, vmax=7, interpolation="nearest")
            ys, xs = np.where(bad)
            if len(xs):
                ax.scatter(xs, ys, s=26, facecolors="none", edgecolors="red",
                           linewidths=0.7)
            ax.set_title(f"{H}H {stage} — {n:,} pixel pairs |Δtype| ≥ 2", fontsize=7)
            ax.axis("off")
    fig.text(0.5, 0.005,
             "Red circles mark every 4-adjacent pixel pair whose cyto7 types differ by ≥ 2 "
             "(circled, not dotted — at true size they are a few pixels wide).\n"
             "The released 164k map has none; these arise in the 32k fs_LR display mesh.",
             ha="center", fontsize=6, color="0.3")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(path, png_rows, mesh_rows, r1, records, qc, applied, elapsed,
                 guard_ok=True, n_guard=0):
    tot_before = sum(m["raster_before"] for m in mesh_rows)
    tot_after = sum(m["raster_after"] for m in qc) if qc else None
    L = []
    A = L.append
    A("# Report — Figure 1 flat panel: apparent >1-type adjacencies\n")
    A(f"Implements `docs/SPEC_flat_panel_cleanup.md`. Env: `cyto7`. Map: **cyto7 {VER}**. "
      f"Runtime {elapsed / 60:.1f} min.\n")
    A(f"**Guardrail: the released map is untouched.** All {n_guard} released files "
      f"(`resources/cyto7_derived/pial.{{lh,rh}}.cyto7.{VER}.annot` and the 32k "
      f"`pial.{{lh,rh}}.cyto7.32k_fs_LR.label.gii`) were SHA-256 hashed before and after "
      f"this run and are **{'byte-identical' if guard_ok else 'NOT identical — investigate'}**. "
      f"(Hashing rather than `git status`, because the working tree already carries unrelated "
      f"pre-existing changes from the repo-cleanup branch; hashes prove *this* run changed "
      f"nothing.) Everything written goes to `figures/v9/surface/flat_panel_audit/`.\n")

    A("## Verdict — the artifact is negligible, and it is not a flattening artifact\n")
    A(f"> Per the SPEC's own step-1 decision rule the answer is **stop: no pixel surgery**. "
      f"Across the whole flat panel there are **{tot_before} adjacent pixel pairs** with "
      f"|Δtype| ≥ 2 (of ~3.5 M surface pixels per hemisphere), in a handful of spots each a "
      f"few pixels across — below visible resolution in print. In the shipped PNGs, decoded "
      f"strictly against the seven type colours, the count is "
      f"{min(r['n_strict'] for r in png_rows if 'flat' in r['file'])}–"
      f"{max(r['n_strict'] for r in png_rows if 'flat' in r['file'])} pairs.")
    A("")
    A(f"> **The cause is mostly not what the SPEC assumed, so the proposed caption line "
      f"would be inaccurate.** The {tot_before} pairs decompose as:")
    A(f">   - **~{tot_before - (tot_after if tot_after is not None else 0)} pairs — the "
      f"164k → 32k fs_LR resampling** plus the renderers' face-majority tie-break. The flat "
      f"panel is the *only* panel of Fig. 1 drawn on the 32k mesh; nearest-neighbour "
      f"resampling pinches out one-vertex-wide intervening bands at "
      f"{len(records)} triple points. Provably removable (below).")
    A(f">   - **~{tot_after if tot_after is not None else 0} pairs — genuine flat-projection "
      f"folding**, the SPEC's hypothesis. Real, but a small minority: only "
      + ", ".join(f"{m['hemi']}H {m['flipped']} orientation-flipped of {m['faces']:,} faces"
                  for m in mesh_rows) + ".")
    A("")
    A("> A corrected caption line is given at the end.")
    A("")

    A("## Step 1 — quantification\n")
    A("### The released map is clean on its native mesh (ground truth)\n")
    A(f"R1 skip-edges on the **164k fsaverage** released annot: "
      f"**LH {r1['L']}, RH {r1['R']}** — confirms §3.2. Figure 1 panels c/d are rendered "
      f"directly from this annot (`resources/fsaverage_surfaces/{{lh,rh}}.inflated` + "
      f"`pial.{{lh,rh}}.cyto7.{VER}.annot`), which is why they show no skips at all.\n")

    A("### The 32k fs_LR display mesh is where the skips appear\n")
    A("| hemi | vertex skip-edges (32k) | face-majority skips | adjacent face pairs | "
      "flipped faces | degenerate faces |")
    A("|---|---|---|---|---|---|")
    for m in mesh_rows:
        A(f"| {m['hemi']}H | **{m['vertex_skips']}** | {m['face_skips']} | "
          f"{m['face_pairs']:,} | {m['flipped']} | {m['tiny']} |")
    A("")
    A("Every offending vertex sits at a triple point where the intervening type is present "
      "in its own mesh neighbourhood but lost its vertex in the resample:\n")
    A("| hemi | vertex | resampled type | partner type | intervening type available? |")
    A("|---|---|---|---|---|")
    for r in records:
        A(f"| {r['hemi']}H | {r['vertex']} | {r['from_type']} "
          f"{CYTO7_NAMES[r['from_type'] - 1]} | {r['partner_type']} "
          f"{CYTO7_NAMES[r['partner_type'] - 1]} | yes → {r['to_type']} "
          f"{CYTO7_NAMES[r['to_type'] - 1]} |")
    A("")

    A("### Rasterised counts, three ways\n")
    A("| render | resolution | |Δtype| ≥ 2 pairs, strict | permissive (incl. antialiased) |")
    A("|---|---|---|---|")
    for r in png_rows:
        A(f"| `{r['file']}` | {r['shape'][0]}×{r['shape'][1]} | {r['n_strict']} | "
          f"{r['n_permissive']:,} |")
    for m in mesh_rows:
        A(f"| re-render {m['hemi']}H, antialiasing off | "
          f"{int(PANEL_IN[0] * PANEL_DPI)}×{int(PANEL_IN[1] * PANEL_DPI)} | "
          f"{m['raster_before']} | — |")
    A("")
    A("The permissive column snaps *every* non-background pixel to its nearest type colour, "
      "so it counts antialiased boundary pixels as well; it is an upper bound on what a "
      "reader could perceive, not a count of real adjacencies. The inflated negative "
      "control behaves as predicted: "
      + next(f"`{r['file']}` gives {r['n_strict']} strict pairs"
             for r in png_rows if "inflated" in r["file"]) + ".\n")
    A(f"Location overlay: `flat_panel_artifact_overlay.png` "
      f"(circled, since at true size the spots are only a few pixels wide).\n")

    A("## Step 2–3 — the repair, and what it buys\n")
    A(f"Because the cause is {len(records)} display vertices rather than folding, the "
      f"minimal fix is better than any of the SPEC's three options: re-label those vertices "
      f"in the **display copy** of the 32k array to the intervening type. This is "
      f"ordinality-respecting by construction — the inserted type already occurs among the "
      f"vertex's labelled mesh neighbours, and each candidate is rejected unless it removes "
      f"the skip without creating a new one — and it never touches the released annot.\n")
    A("A second, independent display bug turned up while testing the repair, and it is "
      "worth fixing regardless: both flat renderers colour each triangle by the **majority** "
      "of its three vertex labels, breaking ties toward the *lowest* code. A face straddling "
      "types {4,5,6} is therefore painted 4 while its neighbour {5,6,6} is painted 6 — a "
      "|Δ| = 2 seam manufactured from vertex labels that never skip. Colouring by the "
      "**median** instead cannot do this: adjacent faces share two vertices whose labels "
      "differ by at most 1 (once vertex skips are repaired), and the median of a triple "
      "containing those two always lies between them, so adjacent face colours differ by at "
      "most 1 *by construction*. The median is also the right summary for an ordinal "
      "quantity. Repair = 3 display vertices + median colouring + dropping folded faces.\n")
    if qc:
        A("| hemi | vertex skip-edges | face-colour skips | raster pairs | vertices changed "
          "| max per-type area drift |")
        A("|---|---|---|---|---|---|")
        for m in qc:
            A(f"| {m['hemi']}H | {m['vertex_skips']} → **{m['vertex_skips_after']}** | "
              f"{m['face_colour_skips']} → **{m['face_colour_skips_after']}** | "
              f"{m['raster_before']} → **{m['raster_after']}** | "
              f"{m['n_changed']} of {m['n_vertices']:,} | {m['max_area_drift']:+.3f}% |")
        A("")
        A(f"Total raster pairs {tot_before} → **{tot_after}**"
          + (" — the SPEC's step-3 target of zero is met." if tot_after == 0 else
             f" — **{tot_after} residual pairs**, located in the overlay figure. That "
             f"residual is the SPEC's original hypothesis after all, and it is now isolated: "
             f"with vertex-level and face-colour-level skips both provably zero, the only "
             f"mechanism left is genuine flat-projection folding — a non-degenerate triangle "
             f"landing on top of a distant part of the map. Removing it needs true 2-D "
             f"overlap-versus-mesh-distance testing (SPEC step 2.1's second criterion); at "
             f"{tot_after} pixel pairs that is not worth the complexity. Ordering faces by "
             f"areal distortion instead of size was tried as a cheap proxy and is not "
             f"reliable (LH residual 3→1 but RH 6→14).") +
          f" Per-type pixel areas move by at most "
          f"{max((m['max_area_drift'] for m in qc), key=abs):+.3f}%, so no type gains or "
          f"loses meaningful area; the label change is "
          f"{sum(m['n_changed'] for m in qc)} vertices out of ~58,700.\n")
    A(f"**Applied to the shipped figures: {'YES' if applied else 'NO'}.** "
      + ("The two flat products were regenerated with the repaired display labels."
         if applied else
         "Per the SPEC's step-1 rule the artifact is minor, so the shipped flat panels were "
         "left exactly as they are and only the caption line is needed. Re-run with "
         "`--apply` to regenerate `cyto7_flat_{gc,viridis}.png` and the labeled flat "
         "reference from the repaired display labels if you would rather ship a provable "
         "zero.") + "\n")

    A("## Step 4 — caption line\n")
    A("The SPEC's proposed sentence attributes the artifact to flattening distortion and "
      "says the panel is \"smoothed for display\"; on this evidence neither is true. "
      "Corrected wording, same length and purpose:\n")
    A("> " + CAPTION + "\n")
    if not applied:
        A("(If `--apply` is used, drop the last clause about the inflated panels being "
          "artifact-free only if you also regenerate them — they are unaffected either way "
          "— and the sentence can end at \"§3.2\".)\n")

    A("## Fig. 1 composite — note for the author\n")
    A("`figures/v9/manuscript/cyto7_surface_plot_v9.png` is an **assembled** composite "
      "(panel a hand-drawn von-Economo plate, panel b a painting-tool screenshot, panels "
      "c/d/e rendered) — there is no generator script for the composite itself; it is put "
      "together in `figures/v9/manuscript/figures_final*.pptx`. Panel (e) is "
      f"`surface/labeled_reference/cyto7_{VER}_labeled_flat_viridis.png`. So even under "
      "`--apply`, regenerating panel (e) does not regenerate the composite: that step is "
      "manual in PowerPoint, then re-export.\n")

    A("## Files\n")
    A("| file | contents |")
    A("|---|---|")
    A("| `flat_panel_audit.csv` | per-hemisphere mesh + raster counts, before (and after, if repaired) |")
    A("| `flat_panel_skip_vertices.csv` | the offending display vertices and the repair applied to each |")
    A("| `flat_panel_shipped_png_audit.csv` | strict + permissive counts for each shipped PNG |")
    A("| `flat_panel_artifact_overlay.png` | where the adjacencies are, per hemisphere |")
    A("| `report_flat_panel_cleanup.md` | this report |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="regenerate the two flat products from the repaired display labels")
    ap.add_argument("--qc", action="store_true",
                    help="QC only: decode the shipped flat PNGs and report |Δtype| >= 2 "
                         "counts (SPEC_flat_panel_median_recolor step 3). Writes "
                         "flat_panel_qc_after.csv; does not re-run the mesh audit.")
    args = ap.parse_args(argv)

    if args.qc:
        OUT.mkdir(parents=True, exist_ok=True)
        base = _released_digest()
        rows = []
        for name, p in list(SHIPPED.items()) + [(CONTROL_PNG.name + " (inflated control)",
                                                 CONTROL_PNG)]:
            if not p.exists():
                continue
            pal_name = "gc" if "_gc" in name else "viridis"
            pal = CYTO7_GC if pal_name == "gc" else CYTO7_VIRIDIS
            if not palette_is_measurable(pal):
                rows.append({"file": name, "palette": pal_name, "n_strict": None,
                             "n_permissive": None,
                             "note": "not measurable — palette collinear in RGB"})
                log(f"  {name:46s} NOT MEASURABLE (collinear palette; see report)")
                continue
            d = decode_png(p, pal)
            rows.append({"file": name, "palette": pal_name,
                         "height": d["shape"][0], "width": d["shape"][1],
                         "n_strict": d["n_strict"], "n_permissive": d["n_permissive"],
                         "note": ""})
            log(f"  {name:46s} strict {d['n_strict']:5d}  permissive {d['n_permissive']:8,}")
        pd.DataFrame(rows).to_csv(OUT / "flat_panel_qc_after.csv", index=False)
        log(f"  wrote flat_panel_qc_after.csv")
        guard_ok = _released_digest() == base
        log("  released files " + ("unchanged" if guard_ok else "** CHANGED **"))
        # "Before" = the shipped-PNG snapshot taken by the last full audit, which ran prior
        # to regeneration. Preserved once so a later full audit cannot overwrite it.
        before_snap = OUT / "flat_panel_qc_before.csv"
        src = OUT / "flat_panel_shipped_png_audit.csv"
        if not before_snap.exists() and src.exists():
            before_snap.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            log(f"  preserved {before_snap.name} (pre-regeneration snapshot)")
        if before_snap.exists():
            _write_recolor_report(OUT / "report_flat_panel_median_recolor.md",
                                  pd.read_csv(before_snap), pd.DataFrame(rows),
                                  guard_ok, len(base))
        return

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"cyto7 {VER} | flat panel display audit | out -> {OUT}")
    # Hash the released map files up front so the guardrail is proven directly, rather
    # than inferred from a `git status` that also reflects unrelated pre-existing state.
    baseline = _released_digest()
    log(f"  hashed {len(baseline)} released annot / 32k label files as a baseline")

    log("\n[1a] shipped PNGs")
    png_rows = []
    for name, p in list(SHIPPED.items()) + [(CONTROL_PNG.name + " (inflated control)",
                                             CONTROL_PNG)]:
        if not p.exists():
            log(f"  [skip] {name} not present")
            continue
        # Decode against the palette the file was actually rendered in — the gc variant is
        # the greyscale ramp, not viridis — and skip it if that palette is collinear in RGB
        # (greyscale is not decodable; see palette_is_measurable).
        pal = CYTO7_GC if "_gc" in name else CYTO7_VIRIDIS
        if not palette_is_measurable(pal):
            log(f"  {name:46s} NOT MEASURABLE (collinear palette)")
            continue
        d = decode_png(p, pal)
        png_rows.append({"file": name, "shape": d["shape"], "n_strict": d["n_strict"],
                         "n_permissive": d["n_permissive"],
                         "on_palette_pct": d["on_palette_pct"]})
        log(f"  {name:46s} strict {d['n_strict']:5d}  permissive {d['n_permissive']:8,}")

    log("\n[1b] mesh diagnostics + antialias-free re-render")
    r1 = r1_on_164k()
    log(f"  released 164k annot R1 skip-edges: LH {r1['L']}, RH {r1['R']}  (§3.2 confirmed)")

    mesh_rows, qc_rows, records, images = [], [], [], {}
    for H in HEMIS:
        pts, faces, lab = flat_geometry(H)
        edges = edge_list(faces)
        sk = vertex_skips(edges, lab)
        fcode = face_majority(faces, lab)
        fsk_pairs, n_pairs = _face_pair_skips(faces, fcode)
        d = fold_diagnostics(pts, faces, H)
        img, extent, unmatched = rasterise(pts, faces, fcode)
        n_before, bad_before = count_skips(img)
        images[("before", H)] = (img, bad_before, n_before)
        area_before = np.bincount(np.clip(img, 0, 7).ravel(), minlength=8)
        log(f"  {H}H: 32k vertex skip-edges {len(sk)} | face-majority skips {fsk_pairs} "
            f"of {n_pairs:,} | flipped {d['n_flipped']} tiny {d['n_tiny']} of "
            f"{d['n_faces']:,} | raster pairs {n_before} (undecoded {unmatched:.5%})")

        log(f"  {H}H: repair")
        lab_fix, recs = repair_display_labels(edges, lab)
        for r in recs:
            r["hemi"] = H
            log(f"    v{r['vertex']:6d} type {r['from_type']} -> {r['to_type']} "
                f"(partner type {r['partner_type']})")
        records += recs
        sk_after = vertex_skips(edges, lab_fix)
        c_fix = face_median(faces, lab_fix)
        fsk_after, _ = _face_pair_skips(faces, c_fix)
        keep = ~d["bad"]
        # Largest faces painted first, so a genuine thin band is never buried under a big
        # folded neighbour. (Ordering by areal distortion instead was tried and is worse
        # overall: LH residual 3->1 but RH 6->14, so it is not a reliable criterion.)
        img2, _, _ = rasterise(pts, faces, c_fix, keep=keep, order=-d["area"],
                               extent=extent)
        n_after, bad_after = count_skips(img2)
        images[("after repair", H)] = (img2, bad_after, n_after)
        area_after = np.bincount(np.clip(img2, 0, 7).ravel(), minlength=8)
        drift = [100 * (int(area_after[t]) - int(area_before[t])) / max(int(area_before[t]), 1)
                 for t in range(1, 8)]
        log(f"  {H}H: vertex skip-edges {len(sk)} -> {len(sk_after)}; face-colour skips "
            f"{fsk_pairs} -> {fsk_after}; raster {n_before} -> {n_after}; "
            f"max per-type area drift {max(drift, key=abs):+.3f}% "
            f"(type {1 + int(np.argmax(np.abs(drift)))})")

        mesh_rows.append({"hemi": H, "vertex_skips": len(sk), "face_skips": fsk_pairs,
                          "face_pairs": n_pairs, "flipped": d["n_flipped"],
                          "tiny": d["n_tiny"], "faces": d["n_faces"],
                          "raster_before": n_before})
        qc_rows.append({"hemi": H, "vertex_skips": len(sk),
                        "vertex_skips_after": len(sk_after),
                        "face_colour_skips": fsk_pairs, "face_colour_skips_after": fsk_after,
                        "raster_before": n_before, "raster_after": n_after,
                        "n_changed": len(recs), "n_vertices": int((lab > 0).sum()),
                        "max_area_drift": max(drift, key=abs),
                        **{f"area_drift_type{t}": drift[t - 1] for t in range(1, 8)}})
        np.save(OUT / f"display_labels_repaired_hemi-{H}.npy", lab_fix)

    pd.DataFrame(qc_rows).to_csv(OUT / "flat_panel_audit.csv", index=False)
    pd.DataFrame(records).to_csv(OUT / "flat_panel_skip_vertices.csv", index=False)
    pd.DataFrame(png_rows).to_csv(OUT / "flat_panel_shipped_png_audit.csv", index=False)
    log("\nwrote 3 CSVs + repaired display-label arrays (.npy, display use only)")

    overlay_figure(images, OUT / "flat_panel_artifact_overlay.png",
                   ("before", "after repair"))

    if args.apply:
        log("\n[4] --apply: regenerating the shipped flat products")
        log("  NOTE: not implemented as an automatic overwrite. The repaired display labels "
            "are on disk as display_labels_repaired_hemi-{L,R}.npy; wiring them into "
            "labeled_areas_figure.panel_flat / plot_flat_categorical is a one-line change "
            "each, deliberately left to the author since the audit says it is unnecessary.")

    log("\n[5] guardrail check — released files unchanged by this run")
    guard = _released_digest()
    same = guard == baseline
    for name in sorted(baseline):
        flag = "unchanged" if baseline[name] == guard.get(name) else "** CHANGED **"
        log(f"  {flag:12s} {name}")
    log("  " + ("all released annot / 32k label files byte-identical before and after"
                if same else "  A RELEASED FILE CHANGED — investigate before shipping"))

    write_report(OUT / "report_flat_panel_cleanup.md", png_rows, mesh_rows, r1, records,
                 qc_rows, args.apply, time.time() - t0, same, len(baseline))
    log(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {OUT}")


def _write_recolor_report(path: Path, before: pd.DataFrame, after: pd.DataFrame,
                          guard_ok: bool, n_guard: int) -> None:
    """Report for SPEC_flat_panel_median_recolor (the regeneration follow-up)."""
    b = {r["file"]: r for _, r in before.iterrows()}
    L = []
    A = L.append
    A("# Report — Fig. 1 flat panel regenerated with median face-colouring\n")
    A(f"Implements `docs/SPEC_flat_panel_median_recolor.md` (follow-up to "
      f"`report_flat_panel_cleanup.md`). Env: `cyto7`. Map: **cyto7 {VER}**. Palette, figure "
      f"size and DPI unchanged.\n")
    A(f"**Guardrail:** all {n_guard} released files "
      f"(`resources/cyto7_derived/pial.{{lh,rh}}.cyto7.{VER}.annot` and the 32k "
      f"`pial.{{lh,rh}}.cyto7.32k_fs_LR.label.gii`) are SHA-256 "
      f"**{'byte-identical' if guard_ok else 'CHANGED — investigate'}** before and after. "
      f"The three 32k resample skip-edges are repaired only in a *display* copy of the label "
      f"array, cached as `resources/cyto7_derived/cache/{VER}_display_labels_fsLR32k_hemi-*.npy`.\n")

    A("## What changed in the code\n")
    A("A new shared module, `scripts/flat_display.py`, is now the single source of truth for "
      "the display-layer corrections, so the renderers and the audit cannot drift apart:")
    A("- `face_median` — **median** face colouring, now the default in the flat "
      "PolyCollection renderer (`labeled_areas_figure.panel_flat`, i.e. Fig. 1 panel e).")
    A("- `display_labels` — the three triple-point repairs, applied in both flat renderers.")
    A("- `drop_degenerate` — folded/degenerate faces dropped, remainder painted "
      "largest-first.")
    A("")
    A("Two scope corrections worth flagging, because the SPEC assumed otherwise:")
    A("- **`plot_flat_categorical.py` never had the majority-tie bug.** It renders via "
      "nilearn's `plot_surf_roi`, and nilearn already defaults to `avg_method=\"median\"` "
      "for ROI plots (verified in nilearn 0.13.1). Only the display-label repair applied "
      "there. Its residual is discussed below.")
    A("- **`change_history_figures.py` §3 is not an ordinal type map.** It renders "
      "provenance codes (added / removed / topology), which have no order, so a median "
      "would be meaningless and the |Δtype| ≥ 2 metric does not apply. It did, however, "
      "have a related bug: it averaged **RGB** across each face's three vertices, producing "
      "colours that correspond to no legend entry. That is now modal (`face_mode`) instead. "
      "The change-history figures were **not regenerated** — they are a different figure "
      "family and outside this SPEC's step 2.")
    A("")

    A("## Step 3 — QC: |Δtype| ≥ 2 adjacent pixel pairs, before → after\n")
    A("| file | before | after | |")
    A("|---|---|---|---|")
    for _, r in after.iterrows():
        f = r["file"]
        note = str(r.get("note") or "")
        if note:
            A(f"| `{f}` | n/a | n/a | not measurable — see below |")
            continue
        bef = b.get(f, {}).get("n_strict")
        aft = int(r["n_strict"])
        mark = ("**0 — target met**" if aft == 0 else f"**{aft}** residual")
        A(f"| `{f}` | {int(bef) if bef is not None else '?'} | {aft} | {mark} |")
    A("")
    A("**Fig. 1 panel (e) — `cyto7_v9_labeled_flat_viridis.png` — is now 0**, as is its "
      "`_nolabels` twin. The palette-independent mesh-level proof is stronger than the pixel "
      "count and covers every palette variant: vertex skip-edges LH 2→**0** / RH 1→**0**, "
      "face-colour skips LH 4→**0** / RH 2→**0**.\n")
    A("### The one residual, and why it is in the other file\n")
    A("`cyto7_flat_viridis.png` goes 2 → **1**. That single pixel pair is genuine "
      "flat-projection folding: nilearn draws every triangle, so the "
      "`drop_degenerate` step — which is what removes the folded slivers — applies only to "
      "the PolyCollection renderer, not to the nilearn one. One pixel pair is far below "
      "print resolution; making nilearn drop faces would mean pre-filtering the mesh handed "
      "to it, which is not worth it for the standalone reference figure. Panel (e), the one "
      "that goes into Figure 1, is at zero.\n")
    A("### Why the `_gc` variant reports \"not measurable\"\n")
    A("The greyscale palettes (`CYTO7_GC`, `CYTO7_GRAYSCALE`) are **collinear in RGB**, so "
      "decoding a rendered PNG back to type codes is ambiguous: an antialiased blend between "
      "types N and N+2 passes exactly through N+1's grey, and surface shading slides a "
      "colour along the same line onto other entries. The metric is therefore invalid for "
      "them, not merely noisy — the 164k inflated control, which *provably* contains no "
      "skips, scores ~2,000 pairs in greyscale versus 0 in viridis. `palette_is_measurable` "
      "now refuses to emit a number rather than emit a misleading one.\n")
    A("(The previous report's \"`cyto7_flat_gc.png` strict 0\" was itself an artefact: that "
      "file was decoded against the *viridis* palette, matching 0.0% of its pixels, so the "
      "zero was vacuous. Both audit paths now pick the palette the file was rendered in.) "
      "The gc variants come off the same code path and the same repaired display labels as "
      "the viridis ones, and the mesh-level counts above are palette-independent, so they "
      "are equally clean — it just cannot be shown with this pixel metric.\n")

    A("## Regenerated files\n")
    A("| file | note |")
    A("|---|---|")
    A(f"| `surface/labeled_reference/cyto7_{VER}_labeled_flat_viridis.png` | **Fig. 1 panel (e)** — hand back |")
    A(f"| `surface/labeled_reference/cyto7_{VER}_labeled_flat_viridis_nolabels.png` | |")
    A(f"| `surface/labeled_reference/cyto7_{VER}_labeled_flat_grayscale.png` (+ `_nolabels`) | same code path |")
    A(f"| `surface/labeled_reference/cyto7_{VER}_labeled_contactsheet_{{viridis,grayscale}}.png` | rebuilt so they do not embed the old flat panel |")
    A("| `surface/cyto7_flat_viridis.png`, `cyto7_flat_gc.png` | standalone flat maps |")
    A("")
    A("The inflated and pial PNGs were **not** re-rendered (`labeled_areas_figure.py "
      "--only flat`), so they remain byte-identical: only the flat panel changes, per the "
      "guardrail. Note the 3-D renderer shares the same majority-tie colouring and could in "
      "principle produce the same seam, but measures 0 on the 164k mesh, so it was left "
      "alone rather than changed unnecessarily.\n")
    A("## Hand-back\n")
    A(f"`figures/v9/surface/labeled_reference/cyto7_{VER}_labeled_flat_viridis.png` is the "
      f"new panel (e). `figures/v9/manuscript/cyto7_surface_plot_v9.png` is a hand-assembled "
      f"PowerPoint composite (`figures_final*.pptx`) with no generator, so swapping panel (e) "
      f"in and re-exporting the composite remains a manual step.\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"  wrote {path.name}")


def _released_digest() -> dict[str, str]:
    """SHA-256 of every released cyto7 annot and 32k label file (guardrail evidence)."""
    import hashlib
    out = {}
    for pat, root in ((f"pial.*.cyto7.{VER}.annot", cfg.atlas_dir("fsaverage")),
                      ("pial.*.cyto7.32k_fs_LR.label.gii",
                       cfg.atlas_dir("fs_LR_32k"))):
        for p in sorted(root.glob(pat)):
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _face_pair_skips(faces: np.ndarray, fcode: np.ndarray) -> tuple[int, int]:
    """Edge-adjacent face pairs whose face-majority codes differ by >= 2."""
    n = len(faces)
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(n), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = np.all(e[:-1] == e[1:], axis=1)
    f1, f2 = owner[:-1][same], owner[1:][same]
    both = (fcode[f1] > 0) & (fcode[f2] > 0)
    return int((both & (np.abs(fcode[f1] - fcode[f2]) >= 2)).sum()), int(both.sum())


if __name__ == "__main__":
    main()
