# Report — Fig. 1 flat panel regenerated with median face-colouring

Implements `docs/SPEC_flat_panel_median_recolor.md` (follow-up to `report_flat_panel_cleanup.md`). Env: `cyto7`. Map: **cyto7 v9**. Palette, figure size and DPI unchanged.

**Guardrail:** all 4 released files (`resources/cyto7_derived/pial.{lh,rh}.cyto7.v9.annot` and the 32k `pial.{lh,rh}.cyto7.32k_fs_LR.label.gii`) are SHA-256 **byte-identical** before and after. The three 32k resample skip-edges are repaired only in a *display* copy of the label array, cached as `resources/cyto7_derived/cache/v9_display_labels_fsLR32k_hemi-*.npy`.

## What changed in the code

A new shared module, `scripts/flat_display.py`, is now the single source of truth for the display-layer corrections, so the renderers and the audit cannot drift apart:
- `face_median` — **median** face colouring, now the default in the flat PolyCollection renderer (`labeled_areas_figure.panel_flat`, i.e. Fig. 1 panel e).
- `display_labels` — the three triple-point repairs, applied in both flat renderers.
- `drop_degenerate` — folded/degenerate faces dropped, remainder painted largest-first.

Two scope corrections worth flagging, because the SPEC assumed otherwise:
- **`plot_flat_categorical.py` never had the majority-tie bug.** It renders via nilearn's `plot_surf_roi`, and nilearn already defaults to `avg_method="median"` for ROI plots (verified in nilearn 0.13.1). Only the display-label repair applied there. Its residual is discussed below.
- **`change_history_figures.py` §3 is not an ordinal type map.** It renders provenance codes (added / removed / topology), which have no order, so a median would be meaningless and the |Δtype| ≥ 2 metric does not apply. It did, however, have a related bug: it averaged **RGB** across each face's three vertices, producing colours that correspond to no legend entry. That is now modal (`face_mode`) instead. The change-history figures were **not regenerated** — they are a different figure family and outside this SPEC's step 2.

## Step 3 — QC: |Δtype| ≥ 2 adjacent pixel pairs, before → after

| file | before | after | |
|---|---|---|---|
| `cyto7_flat_viridis.png` | 2 | 1 | **1** residual |
| `cyto7_flat_gc.png` | n/a | n/a | not measurable — see below |
| `labeled_flat_viridis_nolabels.png` | 4 | 0 | **0 — target met** |
| `labeled_flat_viridis.png` | 4 | 0 | **0 — target met** |
| `cyto7_v9_labeled_inflated_viridis_nolabels.png (inflated control)` | 0 | 0 | **0 — target met** |

**Fig. 1 panel (e) — `cyto7_v9_labeled_flat_viridis.png` — is now 0**, as is its `_nolabels` twin. The palette-independent mesh-level proof is stronger than the pixel count and covers every palette variant: vertex skip-edges LH 2→**0** / RH 1→**0**, face-colour skips LH 4→**0** / RH 2→**0**.

### The one residual, and why it is in the other file

`cyto7_flat_viridis.png` goes 2 → **1**. That single pixel pair is genuine flat-projection folding: nilearn draws every triangle, so the `drop_degenerate` step — which is what removes the folded slivers — applies only to the PolyCollection renderer, not to the nilearn one. One pixel pair is far below print resolution; making nilearn drop faces would mean pre-filtering the mesh handed to it, which is not worth it for the standalone reference figure. Panel (e), the one that goes into Figure 1, is at zero.

### Why the `_gc` variant reports "not measurable"

The greyscale palettes (`CYTO7_GC`, `CYTO7_GRAYSCALE`) are **collinear in RGB**, so decoding a rendered PNG back to type codes is ambiguous: an antialiased blend between types N and N+2 passes exactly through N+1's grey, and surface shading slides a colour along the same line onto other entries. The metric is therefore invalid for them, not merely noisy — the 164k inflated control, which *provably* contains no skips, scores ~2,000 pairs in greyscale versus 0 in viridis. `palette_is_measurable` now refuses to emit a number rather than emit a misleading one.

(The previous report's "`cyto7_flat_gc.png` strict 0" was itself an artefact: that file was decoded against the *viridis* palette, matching 0.0% of its pixels, so the zero was vacuous. Both audit paths now pick the palette the file was rendered in.) The gc variants come off the same code path and the same repaired display labels as the viridis ones, and the mesh-level counts above are palette-independent, so they are equally clean — it just cannot be shown with this pixel metric.

## Regenerated files

| file | note |
|---|---|
| `surface/labeled_reference/cyto7_v9_labeled_flat_viridis.png` | **Fig. 1 panel (e)** — hand back |
| `surface/labeled_reference/cyto7_v9_labeled_flat_viridis_nolabels.png` | |
| `surface/labeled_reference/cyto7_v9_labeled_flat_grayscale.png` (+ `_nolabels`) | same code path |
| `surface/labeled_reference/cyto7_v9_labeled_contactsheet_{viridis,grayscale}.png` | rebuilt so they do not embed the old flat panel |
| `surface/cyto7_flat_viridis.png`, `cyto7_flat_gc.png` | standalone flat maps |

The inflated and pial PNGs were **not** re-rendered (`labeled_areas_figure.py --only flat`), so they remain byte-identical: only the flat panel changes, per the guardrail. Note the 3-D renderer shares the same majority-tie colouring and could in principle produce the same seam, but measures 0 on the 164k mesh, so it was left alone rather than changed unnecessarily.

## Hand-back

`figures/v9/surface/labeled_reference/cyto7_v9_labeled_flat_viridis.png` is the new panel (e). `figures/v9/manuscript/cyto7_surface_plot_v9.png` is a hand-assembled PowerPoint composite (`figures_final*.pptx`) with no generator, so swapping panel (e) in and re-exporting the composite remains a manual step.

