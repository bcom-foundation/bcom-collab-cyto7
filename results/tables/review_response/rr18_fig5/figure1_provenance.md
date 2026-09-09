# RR18 (c): what produced Figure 1, and whether it can be regenerated

Reported, not fixed, as the task requires. Figure 1 was not re-rendered.

## The answer, plainly

**Figure 1 is a hand-assembled figure with no generator, and it cannot be reproduced from this
repository in the state that produced the shipped file.** The repo currently gives a reader no
indication of that, which is the part worth correcting.

## What produced it

`grep` across the whole of `scripts/` finds **no script that writes
`cyto7_surface_plot_v9.png`**. The only two references to the name are notes in
`flat_panel_cleanup.py` that say so explicitly:

> `figures/v9/manuscript/cyto7_surface_plot_v9.png` is an **assembled** composite (panel a
> hand-drawn von-Economo plate, panel b a painting-tool screenshot, panels c/d/e rendered) —
> there is no generator script for the composite itself; it is put together in
> `figures/v9/manuscript/figures_final*.pptx`.
> — `scripts/flat_panel_cleanup.py:464`

> `figures/v9/manuscript/cyto7_surface_plot_v9.png` is a hand-assembled PowerPoint composite
> (`figures_final*.pptx`) with no generator, so swapping panel (e) in and re-exporting the
> composite remains a manual step.
> — `scripts/flat_panel_cleanup.py:750`

So the composition is: **(a)** a hand-drawn von Economo plate, **(b)** a screenshot of the
painting tool, **(c)**, **(d)**, **(e)** rendered panels, assembled and exported from PowerPoint.
Only the rendered panels have generators. Panel (e) is
`figures/v9/surface/labeled_reference/cyto7_v9_labeled_flat_viridis.png`, produced by
`scripts/flat_display.py`.

## Why the shipped file and the source tree disagree, and which is right

| artefact | date | size | what it is |
|---|---|---|---|
| `manuscript/preprint/26th_August_2026/figures/cyto7_surface_plot_v9.png` | 2026-08-27 | 3,714,978 | **shipped, correct** |
| `figures/v9/manuscript/cyto7_surface_plot_v9.png` | 2026-07-22 | 3,309,542 | stale export |
| `manuscript/preprint/figures/cyto7_surface_plot_v9.png` | 2026-07-22 | 3,309,542 | same stale export |
| `figures/v9/surface/labeled_reference/cyto7_v9_labeled_flat_viridis.png` | 2026-07-28 | 523,165 | panel (e), **after** the median-recolour fix |
| `figures/v9/manuscript/figures_final.pptx` | 2026-07-22 | 16,720,397 | assembly source |
| `figures/v9/manuscript/figures_final_v2.pptx` | 2026-07-22 | 7,796,144 | assembly source |

The sequence reads: the composite was exported on 22 July; panel (e) was regenerated on 28 July
by the flat-panel median-recolour work; the composite was re-exported on 27 August with the new
panel (e) swapped in. That 27 August export is what the manuscript ships, and it is the correct
one.

**The gap that matters: both `.pptx` files are still dated 22 July**, five weeks before the
shipped export. The assembly source in the repository is therefore **not** the state that
produced the shipped figure. Re-exporting from the saved `.pptx` today would reinstate the old
panel (e) and produce the 22 July composite, not the shipped one.

That is why this must not be regenerated, and it is a stronger reason than "a regeneration
probably would not reproduce the manual fix": the inputs that would be used are demonstrably the
pre-fix ones.

## What the release should ship

The repository should stop implying that Figure 1 is reproducible. Two changes, neither of which
this task performs:

1. **Save the assembly source in the state that produced the shipped export.** Whoever re-exported
   on 27 August had a PowerPoint file with the new panel (e) in it. Saving that file into
   `figures/v9/manuscript/` closes the gap and makes the figure re-exportable, even if not
   scriptable.
2. **Ship a provenance note beside the figure**, so nobody tries to regenerate it and nobody reads
   the absence of a generator as an oversight.

Also worth doing while there: `figures/v9/manuscript/cyto7_surface_plot_v9.png` should be brought
up to the shipped 27 August export, so the analysis tree and the manuscript agree. This is the
reverse of the RR17 defect, where the tree was right and the release was stale.

## Proposed provenance note

To sit at `figures/v9/manuscript/FIGURE_1_PROVENANCE.md`, and to be summarised in one sentence in
the public release's provenance notes:

> **Figure 1 is a hand-assembled composite and has no generator script.**
>
> Panels (c), (d) and (e) are rendered from the v9 atlas by scripts in this repository; panel (e)
> is `figures/v9/surface/labeled_reference/cyto7_v9_labeled_flat_viridis.png`, produced by
> `scripts/flat_display.py`. Panel (a) is a hand-drawn plate after von Economo and Koskinas and
> panel (b) is a screenshot of the surface painting tool; neither is derived from the atlas data
> and neither can be regenerated from it.
>
> The panels are assembled and exported manually in
> `figures/v9/manuscript/figures_final_v2.pptx`. The shipped export dates from 2026-08-27 and
> incorporates the corrected panel (e) from the flat-panel median-recolour fix of 2026-07-28.
>
> Running the analysis pipeline reproduces every rendered panel but does **not** reproduce this
> composite. Re-exporting it is a manual step in PowerPoint. The figure is included in the release
> as a published image rather than as a reproducible product, and the underlying atlas data for
> panels (c) to (e) ships in full.

## One consequence for the public release

The release currently presents all figures alike. If a user regenerates the figure set from the
code, every figure will match except Figure 1, with no explanation. Stating the exception is
cheaper than defending it later, and it costs the paper nothing: panels (a) and (b) are
illustrative, and every quantitative panel is reproducible.
