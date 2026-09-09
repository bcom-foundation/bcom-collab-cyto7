# RR18 (b): `manuscript/preprint/figures/` is a superseded staging folder

Marked as superseded, **not deleted**, as instructed. The decision on whether to remove it is
yours; this records what it would cost.

## What the two folders hold

| | folder | PNG files |
|---|---|---|
| stale | `manuscript/preprint/figures/` | 31 |
| shipped | `manuscript/preprint/26th_August_2026/figures/` | 19 |

13 files are byte-identical in both. The shipped folder is the one the live
`cyto7_manuscript.tex` resolves against, confirmed by the fact that its Figure 3 reference
(`figures/cyto7_support_map_myelin.png`, line 375) exists only there.

## The four that differ

Uploading from the stale folder today would regress all four.

| figure | stale version | shipped version |
|---|---|---|
| **Fig. 1** `cyto7_surface_plot_v9.png` | `148fe7090c62` · 2026-07-22 · 3,309,542 B | `1c5187d0d137` · 2026-08-27 · 3,714,978 B |
| **Fig. 3** myelin support map | `cyto7_confidence_map_myelin.png` · `c5ad475c29d1` · 2026-07-22 · 10,972,657 B | `cyto7_support_map_myelin.png` · `e0fe66673def` · 2026-09-02 · 1,449,812 B |
| **Fig. 5** `cyto7_structural_model_gradients.png` | `2dc75842e21d` · 2026-07-22 · 370,506 B | `36cdb006e29b` · 2026-09-02 · 573,738 B |
| **Fig. S3** `cyto7_supp_per_type_connectivity.png` | `708a3d267921` · 2026-07-22 · 297,930 B | `f81dc69b0d41` · 2026-08-26 · 317,729 B |

Figure 3 differs by name as well as content, because of the RR17 rename, so a name-matched
comparison alone would have missed it and reported only three.

Figure 5's shipped hash is the file this task just produced. At the time of the audit the
shipped version was `9aaefc583f3b`; the difference against the stale folder was already there
and is unchanged in kind.

## The trap that made this worth reporting

`manuscript/preprint/revision/FIGURE_UPLOAD_RULE.md` instructs uploading to Overleaf **from the
stale folder**. Following that document today silently regresses Figure 1 to the 22 July render
instead of the 27 August fix that is in the paper. A superseded notice naming the correct source
has been added at the top of that file; nothing else in it was changed.

## Evidence that the folder actively causes confusion

The stale folder contains `cyto7_structural_model_gradients_NEW_fig6.png` at `9aaefc583f3b`,
2026-08-26, 576,287 B. That hash is the **previous shipped Figure 5**, staged into the old folder
under a different filename. So the stale folder holds two files that both claim to be Figure 5,
under different names, and neither is the current one.

## The other 15 files

Present only in the stale folder, all dated 2026-07-22. They are an earlier naming generation
(`figure_1_atlas_surface.png`, `figure_2_vs_voneconomo.png`, `figure_3_confidence.png`,
`figure_4_myelin_regression.png`, `figure_5_fingerprint.png`, `figure_6_frequency_timescale.png`,
`figure_7_tractography.png`, `figure_8_allocortex.png`, the two `figure_1c_flat` variants, the two
`figure_1_atlas_surface` palette variants, `cyto7_confidence_map_myelin.png`,
`cyto7_supp_feature_gallery.png`, `cyto7_supp_frequency_timescale.png`).

Nothing in the live manuscript references any of these names. `figures/v9/manuscript/FIGURE_MANIFEST.md`
still documents that generation, which is a second reason the old folder looks current to a
reader: the manifest and the folder agree with each other, and both disagree with the paper.

## Recommendation

Delete the folder, once you are satisfied nothing outside the repository points at it. Three
files there have no counterpart anywhere else (`cyto7_supp_feature_gallery.png`,
`cyto7_supp_frequency_timescale.png` and the `_NEW_fig6` duplicate), so check those first if you
want to keep an archive. `FIGURE_MANIFEST.md` needs the same treatment as the upload rule, since
it describes the superseded filenames as if they were current; that is a documentation pass rather
than a figure change and was left alone here.
