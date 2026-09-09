# Report — Figure 1 flat panel: apparent >1-type adjacencies

Implements `docs/SPEC_flat_panel_cleanup.md`. Env: `cyto7`. Map: **cyto7 v9**. Runtime 0.6 min.

**Guardrail: the released map is untouched.** All 4 released files (`resources/cyto7_derived/pial.{lh,rh}.cyto7.v9.annot` and the 32k `pial.{lh,rh}.cyto7.32k_fs_LR.label.gii`) were SHA-256 hashed before and after this run and are **byte-identical**. (Hashing rather than `git status`, because the working tree already carries unrelated pre-existing changes from the repo-cleanup branch; hashes prove *this* run changed nothing.) Everything written goes to `figures/v9/surface/flat_panel_audit/`.

## Verdict — the artifact is negligible, and it is not a flattening artifact

> Per the SPEC's own step-1 decision rule the answer is **stop: no pixel surgery**. Across the whole flat panel there are **58 adjacent pixel pairs** with |Δtype| ≥ 2 (of ~3.5 M surface pixels per hemisphere), in a handful of spots each a few pixels across — below visible resolution in print. In the shipped PNGs, decoded strictly against the seven type colours, the count is 0–4 pairs.

> **The cause is mostly not what the SPEC assumed, so the proposed caption line would be inaccurate.** The 58 pairs decompose as:
>   - **~49 pairs — the 164k → 32k fs_LR resampling** plus the renderers' face-majority tie-break. The flat panel is the *only* panel of Fig. 1 drawn on the 32k mesh; nearest-neighbour resampling pinches out one-vertex-wide intervening bands at 3 triple points. Provably removable (below).
>   - **~9 pairs — genuine flat-projection folding**, the SPEC's hypothesis. Real, but a small minority: only LH 0 orientation-flipped of 59,013 faces, RH 9 orientation-flipped of 58,914 faces.

> A corrected caption line is given at the end.

## Step 1 — quantification

### The released map is clean on its native mesh (ground truth)

R1 skip-edges on the **164k fsaverage** released annot: **LH 0, RH 0** — confirms §3.2. Figure 1 panels c/d are rendered directly from this annot (`resources/fsaverage_surfaces/{lh,rh}.inflated` + `pial.{lh,rh}.cyto7.v9.annot`), which is why they show no skips at all.

### The 32k fs_LR display mesh is where the skips appear

| hemi | vertex skip-edges (32k) | face-majority skips | adjacent face pairs | flipped faces | degenerate faces |
|---|---|---|---|---|---|
| LH | **2** | 4 | 87,347 | 0 | 9 |
| RH | **1** | 2 | 87,059 | 9 | 27 |

Every offending vertex sits at a triple point where the intervening type is present in its own mesh neighbourhood but lost its vertex in the resample:

| hemi | vertex | resampled type | partner type | intervening type available? |
|---|---|---|---|---|
| LH | 1193 | 3 Dysgranular | 5 Eulaminate II | yes → 4 Eulaminate I |
| LH | 11263 | 4 Eulaminate I | 6 Eulaminate III | yes → 5 Eulaminate II |
| RH | 1470 | 4 Eulaminate I | 6 Eulaminate III | yes → 5 Eulaminate II |

### Rasterised counts, three ways

| render | resolution | |Δtype| ≥ 2 pairs, strict | permissive (incl. antialiased) |
|---|---|---|---|
| `cyto7_flat_viridis.png` | 2602×4709 | 2 | 33,305 |
| `cyto7_flat_gc.png` | 2602×4709 | 0 | 44,345 |
| `labeled_flat_viridis_nolabels.png` | 1821×3173 | 4 | 14,317 |
| `labeled_flat_viridis.png` | 1821×3173 | 4 | 25,745 |
| `cyto7_v9_labeled_inflated_viridis_nolabels.png (inflated control)` | 1583×4710 | 0 | 23,215 |
| re-render LH, antialiasing off | 1950×1800 | 37 | — |
| re-render RH, antialiasing off | 1950×1800 | 21 | — |

The permissive column snaps *every* non-background pixel to its nearest type colour, so it counts antialiased boundary pixels as well; it is an upper bound on what a reader could perceive, not a count of real adjacencies. The inflated negative control behaves as predicted: `cyto7_v9_labeled_inflated_viridis_nolabels.png (inflated control)` gives 0 strict pairs.

Location overlay: `flat_panel_artifact_overlay.png` (circled, since at true size the spots are only a few pixels wide).

## Step 2–3 — the repair, and what it buys

Because the cause is 3 display vertices rather than folding, the minimal fix is better than any of the SPEC's three options: re-label those vertices in the **display copy** of the 32k array to the intervening type. This is ordinality-respecting by construction — the inserted type already occurs among the vertex's labelled mesh neighbours, and each candidate is rejected unless it removes the skip without creating a new one — and it never touches the released annot.

A second, independent display bug turned up while testing the repair, and it is worth fixing regardless: both flat renderers colour each triangle by the **majority** of its three vertex labels, breaking ties toward the *lowest* code. A face straddling types {4,5,6} is therefore painted 4 while its neighbour {5,6,6} is painted 6 — a |Δ| = 2 seam manufactured from vertex labels that never skip. Colouring by the **median** instead cannot do this: adjacent faces share two vertices whose labels differ by at most 1 (once vertex skips are repaired), and the median of a triple containing those two always lies between them, so adjacent face colours differ by at most 1 *by construction*. The median is also the right summary for an ordinal quantity. Repair = 3 display vertices + median colouring + dropping folded faces.

| hemi | vertex skip-edges | face-colour skips | raster pairs | vertices changed | max per-type area drift |
|---|---|---|---|---|---|
| LH | 2 → **0** | 4 → **0** | 37 → **3** | 2 of 29,387 | -0.803% |
| RH | 1 → **0** | 2 → **0** | 21 → **6** | 1 of 29,344 | -1.446% |

Total raster pairs 58 → **9** — **9 residual pairs**, located in the overlay figure. That residual is the SPEC's original hypothesis after all, and it is now isolated: with vertex-level and face-colour-level skips both provably zero, the only mechanism left is genuine flat-projection folding — a non-degenerate triangle landing on top of a distant part of the map. Removing it needs true 2-D overlap-versus-mesh-distance testing (SPEC step 2.1's second criterion); at 9 pixel pairs that is not worth the complexity. Ordering faces by areal distortion instead of size was tried as a cheap proxy and is not reliable (LH residual 3→1 but RH 6→14). Per-type pixel areas move by at most -1.446%, so no type gains or loses meaningful area; the label change is 3 vertices out of ~58,700.

**Applied to the shipped figures: NO.** Per the SPEC's step-1 rule the artifact is minor, so the shipped flat panels were left exactly as they are and only the caption line is needed. Re-run with `--apply` to regenerate `cyto7_flat_{gc,viridis}.png` and the labeled flat reference from the repaired display labels if you would rather ship a provable zero.

## Step 4 — caption line

The SPEC's proposed sentence attributes the artifact to flattening distortion and says the panel is "smoothed for display"; on this evidence neither is true. Corrected wording, same length and purpose:

> On the flat surface, a small number of apparent adjacencies between non-neighbouring cyto7 types are rendering artifacts of the flat display mesh — chiefly the resampling of the atlas from its native 164k fsaverage mesh to the 32k fs_LR mesh used for flattening, which can pinch out an intervening band only one vertex wide, and to a lesser extent local folding of the flat projection. They are not real transitions: the released map is topologically clean on its native mesh (no type skips, §3.2), and the inflated panels, drawn directly from that mesh, contain none.

(If `--apply` is used, drop the last clause about the inflated panels being artifact-free only if you also regenerate them — they are unaffected either way — and the sentence can end at "§3.2".)

## Fig. 1 composite — note for the author

`figures/v9/manuscript/cyto7_surface_plot_v9.png` is an **assembled** composite (panel a hand-drawn von-Economo plate, panel b a painting-tool screenshot, panels c/d/e rendered) — there is no generator script for the composite itself; it is put together in `figures/v9/manuscript/figures_final*.pptx`. Panel (e) is `surface/labeled_reference/cyto7_v9_labeled_flat_viridis.png`. So even under `--apply`, regenerating panel (e) does not regenerate the composite: that step is manual in PowerPoint, then re-export.

## Files

| file | contents |
|---|---|
| `flat_panel_audit.csv` | per-hemisphere mesh + raster counts, before (and after, if repaired) |
| `flat_panel_skip_vertices.csv` | the offending display vertices and the repair applied to each |
| `flat_panel_shipped_png_audit.csv` | strict + permissive counts for each shipped PNG |
| `flat_panel_artifact_overlay.png` | where the adjacencies are, per hemisphere |
| `report_flat_panel_cleanup.md` | this report |
