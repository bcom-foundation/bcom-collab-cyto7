# cyto7 vs von-Economo-derived cortical types — comparison report

*Mesh: **fsaverage** (163,842 verts/hemi, exact match — no resampling).*

*cyto7 source: **v9**.*

> **Disagreement is not error.** The von-Economo-derived map is *area-level* (piecewise-constant within von Economo areas); cyto7 is drawn at *vertex* resolution. Most disagreement is cyto7 legitimately refining a coarse areal scaffold. Only large, interior, high-ordinal-gap, independently-discordant patches are genuine review candidates — and the final call belongs to the human authors against histology and the García-Cabezas protocol.

## 1. Headline numbers (pooled, common support)

- Common-support vertices: **291068** (cyto7 ∈ types 2–7 **and** von Economo has an isocortical 6-type).
- Overall agreement: **52.5%**
- **Quadratic-weighted κ (headline): 0.699**
- Cohen's κ (unweighted): 0.361
- Adjusted Rand Index: 0.188
- Mean |Δordinal|: 0.558; mean signed Δordinal: +0.120 (+ ⇒ cyto7 more differentiated than the von-Economo area on average).

Per-hemisphere values and per-type Dice/IoU/Δ are in `agreement_summary.csv` / `.txt`; the full confusion matrix is in `confusion_matrix.csv` and `confusion_matrix.png`.

## 2. Figures

- `sidebyside_surface.png` — cyto7 vs von-Economo-derived type, shared 6-colour scale.
- `difference_map.png` — per-vertex disagreement category (agree / off-by-one ± / off-by-≥2).
- `delta_ordinal_signed.png` — signed Δordinal (red = cyto7 more differentiated).
- `cyto7_allocortex_coverage.png` — cyto7 allocortex, the scope the von-Economo 6-type map lacks (LH=4147, RH=3193 vertices).
- `review_candidates.png` — top clusters outlined, coloured by priority rank.

## 3. Candidate regions for review

`review_candidates.csv` ranks **27** disagreement clusters (≥ 30 vertices) by a composite `review_priority` (normalised size + ordinal gap + 0.5·interiorness).

- **Genuine review candidates** (large ordinal gap |Δ|≥2, or a large *systematic* off-by-one patch whose independent T1w/T2w myelin signal favours the von Economo type): **3**. These are worth re-checking against histology / the protocol — *not* presumed errors.
- **Expected refinements** (off-by-one and/or boundary-adjacent): **24**. These are the anticipated consequence of vertex-level painting over coarse areal edges.

Top 10 by review priority:

| hemi | dominant_aparc | area_mm2 | cyto7_type | voneconomo_type | mean_signed_delta_ordinal | median_dist_to_areal_edge_mm | myelin_adjudication | review_priority | category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lh | lateraloccipital | 3288.2 | eulaminate II | eulaminate I | 0.365 | 3.35 | cyto7 supported by myelin | 2.1983 | refinement (off-by-one, interior) |
| rh | postcentral | 24062.4 | eulaminate III | eulaminate II | 0.249 | 3.26 | von-Economo supported by myelin | 2.1879 | review candidate (systematic off-by-one, myelin favours von Economo) |
| lh | postcentral | 20086.8 | eulaminate III | eulaminate II | 0.293 | 3.17 | von-Economo supported by myelin | 2.0515 | review candidate (systematic off-by-one, myelin favours von Economo) |
| lh | precentral | 753.7 | eulaminate II | eulaminate III | -1.0 | 3.65 | von-Economo supported by myelin | 1.0467 | review candidate (systematic off-by-one, myelin favours von Economo) |
| rh | rostralmiddlefrontal | 472.1 | eulaminate II | eulaminate I | 1.0 | 3.31 | von-Economo supported by myelin | 0.939 | refinement (off-by-one, interior) |
| rh | inferiorparietal | 298.4 | eulaminate I | eulaminate II | -1.0 | 3.39 | cyto7 supported by myelin | 0.89 | refinement (off-by-one, interior) |
| lh | rostralmiddlefrontal | 426.6 | eulaminate II | eulaminate I | 1.0 | 2.88 | von-Economo supported by myelin | 0.8668 | expected refinement (off-by-one, boundary-adjacent) |
| lh | supramarginal | 193.1 | eulaminate I | eulaminate II | -0.482 | 2.08 | cyto7 supported by myelin | 0.6538 | expected refinement (off-by-one, boundary-adjacent) |
| rh | cuneus | 88.9 | koniocortex | eulaminate III | 1.0 | 2.21 | cyto7 supported by myelin | 0.5709 | expected refinement (off-by-one, boundary-adjacent) |
| lh | rostralmiddlefrontal | 111.5 | eulaminate II | eulaminate I | 0.641 | 0.95 | von-Economo supported by myelin | 0.4276 | expected refinement (off-by-one, boundary-adjacent) |

The `myelin_adjudication` column uses the independent T1w/T2w myelin ordering (myelin rises with differentiation): clusters tagged *von-Economo supported by myelin* are the strongest review candidates; *cyto7 supported by myelin* are most likely legitimate refinements.

## 4. Caveats (must be read with any quoted number)

1. **Lookup is a verification checkpoint.** The von-Economo-area → type assignment (`von_economo_cortical_types.csv`, **verified=FALSE**) was read from García-Cabezas et al. (2020) Tables 4–7. **16 of 40** typed areas aggregate García-Cabezas sub-areas of differing type and are flagged `ambiguous`; the authors must confirm these before quoting.
2. **6 vs 7 types.** The von-Economo-derived map is isocortex-only. cyto7 allocortex (code 1) has no counterpart and is excluded from agreement, reported separately as cyto7's added coverage.
3. **Area-level vs vertex-level.** The comparison map's boundaries are von Economo areal edges; disagreement near those edges is largely expected refinement, not error.
4. **Mesh / resampling.** This run used **fsaverage**. von Economo labels were produced directly on fsaverage via `mris_ca_label` (exact); the fsaverage5 option uses a nested-icosahedron label subset (nearest-neighbour).
5. **Descriptive statistics.** No correction for spatial autocorrelation; no naive p-values.
6. **Excluded limbic areas.** HA, HB, HC (periallocortex) carry no isocortical 6-type and are excluded.
7. **Native 5-type scheme is not used here.** The micaopen `economo7` map is the native von Economo types (agranular/frontal/parietal/polar/granular + limbic/insular), which are *not* ordinally aligned with cyto7; the headline comparison uses the García-Cabezas 6-type assignment only.

## 5. Reproduce

```bash
python scripts/fetch_voneconomo_atlas.py            # atlas + lookup CSV
wsl.exe -d Ubuntu-20.04 bash scripts/make_economo_annot_fsaverage.sh  # annot on fsaverage
python scripts/compare_cyto7_vs_voneconomo.py --mesh fsaverage --annot-version v9
```