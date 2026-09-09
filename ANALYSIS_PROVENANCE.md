# Analysis provenance

This document maps every analysis reported in the paper to the script that ran it and the file
in this repository that holds its numbers. It is written for a reader who wants to check a
figure or a table: find the row, open the output file, and if you want to re-run it, use the
script named beside it.

It does not restate the reasoning behind any analysis. That is in the manuscript and its
annexes, which are cross-referenced here rather than summarised.

## How to read an entry

| column | meaning |
|---|---|
| **Analysis** | what was computed |
| **Script** | path under `code/`, run from the repository root |
| **Inputs** | what it reads. Third-party datasets are named but not redistributed; see "Inputs not in this repository" |
| **Null / seed** | the resampling null and its parameters, where inference was done |
| **Output** | the file holding the numbers, under `results/` |

## Conventions that apply throughout

- **Surfaces.** The atlas is defined on fsaverage 164k. Feature analyses run on fs_LR 32k after
  resampling with Connectome Workbench; geometry uses the fsaverage white surface.
- **Spin nulls.** Alexander-Bloch spherical rotations, **N = 1000, seed 0**, with the two-tailed
  plus-one convention `p = (1 + #{|rho_null| >= |rho_obs|}) / (N + 1)`, so the smallest
  attainable value is `1/1001 = 0.000999`. Vertex-level nulls are cached and reused across
  analyses so that families are comparable.
- **Parcel-level nulls.** Where a statistic is defined on the 68 Desikan parcels, the null
  rotates the parcellation itself, 1000 rotations at numpy seed 0. Section 2.6 of the paper and
  the p-value convention audit below both concern this distinction; read that entry before
  comparing a parcel-level p to a vertex-level one.
- **Multiple comparisons.** Benjamini-Hochberg within declared families, never pooled across
  them. Every test, its family, its null and its status is tabulated in
  `results/tables/review_response/rr2_table/outcome_table.csv`, which is the supplementary
  outcome table.
- **Atlas version.** All released results use **v9**. Superseded versions are kept under
  `atlas/provenance/versions/` for provenance only.

---

# Primary analyses

## The atlas and its quality control

| | |
|---|---|
| **Analysis** | Topological consistency audit of the released map: skip-edges, ring structure, component and hole counts per type |
| **Script** | `code/pipeline/audit_topology.py` |
| **Inputs** | `atlas/fsaverage/pial.{lh,rh}.cyto7.v9.annot` |
| **Null / seed** | none; deterministic geometry |
| **Output** | `results/tables/topology/topology_audit_v9.json`, `results/tables/topology/ring_analysis_v9.csv` |

| | |
|---|---|
| **Analysis** | Anatomical support score: a weighted geometric mean of four anatomy-only components (atlas concordance, topology, geometry, prior), with a data overlay computed but deliberately excluded from the score |
| **Script** | `code/pipeline/build_support_map.py` |
| **Inputs** | the v9 annot; the von Economo derived map; FreeSurfer ex-vivo and Glasser reference labels |
| **Null / seed** | none for the score itself; see the calibration entry below |
| **Output** | `atlas/fsaverage/support/pial.{lh,rh}.cyto7.support*.shape.gii` (combined map, four components, data overlay, categorical annot) and `results/tables/support/support_summary.csv` for per-type medians |

The score is **not** a statistical confidence measure and was renamed to say so. The
atlas-free variant that section 3.4 promises can be rebuilt from the shipped components as
`exp(mean(log(topo, geom, prior)))`; that this reproduces the internally used variant exactly
is recorded in `results/tables/review_response/rr17_confidence_promotion/`.

| | |
|---|---|
| **Analysis** | Calibration of the support score against disagreement with the area-level map, and against an independent ex-vivo histological patch |
| **Script** | `code/pipeline/reviewer_response.py` |
| **Inputs** | the support components; the von Economo derived map; FreeSurfer ex-vivo entorhinal and perirhinal labels |
| **Null / seed** | none; AUC and per-tertile error rates are descriptive |
| **Output** | `results/tables/support_calibration.csv` (disagreement AUC per component) and `results/tables/support_histology_patch.csv` (labelling error by score tertile) |

Both quote the **benchmark-independent** score, which excludes the atlas-concordance term
because that term is derived from the very map the benchmark compares against.

## The area-level benchmark

| | |
|---|---|
| **Analysis** | Vertex-level comparison against the von Economo derived map: agreement, weighted kappa, confusion matrix, and the added-value test on the vertices where the two maps disagree |
| **Script** | `code/pipeline/compare_cyto7_vs_voneconomo.py`, then `code/pipeline/reviewer_response.py` |
| **Inputs** | the v9 annot; `atlas/fsaverage/voneconomo/`; the three arbiter feature maps (T1w/T2w myelin, cortical thickness, principal functional gradient) |
| **Null / seed** | spin null, N = 1000, seed 0, rotating cyto7 at its own vertex granularity and rebuilding per-type medians from the rotated labels |
| **Output** | `results/tables/vs_voneconomo/agreement_summary.csv`, `confusion_matrix.csv`; `results/tables/added_value_global.csv`, `added_value_localized.csv`, `added_value_by_support.csv` |

The three arbiters were fixed in advance as the axis features. The win fraction is reported
against a chance baseline computed from the same null, not against 50 per cent.

## Structure and function

| | |
|---|---|
| **Analysis** | Association of each macroscale feature with cortical type: myelin, thickness, functional gradient, gene expression PC1, MEG band powers, intrinsic timescale, spectral centroid |
| **Script** | `code/pipeline/summarise_functional_features.py`, with `code/pipeline/extend_structure_function.py` for the tier and per-version variants |
| **Inputs** | neuromaps feature maps and HCP-derived MEG maps, both fetched by `code/fetch/fetch_neuromaps_features.py`; the v9 labels |
| **Null / seed** | spin null, N = 1000, seed 0; BH-FDR within the structure-function family |
| **Output** | `results/tables/structure_function/functional_summary_table_v9.csv` |

| | |
|---|---|
| **Analysis** | Myelin progression across the type axis, and the regression used for Figure 4 |
| **Script** | `code/figures/plot_myelin_progression_regression.py` |
| **Inputs** | the T1w/T2w map; the v9 labels |
| **Null / seed** | spin null, N = 1000, seed 0 |
| **Output** | `results/tables/structure_function/` (regression summary written beside the figure) |

## MEG dynamics

| | |
|---|---|
| **Analysis** | Group-level source-reconstructed MEG dynamics along the type axis: intrinsic timescale, aperiodic exponent, spectral centroid, aperiodic-corrected band powers |
| **Script** | `code/pipeline/meg_dynamics_v2.py` (per subject, MATLAB and FieldTrip), `code/pipeline/meg_v2_build_maps.py` (group maps), `code/pipeline/meg_dynamics_v2_analyse.py` (statistics) |
| **Inputs** | HCP-MEG resting-state recordings for the subjects listed in `results/subjects_used.txt`; FieldTrip; a single-shell forward model |
| **Null / seed** | spin null, N = 1000, seed 0; BH-FDR within the dynamics family |
| **Output** | `results/tables/structure_function/meg_dynamics_v2_summary.csv`, and `meg_dynamics_v2_summary_fdr_singlemember.csv` for the single-member family coding |

| | |
|---|---|
| **Analysis** | Per-subject rather than group-level association, as a generalisation check |
| **Script** | `code/pipeline/meg_subjectlevel.py` |
| **Inputs** | the same per-subject outputs |
| **Null / seed** | per-subject Spearman, then a group-level test |
| **Output** | `results/report_hcp_meg_v2.md` |

Two MEG entries are reported in the paper with an explicit caveat: the aperiodic-exponent
analysis was stopped at its data gate, recorded in `results/report_hcp_meg_aperiodic.md`, and
two band-power maps are near-constant and are flagged as not estimable in the outcome table.

## Receptors, genes and disease

| | |
|---|---|
| **Analysis** | Neurotransmitter receptor and transporter densities along the type axis: 19 individual PET maps, their first principal component, and pharmacological-class composites |
| **Script** | `code/pipeline/receptor_pc1.py`, `code/pipeline/receptor_type_connectivity.py` |
| **Inputs** | the Hansen PET receptor collection, via neuromaps |
| **Null / seed** | spin null, N = 1000, seed 0; BH-FDR within the 19-map family and within the composites family separately |
| **Output** | `results/tables/structure_function/receptor_composites.csv`, `receptor_diversity.csv`; `results/report_receptor_type.md` |

| | |
|---|---|
| **Analysis** | Laminar marker genes as a non-circular test of the defining criterion, RORB for layer IV among them |
| **Script** | `code/pipeline/layer_marker_genes.py` |
| **Inputs** | AHBA microarray expression, parcellated to Desikan-68 with abagen |
| **Null / seed** | **parcel-level** null: 1000 rotations of the Desikan parcellation at numpy seed 0. See the p-value convention audit below before comparing these p-values with vertex-level ones |
| **Output** | `results/tables/definitional/layer_marker_genes.csv`, `layer_marker_expression.csv` |

| | |
|---|---|
| **Analysis** | Disease vulnerability against cortical type, for eight ENIGMA case-control effect-size maps |
| **Script** | `code/figures/fig9_predictions.py` |
| **Inputs** | ENIGMA cortical effect-size maps via the ENIGMA toolbox |
| **Null / seed** | parcel-level rotations, 1000, seed 0; a partial variant controls the functional gradient |
| **Output** | `results/tables/external/disease_vulnerability.csv` |

| | |
|---|---|
| **Analysis** | Negative control: a feature with no expected relation to type, run through the identical pipeline |
| **Script** | `code/pipeline/negative_control.py` |
| **Inputs** | sulcal depth |
| **Null / seed** | spin null, N = 1000, seed 0 |
| **Output** | `results/tables/definitional/negative_control.csv` |

## External validation

| | |
|---|---|
| **Analysis** | Molecular and histological reference comparison: gene expression PC1, receptor PC1, BigBrain staining-intensity profiles and their skewness |
| **Script** | `code/pipeline/external_validation.py`, `code/pipeline/bigbrain_profile_validation.py`, `code/pipeline/bigbrain_added_value.py` |
| **Inputs** | AHBA; the PET collection; BigBrain profiles via BigBrainWarp |
| **Null / seed** | spin null, N = 1000, seed 0 |
| **Output** | `results/tables/structure_function/external_validation_table.csv`; `results/tables/structure_function/bigbrain_addedvalue/` |

The BigBrain comparison is reported as inconclusive in a single specimen, and its
pre-specified validity gate is recorded with the outputs rather than only in the text.

## Tractography

| | |
|---|---|
| **Analysis** | Short-range cortico-cortical streamline counts between type pairs, and their falloff with type distance |
| **Script** | `code/pipeline/compute_tractography_connectivity.py`, `code/pipeline/write_tracto_report_v9.py` |
| **Inputs** | a population-averaged tractogram; the v9 labels; a short/long cutoff of 80 mm |
| **Null / seed** | permutation over type orderings; the per-type family uses an exhaustive permutation over all 7! = 5040 orderings, so its resolution floor is 1/5041 and it is the one family whose p-values are printed to four decimals |
| **Output** | `results/tables/tractography/`, summarised in `results/tables/tractography/REPORT_tractography_v9.md` |

This analysis is reported in an annex rather than the main results, and no inferential claim is
drawn from it. The controls that led to that decision are listed under follow-up analyses.

---

# Follow-up analyses

These were run after the first submission. Each produced numbers that are in the current paper,
so each is listed with the same detail. Outputs are under
`results/tables/review_response/`.

| Analysis | Script | Null / seed | Output directory |
|---|---|---|---|
| **Topology repair audit.** Which vertices were relabelled purely to satisfy the topology rules, attributed to the step that changed them, with a leave-out robustness check | `code/pipeline/rr1_topology_audit.py` | leave-out margin 2 mesh hops; no resampling null | `rr1_topology/` |
| **Outcome table.** Assembly of every test in the study into one table with family, null, status, raw p and within-family q | `code/pipeline/rr2_outcome_table.py` | none; reads existing outputs, computes nothing | `rr2_table/` |
| **Headline panel source values.** The measures plotted in the structural-model figure, read from the outcome table so the figure cannot disagree with it | `code/figures/rr3_fig6_bands.py` | none | `rr3_fig6/` |
| **Benchmark holdout and areal interior.** Fitting per-type medians on one hemisphere and scoring the other; and the win fraction restricted to vertices more than five mesh hops inside a von Economo area, with the 1 to 10 hop profile | `code/pipeline/rr4_benchmark_holdout.py` | spin null, N = 1000, seed 0 | `rr4_benchmark/` |
| **MEG controls.** Whether the type-timescale relation survives control for the aperiodic slope and for band power, with collinearity and degeneracy diagnostics | `code/pipeline/rr5_meg_controls.py` | partial Spearman by rank residualisation, per subject then group mean | `rr5_meg_controls/` |
| **Belt-excluded sensitivity.** Every primary measure re-run with the Glasser-adjudicated limbic belt removed, to test whether results depend on the least certain tissue | `code/pipeline/rr6_belt_excluded.py` | spin null, N = 1000, seed 0 | `rr6_belt/` |
| **Tractography controls.** A topology-preserving null on the raw-count falloff, and a bundle-level model with the endpoint-pair offset | `code/pipeline/rr7_tracto_controls.py` | negative binomial, log link, log offset; tract random intercept | `rr7_tracto/` |
| **Tractography proximity control.** The falloff re-tested against a spatial-opportunity offset counting only vertex pairs within 80 mm, which is the control the published offset did not provide | `code/pipeline/rr9_proximity_control.py` | as above, with the corrected offset | `rr9_proximity/` |
| **Crossed parcellation.** Desikan and von Economo parcels subdivided by cyto7 type, with a 200-vertex minimum node size applied bilaterally and residue absorbed into the largest neighbouring node in the same parcel | `code/pipeline/rr10_crossed_parcellation.py` | none; deterministic | `rr10_crossed/`, products in `atlas/fsaverage/crossed/` |
| **Resolution cascade.** What is lost when the vertex-level map is reduced to parcel-level and to a majority-type lookup | `code/pipeline/rr11_resolution_cascade.py` | none; misassignment fractions are descriptive | `rr11_cascade/` |
| **p-value convention audit.** A sweep of every resampling p-value in the repository against the convention section 2.6 states, distinguishing the vertex-level plus-one two-tailed convention from the parcel-level one-tailed function used for the layer-marker and disease families | `code/pipeline/rr13_pvalue_audit.py` | audits existing nulls; adds none | `rr13_pvalues/` |
| **Disease p-values recomputed.** All eight ENIGMA disorders recomputed under the section 2.6 convention on the same 1000 parcel rotations | `code/pipeline/rr13b_disease_convention.py` | parcel rotations, N = 1000, seed 0, two-tailed with plus-one | `rr13b_disease/` |
| **Comparator and granularity-matched null.** cyto7 against a myelin septile, a gradient septile, a BigBrain septile, and against size-matched contiguous random partitions, on the same disagreement vertices | `code/pipeline/rr14_comparator.py` | 1000 matched partitions grown per hemisphere from farthest-point seeds | `rr14_comparator/` |
| **Atlas-concordance harmonisation.** Whether the two atlas-concordance source sets can be harmonised without changing the released score | `code/pipeline/rr15_concordance.py` | none; a reconstruction check against the released map | `rr15_concordance/` |
| **Support-map promotion.** Verification that the released support products reproduce the published per-type medians exactly, for all seven types in both hemispheres, and that the atlas-free variant rebuilds identically | recorded outputs only; see below | none; exact reproduction check | `rr17_confidence_promotion/` |

The support-map promotion entry has no script in this repository because it is a release step
rather than an analysis: it copied the current build into the published location and verified
it. `rr17_confidence_promotion/acceptance_test.csv` records the per-type medians reproduced
from the shipped files, and `sha_before_after.csv` records the checksums.

The figure-text corrections recorded in `rr18_fig5/` and `rr19_fig5/` are documentation of two
stale strings rendered into the structural-model figure, with the pixel-level evidence that
nothing else in the image changed. They are included because they explain why that figure's
checksum changed after the outcome table was regenerated.

---

# Inputs not in this repository

These are third-party datasets under their own terms. None is redistributed here. Every script
that needs one reads it from `CYTO7_DATA_DIR`; see `docs/REPRODUCING.md` and `code/.env.example`.

| Dataset | Needed by | How to obtain |
|---|---|---|
| HCP S1200, structural and MEG | myelin and thickness maps, all MEG analyses | ConnectomeDB, with its data-use terms. The subjects used are listed in `results/subjects_used.txt` |
| AHBA microarray | gene expression PC1, layer marker genes | `abagen`, which downloads it |
| ENIGMA effect-size maps | disease vulnerability | the ENIGMA toolbox |
| PET receptor collection | receptor analyses | `neuromaps` |
| BigBrain profiles | histological comparison | BigBrainWarp |
| Population-averaged tractogram | tractography | as cited in the paper |
| FreeSurfer, Connectome Workbench, FieldTrip, MATLAB | resampling and MEG source reconstruction | their own distributions |

Only aggregate, group-level values are published here. No per-subject value is included, and
`results/subjects_used.txt` contains identifiers only.

# Checksums

`dist/SHA256SUMS.txt` lists a SHA-256 for every file in this repository. Verify with
`sha256sum -c dist/SHA256SUMS.txt` from the repository root.
