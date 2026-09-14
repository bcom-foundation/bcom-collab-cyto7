# Which file backs which Annex G number

This directory holds several similar-looking tables, and nothing previously said which
one supports which published figure. This file does.

**Every number in Annex G reproduces from files in this release.** Each row below names
the file it comes from and the script that computes it. Values are as published in the
supplement's Annex G; the "in the release" column is what the named file actually
contains.

Map version throughout: **v9**, the released map. Short/long split at **80 mm** on
`Mean_Length_mm`. Seed 0, N = 1000 for every null.

## The headline fit

| Published | In the release | File | Script |
|---|---|---|---|
| short-range slope α ≈ −1.23 | **−1.2306** | `connectivity_short.csv` | `code/figures/plot_tractography_analysis.py` |
| observed Pearson r = −0.765 | **−0.7649** | `connectivity_short.csv` | same |
| label-permutation p = 0.003 | **0.003** | `tractography_null.csv` | `code/pipeline/reviewer_response.py` |

### How the slope is fitted, exactly

This is the part that is easy to get wrong, and getting it wrong gives a materially
different number. The fit is in `code/figures/plot_tractography_analysis.py`, function
`plot_matrix` (the log-linear fit at lines 311-318):

1. Build the symmetric 7 × 7 type-pair matrix from `cyto7.v9_tract_geometry_per_bundle.csv`
   with `stratified_matrices(geometry, 80.0)`. Each record is assigned to short or long by
   its `Mean_Length_mm`, and its `Streamline_Count` is added to **both** `[i, j]` and
   `[j, i]`. That is `connectivity_short.csv`.
2. Take the **21 off-diagonal upper-triangle cells only**. The 7 diagonal cells are not
   in the fit.
3. x = |type-distance| = `|index(i) − index(j)|` over
   `Allocortex, Agranular, Dysgranular, Eulaminate-I, Eulaminate-II, Eulaminate-III, Koniocortex`.
4. Fit **`log1p(y)`** against x with `np.polyfit(x, log1p(y), 1)`, **unweighted**, and
   **keep the zero cells** (6 of the 21 are zero; `log1p(0) = 0`).

Three details each change the answer on their own:

- using `log` instead of `log1p` and dropping the zero cells gives **−0.688**
- additionally including the 7 diagonal cells gives **−0.405**
- fitting per-bundle records rather than the 7 × 7 matrix answers a different question
  entirely; that is the bundle-level model below, whose coefficient is **+0.01**

## Nulls, controls and the opportunity correction

| Published | In the release | File | Script |
|---|---|---|---|
| rotation null p = 0.040, null mean r = −0.334 | **0.040**, **−0.3337** | `../review_response/rr9_proximity/published_statistic_with_offset.csv`, row `raw_count_published` | `code/pipeline/rr9_proximity_control.py` |
| opportunity-corrected slope −0.80, r = −0.597, p = 0.070 | **−0.7999**, **−0.5972**, **0.0699** | same file, row `rate_per_opportunity_80mm` | same |
| contact-area partial r = −0.64, p = 0.002 | **−0.6358**, **0.0019** | `tractography_null.csv` | `code/pipeline/reviewer_response.py` |
| connectivity vs contact-area r = +0.575 | **0.5754** | `tractography_null.csv` | same |

## Aggregate and bundle-level models

Negative-binomial, log link, `Tract_Name` random intercept at bundle level; offsets as
named. `p_spin` is the topology-preserving rotation null.

| Published | In the release | File, row | Script |
|---|---|---|---|
| aggregate coefficient −1.15, p = 1.7e−6, spin p = 0.067 | **−1.1526**, **1.737e−06**, **0.0669** | `../review_response/rr9_proximity/model_coefficients_rr9.csv`, `level=aggregate, model=R1` (offset `log(pairs ≤ 80 mm)`) | `code/pipeline/rr9_proximity_control.py` |
| against −1.36, spin p = 0.060 | **−1.3631**, **0.0599** | same file, `level=aggregate, model=R0` (RR7's `log(n_i n_j)` offset) | same |
| contact-area specification −1.47, spin p = 0.021 | **−1.4666**, **0.0210** | same file, `level=aggregate, model=R2` (+ contact area) | same |
| contact-area specification −1.68, spin p = 0.012 | **−1.6794**, **0.0120** | same file, `level=aggregate, model=R3` (+ contact area + NN median Euclidean) | same |
| bundle-level +0.13, p = 0.43 | **0.1263**, **0.4330** | same file, `level=bundle, model=R1` | same |
| bundle-level +0.01, p = 0.96 | **0.0089**, **0.9571** | `../review_response/rr7_tracto/model_coefficients.csv`, `design=bundle x type-pair, zero-completed (primary)`, `spec=S1  + endpoint-pair offset` | `code/pipeline/rr7_tracto_controls.py` |

`../review_response/rr7_tracto/null_comparison.csv` carries the same spin and
label-permutation nulls in one place for the specifications RR7 tested.

## What the other files in this directory are

| File | What it is |
|---|---|
| `connectivity_short.csv`, `connectivity_long.csv` | the 7 × 7 type-pair matrices, short and long range. **The published fit is on the short one.** |
| `connectivity_all_recomputed.csv` | the same matrix without the length split |
| `cyto7.v9_connectivity_per_bundle.csv` | one row per (bundle, source type, target type), 223 records |
| `cyto7.v9_tract_geometry_per_bundle.csv` | the same records plus mean length, tortuosity and direction, 174 records. **The matrices above are built from this.** |
| `cyto7.v9_connectivity_per_bundle_aggregate.csv` | the 7 × 7 matrix over **all** bundles, not the short subset. Fitting this gives −0.435, not the published value. |
| `cyto7.{v3,v3_clean,v6,v7,v8}_*` | the same three products for superseded map versions, kept for provenance. Not the published run. |
| `boundary_surface_mm2.csv` | pairwise contact area between types, from the label volume; the covariate in the partial and contact-area models |
| `tractography_null.csv` | the headline statistics with their permutation and partial-correlation results |
| `orientation_tortuosity_summary.csv`, `per_type_connectivity*.csv` | the orientation, tortuosity and per-type descriptives |
| `summary.txt` | the run summary. **Its streamline sums are matrix totals**, so off-diagonal pairs are counted twice: 8,749 short is 1,309 on the diagonal plus 7,440 off it, and halving the off-diagonal recovers the 5,029 raw per-record sum. |
| `REPORT_tractography_v9.md` | the v9 run report, and the v8-to-v9 comparison |

## What this analysis does and does not support

Annex G reports this as exploratory and draws no inferential claim from it. The
falloff does not survive the opportunity-corrected rotation null (p = 0.070), is
absent at bundle level (+0.01, p = 0.96; +0.13, p = 0.43), and cannot separate
cytoarchitectural from physical distance. **No conclusion in the paper rests on
−1.23.** This file exists so the number can be traced, not because anything depends
on it.

Added by RR28, 2026-09-14.
