# Report — per-subject T1w/T2w vs cyto7, and a microstructure-informed (T1w/T2w) inter-subject variability layer

Implements `docs/SPEC_individual_myelin_variability.md`. Env: `cyto7`. Map: **cyto7 v9** (canonical). Mesh: fs_LR 32k. Runtime 2.9 min. **Proof-of-concept — not wired into the manuscript, and nothing released was modified.**

## Anti-circularity statement

T1w/T2w is a modality cyto7 is *validated against* (§3.5) and the released per-vertex support map is deliberately **anatomy-only** (Annex C: atlas + topology + geometry + prior; the T1w/T2w overlay is computed but excluded from the score). The layer produced here is therefore a **separate, microstructure-informed (T1w/T2w) inter-subject variability** product. It is **not** folded into the anatomy-only support, and it is **not** used to re-validate cyto7 — the §3.5 T1w/T2w validation stays on the anatomy-only map. Step 4 below only *compares* the two maps, to ask whether they carry the same information. Any future manuscript use must keep this labelling.

## Data and provenance

- **Source:** `s3://hcp-openaccess/HCP_1200/<subj>/MNINonLinear/fsaverage_LR32k/<subj>.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii` — the individual bias-corrected T1w/T2w map, MSMAll-aligned, cortex-only (59,412 grayordinates). Nothing else was downloaded. Credentials came from the AWS `[default]` profile and appear nowhere in this repository or in any output.
- **N = 200 subjects.** IDs in `subjects_used.txt`. 3 candidate subject(s) were stop-and-logged as missing/unreadable and skipped (126931, 129432, 131621); see `subjects_missing.txt`.
- **Subject selection:** the 210 *RelatedValidation* IDs are **not obtainable** programmatically — the group membership list is not in the S1200 group-average package, not in the local Glasser RVVG package, and not derivable from the S3 listing (it lives in the restricted ConnectomeDB release). Per the SPEC fallback, the first 200 S1200 subjects (ascending subject ID) that have the myelin map were used. This is an unselected S1200 sample, so it is **not** the exact set behind the group average; that is a deliberate, logged deviation and `--subject-list` accepts the real list if the author supplies it later.
- **Group-average reference:** `Q1-Q6_RelatedValidation210.MyelinMap_BC_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii` (Q1-Q6_RelatedValidation210, the dataset the cyto7 labels were resampled against).
- **Valid vertices:** 58,731 of 64,984 — labelled cyto7 cortex (`label > 0`) that is also a CIFTI grayordinate in the group map *and in every subject*, so all subjects contribute at every vertex. Allocortex-excluded sensitivity uses 57,952 vertices (types 2–7). Validity is taken from the CIFTI vertex index, not `value != 0`, because bias-corrected individual T1w/T2w legitimately contains values ≤ 0.
- **Downloads cached outside the repository** at `$CYTO7_SCRATCH_DIR (~118 MB); re-runs are resumable.

## Step 2 — per-subject alignment with cyto7 type

| | median | IQR | min | max | group-average map |
|---|---|---|---|---|---|
| Spearman ρ (all 7 types) | **+0.527** | +0.516 – +0.538 | +0.474 | +0.565 | +0.589 |
| ρ, allocortex-excluded (2–7) | **+0.539** | +0.527 – +0.550 | +0.485 | +0.573 | +0.604 |

Every one of the 200 subjects has ρ > 0 (200/200); the individual ρ is systematically **lower** than the group-average value, which is expected — averaging 210 brains removes individual noise that the ordinal type map cannot predict, so the group ρ is an upper bound, not a typical individual.
Type-progression stability: Kendall τ between each subject's seven per-type median myelin values and the type rank is median **+0.714** (allocortex-excluded +1.000); **200/200** subjects show a strictly monotonic agranular→koniocortex increase in median myelin. The all-7 τ is lower than the allocortex-excluded τ in essentially every subject for the same reason it is in the released group table: allocortex sits at rank 1 but its T1w/T2w is *elevated* relative to agranular cortex, so it breaks the ramp — which is exactly why the paper excludes allocortex from the trend fits. The individualisation result reproduces that group-level quirk subject by subject. Per-subject numbers: `persubject_rho.csv`, `persubject_type_medians.csv`.

## Step 3 — between-subject variability map

Per subject: z-score T1w/T2w across valid cortical vertices; subtract that subject's mean z within each cyto7 type (the level the group label predicts *in that subject*); the residual `r_s(v)` is ~0 where the group label fits the individual well. Across subjects, per vertex:
- **`between_subject_variability`** = SD over subjects of `r_s(v)` — the headline layer. Median 0.364, IQR 0.321–0.442, 99.9th pct 5.079, range 0.204–14.610 z units. The long upper tail is a small set of artefact vertices in individual T1w/T2w maps, which is why an IQR-based twin (`between_subject_variability_robustIQR.npy`) is carried alongside and used as the step-4 sensitivity check.
- **`systematic_misfit`** = mean over subjects of `r_s(v)` (signed; range -1.842…+5.348) and mean `|r_s(v)|` (median 0.485), written as two maps in the same file family.

Median variability by cyto7 type (z units):

| type | median SD |
|---|---|
| 1 Allocortex | 0.667 |
| 2 agranular | 0.385 |
| 3 dysgranular | 0.340 |
| 4 eulaminate I | 0.330 |
| 5 eulaminate II | 0.345 |
| 6 eulaminate III | 0.397 |
| 7 koniocortex | 0.562 |

Outputs: `between_subject_variability.{npy,dscalar.nii}`, `systematic_misfit.{npy,dscalar.nii}` (signed mean; the absolute mean is `absolute_misfit.{npy,dscalar.nii}`). All on fs_LR 32k, NaN outside valid cortex.

## Step 4 — cross-check against the released anatomy-only support (the scientific payoff)

Over 58,731 valid vertices, between-subject variability vs `1 − anatomy-only support` (cyto7 v9):
- Spearman ρ = **+0.045**, spin p = 0.663 (Alexander-Bloch on the variability map, n_perm=1000, seed=0; null SD 0.099).
- Pearson r = +0.216 (R² = 0.047). Pearson runs well above Spearman here because the SD layer has a long artefact tail that a rank correlation ignores; **Spearman is the number to quote**.
- **Outlier robustness:** repeating this with the IQR-based variability twin (normalised inter-quartile range of `r_s(v)` over subjects, immune to the handful of artefact vertices whose SD is an order of magnitude above the bulk) gives ρ = **+0.019**; the two variability estimates agree with each other at ρ = +0.912. The conclusion does not rest on the SD's tail sensitivity.
- Within-type ρ, the allocortex-excluded row, and the |misfit| / signed-misfit variants: `variability_vs_support.csv`.
- Border control: geodesic distance to the nearest cyto7 type boundary vs variability, ρ = -0.043 — variability is not simply concentrated near type borders, so it is not a boundary-placement effect.

**The pooled ρ is a cancellation, not an absence of relationship.** Within cyto7 types the correlation is substantial and *changes sign*: it is positive in the less-differentiated types (Allocortex +0.62, agranular +0.32) — there, low anatomy-only support does mark where individuals' microstructure disagrees with the group label — and negative in the most-differentiated types (eulaminate III -0.29, koniocortex -0.32), where low-support vertices are actually the *more* consistent ones across individuals. Pooled over cortex these opposite trends cancel to ρ = +0.045. The allocortex-excluded pooled value (+0.015) confirms the pooled number is not carried by allocortex alone. Any use of this layer should therefore be *within-type*, not global.

**Interpretation.** Globally, the geometric/topological support and the microstructure-informed variability are **independent** (ρ = +0.045, spin p = 0.663): a whole-cortex map of the anatomy-only support tells you essentially nothing about where individuals' microstructure disagrees with the group label, and the within-type breakdown above shows why — the two are coupled inside types, but in opposite directions at the two ends of the differentiation gradient, so the pooled signal cancels. That is the substantive result for §4.10: a between-subject variability layer would be a genuinely **additional** axis of uncertainty, not a re-expression of the support map already released — but it should be read per cyto7 type rather than as a single cortex-wide overlay.

(All 58,731 valid vertices carry a finite anatomy-only support value after the 164k→32k resample.)

## Files

| file | contents |
|---|---|
| `subjects_used.txt` | the N subject IDs actually used |
| `persubject_rho.csv` | per subject: ρ, ρ allocortex-excluded, Kendall τ of the type progression, monotonicity flag |
| `persubject_type_medians.csv` | per subject: median T1w/T2w per cyto7 type |
| `persubject_rho_summary.txt` | the one-line summary (median, IQR, group ρ) |
| `between_subject_variability.{npy,dscalar.nii}` | headline layer: SD over subjects of the residual |
| `between_subject_variability_robustIQR.npy` | outlier-resistant twin (normalised IQR over subjects) |
| `systematic_misfit.{npy,dscalar.nii}` | signed mean residual over subjects |
| `absolute_misfit.{npy,dscalar.nii}` | mean absolute residual over subjects |
| `variability_vs_support.csv` | step-4 correlations, overall and per type |
| `figure.png` | (a) per-subject ρ histogram, (b) variability on the inflated 32k surface with cyto7 borders, (c) support vs variability |

## Caveats

- The sample is the first 200 S1200 subjects with a myelin map, not the RelatedValidation210 set behind the group average (see Provenance).
- `MyelinMap_BC_MSMAll` inherits HCP's bias-field correction and MSMAll areal alignment; MSMAll uses myelin as one of its alignment features, so part of the *reduction* in between-subject variability is registration-induced. The layer is therefore a conservative (lower-bound) estimate of true anatomical variability.
- Residuals are relative to a *within-subject* z-score, so the layer measures the spatial pattern of misfit, not absolute T1w/T2w differences (global intensity differences between subjects are removed by construction).
- The variability map is a group-level summary over N subjects; it is not an individualised parcellation and does not license per-subject relabelling.
