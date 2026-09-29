# Task — BigBrain histological validation done right (intensity profiles, not the G1 gradient)

**Origin:** the written analysis plan for the BigBrain profile validation, fixing the differentiation
index and the arbiter-validity gate before the comparison was made. Reproduced from the working
repository; the local path in its original header has been removed.
**Why:** the earlier BigBrain test used the Hist-G1 gradient as arbiter, which itself barely tracks the
type axis (ρ ≈ 0.21, spin p 0.37) — an underpowered referee, so its added-value null (49.9%) is
uninformative, not evidence against the refinement. Hist-G1 is a *derived* covariance-embedding that
dissociates from the type hierarchy by design (Paquola 2019). The correct histological signal is the
**raw intracortical staining-intensity depth profile**, whose *shape* directly encodes laminar
differentiation (layer-IV granularity) — i.e. exactly what cortical type indexes.

**Key property of this design:** it is informative *either way*. Either the refinement confirms against
real histology (the strongest answer to the "expert-scheme + MRI-only" objection), or Step 1 shows the
single-specimen profiles can't resolve the 7-type scale — a clean, honest limitation rather than an
ambiguous null.

## Data
BigBrain **intracortical intensity profiles**: intensity sampled at ~50 equivolumetric depths
(pial→white) per vertex, from **BigBrainWarp** (same source as Hist-G1; `github.com/caseypaquola/BigBrainWarp`
/ the BigBrain project servers). Prefer a version already on fs_LR 32k; otherwise fetch native BigBrain
surfaces and resample to 32k fs_LR (`neuromaps.transforms`, nearest/linear). Cache as a
(depths × 32492) array per hemi under `resources/neuromaps_cache/bigbrain_profiles_fsLR32k_hemi-{L,R}.npy`.
If the profiles can't be reached, **report and stop — do not substitute** (guardrail).

## Step 1 — build a differentiation index and gate its validity (do this first)
From each vertex profile `p(d)` compute the standard microstructural-profile features:
- **central moments**: mean intensity, SD, skewness, kurtosis (Paquola-style);
- **layer-IV peak prominence**: prominence of a local maximum in the mid-cortical band (~30–60% depth).

**Pre-register the index** to avoid selection bias: report *every* feature's per-type trend, and designate
a single a-priori differentiation index **before** Step 2 (recommended: profile **skewness** and/or **mean
intensity**, the conventional microstructure summaries) — do **not** pick the feature that maximises the
cyto7 correlation.

Then run the gradient-by-type test (same spin+FDR machinery as the structure–function panel): Spearman ρ of
each feature vs ordinal cyto7 type, spin p (N=1000), FDR q. Add a BigBrain-profile row to the supplementary
gallery / Table S2.

**Gate:** the arbiter is only valid if the chosen index tracks type significantly (target ρ ≳ 0.4, spin-sig).
- If **yes** → proceed to Step 2 (the test is now powered).
- If **no** → stop and report as a limitation: single-specimen BigBrain profiles at this projection do not
  resolve the 7-type scale; do not run/interpret Step 2 as evidence either way.

## Step 2 — histology added-value test (only if the gate passes)
Reuse `reviewer_response.analysis1` **verbatim**, swapping the MRI multimodal profile for the pre-registered
histology index as the sole arbiter. On the ~27,298 isocortex disagreement vertices, compute the fraction
where the **cyto7** ordinal type predicts the local histology index better than the **von-Economo** type,
with the spin null. Write to a dedicated subdir (do **not** clobber the canonical v9 added-value files).
This is not circular: the arbiter is histology-derived, independent of both label maps (that's why the index
must be pre-registered, not label-selected).

## Step 3 (optional) — cross-validated profile matching
Per-type mean intensity profile ("histological fingerprint"). To avoid circularity, build templates with
**spatial hold-out** (Alexander-Bloch spin blocks / geodesic folds) so no vertex informs its own template.
At each disagreement vertex, whichever type-fingerprint the observed profile is closer to (correlation /
Euclidean) "wins"; report cyto7 vs von-Economo win-fraction with the spin null. Most direct "does the
histology look like the assigned type" test.

## Deliverables
- `resources/neuromaps_cache/bigbrain_profiles_fsLR32k_hemi-{L,R}.npy` + a small `bigbrain_profile_features` cache.
- `figures/v9/structure_function/bigbrain_profiles/REPORT_bigbrain_profiles.md`: per-feature ρ/p/q (Step 1),
  the gate decision, and (if passed) the Step 2 (and Step 3) win-fractions with spin p.
- BigBrain-profile row added to the supplementary gallery + Table S2 (replacing/complementing the Hist-G1 row).
- `scripts/bigbrain_profile_validation.py`.

## Acceptance
- [ ] Profiles fetched + cached (depths × 32492/hemi); features computed; index pre-registered and named.
- [ ] Step 1 gradient-by-type ρ/p/q reported for all features + the gate decision stated explicitly.
- [ ] If gate passes: Step 2 win-fraction + spin p (dedicated subdir; canonical v9 files untouched); optional Step 3.
- [ ] If gate fails: reported as a limitation, Step 2 not over-interpreted.
- [ ] Honest verdict in the report; no map substitution; manuscript files untouched.

## Guardrails
- Single specimen (n=1) — state as a limitation throughout; BigBrain→fs_LR warp error noted.
- Pre-register the differentiation index (no label-driven feature selection).
- Reuse the existing spin+FDR + added-value machinery; 32k fs_LR; magma surfaces; report, don't substitute.
