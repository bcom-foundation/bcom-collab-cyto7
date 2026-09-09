# Report — HCP-MEG dynamics v2 (FieldTrip singleshell forward + true INT)

Implements `docs/SPEC_hcp_meg_dynamics_v2.md`. Route **1a: FieldTrip 20200607 singleshell LCMV** (MATLAB R2018a). cyto7 **v9**; released spin fsLR 32k, n_perm=1000, seed=0. Subjects **n=89**.

## Forward validation (better forward should not degrade this)
- source rel-alpha vs neuromaps `megalpha`: **rho=+0.978** (v1 single-sphere ~+0.51 -> singleshell improves localization).
- true INT vs neuromaps `megtimescale` (expect POSITIVE; envelope proxy failed at -0.84): 1e=+0.463, area=+0.852, tau=+0.651.

## Fit QC + reliability
- retained fraction (fixed-fit R2>=0.9) = 0.999; valid 4k vertices 7234.
- across-run reliability (mean pairwise Spearman): exponent=0.872, INT-1e=0.863, INT-area=0.828, INT-tau=0.878. (A null is only informative for a reliable metric.)

## Results vs cyto7 v9 (per-subject rho; released spin; BH within family)
### Pre-registered primaries
- **Aperiodic exponent (fixed 2-40 Hz):** rho=-0.119 (med -0.129), spin p=0.499, q=0.582, allo-excl -0.101 (p=0.563), frac-pred=0.80, v1=-0.01256197401939
- **True intrinsic timescale (area def):** rho=-0.459 (med -0.471), spin p=0.004, q=0.028, allo-excl -0.451 (p=0.005), frac-pred=1.00, v1=nan
### Dynamics-core family
- exponent (knee): rho=+0.339 (med +0.358), spin p=0.040, q=0.080, allo-excl +0.330 (p=0.047), frac-pred=0.03, v1=nan
- exponent (broad 1-100): rho=+0.095 (med +0.097), spin p=0.469, q=0.469, allo-excl +0.089 (p=0.511), frac-pred=0.18, v1=nan
- peak frequency: rho=+0.028 (med +0.021), spin p=0.824, q=0.824, allo-excl +0.019 (p=0.878), frac-pred=0.54, v1=0.0043411442336908
- offset: rho=-0.162 (med -0.166), spin p=0.213, q=0.298, allo-excl -0.147 (p=0.248), frac-pred=nan, v1=-0.1073969660242624
- knee-timescale: rho=-0.343 (med -0.384), spin p=0.027, q=0.047, allo-excl -0.331 (p=0.034), frac-pred=0.94, v1=nan
- INT (lag-1/e): rho=-0.315 (med -0.325), spin p=0.025, q=0.047, allo-excl -0.298 (p=0.040), frac-pred=0.98, v1=nan
- INT (exp-tau): rho=-0.380 (med -0.377), spin p=0.013, q=0.045, allo-excl -0.361 (p=0.022), frac-pred=1.00, v1=nan
### Oscillatory-band family (aperiodic-corrected) + summaries
- osc delta: rho=-0.030 (med -0.011), spin p=0.284, q=0.397, allo-excl -0.029 (p=0.295), frac-pred=0.94, v1=nan
- osc theta: rho=-0.092 (med -0.105), spin p=0.493, q=0.575, allo-excl -0.098 (p=0.465), frac-pred=0.74, v1=nan
- osc alpha: rho=+0.260 (med +0.269), spin p=0.242, q=0.397, allo-excl +0.263 (p=0.240), frac-pred=nan, v1=nan
- osc beta: rho=+0.222 (med +0.228), spin p=0.012, q=0.028, allo-excl +0.215 (p=0.017), frac-pred=0.90, v1=nan
- osc gamma1: rho=-0.027 (med -0.019), spin p=0.591, q=0.591, allo-excl -0.030 (p=0.539), frac-pred=0.28, v1=nan
- SF ratio: rho=-0.356 (med -0.359), spin p=0.009, q=0.028, allo-excl -0.339 (p=0.011), frac-pred=0.98, v1=nan
- centroid: rho=+0.409 (med +0.423), spin p=0.004, q=0.028, allo-excl +0.392 (p=0.005), frac-pred=0.99, v1=nan

## Interpretation
Group statistic = mean over subjects of the per-source Spearman rho vs ordinal cyto7 type; released spin null; BH within each FDR family (dynamics_core, dynamics_robust, osc_band) — kept separate from the published families. The v1 column is the single-sphere value for the same metric; comparing them isolates the effect of the realistic forward. The INT here is the correctly-specified broadband-ACF timescale (validity-gated vs megtimescale), superseding the Option-B envelope proxy. A properly-powered null is an acceptable, publishable outcome and feeds the C2 E/I strand.

## Provenance
- FieldTrip 20200607 (R2018a) singleshell LCMV (unit-noise-gain, fixedori, 5% lambda); specparam v2; seed=0; n_perm=1000. Persisted PSD/INT scratch kept at `$CYTO7_DATA_DIR Young MEG data\_scratch\v2\persist` (path+size printed in the run log). Released tables untouched.