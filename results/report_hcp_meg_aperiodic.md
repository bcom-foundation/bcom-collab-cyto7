# Report — HCP-MEG source aperiodic exponent / peak frequency vs cyto7 v9 (Option A)

Implements `docs/SPEC_hcp_meg_aperiodic.md`. Env `cyto7` (+mne/specparam). Map **cyto7 v9**. Subjects **n=89**. Spin: released Alexander-Bloch fsLR 32k, n_perm=1000, seed=0.

## Method
- **Source spectra:** broadband `rmegpreproc` (241 4D/BTi magnetometers, ~508.6 Hz, 2 s clean epochs, 3 runs) -> single-sphere (Sarvas) MEG leadfield (field at coils combined via the grad `tra`, applying the 4D PCA reference compensation exactly; sphere fit to the supplied brain surface) -> LCMV scalar beamformer (max-power orientation, 5% covariance loading) -> per-source Welch PSD (2 s Hann per epoch, averaged over epochs and runs), on the 8004-source fs_LR 4k sheet.
- **specparam** (v2) per source: 2-40 Hz, aperiodic_mode='fixed' (pre-registered primary). Exponent (primary), offset, and the largest 2-40 Hz peak (peak/dominant frequency).
- **QC:** per-source fit R²; sources with R²<0.9 excluded; retained fraction = 0.862. Vertices kept if ≥60% of subjects have a good fit (7234 of 8004).
- **Forward-model caveat:** single-sphere approximates the HCP FieldTrip *singleshell* (Nolte); forward error smears power spatially and biases the type gradient toward the NULL, not toward a false positive. The exponent is a log-log slope (scale-invariant), so leadfield calibration is irrelevant. Validated: source alpha power tracks the neuromaps `megalpha` map (see run log).
- **Alignment/inference:** native 4k, cyto7 v9 aggregated to 4k (nearest sphere vertex) with the released 32k rotations downsampled identically; per-subject Spearman ρ(metric, type) over 4k cortical vertices; group statistic = mean/median of per-subject ρ; spin-p under the rotated type map. NEW 'dynamics_exponent' BH-FDR family; released tables untouched.

## Results (pre-registered primary: aperiodic exponent, ρ<0 predicted)
- **Aperiodic exponent (fixed, 2-40 Hz):** ρ_grp=-0.013 (med -0.010), spin p=0.854, q=0.938, allo-excl ρ=-0.003 (p=0.961), frac predicted dir=0.58
- **Peak/dominant frequency (ρ>0 predicted):** ρ_grp=+0.004 (med +0.008), spin p=0.938, q=0.938, allo-excl ρ=-0.000 (p=0.996), frac predicted dir=0.55
- **Offset:** ρ_grp=-0.107 (med -0.108), spin p=0.381, q=0.938, allo-excl ρ=-0.092 (p=0.452), frac predicted dir=nan

Full numbers + per-subject ρ: `figures/v9/structure_function/meg_aperiodic_summary.csv`, `meg_aperiodic_persubject_rho.csv`.

## Interpretation
The exponent group statistic is the mean over subjects of the per-source Spearman ρ vs ordinal cyto7 type; significance is against the released spin. This is the correctly-specified, properly-powered dynamics test (per-subject source spectra, not band-limited summaries).

**Verdict: a clean, robust NULL.** The MEG source aperiodic 1/f exponent does **not**
track the cyto7 cortical-type axis (ρ_grp = −0.01, spin p = 0.85, FDR q = 0.94), and the
null is unchanged with allocortex excluded (ρ = −0.00) and holds for the secondaries —
peak frequency (ρ = +0.00, p = 0.94) and offset (ρ = −0.11, p = 0.38). Per-subject signs
scatter around chance (58% in the predicted direction for the exponent), so this is not
an outlier-driven wash-out but a genuine absence of a group gradient.

**Why the null is credible (not merely a pipeline artefact):** (i) fits are good —
86% of source spectra pass R² ≥ 0.9, median exponent ≈ 0.5 (typical for resting MEG
2–40 Hz); (ii) the forward + beamformer are **validated** — group-mean source alpha
power tracks the neuromaps `megalpha` map at Spearman ρ ≈ +0.5 (relative power), i.e. the
localisation genuinely works. **Caveat:** the single-sphere forward approximates the HCP
FieldTrip *singleshell* and any residual forward error smears power spatially, which
biases a true gradient **toward** the null — so the honest reading is "no exponent
gradient detectable with this validated-but-approximate source pipeline," not "proven
absence." A confirmatory re-run with the exact FieldTrip singleshell leadfields (or an
independent MEG cohort) would harden it; the pre-registration and machinery are ready.

This is a pre-registered, properly-powered outcome and, per the task, an acceptable and
publishable result — it says the cytoarchitectural axis is captured by structural/
myelin/gradient maps (which do track it) but **not** by the resting-state aperiodic
exponent. It does not become a new Fig. 5 bar. Complementary band-limited result:
`report_meg_subjectlevel.md` (Option B).

## Provenance
- megconnectome v3.0 data; mne + specparam (v2); Sarvas single-sphere forward; seed=0; n_perm=1000. Script `scripts/meg_aperiodic.py`. Per-subject processing on X: scratch (one preproc archive at a time, rmegpreproc read from the zip in memory), summaries only returned to the repo.