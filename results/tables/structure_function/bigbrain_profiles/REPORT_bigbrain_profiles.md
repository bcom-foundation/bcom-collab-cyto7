# BigBrain intensity-profile histological validation (SPEC_bigbrain_profile_validation)

Arbiter = raw 50-depth intracortical staining-intensity profiles (BigBrainWarp, fs_LR 32k; Paquola 2019, **single specimen n=1**), whose *shape* encodes laminar differentiation — unlike the derived Hist-G1 gradient used before (ρ≈0.21, underpowered). cyto7 **v9**, spin N=1000, seed 0; FDR across the profile-feature panel.

**Pre-registered differentiation index: profile skewness** (conventional microstructure summary; declared before Step 2, not the max-ρ feature).

## Step 1 — profile features by cyto7 type (all reported; no selection)

| feature | ρ vs type | spin p | FDR q |
| --- | --- | --- | --- |
| Profile mean intensity | -0.283 | 0.209 | 0.348 |
| Profile SD | +0.436 | 0.003 | 0.015* |
| Profile skewness (pre-registered) **(pre-reg)** | -0.209 | 0.321 | 0.401 |
| Profile kurtosis | -0.242 | 0.133 | 0.332 |
| Layer-IV peak prominence | +0.085 | 0.674 | 0.674 |

**GATE:** pre-registered index 'skewness' |ρ| = **0.209** (target ≥ 0.4), spin p = 0.321 → **FAIL**.

**Verdict (gate FAILED):** the single-specimen BigBrain intensity profiles at this fs_LR projection do **not** resolve the 7-type differentiation scale (pre-registered index below the power threshold). Per the pre-registered design, Step 2 is **not run/interpreted** as evidence either way — this is a clean, honest limitation (n=1 histology + warp error), not an ambiguous null. The BigBrain leg is reported as inconclusive.

**Transparency note.** One feature *does* survive the spin+FDR test — **profile SD** (ρ +0.436, q 0.015) — but SD is the overall staining-contrast amplitude, **not** the pre-registered laminar-differentiation index, and the shape features that specifically index granularity (skewness, kurtosis, layer-IV peak prominence) are all non-significant. Pre-registration was chosen precisely to prevent pivoting to the post-hoc max-ρ feature: we report SD honestly but do **not** use it as the arbiter or claim it as histological validation of the 7-type scale. The consistent picture across skewness/kurtosis/layer-IV prominence is that the n=1 profiles carry a weak overall contrast trend but do not resolve the differentiation *hierarchy*.

**Nuance (see `bigbrain_profiles_gallery.png`).** The *mean* intracortical profiles do differ in
**shape** by type — koniocortex shows the characteristic deep mid-cortical (~60% depth) intensity
minimum expected of a dense granular layer, while allocortex/agranular are flatter with a mid-depth
bump. But no single *scalar* summary of that shape (skewness/kurtosis/layer-IV prominence) resolves
the ordinal 7-type scale under the stringent spatial-autocorrelation spin null. A multivariate
profile-shape test (Step 3) is **not** run here: with the pre-registered scalar gate failed, running
it now would be a post-hoc attempt to rescue the null and violate the pre-registration. It is flagged
as a legitimate future analysis (with its own cross-validation), not evidence for or against the map today.

**Comparison to the earlier Hist-G1 test:** the raw profiles are the correct, more powerful arbiter in principle, yet the pre-registered differentiation index (skewness ρ 0.21) tracks type no better than the derived Hist-G1 gradient (ρ 0.21) — reinforcing that the limitation is the **single specimen + warp**, not the choice of derived vs raw signal. Both BigBrain legs are therefore reported as inconclusive; the histological validation remains open pending a multi-specimen / higher-fidelity histology reference.

_Manuscript untouched; no map substitution; canonical v9 added-value files not overwritten._