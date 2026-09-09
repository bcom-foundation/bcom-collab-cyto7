# Report — signal-controlled between-subject T1w/T2w variability (does koniocortex survive?)

Implements `docs/SPEC_individual_myelin_variability_signalcontrol.md`; follow-up to `report_individual_myelin_variability.md`. Env: `cyto7`. Map: **cyto7 v9**, fs_LR 32k, **N = 200** subjects reused from scratch (no S3 access, no new download). 58,731 valid vertices. Runtime 1.5 min. **Proof-of-concept; nothing released or previously produced was modified.**

## Verdict

> **Koniocortex does NOT survive the signal control.** Its elevated raw between-subject SD was largely heteroscedasticity: koniocortex has the highest group-mean T1w/T2w in the cortex (1.55 vs ~1.25 mid-cortex), and higher signal carries higher between-subject spread (ρ = +0.552 cortex-wide). Under the **only control that actually neutralises signal level** ((A2) binned mean removal; residual ρ = -0.016), the koniocortex offset falls from **+2.20** to **+0.53** standardised units — a 76% reduction — and is **no longer distinguishable from chance** (spin p = 0.109).

The two controls that *do* leave koniocortex elevated ((B) coefficient of variation, (C) per-type-SD residual) are exactly the ones that failed the Step-2 check (residual ρ = +0.243, +0.401), i.e. they still carry the confound they were meant to remove, so their support for the claim is not independent evidence.

**Recommendation for N10:** drop the koniocortex "crisp but variable" headline and reframe the individualisation item as *residual variability concentrates in the limbic belt* — that result is large, control-independent and, unlike koniocortex, strengthens rather than weakens when signal level is removed (see Step 3a).

**Read the Step-2 table before the Step-3 tables.** Only 1 of the 4 controls actually neutralised the confound ((A2) binned mean removal); (A) regress-out mean, (B) coefficient of variation, (C) per-type-SD residual did not, and their per-type numbers below are reported for completeness rather than as evidence.

## The confound being controlled

The raw between-subject SD map from the previous run correlates with the group-mean T1w/T2w at Spearman **ρ = +0.552** over 58,731 vertices. That is a mean–variance (heteroscedasticity) effect: vertices with more signal have more between-subject spread. It matters because the per-type ordering of group-mean T1w/T2w (koniocortex highest, allocortex second) is almost the same ordering as the raw SD, so the raw map cannot distinguish "individuals disagree here" from "the signal is large here".

## Step 1–2 — the controlled maps, and whether the control worked

| map | ρ vs group-mean T1w/T2w | Pearson vs mean | control effective? |
|---|---|---|---|
| raw SD | +0.552 | +0.316 | — |
| (A) regress-out mean | -0.224 | -0.000 | **no** — over-corrected (sign flipped) |
| (A2) binned mean removal | -0.016 | +0.131 | **yes** (|ρ| < 0.10) |
| (B) coefficient of variation | +0.243 | +0.185 | **no** — still tracks signal |
| (C) per-type-SD residual | +0.401 | +0.375 | **no** — still tracks signal |

**This table is the most important result in the report**, because it decides which of the other numbers mean anything. Only (A2) binned mean removal reached ρ ≈ 0.

- **(A), the SPEC's primary, over-corrects** (ρ = -0.224, now *negative*). It fits SD = -0.313 +0.487·mean +0.867·|mean − median| (model R² = 0.132), but the SD-vs-signal relation is not linear, so a straight line subtracts too much at the high-myelin end — precisely where koniocortex sits. That is why (A) alone would have pushed koniocortex to -0.77, an artefact of the correction rather than a result.
- **(A2) is the only clean control** (ρ = -0.016). Subtracting the median SD within each of 50 equal-count bins of the group mean removes *any* monotone dependence on signal level, linear or not. It was added as a supplement to (A); on this evidence it should be treated as the reference control.
- **(B) under-corrects** (ρ = +0.243): dividing by the mean is the right correction only if SD scales exactly proportionally to the mean, which it does not here.
- **(C) under-corrects most** (ρ = +0.401). Standardising by the subject's *within-type* SD removes between-type differences in scale but leaves the mean–variance relation that operates *within* each type, which is the larger part of the confound. (C) also has a second problem for this question — see the caveats: by normalising each type to unit within-type spread it partly defines away the between-type comparison it is being used to make.

## Step 3a — per-type medians, raw vs controlled

Standardised per-type offsets, `(median_type − median_cortex) / robust SD_cortex`, so maps in different units are comparable. Positive = more variable than cortex typically is.

| type | raw SD | (A) regress-out mean | (A2) binned mean removal | (B) coefficient of variation | (C) per-type-SD residual |
|---|---|---|---|---|---|
| 1 Allocortex | +3.38 | +3.99 | +4.88 | +4.24 | -0.82 |
| 2 agranular | +0.23 | +0.42 | +0.78 | +1.04 | -1.45 |
| 3 dysgranular | -0.27 | -0.27 | -0.06 | +0.10 | -1.74 |
| 4 eulaminate I | -0.38 | +0.06 | -0.15 | -0.36 | +0.49 |
| 5 eulaminate II | -0.21 | +0.04 | -0.08 | -0.24 | +0.16 |
| 6 eulaminate III | +0.37 | -0.05 | +0.06 | +0.20 | -0.53 |
| 7 koniocortex | +2.20 | -0.77 | +0.53 | +1.53 | +1.74 |

Koniocortex-vs-eulaminate I/II contrast (the claim under test), standardised, with a two-sided spin test that rotates the cyto7 type map (released Alexander-Bloch set, n_perm=1000, seed 0):

| map | konio − eulaminate I/II (std) | spin p |
|---|---|---|
| raw SD | +2.49 | 0.001 |
| (A) regress-out mean | -0.82 | 0.060 |
| (A2) binned mean removal | +0.65 | 0.109 |
| (B) coefficient of variation | +1.83 | 0.002 |
| (C) per-type-SD residual | +1.42 | 0.003 |

**Allocortex behaves in the opposite way to koniocortex, and that is the result that survives.** Its standardised offset goes from +3.38 raw to +4.88 under (A2) binned mean removal — controlling for signal level makes the limbic end **more** extreme, not less, because allocortex is only moderately myelinated and so its very high between-subject spread is not what signal level predicts. Agranular cortex moves +0.23 → +0.78. Koniocortex is the mirror image: high signal, and once that is removed the excess variability largely goes with it. So the honest one-line summary of the layer is *residual inter-subject disagreement concentrates in the limbic belt (allocortex ≫ agranular), not at the sensory end*.

Note (C) is the exception (-0.82 for allocortex): that is expected and is a property of the method, not a contradiction — see the caveats.

## Step 3b — vs the released anatomy-only support

Spearman ρ between each variability map and `1 − anatomy-only support`. The question is whether the within-type sign reversal reported previously (positive in the limbic belt, negative in koniocortex) was carried by the signal confound.

| scope | raw SD | (A) regress-out mean | (A2) binned mean removal | (B) coefficient of variation | (C) per-type-SD residual |
|---|---|---|---|---|---|
| overall | +0.045 | +0.096 | +0.123 | +0.114 | -0.181 |
| allocortex-excluded | +0.015 | +0.066 | +0.093 | +0.083 | -0.177 |
| type 1 Allocortex | +0.617 | +0.602 | +0.616 | +0.640 | +0.622 |
| type 2 agranular | +0.319 | +0.331 | +0.328 | +0.291 | +0.301 |
| type 3 dysgranular | +0.095 | +0.151 | +0.158 | +0.062 | +0.078 |
| type 4 eulaminate I | +0.109 | +0.043 | +0.101 | +0.143 | +0.099 |
| type 5 eulaminate II | +0.049 | -0.052 | -0.029 | +0.003 | +0.043 |
| type 6 eulaminate III | -0.290 | +0.251 | +0.147 | -0.151 | -0.280 |
| type 7 koniocortex | -0.320 | -0.100 | -0.200 | -0.273 | -0.300 |

Judged against the control(s) that actually worked ((A2) binned mean removal), and looking only at types where the raw |ρ| exceeded 0.15:
- **Survives:** 1 Allocortex, 2 agranular, 7 koniocortex.
- **Does not survive:** 6 eulaminate III.

The **positive limbic-belt association is solid**: in allocortex (ρ = +0.616) and agranular cortex (ρ = +0.328) low anatomy-only support genuinely marks where individuals' microstructure disagrees with the group label, and this is untouched by the signal control.
The **eulaminate III association does not survive**: raw ρ = -0.290 becomes +0.147 — it *changes sign* — so that part of the previously reported reversal was carried by the signal confound.
The **koniocortex negative association does survive** (ρ = -0.320 → -0.200), even though the koniocortex *level* does not. These are different claims: where variability is high, versus how it covaries with support inside the type.

So the parent report's "sign-reversing within type" framing is **half right** and must be narrowed: the positive limbic-belt end is robust, the negative koniocortex end is robust, but the eulaminate III contribution was a signal artefact. The overall conclusion — read this layer within-type, never as one cortex-wide overlay — is unchanged and if anything reinforced.

Border-distance control (geodesic mm to the nearest cyto7 type boundary): raw SD ρ = -0.043, (A) regress-out mean ρ = -0.023, (A2) binned mean removal ρ = -0.031, (B) coefficient of variation ρ = -0.056, (C) per-type-SD residual ρ = -0.021. No variant is driven by proximity to type borders.

## Files

| file | contents |
|---|---|
| `variability_regressout.{npy,dscalar.nii}` | (A) the SPEC's primary: SD residual after linear regression on signal level. **Over-corrects — do not use as the reference map.** |
| `variability_binned.{npy,dscalar.nii}` | (A2) **the map to use**: SD minus the median SD in its group-mean quantile bin (the only control that reached ρ ≈ 0) |
| `variability_cov.{npy,dscalar.nii}` | (B) coefficient of variation, SD / group-mean (under-corrects) |
| `variability_pertypeSD.{npy,dscalar.nii}` | (C) SD across subjects of the within-type-standardised deviation (within-type measure only; see caveats) |
| `signalcontrol_summary.csv` | every number in this report, per map variant |
| `figure_signalcontrol.png` | (a) per-type standardised variability, raw vs all controls, koniocortex highlighted; (b) the **(A2)** map on the inflated surface |

Note the SPEC designated (A) primary and (A2) supplementary; the Step-2 evidence reverses that, so (A2) is written as a full dscalar too and is the map the report reasons from. (A) is kept for completeness and traceability against the SPEC.

## Caveats

- Controlling for the group mean also removes any *genuine* variability that happens to be co-located with high myelin. These controls are deliberately conservative: they answer "is there variability structure beyond signal level?", not "how much variability is there?".
- **(C) should not be used to compare types**, despite being the only control applied at the per-subject source level. Dividing each subject's deviations by that subject's within-type SD forces unit spread inside every type, which partly defines away the between-type differences the koniocortex/allocortex question is about — visible in allocortex collapsing from +3.38 to -0.82. It also under-corrects the confound itself (ρ = +0.401). It remains a reasonable measure of *within-type* spatial disagreement, which is how its vs-support column should be read.
- The spin test rotates the cyto7 type map, which tests whether koniocortex's value is unusual *for a region of that size and spatial configuration*. It does not test the size of the effect, which the standardised offsets give.
- Sample and MSMAll caveats from the parent report carry over unchanged (first 200 S1200 subjects; MSMAll uses myelin as an alignment feature, so all variability estimates are lower bounds).
