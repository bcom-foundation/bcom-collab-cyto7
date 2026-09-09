# cyto7 final re-run on the clean map (v3_clean) — consolidated REPORT

Every analysis re-run on **v3_clean** (speckle-cleaned; 24 vtx moved vs v3), 32k fs_LR for feature analyses / 164k for topology, seed 0, spin/perm N=1000, short/long cutoff 80 mm. Paste-ready numbers below; a **what-changed-vs-draft** diff closes the report. No manuscript files were edited.

## §3.1 Topology (Annex A)

- v3_clean audit: lh: R1 skip-edges=0; EuII 1 comp/3 holes (ok=True); R4 annular-violations=0; belt components {'Allocortex': 2, 'Agranular': 1, 'Dysgranular': 3} | rh: R1 skip-edges=0; EuII 1 comp/3 holes (ok=True); R4 annular-violations=0; belt components {'Allocortex': 2, 'Agranular': 2, 'Dysgranular': 3}
- Provenance: R1=0 skip-edges; limbic-belt union {allo+agranular+dysgranular} = single closed annulus (R2); eulaminate-II = 1 component, 3 holes; eulaminate-III & koniocortex non-annular islands (see `PROVENANCE_v3_clean.txt`). Figure: `topology_audit.png`.

## §3.2 Benchmark vs von-Economo-derived (Annex B)

- Overall agreement **0.535**, Cohen's κ **0.375**, **quadratic-weighted κ 0.711**, ARI 0.192, mean |Δordinal| 0.544.
> **Paste (§3.2):** "On the standard fsaverage mesh, cyto7 (v3_clean) and the von-Economo-derived cortical-type map agree at quadratic-weighted κ = 0.71 (Cohen's κ 0.37); most disagreement is off-by-one refinement."

## §3.3 Added value over the area-level map

- Disagreement set: **26634** vertices (22706 off-by-one, 3928 off-by-≥2).
- **Localized win-fraction (myelin/thickness/gradient):** all 60.3% (spin-p 0.0010); off-by-1 57.6% (spin-p 0.0020); off-by-≥2 75.5% (spin-p 0.0010).
- Global ρ (context): myelin +0.61 vs +0.50 (q=0.006), thickness -0.51 vs -0.42 (q=0.006), gradient -0.59 vs -0.42 (q=0.006) — cyto7 ≥ area map even whole-cortex.
> **Paste (§3.3):** "Where cyto7 (v3_clean) and the area-level map disagree (26634 vertices), cyto7's label better matches the independent multimodal feature profile in 60% of cases (spin-p 0.001)."

## §3.4 Support calibration (Annex C)

- Benchmark-independent disagreement-AUC (topo·geom·prior) = **0.5622** (geom 0.5685).
- Histology-patch error by support tertile: low 0.2765, mid 0.0667, high 0.1636 (509 belt vertices).

## §3.5–3.6 Structure–function (Annexes D–F)

- T1w/T2w myelin: ρ=+0.591, spin-p=0.001, q=0.004
- thickness: ρ=-0.504, spin-p=0.002, q=0.006
- fMRI gradient: ρ=-0.580, spin-p=0.001, q=0.004
- intrinsic timescale: ρ=-0.482, spin-p=0.047, q=0.106
- Myelin progression fit (dys→konio): R²med=0.930, R²vtx=0.556, spin-p=0.001.
- intrinsic timescale (Sec.7, allo-excl): ρ=-0.511, spin-p=0.026, q=0.068
- spectral centroid (midpoint f): ρ=+0.463, spin-p=0.068, q=0.068
- spectral centroid (geometric f): ρ=+0.476, spin-p=0.061, q=0.068
- spectral centroid (logspaced f): ρ=+0.465, spin-p=0.063, q=0.068

> **Reconciled timescale (authoritative, 9-panel masking, allo incl):** ρ=-0.482, spin-p=0.047, q=0.106. Propagate to Abstract / §3.6 / Annex F.

## §3.7 Tractography (Annex G) — v3_clean tables

- Short-range type-distance slope **-1.11**, label-permutation **perm-p 0.006**; conn-vs-typedist r -0.6051, contact-area r 0.5908, **partial r -0.383** (p 0.086).
- Verdict: the type-distance effect **survives the permutation null; contact-area attenuates the partial**.

## Sensitivity / independence (Annexes)

- Inter-feature r: myelin–thickness -0.574, myelin–gradient -0.544, thickness–gradient 0.366.
- Myelin ρ: 0.6176 (dys→konio) vs 0.5956 (+agranular).
- 164k↔32k resample Dice ≥ 0.790 all types (see `resample_agreement.csv`, `sensitivity_exclusions.csv`).

## What changed vs the current draft (v3 → v3_clean)

| quantity | v3 (draft) | v3_clean | note |
| --- | --- | --- | --- |
| von-Economo κ_w | 0.71 | 0.711 | ≈ unchanged |
| support AUC | 0.562 | 0.5622 | ≈ unchanged |
| added-value win-frac | 0.603 | 0.603 | ≈ unchanged |
| timescale ρ / q | -0.482 / 0.106 | -0.482 / 0.106 | reconciled 9-panel |
| myelin R²med | 0.930 | 0.930 | ≈ unchanged |
| tractography slope | -1.11 | -1.11 | re-run dMRI |
| tractography perm-p | 0.006 | 0.006 | |
| tractography partial r (p) | -0.38 (0.09) | -0.38 (0.09) | contact-area control |

_Seeds/settings identical to prior runs (seed 0, N=1000, cutoff 80 mm). Full figure set under `figures/final/`. No manuscript files touched._