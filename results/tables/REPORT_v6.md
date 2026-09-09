# cyto7 v6 — full re-run report (hand-painted release: cingulate rings completed)

Map **v6** (hand-painted by RS from v5: cingulate dysgranular outer ring filled + agranular inner ring completed on genuine cortex; strict R1=0; isocortex unchanged). 32k fs_LR, seed 0, spin/perm N=1000. Tractography on the v3_clean tables. Numbers below + changed-vs-v5 diff (v4 noted).

## §3.1 Map

- Surface allocortex LH 2131 / RH 1828 vertices; strict R1=0 (topology_audit_v6); agranular & dysgranular now each a single encircling ring (inner / outer) — the v5 retrosplenial interruption is closed.
## §3.2 Benchmark vs von-Economo

- quadratic-weighted κ_w = **0.703**, Cohen's κ 0.370, ARI 0.190, exact agreement 53.1%.
## §3.3 Added value

- disagreement set 26858 vtx; localized win-fraction **60.2%** (spin-p 0.0010); off-by-≥2 75.1%.
## §3.4 Support (strict topo)

- benchmark-independent disagreement AUC **0.5686**; histology-patch error by tertile {'low': 0.1059, 'mid': 0.0377, 'high': 0.0}.
## §3.5–3.6 Structure–function

- myelin: ρ=+0.586, spin-p=0.001, q=0.004
- thickness: ρ=-0.498, spin-p=0.002, q=0.006
- gradient: ρ=-0.580, spin-p=0.001, q=0.004
- timescale: ρ=-0.483, spin-p=0.046, q=0.103
- myelin regression R²med=0.929, R²vtx=0.557 (spin-p 0.001).
- spectral centroid ρ=+0.456 (p_spin 0.070).
## §3.7 Tractography (v6's own tractogram — SUPERSEDES the v3_clean tables)

- **v6 own tractogram** (`resources/tractography/v6/`, seed 0, N=1000, cutoff 80 mm):
  short-range type-distance slope **−1.278**, label-permutation perm-p **0.002**,
  connectivity-vs-type-distance r **−0.790** (p < 0.001), contact-area r **0.570**,
  **partial r (type-distance | contact area) −0.677 (p 0.0008 — SURVIVES)**.
- Changed vs the borrowed v3_clean tables (slope −1.11, perm-p 0.006, partial r −0.383
  p 0.086): the contact-area-controlled effect **now survives** on v6's own tractogram.
  Full detail + caveats in `tractography/REPORT_tractography_v6.md`.
- (Prior borrowed-table line, retained for the record: slope −1.11, perm-p 0.006,
  partial r −0.383, p 0.086.)

## Changed vs v5 (v4 noted)

| quantity | v4 | v5 | v6 | note |
| --- | --- | --- | --- | --- |
| allocortex (LH/RH vtx) | 2130/1770 | 2131/1769 | 2131/1828 | RH allocortex +59 (hand paint) |
| von-Economo κ_w | 0.710 | 0.703 | 0.703 | ≈ unchanged |
| support AUC | — | 0.570 | 0.5686 | strict topo |
| added-value win | — | 0.604 | 0.602 | new disagreement set |
| myelin ρ / q | — | +0.593/0.004 | +0.586/0.004 | dysgranular enlarged |
| thickness ρ / q | — | -0.506/0.006 | -0.498/0.006 |  |
| gradient ρ / q | — | -0.578/0.004 | -0.580/0.004 |  |
| timescale ρ / q | — | -0.485/0.106 | -0.483/0.103 |  |
| myelin R²med | — | 0.928 | 0.929 | ≈ unchanged |
| spectral centroid ρ | — | +0.460 | +0.456 | ≈ unchanged |
| tractography slope/perm-p | — | -1.11/0.006 (v3_clean tables) | **-1.28/0.002 (v6 own tractogram)** | first run on v6's own tractogram; partial r -0.68 p 0.0008 SURVIVES (was -0.38 p 0.086) — see §3.7 |

_v6 enlarges the cingulate dysgranular belt (hand-painted), so mesocortex-driven structure–function and the benchmark may shift slightly vs v5; reported honestly above. Isocortex is identical to v5. Seeds fixed; figures/v6/ regenerated on v6; manuscript files untouched._