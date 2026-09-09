# cyto7 v5 — full re-run report (cingulate rings + closure + smoothing + strict topology)

Map **v5** (from v4: posterior/isthmus outer agranular→dysgranular; inner agranular ring closed into genuine cortex; strict R1=0; speck-cleanup + gentle boundary smoothing). 32k fs_LR, seed 0, spin/perm N=1000. Tractography on the v3_clean tables. Numbers below + changed-vs-v4 diff.

## §3.1 Map

- Surface allocortex LH 2131 / RH 1769 vertices; strict R1=0 (topology_audit_v5); cingulate dysgranular now a single encircling outer ring, agranular a near-complete inner ring (retrosplenial break).
## §3.2 Benchmark vs von-Economo

- quadratic-weighted κ_w = **0.703**, Cohen's κ 0.369, ARI 0.190, exact agreement 53.1%.
## §3.3 Added value

- disagreement set 26873 vtx; localized win-fraction **60.4%** (spin-p 0.0010); off-by-≥2 74.7%.
## §3.4 Support (strict topo)

- benchmark-independent disagreement AUC **0.5699**; histology-patch error by tertile {'low': 0.1471, 'mid': 0.0361, 'high': 0.0}.
## §3.5–3.6 Structure–function

- myelin: ρ=+0.593, spin-p=0.001, q=0.004
- thickness: ρ=-0.506, spin-p=0.002, q=0.006
- gradient: ρ=-0.578, spin-p=0.001, q=0.004
- timescale: ρ=-0.485, spin-p=0.047, q=0.106
- myelin regression R²med=0.928, R²vtx=0.557 (spin-p 0.001).
- spectral centroid ρ=+0.460 (p_spin 0.064).
## §3.7 Tractography (v3_clean tables)

- slope -1.11, perm-p 0.006, partial r -0.383 (p 0.086).

## Changed vs v4

| quantity | v4 | v5 | note |
| --- | --- | --- | --- |
| allocortex (LH/RH vtx) | 2130/1770 | 2131/1769 | cingulate-scoped (allo ~unchanged) |
| von-Economo κ_w | 0.710 | 0.703 | ≈ unchanged |
| support AUC | 0.561 | 0.5699 | strict topo |
| added-value win | 0.602 | 0.604 | new disagreement set |
| myelin ρ / q | +0.591/0.004 | +0.593/0.004 | mesocortex membership changed |
| thickness ρ / q | -0.504/0.006 | -0.506/0.006 |  |
| gradient ρ / q | -0.580/0.004 | -0.578/0.004 |  |
| timescale ρ / q | -0.482/0.106 | -0.485/0.106 |  |
| myelin R²med | 0.930 | 0.928 | ≈ unchanged |
| spectral centroid ρ | +0.456 | +0.460 | ≈ unchanged |
| tractography slope/perm-p | -1.11/0.006 | -1.11/0.006 | v3_clean tractogram (unchanged) |

_The cingulate reassignment shifts mesocortex (agranular↔dysgranular) membership, so small shifts in the structure–function correlations are expected and reported honestly above. Seeds fixed; figures/v6/ regenerated on v5; manuscript files untouched._