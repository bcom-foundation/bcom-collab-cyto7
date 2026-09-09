# cyto7 v7 — full re-run report (hand-painted release: agranular ventro-anterior insula)

Map **v7** (hand-painted by RS from v6: agranular ventro-anterior insula / von Economo frontoinsular FJK/FI added, L/R-balanced, with a ±1 dysgranular buffer; strict R1=0; isocortex above eulaminate-I unchanged). 32k fs_LR, seed 0, spin/perm N=1000. Tractography on v7's own tractogram. Numbers below + changed-vs-v6 diff (v5 noted).

## §3.1 Map

- Surface allocortex LH 2131 / RH 1828 vertices (unchanged from v6); strict R1=0 (topology_audit_v7); agranular & dysgranular each a single encircling ring; insula now carries an agranular ventro-anterior sector.
## §3.2 Benchmark vs von-Economo

- quadratic-weighted κ_w = **0.702**, Cohen's κ 0.366, ARI 0.189, exact agreement 52.7%.
## §3.3 Added value

- disagreement set 27154 vtx; localized win-fraction **60.2%** (spin-p 0.0010); off-by-≥2 74.3%.
## §3.4 Support (strict topo)

- benchmark-independent disagreement AUC **0.5745**; histology-patch error by tertile {'low': 0.1059, 'mid': 0.0377, 'high': 0.0}.
## §3.5–3.6 Structure–function

- myelin: ρ=+0.589, spin-p=0.001, q=0.004
- thickness: ρ=-0.501, spin-p=0.002, q=0.006
- gradient: ρ=-0.576, spin-p=0.001, q=0.004
- timescale: ρ=-0.484, spin-p=0.047, q=0.106
- myelin regression R²med=0.928, R²vtx=0.556 (spin-p 0.001).
- spectral centroid ρ=+0.457 (p_spin 0.069).
## §3.7 Tractography (v7 own tractogram)

- slope -1.26, perm-p 0.002, partial r -0.659 (p 0.001). Full detail: REPORT_tractography_v7.md.

## Changed vs v6 (v5 noted)

| quantity | v5 | v6 | v7 | note |
| --- | --- | --- | --- | --- |
| allocortex (LH/RH vtx) | 2131/1769 | 2131/1828 | 2131/1828 | unchanged (insula edit is agr/dys/EulI only) |
| von-Economo κ_w | 0.703 | 0.703 | 0.702 | ≈ unchanged |
| support AUC | — | 0.569 | 0.5745 | strict topo |
| added-value win | — | 0.602 | 0.602 | new disagreement set |
| myelin ρ / q | — | +0.586/0.004 | +0.589/0.004 | agranular/insula enlarged |
| thickness ρ / q | — | -0.498/0.006 | -0.501/0.006 |  |
| gradient ρ / q | — | -0.580/0.004 | -0.576/0.004 |  |
| timescale ρ / q | — | -0.483/0.103 | -0.484/0.106 |  |
| myelin R²med | — | 0.929 | 0.928 | ≈ unchanged |
| spectral centroid ρ | — | +0.456 | +0.457 | ≈ unchanged |
| tractography slope/perm-p | — | -1.28/0.002 | -1.26/0.002 | v7 own tractogram |

_v7 adds an agranular ventro-anterior insula sector (EulI/Dys→Agr) with a ±1 dysgranular buffer; mesocortex-driven structure–function and the benchmark may shift slightly vs v6; reported honestly. Isocortex above eulaminate-I is unchanged. Seeds fixed; figures/v6/ regenerated on v7; manuscript files untouched._