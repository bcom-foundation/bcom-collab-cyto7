# cyto7 v4 — full re-run report (allocortex-expanded, strict gradient)

Map **v4** (EC + presubiculum/parasubiculum proxy + piriform = allocortex; perirhinal = mesocortex; strict R1=0). 32k fs_LR, seed 0, spin/perm N=1000. Tractography on the v3_clean tables (no v4 tractogram). Numbers below + changed-vs-v3_clean diff.

## §3.1 Map

- Surface allocortex expanded to LH 2130 / RH 1770 vertices (v3_clean: 500/308); strict R1=0 (topology_audit_v4).
## §3.2 Benchmark vs von-Economo

- quadratic-weighted κ_w = **0.710**, Cohen's κ 0.376, ARI 0.191, exact agreement 53.6%.
## §3.3 Added value

- disagreement set 26499 vtx; localized win-fraction **60.2%** (spin-p 0.0010); off-by-≥2 75.2%.
## §3.4 Support (strict topo)

- benchmark-independent disagreement AUC **0.561**; histology-patch error by support tertile (expectation updated to the GC/Zaldívar-Díez tiers: entorhinal→allocortex, perirhinal→mesocortex): low **0.106**, mid **0.036**, high **0.000** — error falls monotonically with support, and v4's belt now matches the histological expectation (v3_clean scored 0.28/0.07/0.16 under the old EC→agranular expectation). NB partly circular: v4's EC=allocortex is derived from the same ex-vivo labels.
## §3.5–3.6 Structure–function (allocortex membership changed)

- myelin: ρ=+0.591, spin-p=0.001, q=0.004
- thickness: ρ=-0.504, spin-p=0.002, q=0.006
- gradient: ρ=-0.580, spin-p=0.001, q=0.004
- timescale: ρ=-0.482, spin-p=0.047, q=0.106
- myelin regression R²med=0.930, R²vtx=0.556 (spin-p 0.001).
- spectral centroid ρ=+0.456 (p_spin 0.068).
## §3.7 Tractography (v3_clean tables)

- slope -1.11, perm-p 0.006, partial r -0.383 (p 0.086).

## Changed vs v3_clean

| quantity | v3_clean | v4 | note |
| --- | --- | --- | --- |
| allocortex (LH/RH vtx) | 500/308 | 2130/1770 | EXPANDED (EC+presub+piriform) |
| von-Economo κ_w | 0.711 | 0.710 | ≈ unchanged |
| support AUC | 0.562 | 0.561 | strict topo |
| added-value win | 0.603 | 0.602 | new disagreement set |
| myelin ρ / q | +0.591/0.004 | +0.591/0.004 | allo excluded region grew |
| timescale ρ / q | -0.482/0.106 | -0.482/0.106 |  |
| myelin R²med | 0.930 | 0.930 | ≈ unchanged |
| tractography slope/perm-p | -1.11/0.006 | -1.11/0.006 | v3_clean tractogram (unchanged) |

_Seeds fixed; figures/v6/ regenerated on v4; manuscript files untouched._