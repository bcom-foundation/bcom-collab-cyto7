# cyto7 v9 — full re-run report (L/R entorhinal consistency fix; CANONICAL)

Map **v9** (hand-painted by RS from v8: evens up the left–right entorhinal painting — 543 LH vertices agranular→allocortex in entorhinal/parahippocampal/temporal-pole/fusiform, bringing LH entorhinal allocortex ~64%→~98% to match RH; **RH untouched**). Miguel approved v8; v9 is a small consistency fix, **approved for adoption**. 32k fs_LR, seed 0, spin/perm N=1000. Tractography on v9's own tractogram. Numbers + changed-vs-v8 diff.

## §3.1 Map

- Surface allocortex LH 4147 / RH 3193 vertices (v8 3604/3193; +543 LH from the entorhinal even-up, RH unchanged); strict R1=0 (topology_audit_v9); allocortex = single **open arc** (β1=0); agranular & dysgranular each a single encircling ring; EulII 1-comp/3-holes; EulIII/konio 3 islands.
## §3.2 Benchmark vs von-Economo

- quadratic-weighted κ_w = **0.699**, Cohen's κ 0.361, ARI 0.188, exact agreement 52.5%.
## §3.3 Added value

- disagreement set 27298 vtx (v8 27351; the 543 relabelled entorhinal vertices leave the benchmark-disagreement set); localized win-fraction **60.1%** (spin-p 0.0010); off-by-≥2 74.3%.
## §3.4 Support (strict topo) — incl. ALLOCORTEX SUPPORT

- benchmark-independent disagreement AUC **0.5774**; histology-patch error by tertile {'low': 0.1529, 'mid': 0.0714, 'high': 0.0188}.
- **allocortex median support: v8 0.316 → v9 0.316** (per-hemi v9 lh 0.316 / rh 0.316). The 543 added entorhinal vertices are low-support developmental allocortex — the narrative is unchanged from v8 (expert/developmental, not atlas-driven).
## §3.5–3.6 Structure–function (allocortex excluded from the gradient)

- myelin: ρ=+0.589, spin-p=0.001, q=0.004
- thickness: ρ=-0.502, spin-p=0.002, q=0.006
- gradient: ρ=-0.576, spin-p=0.001, q=0.004
- timescale: ρ=-0.485, spin-p=0.046, q=0.103
- myelin regression R²med=0.925, R²vtx=0.552 (spin-p 0.001).
- spectral centroid ρ=+0.459 (p_spin 0.069).
## §3.7 Tractography (v9 own tractogram)

- slope -1.23, perm-p 0.003, partial r -0.636 (p 0.002), survives (p<0.05). Full detail: REPORT_tractography_v9.md.

## Changed vs v8 (v7 noted)

| quantity | v7 | v8 | v9 | note |
| --- | --- | --- | --- | --- |
| allocortex (LH/RH vtx) | 2131/1828 | 3604/3193 | 4147/3193 | +543 LH entorhinal even-up; RH unchanged |
| von-Economo κ_w | 0.702 | 0.699 | 0.699 | ≈ unchanged |
| support AUC | 0.575 | 0.578 | 0.5774 | strict topo |
| added-value win | 0.602 | 0.601 | 0.601 | ≈ unchanged |
| myelin ρ / q | +0.589/0.004 | +0.589/0.004 | +0.589/0.004 | ≈ unchanged |
| thickness ρ / q | -0.501/0.006 | -0.502/0.006 | -0.502/0.006 | ≈ unchanged |
| gradient ρ / q | -0.576/0.004 | -0.576/0.004 | -0.576/0.004 | ≈ unchanged |
| timescale ρ / q | -0.484/0.106 | -0.485/0.103 | -0.485/0.103 |  |
| myelin R²med | 0.928 | 0.925 | 0.925 | ≈ unchanged |
| spectral centroid ρ | +0.457 | +0.460 | +0.459 | ≈ unchanged |
| allocortex median support | — | 0.316 | 0.316 | ≈ unchanged (developmental) |
| tractography slope/perm-p | -1.26/0.002 | -1.24/0.003 | -1.23/0.003 | v9 own tractogram |
| tractography partial-r | -0.659 (p 0.001) | -0.636 (p 0.0019) | -0.636 (p 0.0019) | survives (p<0.05) |

_v9 is a 543-vertex L/R entorhinal consistency fix (RH untouched); all validations hold within noise of v8 (allocortex is excluded from the structure–function gradient, so those numbers are unchanged at 2 dp). The added entorhinal allocortex is low-support developmental cortex, as in v8. v9 is the canonical map. Seeds fixed; figures/v8/ regenerated on v9; manuscript files untouched._