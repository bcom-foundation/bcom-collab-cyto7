# Tractography on v9 — §3.7 numbers (v9's own tractogram)

Repo pipeline (`plot_tractography_analysis.py` + `reviewer_response.py` Analysis 3),
inputs `resources/tractography/v9/cyto7.v9_*` (v9's **own** HCP-1065 tractogram fit).
Short/long cutoff **80 mm**, seed **0**, permutation **N=1000**.

## Headline (short-range cortico-cortical connectivity vs cytoarchitectural type-distance)

| quantity | v8 (own tractogram) | **v9 (own tractogram)** |
| --- | --- | --- |
| short-range type-distance log-linear slope | -1.238 | **-1.231** |
| label-permutation null p | 0.003 | **0.003** |
| connectivity vs type-distance r | -0.768 | **-0.765** (p 0.0001) |
| contact-area r (log-conn vs contact area) | +0.585 | **+0.575** |
| partial r (type-distance \| contact area) | -0.636 (p 0.0019, survives) | **-0.636 (p 0.0019, SURVIVES)** |

## Changed vs v8

- slope ≈ unchanged; perm-p 0.003 (v8 0.003); partial r -0.636 (v8 -0.636), ≈ unchanged, **SURVIVES** at p<0.05.
- The core §3.7 claim — short-range connectivity decays with cytoarchitectural type-distance and **survives the contact-area control** — HOLDS on v9.

_v9 = v8 + a 543-vertex L/R entorhinal even-up (RH untouched). Tractography on v9's own tractogram; no v8 fallback. Seeds fixed; manuscript files untouched._