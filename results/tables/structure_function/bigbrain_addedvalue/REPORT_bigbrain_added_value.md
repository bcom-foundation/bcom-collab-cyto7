# BigBrain histology-based added value (SPEC_external_validation Tier 1A, Analysis 2)

Repeat of the §3.3 added-value test with the **BigBrain Hist-G1 histological gradient** as the arbiter (BigBrainWarp, fs_LR 32k; Paquola 2019, single specimen). cyto7 **v9** vs the von-Economo-derived type map, on the isocortex disagreement set. Spin null n=1000, seed 0.

## Global tracking of the BigBrain gradient

- ρ(cyto7 type, BigBrain Hist-G1) = **+0.209**; ρ(von-Economo, BigBrain) = +0.158; Δ|ρ| = +0.051 (cyto7 better spin-p 0.343).
## Localized head-to-head on the disagreement set — the decisive test

- disagreement set n = 27298 isocortex vertices (off-by-1 23168, off-by-≥2 4130).
| subset | n | cyto7 win-fraction | spin-p | null mean |
| --- | --- | --- | --- | --- |
| all | 27298 | **0.499** | 0.7403 | 0.519 |
| off1 | 23168 | **0.492** | 0.7862 | 0.521 |
| off2 | 4130 | **0.542** | 0.2797 | 0.517 |

**Verdict:** on the disagreement set cyto7 wins **49.9%** (spin-p 0.7403) with BigBrain as arbiter — cyto7 does NOT beat von-Economo against BigBrain histology (win ≤ 50%).

_BigBrain is a single specimen (n=1) — an independent histological reference, not a population map; the Hist-G1 surface is spatially noisy at 32k. Reported honestly. Outputs in `figures/v8/structure_function/bigbrain_addedvalue/`. Manuscript untouched._