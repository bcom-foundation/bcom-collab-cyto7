# cyto7 v8 — full re-run report (expert-consensus revision; DECISION-GATE ARBITER)

Map **v8** (hand-painted by RS from v7 after the 2026-07-17 García-Cabezas meeting: entorhinal + pre-/parasubiculum → allocortex; belt bands widened for gradient visibility; eulaminate-II thickened so EulI/EulIII don't abut; agranular insula enlarged; piriform merged into the main allocortex arc). The meeting **prioritised visual clarity over strict anatomical realism**, so these numbers are the arbiter. 32k fs_LR, seed 0, spin/perm N=1000. Tractography on v8's own tractogram. Numbers below + changed-vs-v7 diff (v6 noted).

> **NOTE — candidate, not canonical.** v6/v7 retained; v7 stays default until the decision gate below is reviewed.

## §3.1 Map

- Surface allocortex LH 3604 / RH 3193 vertices — **≈ doubled vs v7** (2131/1828) via entorhinal/pre-parasubiculum→allocortex + piriform merge; strict R1=0 (topology_audit_v8); allocortex = single **open arc** (β1=0); agranular & dysgranular each a single encircling ring; EulII 1-comp/3-holes; EulIII/konio 3 islands.
## §3.2 Benchmark vs von-Economo

- quadratic-weighted κ_w = **0.699**, Cohen's κ 0.361, ARI 0.188, exact agreement 52.4%.
## §3.3 Added value

- disagreement set 27351 vtx; localized win-fraction **60.1%** (spin-p 0.0010); off-by-≥2 74.4%.
## §3.4 Support (strict topo) — incl. ALLOCORTEX SUPPORT

- benchmark-independent disagreement AUC **0.5778**; histology-patch error by tertile {'low': 0.1529, 'mid': 0.0662, 'high': 0.1527} — **FLAG: no longer monotone** (v7 was 0.106 → 0.038 → 0.000; the high-support tertile error rose to 0.153), i.e. support is a weaker predictor of histology error on v8.
- **allocortex median support: v7 0.760 → v8 0.316** (per-hemi v8 lh 0.316 / rh 0.316) — **FLAG: the ≈-doubled allocortex is low-support.** The entorhinal/pre-parasubiculum expansion rests on a developmental/expert argument, not atlas evidence, so the support model (atlas/geom/prior) does not support it. See decision gate below.
## §3.5–3.6 Structure–function

- myelin: ρ=+0.589, spin-p=0.001, q=0.004
- thickness: ρ=-0.502, spin-p=0.002, q=0.006
- gradient: ρ=-0.576, spin-p=0.001, q=0.004
- timescale: ρ=-0.485, spin-p=0.046, q=0.103
- myelin regression R²med=0.925, R²vtx=0.552 (spin-p 0.001).
- spectral centroid ρ=+0.460 (p_spin 0.068).
## §3.7 Tractography (v8 own tractogram)

- slope -1.24, perm-p 0.003, partial r -0.636 (p 0.002). Full detail: REPORT_tractography_v8.md.

## Changed vs v7 (v6 noted) — FOUR HEADLINE VALIDATIONS FLAGGED

| quantity | v6 | v7 | v8 | note |
| --- | --- | --- | --- | --- |
| allocortex (LH/RH vtx) | 2131/1828 | 2131/1828 | 3604/3193 | **≈ doubled** (entorhinal/pre-parasub→allo + piriform) |
| von-Economo κ_w ⚑ | 0.703 | 0.702 | 0.699 | ≈ unchanged — EulI→Dys moves isocortex boundary |
| support AUC ⚑ | 0.569 | 0.575 | 0.5778 | strict topo |
| added-value win ⚑ | 0.602 | 0.602 | 0.601 | ≈ unchanged |
| myelin ρ / q ⚑ | +0.586/0.004 | +0.589/0.004 | +0.589/0.004 | ≈ unchanged |
| thickness ρ / q ⚑ | -0.498/0.006 | -0.501/0.006 | -0.502/0.006 | ≈ unchanged |
| gradient ρ / q ⚑ | -0.580/0.004 | -0.576/0.004 | -0.576/0.004 | ≈ unchanged |
| timescale ρ / q | -0.483/0.103 | -0.484/0.106 | -0.485/0.103 |  |
| myelin R²med ⚑ | 0.929 | 0.928 | 0.925 | ≈ unchanged |
| spectral centroid ρ | +0.456 | +0.457 | +0.460 | ≈ unchanged |
| allocortex median support | — | 0.760 | 0.316 | CHANGED (-0.444) |
| tractography slope/perm-p ⚑ | -1.28/0.002 | -1.26/0.002 | -1.24/0.003 | v8 own tractogram |
| tractography partial-r ⚑ | — | -0.659 (p 0.001) | -0.636 (p 0.002) | survives (p<0.05) |

## Decision gate — do the four headline validations HOLD vs v7?

- **κ_w**: HOLD/improve (0.699 vs 0.702, Δ-0.003)
- **structure–function (myelin ρ)**: HOLD/improve (0.589 vs 0.589, Δ+0.000)
- **structure–function (myelin R²med)**: HOLD/improve (0.925 vs 0.928, Δ-0.003)
- **structure–function (gradient ρ)**: HOLD/improve (0.576 vs 0.576, Δ+0.000)
- **support (AUC + allocortex support)**: AUC HOLD/improve (0.578 vs 0.575, Δ+0.003); **allocortex support DEGRADES (0.316 vs 0.760, Δ-0.444) — low-support allocortex painted**
- **added-value win**: HOLD/improve (0.601 vs 0.602, Δ-0.001)
- **tractography partial-r**: HOLD/improve (0.636 vs 0.659, Δ-0.023); survives (p<0.05)

**Verdict: KEEP v7 as default pending expert review.** Three of the four headline validations hold cleanly — the von-Economo benchmark (κ_w 0.699), the structure–function gradient (myelin ρ +0.589 / R²med 0.925, gradient ρ −0.576, all q≤0.006), and tractography (partial r −0.636, p 0.002, **survives** the contact-area control) are all within noise of v7. **But the support validation fails on allocortex support:** the ≈-doubled allocortex has far lower atlas/histology support (median **0.316 vs v7 0.760**), and the histology-patch tertile error is no longer monotone (v8 low 0.153 / mid 0.066 / **high 0.153** vs v7 0.106 / 0.038 / **0.000**). The entorhinal/pre-parasubiculum expansion was painted where the support model has weak support — expected, since it rests on a **developmental/expert argument, not atlas evidence**. This is a *visual-clarity-vs-support* trade-off for RS + García-Cabezas to weigh; **v6/v7 remain intact and v7 stays the default map**. If the expert argument is accepted as authoritative (the meeting mandate), v8 can be adopted with the low allocortex support documented as a known, intended feature.

_v8 is a large, visual-clarity-driven relabel (LH 4315 / RH 2454 vtx; allocortex ≈ doubled). Numbers reported honestly for review; defaults NOT switched. Seeds fixed; figures/v6/ regenerated on v8; manuscript files untouched._