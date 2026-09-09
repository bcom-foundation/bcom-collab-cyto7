# Report — per-receptor chemoarchitecture, per-type connectivity, pre-registered MEG ratio

Implements `docs/SPEC_receptor_and_type_connectivity.md` (reviewer G. Ruffini).
Run in the `cyto7` conda env. Script: `scripts/receptor_type_connectivity.py`
(`--n-spin 1000`, all parts). Date: 2026-07-22. Cortical-type map: **cyto7 v9**, used
directly (no area-level García-Cabezas/Scholtens crosswalk built). No NEMO transform.

## Reproducibility / method reuse
- **Spin test:** `neuromaps.nulls.alexander_bloch(rank_full, atlas="fsLR", density="32k",
  n_perm=1000, seed=0)` — the *identical* call, seed and rotation set used by
  `external_validation.py` / `summarise_functional_features.py`. The v9 type-rank map
  (1–7, NaN outside labelled cortex) is spun once; the rotated type-rank is correlated
  against each fixed feature map. Rotations are shared across every new map (receptors,
  composites, MEG ratio), so all statistics are directly comparable to the paper's.
- **ρ:** Spearman over all labelled cortical vertices (types 1–7), both hemispheres
  pooled (n = 51 796). An **allocortex-excluded** sensitivity ρ (types 2–7) is reported
  alongside (Methods §2.6).
- **Labels:** `resolve_target_map("v9","fs_LR")` — verified byte-identical to
  `resources/cyto7_derived/cache/v9_labels_fsLR32k_hemi-{L,R}.npy`.
- **FDR families kept separate** (Benjamini–Hochberg within each; existing q-values NOT
  recomputed or modified): structure–function **9** and external-validation **3** are
  untouched; this task adds a **19-receptor** family, a separate **8-composite** family,
  and adds the MEG slow/fast ratio to the **frequency family** (timescale + SF).
- **Seed:** 0 everywhere. Part-B label-permutation uses the exact 7! = 5040 type
  permutations (deterministic, no seed needed).
- **Existing files not modified:** confirmed `functional_summary_table_v9.csv` and
  `external_validation_table.csv` unchanged (git clean). All outputs are new files.

## QC-anchor check (Part A gate — passed)
Descriptive ρ (types 1–7, allocortex included) for all 19 receptors matched the spec's
QC anchors within **|Δ| ≤ 0.005** (tolerance 0.03). No label/hemisphere/order mismatch.

---

## Part A — per-receptor receptor gradients
Output: `figures/v9/structure_function/receptor_type_association.csv`,
`receptor_composites.csv`, `receptor_gradients.png` (heatmap + forest, 190 mm @ 600 dpi),
`receptor_composites_panel.png`.

**Per-receptor ρ vs cyto7 type (19-receptor FDR family).** Four survive FDR (q < 0.05):

| receptor | system | class | ρ | spin p | q | ρ (allo-excl) | reversals |
|---|---|---|---|---|---|---|---|
| MOR | opioid | metabotropic | −0.548 | 0.005 | **0.043** | −0.570 | 2 |
| 5-HT1a | serotonin | metabotropic | −0.537 | 0.007 | **0.043** | −0.519 | 0 |
| CB1 | endocannabinoid | metabotropic | −0.406 | 0.009 | **0.043** | −0.421 | 1 |
| NET | noradrenaline | transporter | +0.441 | 0.003 | **0.043** | +0.420 | 0 |
| D1 | dopamine | metabotropic | −0.394 | 0.017 | 0.065 | −0.422 | 2 |
| 5-HT4 | serotonin | metabotropic | −0.373 | 0.048 | 0.152 | −0.395 | 2 |
| DAT | dopamine | transporter | −0.340 | 0.061 | 0.165 | −0.319 | 2 |
| D2 | dopamine | metabotropic | −0.355 | 0.108 | 0.256 | −0.361 | 2 |
| mGluR5 | glutamate | metabotropic | −0.251 | 0.127 | 0.268 | −0.289 | 4 |
| … (10 more, all n.s.: H3, 5-HT2a, 5-HT6, 5-HTT, NMDA, α4β2, M1, VAChT, GABA-A, 5-HT1b) |

Signs and rank order match the anchors; the gradient is carried by **inhibitory/modulatory
metabotropic receptors falling** with cortical type (MOR, 5-HT1a, CB1, D1) and the
**noradrenaline transporter NET rising** — i.e. neuromodulator density is highest in the
least-differentiated (limbic/agranular) cortex and lowest in koniocortex, with NET
opposite. Allocortex-excluded ρ are near-identical (signs and significance unchanged).

**Composites (separate 8-item FDR family)** — the Fig. 9 reproduction:

| composite | members | ρ | spin p | q |
|---|---|---|---|---|
| iono − metabo index | (iono)−(metabo) | **+0.426** | 0.002 | **0.016** |
| dopamine | D1, D2 | −0.460 | 0.008 | **0.032** |
| metabotropic | 12 receptors | −0.408 | 0.016 | **0.043** |
| serotonin | 5 receptors | −0.290 | 0.108 | 0.216 |
| glutamate − GABA | (mGluR5,NMDA)−(GABA-A) | −0.163 | 0.184 | 0.294 |
| reuptake/innervation | 4 transporters | +0.037 | 0.831 | 0.844 |
| ionotropic | NMDA, α4β2, GABA-A | −0.035 | 0.844 | 0.844 |
| acetylcholine | α4β2, M1 | −0.025 | 0.807 | 0.844 |

**Fig. 9 axis reproduced:** the **ionotropic/metabotropic index rises with cortical type**
(ρ = +0.43, q = 0.016) — metabotropic (neuromodulator) dominance in low-differentiation
cortex gives way to relatively more ionotropic tone toward koniocortex, exactly the
predicted direction. The **metabotropic composite falls** (ρ = −0.41) while the
**ionotropic composite is flat** (ρ = −0.03, n.s.), so the index is driven by the
metabotropic decline.

**Decisions / caveats (stated per spec A.4.5):**
- Composite membership is non-overlapping: system composites (serotonin/dopamine/
  acetylcholine) use their **receptor** members only; the 4 transporters form the separate
  "reuptake/innervation" composite (so 5-HTT is not double-counted in serotonin, etc.).
- PET densities are tracer-specific units → composites are means of **z-scored** maps and
  the index is a **contrast of z-composites**, never a raw ratio of densities.
- The Hansen panel is **neuromodulator-heavy**. It reproduces the ionotropic/metabotropic
  axis of Fig. 9 well, but only partially addresses a clean excitatory/inhibitory
  (glutamate/GABA) contrast: the **glutamate − GABA composite is under-powered** (3 maps:
  mGluR5 + NMDA vs GABA-A; ρ = −0.16, n.s.) and is **flagged** as such in
  `receptor_composites.csv` (`underpowered_flag`).

---

## Part B — per-type connectivity from the tractogram
Reuses the cached HCP-1065 type×type matrices (`connectivity_short.csv` <80 mm,
`connectivity_long.csv` >80 mm) and per-bundle geometry
(`resources/tractography/v9/cyto7.v9_tract_geometry_per_bundle.csv`). No tractogram
recompute. Outputs: `figures/v9/tractography/per_type_connectivity{,_trends,_myelin_xref}.csv`
and `per_type_connectivity.png`. Trend test = exact 7! = 5040 label-permutation null.

| summary | ρ vs type | perm p |
|---|---|---|
| mean tortuosity | **+1.000** | 0.0004 |
| mean bundle length (mm) | **+0.964** | 0.003 |
| degree per vertex (total) | **−0.821** | 0.034 |
| degree per vertex (short) | **−0.893** | 0.012 |
| degree per vertex (long) | **−0.821** | 0.034 |
| degree short (raw) | +0.107 | 0.84 |
| degree long (raw) | +0.429 | 0.35 |
| degree total (raw) | +0.179 | 0.71 |

**Read:** once normalised by type size (surface vertex count), **connectional degree
falls with cortical type** — least-differentiated cortex (allo/agranular) is the most
densely connected per unit area, especially at short range — while **bundles get longer
and more tortuous** toward koniocortex. Raw (un-normalised) degree shows no trend, i.e. the
per-vertex effect is a genuine density effect, not a size artefact.

**Myelin cross-reference (per-type medians from `functional_summary_table_v9.csv`):**
short-range degree-per-vertex is strongly **anti**-correlated with T1w/T2w myelin
(ρ = −0.96); bundle length (+0.89) and tortuosity (+0.79) co-vary **with** myelin.

**Caveat carried in the text (spec B.3):** streamline count and length are
connectivity-density proxies, **not** myelination. Results are stated as connectional
properties *by type*, cross-referenced to the myelin gradient — we do **not** claim
"more myelinated connections" from tractography, and note the standard tractography
biases (gyral/length bias, false positives; template-space population average).

---

## Part C — pre-registered MEG slow/fast ratio
Single committed metric (fixed before computing): **SF(v) = (δ+θ)/(β+γ₁)**, using the
cached `hcps1200` band maps (γ = γ₁, as elsewhere). Pre-registered direction: **SF falls
with cortical type**. One test only; no band-combination search. Output:
`figures/v9/structure_function/meg_ratio_summary.csv`, `meg_slow_fast_ratio.png`.

- ρ vs type = **−0.409**, spin p = **0.134**, q (frequency family, with timescale) = 0.134.
- Allocortex-excluded: ρ = −0.396, spin p = 0.157.

**Outcome (reported plainly, per Ricardo's negative-result style):** the ratio is in the
**predicted direction** — slower dynamics in less-differentiated cortex — but **does not
reach significance** against the spatial-autocorrelation null (spin p = 0.13). This mirrors
the existing intrinsic-timescale result (suggestive, does not survive the spin): the MEG
band-power maps carry a clear naive gradient that the spatial null largely explains. We do
not over-claim; SF is reported as a single pre-registered, directionally-consistent but
non-significant test alongside timescale in the frequency family.

---

## Acceptance checklist
- [x] `receptor_type_association.csv` — 19 receptors (ρ, spin_p, fdr_q, per-type medians,
  allo-excluded ρ/p, reversals); per-receptor ρ match QC anchors within 0.005.
- [x] Receptor×type heatmap + forest plot (with spin-null 95% CIs) + metabotropic/ionotropic
  composite panel (190 mm @ 600 dpi, no title/caption line).
- [x] `per_type_connectivity.csv` (+ trends, + myelin xref) + figure; streamlines≠myelin
  caveat recorded.
- [x] Pre-registered MEG SF ratio computed once; ρ/spin_p/fdr_q reported (n.s., stated plainly).
- [x] Existing feature/external q-values unchanged; three FDR families kept separate.
- [x] Seeds (0) and rotation set (alexander_bloch fsLR 32k, n=1000) recorded.

## Hand-back
Three CSVs + figures + this report, ready for the §3.8 + Annex H replacement text
(per-receptor gradients reproducing Fig. 9) and the per-type-connectivity note.
