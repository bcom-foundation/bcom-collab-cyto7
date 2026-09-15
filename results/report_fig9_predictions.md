# Report — more Fig. 9 (Zaldívar-Díez & García-Cabezas 2026) predictions vs cyto7 v9

Implements `docs/SPEC_fig9_predictions_batch.md` + the follow-up `docs/SPEC_fig9_batch_fixes.md`.
Run in the `cyto7` conda env. Scripts: `scripts/fig9_predictions.py` (Parts A/B/C) and the
Part-D fix in `scripts/receptor_type_connectivity.py`. Type map: **cyto7 v9**, used directly
(no area-level crosswalk). Builds on the ionotropic/metabotropic result in Supp Fig S5.
Updated 2026-07-22 with the batch fixes (schizophrenia added, bilateral expansion, gradient
specificity, spin validation, dependency pin).

## Two distinct spins are used (methods clarity)
- **Parts A & C (vertex-level):** the released feature-validation machinery —
  `neuromaps.nulls.alexander_bloch(rank_full, atlas="fsLR", density="32k", n_perm=1000,
  seed=0)` — the identical rotation set behind `functional_summary_table_v9.csv` /
  `external_validation_table.csv`. The v9 type-rank map is spun once; rotated type-rank is
  correlated against each fixed vertexwise feature.
- **Part B (parcel-level):** because the ENIGMA data are Desikan-parcellated, a **parcel
  spin** on the fsaverage5 aparc sphere — ENIGMA's `rotate_parcellation` on aparc centroids,
  `np.random.seed(0)`, `n_rot=1000`. The toolbox's high-level `spin_test` is broken in this
  env (a VTK `PointSet.__vtkname__` incompatibility when it loads the fsa5 surface); we
  reproduced its pipeline **VTK-free** by supplying fsa5 sphere coords via
  `nilearn.datasets.fetch_surf_fsaverage("fsaverage5")` and calling the numpy-only
  `centroid_extraction_sphere` / `rotate_parcellation` / `perm_sphere_p` directly (same
  algorithm, same parcel order). These are **not** the same spin — do not conflate them.

**Spin validation (new).** Under the null, the reimplemented parcel spin is well-behaved:
correlating the parcel type score against **500 random maps** gives spin-p mean = **0.504**,
median = 0.496, fraction < 0.05 = **0.052** — i.e. approximately uniform, as a valid null
must be (`python scripts/fig9_predictions.py --spin-sanity`).

## FDR families, seeds, dependency
- Separate BH-FDR families, none merged with the published SF-9 / ext-3 / receptor-19: a
  pooled **Fig-9 family** (H + xu2020 evoexp + hill2010 devexp, 3 metrics) and the
  **disorder family** (8 disorders; a separate BH for the gradient-partialled p's).
- Seeds: vertex spin `seed=0`; parcel rotation `np.random.seed(0)`.
- **Dependency pinned:** `enigmatoolbox` is not on PyPI; installed from
  `git+https://github.com/MICA-MNI/ENIGMA.git@b08974b55243060cbc1fad12c87048037446e8f7`
  (recorded in `environment.yml`).
- **Released tables untouched** (verified git-clean): `functional_summary_table_v9.csv`,
  `external_validation_table.csv`. All outputs are new files.

---

## Part A — receptor diversity (Shannon entropy)  *(unchanged from parent run)*
Pre-registered metric: per-vertex Shannon entropy `H` over the 19 min-max-normalised Hansen
maps; `exp(H)` = effective number of receptors.

| metric | ρ vs type | spin p | ρ (allo-excl) |
|---|---|---|---|
| Shannon entropy H (primary) | **−0.381** | 0.046 | −0.396 |
| CV robustness (↑CV = ↓diversity) | +0.414 | 0.033 | +0.435 |

Direction correct and nominally significant, but the effect is **small** (effective #receptors
≈ 16.9 → 16.1 of 19). Caveat: diversity is over the 19 PET-measured (neuromodulator-heavy)
targets, normative group maps, mostly volume-projected.

## Part C — cortical expansion  *(fixed: now includes a bilateral map)*
**Decision (§2):** the evolutionary axis is tested **bilaterally** with the **Xu-2020
evolutionary-expansion** map (neuromaps `xu2020/evoexp`, fsLR 32k, both hemispheres — §2.1);
the developmental axis uses **Hill-2010 devexp**, which neuromaps ships **right-hemisphere
only**, run consistently **RH-only and labelled exploratory** (§2.2). The RH-only hill2010
*evoexp* used in the parent run is **dropped** (superseded by the bilateral xu2020). No map
was mirrored across hemispheres.

| map | scope | ρ vs type | spin p | ρ (allo-excl) | Fig-9 FDR q |
|---|---|---|---|---|---|
| xu2020 evolutionary expansion | **bilateral** | **−0.266** | 0.028 | −0.303 (p=0.010) | 0.069 |
| hill2010 developmental expansion | RH-only (exploratory) | −0.115 | 0.416 | −0.144 | 0.416 |

**Outcome:** with a proper bilateral map, evolutionary expansion runs in the predicted
direction (ρ=−0.27, p=0.028; allo-excluded p=0.010) — greater expansion in
less-differentiated cortex — a change from the RH-only parent result (which was
n.s., p≈0.49). It sits just above the Fig-9 FDR threshold (q=0.069). Developmental expansion
(RH-only) remains n.s.

> **What the paper does with this (as of 15 Sept 2026).** Both comparisons were run and both
> are reported: they are rows 81 and 82 of the outcome table (Table S8), and §3.7 gives their
> values (ρ = −0.27 and −0.11). The paper **does not build on them**, because the available
> expansion estimates rest on cross-species alignments whose phylogenetic assumptions differ
> from the cortical-type framework. The comparison was therefore removed from Figure 5 and
> moved to §4.10 as future work: a direct human-to-macaque comparison in matched types.
> This report is the record of what was computed, which is why the numbers stay here.

**Fig-9 family (BH-FDR across H + xu2020 evoexp + devexp):** H q=0.069, xu2020 evoexp q=0.069,
devexp q=0.416 — the two evolutionary/diversity axes are nominally significant and near-FDR;
none formally clears q<0.05 in the 3-metric family, but both point the predicted way.

---

## Part B — disease vulnerability (ENIGMA), now 8 disorders + specificity check
Per disorder: parcelwise Cohen's d (cortical thickness, Desikan) vs the continuous per-parcel
type score `c_rank = Σ_t t·fraction_t`; parcel spin (1000 rot); BH-FDR across the **8**
disorders. Output: `figures/v9/external/disease_vulnerability.csv`, `disease_vulnerability.png`.

**Schizophrenia sourcing (§1).** `load_summary_stats("schizophrenia")` is broken in
enigmatoolbox 2.0.3, but the van Erp 2018 case-control thickness table **ships** in the
package (`datasets/summary_statistics/Schizophrenia_case-controls_CortThick.csv`). We read it
directly (semicolon-delimited). It was **verified** to be the case-control d table — its |d|
magnitudes and spatial pattern match van Erp 2018 (largest thinning in fusiform ≈0.40,
superior temporal ≈0.40, inferior temporal ≈0.39). It uses the **opposite sign convention**
(positive = thinning) to the toolbox-loaded tables (depression mean d=−0.04, negative =
thinning), so we **sign-flipped** it to match (negative = thinning). Documented in code.

**Sign convention.** ENIGMA d = case − control ⇒ negative d = atrophy. With atrophy,
**ρ(d, type) > 0 ⇒ more atrophy in LOWER-type cortex = vulnerability falls with type** (Fig. 9).
`ρ(|d|, 7−type) > 0` gives the unambiguous magnitude reading.

| disorder | ρ(d,type) | spin p | FDR q | partial ρ \| gradient | partial p | partial q |
|---|---|---|---|---|---|---|
| depression | +0.597 | 0.001 | **0.004** | +0.457 | 0.004 | **0.016** |
| adhd | +0.414 | 0.001 | **0.004** | +0.546 | 0.003 | **0.016** |
| ocd | +0.317 | 0.005 | **0.012** | +0.124 | 0.342 | 0.456 |
| 22q | +0.356 | 0.011 | **0.018** | +0.357 | 0.026 | 0.069 |
| epilepsy | −0.423 | 0.011 | **0.018** | −0.357 | 0.043 | 0.086 |
| schizophrenia | −0.182 | 0.250 | 0.333 | −0.218 | 0.422 | 0.482 |
| asd | −0.025 | 0.446 | 0.498 | +0.180 | 0.625 | 0.625 |
| bipolar | +0.003 | 0.498 | 0.498 | −0.248 | 0.154 | 0.246 |

**Outcome.** For the canonical atrophy disorders — **depression, ADHD, OCD** (and **22q**) —
cortical thinning is significantly concentrated in **lower-type (limbic/agranular) cortex**
(FDR q<0.05), consistent with Fig. 9. **Epilepsy runs opposite** (thinning weighted to
eulaminate cortex). **Schizophrenia** is n.s. and if anything reversed — its thinning sits in
higher-type temporal/frontal association cortex, so it does **not** follow the low-type
vulnerability pattern (an honest non-confirmation, not a failure). ASD/bipolar null.

**Specificity check (§3) — is it cyto7-specific or just the shared gradient?** Partial Spearman
of d vs type controlling the per-parcel **principal functional gradient** (`margulies2016
fcgradient01`, parcellated on fsa5 aparc), with the type map spun:
- **depression and ADHD survive** the gradient-partialling with FDR q<0.05 (partial ρ=+0.46
  and +0.55) — there is cyto7-specific signal beyond the sensorimotor–association gradient.
- **22q and epilepsy** attenuate to borderline (partial q≈0.07–0.09).
- **OCD attenuates strongly** (p=0.005 → partial p=0.342): that association is largely the
  shared gradient, not cyto7-specific.
This is convergent context, **not** a claim that cyto7 predicts disease.

**Caveats:** Desikan-parcel resolution is coarse and a single continuous type per parcel is
lossy; effect sizes are cross-sectional and heterogeneous across consortia.

---

## Part D — per-type connectivity figure fix  *(done in the parent batch)*
`figures/v9/tractography/per_type_connectivity.png`: length/tortuosity markers carry
**count-weighted SD (not SEM)**; both panel titles show the label-permutation p (degree/vtx
ρ=−0.82, p=0.034; length ρ=+0.96, p=0.003; tortuosity ρ=+1.00, p<0.001). Restaged to
`manuscript/preprint/figures/cyto7_supp_per_type_connectivity.png` (190 mm @ 600 dpi).

---

## Bottom line for the manuscript paragraph
cyto7 recovers **several** Fig. 9 axes from independent data, with honestly varying strength:
- **Disease vulnerability** (Part B) is the strongest convergent signal — depression, ADHD,
  OCD, 22q show atrophy concentrated in lower-type cortex; depression and ADHD remain
  significant even after controlling for the functional gradient (cyto7-specific), while OCD
  is carried by the shared gradient; epilepsy is opposite and schizophrenia does not follow.
- **Evolutionary cortical expansion** (Part C, now bilateral) runs in the predicted direction
  (ρ=−0.27, p=0.028; allo-excluded p=0.010), but **the paper does not build on it**. Both
  expansion comparisons are reported as rows 81 and 82 of the outcome table (Table S8) and
  their values are given in §3.7; the result was removed from Figure 5 and the direct
  human-to-macaque comparison in matched types is listed in §4.10 as future work, because the
  available expansion estimates rest on cross-species alignments whose phylogenetic
  assumptions differ from the cortical-type framework.
- **Receptor diversity** (Part A) is directionally correct but weak.
Together with the ionotropic/metabotropic ratio (Supp Fig S5), cyto7 aligns with **multiple —
not all — of the molecular/cellular gradients Fig. 9 sketches**, and the disease-vulnerability
link is partly (depression, ADHD) independent of the functional gradient.

## Not run this batch (per spec)
Dendritic field size (not in-vivo mappable); epigenetic/plasticity/excitatory gene-set scores
(needs curated AHBA panels — deferred); aperiodic exponent / peak frequency (needs raw
source-localised MEG under DUA — future work).

## Acceptance
- [x] Schizophrenia sourced verifiably (shipped van Erp 2018 table, sign-verified + flipped), included; FDR recomputed across 8 disorders.
- [x] Expansion is **bilateral** for the evolutionary axis (xu2020); developmental kept RH-only and labelled exploratory; hill2010 evoexp dropped — decision stated (§2.1 + §2.2).
- [x] Partial-correlation (controlling the functional gradient) reported for all disorders incl. the FDR-significant ones; specificity framed honestly.
- [x] Spin reimplementation sanity-checked (null ≈ uniform: mean p 0.504, frac<0.05 = 0.052); enigmatoolbox commit pinned in environment.yml; vertex- vs parcel-spin usage described.
- [x] Released SF/external/receptor tables untouched (git-clean); FDR families separate.
