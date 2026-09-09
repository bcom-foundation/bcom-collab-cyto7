# Report — definitional-validation add-ons (3 light analyses)

Implements `docs/SPEC_definitional_validation.md`. Env: `cyto7`. Map: **cyto7 v9**,
32k fs_LR. Date: 2026-07-23. Run **concurrently with the two MEG jobs** (Option A
`meg_aperiodic.py`, Option B `meg_subjectlevel.py`) without disturbing them:
BLAS threads pinned to 1; new packages installed `--no-deps` only; **numpy/scipy
unchanged (2.4.6 / 1.17.1)**; all outputs confined to `figures/v9/definitional/`;
no released table modified (verified git-clean, see end).

## Shared machinery (reused, not recomputed)
- The released Alexander-Bloch spin rotations for the v9 type-rank map (fsLR 32k,
  **seed 0, n_spin 1000**) were generated **once** and cached to
  `figures/v9/definitional/_spin_nulls_v9_fsLR32k_seed0_n1000.npy`; all vertex-level
  analyses reuse that identical rotation set. `definitional_common.evaluate_cached()`
  mirrors `external_validation.evaluate()`'s per-type Spearman + spin-p + BH-FDR math
  exactly (imports the released `_bh`).
- Each analysis is its **own FDR family** (published q-values unchanged).

---

## Analysis 2 — negative control (sulcal depth) — **COMPLETED**

Pre-registered as an **expected null**: a geometric map with no cytoarchitectural
meaning should not track type; demonstrates the spin test does not manufacture
positives on spatially-autocorrelated maps. Reported standalone (not FDR-pooled with
the definitional tests). Sulcal depth = neuromaps fs_LR 32k `desc-sulc`; curvature =
discrete mean-curvature proxy (uniform-Laplacian magnitude) from the fs_LR 32k
midthickness mesh.

| map | ρ vs type | naive p_param | spin p | ρ (allo-excl) | spin p (allo-excl) |
| --- | --- | --- | --- | --- | --- |
| **sulcal depth** (pre-reg expected-null) | **−0.039** | 1.9e−21 | **0.661** | −0.020 | 0.814 |
| cortical curvature (secondary) | −0.051 | 8.9e−36 | 0.529 | −0.033 | 0.689 |

n = 58 731 labelled cortical vertices. **Result matches the pre-registration:** no
alignment with cyto7 type under the spin null. Note the instructive contrast — the
naive parametric p is astronomically "significant" (1e−21) purely from the ~59k
autocorrelated vertices, yet the spatial-autocorrelation-preserving spin gives
p = 0.66. This is exactly why the spin test is used throughout the paper.

**Paper line (Methods / §3.5):** "A pre-registered negative control (sulcal depth)
showed no alignment with cortical type (ρ = −0.04, spin p = 0.66), confirming the
spin test does not manufacture positives on spatially-autocorrelated maps."

Outputs: `figures/v9/definitional/negative_control.csv`, `negative_control.png`.

---

## Analysis 1 — layer-marker genes (RORB) — **COMPLETED (RORB strongly supports the L4 criterion)**

**Pre-registration:** RORB (layer-IV excitatory marker) predicted **ρ > 0** with type —
a falsifiable molecular test of the layer-IV definitional criterion. CUX2 (upper),
FEZF2/FOXP2/TLE4 (deep) + upper/granular/deep z-composites form the rest of a single
**layer-marker-gene FDR family** (BH within family).

**Resolution choice (documented): parcel-level.** `abagen` on the volumetric
Desikan-Killiany atlas → region×gene (5 AHBA donors, `lr_mirror='bidirectional'`,
`norm_matched=False`, `missing='interpolate'`); each of the 68 Desikan cortical
parcels gets the released continuous cyto7 type score
(`figures/v9/crossed/desikan_x_cyto7_composition.csv`, c_rank = Σ_t t·fraction_t —
identical to the ENIGMA disease test); parcel spin = fsa5 aparc `rotate_parcellation`
(seed 0, n=1000), the same machinery as `fig9_predictions.py` Part B.

**Environment note (how it was run without disturbing the MEG jobs):** `abagen` was
**not** installed in `cyto7` (contrary to the task premise), and abagen 0.1.3 is
incompatible with the env's **pandas 3.0.3** (uses removed `DataFrame.append`,
`groupby(axis=1)`, `set_axis(inplace=)`). Rather than change the shared pandas, the
expression was computed in an **isolated `abagen_env`** (`conda create ... python=3.10
"pandas<2" "numpy<2" scipy nibabel nilearn` + `pip install --no-deps abagen`) and
exported to `figures/v9/definitional/layer_marker_expression.csv`; the cyto7 spin/FDR
step reads that CSV (no abagen import in cyto7). The MEG env's numpy/scipy/pandas were
never modified.

**Donor note:** 5 of 6 AHBA donors (9861/10021/12876/14380/15697). Donor **15496** has
a dead Allen download URL in abagen 0.1.3 (HTTP 404, retried once — still 404); 5/6 is
standard for AHBA and documented.

**Results (n = 68 Desikan parcels; layer-marker-gene FDR family):**

| gene / composite | layer | ρ vs type | spin p | FDR q | ρ (allo-excl, n=66) |
| --- | --- | --- | --- | --- | --- |
| **RORB** (pre-reg primary) | IV (granular) | **+0.776** | <0.001 | **<0.001** | **+0.756** |
| CUX2 | II/III (upper) | +0.711 | <0.001 | <0.001 | +0.687 |
| FEZF2 | V (deep) | −0.593 | <0.001 | 0.0008 | −0.559 |
| FOXP2 | VI (deep) | +0.689 | 0.001 | 0.0013 | +0.660 |
| TLE4 | VI (deep) | +0.294 | 0.028 | 0.031 | +0.330 |
| upper composite (CUX2) | — | +0.711 | <0.001 | <0.001 | +0.687 |
| granular composite (RORB) | — | +0.776 | <0.001 | <0.001 | +0.756 |
| deep composite (FEZF2/FOXP2/TLE4) | — | +0.335 | 0.031 | 0.031 | +0.290 |

**Verdict:** the pre-registered primary **RORB rises steeply with cyto7 type
(ρ = +0.78, spin p < 0.001, FDR q < 0.001)** and is robust to allocortex exclusion
(ρ = +0.76) — an independent **molecular confirmation of the layer-IV granularity
criterion** that defines the atlas. The upper-layer marker CUX2 also rises; among deep
markers the signal is mixed (FOXP2/TLE4 rise, the L5 marker FEZF2 falls, ρ = −0.59),
so the "deep composite" is weak/mixed and should be read per-gene, not pooled. RORB is
a strong candidate for a new **Fig. 5 bar** (parcel-level ρ; flagged, Fig. 5 not
modified).

**Caveats (stated):** AHBA six donors (here five; mostly LH, five male); expression is
transcript, not protein; Desikan parcels are coarse and one type score per parcel is a
simplification. Direction and significance are nonetheless unambiguous.

Outputs: `figures/v9/definitional/layer_marker_genes.csv`, `layer_marker_genes.png`,
`layer_marker_expression.csv` (the exported abagen table).

---

## Analysis 3 — BigBrain per-layer thickness (L4) — **STOP-AND-LOG (data unavailable)**

**Pre-registration (unchanged):** layer-IV **thickness** (and L4 fraction = L4/total)
predicted **ρ > 0** with type — the direct histological counterpart of the
definitional criterion; own **BigBrain-L4 FDR family** {L4 thickness, L4 fraction};
allocortex-excluded sensitivity.

**Why not completed now:** the Wagstyl-2020 per-layer thickness maps (6 layers, tpl-
fs_LR den-32k) are **not in the repo cache** (only the 50-depth intensity *profiles*
and Hist-G1 were retained) and are **not distributed via neuromaps**. They require
re-downloading BigBrainWarp's large `BBW_BigData.zip` (sciebo), which was out of scope
for these light add-ons and not run while the MEG jobs occupy the machine. Script
`scripts/bigbrain_layer_thickness.py` is written and will compute L4 thickness + L4
fraction (released spin, own family) the instant the six layer maps are dropped into
`resources/neuromaps_cache/bigbrain_layer{1..6}_thickness_fsLR32k_hemi-{L,R}.npy`.

**Honest caveat (as in §4.9):** BigBrain is a **single specimen** warped to a group
surface — the exact limitation that made §4.9 inconclusive. Even when run, it may not
resolve under the spin null. For now the single-specimen limitation stands and §4.9's
honest framing is retained (an acceptable, publishable outcome per the SPEC). Stop
note: `figures/v9/definitional/bigbrain_layer_thickness.STOP.txt`.

---

## Provenance / reproducibility
- cyto7 **v9** type map; released spin rotations (Alexander-Bloch, fsLR 32k, **seed 0,
  n_spin 1000**), cached once and reused.
- Packages: neuromaps 0.0.7, nibabel 5.4.2, numpy 2.4.6, scipy 1.17.1, pandas 3.0.3;
  **abagen 0.1.3** (added `--no-deps`; incompatible with pandas 3 — see Analysis 1);
  enigmatoolbox (fsa5 aparc `rotate_parcellation`, for the gene parcel spin).
- Cache paths: `resources/neuromaps_cache/` (sulc via neuromaps `fetch_atlas`);
  AHBA microarray in `~/abagen-data` (5 donors); spin nulls in
  `figures/v9/definitional/_spin_nulls_v9_fsLR32k_seed0_n1000.npy`.
- Scripts: `scripts/definitional_common.py`, `negative_control.py`,
  `layer_marker_genes.py`, `bigbrain_layer_thickness.py`.

## Summary
- **Analysis 1 (RORB layer-marker genes): completed** — pre-registered primary **RORB
  ρ = +0.78 (spin p < 0.001, FDR q < 0.001)**, allo-excl +0.76: molecular confirmation
  of the layer-IV criterion. CUX2 (+0.71) also rises; deep markers mixed (FOXP2/TLE4 up,
  FEZF2 down). Candidate new Fig. 5 bar. Run via an isolated pandas<2 `abagen_env`; the
  cyto7/MEG environment was never modified. 5/6 AHBA donors (15496 URL dead).
- **Analysis 2 (negative control): completed** — clean pre-registered null (sulc ρ=−0.04,
  spin p=0.66); ready as a one-line robustness statement.
- **Analysis 3 (BigBrain L4): held / stop-and-log** — layer-thickness data not available
  locally / not in neuromaps; per instruction, `BBW_BigData.zip` was NOT downloaded
  mid-run (may be dropped). §4.9's single-specimen framing stands; script ready if revisited.
