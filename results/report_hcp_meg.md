# Report — HCP-MEG aperiodic-exponent test: Stage-0 audit and stop-and-log

Implements `docs/SPEC_hcp_meg_aperiodic.md`. Env: `cyto7`. Date: 2026-07-23.
Cortical-type map intended for the test: **cyto7 v9**. Server:
`$CYTO7_DATA_DIR Young MEG data`.

**Bottom line.** The pre-registered primary metric — the `specparam` aperiodic
1/f exponent — **cannot be computed from the HCP-MEG packages that were
downloaded.** Those packages contain only *band-limited power* products
(band-power envelopes and band-power connectivity), not the broadband source
(or sensor) time series a spectral fit requires. This is a data-availability
blocker, **not** a null result. No cyto7 statistical test was run. Per the SPEC
guardrail ("stop-and-log on any missing/failed input rather than fabricating")
and Ricardo's decision (strict stop-and-log), Stage 0 was completed — audit +
integrity manifest — and Stages 1–5 were not attempted on these packages.

---

## Stage 0 — inventory

Three package types are present on the server, for HCP young-adult subjects,
all produced with **megconnectome version 3.0** (from each archive's
`release-notes/MEG.txt`):

| Package (archive suffix) | n subjects | Size each | What it actually contains |
|---|---|---|---|
| `*_MEG_Restin_dtseries.zip` | 89 | ~20 GB | `bfblpenv/` (beamformer) and `icablpenv/` (ICA-MNE) **band-power *envelope*** dense time series, 8 bands (δ, θ, α, β-low, β-high, γ-low, γ-mid, γ-high; plus `whole` for icablpenv), 3 runs (3-/4-/5-Restin). No broadband signal. |
| `*_MEG_Restin_parcel_yeo.zip` | 89 | ~0.28 GB | `blpcorr`/`imagcoh` band-power **connectivity** matrices (`.pconn.nii`) between Yeo-2011 17-network parcels, per band. No spectra, no time series. |
| `*_MEG_anatomy.zip` | 95 | ~0.07 GB | Per-subject head model + source models (2D and 3D 4/6/8 mm) + 4k_fs_LR surfaces + T1w. No functional data. |

Subject counts and the subject×package coverage are recorded in
`results/hcp_meg_manifest.tsv` (one row per archive).

### Evidence for the "band-power envelope, not spectrum" characterisation
One `bfblpenv` file was extracted to server scratch and its CIFTI header read
(`100307_MEG_3-Restin_bfblpenv_delta.power.dtseries.nii`):

- Shape **(14690 time points × 8004 sources)**; `SeriesAxis` step ≈ 0.0202 s
  → the *envelope* is sampled at ~49.6 Hz on the 8004-grayordinate 4k_fs_LR
  source mesh. (BrainModel structure labels are `CIFTI_STRUCTURE_INVALID` — a
  known megconnectome quirk; the sources correspond to the anatomy
  `sourcemodel_2d`.)
- Each such file is the *time-varying power in one frequency band*. Averaging
  over time yields **one power value per source per band → 8 band-power values
  per source**. That is the same *kind* of quantity as the neuromaps
  `hcps1200` band-power maps already used in the paper (there: 5 bands, one
  group map), only per-subject and with 8 bands and two source-projection
  variants.

The probe file was deleted immediately; server scratch is empty (peak extra
disk use = one 470 MB file, well within the one-subject-at-a-time budget).

---

## Why these packages cannot yield the pre-registered metric

The `specparam` aperiodic exponent is the slope of the 1/f background of a
**resolved power spectrum**, fit continuously over 2–40 Hz after separating
oscillatory peaks. Producing it needs a broadband signal per source per
subject, from which a Welch PSD is computed.

- **Band-power envelopes discard exactly what the fit needs.** An amplitude
  envelope is the (Hilbert/RMS) magnitude of a band-passed signal over time; it
  retains only slow power fluctuations within a band and throws away phase and
  the within-band spectral shape. You cannot reconstruct a 2–40 Hz PSD from
  band envelopes.
- **Eight band values are not a spectrum.** In principle one could regress
  log-power on log-band-centre across the 8 bands and call the slope a "spectral
  exponent," but (i) that is a coarse 8-point summary contaminated by
  oscillatory peaks (α, β), which is precisely what `specparam` exists to
  separate; and (ii) it is the same band-limited quantity the SPEC set out to
  *replace* — reporting it as "the aperiodic exponent" would be the fabrication
  the guardrail forbids. It is not run here.
- **`parcel_yeo` is further removed** — it is connectivity between parcels, with
  no per-location power at all.

The SPEC's own "heaviest route" anticipated this fallback: LCMV-beamform the
sensor broadband data to fs_LR sources using the supplied head/source models,
then PSD. That route needs the sensor package **`rmegpreproc`** (broadband,
cleaned, sensor-level). **`rmegpreproc` is not present on the server** — a full
scan of `*.zip` returns only the three package types above for every subject.
So even the fallback cannot run on what was downloaded. (This matches the
pre-existing note in `docs/SPEC_fig9_predictions_batch.md` §"Aperiodic": the
exponent "requires raw/source per-subject MEG … a full source-localisation +
PSD + specparam pipeline", flagged as future work, §4.5 / Annex F.)

Additionally, the `cyto7` env does **not** currently have `mne`, `specparam`, or
`fooof` installed (checked): they would need adding to `environment.yml` before
any real run.

---

## Integrity / manifest

`results/hcp_meg_manifest.tsv` (columns: subject, package, file, size_bytes,
megconnectome_version, expected_md5, actual_md5, status), built by
`scripts/manifest_hcp_meg.py`.

- **Expected MD5** for every archive is read from its shipped `.md5` sidecar.
- **Recomputed and compared** for all `anatomy` and all `parcel_yeo` archives
  (small), plus a **dtseries spot-check** of subjects 100307 / 512835 / 990366.
- **Deferred** (expected MD5 recorded, not recomputed) for the remaining
  dtseries archives: re-hashing all 89 (~1.8 TB) was not justified for data that
  Stage 0 shows is unusable for this test. Run
  `python scripts/manifest_hcp_meg.py --full` to force full verification (e.g.
  before any future beamform run).

Checksum tally: **187 pass, 0 FAIL, 86 deferred, 0 missing-md5** across 273
archives (all 95 `anatomy` + all 89 `parcel_yeo` + 3 `dtseries` spot-checks =
187 recomputed and matched; 86 `dtseries` deferred). Coverage: 95 subjects have
`anatomy`, 89 have `dtseries`, 89 have `parcel_yeo`.

If any recomputed checksum had FAILed, it is flagged `FAIL` in the manifest and
the run exits non-zero; the report would name the offending archive rather than
proceed.

---

## Pre-registration statement (retained for the real run)

Kept verbatim so the test is fixed before data exist:

- **Primary:** Spearman ρ of the **group aperiodic exponent** (`specparam`,
  `aperiodic_mode='fixed'`, fit 2–40 Hz) vs ordinal cyto7 v9 type, with the
  released spin machinery (`neuromaps.nulls.alexander_bloch`, fsLR 32k,
  `n_perm=1000`, `seed=0`) and a new FDR family ("dynamics"), separate from the
  published families. **Predicted ρ < 0** (exponent flattens toward
  koniocortex). Allocortex-excluded sensitivity reported alongside.
- **Secondary:** peak/dominant frequency (predicted ρ > 0) and offset.
- Robustness: `'knee'` mode and a broadband 1–100 Hz variant. QC: per-source
  fit R²; exclude R² < 0.9 and report retained fraction.

---

## Honest interpretation

- This is **not** a scientific null. The dynamics hypothesis was not tested; the
  correctly-specified data were not available in the downloaded packages.
- The paper's current dynamics evidence is unchanged: the band-power maps fail
  the spin test, the intrinsic timescale is suggestive but fails FDR, and the
  spectral-centroid / slow-fast-ratio summaries are reported plainly in Annex F.
  Nothing here weakens or strengthens those; §3.6 / Annex F stand as they are.
- A properly-powered result — positive or null — still requires the broadband
  source data.

## What Ricardo needs to do to enable the real test

1. **Download the HCP-MEG `rmegpreproc` package** (sensor-level, broadband,
   cleaned; "MEG Resting State" preprocessed) for the same subjects — ~1–2 GB
   per subject, one at a time onto `X:` scratch. The `anatomy` package (head +
   source models) needed for beamforming is **already present** for 95 subjects.
   *(Alternatively, if HCP distributes a broadband source-level resting product,
   that would let us skip beamforming — but the "dtseries" release is envelopes
   only, as shown above.)*
2. Add `mne` + `specparam` (and `h5py` for the FieldTrip `.mat` I/O) to
   `environment.yml`.
3. Re-run: I then build Stage 1 (LCMV to fs_LR) → Stage 2 (`specparam` 2–40 Hz,
   fixed mode) → Stages 3–5 exactly as pre-registered above, one subject at a
   time, checksum-gated, deleting raw per subject.

## Provenance

- HCP megconnectome pipeline **version 3.0** (per release notes).
- `specparam` version: **not installed** (no fit performed; to be pinned at the
  real run).
- Seeds: none used (no computation beyond checksums + one header read).
- Scripts: `scripts/manifest_hcp_meg.py`. Server scratch cleared; only
  `results/hcp_meg_manifest.tsv` + this report were written into the repo.
