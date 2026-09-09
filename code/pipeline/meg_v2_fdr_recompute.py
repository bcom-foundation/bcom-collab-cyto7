#!/usr/bin/env python
"""SPEC_meg_v2_fdr_recompute — dynamics FDR with one member per construct.

Part A (required): recompute Benjamini-Hochberg q over the committed dynamics
family (one member per construct: intrinsic timescale = ACF-area, spectral
centroid, slow/fast ratio, aperiodic-corrected beta, aperiodic exponent = fixed
mode, peak frequency, offset). The timescale/exponent robustness variants
(1/e-lag, exp-decay, knee-derived timescale; knee/broadband exponent) report
their raw spin-p but are EXCLUDED from the family (they must not consume FDR dof),
matching §2.6. Confirms verdicts are unchanged; stop-and-log on any flip. Writes
`meg_dynamics_v2_summary_fdr_singlemember.csv` (the original CSV is not modified).

Part B (optional): v2 source-reconstructed per-type (7 cyto7 types) hemisphere-
averaged medians of the intrinsic timescale (ms), spectral centroid, and
aperiodic-corrected beta from the persisted group-mean 32k maps (Table S3 refresh).
`functional_summary_table_v9.csv` is NOT overwritten.

Run:  conda run -n cyto7 python scripts/meg_v2_fdr_recompute.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from pathlib import Path

import numpy as np
import pandas as pd

from cyto7_surface_io import REPO_ROOT, LABEL_NAMES
from external_validation import _bh  # identical BH-FDR used by the released tables

SF = cfg.results_dir("tables") / "structure_function"
SUMMARY = SF / "meg_dynamics_v2_summary.csv"
MAPS = SF / "meg_v2_maps"
LABCACHE = cfg.scratch_dir("resample_cache")
OUT_CSV = SF / "meg_dynamics_v2_summary_fdr_singlemember.csv"

# committed family: one member per construct
FAMILY = ["int_area", "centroid", "sf_ratio", "osc_beta", "exponent_fixed", "peak_freq", "offset"]
FAMILY_LABEL = {
    "int_area": "Intrinsic timescale (ACF-area)", "centroid": "Spectral centroid",
    "sf_ratio": "Slow/fast ratio", "osc_beta": "Aperiodic-corrected beta power",
    "exponent_fixed": "Aperiodic exponent (fixed)", "peak_freq": "Peak frequency", "offset": "Offset"}
# robustness variants: raw p reported, EXCLUDED from the family
ROBUSTNESS = {
    "int_1e": "Intrinsic timescale (1/e-lag) [robustness]",
    "int_tau": "Intrinsic timescale (exp-decay) [robustness]",
    "knee_tau": "Intrinsic timescale (knee-derived) [robustness]",
    "exponent_knee": "Aperiodic exponent (knee mode) [robustness]",
    "exponent_broad": "Aperiodic exponent (broadband 1-100) [robustness]"}
TS_VARIANTS = ["int_1e", "int_tau", "knee_tau"]  # must agree in direction (rho<0) at raw p<0.05
EXPECT_SURVIVE = {"int_area", "centroid", "sf_ratio", "osc_beta"}
EXPECT_NULL = {"exponent_fixed", "peak_freq", "offset"}


def part_a(log):
    df = pd.read_csv(SUMMARY).set_index("metric")
    fam = df.loc[FAMILY, ["group_mean_rho", "spin_p", "predicted_sign", "family"]].copy()
    fam["fdr_q_singlemember"] = _bh(fam["spin_p"].values)          # BH over exactly 7 members
    fam["verdict"] = np.where(fam["fdr_q_singlemember"] < 0.05, "survives", "null")

    log("== Part A — single-member dynamics family (BH over 7 committed members) ==")
    for m in FAMILY:
        r = fam.loc[m]
        log(f"  {FAMILY_LABEL[m]:34s} rho={r.group_mean_rho:+.3f}  spin_p={r.spin_p:.3f}  "
            f"q_single={r.fdr_q_singlemember:.3f}  -> {r.verdict}")

    # verdict-flip guard
    surv = set(fam.index[fam.verdict == "survives"])
    nul = set(fam.index[fam.verdict == "null"])
    flips = (EXPECT_SURVIVE - surv) | (EXPECT_NULL - nul)
    if flips:
        log(f"  [STOP-AND-LOG] verdict FLIP for {sorted(flips)} — NOT writing; investigate before "
            f"changing any manuscript claim.")
        return None
    log("  verdicts UNCHANGED: timescale / centroid / slow-fast / aperiodic-corrected-beta survive "
        "(q<0.05); exponent / peak-freq / offset null.")

    # robustness variants: raw p + direction agreement
    log("\n  Robustness variants (raw spin-p; excluded from the family):")
    for m, lab in ROBUSTNESS.items():
        r = df.loc[m]
        log(f"    {lab:52s} rho={r.group_mean_rho:+.3f}  spin_p={r.spin_p:.3f}")
    tv = df.loc[TS_VARIANTS]
    agree = bool((tv["group_mean_rho"] < 0).all() and (tv["spin_p"] < 0.05).all())
    log(f"  Timescale robustness variants agree in direction (rho<0) at raw p<0.05: "
        f"{'YES' if agree else 'NO'} "
        f"[1/e {df.loc['int_1e','group_mean_rho']:+.3f} p={df.loc['int_1e','spin_p']:.3f}; "
        f"exp-decay {df.loc['int_tau','group_mean_rho']:+.3f} p={df.loc['int_tau','spin_p']:.3f}; "
        f"knee {df.loc['knee_tau','group_mean_rho']:+.3f} p={df.loc['knee_tau','spin_p']:.3f}]")

    # assemble output CSV: family (with new q) + robustness (q=NaN, flagged)
    rows = []
    for m in FAMILY:
        r = fam.loc[m]
        rows.append(dict(metric=m, construct=FAMILY_LABEL[m], role="family_member",
                         group_mean_rho=round(float(r.group_mean_rho), 4),
                         spin_p=round(float(r.spin_p), 4),
                         fdr_q_singlemember=round(float(r.fdr_q_singlemember), 4),
                         verdict=r.verdict))
    for m, lab in ROBUSTNESS.items():
        r = df.loc[m]
        rows.append(dict(metric=m, construct=lab, role="robustness_variant_excluded",
                         group_mean_rho=round(float(r.group_mean_rho), 4),
                         spin_p=round(float(r.spin_p), 4),
                         fdr_q_singlemember=np.nan, verdict="raw p only"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    log(f"\n  wrote {OUT_CSV}")
    log(f"  KEY: ACF-area intrinsic-timescale q (single-member family) = "
        f"{fam.loc['int_area','fdr_q_singlemember']:.3f} "
        f"(was {df.loc['int_area','fdr_q']:.3f} under the over-split families).")
    return fam


def part_b(log):
    labs = {H: np.load(LABCACHE / f"v9_labels_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}
    maps = {"Intrinsic timescale (ms)": "int_area_ms", "Spectral centroid (Hz)": "centroid",
            "Aperiodic-corrected beta (a.u.)": "osc_beta"}
    log("\n== Part B — v2 per-type hemisphere-averaged medians (Table S3 refresh) ==")
    header = "  type          " + "".join(f"{k:>34s}" for k in maps)
    log(header)
    rows = []
    for t in range(1, 8):
        cells = {}
        for name, key in maps.items():
            d = {H: np.load(MAPS / f"{key}_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}
            meds = []
            for H in ("L", "R"):
                m = (labs[H] == t) & np.isfinite(d[H])
                meds.append(np.median(d[H][m]) if m.any() else np.nan)
            cells[name] = float(np.nanmean(meds))
        rows.append(dict(type=LABEL_NAMES[t - 1], **cells))
        log("  " + f"{LABEL_NAMES[t-1]:14s}" + "".join(f"{cells[k]:>34.3f}" for k in maps))
    return pd.DataFrame(rows)


def main():
    log = print
    fam = part_a(log)
    if fam is None:
        raise SystemExit(3)
    part_b(log)
    log("\nDone. (report_hcp_meg_v2.md / functional_summary_table_v9.csv NOT modified.)")


if __name__ == "__main__":
    main()
