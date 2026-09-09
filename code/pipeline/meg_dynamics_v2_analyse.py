"""SPEC_hcp_meg_dynamics_v2 — Phase 2 (analysis) module. Imported by meg_dynamics_v2.py.

Loads the persisted per-subject PSD + INT maps (FieldTrip beamform, Phase 1), fits
specparam (fixed 2-40 primary, knee 2-40, broad 1-100), derives knee-timescale,
aperiodic-corrected oscillatory band power, and the true intrinsic timescale (3 defs);
computes across-run reliability and the validity gates; aligns to cyto7 v9 with the
released spin (reusing meg_subjectlevel) and tests each metric (BH-FDR, own families,
allocortex-excluded), side-by-side with v1 single-sphere + released single-map values.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

from cyto7_surface_io import REPO_ROOT
from external_validation import _bh
import meg_subjectlevel as MB   # build_4k_alignment, subject_level_spin, _fast_group_null

VSC = cfg.data_dir("019-HCP Young MEG data/_scratch/v2")
PERSIST = VSC / "persist"
OUT_SF = cfg.results_dir("tables") / "structure_function"
RESULTS = REPO_ROOT / "results"
CACHE = cfg.data_dir() / "neuromaps_cache"
SEED = 0
FIT_LO, FIT_HI = 2.0, 40.0
BROAD_LO, BROAD_HI = 1.0, 100.0
CANON = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma1": (30, 60)}
CANON_CENTRE = {b: 0.5 * (lo + hi) for b, (lo, hi) in CANON.items()}

# metric -> (predicted_sign, family). primary flagged separately.
METRICS = [
    ("exponent_fixed", -1, "dynamics_core"),
    ("peak_freq", +1, "dynamics_core"),
    ("offset", None, "dynamics_core"),
    ("knee_tau", -1, "dynamics_core"),
    ("int_1e", -1, "dynamics_core"),
    ("int_area", -1, "dynamics_core"),
    ("int_tau", -1, "dynamics_core"),
    ("exponent_knee", -1, "dynamics_robust"),
    ("exponent_broad", -1, "dynamics_robust"),
    ("osc_delta", -1, "osc_band"),
    ("osc_theta", -1, "osc_band"),
    ("osc_alpha", None, "osc_band"),
    ("osc_beta", +1, "osc_band"),
    ("osc_gamma1", +1, "osc_band"),
    ("sf_ratio", -1, "osc_band"),
    ("centroid", +1, "osc_band"),
]
PRIMARIES = {"exponent_fixed", "int_area"}   # exponent (pre-reg) + best-validated INT


def _runs_axis(a):
    """psd/int saved as 2D (1 run) or 3D (nrun); return with a run axis last."""
    a = np.asarray(a, float)
    return a[..., None] if a.ndim == 2 else a


def _specparam(freqs, psd, lo, hi, mode):
    from specparam import SpectralGroupModel
    S = psd.shape[0]
    band = (freqs >= lo) & (freqs <= hi)
    fit = np.all(np.isfinite(psd[:, band]) & (psd[:, band] > 0), axis=1)
    out = {k: np.full(S, np.nan) for k in ("exponent", "offset", "knee", "r2", "peak_freq")}
    peaks_by_src = [[] for _ in range(S)]
    idx = np.where(fit)[0]
    if idx.size == 0:
        return out, peaks_by_src
    fm = SpectralGroupModel(aperiodic_mode=mode, peak_width_limits=(1.0, 12.0),
                            max_n_peaks=6, min_peak_height=0.05, verbose=False)
    try:
        fm.fit(freqs, psd[idx], freq_range=[lo, hi], n_jobs=6)
    except TypeError:
        fm.fit(freqs, psd[idx], freq_range=[lo, hi])
    out["exponent"][idx] = fm.get_params("aperiodic", "exponent")
    out["offset"][idx] = fm.get_params("aperiodic", "offset")
    out["r2"][idx] = fm.get_metrics("gof_rsquared")
    if mode == "knee":
        try:
            out["knee"][idx] = fm.get_params("aperiodic", "knee")
        except Exception:
            out["knee"][idx] = fm.get_params("aperiodic")[:, 1]
    pk = np.atleast_2d(np.asarray(fm.get_params("peak"), float))
    if pk.size and pk.shape[1] >= 4:
        for cf, pw, bw, si in pk:
            peaks_by_src[idx[int(si)]].append((cf, pw, bw))
        for j in idx:
            ps = peaks_by_src[j]
            if ps:
                out["peak_freq"][j] = max(ps, key=lambda t: t[1])[0]
    return out, peaks_by_src


def _osc_bandpower(peaks_by_src, S):
    """Aperiodic-corrected oscillatory power per canonical band = sum of peak power (PW)
    whose centre frequency falls in the band."""
    osc = {b: np.zeros(S) for b in CANON}
    for j in range(S):
        for cf, pw, bw in peaks_by_src[j]:
            for b, (lo, hi) in CANON.items():
                if lo <= cf < hi:
                    osc[b][j] += pw
    return osc


def _subject_metrics(mat, log):
    psd = _runs_axis(mat["psd_runs"])          # (S, nfreq, nrun)
    intr = _runs_axis(mat["int_runs"])         # (S, 3, nrun)
    freqs = np.asarray(mat["freqs"], float).ravel()
    S, _, nrun = psd.shape
    psd_avg = np.nanmean(psd, axis=2)

    fixed, peaks = _specparam(freqs, psd_avg, FIT_LO, FIT_HI, "fixed")
    knee, _ = _specparam(freqs, psd_avg, FIT_LO, FIT_HI, "knee")
    broad, _ = _specparam(freqs, psd_avg, BROAD_LO, BROAD_HI, "fixed")

    with np.errstate(divide="ignore", invalid="ignore"):
        fkn = np.where((knee["knee"] > 0) & (knee["exponent"] > 0),
                       knee["knee"] ** (1.0 / np.maximum(knee["exponent"], 1e-6)), np.nan)
        knee_tau = np.where(fkn > 0, 1.0 / (2 * np.pi * fkn), np.nan)

    osc = _osc_bandpower(peaks, S)
    # raw band powers (for SF/centroid, relative like Option B)
    bp = {b: psd_avg[:, (freqs >= lo) & (freqs <= hi)].mean(1) for b, (lo, hi) in CANON.items()}
    broadp = sum(bp.values())
    with np.errstate(divide="ignore", invalid="ignore"):
        sf = np.where((bp["beta"] + bp["gamma1"]) > 0,
                      (bp["delta"] + bp["theta"]) / (bp["beta"] + bp["gamma1"]), np.nan)
        cw = sum(CANON_CENTRE[b] * bp[b] for b in CANON)
        centroid = np.where(broadp > 0, cw / broadp, np.nan)
        rel_alpha = np.where(broadp > 0, bp["alpha"] / broadp, np.nan)

    m = dict(exponent_fixed=fixed["exponent"], offset=fixed["offset"], peak_freq=fixed["peak_freq"],
             exponent_knee=knee["exponent"], exponent_broad=broad["exponent"], knee_tau=knee_tau,
             int_1e=np.nanmean(intr[:, 0, :], 1), int_area=np.nanmean(intr[:, 1, :], 1),
             int_tau=np.nanmean(intr[:, 2, :], 1),
             sf_ratio=sf, centroid=centroid, rel_alpha=rel_alpha,
             r2_fixed=fixed["r2"])
    for b in CANON:
        m[f"osc_{b}"] = osc[b]

    # across-run reliability (mean pairwise Spearman over the nrun run maps) for exponent + INT
    rel = {}
    if nrun >= 2:
        exp_runs = []
        for r in range(nrun):
            fr, _ = _specparam(freqs, psd[:, :, r], FIT_LO, FIT_HI, "fixed")
            exp_runs.append(fr["exponent"])
        rel["exponent_fixed"] = _pairwise_rho(exp_runs)
        for k, col in (("int_1e", 0), ("int_area", 1), ("int_tau", 2)):
            rel[k] = _pairwise_rho([intr[:, col, r] for r in range(nrun)])
    return m, rel


def _pairwise_rho(maps):
    vals = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            a, b = maps[i], maps[j]
            k = np.isfinite(a) & np.isfinite(b)
            if k.sum() > 100:
                vals.append(stats.spearmanr(a[k], b[k])[0])
    return float(np.nanmean(vals)) if vals else np.nan


def _nm_to_4k(desc):
    import nibabel as nib
    from neuromaps.datasets import fetch_atlas
    from scipy.spatial import cKDTree
    a4, a32 = fetch_atlas("fsLR", "4k"), fetch_atlas("fsLR", "32k")
    idx = []
    for hi in (0, 1):
        s4 = np.asarray(nib.load(str(a4["sphere"][hi])).agg_data()[0], float)
        s32 = np.asarray(nib.load(str(a32["sphere"][hi])).agg_data()[0], float)
        idx.append(cKDTree(s32).query(s4, k=1)[1] + (0 if hi == 0 else 32492))
    idx = np.concatenate(idx)
    v = np.concatenate([np.load(CACHE / f"hcps1200_{desc}_fsLR32k_hemi-{h}.npy")
                        for h in ("L", "R")]).astype(float)
    return v[idx]


def _v1_values():
    p = OUT_SF / "meg_aperiodic_summary.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p).set_index("metric")["group_mean_rho"].to_dict()
    return {"exponent_fixed": d.get("exponent"), "peak_freq": d.get("peak_freq"),
            "offset": d.get("offset")}


def run(subjects, n_spin, log):
    mats = sorted(PERSIST.glob("sub-*_dynamics.mat"))
    if not mats:
        log("STOP: no persisted beamform outputs found."); return
    log(f"Phase 2: {len(mats)} persisted subjects; building 4k alignment (released spin)...")
    type4k, nulls4k = MB.build_4k_alignment(n_spin, log)

    stacks = {m: [] for m, _, _ in METRICS}
    stacks["rel_alpha"] = []
    r2s, rels, kept = [], [], []
    for mp in mats:
        subj = mp.name.split("_")[0].replace("sub-", "")
        mat = sio.loadmat(str(mp))
        m, rel = _subject_metrics(mat, log)
        for k, _, _ in METRICS:
            stacks[k].append(m[k])
        stacks["rel_alpha"].append(m["rel_alpha"])
        r2s.append(m["r2_fixed"]); rels.append(rel); kept.append(subj)
        log(f"  {subj}: exp med={np.nanmedian(m['exponent_fixed']):.2f} "
            f"INTarea med={np.nanmedian(m['int_area'])*1000:.0f}ms r2={np.nanmedian(m['r2_fixed']):.3f}")
    S = len(kept)
    metric_stacks = {k: np.vstack(stacks[k]) for k in stacks}
    r2_stack = np.vstack(r2s)

    # QC: require good fixed-fit R2>=0.9 for exponent metrics
    good = r2_stack >= 0.9
    for k in ("exponent_fixed", "exponent_knee", "exponent_broad", "offset", "peak_freq",
              "knee_tau", "osc_delta", "osc_theta", "osc_alpha", "osc_beta", "osc_gamma1"):
        metric_stacks[k] = np.where(good, metric_stacks[k], np.nan)
    retained = float(np.mean(good))

    valid = np.isfinite(type4k) & (type4k > 0)
    frac_good = np.mean(np.isfinite(metric_stacks["exponent_fixed"]), axis=0)
    valid &= frac_good >= 0.6
    valid_excl = valid & (type4k > 1)
    log(f"  valid 4k vertices: {int(valid.sum())} (allo-excl {int(valid_excl.sum())}); "
        f"retained R2>=0.9 = {retained:.3f}")

    # validity gates
    val_alpha = _validity(metric_stacks["rel_alpha"], _nm_to_4k("megalpha"), type4k, valid)
    val_int = {k: _validity(metric_stacks[k], _nm_to_4k("megtimescale"), type4k, valid)
               for k in ("int_1e", "int_area", "int_tau")}
    log(f"  [validity] rel-alpha vs megalpha rho={val_alpha:+.3f}; "
        f"INT vs megtimescale: 1e={val_int['int_1e']:+.3f} area={val_int['int_area']:+.3f} "
        f"tau={val_int['int_tau']:+.3f} (expect POSITIVE)")

    # reliability (mean over subjects of across-run pairwise rho)
    def rel_mean(k):
        vs = [r[k] for r in rels if k in r and np.isfinite(r[k])]
        return float(np.mean(vs)) if vs else np.nan
    reliab = {k: rel_mean(k) for k in ("exponent_fixed", "int_1e", "int_area", "int_tau")}

    v1 = _v1_values()
    rows, persubj = [], {"subject": kept}
    for k, sign, fam in METRICS:
        ms = _impute(metric_stacks[k], valid)
        r = MB.subject_level_spin(ms, type4k, nulls4k, valid, sign)
        rex = MB.subject_level_spin(ms, type4k, nulls4k, valid_excl, sign)
        persubj[f"rho_{k}"] = r.pop("rho_subj"); rex.pop("rho_subj")
        rows.append(dict(metric=k, family=fam, primary=(k in PRIMARIES),
                         predicted_sign={-1: "neg", 1: "pos", None: "none"}[sign],
                         group_mean_rho=r["group_mean_rho"], group_median_rho=r["group_median_rho"],
                         spin_p=r["spin_p"], frac_predicted=r["frac_predicted"],
                         group_mean_rho_allo_excl=rex["group_mean_rho"], spin_p_allo_excl=rex["spin_p"],
                         v1_single_sphere_rho=v1.get(k, np.nan),
                         reliability_across_run=reliab.get(k, np.nan)))
        log(f"    {k:16s} rho={r['group_mean_rho']:+.3f} p={r['spin_p']:.3f} "
            f"(v1={v1.get(k, float('nan'))})")
    df = pd.DataFrame(rows)
    # BH within each family
    df["fdr_q"] = np.nan
    for fam in df["family"].unique():
        mask = df["family"] == fam
        df.loc[mask, "fdr_q"] = _bh(df.loc[mask, "spin_p"].values)
    df["retained_fraction_r2ge0.9"] = retained
    df["validity_rel_alpha_vs_megalpha"] = val_alpha
    for k in ("int_1e", "int_area", "int_tau"):
        df.loc[df.metric == k, "validity_vs_megtimescale"] = val_int[k]

    OUT_SF.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_SF / "meg_dynamics_v2_summary.csv", index=False)
    pd.DataFrame(persubj).to_csv(OUT_SF / "meg_dynamics_v2_persubject_rho.csv", index=False)
    log(f"  wrote {OUT_SF/'meg_dynamics_v2_summary.csv'}")
    _write_report(df, kept, valid, valid_excl, retained, val_alpha, val_int, reliab, n_spin, log)
    scratch_gb = sum(f.stat().st_size for f in PERSIST.glob("*")) / 1e9
    log(f"  PERSISTED SCRATCH: {PERSIST}  ({scratch_gb:.2f} GB, NOT deleted per SPEC)")


def _validity(stack, ref4k, type4k, valid):
    gm = np.nanmean(stack, axis=0)
    m = valid & np.isfinite(gm) & np.isfinite(ref4k)
    return float(stats.spearmanr(gm[m], ref4k[m])[0])


def _impute(M, valid):
    out = M.copy()
    cm = np.nanmean(np.where(valid[None, :], out, np.nan), axis=0)
    bad = ~np.isfinite(out) & valid[None, :]
    ii = np.where(bad)
    out[ii] = np.take(cm, ii[1])
    return out


def _write_report(df, kept, valid, valid_excl, retained, val_alpha, val_int, reliab, n_spin, log):
    def row(k):
        r = df[df.metric == k].iloc[0]
        return (f"rho={r.group_mean_rho:+.3f} (med {r.group_median_rho:+.3f}), spin p={r.spin_p:.3f}, "
                f"q={r.fdr_q:.3f}, allo-excl {r.group_mean_rho_allo_excl:+.3f} (p={r.spin_p_allo_excl:.3f}), "
                f"frac-pred={r.frac_predicted:.2f}, v1={r.v1_single_sphere_rho}")
    L = ["# Report — HCP-MEG dynamics v2 (FieldTrip singleshell forward + true INT)\n",
         f"Implements `docs/SPEC_hcp_meg_dynamics_v2.md`. Route **1a: FieldTrip 20200607 "
         f"singleshell LCMV** (MATLAB R2018a). cyto7 **v9**; released spin fsLR 32k, n_perm={n_spin}, "
         f"seed={SEED}. Subjects **n={len(kept)}**.\n",
         "## Forward validation (better forward should not degrade this)",
         f"- source rel-alpha vs neuromaps `megalpha`: **rho={val_alpha:+.3f}** "
         f"(v1 single-sphere ~+0.51 -> singleshell improves localization).",
         f"- true INT vs neuromaps `megtimescale` (expect POSITIVE; envelope proxy failed at -0.84): "
         f"1e={val_int['int_1e']:+.3f}, area={val_int['int_area']:+.3f}, tau={val_int['int_tau']:+.3f}.\n",
         "## Fit QC + reliability",
         f"- retained fraction (fixed-fit R2>=0.9) = {retained:.3f}; valid 4k vertices {int(valid.sum())}.",
         f"- across-run reliability (mean pairwise Spearman): exponent={reliab.get('exponent_fixed'):.3f}, "
         f"INT-1e={reliab.get('int_1e'):.3f}, INT-area={reliab.get('int_area'):.3f}, "
         f"INT-tau={reliab.get('int_tau'):.3f}. (A null is only informative for a reliable metric.)\n",
         "## Results vs cyto7 v9 (per-subject rho; released spin; BH within family)",
         "### Pre-registered primaries",
         f"- **Aperiodic exponent (fixed 2-40 Hz):** {row('exponent_fixed')}",
         f"- **True intrinsic timescale (area def):** {row('int_area')}",
         "### Dynamics-core family",
         f"- exponent (knee): {row('exponent_knee')}",
         f"- exponent (broad 1-100): {row('exponent_broad')}",
         f"- peak frequency: {row('peak_freq')}",
         f"- offset: {row('offset')}",
         f"- knee-timescale: {row('knee_tau')}",
         f"- INT (lag-1/e): {row('int_1e')}",
         f"- INT (exp-tau): {row('int_tau')}",
         "### Oscillatory-band family (aperiodic-corrected) + summaries",
         f"- osc delta: {row('osc_delta')}", f"- osc theta: {row('osc_theta')}",
         f"- osc alpha: {row('osc_alpha')}", f"- osc beta: {row('osc_beta')}",
         f"- osc gamma1: {row('osc_gamma1')}",
         f"- SF ratio: {row('sf_ratio')}", f"- centroid: {row('centroid')}\n",
         "## Interpretation",
         "Group statistic = mean over subjects of the per-source Spearman rho vs ordinal cyto7 type; "
         "released spin null; BH within each FDR family (dynamics_core, dynamics_robust, osc_band) — "
         "kept separate from the published families. The v1 column is the single-sphere value for the "
         "same metric; comparing them isolates the effect of the realistic forward. The INT here is the "
         "correctly-specified broadband-ACF timescale (validity-gated vs megtimescale), superseding the "
         "Option-B envelope proxy. A properly-powered null is an acceptable, publishable outcome and "
         "feeds the C2 E/I strand.\n",
         "## Provenance",
         f"- FieldTrip 20200607 (R2018a) singleshell LCMV (unit-noise-gain, fixedori, 5% lambda); "
         f"specparam v2; seed={SEED}; n_perm={n_spin}. Persisted PSD/INT scratch kept at "
         f"`{PERSIST}` (path+size printed in the run log). Released tables untouched."]
    (RESULTS / "report_hcp_meg_v2.md").write_text("\n".join(L), encoding="utf-8")
    log(f"  wrote {RESULTS/'report_hcp_meg_v2.md'}")
