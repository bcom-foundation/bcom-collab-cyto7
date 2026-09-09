"""RR5 - is the type/timescale relation independent of the aperiodic slope and band power?

Adds a partial-correlation variant of ``meg_subjectlevel.subject_level_spin``:
per subject, the Spearman partial correlation between a metric and ordinal cyto7
type at vertex level, controlling for subject-specific spectral covariates; then
the same group mean over subjects and the same hierarchical spin null (only the
type map rotates, the covariates never do).

Reuses the persisted per-subject beamformer PSD/INT maps (phase 1 of
SPEC_hcp_meg_dynamics_v2) and the specparam code in
``meg_dynamics_v2_analyse``; it does not re-derive spectra or re-run the
beamformer. The per-subject metric stacks are cached so re-runs are cheap.

Run::
    conda run -n cyto7 python scripts/rr5_meg_controls.py --n-spin 1000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

import meg_dynamics_v2_analyse as MD
import meg_subjectlevel as MB
import rr_common as rc

OUTDIR = rc.OUT / "rr5_meg_controls"
CACHEDIR = rc.OUT / "_cache"
STACK_CACHE = CACHEDIR / "rr5_metric_stacks.npz"

BANDS = [f"osc_{b}" for b in ("delta", "theta", "alpha", "beta", "gamma1")]
NEEDED = ["int_area", "int_1e", "int_tau", "exponent_fixed", "offset", "peak_freq",
          "centroid", "sf_ratio", "rel_alpha", "r2_fixed"] + BANDS

# Published values (figures/v9/structure_function/meg_dynamics_v2_summary.csv).
PUBLISHED = {"int_area": {"rho": -0.459219, "spin_p": 0.003996, "q": 0.027972,
                          "frac_predicted": 1.0, "median": -0.471195},
             "centroid": {"rho": 0.408791, "spin_p": 0.003996},
             "sf_ratio": {"rho": -0.356376, "spin_p": 0.008991},
             "exponent_fixed": {"rho": -0.119234, "spin_p": 0.498501}}


# --------------------------------------------------------------------------- #
# Per-subject metric stacks (cached)
# --------------------------------------------------------------------------- #


def _subject_metrics_fixed_only(mat):
    """The subset of ``MD._subject_metrics`` that RR5 needs.

    Only the fixed 2-40 Hz specparam fit is run (the knee and broad fits and the
    per-run reliability refits are not used by any RR5 model), so this is about
    four times cheaper per subject and returns numerically identical values for
    the metrics it does emit.
    """
    psd = MD._runs_axis(mat["psd_runs"])
    intr = MD._runs_axis(mat["int_runs"])
    freqs = np.asarray(mat["freqs"], float).ravel()
    S = psd.shape[0]
    psd_avg = np.nanmean(psd, axis=2)

    fixed, peaks = MD._specparam(freqs, psd_avg, MD.FIT_LO, MD.FIT_HI, "fixed")
    osc = MD._osc_bandpower(peaks, S)
    bp = {b: psd_avg[:, (freqs >= lo) & (freqs <= hi)].mean(1) for b, (lo, hi) in MD.CANON.items()}
    broadp = sum(bp.values())
    with np.errstate(divide="ignore", invalid="ignore"):
        sf = np.where((bp["beta"] + bp["gamma1"]) > 0,
                      (bp["delta"] + bp["theta"]) / (bp["beta"] + bp["gamma1"]), np.nan)
        cw = sum(MD.CANON_CENTRE[b] * bp[b] for b in MD.CANON)
        centroid = np.where(broadp > 0, cw / broadp, np.nan)
        rel_alpha = np.where(broadp > 0, bp["alpha"] / broadp, np.nan)
    m = dict(exponent_fixed=fixed["exponent"], offset=fixed["offset"],
             peak_freq=fixed["peak_freq"], r2_fixed=fixed["r2"],
             int_1e=np.nanmean(intr[:, 0, :], 1), int_area=np.nanmean(intr[:, 1, :], 1),
             int_tau=np.nanmean(intr[:, 2, :], 1),
             sf_ratio=sf, centroid=centroid, rel_alpha=rel_alpha)
    for b in MD.CANON:
        m[f"osc_{b}"] = osc[b]
    return m


def build_stacks(log=print) -> tuple[dict[str, np.ndarray], list[str]]:
    if STACK_CACHE.exists():
        z = np.load(STACK_CACHE, allow_pickle=True)
        subs = list(z["subjects"])
        log(f"  reusing cached metric stacks {STACK_CACHE.name} ({len(subs)} subjects)")
        return {k: z[k] for k in NEEDED}, subs
    mats = sorted(MD.PERSIST.glob("sub-*_dynamics.mat"))
    if not mats:
        raise RuntimeError(f"no persisted beamform outputs in {MD.PERSIST}")
    log(f"  building metric stacks from {len(mats)} persisted subjects (fixed fit only)...")
    acc = {k: [] for k in NEEDED}
    subs = []
    for i, mp in enumerate(mats, 1):
        subj = mp.name.split("_")[0].replace("sub-", "")
        m = _subject_metrics_fixed_only(sio.loadmat(str(mp)))
        for k in NEEDED:
            acc[k].append(m[k])
        subs.append(subj)
        log(f"    [{i}/{len(mats)}] {subj}: INTarea med={np.nanmedian(m['int_area'])*1000:.0f}ms "
            f"exp med={np.nanmedian(m['exponent_fixed']):.2f} r2={np.nanmedian(m['r2_fixed']):.3f}")
    stacks = {k: np.vstack(acc[k]) for k in NEEDED}
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STACK_CACHE, subjects=np.array(subs), **stacks)
    log(f"  cached {STACK_CACHE.name}")
    return stacks, subs


def prepare(stacks, type4k, log=print):
    """Apply the published QC masking, validity gate and imputation."""
    good = stacks["r2_fixed"] >= 0.9
    out = {}
    for k in NEEDED:
        if k == "r2_fixed":
            continue
        v = stacks[k]
        if k in ("exponent_fixed", "offset", "peak_freq") or k in BANDS:
            v = np.where(good, v, np.nan)
        out[k] = v
    valid = np.isfinite(type4k) & (type4k > 0)
    frac_good = np.mean(np.isfinite(out["exponent_fixed"]), axis=0)
    valid &= frac_good >= 0.6
    log(f"  retained R2>=0.9 = {float(np.mean(good)):.3f}; valid 4k vertices {int(valid.sum())}")
    imputed = {k: MD._impute(v, valid) for k, v in out.items()}
    return imputed, valid


# --------------------------------------------------------------------------- #
# Partial-correlation group statistic + hierarchical spin null
# --------------------------------------------------------------------------- #


def _ranks(a: np.ndarray) -> np.ndarray:
    """Row-wise rank transform, mean-centred."""
    r = stats.rankdata(a, axis=-1).astype(np.float64)
    return r - r.mean(axis=-1, keepdims=True)


def _partial_rho_subjects(metric: np.ndarray, type_vec: np.ndarray,
                          covs: np.ndarray | None) -> np.ndarray:
    """Per-subject Spearman partial correlation of metric and type given covs.

    metric: (S, n); type_vec: (n,); covs: (S, n, k) or None. Ranks are taken
    within subject, both metric and type are residualised on the covariates, and
    the Pearson correlation of the residuals is returned per subject.
    """
    S, n = metric.shape
    ym = _ranks(metric)                                   # (S, n)
    yt = np.broadcast_to(_ranks(type_vec), (S, n)).copy()  # (S, n)
    if covs is not None and covs.shape[-1] > 0:
        X = np.concatenate([np.ones((S, n, 1)), _ranks(np.moveaxis(covs, -1, 1)).transpose(0, 2, 1)],
                           axis=2)                        # (S, n, k+1)
        G = np.einsum("snk,snl->skl", X, X)
        Bm = np.einsum("snk,sn->sk", X, ym)
        Bt = np.einsum("snk,sn->sk", X, yt)
        # The pseudo-inverse of the Gram matrix, not a solve: two of the five
        # aperiodic-corrected band powers (delta and gamma) are constant-zero for
        # most subjects, because specparam is fitted over 2-40 Hz and finds almost
        # no peaks at the edges of that range. X is therefore rank-deficient within
        # subject, and X pinv(X'X) X' is still exactly the orthogonal projector onto
        # the column space, so the residuals are the same as they would be if the
        # degenerate columns had been dropped.
        Gi = np.linalg.pinv(G)
        cm = np.einsum("skl,sl->sk", Gi, Bm)
        ct = np.einsum("skl,sl->sk", Gi, Bt)
        ym = ym - np.einsum("snk,sk->sn", X, cm)
        yt = yt - np.einsum("snk,sk->sn", X, ct)
    ym = ym - ym.mean(axis=1, keepdims=True)
    yt = yt - yt.mean(axis=1, keepdims=True)
    num = np.einsum("sn,sn->s", ym, yt)
    den = np.sqrt(np.einsum("sn,sn->s", ym, ym) * np.einsum("sn,sn->s", yt, yt))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def subject_level_spin_partial(metric_stack, type4k, covariate_stacks, nulls4k, valid,
                               sign, log=print) -> dict:
    """Group-mean per-subject partial rho + the hierarchical spin null.

    Only the type map rotates. The covariates are subject-specific data maps and
    stay fixed, exactly as the metric does. Plus-one corrected two-sided tail.
    """
    covs = (np.stack([c[:, valid] for c in covariate_stacks], axis=-1)
            if covariate_stacks else None)
    rho_subj = _partial_rho_subjects(metric_stack[:, valid], type4k[valid], covs)
    gmean = float(np.nanmean(rho_subj))
    gmed = float(np.nanmedian(rho_subj))
    n_spin = nulls4k.shape[1]
    gnull = np.empty(n_spin)
    for i in range(n_spin):
        col = nulls4k[:, i]
        base = valid & np.isfinite(col)
        cv = (np.stack([c[:, base] for c in covariate_stacks], axis=-1)
              if covariate_stacks else None)
        gnull[i] = float(np.nanmean(_partial_rho_subjects(metric_stack[:, base], col[base], cv)))
        if (i + 1) % 200 == 0:
            log(f"      spin {i + 1}/{n_spin}")
    gn = gnull[np.isfinite(gnull)]
    p = float((np.sum(np.abs(gn) >= abs(gmean)) + 1) / (gn.size + 1)) if gn.size else np.nan
    fin = np.isfinite(rho_subj)
    frac = float(np.mean(np.sign(rho_subj[fin]) == sign)) if sign is not None else np.nan
    return dict(group_mean_partial_rho=gmean, group_median=gmed, spin_p=p,
                frac_predicted_sign=frac, n_subj=int(fin.sum()),
                null_mean=float(np.mean(gn)) if gn.size else np.nan,
                rho_subj=rho_subj)


# --------------------------------------------------------------------------- #
# Collinearity diagnostics
# --------------------------------------------------------------------------- #


def collinearity(stacks, valid, keys):
    """Mean-across-subjects vertex-level Spearman correlation matrix."""
    S = stacks[keys[0]].shape[0]
    acc = np.zeros((len(keys), len(keys)))
    for s in range(S):
        M = np.vstack([stacks[k][s, valid] for k in keys])
        acc += stats.spearmanr(M, axis=1)[0]
    return pd.DataFrame(acc / S, index=keys, columns=keys)


def vifs(stacks, valid, cov_keys):
    """Variance inflation for the rank-transformed design, averaged over the
    subjects in which the covariate actually varies.

    A covariate that is constant within a subject has no variance to inflate, and
    including such subjects returns meaningless values of order 1e11. Those
    subjects are counted and excluded per covariate instead.
    """
    S = stacks[cov_keys[0]].shape[0]
    acc = np.zeros(len(cov_keys))
    used = np.zeros(len(cov_keys), int)
    for s in range(S):
        cols = [stats.rankdata(stacks[k][s, valid]) for k in cov_keys]
        R = np.vstack(cols).T.astype(float)
        R = R - R.mean(0)
        sd = R.std(0)
        for j in range(len(cov_keys)):
            if sd[j] <= 0:
                continue
            keep = [i for i in range(len(cov_keys)) if i != j and sd[i] > 0]
            y = R[:, j]
            X = np.c_[np.ones(len(R)), R[:, keep]] if keep else np.ones((len(R), 1))
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            r2 = 1 - resid.var() / y.var()
            acc[j] += 1.0 / max(1e-12, 1 - r2)
            used[j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        vif = np.where(used > 0, acc / np.maximum(used, 1), np.nan)
    return pd.DataFrame({"vif_M3": vif, "n_subjects_with_variance": used}, index=cov_keys)


def degeneracy(stacks, valid, keys) -> pd.DataFrame:
    """Per-metric diagnostic: is the map constant within subject, and how often?"""
    rows = []
    for k in keys:
        v = stacks[k][:, valid]
        sd = np.nanstd(v, axis=1)
        rows.append({"metric": k,
                     "frac_subject_vertices_nonzero": round(float(np.mean(v != 0)), 4),
                     "n_subjects_constant": int(np.sum(~(sd > 0))),
                     "n_subjects": int(v.shape[0]),
                     "median_within_subject_sd": float(np.nanmedian(sd))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=1000)
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frozen_before = rc.frozen_hashes()

    stacks_raw, subs = build_stacks()
    print(f"  subjects: {len(subs)}")
    type4k, nulls4k = MB.build_4k_alignment(args.n_spin, print)
    stacks, valid = prepare(stacks_raw, type4k)

    models = [
        ("int_area", "M0", [], -1),
        ("int_area", "M1", ["exponent_fixed"], -1),
        ("int_area", "M2", BANDS, -1),
        ("int_area", "M3", ["exponent_fixed"] + BANDS, -1),
        ("centroid", "M0", [], +1),
        ("centroid", "M3", ["exponent_fixed"] + BANDS, +1),
        ("sf_ratio", "M0", [], -1),
        ("sf_ratio", "M3", ["exponent_fixed"] + BANDS, -1),
        ("exponent_fixed", "M0", [], -1),
        ("exponent_fixed", "R1", ["int_area"], -1),
        ("exponent_fixed", "R2", ["int_area"] + BANDS, -1),
    ]

    rows = []
    persubj = {"subject": subs}
    for outcome, name, covs, sign in models:
        print(f"  {outcome} {name}: covariates {covs or 'none'}")
        r = subject_level_spin_partial(stacks[outcome], type4k,
                                       [stacks[c] for c in covs], nulls4k, valid, sign)
        persubj[f"{outcome}_{name}"] = r.pop("rho_subj")
        rows.append(dict(outcome=outcome, model=name,
                         covariates=("none" if not covs else "+".join(covs)),
                         n_covariates=len(covs), **r))
        print(f"    partial rho={r['group_mean_partial_rho']:+.4f} (median "
              f"{r['group_median']:+.4f}) spin p={r['spin_p']:.4f} "
              f"frac sign={r['frac_predicted_sign']:.3f} null mean={r['null_mean']:+.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "partial_models.csv", index=False)
    pd.DataFrame(persubj).to_csv(OUTDIR / "partial_persubject_rho.csv", index=False)

    ck = ["int_area", "exponent_fixed", "centroid", "sf_ratio"] + BANDS
    cm = collinearity(stacks, valid, ck)
    cm.round(4).to_csv(OUTDIR / "collinearity_matrix.csv")
    v = vifs(stacks, valid, ["exponent_fixed"] + BANDS)
    v.round(4).to_csv(OUTDIR / "vif_m3.csv")
    deg = degeneracy(stacks, valid, ck)
    deg.to_csv(OUTDIR / "degeneracy_diagnostics.csv", index=False)
    print("\ncollinearity (mean across subjects, Spearman):")
    print(cm.round(3).to_string())
    print("\nM3 VIFs:"); print(v.round(3).to_string())
    print("\ndegeneracy diagnostics:"); print(deg.to_string(index=False))

    m0 = df[(df.outcome == "int_area") & (df.model == "M0")].iloc[0]
    repro = {"int_area_M0": {"published_rho": PUBLISHED["int_area"]["rho"],
                             "recomputed_rho": float(m0.group_mean_partial_rho),
                             "published_spin_p": PUBLISHED["int_area"]["spin_p"],
                             "recomputed_spin_p": float(m0.spin_p),
                             "published_frac_predicted": PUBLISHED["int_area"]["frac_predicted"],
                             "recomputed_frac_predicted": float(m0.frac_predicted_sign),
                             "n_subj": int(m0.n_subj)}}
    ok, detail = rc.check_frozen(frozen_before)
    (OUTDIR / "reproduction_check.json").write_text(
        json.dumps({"reproduction": repro, "frozen_unchanged": ok, "frozen": detail,
                    "n_spin": args.n_spin, "n_subjects": len(subs),
                    "n_valid_vertices": int(valid.sum())}, indent=2), encoding="utf-8")
    print("\nreproduction:", json.dumps(repro, indent=2))
    print("frozen files unchanged:", ok)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
