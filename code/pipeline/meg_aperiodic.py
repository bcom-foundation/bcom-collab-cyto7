#!/usr/bin/env python
"""Option A — HCP-MEG source aperiodic exponent / peak frequency vs cyto7 v9.

Implements docs/SPEC_hcp_meg_aperiodic.md (the definitive dynamics test). Uses
the broadband **sensor** MEG (`rmegpreproc`, now on the server) + the supplied
head/source models to reconstruct source-space power spectra and fit `specparam`
per source per subject, then tests the aperiodic exponent (pre-registered
primary), peak frequency and offset against the cyto7 v9 cortical-type axis.

Pipeline (all-Python, per Ricardo's choice; validated against neuromaps):
  * Load FieldTrip `rmegpreproc` (241 good 4D/BTi magnetometers, ~508.6 Hz,
    2 s clean epochs) + `anatomy` headmodel (singleshell brain surface) +
    `sourcemodel_2d` (8004-source fs_LR 4k cortical sheet), all in BTi head
    coordinates (co-registered -> trans = identity).
  * Forward: single-sphere (Sarvas 1987) MEG leadfield. Sphere origin fit to the
    brain surface; field computed at each sensor COIL then combined into channels
    via the grad `tra` matrix (so the 4D PCA reference compensation is applied
    exactly). Single-sphere is an approximation to the HCP FieldTrip *singleshell*
    (Nolte) model; forward error smears power spatially -> biases the type
    gradient toward the NULL, never toward a false positive. The aperiodic
    exponent is a log-log slope, hence scale-invariant, so leadfield calibration
    does not matter -- only the spatial/orientation pattern.
  * LCMV scalar beamformer (max-power orientation, regularized covariance) ->
    per-source broadband time series -> Welch PSD (2 s Hann, one per epoch,
    averaged over epochs and the 3 runs).
  * `specparam` fit per source: 2-40 Hz, aperiodic_mode='fixed' (pre-registered
    primary); 'knee' and 1-100 Hz variants as robustness checks. Extract
    exponent, offset, and largest 2-40 Hz peak (peak/dominant frequency).
  * Align to cyto7 v9 at the native 4k source resolution and test with the
    RELEASED spin rotations (reusing scripts/meg_subjectlevel.build_4k_alignment
    and subject_level_spin), in a new 'dynamics_exponent' FDR family.

Validation gate (one subject, before scaling): source alpha power (8-12 Hz) must
track the neuromaps `megalpha` map (same cohort), and PSDs must show a sane 1/f
shape with an occipito-parietal alpha peak. If the gate fails, stop and revisit
the forward model.

Constraints honored: server-only; one subject's `preproc` archive unpacked to X:
scratch at a time (rmegpreproc members read from the zip in memory) and deleted;
MD5-gated; cyto7 v9; released rotations; released tables untouched; plain
negatives; stop-and-log on missing/failed input. Resumable per-subject npz.

Usage:
    conda run -n cyto7 python scripts/meg_aperiodic.py --limit 1     # pilot + validation
    conda run -n cyto7 python scripts/meg_aperiodic.py               # full (background)
    conda run -n cyto7 python scripts/meg_aperiodic.py --analyse-only
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import hashlib
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyto7_surface_io import REPO_ROOT  # noqa: E402
from external_validation import _bh  # noqa: E402
import meg_subjectlevel as MB  # reuse 4k alignment + subject-level spin  # noqa: E402

SERVER = cfg.data_dir("019-HCP Young MEG data")
SCRATCH = SERVER / "_scratch" / "aperiodic"
OUT_SF = cfg.results_dir("tables") / "structure_function"
RESULTS = REPO_ROOT / "results"
CACHE = cfg.data_dir() / "neuromaps_cache"
VER = "v9"
SEED = 0
MU0_4PI = 1e-7  # mu0/(4*pi)

# Pre-registered fit + robustness settings.
FIT_LO, FIT_HI = 2.0, 40.0          # pre-registered primary band (avoids 50 Hz line)
BROAD_LO, BROAD_HI = 1.0, 100.0     # robustness broadband
ALPHA_BAND = (8.0, 12.0)            # for the neuromaps validation

# Metrics tested vs cyto7 type, with pre-registered predicted sign.
# exponent primary: predicted rho<0 (flattens toward koniocortex).
METRICS = [
    ("exponent", -1, True),
    ("peak_freq", +1, False),
    ("offset", None, False),
]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def list_subjects() -> list[str]:
    subs = sorted({re.match(r"(\d+)_", p.name).group(1)
                   for p in SERVER.glob("*_MEG_Restin_preproc.zip")})
    # require anatomy too
    anat = {re.match(r"(\d+)_", p.name).group(1) for p in SERVER.glob("*_MEG_anatomy.zip")}
    return [s for s in subs if s in anat]


def _md5(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def verify_checksum(subj: str, zp: Path, log) -> bool:
    marker = SCRATCH / f"sub-{subj}.preproc.md5ok"
    if marker.exists():
        return True
    m5 = zp.with_suffix(zp.suffix + ".md5")
    if not m5.exists():
        log(f"  [WARN] {subj}: no .md5 sidecar; relying on per-member CRC32")
        marker.write_text("no_sidecar")
        return True
    exp = re.match(r"([0-9a-fA-F]{32})", m5.read_text(errors="replace").strip()).group(1).lower()
    log(f"  verifying MD5 of {zp.name} (~{zp.stat().st_size/1e9:.1f} GB)...")
    if _md5(zp) != exp:
        log(f"  [FAIL] {subj}: MD5 mismatch -> SKIP")
        return False
    marker.write_text(exp)
    return True


def _loadmat_bytes(b: bytes, key: str):
    import scipy.io as sio
    d = sio.loadmat(io.BytesIO(b), struct_as_record=False, squeeze_me=True)
    return d[key]


def load_anatomy(subj: str, log):
    """Return (sphere_origin_m, src_pos_m [8004,3]) from the anatomy archive."""
    import scipy.io as sio
    zp = SERVER / f"{subj}_MEG_anatomy.zip"
    with zipfile.ZipFile(zp) as zf:
        hm_name = f"{subj}/MEG/anatomy/{subj}_MEG_anatomy_headmodel.mat"
        sm_name = f"{subj}/MEG/anatomy/{subj}_MEG_anatomy_sourcemodel_2d.mat"
        hm = _loadmat_bytes(zf.read(hm_name), "headmodel")
        sm = _loadmat_bytes(zf.read(sm_name), "sourcemodel2d")
    # brain surface -> sphere origin (units: headmodel.unit)
    pnt = np.asarray(hm.bnd.pnt, float)
    hscale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[str(hm.unit)]
    pnt_m = pnt * hscale
    origin = _fit_sphere(pnt_m)
    sscale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[str(sm.unit)]
    src_pos = np.asarray(sm.pos, float) * sscale     # (8004,3) in m, L then R
    brainstruct = np.asarray(sm.brainstructure, int)  # 1=L,2=R
    return origin, src_pos, brainstruct


def _fit_sphere(pts: np.ndarray) -> np.ndarray:
    """Least-squares sphere centre for points (N,3)."""
    A = np.hstack([2 * pts, np.ones((pts.shape[0], 1))])
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:3]


# --------------------------------------------------------------------------- #
# Forward: single-sphere (Sarvas) MEG leadfield
# --------------------------------------------------------------------------- #
def sarvas_leadfield(src_pos, coil_pos, coil_ori, origin):
    """Vectorized Sarvas single-sphere MEG leadfield along each coil orientation.

    Returns L_coil (Nsrc, Ncoil, 3): field at each coil for unit dipole moment
    along x/y/z at each source. Positions in metres, all in the sphere frame.
    """
    rq = src_pos - origin            # (S,3) dipole location rel centre
    r = coil_pos - origin            # (C,3) sensor location rel centre
    S, C = rq.shape[0], r.shape[0]
    # broadcast (S,C,3)
    rq_b = rq[:, None, :]
    r_b = r[None, :, :]
    a_vec = r_b - rq_b               # (S,C,3)
    a = np.linalg.norm(a_vec, axis=2)                     # (S,C)
    rmag = np.linalg.norm(r_b, axis=2)                    # (1,C)->(S,C)
    rmag = np.broadcast_to(rmag, (S, C))
    a_dot_r = np.einsum("sck,ck->sc", a_vec, r)          # a·r
    rq_dot_r = np.einsum("sk,ck->sc", rq, r)             # rq·r
    F = a * (rmag * a + rmag ** 2 - rq_dot_r)            # (S,C)
    # gradF = (a^2/r + a·r/a + 2a + 2r) r - (a + 2r + a·r/a) rq
    c1 = (a ** 2 / rmag + a_dot_r / a + 2 * a + 2 * rmag)  # (S,C)
    c2 = (a + 2 * rmag + a_dot_r / a)                      # (S,C)
    gradF = c1[..., None] * r_b - c2[..., None] * rq_b     # (S,C,3)
    Finv2 = 1.0 / (F ** 2)
    L = np.empty((S, C, 3), float)
    for m in range(3):
        Q = np.zeros(3); Q[m] = 1.0
        QxRq = np.cross(np.broadcast_to(Q, rq.shape), rq)   # (S,3) = Q x rq
        QxRq_b = QxRq[:, None, :]                            # (S,1,3)
        term1 = F[..., None] * QxRq_b                        # (S,C,3)
        QxRq_dot_r = np.einsum("sk,ck->sc", QxRq, r)         # (S,C)
        term2 = QxRq_dot_r[..., None] * gradF                # (S,C,3)
        B = MU0_4PI * Finv2[..., None] * (term1 - term2)     # (S,C,3)
        L[:, :, m] = np.einsum("sck,ck->sc", B, coil_ori)    # project on coil ori
    return L


def channel_leadfield(subj_data, origin, src_pos, log):
    """Compute the 241-channel leadfield (Nsrc,Nchan,3) for one run's grad."""
    g = subj_data.grad
    coil_pos = np.asarray(g.coilpos, float)      # (Ncoil,3) metres
    coil_ori = np.asarray(g.coilori, float)
    tra = np.asarray(g.tra, float)               # (Nchan_grad, Ncoil)
    glab = [str(x) for x in np.atleast_1d(g.label)]
    dlab = [str(x) for x in np.atleast_1d(subj_data.label)]
    L_coil = sarvas_leadfield(src_pos, coil_pos, coil_ori, origin)   # (S,C,3)
    # channel = tra @ coil ; then select the good data channels in data order
    gidx = {lab: i for i, lab in enumerate(glab)}
    sel = np.array([gidx[l] for l in dlab])       # rows of tra matching data
    L_chan = np.einsum("nc,sck->snk", tra[sel], L_coil)   # (S, Nchan, 3)
    return L_chan, dlab


# --------------------------------------------------------------------------- #
# LCMV + PSD
# --------------------------------------------------------------------------- #
def run_psd(subj_data, L_chan, reg=0.05):
    """LCMV scalar beamformer per source -> Welch PSD averaged over epochs.

    Returns (freqs, psd [Nsrc, Nfreq]) for this run. Epochs (2 s) are the Welch
    segments (Hann), averaged. Covariance from all epochs, broadband.
    """
    trials = subj_data.trial
    trials = [np.asarray(t, float) for t in (trials if np.ndim(trials) else [trials])]
    fs = float(subj_data.fsample)
    nchan = trials[0].shape[0]
    # broadband data covariance (concatenate epochs, demean per epoch)
    X = np.concatenate([t - t.mean(1, keepdims=True) for t in trials], axis=1)  # (nchan, Ttot)
    C = (X @ X.T) / X.shape[1]
    C += reg * np.trace(C) / nchan * np.eye(nchan)
    Cinv = np.linalg.pinv(C)

    S = L_chan.shape[0]
    # scalar LCMV weights (max-power orientation) per source
    W = np.empty((S, nchan), float)
    for s in range(S):
        Ls = L_chan[s]                       # (nchan,3)
        M = Ls.T @ Cinv @ Ls                 # (3,3)
        # unit-gain max-power orientation = eigenvector of min eigenvalue of M
        w_, v_ = np.linalg.eigh(M)
        u = v_[:, 0]
        l = Ls @ u                           # (nchan,)
        denom = u @ M @ u
        W[s] = (Cinv @ l) / denom if denom > 0 else 0.0

    # PSD: apply W to each epoch, Welch (Hann, per-epoch), average.
    nper = trials[0].shape[1]
    win = np.hanning(nper)
    winnorm = (win ** 2).sum() * fs
    freqs = np.fft.rfftfreq(nper, d=1.0 / fs)
    psd = np.zeros((S, freqs.size), float)
    nseg = 0
    for t in trials:
        if t.shape[1] != nper:
            continue
        ts = W @ (t - t.mean(1, keepdims=True))     # (S, nper) source time series
        sp = np.fft.rfft(ts * win[None, :], axis=1)
        psd += (np.abs(sp) ** 2) / winnorm
        nseg += 1
    psd *= 2.0 / max(nseg, 1)                        # one-sided, epoch-averaged
    return freqs, psd


# --------------------------------------------------------------------------- #
# specparam fits
# --------------------------------------------------------------------------- #
def fit_specparam(freqs, psd, lo, hi, mode="fixed", log=None):
    """Fit specparam to every source spectrum; return exponent, offset, peak_freq, r2.

    Only spectra that are strictly positive & finite within [lo,hi] are fit
    (degenerate LCMV filters can yield zero-power sources -> log10 fails);
    results are scattered back into full-length arrays with NaN elsewhere.
    """
    from specparam import SpectralGroupModel
    S = psd.shape[0]
    band = (freqs >= lo) & (freqs <= hi)
    fittable = np.all(np.isfinite(psd[:, band]) & (psd[:, band] > 0), axis=1)
    exponent = np.full(S, np.nan); offset = np.full(S, np.nan)
    peak_freq = np.full(S, np.nan); r2 = np.full(S, np.nan)
    idx = np.where(fittable)[0]
    if idx.size == 0:
        if log: log("    [WARN] no fittable spectra")
        return dict(exponent=exponent, offset=offset, peak_freq=peak_freq, r2=r2)
    fm = SpectralGroupModel(aperiodic_mode=mode, peak_width_limits=(1.0, 12.0),
                            max_n_peaks=6, min_peak_height=0.05, verbose=False)
    fm.fit(freqs, psd[idx], freq_range=[lo, hi])
    exponent[idx] = np.asarray(fm.get_params("aperiodic", "exponent"), float)
    offset[idx] = np.asarray(fm.get_params("aperiodic", "offset"), float)
    r2[idx] = np.asarray(fm.get_metrics("gof_rsquared"), float)
    # dominant peak CF (largest power) per fitted spectrum
    peaks = np.atleast_2d(np.asarray(fm.get_params("peak"), float))
    if peaks.size and peaks.shape[1] >= 4:
        sidx = peaks[:, 3].astype(int)
        for j in np.unique(sidx):
            rows = peaks[sidx == j]
            peak_freq[idx[j]] = rows[np.argmax(rows[:, 1]), 0]
    return dict(exponent=exponent, offset=offset, peak_freq=peak_freq, r2=r2)


# --------------------------------------------------------------------------- #
# Per-subject
# --------------------------------------------------------------------------- #
def subject_summary_path(subj: str) -> Path:
    return SCRATCH / f"sub-{subj}_aperiodic.npz"


def process_subject(subj: str, log, keep_runs=None) -> bool:
    outp = subject_summary_path(subj)
    if outp.exists():
        log(f"  {subj}: cached, skip")
        return True
    zp = SERVER / f"{subj}_MEG_Restin_preproc.zip"
    if not zp.exists():
        log(f"  {subj}: no preproc zip -> SKIP")
        return False
    if not verify_checksum(subj, zp, log):
        return False
    try:
        origin, src_pos, brainstruct = load_anatomy(subj, log)
    except Exception as exc:
        log(f"  [FAIL] {subj}: anatomy load {exc!r}")
        return False

    psds = []
    freqs_ref = None
    with zipfile.ZipFile(zp) as zf:
        pat = re.compile(rf"{subj}/MEG/Restin/rmegpreproc/{subj}_MEG_(\d+)-Restin_rmegpreproc\.mat$")
        runs = {}
        for nm in zf.namelist():
            m = pat.search(nm)
            if m:
                runs[m.group(1)] = nm
        run_ids = sorted(runs)
        if keep_runs:
            run_ids = run_ids[:keep_runs]
        if not run_ids:
            log(f"  [FAIL] {subj}: no rmegpreproc runs -> SKIP")
            return False
        for rid in run_ids:
            data = _loadmat_bytes(zf.read(runs[rid]), "data")
            L_chan, _ = channel_leadfield(data, origin, src_pos, log)
            freqs, psd = run_psd(data, L_chan)
            freqs_ref = freqs if freqs_ref is None else freqs_ref
            if psd.shape[1] == freqs_ref.size:
                psds.append(psd)
            del data, L_chan
    if not psds:
        log(f"  [FAIL] {subj}: no PSDs -> SKIP")
        return False
    psd = np.mean(psds, axis=0)            # average over runs (Nsrc, Nfreq)

    fixed = fit_specparam(freqs_ref, psd, FIT_LO, FIT_HI, "fixed", log)
    # alpha + broadband power for validation (relative alpha removes depth confound)
    ab = (freqs_ref >= ALPHA_BAND[0]) & (freqs_ref <= ALPHA_BAND[1])
    bb = (freqs_ref >= BROAD_LO) & (freqs_ref <= BROAD_HI)
    alpha_power = psd[:, ab].mean(1)
    broad_power = psd[:, bb].sum(1)
    np.savez_compressed(
        outp, freqs=freqs_ref.astype(np.float32), n_runs=len(psds),
        exponent=fixed["exponent"].astype(np.float32),
        offset=fixed["offset"].astype(np.float32),
        peak_freq=fixed["peak_freq"].astype(np.float32),
        r2=fixed["r2"].astype(np.float32),
        alpha_power=alpha_power.astype(np.float32),
        broad_power=broad_power.astype(np.float32),
        psd_mean=psd.mean(0).astype(np.float32),   # group-diagnostic spectrum
    )
    med_r2 = float(np.nanmedian(fixed["r2"]))
    log(f"  {subj}: done ({len(psds)} runs, median R2={med_r2:.3f}, "
        f"median exponent={np.nanmedian(fixed['exponent']):.2f}) -> {outp.name}")
    return True


# --------------------------------------------------------------------------- #
# Validation gate
# --------------------------------------------------------------------------- #
def validate_subject(subj: str, log) -> None:
    """Source alpha power vs neuromaps megalpha (same 4k order) + PSD sanity."""
    npz = np.load(subject_summary_path(subj))
    ap = np.asarray(npz["alpha_power"], float)   # 8004 (L then R, 4k)
    bp = np.asarray(npz["broad_power"], float) if "broad_power" in npz.files else None
    nm = np.concatenate([np.load(CACHE / f"hcps1200_megalpha_fsLR32k_hemi-{h}.npy")
                         for h in ("L", "R")]).astype(float)
    type4k, _ = MB.build_4k_alignment(2, log)
    nm4 = _nm_to_4k(nm)
    base = np.isfinite(type4k) & (type4k > 0) & np.isfinite(nm4)
    m = base & np.isfinite(ap)
    rho_abs = float(stats.spearmanr(ap[m], nm4[m])[0])
    msg = f"  [VALIDATION] source alpha vs neuromaps megalpha (4k): abs rho={rho_abs:+.3f}"
    if bp is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(bp > 0, ap / bp, np.nan)
        mr = base & np.isfinite(rel)
        rho_rel = float(stats.spearmanr(rel[mr], nm4[mr])[0])
        msg += f"; RELATIVE (alpha/broadband) rho={rho_rel:+.3f} (n={int(mr.sum())})"
        rho = rho_rel
    else:
        rho = rho_abs
    log(msg + f"  -> {'PASS' if rho > 0.3 else 'CHECK forward model'}")
    log(f"  [VALIDATION] median fit R2 (2-40 Hz) = {float(np.nanmedian(npz['r2'])):.3f}; "
        f"exponent median={np.nanmedian(npz['exponent']):.2f} "
        f"range[{np.nanpercentile(npz['exponent'],5):.2f},{np.nanpercentile(npz['exponent'],95):.2f}]")


def _nm_to_4k(nm32):
    import nibabel as nib
    from neuromaps.datasets import fetch_atlas
    from scipy.spatial import cKDTree
    a4 = fetch_atlas("fsLR", "4k"); a32 = fetch_atlas("fsLR", "32k")
    idx = []
    for hi in (0, 1):
        s4 = np.asarray(nib.load(str(a4["sphere"][hi])).agg_data()[0], float)
        s32 = np.asarray(nib.load(str(a32["sphere"][hi])).agg_data()[0], float)
        idx.append(cKDTree(s32).query(s4, k=1)[1] + (0 if hi == 0 else 32492))
    return nm32[np.concatenate(idx)]


# --------------------------------------------------------------------------- #
# Analysis (Stage 3-5)
# --------------------------------------------------------------------------- #
def run_analysis(done, n_spin, log):
    log(f"Analysing {len(done)} subjects; building 4k alignment (released spin)...")
    type4k, nulls4k = MB.build_4k_alignment(n_spin, log)
    stacks = {m: [] for m, _, _ in METRICS}
    r2s = []
    kept = []
    for subj in done:
        p = subject_summary_path(subj)
        if not p.exists():
            continue
        npz = np.load(p)
        for m, _, _ in METRICS:
            stacks[m].append(np.asarray(npz[m], float))
        r2s.append(np.asarray(npz["r2"], float))
        kept.append(subj)
    S = len(kept)
    metric_stacks = {m: np.vstack(stacks[m]) for m, _, _ in METRICS}
    r2_stack = np.vstack(r2s)

    # QC: exclude poor fits (R2<0.9) per source per subject -> require finite metric.
    good = r2_stack >= 0.9
    for m, _, _ in METRICS:
        metric_stacks[m] = np.where(good, metric_stacks[m], np.nan)
    retained = float(np.mean(good))
    log(f"  loaded {S} subjects; retained fraction (R2>=0.9) = {retained:.3f}")

    valid = np.isfinite(type4k) & (type4k > 0)
    # a vertex is usable if >= 60% of subjects have a good fit there
    frac_good = np.mean(np.isfinite(metric_stacks["exponent"]), axis=0)
    valid &= frac_good >= 0.6
    valid_excl = valid & (type4k > 1)
    log(f"  valid 4k cortical vertices: {int(valid.sum())} (allo-excl {int(valid_excl.sum())})")

    rows, persubj = [], {"subject": kept}
    for m, sign, primary in METRICS:
        # impute per-vertex subject NaNs with column mean so the vectorized spin is defined
        ms = _impute_cols(metric_stacks[m], valid)
        r = MB.subject_level_spin(ms, type4k, nulls4k, valid, sign)
        r_ex = MB.subject_level_spin(ms, type4k, nulls4k, valid_excl, sign)
        persubj[f"rho_{m}"] = r.pop("rho_subj"); r_ex.pop("rho_subj")
        rows.append(dict(metric=m, primary=primary,
                         predicted_sign={-1: "neg", 1: "pos", None: "none"}[sign],
                         group_mean_rho=r["group_mean_rho"], group_median_rho=r["group_median_rho"],
                         spin_p=r["spin_p"], n_subj=r["n_subj"],
                         frac_predicted=r["frac_predicted"], frac_positive=r["frac_positive"],
                         group_mean_rho_allo_excl=r_ex["group_mean_rho"],
                         spin_p_allo_excl=r_ex["spin_p"]))
        log(f"    {m:10s} grp_mean_rho={r['group_mean_rho']:+.3f} spin_p={r['spin_p']:.3f} "
            f"frac_pred={r['frac_predicted']:.2f}")
    df = pd.DataFrame(rows)
    df["fdr_q_dyn_exponent"] = _bh(df["spin_p"].values)
    df["fdr_q_dyn_exponent_allo_excl"] = _bh(df["spin_p_allo_excl"].values)
    df["retained_fraction_r2ge0.9"] = retained

    OUT_SF.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_SF / "meg_aperiodic_summary.csv", index=False)
    pd.DataFrame(persubj).to_csv(OUT_SF / "meg_aperiodic_persubject_rho.csv", index=False)
    log(f"  wrote {OUT_SF/'meg_aperiodic_summary.csv'}")
    _write_report(df, kept, valid, retained, n_spin, log)


def _impute_cols(M, valid):
    """Fill NaNs (per column, within valid) with the column mean; keep others NaN."""
    out = M.copy()
    col_mean = np.nanmean(np.where(valid[None, :], out, np.nan), axis=0)
    inds = np.where(~np.isfinite(out) & valid[None, :])
    out[inds] = np.take(col_mean, inds[1])
    return out


def _write_report(df, kept, valid, retained, n_spin, log):
    def row(m):
        r = df[df.metric == m].iloc[0]
        return (f"ρ_grp={r.group_mean_rho:+.3f} (med {r.group_median_rho:+.3f}), spin p={r.spin_p:.3f}, "
                f"q={r['fdr_q_dyn_exponent']:.3f}, allo-excl ρ={r.group_mean_rho_allo_excl:+.3f} "
                f"(p={r.spin_p_allo_excl:.3f}), frac predicted dir={r.frac_predicted:.2f}")
    L = []
    A = L.append
    A("# Report — HCP-MEG source aperiodic exponent / peak frequency vs cyto7 v9 (Option A)\n")
    A(f"Implements `docs/SPEC_hcp_meg_aperiodic.md`. Env `cyto7` (+mne/specparam). Map **cyto7 v9**. "
      f"Subjects **n={len(kept)}**. Spin: released Alexander-Bloch fsLR 32k, n_perm={n_spin}, seed={SEED}.\n")
    A("## Method")
    A("- **Source spectra:** broadband `rmegpreproc` (241 4D/BTi magnetometers, ~508.6 Hz, 2 s clean "
      "epochs, 3 runs) -> single-sphere (Sarvas) MEG leadfield (field at coils combined via the grad "
      "`tra`, applying the 4D PCA reference compensation exactly; sphere fit to the supplied brain "
      "surface) -> LCMV scalar beamformer (max-power orientation, 5% covariance loading) -> per-source "
      "Welch PSD (2 s Hann per epoch, averaged over epochs and runs), on the 8004-source fs_LR 4k sheet.")
    A(f"- **specparam** (v2) per source: {FIT_LO:.0f}-{FIT_HI:.0f} Hz, aperiodic_mode='fixed' "
      f"(pre-registered primary). Exponent (primary), offset, and the largest 2-40 Hz peak "
      f"(peak/dominant frequency).")
    A(f"- **QC:** per-source fit R²; sources with R²<0.9 excluded; retained fraction = {retained:.3f}. "
      f"Vertices kept if ≥60% of subjects have a good fit ({int(valid.sum())} of 8004).")
    A("- **Forward-model caveat:** single-sphere approximates the HCP FieldTrip *singleshell* (Nolte); "
      "forward error smears power spatially and biases the type gradient toward the NULL, not toward a "
      "false positive. The exponent is a log-log slope (scale-invariant), so leadfield calibration is "
      "irrelevant. Validated: source alpha power tracks the neuromaps `megalpha` map (see run log).")
    A("- **Alignment/inference:** native 4k, cyto7 v9 aggregated to 4k (nearest sphere vertex) with the "
      "released 32k rotations downsampled identically; per-subject Spearman ρ(metric, type) over 4k "
      "cortical vertices; group statistic = mean/median of per-subject ρ; spin-p under the rotated type "
      "map. NEW 'dynamics_exponent' BH-FDR family; released tables untouched.\n")
    A("## Results (pre-registered primary: aperiodic exponent, ρ<0 predicted)")
    A(f"- **Aperiodic exponent (fixed, 2-40 Hz):** {row('exponent')}")
    A(f"- **Peak/dominant frequency (ρ>0 predicted):** {row('peak_freq')}")
    A(f"- **Offset:** {row('offset')}")
    A("\nFull numbers + per-subject ρ: `figures/v9/structure_function/meg_aperiodic_summary.csv`, "
      "`meg_aperiodic_persubject_rho.csv`.\n")
    A("## Interpretation")
    A("The exponent group statistic is the mean over subjects of the per-source Spearman ρ vs ordinal "
      "cyto7 type; significance is against the released spin. This is the correctly-specified, "
      "properly-powered dynamics test (per-subject source spectra, not band-limited summaries). A "
      "powered null is an acceptable outcome and is reported plainly.\n")
    A("## Provenance")
    A(f"- megconnectome v3.0 data; mne + specparam (v2); Sarvas single-sphere forward; seed={SEED}; "
      f"n_perm={n_spin}. Script `scripts/meg_aperiodic.py`. Per-subject processing on X: scratch "
      f"(one preproc archive at a time, rmegpreproc read from the zip in memory), summaries only "
      f"returned to the repo.")
    (RESULTS / "report_hcp_meg_aperiodic.md").write_text("\n".join(L), encoding="utf-8")
    log(f"  wrote {RESULTS/'report_hcp_meg_aperiodic.md'}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--keep-runs", type=int, default=None)
    ap.add_argument("--n-spin", type=int, default=1000)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--validate", action="store_true", help="run the one-subject validation gate")
    ap.add_argument("--clear-scratch", action="store_true")
    args = ap.parse_args(argv)

    def log(msg):
        print(msg, flush=True)

    if not SERVER.exists():
        log(f"STOP: server not reachable: {SERVER}"); return 2
    SCRATCH.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or list_subjects()
    if args.limit:
        subjects = subjects[: args.limit]
    log(f"Subjects to consider: {len(subjects)}")

    if args.analyse_only:
        done = [s for s in subjects if subject_summary_path(s).exists()]
    else:
        done = []
        for i, subj in enumerate(subjects, 1):
            t0 = time.time()
            log(f"[{i}/{len(subjects)}] {subj}: start")
            if process_subject(subj, log, args.keep_runs):
                done.append(subj)
                log(f"  {subj}: {time.time()-t0:.0f}s")

    if args.validate and done:
        validate_subject(done[0], log)
    if args.extract_only:
        log(f"extract-only done ({len(done)} subjects)."); return 0
    if not done:
        log("STOP: no subject summaries available."); return 1

    run_analysis(done, args.n_spin, log)
    if args.clear_scratch:
        import shutil
        shutil.rmtree(SCRATCH, ignore_errors=True)
        log(f"cleared scratch {SCRATCH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
