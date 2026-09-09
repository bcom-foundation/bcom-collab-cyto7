#!/usr/bin/env python
"""Subject-level MEG band-power / timescale gradients vs cyto7 v9 (Option B).

Implements docs/SPEC_hcp_meg_bandpower_subjectlevel.md. Re-runs the dynamics
metrics the paper currently reports on a single group-average map (intrinsic
timescale, slow/fast ratio, spectral centroid, band powers) **per subject**
(~89 HCP-MEG subjects), for statistical power and individual-level replication.
This is band-limited and does NOT give the aperiodic exponent (that needs the
broadband sensor `rmegpreproc` + beamformer + specparam pipeline, Option A).

Data: the HCP-MEG `bfblpenv` "dtseries" package = beamformer band-limited power
*envelopes*, 8 megconnectome bands x 3 resting runs, on the 8004-source
fs_LR 4k mesh (rows 0-4001 = L, 4002-8003 = R; verified against neuromaps
hcps1200 4k). Envelope sampling fs ~= 49.6 Hz, ~296 s/run.

Per-subject metrics (per source, run-averaged):
  * relative band power  r_b = P_b / sum_b P_b   for canonical delta/theta/alpha/beta/gamma1
    (relative removes the per-vertex leadfield/depth multiplicative confound that
     dominates absolute source power; SF and centroid are ratios so unaffected).
  * spectral centroid    = sum_b(centre_b * P_b) / sum_b(P_b) over canonical 5 bands.
  * slow/fast ratio      SF = (P_delta + P_theta) / (P_beta + P_gamma1)  (as pre-registered).
  * intrinsic timescale  tau = lag at which the ACF of the broadband power envelope
    (sum of all 8 band envelopes) falls to 1/e (envelope-based approximation of
    neuromaps `megtimescale`; validity-checked against it).

Alignment (Stage 2): work at the native 4k MEG resolution and aggregate the
cyto7 v9 type map DOWN to 4k (nearest fs_LR sphere vertex). The RELEASED 32k
Alexander-Bloch rotation set (seed=0, n_perm) is applied to the 32k type map and
each rotated map is downsampled 32k->4k the same way, so the null preserves both
the released rotations and the observed/null consistency (SPEC-permitted route).

Inference (Stage 3): per subject Spearman rho(metric, ordinal type over 4k
cortical vertices); group statistic = mean (and median) of per-subject rho;
spin-p from the group statistic under the rotated type map; BH-FDR within a new
"dynamics" family kept separate from the released tables; fraction of subjects
in the predicted direction; side-by-side with the released single-group-map rho.

Hard constraints honored: server-only, one band-file (~470 MB) unpacked at a
time on X: scratch and deleted immediately, whole-archive MD5 gate before use,
released tables never modified. Resumable: per-subject summaries cached on
scratch; re-runs skip completed subjects.

Usage:
    conda run -n cyto7 python scripts/meg_subjectlevel.py --limit 3        # pilot
    conda run -n cyto7 python scripts/meg_subjectlevel.py                  # full (background)
    conda run -n cyto7 python scripts/meg_subjectlevel.py --analyse-only   # skip extraction
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import hashlib
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyto7_surface_io import REPO_ROOT, LABEL_NAMES, resolve_target_map  # noqa: E402
from external_validation import _bh  # identical BH-FDR used for the released tables  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SERVER = cfg.data_dir("019-HCP Young MEG data")
SCRATCH = SERVER / "_scratch" / "meg_subjectlevel"
OUT_SF = cfg.results_dir("tables") / "structure_function"
RESULTS = REPO_ROOT / "results"
CACHE = cfg.data_dir() / "neuromaps_cache"
VER = "v9"
SEED = 0
N_HEMI_4K = 4002            # fs_LR 4k vertices per hemisphere (verified)
FS_HZ = 49.623              # envelope sampling rate (1/0.020152 s); confirmed/overwritten per file at extraction

# 8 megconnectome bfblpenv bands (order used in files).
ALL_BANDS = ["delta", "theta", "alpha", "betalow", "betahigh",
             "gammalow", "gammamid", "gammahigh"]

# megconnectome 3.0 pipeline band edges (Hz). NOT stored in the CIFTI metadata
# (verified: matrix metadata empty, provenance XML has only pipeline version);
# these are the documented megconnectome/HCP-MEG defaults used by hcp_bfblpenv.m.
# Used only for the spectral centroid band centres; edges reported in the report.
BAND_EDGES = {
    "delta": (1.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 15.0),
    "betalow": (15.0, 26.0), "betahigh": (26.0, 35.0),
    "gammalow": (35.0, 50.0), "gammamid": (50.0, 76.0), "gammahigh": (76.0, 120.0),
}

# Canonical 5-band mapping (paper's delta/theta/alpha/beta/gamma1).
CANON = {
    "delta": ["delta"], "theta": ["theta"], "alpha": ["alpha"],
    "beta": ["betalow", "betahigh"], "gamma1": ["gammalow"],
}
CANON_ORDER = ["delta", "theta", "alpha", "beta", "gamma1"]


def _canon_centre(name: str) -> float:
    lo = min(BAND_EDGES[b][0] for b in CANON[name])
    hi = max(BAND_EDGES[b][1] for b in CANON[name])
    return 0.5 * (lo + hi)


CANON_CENTRE = {b: _canon_centre(b) for b in CANON_ORDER}

# Metrics tested, with pre-registered predicted sign (None = no crisp prediction).
# timescale & SF are the pre-registered primaries.
METRICS = [
    ("timescale", -1, True),
    ("sf_ratio", -1, True),
    ("centroid", +1, False),
    ("rel_delta", -1, False),
    ("rel_theta", -1, False),
    ("rel_alpha", None, False),
    ("rel_beta", +1, False),
    ("rel_gamma1", +1, False),
]


# --------------------------------------------------------------------------- #
# Stage 0 / 1 — per-subject extraction & metrics
# --------------------------------------------------------------------------- #
def list_subjects() -> list[str]:
    subs = sorted({re.match(r"(\d+)_", p.name).group(1)
                   for p in SERVER.glob("*_MEG_Restin_dtseries.zip")})
    return subs


def expected_md5(zp: Path) -> str | None:
    m5 = zp.with_suffix(zp.suffix + ".md5")
    if not m5.exists():
        return None
    t = m5.read_text(encoding="utf-8", errors="replace").strip()
    mm = re.match(r"([0-9a-fA-F]{32})", t)
    return mm.group(1).lower() if mm else None


def md5_of(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def verify_checksum(subj: str, zp: Path, log) -> bool:
    """MD5-gate the archive before use; cache a marker so it runs once."""
    marker = SCRATCH / f"sub-{subj}.md5ok"
    if marker.exists():
        return True
    exp = expected_md5(zp)
    if exp is None:
        log(f"  [WARN] {subj}: no .md5 sidecar -> relying on per-member CRC32 only")
        marker.write_text("no_sidecar")
        return True
    log(f"  verifying MD5 of {zp.name} (~{zp.stat().st_size/1e9:.1f} GB)...")
    act = md5_of(zp)
    if act != exp:
        log(f"  [FAIL] {subj}: MD5 mismatch (expected {exp}, got {act}) -> SKIP")
        return False
    marker.write_text(exp)
    return True


def _parse_members(zf: zipfile.ZipFile, subj: str) -> dict[tuple[str, str], str]:
    """Map (run, band) -> member path for bfblpenv .nii data files."""
    out: dict[tuple[str, str], str] = {}
    pat = re.compile(rf"{subj}/MEG/Restin/bfblpenv/{subj}_MEG_(\d+)-Restin_"
                     r"bfblpenv_(\w+)\.power\.dtseries\.nii$")
    for nm in zf.namelist():
        m = pat.search(nm)
        if m:
            out[(m.group(1), m.group(2))] = nm
    return out


def _acf_tau(E: np.ndarray, dt: float, max_lag_s: float = 10.0,
             chunk: int = 1000) -> np.ndarray:
    """Per-vertex intrinsic timescale: lag where the ACF falls to 1/e (seconds).

    E: (T, V) broadband power envelope. FFT autocorrelation, chunked over
    vertices to bound memory; linear interpolation of the 1/e crossing.
    """
    T, V = E.shape
    Em = E - E.mean(axis=0, keepdims=True)
    n = 1
    while n < 2 * T:
        n <<= 1
    Lmax = min(T - 1, int(round(max_lag_s / dt)))
    thr = 1.0 / np.e
    tau = np.full(V, np.nan, dtype=np.float64)
    for a in range(0, V, chunk):
        b = min(a + chunk, V)
        f = np.fft.rfft(Em[:, a:b], n=n, axis=0)
        ac = np.fft.irfft(f * np.conj(f), n=n, axis=0)[: Lmax + 1]  # (Lmax+1, nc)
        ac0 = ac[0:1].copy()
        ac0[ac0 == 0] = np.nan
        ac = ac / ac0
        below = ac < thr
        has = below.any(axis=0)
        k = np.argmax(below, axis=0)  # first True lag (0 if none, guarded by `has`)
        kk = np.clip(k, 1, Lmax)
        prev = ac[kk - 1, np.arange(b - a)]
        cur = ac[kk, np.arange(b - a)]
        denom = prev - cur
        frac = np.where(denom != 0, (prev - thr) / denom, 0.0)
        lag = (kk - 1 + frac).astype(np.float64)
        lag[~has] = np.nan
        tau[a:b] = lag * dt
    return tau


def compute_subject(subj: str, zp: Path, log, keep_runs: int | None) -> dict | None:
    """Compute run-averaged 8-band absolute power + tau (8004 vectors).

    Reads each ~470 MB band-envelope member fully into memory via ZipFile.read
    (which validates the member CRC32) and parses it with Cifti2Image.from_bytes
    -- no temp file on disk, so peak extra disk use is nil and there is no
    Windows mmap/unlink hazard. Peak RAM ~= member bytes + one float32 array +
    the broadband accumulator (~1.4 GB).
    """
    global FS_HZ
    import gc
    import nibabel as nib
    with zipfile.ZipFile(zp) as zf:
        members = _parse_members(zf, subj)
        runs = sorted({r for (r, _b) in members})
        if keep_runs:
            runs = runs[:keep_runs]
        if not runs:
            log(f"  [FAIL] {subj}: no bfblpenv members found -> SKIP")
            return None
        band_sum: dict[str, np.ndarray] = {b: None for b in ALL_BANDS}
        tau_runs: list[np.ndarray] = []
        n_used_runs = 0
        for run in runs:
            missing = [b for b in ALL_BANDS if (run, b) not in members]
            if missing:
                log(f"  [WARN] {subj} run {run}: missing bands {missing} -> skip run")
                continue
            E = None
            for band in ALL_BANDS:
                img = nib.Cifti2Image.from_bytes(zf.read(members[(run, band)]))
                FS_HZ = 1.0 / float(img.header.get_axis(0).step)
                arr = np.asarray(img.get_fdata(dtype=np.float32))  # (T, 8004)
                bm = arr.mean(axis=0)
                band_sum[band] = bm if band_sum[band] is None else band_sum[band] + bm
                E = arr.copy() if E is None else E + arr
                del arr, img
                gc.collect()
            dt = 1.0 / FS_HZ
            tau_runs.append(_acf_tau(E, dt))
            del E
            gc.collect()
            n_used_runs += 1
        if n_used_runs == 0:
            log(f"  [FAIL] {subj}: no complete runs -> SKIP")
            return None
        absmean = {b: band_sum[b] / n_used_runs for b in ALL_BANDS}
        tau = np.nanmean(np.vstack(tau_runs), axis=0)
    return {"absmean": absmean, "tau": tau, "n_runs": n_used_runs,
            "n_verts": int(tau.size)}


def subject_summary_path(subj: str) -> Path:
    return SCRATCH / f"sub-{subj}_meg4k.npz"


def extract_all(subjects: list[str], log, keep_runs: int | None) -> list[str]:
    """Produce per-subject summary npz on scratch (resumable). Returns done ids."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    done = []
    for i, subj in enumerate(subjects, 1):
        outp = subject_summary_path(subj)
        if outp.exists():
            log(f"[{i}/{len(subjects)}] {subj}: cached, skip")
            done.append(subj)
            continue
        zp = SERVER / f"{subj}_MEG_Restin_dtseries.zip"
        if not zp.exists():
            log(f"[{i}/{len(subjects)}] {subj}: no dtseries zip -> SKIP")
            continue
        t0 = time.time()
        log(f"[{i}/{len(subjects)}] {subj}: start")
        if not verify_checksum(subj, zp, log):
            continue
        try:
            res = compute_subject(subj, zp, log, keep_runs)
        except (zipfile.BadZipFile, OSError) as exc:
            log(f"  [FAIL] {subj}: read error {exc!r} -> SKIP")
            continue
        if res is None:
            continue
        np.savez_compressed(
            outp,
            tau=res["tau"].astype(np.float32),
            n_runs=res["n_runs"],
            **{f"abs_{b}": res["absmean"][b].astype(np.float32) for b in ALL_BANDS},
        )
        done.append(subj)
        log(f"  {subj}: done ({res['n_runs']} runs, {time.time()-t0:.0f}s) -> {outp.name}")
    return done


# --------------------------------------------------------------------------- #
# Stage 1b — derive metrics from a subject summary
# --------------------------------------------------------------------------- #
def derive_metrics(npz: dict) -> dict[str, np.ndarray]:
    absb = {b: np.asarray(npz[f"abs_{b}"], dtype=np.float64) for b in ALL_BANDS}
    canon = {c: sum(absb[m] for m in CANON[c]) for c in CANON_ORDER}
    broadband = sum(absb[b] for b in ALL_BANDS)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = {c: np.where(broadband > 0, canon[c] / broadband, np.nan) for c in CANON_ORDER}
        num = canon["delta"] + canon["theta"]
        den = canon["beta"] + canon["gamma1"]
        sf = np.where(den > 0, num / den, np.nan)
        cw = sum(CANON_CENTRE[c] * canon[c] for c in CANON_ORDER)
        cd = sum(canon[c] for c in CANON_ORDER)
        centroid = np.where(cd > 0, cw / cd, np.nan)
    out = {"timescale": np.asarray(npz["tau"], dtype=np.float64),
           "sf_ratio": sf, "centroid": centroid}
    for c in CANON_ORDER:
        out[f"rel_{c}"] = rel[c]
    return out


# --------------------------------------------------------------------------- #
# Stage 2 — cyto7 v9 aggregated to 4k + released 32k nulls downsampled to 4k
# --------------------------------------------------------------------------- #
def build_4k_alignment(n_spin: int, log) -> tuple[np.ndarray, np.ndarray]:
    """Return (type4k [8004], nulls4k [8004, n_spin]) from the released 32k spin.

    v9 32k type map is aggregated to fs_LR 4k by nearest sphere vertex; the same
    nearest map is applied to each Alexander-Bloch-rotated 32k rank map.
    """
    from neuromaps.datasets import fetch_atlas
    from neuromaps.nulls import alexander_bloch
    import nibabel as nib

    labels = resolve_target_map(VER, "fs_LR")  # {"L":32492, "R":32492}
    rank32 = {h: np.where(labels[h] > 0, labels[h].astype(float), np.nan) for h in ("L", "R")}
    rank32_concat = np.concatenate([rank32["L"], rank32["R"]])

    log(f"  generating released spin nulls (alexander_bloch fsLR 32k, "
        f"n_perm={n_spin}, seed={SEED})...")
    nulls32 = alexander_bloch(rank32_concat, atlas="fsLR", density="32k",
                              n_perm=n_spin, seed=SEED)  # (64984, n_spin)

    # nearest 4k->32k per hemisphere using fs_LR spheres
    atl4 = fetch_atlas("fsLR", "4k")
    atl32 = fetch_atlas("fsLR", "32k")
    nn = {}
    for hi, h in enumerate(("L", "R")):
        sph4 = np.asarray(nib.load(str(atl4["sphere"][hi])).agg_data()[0], dtype=np.float64)
        sph32 = np.asarray(nib.load(str(atl32["sphere"][hi])).agg_data()[0], dtype=np.float64)
        if sph4.shape[0] != N_HEMI_4K or sph32.shape[0] != 32492:
            raise RuntimeError(f"unexpected sphere sizes {sph4.shape} {sph32.shape} ({h})")
        nn[h] = cKDTree(sph32).query(sph4, k=1)[1]  # (4002,) idx into 32k hemi
    off = {"L": 0, "R": 32492}
    idx = np.concatenate([nn["L"] + off["L"], nn["R"] + off["R"]])  # (8004,) into 64984
    type4k = rank32_concat[idx]
    nulls4k = nulls32[idx, :]
    return type4k, nulls4k


# --------------------------------------------------------------------------- #
# Stage 3 — subject-level inference
# --------------------------------------------------------------------------- #
def _fast_group_null(M: np.ndarray, spun: np.ndarray, base: np.ndarray) -> float:
    """Mean over subjects of Spearman(metric_s, spun) on vertices `base`.

    M: (S, V) metric stack; spun: (V,) rotated rank; base: (V,) bool valid set.
    Uses rank-transform + vectorized Pearson (== Spearman) across subjects.
    """
    sub = M[:, base]                                   # (S, nb)
    rt = stats.rankdata(spun[base]).astype(np.float64)  # (nb,)
    rm = stats.rankdata(sub, axis=1).astype(np.float64)  # (S, nb)
    rt -= rt.mean()
    rm -= rm.mean(axis=1, keepdims=True)
    num = rm @ rt
    den = np.sqrt((rm * rm).sum(axis=1) * (rt * rt).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rhos = np.where(den > 0, num / den, np.nan)
    return float(np.nanmean(rhos))


def subject_level_spin(metric_stack: np.ndarray, type4k: np.ndarray,
                       nulls4k: np.ndarray, valid: np.ndarray,
                       predicted_sign: int | None) -> dict:
    """Group mean/median of per-subject rho + spin-p under the rotated type map."""
    S = metric_stack.shape[0]
    rt_obs = stats.rankdata(type4k[valid]).astype(np.float64)
    rt_obs -= rt_obs.mean()
    sub = metric_stack[:, valid]
    rm = stats.rankdata(sub, axis=1).astype(np.float64)
    rm -= rm.mean(axis=1, keepdims=True)
    num = rm @ rt_obs
    den = np.sqrt((rm * rm).sum(axis=1) * (rt_obs * rt_obs).sum())
    rho_subj = np.where(den > 0, num / den, np.nan)     # (S,)
    group_mean = float(np.nanmean(rho_subj)) if np.isfinite(rho_subj).any() else float("nan")
    group_med = float(np.nanmedian(rho_subj)) if np.isfinite(rho_subj).any() else float("nan")

    # Degenerate metric (e.g. near-constant/all-zero map: no oscillatory peaks in a band)
    # -> observed rho is undefined; report NaN p rather than a spurious significance.
    if not np.isfinite(group_mean):
        return dict(group_mean_rho=float("nan"), group_median_rho=float("nan"),
                    spin_p=float("nan"), n_subj=int(np.isfinite(rho_subj).sum()),
                    frac_predicted=float("nan"), frac_positive=float("nan"),
                    rho_subj=rho_subj)

    n_spin = nulls4k.shape[1]
    group_null = np.empty(n_spin)
    for i in range(n_spin):
        col = nulls4k[:, i]
        base = valid & np.isfinite(col)
        group_null[i] = _fast_group_null(metric_stack, col, base)
    gn = group_null[np.isfinite(group_null)]
    p_spin = float((np.sum(np.abs(gn) >= abs(group_mean)) + 1) / (gn.size + 1)) if gn.size else float("nan")

    frac_pred = float("nan")
    if predicted_sign is not None:
        fin = np.isfinite(rho_subj)
        frac_pred = float(np.mean(np.sign(rho_subj[fin]) == predicted_sign))
    frac_pos = float(np.mean(rho_subj[np.isfinite(rho_subj)] > 0))
    return dict(group_mean_rho=group_mean, group_median_rho=group_med,
                spin_p=p_spin, n_subj=int(np.isfinite(rho_subj).sum()),
                frac_predicted=frac_pred, frac_positive=frac_pos,
                rho_subj=rho_subj)


# --------------------------------------------------------------------------- #
# Released single-group-map reference rho (neuromaps 32k, released spin)
# --------------------------------------------------------------------------- #
def reference_group_map_rhos(n_spin: int, log) -> dict[str, dict]:
    """Group-map rho for each metric from the cached neuromaps 32k maps (the
    quantities the paper used), with the identical released 32k spin. Reported
    side-by-side with the subject-level statistic."""
    from neuromaps.nulls import alexander_bloch
    labels = resolve_target_map(VER, "fs_LR")
    rank_full = np.concatenate([np.where(labels[h] > 0, labels[h].astype(float), np.nan)
                                for h in ("L", "R")])
    nulls = alexander_bloch(rank_full, atlas="fsLR", density="32k",
                            n_perm=n_spin, seed=SEED)

    def loadm(desc):
        return np.concatenate([np.load(CACHE / f"hcps1200_{desc}_fsLR32k_hemi-{h}.npy")
                               for h in ("L", "R")]).astype(float)

    band = {b: loadm(f"meg{b}") for b in ("delta", "theta", "alpha", "beta")}
    band["gamma1"] = loadm("meggamma1")
    ts = loadm("megtimescale")
    broad = sum(band[b] for b in CANON_ORDER)
    with np.errstate(divide="ignore", invalid="ignore"):
        relmap = {b: np.where(broad > 0, band[b] / broad, np.nan) for b in CANON_ORDER}
        sfmap = np.where((band["beta"] + band["gamma1"]) > 0,
                         (band["delta"] + band["theta"]) / (band["beta"] + band["gamma1"]), np.nan)
        cw = sum(CANON_CENTRE[b] * band[b] for b in CANON_ORDER)
        cd = sum(band[b] for b in CANON_ORDER)
        centroidmap = np.where(cd > 0, cw / cd, np.nan)
    feat = {"timescale": ts, "sf_ratio": sfmap, "centroid": centroidmap,
            **{f"rel_{b}": relmap[b] for b in CANON_ORDER}}

    valid = (rank_full > 0)
    obs_rank = rank_full[valid]
    out = {}
    for key, vals in feat.items():
        vv = vals[valid]
        ok0 = np.isfinite(vv)
        rho = float(stats.spearmanr(vv[ok0], obs_rank[ok0])[0])
        nr = np.empty(n_spin)
        for i in range(n_spin):
            spun = nulls[:, i][valid]
            ok = np.isfinite(spun) & np.isfinite(vv)
            nr[i] = stats.spearmanr(vv[ok], spun[ok])[0]
        p = float((np.sum(np.abs(nr) >= abs(rho)) + 1) / (n_spin + 1))
        out[key] = dict(ref_rho=rho, ref_spin_p=p)
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def run_analysis(done: list[str], n_spin: int, log) -> None:
    log(f"Analysing {len(done)} subjects; building 4k alignment...")
    type4k, nulls4k = build_4k_alignment(n_spin, log)

    # Load subject metric stacks.
    stacks: dict[str, list[np.ndarray]] = {m: [] for m, _, _ in METRICS}
    kept = []
    for subj in done:
        p = subject_summary_path(subj)
        if not p.exists():
            continue
        npz = np.load(p)
        met = derive_metrics(npz)
        for m, _, _ in METRICS:
            stacks[m].append(met[m])
        kept.append(subj)
    S = len(kept)
    log(f"  loaded {S} subject metric maps ({type4k.size} verts)")
    metric_stacks = {m: np.vstack(stacks[m]) for m, _, _ in METRICS}

    # Global valid mask: labelled 4k cortex, finite across all subjects & metrics.
    valid = np.isfinite(type4k) & (type4k > 0)
    for m, _, _ in METRICS:
        valid &= np.all(np.isfinite(metric_stacks[m]), axis=0)
    valid_excl = valid & (type4k > 1)
    log(f"  valid 4k cortical vertices: {int(valid.sum())} "
        f"(allo-excluded {int(valid_excl.sum())})")

    # Validity check: group-mean rel_delta & tau vs neuromaps (spatial Spearman at 4k).
    _validity_check(metric_stacks, type4k, valid, log)

    log("  computing released group-map reference rhos (neuromaps 32k)...")
    ref = reference_group_map_rhos(n_spin, log)

    rows, persubj = [], {"subject": kept}
    for m, sign, primary in METRICS:
        r = subject_level_spin(metric_stacks[m], type4k, nulls4k, valid, sign)
        r_ex = subject_level_spin(metric_stacks[m], type4k, nulls4k, valid_excl, sign)
        persubj[f"rho_{m}"] = r.pop("rho_subj")
        r_ex.pop("rho_subj")
        rows.append(dict(
            metric=m, primary=primary,
            predicted_sign=("neg" if sign == -1 else "pos" if sign == 1 else "none"),
            group_mean_rho=r["group_mean_rho"], group_median_rho=r["group_median_rho"],
            spin_p=r["spin_p"], n_subj=r["n_subj"],
            frac_predicted=r["frac_predicted"], frac_positive=r["frac_positive"],
            group_mean_rho_allo_excl=r_ex["group_mean_rho"], spin_p_allo_excl=r_ex["spin_p"],
            released_map_rho=ref[m]["ref_rho"], released_map_spin_p=ref[m]["ref_spin_p"],
        ))
        log(f"    {m:11s} grp_mean_rho={r['group_mean_rho']:+.3f} spin_p={r['spin_p']:.3f} "
            f"| ref_map_rho={ref[m]['ref_rho']:+.3f} | frac_pred={r['frac_predicted']:.2f}")

    df = pd.DataFrame(rows)
    # Dynamics FDR family (BH across all subject-level metrics; separate family).
    df["fdr_q_dynamics"] = _bh(df["spin_p"].values)
    df["fdr_q_dynamics_allo_excl"] = _bh(df["spin_p_allo_excl"].values)

    OUT_SF.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    spath = OUT_SF / "meg_subjectlevel_summary.csv"
    df.to_csv(spath, index=False)
    log(f"  wrote {spath}")
    pspath = OUT_SF / "meg_subjectlevel_persubject_rho.csv"
    pd.DataFrame(persubj).to_csv(pspath, index=False)
    log(f"  wrote {pspath}")

    _plot(df, persubj, log)
    _write_report(df, kept, valid, valid_excl, n_spin, log)


def _validity_check(metric_stacks, type4k, valid, log) -> None:
    """Group-mean rel_delta and tau vs neuromaps maps (downsampled to 4k)."""
    try:
        import nibabel as nib
        from neuromaps.datasets import fetch_atlas
        atl4 = fetch_atlas("fsLR", "4k")
        atl32 = fetch_atlas("fsLR", "32k")
        nn = []
        for hi in (0, 1):
            s4 = np.asarray(nib.load(str(atl4["sphere"][hi])).agg_data()[0], float)
            s32 = np.asarray(nib.load(str(atl32["sphere"][hi])).agg_data()[0], float)
            nn.append(cKDTree(s32).query(s4, k=1)[1])
        idx = np.concatenate([nn[0], nn[1] + 32492])

        def nm4(desc):
            v = np.concatenate([np.load(CACHE / f"hcps1200_{desc}_fsLR32k_hemi-{h}.npy")
                                for h in ("L", "R")]).astype(float)
            return v[idx]
        gm_reldelta = np.nanmean(metric_stacks["rel_delta"], axis=0)
        gm_tau = np.nanmean(metric_stacks["timescale"], axis=0)
        for lab, mine, desc in [("rel_delta", gm_reldelta, "megdelta"),
                                ("timescale", gm_tau, "megtimescale")]:
            ref = nm4(desc)
            mm = valid & np.isfinite(mine) & np.isfinite(ref)
            rho = float(stats.spearmanr(mine[mm], ref[mm])[0])
            log(f"  [validity] group-mean {lab} vs neuromaps {desc} (4k): "
                f"Spearman rho={rho:+.3f} (n={int(mm.sum())})")
    except Exception as exc:  # pragma: no cover
        log(f"  [validity] check skipped: {exc!r}")


# --------------------------------------------------------------------------- #
# Figure + report
# --------------------------------------------------------------------------- #
def _plot(df: pd.DataFrame, persubj: dict, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [m for m, _, _ in METRICS]
    fig_w = 190 / 25.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.55))
    data = [persubj[f"rho_{m}"][np.isfinite(persubj[f"rho_{m}"])] for m in order]
    parts = ax.violinplot(data, showmeans=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#9ecae1"); pc.set_edgecolor("#3182bd"); pc.set_alpha(0.7)
    for i, m in enumerate(order, 1):
        row = df[df.metric == m].iloc[0]
        ax.plot(i, row.group_mean_rho, "o", color="black", ms=4, zorder=5)
        q = row.fdr_q_dynamics
        star = "*" if np.isfinite(q) and q < 0.05 else ""
        ax.annotate(f"{row.group_mean_rho:+.2f}{star}", (i, row.group_mean_rho),
                    textcoords="offset points", xytext=(6, 0), fontsize=6.5, va="center")
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("per-subject Spearman ρ (metric vs cyto7 type)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Subject-level MEG dynamics gradients vs cyto7 v9 "
                 "(● group mean; * FDR q<0.05, spin)", fontsize=7.5)
    fig.tight_layout()
    p = OUT_SF / "meg_subjectlevel_rho.png"
    fig.savefig(p, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {p}")


def _write_report(df, kept, valid, valid_excl, n_spin, log) -> None:
    def fmt(m):
        r = df[df.metric == m].iloc[0]
        return (f"ρ_grp={r.group_mean_rho:+.3f} (med {r.group_median_rho:+.3f}), "
                f"spin p={r.spin_p:.3f}, q_dyn={r.fdr_q_dynamics:.3f}, "
                f"frac in predicted dir={r.frac_predicted:.2f}, "
                f"released group-map ρ={r.released_map_rho:+.3f} (p={r.released_map_spin_p:.3f})")
    lines = []
    A = lines.append
    A("# Report — subject-level MEG band-power / timescale gradients vs cyto7 (Option B)\n")
    A(f"Implements `docs/SPEC_hcp_meg_bandpower_subjectlevel.md`. Env: `cyto7`. "
      f"Map: **cyto7 v9**. Subjects: **n={len(kept)}** HCP-MEG. "
      f"Spin: released Alexander-Bloch fsLR 32k, n_perm={n_spin}, seed={SEED}.\n")
    A("**Not the aperiodic exponent.** This is band-limited (8 megconnectome power "
      "envelopes); it re-tests the dynamics metrics per subject for power, and does "
      "not replace the Option-A sensor→beamformer→specparam analysis.\n")
    A("## Method")
    A(f"- **Data:** HCP-MEG `bfblpenv` beamformer band-power envelopes, "
      f"{FS_HZ:.2f} Hz envelope sampling, ~296 s/run, up to 3 runs (run-averaged), "
      f"fs_LR 4k source mesh (8004 sources; L 0–4001, R 4002–8003, verified vs "
      f"neuromaps hcps1200 4k).")
    A(f"- **Bands (megconnectome 3.0 defaults; not in CIFTI metadata):** " +
      ", ".join(f"{b} {BAND_EDGES[b][0]}–{BAND_EDGES[b][1]} Hz" for b in ALL_BANDS) + ".")
    A(f"- **Canonical mapping:** δ=delta, θ=theta, α=alpha, β=betalow+betahigh, "
      f"γ1=gammalow; γmid/γhigh enter only the broadband envelope for the timescale. "
      f"Centroid band centres (Hz): " +
      ", ".join(f"{c}={CANON_CENTRE[c]:.1f}" for c in CANON_ORDER) + ".")
    A("- **Relative band power** (band/broadband per vertex) is used for the band-power "
      "metrics to remove the per-vertex leadfield/depth multiplicative confound that "
      "dominates absolute MEG source power; SF ratio and centroid are ratios so are "
      "unaffected. (Absolute single-subject power anti-correlates with the normalized "
      "neuromaps band maps precisely because of this confound.)")
    A("- **Timescale:** lag at which the ACF of the broadband power envelope (sum of 8 "
      "band envelopes) falls to 1/e; an envelope-based approximation of neuromaps "
      "`megtimescale`.")
    A("- **Alignment (Stage 2):** cyto7 v9 32k type map aggregated to 4k by nearest "
      "fs_LR sphere vertex; the released 32k rotations are downsampled the same way, so "
      "the null keeps the published rotations and observed/null consistency.")
    A(f"- **Valid vertices:** {int(valid.sum())} labelled 4k cortical sources "
      f"(allocortex-excluded {int(valid_excl.sum())}).")
    A("- **FDR:** Benjamini-Hochberg within a NEW 'dynamics' family (all subject-level "
      "metrics); released structure-function / external-validation / receptor / "
      "frequency families are untouched.\n")
    A("## Results (pre-registered primaries: intrinsic timescale, slow/fast ratio; ρ<0 predicted)")
    A(f"- **Intrinsic timescale:** {fmt('timescale')}")
    A(f"- **Slow/fast ratio (δ+θ)/(β+γ1):** {fmt('sf_ratio')}")
    A("\n### Secondary (reported alongside)")
    A(f"- **Spectral centroid (ρ>0 predicted):** {fmt('centroid')}")
    for c in CANON_ORDER:
        A(f"- **Relative {c} power:** {fmt('rel_'+c)}")
    A("\nFull numbers, per-subject ρ distributions and the allocortex-excluded "
      "sensitivity are in `figures/v9/structure_function/meg_subjectlevel_summary.csv` "
      "and `..._persubject_rho.csv`; the ρ-distribution figure is "
      "`..._meg_subjectlevel_rho.png`.\n")
    A("## Interpretation")
    A("Each metric's group statistic is the mean over subjects of the per-subject "
      "vertexwise Spearman ρ vs ordinal cyto7 type; significance is against the "
      "released spatial-autocorrelation-preserving spin. The `released group-map ρ` "
      "column is the single-group-map value (neuromaps 32k, identical spin) for direct "
      "before/after comparison. A powered null is an acceptable outcome; directions and "
      "FDR are reported plainly whichever way they fall.\n")
    A("## Provenance")
    A(f"- megconnectome pipeline v3.0; envelope fs={FS_HZ:.3f} Hz; seed={SEED}; "
      f"n_perm={n_spin}; specparam NOT used (band-limited). Script: "
      f"`scripts/meg_subjectlevel.py`. Per-subject summaries computed on X: scratch "
      f"(one ~470 MB band-file unpacked at a time, deleted immediately); only group "
      f"CSVs + figure copied into the repo.")
    (RESULTS / "report_meg_subjectlevel.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"  wrote {RESULTS/'report_meg_subjectlevel.md'}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N subjects (pilot)")
    ap.add_argument("--subjects", nargs="*", default=None, help="explicit subject IDs")
    ap.add_argument("--keep-runs", type=int, default=None, help="use only the first K runs (default: all)")
    ap.add_argument("--n-spin", type=int, default=1000)
    ap.add_argument("--analyse-only", action="store_true", help="skip extraction; use cached summaries")
    ap.add_argument("--extract-only", action="store_true", help="extract per-subject summaries, no analysis")
    ap.add_argument("--clear-scratch", action="store_true", help="delete scratch summaries at the end")
    args = ap.parse_args(argv)

    def log(msg):
        print(msg, flush=True)

    if not SERVER.exists():
        log(f"STOP: server not reachable: {SERVER}")
        return 2

    subjects = args.subjects or list_subjects()
    if args.limit:
        subjects = subjects[: args.limit]
    log(f"Subjects to consider: {len(subjects)}")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    if args.analyse_only:
        done = [s for s in subjects if subject_summary_path(s).exists()]
        log(f"analyse-only: {len(done)} cached summaries")
    else:
        done = extract_all(subjects, log, args.keep_runs)

    if args.extract_only:
        log(f"extract-only done ({len(done)} subjects).")
        return 0
    if not done:
        log("STOP: no subject summaries available.")
        return 1

    run_analysis(done, args.n_spin, log)

    if args.clear_scratch:
        import shutil
        shutil.rmtree(SCRATCH, ignore_errors=True)
        log(f"cleared scratch: {SCRATCH}")
    else:
        log(f"NOTE: per-subject summaries kept on scratch ({SCRATCH}); "
            f"rerun with --clear-scratch to remove.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
