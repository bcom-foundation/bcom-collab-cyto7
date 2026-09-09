#!/usr/bin/env python
"""Build group-mean v2 per-source maps (32k fs_LR) for the MEG figures.

SPEC_meg_v2_figures. Reads the persisted per-subject v2 outputs
($CYTO7_DATA_DIR/_scratch/v2/persist/sub-*_dynamics.mat) — NO re-beamforming — and
produces group-mean maps of the intrinsic timescale (ACF-area, ms), aperiodic
exponent, spectral centroid, aperiodic-corrected beta power, and the alpha power
(for a localization sanity map), on the 4k MEG source mesh, then upsamples to
32k fs_LR (nearest sphere vertex) so the figure scripts can overlay them on the
32k surfaces with cyto7 borders. Saves .npy per hemi under
figures/v9/structure_function/meg_v2_maps/.

The maps are for VISUALISATION of the spatial gradient; the ρ/spin-p/FDR-q that
annotate the figures come from the per-subject analysis (meg_dynamics_v2_summary.csv).
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from pathlib import Path

import numpy as np
import scipy.io as sio

from cyto7_surface_io import REPO_ROOT

PERSIST = cfg.data_dir("019-HCP Young MEG data/_scratch/v2/persist")
OUT = cfg.results_dir("tables") / "structure_function" / "meg_v2_maps"
CANON = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma1": (30, 60)}
CANON_CENTRE = {b: 0.5 * (lo + hi) for b, (lo, hi) in CANON.items()}
FIT_LO, FIT_HI = 2.0, 40.0


def _runs(a):
    a = np.asarray(a, float)
    return a[..., None] if a.ndim == 2 else a


def build_4k_maps(log=print):
    mats = sorted(PERSIST.glob("sub-*_dynamics.mat"))
    if not mats:
        raise SystemExit(f"STOP: no persisted mats in {PERSIST}")
    log(f"  loading {len(mats)} persisted subjects...")
    psd_sum = None
    int_area_sum = None
    n = 0
    freqs = None
    for mp in mats:
        m = sio.loadmat(str(mp))
        psd = _runs(m["psd_runs"]).mean(axis=2)          # (8004, nfreq) run-avg
        intr = _runs(m["int_runs"])                       # (8004, 3, nrun)
        area = np.nanmean(intr[:, 1, :], axis=1)          # (8004,) ACF-area, run-avg
        if freqs is None:
            freqs = np.asarray(m["freqs"], float).ravel()
            psd_sum = np.zeros_like(psd)
            int_area_sum = np.zeros_like(area)
        psd_sum += np.nan_to_num(psd)
        int_area_sum += np.nan_to_num(area)
        n += 1
    mean_psd = psd_sum / n                                 # (8004, nfreq)
    int_area = int_area_sum / n                            # seconds

    # specparam (fixed 2-40) on the group-mean PSD -> exponent + osc-beta peaks
    from specparam import SpectralGroupModel
    S = mean_psd.shape[0]
    band = (freqs >= FIT_LO) & (freqs <= FIT_HI)
    fit = np.all(np.isfinite(mean_psd[:, band]) & (mean_psd[:, band] > 0), axis=1)
    idx = np.where(fit)[0]
    exponent = np.full(S, np.nan)
    osc_beta = np.zeros(S)
    fm = SpectralGroupModel(aperiodic_mode="fixed", peak_width_limits=(1.0, 12.0),
                            max_n_peaks=6, min_peak_height=0.05, verbose=False)
    try:
        fm.fit(freqs, mean_psd[idx], freq_range=[FIT_LO, FIT_HI], n_jobs=6)
    except TypeError:
        fm.fit(freqs, mean_psd[idx], freq_range=[FIT_LO, FIT_HI])
    exponent[idx] = fm.get_params("aperiodic", "exponent")
    pk = np.atleast_2d(np.asarray(fm.get_params("peak"), float))
    if pk.size and pk.shape[1] >= 4:
        for cf, pw, bw, si in pk:
            if CANON["beta"][0] <= cf < CANON["beta"][1]:
                osc_beta[idx[int(si)]] += pw

    # centroid + alpha (from mean PSD band means)
    bp = {b: mean_psd[:, (freqs >= lo) & (freqs <= hi)].mean(1) for b, (lo, hi) in CANON.items()}
    broad = sum(bp.values())
    with np.errstate(divide="ignore", invalid="ignore"):
        centroid = np.where(broad > 0, sum(CANON_CENTRE[b] * bp[b] for b in CANON) / broad, np.nan)
        rel_alpha = np.where(broad > 0, bp["alpha"] / broad, np.nan)

    return {"int_area_ms": int_area * 1000.0, "exponent": exponent, "centroid": centroid,
            "osc_beta": osc_beta, "rel_alpha": rel_alpha}


def upsample_4k_to_32k(map4: np.ndarray) -> dict[str, np.ndarray]:
    """4k (8004 = L4002+R4002) -> 32k per hemi via nearest fs_LR sphere vertex."""
    import nibabel as nib
    from neuromaps.datasets import fetch_atlas
    from scipy.spatial import cKDTree
    a4, a32 = fetch_atlas("fsLR", "4k"), fetch_atlas("fsLR", "32k")
    out = {}
    for hi, H in enumerate(("L", "R")):
        s4 = np.asarray(nib.load(str(a4["sphere"][hi])).agg_data()[0], float)
        s32 = np.asarray(nib.load(str(a32["sphere"][hi])).agg_data()[0], float)
        nn = cKDTree(s4).query(s32, k=1)[1]               # (32492,) idx into 4k hemi
        seg = map4[:4002] if H == "L" else map4[4002:8004]
        out[H] = seg[nn]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = print
    log("Building v2 group-mean maps from persisted mats...")
    maps4 = build_4k_maps(log)
    for key, m4 in maps4.items():
        m32 = upsample_4k_to_32k(m4)
        for H in ("L", "R"):
            np.save(OUT / f"{key}_fsLR32k_hemi-{H}.npy", m32[H])
        log(f"  {key}: 4k median={np.nanmedian(m4):.4g} -> 32k saved")
    log(f"wrote v2 maps -> {OUT}")


if __name__ == "__main__":
    main()
