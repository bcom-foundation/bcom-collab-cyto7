"""Shared helpers for the definitional-validation add-ons (SPEC_definitional_validation).

Generates the released Alexander-Bloch spin rotations for the cyto7 v9 type-rank
map (32k fs_LR) **once** (seed 0, n_perm 1000 — identical to the released
pipeline) and caches them, so the three analyses reuse the same rotation set
without recomputing (and without disturbing the running MEG jobs). Provides
`evaluate_cached()`, which mirrors `external_validation.evaluate()`'s per-type
Spearman + spin-p + BH-FDR math exactly but takes the precomputed nulls.

Outputs are confined to figures/v9/definitional/. No released table is touched.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from pathlib import Path

import numpy as np
from scipy import stats

from cyto7_surface_io import REPO_ROOT, resolve_target_map
from external_validation import _bh  # identical BH-FDR used by the released tables

OUT = cfg.results_dir("tables/definitional")
NULLS_CACHE = OUT / "_spin_nulls_v9_fsLR32k_seed0_n1000.npy"
VER = "v9"
SEED = 0
N_SPIN = 1000


def labels_32k() -> dict[str, np.ndarray]:
    lab = resolve_target_map(VER, "fs_LR")
    return {"L": np.asarray(lab["L"]), "R": np.asarray(lab["R"])}


def get_nulls(labels: dict[str, np.ndarray] | None = None,
              n_spin: int = N_SPIN, log=print) -> np.ndarray:
    """Return the (64984, n_spin) rotated type-rank nulls; cache to disk once.

    Identical call to external_validation.evaluate: alexander_bloch on the full
    type-rank map (NaN outside labelled cortex), atlas=fsLR density=32k, seed 0.
    """
    if NULLS_CACHE.exists():
        arr = np.load(NULLS_CACHE)
        if arr.shape[1] >= n_spin:
            log(f"  reusing cached spin nulls {NULLS_CACHE.name} {arr.shape}")
            return arr[:, :n_spin]
    labels = labels or labels_32k()
    from neuromaps.nulls import alexander_bloch
    rank_full = np.concatenate([
        np.where(labels[h] > 0, labels[h].astype(float), np.nan) for h in ("L", "R")])
    log(f"  generating spin nulls (alexander_bloch fsLR 32k, n_perm={n_spin}, seed={SEED}) ONCE...")
    nulls = alexander_bloch(rank_full, atlas="fsLR", density="32k", n_perm=n_spin, seed=SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(NULLS_CACHE, nulls.astype(np.float32))
    log(f"  cached {NULLS_CACHE.name} {nulls.shape}")
    return nulls


def evaluate_cached(features: dict[str, dict[str, np.ndarray]],
                    nulls: np.ndarray, labels: dict[str, np.ndarray] | None = None):
    """Per-type Spearman ρ + spin-p (via precomputed nulls) + BH-FDR (one family).

    Mirrors external_validation.evaluate() exactly; BH is computed across the
    feature set passed in this call (so call once per FDR family).
    """
    import pandas as pd
    labels = labels or labels_32k()
    valid = {k: {h: (labels[h] > 0) & np.isfinite(fh[h]) for h in ("L", "R")}
             for k, fh in features.items()}
    rows = {}
    for k, fh in features.items():
        m = valid[k]
        rank = np.concatenate([labels[h][m[h]].astype(float) for h in ("L", "R")])
        vals = np.concatenate([fh[h][m[h]] for h in ("L", "R")])
        rho, p_param = stats.spearmanr(vals, rank)
        rows[k] = {"spearman_rho": float(rho), "p_param": float(p_param),
                   "n": int(vals.size), "p_spin": float("nan")}
    n_spin = nulls.shape[1]
    concat = {k: np.concatenate([features[k][h] for h in ("L", "R")]) for k in features}
    vmask = {k: np.concatenate([valid[k][h] for h in ("L", "R")]) for k in features}
    for k in features:
        vv = concat[k][vmask[k]]
        obs = rows[k]["spearman_rho"]
        null = np.empty(n_spin)
        for i in range(n_spin):
            spun = nulls[:, i][vmask[k]]
            ok = np.isfinite(spun)
            null[i] = stats.spearmanr(vv[ok], spun[ok])[0]
        rows[k]["p_spin"] = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_spin + 1))
    keys = list(features)
    q = _bh(np.array([rows[k]["p_spin"] for k in keys]))
    for k, qk in zip(keys, q):
        rows[k]["p_spin_fdr"] = float(qk)
    return pd.DataFrame([{"FeatureKey": k, **rows[k]} for k in keys])


def evaluate_allo_excluded(features, nulls, labels=None):
    """Same as evaluate_cached but restricted to types 2-7 (allocortex-excluded)."""
    labels = labels or labels_32k()
    lab2 = {h: np.where(labels[h] > 1, labels[h], 0) for h in ("L", "R")}
    return evaluate_cached(features, nulls, lab2)
