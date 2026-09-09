"""External-validation feature rows (SPEC_external_validation.md) — molecular tier.

Adds independent, non-MRI feature maps as extra structure–function rows, run through the
**same** per-type Spearman + spatial-autocorrelation-preserving spin test (Alexander-Bloch,
neuromaps) + Benjamini–Hochberg FDR used by ``summarise_functional_features.py``, against the
canonical cyto7 **v9** type rank (32k fs_LR). Writes a supplementary table (Table S2) and a
magma feature gallery.

Feature maps (each cached to ``resources/neuromaps_cache/`` as 32k fs_LR ``.npy``):
* **gene PC1** — AHBA gene-expression PC1 (``abagen/genepc1``, fsaverage 10k → fs_LR 32k;
  Burt 2018). Caveat: 6 donors, mostly LH; convergent evidence, not primary.
* **receptor PC1** — first PC across the neuromaps PET receptor/transporter maps
  (Hansen-2022 collection); enabled with ``--receptors`` (assembles many mixed-space maps).

Outputs (``figures/v9/structure_function/``):
* ``external_validation_table.csv`` — Table S2 rows (ρ, p_param, p_spin, p_spin_fdr, n).
* ``external_validation_gallery.png`` — surface (magma) + box-by-type per feature.

Run::  conda activate cyto7 && python scripts/external_validation.py --n-spin 1000
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import os
from pathlib import Path

import numpy as np

from cyto7_surface_io import REPO_ROOT, resolve_target_map, surface_path

CACHE = cfg.data_dir() / "neuromaps_cache"
OUT = cfg.results_dir("tables") / "structure_function"
VER = "v9"
DATASET = "Validation210"
WB_DEFAULT = cfg.workbench_dir()


def _wb_on_path():
    wb = os.environ.get("WORKBENCH_BIN") or WB_DEFAULT
    if wb and Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")


def _cache(source, desc, H):
    return CACHE / f"{source}_{desc}_fsLR32k_hemi-{H}.npy"


# --------------------------------------------------------------------------- #
# Fetch + cache
# --------------------------------------------------------------------------- #


def fetch_gene_pc1():
    """AHBA gene PC1 (abagen/genepc1, fsaverage 10k) -> fs_LR 32k, cached."""
    if _cache("abagen", "genepc1", "L").exists() and _cache("abagen", "genepc1", "R").exists():
        print("  gene PC1 already cached.")
        return
    _wb_on_path()
    import nibabel as nib
    from neuromaps import datasets, transforms
    print("  fetching abagen/genepc1 (fsaverage 10k)...")
    src = datasets.fetch_annotation(source="abagen", desc="genepc1")  # (L, R) fsaverage 10k
    print("  resampling fsaverage 10k -> fs_LR 32k (linear)...")
    for H, gii in zip(("L", "R"), src):
        res = transforms.fsaverage_to_fslr(gii, "32k", hemi=H, method="linear")
        data = np.asarray(res[0].agg_data(), float)
        assert data.shape[0] == 32492, data.shape
        np.save(_cache("abagen", "genepc1", H), data)
        print(f"    cached {_cache('abagen','genepc1',H).name} "
              f"(finite={int(np.isfinite(data).sum())})")


# --------------------------------------------------------------------------- #
# Stats — same spin machinery as summarise_functional_features
# --------------------------------------------------------------------------- #


def _labels_32k():
    lab = resolve_target_map(VER, "fs_LR")
    return {"L": np.asarray(lab["L"]), "R": np.asarray(lab["R"])}


def evaluate(features: dict[str, dict[str, np.ndarray]], n_spin: int) -> "pd.DataFrame":
    """Per-type Spearman ρ vs cyto7 type-rank + spin p + BH-FDR, for each feature.

    *features* = ``{key: {"L": arr32k, "R": arr32k}}``. Validity = labelled cortex (1–7)
    & finite feature value (per feature). Reuses the Alexander-Bloch spin on the type-rank map.
    """
    import pandas as pd
    from scipy import stats
    labels = _labels_32k()

    # per-feature validity mask
    valid = {}
    for key, fh in features.items():
        valid[key] = {h: (labels[h] > 0) & np.isfinite(fh[h]) for h in ("L", "R")}

    rows = {}
    for key, fh in features.items():
        m = valid[key]
        rank = np.concatenate([labels[h][m[h]].astype(float) for h in ("L", "R")])
        vals = np.concatenate([fh[h][m[h]] for h in ("L", "R")])
        rho, p_param = stats.spearmanr(vals, rank)
        rows[key] = {"spearman_rho": float(rho), "p_param": float(p_param),
                     "n": int(vals.size), "p_spin": float("nan")}

    if n_spin > 0:
        try:
            from neuromaps.nulls import alexander_bloch
            # spin the FULL type-rank map once (shared rotations); NaN outside labelled cortex
            base_valid = {h: labels[h] > 0 for h in ("L", "R")}
            rank_full = np.concatenate([
                np.where(base_valid[h], labels[h].astype(float), np.nan) for h in ("L", "R")])
            nulls = alexander_bloch(rank_full, atlas="fsLR", density="32k",
                                    n_perm=n_spin, seed=0)
            concat = {key: np.concatenate([features[key][h] for h in ("L", "R")])
                      for key in features}
            vmask = {key: np.concatenate([valid[key][h] for h in ("L", "R")]) for key in features}
            for key in features:
                vv = concat[key][vmask[key]]
                obs = rows[key]["spearman_rho"]
                null = np.empty(n_spin)
                for i in range(n_spin):
                    spun = nulls[:, i][vmask[key]]
                    ok = np.isfinite(spun)
                    null[i] = stats.spearmanr(vv[ok], spun[ok])[0]
                rows[key]["p_spin"] = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_spin + 1))
        except Exception as exc:  # pragma: no cover
            print(f"WARNING: spin test unavailable ({exc!r}); p_spin=NaN.")

    # BH-FDR across the evaluated feature set
    keys = list(features)
    p = np.array([rows[k]["p_spin"] for k in keys])
    q = _bh(p)
    for k, qk in zip(keys, q):
        rows[k]["p_spin_fdr"] = float(qk)
    return pd.DataFrame([{"FeatureKey": k, **rows[k]} for k in keys])


def _bh(p):
    p = np.asarray(p, float); q = np.full(p.shape, np.nan)
    fin = np.isfinite(p)
    if not fin.any():
        return q
    idx = np.where(fin)[0]; m = idx.size
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def render_gallery(features, meta, stats_df, out_path, dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from nilearn import plotting
    from cyto7_surface_io import LABEL_NAMES

    labels = _labels_32k()
    sdf = stats_df.set_index("FeatureKey")
    nfeat = len(features)
    fig = plt.figure(figsize=(11, 4.6 * nfeat)); fig.patch.set_facecolor("white")
    for r, (key, fh) in enumerate(features.items()):
        # surface (LH lateral, magma) — continuous map, so stat_map (allows negatives)
        ax1 = fig.add_subplot(nfeat, 2, r * 2 + 1, projection="3d")
        d = fh["L"].astype(float).copy(); d[labels["L"] == 0] = np.nan
        plotting.plot_surf_stat_map(str(surface_path(DATASET, "L", "inflated")), d, hemi="left",
                                    view="lateral", axes=ax1, cmap="magma", colorbar=True,
                                    symmetric_cbar=False, threshold=None)
        ax1.set_title("LH surface (magma)", fontsize=10, loc="left")
        # box-by-type
        ax2 = fig.add_subplot(nfeat, 2, r * 2 + 2)
        m = {h: (labels[h] > 0) & np.isfinite(fh[h]) for h in ("L", "R")}
        rank = np.concatenate([labels[h][m[h]] for h in ("L", "R")]).astype(int)
        vals = np.concatenate([fh[h][m[h]] for h in ("L", "R")])
        data = [vals[rank == t] for t in range(1, 8)]
        ax2.boxplot(data, showfliers=False, widths=0.6)
        ax2.set_xticklabels([LABEL_NAMES[t - 1][:4] for t in range(1, 8)], rotation=30, fontsize=8)
        s = sdf.loc[key]; sig = "*" if (np.isfinite(s.p_spin_fdr) and s.p_spin_fdr < 0.05) else ""
        ax2.set_title(f"{meta[key]['label']} by cyto7 type "
                      f"(ρ={s.spearman_rho:+.3f}, spin p={s.p_spin:.3f}, q={s.p_spin_fdr:.3f}{sig})",
                      fontsize=9.5)
        ax2.set_xlabel("cyto7 type (allo→konio)", fontsize=8)
    fig.suptitle("External validation — molecular feature maps vs cyto7 v9 type (magma; box-by-type + spin ρ)",
                 fontsize=12, weight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=1000)
    ap.add_argument("--receptors", action="store_true",
                    help="also assemble receptor PC1 from the neuromaps PET maps (heavy).")
    args = ap.parse_args(argv)
    import pandas as pd

    meta = {"genepc1": {"source": "abagen", "desc": "genepc1",
                        "label": "AHBA gene expression PC1"}}
    fetch_gene_pc1()
    features = {"genepc1": {H: np.load(_cache("abagen", "genepc1", H)) for H in ("L", "R")}}

    # BigBrain Hist-G1 histological gradient (BigBrainWarp, fs_LR 32k) — Analysis 1 row.
    if _cache("bigbrain", "histg1", "L").exists():
        features["bigbrain_histg1"] = {H: np.load(_cache("bigbrain", "histg1", H)) for H in ("L", "R")}
        meta["bigbrain_histg1"] = {"source": "bigbrain", "desc": "histg1",
                                   "label": "BigBrain Hist-G1 (histology)"}

    if args.receptors:
        from receptor_pc1 import build_receptor_pc1  # separate module (heavy assembly)
        build_receptor_pc1()
        features["receptorpc1"] = {H: np.load(_cache("hansen", "receptorpc1", H)) for H in ("L", "R")}
        meta["receptorpc1"] = {"source": "hansen", "desc": "receptorpc1",
                               "label": "PET receptor PC1 (Hansen-2022 collection)"}

    print("== per-type Spearman + spin + FDR ==")
    stats_df = evaluate(features, args.n_spin)
    OUT.mkdir(parents=True, exist_ok=True)
    stats_df.insert(1, "Feature", [meta[k]["label"] for k in stats_df["FeatureKey"]])
    stats_df.to_csv(OUT / "external_validation_table.csv", index=False)
    print(stats_df.to_string(index=False))
    print(f"  wrote {OUT/'external_validation_table.csv'}")

    render_gallery(features, meta, stats_df, OUT / "external_validation_gallery.png")
    print("Done.")


if __name__ == "__main__":
    main()
