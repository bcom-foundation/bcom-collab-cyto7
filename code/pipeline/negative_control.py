"""Analysis 2 — pre-declared negative control (spin-test specificity).

SPEC_definitional_validation, Analysis 2. A geometric map with no
cytoarchitectural meaning should NOT track cyto7 type: pre-registered
expected-null (ρ ≈ 0, non-significant under the spin). Demonstrates the released
spin test does not manufacture positives on spatially-autocorrelated maps.

- **sulcal depth** (primary, pre-registered expected-null): neuromaps fs_LR 32k
  `desc-sulc` surface shape.
- **cortical curvature** (secondary): discrete mean-curvature proxy from the
  fs_LR 32k midthickness mesh (uniform-Laplacian magnitude).

Reused: definitional_common.evaluate_cached() (identical to
external_validation.evaluate) with the cached seed-0 / n=1000 rotations.
Reported standalone (NOT FDR-pooled with the definitional tests).

Run::  conda activate cyto7 && python scripts/negative_control.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from pathlib import Path

import numpy as np

from cyto7_surface_io import REPO_ROOT, LABEL_NAMES
import definitional_common as dc

OUT = cfg.results_dir("tables/definitional")


def load_sulc() -> dict[str, np.ndarray]:
    import nibabel as nib
    from neuromaps.datasets import fetch_atlas
    a = fetch_atlas("fsLR", "32k")
    out = {}
    for hi, H in enumerate(("L", "R")):
        g = nib.load(str(a["sulc"][hi]))
        out[H] = np.asarray(g.agg_data(), float)
        assert out[H].shape[0] == 32492, out[H].shape
    return out


def load_curvature() -> dict[str, np.ndarray]:
    """Uniform-Laplacian mean-curvature proxy from the fs_LR 32k midthickness mesh."""
    import nibabel as nib
    from neuromaps.datasets import fetch_atlas
    a = fetch_atlas("fsLR", "32k")
    out = {}
    for hi, H in enumerate(("L", "R")):
        g = nib.load(str(a["midthickness"][hi]))
        coords, faces = g.agg_data()
        coords = np.asarray(coords, float)
        faces = np.asarray(faces, int)
        n = coords.shape[0]
        # accumulate neighbour-position sums and counts
        neigh_sum = np.zeros_like(coords)
        deg = np.zeros(n)
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            i, j = faces[:, a_], faces[:, b_]
            np.add.at(neigh_sum, i, coords[j]); np.add.at(deg, i, 1)
            np.add.at(neigh_sum, j, coords[i]); np.add.at(deg, j, 1)
        deg[deg == 0] = 1
        lap = neigh_sum / deg[:, None] - coords          # umbrella Laplacian
        out[H] = 0.5 * np.linalg.norm(lap, axis=1)        # |mean curvature| proxy
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = print
    labels = dc.labels_32k()
    nulls = dc.get_nulls(labels, log=log)

    log("== Analysis 2 — negative control (sulcal depth primary; curvature secondary) ==")
    feats = {"sulc": load_sulc()}
    try:
        feats["curv"] = load_curvature()
    except Exception as exc:
        log(f"  [WARN] curvature proxy unavailable ({exc!r}); reporting sulc only")

    # Reported standalone: evaluate each on its own (no cross-FDR pooling with definitional tests).
    df = dc.evaluate_cached(feats, nulls, labels)
    df_ex = dc.evaluate_allo_excluded(feats, nulls, labels)
    df = df.merge(df_ex[["FeatureKey", "spearman_rho", "p_spin"]]
                  .rename(columns={"spearman_rho": "spearman_rho_allo_excl",
                                   "p_spin": "p_spin_allo_excl"}), on="FeatureKey")
    LABELS = {"sulc": "Sulcal depth (pre-registered expected-null)",
              "curv": "Cortical curvature (|mean-curv| proxy)"}
    df.insert(1, "Feature", [LABELS.get(k, k) for k in df["FeatureKey"]])
    df["prereg"] = ["expected-null" if k == "sulc" else "" for k in df["FeatureKey"]]
    df.to_csv(OUT / "negative_control.csv", index=False)
    log(df.to_string(index=False))
    log(f"  wrote {OUT/'negative_control.csv'}")

    _plot(feats, df, labels)
    _one_liner(df, log)
    return 0


def _plot(feats, df, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sdf = df.set_index("FeatureKey")
    keys = list(feats)
    fig_w = 190 / 25.4
    fig, axes = plt.subplots(1, len(keys), figsize=(fig_w, fig_w * 0.42), squeeze=False)
    for ax, k in zip(axes[0], keys):
        m = {h: (labels[h] > 0) & np.isfinite(feats[k][h]) for h in ("L", "R")}
        rank = np.concatenate([labels[h][m[h]] for h in ("L", "R")]).astype(int)
        vals = np.concatenate([feats[k][h][m[h]] for h in ("L", "R")])
        ax.boxplot([vals[rank == t] for t in range(1, 8)], showfliers=False, widths=0.6)
        ax.set_xticklabels([LABEL_NAMES[t - 1][:4] for t in range(1, 8)], rotation=35, ha="right",
                           fontsize=6.5)
        s = sdf.loc[k]
        ax.set_title(f"{k}: rho={s.spearman_rho:+.3f}, spin p={s.p_spin:.3f}", fontsize=7.5)
        ax.set_xlabel("cyto7 type (allo->konio)", fontsize=7)
        ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(str(OUT / "negative_control.png"), dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT/'negative_control.png'}")


def _one_liner(df, log):
    s = df.set_index("FeatureKey").loc["sulc"]
    log(f"\n  PAPER LINE: a pre-registered negative control (sulcal depth) showed "
        f"{'no' if s.p_spin >= 0.05 else 'UNEXPECTED'} alignment with type "
        f"(rho = {s.spearman_rho:+.3f}, spin p = {s.p_spin:.3f}), confirming the spin "
        f"test does not manufacture positives on autocorrelated maps.")


if __name__ == "__main__":
    raise SystemExit(main())
