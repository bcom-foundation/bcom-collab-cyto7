"""Analysis 3 — BigBrain per-layer thickness (redo the inconclusive §4.9 histological check).

SPEC_definitional_validation, Analysis 3. Layer-IV **thickness** (and its fraction of
total cortical thickness) should RISE with cyto7 type — the direct histological
counterpart of the definitional criterion. Pre-registered primary: L4 thickness ρ > 0.

Data: Wagstyl 2020 per-layer thickness (6 layers) on tpl-fs_LR den-32k, via
BigBrainWarp. Cached as
`resources/neuromaps_cache/bigbrain_layer{1..6}_thickness_fsLR32k_hemi-{L,R}.npy`
(same cache location/convention as the intensity profiles used by
`bigbrain_profile_validation.py`).

Reuses definitional_common.evaluate_cached() (identical to external_validation.evaluate)
with the cached seed-0 / n=1000 rotations. Own BigBrain-layer-IV FDR family
{L4 thickness, L4 fraction}. Allocortex-excluded sensitivity reported.

**Honest caveat (per SPEC):** BigBrain is a SINGLE specimen warped to a group surface —
the exact limitation that made §4.9 inconclusive. If L4 thickness survives the spin,
§4.9 upgrades to a positive histological validation; if not, the single-specimen
limitation stands. Either outcome is publishable.

STOP-AND-LOG: the six layer-thickness maps are not present in this repo's cache and
are not distributed via neuromaps; they require BigBrainWarp's `BBW_BigData.zip`
(sciebo) layer-thickness files. If absent, this script writes a STOP note rather than
fabricating. To enable: fetch the Wagstyl layer thicknesses on tpl-fs_LR den-32k and
save them to the cache paths below, then re-run.

Run::  conda activate cyto7 && python scripts/bigbrain_layer_thickness.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from pathlib import Path

import numpy as np

from cyto7_surface_io import REPO_ROOT, LABEL_NAMES
import definitional_common as dc

CACHE = cfg.data_dir() / "neuromaps_cache"
OUT = cfg.results_dir("tables/definitional")
LAYERS = [1, 2, 3, 4, 5, 6]


def _layer_path(n: int, H: str) -> Path:
    return CACHE / f"bigbrain_layer{n}_thickness_fsLR32k_hemi-{H}.npy"


def _have_layers() -> bool:
    return all(_layer_path(n, H).exists() for n in LAYERS for H in ("L", "R"))


def load_layers() -> dict[int, dict[str, np.ndarray]]:
    return {n: {H: np.load(_layer_path(n, H)).astype(float) for H in ("L", "R")} for n in LAYERS}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = print
    if not _have_layers():
        msg = (
            "STOP-AND-LOG (Analysis 3): BigBrain per-layer thickness maps are not in the "
            "cache and are not available via neuromaps. Expected files (not found):\n  "
            + "\n  ".join(str(_layer_path(n, H)) for n in LAYERS for H in ("L", "R"))
            + "\n\nThese require BigBrainWarp's BBW_BigData.zip (sciebo) Wagstyl-2020 layer "
              "thickness on tpl-fs_LR den-32k (the profiles/Hist-G1 were cached from the same "
              "source, but the raw download is no longer on disk and layer thickness was never "
              "cached). Re-downloading BBW_BigData.zip was out of scope for these light add-ons "
              "and would not be run while the MEG jobs occupy the machine. Once the six layer "
              "maps are saved to the paths above, re-running this script computes L4 thickness "
              "(primary, rho>0) and L4 fraction vs cyto7 v9 with the released spin. The §4.9 "
              "single-specimen limitation therefore stands for now (an acceptable, honest "
              "outcome per the SPEC)."
        )
        (OUT / "bigbrain_layer_thickness.STOP.txt").write_text(msg, encoding="utf-8")
        log(msg)
        return 3

    labels = dc.labels_32k()
    nulls = dc.get_nulls(labels, log=log)
    lay = load_layers()
    total = {H: sum(lay[n][H] for n in LAYERS) for H in ("L", "R")}
    with np.errstate(divide="ignore", invalid="ignore"):
        l4_frac = {H: np.where(total[H] > 0, lay[4][H] / total[H], np.nan) for H in ("L", "R")}
    feats = {"L4_thickness": lay[4], "L4_fraction": l4_frac}

    df = dc.evaluate_cached(feats, nulls, labels)
    df_ex = dc.evaluate_allo_excluded(feats, nulls, labels)
    df = df.merge(df_ex[["FeatureKey", "spearman_rho", "p_spin"]].rename(
        columns={"spearman_rho": "spearman_rho_allo_excl", "p_spin": "p_spin_allo_excl"}),
        on="FeatureKey")
    df["prereg"] = ["L4 thickness rho>0 (primary)" if k == "L4_thickness" else ""
                    for k in df["FeatureKey"]]
    df.to_csv(OUT / "bigbrain_layer_thickness.csv", index=False)
    log(df.to_string(index=False))
    log(f"  wrote {OUT/'bigbrain_layer_thickness.csv'}")
    _plot(feats, df, labels)
    r = df[df.FeatureKey == "L4_thickness"].iloc[0]
    log(f"\n  PRE-REGISTERED PRIMARY L4 thickness: rho={r.spearman_rho:+.3f}, spin p={r.p_spin:.3f}, "
        f"q={r.p_spin_fdr:.3f} -> {'UPGRADES §4.9' if r.spearman_rho>0 and r.p_spin_fdr<0.05 else 'single-specimen limitation stands'}.")
    return 0


def _plot(feats, df, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sdf = df.set_index("FeatureKey")
    fig_w = 190 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w * 0.42))
    for ax, k in zip(axes, ["L4_thickness", "L4_fraction"]):
        m = {H: (labels[H] > 0) & np.isfinite(feats[k][H]) for H in ("L", "R")}
        rank = np.concatenate([labels[H][m[H]] for H in ("L", "R")]).astype(int)
        vals = np.concatenate([feats[k][H][m[H]] for H in ("L", "R")])
        ax.boxplot([vals[rank == t] for t in range(1, 8)], showfliers=False, widths=0.6)
        ax.set_xticklabels([LABEL_NAMES[t - 1][:4] for t in range(1, 8)], rotation=35, ha="right",
                           fontsize=6.5)
        s = sdf.loc[k]
        ax.set_title(f"{k}: rho={s.spearman_rho:+.3f}, spin p={s.p_spin:.3f}, q={s.p_spin_fdr:.3f}",
                     fontsize=7.5)
        ax.set_xlabel("cyto7 type (allo->konio)", fontsize=7)
        ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(str(OUT / "bigbrain_layer_thickness.png"), dpi=600, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT/'bigbrain_layer_thickness.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
