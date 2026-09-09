"""Reproduce the myeloarchitectural progression / validation regression figure.

This regenerates ``final_validation_regression.png``: a per-hemisphere boxplot
of the T1w/T2w myelin proxy across the cyto7 cortical types, ordered by laminar
differentiation, with a linear fit through the neocortical progression
(dysgranular -> koniocortex) overlaid. It tests the hypothesised monotonic
increase of intracortical myelination with cortical-type differentiation.

Methodological lineage and choices
----------------------------------
Several historical figures explored this relationship with different pipelines
(see the repository README). This consolidated script adopts, by default, the
"final" recipe but works entirely on the **32k fs_LR surface** (rather than
sampling the T1w/T2w *volume* at native pial vertices), so it is fully
reproducible from the files in this repository and is co-registered with the
contour-overlay figures:

* **Myelin source.** The HCP bias-corrected surface myelin map
  (``MyelinMap_BC``), read from the CIFTI dscalar (same data as the
  contour-overlay figures).
* **Data-validity mask (always on).** Vertices with myelin == 0 are dropped:
  on the surface map a 0 means "no data" (medial wall), not zero myelin. This
  corrects a flaw in the earlier ``bilateral_neocortical_progression`` figure,
  which left these zeros in and so inflated the allocortex/agranular boxes.
* **Gyral domes & walls** (``--curv-mask``, on by default): keep ``curv <= 0``,
  excluding sulcal fundi (replicates Garcia-Cabezas Fig. 8D).
* **Medial-wall buffer** (``--medial-buffer-mm``, default 3 mm): drop a rim of
  cortex within a geodesic distance of the medial wall.
* **Targeted outlier trimming** (``--trim``, on by default): per type x
  hemisphere, trim the 10% tails for the noisier least-differentiated types
  (allocortex, agranular, dysgranular) and the 2.5% tails for the rest.

The linear fit covers dysgranular -> koniocortex (agranular and allocortex are
plotted for context but excluded from the fit, as they carry motor/limbic
variance unrelated to the sensory-differentiation axis). Both R-squared values
are reported, because they answer different questions and are NOT comparable:

* **per-median R^2** -- fit through the 5 type medians; measures how cleanly the
  *type-level* progression follows a straight line (this is the number shown in
  the original ``final_validation_regression.png``).
* **per-vertex R^2** -- fit through every retained vertex; far lower because it
  also includes within-type scatter (this is the number shown in
  ``bilateral_neocortical_progression.png``).

Run::

    conda activate cyto7
    python scripts/plot_myelin_progression_regression.py

See ``--help`` for options.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed (works under WSL/CI)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from cyto7_surface_io import (
    LABEL_NAMES,
    REPO_ROOT,
    load_curvature_on_surface,
    load_cyto7_labels,
    load_myelin_on_surface,
    load_surface_geometry,
    medial_wall_buffer_mask,
    resolve_target_map,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Types plotted on the x-axis (allocortex is excluded: it sits on the medial
#: wall, where the surface myelin map has little/no data).
PLOT_TYPES: list[str] = LABEL_NAMES[1:]  # agranular ... koniocortex

#: Types entering the linear fit (the monotonic neocortical progression).
FIT_TYPES: list[str] = LABEL_NAMES[2:]  # dysgranular ... koniocortex

#: Types given more aggressive (10%) tail trimming because they are noisier;
#: all other types are trimmed at 2.5%.
NOISY_TYPES: set[str] = {"Allocortex", "agranular", "dysgranular"}


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #


def build_vertex_table(
    dataset: str,
    use_curv_mask: bool,
    medial_buffer_mm: float,
    labels_override: dict[str, np.ndarray] | None = None,
    conf_mask: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Assemble a per-vertex table of myelin values labelled by cyto7 type.

    Applies, in order: the data-validity mask (myelin > 0), the optional gyral
    domes-&-walls mask (``curv <= 0``) and the optional medial-wall geodesic
    buffer.

    Parameters
    ----------
    dataset:
        ``"Validation210"`` or ``"Parcellation210"``.
    use_curv_mask:
        If True, restrict to gyral domes and walls (``curv <= 0``).
    medial_buffer_mm:
        Geodesic buffer (mm) around the medial wall; 0 disables it.

    Returns
    -------
    pandas.DataFrame
        Columns ``Hemi`` ("LH"/"RH"), ``Type`` (cyto7 name), ``T1T2`` (myelin).
    """
    myelin = load_myelin_on_surface(dataset)
    curv = load_curvature_on_surface(dataset) if use_curv_mask else None

    frames: list[pd.DataFrame] = []
    for hemi, gii_hemi in [("LH", "lh"), ("RH", "rh")]:
        key = gii_hemi[0].upper()
        labels = labels_override[key] if labels_override is not None else load_cyto7_labels(gii_hemi)

        # Data-validity mask: labelled cortex with actual myelin data.
        mask = (labels > 0) & (myelin[key] > 0)
        if use_curv_mask:
            mask &= curv[key] <= 0
        if medial_buffer_mm > 0:
            coords, faces = load_surface_geometry(dataset, key, "midthickness")
            mask &= medial_wall_buffer_mask(coords, faces, labels, medial_buffer_mm)
        if conf_mask is not None:
            mask &= conf_mask[key]

        frames.append(
            pd.DataFrame(
                {
                    "Hemi": hemi,
                    "Type": [LABEL_NAMES[int(lbl) - 1] for lbl in labels[mask]],
                    "T1T2": myelin[key][mask],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def trim_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Targeted per-(type, hemisphere) percentile trimming.

    Trims the 10% tails for the noisier least-differentiated types
    (:data:`NOISY_TYPES`) and the 2.5% tails for all others.
    """
    chunks: list[pd.DataFrame] = []
    for (type_name, _hemi), group in df.groupby(["Type", "Hemi"]):
        p_cut = 10.0 if type_name in NOISY_TYPES else 2.5
        lo, hi = np.percentile(group["T1T2"], [p_cut, 100.0 - p_cut])
        chunks.append(group[(group["T1T2"] >= lo) & (group["T1T2"] <= hi)])
    return pd.concat(chunks, ignore_index=True)


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def fit_progression(df: pd.DataFrame) -> dict[str, float]:
    """Fit the neocortical progression two ways and return both R^2 values.

    The x-coordinate of each type is its index within :data:`PLOT_TYPES`
    (agranular = 0, dysgranular = 1, ...), so the fit and the boxplot share an
    axis. Only :data:`FIT_TYPES` (dysgranular -> koniocortex) enter the fit.

    Returns
    -------
    dict
        ``slope``, ``intercept`` and ``p_value`` of the per-vertex fit, plus
        ``r2_vertex`` and ``r2_median``.
    """
    x_of_type = {name: i for i, name in enumerate(PLOT_TYPES)}
    fit_df = df[df["Type"].isin(FIT_TYPES)].copy()
    fit_df["x"] = fit_df["Type"].map(x_of_type)

    # Per-vertex fit (includes within-type scatter).
    vtx = stats.linregress(fit_df["x"], fit_df["T1T2"])

    # Per-median fit (one point per type).
    medians = fit_df.groupby("Type")["T1T2"].median().reindex(FIT_TYPES)
    x_med = np.array([x_of_type[name] for name in FIT_TYPES])
    med = stats.linregress(x_med, medians.values)

    return {
        "slope": vtx.slope,
        "intercept": vtx.intercept,
        "p_value": vtx.pvalue,
        "r2_vertex": vtx.rvalue**2,
        "r2_median": med.rvalue**2,
    }


def myelin_trend_spin(
    labels: dict[str, np.ndarray], dataset: str, n_spin: int, seed: int = 0
) -> float:
    """Spin-test p for the myelin↑type trend across the fitted neocortical types.

    Spatial-autocorrelation-aware (Alexander-Bloch rotations of the cyto7-type
    map on fsLR 32k), matching the functional panel. Spearman(myelin, type-rank)
    over the dysgranular->koniocortex vertices with myelin>0; two-tailed.
    """
    if n_spin <= 0:
        return float("nan")
    from neuromaps.nulls import alexander_bloch

    myelin = load_myelin_on_surface(dataset)
    fit_codes = [LABEL_NAMES.index(n) + 1 for n in FIT_TYPES]
    valid = {H: np.isin(labels[H], fit_codes) & (myelin[H] > 0) for H in ("L", "R")}
    rank = []
    for H in ("L", "R"):
        r = np.full(labels[H].shape, np.nan)
        r[valid[H]] = labels[H][valid[H]].astype(float)
        rank.append(r)
    rank_concat = np.concatenate(rank)
    valid_concat = np.concatenate([valid[H] for H in ("L", "R")])
    mye_concat = np.concatenate([myelin[H] for H in ("L", "R")])
    nulls = alexander_bloch(rank_concat, atlas="fsLR", density="32k",
                            n_perm=n_spin, seed=seed)
    obs = stats.spearmanr(mye_concat[valid_concat], rank_concat[valid_concat])[0]
    nr = np.empty(n_spin)
    for i in range(n_spin):
        spun = nulls[:, i][valid_concat]
        ok = np.isfinite(spun)
        nr[i] = stats.spearmanr(mye_concat[valid_concat][ok], spun[ok])[0]
    return float((np.sum(np.abs(nr) >= abs(obs)) + 1) / (n_spin + 1))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plot_regression(df: pd.DataFrame, fit: dict[str, float], output_path: Path, dpi: int) -> None:
    """Draw the boxplot + neocortical fit and save it."""
    plot_df = df[df["Type"].isin(PLOT_TYPES)]

    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=plot_df,
        x="Type",
        y="T1T2",
        hue="Hemi",
        order=PLOT_TYPES,
        palette="Greys_r",
        width=0.6,
        showfliers=False,
    )

    # Fit line spanning the fitted types (per-vertex slope/intercept).
    x_fit = np.array([PLOT_TYPES.index(name) for name in FIT_TYPES], dtype=float)
    y_fit = fit["slope"] * x_fit + fit["intercept"]
    plt.plot(
        x_fit,
        y_fit,
        "r--",
        linewidth=2.5,
        zorder=5,
        label=(
            f"Neocortical fit  "
            f"$R^2_{{median}}$ = {fit['r2_median']:.3f},  "
            f"$R^2_{{vertex}}$ = {fit['r2_vertex']:.3f}"
        ),
    )

    plt.title("Myeloarchitectural Validation: Neocortical Progression", fontsize=16)
    plt.xlabel("Laminar Differentiation (Dysgranular $\\rightarrow$ Koniocortex)", fontsize=13)
    plt.ylabel("T1/T2 Ratio (Myelin Proxy)", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="lower right", framealpha=1)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        choices=["Validation210", "Parcellation210"],
        default="Validation210",
        help="Glasser HCP group-average dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--curv-mask",
        dest="curv_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict to gyral domes & walls (curv <= 0). Default: on.",
    )
    parser.add_argument(
        "--medial-buffer-mm",
        type=float,
        default=3.0,
        help="Geodesic medial-wall buffer in mm; 0 disables it (default: %(default)s).",
    )
    parser.add_argument(
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Targeted per-type percentile outlier trimming. Default: on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.figures_dir(),
        help="Directory to write the PNG to (default: <repo>/figures).",
    )
    parser.add_argument(
        "--output-name",
        default="final_validation_regression.png",
        help="Output file name (default: %(default)s).",
    )
    parser.add_argument(
        "--annot-version", default=None,
        help="cyto7 target-map version (v1|v3|...) or {hemi}-path; resolved 164k->32k nearest.",
    )
    parser.add_argument("--out-suffix", default="", help="Suffix appended to the output name stem.")
    parser.add_argument(
        "--support", choices=["none", "anatomy"], default="none",
        help="Mask low anatomy-only-support vertices before the fit.",
    )
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument(
        "--n-spin", type=int, default=0,
        help="Spin-test permutations for the myelin trend (0 disables; use 1000 to match the panel).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output resolution (default: %(default)s)."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Assemble the data, fit the progression and render the figure."""
    args = parse_args(argv)

    print(f"Dataset: {args.dataset}")
    print(
        "Masks: data-validity (myelin>0) always on; "
        f"curv<=0={args.curv_mask}; medial_buffer={args.medial_buffer_mm} mm; "
        f"targeted_trim={args.trim}"
    )

    labels_override = None
    if args.annot_version:
        print(f"Target map: {args.annot_version} (resolved 164k->32k nearest)")
        labels_override = resolve_target_map(args.annot_version, "fs_LR")
    conf_mask = None
    if args.support == "anatomy":
        from support_io import anatomy_support_32k
        ver = args.annot_version or "v1"
        print(f"Masking vertices with anatomy support < {args.support_threshold} ({ver})")
        conf = anatomy_support_32k(ver)
        conf_mask = {H: (conf[H] >= args.support_threshold) for H in ("L", "R")}

    df = build_vertex_table(args.dataset, args.curv_mask, args.medial_buffer_mm,
                            labels_override=labels_override, conf_mask=conf_mask)
    if args.trim:
        df = trim_outliers(df)

    fit = fit_progression(df)
    labels_for_spin = labels_override or {"L": load_cyto7_labels("lh"), "R": load_cyto7_labels("rh")}
    spin_p = myelin_trend_spin(labels_for_spin, args.dataset, args.n_spin)
    print(
        f"Neocortical fit (dysgranular->koniocortex): "
        f"R2_median = {fit['r2_median']:.4f}, "
        f"R2_vertex = {fit['r2_vertex']:.4f}, "
        f"p (per-vertex) = {fit['p_value']:.3e}, spin_p = {spin_p:.3f}"
    )

    stem = Path(args.output_name)
    out_name = f"{stem.stem}{args.out_suffix}{stem.suffix}"
    plot_regression(df, fit, args.output_dir / out_name, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
