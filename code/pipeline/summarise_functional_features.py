"""Summarise structural & functional features across the cyto7 cortical types.

This consolidates the old ``neurophysiological_features_*`` scripts into a
single, reproducible analysis. For each cyto7 cortical type it summarises a
panel of nine vertexwise feature maps spanning three modalities, and tests
whether each feature varies systematically along the cytoarchitectural
differentiation axis (allocortex -> koniocortex).

Feature panel
-------------
* **Structural** -- T1w/T2w myelin and cortical thickness (HCP CIFTI dscalars,
  local to this repository).
* **MEG electrophysiology** (Shafiei et al., 2022; ``hcps1200``) -- intrinsic
  timescale and band power (delta, theta, alpha, beta, gamma).
* **fMRI** (Margulies et al., 2016) -- the principal functional connectivity
  gradient.

The MEG/fMRI maps are read from ``resources/neuromaps_cache/`` (populated once
by ``fetch_neuromaps_features.py``), so this script runs offline.

Outputs (written to ``figures/``)
----------------------------------
* ``functional_signatures.png`` -- the main composite figure: representative
  surface maps, a z-scored feature x type "fingerprint" heatmap, and a z-scored
  trend plot along the differentiation axis.
* ``functional_distributions.png`` -- supplementary per-feature split-violin
  dashboard (type x hemisphere).
* ``functional_summary_table.csv`` -- per type x hemisphere descriptive
  statistics plus, per feature, the Spearman trend (value vs cortical-type
  rank) with an optional spin-test (spatial-autocorrelation-preserving) p-value.

Methodological notes
--------------------
* **Validity mask.** Only vertices with genuine data in *every* modality are
  used: labelled cortex (cyto7 1-7) with myelin > 0, thickness > 0 and a
  non-zero functional gradient (the gradient's zeros mark the fs_LR medial
  wall, where surface functional maps have no cortical data). Per-type coverage
  is reported, because allocortex/agranular are only partially sampled.
* **Cross-modal comparability.** Units differ across features, so the heatmap
  and trend plot use per-feature z-scores (computed over the pooled valid
  vertices); the table reports raw units.
* **Statistics.** Per-vertex values are spatially autocorrelated, so a naive
  p-value is anticonservative. The reported significance uses a spin test
  (Alexander-Bloch et al.) that rotates the cortical-type map to build a
  spatially-constrained null; disable with ``--n-spin 0``.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed (works under WSL/CI)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from nilearn import plotting
from scipy import stats

from cyto7_surface_io import (
    LABEL_LEVELS,
    LABEL_NAMES,
    REPO_ROOT,
    load_cached_feature,
    load_cyto7_labels,
    load_myelin_on_surface,
    load_thickness_on_surface,
    resolve_target_map,
    surface_path,
)

# --------------------------------------------------------------------------- #
# Feature registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Feature:
    """A vertexwise feature map and how to load it."""

    key: str  # short identifier used in outputs
    label: str  # full human-readable name
    short: str  # compact label for the heatmap axis
    group: str  # modality group: "Structural" / "MEG" / "fMRI"
    loader: Callable[[str], dict[str, np.ndarray]]  # dataset -> {"L":..,"R":..}


def _neuromaps_loader(source: str, desc: str) -> Callable[[str], dict[str, np.ndarray]]:
    """Build a loader for a cached neuromaps feature (ignores the dataset arg)."""
    return lambda _dataset: load_cached_feature(source, desc)


#: The nine-feature panel, in display order.
FEATURES: list[Feature] = [
    Feature("myelin", "T1w/T2w myelin", "Myelin", "Structural", load_myelin_on_surface),
    Feature("thickness", "Cortical thickness", "Thickness", "Structural", load_thickness_on_surface),
    Feature("timescale", "Intrinsic timescale", "Timescale", "MEG", _neuromaps_loader("hcps1200", "megtimescale")),
    Feature("delta", "Delta power (0.5-4 Hz)", "Delta", "MEG", _neuromaps_loader("hcps1200", "megdelta")),
    Feature("theta", "Theta power (4-8 Hz)", "Theta", "MEG", _neuromaps_loader("hcps1200", "megtheta")),
    Feature("alpha", "Alpha power (8-12 Hz)", "Alpha", "MEG", _neuromaps_loader("hcps1200", "megalpha")),
    Feature("beta", "Beta power (12-30 Hz)", "Beta", "MEG", _neuromaps_loader("hcps1200", "megbeta")),
    Feature("gamma", "Gamma power (30-60 Hz)", "Gamma", "MEG", _neuromaps_loader("hcps1200", "meggamma1")),
    Feature("gradient", "Principal functional gradient", "Gradient", "fMRI", _neuromaps_loader("margulies2016", "fcgradient01")),
]

#: Colour per modality group (used in the trend plot).
GROUP_COLORS = {"Structural": "#1b7837", "MEG": "#762a83", "fMRI": "#2166ac"}


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #


def load_all_features(dataset: str) -> dict[str, dict[str, np.ndarray]]:
    """Load every feature map. Returns ``{feature_key: {"L":.., "R":..}}``."""
    return {feat.key: feat.loader(dataset) for feat in FEATURES}


def build_validity_mask(
    labels: dict[str, np.ndarray], feats: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    """Per-hemisphere boolean mask of vertices with valid data in all modalities.

    A vertex is valid if it is labelled cortex (cyto7 1-7), has positive myelin
    and thickness (their zeros mark the medial wall), a non-zero functional
    gradient (its zeros mark the fs_LR medial wall) and finite values for every
    feature.
    """
    masks: dict[str, np.ndarray] = {}
    for hemi in ("L", "R"):
        mask = labels[hemi] > 0
        mask &= feats["myelin"][hemi] > 0
        mask &= feats["thickness"][hemi] > 0
        mask &= feats["gradient"][hemi] != 0
        for feat in FEATURES:
            mask &= np.isfinite(feats[feat.key][hemi])
        masks[hemi] = mask
    return masks


def build_long_table(
    labels: dict[str, np.ndarray],
    feats: dict[str, dict[str, np.ndarray]],
    valid: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Assemble a tidy per-vertex table (one row per vertex x feature).

    Columns: ``Hemi``, ``Type``, ``TypeRank`` (1-7), ``Feature``, ``Value`` and
    ``Z`` (per-feature z-score over the pooled valid vertices).
    """
    rows = []
    for hemi_key, hemi_name in [("L", "LH"), ("R", "RH")]:
        m = valid[hemi_key]
        lab = labels[hemi_key][m]
        type_names = np.array([LABEL_NAMES[int(v) - 1] for v in lab])
        for feat in FEATURES:
            vals = feats[feat.key][hemi_key][m]
            rows.append(
                pd.DataFrame(
                    {
                        "Hemi": hemi_name,
                        "Type": type_names,
                        "TypeRank": lab.astype(int),
                        "Feature": feat.label,
                        "FeatureKey": feat.key,
                        "Value": vals,
                    }
                )
            )
    df = pd.concat(rows, ignore_index=True)

    # Per-feature z-score over pooled valid vertices.
    df["Z"] = df.groupby("FeatureKey")["Value"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    return df


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values, preserving input order.

    NaN p-values (e.g. when the spin test is disabled) are ignored in the
    ranking and returned as NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if not finite.any():
        return q

    idx = np.where(finite)[0]
    m = idx.size
    order = idx[np.argsort(p[idx])]  # finite indices, ascending p
    ranked = p[order] * m / (np.arange(1, m + 1))
    # Enforce monotonicity from the largest p downwards, then clip to 1.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q


def compute_trend_stats(
    labels: dict[str, np.ndarray],
    feats: dict[str, dict[str, np.ndarray]],
    valid: dict[str, np.ndarray],
    n_spin: int,
    dataset: str,
) -> dict[str, dict[str, float]]:
    """Spearman trend (feature vs cortical-type rank) per feature, with spin test.

    The Spearman correlation is computed vertexwise over the pooled valid
    vertices, between each feature and the ordinal cortical-type rank (1-7).
    When ``n_spin > 0`` a spatial-autocorrelation-preserving spin test rotates
    the cortical-type map to produce a null distribution; otherwise only the
    naive parametric p-value is reported.
    """
    # Pooled valid feature/label vectors for the parametric Spearman.
    rank_vec = np.concatenate([labels[h][valid[h]].astype(float) for h in ("L", "R")])

    out: dict[str, dict[str, float]] = {}
    for feat in FEATURES:
        vals = np.concatenate([feats[feat.key][h][valid[h]] for h in ("L", "R")])
        rho, p_param = stats.spearmanr(vals, rank_vec)
        out[feat.key] = {"spearman_rho": float(rho), "p_param": float(p_param)}

    if n_spin <= 0:
        for feat in FEATURES:
            out[feat.key]["p_spin"] = float("nan")
            out[feat.key]["p_spin_fdr"] = float("nan")
        return out

    # Spin test: build a null by rotating the cortical-type map on the sphere.
    try:
        p_spin = _spin_test_pvalues(labels, feats, valid, n_spin, dataset)
        for feat in FEATURES:
            out[feat.key]["p_spin"] = p_spin[feat.key]
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"WARNING: spin test unavailable ({exc!r}); reporting NaN p_spin.")
        for feat in FEATURES:
            out[feat.key]["p_spin"] = float("nan")

    # Benjamini-Hochberg FDR correction across the feature panel.
    keys = [feat.key for feat in FEATURES]
    q = benjamini_hochberg(np.array([out[k]["p_spin"] for k in keys]))
    for k, qk in zip(keys, q):
        out[k]["p_spin_fdr"] = float(qk)
    return out


def _spin_test_pvalues(
    labels: dict[str, np.ndarray],
    feats: dict[str, dict[str, np.ndarray]],
    valid: dict[str, np.ndarray],
    n_spin: int,
    dataset: str,
) -> dict[str, float]:
    """Compute spin-test p-values for all features (shared rotation set).

    The cortical-type-rank map is spun ``n_spin`` times using sphere rotations
    (Alexander-Bloch et al.); the same rotations serve every feature. Vertices
    outside the validity mask are set to NaN so they are ignored.
    """
    from neuromaps.nulls import alexander_bloch

    # Full-surface (L, R) cortical-type-rank maps with NaN outside valid cortex.
    rank_full = []
    valid_full = []
    for hemi in ("L", "R"):
        r = np.full(labels[hemi].shape, np.nan)
        r[valid[hemi]] = labels[hemi][valid[hemi]].astype(float)
        rank_full.append(r)
        valid_full.append(valid[hemi])
    rank_concat = np.concatenate(rank_full)
    valid_concat = np.concatenate(valid_full)

    # n_spin rotated versions of the rank map (NaNs preserved by the rotation).
    nulls = alexander_bloch(
        rank_concat, atlas="fsLR", density="32k", n_perm=n_spin, seed=0
    )

    observed_rank = rank_concat[valid_concat]
    results: dict[str, float] = {}
    for feat in FEATURES:
        vals = np.concatenate([feats[feat.key][h] for h in ("L", "R")])[valid_concat]
        rho_obs, _ = stats.spearmanr(vals, observed_rank)

        null_rhos = np.empty(n_spin)
        for i in range(n_spin):
            spun = nulls[:, i][valid_concat]
            ok = np.isfinite(spun)
            null_rhos[i], _ = stats.spearmanr(vals[ok], spun[ok])

        # Two-tailed: fraction of |null| >= |observed|.
        p = (np.sum(np.abs(null_rhos) >= abs(rho_obs)) + 1) / (n_spin + 1)
        results[feat.key] = float(p)
    return results


def build_summary_table(
    df: pd.DataFrame, trend: dict[str, dict[str, float]], coverage: pd.DataFrame
) -> pd.DataFrame:
    """Per (feature, type, hemisphere) descriptive stats + trend columns."""
    grouped = (
        df.groupby(["FeatureKey", "Feature", "Type", "Hemi"])["Value"]
        .agg(mean="mean", std="std", median="median", n="count")
        .reset_index()
    )
    grouped["spearman_rho"] = grouped["FeatureKey"].map(lambda k: trend[k]["spearman_rho"])
    grouped["p_param"] = grouped["FeatureKey"].map(lambda k: trend[k]["p_param"])
    grouped["p_spin"] = grouped["FeatureKey"].map(lambda k: trend[k]["p_spin"])
    grouped["p_spin_fdr"] = grouped["FeatureKey"].map(lambda k: trend[k]["p_spin_fdr"])

    # Order rows by the display order of features then the cyto7 axis.
    feat_order = {f.label: i for i, f in enumerate(FEATURES)}
    type_order = {n: i for i, n in enumerate(LABEL_NAMES)}
    grouped["_f"] = grouped["Feature"].map(feat_order)
    grouped["_t"] = grouped["Type"].map(type_order)
    return grouped.sort_values(["_f", "_t", "Hemi"]).drop(columns=["_f", "_t"])


def compute_coverage(
    labels: dict[str, np.ndarray], valid: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Per-type vertex coverage (valid / total, pooled across hemispheres)."""
    rows = []
    for lvl in LABEL_LEVELS:
        total = sum(int(np.sum(labels[h] == lvl)) for h in ("L", "R"))
        kept = sum(int(np.sum((labels[h] == lvl) & valid[h])) for h in ("L", "R"))
        rows.append(
            {
                "Type": LABEL_NAMES[lvl - 1],
                "total": total,
                "valid": kept,
                "coverage_pct": 100.0 * kept / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def _zscore_type_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matrix of mean z-score per (feature x type), for the heatmap."""
    mat = (
        df.groupby(["FeatureKey", "Type"])["Z"]
        .mean()
        .unstack("Type")
        .reindex(index=[f.key for f in FEATURES], columns=LABEL_NAMES)
    )
    mat.index = [f.short for f in FEATURES]
    return mat


def plot_main_composite(
    df: pd.DataFrame,
    labels: dict[str, np.ndarray],
    feats: dict[str, dict[str, np.ndarray]],
    valid: dict[str, np.ndarray],
    coverage: pd.DataFrame,
    trend: dict[str, dict[str, float]],
    dataset: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Render the main composite figure (surface maps + heatmap + trend)."""
    print("Rendering main composite figure...")
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.25], hspace=0.22, wspace=0.25)

    # --- Row 1: representative LH surface maps (inflated) --------------------
    surf = str(surface_path(dataset, "L", "inflated"))
    lab_l = labels["L"].astype(float)
    lab_l_masked = np.where(valid["L"], lab_l, np.nan)

    # cyto7 map (discrete) — central GC categorical palette.
    from figure_style import BANDPOWER, GRADIENT, TIMESCALE, cyto7_listed_cmap
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    plotting.plot_surf_roi(
        surf, lab_l_masked, hemi="left", view="lateral", axes=ax,
        cmap=cyto7_listed_cmap("gc"), vmin=1, vmax=7, colorbar=False,
    )
    ax.set_title("cyto7 types", fontsize=13, fontweight="bold")

    # Three representative features — colormaps from figure_style (sequential for
    # magnitudes: timescale=magma, band power=viridis; diverging for the gradient).
    for col, (key, name, cmap) in enumerate(
        [("timescale", "Intrinsic timescale", TIMESCALE.cmap),
         ("gamma", "Gamma power", BANDPOWER.cmap),
         ("gradient", "Functional gradient", GRADIENT.cmap)],
        start=1,
    ):
        data = np.where(valid["L"], feats[key]["L"], np.nan)
        ax = fig.add_subplot(gs[0, col], projection="3d")
        plotting.plot_surf_stat_map(
            surf, data, hemi="left", view="lateral", axes=ax, cmap=cmap, colorbar=True,
        )
        ax.set_title(name, fontsize=13, fontweight="bold")

    # --- Row 2 left: z-scored fingerprint heatmap ---------------------------
    ax_heat = fig.add_subplot(gs[1, :2])
    mat = _zscore_type_matrix(df)
    sns.heatmap(
        mat, ax=ax_heat, cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5,
        annot=True, fmt=".2f", annot_kws={"fontsize": 8},
        cbar_kws={"label": "mean z-score", "shrink": 0.7},
        linewidths=0.5, linecolor="white",
    )
    ax_heat.set_title("Feature fingerprint per cortical type (z-scored)", fontsize=14, fontweight="bold")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=35, ha="right")

    # --- Row 2 right: z-scored trend along the differentiation axis ---------
    ax_trend = fig.add_subplot(gs[1, 2:])
    x = np.array(LABEL_LEVELS)
    for feat in FEATURES:
        sub = df[df["FeatureKey"] == feat.key]
        gmean = sub.groupby("TypeRank")["Z"].mean().reindex(x)
        gsem = sub.groupby("TypeRank")["Z"].sem().reindex(x)
        # FDR-significant trends are drawn solid+opaque; non-significant ones
        # faint+dashed, with an asterisk in the legend.
        q = trend[feat.key].get("p_spin_fdr", float("nan"))
        sig = np.isfinite(q) and q < 0.05
        ax_trend.errorbar(
            x, gmean.values, yerr=1.96 * gsem.values,
            marker="o", markersize=4, capsize=2,
            color=GROUP_COLORS[feat.group],
            linewidth=2.2 if sig else 1.0,
            linestyle="-" if sig else "--",
            alpha=0.95 if sig else 0.5,
            label=f"{feat.short}*" if sig else feat.short,
        )
    ax_trend.axhline(0, color="0.6", linewidth=0.8, linestyle="--")
    ax_trend.set_xticks(x)
    ax_trend.set_xticklabels([LABEL_NAMES[i - 1] for i in x], rotation=35, ha="right")
    ax_trend.set_ylabel("z-score (mean ± 95% CI)")
    ax_trend.set_title("Feature progression along the cyto7 axis", fontsize=14, fontweight="bold")
    ax_trend.grid(axis="y", alpha=0.3)
    ax_trend.legend(
        ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        title="solid + * : FDR q<0.05 (spin test);  dashed : n.s.",
        title_fontsize=8,
    )

    # Coverage caption.
    low = coverage[coverage["coverage_pct"] < 95]
    cap = "  ".join(f"{r.Type}: {r.coverage_pct:.0f}%" for r in low.itertuples())
    fig.suptitle(
        "Structural & functional signatures of cytoarchitectural types",
        fontsize=18, fontweight="bold", y=0.98,
    )
    if cap:
        fig.text(0.5, 0.005, f"Partial data coverage (valid vertices): {cap}",
                 ha="center", fontsize=9, style="italic", color="0.3")

    _save(fig, output_path, dpi)


def plot_distribution_dashboard(df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Render the supplementary per-feature split-violin dashboard."""
    print("Rendering distribution dashboard (supplementary)...")
    n = len(FEATURES)
    ncols, nrows = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    hemi_colors = {"LH": "#4f75f2", "RH": "#f24f4f"}

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        sub = df[df["FeatureKey"] == feat.key]
        sns.violinplot(
            ax=ax, data=sub, x="Type", y="Value", hue="Hemi",
            order=LABEL_NAMES, palette=hemi_colors, split=True, inner="quart",
            density_norm="width",
        )
        ax.set_title(f"{feat.label}  [{feat.group}]", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("value")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        if i > 0:
            leg = ax.get_legend()
            if leg:
                leg.remove()

    for j in range(len(FEATURES), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Feature distributions per cortical type and hemisphere",
        fontsize=18, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, output_path, dpi)


def _save(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    """Save *fig* to *output_path* and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
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
        "--dataset", choices=["Validation210", "Parcellation210"], default="Validation210",
        help="Glasser dataset for the local structural maps and surfaces (default: %(default)s).",
    )
    parser.add_argument(
        "--n-spin", type=int, default=1000,
        help="Spin-test permutations for the trend p-value; 0 disables (default: %(default)s).",
    )
    parser.add_argument(
        "--annot-version", default=None,
        help="cyto7 target-map version (v1|v3|...) or {hemi}-path. If set, labels are "
        "resolved (164k->32k nearest) via resolve_target_map instead of the bundled 32k file.",
    )
    parser.add_argument(
        "--out-suffix", default="",
        help="Suffix appended to output filenames (e.g. '_v3').",
    )
    parser.add_argument(
        "--support", choices=["none", "anatomy"], default="none",
        help="Mask low anatomy-only-support vertices before the summaries/trend.",
    )
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir", type=Path, default=cfg.figures_dir(),
        help="Directory for the figures and table (default: <repo>/figures).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Figure resolution (default: %(default)s)."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load features, compute statistics, and render all outputs."""
    args = parse_args(argv)
    print(f"Dataset: {args.dataset}")

    if args.annot_version:
        print(f"Target map: {args.annot_version} (resolved 164k->32k nearest)")
        labels = resolve_target_map(args.annot_version, "fs_LR")
    else:
        labels = {"L": load_cyto7_labels("lh"), "R": load_cyto7_labels("rh")}
    print("Loading features (structural + cached neuromaps)...")
    feats = load_all_features(args.dataset)

    valid = build_validity_mask(labels, feats)
    if args.support == "anatomy":
        from support_io import anatomy_support_32k
        ver = args.annot_version or "v1"
        print(f"Masking vertices with anatomy support < {args.support_threshold} ({ver})")
        conf = anatomy_support_32k(ver)
        valid = {H: valid[H] & (conf[H] >= args.support_threshold) for H in ("L", "R")}
    coverage = compute_coverage(labels, valid)
    print("Per-type coverage:")
    for r in coverage.itertuples():
        print(f"  {r.Type:14s} {r.valid:6d}/{r.total:<6d} ({r.coverage_pct:5.1f}%)")

    df = build_long_table(labels, feats, valid)

    print(f"Computing trend statistics (n_spin={args.n_spin})...")
    trend = compute_trend_stats(labels, feats, valid, args.n_spin, args.dataset)
    for feat in FEATURES:
        t = trend[feat.key]
        sig = "*" if np.isfinite(t["p_spin_fdr"]) and t["p_spin_fdr"] < 0.05 else " "
        print(f"  {feat.short:10s} rho={t['spearman_rho']:+.3f}  "
              f"p_param={t['p_param']:.2e}  p_spin={t['p_spin']:.3f}  "
              f"q_fdr={t['p_spin_fdr']:.3f} {sig}")

    # Outputs (version-suffixed when requested).
    sfx = args.out_suffix
    table = build_summary_table(df, trend, coverage)
    table_path = args.output_dir / f"functional_summary_table{sfx}.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    print(f"  saved {table_path}")

    plot_main_composite(df, labels, feats, valid, coverage, trend, args.dataset,
                        args.output_dir / f"functional_signatures{sfx}.png", args.dpi)
    plot_distribution_dashboard(df, args.output_dir / f"functional_distributions{sfx}.png", args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
