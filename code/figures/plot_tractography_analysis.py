"""Tractography analysis of cyto7 connectivity and bundle geometry (figure layer).

This is the **figure-producing** half of the tractography pipeline. It consumes
three small CSV tables (and one label NIfTI) and tests the Structural-Model
prediction that cortico-cortical connectivity is organised by *cytoarchitectural
type-distance*: regions of similar laminar type connect more strongly and over
shorter, straighter paths than regions far apart on the differentiation axis
(Project Summary section 1).

Inputs (defaults point to the cached tables in ``resources/tractography/``):

* ``cyto7_connectivity_per_bundle.csv`` -- one row per (named HCP bundle,
  source-type, target-type) with a streamline count.
* ``cyto7_connectivity_per_bundle_aggregate.csv`` -- the symmetric 7x7
  type x type connectivity matrix.
* ``cyto7_tract_geometry_per_bundle.csv`` -- per-bundle mean length, mean
  tortuosity, and mean absolute direction cosines (L-R / A-P / I-S).
* ``cyto7_in_reference.nii.gz`` -- the cyto7 *label volume* on the tractography
  reference grid, used for the inter-region boundary-surface analysis.

These CSVs are produced upstream by ``compute_tractography_connectivity.py``
(DIPY + FreeSurfer, against a whole-brain tractogram). That step is heavy and
needs external data, so the tables are cached here and this script reproduces
the figures offline.

Outputs (written to ``figures/tractography/`` by default): length-stratified
connectivity matrices, length distribution, type-distance vs connectivity
scatters, bundle orientation/tortuosity vs type-distance, and -- when the label
volume is given -- the boundary-surface matrix and its correlation with
short-range connectivity. A ``summary.txt`` and several CSVs are also written.

Usage
-----
    conda activate cyto7
    python scripts/plot_tractography_analysis.py            # uses cached resources
    python scripts/plot_tractography_analysis.py --short-cutoff 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

# Some summaries contain non-ASCII characters (e.g. the Spearman rho symbol).
# The Windows console defaults to cp1252 and would crash on them; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover - non-standard stdout
    pass

# Repository root (this file lives in ``<root>/scripts/``).
REPO_ROOT = Path(__file__).resolve().parent.parent
# Directory holding the cached tractography tables + label volume.
TRACTO_DIR = cfg.data_dir() / "tractography"

# --- Constants ---------------------------------------------------------

# Cyto-type ordering used for all matrices and plots. Matches the
# order in García-Cabezas et al. (2020) from least to most laminar
# elaboration. "Allocortex" sits before Agranular by convention here.
CYTO_ORDER = [
    "Allocortex", "Agranular", "Dysgranular",
    "Eulaminate-I", "Eulaminate-II", "Eulaminate-III",
    "Koniocortex",
]
CYTO_LABEL_VALUES = {name: i + 1 for i, name in enumerate(CYTO_ORDER)}

# Length cutoff (mm) separating "short-range" from "long-range" bundles.
# 80 mm reproduces the cached reference figures; override with --short-cutoff.
# (Shorter values, e.g. 40 mm, emphasise U-fibre-like local connections.)
DEFAULT_SHORT_CUTOFF_MM = 80.0

# Seed for the deterministic horizontal jitter in the type-distance scatter.
JITTER_SEED = 0


# --- Loading -----------------------------------------------------------

def load_per_bundle(path: Path) -> pd.DataFrame:
    """Load and tidy the per-bundle connectivity CSV."""
    df = pd.read_csv(path)
    df["Source_Cyto"] = df["Source_Cyto"].astype("category")
    df["Target_Cyto"] = df["Target_Cyto"].astype("category")
    return df


def load_aggregate(path: Path) -> pd.DataFrame:
    """Load the aggregated symmetric matrix; reindex to CYTO_ORDER."""
    df = pd.read_csv(path, index_col=0)
    available = [c for c in CYTO_ORDER if c in df.columns]
    return df.loc[available, available]


def load_geometry(path: Path) -> pd.DataFrame:
    """Load the per-bundle geometry CSV."""
    df = pd.read_csv(path)
    df["Source_Cyto"] = df["Source_Cyto"].astype("category")
    df["Target_Cyto"] = df["Target_Cyto"].astype("category")
    return df


# --- Length-stratified aggregation ------------------------------------

def stratified_matrices(
    geometry: pd.DataFrame,
    short_cutoff_mm: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split bundles by Mean_Length_mm and aggregate streamline counts.

    Each row in `geometry` is one (bundle, source-cyto, target-cyto)
    record with a streamline count and a mean length. We weight each
    record by its streamline count and assign it to short or long
    based on its mean length.

    Returns three symmetric matrices (short, long, all) indexed by
    CYTO_ORDER.
    """
    df = geometry.copy()
    df["Stratum"] = np.where(
        df["Mean_Length_mm"] < short_cutoff_mm, "short", "long")

    available = [c for c in CYTO_ORDER if c in
                 set(df["Source_Cyto"]).union(df["Target_Cyto"])]

    def aggregate(sub: pd.DataFrame) -> pd.DataFrame:
        m = pd.DataFrame(0.0, index=available, columns=available)
        for _, r in sub.iterrows():
            i, j = r["Source_Cyto"], r["Target_Cyto"]
            if i not in m.index or j not in m.columns:
                continue
            n = float(r["Streamline_Count"])
            if i == j:
                m.loc[i, j] += n
            else:
                m.loc[i, j] += n
                m.loc[j, i] += n
        return m

    short = aggregate(df[df["Stratum"] == "short"])
    long_ = aggregate(df[df["Stratum"] == "long"])
    all_ = aggregate(df)
    return short, long_, all_


# --- Boundary surface from label volume -------------------------------

def boundary_surface_matrix(labels_nifti: Path) -> pd.DataFrame:
    """Compute pairwise boundary-surface area between cyto regions.

    For each pair (i, j) of distinct cyto labels, count the number of
    voxel faces where one side has label i and the other has label j.
    Multiply by the in-plane voxel area to convert to mm². The diagonal
    (i, i) is left at zero — adjacency to oneself isn't a boundary.

    Returns a symmetric matrix indexed by CYTO_ORDER labels that are
    actually present in the volume.
    """
    import nibabel as nib

    img = nib.load(str(labels_nifti))
    data = np.asanyarray(img.dataobj).astype(np.int32)
    zooms = img.header.get_zooms()[:3]
    dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    # Face area for each axis: the face perpendicular to axis k has
    # area equal to the product of the other two zooms.
    face_areas = {0: dy * dz, 1: dx * dz, 2: dx * dy}

    label_values = sorted(set(CYTO_LABEL_VALUES.values()))
    name_by_value = {v: k for k, v in CYTO_LABEL_VALUES.items()}
    present = sorted(int(v) for v in np.unique(data) if v in label_values)
    names = [name_by_value[v] for v in present]
    M = pd.DataFrame(0.0, index=names, columns=names)

    for axis, area in face_areas.items():
        a = np.take(data, np.arange(data.shape[axis] - 1), axis=axis)
        b = np.take(data, np.arange(1, data.shape[axis]), axis=axis)
        # Pairs of labels across this axis (both nonzero, different).
        mask = (a > 0) & (b > 0) & (a != b)
        if not mask.any():
            continue
        pairs_a = a[mask]
        pairs_b = b[mask]
        # Use a (label_i, label_j) sorted-pair key to count symmetric.
        lo = np.minimum(pairs_a, pairs_b)
        hi = np.maximum(pairs_a, pairs_b)
        df_pairs = pd.DataFrame({"lo": lo, "hi": hi})
        counts = df_pairs.groupby(["lo", "hi"]).size().reset_index(
            name="n")
        for _, row in counts.iterrows():
            li, lj = int(row["lo"]), int(row["hi"])
            if li in name_by_value and lj in name_by_value:
                ni, nj = name_by_value[li], name_by_value[lj]
                if ni in M.index and nj in M.columns:
                    M.loc[ni, nj] += row["n"] * area
                    M.loc[nj, ni] += row["n"] * area
    return M


# --- Plotting helpers --------------------------------------------------

def plot_matrix(
    M: pd.DataFrame,
    title: str,
    out: Path,
    *,
    cmap: str = "viridis",
    annotate: bool = True,
    log_color: bool = False,
    cbar_label: str = "streamline count",
) -> None:
    """Heatmap with cells annotated."""
    fig, ax = plt.subplots(figsize=(7, 6.2))
    arr = M.values.astype(float)
    if log_color:
        # log1p so zeros don't break the colormap, but show raw counts
        # in the annotations.
        im = ax.imshow(np.log1p(arr), cmap=cmap, aspect="equal")
    else:
        im = ax.imshow(arr, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(M.columns)))
    ax.set_xticklabels(M.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels(M.index)
    if annotate:
        amax = arr.max() if arr.size else 1.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if v == 0:
                    txt = "0"
                elif v >= 1000:
                    txt = f"{int(round(v))}"
                elif v >= 10:
                    txt = f"{v:.0f}"
                else:
                    txt = f"{v:.1f}"
                # color the annotation for contrast against the cell
                ref = np.log1p(v) if log_color else v
                ref_max = np.log1p(amax) if log_color else amax
                col = "white" if (ref_max > 0 and ref > 0.55 * ref_max) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=col)
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(("log(1+x), " if log_color else "") + cbar_label)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_length_distribution(
    geometry: pd.DataFrame, out: Path, short_cutoff_mm: float,
) -> None:
    """Distribution of mean bundle length, weighted by streamline count."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    weights = geometry["Streamline_Count"].astype(float)
    lengths = geometry["Mean_Length_mm"].astype(float)
    ax.hist(lengths, bins=40, weights=weights,
            color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(short_cutoff_mm, ls="--", color="firebrick",
               label=f"short / long cutoff at {short_cutoff_mm:.0f} mm")
    ax.set_xlabel("Mean bundle length (mm)")
    ax.set_ylabel("Streamline count (sum within bin)")
    ax.set_title("Bundle length distribution (streamline-weighted)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_type_distance_vs_count(
    matrix: pd.DataFrame, out: Path, title: str,
) -> None:
    """Scatter: |type_distance| vs (off-diagonal) connection count.

    Tests the gradient prediction: connectivity should fall off with
    type-distance (number of steps along the laminar gradient).
    """
    rows, cols = matrix.index.tolist(), matrix.columns.tolist()
    type_index = {n: i for i, n in enumerate(CYTO_ORDER)}
    xs, ys = [], []
    for i, ri in enumerate(rows):
        for j, rj in enumerate(cols):
            if j <= i:
                continue
            if ri not in type_index or rj not in type_index:
                continue
            d = abs(type_index[ri] - type_index[rj])
            v = matrix.iloc[i, j]
            xs.append(d)
            ys.append(v)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    jitter = np.random.default_rng(JITTER_SEED).normal(0, 0.05, size=xs.shape)
    ax.scatter(xs + jitter,
               ys, alpha=0.7, s=45, color="darkslateblue")
    ax.set_xlabel("Type-distance |i − j| (steps along the gradient)")
    ax.set_ylabel("Streamline count")
    ax.set_title(title)
    if (ys > 0).any():
        # Fit log(1+y) ~ a + b*d for a robust trend line.
        ly = np.log1p(ys)
        if xs.std() > 0:
            b, a = np.polyfit(xs, ly, 1)
            xfit = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xfit, np.expm1(a + b * xfit), color="firebrick",
                    label=f"log-linear fit (slope={b:+.2f})")
            ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_vs_short(
    boundary: pd.DataFrame,
    short_conn: pd.DataFrame,
    out: Path,
) -> Tuple[float, float]:
    """Scatter boundary surface (mm²) vs short-range streamline count.

    Returns (Pearson_r, Spearman_rho) for the off-diagonal pairs that
    appear in both matrices.
    """
    common = [c for c in CYTO_ORDER
              if c in boundary.index and c in short_conn.index]
    B = boundary.loc[common, common].values
    C = short_conn.loc[common, common].values
    xs, ys, labels = [], [], []
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            xs.append(B[i, j])
            ys.append(C[i, j])
            labels.append(f"{common[i][:3]}-{common[j][:3]}")
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    from scipy.stats import pearsonr, spearmanr
    r_p, p_p = pearsonr(xs, ys) if xs.size > 2 else (np.nan, np.nan)
    r_s, p_s = spearmanr(xs, ys) if xs.size > 2 else (np.nan, np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=55, color="seagreen", edgecolor="black",
               linewidth=0.4)
    for x, y, lb in zip(xs, ys, labels):
        ax.annotate(lb, (x, y), fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points",
                    color="dimgray")
    ax.set_xlabel("Boundary surface area (mm²)")
    ax.set_ylabel("Short-range streamline count")
    ax.set_title(
        "Inter-region boundary surface vs short-range connectivity\n"
        f"Pearson r = {r_p:+.2f} (p={p_p:.2g}); "
        f"Spearman ρ = {r_s:+.2f} (p={p_s:.2g})"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return float(r_p), float(r_s)


# --- Orientation and tortuosity by type-distance -----------------------

def _type_distance(source: str, target: str) -> Optional[int]:
    """Return |i − j| along CYTO_ORDER, or None if either label unknown."""
    idx = {n: i for i, n in enumerate(CYTO_ORDER)}
    if source not in idx or target not in idx:
        return None
    return abs(idx[source] - idx[target])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = weights.sum()
    return float((values * weights).sum() / w) if w > 0 else float("nan")


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    w = weights.sum()
    if w <= 0:
        return float("nan")
    m = (values * weights).sum() / w
    var = ((values - m) ** 2 * weights).sum() / w
    return float(np.sqrt(var))


def orientation_tortuosity_by_type_distance(
    geometry: pd.DataFrame,
    short_cutoff_mm: float,
) -> pd.DataFrame:
    """Aggregate orientation components and tortuosity by type-distance.

    Each row of `geometry` is weighted by its Streamline_Count. We
    aggregate within strata defined by (type_distance, length_stratum)
    and report streamline-weighted means and standard deviations.
    """
    df = geometry.copy()
    df["TypeDistance"] = [
        _type_distance(s, t) for s, t in zip(df["Source_Cyto"], df["Target_Cyto"])
    ]
    df = df.dropna(subset=["TypeDistance"]).copy()
    df["TypeDistance"] = df["TypeDistance"].astype(int)
    df["Stratum"] = np.where(
        df["Mean_Length_mm"] < short_cutoff_mm, "short", "long")

    rows = []
    for (td, stratum), sub in df.groupby(["TypeDistance", "Stratum"]):
        w = sub["Streamline_Count"].to_numpy(dtype=float)
        rows.append({
            "TypeDistance": td,
            "Stratum": stratum,
            "n_records": len(sub),
            "n_streamlines": int(w.sum()),
            "mean_LR": _weighted_mean(sub["Dir_X_LR_axis"].to_numpy(), w),
            "mean_AP": _weighted_mean(sub["Dir_Y_AP_axis"].to_numpy(), w),
            "mean_IS": _weighted_mean(sub["Dir_Z_IS_axis"].to_numpy(), w),
            "mean_tortuosity": _weighted_mean(
                sub["Mean_Tortuosity"].to_numpy(), w),
            "std_tortuosity": _weighted_std(
                sub["Mean_Tortuosity"].to_numpy(), w),
            "mean_length_mm": _weighted_mean(
                sub["Mean_Length_mm"].to_numpy(), w),
        })
    return pd.DataFrame(rows).sort_values(
        ["Stratum", "TypeDistance"]).reset_index(drop=True)


def plot_orientation_by_type_distance(
    summary: pd.DataFrame, out: Path, title: str,
) -> None:
    """Stacked bar of normalized orientation components per type-distance.

    Within each stratum (short, long), for each type-distance we show
    a stacked bar of (LR, AP, IS) components, normalized so the three
    stack to 1 (since each is a per-bundle absolute axis component, the
    raw sum is close to but not exactly 1; normalizing keeps the visual
    interpretation honest).
    """
    strata = ["short", "long"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, stratum in zip(axes, strata):
        sub = summary[summary["Stratum"] == stratum].sort_values(
            "TypeDistance")
        if sub.empty:
            ax.set_title(f"{stratum} bundles — no data")
            continue
        td = sub["TypeDistance"].to_numpy()
        lr = sub["mean_LR"].to_numpy()
        ap = sub["mean_AP"].to_numpy()
        is_ = sub["mean_IS"].to_numpy()
        total = lr + ap + is_
        total = np.where(total > 0, total, 1.0)
        lr_n, ap_n, is_n = lr / total, ap / total, is_ / total

        ax.bar(td, lr_n, label="L–R", color="#d95f5f")
        ax.bar(td, ap_n, bottom=lr_n, label="A–P", color="#5fa05f")
        ax.bar(td, is_n, bottom=lr_n + ap_n, label="I–S", color="#5f7fd9")
        ax.set_xticks(td)
        ax.set_xlabel("Type-distance |i − j|")
        ax.set_title(f"{stratum} bundles")
        # Annotate each bar with its streamline count for context.
        ns = sub["n_streamlines"].to_numpy()
        for x, n in zip(td, ns):
            ax.text(x, 1.02, f"n={n}", ha="center", va="bottom",
                    fontsize=8, color="dimgray")
        ax.set_ylim(0, 1.18)
    axes[0].set_ylabel("Fraction of axis-alignment")
    axes[1].legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_is_alignment_vs_typedist(
    summary: pd.DataFrame, out: Path,
) -> Tuple[float, float]:
    """Mean I–S axis alignment vs type-distance, separately by stratum.

    Linear fits are computed over type-distance >= 1 only. Type-distance
    0 (within-region) is shown but excluded from the fit because the
    end-to-end direction of within-region bundles is artifact-dominated
    (a streamline that loops back has near-zero end-to-end vector and
    its sign components are noise).

    Returns (slope_short, slope_long) of weighted linear fits.
    """
    fig, ax = plt.subplots(figsize=(7, 4.4))
    slopes = {}
    colors = {"short": "#1f77b4", "long": "#d62728"}
    for stratum in ["short", "long"]:
        sub = summary[summary["Stratum"] == stratum].sort_values(
            "TypeDistance")
        if sub.empty:
            slopes[stratum] = float("nan")
            continue
        td = sub["TypeDistance"].to_numpy(dtype=float)
        is_ = sub["mean_IS"].to_numpy(dtype=float)
        w = sub["n_streamlines"].to_numpy(dtype=float)
        # Plot all points (including TD=0) but mark TD=0 differently.
        is_zero = td == 0
        ax.scatter(td[is_zero], is_[is_zero],
                   s=np.sqrt(w[is_zero]) * 4, alpha=0.4,
                   color=colors[stratum], marker="x",
                   label=f"{stratum}, TD=0 (excluded from fit)")
        ax.scatter(td[~is_zero], is_[~is_zero],
                   s=np.sqrt(w[~is_zero]) * 4, alpha=0.7,
                   color=colors[stratum], label=f"{stratum}, TD≥1")
        # Fit on TD >= 1 only.
        td_fit = td[~is_zero]
        is_fit = is_[~is_zero]
        w_fit = w[~is_zero]
        if len(td_fit) >= 2 and w_fit.sum() > 0:
            b, a = np.polyfit(td_fit, is_fit, 1, w=w_fit)
            xfit = np.linspace(td_fit.min(), td_fit.max(), 50)
            ax.plot(xfit, a + b * xfit, color=colors[stratum], lw=1.2,
                    alpha=0.85, ls="--")
            slopes[stratum] = float(b)
        else:
            slopes[stratum] = float("nan")
    ax.set_xlabel("Type-distance |i − j|")
    ax.set_ylabel("Mean I–S axis alignment (|Z|)")
    ax.set_title(
        "I–S orientation component vs type-distance\n"
        f"(short slope, TD≥1 = {slopes.get('short', float('nan')):+.3f}; "
        f"long slope, TD≥1 = {slopes.get('long', float('nan')):+.3f})"
    )
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, max(0.85, ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return slopes.get("short", float("nan")), slopes.get("long", float("nan"))


def plot_tortuosity_by_type_distance(
    summary: pd.DataFrame, out: Path,
) -> Tuple[float, float]:
    """Streamline-weighted mean tortuosity vs type-distance.

    Linear fits over type-distance >= 1 only. TD=0 (within-region) is
    artifact-dominated for tortuosity: a streamline that loops back
    near its origin has small end-to-end distance and therefore an
    inflated path/end-to-end ratio.

    Returns (slope_short, slope_long) of weighted linear fits.
    """
    fig, ax = plt.subplots(figsize=(7, 4.4))
    slopes = {}
    colors = {"short": "#1f77b4", "long": "#d62728"}
    for stratum in ["short", "long"]:
        sub = summary[summary["Stratum"] == stratum].sort_values(
            "TypeDistance")
        if sub.empty:
            slopes[stratum] = float("nan")
            continue
        td = sub["TypeDistance"].to_numpy(dtype=float)
        tau = sub["mean_tortuosity"].to_numpy(dtype=float)
        sd = sub["std_tortuosity"].to_numpy(dtype=float)
        w = sub["n_streamlines"].to_numpy(dtype=float)
        # Error bars and points, with TD=0 marked differently.
        is_zero = td == 0
        ax.errorbar(td, tau, yerr=sd, fmt="none",
                    ecolor=colors[stratum], alpha=0.3, elinewidth=1)
        ax.scatter(td[is_zero], tau[is_zero],
                   s=np.sqrt(w[is_zero]) * 4, alpha=0.4,
                   color=colors[stratum], marker="x",
                   label=f"{stratum}, TD=0 (excluded from fit)")
        ax.scatter(td[~is_zero], tau[~is_zero],
                   s=np.sqrt(w[~is_zero]) * 4, alpha=0.85,
                   color=colors[stratum], label=f"{stratum}, TD≥1")
        td_fit = td[~is_zero]
        tau_fit = tau[~is_zero]
        w_fit = w[~is_zero]
        if len(td_fit) >= 2 and w_fit.sum() > 0:
            b, a = np.polyfit(td_fit, tau_fit, 1, w=w_fit)
            xfit = np.linspace(td_fit.min(), td_fit.max(), 50)
            ax.plot(xfit, a + b * xfit, color=colors[stratum], lw=1.2,
                    alpha=0.85, ls="--")
            slopes[stratum] = float(b)
        else:
            slopes[stratum] = float("nan")
    ax.axhline(1.0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("Type-distance |i − j|")
    ax.set_ylabel("Mean tortuosity (path length / end-to-end)")
    ax.set_title(
        "Tortuosity vs type-distance\n"
        f"(short slope, TD≥1 = {slopes.get('short', float('nan')):+.3f}; "
        f"long slope, TD≥1 = {slopes.get('long', float('nan')):+.3f})"
    )
    ax.legend(loc="best", fontsize=8)
    # Cap y-axis so the error bars at TD=0 don't compress everything.
    ax.set_ylim(0, 6)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return slopes.get("short", float("nan")), slopes.get("long", float("nan"))


# --- Main --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # RR33: these four defaults pointed at the untagged 2026-06-19 files that sat directly in
    # resources/tractography/, one level above the per-version directories, so running this
    # script with no arguments analysed a v1-era tractogram. They are now in
    # resources/tractography/_superseded_pre_v9/ and the defaults name v9 explicitly. Every
    # other reader (rr7_tracto_controls.py, rr28_annexg_reproduce.py, rerun_all_v9.py) already
    # named the v9 files, so no released number depends on the old defaults.
    parser.add_argument("--conn-bundle", type=Path,
                        default=TRACTO_DIR / "v9" / "cyto7.v9_connectivity_per_bundle.csv")
    parser.add_argument("--conn-aggregate", type=Path,
                        default=TRACTO_DIR / "v9" / "cyto7.v9_connectivity_per_bundle_aggregate.csv")
    parser.add_argument("--geometry", type=Path,
                        default=TRACTO_DIR / "v9" / "cyto7.v9_tract_geometry_per_bundle.csv")
    parser.add_argument("--labels-nifti", type=Path,
                        default=TRACTO_DIR / "v9" / "cyto7.v9_in_reference.nii.gz",
                        help="cyto7 label volume for the boundary-surface analysis; "
                             "pass a non-existent path to skip it.")
    parser.add_argument("--short-cutoff", type=float,
                        default=DEFAULT_SHORT_CUTOFF_MM)
    parser.add_argument("--out-dir", type=Path,
                        # RR33: was figures/tractography, above the version tree; the released
                        # panels live in figures/v9/tractography and that is where a v9 input
                        # must write, so an output cannot be mistaken for a different run's.
                        default=cfg.figures_dir() / "tractography")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.conn_bundle}")
    per_bundle = load_per_bundle(args.conn_bundle)
    print(f"[load] {args.conn_aggregate}")
    aggregate = load_aggregate(args.conn_aggregate)
    print(f"[load] {args.geometry}")
    geometry = load_geometry(args.geometry)

    # 1. Aggregated connectivity heatmap (as supplied).
    plot_matrix(
        aggregate,
        title="Aggregated connectivity (all bundles, all lengths)",
        out=args.out_dir / "01_connectivity_aggregate.png",
        cmap="viridis", log_color=True,
    )

    # 2. Length-stratified matrices.
    short_M, long_M, all_M = stratified_matrices(
        geometry, args.short_cutoff)
    plot_length_distribution(
        geometry, args.out_dir / "02_length_distribution.png",
        args.short_cutoff)
    plot_matrix(
        short_M,
        title=f"Short-range connectivity (< {args.short_cutoff:.0f} mm)",
        out=args.out_dir / "03_connectivity_short.png",
        cmap="Blues", log_color=True,
    )
    plot_matrix(
        long_M,
        title=f"Long-range connectivity (≥ {args.short_cutoff:.0f} mm)",
        out=args.out_dir / "04_connectivity_long.png",
        cmap="Reds", log_color=True,
    )

    # 3. Type-distance vs connection count, for both strata.
    plot_type_distance_vs_count(
        short_M, args.out_dir / "05_typedist_short.png",
        "Type-distance vs short-range connection count")
    plot_type_distance_vs_count(
        long_M, args.out_dir / "06_typedist_long.png",
        "Type-distance vs long-range connection count")

    # 4. Orientation and tortuosity by type-distance.
    ot_summary = orientation_tortuosity_by_type_distance(
        geometry, args.short_cutoff)
    ot_summary.to_csv(
        args.out_dir / "orientation_tortuosity_summary.csv", index=False)
    plot_orientation_by_type_distance(
        ot_summary,
        args.out_dir / "10_orientation_by_typedist.png",
        "Bundle orientation composition by type-distance "
        "(streamline-weighted)",
    )
    is_short_slope, is_long_slope = plot_is_alignment_vs_typedist(
        ot_summary, args.out_dir / "11_IS_alignment_vs_typedist.png")
    tau_short_slope, tau_long_slope = plot_tortuosity_by_type_distance(
        ot_summary, args.out_dir / "12_tortuosity_vs_typedist.png")

    # 5. Boundary surface analysis (only if a label NIfTI is provided).
    summary_lines = [
        "cyto7 connectivity analysis — summary",
        "=" * 44,
        f"Per-bundle records:    {len(per_bundle):>6}",
        f"Geometry records:      {len(geometry):>6}",
        f"Short/long cutoff:     {args.short_cutoff:.1f} mm",
        f"Sum streamlines short: {short_M.values.sum():>10.0f}",
        f"Sum streamlines long:  {long_M.values.sum():>10.0f}",
        "",
        "Orientation / tortuosity vs type-distance (weighted slopes):",
        f"  I–S alignment, short bundles: {is_short_slope:+.4f} per step",
        f"  I–S alignment, long bundles:  {is_long_slope:+.4f} per step",
        f"  Tortuosity,    short bundles: {tau_short_slope:+.4f} per step",
        f"  Tortuosity,    long bundles:  {tau_long_slope:+.4f} per step",
    ]
    if args.labels_nifti is not None and args.labels_nifti.is_file():
        print(f"[load] {args.labels_nifti}")
        boundary = boundary_surface_matrix(args.labels_nifti)
        boundary.to_csv(args.out_dir / "boundary_surface_mm2.csv")
        plot_matrix(
            boundary,
            title="Inter-region boundary surface (mm²) — the political map",
            out=args.out_dir / "07_boundary_matrix.png",
            cmap="cividis", log_color=False,
            cbar_label="boundary surface area (mm²)",
        )
        r_p, r_s = plot_boundary_vs_short(
            boundary, short_M,
            args.out_dir / "08_boundary_vs_short.png")
        r_pl, r_sl = plot_boundary_vs_short(
            boundary, long_M,
            args.out_dir / "09_boundary_vs_long.png")
        summary_lines += [
            f"Boundary vs short: Pearson r = {r_p:+.3f}, Spearman ρ = {r_s:+.3f}",
            f"Boundary vs long:  Pearson r = {r_pl:+.3f}, Spearman ρ = {r_sl:+.3f}",
        ]
    else:
        summary_lines.append(
            "No --labels-nifti provided; boundary analysis skipped.")

    summary = "\n".join(summary_lines)
    print(summary)
    (args.out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    short_M.to_csv(args.out_dir / "connectivity_short.csv")
    long_M.to_csv(args.out_dir / "connectivity_long.csv")
    all_M.to_csv(args.out_dir / "connectivity_all_recomputed.csv")
    print(f"\n[done] figures and CSVs written to {args.out_dir}")


if __name__ == "__main__":
    main()
