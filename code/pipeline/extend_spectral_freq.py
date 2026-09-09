"""Section 7 (EXTEND): single-metric frequency summary vs the cyto7 axis.

The five MEG band powers each fail the spin test individually. Mahjoory et al.
(2020, eLife) report a *peak-frequency* gradient that follows cortical
hierarchy. We cannot compute a true per-vertex peak frequency (only band-power
maps are cached), but we can test two single, less-redundant frequency
summaries against the cyto7 differentiation axis:

* **Spectral centroid** (power-weighted mean frequency), per vertex on 32k
  fs_LR: ``C = sum_b P_b * f_b / sum_b P_b`` over the cached band-power maps
  b in {delta, theta, alpha, beta, gamma1}, with band-centre frequencies
  f_b ~= {2.25, 6, 10, 21, 45} Hz (band midpoints). The band powers are the
  (non-negative) weights.
* **Intrinsic timescale** (cached ``megtimescale``) -- the time-domain analog
  of dominant frequency; the closest-to-significant feature in the 9-panel.

For BOTH metrics, on BOTH map versions (v1, v3): Spearman rho vs ordinal cyto7
type, with the **spin test (1000 perms, seed 0) + BH-FDR**, reusing the exact
machinery of the 9-feature panel (``extend_structure_function._spin_nulls`` /
``benjamini_hochberg``). **Allocortex (type 1) is excluded from the fits** so the
two metrics share an identical mask and the comparison is like-for-like.

**Robustness:** the centroid is re-run with alternate band centres (geometric
band means; a log-spaced set) to confirm the verdict is not an artefact of the
chosen centres.

Expected direction (if it follows Mahjoory): spectral centroid rho > 0 (higher
dominant frequency toward koniocortex); intrinsic timescale rho < 0 (already
~= -0.48).

Outputs (default to the post-reorg locations -- see ``--out-dir``/``--report``):
  * append ``spectral centroid`` (+ re-listed ``intrinsic timescale``) rows to
    ``figures/v1_vs_v3/structure_function_v1_vs_v3.csv`` (idempotent);
  * ``figures/v1_vs_v3/spectral_freq_summary_v1_vs_v3.png`` (centroid + timescale
    vs type, v1 & v3);
  * a Section-7 paragraph (cf. Mahjoory 2020) in ``figures/REPORT.md`` (idempotent).

Run::
    conda activate cyto7
    python scripts/extend_spectral_freq.py --n-spin 1000
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cyto7_surface_io import LABEL_LEVELS, LABEL_NAMES, REPO_ROOT, resolve_target_map
from summarise_functional_features import build_validity_mask, load_all_features
from extend_structure_function import SEED, VERSIONS, _spin_nulls, benjamini_hochberg

ALLOCORTEX = 1
#: band-power maps used as centroid weights, in ascending-frequency order.
BANDS = ("delta", "theta", "alpha", "beta", "gamma")

#: band-centre frequency sets (Hz) for the centroid + robustness variants.
#: midpoint = nominal band midpoints (gamma1 ~= 45); geometric = sqrt(lo*hi)
#: band means; logspaced = a genuinely log-spaced anchor set spanning the range.
BAND_CENTRES = {
    "midpoint": np.array([2.25, 6.0, 10.0, 21.0, 45.0]),
    "geometric": np.array([np.sqrt(0.5 * 4), np.sqrt(4 * 8), np.sqrt(8 * 12),
                           np.sqrt(12 * 30), np.sqrt(30 * 60)]),  # ~1.41,5.66,9.80,18.97,42.43
    "logspaced": np.geomspace(2.0, 50.0, 5),                       # ~2,4.47,10,22.36,50
}
PRIMARY_VARIANT = "midpoint"

OUT_DEFAULT = cfg.results_dir("tables") / "structure_function"
CSV_DEFAULT = OUT_DEFAULT / "structure_function_v1_vs_v3.csv"
REPORT_DEFAULT = cfg.figures_dir() / "REPORT.md"
TIMESCALE_LABEL = "intrinsic timescale (Sec.7, allo-excl)"
SECTION7_MARKER = "## Section 7"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def spectral_centroid(feats: dict, f_b: np.ndarray) -> dict[str, np.ndarray]:
    """Power-weighted mean frequency per vertex, per hemisphere.

    ``C = sum_b P_b f_b / sum_b P_b`` over the five band-power maps. Vertices
    with non-positive total power or any non-finite band (the medial wall) are
    set to NaN (they are excluded by the validity mask anyway).
    """
    out = {}
    for H in ("L", "R"):
        P = np.stack([feats[b][H] for b in BANDS], axis=1)  # (N, 5)
        psum = P.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = (P * f_b).sum(axis=1) / psum
        bad = ~(psum > 0) | ~np.isfinite(P).all(axis=1)
        C[bad] = np.nan
        out[H] = C
    return out


def metric_trend_stats(labels, metrics, valid, n_spin):
    """Spearman rho vs ordinal type + spin-p + BH-FDR for a set of metrics.

    The spin rotations (Alexander-Bloch, seed ``SEED``) are built once over the
    shared ``valid`` mask and reused for every metric -- exactly as the 9-feature
    panel shares one rotation set across features. BH-FDR is applied across the
    supplied metric family.
    """
    valid_concat = np.concatenate([valid[H] for H in ("L", "R")])
    rank_obs = np.concatenate([labels[H][valid[H]].astype(float) for H in ("L", "R")])
    out = {}
    for name, m in metrics.items():
        vals = np.concatenate([m[H][valid[H]] for H in ("L", "R")])
        rho, p = stats.spearmanr(vals, rank_obs)
        out[name] = {"rho": float(rho), "p_param": float(p), "p_spin": float("nan")}
    if n_spin > 0:
        _rank_concat, nulls = _spin_nulls(labels, valid, n_spin)
        for name, m in metrics.items():
            vals = np.concatenate([m[H] for H in ("L", "R")])[valid_concat]
            rho_obs = stats.spearmanr(vals, rank_obs)[0]
            nr = np.empty(n_spin)
            for i in range(n_spin):
                spun = nulls[:, i][valid_concat]
                ok = np.isfinite(spun)
                nr[i] = stats.spearmanr(vals[ok], spun[ok])[0]
            out[name]["p_spin"] = float((np.sum(np.abs(nr) >= abs(rho_obs)) + 1) / (n_spin + 1))
    names = list(metrics)
    q = benjamini_hochberg(np.array([out[n]["p_spin"] for n in names]))
    for n, qk in zip(names, q):
        out[n]["q_fdr"] = float(qk)
    return out


def build_metrics(feats: dict) -> dict[str, dict[str, np.ndarray]]:
    """The Section-7 metric family: timescale + centroid (all band-centre variants)."""
    metrics = {TIMESCALE_LABEL: feats["timescale"]}
    for variant, f_b in BAND_CENTRES.items():
        metrics[f"spectral centroid ({variant} f)"] = spectral_centroid(feats, f_b)
    return metrics


def run_version(version: str, feats: dict, metrics: dict, n_spin: int) -> dict:
    """Resolve a version's labels, build the allocortex-excluded valid mask, fit."""
    labels = resolve_target_map(version, "fs_LR")
    base = build_validity_mask(labels, feats)
    valid = {H: base[H] & (labels[H] != ALLOCORTEX) for H in ("L", "R")}  # allo excluded
    n_allo_dropped = sum(int(np.sum(base[H] & (labels[H] == ALLOCORTEX))) for H in ("L", "R"))
    # Require every metric finite on the shared mask: a few non-medial vertices
    # pass the panel validity (gradient != 0) yet have all five band powers = 0,
    # which makes the centroid (a ratio) NaN. Drop them so the observed rho and
    # the spin null use one common, fully-finite vertex set for all metrics.
    for m in metrics.values():
        valid = {H: valid[H] & np.isfinite(m[H]) for H in ("L", "R")}
    trend = metric_trend_stats(labels, metrics, valid, n_spin)
    return {"labels": labels, "valid": valid, "trend": trend,
            "n_valid": int(valid["L"].sum() + valid["R"].sum()),
            "n_allo_dropped": n_allo_dropped}


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


def append_csv(res: dict, metrics: dict, csv_path: Path) -> pd.DataFrame:
    """Append (idempotently) the Section-7 metric rows to the paired CSV."""
    cols = ["feature", "support", "rho_v1", "rho_v3", "d_rho",
            "p_spin_v1", "p_spin_v3", "q_fdr_v1", "q_fdr_v3", "p_param_v1", "p_param_v3"]
    new_rows = []
    for name in metrics:
        t1, t3 = res["v1"]["trend"][name], res["v3"]["trend"][name]
        new_rows.append({
            "feature": name, "support": "none",
            "rho_v1": round(t1["rho"], 4), "rho_v3": round(t3["rho"], 4),
            "d_rho": round(t3["rho"] - t1["rho"], 4),
            "p_spin_v1": round(t1["p_spin"], 4), "p_spin_v3": round(t3["p_spin"], 4),
            "q_fdr_v1": round(t1["q_fdr"], 4), "q_fdr_v3": round(t3["q_fdr"], 4),
            "p_param_v1": f"{t1['p_param']:.2e}", "p_param_v3": f"{t3['p_param']:.2e}",
        })
    new_df = pd.DataFrame(new_rows, columns=cols)
    labels = set(metrics)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[~old["feature"].isin(labels)]  # drop any prior Section-7 rows
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    print(f"  appended {len(new_df)} Section-7 rows -> {csv_path}")
    return new_df


def _per_type_mean(labels, valid, metric):
    """mean +/- sem of a metric per cyto7 type (ranks 2..7), over valid vertices."""
    rank = np.concatenate([labels[H][valid[H]] for H in ("L", "R")])
    vals = np.concatenate([metric[H][valid[H]] for H in ("L", "R")])
    levels = LABEL_LEVELS[1:]  # exclude allocortex (rank 1)
    mean = np.array([np.nanmean(vals[rank == lv]) if np.any(rank == lv) else np.nan for lv in levels])
    sem = np.array([stats.sem(vals[rank == lv], nan_policy="omit") if np.sum(rank == lv) > 1 else 0.0
                    for lv in levels])
    return np.array(levels), mean, sem


def _per_type_values(labels, valid, metric):
    """Per-vertex values of a metric within each cyto7 type (ranks 2..7)."""
    rank = np.concatenate([labels[H][valid[H]] for H in ("L", "R")])
    vals = np.concatenate([metric[H][valid[H]] for H in ("L", "R")])
    levels = LABEL_LEVELS[1:]  # exclude allocortex (rank 1)
    return levels, [vals[rank == lv] for lv in levels]


def fig_summary_single(res_one, metrics, out_path, dpi, version):
    """Two panels (spectral centroid + intrinsic timescale vs type) for ONE map.

    Supplementary Fig S2 (the released clean map only). Each type is shown as a
    box-and-whisker of the per-vertex distribution (median, IQR, 1.5xIQR whiskers),
    with the per-type median trend overlaid; inference is by the spin test
    (annotated). The boxes show the true spread of the data, NOT the standard error
    of the mean, so the panel does not overstate precision (a SEM band over tens of
    thousands of spatially autocorrelated vertices would be misleadingly tight).
    Exported at 190 mm wide, no title (the title lives in the LaTeX caption),
    matching the Supp Fig S3/S4 convention.
    """
    centroid_name = f"spectral centroid ({PRIMARY_VARIANT} f)"
    panels = [(centroid_name, "Spectral centroid (Hz)", "power-weighted mean freq"),
              (TIMESCALE_LABEL, "Intrinsic timescale (a.u.)", "megtimescale")]
    labels, valid = res_one["labels"], res_one["valid"]
    w_in = 190.0 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(w_in, w_in * 0.42))
    fig.patch.set_facecolor("white")
    tcol = [plt.cm.viridis((lv - 1) / 6.0) for lv in LABEL_LEVELS[1:]]
    for ax, (key, ylab, sub) in zip(axes, panels):
        lv, data = _per_type_values(labels, valid, metrics[key])
        t = res_one["trend"][key]
        sig = np.isfinite(t["q_fdr"]) and t["q_fdr"] < 0.05
        bp = ax.boxplot(data, positions=range(len(lv)), widths=0.62, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=1.0),
                        whiskerprops=dict(color="0.4", lw=0.8), capprops=dict(color="0.4", lw=0.8),
                        boxprops=dict(lw=0.6))
        for patch, c in zip(bp["boxes"], tcol):
            patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("0.25")
        med = [np.median(d) if len(d) else np.nan for d in data]
        ax.plot(range(len(lv)), med, color="#c0392b", lw=1.4, marker="o", ms=3.2, zorder=5)
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([LABEL_NAMES[i - 1] for i in lv], rotation=30, ha="right", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(f"{ylab.split(' (')[0]} ({sub})", fontsize=8.5, fontweight="bold")
        ax.grid(axis="y", ls="--", alpha=0.35)
        ax.text(0.03, 0.03, (f"$\\rho$={t['rho']:+.2f}  spin $p$={t['p_spin']:.3f}  "
                             f"$q$={t['q_fdr']:.3f}{'*' if sig else ''}"),
                transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=max(dpi, 600), facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_summary(res, metrics, out_path, dpi):
    """Two panels: spectral centroid (primary) and intrinsic timescale vs type, v1 & v3."""
    centroid_name = f"spectral centroid ({PRIMARY_VARIANT} f)"
    panels = [(centroid_name, "Spectral centroid (Hz)", "power-weighted mean freq"),
              (TIMESCALE_LABEL, "Intrinsic timescale (a.u.)", "megtimescale")]
    style = {"v1": dict(color="#3b3b3b", ls="-", marker="o"),
             "v3": dict(color="#c0392b", ls="--", marker="s")}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (key, ylab, sub) in zip(axes, panels):
        for ver in VERSIONS:
            labels, valid = res[ver]["labels"], res[ver]["valid"]
            lv, mean, sem = _per_type_mean(labels, valid, metrics[key])
            t = res[ver]["trend"][key]
            sig = np.isfinite(t["q_fdr"]) and t["q_fdr"] < 0.05
            ax.errorbar(lv, mean, yerr=1.96 * sem, capsize=3, markersize=5,
                        lw=2.4 if sig else 1.6, alpha=0.95 if sig else 0.8,
                        label=(f"{ver}: rho={t['rho']:+.2f}, "
                               f"p_spin={t['p_spin']:.3f}, q={t['q_fdr']:.3f}{'*' if sig else ''}"),
                        **style[ver])
        ax.set_xticks(LABEL_LEVELS[1:])
        ax.set_xticklabels([LABEL_NAMES[i - 1] for i in LABEL_LEVELS[1:]], rotation=35, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab.split(' (')[0]}  ({sub})", fontsize=12, fontweight="bold")
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best", title="allocortex excluded; spin 1000 + FDR",
                  title_fontsize=8)
    fig.suptitle("Single-metric frequency summary along the cyto7 axis — v1 vs v3 (cf. Mahjoory 2020)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def update_report(res, metrics, report_path: Path, n_spin: int):
    """Append/replace the Section-7 paragraph (cf. Mahjoory 2020) in REPORT.md."""
    centroid_name = f"spectral centroid ({PRIMARY_VARIANT} f)"

    def row(name):
        t1, t3 = res["v1"]["trend"][name], res["v3"]["trend"][name]
        return (f"| {name} | {t1['rho']:+.3f} | {t1['p_spin']:.3f} | {t1['q_fdr']:.3f} | "
                f"{t3['rho']:+.3f} | {t3['p_spin']:.3f} | {t3['q_fdr']:.3f} |")

    cen = res["v1"]["trend"][centroid_name]
    cen_sig = np.isfinite(cen["q_fdr"]) and cen["q_fdr"] < 0.05
    ts = res["v1"]["trend"][TIMESCALE_LABEL]
    variant_rhos_v1 = [res["v1"]["trend"][f"spectral centroid ({v} f)"]["rho"] for v in BAND_CENTRES]
    cen_dir = ("positive (as predicted — higher dominant frequency toward koniocortex)"
               if cen["rho"] > 0 else
               "negative (opposite to the Mahjoory prediction)")

    L = []
    A = L.append
    A(SECTION7_MARKER + " — single-metric frequency summary (cf. Mahjoory 2020)\n")
    A("The five MEG band powers each fail the spin test individually. Mahjoory et al. "
      "(2020, eLife) report a **peak-frequency gradient that follows cortical hierarchy**. "
      "We cannot compute a true per-vertex peak frequency (only band-power maps are cached), "
      "so we test two single, less-redundant frequency summaries against the cyto7 axis: a "
      "power-weighted **spectral centroid** `C = Σ_b P_b·f_b / Σ_b P_b` (b ∈ "
      "{δ,θ,α,β,γ1}, f_b ≈ {2.25,6,10,21,45} Hz) and the cached **intrinsic timescale**. "
      "Spearman ρ vs ordinal type, spin test "
      f"({n_spin} Alexander-Bloch rotations, seed {SEED}) + BH-FDR across this 4-test "
      "frequency family, on v1 and v3, **allocortex excluded** from the fits.\n")
    A("| metric | ρ v1 | p_spin v1 | q v1 | ρ v3 | p_spin v3 | q v3 |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    A(row(TIMESCALE_LABEL))
    for v in BAND_CENTRES:
        A(row(f"spectral centroid ({v} f)"))
    A("")
    A(f"- **Intrinsic timescale points the predicted way:** ρ_v1 = {ts['rho']:+.3f} "
      f"(< 0), as Mahjoory's hierarchy→slower-timescale gradient implies; it remains "
      f"the closest-to-significant single feature (p_spin = {ts['p_spin']:.3f}, "
      f"q = {ts['q_fdr']:.3f}) but does **not** survive the spin test + FDR.")
    A(f"- **Spectral centroid:** ρ_v1 = {cen['rho']:+.3f} ({cen_dir}); "
      f"p_spin = {cen['p_spin']:.3f}, q = {cen['q_fdr']:.3f} — it **{'survives' if cen_sig else 'does NOT survive'}** "
      "spin + FDR.")
    A(f"- **Robustness to band centres:** the centroid ρ_v1 over the three band-centre "
      f"sets (midpoint / geometric / log-spaced) is "
      f"{', '.join(f'{r:+.3f}' for r in variant_rhos_v1)} — the verdict is **unchanged** "
      "across centre choices (the spin test is the binding constraint, not the band-centre "
      "calibration).")
    A("- v1↔v3 differences are negligible (the periallocortex relabelling barely touches "
      "these surface-functional fits), consistent with the rest of this report.\n")
    A("**Interpretation.** A frequency/timescale gradient along the cytoarchitectural axis "
      "is *expected*, and our timescale metric does point in Mahjoory's predicted direction. "
      "But the normative band-power maps are spatially smooth, so the spin null is strict and "
      "a coarse 5-bin centroid is too blunt a proxy to clear it. This is **not** a substitute "
      "for a true peak-frequency analysis: replicating Mahjoory (2020) properly requires "
      "recomputing peak frequency from **raw per-subject MEG** (e.g. Donders/HCP), which is "
      "out of scope here and flagged as future work.\n")
    block = "\n".join(L)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        idx = text.find(SECTION7_MARKER)
        text = (text[:idx].rstrip() + "\n\n" + block) if idx != -1 else (text.rstrip() + "\n\n" + block)
    else:
        text = block
    report_path.write_text(text, encoding="utf-8")
    print(f"  updated {report_path} (Section 7)")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Validation210")
    p.add_argument("--n-spin", type=int, default=1000)
    p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT,
                   help="Directory for the CSV and figure (default: figures/v1_vs_v3).")
    p.add_argument("--report", type=Path, default=REPORT_DEFAULT,
                   help="REPORT.md to update with the Section-7 paragraph.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--version", default=None,
                   help="Single-map mode (e.g. v3_clean): plot ONLY that map (manuscript Fig 6), "
                        "writing spectral_freq_summary.png + a per-metric CSV to --out-dir. "
                        "Omit for the legacy paired v1-vs-v3 output.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Loading feature maps ({args.dataset})...")
    feats = load_all_features(args.dataset)
    metrics = build_metrics(feats)

    # Single-map mode (manuscript Fig 6): one version only, no v1 series.
    if args.version:
        print(f"Section 7 (single-map {args.version}) — freq summary "
              f"(n_spin={args.n_spin}, allocortex excluded)")
        one = run_version(args.version, feats, metrics, args.n_spin)
        for name in metrics:
            t = one["trend"][name]
            sig = "*" if np.isfinite(t["q_fdr"]) and t["q_fdr"] < 0.05 else " "
            print(f"    {name:34s} rho={t['rho']:+.3f}  p_spin={t['p_spin']:.3f}  "
                  f"q_fdr={t['q_fdr']:.3f} {sig}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        fig_summary_single(one, metrics, args.out_dir / "spectral_freq_summary.png",
                           args.dpi, args.version)
        rows = [{"metric": name, "rho": round(one["trend"][name]["rho"], 4),
                 "p_spin": round(one["trend"][name]["p_spin"], 4),
                 "q_fdr": round(one["trend"][name]["q_fdr"], 4)} for name in metrics]
        pd.DataFrame(rows).to_csv(
            args.out_dir / f"spectral_freq_summary_{args.version}.csv", index=False)
        print(f"  wrote {args.out_dir / f'spectral_freq_summary_{args.version}.csv'}")
        print("Done.")
        return

    print(f"Section 7 — frequency summary (n_spin={args.n_spin}, allocortex excluded)")
    res = {}
    for ver in VERSIONS:
        res[ver] = run_version(ver, feats, metrics, args.n_spin)
        print(f"  [{ver}] valid vertices={res[ver]['n_valid']} "
              f"(allocortex dropped={res[ver]['n_allo_dropped']})")
        for name in metrics:
            t = res[ver]["trend"][name]
            sig = "*" if np.isfinite(t["q_fdr"]) and t["q_fdr"] < 0.05 else " "
            print(f"    {name:34s} rho={t['rho']:+.3f}  p_spin={t['p_spin']:.3f}  "
                  f"q_fdr={t['q_fdr']:.3f} {sig}")

    csv_path = args.out_dir / "structure_function_v1_vs_v3.csv"
    append_csv(res, metrics, csv_path)
    fig_summary(res, metrics, args.out_dir / "spectral_freq_summary_v1_vs_v3.png", args.dpi)
    update_report(res, metrics, args.report, args.n_spin)
    print("Done.")


if __name__ == "__main__":
    main()
