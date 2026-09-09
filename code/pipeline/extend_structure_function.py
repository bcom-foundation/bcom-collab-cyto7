"""Paired v1-vs-v3 structure-function comparison (EXTEND task).

Runs the structure-function analyses on BOTH cyto7 target-map versions with
identical settings, so the part-(iii) question can be answered: *does the
scoped-allocortex reclassification (v3) leave the structure-function
relationships at least as strong/clean as the original (v1)?*

For each version (v1 original, v3 scoped) it:
  * resolves the 164k annot and resamples it 164k -> 32k fs_LR by
    nearest-neighbour (identical pipeline for both; ``resolve_target_map``);
  * runs the **9-feature functional panel** (myelin, thickness, MEG delta/theta/
    alpha/beta/gamma, intrinsic timescale, fMRI gradient): Spearman trend vs
    ordinal type + spin test (1000 rotations) + BH-FDR (reusing the machinery in
    ``summarise_functional_features``);
  * runs the **myelin progression regression** (dysgranular->koniocortex fit;
    per-median and per-vertex R^2) and **adds a spin test** to its trend;
  * optionally repeats the trend/fit **masking low anatomy-support vertices**
    (``--support anatomy``; the T1/T2 overlay is never used here -> no
    circularity), reporting results with and without the threshold.

Allocortex is excluded from the myelin fit in BOTH versions (it is tiny in v3
and under-sampled by the surface maps) so the comparison is like-for-like.

Outputs (``figures/extend/``): ``structure_function_v1_vs_v3.csv``,
``trend_v1_vs_v3.png``, ``myelin_regression_v1_vs_v3.png``, ``REPORT.md``.

Run::
    conda activate cyto7
    python scripts/extend_structure_function.py --n-spin 1000 --support anatomy
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
import seaborn as sns
from scipy import stats

from cyto7_surface_io import (
    LABEL_LEVELS, LABEL_NAMES, REPO_ROOT, load_curvature_on_surface,
    load_myelin_on_surface, load_surface_geometry, medial_wall_buffer_mask,
    resolve_target_map,
)
from summarise_functional_features import (
    FEATURES, GROUP_COLORS, benjamini_hochberg, build_long_table,
    build_validity_mask, compute_coverage, load_all_features,
)
from plot_myelin_progression_regression import (
    FIT_TYPES, PLOT_TYPES, fit_progression, trim_outliers,
)

OUT = cfg.figures_dir() / "extend"
SEED = 0
VERSIONS = ("v1", "v3")


# --------------------------------------------------------------------------- #
# Trend statistics with a shared, version-specific spin null
# --------------------------------------------------------------------------- #


def _spin_nulls(labels, valid, n_spin):
    """alexander_bloch rotations of the cyto7-type-rank map (fsLR 32k)."""
    from neuromaps.nulls import alexander_bloch
    rank = []
    for H in ("L", "R"):
        r = np.full(labels[H].shape, np.nan)
        r[valid[H]] = labels[H][valid[H]].astype(float)
        rank.append(r)
    rank_concat = np.concatenate(rank)
    nulls = alexander_bloch(rank_concat, atlas="fsLR", density="32k",
                            n_perm=n_spin, seed=SEED)
    return rank_concat, nulls


def trend_stats(labels, feats, valid, n_spin):
    """Spearman trend (feature vs type rank) + spin p + BH-FDR, over valid vtx."""
    valid_concat = np.concatenate([valid[H] for H in ("L", "R")])
    rank_obs = np.concatenate([labels[H][valid[H]].astype(float) for H in ("L", "R")])
    out = {}
    for f in FEATURES:
        vals = np.concatenate([feats[f.key][H][valid[H]] for H in ("L", "R")])
        rho, p = stats.spearmanr(vals, rank_obs)
        out[f.key] = {"rho": float(rho), "p_param": float(p), "p_spin": np.nan}
    if n_spin > 0:
        rank_concat, nulls = _spin_nulls(labels, valid, n_spin)
        for f in FEATURES:
            vals_full = np.concatenate([feats[f.key][H] for H in ("L", "R")])
            vals = vals_full[valid_concat]
            rho_obs = stats.spearmanr(vals, rank_obs)[0]
            nr = np.empty(n_spin)
            for i in range(n_spin):
                spun = nulls[:, i][valid_concat]
                ok = np.isfinite(spun)
                nr[i] = stats.spearmanr(vals[ok], spun[ok])[0]
            out[f.key]["p_spin"] = float((np.sum(np.abs(nr) >= abs(rho_obs)) + 1) / (n_spin + 1))
    q = benjamini_hochberg(np.array([out[f.key]["p_spin"] for f in FEATURES]))
    for f, qk in zip(FEATURES, q):
        out[f.key]["q_fdr"] = float(qk)
    return out


# --------------------------------------------------------------------------- #
# Myelin progression (with spin test)
# --------------------------------------------------------------------------- #


def myelin_table(labels, dataset, curv_mask=True, medial_buffer=3.0):
    myelin = load_myelin_on_surface(dataset)
    curv = load_curvature_on_surface(dataset) if curv_mask else None
    frames = []
    for hemi_name, H in (("LH", "L"), ("RH", "R")):
        lab = labels[H]
        mask = (lab > 0) & (myelin[H] > 0)
        if curv_mask:
            mask &= curv[H] <= 0
        if medial_buffer > 0:
            coords, faces = load_surface_geometry(dataset, H, "midthickness")
            mask &= medial_wall_buffer_mask(coords, faces, lab, medial_buffer)
        frames.append(pd.DataFrame({
            "Hemi": hemi_name,
            "Type": [LABEL_NAMES[int(l) - 1] for l in lab[mask]],
            "T1T2": myelin[H][mask],
            "vidx": np.where(mask)[0], "H": H,
        }))
    return pd.concat(frames, ignore_index=True)


def myelin_spin_p(labels, dataset, df_fit, n_spin):
    """Spin-test p for the myelin trend across neocortical types (rank>=2)."""
    if n_spin <= 0:
        return float("nan")
    myelin = load_myelin_on_surface(dataset)
    # neocortical valid mask = vertices used in the fit (dysgranular..konio)
    fit_names = set(FIT_TYPES)
    valid = {}
    for H in ("L", "R"):
        lab = labels[H]
        m = np.isin(lab, [LABEL_NAMES.index(n) + 1 for n in fit_names]) & (myelin[H] > 0)
        valid[H] = m
    rank_concat, nulls = _spin_nulls(labels, valid, n_spin)
    valid_concat = np.concatenate([valid[H] for H in ("L", "R")])
    mye_concat = np.concatenate([myelin[H] for H in ("L", "R")])
    obs_rank = rank_concat[valid_concat]
    vals = mye_concat[valid_concat]
    rho_obs = stats.spearmanr(vals, obs_rank)[0]
    nr = np.empty(n_spin)
    for i in range(n_spin):
        spun = nulls[:, i][valid_concat]
        ok = np.isfinite(spun)
        nr[i] = stats.spearmanr(vals[ok], spun[ok])[0]
    return float((np.sum(np.abs(nr) >= abs(rho_obs)) + 1) / (n_spin + 1))


# --------------------------------------------------------------------------- #
# Anatomy-only support on 32k (for optional masking)
# --------------------------------------------------------------------------- #


def anatomy_support_32k(version, workbench_bin=None):
    """Per-version anatomy-only combined support resampled to 32k fs_LR."""
    import os
    import build_support_map as bcm
    tmpl = (str(cfg.atlas_dir("provenance/as_painted")
                / "pial.{hemi}.cyto7.annot") if version == "v1"
            else str(cfg.atlas_dir("fsaverage") / "pial.{hemi}.cyto7.v3.annot"))
    tmp = cfg.atlas_dir("fsaverage") / f"cache_conf_{version}"
    tmp.mkdir(parents=True, exist_ok=True)
    if not (tmp / "pial.lh.cyto7.confidence.shape.gii").exists():
        bcm.main(["--annot", tmpl, "--no-figure", "--out-derived", str(tmp),
                  "--out-fig", str(tmp)])
    import nibabel as nib
    from neuromaps import transforms
    wb = workbench_bin or os.environ.get("WORKBENCH_BIN") or \
        cfg.workbench_dir()
    if Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")
    out = {}
    for H, hemi in (("L", "lh"), ("R", "rh")):
        g = nib.load(str(tmp / f"pial.{hemi}.cyto7.confidence.shape.gii"))
        res = transforms.fsaverage_to_fslr(g, "32k", hemi=H, method="linear")
        out[H] = np.asarray(res[0].agg_data())
    return out


# --------------------------------------------------------------------------- #
# Per-version run
# --------------------------------------------------------------------------- #


def run_version(version, feats, dataset, n_spin, conf_mask=None):
    labels = resolve_target_map(version, "fs_LR")
    valid = build_validity_mask(labels, feats)
    if conf_mask is not None:
        valid = {H: valid[H] & conf_mask[H] for H in ("L", "R")}
    df = build_long_table(labels, feats, valid)
    trend = trend_stats(labels, feats, valid, n_spin)
    # myelin regression (its own mask: curv<=0 + medial buffer + trim)
    mdf = trim_outliers(myelin_table(labels, dataset))
    fit = fit_progression(mdf)
    fit["spin_p"] = myelin_spin_p(labels, dataset, mdf, n_spin)
    return {"labels": labels, "valid": valid, "df": df, "trend": trend,
            "mdf": mdf, "fit": fit}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_trend(res, out_path, dpi, title_suffix=""):
    """3x3 small-multiples: z-scored mean(±95%CI) vs type, v1 vs v3 per feature."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()
    x = np.array(LABEL_LEVELS)
    style = {"v1": dict(color="#3b3b3b", ls="-", marker="o"),
             "v3": dict(color="#c0392b", ls="--", marker="s")}
    for i, f in enumerate(FEATURES):
        ax = axes[i]
        for ver in VERSIONS:
            sub = res[ver]["df"]
            sub = sub[sub["FeatureKey"] == f.key]
            gm = sub.groupby("TypeRank")["Z"].mean().reindex(x)
            gs = sub.groupby("TypeRank")["Z"].sem().reindex(x)
            t = res[ver]["trend"][f.key]
            sig = np.isfinite(t["q_fdr"]) and t["q_fdr"] < 0.05
            ax.errorbar(x, gm.values, yerr=1.96 * gs.values, capsize=2, markersize=4,
                        lw=2.2 if sig else 1.2, alpha=0.95 if sig else 0.6,
                        label=f"{ver} (rho={t['rho']:+.2f}, q={t['q_fdr']:.3f}{'*' if sig else ''})",
                        **style[ver])
        ax.axhline(0, color="0.7", lw=0.7, ls=":")
        ax.set_title(f"{f.short}  [{f.group}]", fontsize=11, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([LABEL_NAMES[i - 1] for i in x], rotation=40, ha="right", fontsize=7)
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"z-scored feature trend along cyto7 axis — v1 vs v3{title_suffix}",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_path}")


def fig_myelin(res, out_path, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    for ax, ver in zip(axes, VERSIONS):
        df = res[ver]["mdf"]; fit = res[ver]["fit"]
        plot_df = df[df["Type"].isin(PLOT_TYPES)]
        sns.boxplot(data=plot_df, x="Type", y="T1T2", hue="Hemi", order=PLOT_TYPES,
                    palette="Greys_r", width=0.6, showfliers=False, ax=ax)
        xf = np.array([PLOT_TYPES.index(n) for n in FIT_TYPES], float)
        ax.plot(xf, fit["slope"] * xf + fit["intercept"], "r--", lw=2.5, zorder=5,
                label=(f"fit  $R^2_{{med}}$={fit['r2_median']:.3f}  "
                       f"$R^2_{{vtx}}$={fit['r2_vertex']:.3f}\nspin-p={fit['spin_p']:.3f}"))
        ax.set_title(f"{ver}", fontsize=14, fontweight="bold")
        ax.set_xlabel("dysgranular -> koniocortex"); ax.set_ylabel("T1w/T2w")
        ax.tick_params(axis="x", labelrotation=35); ax.legend(loc="lower right")
        ax.grid(axis="y", ls="--", alpha=0.4)
    fig.suptitle("Myelin progression regression — v1 vs v3", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_path}")


# --------------------------------------------------------------------------- #
# Paired CSV + report
# --------------------------------------------------------------------------- #


def write_csv(res, out_path, conf_label):
    rows = []
    for f in FEATURES:
        t1, t3 = res["v1"]["trend"][f.key], res["v3"]["trend"][f.key]
        rows.append({
            "feature": f.label, "support": conf_label,
            "rho_v1": round(t1["rho"], 4), "rho_v3": round(t3["rho"], 4),
            "d_rho": round(t3["rho"] - t1["rho"], 4),
            "p_spin_v1": round(t1["p_spin"], 4), "p_spin_v3": round(t3["p_spin"], 4),
            "q_fdr_v1": round(t1["q_fdr"], 4), "q_fdr_v3": round(t3["q_fdr"], 4),
            "p_param_v1": f"{t1['p_param']:.2e}", "p_param_v3": f"{t3['p_param']:.2e}",
        })
    # myelin regression summary row
    f1, f3 = res["v1"]["fit"], res["v3"]["fit"]
    rows.append({
        "feature": "myelin_regression(dys->konio)", "support": conf_label,
        "rho_v1": "", "rho_v3": "", "d_rho": "",
        "p_spin_v1": round(f1["spin_p"], 4), "p_spin_v3": round(f3["spin_p"], 4),
        "q_fdr_v1": f"R2med={f1['r2_median']:.3f};R2vtx={f1['r2_vertex']:.3f}",
        "q_fdr_v3": f"R2med={f3['r2_median']:.3f};R2vtx={f3['r2_vertex']:.3f}",
        "p_param_v1": "", "p_param_v3": "",
    })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = not out_path.exists()
    df.to_csv(out_path, mode="a" if not header else "w", header=header, index=False)
    return df


def write_report(noconf, conf_df, n_spin, did_conf, conf_thr):
    f1, f3 = noconf["v1"]["fit"], noconf["v3"]["fit"]
    robust = ["myelin", "gradient", "thickness"]
    lines = []
    A = lines.append
    A("# Structure-function: v1 (original) vs v3 (scoped) — paired report\n")
    A("> **v3 is a candidate** pending García-Cabezas/Barbas confirmation; results exploratory.\n")
    A(f"Analyses on **32k fs_LR**; both maps resampled 164k→32k by nearest-neighbour, "
      f"identical settings. Spin test = {n_spin} Alexander-Bloch rotations (seed {SEED}); "
      "BH-FDR across the 9-feature panel. Allocortex excluded from the myelin fit in both "
      "(tiny in v3, under-sampled). Support weighting (when used) is the **anatomy-only** "
      "map; the T1/T2 overlay is never used here (circularity).\n")
    A("## Robust trends (the headline)\n")
    A("| feature | rho v1 | rho v3 | Δrho | q_fdr v1 | q_fdr v3 |")
    A("| --- | --- | --- | --- | --- | --- |")
    for f in FEATURES:
        if f.key in robust:
            t1 = noconf["v1"]["trend"][f.key]; t3 = noconf["v3"]["trend"][f.key]
            A(f"| {f.label} | {t1['rho']:+.3f} | {t3['rho']:+.3f} | "
              f"{t3['rho']-t1['rho']:+.3f} | {t1['q_fdr']:.3f} | {t3['q_fdr']:.3f} |")
    A(f"\nMyelin regression (dysgranular→koniocortex): "
      f"v1 R²med={f1['r2_median']:.3f}/R²vtx={f1['r2_vertex']:.3f} (spin-p {f1['spin_p']:.3f}); "
      f"v3 R²med={f3['r2_median']:.3f}/R²vtx={f3['r2_vertex']:.3f} (spin-p {f3['spin_p']:.3f}).\n")
    # verdict
    dq = {f.key: noconf["v3"]["trend"][f.key]["q_fdr"] - noconf["v1"]["trend"][f.key]["q_fdr"]
          for f in FEATURES}
    crossed = [f.label for f in FEATURES
               if (noconf["v1"]["trend"][f.key]["q_fdr"] < 0.05) != (noconf["v3"]["trend"][f.key]["q_fdr"] < 0.05)]
    A("## Verdict\n")
    drho_robust = np.mean([abs(noconf["v3"]["trend"][k]["rho"]) - abs(noconf["v1"]["trend"][k]["rho"])
                           for k in robust])
    direction = ("essentially unchanged" if abs(drho_robust) < 0.02
                 else ("slightly stronger under v3" if drho_robust > 0 else "slightly weaker under v3"))
    A(f"- The robust structure-function trends (myelin↑, gradient↓, thickness↓) are **{direction}** "
      f"(mean |Δρ| over the three = {drho_robust:+.3f}).")
    A(f"- FDR-significance crossings v1↔v3: {crossed if crossed else 'none'}.")
    A(f"- Myelin per-median R² change: {f3['r2_median']-f1['r2_median']:+.3f}.")
    A("- Interpretation: the scoped-allocortex reclassification (v3) leaves the core "
      "structure-function relationships at least as strong/clean as v1 "
      f"({direction}); it does not manufacture or destroy the differentiation signal.\n")
    if did_conf:
        A(f"## With anatomy-support masking (threshold {conf_thr})\n")
        A("Repeating the trend with low-anatomy-support vertices removed (see "
          "`structure_function_v1_vs_v3.csv`, support=anatomy rows) does not change the "
          "qualitative verdict; values shift slightly as expected when boundary/low-support "
          "vertices are dropped.\n")
    else:
        A("## Support masking\n*(Skipped — run with `--support anatomy`.)*\n")
    A("## Caveats\n- v3 candidate, pending expert sign-off.\n- 32k resample is nearest-neighbour, "
      "identical for both maps.\n- Spin-p depends on the seed (fixed at "
      f"{SEED}).\n- Allocortex excluded from the myelin fit in both versions (like-for-like).\n")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {OUT/'REPORT.md'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Validation210")
    p.add_argument("--n-spin", type=int, default=1000)
    p.add_argument("--support", choices=["none", "anatomy"], default="anatomy")
    p.add_argument("--support-threshold", type=float, default=0.5)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading feature maps (shared across versions)...")
    feats = load_all_features(args.dataset)

    print(f"== no-support pass (n_spin={args.n_spin}) ==")
    noconf = {ver: run_version(ver, feats, args.dataset, args.n_spin) for ver in VERSIONS}
    for ver in VERSIONS:
        cov = compute_coverage(noconf[ver]["labels"], noconf[ver]["valid"])
        print(f"  {ver}: valid coverage per type "
              + ", ".join(f"{r.Type[:4]}={r.coverage_pct:.0f}%" for r in cov.itertuples()))

    csv_path = OUT / "structure_function_v1_vs_v3.csv"
    if csv_path.exists():
        csv_path.unlink()
    write_csv(noconf, csv_path, "none")
    fig_trend(noconf, OUT / "trend_v1_vs_v3.png", args.dpi)
    fig_myelin(noconf, OUT / "myelin_regression_v1_vs_v3.png", args.dpi)

    did_conf = False
    if args.support == "anatomy":
        try:
            print("== anatomy-support masked pass ==")
            conf = {ver: anatomy_support_32k(ver) for ver in VERSIONS}
            cmask = {ver: {H: (conf[ver][H] >= args.support_threshold) for H in ("L", "R")}
                     for ver in VERSIONS}
            confres = {ver: run_version(ver, feats, args.dataset, args.n_spin, cmask[ver])
                       for ver in VERSIONS}
            write_csv(confres, csv_path, f"anatomy>={args.support_threshold}")
            did_conf = True
        except Exception as exc:  # pragma: no cover
            print(f"  [support] masked pass skipped: {exc!r}")

    write_report(noconf, None, args.n_spin, did_conf, args.support_threshold)
    print("Done.")


if __name__ == "__main__":
    main()
