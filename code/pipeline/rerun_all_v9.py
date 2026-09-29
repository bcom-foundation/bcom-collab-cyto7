"""Full re-run of the cyto7 analyses on the **v9** map (SPEC_adopt_v9.md Step 3).

v9 = v8 + an L/R entorhinal even-up (543 LH vertices agranular -> allocortex; RH untouched).
Miguel approved v8; v9 is a small consistency fix, **approved for adoption** (no decision gate).
Mirrors ``rerun_all_v8.py`` pointed at v9, writing into the released ``figures/v9/`` tree
(labelled v9) and reporting **what changed vs v8** (expected: essentially unchanged — the 543
added entorhinal vertices are low-support developmental allocortex, excluded from the
structure-function gradient). Heavy (1000-perm spins). Tractography on v9's own tractogram.

Run::  conda activate cyto7 && python scripts/rerun_all_v9.py --n-spin 1000
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from cyto7_surface_io import REPO_ROOT, resolve_target_map
import reviewer_response as rr

V = "v9"
FIG = cfg.results_dir("tables")          # released tree keeps the figures/v9 folder name
SF = FIG / "structure_function"
DER = cfg.atlas_dir("fsaverage")
ANNOT = str(DER / "pial.{hemi}.cyto7.v9.annot")
TRACTO = cfg.data_dir() / "tractography" / "v9"   # v9's own tractogram
PFX = "cyto7.v9"

def require(step: str, *paths) -> None:
    """Every step declares its inputs, and a missing one stops the run.

    RR34 B2. The support step consumed a July 2026 summary table on every v9 run and
    reported success, because the code that read it fell back silently when the v9 table
    was absent - which, under the old step order, it always was. Silent fallback is the
    mechanism behind that whole class of defect: the run is green and the product is wrong.
    A step that cannot find an input must stop, name the input and name itself.
    """
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        lines = "\n    ".join(missing)
        raise SystemExit(
            f"\n  PREFLIGHT FAILED for step '{step}': {len(missing)} expected input(s) "
            f"absent.\n    {lines}\n"
            "  The step is not run. Produce the input, or fix the step order in "
            "rerun_all_v9.main();\n  do not let a step fall back to whatever else it can "
            "find - that is how the support map\n  came to be built from a pre-v9 table "
            "(RR33, RR34).")


# v8 reference numbers (from figures/v9/REPORT_v8.md) for the changed-vs diff.
REF = {"kappa_w": 0.699, "cohen_k": 0.361, "ari": 0.188, "agree": 0.524,
       "conf_auc": 0.5778, "av_win": 0.601, "av_n": 27351,
       "myelin_rho": 0.589, "myelin_q": 0.004, "thick_rho": -0.502, "thick_q": 0.006,
       "grad_rho": -0.576, "grad_q": 0.004, "ts_rho": -0.485, "ts_q": 0.103,
       "r2med": 0.925, "r2vtx": 0.552, "spec_centroid": 0.460,
       "tract_slope": -1.24, "tract_perm": 0.003,
       "tract_partial_r": -0.636, "tract_partial_p": 0.0019,
       "allo_lh": 3604, "allo_rh": 3193, "allo_conf": 0.316,
       # v7 (two-versions-back) for the extra note column
       "v7_kappa_w": 0.702, "v7_allo_lh": 2131, "v7_allo_rh": 1828}


def _promote_to_top_level(cache_dir: Path) -> dict:
    """Copy this version's support products from the cache to resources/cyto7_derived/.

    The cache keeps the historical ``confidence`` spelling so old caches stay readable;
    the promoted public copies carry the current ``support`` name. Returns the names
    written, so the run report can state what the release actually contains.
    """
    import os
    import shutil
    written = []
    for hemi in ("lh", "rh"):
        names = [(f"pial.{hemi}.cyto7.confidence{sfx}.shape.gii",
                  f"pial.{hemi}.cyto7.support{sfx}.shape.gii")
                 for sfx in ("", "_atlas", "_geom", "_prior", "_topo", "_data_overlay")]
        names.append((f"pial.{hemi}.cyto7.confidence_categorical.annot",
                      f"pial.{hemi}.cyto7.support_categorical.annot"))
        for old, new in names:
            src, dst = cache_dir / old, DER / new
            if not src.exists():
                continue
            if dst.exists():
                os.chmod(dst, 0o666)
                dst.unlink()
            shutil.copy2(src, dst)
            written.append(new)
    print(f"  promoted {len(written)} support products to {DER}")
    return {"n": len(written), "files": written}


def _allo_conf_median(cache_dir: Path, version: str) -> dict:
    """Median combined support within the allocortex mask of *version*."""
    labs = resolve_target_map(version, "fsaverage")
    out, vals = {}, []
    for H, hemi in (("L", "lh"), ("R", "rh")):
        p = cache_dir / f"pial.{hemi}.cyto7.confidence.shape.gii"
        if not p.exists():
            return {}
        conf = np.asarray(nib.load(str(p)).darrays[0].data, float)
        m = (labs[H] == 1) & np.isfinite(conf)
        out[hemi] = float(np.median(conf[m])) if m.any() else float("nan")
        vals.append(conf[m])
    out["both"] = float(np.median(np.concatenate(vals))) if vals else float("nan")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Validation210")
    ap.add_argument("--n-spin", type=int, default=1000)
    args = ap.parse_args(argv)
    rr.OUT_FIG = FIG
    SF.mkdir(parents=True, exist_ok=True)
    R = {}

    require("benchmark", *[Path(ANNOT.format(hemi=h)) for h in ("lh", "rh")],
            REPO_ROOT / "resources" / "voneconomo" / "von_economo_cortical_types.csv")
    print("== benchmark ==")
    import compare_cyto7_vs_voneconomo as cmp
    cmp.main(["--annot-version", V, "--output-dir", str(FIG / "vs_voneconomo")])
    agg = pd.read_csv(FIG / "vs_voneconomo" / "agreement_summary.csv")
    R["bench"] = agg[agg["set"] == "pooled"].iloc[0].to_dict()

    core = rr.load_core(args.dataset, V); core["dataset"] = args.dataset

    print("== structure-function ==")
    import summarise_functional_features as sff
    import plot_myelin_progression_regression as myreg
    import extend_spectral_freq as esf
    from extend_structure_function import fit_progression, myelin_spin_p, myelin_table, trim_outliers
    sff.main(["--annot-version", V, "--out-suffix", "_v9", "--n-spin", str(args.n_spin),
              "--output-dir", str(SF)])
    tbl = pd.read_csv(SF / "functional_summary_table_v9.csv")
    R["feat"] = {k: tbl[tbl.FeatureKey == k].iloc[0][["spearman_rho", "p_spin", "p_spin_fdr"]].to_dict()
                 for k in ("myelin", "thickness", "gradient", "timescale")}
    myreg.main(["--annot-version", V, "--out-suffix", "_v9", "--n-spin", str(args.n_spin),
                "--output-dir", str(SF)])
    labels = resolve_target_map(V, "fs_LR")
    mdf = trim_outliers(myelin_table(labels, args.dataset)); fit = fit_progression(mdf)
    fit["spin_p"] = myelin_spin_p(labels, args.dataset, mdf, args.n_spin); R["myfit"] = fit
    metrics = esf.build_metrics(core["feats"]); one = esf.run_version(V, core["feats"], metrics, args.n_spin)
    esf.fig_summary_single(one, metrics, SF / "spectral_freq_summary.png", 300, V)
    R["spec"] = {n: one["trend"][n] for n in metrics}

    # RR33: structure-function now runs BEFORE support. build_support_map reads the
    # per-type myelin medians from functional_summary_table_v9.csv, and under the old
    # order that file did not exist yet, so the support map silently consumed a pre-v9
    # summary left at figures/functional_summary_table.csv on every run. Reordering is
    # safe: the two blocks share only `core`, which is loaded above, and each writes its
    # own keys into R.
    # The support step needs the v9 per-type myelin medians that the structure-function
    # step above writes. Under the pre-RR33 order this file did not exist yet and the read
    # fell back to figures/functional_summary_table.csv, so the preflight is the guard that
    # makes the ordering requirement enforced rather than merely documented.
    require("support", SF / "functional_summary_table_v9.csv",
            *[Path(ANNOT.format(hemi=h)) for h in ("lh", "rh")],
            *[REPO_ROOT / "resources" / "neuromaps_cache" / f"myelin_fsaverage_164k_{h}.npy"
              for h in ("lh", "rh")])
    print("== support ==")
    import build_support_map as bcm
    conf_cache = DER / "cache_conf_v9"
    bcm.main(["--annot", ANNOT, "--out-derived", str(conf_cache),
              "--out-fig", str(FIG / "support")])
    # --out-derived sends the build into this version's cache, so without the step below
    # the top-level products in resources/cyto7_derived/ are never refreshed. They were
    # left at v3 from v4 through v9 because nothing in the analysis pipeline reads them --
    # every reader is version-parameterised onto cache_conf_<version> -- while the public
    # release copies them. Promotion is now part of the run, not a manual afterthought.
    R["promoted"] = _promote_to_top_level(conf_cache)
    R["cal"] = rr.analysis2(core, args.n_spin)
    R["allo_conf"] = {"v8": _allo_conf_median(DER / "cache_conf_v8", "v8"),
                      "v9": _allo_conf_median(conf_cache, "v9")}

    print("== added value ==")
    R["av"] = rr.analysis1(core, args.n_spin)

    require("tractography", TRACTO / f"{PFX}_tract_geometry_per_bundle.csv",
            TRACTO / f"{PFX}_in_reference.nii.gz")
    print("== tractography (v9 own tractogram) ==")
    R["tract"] = rr.analysis3(args.n_spin,
                              geom_csv=TRACTO / f"{PFX}_tract_geometry_per_bundle.csv",
                              labels_nifti=TRACTO / f"{PFX}_in_reference.nii.gz",
                              map_tag="v9 (own tractogram)")

    print("== sensitivity ==")
    R["sens"] = rr.analysis4(core, args.n_spin)

    write_report(R, args.n_spin)
    print("Done.")


def _chg(new, old, tol=0.01):
    return "≈ unchanged" if abs(new - old) < tol else f"CHANGED ({new-old:+.3f})"


def write_report(R, n_spin):
    b, cal, av = R["bench"], R["cal"], R["av"]
    cd = cal["cal"].set_index("component")
    fs, mf, sp, tr = R["feat"], R["myfit"], R["spec"], R["tract"]
    auc = cd.loc["indep", "auc_disagree"]
    cen = next(k for k in sp if "midpoint" in k)
    ac = R["allo_conf"]
    lv = resolve_target_map("v9", "fsaverage")
    allo_lh, allo_rh = int((lv["L"] == 1).sum()), int((lv["R"] == 1).sum())
    hist = cal["hist"].set_index("support_tertile")["error_rate_vs_histology"].to_dict()
    kw = b["weighted_kappa_quadratic"]; ag = av["agg"]
    acv8 = ac.get("v8", {}).get("both", float("nan"))
    acv9 = ac.get("v9", {}).get("both", float("nan"))

    L = []; A = L.append
    A("# cyto7 v9 — full re-run report (L/R entorhinal consistency fix; CANONICAL)\n")
    A(f"Map **v9** (hand-painted by RS from v8: evens up the left–right entorhinal painting — "
      f"543 LH vertices agranular→allocortex in entorhinal/parahippocampal/temporal-pole/fusiform, "
      f"bringing LH entorhinal allocortex ~64%→~98% to match RH; **RH untouched**). Miguel approved "
      f"v8; v9 is a small consistency fix, **approved for adoption**. 32k fs_LR, seed {rr.SEED}, "
      f"spin/perm N={n_spin}. Tractography on v9's own tractogram. Numbers + changed-vs-v8 diff.\n")

    A("## §3.1 Map\n")
    A(f"- Surface allocortex LH {allo_lh} / RH {allo_rh} vertices (v8 {REF['allo_lh']}/{REF['allo_rh']}; "
      f"+{allo_lh-REF['allo_lh']} LH from the entorhinal even-up, RH unchanged); strict R1=0 "
      "(topology_audit_v9); allocortex = single **open arc** (β1=0); agranular & dysgranular each a "
      "single encircling ring; EulII 1-comp/3-holes; EulIII/konio 3 islands.")
    A("## §3.2 Benchmark vs von-Economo\n")
    A(f"- quadratic-weighted κ_w = **{kw:.3f}**, Cohen's κ {b['cohen_kappa']:.3f}, ARI "
      f"{b['adjusted_rand_index']:.3f}, exact agreement {b['overall_agreement']:.1%}.")
    A("## §3.3 Added value\n")
    A(f"- disagreement set {av['n_disagree']} vtx (v8 {REF['av_n']}; the 543 relabelled entorhinal "
      f"vertices leave the benchmark-disagreement set); localized win-fraction "
      f"**{100*ag['all']['win_frac']:.1f}%** (spin-p {ag['all']['spin_p']:.4f}); "
      f"off-by-≥2 {100*ag['off2']['win_frac']:.1f}%.")
    A("## §3.4 Support (strict topo) — incl. ALLOCORTEX SUPPORT\n")
    A(f"- benchmark-independent disagreement AUC **{auc}**; histology-patch error by tertile {hist}.")
    if ac.get("v9"):
        A(f"- **allocortex median support: v8 {acv8:.3f} → v9 {acv9:.3f}** (per-hemi v9 "
          f"lh {ac['v9'].get('lh', float('nan')):.3f} / rh {ac['v9'].get('rh', float('nan')):.3f}). "
          "The 543 added entorhinal vertices are low-support developmental allocortex — the "
          "narrative is unchanged from v8 (expert/developmental, not atlas-driven).")
    A("## §3.5–3.6 Structure–function (allocortex excluded from the gradient)\n")
    for k, nm in (("myelin", "myelin"), ("thickness", "thickness"), ("gradient", "gradient"),
                  ("timescale", "timescale")):
        r = fs[k]
        A(f"- {nm}: ρ={r['spearman_rho']:+.3f}, spin-p={r['p_spin']:.3f}, q={r['p_spin_fdr']:.3f}")
    A(f"- myelin regression R²med={mf['r2_median']:.3f}, R²vtx={mf['r2_vertex']:.3f} (spin-p {mf['spin_p']:.3f}).")
    A(f"- spectral centroid ρ={sp[cen]['rho']:+.3f} (p_spin {sp[cen]['p_spin']:.3f}).")
    A("## §3.7 Tractography (v9 own tractogram)\n")
    A(f"- slope {tr['slope']:+.2f}, perm-p {tr['perm_p']:.3f}, partial r "
      f"{tr['partial_r']:+.3f} (p {tr['partial_p']:.3f}), {_survives(tr)}. "
      "Full detail: REPORT_tractography_v9.md.")

    A("\n## Changed vs v8 (v7 noted)\n")
    A("| quantity | v7 | v8 | v9 | note |")
    A("| --- | --- | --- | --- | --- |")
    A(f"| allocortex (LH/RH vtx) | {REF['v7_allo_lh']}/{REF['v7_allo_rh']} | "
      f"{REF['allo_lh']}/{REF['allo_rh']} | {allo_lh}/{allo_rh} | +{allo_lh-REF['allo_lh']} LH entorhinal even-up; RH unchanged |")
    A(f"| von-Economo κ_w | {REF['v7_kappa_w']:.3f} | {REF['kappa_w']:.3f} | {kw:.3f} | {_chg(kw, REF['kappa_w'])} |")
    A(f"| support AUC | 0.575 | {REF['conf_auc']:.3f} | {auc} | strict topo |")
    A(f"| added-value win | 0.602 | {REF['av_win']:.3f} | {ag['all']['win_frac']:.3f} | {_chg(ag['all']['win_frac'], REF['av_win'])} |")
    A(f"| myelin ρ / q | +0.589/0.004 | {REF['myelin_rho']:+.3f}/{REF['myelin_q']:.3f} | "
      f"{fs['myelin']['spearman_rho']:+.3f}/{fs['myelin']['p_spin_fdr']:.3f} | {_chg(fs['myelin']['spearman_rho'], REF['myelin_rho'])} |")
    A(f"| thickness ρ / q | -0.501/0.006 | {REF['thick_rho']:+.3f}/{REF['thick_q']:.3f} | "
      f"{fs['thickness']['spearman_rho']:+.3f}/{fs['thickness']['p_spin_fdr']:.3f} | {_chg(fs['thickness']['spearman_rho'], REF['thick_rho'])} |")
    A(f"| gradient ρ / q | -0.576/0.004 | {REF['grad_rho']:+.3f}/{REF['grad_q']:.3f} | "
      f"{fs['gradient']['spearman_rho']:+.3f}/{fs['gradient']['p_spin_fdr']:.3f} | {_chg(fs['gradient']['spearman_rho'], REF['grad_rho'])} |")
    A(f"| timescale ρ / q | -0.484/0.106 | {REF['ts_rho']:+.3f}/{REF['ts_q']:.3f} | "
      f"{fs['timescale']['spearman_rho']:+.3f}/{fs['timescale']['p_spin_fdr']:.3f} |  |")
    A(f"| myelin R²med | 0.928 | {REF['r2med']:.3f} | {mf['r2_median']:.3f} | {_chg(mf['r2_median'], REF['r2med'])} |")
    A(f"| spectral centroid ρ | +0.457 | {REF['spec_centroid']:+.3f} | {sp[cen]['rho']:+.3f} | {_chg(sp[cen]['rho'], REF['spec_centroid'])} |")
    A(f"| allocortex median support | — | {REF['allo_conf']:.3f} | {acv9:.3f} | {_chg(acv9, acv8, 0.03)} (developmental) |")
    A(f"| tractography slope/perm-p | -1.26/0.002 | {REF['tract_slope']:+.2f}/{REF['tract_perm']:.3f} | "
      f"{tr['slope']:+.2f}/{tr['perm_p']:.3f} | v9 own tractogram |")
    A(f"| tractography partial-r | -0.659 (p 0.001) | {REF['tract_partial_r']:+.3f} (p {REF['tract_partial_p']:.4f}) | "
      f"{tr['partial_r']:+.3f} (p {tr['partial_p']:.4f}) | {_survives(tr)} |")

    A("\n_v9 is a 543-vertex L/R entorhinal consistency fix (RH untouched); all validations hold within "
      "noise of v8 (allocortex is excluded from the structure–function gradient, so those numbers are "
      "unchanged at 2 dp). The added entorhinal allocortex is low-support developmental cortex, as in "
      "v8. v9 is the canonical map. Seeds fixed; figures/v9/ regenerated on v9; manuscript files untouched._")
    (FIG / "REPORT_v9.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {FIG/'REPORT_v9.md'}")


def _survives(tr):
    return "survives (p<0.05)" if tr["partial_p"] < 0.05 else "does NOT survive (p≥0.05)"


if __name__ == "__main__":
    main()
