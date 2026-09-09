"""SPEC_adopt_v9 Step 3: v9 tractography report from v9's OWN tractogram.

Copy of write_tracto_report_v8.py repointed to resources/tractography/v9/cyto7.v9_*.
Writes figures/v9/tractography/REPORT_tractography_v9.md (folder name kept figures/v9)
with a changed-vs-v8 line. Stops if any v9 input is missing (no v8 fallback).

Run::  conda activate cyto7 && python scripts/write_tracto_report_v9.py --n-perm 1000
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg
import argparse
import sys
from pathlib import Path

from cyto7_surface_io import REPO_ROOT
import reviewer_response as rr

TR = cfg.data_dir() / "tractography" / "v9"
OUT = cfg.results_dir("tables") / "tractography"   # released tree keeps figures/v9 name
PFX = "cyto7.v9"

# v8 reference (from REPORT_tractography_v8.md, v8's own tractogram).
REF = {"slope": -1.238, "perm_p": 0.003, "conn_r": -0.768, "contact_r": 0.585,
       "partial_r": -0.636, "partial_p": 0.0019}

REQUIRED = [
    TR / f"{PFX}_connectivity_per_bundle.csv",
    TR / f"{PFX}_connectivity_per_bundle_aggregate.csv",
    TR / f"{PFX}_tract_geometry_per_bundle.csv",
    TR / f"{PFX}_in_reference.nii.gz",
]


def _chg(new, old, tol=0.01):
    return "≈ unchanged" if abs(new - old) < tol else f"CHANGED ({new-old:+.3f})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--short-cutoff", type=float, default=80.0)
    args = ap.parse_args(argv)

    missing = [str(p) for p in REQUIRED if not p.exists()]
    if missing:
        print("STOP: missing v9 tractography input(s):")
        for m in missing:
            print("  -", m)
        print("Not falling back to v8 tables (per spec).")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)

    print("== tractography figures (v9) ==")
    import plot_tractography_analysis as pta
    sys.argv = ["plot_tractography_analysis.py",
                "--conn-bundle", str(TR / f"{PFX}_connectivity_per_bundle.csv"),
                "--conn-aggregate", str(TR / f"{PFX}_connectivity_per_bundle_aggregate.csv"),
                "--geometry", str(TR / f"{PFX}_tract_geometry_per_bundle.csv"),
                "--labels-nifti", str(TR / f"{PFX}_in_reference.nii.gz"),
                "--short-cutoff", str(args.short_cutoff),
                "--out-dir", str(OUT)]
    pta.main()

    print("== Analysis 3 (v9 own tractogram) ==")
    rr.OUT_FIG = OUT
    res = rr.analysis3(args.n_perm,
                       geom_csv=TR / f"{PFX}_tract_geometry_per_bundle.csv",
                       labels_nifti=TR / f"{PFX}_in_reference.nii.gz",
                       map_tag="v9 (own tractogram)")
    row = res["df"].iloc[0]
    slope = float(row["short_slope_obs"]); perm = float(row["slope_perm_p"])
    conn_r = float(row["conn_vs_typedist_r"]); conn_p = float(row["conn_vs_typedist_p"])
    contact_r = float(row["conn_vs_contactarea_r"])
    pr = float(row["partial_r_typedist_given_contact"]); pp = float(row["partial_p"])
    survives = "SURVIVES" if pp < 0.05 else "does NOT survive"

    L = []; A = L.append
    A("# Tractography on v9 — §3.7 numbers (v9's own tractogram)\n")
    A("Repo pipeline (`plot_tractography_analysis.py` + `reviewer_response.py` Analysis 3),")
    A("inputs `resources/tractography/v9/cyto7.v9_*` (v9's **own** HCP-1065 tractogram fit).")
    A(f"Short/long cutoff **{args.short_cutoff:.0f} mm**, seed **{rr.SEED}**, permutation **N={args.n_perm}**.\n")
    A("## Headline (short-range cortico-cortical connectivity vs cytoarchitectural type-distance)\n")
    A("| quantity | v8 (own tractogram) | **v9 (own tractogram)** |")
    A("| --- | --- | --- |")
    A(f"| short-range type-distance log-linear slope | {REF['slope']:+.3f} | **{slope:+.3f}** |")
    A(f"| label-permutation null p | {REF['perm_p']:.3f} | **{perm:.3f}** |")
    A(f"| connectivity vs type-distance r | {REF['conn_r']:+.3f} | **{conn_r:+.3f}** (p {conn_p:.3g}) |")
    A(f"| contact-area r (log-conn vs contact area) | {REF['contact_r']:+.3f} | **{contact_r:+.3f}** |")
    A(f"| partial r (type-distance \\| contact area) | {REF['partial_r']:+.3f} (p {REF['partial_p']:.4f}, survives) | "
      f"**{pr:+.3f} (p {pp:.4f}, {survives})** |")
    A("\n## Changed vs v8\n")
    A(f"- slope {_chg(slope, REF['slope'])}; perm-p {perm:.3f} (v8 {REF['perm_p']:.3f}); "
      f"partial r {pr:+.3f} (v8 {REF['partial_r']:+.3f}), {_chg(pr, REF['partial_r'])}, "
      f"**{survives}** at p<0.05.")
    A(f"- The core §3.7 claim — short-range connectivity decays with cytoarchitectural type-distance "
      f"and **survives the contact-area control** — {'HOLDS on v9.' if pp < 0.05 else 'does NOT survive on v9 (FLAG).'}")
    A("\n_v9 = v8 + a 543-vertex L/R entorhinal even-up (RH untouched). Tractography on v9's own "
      "tractogram; no v8 fallback. Seeds fixed; manuscript files untouched._")
    (OUT / "REPORT_tractography_v9.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {OUT/'REPORT_tractography_v9.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
