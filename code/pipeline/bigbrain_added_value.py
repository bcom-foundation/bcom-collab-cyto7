"""BigBrain histology-based added-value test (SPEC_external_validation.md Tier 1A, Analysis 2).

The decisive external test: repeat the §3.3 added-value analysis (`reviewer_response.analysis1`)
but with the **BigBrain Hist-G1 histological gradient** as the sole arbiter instead of the MRI
axis features. On the isocortex vertices where cyto7 (v9) and the von-Economo-derived type map
disagree, does cyto7's label sit closer to its type's BigBrain-gradient median than the
von-Economo label does? A win-fraction > 50% (with a spin null) means the vertex refinement
matches independent **histology**, not just MRI proxies.

Reuses `analysis1` verbatim (identical disagreement set, win rule, spin null) with
``FEATURES`` / ``AXIS_KEYS`` swapped to the single BigBrain feature. Writes to a dedicated
subdir so the canonical v9 added-value outputs are not overwritten.

Run::  conda activate cyto7 && python scripts/bigbrain_added_value.py --n-spin 1000
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from pathlib import Path

import numpy as np

from cyto7_surface_io import REPO_ROOT
import reviewer_response as rr
from summarise_functional_features import Feature

CACHE = cfg.data_dir() / "neuromaps_cache"
OUT = cfg.results_dir("tables") / "structure_function" / "bigbrain_addedvalue"
DATASET = "Validation210"
KEY = "bigbrain_histg1"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=1000)
    args = ap.parse_args(argv)
    import pandas as pd

    bb_l = CACHE / f"bigbrain_histg1_fsLR32k_hemi-L.npy"
    if not bb_l.exists():
        print("STOP: BigBrain Hist-G1 cache missing — run the acquisition first.")
        return 1

    print("Loading core (cyto7 v9, vE, 9 features) on 32k fs_LR...")
    core = rr.load_core(DATASET, "v9")            # builds the isocortex disagreement support
    bb = {H: np.load(CACHE / f"bigbrain_histg1_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}
    core["feats"][KEY] = bb

    # Swap the feature panel to the single BigBrain arbiter (analysis1 uses the module globals).
    rr.FEATURES = [Feature(KEY, "BigBrain Hist-G1 (histology)", "BigBrain", "Histology",
                           lambda _d: bb)]
    rr.AXIS_KEYS = [KEY]
    OUT.mkdir(parents=True, exist_ok=True)
    rr.OUT_FIG = OUT

    print("== Analysis 2: histology-based added value (BigBrain arbiter) ==")
    res = rr.analysis1(core, args.n_spin)

    g = res["global"].iloc[0]                     # single BigBrain row
    agg = res["agg"]
    lines = []
    A = lines.append
    A("# BigBrain histology-based added value (SPEC_external_validation Tier 1A, Analysis 2)\n")
    A("Repeat of the §3.3 added-value test with the **BigBrain Hist-G1 histological gradient** as "
      "the arbiter (BigBrainWarp, fs_LR 32k; Paquola 2019, single specimen). cyto7 **v9** vs the "
      "von-Economo-derived type map, on the isocortex disagreement set. Spin null n="
      f"{args.n_spin}, seed {rr.SEED}.\n")
    A("## Global tracking of the BigBrain gradient\n")
    A(f"- ρ(cyto7 type, BigBrain Hist-G1) = **{g.rho_cyto7:+.3f}**; ρ(von-Economo, BigBrain) = "
      f"{g.rho_voneconomo:+.3f}; Δ|ρ| = {g.d_rho_abs:+.3f} (cyto7 better spin-p {g.spin_p_cyto_better:.3f}).")
    A("## Localized head-to-head on the disagreement set — the decisive test\n")
    A(f"- disagreement set n = {res['n_disagree']} isocortex vertices "
      f"(off-by-1 {res['n_off1']}, off-by-≥2 {res['n_off2']}).")
    A("| subset | n | cyto7 win-fraction | spin-p | null mean |")
    A("| --- | --- | --- | --- | --- |")
    for name in ("all", "off1", "off2"):
        a = agg[name]
        A(f"| {name} | {a['n_events']} | **{a['win_frac']:.3f}** | {a['spin_p']:.4f} | {a['null_mean']:.3f} |")
    win = agg["all"]["win_frac"]; p = agg["all"]["spin_p"]
    verdict = ("cyto7's refinement matches independent histology (win > 50%, "
               + ("spin-significant)." if p < 0.05 else "not spin-significant).")) if win > 0.5 else \
              ("cyto7 does NOT beat von-Economo against BigBrain histology (win ≤ 50%).")
    A(f"\n**Verdict:** on the disagreement set cyto7 wins **{100*win:.1f}%** "
      f"(spin-p {p:.4f}) with BigBrain as arbiter — {verdict}")
    A("\n_BigBrain is a single specimen (n=1) — an independent histological reference, not a "
      "population map; the Hist-G1 surface is spatially noisy at 32k. Reported honestly. "
      "Outputs in `figures/v9/structure_function/bigbrain_addedvalue/`. Manuscript untouched._")
    (OUT / "REPORT_bigbrain_added_value.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {OUT/'REPORT_bigbrain_added_value.md'}")
    print(f"  win(all)={win:.3f} spin-p={p:.4f}; global ρ_cyto {g.rho_cyto7:+.3f} vs ρ_vE {g.rho_voneconomo:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
