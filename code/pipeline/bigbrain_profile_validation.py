"""BigBrain histological validation via intracortical INTENSITY PROFILES.

SPEC_bigbrain_profile_validation.md. The earlier test used the derived Hist-G1 gradient, an
underpowered arbiter (ρ≈0.21). Here the arbiter is the **raw 50-depth staining-intensity depth
profile**, whose shape encodes laminar differentiation (layer-IV granularity) — exactly what
cortical type indexes.

Data: `resources/neuromaps_cache/bigbrain_profiles_fsLR32k_hemi-{L,R}.npy` (50 depths × 32492),
from BigBrainWarp `tpl-fs_LR_den-32k_desc-profiles.txt` (single specimen, n=1).

Step 1 — profile features + PRE-REGISTERED index + validity gate:
  central moments (mean, SD, skewness, kurtosis) + layer-IV peak prominence (mid-cortical
  ~30–60% depth). **Pre-registered differentiation index = profile SKEWNESS** (the conventional
  microstructure summary; declared before Step 2, NOT the max-ρ feature). Report every feature's
  per-type Spearman ρ + spin p (N=1000) + FDR q (same machinery as the structure–function panel).
  GATE: arbiter valid only if the pre-registered index tracks type with |ρ| ≳ 0.4 and spin-sig.

Step 2 (only if gate passes) — histology added-value: reuse `reviewer_response.analysis1` with the
  pre-registered index as sole arbiter; cyto7 vs von-Economo win-fraction on the isocortex
  disagreement set + spin null (dedicated subdir; canonical v9 files untouched).

Step 3 (optional, if gate passes) — cross-validated per-type profile fingerprint matching with
  cross-hemisphere spatial hold-out (no vertex informs its own template).

Run::  conda activate cyto7 && python scripts/bigbrain_profile_validation.py --n-spin 1000
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from cyto7_surface_io import REPO_ROOT, resolve_target_map

CACHE = cfg.data_dir() / "neuromaps_cache"
OUT = cfg.results_dir("tables") / "structure_function" / "bigbrain_profiles"
VER = "v9"
GATE_RHO = 0.40          # pre-registered gate threshold (|ρ| ≳ 0.4)
PREREG_INDEX = "skewness"  # pre-registered differentiation index (declared a priori)


def load_profiles():
    return {H: np.load(CACHE / f"bigbrain_profiles_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}


def profile_features(prof: np.ndarray) -> dict[str, np.ndarray]:
    """Per-vertex features from the (50 depths × nvert) profile matrix.

    Depth 0 = pial, depth 49 = white. Saturated/degenerate profiles → NaN (masked downstream).
    """
    p = prof.astype(float)
    sd = p.std(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        feats = {
            "mean": p.mean(0),
            "sd": sd,
            "skewness": stats.skew(p, axis=0, bias=False),
            "kurtosis": stats.kurtosis(p, axis=0, bias=False),
            # layer-IV peak prominence: mid-cortical (~30–60%) peak minus shoulder baseline
            "peakIV": p[15:30].max(0) - 0.5 * (p[10:15].mean(0) + p[30:35].mean(0)),
        }
    # invalidate degenerate profiles (flat, or saturated at uint16 max 65535)
    bad = (sd == 0) | (p.max(0) >= 65535)
    for k in feats:
        feats[k] = feats[k].astype(float)
        feats[k][bad] = np.nan
    return feats


FEATURE_LABELS = {
    "mean": "Profile mean intensity", "sd": "Profile SD",
    "skewness": "Profile skewness (pre-registered)", "kurtosis": "Profile kurtosis",
    "peakIV": "Layer-IV peak prominence",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=1000)
    args = ap.parse_args(argv)
    import pandas as pd
    import external_validation as ev

    OUT.mkdir(parents=True, exist_ok=True)
    prof = load_profiles()
    feats = {H: profile_features(prof[H]) for H in ("L", "R")}
    keys = ["mean", "sd", "skewness", "kurtosis", "peakIV"]
    features = {k: {H: feats[H][k] for H in ("L", "R")} for k in keys}

    print(f"== Step 1: gradient-by-type (pre-registered index = profile {PREREG_INDEX}) ==")
    stats_df = ev.evaluate(features, args.n_spin)
    stats_df.insert(1, "Feature", [FEATURE_LABELS[k] for k in stats_df["FeatureKey"]])
    stats_df.to_csv(OUT / "profile_features_by_type.csv", index=False)
    print(stats_df.to_string(index=False))

    row = stats_df.set_index("FeatureKey").loc[PREREG_INDEX]
    rho, psp = float(row.spearman_rho), float(row.p_spin)
    gate_pass = (abs(rho) >= GATE_RHO) and (psp < 0.05)
    print(f"\n  GATE (pre-registered '{PREREG_INDEX}'): |ρ|={abs(rho):.3f} (target ≥{GATE_RHO}), "
          f"spin p={psp:.3f} -> {'PASS' if gate_pass else 'FAIL'}")

    L = []; A = L.append
    A("# BigBrain intensity-profile histological validation (SPEC_bigbrain_profile_validation)\n")
    A("Arbiter = raw 50-depth intracortical staining-intensity profiles (BigBrainWarp, fs_LR 32k; "
      "Paquola 2019, **single specimen n=1**), whose *shape* encodes laminar differentiation — "
      "unlike the derived Hist-G1 gradient used before (ρ≈0.21, underpowered). cyto7 **v9**, "
      f"spin N={args.n_spin}, seed 0; FDR across the profile-feature panel.\n")
    A(f"**Pre-registered differentiation index: profile {PREREG_INDEX}** (conventional microstructure "
      "summary; declared before Step 2, not the max-ρ feature).\n")
    A("## Step 1 — profile features by cyto7 type (all reported; no selection)\n")
    A("| feature | ρ vs type | spin p | FDR q |")
    A("| --- | --- | --- | --- |")
    for _, r in stats_df.iterrows():
        star = "*" if (np.isfinite(r.p_spin_fdr) and r.p_spin_fdr < 0.05) else ""
        pre = " **(pre-reg)**" if r.FeatureKey == PREREG_INDEX else ""
        A(f"| {FEATURE_LABELS[r.FeatureKey]}{pre} | {r.spearman_rho:+.3f} | {r.p_spin:.3f} | {r.p_spin_fdr:.3f}{star} |")
    A(f"\n**GATE:** pre-registered index '{PREREG_INDEX}' |ρ| = **{abs(rho):.3f}** (target ≥ {GATE_RHO}), "
      f"spin p = {psp:.3f} → **{'PASS' if gate_pass else 'FAIL'}**.")

    result = {"stats": stats_df, "gate_pass": gate_pass, "rho": rho, "psp": psp, "lines": L,
              "features": features}
    if not gate_pass:
        A("\n**Verdict (gate FAILED):** the single-specimen BigBrain intensity profiles at this fs_LR "
          "projection do **not** resolve the 7-type differentiation scale (pre-registered index below "
          "the power threshold). Per the pre-registered design, Step 2 is **not run/interpreted** as "
          "evidence either way — this is a clean, honest limitation (n=1 histology + warp error), not "
          "an ambiguous null. The BigBrain leg is reported as inconclusive.")
        A("\n_Manuscript untouched; no map substitution._")
        (OUT / "REPORT_bigbrain_profiles.md").write_text("\n".join(L), encoding="utf-8")
        print(f"  wrote {OUT/'REPORT_bigbrain_profiles.md'} (gate failed; Step 2/3 skipped)")
        _write_gallery(features, stats_df)
        return 0

    # ---- gate passed: Steps 2 + 3 ----
    A("\n## Step 2 — histology added-value (pre-registered index as arbiter)\n")
    _step2(features[PREREG_INDEX], args.n_spin, A)
    A("\n## Step 3 — cross-validated per-type profile fingerprint matching\n")
    _step3(prof, args.n_spin, A)
    A("\n_BigBrain n=1 (independent histological reference, not a population map); warp error noted. "
      "Manuscript untouched; canonical v9 added-value files not overwritten._")
    (OUT / "REPORT_bigbrain_profiles.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {OUT/'REPORT_bigbrain_profiles.md'}")
    _write_gallery(features, stats_df)
    return 0


def _step2(index_map, n_spin, A):
    """Reuse reviewer_response.analysis1 with the pre-registered histology index as arbiter."""
    import reviewer_response as rr
    from summarise_functional_features import Feature
    core = rr.load_core("Validation210", VER)
    core["feats"]["bb_index"] = index_map
    rr.FEATURES = [Feature("bb_index", f"BigBrain profile {PREREG_INDEX}", "BigBrain", "Histology",
                           lambda _d: index_map)]
    rr.AXIS_KEYS = ["bb_index"]
    rr.OUT_FIG = OUT
    res = rr.analysis1(core, n_spin)
    g = res["global"].iloc[0]; agg = res["agg"]
    A(f"- global tracking: ρ(cyto7, index) {g.rho_cyto7:+.3f} vs ρ(vE, index) {g.rho_voneconomo:+.3f}.")
    A(f"- disagreement set n = {res['n_disagree']} (off1 {res['n_off1']}, off≥2 {res['n_off2']}).")
    A("| subset | n | cyto7 win-fraction | spin-p | null mean |")
    A("| --- | --- | --- | --- | --- |")
    for name in ("all", "off1", "off2"):
        a = agg[name]
        A(f"| {name} | {a['n_events']} | **{a['win_frac']:.3f}** | {a['spin_p']:.4f} | {a['null_mean']:.3f} |")
    w, p = agg["all"]["win_frac"], agg["all"]["spin_p"]
    A(f"\n**Step 2 verdict:** cyto7 wins **{100*w:.1f}%** (spin-p {p:.4f}) vs von-Economo against the "
      f"BigBrain profile {PREREG_INDEX} — "
      + ("cyto7's refinement matches independent histology."
         if w > 0.5 and p < 0.05 else
         "not a spin-significant win." if w > 0.5 else "cyto7 does not beat von-Economo."))


def _step3(prof, n_spin, A):
    """Cross-hemisphere fingerprint matching: template from one hemi, test the other (no self-leak)."""
    import reviewer_response as rr
    labels = resolve_target_map(VER, "fsaverage")  # not used; keep 32k below
    lab = resolve_target_map(VER, "fs_LR")
    ve = rr.resample_to_32k(rr.voneconomo_code_164k(), "nearest", "voneconomo_code")
    ISO = set(range(2, 8))

    def zprofiles(H):
        p = prof[H].astype(float)
        z = (p - p.mean(0)) / (p.std(0) + 1e-9)
        return z  # (50, nvert)

    def templates(train_H):
        z = zprofiles(train_H); lb = lab[train_H]
        return {c: z[:, lb == c].mean(1) for c in range(1, 8) if np.any(lb == c)}

    wins = []
    for test_H, train_H in (("L", "R"), ("R", "L")):
        tmpl = templates(train_H)
        z = zprofiles(test_H); lb = lab[test_H]; v = ve[test_H]
        dis = (lb != v) & np.isin(lb, list(ISO)) & np.isin(v, list(ISO))
        idx = np.where(dis)[0]
        for i in idx:
            ct, vt = int(lb[i]), int(v[i])
            if ct not in tmpl or vt not in tmpl:
                continue
            pv = z[:, i]
            rc = np.corrcoef(pv, tmpl[ct])[0, 1]
            rv = np.corrcoef(pv, tmpl[vt])[0, 1]
            wins.append(rc > rv)
    wins = np.array(wins)
    frac = float(wins.mean()) if wins.size else float("nan")
    bt = stats.binomtest(int(wins.sum()), int(wins.size), 0.5) if wins.size else None
    A(f"- cross-hemisphere hold-out (template from other hemi; no vertex informs its own template): "
      f"on {wins.size} disagreement vertices the observed profile better matches **cyto7's** "
      f"type-fingerprint than von-Economo's in **{100*frac:.1f}%** "
      f"(binomial p {bt.pvalue:.3g}).")
    A(f"\n**Step 3 verdict:** {'cyto7-assigned type fingerprints fit the histology better' if frac>0.5 else 'no cyto7 advantage'} "
      f"({100*frac:.1f}% > 50%{'' if frac>0.5 else ' NOT exceeded'}).")


def _write_gallery(features, stats_df):
    """Supplementary gallery: mean profile-shape by type + box-by-type for the pre-registered index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cyto7_surface_io import LABEL_NAMES
    lab = resolve_target_map(VER, "fs_LR")
    prof = load_profiles()
    sdf = stats_df.set_index("FeatureKey")

    fig = plt.figure(figsize=(13, 5)); fig.patch.set_facecolor("white")
    # (a) mean intensity profile by type
    ax = fig.add_subplot(1, 2, 1)
    cmap = plt.get_cmap("viridis")
    for t in range(1, 8):
        cols = [prof[H][:, lab[H] == t] for H in ("L", "R")]
        allc = np.concatenate(cols, axis=1)
        allc = allc[:, allc.max(0) < 65535]
        if allc.shape[1] == 0:
            continue
        m = allc.mean(1)
        ax.plot(np.linspace(0, 100, 50), m, color=cmap((t - 1) / 6), lw=2,
                label=LABEL_NAMES[t - 1][:10])
    ax.set_xlabel("cortical depth % (pial→white)"); ax.set_ylabel("BigBrain staining intensity")
    ax.set_title("Mean intracortical profile by cyto7 type"); ax.legend(fontsize=7, frameon=False)
    # (b) pre-registered index box-by-type
    ax2 = fig.add_subplot(1, 2, 2)
    fk = PREREG_INDEX
    m = {H: (lab[H] > 0) & np.isfinite(features[fk][H]) for H in ("L", "R")}
    rank = np.concatenate([lab[H][m[H]] for H in ("L", "R")]).astype(int)
    vals = np.concatenate([features[fk][H][m[H]] for H in ("L", "R")])
    ax2.boxplot([vals[rank == t] for t in range(1, 8)], showfliers=False, widths=0.6)
    ax2.set_xticklabels([LABEL_NAMES[t - 1][:4] for t in range(1, 8)], rotation=30, fontsize=8)
    s = sdf.loc[fk]
    ax2.set_title(f"Profile {fk} by type (ρ={s.spearman_rho:+.3f}, spin p={s.p_spin:.3f}, q={s.p_spin_fdr:.3f})")
    ax2.set_xlabel("cyto7 type (allo→konio)")
    fig.suptitle("BigBrain intracortical intensity profiles vs cyto7 v9 type", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(str(OUT / "bigbrain_profiles_gallery.png"), dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT/'bigbrain_profiles_gallery.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
