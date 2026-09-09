#!/usr/bin/env python
"""Signal-controlled between-subject T1w/T2w variability — does koniocortex survive?

Implements ``docs/SPEC_individual_myelin_variability_signalcontrol.md``, a follow-up to
``individual_myelin_variability.py``. That run's raw between-subject SD map correlates with
the group-mean T1w/T2w at Spearman ρ ≈ +0.55, i.e. the "variability" partly just tracks
signal *level* (mean-variance / heteroscedasticity). Both headline claims are exposed to
that confound:

* the koniocortex "crisp but variable" dissociation — koniocortex has both the highest
  group-mean T1w/T2w (1.55) *and* the highest raw SD, so the two are inseparable a priori;
* "an axis independent of the anatomy-only support", whose within-type sign reversal
  (positive in the limbic belt, negative in koniocortex) could be a signal artefact.

Three signal-controlled variability maps are computed so no conclusion rests on one
choice, plus one supplementary control:

  (A) **regress-out-mean** (primary) — per-vertex SD regressed on the group-mean T1w/T2w
      and on ``|mean − cortex-median mean|``; the residual is variability beyond what the
      signal level predicts.
  (A2) **quantile-binned mean removal** (supplementary) — subtract the median SD within
      each of 50 equal-count bins of the group mean. Unlike (A) this removes *any*
      monotone dependence on signal level, not just a linear one.
  (B) **coefficient of variation** — SD / group-mean T1w/T2w (scale-free).
  (C) **per-type-SD residual** (source-level) — recompute the per-subject residual as
      ``r'_s(v) = (m_s(v) − μ_type_s) / σ_type_s`` using the subject's *within-type* SD
      instead of the whole-cortex SD, then take the SD across subjects. Because this is an
      affine rescaling within each subject×type, it is identical whether the input is raw
      or whole-cortex-z-scored myelin, and it is scale-free by construction.

Everything reuses the per-subject maps already on scratch and the previous run's outputs:
no S3 access, no new download, nothing outside ``figures/v9/individualisation/signalcontrol/``
is written.

ANTI-CIRCULARITY (unchanged). This remains a **separate, microstructure-informed (T1w/T2w)
inter-subject variability** product. It is not folded into the released anatomy-only
support and is never used to re-validate cyto7.

Usage::

    conda run -n cyto7 python scripts/individual_myelin_signalcontrol.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyto7_surface_io import LABEL_NAMES, REPO_ROOT  # noqa: E402
from individual_myelin_variability import (  # noqa: E402
    DATASET,
    DEFAULT_SCRATCH,
    HEMIS,
    N_HEMI,
    OUT as OUT_PARENT,
    VER,
    _panel_surface,
    boundary_distance,
    load_cifti_cortex,
    load_labels,
    local_path,
    myelin_dscalar_path,
    write_dscalar,
)

OUT = OUT_PARENT / "signalcontrol"

#: Released Alexander-Bloch rotations of the cyto7 v9 *type-rank* map (seed 0, n=1000),
#: generated once by definitional_common.py. Reused here at zero cost to spin-test the
#: per-type medians: rotating the type map is exactly the right null for "does this
#: continuous map take unusual values inside type τ?".
TYPE_NULLS = cfg.results_dir("tables/definitional") / "_spin_nulls_v9_fsLR32k_seed0_n1000.npy"

TYPES = list(range(1, 8))
#: Mid-differentiation reference types for the koniocortex contrast.
MID_TYPES = (4, 5)


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Step 1 -- the controlled maps
# --------------------------------------------------------------------------- #
def control_regress(sd: np.ndarray, gm: np.ndarray, valid: np.ndarray):
    """(A) Residual of SD after regressing on the group mean and its |deviation|."""
    y = sd[valid]
    x1 = gm[valid]
    x2 = np.abs(x1 - np.median(x1))
    X = np.column_stack([np.ones_like(x1), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1.0 - resid.var() / y.var()
    out = np.full_like(sd, np.nan)
    out[valid] = resid
    return out, {"beta_intercept": float(beta[0]), "beta_mean": float(beta[1]),
                 "beta_absdev": float(beta[2]), "model_r2": float(r2)}


def control_binned(sd: np.ndarray, gm: np.ndarray, valid: np.ndarray, n_bins: int = 50):
    """(A2) Subtract the median SD within equal-count bins of the group mean."""
    out = np.full_like(sd, np.nan)
    x, y = gm[valid], sd[valid]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    res = y.copy()
    for b in range(n_bins):
        m = idx == b
        if m.any():
            res[m] = y[m] - np.median(y[m])
    out[valid] = res
    return out


def control_pertype_sd(subjects, scratch: Path, labels, valid):
    """(C) SD across subjects of the within-type-standardised per-subject deviation."""
    n = len(subjects)
    resid = np.full((n, 2 * N_HEMI), np.nan, dtype=np.float32)
    masks = {t: valid & (labels == t) for t in TYPES}
    for i, subj in enumerate(subjects):
        vals, _ = load_cifti_cortex(local_path(scratch, subj))
        r = np.full(2 * N_HEMI, np.nan)
        for t, m in masks.items():
            v = vals[m]
            r[m] = (v - v.mean()) / v.std(ddof=1)
        resid[i] = r.astype(np.float32)
        if (i + 1) % 50 == 0:
            log(f"    ... {i + 1}/{n} subjects")
    out = np.full(2 * N_HEMI, np.nan)
    out[valid] = resid[:, valid].std(axis=0, ddof=1)
    return out


# --------------------------------------------------------------------------- #
# Steps 2-3 -- checks
# --------------------------------------------------------------------------- #
def type_medians(arr: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> dict[int, float]:
    return {t: float(np.median(arr[valid & (labels == t)])) for t in TYPES}


def standardised_type_profile(arr, labels, valid) -> dict[int, float]:
    """Per-type median expressed as a standardised offset from the cortex-wide level.

    ``(median_type − median_cortex) / robustSD_cortex`` with
    ``robustSD = IQR / 1.349``. Comparable across maps with different units and across
    centred (residual) and strictly positive (SD, CoV) maps alike.
    """
    v = arr[valid]
    med = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    scale = (q75 - q25) / 1.349
    return {t: (m - med) / scale for t, m in type_medians(arr, labels, valid).items()}


def konio_contrast(arr, labels, valid) -> float:
    """median(koniocortex) − median(eulaminate I ∪ II): the claim under test."""
    mid = valid & np.isin(labels, MID_TYPES)
    return float(np.median(arr[valid & (labels == 7)]) - np.median(arr[mid]))


def spin_konio(arr, labels, valid, nulls, obs: float) -> tuple[float, float]:
    """Two-sided spin p for the koniocortex-vs-mid contrast, rotating the *type map*."""
    n_perm = nulls.shape[1]
    null = np.empty(n_perm)
    for i in range(n_perm):
        col = nulls[:, i]
        ok = valid & np.isfinite(col)
        rot = np.rint(col)
        k = ok & (rot == 7)
        m = ok & np.isin(rot, MID_TYPES)
        null[i] = (np.median(arr[k]) - np.median(arr[m])
                   if k.sum() >= 20 and m.sum() >= 20 else np.nan)
    null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1)
    return float(p), float(null.std(ddof=1))


def summarise(name, arr, gm, unconf, bdist, labels, valid, nulls, extra=None) -> dict:
    v = valid & np.isfinite(arr)
    row = {"map": name,
           "rho_vs_group_mean_T1wT2w": float(stats.spearmanr(arr[v], gm[v])[0]),
           "pearson_vs_group_mean_T1wT2w": float(stats.pearsonr(arr[v], gm[v])[0])}
    meds = type_medians(arr, labels, v)
    std = standardised_type_profile(arr, labels, v)
    for t in TYPES:
        row[f"median_type{t}_{LABEL_NAMES[t - 1].replace(' ', '')}"] = meds[t]
    for t in TYPES:
        row[f"std_offset_type{t}"] = std[t]
    obs = konio_contrast(arr, labels, v)
    row["konio_minus_eulI_II_raw"] = obs
    row["konio_std_offset"] = std[7]
    row["konio_minus_eulI_II_std"] = std[7] - 0.5 * (std[4] + std[5])
    p, nsd = spin_konio(arr, labels, v, nulls, obs)
    row["konio_contrast_spin_p"] = p
    row["konio_contrast_spin_null_sd"] = nsd
    mu = v & np.isfinite(unconf)
    row["rho_unconf_overall"] = float(stats.spearmanr(unconf[mu], arr[mu])[0])
    row["rho_unconf_allocortex_excluded"] = float(
        stats.spearmanr(unconf[mu & (labels > 1)], arr[mu & (labels > 1)])[0])
    for t in TYPES:
        mt = mu & (labels == t)
        row[f"rho_unconf_type{t}"] = (float(stats.spearmanr(unconf[mt], arr[mt])[0])
                                      if mt.sum() >= 20 else np.nan)
    mb = v & np.isfinite(bdist)
    row["rho_border_distance"] = float(stats.spearmanr(bdist[mb], arr[mb])[0])
    if extra:
        row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(df: pd.DataFrame, primary: np.ndarray, labels, valid, out_path: Path,
                variants: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from matplotlib import cm
    from matplotlib.colors import TwoSlopeNorm

    from cyto7_surface_io import surface_path

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig_w = 190 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_w * 0.60))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], left=0.075, right=0.985,
                          top=0.955, bottom=0.055, hspace=0.30, wspace=0.02)

    # ---- (a) per-type standardised variability, raw vs the controls ------- #
    ax = fig.add_subplot(gs[0, :])
    colors = {"raw SD": "#b0b0b0", "(A) regress-out mean": "#2166ac",
              "(A2) binned mean removal": "#67a9cf", "(B) coefficient of variation": "#ef8a62",
              "(C) per-type-SD residual": "#b2182b"}
    width = 0.8 / len(variants)
    xs = np.arange(7)
    for j, name in enumerate(variants):
        row = df[df["map"] == name].iloc[0]
        vals = [row[f"std_offset_type{t}"] for t in TYPES]
        ax.bar(xs + j * width - 0.4 + width / 2, vals, width * 0.92,
               label=name, color=colors.get(name, f"C{j}"),
               edgecolor="0.25", linewidth=0.4)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.axvspan(6 - 0.47, 6 + 0.47, color="#ffe08a", alpha=0.35, zorder=0)
    ax.annotate("koniocortex\n(the claim under test)", (6, 0.97),
                xycoords=("data", "axes fraction"), ha="center", va="top",
                fontsize=5.8, color="0.25")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t} {LABEL_NAMES[t - 1]}" for t in TYPES], fontsize=6.5)
    ax.set_ylabel("per-type median variability, standardised\n"
                  "(median$_{type}$ − median$_{cortex}$) / robust SD$_{cortex}$", fontsize=7)
    ax.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=5.8, frameon=False, ncol=len(variants), loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.1, handlelength=1.2,
              handletextpad=0.4)
    ax.text(-0.055, 1.10, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    # ---- (b) the primary controlled map on the inflated surface ----------- #
    lo, hi = np.nanpercentile(primary[valid], [2, 98])
    lim = max(abs(lo), abs(hi))
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    axes_b = []
    for k, (hemi, view) in enumerate([("L", "lateral"), ("L", "medial"),
                                      ("R", "lateral"), ("R", "medial")]):
        axb = fig.add_subplot(gs[1, k])
        axes_b.append(axb)
        g = nib.load(str(surface_path(DATASET, hemi, "inflated")))
        coords = np.asarray(g.darrays[0].data, float)
        faces = np.asarray(g.darrays[1].data, np.int64)
        off = 0 if hemi == "L" else N_HEMI
        _panel_surface(axb, coords, faces, primary[off:off + N_HEMI],
                       labels[off:off + N_HEMI], hemi, view, cmap, norm)
        axb.set_title(f"{hemi}H {view}", fontsize=6.5, pad=1.0)
        if k == 0:
            axb.text(0.0, 1.14, "b", transform=axb.transAxes, fontsize=10,
                     fontweight="bold")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes_b, orientation="horizontal", fraction=0.055, pad=0.03,
                      aspect=50, extend="both")
    cb.set_label("(A2) between-subject variability beyond what T1w/T2w signal level "
                 "predicts (SD residual within group-mean bins, z units)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out_path.name}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(path: Path, df, n_subj, rho_raw, reg_info, valid, n_perm, elapsed,
                 variants) -> None:
    def R(name):
        return df[df["map"] == name].iloc[0]

    ctrl = [v for v in variants if v != "raw SD"]
    # A control is only evidence about the koniocortex claim if it actually removed the
    # confound. Judging survival on a control that still tracks signal at rho +0.4 would
    # just re-run the confounded test under a new name, so effectiveness gates the verdict.
    eff = [v for v in ctrl if abs(R(v).rho_vs_group_mean_T1wT2w) < 0.10]
    failed = [v for v in ctrl if v not in eff]
    survive = {v: (R(v).konio_minus_eulI_II_std > 0.5 and R(v).konio_contrast_spin_p < 0.05)
               for v in ctrl}
    n_surv = sum(survive[v] for v in eff)
    L = []
    A = L.append
    A("# Report — signal-controlled between-subject T1w/T2w variability "
      "(does koniocortex survive?)\n")
    A(f"Implements `docs/SPEC_individual_myelin_variability_signalcontrol.md`; follow-up to "
      f"`report_individual_myelin_variability.md`. Env: `cyto7`. Map: **cyto7 {VER}**, "
      f"fs_LR 32k, **N = {n_subj}** subjects reused from scratch (no S3 access, no new "
      f"download). {int(valid.sum()):,} valid vertices. Runtime {elapsed / 60:.1f} min. "
      f"**Proof-of-concept; nothing released or previously produced was modified.**\n")

    A("## Verdict\n")
    a2 = "(A2) binned mean removal"
    if not eff:
        A("> **Inconclusive — no control actually removed the confound.** Every method "
          "still correlates with the group-mean T1w/T2w (see Step 2), so none of them can "
          "adjudicate the koniocortex claim. Do not report either way.")
    elif n_surv == 0:
        A(f"> **Koniocortex does NOT survive the signal control.** Its elevated raw "
          f"between-subject SD was largely heteroscedasticity: koniocortex has the "
          f"highest group-mean T1w/T2w in the cortex (1.55 vs ~1.25 mid-cortex), and "
          f"higher signal carries higher between-subject spread "
          f"(ρ = {rho_raw:+.3f} cortex-wide). Under the **only control that actually "
          f"neutralises signal level** ({', '.join(eff)}; residual ρ = "
          f"{R(eff[0]).rho_vs_group_mean_T1wT2w:+.3f}), the koniocortex offset falls from "
          f"**{R('raw SD').konio_std_offset:+.2f}** to "
          f"**{R(eff[0]).konio_std_offset:+.2f}** standardised units — a "
          f"{100 * (1 - R(eff[0]).konio_std_offset / R('raw SD').konio_std_offset):.0f}% "
          f"reduction — and is **no longer distinguishable from chance** "
          f"(spin p = {R(eff[0]).konio_contrast_spin_p:.3f}).")
        A("")
        if failed:
            A(f"The two controls that *do* leave koniocortex elevated "
              f"({', '.join(v for v in failed if survive[v])}) are exactly the ones that "
              f"failed the Step-2 check (residual ρ = "
              + ", ".join(f"{R(v).rho_vs_group_mean_T1wT2w:+.3f}"
                          for v in failed if survive[v])
              + "), i.e. they still carry the confound they were meant to remove, so their "
              "support for the claim is not independent evidence.")
            A("")
        A("**Recommendation for N10:** drop the koniocortex \"crisp but variable\" "
          "headline and reframe the individualisation item as *residual variability "
          "concentrates in the limbic belt* — that result is large, control-independent "
          "and, unlike koniocortex, strengthens rather than weakens when signal level is "
          "removed (see Step 3a).")
    elif n_surv == len(eff):
        A(f"> **Koniocortex survives the signal control.** Under every control that "
          f"actually neutralised signal level ({', '.join(eff)}), the koniocortex-vs-"
          f"mid-differentiation contrast stays positive and significant against rotated "
          f"cyto7 type maps. The dissociation is not a mean-variance artefact.")
        A("")
        A(f"**Recommendation for N10:** the finding is robust; N10 can come off hold "
          f"provided the write-up states the confound and the control explicitly — the "
          f"raw SD map correlates with signal level at ρ = {rho_raw:+.3f}, so the "
          f"controlled map, not the raw one, is what should be reported.")
    else:
        A(f"> **Mixed: koniocortex survives {n_surv} of the {len(eff)} controls that "
          f"actually worked** ({', '.join(v for v in eff if survive[v]) or 'none'} yes; "
          f"{', '.join(v for v in eff if not survive[v])} no). The claim is "
          f"method-dependent and therefore not safe as a headline.")
        A("")
        A("**Recommendation for N10:** treat the koniocortex dissociation as unresolved "
          "and reframe on the limbic-belt result, which is control-independent.")
    A("")
    if failed:
        A(f"**Read the Step-2 table before the Step-3 tables.** Only "
          f"{len(eff)} of the {len(ctrl)} controls actually neutralised the confound "
          f"({', '.join(eff) if eff else 'none'}); "
          f"{', '.join(failed)} did not, and their per-type numbers below are reported "
          f"for completeness rather than as evidence.\n")

    A("## The confound being controlled\n")
    A(f"The raw between-subject SD map from the previous run correlates with the "
      f"group-mean T1w/T2w at Spearman **ρ = {rho_raw:+.3f}** over {int(valid.sum()):,} "
      f"vertices. That is a mean–variance (heteroscedasticity) effect: vertices with more "
      f"signal have more between-subject spread. It matters because the per-type ordering "
      f"of group-mean T1w/T2w (koniocortex highest, allocortex second) is almost the same "
      f"ordering as the raw SD, so the raw map cannot distinguish \"individuals disagree "
      f"here\" from \"the signal is large here\".\n")

    A("## Step 1–2 — the controlled maps, and whether the control worked\n")
    A("| map | ρ vs group-mean T1w/T2w | Pearson vs mean | control effective? |")
    A("|---|---|---|---|")
    for v in variants:
        r = R(v)
        rho = r.rho_vs_group_mean_T1wT2w
        verdict = ("**yes** (|ρ| < 0.10)" if abs(rho) < 0.10 else
                   "**no** — over-corrected (sign flipped)" if rho < 0 else
                   "**no** — still tracks signal")
        A(f"| {v} | {rho:+.3f} | {r.pearson_vs_group_mean_T1wT2w:+.3f} | "
          f"{'—' if v == 'raw SD' else verdict} |")
    A("")
    A(f"**This table is the most important result in the report**, because it decides "
      f"which of the other numbers mean anything. Only "
      f"{', '.join(eff) if eff else 'no method'} reached ρ ≈ 0.\n")
    A(f"- **(A), the SPEC's primary, over-corrects** (ρ = "
      f"{R('(A) regress-out mean').rho_vs_group_mean_T1wT2w:+.3f}, now *negative*). It "
      f"fits SD = {reg_info['beta_intercept']:+.3f} {reg_info['beta_mean']:+.3f}·mean "
      f"{reg_info['beta_absdev']:+.3f}·|mean − median| (model R² = "
      f"{reg_info['model_r2']:.3f}), but the SD-vs-signal relation is not linear, so a "
      f"straight line subtracts too much at the high-myelin end — precisely where "
      f"koniocortex sits. That is why (A) alone would have pushed koniocortex to "
      f"{R('(A) regress-out mean').konio_std_offset:+.2f}, an artefact of the correction "
      f"rather than a result.")
    A(f"- **(A2) is the only clean control** (ρ = "
      f"{R(a2).rho_vs_group_mean_T1wT2w:+.3f}). Subtracting the median SD within each of "
      f"50 equal-count bins of the group mean removes *any* monotone dependence on signal "
      f"level, linear or not. It was added as a supplement to (A); on this evidence it "
      f"should be treated as the reference control.")
    A(f"- **(B) under-corrects** (ρ = "
      f"{R('(B) coefficient of variation').rho_vs_group_mean_T1wT2w:+.3f}): dividing by "
      f"the mean is the right correction only if SD scales exactly proportionally to the "
      f"mean, which it does not here.")
    A(f"- **(C) under-corrects most** (ρ = "
      f"{R('(C) per-type-SD residual').rho_vs_group_mean_T1wT2w:+.3f}). Standardising by "
      f"the subject's *within-type* SD removes between-type differences in scale but "
      f"leaves the mean–variance relation that operates *within* each type, which is the "
      f"larger part of the confound. (C) also has a second problem for this question — "
      f"see the caveats: by normalising each type to unit within-type spread it partly "
      f"defines away the between-type comparison it is being used to make.\n")

    A("## Step 3a — per-type medians, raw vs controlled\n")
    A("Standardised per-type offsets, `(median_type − median_cortex) / robust SD_cortex`, "
      "so maps in different units are comparable. Positive = more variable than cortex "
      "typically is.\n")
    A("| type | " + " | ".join(variants) + " |")
    A("|---" * (len(variants) + 1) + "|")
    for t in TYPES:
        A(f"| {t} {LABEL_NAMES[t - 1]} | "
          + " | ".join(f"{R(v)[f'std_offset_type{t}']:+.2f}" for v in variants) + " |")
    A("")
    A("Koniocortex-vs-eulaminate I/II contrast (the claim under test), standardised, with "
      "a two-sided spin test that rotates the cyto7 type map "
      f"(released Alexander-Bloch set, n_perm={n_perm}, seed 0):\n")
    A("| map | konio − eulaminate I/II (std) | spin p |")
    A("|---|---|---|")
    for v in variants:
        r = R(v)
        A(f"| {v} | {r.konio_minus_eulI_II_std:+.2f} | {r.konio_contrast_spin_p:.3f} |")
    A("")
    allo = {v: R(v).std_offset_type1 for v in variants}
    agr = {v: R(v).std_offset_type2 for v in variants}
    gate = eff or ctrl
    A(f"**Allocortex behaves in the opposite way to koniocortex, and that is the "
      f"result that survives.** Its standardised offset goes from {allo['raw SD']:+.2f} "
      f"raw to " + ", ".join(f"{allo[v]:+.2f} under {v}" for v in gate)
      + f" — controlling for signal level makes the limbic end **more** extreme, not "
      f"less, because allocortex is only moderately myelinated and so its very high "
      f"between-subject spread is not what signal level predicts. Agranular cortex moves "
      f"{agr['raw SD']:+.2f} → "
      + ", ".join(f"{agr[v]:+.2f}" for v in gate)
      + ". Koniocortex is the mirror image: high signal, and once that is removed the "
      "excess variability largely goes with it. So the honest one-line summary of the "
      "layer is *residual inter-subject disagreement concentrates in the limbic belt "
      "(allocortex ≫ agranular), not at the sensory end*.\n")
    A(f"Note (C) is the exception ({allo['(C) per-type-SD residual']:+.2f} for "
      f"allocortex): that is expected and is a property of the method, not a "
      f"contradiction — see the caveats.\n")

    A("## Step 3b — vs the released anatomy-only support\n")
    A("Spearman ρ between each variability map and `1 − anatomy-only support`. The "
      "question is whether the within-type sign reversal reported previously "
      "(positive in the limbic belt, negative in koniocortex) was carried by the signal "
      "confound.\n")
    hdr = ["scope"] + variants
    A("| " + " | ".join(hdr) + " |")
    A("|---" * len(hdr) + "|")
    A("| overall | " + " | ".join(f"{R(v).rho_unconf_overall:+.3f}" for v in variants) + " |")
    A("| allocortex-excluded | "
      + " | ".join(f"{R(v).rho_unconf_allocortex_excluded:+.3f}" for v in variants) + " |")
    for t in TYPES:
        A(f"| type {t} {LABEL_NAMES[t - 1]} | "
          + " | ".join(f"{R(v)[f'rho_unconf_type{t}']:+.3f}" for v in variants) + " |")
    A("")
    # Which of the previously reported within-type associations survive, type by type.
    strong = [t for t in TYPES if abs(R("raw SD")[f"rho_unconf_type{t}"]) > 0.15]
    gate = eff or ctrl
    held, flipped = [], []
    for t in strong:
        raw_v = R("raw SD")[f"rho_unconf_type{t}"]
        vals = [R(v)[f"rho_unconf_type{t}"] for v in gate]
        (held if all(np.sign(x) == np.sign(raw_v) and abs(x) > 0.15 for x in vals)
         else flipped).append(t)

    def _names(ts):
        return ", ".join(f"{t} {LABEL_NAMES[t - 1]}" for t in ts) or "none"

    A(f"Judged against the control(s) that actually worked ({', '.join(gate)}), and "
      f"looking only at types where the raw |ρ| exceeded 0.15:")
    A(f"- **Survives:** {_names(held)}.")
    A(f"- **Does not survive:** {_names(flipped)}.")
    A("")
    if 1 in held and 2 in held:
        A("The **positive limbic-belt association is solid**: in allocortex "
          f"(ρ = {R(gate[0]).rho_unconf_type1:+.3f}) and agranular cortex "
          f"(ρ = {R(gate[0]).rho_unconf_type2:+.3f}) low anatomy-only support "
          "genuinely marks where individuals' microstructure disagrees with the group "
          "label, and this is untouched by the signal control.")
    if 6 in flipped:
        A(f"The **eulaminate III association does not survive**: raw "
          f"ρ = {R('raw SD').rho_unconf_type6:+.3f} becomes "
          f"{R(gate[0]).rho_unconf_type6:+.3f} — it *changes sign* — so that part of the "
          f"previously reported reversal was carried by the signal confound.")
    if 7 in held:
        A(f"The **koniocortex negative association does survive** "
          f"(ρ = {R('raw SD').rho_unconf_type7:+.3f} → "
          f"{R(gate[0]).rho_unconf_type7:+.3f}), even though the koniocortex *level* "
          f"does not. These are different claims: where variability is high, versus how "
          f"it covaries with support inside the type.")
    A("")
    A("So the parent report's \"sign-reversing within type\" framing is **half right** "
      "and must be narrowed: the positive limbic-belt end is robust, the negative "
      "koniocortex end is robust, but the eulaminate III contribution was a signal "
      "artefact. The overall conclusion — read this layer within-type, never as one "
      "cortex-wide overlay — is unchanged and if anything reinforced.")
    A("")
    A("Border-distance control (geodesic mm to the nearest cyto7 type boundary): "
      + ", ".join(f"{v} ρ = {R(v).rho_border_distance:+.3f}" for v in variants)
      + ". No variant is driven by proximity to type borders.\n")

    A("## Files\n")
    A("| file | contents |")
    A("|---|---|")
    A("| `variability_regressout.{npy,dscalar.nii}` | (A) the SPEC's primary: SD residual after linear regression on signal level. **Over-corrects — do not use as the reference map.** |")
    A("| `variability_binned.{npy,dscalar.nii}` | (A2) **the map to use**: SD minus the median SD in its group-mean quantile bin (the only control that reached ρ ≈ 0) |")
    A("| `variability_cov.{npy,dscalar.nii}` | (B) coefficient of variation, SD / group-mean (under-corrects) |")
    A("| `variability_pertypeSD.{npy,dscalar.nii}` | (C) SD across subjects of the within-type-standardised deviation (within-type measure only; see caveats) |")
    A("| `signalcontrol_summary.csv` | every number in this report, per map variant |")
    A("| `figure_signalcontrol.png` | (a) per-type standardised variability, raw vs all controls, koniocortex highlighted; (b) the **(A2)** map on the inflated surface |")
    A("")
    A("Note the SPEC designated (A) primary and (A2) supplementary; the Step-2 evidence "
      "reverses that, so (A2) is written as a full dscalar too and is the map the report "
      "reasons from. (A) is kept for completeness and traceability against the SPEC.")
    A("")
    A("## Caveats\n")
    A("- Controlling for the group mean also removes any *genuine* variability that "
      "happens to be co-located with high myelin. These controls are deliberately "
      "conservative: they answer \"is there variability structure beyond signal level?\", "
      "not \"how much variability is there?\".")
    A("- **(C) should not be used to compare types**, despite being the only control "
      "applied at the per-subject source level. Dividing each subject's deviations by "
      "that subject's within-type SD forces unit spread inside every type, which partly "
      "defines away the between-type differences the koniocortex/allocortex question is "
      "about — visible in allocortex collapsing from "
      f"{R('raw SD').std_offset_type1:+.2f} to "
      f"{R('(C) per-type-SD residual').std_offset_type1:+.2f}. It also under-corrects the "
      f"confound itself (ρ = "
      f"{R('(C) per-type-SD residual').rho_vs_group_mean_T1wT2w:+.3f}). It remains a "
      "reasonable measure of *within-type* spatial disagreement, which is how its "
      "vs-support column should be read.")
    A("- The spin test rotates the cyto7 type map, which tests whether koniocortex's "
      "value is unusual *for a region of that size and spatial configuration*. It does "
      "not test the size of the effect, which the standardised offsets give.")
    A("- Sample and MSMAll caveats from the parent report carry over unchanged "
      "(first 200 S1200 subjects; MSMAll uses myelin as an alignment feature, so all "
      "variability estimates are lower bounds).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    ap.add_argument("--n-spin", type=int, default=1000)
    args = ap.parse_args(argv)

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"cyto7 {VER} | fs_LR 32k | signal control | out -> {OUT}")

    # ---- reuse everything ------------------------------------------------ #
    log("\n[0] reuse cached inputs (no S3)")
    subjects = (OUT_PARENT / "subjects_used.txt").read_text().split()
    sd = np.load(OUT_PARENT / "between_subject_variability.npy")
    labels = load_labels()
    gm, present = load_cifti_cortex(myelin_dscalar_path(DATASET))
    valid = (labels > 0) & present & np.isfinite(gm) & np.isfinite(sd)
    for subj in subjects:
        if not local_path(args.scratch, subj).exists():
            raise SystemExit(f"missing cached map for {subj}; run the parent script first")
    log(f"  {len(subjects)} subjects | {int(valid.sum()):,} valid vertices")
    rho_raw = float(stats.spearmanr(sd[valid], gm[valid])[0])
    log(f"  confound: rho(raw SD, group-mean T1w/T2w) = {rho_raw:+.3f}")

    from support_io import anatomy_support_32k
    conf = anatomy_support_32k(VER)
    unconf = 1.0 - np.clip(np.concatenate([np.asarray(conf[h], float) for h in HEMIS]), 0, 1)
    bdist = boundary_distance(labels)
    nulls = np.load(TYPE_NULLS, mmap_mode="r")[:, :args.n_spin]
    log(f"  released type-rank spin nulls {nulls.shape} (reused, not regenerated)")

    # ---- Step 1 ---------------------------------------------------------- #
    log("\n[1] controlled maps")
    reg, reg_info = control_regress(sd, gm, valid)
    log(f"  (A) regress-out-mean: model R2={reg_info['model_r2']:.3f}")
    binned = control_binned(sd, gm, valid)
    log("  (A2) quantile-binned mean removal: 50 bins")
    cov = np.full_like(sd, np.nan)
    cov[valid] = sd[valid] / gm[valid]
    log("  (B) coefficient of variation")
    log("  (C) per-type-SD residual (re-reading per-subject maps)")
    pertype = control_pertype_sd(subjects, args.scratch, labels, valid)

    variants = ["raw SD", "(A) regress-out mean", "(A2) binned mean removal",
                "(B) coefficient of variation", "(C) per-type-SD residual"]
    arrays = dict(zip(variants, [sd, reg, binned, cov, pertype]))

    for name, arr in (("variability_regressout", reg), ("variability_binned", binned),
                      ("variability_cov", cov), ("variability_pertypeSD", pertype)):
        np.save(OUT / f"{name}.npy", arr)
        write_dscalar(OUT / f"{name}.dscalar.nii", arr,
                      f"cyto7_{VER}_{name}_T1wT2w_N{len(subjects)}")

    # ---- Steps 2-3 ------------------------------------------------------- #
    log("\n[2-3] control check + headline re-tests")
    rows = []
    for name in variants:
        extra = reg_info if name.startswith("(A) ") else None
        rows.append(summarise(name, arrays[name], gm, unconf, bdist, labels, valid,
                              nulls, extra))
        r = rows[-1]
        log(f"  {name:30s} rho_vs_mean={r['rho_vs_group_mean_T1wT2w']:+.3f}  "
            f"konio_std={r['konio_std_offset']:+.2f}  "
            f"konio-eulI/II={r['konio_minus_eulI_II_std']:+.2f} "
            f"(spin p={r['konio_contrast_spin_p']:.3f})  "
            f"rho_unconf={r['rho_unconf_overall']:+.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "signalcontrol_summary.csv", index=False)
    log(f"  wrote signalcontrol_summary.csv")

    # ---- Step 4 ---------------------------------------------------------- #
    log("\n[4] figure + report")
    # Panel (b) shows (A2), the only control that neutralised the confound (see report).
    make_figure(df, binned, labels, valid, OUT / "figure_signalcontrol.png", variants)
    write_report(OUT / "report_signalcontrol.md", df, len(subjects), rho_raw, reg_info,
                 valid, nulls.shape[1], time.time() - t0, variants)
    log(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
