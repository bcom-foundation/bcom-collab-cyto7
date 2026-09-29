"""Per-receptor chemoarchitecture, per-type connectivity, pre-registered MEG ratio.

Implements ``docs/SPEC_receptor_and_type_connectivity.md`` (reviewer G. Ruffini).
Uses the canonical cyto7 **v9** type map directly (no area-level crosswalk) and reuses
the **exact** spin+FDR machinery + the same N=1000 rotations as
``summarise_functional_features.py`` / ``external_validation.py``: the
Alexander-Bloch spin (`neuromaps.nulls.alexander_bloch`, atlas="fsLR", density="32k",
n_perm=1000, seed=0) is applied to the v9 type-rank map, and the rotated type-rank is
correlated against each fixed feature — so every new statistic is directly comparable
to the numbers already in the paper.

Three FDR families are kept **separate** (Benjamini-Hochberg within each): the existing
structure-function 9 and external-validation 3 are NOT recomputed here; this script adds
a new **19-receptor** family, a separate **composites** family, and adds the MEG slow/fast
ratio to the frequency family (timescale + SF). Existing CSVs are never modified.

Parts:
  A  per-receptor receptor gradients + composites  (Fig. 9 reproduction)
  B  per-type connectivity from the cached HCP-1065 tractogram matrices
  C  pre-registered MEG slow/fast band-power ratio SF=(delta+theta)/(beta+gamma1)

Run::  conda activate cyto7 && python scripts/receptor_type_connectivity.py --n-spin 1000
       python scripts/receptor_type_connectivity.py --qc-only       # Part-A anchor check, no spin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cyto7_surface_io import REPO_ROOT, resolve_target_map, LABEL_NAMES
from external_validation import _bh  # identical BH-FDR used for the existing tables

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

CACHE = cfg.data_dir() / "neuromaps_cache"
SF_DIR = cfg.results_dir("tables") / "structure_function"
TR_DIR = cfg.results_dir("tables") / "tractography"
TR_RES = cfg.data_dir() / "tractography" / "v9"
RESULTS = REPO_ROOT / "results"
VER = "v9"
SEED = 0                      # same seed as external_validation / summarise_functional_features
TYPE_NAMES = ["Allocortex", "Agranular", "Dysgranular", "Eulaminate I",
              "Eulaminate II", "Eulaminate III", "Koniocortex"]
TYPE_SHORT = ["Allo", "Agr", "Dys", "EulI", "EulII", "EulIII", "Konio"]

# --------------------------------------------------------------------------- #
# A.2 — authoritative tracer -> receptor -> system -> class mapping (do not guess)
# --------------------------------------------------------------------------- #
RECEPTORS = [
    dict(key="aghourian2017_feobv",       receptor="VAChT",  system="acetylcholine",   klass="transporter",  transporter=True),
    dict(key="beliveau2017_cimbi36",      receptor="5-HT2a", system="serotonin",       klass="metabotropic", transporter=False),
    dict(key="beliveau2017_cumi101",      receptor="5-HT1a", system="serotonin",       klass="metabotropic", transporter=False),
    dict(key="beliveau2017_dasb",         receptor="5-HTT",  system="serotonin",       klass="transporter",  transporter=True),
    dict(key="beliveau2017_sb207145",     receptor="5-HT4",  system="serotonin",       klass="metabotropic", transporter=False),
    dict(key="ding2010_mrb",              receptor="NET",    system="noradrenaline",   klass="transporter",  transporter=True),
    dict(key="dubois2015_abp688",         receptor="mGluR5", system="glutamate",       klass="metabotropic", transporter=False),
    dict(key="dukart2018_fpcit",          receptor="DAT",    system="dopamine",        klass="transporter",  transporter=True),
    dict(key="gallezot2010_p943",         receptor="5-HT1b", system="serotonin",       klass="metabotropic", transporter=False),
    dict(key="gallezot2017_gsk189254",    receptor="H3",     system="histamine",       klass="metabotropic", transporter=False),
    dict(key="galovic2021_ge179",         receptor="NMDA",   system="glutamate",       klass="ionotropic",   transporter=False),
    dict(key="hillmer2016_flubatine",     receptor="a4b2",   system="acetylcholine",   klass="ionotropic",   transporter=False),
    dict(key="kaller2017_sch23390",       receptor="D1",     system="dopamine",        klass="metabotropic", transporter=False),
    dict(key="kantonen2020_carfentanil",  receptor="MOR",    system="opioid",          klass="metabotropic", transporter=False),
    dict(key="naganawa2020_lsn3172176",   receptor="M1",     system="acetylcholine",   klass="metabotropic", transporter=False),
    dict(key="norgaard2021_flumazenil",   receptor="GABA-A", system="GABA",            klass="ionotropic",   transporter=False),
    dict(key="normandin2015_omar",        receptor="CB1",    system="endocannabinoid", klass="metabotropic", transporter=False),
    dict(key="radnakrishnan2018_gsk215083", receptor="5-HT6", system="serotonin",      klass="metabotropic", transporter=False),
    dict(key="smith2017_flb457",          receptor="D2",     system="dopamine",        klass="metabotropic", transporter=False),
]

# A.3 QC anchors: descriptive rho over types 1-7 (allo included), match within ~0.03.
QC_ANCHORS = {
    "MOR": -0.55, "5-HT1a": -0.54, "NET": +0.44, "CB1": -0.41, "D1": -0.39,
    "5-HT4": -0.37, "D2": -0.36, "DAT": -0.34, "mGluR5": -0.25, "H3": -0.21,
    "5-HT2a": -0.18, "5-HT6": -0.12, "5-HTT": -0.09, "NMDA": -0.08, "a4b2": -0.02,
    "M1": -0.01, "VAChT": +0.01, "GABA-A": +0.04, "5-HT1b": +0.10,
}
QC_TOL = 0.03


#: This generator's outputs, under the names the manuscript includes them by. RR38: the three
#: figures below are `\includegraphics` targets in the supplement, and until now nothing carried
#: them from this script's own output tree to the folder the manuscript compiles from. That gap
#: is what RR37 found for Figures 4, S9 and S1, and the fix is the same one: the generator
#: stages its own output, because a copy step outside the generator is the step that gets
#: forgotten.
COMPILE_DIR = REPO_ROOT / "manuscript" / "preprint" / "26th_August_2026" / "figures"
STAGED_AS = {
    "receptor_gradients.png": "cyto7_supp_receptor_gradients.png",
    "receptor_composites_panel.png": "cyto7_supp_receptor_composites.png",
    "per_type_connectivity.png": "cyto7_supp_per_type_connectivity.png",
}


def stage(path: Path) -> None:
    """Copy a rendered figure to the compile folder under its manuscript filename."""
    name = STAGED_AS.get(path.name)
    if name is None or not COMPILE_DIR.is_dir():
        return
    (COMPILE_DIR / name).write_bytes(path.read_bytes())
    print("  staged", COMPILE_DIR / name)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def load_labels():
    lab = resolve_target_map(VER, "fs_LR")
    return {h: np.asarray(lab[h]) for h in ("L", "R")}


def concat(dh):
    return np.concatenate([dh["L"], dh["R"]])


def load_receptor(key):
    return {h: np.load(CACHE / f"receptor_{key}_fsLR32k_hemi-{h}.npy").astype(float)
            for h in ("L", "R")}


def load_meg(desc):
    return {h: np.load(CACHE / f"hcps1200_{desc}_fsLR32k_hemi-{h}.npy").astype(float)
            for h in ("L", "R")}


def compute_nulls(labels, n_spin):
    """The shared Alexander-Bloch rotation set applied to the v9 type-rank map.

    Identical call to external_validation.evaluate: spins the full type-rank map
    (NaN outside labelled cortex) once with seed=0; returns (nverts, n_spin)."""
    from neuromaps.nulls import alexander_bloch
    rank_full = concat({h: np.where(labels[h] > 0, labels[h].astype(float), np.nan)
                        for h in ("L", "R")})
    return alexander_bloch(rank_full, atlas="fsLR", density="32k",
                           n_perm=n_spin, seed=SEED)


def rho_and_spin(vals_c, labels_c, valid_c, nulls):
    """Observed Spearman rho (feature vs type-rank over *valid_c*) + spin p.

    Mirrors external_validation.evaluate exactly: obs = corr(feature, true rank);
    null_i = corr(feature, rotated rank) over the same valid vertices."""
    vv = vals_c[valid_c]
    rank = labels_c[valid_c].astype(float)
    obs = float(stats.spearmanr(vv, rank)[0])
    p_spin, null = np.nan, None
    if nulls is not None:
        n = nulls.shape[1]
        null = np.empty(n)
        for i in range(n):
            spun = nulls[:, i][valid_c]
            ok = np.isfinite(spun)
            null[i] = stats.spearmanr(vv[ok], spun[ok])[0]
        p_spin = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))
    return obs, p_spin, null


def per_type_medians(dh, labels, exclude_allo=False):
    """Hemisphere-averaged per-type median (avg of LH,RH medians) for types 1-7."""
    lo = 2 if exclude_allo else 1
    per_hemi = []
    for h in ("L", "R"):
        m = (labels[h] > 0) & np.isfinite(dh[h])
        meds = [np.median(dh[h][m & (labels[h] == c)]) if np.any(m & (labels[h] == c)) else np.nan
                for c in range(1, 8)]
        per_hemi.append(meds)
    med = np.nanmean(np.array(per_hemi, float), axis=0)
    if exclude_allo:
        med[0] = np.nan
    _ = lo
    return med


def count_reversals(med):
    """Number of monotonicity reversals in the ordered (allo->konio) per-type medians."""
    d = np.diff(med[np.isfinite(med)])
    if d.size == 0:
        return 0
    signs = np.sign(d)
    dom = 1 if np.nansum(d) >= 0 else -1
    return int(np.sum(signs == -dom))


# =========================================================================== #
# Part A
# =========================================================================== #
def part_a(n_spin, qc_only=False):
    print("\n== Part A — per-receptor gradients ==")
    labels = load_labels()
    labels_c = concat(labels)
    valid_all = labels_c > 0                 # types 1-7
    valid_excl = labels_c > 1                # allocortex excluded (types 2-7)

    maps = {r["receptor"]: load_receptor(r["key"]) for r in RECEPTORS}

    # ---- QC anchor gate (descriptive rho, no spin) ------------------------- #
    print("  QC anchor check (descriptive rho, allo included):")
    qc_fail = []
    desc_rho = {}
    for r in RECEPTORS:
        name = r["receptor"]
        vc = concat(maps[name])
        rho = float(stats.spearmanr(vc[valid_all], labels_c[valid_all])[0])
        desc_rho[name] = rho
        anc = QC_ANCHORS[name]
        d = abs(rho - anc)
        flag = "  <-- EXCEEDS" if d > QC_TOL else ""
        print(f"    {name:8s} rho={rho:+.3f}  anchor={anc:+.2f}  |d|={d:.3f}{flag}")
        if d > QC_TOL:
            qc_fail.append((name, rho, anc, d))
    if qc_fail:
        raise SystemExit(f"QC ANCHOR FAILURE ({len(qc_fail)} receptors exceed {QC_TOL}); "
                         "investigate label/hemisphere/order mismatch before spin.")
    print("  QC anchors OK (all within {:.2f}).".format(QC_TOL))
    if qc_only:
        return None

    nulls = compute_nulls(labels, n_spin)

    # ---- per-receptor rho + spin ------------------------------------------- #
    rows = []
    null_ci = {}
    for r in RECEPTORS:
        name = r["receptor"]
        vc = concat(maps[name])
        rho, p_spin, null = rho_and_spin(vc, labels_c, valid_all, nulls)
        rho_ex, p_ex, _ = rho_and_spin(vc, labels_c, valid_excl, nulls)
        null_ci[name] = np.percentile(null, [2.5, 97.5]) if null is not None else (np.nan, np.nan)
        med = per_type_medians(maps[name], labels)
        row = dict(receptor=name, system=r["system"], klass=r["klass"],
                   transporter=r["transporter"], n=int(valid_all.sum()),
                   spearman_rho=rho, spin_p=p_spin,
                   spearman_rho_allo_excl=rho_ex, spin_p_allo_excl=p_ex,
                   n_reversals=count_reversals(med))
        for c in range(1, 8):
            row[f"median_{TYPE_SHORT[c-1]}"] = med[c-1]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["fdr_q"] = _bh(df["spin_p"].values)              # 19-receptor family
    df["fdr_q_allo_excl"] = _bh(df["spin_p_allo_excl"].values)
    df = df.sort_values("spearman_rho").reset_index(drop=True)

    # ---- composites (own FDR family) --------------------------------------- #
    zmaps = {}
    for r in RECEPTORS:
        vc = concat(maps[r["receptor"]])
        mu, sd = np.nanmean(vc[valid_all]), np.nanstd(vc[valid_all])
        zmaps[r["receptor"]] = {h: (maps[r["receptor"]][h] - mu) / sd for h in ("L", "R")}

    def composite(members):
        return {h: np.nanmean(np.stack([zmaps[m][h] for m in members]), axis=0) for h in ("L", "R")}

    def contrast(a, b):
        ca, cb = composite(a), composite(b)
        return {h: ca[h] - cb[h] for h in ("L", "R")}

    metabo = [r["receptor"] for r in RECEPTORS if r["klass"] == "metabotropic"]
    iono = [r["receptor"] for r in RECEPTORS if r["klass"] == "ionotropic"]
    transp = [r["receptor"] for r in RECEPTORS if r["transporter"]]
    sero = [r["receptor"] for r in RECEPTORS if r["system"] == "serotonin" and not r["transporter"]]
    dopa = [r["receptor"] for r in RECEPTORS if r["system"] == "dopamine" and not r["transporter"]]
    ach = [r["receptor"] for r in RECEPTORS if r["system"] == "acetylcholine" and not r["transporter"]]
    glu = [r["receptor"] for r in RECEPTORS if r["system"] == "glutamate"]
    gaba = [r["receptor"] for r in RECEPTORS if r["system"] == "GABA"]

    comp_defs = {
        "metabotropic": composite(metabo), "ionotropic": composite(iono),
        "serotonin": composite(sero), "dopamine": composite(dopa),
        "acetylcholine": composite(ach), "reuptake_innervation": composite(transp),
        "iono_minus_metabo_index": contrast(iono, metabo),
        "glutamate_minus_GABA": contrast(glu, gaba),
    }
    comp_members = {"metabotropic": metabo, "ionotropic": iono, "serotonin": sero,
                    "dopamine": dopa, "acetylcholine": ach, "reuptake_innervation": transp,
                    "iono_minus_metabo_index": iono + ["-"] + metabo,
                    "glutamate_minus_GABA": glu + ["-"] + gaba}
    crows = []
    comp_med = {}
    for cname, cmap in comp_defs.items():
        vc = concat(cmap)
        rho, p_spin, _ = rho_and_spin(vc, labels_c, valid_all, nulls)
        med = per_type_medians(cmap, labels)
        comp_med[cname] = med
        cr = dict(composite=cname, members="|".join(comp_members[cname]),
                  n_members=len([m for m in comp_members[cname] if m != "-"]),
                  spearman_rho=rho, spin_p=p_spin, n_reversals=count_reversals(med))
        for c in range(1, 8):
            cr[f"median_{TYPE_SHORT[c-1]}"] = med[c-1]
        crows.append(cr)
    cdf = pd.DataFrame(crows)
    cdf["fdr_q"] = _bh(cdf["spin_p"].values)            # composites family (separate)
    cdf["underpowered_flag"] = cdf["composite"].eq("glutamate_minus_GABA")

    # ---- write CSVs -------------------------------------------------------- #
    SF_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SF_DIR / "receptor_type_association.csv", index=False)
    cdf.to_csv(SF_DIR / "receptor_composites.csv", index=False)
    print("  wrote", SF_DIR / "receptor_type_association.csv")
    print("  wrote", SF_DIR / "receptor_composites.csv")
    print(df[["receptor", "system", "klass", "spearman_rho", "spin_p", "fdr_q",
              "spearman_rho_allo_excl", "n_reversals"]].to_string(index=False))
    print(cdf[["composite", "n_members", "spearman_rho", "spin_p", "fdr_q"]].to_string(index=False))

    render_receptor_figures(df, cdf, comp_med, labels, maps, null_ci)
    return df, cdf


def _fig_dims(w_mm=190.0, ratio=0.62):
    w_in = w_mm / 25.4
    return w_in, w_in * ratio


# --------------------------------------------------------------------------- #
# RR34 Part D: printed statistics come from the record, not from this script's own
# in-memory results. The values agreed - every row of receptor_type_association.csv,
# receptor_composites.csv and per_type_connectivity_trends.csv reproduces the outcome
# table exactly - but agreeing by coincidence is not the same as being the same number.
# Figures S3, S4 and S5 now read rho, p and the survivor marks from
# figures/v9/review_response/rr2_table/outcome_table.csv, like Figures 2, 4, 5, S2, S6,
# S9, S10 and S11.
# --------------------------------------------------------------------------- #


def _pfmt(p):
    """Print a permutation p, never as a bound.

    The per-type trends use an exhaustive 7! label permutation, so a value below 0.001 is
    exact rather than censored: 1/2520 = 0.000397 is a number, and "<0.001" throws it away.
    """
    import numpy as _np
    if p is None or not _np.isfinite(float(p)):
        return "n/a"
    p = float(p)
    return f"{p:.4f}" if p < 0.0005 else f"{p:.3f}"


def render_receptor_figures(df, cdf, comp_med, labels, maps, null_ci):
    import rr32_outcome_stats as rs
    rec = rs.load_outcomes()
    rs.reset_ledger()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- heatmap (receptors sorted by rho) + forest plot ------------------- #
    order = df.sort_values("spearman_rho", ascending=True)["receptor"].tolist()
    med_cols = [f"median_{s}" for s in TYPE_SHORT]
    Z = np.full((len(order), 7), np.nan)
    for i, name in enumerate(order):
        med = df.set_index("receptor").loc[name, med_cols].values.astype(float)
        Z[i] = (med - np.nanmean(med)) / np.nanstd(med)
    vmax = np.ceil(np.nanmax(np.abs(Z)) * 10) / 10

    w_in, h_in = _fig_dims(190.0, 0.70)
    fig, (axh, axf) = plt.subplots(1, 2, figsize=(w_in, h_in),
                                   gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.42))
    fig.patch.set_facecolor("white")
    im = axh.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axh.set_xticks(range(7)); axh.set_xticklabels(TYPE_SHORT, rotation=35, ha="right", fontsize=7)
    import rr32_outcome_stats as _rs
    ylab = []
    for n in order:
        _rs.render(f"receptor_{n}", "q", _rs.stat(rec, f"receptor_{n}")["q"])
        ylab.append(f"{n}{' *' if _rs.survives(rec, f'receptor_{n}') else ''}")
    axh.set_yticks(range(len(order))); axh.set_yticklabels(ylab, fontsize=7)
    axh.set_xlabel("cyto7 type (allo→konio)", fontsize=7.5)
    for i in range(len(order)):
        for j in range(7):
            v = Z[i, j]
            if np.isfinite(v):
                axh.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=5.2,
                         color="white" if abs(v) > 0.6 * vmax else "0.15")
    cb = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.02)
    cb.set_label("per-type median (z)", fontsize=7); cb.ax.tick_params(labelsize=6)

    # forest: observed rho + spin-null 2.5-97.5% CI band (per receptor)
    fo = df.sort_values("spearman_rho", ascending=True).reset_index(drop=True)
    y = np.arange(len(fo))
    for yi, name in zip(y, fo["receptor"]):
        lo, hi = null_ci.get(name, (np.nan, np.nan))
        if np.isfinite(lo):
            axf.plot([lo, hi], [yi, yi], color="0.72", lw=3.2, solid_capstyle="butt", zorder=1)
    axf.plot(fo["spearman_rho"], y, "o", ms=3.4, color="#333333", zorder=3)
    for yi, q, rho in zip(y, fo["fdr_q"], fo["spearman_rho"]):
        if np.isfinite(q) and q < 0.05:
            axf.plot(rho, yi, "o", ms=6.5, mfc="none", mec="#c1272d", mew=1.2, zorder=4)
    axf.axvline(0, color="0.6", lw=0.8)
    axf.set_yticks(y); axf.set_yticklabels(fo["receptor"], fontsize=7)
    axf.set_xlabel("Spearman ρ vs cyto7 type", fontsize=7.5)
    axf.tick_params(axis="x", labelsize=6.5)
    axf.set_xlim(-0.65, 0.55); axf.grid(axis="x", color="0.9", lw=0.6)
    # RR37: was "grey = spin-null 95% CI". The grey bars are the central 95% of the spin-null
    # distribution, which is a null interval and not a confidence interval for the observed
    # rho - the S4 caption now says so, and the artwork has to say the same thing. This is a
    # label in the image, so it needs the generator, not the caption.
    axf.text(0.0, 1.01, "grey = central 95% of the spin null;  ○ = FDR q<0.05",
             transform=axf.transAxes, fontsize=6.2)
    _rs.verify_renders("figure_S4_receptor_gradients")
    _rs.reset_ledger()
    fig.savefig(SF_DIR / "receptor_gradients.png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", SF_DIR / "receptor_gradients.png")
    stage(SF_DIR / "receptor_gradients.png")

    # ---- composites panel: metabotropic vs ionotropic (+ index) along type -- #
    w_in, h_in = _fig_dims(190.0, 0.42)
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(w_in, h_in), gridspec_kw=dict(wspace=0.28))
    fig2.patch.set_facecolor("white")
    x = np.arange(1, 8)
    for cname, style in [("metabotropic", dict(color="#2166ac", marker="o")),
                         ("ionotropic", dict(color="#b2182b", marker="s"))]:
        a1.plot(x, comp_med[cname], label=cname, lw=1.6, ms=4, **style)
    a1.set_xticks(x); a1.set_xticklabels(TYPE_SHORT, rotation=35, ha="right", fontsize=7)
    a1.set_ylabel("composite z (per-type median)", fontsize=7.5)
    a1.axhline(0, color="0.7", lw=0.7); a1.legend(fontsize=6.5, frameon=False)
    a1.tick_params(labelsize=6.5)
    a1.set_title("Metabotropic vs ionotropic composite", fontsize=8)
    a2.plot(x, comp_med["iono_minus_metabo_index"], color="#4d004b", lw=1.8, marker="D", ms=4)
    a2.set_xticks(x); a2.set_xticklabels(TYPE_SHORT, rotation=35, ha="right", fontsize=7)
    a2.set_ylabel("ionotropic − metabotropic (z)", fontsize=7.5)
    a2.axhline(0, color="0.7", lw=0.7); a2.tick_params(labelsize=6.5)
    _k = "composite_iono_minus_metabo_index"
    _row = _rs.stat(rec, _k)
    _rho, _p = float(_row["effect_value"]), _rs.p_display(_row["p_raw"])
    _rs.render(_k, "rho", _rho); _rs.render(_k, "p", _p)
    a2.set_title(f"Iono/metabo index (ρ={_rho:+.2f}, spin p={_rs.fmt_p(_p)})", fontsize=8)
    _rs.verify_renders("figure_S5_receptor_composites")
    fig2.savefig(SF_DIR / "receptor_composites_panel.png", dpi=600, facecolor="white",
                 bbox_inches="tight")
    plt.close(fig2)
    print("  wrote", SF_DIR / "receptor_composites_panel.png")
    stage(SF_DIR / "receptor_composites_panel.png")


# =========================================================================== #
# Part B
# =========================================================================== #
def part_b(n_perm=5040):
    print("\n== Part B — per-type connectivity ==")
    labels = load_labels()
    vtx = np.array([int((labels["L"] == c).sum() + (labels["R"] == c).sum())
                    for c in range(1, 8)], float)

    short = pd.read_csv(TR_DIR / "connectivity_short.csv", index_col=0).values.astype(float)
    long = pd.read_csv(TR_DIR / "connectivity_long.csv", index_col=0).values.astype(float)
    deg_s = short.sum(axis=1)
    deg_l = long.sum(axis=1)
    deg_t = deg_s + deg_l

    geom = pd.read_csv(TR_RES / "cyto7.v9_tract_geometry_per_bundle.csv")
    # count-weighted per-type mean length + tortuosity: each bundle contributes to both
    # its source and target type, weighted by streamline count.
    mlen = np.full(7, np.nan); mtor = np.full(7, np.nan)
    slen = np.full(7, np.nan); stor = np.full(7, np.nan)   # count-weighted SD (dispersion)
    for c in range(1, 8):
        name = ["Allocortex", "Agranular", "Dysgranular", "Eulaminate-I",
                "Eulaminate-II", "Eulaminate-III", "Koniocortex"][c-1]
        inc = geom[(geom.Source_Cyto == name) | (geom.Target_Cyto == name)]
        w = inc.Streamline_Count.values.astype(float)
        if w.sum() > 0:
            L = inc.Mean_Length_mm.values.astype(float)
            T = inc.Mean_Tortuosity.values.astype(float)
            mlen[c-1] = np.average(L, weights=w); mtor[c-1] = np.average(T, weights=w)
            slen[c-1] = float(np.sqrt(np.average((L - mlen[c-1])**2, weights=w)))
            stor[c-1] = float(np.sqrt(np.average((T - mtor[c-1])**2, weights=w)))

    rank = np.arange(1, 8, dtype=float)

    def perm_p(vals):
        """Exact 7! label-permutation p for |Spearman(vals, rank)| (types re-labelled)."""
        from itertools import permutations
        v = np.asarray(vals, float)
        ok = np.isfinite(v)
        obs = stats.spearmanr(v[ok], rank[ok])[0]
        perms = list(permutations(range(7)))
        cnt = 0
        for pp in perms:
            vp = v[list(pp)]
            okp = np.isfinite(vp)
            r = stats.spearmanr(vp[okp], rank[okp])[0]
            if abs(r) >= abs(obs) - 1e-12:
                cnt += 1
        return float(obs), float(cnt / len(perms))

    summaries = {
        "degree_short": deg_s, "degree_long": deg_l, "degree_total": deg_t,
        "degree_per_vertex": deg_t / vtx,
        "degree_short_per_vertex": deg_s / vtx, "degree_long_per_vertex": deg_l / vtx,
        "mean_length_mm": mlen, "mean_tortuosity": mtor,
    }
    rows = []
    for c in range(1, 8):
        row = dict(type=TYPE_NAMES[c-1], type_rank=c, n_vertices=int(vtx[c-1]),
                   degree_short=deg_s[c-1], degree_long=deg_l[c-1], degree_total=deg_t[c-1],
                   degree_per_vertex=deg_t[c-1]/vtx[c-1],
                   degree_short_per_vertex=deg_s[c-1]/vtx[c-1],
                   degree_long_per_vertex=deg_l[c-1]/vtx[c-1],
                   mean_length_mm=mlen[c-1], mean_length_sd=slen[c-1],
                   mean_tortuosity=mtor[c-1], mean_tortuosity_sd=stor[c-1])
        rows.append(row)
    df = pd.DataFrame(rows)

    # per-summary Spearman vs type + label-permutation p
    trend = []
    for key, vals in summaries.items():
        rho, p = perm_p(vals)
        trend.append(dict(summary=key, spearman_rho_vs_type=rho, perm_p=p, n_perm=len(list(__import__('itertools').permutations(range(7))))))
    trend_df = pd.DataFrame(trend)

    # cross-reference with per-type myelin medians (hemisphere-averaged)
    ft = pd.read_csv(SF_DIR / "functional_summary_table_v9.csv")
    my = ft[ft.FeatureKey == "myelin"]
    myelin_med = np.array([np.mean(my[my.Type == t]["median"].values) for t in
                           ["Allocortex", "agranular", "dysgranular", "eulaminate I",
                            "eulaminate II", "eulaminate III", "koniocortex"]], float)
    xref = []
    for key, vals in summaries.items():
        ok = np.isfinite(vals) & np.isfinite(myelin_med)
        r = stats.spearmanr(vals[ok], myelin_med[ok])[0]
        xref.append(dict(summary=key, spearman_rho_vs_myelin=float(r), n_types=int(ok.sum())))
    xref_df = pd.DataFrame(xref)

    TR_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TR_DIR / "per_type_connectivity.csv", index=False)
    trend_df.to_csv(TR_DIR / "per_type_connectivity_trends.csv", index=False)
    xref_df.to_csv(TR_DIR / "per_type_connectivity_myelin_xref.csv", index=False)
    print("  wrote", TR_DIR / "per_type_connectivity.csv")
    print(df.to_string(index=False))
    print(trend_df.to_string(index=False))
    print("  myelin cross-ref:\n", xref_df.to_string(index=False))

    render_connectivity_figure(df, trend_df)
    return df, trend_df, xref_df


def render_connectivity_figure(df, trend_df):
    import rr32_outcome_stats as _rsr
    _rsr.reset_ledger()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w_in, h_in = _fig_dims(190.0, 0.46)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(w_in, h_in), gridspec_kw=dict(wspace=0.40))
    fig.patch.set_facecolor("white")
    x = np.arange(1, 8)
    tr = trend_df.set_index("summary")

    def _pf(p):
        return "<0.001" if (np.isfinite(p) and p < 0.001) else f"{p:.3f}"
    a1.bar(x - 0.2, df.degree_short_per_vertex, width=0.4, label="short (<80 mm)", color="#4393c3")
    a1.bar(x + 0.2, df.degree_long_per_vertex, width=0.4, label="long (>80 mm)", color="#d6604d")
    a1.set_xticks(x); a1.set_xticklabels(TYPE_SHORT, rotation=35, ha="right", fontsize=7)
    a1.set_ylabel("streamlines per vertex (degree/size)", fontsize=7.5)
    a1.legend(fontsize=6.5, frameon=False); a1.tick_params(labelsize=6.5)
    import rr32_outcome_stats as _rs
    _rec = _rs.load_outcomes()

    def _t(summary):
        r = _rs.stat(_rec, f"tract_{summary}")
        rho, pv = float(r["effect_value"]), float(r["p_raw"])
        _rs.render(f"tract_{summary}", "rho", rho)
        _rs.render(f"tract_{summary}", "p", _rs.p_display(pv))
        return rho, pv

    _rho_dv, _p_dv = _t("degree_per_vertex")
    a1.set_title("Degree (size-norm.):  "
                 f"deg/vtx ρ={_rho_dv:+.2f}, p={_pfmt(_p_dv)}", fontsize=7.5)
    # length + tortuosity markers now carry count-weighted SD (per-bundle dispersion), NOT SEM
    a2.errorbar(x, df.mean_length_mm, yerr=df.mean_length_sd, color="#1b7837", lw=1.6,
                marker="o", ms=4, capsize=2, elinewidth=0.9, label="mean length (mm)")
    a2b = a2.twinx()
    a2b.errorbar(x, df.mean_tortuosity, yerr=df.mean_tortuosity_sd, color="#762a83", lw=1.4,
                 marker="s", ms=3.5, ls="--", capsize=2, elinewidth=0.9, label="tortuosity")
    a2.set_xticks(x); a2.set_xticklabels(TYPE_SHORT, rotation=35, ha="right", fontsize=7)
    a2.set_ylabel("mean bundle length (mm) ± SD", fontsize=7.5, color="#1b7837")
    a2b.set_ylabel("mean tortuosity ± SD", fontsize=7.5, color="#762a83")
    a2.tick_params(labelsize=6.5); a2b.tick_params(labelsize=6.5)
    _rho_len, _p_len = _t("mean_length_mm")
    _rho_tort, _p_tort = _t("mean_tortuosity")
    a2.set_title(f"Length/tortuosity:  "
                 f"len ρ={_rho_len:+.2f}, p={_pfmt(_p_len)};  "
                 f"tort ρ={_rho_tort:+.2f}, p={_pfmt(_p_tort)}", fontsize=7.5)
    _rs.verify_renders("figure_S3_per_type_connectivity")
    fig.savefig(TR_DIR / "per_type_connectivity.png", dpi=600, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote", TR_DIR / "per_type_connectivity.png")
    # Part D: restage into the LaTeX search path as the supplementary figure.
    # RR38: this used to write only manuscript/preprint/figures/, which is a *sibling* of the
    # folder the manuscript compiles from - the same mis-staging RR37 found behind Figures 4,
    # S9 and S1. Both are written now, the compile folder via stage().
    import shutil
    sibling = REPO_ROOT / "manuscript" / "preprint" / "figures" / "cyto7_supp_per_type_connectivity.png"
    sibling.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TR_DIR / "per_type_connectivity.png", sibling)
    print("  staged", sibling)
    stage(TR_DIR / "per_type_connectivity.png")


# =========================================================================== #
# Part C
# =========================================================================== #
def part_c(n_spin):
    print("\n== Part C — pre-registered MEG slow/fast ratio ==")
    labels = load_labels()
    labels_c = concat(labels)
    bands = {b: load_meg(f"meg{b}") for b in ("delta", "theta", "beta")}
    bands["gamma"] = load_meg("meggamma1")   # gamma = gamma1 (as used elsewhere)
    # SF(v) = (delta+theta)/(beta+gamma1)
    sf = {}
    for h in ("L", "R"):
        num = bands["delta"][h] + bands["theta"][h]
        den = bands["beta"][h] + bands["gamma"][h]
        with np.errstate(divide="ignore", invalid="ignore"):
            sf[h] = np.where(den > 0, num / den, np.nan)
    sf_c = concat(sf)
    valid_all = (labels_c > 0) & np.isfinite(sf_c)
    valid_excl = (labels_c > 1) & np.isfinite(sf_c)

    nulls = compute_nulls(labels, n_spin)
    rho, p_spin, _ = rho_and_spin(sf_c, labels_c, valid_all, nulls)
    rho_ex, p_ex, _ = rho_and_spin(sf_c, labels_c, valid_excl, nulls)
    med = per_type_medians(sf, labels)

    # frequency family: timescale (existing) + SF -> BH within family
    ft = pd.read_csv(SF_DIR / "functional_summary_table_v9.csv")
    ts_p = float(ft[ft.FeatureKey == "timescale"]["p_spin"].dropna().iloc[0])
    fam_q = _bh(np.array([ts_p, p_spin]))
    q_sf = float(fam_q[1])

    row = dict(FeatureKey="meg_slow_fast_ratio",
               Feature="MEG slow/fast band-power ratio (delta+theta)/(beta+gamma1)",
               prereg_direction="SF falls with type (negative rho)",
               spearman_rho=rho, spin_p=p_spin, fdr_q_freq_family=q_sf,
               spearman_rho_allo_excl=rho_ex, spin_p_allo_excl=p_ex,
               n=int(valid_all.sum()), n_spin=n_spin,
               freq_family="timescale|meg_slow_fast_ratio")
    for c in range(1, 8):
        row[f"median_{TYPE_SHORT[c-1]}"] = med[c-1]
    df = pd.DataFrame([row])
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = SF_DIR / "meg_ratio_summary.csv"
    df.to_csv(out, index=False)
    print("  wrote", out)
    verdict = ("supports" if (rho < 0 and p_spin < 0.05) else
               "does NOT support (n.s.)" if rho < 0 else "opposite sign")
    print(f"  SF vs type: rho={rho:+.3f}, spin p={p_spin:.3f}, q(freq)={q_sf:.3f} "
          f"[pre-reg: SF falls with type -> {verdict}]")
    print(f"  allo-excluded: rho={rho_ex:+.3f}, spin p={p_ex:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w_in, h_in = _fig_dims(120.0, 0.75)
    fig, ax = plt.subplots(figsize=(w_in, h_in)); fig.patch.set_facecolor("white")
    data = []
    for c in range(1, 8):
        vals = []
        for h in ("L", "R"):
            m = (labels[h] == c) & np.isfinite(sf[h])
            vals.append(sf[h][m])
        data.append(np.concatenate(vals))
    ax.boxplot(data, showfliers=False, widths=0.6)
    ax.set_xticklabels(TYPE_SHORT, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("slow/fast ratio (δ+θ)/(β+γ₁)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.set_title(f"MEG slow/fast ratio by type (ρ={rho:+.2f}, spin p={p_spin:.3f})", fontsize=8)
    fig.savefig(SF_DIR / "meg_slow_fast_ratio.png", dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", SF_DIR / "meg_slow_fast_ratio.png")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=1000)
    ap.add_argument("--parts", default="ABC", help="which parts to run, e.g. 'A', 'ABC'")
    ap.add_argument("--qc-only", action="store_true", help="Part-A anchor check only (no spin)")
    args = ap.parse_args(argv)
    if args.qc_only:
        part_a(args.n_spin, qc_only=True)
        return
    if "A" in args.parts:
        part_a(args.n_spin)
    if "B" in args.parts:
        part_b()
    if "C" in args.parts:
        part_c(args.n_spin)
    print("\nDone.")


if __name__ == "__main__":
    main()
