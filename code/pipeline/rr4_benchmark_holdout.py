"""RR4 - held-out and stratified tests of the 60% win-fraction benchmark.

Extends ``reviewer_response.analysis1`` (it does not rewrite it): the same
per-type-median arbiter test on the cyto7-vs-von-Economo disagreement set, but

  1. reproduced first, unchanged, as a sanity check,
  2. with the type medians fitted on one hemisphere and scored on the other,
  3. split by mesh-hop distance from the nearest von-Economo areal boundary
     (interior vs border band, plus the 1-to-10 hop profile),
  4. per arbiter feature as well as the published majority-of-three aggregate,
  5. with the existing support stratification re-reported alongside.

Every win fraction carries its own spin null, recomputed on the vertex set
actually scored, and its own chance level (the mean of that null). Outputs go to
``figures/v9/review_response/rr4_benchmark/``; nothing published is touched.

Run::
    conda run -n cyto7 python scripts/rr4_benchmark_holdout.py --n-spin 1000
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import reviewer_response as rr
import rr_common as rc
from cyto7_surface_io import REPO_ROOT

OUTDIR = rc.OUT / "rr4_benchmark"
CACHEDIR = rc.OUT / "_cache"
AXIS_KEYS = rr.AXIS_KEYS            # myelin, thickness, gradient
ISO = rr.ISO_CODES                  # 2..7
ARBITER_LABEL = {"myelin": "T1w/T2w myelin", "thickness": "Cortical thickness",
                 "gradient": "Principal functional gradient"}

# Published values to reproduce (figures/v9/REPORT_v9.md + added_value_localized.csv).
PUBLISHED = {"n_disagree": 27298, "win_all": 0.601, "win_off2": 0.743,
             "spin_p_all": 0.001, "chance": 0.46}


# --------------------------------------------------------------------------- #
# Spin nulls on the published (common-support) mask
# --------------------------------------------------------------------------- #


def published_nulls(cyto, common, n_spin: int) -> np.ndarray:
    """The rotations used by the published win-fraction test, cached.

    ``reviewer_response.analysis1`` calls ``_spin_nulls(cyto, common, n_spin)``,
    i.e. alexander_bloch (seed 0) on the type-rank map masked to the *common*
    support (isocortex where both maps are defined). That support differs from
    the released whole-cortex null file, so reproducing the published p requires
    replaying this construction; it is cached here and reused for every test.
    """
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    cpath = CACHEDIR / f"_spin_nulls_v9_fsLR32k_commonsupport_seed0_n{n_spin}.npy"
    if cpath.exists():
        arr = np.load(cpath, mmap_mode="r")
        if arr.shape[1] >= n_spin:
            print(f"  reusing {cpath.name} {arr.shape}")
            return np.asarray(arr[:, :n_spin])
    print(f"  generating common-support spin nulls (alexander_bloch, seed {rr.SEED}, "
          f"n={n_spin}) ONCE...")
    from extend_structure_function import _spin_nulls
    _rank, nulls = _spin_nulls(cyto, common, n_spin)
    np.save(cpath, nulls.astype(np.float32))
    print(f"  cached {cpath.name} {nulls.shape}")
    return nulls


# --------------------------------------------------------------------------- #
# von-Economo areal boundaries and hop distance
# --------------------------------------------------------------------------- #


def economo_hop_distance() -> np.ndarray:
    """Mesh-hop distance (concatenated 64984) to the nearest von-Economo areal boundary.

    A vertex is on a boundary when any mesh neighbour carries a different
    von-Economo *area* code. Code 0 (unknown / medial wall) counts as a distinct
    area, so the atlas edge is a boundary too: that is the conservative choice
    for an interior claim, since it can only move vertices out of the interior.
    """
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    cpath = CACHEDIR / "_economo_hopdist_fsLR32k.npy"
    if cpath.exists():
        return np.load(cpath)
    from labeled_areas_figure import _annot_32k
    geom = rc.fslr_geometry("midthickness")
    out = []
    for H in ("L", "R"):
        codes, _names = _annot_32k(H, "economo")
        coords, faces = geom[H]
        A = rc.adjacency(faces, len(coords))
        # boundary = a vertex with a neighbour of a different areal code
        Acsr = A.tocsr()
        bnd = np.zeros(len(coords), bool)
        indptr, indices = Acsr.indptr, Acsr.indices
        for v in range(len(coords)):
            nb = indices[indptr[v]:indptr[v + 1]]
            if nb.size and np.any(codes[nb] != codes[v]):
                bnd[v] = True
        out.append(rc.hop_distance(bnd, A))
    hop = np.concatenate(out)
    np.save(cpath, hop)
    return hop


# --------------------------------------------------------------------------- #
# Win-fraction machinery (generalised from analysis1)
# --------------------------------------------------------------------------- #


def _median_vector(fv: np.ndarray, types: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-code medians of *fv* over ``mask``, as a lookup array indexed by code."""
    lut = np.full(9, np.nan)
    for c in ISO:
        m = mask & (types == c)
        if m.any():
            lut[c] = np.median(fv[m])
    return lut


def win_arrays(fvs, cyto_types, ve_types, median_mask, ve_luts):
    """Boolean win-vector per arbiter over *all* vertices, for one median mask."""
    out = {}
    for k, fv in fvs.items():
        mc = _median_vector(fv, cyto_types, median_mask)
        dc = np.abs(fv - mc[np.clip(cyto_types, 0, 8)])
        dv = np.abs(fv - ve_luts[k][np.clip(ve_types, 0, 8)])
        out[k] = dc < dv
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Validation210")
    ap.add_argument("--n-spin", type=int, default=1000)
    ap.add_argument("--hop-threshold", type=int, default=5)
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    frozen_before = rc.frozen_hashes()

    core = rr.load_core(args.dataset, "v9")
    cyto, ve, feats, common = core["cyto"], core["ve"], core["feats"], core["common"]
    cyto_c = rc.cat(cyto, int).astype(int)
    ve_c = rc.cat(ve, int).astype(int)
    common_c = rc.cat(common, bool).astype(bool)
    n_half = cyto["L"].size
    lh = np.zeros(common_c.size, bool); lh[:n_half] = True
    rh = ~lh

    fvs = {k: rc.cat(feats[k]) for k in AXIS_KEYS}
    ve_luts = {k: _median_vector(fvs[k], ve_c, common_c) for k in AXIS_KEYS}

    hop = economo_hop_distance()
    thr = args.hop_threshold
    interior = hop > thr
    border = (hop >= 0) & (hop <= thr)
    print(f"  von-Economo areal boundary hop distance: median {np.median(hop[common_c]):.0f}, "
          f"max {hop[common_c].max()}; interior(> {thr} hops) {int((interior & common_c).sum())} vtx")

    nulls = published_nulls(cyto, common, args.n_spin)

    # ------------------------------------------------------------------ #
    # test specifications
    # ------------------------------------------------------------------ #
    #   name, delta filter, geometric subset, score mask, median mask key, feature
    ALL = np.ones(common_c.size, bool)
    tests = []

    def add(name, subset_label, delta, geo, score, mmask, feature):
        tests.append(dict(test=name, subset=subset_label, delta=delta, geo=geo,
                          score=score, mmask=mmask, feature=feature))

    # step 1: reproduce
    for d in ("all", "off1", "off2"):
        add("published (whole cortex)", d, d, ALL, ALL, "common", "aggregate")
    # step 4: per arbiter (whole cortex, all disagreements)
    for k in AXIS_KEYS:
        add("per-arbiter", f"all / {ARBITER_LABEL[k]}", "all", ALL, ALL, "common", k)
    # step 2: hemisphere holdout
    for d in ("all", "off2"):
        add("holdout: fit LH, score RH", d, d, ALL, rh, "commonL", "aggregate")
        add("holdout: fit RH, score LH", d, d, ALL, lh, "commonR", "aggregate")
    # step 3: interior vs border
    for d in ("all", "off2"):
        add(f"areal interior (> {thr} hops)", d, d, interior, ALL, "common", "aggregate")
        add(f"areal border band (<= {thr} hops)", d, d, border, ALL, "common", "aggregate")

    median_masks = {"common": common_c, "commonL": common_c & lh, "commonR": common_c & rh}

    # ------------------------------------------------------------------ #
    # observed statistics
    # ------------------------------------------------------------------ #
    disagree = common_c & (cyto_c != ve_c)
    delta_abs = np.abs(cyto_c - ve_c)
    dmask = {"all": disagree, "off1": disagree & (delta_abs == 1),
             "off2": disagree & (delta_abs >= 2)}
    print(f"  disagreement set n={int(disagree.sum())} "
          f"(off1={int(dmask['off1'].sum())}, off2={int(dmask['off2'].sum())})")

    obs_wins = {mk: win_arrays(fvs, cyto_c, ve_c, mm, ve_luts) for mk, mm in median_masks.items()}

    def stat(t, wins, dsets):
        sel = dsets[t["delta"]] & t["geo"] & t["score"]
        keys = AXIS_KEYS if t["feature"] == "aggregate" else [t["feature"]]
        w = np.concatenate([wins[t["mmask"]][k][sel] for k in keys])
        return (float(w.mean()) if w.size else np.nan), int(sel.sum()), int(w.size)

    for t in tests:
        t["win_fraction"], t["n_vertices"], t["n_events"] = stat(t, obs_wins, dmask)

    # ------------------------------------------------------------------ #
    # spin nulls: one pass over rotations, all tests scored per rotation
    # ------------------------------------------------------------------ #
    n_spin = args.n_spin
    null_mat = np.full((len(tests), n_spin), np.nan)
    print(f"  spin nulls: {n_spin} rotations x {len(tests)} tests ...")
    for i in range(n_spin):
        spun = nulls[:, i]
        rot = np.rint(spun)
        defined = np.isfinite(spun) & common_c
        rot_i = np.where(defined, rot, -1).astype(int)
        d_rot = defined & (rot_i != ve_c) & np.isin(rot_i, ISO)
        dl = np.abs(rot_i - ve_c)
        dsets = {"all": d_rot, "off1": d_rot & (dl == 1), "off2": d_rot & (dl >= 2)}
        wins = {mk: win_arrays(fvs, rot_i, ve_c, mm & defined, ve_luts)
                for mk, mm in median_masks.items()}
        for j, t in enumerate(tests):
            null_mat[j, i] = stat(t, wins, dsets)[0]
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_spin}")

    for j, t in enumerate(tests):
        nl = null_mat[j][np.isfinite(null_mat[j])]
        t["chance"] = float(nl.mean()) if nl.size else np.nan
        t["spin_p"] = float((np.sum(nl >= t["win_fraction"]) + 1) / (nl.size + 1)) if nl.size else np.nan
        t["n_spin"] = int(nl.size)
        bt = stats.binomtest(int(round(t["win_fraction"] * t["n_events"])), t["n_events"], 0.5)
        t["binom_p"] = float(bt.pvalue)

    df = pd.DataFrame([{k: t[k] for k in ("test", "subset", "n_vertices", "n_events",
                                          "win_fraction", "chance", "spin_p", "n_spin",
                                          "binom_p")} for t in tests])
    # holdout mean of the two directions
    for d in ("all", "off2"):
        sel = df[df.test.str.startswith("holdout") & (df.subset == d)]
        if len(sel) == 2:
            df.loc[len(df)] = {"test": "holdout: mean of both directions", "subset": d,
                               "n_vertices": int(sel.n_vertices.sum()),
                               "n_events": int(sel.n_events.sum()),
                               "win_fraction": float(sel.win_fraction.mean()),
                               "chance": float(sel.chance.mean()),
                               "spin_p": np.nan, "n_spin": n_spin, "binom_p": np.nan}
    df.to_csv(OUTDIR / "holdout_and_strata.csv", index=False)
    print(df.to_string(index=False))

    # ------------------------------------------------------------------ #
    # hop-distance profile (observed only, 1-hop bins out to 10)
    # ------------------------------------------------------------------ #
    rows = []
    for b in list(range(0, 11)) + ["11+"]:
        geo = (hop == b) if b != "11+" else (hop >= 11)
        for d in ("all", "off2"):
            sel = dmask[d] & geo
            w = np.concatenate([obs_wins["common"][k][sel] for k in AXIS_KEYS])
            rows.append({"hop_bin": b, "subset": d, "n_vertices": int(sel.sum()),
                         "n_events": int(w.size),
                         "win_fraction": float(w.mean()) if w.size else np.nan})
    hop_df = pd.DataFrame(rows)
    hop_df.to_csv(OUTDIR / "win_fraction_by_hop_distance.csv", index=False)
    _plot_hop(hop_df, df, OUTDIR / "win_fraction_by_hop_distance.png", thr)

    # ------------------------------------------------------------------ #
    # support stratification, re-reported
    # ------------------------------------------------------------------ #
    conf = pd.read_csv(cfg.results_dir("tables") / "added_value_by_support.csv")
    conf.insert(0, "test", "support tertile (anatomical support score)")
    conf.to_csv(OUTDIR / "support_strata_rereported.csv", index=False)

    # ------------------------------------------------------------------ #
    # reproduction check
    # ------------------------------------------------------------------ #
    repro = {
        "n_disagree": {"published": PUBLISHED["n_disagree"], "recomputed": int(disagree.sum())},
        "win_all": {"published": PUBLISHED["win_all"],
                    "recomputed": round(float(df.loc[(df.test == "published (whole cortex)") &
                                                     (df.subset == "all"), "win_fraction"].iloc[0]), 4)},
        "win_off2": {"published": PUBLISHED["win_off2"],
                     "recomputed": round(float(df.loc[(df.test == "published (whole cortex)") &
                                                      (df.subset == "off2"), "win_fraction"].iloc[0]), 4)},
        "spin_p_all": {"published": PUBLISHED["spin_p_all"],
                       "recomputed": float(df.loc[(df.test == "published (whole cortex)") &
                                                  (df.subset == "all"), "spin_p"].iloc[0])},
        "chance_all": {"published": PUBLISHED["chance"],
                       "recomputed": round(float(df.loc[(df.test == "published (whole cortex)") &
                                                        (df.subset == "all"), "chance"].iloc[0]), 4)},
    }
    ok, detail = rc.check_frozen(frozen_before)
    (OUTDIR / "reproduction_check.json").write_text(
        json.dumps({"reproduction": repro, "frozen_unchanged": ok, "frozen": detail,
                    "hop_threshold": thr, "n_spin": n_spin}, indent=2), encoding="utf-8")
    print("\nreproduction:", json.dumps(repro, indent=2))
    print("frozen files unchanged:", ok)
    return df, hop_df, repro


def _plot_hop(hop_df, df, out_path: Path, thr: int):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for d, col, lab in (("all", "#1b1b6f", "all disagreements"),
                        ("off2", "#d95f02", "off-by->=2")):
        s = hop_df[(hop_df.subset == d) & (hop_df.hop_bin != "11+")]
        x = s.hop_bin.astype(int).to_numpy()
        ax.plot(x, s.win_fraction.to_numpy(), "o-", color=col, label=lab)
    ch = df.loc[(df.test == "published (whole cortex)") & (df.subset == "all"), "chance"]
    if len(ch):
        ax.axhline(float(ch.iloc[0]), ls=":", color="#555555",
                   label=f"spin-null chance ({100*float(ch.iloc[0]):.1f}%)")
    ax.axhline(0.5, ls="--", lw=0.8, color="#999999")
    ax.axvline(thr + 0.5, ls="-", lw=0.8, color="#999999")
    ax.set_xlabel("mesh hops from the nearest von-Economo areal boundary")
    ax.set_ylabel("cyto7 win fraction")
    ax.set_title("Win fraction by distance from an areal boundary")
    ax.set_xticks(range(0, 11))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
