"""RR13 - p-value convention audit.

Section 2.6 states the spin p-value as the plus-one corrected two-tailed tail,

    p = (1 + #{|rho_null| >= |rho_obs|}) / (N + 1),   N = 1000,

so the smallest attainable value is 1/1001 and an exact zero is impossible. But
``figures/v9/definitional/layer_marker_genes.csv`` reports p_spin = 0.0 for RORB
and CUX2. This audit finds out which convention that family actually used, sweeps
every family for the same problem, and recomputes the layer-marker family under
the stated convention.

Nothing published is changed in place. Corrected values are emitted alongside.

Run::
    conda run -n cyto7 python scripts/rr13_pvalue_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import rr_common as rc
from cyto7_surface_io import REPO_ROOT

OUTDIR = rc.OUT / "rr13_pvalues"
FIGV9 = REPO_ROOT / "figures" / "v9"
DEF = FIGV9 / "definitional"
CROSSED = REPO_ROOT / "resources" / "cyto7_derived" / "crossed"

#: RR10 outputs are inputs here and must come out unchanged.
FROZEN_EXTRA = sorted(CROSSED.glob("*.annot")) + sorted(CROSSED.glob("*.label.gii")) + \
    [CROSSED / "aparc_cyto7_nodes.tsv", CROSSED / "voneconomo_cyto7_nodes.tsv"]

#: columns that hold a p or q value. Deliberately explicit rather than a loose
#: regex, so the audit can state what it looked at and a reader can check it.
PQ_PAT = re.compile(
    r"(^|_)(p|q)$|p_?spin|p_?perm|p_?val|pvalue|p_?boot|p_?model|p_?fdr|"
    r"(^|_)q_?(val|value|fdr|bh)?$|fdr|_p$|^p_|_q$|^q_|spin_p|perm_p|p_label",
    re.I)

#: N used by each family, where it is known from the code or the spec. The repo
#: standard is 1000 rotations (runner section 4); families that differ are named.
FAMILY_N = {
    "definitional/layer_marker_genes.csv": (1000, "ENIGMA rotate_parcellation, "
                                            "nrot=1000, seed 0, two directions averaged"),
    "definitional/negative_control.csv": (1000, "released vertex spin nulls, seed 0"),
}
DEFAULT_N = 1000

#: A floor argument only applies to a resampling test. A parametric or binomial
#: p-value has no floor: it can legitimately underflow to zero at large n, and
#: flagging those would bury the real finding in false positives.
RESAMPLING_COLS = re.compile(r"spin|perm|boot|shuffl", re.I)
PARAMETRIC_COLS = re.compile(r"param|binom|pearson_p|spearman_p|p_model|_t$|chi2", re.I)


def classify_test(col: str, fname: str) -> str:
    c = str(col)
    if RESAMPLING_COLS.search(c):
        return "resampling"
    if PARAMETRIC_COLS.search(c):
        return "parametric"
    if "outcome_table" in fname:
        return "harvested"      # rr2 copies whatever its source reported
    return "unclassified"


def _log_factory(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a)
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()
    return log, fh


# --------------------------------------------------------------------------- #
# Step 2 - sweep every family
# --------------------------------------------------------------------------- #


def sweep(log) -> list[dict]:
    """Every p or q value that is exactly 0, or below the floor for its N."""
    rows = []
    files = sorted(p for p in FIGV9.rglob("*.csv") if "_cache" not in p.parts)
    log(f"  scanning {len(files)} CSV files under figures/v9/")
    for f in files:
        rel = f.relative_to(FIGV9).as_posix()
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            log(f"    [skip] {rel}: {type(exc).__name__}")
            continue
        n_used, n_note = FAMILY_N.get(rel, (DEFAULT_N, "assumed repo default"))
        # a table that states its own permutation count is authoritative over any
        # assumption: per_type_connectivity_trends.csv carries n_perm = 5040
        # (exhaustive 7! orderings), whose floor is nothing like 1/1001
        n_col = next((c for c in df.columns
                      if str(c).lower() in ("n_perm", "n_spin", "n_rot", "n_rotations")),
                     None)
        if n_col is not None:
            stated = pd.to_numeric(df[n_col], errors="coerce").dropna()
            if stated.size and stated.nunique() == 1:
                n_used = int(stated.iloc[0])
                n_note = f"stated in the file's {n_col} column"
        floor_plus1 = 1.0 / (n_used + 1)
        floor_raw = 1.0 / n_used
        for col in df.columns:
            if not PQ_PAT.search(str(col)):
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            if not s.notna().any():
                continue
            if s.min(skipna=True) < 0 or s.max(skipna=True) > 1:
                continue          # not a probability column
            for i, v in s.items():
                if not np.isfinite(v):
                    continue
                is_zero = (v == 0.0)
                # CSV round-tripping truncates 1/1001 to 0.0009990009990009, which is
                # below the true floor by about 1e-19. A bare "<" calls that a
                # violation; a relative tolerance does not, which matters because
                # otherwise 39 correct values drown the real finding.
                at_floor = abs(v - floor_plus1) <= 1e-9 * max(floor_plus1, 1e-12)
                below = (v < floor_plus1) and not at_floor
                if not (is_zero or below):
                    continue
                label = ""
                for key in ("feature", "measure", "name", "test", "map", "metric",
                            "disorder", "spec", "resolution", "atlas", "gene"):
                    if key in df.columns:
                        label = str(df[key].iloc[i])
                        break
                kind = classify_test(col, rel)
                rows.append({
                    "file": rel, "column": col, "test_kind": kind,
                    "row_index": int(i),
                    "row_label": label, "value": float(v),
                    "n_used": n_used, "n_basis": n_note,
                    "floor_plus_one": round(floor_plus1, 8),
                    "floor_raw_tail": round(floor_raw, 8),
                    "exactly_zero": bool(is_zero),
                    "equals_floor_within_tolerance": bool(at_floor),
                    # a floor only exists for a resampling test; a parametric or
                    # binomial p may legitimately underflow to zero
                    "violates_stated_floor": bool(below and kind == "resampling"),
                })
    viol = [r for r in rows if r["violates_stated_floor"]]
    log(f"  {len(rows)} values at or below the stated floor overall "
        f"({sum(r['exactly_zero'] for r in rows)} exactly zero)")
    log(f"  of those, {len(viol)} are RESAMPLING tests and so genuinely violate a floor:")
    seen = {}
    for r in viol:
        k = (r["file"], r["column"])
        seen[k] = seen.get(k, 0) + 1
    for (f_, c_), n_ in sorted(seen.items()):
        log(f"    {f_}:{c_}  x{n_}")
    bykind = {}
    for r in rows:
        bykind[r["test_kind"]] = bykind.get(r["test_kind"], 0) + 1
    log(f"  breakdown by test kind: {bykind}")
    return rows


# --------------------------------------------------------------------------- #
# Step 5 - families whose smallest p sits at the floor for many members
# --------------------------------------------------------------------------- #


def floor_pileup(log) -> list[dict]:
    """Members sitting exactly at a family's minimum, which can mean too few rotations."""
    out = []
    files = sorted(p for p in FIGV9.rglob("*.csv") if "_cache" not in p.parts)
    for f in files:
        rel = f.relative_to(FIGV9).as_posix()
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        n_used, _note = FAMILY_N.get(rel, (DEFAULT_N, ""))
        for col in df.columns:
            if not PQ_PAT.search(str(col)):
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.size < 3 or s.min() < 0 or s.max() > 1:
                continue
            mn = s.min()
            at_min = int((s == mn).sum())
            floors = {0.0, 1.0 / n_used, 1.0 / (n_used + 1), 1.0 / (2 * n_used),
                      1.0 / (2 * n_used + 1)}
            near_floor = any(abs(mn - fl) < 1e-9 for fl in floors)
            if at_min >= 2 and near_floor:
                out.append({"file": rel, "column": col, "n_members": int(s.size),
                            "min_value": float(mn), "n_at_min": at_min,
                            "frac_at_min": round(at_min / s.size, 3),
                            "n_used": n_used,
                            "interpretation": ("minimum is a resolution floor, not a "
                                               "measured value; more rotations would "
                                               "separate these members")})
    for r in out:
        log(f"    {r['file']}:{r['column']}  {r['n_at_min']}/{r['n_members']} members at "
            f"{r['min_value']:.6g}")
    return out


# --------------------------------------------------------------------------- #
# Step 3 - recompute the layer-marker family under the stated convention
# --------------------------------------------------------------------------- #


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman with pairwise NaN handling, matching what perm_sphere_p does."""
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def build_nulls(vec: np.ndarray, cr: np.ndarray, perm: np.ndarray):
    """Reproduce ENIGMA's two null directions, vectorised.

    perm_sphere_p permutes x against y and y against x, giving two null vectors of
    length nperm; the published p is the mean of the two one-tailed fractions. The
    same two vectors are what the section 2.6 convention needs, so they are built
    once here and reused for every variant.
    """
    nperm = perm.shape[1]
    xy = np.empty(nperm)
    yx = np.empty(nperm)
    for r in range(nperm):
        idx = perm[:, r].astype(int)
        xy[r] = rank_corr(vec[idx], cr)
        yx[r] = rank_corr(vec, cr[idx])
    return xy, yx


def p_variants(obs: float, xy: np.ndarray, yx: np.ndarray, nperm: int) -> dict:
    """The published value and the section 2.6 value, from the same nulls."""
    # as published: one-tailed, signed, strict >, no plus-one, averaged
    if obs >= 0:
        pub = (np.sum(xy > obs) / nperm + np.sum(yx > obs) / nperm) / 2.0
    else:
        pub = (np.sum(xy < obs) / nperm + np.sum(yx < obs) / nperm) / 2.0
    both = np.concatenate([xy, yx])
    # section 2.6 as written: two-tailed on |rho|, >=, plus-one, one null distribution
    stated_2000 = (1 + int(np.sum(np.abs(both) >= abs(obs)))) / (both.size + 1)
    # the same convention applied per direction at N=1000, then averaged
    a = (1 + int(np.sum(np.abs(xy) >= abs(obs)))) / (nperm + 1)
    b = (1 + int(np.sum(np.abs(yx) >= abs(obs)))) / (nperm + 1)
    stated_1000 = (a + b) / 2.0
    return {"p_published": float(pub),
            "p_stated_2tail_plus1_N2000": float(stated_2000),
            "p_stated_2tail_plus1_N1000_avg": float(stated_1000),
            "n_null_values": int(both.size)}


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = o[rank]
        val = p[i] * n / (rank + 1)
        prev = min(prev, val)
        q[i] = prev
    return np.minimum(q, 1.0)


def recompute_layer_markers(log) -> tuple[pd.DataFrame, dict]:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import layer_marker_genes as lmg

    published = pd.read_csv(DEF / "layer_marker_genes.csv")
    expr = lmg.get_gene_expression(log)
    names, perm = lmg.aparc_order_and_perm(1000)
    crank = lmg.crank_by_name()
    cr = np.array([crank.get(nm, np.nan) for (_h, nm) in names], float)
    nperm = perm.shape[1]
    log(f"  parcel rotations: {nperm}, seed {lmg.SEED}, {len(names)} parcels")

    def gene_vec(g):
        return np.array([expr.loc[(h, nm), g] if (h, nm) in expr.index else np.nan
                         for (h, nm) in names], float)

    genes = [g for g in lmg.GENES if g in expr.columns]
    z = {g: (gene_vec(g) - np.nanmean(gene_vec(g))) / np.nanstd(gene_vec(g)) for g in genes}
    vectors = {g: gene_vec(g) for g in genes}
    if "CUX2" in z:
        vectors["upper(CUX2)"] = z["CUX2"]
    if "RORB" in z:
        vectors["granular(RORB)"] = z["RORB"]
    deep = [g for g in ("FEZF2", "FOXP2", "TLE4") if g in z]
    if deep:
        vectors["deep(FEZF2/FOXP2/TLE4)"] = np.nanmean([z[g] for g in deep], axis=0)

    rows = []
    for feat in published["feature"]:
        v = vectors.get(feat)
        if v is None:
            log(f"    [WARN] {feat} not reconstructable")
            continue
        keep = np.isfinite(v) & np.isfinite(cr)
        rho = float(stats.spearmanr(v[keep], cr[keep])[0])
        t0 = time.time()
        xy, yx = build_nulls(v, cr, perm)
        pv = p_variants(rank_corr(v, cr), xy, yx, nperm)
        pub_row = published[published["feature"] == feat].iloc[0]
        rows.append({"feature": feat, "kind": pub_row["kind"],
                     "spearman_rho_recomputed": rho,
                     "spearman_rho_published": float(pub_row["spearman_rho"]),
                     "p_spin_published": float(pub_row["p_spin"]),
                     **pv,
                     "seconds": round(time.time() - t0, 1)})
        log(f"    {feat:26s} rho {rho:+.4f}  published p {float(pub_row['p_spin']):.4f}  "
            f"recomputed-as-published {pv['p_published']:.4f}  "
            f"stated-convention {pv['p_stated_2tail_plus1_N2000']:.5f}")

    df = pd.DataFrame(rows)
    df["q_published_recomputed"] = bh(df["p_spin_published"].values)
    df["q_stated_2tail_plus1_N2000"] = bh(df["p_stated_2tail_plus1_N2000"].values)
    df["q_stated_2tail_plus1_N1000_avg"] = bh(df["p_stated_2tail_plus1_N1000_avg"].values)
    df["q_published_in_csv"] = published.set_index("feature").loc[
        df["feature"], "p_spin_fdr"].values
    reproduced = bool(np.allclose(df["p_published"], df["p_spin_published"], atol=1e-9))
    log(f"  reproduces the published p values exactly: {reproduced}")
    return df, {"n_rotations": nperm, "seed": lmg.SEED,
                "n_parcels": len(names),
                "reproduces_published_p": reproduced,
                "floor_published_convention": 0.0,
                "floor_stated_convention_N2000": 1.0 / 2001,
                "floor_stated_convention_N1000": 1.0 / 1001}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

FORMULA = """As implemented (enigmatoolbox.permutation_testing.perm_sphere_p, called by
scripts/layer_marker_genes.py with nrot=1000, numpy seed 0):

    rho_emp = spearman(x, y)
    for each of nperm rotations r:
        rho_null_xy[r] = spearman(x[perm[:, r]], y)
        rho_null_yx[r] = spearman(x, y[perm[:, r]])

    if rho_emp >= 0:
        p_xy = sum(rho_null_xy >  rho_emp) / nperm
        p_yx = sum(rho_null_yx >  rho_emp) / nperm
    else:
        p_xy = sum(rho_null_xy <  rho_emp) / nperm
        p_yx = sum(rho_null_yx <  rho_emp) / nperm

    p = (p_xy + p_yx) / 2

Section 2.6 states instead:

    p = (1 + #{|rho_null| >= |rho_obs|}) / (N + 1)

Four differences: no plus-one correction (so zero is attainable); one-tailed on the
signed correlation rather than two-tailed on |rho|; strict > rather than >=; and an
average of two null directions rather than one null distribution. The granularity of
the published value is 1/(2*nperm) = 0.0005, which is why the reported values are all
multiples of 0.0005."""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-recompute", action="store_true")
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log, fh = _log_factory(OUTDIR / "run_rr13.log")
    t0 = time.time()

    frozen_before = rc.frozen_hashes()
    extra_before = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    log(f"frozen: {len(frozen_before)} released v9 + {len(extra_before)} RR10 files")

    log("\n== Step 1: the formula actually used ==")
    log(FORMULA)
    (OUTDIR / "formula_found.txt").write_text(FORMULA + "\n", encoding="utf-8")

    log("\n== Step 2: sweep every family ==")
    rows = sweep(log)
    with open(OUTDIR / "pvalue_floor_audit.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["file", "column", "test_kind", "row_index", "row_label", "value",
                "n_used", "n_basis", "floor_plus_one", "floor_raw_tail",
                "exactly_zero", "equals_floor_within_tolerance",
                "violates_stated_floor"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    log("\n== Step 5: floor pile-up (too few rotations to resolve the tail) ==")
    pile = floor_pileup(log)

    corrected, meta = (None, {})
    if not args.skip_recompute:
        log("\n== Step 3: recompute the layer-marker family under the stated convention ==")
        corrected, meta = recompute_layer_markers(log)
        corrected.to_csv(OUTDIR / "layer_marker_corrected.csv", index=False)

    ok, detail = rc.check_frozen(frozen_before)
    extra_after = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    ok_extra = extra_before == extra_after
    log(f"\nreleased v9 unchanged: {ok}; RR10 crossed files unchanged: {ok_extra}")

    (OUTDIR / "rr13_summary.json").write_text(json.dumps({
        "formula": FORMULA, "layer_marker_meta": meta,
        "n_floor_violations": len(rows),
        "n_exact_zeros": sum(r["exactly_zero"] for r in rows),
        "floor_audit": rows, "floor_pileup": pile,
        "corrected": (corrected.to_dict("records") if corrected is not None else None),
        "frozen_v9_unchanged": ok, "frozen_rr10_unchanged": ok_extra,
        "runtime_seconds": round(time.time() - t0, 1)}, indent=2, default=float),
        encoding="utf-8")
    log(f"total runtime {(time.time()-t0)/60:.1f} min")
    fh.close()
    return rows, corrected


if __name__ == "__main__":
    main()
