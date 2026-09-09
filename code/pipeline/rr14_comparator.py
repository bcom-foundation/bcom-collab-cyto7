"""RR14 - the comparator analysis, the granularity-matched null, and the M1 sensitivity.

Three referee points that share one computation. A plausible alternative seven-class
partition of cortex is both the comparator M2 asks for ("does cyto7 beat a seven-way
quantisation of the T1w/T2w map itself?") and the granularity-matched null M3 asks for
("the 46% chance baseline is scrambled cyto7, not a plausible alternative").

The expected result is stated in the spec and is not a failure: a myelin septile will
match or beat cyto7 on myelin and on anything correlated with myelin. That is
arithmetic. What matters is RORB and the layer-marker family, which is why they are in
the panel.

Nothing published is changed. The comparators are analysis objects, not products.

Run::
    conda run -n cyto7 python scripts/rr14_comparator.py --n-part 200
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import rr_common as rc
from cyto7_surface_io import REPO_ROOT, resolve_target_map

OUTDIR = rc.OUT / "rr14_comparator"
CACHEDIR = rc.OUT / "_cache"
NM = REPO_ROOT / "resources" / "neuromaps_cache"
C7CACHE = REPO_ROOT / "resources" / "cyto7_derived" / "cache"
CROSSED_FIG = REPO_ROOT / "figures" / "v9" / "crossed"
DEF = REPO_ROOT / "figures" / "v9" / "definitional"
SF = REPO_ROOT / "figures" / "v9" / "structure_function"
CROSSED_RES = REPO_ROOT / "resources" / "cyto7_derived" / "crossed"

FROZEN_EXTRA = sorted(CROSSED_RES.glob("*.annot")) + \
    sorted(CROSSED_RES.glob("*.label.gii")) + \
    [CROSSED_RES / "aparc_cyto7_nodes.tsv", CROSSED_RES / "voneconomo_cyto7_nodes.tsv"]

HEMIS = ("L", "R")
SEED = 0
N_SPIN = 1000
ISO = list(range(2, 8))          # the win-fraction test runs on isocortex, codes 2..7


def _log_factory(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a)
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()
    return log, fh


def cat(d):
    return np.concatenate([np.asarray(d[H]) for H in HEMIS])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_core(log):
    from summarise_functional_features import (FEATURES, build_validity_mask,
                                               load_all_features)
    labels = {H: np.asarray(v, int) for H, v in
              resolve_target_map("v9", "fs_LR").items()}
    feats = load_all_features("Validation210")
    valid = build_validity_mask(labels, feats)
    aparc = {H: np.load(C7CACHE / f"aparc_fsLR32k_hemi-{H}.npy") for H in HEMIS}
    # the decoded von-Economo-derived cyto7 code (0, 2..7), which is what the
    # published benchmark compares against. economo_fsLR32k_*.npy is the raw area
    # id (1..43) and using it makes almost every vertex "disagree".
    ve_cache = REPO_ROOT / "resources" / "reviewer_response" / "cache"
    econ = {H: np.load(ve_cache / f"voneconomo_code_nearest_{H}.npy").astype(int)
            for H in HEMIS}
    log(f"  32k fs_LR: {cat(valid).sum()} valid vertices of {cat(valid).size}")
    return FEATURES, labels, feats, valid, aparc, econ


def load_extra_maps(log) -> dict:
    """Maps used to define comparators but not part of the released FEATURES panel."""
    out = {}
    for key, stem in (("gene_pc1", "abagen_genepc1"),
                      ("histg1", "bigbrain_histg1"),
                      ("receptor_pc1", "hansen_receptorpc1")):
        try:
            out[key] = {H: np.load(NM / f"{stem}_fsLR32k_hemi-{H}.npy") for H in HEMIS}
        except OSError:
            log(f"  [skip] {key}: {stem} not cached")
    log(f"  extra maps loaded: {sorted(out)}")
    return out


# --------------------------------------------------------------------------- #
# Step 1 - comparators
# --------------------------------------------------------------------------- #


def septile(values: np.ndarray, valid: np.ndarray, n_levels: int = 7) -> np.ndarray:
    """Equal-count quantisation into ordered levels 1..n over the valid set.

    Equal-count, not equal-width, so the comparator matches cyto7's ordinal
    structure without importing cyto7's type-size distribution.
    """
    out = np.zeros(values.shape[0], int)
    v = values[valid]
    edges = np.quantile(v, np.linspace(0, 1, n_levels + 1)[1:-1])
    out[valid] = np.searchsorted(edges, v, side="right") + 1
    return out


def mesh_adjacency():
    """Sparse vertex adjacency of the 32k fs_LR mesh, both hemispheres stacked.

    The two hemispheres are kept disconnected, which is correct: a region cannot
    grow across the midline, and cyto7's own types are per-hemisphere structures.
    """
    from scipy import sparse
    geom = rc.fslr_geometry("midthickness")
    blocks, off = [], 0
    n_tot = sum(geom[H][0].shape[0] for H in HEMIS)
    rows, cols = [], []
    for H in HEMIS:
        coords, faces = geom[H]
        f = faces + off
        e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
        e = np.vstack([e, e[:, ::-1]])
        rows.append(e[:, 0]); cols.append(e[:, 1])
        off += coords.shape[0]
    del blocks
    r = np.concatenate(rows); c = np.concatenate(cols)
    A = sparse.coo_matrix((np.ones(r.size, np.int8), (r, c)),
                          shape=(n_tot, n_tot)).tocsr()
    A.data[:] = 1
    return A


def grow_partition(A, valid: np.ndarray, sizes: list[int], rng,
                   hemi_offsets: list[tuple[int, int]]) -> np.ndarray:
    """One spatially contiguous, size-matched seven-class partition.

    Grown separately in each hemisphere, with that hemisphere's own cyto7 type
    sizes as targets. The hemispheres are disconnected on the mesh, so growing
    across both at once strands whichever regions were seeded in the other
    hemisphere and destroys the size matching; that was worth getting right,
    because a ragged partition would make the baseline artificially easy to beat.

    Within a hemisphere: seven random seeds, then repeatedly give one frontier
    vertex to whichever region is furthest below its quota in relative terms. A
    region that reaches its target stops growing. Anything left stranded is
    assigned to an adjacent region afterwards.
    """
    lab = np.zeros(valid.size, int)
    indptr, indices = A.indptr, A.indices
    for (lo, hi), sz in zip(hemi_offsets, sizes):
        vmask = np.zeros(valid.size, bool)
        vmask[lo:hi] = valid[lo:hi]
        idx = np.where(vmask)[0]
        k = len(sz)
        if idx.size < k * 2:
            continue
        # farthest-point seeding: a random first seed, then each next seed as far
        # as possible from those already chosen. Well-separated seeds stop a region
        # being boxed in before it reaches its quota, which is what left the sizes
        # ragged under purely random seeding.
        seeds = [int(rng.choice(idx))]
        d = np.full(valid.size, np.inf)
        for _ in range(k - 1):
            frontier_bfs = {seeds[-1]}
            dist = 0
            seen = {seeds[-1]}
            while frontier_bfs:
                for u in frontier_bfs:
                    if dist < d[u]:
                        d[u] = dist
                nxt = set()
                for u in frontier_bfs:
                    for nb in indices[indptr[u]:indptr[u + 1]]:
                        if vmask[nb] and nb not in seen:
                            seen.add(int(nb)); nxt.add(int(nb))
                frontier_bfs = nxt
                dist += 1
            cand = d[idx]
            seeds.append(int(idx[int(np.argmax(cand))]))
        seeds = np.array(seeds)
        frontier = [set() for _ in range(k)]
        count = np.zeros(k, int)
        for j, sd in enumerate(seeds):
            lab[sd] = j + 1
            count[j] = 1
            for nb in indices[indptr[sd]:indptr[sd + 1]]:
                if vmask[nb] and lab[nb] == 0:
                    frontier[j].add(int(nb))
        target = np.asarray(sz, float)
        remaining = int(vmask.sum()) - k
        while remaining > 0:
            deficit = np.where(count < target, (target - count) / target, -np.inf)
            placed = False
            for j in np.argsort(-deficit):
                if not np.isfinite(deficit[j]):
                    continue
                fr = frontier[j]
                while fr:
                    v = fr.pop()
                    if lab[v] != 0:
                        continue
                    lab[v] = j + 1
                    count[j] += 1
                    remaining -= 1
                    for nb in indices[indptr[v]:indptr[v + 1]]:
                        if vmask[nb] and lab[nb] == 0:
                            fr.add(int(nb))
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                break
        # anything stranded (an island, or boxed in once quotas were met) goes to
        # an adjacent region, preferring the one still furthest below its quota
        left = np.where(vmask & (lab == 0))[0]
        guard = 0
        while left.size and guard < 200:
            guard += 1
            progressed = False
            for v in left:
                nbl = lab[indices[indptr[v]:indptr[v + 1]]]
                nbl = nbl[nbl > 0]
                if nbl.size == 0:
                    continue
                cand = np.unique(nbl)
                defs = [(target[c - 1] - count[c - 1]) / target[c - 1] for c in cand]
                j = int(cand[int(np.argmax(defs))])
                lab[v] = j
                count[j - 1] += 1
                progressed = True
            left = np.where(vmask & (lab == 0))[0]
            if not progressed:
                break
    return lab


def build_matched_nulls(A, valid, cyto_c, n_hemi, n_part, log,
                        cache_tag="rr14_matched"):
    """*n_part* size-matched smooth partitions with cyto7's own type-size profile."""
    cpath = CACHEDIR / f"{cache_tag}_n{n_part}.npy"
    if cpath.exists():
        log(f"  reusing cached matched partitions: {cpath.name}")
        return np.load(cpath)
    offs = [(0, n_hemi[0]), (n_hemi[0], n_hemi[0] + n_hemi[1])]
    sizes = []
    for lo, hi in offs:
        sizes.append([int((cyto_c[lo:hi] == t).sum()) for t in range(1, 8)])
    log(f"  cyto7 type sizes, L: {sizes[0]}")
    log(f"  cyto7 type sizes, R: {sizes[1]}")
    rng = np.random.default_rng(SEED)
    out = np.zeros((n_part, valid.size), np.int8)
    t0 = time.time()
    for i in range(n_part):
        out[i] = grow_partition(A, valid, sizes, rng, offs)
        if (i + 1) % 25 == 0:
            log(f"    partition {i + 1}/{n_part}  ({time.time() - t0:.0f}s)")
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    np.save(cpath, out)
    return out


def partition_quality(parts, A, valid, cyto_c, log) -> pd.DataFrame:
    """Prove the matched partitions really are contiguous and size-matched."""
    from scipy import sparse
    sizes_c = np.array([(cyto_c == t).sum() for t in range(1, 8)], float)
    rows = []
    for i in range(min(len(parts), 25)):
        p = parts[i]
        sz = np.array([(p == t).sum() for t in range(1, 8)], float)
        ncomp = []
        for t in range(1, 8):
            idx = np.where(p == t)[0]
            if idx.size == 0:
                ncomp.append(0); continue
            sub = A[idx][:, idx]
            nc, _ = sparse.csgraph.connected_components(sub, directed=False)
            ncomp.append(nc)
        rows.append({"partition": i, "max_size_error_pct":
                     float(np.abs(sz - sizes_c).max() / sizes_c.max() * 100),
                     "n_components_max": int(max(ncomp)),
                     "n_components_mean": float(np.mean(ncomp))})
    df = pd.DataFrame(rows)
    log(f"  matched-partition quality over {len(df)} sampled partitions: "
        f"max size error {df.max_size_error_pct.max():.2f}%, "
        f"components per class max {int(df.n_components_max.max())}, "
        f"mean {df.n_components_mean.mean():.2f}")
    return df


# --------------------------------------------------------------------------- #
# Rotations: recover the permutation once, reuse for every partition
# --------------------------------------------------------------------------- #


def rotation_index(valid_c, log, n_spin=N_SPIN):
    """(n_vert, n_spin) vertex permutation from alexander_bloch, seed 0.

    The rotation depends on the sphere and the seed, not on the data, so rotating
    an index array recovers the permutation itself and it can then be applied to
    any map. Validated below against the released cyto7 null file, which is what
    makes reusing it legitimate rather than merely convenient.
    """
    cpath = CACHEDIR / f"rr14_rotidx_seed{SEED}_n{n_spin}.npy"
    if cpath.exists():
        log(f"  reusing cached rotation index: {cpath.name}")
        return np.load(cpath)
    from neuromaps.nulls import alexander_bloch
    n = valid_c.size
    idx = np.full(n, np.nan)
    idx[valid_c] = np.arange(n, dtype=float)[valid_c]
    rot = alexander_bloch(idx, atlas="fsLR", density="32k", n_perm=n_spin, seed=SEED)
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    np.save(cpath, rot)
    return rot


def validate_rotation(rotidx, labels_c, valid_c, log) -> dict:
    """Applying the recovered permutation to cyto7 must reproduce the released nulls."""
    released = rc.NULLS
    if not released.exists():
        log("  [WARN] released null file absent; rotation index not validated")
        return {"validated": False, "reason": "released null file absent"}
    ref = np.load(released, mmap_mode="r")
    rank = np.full(valid_c.size, np.nan)
    rank[valid_c] = labels_c[valid_c].astype(float)
    ok, checked = 0, 0
    for i in range(min(20, rotidx.shape[1])):
        col = rotidx[:, i]
        m = np.isfinite(col)
        got = np.full(valid_c.size, np.nan)
        got[m] = rank[col[m].astype(int)]
        want = np.asarray(ref[:, i], float)
        both = np.isfinite(got) & np.isfinite(want)
        checked += 1
        if both.sum() and np.allclose(got[both], want[both]):
            ok += 1
    log(f"  rotation index reproduces the released cyto7 nulls in {ok}/{checked} "
        f"checked rotations")
    return {"validated": ok == checked, "n_checked": checked, "n_match": ok}


# --------------------------------------------------------------------------- #
# Step 2 - the panel
# --------------------------------------------------------------------------- #


def spin_p(vals, level, valid_c, rotidx, n_spin=N_SPIN) -> tuple:
    """Spearman rho and the plus-one two-tailed spin p (Annex D convention)."""
    obs = float(stats.spearmanr(vals[valid_c], level[valid_c])[0])
    nr = np.empty(n_spin)
    lv = level.astype(float)
    for i in range(n_spin):
        col = rotidx[:, i]
        m = np.isfinite(col)
        spun = np.full(level.size, np.nan)
        spun[m] = lv[col[m].astype(int)]
        ok = valid_c & np.isfinite(spun) & (spun > 0)
        nr[i] = stats.spearmanr(vals[ok], spun[ok])[0]
    good = nr[np.isfinite(nr)]
    p = float((np.sum(np.abs(good) >= abs(obs)) + 1) / (good.size + 1))
    return obs, p


def bh(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for r in range(n - 1, -1, -1):
        i = o[r]
        prev = min(prev, p[i] * n / (r + 1))
        q[i] = prev
    return np.minimum(q, 1.0)



# --------------------------------------------------------------------------- #
# Step 2b - parcel level: RORB and the layer-marker family
# --------------------------------------------------------------------------- #


def parcel_scores(level: np.ndarray, aparc_c: np.ndarray, valid_c: np.ndarray,
                  names: list[str], va_c: np.ndarray) -> dict[str, float]:
    """Area-weighted mean level per Desikan parcel, hemisphere-pooled.

    The same composition rule the released cyto7 parcel score uses, applied to
    every partition alike so the comparison is like for like. Computed on 32k
    fs_LR because that is where the comparators are defined; cyto7's published
    fsaverage-derived score is carried alongside as a check.
    """
    out = {}
    for code, nm in enumerate(names):
        low = str(nm).lower()
        if low in ("unknown", "corpuscallosum", "", "???"):
            continue
        m = valid_c & (aparc_c == code) & (level > 0)
        if m.sum() < 20:
            continue
        w = va_c[m]
        out[low] = float(np.average(level[m].astype(float), weights=w))
    return out


def parcel_panel(parts_for_panel: dict, aparc_c, valid_c, log) -> pd.DataFrame:
    """Spearman of each partition's parcel score against each layer-marker gene."""
    expr_csv = DEF / "layer_marker_expression.csv"
    if not expr_csv.exists():
        log("  [skip] layer-marker expression CSV absent; parcel panel not run")
        return pd.DataFrame()
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import layer_marker_genes as lmg

    raw = pd.read_csv(expr_csv)
    raw["hemi"] = raw["hemi"].str.lower()
    raw["parcel"] = raw["parcel"].str.lower()
    expr = raw.set_index(["hemi", "parcel"])
    genes = [g for g in lmg.GENES if g in expr.columns]

    names = aparc_names(log)
    geom = rc.fslr_geometry("midthickness")
    va_c = np.concatenate([rc.vertex_areas(*geom[H]) for H in HEMIS])

    # hemisphere-pooled gene value per parcel, matching the pooled parcel score
    gene_by_parcel = {}
    for g in genes:
        d = {}
        for (h, nm), row in expr.iterrows():
            d.setdefault(nm, []).append(row[g])
        gene_by_parcel[g] = {k: float(np.nanmean(v)) for k, v in d.items()}

    published = pd.read_csv(DEF / "layer_marker_genes.csv").set_index("feature")
    corrected = None
    cpath = rc.OUT / "rr13_pvalues" / "layer_marker_corrected.csv"
    if cpath.exists():
        corrected = pd.read_csv(cpath).set_index("feature")

    rows = []
    for pname, lv in parts_for_panel.items():
        sc = parcel_scores(lv, aparc_c, valid_c, names, va_c)
        for g in genes:
            gp = gene_by_parcel[g]
            common = sorted(set(sc) & set(gp))
            x = np.array([sc[k] for k in common], float)
            y = np.array([gp[k] for k in common], float)
            ok = np.isfinite(x) & np.isfinite(y)
            rho = float(stats.spearmanr(x[ok], y[ok])[0])
            rec = {"gene": g, "partition": pname, "n_parcels": int(ok.sum()),
                   "spearman_rho": rho}
            if pname.startswith("cyto7"):
                rec["published_rho_fsaverage"] = float(published.loc[g, "spearman_rho"]) \
                    if g in published.index else np.nan
                if corrected is not None and g in corrected.index:
                    rec["published_p_corrected_rr13"] = float(
                        corrected.loc[g, "p_stated_2tail_plus1_N1000_avg"])
            rows.append(rec)
    df = pd.DataFrame(rows)
    for g in genes:
        sub = df[df.gene == g]
        best = sub.loc[sub.spearman_rho.abs().idxmax()]
        c7 = sub[sub.partition.str.startswith("cyto7")].iloc[0]
        log(f"    {g:6s} cyto7 rho {c7.spearman_rho:+.4f} | " + "  ".join(
            f"{r.partition.split()[0]}={r.spearman_rho:+.3f}"
            for _, r in sub.iterrows() if not r.partition.startswith("cyto7"))
            + f"  -> strongest: {best.partition}")
    return df

# --------------------------------------------------------------------------- #
# Step 3 - the win-fraction test
# --------------------------------------------------------------------------- #

AXIS = ["myelin", "thickness", "gradient"]


def median_lut(fv, codes, mask):
    """Per-code medians of *fv* over *mask*, as a lookup indexed by code."""
    lut = np.full(int(codes.max()) + 2, np.nan)
    for c in np.unique(codes[mask]):
        if c <= 0:
            continue
        m = mask & (codes == c)
        if m.any():
            lut[int(c)] = np.median(fv[m])
    return lut


def win_fraction(fvs, challenger, ve, median_mask, disagree) -> tuple:
    """Win fraction for *challenger* against von Economo on the disagreement set.

    The published statistic pools the three arbiter judgments as separate events
    rather than taking a majority vote: with per-arbiter fractions 0.616, 0.568
    and 0.619 the pooled value is 0.601, which is the published number, whereas a
    majority-of-three vote gives 0.628. Reproducing 0.601 exactly for cyto7 is
    what validates this reimplementation, so the pooled form is used.
    """
    per, allw = {}, []
    for k in AXIS:
        fv = fvs[k]
        lc = median_lut(fv, challenger, median_mask)
        lv = median_lut(fv, ve, median_mask)
        dc = np.abs(fv - lc[np.clip(challenger, 0, lc.size - 1)])
        dv = np.abs(fv - lv[np.clip(ve, 0, lv.size - 1)])
        w = (dc < dv)[disagree]
        per[k] = float(np.mean(w))
        allw.append(w)
    pooled = float(np.mean(np.concatenate(allw)))
    return pooled, per


# --------------------------------------------------------------------------- #
# Step 4 - the M1 sensitivity
# --------------------------------------------------------------------------- #

#: Desikan precentral is M1. The paper's agranular exclusion drops a type; the M1
#: exclusion drops an anatomical territory, so it is defined on the parcellation.
M1_PARCELS = ("precentral",)


def aparc_names(log):
    """Desikan code -> name, from the fsaverage annot the 32k cache was built from."""
    import nibabel as nib
    p = REPO_ROOT / "resources" / "voneconomo" / "lh.aparc.annot"
    _lab, _ctab, names = nib.freesurfer.io.read_annot(str(p))
    return [n.decode() if isinstance(n, bytes) else n for n in names]


def m1_sensitivity(feats, labels_c, valid_c, aparc_c, rotidx, extra, log) -> pd.DataFrame:
    names = aparc_names(log)
    keep_codes = [i for i, n in enumerate(names) if n.lower() in M1_PARCELS]
    if not keep_codes:
        log("  [WARN] precentral not found in the aparc names; M1 sensitivity skipped")
        return pd.DataFrame()
    m1 = np.isin(aparc_c, keep_codes)
    log(f"  M1 (precentral) on the valid set: {int((m1 & valid_c).sum())} vertices "
        f"({100 * (m1 & valid_c).sum() / valid_c.sum():.1f}% of valid cortex)")
    rows = []
    targets = {"myelin": feats["myelin"], "thickness": feats["thickness"],
               "gradient": feats["gradient"]}
    if "gene_pc1" in extra:
        targets["gene_pc1"] = extra["gene_pc1"]
    for key, m in targets.items():
        vals = cat(m) if isinstance(m, dict) else m
        # gene PC1 is not finite everywhere on the 32k mesh, so the base mask has to
        # carry the finite check or every correlation comes back NaN
        base = valid_c & np.isfinite(vals)
        for label, mask in (("with M1", base), ("without M1", base & ~m1)):
            rho, p = spin_p(vals, labels_c, mask, rotidx)
            rows.append({"feature": key, "set": label, "n_vertices": int(mask.sum()),
                         "spearman_rho": rho, "spin_p": p})
        w = [r for r in rows if r["feature"] == key]
        log(f"    {key:10s} with M1 rho {w[0]['spearman_rho']:+.4f} (p {w[0]['spin_p']:.4f}), "
            f"without M1 rho {w[1]['spearman_rho']:+.4f} (p {w[1]['spin_p']:.4f}), "
            f"delta {w[1]['spearman_rho'] - w[0]['spearman_rho']:+.4f}")
    df = pd.DataFrame(rows)
    piv = df.pivot(index="feature", columns="set", values="spearman_rho")
    piv["delta"] = piv["without M1"] - piv["with M1"]
    return df, piv


# --------------------------------------------------------------------------- #
# Step 5 - the BigBrain arbiter, retrieved
# --------------------------------------------------------------------------- #


def bigbrain_arbiter(log) -> pd.DataFrame:
    """Not a new analysis: pull the released bigbrain_added_value_2 family."""
    rows = []
    oc = rc.OUT / "rr2_table" / "outcome_table.csv"
    if oc.exists():
        t = pd.read_csv(oc)
        fam = t[t["family"] == "bigbrain_added_value_2"]
        for _, r in fam.iterrows():
            rows.append({"test": r["measure"], "key": r["key"],
                         "effect": r["effect"], "p": r["p_raw"], "q": r["q"],
                         "survives": r["survives"], "status_in_table_s5": r["status"],
                         "source": r["source_file"]})
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        log(f"    {r['key']}: {r['effect']}, p = {r['p']}, status '{r['status_in_table_s5']}'")
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-part", type=int, default=200)
    ap.add_argument("--n-spin", type=int, default=N_SPIN)
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log, fh = _log_factory(OUTDIR / "run_rr14.log")
    t0 = time.time()
    frozen_before = rc.frozen_hashes()
    extra_before = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    log(f"frozen: {len(frozen_before)} released v9 + {len(extra_before)} RR10 files")

    log("\n== data ==")
    FEATURES, labels, feats, valid, aparc, econ = load_core(log)
    extra = load_extra_maps(log)
    labels_c = cat(labels)
    valid_c = cat(valid)
    aparc_c = cat(aparc)
    econ_c = cat(econ)
    fvs = {k: cat(feats[k]) for k in AXIS}

    log("\n== Step 1: comparators ==")
    comps = {}
    comps["myelin septile"] = septile(cat(feats["myelin"]), valid_c)
    comps["gradient septile"] = septile(cat(feats["gradient"]), valid_c)
    if "histg1" in extra:
        comps["BigBrain Hist-G1 septile"] = septile(cat(extra["histg1"]), valid_c)
    else:
        log("  BigBrain Hist-G1 not available; that comparator is SKIPPED, not substituted")
    for nm, lv in comps.items():
        sizes = [int((lv == t).sum()) for t in range(1, 8)]
        log(f"  {nm}: levels 1-7, sizes {sizes}")

    log("\n== rotations ==")
    rotidx = rotation_index(valid_c, log, args.n_spin)
    rotval = validate_rotation(rotidx, labels_c, valid_c, log)

    log("\n== Step 2: the panel ==")
    panel = []
    parts_for_panel = {"cyto7 (published map)": labels_c, **comps}
    for f in FEATURES:
        vals = cat(feats[f.key])
        for pname, lv in parts_for_panel.items():
            rho, p = spin_p(vals, lv, valid_c, rotidx, args.n_spin)
            panel.append({"feature": f.label, "feature_key": f.key,
                          "partition": pname, "level": "vertex",
                          "spearman_rho": rho, "spin_p": p})
        log(f"    {f.label:32s} " + "  ".join(
            f"{p['partition'].split()[0]}={p['spearman_rho']:+.3f}"
            for p in panel[-len(parts_for_panel):]))
    for key, mp in extra.items():
        vals = cat(mp)
        ok = valid_c & np.isfinite(vals)
        for pname, lv in parts_for_panel.items():
            rho, p = spin_p(vals, lv, ok, rotidx, args.n_spin)
            panel.append({"feature": key, "feature_key": key, "partition": pname,
                          "level": "vertex", "spearman_rho": rho, "spin_p": p})
    pdf = pd.DataFrame(panel)
    for pname, sub in pdf.groupby("partition"):
        pdf.loc[sub.index, "q_fdr_within_partition"] = bh(sub["spin_p"].values)

    log("\n== Step 2b: parcel level (RORB and the layer markers) ==")
    par = parcel_panel(parts_for_panel, aparc_c, valid_c, log)

    log("\n== Step 3: win fraction against von Economo ==")
    # exactly the published support: isocortex where both maps carry a type 2..7
    common = valid_c & np.isin(labels_c, ISO) & np.isin(econ_c, ISO)
    disagree = common & (labels_c != econ_c)
    log(f"  common isocortex support n={int(common.sum())}, "
        f"cyto7-vs-vonEconomo disagreement set n={int(disagree.sum())}")
    log("  the disagreement set is held FIXED at the cyto7-vs-von-Economo vertices, so")
    log("  every challenger is scored on the identical 27,298-vertex set and the")
    log("  numbers stay comparable with the published 60.1%")
    wf_rows = []
    for pname, lv in parts_for_panel.items():
        wf, per = win_fraction(fvs, lv, econ_c, common, disagree)
        wf_rows.append({"challenger": pname, "win_fraction": wf,
                        "n_disagreement_vertices": int(disagree.sum()),
                        **{f"win_{k}": v for k, v in per.items()}})
        log(f"    {pname:28s} win fraction {wf:.4f}  "
            + "  ".join(f"{k}={v:.3f}" for k, v in per.items()))
    wdf = pd.DataFrame(wf_rows)

    log("\n== Step 3b: the granularity-matched null ==")
    A = mesh_adjacency()
    n_hemi = [labels["L"].size, labels["R"].size]
    parts = build_matched_nulls(A, valid_c, np.where(valid_c, labels_c, 0),
                                n_hemi, args.n_part, log)
    qual = partition_quality(parts, A, valid_c, np.where(valid_c, labels_c, 0), log)
    null_wf = []
    for i in range(parts.shape[0]):
        wf, _ = win_fraction(fvs, parts[i].astype(int), econ_c, common, disagree)
        null_wf.append(wf)
        if (i + 1) % 50 == 0:
            log(f"    matched partition {i + 1}/{parts.shape[0]}")
    null_wf = np.asarray(null_wf)
    obs = float(wdf.loc[wdf.challenger == "cyto7 (published map)", "win_fraction"].iloc[0])
    pct = float((null_wf < obs).mean() * 100)
    p_matched = float((np.sum(null_wf >= obs) + 1) / (null_wf.size + 1))
    log(f"  matched-null win fractions: mean {null_wf.mean():.4f}, sd {null_wf.std():.4f}, "
        f"range {null_wf.min():.4f} to {null_wf.max():.4f}")
    log(f"  cyto7 observed {obs:.4f} -> percentile {pct:.1f}, p = {p_matched:.4f}")
    log(f"  the published scrambled-cyto7 chance level was 0.46")

    log("\n== Step 4: M1 sensitivity ==")
    m1df, m1piv = m1_sensitivity(feats, labels_c, valid_c, aparc_c, rotidx, extra, log)

    log("\n== Step 5: BigBrain arbiter (retrieval) ==")
    bb = bigbrain_arbiter(log)

    # ---- write ---- #
    pdf.to_csv(OUTDIR / "comparator_panel.csv", index=False)
    if len(par):
        par.to_csv(OUTDIR / "comparator_panel_parcel.csv", index=False)
    wdf.to_csv(OUTDIR / "winfraction_comparators.csv", index=False)
    pd.DataFrame({"partition": np.arange(null_wf.size),
                  "win_fraction": null_wf}).to_csv(
        OUTDIR / "matched_null_distribution.csv", index=False)
    qual.to_csv(OUTDIR / "matched_null_quality.csv", index=False)
    if len(m1df):
        m1df.to_csv(OUTDIR / "m1_sensitivity.csv", index=False)
    if len(bb):
        bb.to_csv(OUTDIR / "bigbrain_arbiter.csv", index=False)

    ok, detail = rc.check_frozen(frozen_before)
    extra_after = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    ok_extra = extra_before == extra_after
    log(f"\nreleased v9 unchanged: {ok}; RR10 crossed files unchanged: {ok_extra}")

    (OUTDIR / "rr14_summary.json").write_text(json.dumps({
        "comparators_built": sorted(comps),
        "rotation_validation": rotval,
        "n_disagreement": int(disagree.sum()),
        "win_fractions": wdf.to_dict("records"),
        "matched_null": {"n": int(null_wf.size), "mean": float(null_wf.mean()),
                         "sd": float(null_wf.std()), "min": float(null_wf.min()),
                         "max": float(null_wf.max()),
                         "cyto7_observed": obs, "cyto7_percentile": pct,
                         "p_vs_matched_null": p_matched,
                         "published_scrambled_chance": 0.46},
        "frozen_v9_unchanged": ok, "frozen_rr10_unchanged": ok_extra,
        "runtime_seconds": round(time.time() - t0, 1)}, indent=2, default=float),
        encoding="utf-8")
    log(f"total runtime {(time.time() - t0) / 60:.1f} min")
    fh.close()
    return pdf, wdf, null_wf


if __name__ == "__main__":
    main()
