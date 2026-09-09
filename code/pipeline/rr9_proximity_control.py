"""RR9 - control the §3.7 tractography falloff for spatial opportunity, properly.

RR7 demoted the type-distance falloff on three grounds, one of which does not
survive inspection: the geodesic control was a centroid-to-centroid distance
between type footprints, and a cyto7 type is annular or multi-focal, so its
centroid lands in its own hole. That measure calls agranular and dysgranular
54 mm apart although they share 4,728 mm2 of border.

This task replaces it with measures that are valid for annular regions, and,
more importantly, with the right offset. The confound is not distance in the
abstract, it is *opportunity*: how many pairs of cortical locations in types i
and j are close enough that a short-range streamline could join them at all.
The published model offsets by log(n_i x n_j), every possible vertex pair, which
over-counts massively for distant type pairs. R1 replaces that with the number
of vertex pairs actually within 80 mm, the cutoff §3.7 already uses.

  R0  offset log(n_i n_j)          no distance covariate   (RR7's, for comparison)
  R1  offset log(pairs <= 80 mm)   no distance covariate   (the headline)
  R2  offset log(pairs <= 80 mm)   contact area
  R3  offset log(pairs <= 80 mm)   contact area + nearest-neighbour median distance

Every model is fitted at both the aggregate (21 off-diagonal type-pair) and
bundle (924-row, 33-tract-clustered) levels with the negative-binomial and
Poisson machinery from RR7 unchanged, and every model gets the
topology-preserving spin null with the scale-free statistic that RR7 decision 1
requires: the coefficient on a standardised type-distance, never a raw slope.

Run::
    conda run -n cyto7 python scripts/rr9_proximity_control.py --n-spin 1000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import rr_common as rc
import rr7_tracto_controls as rr7
from cyto7_surface_io import resolve_target_map

OUTDIR = rc.OUT / "rr9_proximity"
CACHEDIR = rc.OUT / "_cache"

CYTO_ORDER = rr7.CYTO_ORDER
TIDX = rr7.TIDX
RADII_MM = (40.0, 80.0, 120.0)
PRIMARY_RADIUS = 80.0            # matches §3.7, fixed, never tuned

# The six type-distance-1 pairs, in the order the spec's table lists them.
SPEC_TABLE_PAIRS = [("Eulaminate-III", "Koniocortex"),
                    ("Eulaminate-I", "Eulaminate-II"),
                    ("Dysgranular", "Eulaminate-I"),
                    ("Allocortex", "Agranular"),
                    ("Eulaminate-II", "Eulaminate-III"),
                    ("Agranular", "Dysgranular")]


def _log_factory(path):
    fh = open(path, "w", encoding="utf-8")

    def log(*args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()
    return log, fh


# --------------------------------------------------------------------------- #
# Step 1 - document the existing measure's failure
# --------------------------------------------------------------------------- #


def step1_old_measure_failure(contact: pd.DataFrame, geo_cen: pd.DataFrame,
                              pm: dict | None = None, log=print) -> dict:
    """Reproduce the spec's table and correlate centroid distance with contact area.

    Restricted to the six type-distance-1 pairs, which are the pairs that share a
    border at all, so a distance measure that is valid for these regions has to
    order them roughly by how much border they share.
    """
    rows = []
    for a, b in SPEC_TABLE_PAIRS:
        r = {"type_a": a, "type_b": b,
             "type_distance": abs(TIDX[a] - TIDX[b]),
             "shared_border_mm2": float(contact.loc[a, b]),
             "centroid_geodesic_mm": float(geo_cen.loc[a, b])}
        if pm is not None:
            i, j = TIDX[a] - 1, TIDX[b] - 1
            r["nn_eucl_median_mm"] = float(pm["nn_eucl_median"][i, j])
            r["nn_geo_median_mm"] = float(pm["nn_geo_median"][i, j])
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("centroid_geodesic_mm").reset_index(drop=True)
    log("== Step 1: the centroid measure on the six adjacent type pairs ==")
    log(t.to_string(index=False))
    out = {"table": t}
    # A distance measure that is valid for these regions cannot rank a pair sharing
    # a large border as one of the most distant, so how each measure relates to
    # shared border area over the six adjacent pairs is the direct diagnostic.
    for col, tag in (("centroid_geodesic_mm", "centroid"),
                     ("nn_eucl_median_mm", "nn_eucl"), ("nn_geo_median_mm", "nn_geo")):
        if col not in t.columns:
            continue
        rp = stats.pearsonr(t["shared_border_mm2"], t[col])
        rs = stats.spearmanr(t["shared_border_mm2"], t[col])
        out[f"contact_vs_{tag}_pearson_r"] = float(rp[0])
        out[f"contact_vs_{tag}_pearson_p"] = float(rp[1])
        out[f"contact_vs_{tag}_spearman_rho"] = float(rs[0])
        out[f"contact_vs_{tag}_spearman_p"] = float(rs[1])
        log(f"  contact area vs {col:22s} n = 6 adjacent pairs: "
            f"pearson r = {rp[0]:+.3f} (p {rp[1]:.3f}), spearman rho = {rs[0]:+.3f}")
    if pm is not None:
        rng = t["centroid_geodesic_mm"]
        rng2 = t["nn_eucl_median_mm"]
        log(f"  spread over the six adjacent pairs: centroid {rng.min():.1f} to {rng.max():.1f} mm "
            f"(x{rng.max() / max(rng.min(), 1e-9):.1f}), nearest-neighbour median "
            f"{rng2.min():.1f} to {rng2.max():.1f} mm (x{rng2.max() / rng2.min():.1f})")
        out["centroid_range_mm"] = [float(rng.min()), float(rng.max())]
        out["nn_eucl_range_mm"] = [float(rng2.min()), float(rng2.max())]
    return out


# --------------------------------------------------------------------------- #
# Step 2 - proximity measures valid for annular and multi-focal regions
# --------------------------------------------------------------------------- #


def _weighted_quantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    """Area-weighted quantile of *v* with weights *w*."""
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    v, w = v[ok], w[ok]
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(np.interp(q * cw[-1], cw, v))


def proximity_measures(log=print) -> dict:
    """Nearest-neighbour distances and opportunity counts per unordered type pair.

    On the fsaverage white surface, both hemispheres, from the released v9 annot.

    Euclidean quantities are computed on the bilateral point cloud, so a vertex
    may find its nearest partner in the other hemisphere and a commissural
    endpoint pair counts as an opportunity. Geodesic quantities are within
    hemisphere by construction, since no mesh path crosses the midline; they are
    pooled across hemispheres by area weight.

    The nearest-neighbour summary is symmetric: the distribution is the union of
    {d(u -> j) : u in i} and {d(v -> i) : v in j}, each vertex carrying its own
    surface area as weight, so neither direction of an asymmetric pair dominates.
    """
    cpath = CACHEDIR / "rr9_proximity_measures.npz"
    if cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        log(f"  reusing cached proximity measures from {cpath.name}")
        return {k: z[k] for k in z.files}

    lab = resolve_target_map("v9", "fsaverage")
    geom = rc.fsaverage_geometry("white", "164k")

    coords_h, labels_h, va_h = {}, {}, {}
    for H in ("L", "R"):
        c, f = geom[H]
        coords_h[H] = c
        labels_h[H] = np.asarray(lab[H], int)
        va_h[H] = rc.vertex_areas(c, f)
    coords_all = np.vstack([coords_h["L"], coords_h["R"]])
    labels_all = np.concatenate([labels_h["L"], labels_h["R"]])
    va_all = np.concatenate([va_h["L"], va_h["R"]])
    log(f"  fsaverage white, bilateral: {coords_all.shape[0]} vertices, "
        f"{int((labels_all > 0).sum())} labelled")

    # ---- Euclidean nearest-neighbour distances, bilateral ------------------ #
    trees = {c: cKDTree(coords_all[labels_all == c]) for c in range(1, 8)}
    idx = {c: np.where(labels_all == c)[0] for c in range(1, 8)}
    nn_e = np.full((7, 7), np.nan)          # directional i -> j, area-weighted median
    d_eucl = {}
    log("  Euclidean nearest-neighbour distances ...")
    for i in range(1, 8):
        for j in range(1, 8):
            d, _ = trees[j].query(coords_all[idx[i]], k=1)
            d_eucl[(i, j)] = d
            nn_e[i - 1, j - 1] = _weighted_quantile(d, va_all[idx[i]], 0.5)

    # ---- Geodesic nearest-neighbour distances, per hemisphere -------------- #
    log("  geodesic nearest-neighbour distances (multi-source Dijkstra, 7 passes x 2 hemispheres) ...")
    d_geo = {(i, j): [] for i in range(1, 8) for j in range(1, 8)}
    w_geo = {(i, j): [] for i in range(1, 8) for j in range(1, 8)}
    for H in ("L", "R"):
        c, f = geom[H]
        g = rc.geodesic_graph(c, f)
        l = labels_h[H]
        for j in range(1, 8):
            jdx = np.where(l == j)[0]
            if jdx.size == 0:
                continue
            t0 = time.time()
            dm = dijkstra(g, directed=False, indices=jdx, min_only=True)
            log(f"    {H} target {CYTO_ORDER[j - 1]}: {time.time() - t0:.1f}s")
            for i in range(1, 8):
                iidx = np.where(l == i)[0]
                if iidx.size == 0:
                    continue
                v = dm[iidx]
                d_geo[(i, j)].append(v)
                w_geo[(i, j)].append(va_h[H][iidx])
    nn_g = np.full((7, 7), np.nan)
    for i in range(1, 8):
        for j in range(1, 8):
            if not d_geo[(i, j)]:
                continue
            v = np.concatenate(d_geo[(i, j)])
            w = np.concatenate(w_geo[(i, j)])
            ok = np.isfinite(v)
            nn_g[i - 1, j - 1] = _weighted_quantile(v[ok], w[ok], 0.5)

    # ---- symmetric pair summaries ----------------------------------------- #
    nn_e_med = np.full((7, 7), np.nan); nn_e_p10 = np.full((7, 7), np.nan)
    nn_g_med = np.full((7, 7), np.nan); nn_g_p10 = np.full((7, 7), np.nan)
    for i in range(1, 8):
        for j in range(i, 8):
            ve = np.concatenate([d_eucl[(i, j)], d_eucl[(j, i)]])
            we = np.concatenate([va_all[idx[i]], va_all[idx[j]]])
            nn_e_med[i - 1, j - 1] = nn_e_med[j - 1, i - 1] = _weighted_quantile(ve, we, 0.5)
            nn_e_p10[i - 1, j - 1] = nn_e_p10[j - 1, i - 1] = _weighted_quantile(ve, we, 0.10)
            if d_geo[(i, j)] and d_geo[(j, i)]:
                vg = np.concatenate([np.concatenate(d_geo[(i, j)]), np.concatenate(d_geo[(j, i)])])
                wg = np.concatenate([np.concatenate(w_geo[(i, j)]), np.concatenate(w_geo[(j, i)])])
                ok = np.isfinite(vg)
                nn_g_med[i - 1, j - 1] = nn_g_med[j - 1, i - 1] = _weighted_quantile(vg[ok], wg[ok], 0.5)
                nn_g_p10[i - 1, j - 1] = nn_g_p10[j - 1, i - 1] = _weighted_quantile(vg[ok], wg[ok], 0.10)

    # ---- opportunity counts ------------------------------------------------ #
    # cKDTree.count_neighbors is exact and needs no sampling: it counts the
    # ordered pairs (u in i, v in j) within each radius by dual-tree descent.
    log(f"  opportunity counts at {RADII_MM} mm ...")
    opp = np.full((7, 7, len(RADII_MM)), np.nan)
    opp_within = np.full((7, 7, len(RADII_MM)), np.nan)
    trees_h = {(H, c): cKDTree(coords_h[H][labels_h[H] == c])
               for H in ("L", "R") for c in range(1, 8) if (labels_h[H] == c).any()}
    for i in range(1, 8):
        for j in range(i, 8):
            t0 = time.time()
            n = np.asarray(trees[i].count_neighbors(trees[j], list(RADII_MM)), float)
            opp[i - 1, j - 1, :] = opp[j - 1, i - 1, :] = n
            w = np.zeros(len(RADII_MM))
            for H in ("L", "R"):
                if (H, i) in trees_h and (H, j) in trees_h:
                    w += np.asarray(trees_h[(H, i)].count_neighbors(
                        trees_h[(H, j)], list(RADII_MM)), float)
            opp_within[i - 1, j - 1, :] = opp_within[j - 1, i - 1, :] = w
            log(f"    {CYTO_ORDER[i - 1]:14s} x {CYTO_ORDER[j - 1]:14s} "
                f"{['%.3g' % v for v in n]} ({time.time() - t0:.1f}s)")

    cnt = np.array([int((labels_all == c).sum()) for c in range(1, 8)], float)
    out = {"nn_eucl_directional_median": nn_e, "nn_geo_directional_median": nn_g,
           "nn_eucl_median": nn_e_med, "nn_eucl_p10": nn_e_p10,
           "nn_geo_median": nn_g_med, "nn_geo_p10": nn_g_p10,
           "opportunity": opp, "opportunity_within_hemi": opp_within,
           "radii": np.asarray(RADII_MM, float), "n_vertices": cnt}
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    np.savez(cpath, **out)
    return out


def proximity_table(pm: dict, contact: pd.DataFrame, geo_cen: pd.DataFrame,
                    geo_foot: pd.DataFrame, off_diagonal_only: bool = True) -> pd.DataFrame:
    """One row per unordered type pair with every measure side by side."""
    ri = list(RADII_MM).index(PRIMARY_RADIUS)
    rows = []
    for a in range(1, 8):
        for b in range(a + 1 if off_diagonal_only else a, 8):
            sa, sb = CYTO_ORDER[a - 1], CYTO_ORDER[b - 1]
            row = {"type_a": sa, "type_b": sb, "type_distance": b - a,
                   "contact_area_mm2": float(contact.loc[sa, sb]) if sa in contact.index else np.nan,
                   "centroid_geodesic_mm_SUPERSEDED": float(geo_cen.loc[sa, sb]),
                   "footprint_geodesic_mm": float(geo_foot.loc[sa, sb]),
                   "nn_eucl_median_mm": float(pm["nn_eucl_median"][a - 1, b - 1]),
                   "nn_eucl_p10_mm": float(pm["nn_eucl_p10"][a - 1, b - 1]),
                   "nn_geo_median_mm": float(pm["nn_geo_median"][a - 1, b - 1]),
                   "nn_geo_p10_mm": float(pm["nn_geo_p10"][a - 1, b - 1]),
                   "n_vertices_a": int(pm["n_vertices"][a - 1]),
                   "n_vertices_b": int(pm["n_vertices"][b - 1]),
                   "possible_endpoint_pairs": float(pm["n_vertices"][a - 1] * pm["n_vertices"][b - 1])}
            for k, r in enumerate(RADII_MM):
                row[f"pairs_within_{int(r)}mm"] = float(pm["opportunity"][a - 1, b - 1, k])
                row[f"pairs_within_{int(r)}mm_within_hemi"] = float(
                    pm["opportunity_within_hemi"][a - 1, b - 1, k])
            row["opportunity_fraction_80mm"] = (row[f"pairs_within_{int(PRIMARY_RADIUS)}mm"]
                                                / row["possible_endpoint_pairs"])
            rows.append(row)
    del ri
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 3 - collinearity
# --------------------------------------------------------------------------- #


MEASURES = [("centroid_geodesic_mm_SUPERSEDED", "centroid-to-centroid geodesic (RR7, superseded)"),
            ("footprint_geodesic_mm", "footprint-mean geodesic (RR7, secondary)"),
            ("nn_eucl_median_mm", "nearest-neighbour median, Euclidean"),
            ("nn_eucl_p10_mm", "nearest-neighbour 10th percentile, Euclidean"),
            ("nn_geo_median_mm", "nearest-neighbour median, geodesic"),
            ("nn_geo_p10_mm", "nearest-neighbour 10th percentile, geodesic"),
            ("log_pairs_within_40mm", "log opportunity, 40 mm"),
            ("log_pairs_within_80mm", "log opportunity, 80 mm (primary)"),
            ("log_pairs_within_120mm", "log opportunity, 120 mm"),
            ("log_opportunity_fraction_80mm", "log opportunity fraction, 80 mm")]


def step3_collinearity(tab: pd.DataFrame, log=print) -> pd.DataFrame:
    t = tab.copy()
    for r in RADII_MM:
        t[f"log_pairs_within_{int(r)}mm"] = np.log(np.maximum(t[f"pairs_within_{int(r)}mm"], 1.0))
    t["log_opportunity_fraction_80mm"] = np.log(np.maximum(t["opportunity_fraction_80mm"], 1e-12))
    rows = []
    log("== Step 3: collinearity of type-distance with each distance measure "
        f"(n = {len(t)} off-diagonal pairs) ==")
    for col, desc in MEASURES:
        v = t[col].to_numpy(float)
        ok = np.isfinite(v)
        rp = stats.pearsonr(t["type_distance"].to_numpy(float)[ok], v[ok])
        rs = stats.spearmanr(t["type_distance"].to_numpy(float)[ok], v[ok])
        rows.append({"measure": col, "description": desc, "n_pairs": int(ok.sum()),
                     "pearson_r": float(rp[0]), "pearson_p": float(rp[1]),
                     "spearman_rho": float(rs[0]), "spearman_p": float(rs[1]),
                     "abs_pearson_r": abs(float(rp[0]))})
        log(f"  {desc:52s} pearson r = {rp[0]:+.3f} (p {rp[1]:.4f}), "
            f"spearman rho = {rs[0]:+.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 4 - the RR7 model set with the corrected offset
# --------------------------------------------------------------------------- #


def attach_proximity(d: pd.DataFrame, pm: dict) -> pd.DataFrame:
    """Add opportunity counts and nearest-neighbour distances to a design frame."""
    d = d.copy()
    a = d["source_ord"].to_numpy(int) - 1
    b = d["target_ord"].to_numpy(int) - 1
    for k, r in enumerate(RADII_MM):
        n = pm["opportunity"][a, b, k].astype(float)
        d[f"pairs_within_{int(r)}mm"] = n
        d[f"log_pairs_within_{int(r)}mm"] = np.log(np.maximum(n, 1.0))
    d["nn_eucl_median_mm"] = pm["nn_eucl_median"][a, b]
    d["nn_eucl_p10_mm"] = pm["nn_eucl_p10"][a, b]
    d["nn_geo_median_mm"] = pm["nn_geo_median"][a, b]
    d["log_offset_endpoint_pairs"] = d["log_offset"].to_numpy(float)
    return d


MODELS = [
    ("R0", "log(n_i n_j), RR7's offset", "log_offset_endpoint_pairs", [], "none"),
    ("R1", "log(pairs <= 80 mm)", "log_pairs_within_80mm", [], "none"),
    ("R2", "log(pairs <= 80 mm)", "log_pairs_within_80mm", ["log_contact_area"], "contact area"),
    ("R3", "log(pairs <= 80 mm)", "log_pairs_within_80mm",
     ["log_contact_area", "nn_eucl_median_mm"], "contact area + NN median (Euclidean)"),
]

SENSITIVITY = [
    ("R1-40", "log(pairs <= 40 mm)", "log_pairs_within_40mm", [], "none"),
    ("R1-120", "log(pairs <= 120 mm)", "log_pairs_within_120mm", [], "none"),
    ("R3-geo", "log(pairs <= 80 mm)", "log_pairs_within_80mm",
     ["log_contact_area", "nn_geo_median_mm"], "contact area + NN median (geodesic)"),
]


def _vif(d: pd.DataFrame, covs: list[str]) -> float:
    """Variance-inflation factor of type-distance against the model's covariates.

    With a valid distance measure the covariate is close to collinear with
    type-distance, so this number is what says whether R3 can separate the two at
    all, rather than the coefficient's standard error alone.
    """
    if not covs:
        return 1.0
    y = d["type_distance"].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [
        np.where(np.isfinite(d[c].to_numpy(float)), d[c].to_numpy(float),
                 np.nanmean(d[c].to_numpy(float))) for c in covs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_res = float(((y - X @ beta) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(1.0 / max(1e-12, 1.0 - r2))


def fit_model(d: pd.DataFrame, offset_col: str, covs: list[str], rot: np.ndarray,
              want_null: bool, log=print) -> dict:
    """NB (tract-clustered) + Poisson (cluster-robust) + topology-preserving spin null.

    The offset is swapped into ``log_offset`` so RR7's fitting code is reused
    byte for byte. The null refits with surrogate type-distances induced by the
    released rotations, always on the standardised predictor, per RR7 decision 1:
    a rotation compresses the surrogate type-distances toward their mean, so a
    raw-slope comparison would inflate the null by about 1/compression. The
    offset, contact areas and proximity measures are held at their observed
    values under the null, because the rotation reassigns each type's ordinal
    position while leaving its footprint, and therefore its spatial opportunity,
    exactly where it is. That is precisely the question: does the ordering of the
    types explain connectivity beyond the geometry of where they sit.
    """
    dd = d.copy()
    dd["log_offset"] = dd[offset_col].to_numpy(float)
    nb = rr7._fit_nb(dd, covs, True)
    po = rr7._fit_poisson_cluster(dd, covs, True)
    out = dict(nb); out.update(po)
    out["coef_per_SD"] = rr7._fit_nb(dd, covs, True, td_scale="z")["coef"]
    out["vif_type_distance"] = _vif(dd, covs)
    if not want_null:
        out.update({"p_spin": np.nan, "spin_null_mean_per_SD": np.nan, "n_spin_fits": 0})
        return out, None
    t0 = time.time()
    sp, spmean = rr7.spin_null_coef(dd, covs, True, rot, log=lambda *_a: None)
    good = sp[np.isfinite(sp)]
    p_spin = float((np.sum(np.abs(good) >= abs(out["coef_per_SD"])) + 1) / (good.size + 1)) \
        if good.size else np.nan
    out.update({"p_spin": p_spin, "spin_null_mean_per_SD": float(spmean),
                "spin_null_sd_per_SD": float(np.nanstd(good)) if good.size else np.nan,
                "n_spin_fits": int(good.size), "null_seconds": round(time.time() - t0, 1)})
    return out, sp


def run_models(designs: dict, pm: dict, rot: np.ndarray, n_spin: int, log=print):
    rows, nulls = [], {}
    specs = MODELS + SENSITIVITY
    for level, d in designs.items():
        for tag, off_desc, off_col, covs, cov_desc in specs:
            want_null = True
            f, sp = fit_model(d, off_col, covs, rot, want_null, log=log)
            f.update({"level": level, "model": tag, "offset": off_desc,
                      "distance_covariate": cov_desc,
                      "primary": tag in {m[0] for m in MODELS}})
            rows.append(f)
            if sp is not None:
                nulls[f"{level}_{tag}"] = sp
            log(f"  [{level:9s}] {tag:7s} offset {off_desc:24s} cov {cov_desc:36s} "
                f"NB beta = {f['coef']:+.4f} (se {f['se']:.4f}, p {f['p_model']:.4g}) | "
                f"per SD {f['coef_per_SD']:+.4f} spin p = {f['p_spin']:.4f} "
                f"(null mean {f['spin_null_mean_per_SD']:+.4f}, n {f['n_spin_fits']}) | "
                f"Pois beta = {f['coef_pois']:+.4f} (p {f['p_pois']:.4g})")
    del n_spin
    return pd.DataFrame(rows), nulls


# --------------------------------------------------------------------------- #
# Step 4b - the published statistic itself, opportunity-normalised
# --------------------------------------------------------------------------- #


def published_statistic_with_offset(agg: pd.DataFrame, rot: np.ndarray, log=print) -> dict:
    """The §3.7 statistic, with and without division by spatial opportunity.

    §3.7 does not report a negative-binomial coefficient. It reports the
    log-linear relation between type-distance and short-range streamline count
    over the 21 off-diagonal type pairs, and RR7 tested exactly that against the
    topology-preserving null (Pearson r = -0.765, spin p = 0.040). The
    opportunity control has to be applied to that statistic too, or the answer
    would be about a different quantity from the one the section makes a claim
    about.

    The normalised outcome is log((count + 1) / pairs within 80 mm), a
    connection rate per available endpoint pair rather than a raw count. The null
    statistic is the Pearson correlation, which is scale-free, per RR7 decision 1.
    """
    x = agg["type_distance"].to_numpy(float)
    cnt = agg["streamline_count"].to_numpy(float)
    opp = np.maximum(agg["pairs_within_80mm"].to_numpy(float), 1.0)
    so = agg["source_ord"].to_numpy(int) - 1
    to = agg["target_ord"].to_numpy(int) - 1

    outcomes = {
        "raw_count_published": np.log1p(cnt),
        "rate_per_opportunity_80mm": np.log((cnt + 1.0) / opp),
        "rate_per_possible_pair": np.log((cnt + 1.0)
                                         / agg["possible_endpoint_pairs"].to_numpy(float)),
    }
    out = {}
    for name, y in outcomes.items():
        r_obs = float(stats.pearsonr(x, y)[0])
        slope_obs = float(np.polyfit(x, y, 1)[0])
        null_r = np.full(rot.shape[0], np.nan)
        for k in range(rot.shape[0]):
            rk = rot[k]
            if not np.all(np.isfinite(rk)):
                continue
            xr = np.abs(rk[so] - rk[to])
            if xr.std() == 0:
                continue
            null_r[k] = stats.pearsonr(xr, y)[0]
        g = null_r[np.isfinite(null_r)]
        p = float((np.sum(np.abs(g) >= abs(r_obs)) + 1) / (g.size + 1)) if g.size else np.nan
        out[name] = {"pearson_r": round(r_obs, 4),
                     "slope_per_ordinal_step": round(slope_obs, 4),
                     "p_spin_pearson_r": round(p, 4),
                     "spin_null_mean_r": round(float(g.mean()), 4) if g.size else np.nan,
                     "n_spin": int(g.size)}
        log(f"  {name:28s} r = {r_obs:+.4f}, slope = {slope_obs:+.4f} per ordinal step, "
            f"spin p = {p:.4f} (null mean r {out[name]['spin_null_mean_r']:+.4f})")
    return out


# --------------------------------------------------------------------------- #
# Step 5 - verdict
# --------------------------------------------------------------------------- #


def verdict(mc: pd.DataFrame, alpha: float = 0.05) -> dict:
    def get(level, model):
        r = mc[(mc.level == level) & (mc.model == model)]
        return r.iloc[0] if len(r) else None

    agg = {m: get("aggregate", m) for m in ("R1", "R2", "R3")}
    agg_ok = {m: bool(np.isfinite(r["p_spin"]) and r["p_spin"] < alpha and r["coef"] < 0)
              for m, r in agg.items() if r is not None}
    survives_aggregate = all(agg_ok.get(m, False) for m in ("R1", "R2", "R3"))
    r1 = agg["R1"]
    r1_dies = not agg_ok.get("R1", False)

    b = {m: get("bundle", m) for m in ("R1", "R2", "R3")}
    bundle_nonzero = any(r is not None and np.isfinite(r["p_spin"])
                         and r["p_spin"] < alpha and r["p_model"] < alpha and r["coef"] < 0
                         for r in b.values())

    if r1_dies:
        n = 3
    elif survives_aggregate and bundle_nonzero:
        n = 1
    else:
        n = 2
    return {"outcome": n,
            "aggregate_R1_coef": float(r1["coef"]), "aggregate_R1_p_spin": float(r1["p_spin"]),
            "aggregate_R1_p_model": float(r1["p_model"]),
            "aggregate_survives_R1_R3": bool(survives_aggregate),
            "bundle_coefficient_nonzero": bool(bundle_nonzero),
            "per_model_aggregate_survives": agg_ok, "alpha": alpha}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=1000)
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log, fh = _log_factory(OUTDIR / "run_rr9.log")
    frozen_before = rc.frozen_hashes()
    t_start = time.time()

    from plot_tractography_analysis import boundary_surface_matrix, load_geometry, stratified_matrices
    geometry = load_geometry(rr7.GEOM_CSV)
    short, _long, _all = stratified_matrices(geometry, PRIMARY_RADIUS)
    contact = boundary_surface_matrix(rr7.LABELS_NII)
    geo_cen, geo_foot = rr7.geodesic_type_distances(log=log)

    log("== Step 2: proximity measures on the fsaverage white surface ==")
    pm = proximity_measures(log=log)

    s1 = step1_old_measure_failure(contact, geo_cen, pm=pm, log=log)
    s1["table"].to_csv(OUTDIR / "step1_old_measure_failure.csv", index=False)

    tab = proximity_table(pm, contact, geo_cen, geo_foot, off_diagonal_only=True)
    tab.to_csv(OUTDIR / "proximity_measures.csv", index=False)
    log(tab[["type_a", "type_b", "type_distance", "contact_area_mm2",
             "centroid_geodesic_mm_SUPERSEDED", "nn_eucl_median_mm", "nn_eucl_p10_mm",
             "nn_geo_median_mm", "pairs_within_80mm", "opportunity_fraction_80mm"]]
        .to_string(index=False, float_format=lambda v: f"{v:,.4g}"))

    coll = step3_collinearity(tab, log=log)
    coll.to_csv(OUTDIR / "collinearity.csv", index=False)

    log("== Step 4: model set with the corrected offset ==")
    agg = rr7.aggregate_design(short, contact, geo_cen, geo_foot)
    d_obs = rr7.build_design(contact, geo_cen, geo_foot)
    bundle = rr7.complete_zeros(d_obs, contact, geo_cen, geo_foot)
    agg = attach_proximity(agg, pm)
    bundle = attach_proximity(bundle, pm)
    agg.to_csv(OUTDIR / "aggregate_design_rr9.csv", index=False)
    bundle.to_csv(OUTDIR / "bundle_design_rr9.csv", index=False)
    log(f"  aggregate: {len(agg)} off-diagonal type pairs; "
        f"bundle: {len(bundle)} rows, {bundle.Tract_Name.nunique()} tracts")
    zero_opp = int((agg["pairs_within_80mm"] == 0).sum())
    log(f"  type pairs with zero opportunity at 80 mm: {zero_opp}")

    rot = rr7.rotated_type_ranks(args.n_spin, log=log)
    mc, nulls = run_models({"aggregate": agg, "bundle": bundle}, pm, rot, args.n_spin, log=log)
    cols = ["level", "model", "primary", "offset", "distance_covariate", "n", "coef", "se", "z",
            "p_model", "coef_per_SD", "p_spin", "spin_null_mean_per_SD", "spin_null_sd_per_SD",
            "n_spin_fits", "vif_type_distance", "coef_pois", "se_pois", "p_pois", "alpha",
            "converged"]
    mc = mc[[c for c in cols if c in mc.columns]]
    mc.to_csv(OUTDIR / "model_coefficients_rr9.csv", index=False)
    for k, v in nulls.items():
        np.save(OUTDIR / f"_nulldist_spin_{k}.npy", v)

    log("== Step 4b: the published §3.7 statistic, opportunity-normalised ==")
    pubstat = published_statistic_with_offset(agg, rot, log=log)
    pd.DataFrame(pubstat).T.rename_axis("outcome").reset_index().to_csv(
        OUTDIR / "published_statistic_with_offset.csv", index=False)

    v = verdict(mc)
    log("== Step 5: verdict ==")
    log(json.dumps(v, indent=2, default=float))

    ok, detail = rc.check_frozen(frozen_before)
    (OUTDIR / "rr9_summary.json").write_text(json.dumps({
        "step1_old_measure_failure": {k: val for k, val in s1.items() if k != "table"},
        "step1_table": s1["table"].to_dict("records"),
        "collinearity": coll.to_dict("records"),
        "published_statistic_with_offset": pubstat,
        "verdict": v,
        "primary_radius_mm": PRIMARY_RADIUS, "sensitivity_radii_mm": list(RADII_MM),
        "n_spin": args.n_spin,
        "n_aggregate_pairs": int(len(agg)), "n_bundle_rows": int(len(bundle)),
        "zero_opportunity_pairs_80mm": zero_opp,
        "runtime_seconds": round(time.time() - t_start, 1),
        "frozen_unchanged": ok, "frozen": detail}, indent=2, default=float), encoding="utf-8")
    log(f"frozen files unchanged: {ok}")
    log(f"total runtime: {(time.time() - t_start) / 60:.1f} min")
    fh.close()
    return mc, v


if __name__ == "__main__":
    main()
