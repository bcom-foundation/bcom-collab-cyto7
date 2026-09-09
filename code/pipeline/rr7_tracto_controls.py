"""RR7 Path A - bundle-level model for the tractography type-distance result.

The published §3.7 analysis aggregates the 33 named HCP-1065 atlas bundles into a
7x7 type-pair matrix, so it has n = 28 observations and controls only contact
area. Path A moves the model to the level at which the data actually exist: one
row per bundle x type-pair record (174 rows), with

  outcome     Streamline_Count, negative binomial with a log link
  offset      log(n_vertices(source) x n_vertices(target)), the reviewer's
              "normalize by possible endpoint pairs"
  predictor   type-distance |tau_source - tau_target|
  covariates  Mean_Length_mm, type-pair contact area, endpoint surface areas,
              geodesic distance between the endpoint types, Category
  clustering  Tract_Name (cluster-robust covariance, plus a Poisson random-
              intercept fit as a sensitivity check)

and a null that preserves the type map's spatial structure instead of destroying
it by label permutation. Per-streamline endpoints are not persisted anywhere in
the repo, so the rotation cannot be applied to the true endpoint coordinates; the
null is therefore built on the per-type aggregates (each type's whole footprint
takes the area-weighted mean rotated ordinal rank), which is the documented
fallback. The published label permutation is reported beside it.

Run::
    conda run -n cyto7 python scripts/rr7_tracto_controls.py --n-spin 1000
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import dijkstra

import rr_common as rc
from cyto7_surface_io import REPO_ROOT, resolve_target_map

OUTDIR = rc.OUT / "rr7_tracto"
CACHEDIR = rc.OUT / "_cache"
TRACTO = cfg.data_dir() / "tractography" / "v9"
TFIG = cfg.results_dir("tables") / "tractography"
GEOM_CSV = TRACTO / "cyto7.v9_tract_geometry_per_bundle.csv"
LABELS_NII = TRACTO / "cyto7.v9_in_reference.nii.gz"

# tractography CSV names -> ordinal rank 1..7
CYTO_ORDER = ["Allocortex", "Agranular", "Dysgranular", "Eulaminate-I",
              "Eulaminate-II", "Eulaminate-III", "Koniocortex"]
TIDX = {n: i + 1 for i, n in enumerate(CYTO_ORDER)}

PUBLISHED = {"slope": -1.2306, "perm_p": 0.003, "partial_r": -0.6358, "partial_p": 0.0019,
             "conn_vs_typedist_r": -0.7649, "conn_vs_contact_r": 0.5754}


# --------------------------------------------------------------------------- #
# Step 1 - reproduce the published aggregate numbers
# --------------------------------------------------------------------------- #


def reproduce(n_perm: int) -> dict:
    import reviewer_response as rr
    from plot_tractography_analysis import boundary_surface_matrix, load_geometry, stratified_matrices
    geometry = load_geometry(GEOM_CSV)
    short, _long, _all = stratified_matrices(geometry, 80.0)
    rows = short.index.tolist()

    def pairs(mat):
        xs, ys = [], []
        for i, ri in enumerate(rows):
            for j, rj in enumerate(rows):
                if j <= i:
                    continue
                xs.append(abs(TIDX[ri] - TIDX[rj]))
                ys.append(mat.iloc[i, j])
        return np.array(xs, float), np.array(ys, float)

    x, y = pairs(short)
    ly = np.log1p(y)
    slope = float(np.polyfit(x, ly, 1)[0])
    rng = np.random.default_rng(rr.SEED)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(len(CYTO_ORDER))
        pi = {n: perm[TIDX[n] - 1] for n in CYTO_ORDER}
        xs = np.array([abs(pi[ri] - pi[rj]) for i, ri in enumerate(rows)
                       for j, rj in enumerate(rows) if j > i], float)
        null[k] = np.polyfit(xs, ly, 1)[0]
    perm_p = float((np.sum(np.abs(null) >= abs(slope)) + 1) / (n_perm + 1))

    bnd = boundary_surface_matrix(LABELS_NII)
    common = [c for c in rows if c in bnd.index]
    xs, ys, zs = [], [], []
    for i, ri in enumerate(common):
        for j, rj in enumerate(common):
            if j <= i:
                continue
            xs.append(abs(TIDX[ri] - TIDX[rj]))
            ys.append(np.log1p(short.loc[ri, rj]))
            zs.append(bnd.loc[ri, rj])
    pr, pp = rr._partial_corr(np.array(ys, float), np.array(xs, float), np.array(zs, float))
    return {"slope": slope, "perm_p": perm_p, "partial_r": float(pr), "partial_p": float(pp),
            "n_pairs": len(xs), "contact_matrix": bnd, "short_matrix": short}


# --------------------------------------------------------------------------- #
# Step 2 - the bundle-level design matrix
# --------------------------------------------------------------------------- #


def type_areas_and_counts() -> tuple[dict[int, float], dict[int, int]]:
    """Surface area (mm^2, fsaverage white) and vertex count per cyto7 type."""
    lab = resolve_target_map("v9", "fsaverage")
    geom = rc.fsaverage_geometry("white", "164k")
    area = {c: 0.0 for c in range(1, 8)}
    cnt = {c: 0 for c in range(1, 8)}
    for H in ("L", "R"):
        coords, faces = geom[H]
        va = rc.vertex_areas(coords, faces)
        l = np.asarray(lab[H], int)
        for c in range(1, 8):
            m = l == c
            area[c] += float(va[m].sum())
            cnt[c] += int(m.sum())
    return area, cnt


def geodesic_type_distances(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Geodesic distance (mm) between cyto7 types on the fsaverage white surface.

    Two estimators: centroid-to-centroid (the vertex of each type nearest its
    mean coordinate, as the spec asks) and the mean footprint-to-footprint
    distance (multi-source Dijkstra from all of type a to all of type b), which
    does not assume a type is compact. Computed per hemisphere and averaged,
    since no mesh path crosses hemispheres.
    """
    cpath = CACHEDIR / "rr7_geodesic_type_distance.npz"
    if cpath.exists():
        z = np.load(cpath)
        return (pd.DataFrame(z["centroid"], index=CYTO_ORDER, columns=CYTO_ORDER),
                pd.DataFrame(z["footprint"], index=CYTO_ORDER, columns=CYTO_ORDER))
    lab = resolve_target_map("v9", "fsaverage")
    geom = rc.fsaverage_geometry("white", "164k")
    cen = np.zeros((7, 7)); foot = np.zeros((7, 7)); nh = np.zeros((7, 7))
    for H in ("L", "R"):
        coords, faces = geom[H]
        g = rc.geodesic_graph(coords, faces)
        l = np.asarray(lab[H], int)
        cvert = {}
        for c in range(1, 8):
            idx = np.where(l == c)[0]
            if idx.size:
                cvert[c] = int(idx[np.argmin(np.linalg.norm(coords[idx] - coords[idx].mean(0), axis=1))])
        # centroid-to-centroid
        srcs = [cvert[c] for c in sorted(cvert)]
        dmat = dijkstra(g, directed=False, indices=srcs)
        for a, ca in enumerate(sorted(cvert)):
            for b, cb in enumerate(sorted(cvert)):
                d = dmat[a, cvert[cb]]
                if np.isfinite(d):
                    cen[ca - 1, cb - 1] += d
                    nh[ca - 1, cb - 1] += 1
        # footprint-to-footprint (multi-source per type)
        log(f"    {H}: footprint geodesics ...")
        for ca in sorted(cvert):
            idx = np.where(l == ca)[0]
            dm = dijkstra(g, directed=False, indices=idx, min_only=True)
            for cb in sorted(cvert):
                jdx = np.where(l == cb)[0]
                v = dm[jdx]
                v = v[np.isfinite(v)]
                if v.size:
                    foot[ca - 1, cb - 1] += float(v.mean())
    with np.errstate(invalid="ignore", divide="ignore"):
        cen = np.where(nh > 0, cen / np.maximum(nh, 1), np.nan)
        foot = foot / 2.0
    foot = 0.5 * (foot + foot.T)
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    np.savez(cpath, centroid=cen, footprint=foot)
    return (pd.DataFrame(cen, index=CYTO_ORDER, columns=CYTO_ORDER),
            pd.DataFrame(foot, index=CYTO_ORDER, columns=CYTO_ORDER))


def build_design(contact: pd.DataFrame, geo_cen: pd.DataFrame, geo_foot: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_csv(GEOM_CSV)
    area, cnt = type_areas_and_counts()
    rows = []
    for _, r in g.iterrows():
        s, t = r["Source_Cyto"], r["Target_Cyto"]
        if s not in TIDX or t not in TIDX:
            continue
        si, ti = TIDX[s], TIDX[t]
        ca = float(contact.loc[s, t]) if (s in contact.index and t in contact.columns) else np.nan
        rows.append({
            "Category": r["Category"], "Tract_Name": r["Tract_Name"],
            "source_type": s, "target_type": t, "source_ord": si, "target_ord": ti,
            "type_distance": abs(si - ti),
            "streamline_count": int(r["Streamline_Count"]),
            "mean_length_mm": float(r["Mean_Length_mm"]),
            "mean_tortuosity": float(r["Mean_Tortuosity"]),
            "contact_area_mm2": ca,
            "area_source_mm2": area[si], "area_target_mm2": area[ti],
            "n_vertices_source": cnt[si], "n_vertices_target": cnt[ti],
            "possible_endpoint_pairs": float(cnt[si]) * float(cnt[ti]),
            "geodesic_centroid_mm": float(geo_cen.loc[s, t]),
            "geodesic_footprint_mm": float(geo_foot.loc[s, t]),
        })
    d = pd.DataFrame(rows)
    return _derive(d)


def aggregate_design(short: pd.DataFrame, contact: pd.DataFrame,
                     geo_cen: pd.DataFrame, geo_foot: pd.DataFrame) -> pd.DataFrame:
    """The published 7x7 short-range matrix as 28 unordered type-pair rows.

    Same outcome and offset as the bundle-level model, so the two levels differ
    only in aggregation. There is no tract to cluster on here, so the covariance
    is the ordinary one.
    """
    _area, cnt = type_areas_and_counts()
    rows = []
    for a in range(1, 8):
        for b in range(a + 1, 8):   # off-diagonal only, exactly as the published fit
            sa, sb = CYTO_ORDER[a - 1], CYTO_ORDER[b - 1]
            if sa not in short.index or sb not in short.columns:
                continue
            rows.append({"Category": "aggregate", "Tract_Name": f"{sa}|{sb}",
                         "source_type": sa, "target_type": sb, "source_ord": a, "target_ord": b,
                         "type_distance": b - a,
                         "streamline_count": int(short.loc[sa, sb]),
                         "mean_length_mm": np.nan, "mean_tortuosity": np.nan,
                         "contact_area_mm2": float(contact.loc[sa, sb]),
                         "area_source_mm2": np.nan, "area_target_mm2": np.nan,
                         "n_vertices_source": cnt[a], "n_vertices_target": cnt[b],
                         "possible_endpoint_pairs": float(cnt[a]) * float(cnt[b]),
                         "geodesic_centroid_mm": float(geo_cen.loc[sa, sb]),
                         "geodesic_footprint_mm": float(geo_foot.loc[sa, sb])})
    out = pd.DataFrame(rows)
    out["log_offset"] = np.log(out["possible_endpoint_pairs"])
    out["log_contact_area"] = np.log1p(out["contact_area_mm2"])
    out["is_commissural"] = 0.0
    return out


def _derive(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["log_offset"] = np.log(d["possible_endpoint_pairs"])
    d["log_area_source"] = np.log(d["area_source_mm2"])
    d["log_area_target"] = np.log(d["area_target_mm2"])
    d["log_contact_area"] = np.log1p(d["contact_area_mm2"])
    d["is_commissural"] = (d["Category"] == "commissural").astype(float)
    return d


def complete_zeros(d: pd.DataFrame, contact: pd.DataFrame,
                   geo_cen: pd.DataFrame, geo_foot: pd.DataFrame) -> pd.DataFrame:
    """Add the structural zeros: every tract x unordered type pair with no streamlines.

    The persisted CSV lists only the type pairs a bundle actually reaches, so a
    model fitted on those 174 rows conditions on connectivity being present and
    cannot speak to the published claim, whose 7x7 matrix contains the zeros. Each
    of the 33 tracts is therefore expanded to all 28 unordered type pairs
    (including within-type), absent rows entering as zero counts. Row length for a
    zero row is the tract's mean streamline length, the only length information
    the data contain for a pair that tract does not connect.
    """
    area, cnt = type_areas_and_counts()
    tract_len = d.groupby("Tract_Name")["mean_length_mm"].mean().to_dict()
    tract_cat = d.groupby("Tract_Name")["Category"].first().to_dict()
    have = {(r.Tract_Name, min(r.source_ord, r.target_ord), max(r.source_ord, r.target_ord)): r
            for r in d.itertuples()}
    rows = []
    for tract in sorted(d["Tract_Name"].unique()):
        for a in range(1, 8):
            for b in range(a, 8):
                key = (tract, a, b)
                if key in have:
                    r = have[key]
                    rows.append({c: getattr(r, c) for c in d.columns})
                    continue
                sa, sb = CYTO_ORDER[a - 1], CYTO_ORDER[b - 1]
                rows.append({
                    "Category": tract_cat[tract], "Tract_Name": tract,
                    "source_type": sa, "target_type": sb, "source_ord": a, "target_ord": b,
                    "type_distance": b - a, "streamline_count": 0,
                    "mean_length_mm": float(tract_len[tract]), "mean_tortuosity": np.nan,
                    "contact_area_mm2": float(contact.loc[sa, sb]) if sa in contact.index else np.nan,
                    "area_source_mm2": area[a], "area_target_mm2": area[b],
                    "n_vertices_source": cnt[a], "n_vertices_target": cnt[b],
                    "possible_endpoint_pairs": float(cnt[a]) * float(cnt[b]),
                    "geodesic_centroid_mm": float(geo_cen.loc[sa, sb]),
                    "geodesic_footprint_mm": float(geo_foot.loc[sa, sb]),
                })
    out = pd.DataFrame(rows)
    keep = [c for c in d.columns if c in out.columns]
    return _derive(out[keep])


# --------------------------------------------------------------------------- #
# Step 2/4 - model fits
# --------------------------------------------------------------------------- #


SPECS = [
    ("S0  type-distance only, no offset", []),
    ("S1  + endpoint-pair offset", [], True),
    ("S2  + length", ["mean_length_mm"], True),
    ("S3  + length + contact area", ["mean_length_mm", "log_contact_area"], True),
    ("S4  + length + contact + endpoint areas",
     ["mean_length_mm", "log_contact_area", "log_area_source", "log_area_target"], True),
    ("S5  + length + contact + areas + Category",
     ["mean_length_mm", "log_contact_area", "log_area_source", "log_area_target", "is_commissural"], True),
    ("S6  full + geodesic (centroid)",
     ["mean_length_mm", "log_contact_area", "log_area_source", "log_area_target",
      "is_commissural", "geodesic_centroid_mm"], True),
    ("S7  full + geodesic (footprint)",
     ["mean_length_mm", "log_contact_area", "log_area_source", "log_area_target",
      "is_commissural", "geodesic_footprint_mm"], True),
]


def _design_matrix(d: pd.DataFrame, covs: list[str], td_col: str, td_scale: str = "raw"):
    """[const, type-distance, standardised covariates].

    With ``td_scale='raw'`` type-distance keeps its natural scale, so the
    coefficient is per ordinal step. With ``td_scale='z'`` it is standardised, so
    the coefficient is per standard deviation: that is the form used for every
    null comparison, because the surrogate type-distances a rotation induces are
    compressed toward the mean and an unstandardised slope would be inflated by
    1/compression, biasing the null toward extreme values. The covariates are
    always z-scored, which is what makes the negative-binomial MLE converge on the
    zero-heavy design.
    """
    import statsmodels.api as sm
    td = d[td_col].astype(float).to_numpy()
    if td_scale == "z":
        sd = td.std()
        td = (td - td.mean()) / sd if sd > 0 else td * 0.0
    X = pd.DataFrame({td_col: td}, index=d.index)
    for c in covs:
        v = d[c].astype(float).to_numpy()
        v = np.where(np.isfinite(v), v, np.nanmean(v))
        sd = v.std()
        X[c] = (v - v.mean()) / sd if sd > 0 else 0.0
    return sm.add_constant(X, has_constant="add")


def _fit_nb(d: pd.DataFrame, covs: list[str], use_offset: bool, td_col: str = "type_distance",
            td_scale: str = "raw"):
    """Negative binomial, log link, optional log offset, tract-clustered covariance."""
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial
    X = _design_matrix(d, covs, td_col, td_scale)
    y = d["streamline_count"].astype(float).to_numpy()
    off = d["log_offset"].to_numpy() if use_offset else None
    groups = d["Tract_Name"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois = sm.GLM(y, X.to_numpy(), family=sm.families.Poisson(), offset=off).fit()
        start = np.append(pois.params, 1.0)
        m = NegativeBinomial(y, X.to_numpy(), offset=off, loglike_method="nb2")
        res = None
        for kw in ({"cov_type": "cluster", "cov_kwds": {"groups": groups}}, {}):
            for method in ("bfgs", "nm"):
                try:
                    r = m.fit(start_params=start, method=method, maxiter=5000, disp=0, **kw)
                    if np.isfinite(r.bse).all():
                        res = r
                        break
                except Exception:
                    continue
            if res is not None:
                break
        if res is None:
            res = m.fit(start_params=start, disp=0, maxiter=5000)
    names = list(X.columns)
    k = names.index(td_col)
    conv = bool(res.mle_retvals.get("converged", True)) if hasattr(res, "mle_retvals") else True
    return {"coef": float(res.params[k]), "se": float(res.bse[k]),
            "z": float(res.tvalues[k]), "p_model": float(res.pvalues[k]),
            "converged": conv,
            "alpha": float(res.params[-1]) if len(res.params) > len(names) else np.nan,
            "n": int(len(d)), "covariates": ("none" if not covs else "+".join(covs))}


def _fit_poisson_cluster(d: pd.DataFrame, covs: list[str], use_offset: bool,
                         td_col: str = "type_distance", td_scale: str = "raw"):
    """Poisson log-link with tract-clustered sandwich covariance.

    Reported beside the negative binomial because the sandwich standard errors are
    valid under arbitrary overdispersion without an alpha to estimate, so this
    specification cannot fail to converge on the zero-heavy design.
    """
    import statsmodels.api as sm
    X = _design_matrix(d, covs, td_col, td_scale)
    y = d["streamline_count"].astype(float).to_numpy()
    off = d["log_offset"].to_numpy() if use_offset else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(y, X.to_numpy(), family=sm.families.Poisson(), offset=off).fit(
            cov_type="cluster", cov_kwds={"groups": d["Tract_Name"].to_numpy()})
    k = list(X.columns).index(td_col)
    return {"coef_pois": float(res.params[k]), "se_pois": float(res.bse[k]),
            "z_pois": float(res.tvalues[k]), "p_pois": float(res.pvalues[k])}


def _fit_poisson_mixed(d: pd.DataFrame, covs: list[str]):
    """Poisson random-intercept fit (Tract_Name), the literal random-effects check."""
    from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
    import statsmodels.api as sm
    X = pd.DataFrame({"type_distance": d["type_distance"].astype(float)})
    for c in covs:
        X[c] = d[c].astype(float)
    X = sm.add_constant(X, has_constant="add")
    exog_vc = pd.get_dummies(d["Tract_Name"]).astype(float).to_numpy()
    ident = np.zeros(exog_vc.shape[1], int)
    y = d["streamline_count"].astype(float).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PoissonBayesMixedGLM(y, X.to_numpy(), exog_vc, ident,
                                 vcp_p=2.0, fe_p=2.0)
        # the offset enters as a fixed, known term: absorb it into the linear predictor
        r = m.fit_vb(verbose=False)
    k = list(X.columns).index("type_distance")
    return {"coef": float(r.fe_mean[k]), "se": float(r.fe_sd[k])}


# --------------------------------------------------------------------------- #
# Step 3 - nulls
# --------------------------------------------------------------------------- #


def rotated_type_ranks(n_spin: int, log=print) -> np.ndarray:
    """(n_spin, 7) area-weighted mean rotated ordinal rank of each type's footprint.

    For rotation i, every vertex of type c receives the rotated map's rank; the
    area-weighted mean over c's footprint is the surrogate ordinal position of c.
    This keeps the spatial structure and adjacency of the type map (the reviewer's
    objection to the label permutation) while remaining computable without the
    per-streamline endpoints, which are not persisted.
    """
    cpath = CACHEDIR / f"rr7_rotated_type_ranks_n{n_spin}.npy"
    if cpath.exists():
        return np.load(cpath)
    lab32 = rc.labels_32k()
    labc = rc.cat(lab32, int).astype(int)
    geom = rc.fslr_geometry("midthickness")
    va = np.concatenate([rc.vertex_areas(*geom[H]) for H in ("L", "R")])
    nulls = rc.load_nulls(n_spin)
    out = np.full((n_spin, 7), np.nan)
    for i in range(n_spin):
        col = nulls[:, i]
        ok = np.isfinite(col)
        for c in range(1, 8):
            m = (labc == c) & ok
            if m.any():
                out[i, c - 1] = float(np.average(col[m], weights=va[m]))
        if (i + 1) % 250 == 0:
            log(f"    rotated ranks {i + 1}/{n_spin}")
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    np.save(cpath, out)
    return out


def spin_null_coef(d: pd.DataFrame, covs: list[str], use_offset: bool,
                   rot_ranks: np.ndarray, log=print) -> tuple[np.ndarray, float]:
    """Null distribution of the type-distance coefficient under rotated type ranks."""
    n_spin = rot_ranks.shape[0]
    so = d["source_ord"].to_numpy() - 1
    to = d["target_ord"].to_numpy() - 1
    coefs = np.full(n_spin, np.nan)
    dd = d.copy()
    for i in range(n_spin):
        r = rot_ranks[i]
        if not np.all(np.isfinite(r)):
            continue
        dd["td_rot"] = np.abs(r[so] - r[to])
        if np.allclose(dd["td_rot"].std(), 0):
            continue
        try:
            coefs[i] = _fit_nb(dd, covs, use_offset, td_col="td_rot", td_scale="z")["coef"]
        except Exception:
            coefs[i] = np.nan
        if (i + 1) % 100 == 0:
            log(f"      spin fit {i + 1}/{n_spin}")
    return coefs, float(np.nanmean(coefs))


def published_slope_null(short: pd.DataFrame, rot_ranks: np.ndarray) -> dict:
    """The published log-linear slope, tested against the topology-preserving null.

    This is the statistic §3.7 reports (ordinary least squares of log1p(short-range
    streamline count) on type-distance over the 21 off-diagonal type pairs). The
    published p came from permuting the seven type labels, which destroys the
    map's spatial structure; here the same statistic is recomputed with the
    surrogate type-distances that the released rotations induce.
    """
    rows = [c for c in CYTO_ORDER if c in short.index]
    pairs = [(i, j) for i in range(len(rows)) for j in range(len(rows)) if j > i]
    ly = np.array([np.log1p(short.iloc[i, j]) for i, j in pairs], float)
    oi = np.array([TIDX[rows[i]] - 1 for i, _j in pairs])
    oj = np.array([TIDX[rows[j]] - 1 for _i, j in pairs])
    x_obs = np.abs(oi - oj).astype(float)
    obs = float(np.polyfit(x_obs, ly, 1)[0])

    def _z(v):
        return (v - v.mean()) / v.std()

    # Scale-free forms: the slope on a standardised predictor, and the Pearson r.
    # A rotation compresses the surrogate type-distances toward their mean, so a
    # raw slope would be inflated by 1/compression and the null would be biased.
    obs_z = float(np.polyfit(_z(x_obs), ly, 1)[0])
    obs_r = float(stats.pearsonr(x_obs, ly)[0])
    n = rot_ranks.shape[0]
    null_z = np.full(n, np.nan)
    null_r = np.full(n, np.nan)
    sds = np.full(n, np.nan)
    for k in range(n):
        r = rot_ranks[k]
        if not np.all(np.isfinite(r)):
            continue
        x = np.abs(r[oi] - r[oj])
        if x.std() == 0:
            continue
        sds[k] = x.std()
        null_z[k] = np.polyfit(_z(x), ly, 1)[0]
        null_r[k] = stats.pearsonr(x, ly)[0]
    gz = null_z[np.isfinite(null_z)]
    gr = null_r[np.isfinite(null_r)]
    return {"observed_slope_per_ordinal_step": round(obs, 4),
            "observed_slope_per_SD": round(obs_z, 4),
            "observed_pearson_r": round(obs_r, 4),
            "p_spin_slope_per_SD": round(float((np.sum(np.abs(gz) >= abs(obs_z)) + 1) / (gz.size + 1)), 4),
            "p_spin_pearson_r": round(float((np.sum(np.abs(gr) >= abs(obs_r)) + 1) / (gr.size + 1)), 4),
            "spin_null_mean_slope_per_SD": round(float(gz.mean()), 4),
            "spin_null_sd_slope_per_SD": round(float(gz.std()), 4),
            "spin_null_mean_pearson_r": round(float(gr.mean()), 4),
            "observed_typedistance_SD": round(float(x_obs.std()), 4),
            "surrogate_typedistance_SD_median": round(float(np.nanmedian(sds)), 4),
            "n_spin_fits": int(gz.size)}


def label_perm_coef(d: pd.DataFrame, covs: list[str], use_offset: bool,
                    n_perm: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    so = d["source_ord"].to_numpy() - 1
    to = d["target_ord"].to_numpy() - 1
    coefs = np.full(n_perm, np.nan)
    dd = d.copy()
    for k in range(n_perm):
        p = rng.permutation(7)
        dd["td_rot"] = np.abs(p[so] - p[to])
        if np.allclose(dd["td_rot"].std(), 0):
            continue
        try:
            coefs[k] = _fit_nb(dd, covs, use_offset, td_col="td_rot", td_scale="z")["coef"]
        except Exception:
            coefs[k] = np.nan
    return coefs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--null-spins", type=int, default=None,
                    help="rotations actually refitted for the null (default: --n-spin)")
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frozen_before = rc.frozen_hashes()

    print("== Step 1: reproduce the published aggregate numbers ==")
    rep = reproduce(args.n_perm)
    repro = {k: {"published": PUBLISHED[k], "recomputed": round(rep[k], 4)}
             for k in ("slope", "perm_p", "partial_r", "partial_p")}
    print(json.dumps(repro, indent=2))

    print("== Step 2/4: bundle-level design matrix ==")
    geo_cen, geo_foot = geodesic_type_distances()
    geo_cen.round(2).to_csv(OUTDIR / "geodesic_type_distance_centroid_mm.csv")
    geo_foot.round(2).to_csv(OUTDIR / "geodesic_type_distance_footprint_mm.csv")
    d_obs = build_design(rep["contact_matrix"], geo_cen, geo_foot)
    d = complete_zeros(d_obs, rep["contact_matrix"], geo_cen, geo_foot)
    d_obs.to_csv(OUTDIR / "bundle_level_design_observed_only.csv", index=False)
    d.to_csv(OUTDIR / "bundle_level_design.csv", index=False)
    print(f"  observed records: {len(d_obs)} rows, {d_obs.Tract_Name.nunique()} tracts, "
          f"{d_obs.Category.value_counts().to_dict()}")
    print(f"  zero-completed:   {len(d)} rows ({int((d.streamline_count == 0).sum())} structural zeros)")

    # type-distance vs geodesic distance collinearity
    gd = []
    for a in range(1, 8):
        for b in range(a + 1, 8):
            gd.append({"type_a": CYTO_ORDER[a - 1], "type_b": CYTO_ORDER[b - 1],
                       "type_distance": b - a,
                       "geodesic_centroid_mm": round(float(geo_cen.iloc[a - 1, b - 1]), 2),
                       "geodesic_footprint_mm": round(float(geo_foot.iloc[a - 1, b - 1]), 2)})
    gdf = pd.DataFrame(gd)
    for col in ("geodesic_centroid_mm", "geodesic_footprint_mm"):
        r_p = stats.pearsonr(gdf["type_distance"], gdf[col])
        r_s = stats.spearmanr(gdf["type_distance"], gdf[col])
        gdf.attrs[col] = (r_p, r_s)
        print(f"  type-distance vs {col}: pearson r={r_p[0]:+.3f} (p {r_p[1]:.4f}), "
              f"spearman rho={r_s[0]:+.3f} (p {r_s[1]:.4f})")
    gdf.to_csv(OUTDIR / "geodesic_vs_typedistance.csv", index=False)
    coll = {c: {"pearson_r": round(float(gdf.attrs[c][0][0]), 4),
                "pearson_p": round(float(gdf.attrs[c][0][1]), 4),
                "spearman_rho": round(float(gdf.attrs[c][1][0]), 4),
                "spearman_p": round(float(gdf.attrs[c][1][1]), 4)}
            for c in ("geodesic_centroid_mm", "geodesic_footprint_mm")}

    print("== model specifications ==")
    fits = []
    for design_name, dd in (("bundle x type-pair, zero-completed (primary)", d),
                            ("bundle x type-pair, observed records only", d_obs)):
        for spec in SPECS:
            name, covs = spec[0], spec[1]
            use_off = len(spec) > 2 and spec[2]
            f = _fit_nb(dd, covs, use_off)
            f.update(_fit_poisson_cluster(dd, covs, use_off))
            f.update({"design": design_name, "spec": name, "offset": bool(use_off)})
            fits.append(f)
            print(f"  [{design_name[:34]:34s}] {name:42s} NB beta={f['coef']:+.4f} "
                  f"se={f['se']:.4f} p={f['p_model']:.4g} | Pois beta={f['coef_pois']:+.4f} "
                  f"p={f['p_pois']:.4g}")

    # aggregate 7x7 level, for comparability with the published analysis
    agg = aggregate_design(rep["short_matrix"], rep["contact_matrix"], geo_cen, geo_foot)
    agg.to_csv(OUTDIR / "aggregate_typepair_design.csv", index=False)
    for name, covs, off in (("A0  aggregate, no offset", [], False),
                            ("A1  aggregate + endpoint-pair offset", [], True),
                            ("A2  aggregate + offset + contact", ["log_contact_area"], True),
                            ("A3  aggregate + offset + contact + geodesic",
                             ["log_contact_area", "geodesic_centroid_mm"], True)):
        f = _fit_nb(agg, covs, off)
        f.update(_fit_poisson_cluster(agg, covs, off))
        f.update({"design": f"aggregate 7x7 off-diagonal pairs (n={len(agg)})",
                  "spec": name, "offset": bool(off)})
        fits.append(f)
        print(f"  [aggregate off-diagonal            ] {name:42s} NB beta={f['coef']:+.4f} "
              f"se={f['se']:.4f} p={f['p_model']:.4g} | Pois beta={f['coef_pois']:+.4f} "
              f"p={f['p_pois']:.4g}")

    print("== Step 3: nulls on the fullest specification ==")
    n_null = args.null_spins or args.n_spin
    rot = rotated_type_ranks(args.n_spin)[:n_null]
    full_name, full_covs, full_off = SPECS[-2][0], SPECS[-2][1], True   # S6 (centroid geodesic)
    key_specs = [("S1  + endpoint-pair offset", SPECS[1][1], True),
                 ("S5  full without geodesic", SPECS[5][1], True),
                 (full_name, full_covs, full_off)]
    pub_null = published_slope_null(rep["short_matrix"], rot)
    print(f"  published log-linear slope {pub_null['observed_slope_per_ordinal_step']:+.4f} per "
          f"ordinal step = {pub_null['observed_slope_per_SD']:+.4f} per SD, Pearson r "
          f"{pub_null['observed_pearson_r']:+.4f}")
    print(f"    topology-preserving spin p: {pub_null['p_spin_slope_per_SD']:.4f} (slope per SD, "
          f"null mean {pub_null['spin_null_mean_slope_per_SD']:+.4f} sd "
          f"{pub_null['spin_null_sd_slope_per_SD']:.4f}), "
          f"{pub_null['p_spin_pearson_r']:.4f} (Pearson r, null mean "
          f"{pub_null['spin_null_mean_pearson_r']:+.4f}); published label-permutation p = "
          f"{PUBLISHED['perm_p']}")
    print(f"    type-distance SD observed {pub_null['observed_typedistance_SD']:.3f} vs surrogate "
          f"median {pub_null['surrogate_typedistance_SD_median']:.3f}")
    null_rows = []
    for name, covs, off in (("A1  aggregate + endpoint-pair offset", [], True),
                            ("A3  aggregate + offset + contact + geodesic",
                             ["log_contact_area", "geodesic_centroid_mm"], True)):
        obs = _fit_nb(agg, covs, off)["coef"]
        obs_z = _fit_nb(agg, covs, off, td_scale="z")["coef"]
        sp, spmean = spin_null_coef(agg, covs, off, rot, log=lambda *_a: None)
        good = sp[np.isfinite(sp)]
        p_spin = float((np.sum(np.abs(good) >= abs(obs_z)) + 1) / (good.size + 1)) if good.size else np.nan
        lp = label_perm_coef(agg, covs, off, args.n_perm)
        lpg = lp[np.isfinite(lp)]
        p_perm = float((np.sum(np.abs(lpg) >= abs(obs_z)) + 1) / (lpg.size + 1)) if lpg.size else np.nan
        null_rows.append({"spec": name, "coef": round(obs, 4), "coef_per_SD": round(obs_z, 4),
                          "spin_null_mean": round(spmean, 4), "n_spin_fits": int(good.size),
                          "p_spin_typepreserving": round(p_spin, 4),
                          "label_perm_null_mean": round(float(np.nanmean(lpg)), 4),
                          "n_perm_fits": int(lpg.size), "p_label_perm": round(p_perm, 4)})
        print(f"  {name:48s} obs={obs:+.4f} (per SD {obs_z:+.4f}) spin-p={p_spin:.4f} perm-p={p_perm:.4f}")
    for name, covs, off in key_specs:
        obs = _fit_nb(d, covs, off)["coef"]
        obs_z = _fit_nb(d, covs, off, td_scale="z")["coef"]
        sp, spmean = spin_null_coef(d, covs, off, rot)
        good = sp[np.isfinite(sp)]
        p_spin = float((np.sum(np.abs(good) >= abs(obs_z)) + 1) / (good.size + 1)) if good.size else np.nan
        lp = label_perm_coef(d, covs, off, args.n_perm)
        lpg = lp[np.isfinite(lp)]
        p_perm = float((np.sum(np.abs(lpg) >= abs(obs_z)) + 1) / (lpg.size + 1)) if lpg.size else np.nan
        null_rows.append({"spec": name, "coef": round(obs, 4), "coef_per_SD": round(obs_z, 4),
                          "spin_null_mean": round(spmean, 4), "n_spin_fits": int(good.size),
                          "p_spin_typepreserving": round(p_spin, 4),
                          "label_perm_null_mean": round(float(np.nanmean(lpg)), 4),
                          "n_perm_fits": int(lpg.size), "p_label_perm": round(p_perm, 4)})
        print(f"  {name:48s} obs={obs:+.4f} (per SD {obs_z:+.4f}) spin-p={p_spin:.4f} perm-p={p_perm:.4f}")
        np.save(OUTDIR / f"_nulldist_spin_{name.split()[0]}.npy", sp)

    fdf = pd.DataFrame(fits)[["design", "spec", "covariates", "offset", "n", "coef", "se", "z",
                              "p_model", "alpha", "converged"]]
    ndf = pd.DataFrame(null_rows)
    nsub = ndf[["spec", "coef_per_SD", "p_spin_typepreserving", "p_label_perm",
                "spin_null_mean"]].rename(columns={"spin_null_mean": "spin_null_mean_per_SD"})
    out = fdf.merge(nsub, on="spec", how="left")
    out.loc[out["design"] == "bundle x type-pair, observed records only",
            ["coef_per_SD", "p_spin_typepreserving", "p_label_perm",
             "spin_null_mean_per_SD"]] = np.nan
    ndf.to_csv(OUTDIR / "null_comparison.csv", index=False)
    out.to_csv(OUTDIR / "model_coefficients.csv", index=False)
    print("\n", out.to_string(index=False))

    print("== sensitivity: Poisson random intercept on Tract_Name ==")
    mixed = {}
    try:
        mx = _fit_poisson_mixed(d, SPECS[-2][1])
        mixed = {"coef": round(mx["coef"], 4), "se": round(mx["se"], 4)}
        print(f"  random-intercept Poisson beta={mx['coef']:+.4f} (sd {mx['se']:.4f})")
    except Exception as e:
        mixed = {"error": f"{type(e).__name__}: {e}"}
        print("  mixed fit failed:", mixed["error"])

    ok, detail = rc.check_frozen(frozen_before)
    (OUTDIR / "rr7_summary.json").write_text(json.dumps({
        "reproduction": repro, "collinearity_typedist_vs_geodesic": coll,
        "published_slope_vs_topology_preserving_null": pub_null,
        "poisson_random_intercept": mixed,
        "n_rows": int(len(d)), "n_tracts": int(d.Tract_Name.nunique()),
        "n_spin_requested": args.n_spin, "n_spin_fitted": n_null, "n_perm": args.n_perm,
        "endpoint_note": ("per-streamline endpoints are not persisted; the spin null is built on "
                          "per-type aggregates (area-weighted mean rotated rank per type footprint)"),
        "frozen_unchanged": ok, "frozen": detail}, indent=2), encoding="utf-8")
    print("frozen files unchanged:", ok)
    return out


if __name__ == "__main__":
    main()
