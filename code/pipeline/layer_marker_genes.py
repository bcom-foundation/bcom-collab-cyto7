"""Analysis 1 — layer-marker gene composites (molecular test of the layer-IV criterion).

SPEC_definitional_validation, Analysis 1. cyto7 is defined by layer-IV granularity,
so the layer-IV excitatory marker **RORB** should RISE with type (allo->konio):
a falsifiable molecular test of the definitional criterion, on the AHBA/abagen
pipeline. Pre-registered primary: RORB ρ > 0.

Resolution choice (documented): individual-gene AHBA expression is naturally
**parcel-level**, so we use the **parcel-based spin** — abagen on the volumetric
Desikan-Killiany atlas -> region x gene, each Desikan cortical parcel assigned a
continuous cyto7 type score (the released `desikan_x_cyto7_composition.csv`
c_rank = Σ_t t·fraction_t, identical to the ENIGMA disease test), and the fsa5
aparc rotate_parcellation spin (same machinery as fig9_predictions Part B).

Genes: RORB (L4, primary), CUX2 (upper L2/3), FEZF2 (L5), FOXP2 (L6), TLE4 (L6),
plus z-scored upper/granular/deep composites. BH within this layer-marker family.

Caveats (stated): AHBA six donors (five male), mostly LH; transcript != protein.
Allocortex-excluded sensitivity reported.

Run::  conda activate cyto7 && python scripts/layer_marker_genes.py
(requires the AHBA microarray; abagen.fetch_microarray must have succeeded.)
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import sys
import types
from pathlib import Path

# pkg_resources shim (setuptools>=81 removed it; abagen 0.1.3 imports it). No setuptools change.
try:
    import pkg_resources  # noqa
except ImportError:
    import importlib.resources as _ir
    _m = types.ModuleType("pkg_resources")
    _m.resource_filename = lambda package, resource: str(_ir.files(package).joinpath(resource))
    sys.modules["pkg_resources"] = _m

import numpy as np
import pandas as pd
from scipy import stats

from cyto7_surface_io import REPO_ROOT

CROSSED = cfg.results_dir("tables") / "crossed"
OUT = cfg.results_dir("tables/definitional")
SEED = 0
# AHBA donors: donor 15496 has a dead Allen download URL in abagen 0.1.3 (HTTP 404),
# so we use the five donors that fetched (scientifically standard for AHBA). Documented.
DONORS = ["9861", "10021", "12876", "14380", "15697"]
GENES = ["RORB", "CUX2", "FEZF2", "FOXP2", "TLE4"]
GENE_LAYER = {"RORB": "IV (granular)", "CUX2": "II/III (upper)", "FEZF2": "V (deep)",
              "FOXP2": "VI (deep)", "TLE4": "VI (deep)"}
PRIMARY = "RORB"


def crank_by_name() -> dict[str, float]:
    """Per-Desikan-parcel continuous cyto7 type score (hemi-pooled), from the released CSV."""
    comp = pd.read_csv(CROSSED / "desikan_x_cyto7_composition.csv")
    pct = ["Allo_pct", "Agr_pct", "Dys_pct", "EulI_pct", "EulII_pct", "EulIII_pct", "Kon_pct"]
    frac = comp[pct].values / 100.0
    t = np.arange(1, 8)
    with np.errstate(invalid="ignore"):
        cr = (frac * t).sum(1) / frac.sum(1)
    return {str(p).lower(): float(v) for p, v in zip(comp["anatomical_parcel"], cr)}


def aparc_order_and_perm(n_rot=1000):
    """Canonical 68-parcel order (L then R aparc names) + rotate_parcellation perm ids."""
    import nibabel as nib
    import enigmatoolbox
    from nilearn import datasets as nd
    from enigmatoolbox.permutation_testing.permutation_testing import (
        centroid_extraction_sphere, rotate_parcellation)
    annot_dir = Path(enigmatoolbox.__file__).parent / "permutation_testing" / "annot"
    names = []
    for H, fn in (("l", "fsa5_lh_aparc.annot"), ("r", "fsa5_rh_aparc.annot")):
        _, _, nm = nib.freesurfer.io.read_annot(str(annot_dir / fn))
        nm = [n.decode() if isinstance(n, bytes) else n for n in nm]
        # centroid_extraction_sphere excludes 'unknown'/'corpuscallosum' in ENIGMA's order;
        # replicate: keep the same non-medial parcels, in annot label order.
        for n in nm:
            if n.lower() not in ("unknown", "corpuscallosum"):
                names.append((H, n.lower()))
    fs = nd.fetch_surf_fsaverage("fsaverage5")
    cL = np.asarray(nib.load(fs["sphere_left"]).darrays[0].data, float)
    cR = np.asarray(nib.load(fs["sphere_right"]).darrays[0].data, float)
    ctr_l = centroid_extraction_sphere(cL, str(annot_dir / "fsa5_lh_aparc.annot"))
    ctr_r = centroid_extraction_sphere(cR, str(annot_dir / "fsa5_rh_aparc.annot"))
    np.random.seed(SEED)
    perm = rotate_parcellation(ctr_l, ctr_r, nrot=n_rot)
    return names, perm


EXPR_CSV = OUT / "layer_marker_expression.csv"


def get_gene_expression(log) -> pd.DataFrame:
    """Desikan region x gene expression indexed by (hemi, parcel).

    Prefers the CSV exported by the isolated pandas<2 `abagen_env`
    (scripts run in the cyto7 env cannot import abagen 0.1.3 under pandas 3).
    Falls back to running abagen directly only if the CSV is absent.
    """
    if EXPR_CSV.exists():
        log(f"  loading pre-exported AHBA expression: {EXPR_CSV.name}")
        raw = pd.read_csv(EXPR_CSV)
        raw["hemi"] = raw["hemi"].str.lower()
        raw["parcel"] = raw["parcel"].str.lower()
        df = raw.set_index(["hemi", "parcel"])
        return df
    log("  [WARN] expression CSV absent; attempting direct abagen (needs pandas<2)...")
    import abagen
    atlas = abagen.fetch_desikan_killiany(native=False)
    expr = abagen.get_expression_data(atlas["image"], atlas["info"], donors=DONORS,
                                      lr_mirror="bidirectional", norm_matched=False,
                                      missing="interpolate", n_proc=1, verbose=0)
    info = pd.read_csv(atlas["info"])
    info = info[info["structure"] == "cortex"]
    idmap = {int(r.id): (str(r.hemisphere).lower(), str(r.label).lower()) for r in info.itertuples()}
    rows = {(h, nm): expr.loc[i] for i, (h, nm) in idmap.items() if i in expr.index}
    return pd.DataFrame(rows).T


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = print
    log("== Analysis 1 — layer-marker genes (RORB primary, parcel-level abagen + fsa5 aparc spin) ==")
    try:
        expr = get_gene_expression(log)
    except Exception as exc:
        msg = (f"STOP-AND-LOG: AHBA/abagen expression unavailable ({type(exc).__name__}: "
               f"{str(exc)[:200]}). Analysis 1 not run; see report.")
        log(msg)
        (OUT / "layer_marker_genes.STOP.txt").write_text(msg, encoding="utf-8")
        return 3

    missing = [g for g in GENES if g not in expr.columns]
    if missing:
        log(f"  [WARN] genes not returned by abagen probe selection: {missing}")
    genes = [g for g in GENES if g in expr.columns]

    from enigmatoolbox.permutation_testing.permutation_testing import perm_sphere_p
    names, perm = aparc_order_and_perm(1000)
    crank = crank_by_name()
    cr_vec = np.array([crank.get(nm, np.nan) for (_h, nm) in names])

    def gene_vec(g):
        return np.array([expr.loc[(h, nm), g] if (h, nm) in expr.index else np.nan
                         for (h, nm) in names], float)

    # z-scored composites
    z = {g: (gene_vec(g) - np.nanmean(gene_vec(g))) / np.nanstd(gene_vec(g)) for g in genes}
    comp = {}
    if "CUX2" in z:
        comp["upper(CUX2)"] = z["CUX2"]
    if "RORB" in z:
        comp["granular(RORB)"] = z["RORB"]
    deep = [g for g in ("FEZF2", "FOXP2", "TLE4") if g in z]
    if deep:
        comp["deep(FEZF2/FOXP2/TLE4)"] = np.nanmean([z[g] for g in deep], axis=0)

    def evaluate(vec, cr, perm):
        keep = np.isfinite(vec) & np.isfinite(cr)
        rho = float(stats.spearmanr(vec[keep], cr[keep])[0])
        try:
            p = float(np.ravel(perm_sphere_p(vec, cr, perm, corr_type="spearman"))[0])
        except Exception as exc:
            log(f"    spin failed: {exc!r}"); p = float("nan")
        return rho, p, int(keep.sum())

    def evaluate_allo_excl(vec, cr, perm):
        cr2 = np.where(cr >= 1.5, cr, np.nan)  # drop allocortex-dominated parcels (crank<1.5)
        keep = np.isfinite(vec) & np.isfinite(cr2)
        rho = float(stats.spearmanr(vec[keep], cr2[keep])[0])
        return rho, int(keep.sum())

    rows = []
    for g in genes:
        v = gene_vec(g)
        rho, p, n = evaluate(v, cr_vec, perm)
        rho_ex, n_ex = evaluate_allo_excl(v, cr_vec, perm)
        rows.append(dict(feature=g, kind="gene", layer=GENE_LAYER[g], spearman_rho=rho,
                         p_spin=p, n=n, spearman_rho_allo_excl=rho_ex, n_allo_excl=n_ex,
                         prereg=("RORB rho>0 (primary)" if g == PRIMARY else "")))
    for cname, cvec in comp.items():
        rho, p, n = evaluate(cvec, cr_vec, perm)
        rho_ex, n_ex = evaluate_allo_excl(cvec, cr_vec, perm)
        rows.append(dict(feature=cname, kind="composite", layer="", spearman_rho=rho,
                         p_spin=p, n=n, spearman_rho_allo_excl=rho_ex, n_allo_excl=n_ex, prereg=""))
    df = pd.DataFrame(rows)
    from external_validation import _bh
    df["p_spin_fdr"] = _bh(df["p_spin"].values)  # one layer-marker-gene family
    df.to_csv(OUT / "layer_marker_genes.csv", index=False)
    log(df.to_string(index=False))
    log(f"  wrote {OUT/'layer_marker_genes.csv'}")
    _plot(genes, gene_vec, comp, cr_vec, df, names)
    r = df[df.feature == PRIMARY].iloc[0]
    log(f"\n  PRE-REGISTERED PRIMARY RORB: rho={r.spearman_rho:+.3f}, spin p={r.p_spin:.3f}, "
        f"q={r.p_spin_fdr:.3f} -> {'SUPPORTS' if r.spearman_rho>0 and r.p_spin_fdr<0.05 else 'not FDR-sig'} "
        f"the layer-IV definitional criterion.")
    return 0


def _plot(genes, gene_vec, comp, cr_vec, df, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sdf = df.set_index("feature")
    fig_w = 190 / 25.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_w * 0.42))
    # (a) RORB scatter vs type
    v = gene_vec("RORB") if "RORB" in genes else np.full(len(names), np.nan)
    ax1.scatter(cr_vec, v, s=10, c="#762a83", alpha=0.7)
    ax1.set_xlabel("cyto7 type score (allo->konio)", fontsize=7)
    ax1.set_ylabel("RORB expression (parcel)", fontsize=7)
    s = sdf.loc["RORB"] if "RORB" in sdf.index else None
    if s is not None:
        ax1.set_title(f"RORB (L4): rho={s.spearman_rho:+.3f}, spin p={s.p_spin:.3f}", fontsize=7.5)
    ax1.tick_params(labelsize=6.5)
    # (b) forest of rho per gene/composite
    order = list(df.feature)
    y = np.arange(len(order))
    ax2.barh(y, df["spearman_rho"].values, color=["#762a83" if k == "gene" else "#1b7837"
                                                  for k in df["kind"]])
    ax2.set_yticks(y); ax2.set_yticklabels(order, fontsize=6.5)
    ax2.axvline(0, color="0.5", lw=0.8)
    for i, (_, rr) in enumerate(df.iterrows()):
        star = "*" if (np.isfinite(rr.p_spin_fdr) and rr.p_spin_fdr < 0.05) else ""
        ax2.text(rr.spearman_rho, i, f" {rr.spearman_rho:+.2f}{star}", va="center", fontsize=6)
    ax2.set_xlabel("Spearman rho vs cyto7 type", fontsize=7)
    ax2.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(str(OUT / "layer_marker_genes.png"), dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT/'layer_marker_genes.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
