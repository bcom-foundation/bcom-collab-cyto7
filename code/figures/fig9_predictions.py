"""Test more Zaldivar-Diez & Garcia-Cabezas (2026) Fig. 9 predictions against cyto7 v9.

Implements ``docs/SPEC_fig9_predictions_batch.md``. Builds on the receptor work
(``receptor_type_connectivity.py``) and reuses its **exact** spin machinery + the same
Alexander-Bloch rotation set (fsLR 32k, n_perm=1000, seed=0) as the released tables.

Parts:
  A  receptor diversity (Shannon entropy over the 19 Hansen maps) vs cyto7 type
  C  Hill-2010 evolutionary + developmental cortical-expansion maps vs cyto7 type
  B  ENIGMA case-control cortical-thinning maps vs continuous per-parcel cyto7 type score
  D  handled by re-running receptor_type_connectivity.part_b (dispersion fix) — see that file

FDR families kept separate from the published ones. A + C are pooled into a small
"Fig-9" family (H, evoexp, devexp); B is its own disorder family.

Run::  conda activate cyto7 && python scripts/fig9_predictions.py --parts AC --n-spin 1000
       python scripts/fig9_predictions.py --parts B      # needs enigmatoolbox
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cyto7_surface_io import REPO_ROOT
from external_validation import _bh, _wb_on_path
from receptor_type_connectivity import (
    RECEPTORS, CACHE, SF_DIR, TYPE_SHORT, TYPE_NAMES, VER, SEED,
    load_labels, concat, load_receptor, compute_nulls, rho_and_spin, per_type_medians,
    _fig_dims,
)

EXT_DIR = cfg.results_dir("tables") / "external"
CROSSED = cfg.results_dir("tables") / "crossed"
DATASET = "Validation210"


def eval_map(dh, labels_c, valid_all, valid_excl, nulls):
    rho, p, null = rho_and_spin(concat(dh), labels_c, valid_all, nulls)
    rho_x, p_x, _ = rho_and_spin(concat(dh), labels_c, valid_excl, nulls)
    return dict(spearman_rho=rho, spin_p=p, spearman_rho_allo_excl=rho_x,
                spin_p_allo_excl=p_x, _null=null)


# =========================================================================== #
# Part A — receptor diversity (Shannon entropy)
# =========================================================================== #
def compute_diversity(labels):
    """Per-vertex Shannon entropy H over the 19 min-max-normalised receptor maps."""
    valid = {h: labels[h] > 0 for h in ("L", "R")}
    # min-max each map across pooled labelled cortex -> [0,1]
    norm = {}
    for r in RECEPTORS:
        m = load_receptor(r["key"])
        pooled = np.concatenate([m[h][valid[h]] for h in ("L", "R")])
        lo, hi = float(pooled.min()), float(pooled.max())
        rng = hi - lo if hi > lo else 1.0
        norm[r["receptor"]] = {h: np.clip((m[h] - lo) / rng, 0, 1) for h in ("L", "R")}
    H, effN, CV = {}, {}, {}
    for h in ("L", "R"):
        P = np.stack([norm[nm][h] for nm in [r["receptor"] for r in RECEPTORS]])  # (19,N)
        s = P.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(s > 0, P / s, 0.0)
            ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=0)
            # CV over the 19 normalised (nonneg) values per vertex (defined mean>0)
            mu = P.mean(axis=0); sd = P.std(axis=0)
            cv = np.where(mu > 0, sd / mu, np.nan)
        ent[~valid[h]] = np.nan
        cv[~valid[h]] = np.nan
        H[h] = ent
        effN[h] = np.exp(ent)
        CV[h] = cv
    return H, effN, CV


def part_a(nulls, labels, labels_c, valid_all, valid_excl):
    print("\n== Part A — receptor diversity (Shannon entropy) ==")
    H, effN, CV = compute_diversity(labels)
    cv_c = concat(CV)
    stat_H = eval_map(H, labels_c, valid_all, valid_excl, nulls)
    stat_CV = eval_map(CV, labels_c, valid_all & np.isfinite(cv_c),
                       valid_excl & np.isfinite(cv_c), nulls)
    medH = per_type_medians(H, labels)
    medN = per_type_medians(effN, labels)
    medCV = per_type_medians(CV, labels)

    rows = [
        dict(metric="shannon_entropy_H", primary=True,
             spearman_rho=stat_H["spearman_rho"], spin_p=stat_H["spin_p"],
             spearman_rho_allo_excl=stat_H["spearman_rho_allo_excl"],
             spin_p_allo_excl=stat_H["spin_p_allo_excl"], n=int(valid_all.sum()),
             **{f"median_{TYPE_SHORT[c-1]}": medH[c-1] for c in range(1, 8)},
             **{f"effN_{TYPE_SHORT[c-1]}": medN[c-1] for c in range(1, 8)}),
        dict(metric="cv_robustness", primary=False,
             spearman_rho=stat_CV["spearman_rho"], spin_p=stat_CV["spin_p"],
             spearman_rho_allo_excl=stat_CV["spearman_rho_allo_excl"],
             spin_p_allo_excl=stat_CV["spin_p_allo_excl"], n=int(valid_all.sum()),
             **{f"median_{TYPE_SHORT[c-1]}": medCV[c-1] for c in range(1, 8)},
             **{f"effN_{TYPE_SHORT[c-1]}": np.nan for c in range(1, 8)}),
    ]
    df = pd.DataFrame(rows)
    SF_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SF_DIR / "receptor_diversity.csv", index=False)
    print("  wrote", SF_DIR / "receptor_diversity.csv")
    print(f"  H vs type: rho={stat_H['spearman_rho']:+.3f}, spin p={stat_H['spin_p']:.3f} "
          f"(allo-excl rho={stat_H['spearman_rho_allo_excl']:+.3f}); "
          f"effN allo={medN[0]:.1f} -> konio={medN[6]:.1f}")
    render_diversity_panel(H, labels, stat_H, medH)
    return dict(name="receptor_diversity_H", rho=stat_H["spearman_rho"],
                spin_p=stat_H["spin_p"], rho_excl=stat_H["spearman_rho_allo_excl"],
                spin_p_excl=stat_H["spin_p_allo_excl"], df=df)


def render_diversity_panel(H, labels, stat_H, medH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from cyto7_surface_io import surface_path

    w_in, h_in = _fig_dims(190.0, 0.42)
    fig = plt.figure(figsize=(w_in, h_in)); fig.patch.set_facecolor("white")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    d = H["L"].astype(float).copy()
    lab = labels["L"]
    present = [int(t) for t in range(1, 8) if np.any(lab == t)]
    plotting.plot_surf_stat_map(str(surface_path(DATASET, "L", "inflated")), d, hemi="left",
                                view="lateral", axes=ax1, cmap="magma", colorbar=True,
                                symmetric_cbar=False)
    try:
        plotting.plot_surf_contours(str(surface_path(DATASET, "L", "inflated")), lab,
                                    levels=present, axes=ax1, colors=["w"] * len(present))
    except Exception as exc:
        print("  (contours skipped:", repr(exc)[:80], ")")
    ax1.set_title("LH entropy H (magma) + cyto7 borders", fontsize=8, loc="left")

    ax2 = fig.add_subplot(1, 2, 2)
    data = []
    for c in range(1, 8):
        vals = [H[h][(labels[h] == c) & np.isfinite(H[h])] for h in ("L", "R")]
        data.append(np.concatenate(vals))
    ax2.boxplot(data, showfliers=False, widths=0.6)
    ax2.set_xticklabels(TYPE_SHORT, rotation=35, ha="right", fontsize=7)
    ax2.set_ylabel("receptor Shannon entropy H", fontsize=7.5)
    ax2.tick_params(labelsize=6.5)
    ax2.text(0.02, 0.02, f"ρ={stat_H['spearman_rho']:+.2f}, spin p={stat_H['spin_p']:.3f}",
             transform=ax2.transAxes, fontsize=7)
    fig.savefig(SF_DIR / "receptor_diversity.png", dpi=600, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote", SF_DIR / "receptor_diversity.png")


# =========================================================================== #
# Part C — Hill-2010 cortical expansion
# =========================================================================== #
def fetch_hill2010():
    """Fetch hill2010 evoexp + devexp -> 32k fs_LR .npy cache.

    NOTE: in neuromaps these Van Essen expansion maps are provided as a **single
    right-hemisphere** fsLR 164k file (``hemi-R``). We resample that RH map to 32k and
    analyse RH only (we do NOT mirror to LH — that would fabricate data)."""
    descs = {"evoexp": "hill2010_evoexp", "devexp": "hill2010_devexp"}
    need = any(not (CACHE / f"{stem}_fsLR32k_hemi-R.npy").exists() for stem in descs.values())
    if need:
        _wb_on_path()
        import nibabel as nib
        from neuromaps import datasets, transforms
        for desc, stem in descs.items():
            print(f"  fetching hill2010/{desc} (fsLR 164k, RH-only)...")
            src = datasets.fetch_annotation(source="hill2010", desc=desc)
            path = src[0] if isinstance(src, (tuple, list)) else src
            img = nib.load(path)  # loaded GIFTI (not a path str) + explicit hemi=R
            res = transforms.fslr_to_fslr(img, "32k", hemi="R", method="linear")
            r = res[0] if isinstance(res, (tuple, list)) else res
            data = np.asarray(r.agg_data(), float)
            assert data.shape[0] == 32492, data.shape
            np.save(CACHE / f"{stem}_fsLR32k_hemi-R.npy", data)
            print(f"    cached {stem} R (finite={int(np.isfinite(data).sum())})")
    return descs


def fetch_xu2020_evoexp():
    """Xu-2020 evolutionary expansion — BILATERAL fsLR 32k (no resample needed)."""
    stem = "xu2020_evoexp"
    if all((CACHE / f"{stem}_fsLR32k_hemi-{H}.npy").exists() for H in ("L", "R")):
        return stem
    _wb_on_path()
    import nibabel as nib
    from neuromaps import datasets
    print("  fetching xu2020/evoexp (fsLR 32k, bilateral)...")
    src = datasets.fetch_annotation(source="xu2020", desc="evoexp")  # (L,R) or paths
    items = src if isinstance(src, (tuple, list)) else [src]
    for it in items:
        H = "R" if "hemi-R" in str(it) else "L"
        data = np.asarray(nib.load(it).agg_data(), float)
        assert data.shape[0] == 32492, data.shape
        np.save(CACHE / f"{stem}_fsLR32k_hemi-{H}.npy", data)
        print(f"    cached {stem} {H} (finite={int(np.isfinite(data).sum())})")
    return stem


def part_c(nulls, labels, labels_c, valid_all, valid_excl):
    print("\n== Part C — cortical expansion ==")
    results = []

    # (1) BILATERAL evolutionary expansion — xu2020 (fsLR 32k). Preferred (§2.1).
    fetch_xu2020_evoexp()
    xu = {H: np.load(CACHE / f"xu2020_evoexp_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}
    va = valid_all & np.isfinite(concat(xu)); vx = valid_excl & np.isfinite(concat(xu))
    st = eval_map(xu, labels_c, va, vx, nulls)
    results.append(dict(name="xu2020_evoexp", scope="bilateral", rho=st["spearman_rho"],
                        spin_p=st["spin_p"], rho_excl=st["spearman_rho_allo_excl"],
                        spin_p_excl=st["spin_p_allo_excl"], med=per_type_medians(xu, labels), maps=xu))
    print(f"  xu2020 evoexp (bilateral): rho={st['spearman_rho']:+.3f}, spin p={st['spin_p']:.3f} "
          f"(allo-excl {st['spearman_rho_allo_excl']:+.3f})")

    # (2) developmental expansion — hill2010 devexp, RH-only EXPLORATORY (§2.2; no
    #     bilateral developmental map available in fsLR without CIVET resampling).
    fetch_hill2010()
    dev = {"L": np.full(32492, np.nan),
           "R": np.load(CACHE / "hill2010_devexp_fsLR32k_hemi-R.npy")}
    va = valid_all & np.isfinite(concat(dev)); vx = valid_excl & np.isfinite(concat(dev))
    st = eval_map(dev, labels_c, va, vx, nulls)
    results.append(dict(name="hill2010_devexp", scope="RH-only (exploratory)",
                        rho=st["spearman_rho"], spin_p=st["spin_p"],
                        rho_excl=st["spearman_rho_allo_excl"], spin_p_excl=st["spin_p_allo_excl"],
                        med=per_type_medians(dev, labels), maps=dev))
    print(f"  hill2010 devexp (RH-only): rho={st['spearman_rho']:+.3f}, spin p={st['spin_p']:.3f}")
    render_expansion_panel(results, labels)
    return results


def render_expansion_panel(results, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w_in, h_in = _fig_dims(190.0, 0.42)
    fig, axes = plt.subplots(1, 2, figsize=(w_in, h_in), gridspec_kw=dict(wspace=0.28))
    fig.patch.set_facecolor("white")
    for ax, res in zip(axes, results):
        maps = res["maps"]
        data = []
        for c in range(1, 8):
            vals = [maps[h][(labels[h] == c) & np.isfinite(maps[h])] for h in ("L", "R")]
            data.append(np.concatenate(vals))
        ax.boxplot(data, showfliers=False, widths=0.6)
        ax.set_xticklabels(TYPE_SHORT, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(res["name"] + " expansion", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"{res['name']} [{res['scope']}]  "
                     f"(ρ={res['rho']:+.2f}, spin p={res['spin_p']:.3f})", fontsize=7.5)
    fig.savefig(SF_DIR / "cortical_expansion.png", dpi=600, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote", SF_DIR / "cortical_expansion.png")


def finalize_fig9_family(members):
    """BH-FDR across the pooled Fig-9 family (H + xu2020 evoexp + hill2010 devexp)."""
    p = np.array([m["spin_p"] for m in members])
    q = _bh(p)
    rows = []
    for m, qq in zip(members, q):
        rows.append(dict(metric=m["name"], spearman_rho=m["rho"], spin_p=m["spin_p"],
                         fdr_q_fig9=float(qq), spearman_rho_allo_excl=m["rho_excl"],
                         spin_p_allo_excl=m["spin_p_excl"]))
    df = pd.DataFrame(rows)
    df.to_csv(SF_DIR / "fig9_predictions_summary.csv", index=False)
    print("\n  Fig-9 family (BH-FDR across", len(members), "metrics):")
    print(df.to_string(index=False))
    print("  wrote", SF_DIR / "fig9_predictions_summary.csv")
    return df


# =========================================================================== #
# Part B — ENIGMA disease vulnerability
# =========================================================================== #
# disorder -> preferred CortThick case-vs-controls table key substring
DISORDERS = {
    "22q": "CortThick_case_vs_controls",
    "adhd": "CortThick_case_vs_controls_allages",
    "asd": "CortThick_case_vs_controls_meta_analysis",
    "bipolar": "CortThick_case_vs_controls_adult",
    "depression": "CortThick_case_vs_controls_adult",
    "epilepsy": "CortThick_case_vs_controls_allepilepsy",
    "ocd": "CortThick_case_vs_controls_adult",
    "schizophrenia": "CortThick_case_vs_controls",
}


def _crank():
    """Continuous per-Desikan-parcel cyto7 type score c_rank = sum_t t*fraction_t."""
    comp = pd.read_csv(CROSSED / "desikan_x_cyto7_composition.csv")
    pct = ["Allo_pct", "Agr_pct", "Dys_pct", "EulI_pct", "EulII_pct", "EulIII_pct", "Kon_pct"]
    frac = comp[pct].values / 100.0
    t = np.arange(1, 8)
    cr = (frac * t).sum(axis=1) / frac.sum(axis=1)
    return {str(p).lower(): v for p, v in zip(comp["anatomical_parcel"], cr)}


def _parcel_name(cortthick_label):
    """'L_bankssts' / 'R_entorhinal_asy_thick.csv' -> 'bankssts' / 'entorhinal'."""
    s = str(cortthick_label).lower()
    for pre in ("l_", "r_"):
        if s.startswith(pre):
            s = s[len(pre):]
    for suf in (".csv", "_asy_thick", "_asy_surf", "_thickavg", "_thick", "_surfavg", "_surf"):
        s = s.replace(suf, "")
    return s.strip("_ ")


def _hemi_of(label):
    s = str(label).lower()
    return "l" if s.startswith("l_") else ("r" if s.startswith("r_") else "?")


_PERM_CACHE = {}


def _aparc_perm_id(n_rot, seed):
    """VTK-free ENIGMA parcel spin: fsa5 sphere coords (nilearn) + aparc centroids
    (nibabel) + rotate_parcellation (numpy). Avoids the broken enigma spin_test surface
    load (vtk PointSet incompatibility). Cached; np.random seeded for reproducibility."""
    key = (n_rot, seed)
    if key in _PERM_CACHE:
        return _PERM_CACHE[key]
    import nibabel as nib
    import enigmatoolbox
    from nilearn import datasets as nd
    from enigmatoolbox.permutation_testing.permutation_testing import (
        centroid_extraction_sphere, rotate_parcellation)
    annot_dir = os.path.join(os.path.dirname(enigmatoolbox.__file__),
                             "permutation_testing", "annot")
    fs = nd.fetch_surf_fsaverage("fsaverage5")
    cL = np.asarray(nib.load(fs["sphere_left"]).darrays[0].data, float)
    cR = np.asarray(nib.load(fs["sphere_right"]).darrays[0].data, float)
    ctr_l = centroid_extraction_sphere(cL, os.path.join(annot_dir, "fsa5_lh_aparc.annot"))
    ctr_r = centroid_extraction_sphere(cR, os.path.join(annot_dir, "fsa5_rh_aparc.annot"))
    np.random.seed(seed)
    perm = rotate_parcellation(ctr_l, ctr_r, nrot=n_rot)
    _PERM_CACHE[key] = (perm, ctr_l.shape[0] + ctr_r.shape[0])
    return _PERM_CACHE[key]


_GRAD_CACHE = {}


def _gradient_parc_fsa5():
    """Per-Desikan-parcel principal functional gradient (margulies2016 fcgradient01),
    keyed by (hemi 'l'/'r', parcel-name), parcellated on fsaverage5 aparc — the same
    parcellation used for the parcel spin. Used as the covariate in the specificity check."""
    if "g" in _GRAD_CACHE:
        return _GRAD_CACHE["g"]
    _wb_on_path()
    import nibabel as nib
    import enigmatoolbox
    from neuromaps import datasets as nmd, transforms
    annot_dir = os.path.join(os.path.dirname(enigmatoolbox.__file__),
                             "permutation_testing", "annot")
    src = nmd.fetch_annotation(source="margulies2016", desc="fcgradient01")  # fsLR (L,R)
    out = {}
    for H, gii, fn in zip(("l", "r"), src, ("fsa5_lh_aparc.annot", "fsa5_rh_aparc.annot")):
        res = transforms.fslr_to_fsaverage(nib.load(gii), "10k", hemi=H.upper(), method="linear")
        g = np.asarray((res[0] if isinstance(res, (tuple, list)) else res).agg_data(), float)
        labels, ctab, names = nib.freesurfer.io.read_annot(os.path.join(annot_dir, fn))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        for i, nm in enumerate(names):
            if nm in ("unknown", "corpuscallosum"):
                continue
            m = (labels == i) & np.isfinite(g)
            if m.any():
                out[(H, nm.lower())] = float(np.mean(g[m]))
    _GRAD_CACHE["g"] = out
    return out


def _partial_spearman(x, y, z):
    """Partial Spearman corr of x,y controlling z (Pearson on ranks)."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else np.nan


def _partial_spin_p(d, cr, grad, perm_id):
    """Spin p for partial ρ(d, type | gradient): rotate the type map (cr) via perm_id,
    recompute the partial each rotation, two-tailed against |observed|."""
    obs = _partial_spearman(d, cr, grad)
    n = perm_id.shape[1]
    idx = perm_id.astype(int)
    null = np.empty(n)
    for rr in range(n):
        null[rr] = _partial_spearman(d, cr[idx[:, rr]], grad)
    return obs, float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def _load_disorder(dis, prefer, load_summary_stats):
    """Return (structures, d_icv) for a disorder. Schizophrenia is read directly from the
    shipped van Erp 2018 CSV (its loader entry is broken in 2.0.3) and SIGN-FLIPPED so that
    negative = thinning, matching the case-control convention of the toolbox-loaded tables
    (verified: |d| pattern matches van Erp 2018 — fusiform/superior-temporal largest)."""
    if dis == "schizophrenia":
        import enigmatoolbox
        p = os.path.join(os.path.dirname(enigmatoolbox.__file__), "datasets",
                         "summary_statistics", "Schizophrenia_case-controls_CortThick.csv")
        tab = pd.read_csv(p, sep=";").dropna(subset=["d_icv"])
        struct = list(tab["Structure"])
        d = -pd.to_numeric(tab["d_icv"], errors="coerce").values  # flip to negative=thinning
        return struct, d, "Schizophrenia_case-controls_CortThick.csv (van Erp 2018, sign-flipped)"
    ss = load_summary_stats(dis)
    key = prefer if prefer in ss else next((k for k in ss if k.startswith("CortThick")), None)
    if key is None:
        return None, None, None
    tab = ss[key]
    return list(tab["Structure"]), tab["d_icv"].astype(float).values, key


def part_b(n_spin):
    print("\n== Part B — ENIGMA disease vulnerability ==")
    try:
        from enigmatoolbox.datasets import load_summary_stats
        from enigmatoolbox.permutation_testing.permutation_testing import perm_sphere_p
    except Exception as exc:
        raise SystemExit(f"STOP: enigmatoolbox unavailable ({exc!r}); Part B not run. "
                         "pip install 'git+https://github.com/MICA-MNI/ENIGMA.git' first.")

    perm_id, n_parc = _aparc_perm_id(min(n_spin, 1000), SEED)
    crank = _crank()
    gradp = _gradient_parc_fsa5()
    DIRN = ("ENIGMA d=case-control (neg d=atrophy); with atrophy, rho(d,type)>0 => more "
            "atrophy at LOWER type = vulnerability falls with type (Fig.9). rho(|d|,7-type)>0 "
            "= larger effects at lower type. partial_rho controls the functional gradient.")
    summary = []
    for dis, prefer in DISORDERS.items():
        struct, d_full, key = _load_disorder(dis, prefer, load_summary_stats)
        if struct is None:
            print(f"  SKIP {dis}: no CortThick table"); continue
        names = [_parcel_name(x) for x in struct]
        hemis = [_hemi_of(x) for x in struct]
        cr_full = np.array([crank.get(nm, np.nan) for nm in names])
        gr_full = np.array([gradp.get((h, nm), np.nan) for h, nm in zip(hemis, names)])
        keep = np.isfinite(d_full) & np.isfinite(cr_full) & np.isfinite(gr_full)
        if keep.sum() < 20:
            print(f"  SKIP {dis}: only {int(keep.sum())} matched parcels"); continue
        if keep.sum() != n_parc:
            print(f"  WARN {dis}: {int(keep.sum())} matched != {n_parc} parcels")
        rho = stats.spearmanr(d_full[keep], cr_full[keep])[0]
        rho_absd = stats.spearmanr(np.abs(d_full[keep]), 7 - cr_full[keep])[0]
        try:
            p_spin = float(np.ravel(perm_sphere_p(d_full, cr_full, perm_id,
                                                  corr_type="spearman"))[0])
        except Exception as exc:
            print(f"  spin failed for {dis}: {repr(exc)[:70]}"); p_spin = float("nan")
        # specificity: partial ρ(d, type | functional gradient) + its parcel spin
        prho, pp = _partial_spin_p(d_full[keep], cr_full[keep], gr_full[keep], perm_id)
        summary.append(dict(disorder=dis, n_parcels=int(keep.sum()), source_key=key,
                            spearman_rho=float(rho), spin_p=p_spin,
                            rho_absd_vs_7minus_type=float(rho_absd),
                            partial_rho_gradient=float(prho), partial_spin_p=float(pp),
                            direction=DIRN))
        print(f"  {dis}: n={int(keep.sum())}, rho={rho:+.3f} (p={p_spin:.3f}), "
              f"partial_rho={prho:+.3f} (p={pp:.3f}), rho(|d|,7-type)={rho_absd:+.3f}")
    if not summary:
        raise SystemExit("STOP: no ENIGMA disorders could be loaded; Part B not run.")
    sdf = pd.DataFrame(summary)
    sdf["fdr_q"] = _bh(sdf["spin_p"].values)                    # 8-disorder family
    sdf["partial_fdr_q"] = _bh(sdf["partial_spin_p"].values)
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(EXT_DIR / "disease_vulnerability.csv", index=False)
    print("  wrote", EXT_DIR / "disease_vulnerability.csv")
    render_disease_forest(sdf)
    return sdf


def render_disease_forest(sdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w_in, h_in = _fig_dims(120.0, 0.8)
    fig, ax = plt.subplots(figsize=(w_in, h_in)); fig.patch.set_facecolor("white")
    s = sdf.sort_values("spearman_rho").reset_index(drop=True)
    y = np.arange(len(s))
    # signed ρ (filled) + partial ρ controlling the gradient (open square)
    ax.plot(s["spearman_rho"], y, "o", ms=4.5, color="#333", label="ρ(d, type)", zorder=3)
    ax.plot(s["partial_rho_gradient"], y, "s", ms=4, mfc="none", mec="#2166ac", mew=1.1,
            label="partial ρ | gradient", zorder=3)
    for yi, q, rho in zip(y, s["fdr_q"], s["spearman_rho"]):
        if np.isfinite(q) and q < 0.05:
            ax.plot(rho, yi, "o", ms=8, mfc="none", mec="#c1272d", mew=1.2, zorder=4)
    for yi, r0, r1 in zip(y, s["spearman_rho"], s["partial_rho_gradient"]):
        ax.plot([r0, r1], [yi, yi], color="0.75", lw=0.8, zorder=1)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(s["disorder"], fontsize=8)
    ax.set_xlabel("Spearman ρ (Cohen's d vs parcel type score)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
    ax.set_title("Disease cortical thinning vs cyto7 type (○ red = FDR q<0.05; "
                 "□ = gradient-partialled)", fontsize=7.5)
    fig.savefig(EXT_DIR / "disease_vulnerability.png", dpi=600, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote", EXT_DIR / "disease_vulnerability.png")


def _fast_spin_p(x, y, perm_id):
    """Vectorised parcel spin-p (spin x): Spearman = Pearson on ranks. Equivalent to a
    one-sided-map perm_sphere_p but fast (no python np.append loop)."""
    from scipy.stats import rankdata
    idx = perm_id.astype(int)
    rx, ry = rankdata(x), rankdata(y)
    obs = np.corrcoef(rx, ry)[0, 1]
    rxp = np.apply_along_axis(rankdata, 0, x[idx])          # (nroi, nrot)
    ryc = ry - ry.mean()
    rxpc = rxp - rxp.mean(axis=0)
    nullc = (ryc @ rxpc) / np.sqrt((ryc @ ryc) * (rxpc**2).sum(axis=0))
    p = (np.sum(np.abs(nullc) >= abs(obs)) + 1) / (nullc.size + 1)
    return float(obs), float(p)


def spin_sanity_check(n_rot=1000, seed=0, n_random=500):
    """Validate the reimplemented parcel spin: correlate the parcel type score with
    n_random random maps; spin-p should be ~uniform (mean≈0.5, ~5% below 0.05)."""
    perm_id, n_parc = _aparc_perm_id(n_rot, seed)
    crank = _crank()
    from enigmatoolbox.datasets import load_summary_stats
    tab = load_summary_stats("bipolar")["CortThick_case_vs_controls_adult"]
    names = [_parcel_name(x) for x in tab["Structure"]]
    cr = np.array([crank.get(nm, np.nan) for nm in names])
    rng = np.random.RandomState(seed)
    ps = np.array([_fast_spin_p(rng.randn(n_parc), cr, perm_id)[1] for _ in range(n_random)])
    frac05 = float((ps < 0.05).mean())
    print(f"  spin sanity: {n_random} random maps vs type score — mean p={ps.mean():.3f}, "
          f"median={np.median(ps):.3f}, frac(p<0.05)={frac05:.3f} "
          f"(valid null ≈ mean 0.5, frac 0.05)")
    return dict(n_random=n_random, mean_p=float(ps.mean()), frac_p_lt_05=frac05)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=1000)
    ap.add_argument("--parts", default="ACB")
    ap.add_argument("--spin-sanity", action="store_true", help="run parcel-spin null check")
    args = ap.parse_args(argv)

    if args.spin_sanity:
        print("== spin sanity check ==")
        spin_sanity_check(min(args.n_spin, 1000), SEED)
        return

    if set("AC") & set(args.parts):
        labels = load_labels()
        labels_c = concat(labels)
        valid_all = labels_c > 0
        valid_excl = labels_c > 1
        nulls = compute_nulls(labels, args.n_spin)
        fam = []
        if "A" in args.parts:
            fam.append(part_a(nulls, labels, labels_c, valid_all, valid_excl))
        if "C" in args.parts:
            fam.extend(part_c(nulls, labels, labels_c, valid_all, valid_excl))
        if fam:
            finalize_fig9_family(fam)
    if "B" in args.parts:
        part_b(args.n_spin)
    print("\nDone.")


if __name__ == "__main__":
    main()
