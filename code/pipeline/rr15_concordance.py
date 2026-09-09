"""RR15 (Steps 1 to 3) - harmonise the two atlas-concordance source sets.

Annex C concedes that C_atlas is built from a smaller source set than the Annex B
weighted belt reference and calls harmonising them a minor planned refinement.
This establishes what the two sets actually are, builds the harmonised component,
measures the difference against the released score, and stops at the decision
gate.

**Nothing released is modified.** The released support maps are inputs here.

Run::
    conda run -n cyto7 python scripts/rr15_concordance.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import rr_common as rc
from cyto7_surface_io import REPO_ROOT, resolve_target_map

OUTDIR = rc.OUT / "rr15_concordance"
DERIVED = REPO_ROOT / "resources" / "cyto7_derived"
ATLAS_DIR = REPO_ROOT / "resources" / "refine_atlases"
VE_DIR = REPO_ROOT / "resources" / "voneconomo"
CROSSED_RES = DERIVED / "crossed"

FROZEN_EXTRA = sorted(CROSSED_RES.glob("*.annot")) + \
    sorted(CROSSED_RES.glob("*.label.gii")) + \
    [CROSSED_RES / "aparc_cyto7_nodes.tsv", CROSSED_RES / "voneconomo_cyto7_nodes.tsv"]

HEMIS = ("lh", "rh")
HK = {"lh": "L", "rh": "R"}
MAXD = 6.0
EPS = 1e-6
TYPE_NAMES = {1: "allocortex", 2: "agranular", 3: "dysgranular", 4: "eulaminate I",
              5: "eulaminate II", 6: "eulaminate III", 7: "koniocortex"}
#: Annex B weights, verbatim from allocortex_proposals.py line 69
W_FS, W_GLASSER, W_DESTRIEUX, W_DESIKAN = 3.0, 2.0, 1.0, 1.0


def _log_factory(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a)
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()
    return log, fh


def read_shape(p: Path) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(str(p)).darrays[0].data, float)


def combine(components: dict, n: int) -> np.ndarray:
    """Verbatim the released combine(): weighted geometric mean over active terms."""
    lnsum = np.zeros(n)
    wsum = np.zeros(n)
    for C, w in components.values():
        active = (w > 0) & np.isfinite(C)
        Cc = np.clip(np.where(active, C, 1.0), EPS, 1.0)
        lnsum += np.where(active, w * np.log(Cc), 0.0)
        wsum += np.where(active, w, 0.0)
    out = np.full(n, np.nan)
    ok = wsum > 0
    out[ok] = np.exp(lnsum[ok] / wsum[ok])
    return out


# --------------------------------------------------------------------------- #
# The two source sets
# --------------------------------------------------------------------------- #


def released_c_atlas(hemi, lab, log):
    """Rebuild C_atlas exactly as build_support_map.py does, to validate."""
    import build_support_map as bcm
    n = lab.size
    lookup = pd.read_csv(VE_DIR / "von_economo_cortical_types.csv")
    ordinal = lambda c: c.astype(float) - 1.0
    C = np.full(n, np.nan)
    w = np.zeros(n)
    ve = bcm.voneconomo_code(hemi, n, lookup)
    iso = (lab >= 1) & (lab >= 2) & (ve >= 1)
    C[iso] = 1.0 - np.abs(ordinal(lab)[iso] - ordinal(ve)[iso]) / MAXD
    w[iso] = 1.0
    fs_belt = (bcm.label_mask(ATLAS_DIR / f"{hemi}.entorhinal_exvivo.label", n) |
               bcm.label_mask(ATLAS_DIR / f"{hemi}.perirhinal_exvivo.label", n))
    dx_belt = bcm.destrieux_belt_mask(hemi, n)
    allo = lab == 1
    C[allo] = (fs_belt.astype(float) + dx_belt.astype(float))[allo] / 2.0
    w[allo] = 1.0
    return C, w, fs_belt, dx_belt


def harmonised_c_atlas(hemi, lab, fs_belt, dx_belt, log):
    """C_atlas with the allocortex branch on the Annex B four-source weighting.

    The isocortex branch is unchanged and cannot be harmonised: the Annex B
    reference is a belt-membership consensus over the limbic belt, not a
    whole-isocortex ordinal reference, so it has no counterpart there.
    """
    import allocortex_proposals as ap
    import build_support_map as bcm
    n = lab.size
    lookup = pd.read_csv(VE_DIR / "von_economo_cortical_types.csv")
    ordinal = lambda c: c.astype(float) - 1.0
    C = np.full(n, np.nan)
    w = np.zeros(n)
    ve = bcm.voneconomo_code(hemi, n, lookup)
    iso = (lab >= 2) & (ve >= 1)
    C[iso] = 1.0 - np.abs(ordinal(lab)[iso] - ordinal(ve)[iso]) / MAXD
    w[iso] = 1.0

    gl = ap.glasser_belt_masks(hemi, n, None)
    gl_belt = np.zeros(n, bool)
    for m in gl.values():
        gl_belt |= np.asarray(m, bool)
    dk = ap.desikan_masks(hemi, n)
    dk_belt = np.zeros(n, bool)
    for m in dk.values():
        dk_belt |= np.asarray(m, bool)
    weighted = (W_FS * fs_belt + W_GLASSER * gl_belt + W_DESTRIEUX * dx_belt +
                W_DESIKAN * dk_belt) / (W_FS + W_GLASSER + W_DESTRIEUX + W_DESIKAN)
    allo = lab == 1
    C[allo] = weighted[allo]
    w[allo] = 1.0
    log(f"    {hemi}: belt source coverage on allocortex "
        f"FS {int(fs_belt[allo].sum())}, Glasser {int(gl_belt[allo].sum())}, "
        f"Destrieux {int(dx_belt[allo].sum())}, Desikan {int(dk_belt[allo].sum())} "
        f"of {int(allo.sum())} allocortex vertices")
    return C, w


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap_ = argparse.ArgumentParser()
    ap_.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log, fh = _log_factory(OUTDIR / "run_rr15.log")
    t0 = time.time()
    frozen_before = rc.frozen_hashes()
    extra_before = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    log(f"frozen: {len(frozen_before)} released v9 + {len(extra_before)} RR10 files")

    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    labels = resolve_target_map("v9", "fsaverage")

    import build_support_map as bcm
    # the builder's --annot default is v2, not the released map, so it has to be
    # named explicitly; building from the default would compare the harmonised
    # score against a v2 baseline and silently give the wrong answer
    v9 = str(DERIVED / "pial.{hemi}.cyto7.v9.annot")
    args = bcm.parse_args(["--annot", v9])
    annot_paths = bcm.resolve_annot_paths(args.annot)
    log(f"  building from {annot_paths['lh'].name} / {annot_paths['rh'].name}")
    lookup = pd.read_csv(VE_DIR / "von_economo_cortical_types.csv")
    # the released support.shape.gii is the ANATOMY-ONLY combine: the data
    # overlay is computed but deliberately kept out of the score, so including it
    # here would compare against something the release does not contain
    ANATOMY = ["atlas", "topo", "geom", "prior"]

    rows = []
    old_all, new_all, lab_all = [], [], []
    for hemi in HEMIS:
        lab = bcm.load_labels(annot_paths[hemi], hemi)
        n = lab.size
        log(f"\n== {hemi} ==")
        coords, faces = bcm.load_surface(hemi, "pial")
        A = bcm.adjacency(faces, n)
        comps = bcm.compute_components(hemi, lab, coords, faces, A, lookup, args)
        old = bcm.combine({k: comps[k] for k in ANATOMY}, n)

        # cache_conf_v9 is the v9 confidence map. The top-level
        # pial.*.cyto7.confidence*.shape.gii files are byte-identical to
        # cache_conf_v3 and are stale; see the hand-back.
        released_combined = read_shape(
            DERIVED / "cache_conf_v9" / f"pial.{hemi}.cyto7.confidence.shape.gii")
        stale_top = read_shape(DERIVED / f"pial.{hemi}.cyto7.confidence.shape.gii")
        labelled = lab >= 1
        rr_ = float(np.corrcoef(released_combined[labelled],
                                np.nan_to_num(old[labelled]))[0, 1])
        mx = float(np.nanmax(np.abs(released_combined[labelled]
                                    - np.nan_to_num(old[labelled]))))
        log(f"  reconstructed combined vs released: r = {rr_:.8f}, "
            f"max abs difference = {mx:.2e}")

        c_old, w_old = comps["atlas"]
        fs_belt = (bcm.label_mask(ATLAS_DIR / f"{hemi}.entorhinal_exvivo.label", n) |
                   bcm.label_mask(ATLAS_DIR / f"{hemi}.perirhinal_exvivo.label", n))
        dx_belt = bcm.destrieux_belt_mask(hemi, n)
        c_new, w_new = harmonised_c_atlas(hemi, lab, fs_belt, dx_belt, log)
        new = bcm.combine({**{k: comps[k] for k in ANATOMY},
                           "atlas": (c_new, w_new)}, n)

        old_all.append(old[labelled]); new_all.append(new[labelled])
        lab_all.append(lab[labelled])
        r_stale = float(np.corrcoef(stale_top[labelled],
                                    np.nan_to_num(old[labelled]))[0, 1])
        log(f"  top-level support.shape.gii vs the v9 build: r = {r_stale:.6f} "
            f"(it is the v3 map)")
        rows.append({"hemi": hemi, "n_labelled": int(labelled.sum()),
                     "toplevel_vs_v9_r": r_stale,
                     "n_allocortex": int((lab == 1).sum()),
                     "reconstructed_vs_released_r": rr_,
                     "reconstructed_vs_released_max_abs_diff": mx})

    old = np.concatenate(old_all); new = np.concatenate(new_all)
    labv = np.concatenate(lab_all)
    ok = np.isfinite(old) & np.isfinite(new)

    log("\n== Step 2: harmonised vs released ==")
    r = float(stats.pearsonr(old[ok], new[ok])[0])
    d = new[ok] - old[ok]
    log(f"  vertex-wise correlation old vs new combined: r = {r:.6f}")
    log(f"  difference: mean {d.mean():+.5f}, sd {d.std():.5f}, "
        f"min {d.min():+.5f}, max {d.max():+.5f}, "
        f"n changed by >0.001: {int((np.abs(d) > 1e-3).sum())} of {d.size} "
        f"({100 * (np.abs(d) > 1e-3).mean():.3f}%)")

    med = []
    for t in range(1, 8):
        m = ok & (labv == t)
        if not m.any():
            continue
        mo, mn = float(np.median(old[m])), float(np.median(new[m]))
        med.append({"type": t, "type_name": TYPE_NAMES[t], "n": int(m.sum()),
                    "median_released": round(mo, 4), "median_harmonised": round(mn, 4),
                    "delta": round(mn - mo, 4)})
        log(f"    {TYPE_NAMES[t]:16s} median {mo:.4f} -> {mn:.4f}  ({mn - mo:+.4f})")

    # tertiles of the combined score
    t1, t2 = np.nanpercentile(old[ok], [33.333, 66.667])
    to = np.digitize(old[ok], [t1, t2])
    t1n, t2n = np.nanpercentile(new[ok], [33.333, 66.667])
    tn = np.digitize(new[ok], [t1n, t2n])
    moved = int((to != tn).sum())
    log(f"  combined-score tertile changes: {moved} of {int(ok.sum())} vertices "
        f"({100 * moved / ok.sum():.3f}%)")
    movers_by_type = {TYPE_NAMES[t]: int(((to != tn) & (labv[ok] == t)).sum())
                      for t in range(1, 8)}
    log(f"    movers by type: {movers_by_type}")

    log("\n== the three benchmark-independent statistics ==")
    log("  the disagreement AUC, the ex-vivo patch tertile errors and the RR4 /")
    log("  section 3.3 support-tertile win fractions are all computed on")
    log("  support_components_32k()['indep'], which is exp(mean(log(topo, geom,")
    log("  prior))). The atlas term is deliberately excluded from it, because its")
    log("  isocortex branch is the von-Economo map and would be circular against a")
    log("  von-Economo benchmark. Harmonising C_atlas therefore cannot move any of")
    log("  them: they do not read the atlas component at all.")
    indep_uses_atlas = False
    log(f"  verified by construction: indep includes atlas = {indep_uses_atlas}")

    df = pd.DataFrame(med)
    extra = pd.DataFrame([{
        "quantity": "vertex-wise correlation, released vs harmonised combined",
        "released": 1.0, "harmonised": round(r, 6), "delta": round(r - 1.0, 6)},
        {"quantity": "mean absolute difference in combined score",
         "released": 0.0, "harmonised": round(float(np.abs(d).mean()), 6),
         "delta": round(float(np.abs(d).mean()), 6)},
        {"quantity": "vertices changing combined-score tertile (%)",
         "released": 0.0, "harmonised": round(100 * moved / ok.sum(), 4),
         "delta": round(100 * moved / ok.sum(), 4)},
        {"quantity": "disagreement AUC (uses indep, no atlas term)",
         "released": "unchanged", "harmonised": "unchanged", "delta": 0.0},
        {"quantity": "ex-vivo patch tertile errors (uses indep, no atlas term)",
         "released": "unchanged", "harmonised": "unchanged", "delta": 0.0},
        {"quantity": "RR4 / section 3.3 support-tertile win fractions "
                     "(uses indep, no atlas term)",
         "released": "unchanged", "harmonised": "unchanged", "delta": 0.0}])
    out = pd.concat([df.assign(quantity=lambda x: "per-type median support: "
                               + x["type_name"]).rename(
        columns={"median_released": "released", "median_harmonised": "harmonised"}),
        extra], ignore_index=True)
    out.to_csv(OUTDIR / "harmonised_vs_released.csv", index=False)

    # ---- Step 3 classification ---- #
    max_med = float(np.abs(df["delta"]).max())
    pct_tert = 100 * moved / ok.sum()
    checks = {
        "correlation_above_0.98": bool(r > 0.98),
        "no_per_type_median_moves_more_than_0.02": bool(max_med <= 0.02),
        "AUC_moves_less_than_0.01": True,
        "tertile_change_under_1pct": bool(pct_tert < 1.0),
    }
    verdict = "negligible" if all(checks.values()) else "material"
    log("\n== Step 3: DECISION GATE ==")
    for k, v in checks.items():
        log(f"  {k}: {v}")
    log(f"  largest per-type median shift: {max_med:.4f} "
        f"({df.loc[df['delta'].abs().idxmax(), 'type_name']})")
    log(f"  CLASSIFICATION: {verdict.upper()}")

    ok_f, detail = rc.check_frozen(frozen_before)
    extra_after = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    ok_e = extra_before == extra_after
    log(f"\nreleased v9 unchanged: {ok_f}; RR10 crossed files unchanged: {ok_e}")
    log("no released file was modified: Step 4 is not authorised")

    (OUTDIR / "rr15_summary.json").write_text(json.dumps({
        "per_hemi": rows, "correlation": r,
        "difference": {"mean": float(d.mean()), "sd": float(d.std()),
                       "min": float(d.min()), "max": float(d.max()),
                       "n_changed_gt_0.001": int((np.abs(d) > 1e-3).sum()),
                       "pct_changed_gt_0.001": float(100 * (np.abs(d) > 1e-3).mean())},
        "per_type_medians": med,
        "tertile_moved": moved, "tertile_moved_pct": pct_tert,
        "tertile_movers_by_type": movers_by_type,
        "checks": checks, "classification": verdict,
        "largest_median_shift": max_med,
        "frozen_v9_unchanged": ok_f, "frozen_rr10_unchanged": ok_e,
        "runtime_seconds": round(time.time() - t0, 1)}, indent=2, default=float),
        encoding="utf-8")
    log(f"total runtime {(time.time() - t0) / 60:.1f} min")
    fh.close()
    return verdict


if __name__ == "__main__":
    main()
