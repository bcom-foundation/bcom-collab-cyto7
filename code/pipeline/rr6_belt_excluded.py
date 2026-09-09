"""RR6 - primary results with the Glasser-adjudicated limbic belt excluded.

Recomputes the four primary imaging associations (T1w/T2w myelin, cortical
thickness, principal functional gradient, AHBA gene PC1) and the MEG primary
(int_area, subject-level) on the vertices that remain after removing the limbic
belt, i.e. allocortex + agranular + dysgranular. Two exclusion sets:

  Set A  the whole belt.
  Set B  belt vertices where the Glasser term was decisive in the
         reliability-weighted belt adjudication, replayed at vertex level from
         ``allocortex_proposals.build_consensus`` (weights FS ex-vivo 3,
         Glasser 2, Destrieux 1, Desikan 1; trusted call = FS or Glasser).

Each retained set gets its own spin null: the released rotations of the v9 type
map, restricted to the retained vertices. BH within the pre-existing family.
A contiguous-window control quantifies how much of any attenuation is plain
range restriction rather than circularity.

Run::
    conda run -n cyto7 python scripts/rr6_belt_excluded.py --n-spin 1000
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import rr_common as rc
from cyto7_surface_io import REPO_ROOT, resolve_target_map
from external_validation import _bh

OUTDIR = rc.OUT / "rr6_belt"
CACHEDIR = rc.OUT / "_cache"
NM_CACHE = cfg.data_dir() / "neuromaps_cache"
BELT_CODES = (1, 2, 3)          # allocortex, agranular, dysgranular
DATASET = "Validation210"

PUBLISHED = {"myelin": 0.5894, "thickness": -0.5018, "gradient": -0.5761, "genepc1": 0.6657}
PUBLISHED_ALLO_EXCL = {"myelin": 0.6039, "thickness": -0.5014, "gradient": -0.5822}
SPEC_BELT_164K = {"total_labelled": 304972, "belt": 49034,
                  "allocortex": 7340, "agranular": 15384, "dysgranular": 26310}


# --------------------------------------------------------------------------- #
# Belt definition and the Glasser-decisive subset (Set B)
# --------------------------------------------------------------------------- #


def belt_counts_164k() -> dict:
    lab = resolve_target_map("v9", "fsaverage")
    out = {"per_hemi": {}}
    tot = belt = 0
    per_type = {c: 0 for c in BELT_CODES}
    for H in ("L", "R"):
        l = np.asarray(lab[H], int)
        out["per_hemi"][H] = {"labelled": int((l > 0).sum()),
                              **{rc.TYPE_NAMES[c - 1]: int((l == c).sum()) for c in BELT_CODES}}
        tot += int((l > 0).sum())
        belt += int(np.isin(l, BELT_CODES).sum())
        for c in BELT_CODES:
            per_type[c] += int((l == c).sum())
    out["total_labelled"] = tot
    out["belt"] = belt
    out["belt_fraction"] = belt / tot
    out["per_type"] = {rc.TYPE_NAMES[c - 1]: per_type[c] for c in BELT_CODES}
    return out


def glasser_decisive_164k(log=print) -> dict[str, np.ndarray]:
    """Per-hemisphere 164k mask: would the belt adjudication change without Glasser?

    Replays ``allocortex_proposals.build_consensus``. Two documented decision
    rules depend on the source set, so a vertex is Glasser-decisive when either
    flips: (a) the trusted allocortex-versus-isocortex call (FS or Glasser), and
    (b) the reliability-weighted plurality (score >= 0.5).
    """
    import allocortex_proposals as ap
    cpath = CACHEDIR / "rr6_glasser_decisive_164k.npz"
    if cpath.exists():
        z = np.load(cpath)
        return {"L": z["L"], "R": z["R"], "trusted_L": z["trusted_L"], "trusted_R": z["trusted_R"],
                "weighted_L": z["weighted_L"], "weighted_R": z["weighted_R"]}
    lab = resolve_target_map("v9", "fsaverage")
    out = {}
    for H, hemi in (("L", "lh"), ("R", "rh")):
        n = np.asarray(lab[H]).size
        con = ap.build_consensus(hemi, n, None)
        s = con["sources"]
        fs, gl, dx, dk = s["FS_exvivo"], s["Glasser"], s["Destrieux"], s["Desikan"]
        # (a) trusted call
        trusted_with = fs | gl
        trusted_without = fs
        flip_trusted = trusted_with != trusted_without
        # (b) weighted plurality at the documented threshold
        with_score = (ap.W_FS * fs + ap.W_GLASSER * gl + ap.W_DESTRIEUX * dx + ap.W_DESIKAN * dk) / \
                     (ap.W_FS + ap.W_GLASSER + ap.W_DESTRIEUX + ap.W_DESIKAN)
        without_score = (ap.W_FS * fs + ap.W_DESTRIEUX * dx + ap.W_DESIKAN * dk) / \
                        (ap.W_FS + ap.W_DESTRIEUX + ap.W_DESIKAN)
        flip_weighted = (with_score >= 0.5) != (without_score >= 0.5)
        out[H] = flip_trusted | flip_weighted
        out[f"trusted_{H}"] = flip_trusted
        out[f"weighted_{H}"] = flip_weighted
        log(f"    {hemi}: Glasser-decisive {int(out[H].sum())} vtx "
            f"(trusted flip {int(flip_trusted.sum())}, weighted flip {int(flip_weighted.sum())})")
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cpath, **out)
    return out


def to_32k(mask164: dict[str, np.ndarray], tag: str) -> dict[str, np.ndarray]:
    import reviewer_response as rr
    m = rr.resample_to_32k({H: mask164[H].astype(np.float32) for H in ("L", "R")}, "nearest", tag)
    return {H: m[H].astype(bool) for H in ("L", "R")}


# --------------------------------------------------------------------------- #
# Vertex-level evaluation on a retained set
# --------------------------------------------------------------------------- #


def spearman_with_spin(vals: np.ndarray, rank: np.ndarray, keep: np.ndarray,
                       nulls: np.ndarray) -> tuple[float, float, int]:
    rho = float(stats.spearmanr(vals[keep], rank[keep])[0])
    n_spin = nulls.shape[1]
    nr = np.empty(n_spin)
    for i in range(n_spin):
        spun = nulls[:, i][keep]
        ok = np.isfinite(spun)
        nr[i] = stats.spearmanr(vals[keep][ok], spun[ok])[0]
    p = float((np.sum(np.abs(nr) >= abs(rho)) + 1) / (n_spin + 1))
    return rho, p, int(keep.sum())


def main(argv=None):
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--n-spin", type=int, default=1000)
    ap_.add_argument("--skip-meg", action="store_true",
                     help="skip the MEG primary (needs the RR5 per-subject metric-stack cache)")
    ap_.add_argument("--meg-only", action="store_true")
    args = ap_.parse_args(argv)
    if args.meg_only:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        meg = meg_belt_excluded(args.n_spin)
        pd.DataFrame(meg).to_csv(OUTDIR / "meg_belt_excluded.csv", index=False)
        print(pd.DataFrame(meg).to_string(index=False))
        return None
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frozen_before = rc.frozen_hashes()

    print("== belt counts on the released annot (164k fsaverage) ==")
    counts = belt_counts_164k()
    print(json.dumps(counts, indent=2))
    match = {k: (counts.get(k) if k != "per_type" else None) == v
             for k, v in SPEC_BELT_164K.items() if k in ("total_labelled", "belt")}
    print("  spec counts confirmed:", match, "per-type:", counts["per_type"])

    print("== Set B: Glasser-decisive belt vertices ==")
    dec = glasser_decisive_164k()
    lab164 = resolve_target_map("v9", "fsaverage")
    setB_164 = {H: dec[H] & np.isin(np.asarray(lab164[H], int), BELT_CODES) for H in ("L", "R")}
    nB164 = int(sum(m.sum() for m in setB_164.values()))
    print(f"  Set B (164k): {nB164} vertices "
          f"({100*nB164/counts['total_labelled']:.2f}% of labelled cortex, "
          f"{100*nB164/counts['belt']:.1f}% of the belt)")

    lab32 = rc.labels_32k()
    setB_32 = to_32k(setB_164, "rr6_glasser_decisive")
    labc = rc.cat(lab32, int).astype(int)
    beltA = np.isin(labc, BELT_CODES)
    beltB = rc.cat(setB_32, bool).astype(bool) & beltA
    print(f"  32k: labelled {int((labc>0).sum())}, Set A belt {int(beltA.sum())}, "
          f"Set B {int(beltB.sum())}")

    nulls = rc.load_nulls(args.n_spin)
    rank = labc.astype(float)

    # ---------------- structure-function family (9 features) ---------------- #
    from summarise_functional_features import FEATURES, build_validity_mask, load_all_features
    feats = load_all_features(DATASET)
    base = build_validity_mask(lab32, feats)
    basec = rc.cat(base, bool).astype(bool)
    fv = {f.key: rc.cat(feats[f.key]) for f in FEATURES}

    sets = {
        "published (all labelled)": basec,
        "allocortex excluded": basec & (labc > 1),
        "belt excluded (Set A)": basec & ~beltA,
        "belt excluded (Set B)": basec & ~beltB,
    }
    rows = []
    for sname, keep in sets.items():
        ps, res = [], []
        for f in FEATURES:
            rho, p, n = spearman_with_spin(fv[f.key], rank, keep, nulls)
            ps.append(p); res.append((f.key, f.label, rho, p, n))
        q = _bh(np.array(ps))
        for (key, label, rho, p, n), qk in zip(res, q):
            rows.append({"measure": label, "key": key, "family": "structure-function (9)",
                         "set": sname, "n_retained": n, "rho": round(rho, 4),
                         "spin_p": round(p, 4), "q": round(float(qk), 4)})
        print(f"  [{sname}] myelin {res[0][2]:+.4f} thickness {res[1][2]:+.4f} "
              f"gradient {res[8][2]:+.4f} (n={res[0][4]})")

    # ---------------- external family (3 members) ---------------- #
    ext = {"genepc1": np.concatenate([np.load(NM_CACHE / f"abagen_genepc1_fsLR32k_hemi-{H}.npy")
                                      for H in ("L", "R")]).astype(float),
           "receptorpc1": np.concatenate([np.load(NM_CACHE / f"hansen_receptorpc1_fsLR32k_hemi-{H}.npy")
                                          for H in ("L", "R")]).astype(float)}
    prof = np.concatenate([np.load(NM_CACHE / f"bigbrain_profiles_fsLR32k_hemi-{H}.npy")
                           for H in ("L", "R")], axis=1).astype(float)
    ext["bigbrain_profile_skewness"] = stats.skew(prof, axis=0, bias=False)
    ext_label = {"genepc1": "AHBA gene expression PC1",
                 "receptorpc1": "PET receptor PC1 (Hansen-2022 collection)",
                 "bigbrain_profile_skewness": "BigBrain profile skewness (pre-registered, histology)"}
    ext_sets = {
        "published (all labelled)": (labc > 0),
        "allocortex excluded": (labc > 1),
        "belt excluded (Set A)": (labc > 0) & ~beltA,
        "belt excluded (Set B)": (labc > 0) & ~beltB,
    }
    for sname, base_keep in ext_sets.items():
        ps, res = [], []
        for k, v in ext.items():
            keep = base_keep & np.isfinite(v)
            rho, p, n = spearman_with_spin(v, rank, keep, nulls)
            ps.append(p); res.append((k, rho, p, n))
        q = _bh(np.array(ps))
        for (k, rho, p, n), qk in zip(res, q):
            rows.append({"measure": ext_label[k], "key": k, "family": "external reference (3)",
                         "set": sname, "n_retained": n, "rho": round(rho, 4),
                         "spin_p": round(p, 4), "q": round(float(qk), 4)})
        print(f"  [{sname}] genepc1 {res[0][1]:+.4f} (n={res[0][3]})")

    df = pd.DataFrame(rows)
    pub = {**PUBLISHED, "receptorpc1": -0.3292, "bigbrain_profile_skewness": -0.2091}
    df["published_rho"] = df["key"].map(pub)
    df["delta_rho"] = (df["rho"] - df["published_rho"]).round(4)
    df.to_csv(OUTDIR / "belt_excluded.csv", index=False)

    # ---------------- range-restriction control ---------------- #
    win_rows = []
    for lo in (1, 2, 3, 4):
        wnd = list(range(lo, lo + 4))
        for key, label, values, bmask in (
                ("myelin", "T1w/T2w myelin", fv["myelin"], basec),
                ("thickness", "Cortical thickness", fv["thickness"], basec),
                ("gradient", "Principal functional gradient", fv["gradient"], basec),
                ("genepc1", "AHBA gene expression PC1", ext["genepc1"], (labc > 0) & np.isfinite(ext["genepc1"]))):
            keep = bmask & np.isin(labc, wnd)
            rho = float(stats.spearmanr(values[keep], rank[keep])[0])
            win_rows.append({"measure": label, "key": key,
                             "window": f"types {lo}-{lo + 3}", "n": int(keep.sum()),
                             "rho": round(rho, 4),
                             "published_rho": pub[key],
                             "delta_vs_published": round(rho - pub[key], 4)})
    win = pd.DataFrame(win_rows)
    win.to_csv(OUTDIR / "range_restriction_windows.csv", index=False)
    print("\ncontiguous four-type windows (range-restriction control):")
    print(win.to_string(index=False))

    # ---------------- MEG primary on the retained set ---------------- #
    if args.skip_meg:
        print("\nMEG primary skipped (--skip-meg); run with --meg-only once the RR5 "
              "per-subject metric-stack cache exists.")
    else:
        meg = meg_belt_excluded(args.n_spin)
        pd.DataFrame(meg).to_csv(OUTDIR / "meg_belt_excluded.csv", index=False)
        print("\nMEG int_area:")
        print(pd.DataFrame(meg).to_string(index=False))

    ok, detail = rc.check_frozen(frozen_before)
    (OUTDIR / "belt_definition.json").write_text(json.dumps({
        "counts_164k": counts, "spec_counts": SPEC_BELT_164K,
        "setB_164k": nB164, "setB_32k": int(beltB.sum()),
        "setA_32k": int(beltA.sum()), "labelled_32k": int((labc > 0).sum()),
        "setB_components_164k": {
            "trusted_flip": int(sum(int((dec[f'trusted_{H}'] &
                                         np.isin(np.asarray(lab164[H], int), BELT_CODES)).sum())
                                    for H in ("L", "R"))),
            "weighted_flip": int(sum(int((dec[f'weighted_{H}'] &
                                          np.isin(np.asarray(lab164[H], int), BELT_CODES)).sum())
                                     for H in ("L", "R")))},
        "frozen_unchanged": ok, "frozen": detail, "n_spin": args.n_spin},
        indent=2), encoding="utf-8")
    print("frozen files unchanged:", ok)
    return df


def meg_belt_excluded(n_spin: int) -> list[dict]:
    """int_area (subject-level, hierarchical spin) on belt-excluded 4k vertices."""
    import meg_subjectlevel as MB
    from rr5_meg_controls import build_stacks, prepare
    stacks_raw, subs = build_stacks()
    type4k, nulls4k = MB.build_4k_alignment(n_spin, print)
    stacks, valid = prepare(stacks_raw, type4k)
    out = []
    for sname, keep in (("published (all labelled)", valid),
                        ("allocortex excluded", valid & (type4k > 1)),
                        ("belt excluded (Set A)", valid & (type4k > 3))):
        r = MB.subject_level_spin(stacks["int_area"], type4k, nulls4k, keep, -1)
        r.pop("rho_subj")
        out.append({"measure": "MEG intrinsic timescale (int_area)", "set": sname,
                    "n_retained_4k": int(keep.sum()), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                         for k, v in r.items()}})
        print(f"  [{sname}] int_area rho={r['group_mean_rho']:+.4f} p={r['spin_p']:.4f} "
              f"n4k={int(keep.sum())}")
    return out


if __name__ == "__main__":
    main()
