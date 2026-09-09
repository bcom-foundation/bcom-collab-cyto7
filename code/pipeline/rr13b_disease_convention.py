"""RR13b - put the disease-vulnerability family on the section 2.6 convention.

`figures/v9/external/disease_vulnerability.csv` uses two conventions in adjacent
columns. `spin_p` is ENIGMA `perm_sphere_p`, the raw one-tailed tail fraction with
no plus-one that RR13 found in the layer-marker family. `partial_spin_p`, computed
a few lines later in the same script by `_partial_spin_p`, already uses the
plus-one corrected two-tailed form of section 2.6.

This recomputes `spin_p` and `fdr_q` under the section 2.6 convention, using the
same rotations and seed, so the whole paper states one convention. The null is
built exactly as `_partial_spin_p` builds its own: rotate the per-parcel type score
through `perm_id` and recompute the correlation each rotation, then

    p = (1 + #{|rho_null| >= |rho_obs|}) / (n_rot + 1)

Nothing released is regenerated and nothing published is changed in place.

Run::
    conda run -n cyto7 python scripts/rr13b_disease_convention.py
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
from cyto7_surface_io import REPO_ROOT

OUTDIR = rc.OUT / "rr13b_disease"
PUBLISHED = REPO_ROOT / "figures" / "v9" / "external" / "disease_vulnerability.csv"
CROSSED = REPO_ROOT / "resources" / "cyto7_derived" / "crossed"

FROZEN_EXTRA = sorted(CROSSED.glob("*.annot")) + sorted(CROSSED.glob("*.label.gii")) + \
    [CROSSED / "aparc_cyto7_nodes.tsv", CROSSED / "voneconomo_cyto7_nodes.tsv"]

N_ROT = 1000          # fig9_predictions calls _aparc_perm_id(min(n_spin, 1000), SEED)
SEED = 0              # receptor_type_connectivity.SEED, as RR13 documented


def _log_factory(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")

    def log(*a):
        m = " ".join(str(x) for x in a)
        print(m, flush=True)
        fh.write(m + "\n")
        fh.flush()
    return log, fh


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, matching external_validation._bh."""
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return np.minimum(q, 1.0)


def stated_convention_p(d: np.ndarray, cr: np.ndarray, perm_id: np.ndarray) -> tuple:
    """Section 2.6: plus-one corrected two-tailed tail over the parcel rotations.

    The null rotates the type score, which is the same construction
    ``_partial_spin_p`` already uses in this family, so the recomputed column and
    the existing partial column become the same kind of quantity.
    """
    obs = float(stats.spearmanr(d, cr)[0])
    idx = perm_id.astype(int)
    n = idx.shape[1]
    null = np.empty(n)
    for r in range(n):
        null[r] = stats.spearmanr(d, cr[idx[:, r]])[0]
    k = int(np.sum(np.abs(null) >= abs(obs)))
    return obs, float((k + 1) / (n + 1)), k, null


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rot", type=int, default=N_ROT)
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log, fh = _log_factory(OUTDIR / "run_rr13b.log")
    t0 = time.time()

    frozen_before = rc.frozen_hashes()
    extra_before = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    log(f"frozen: {len(frozen_before)} released v9 + {len(extra_before)} RR10 files")

    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import fig9_predictions as f9
    from enigmatoolbox.datasets import load_summary_stats
    from enigmatoolbox.permutation_testing.permutation_testing import perm_sphere_p

    pub = pd.read_csv(PUBLISHED)
    perm_id, n_parc = f9._aparc_perm_id(min(args.n_rot, 1000), SEED)
    log(f"  parcel rotations: {perm_id.shape[1]}, seed {SEED}, {n_parc} parcels "
        f"(fsa5 aparc centroids via rotate_parcellation)")
    crank = f9._crank()

    rows = []
    for _, prow in pub.iterrows():
        dis = prow["disorder"]
        struct, d_full, key = f9._load_disorder(dis, prow["source_key"], load_summary_stats)
        if struct is None:
            log(f"  SKIP {dis}: no CortThick table")
            continue
        names = [f9._parcel_name(x) for x in struct]
        cr_full = np.array([crank.get(nm, np.nan) for nm in names], float)
        keep = np.isfinite(d_full) & np.isfinite(cr_full)
        d, cr = d_full[keep], cr_full[keep]
        if keep.sum() != perm_id.shape[0]:
            rc.log_decision("RR13b", f"{dis} parcel count differs from the rotation basis",
                            f"used the {int(keep.sum())} finite parcels",
                            "skip the disorder", "the rotation index spans all 68 parcels",
                            "yes", "yes")
            log(f"  WARN {dis}: {int(keep.sum())} finite parcels vs {perm_id.shape[0]}")

        obs, p_new, k, _null = stated_convention_p(d, cr, perm_id)
        # reproduce the published value, to prove these are the same rotations
        p_old_recomputed = float(np.ravel(perm_sphere_p(d_full, cr_full, perm_id,
                                                        corr_type="spearman"))[0])
        rows.append({
            "disorder": dis, "n_parcels": int(keep.sum()),
            "spearman_rho_published": float(prow["spearman_rho"]),
            "spearman_rho_recomputed": obs,
            "spin_p_published": float(prow["spin_p"]),
            "spin_p_published_recomputed": p_old_recomputed,
            "spin_p_stated_convention": p_new,
            "n_null_ge_abs_obs": k,
            "fdr_q_published": float(prow["fdr_q"]),
            "partial_spin_p_published": float(prow["partial_spin_p"]),
            "partial_fdr_q_published": float(prow["partial_fdr_q"]),
        })
        log(f"    {dis:14s} rho {obs:+.4f}  spin_p {float(prow['spin_p']):.4f} -> "
            f"{p_new:.4f}  (#|null|>=|obs| = {k} of {perm_id.shape[1]})")

    df = pd.DataFrame(rows)
    df["fdr_q_stated_convention"] = bh(df["spin_p_stated_convention"].values)
    df["survives_published"] = df["fdr_q_published"] < 0.05
    df["survives_stated"] = df["fdr_q_stated_convention"] < 0.05
    df["changes_status"] = df["survives_published"] != df["survives_stated"]

    reproduced = bool(np.allclose(df["spin_p_published_recomputed"],
                                  df["spin_p_published"], atol=1e-9))
    log(f"\n  reproduces the published spin_p exactly: {reproduced}")
    n_old = int(df["survives_published"].sum())
    n_new = int(df["survives_stated"].sum())
    log(f"  survivors at q < 0.05: {n_old} published -> {n_new} under the stated convention")
    for _, r in df[df["changes_status"]].iterrows():
        log(f"    CROSSES: {r['disorder']}  q {r['fdr_q_published']:.4f} -> "
            f"{r['fdr_q_stated_convention']:.4f}")
    if not df["changes_status"].any():
        log("    no disorder changes status")

    df.to_csv(OUTDIR / "disease_convention_before_after.csv", index=False)

    ok, detail = rc.check_frozen(frozen_before)
    extra_after = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    ok_extra = extra_before == extra_after
    log(f"\nreleased v9 unchanged: {ok}; RR10 crossed files unchanged: {ok_extra}")

    (OUTDIR / "rr13b_summary.json").write_text(json.dumps({
        "n_rotations": int(perm_id.shape[1]), "seed": SEED, "n_parcels": int(n_parc),
        "reproduces_published_spin_p": reproduced,
        "floor_stated_convention": 1.0 / (perm_id.shape[1] + 1),
        "survivors_published": n_old, "survivors_stated": n_new,
        "crossings": df[df["changes_status"]]["disorder"].tolist(),
        "table": df.to_dict("records"),
        "frozen_v9_unchanged": ok, "frozen_rr10_unchanged": ok_extra,
        "runtime_seconds": round(time.time() - t0, 1)}, indent=2, default=float),
        encoding="utf-8")
    log(f"total runtime {(time.time() - t0) / 60:.1f} min")
    fh.close()
    return df


if __name__ == "__main__":
    main()
