"""RR1 - audit the vertices changed solely to satisfy the topology rules.

Classifies every vertex changed along the v3_clean -> v9 chain as `topology`,
`anatomy` or `both`, quantifies the topology share in vertices, per cent of
labelled cortex and mm^2 on the fsaverage white surface, locates it by cyto7 type
and Desikan parcel, renders it on the inflated surface, and repeats the four
primary structure-function correlations with the topology vertices plus a two-hop
mesh margin removed.

Three independent attributions are computed, because the later steps of the chain
were hand-painted rather than script-generated and so cannot be replayed:

  documented  the transition classes each PROVENANCE file attributes to a rule
              (buffer insertion, eulaminate-II thickening, speckle and fragment
              absorption, boundary smoothing) versus to an anatomical decision.
              Where a changelog already carries a reason column it is used
              directly. This is the primary classification.
  replayed    for each step, apply only the documented anatomical changes to the
              earlier map, run the repo's own ``converge_topology``, and diff.
              This is what the task spec asks for and it works exactly for the
              script-generated steps; for the hand-painted steps it gives a
              lower bound, since a human does not paint the minimal buffer.
  required    the revert-one test: a change is topologically required when
              restoring the vertex's old label creates an R1 skip edge against
              its final neighbours. A strict lower bound on the buffer.

Read-only on every annot. Run::
    conda run -n cyto7 python scripts/rr1_topology_audit.py --n-spin 1000
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

import rr_common as rc
from audit_topology import ordinal, unique_edges
from cyto7_surface_io import REPO_ROOT, resolve_target_map
from external_validation import _bh

OUTDIR = rc.OUT / "rr1_topology"
CACHEDIR = rc.OUT / "_cache"
DER = cfg.atlas_dir("fsaverage")
NM_CACHE = cfg.data_dir() / "neuromaps_cache"
DATASET = "Validation210"

CHAIN = ["v3_clean", "v4", "v5", "v6", "v7", "v8", "v9"]
NAME = {0: "unknown", 1: "Allocortex", 2: "Agranular", 3: "Dysgranular",
        4: "Eulaminate I", 5: "Eulaminate II", 6: "Eulaminate III", 7: "Koniocortex"}

TOPO, ANAT, BOTH = "topology", "anatomy", "both"

# --------------------------------------------------------------------------- #
# Documented attribution, per step, from the PROVENANCE files
# --------------------------------------------------------------------------- #
# Each entry maps (old_code, new_code) -> class, with a default for anything else.
# The provenance text that justifies each line is quoted in DOC_NOTES.

DOC_TRANSITIONS = {
    # v3_clean -> v4: allocortex expansion + perirhinal demotion are anatomical;
    # the strict-gradient buffer (16 lh + 9 rh) is topological. The changelog has no
    # reason column, so the buffer is recovered by replaying converge_topology.
    ("v3_clean", "v4"): {"default": ANAT, "replay_topology": True,
                         "extra_topology_csv": "v4_fragment_cleanup_changelog.csv"},
    # v4 -> v5: the changelog carries a reason column.
    ("v4", "v5"): {"reason_column": "v4_to_v5_changelog.csv",
                   "reason_map": {"cingulate_gradient": ANAT, "ring_closure": ANAT,
                                  "boundary_smooth": TOPO, "topology_buffer": TOPO,
                                  "speck_cleanup": TOPO},
                   "default": ANAT},
    # v5 -> v6: hand-painted ring completion; the provenance documents no
    # topology-motivated transition ("Isocortex was not touched").
    ("v5", "v6"): {"default": ANAT},
    # v6 -> v7: "Adds the missing AGRANULAR ventro-anterior insula ..., plus a
    # +-1 dysgranular buffer to preserve the strict sequential gradient."
    ("v6", "v7"): {(4, 3): TOPO, "default": ANAT},
    # v7 -> v8: entorhinal to allocortex and the widened belt bands are anatomical
    # (the expert meeting asked for them); "Eulaminate-II THICKENED so eulaminate-I
    # and -III no longer abut (preserves R1)" is topological.
    ("v7", "v8"): {(6, 5): TOPO, (5, 4): TOPO, (4, 3): BOTH, "default": ANAT},
    # v8 -> v9: the L/R entorhinal even-up only.
    ("v8", "v9"): {"default": ANAT},
}

DOC_NOTES = {
    ("v3_clean", "v4"): "PROVENANCE_v4: 'strict-buffer 16' (lh) / '9' (rh) from converge_topology; "
                        "everything else is the allocortex membership rule and the perirhinal demotion.",
    ("v4", "v5"): "v4_to_v5_changelog.csv reason column. cingulate_gradient and ring_closure are "
                  "anatomical/protocol; boundary_smooth is the R1-constrained regularisation and "
                  "topology_buffer and speck_cleanup are rule-driven.",
    ("v5", "v6"): "PROVENANCE_v6: hand-painted ring completion, 'Isocortex was not touched', no "
                  "topology-motivated transition documented.",
    ("v6", "v7"): "PROVENANCE_v7: 'All changes are EulI->Dys (gradient buffer) and Dys/EulI->Agr "
                  "(the insula sector)'.",
    ("v7", "v8"): "PROVENANCE_v8: 'Eulaminate-II THICKENED so eulaminate-I and -III no longer abut "
                  "(preserves R1)' -> EulIII->EulII and EulII->EulI are topological. The belt "
                  "widening EulI->Dys is an anatomical decision whose extent is also R1-constrained, "
                  "so it is classed as both.",
    ("v8", "v9"): "PROVENANCE_v9: 543 LH entorhinal vertices agranular->allocortex, RH untouched.",
}


# --------------------------------------------------------------------------- #
# Geometry and maps
# --------------------------------------------------------------------------- #


def load_chain() -> dict[str, dict[str, np.ndarray]]:
    return {v: {H: np.asarray(resolve_target_map(v, "fsaverage")[H], int) for H in ("L", "R")}
            for v in CHAIN}


def geometry_164k():
    geom = rc.fsaverage_geometry("white", "164k")
    out = {}
    for H in ("L", "R"):
        coords, faces = geom[H]
        out[H] = {"coords": coords, "faces": faces,
                  "edges": unique_edges(faces),
                  "A": rc.adjacency(faces, len(coords)),
                  "va": rc.vertex_areas(coords, faces)}
    return out


def desikan_164k() -> dict[str, tuple[np.ndarray, list[str]]]:
    out = {}
    for H, hemi in (("L", "lh"), ("R", "rh")):
        lab, _c, names = nib.freesurfer.io.read_annot(
            str(cfg.data_dir() / "voneconomo" / f"{hemi}.aparc.annot"))
        out[H] = (np.asarray(lab, int),
                  [n.decode() if isinstance(n, bytes) else n for n in names])
    return out


# --------------------------------------------------------------------------- #
# Step 1 - classification
# --------------------------------------------------------------------------- #


def r1_violating_edges(lab: np.ndarray, edges: np.ndarray) -> np.ndarray:
    ordv = ordinal(lab)
    both = (lab[edges[:, 0]] >= 1) & (lab[edges[:, 1]] >= 1)
    return edges[both & (np.abs(ordv[edges[:, 0]] - ordv[edges[:, 1]]) >= 2)]


def required_by_r1(before: np.ndarray, after: np.ndarray, changed: np.ndarray,
                   A) -> np.ndarray:
    """Revert-one test: reverting the vertex would create an R1 skip against its
    final neighbours, so the change is topologically required.

    Vertices whose old label was 0 (unpainted cortex) are excluded: R1 constrains
    labelled cortex only, so painting an unlabelled vertex can never be required
    by R1 and counting it would inflate the topology share by the whole of the
    ring-closure and hand-painted fill work.
    """
    ordn = ordinal(after)
    req = np.zeros(before.shape[0], bool)
    idx = np.where(changed & (before >= 1))[0]
    indptr, indices = A.indptr, A.indices
    old_ord = ordinal(before)
    for v in idx:
        nb = indices[indptr[v]:indptr[v + 1]]
        nb = nb[after[nb] >= 1]
        if nb.size == 0:
            continue
        if np.any(np.abs(old_ord[v] - ordn[nb]) >= 2):
            req[v] = True
    return req


def replay_topology(before: np.ndarray, after: np.ndarray, anat: np.ndarray,
                    faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply only the anatomical changes, then run the repo's converge_topology.

    Returns (buffer_mask, n_unresolved). The buffer mask is what the topology pass
    has to move once the anatomical decisions are in place.
    """
    from apply_rule_v3 import converge_topology
    lab = before.copy()
    lab[anat] = after[anat]
    locked = np.zeros(lab.shape[0], bool)     # nothing locked: the maps are already painted
    _new, buf, unresolved = converge_topology(lab.copy(), faces, locked)
    return buf, int(len(unresolved))


def classify(chain, geom, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, step_rows = [], []
    for a, b in zip(CHAIN[:-1], CHAIN[1:]):
        spec = DOC_TRANSITIONS[(a, b)]
        reason_by_vertex = {}
        if "reason_column" in spec:
            cl = pd.read_csv(DER / spec["reason_column"])
            for _, r in cl.iterrows():
                reason_by_vertex[(r["hemi"], int(r["vertex"]))] = r["reason"]
        for H, hemi in (("L", "lh"), ("R", "rh")):
            before, after = chain[a][H], chain[b][H]
            changed = before != after
            n_changed = int(changed.sum())
            cls = np.array([""] * before.shape[0], dtype=object)
            for v in np.where(changed)[0]:
                key = (hemi, int(v))
                if key in reason_by_vertex:
                    cls[v] = spec["reason_map"].get(reason_by_vertex[key], spec["default"])
                else:
                    cls[v] = spec.get((int(before[v]), int(after[v])), spec["default"])
            doc_topo = changed & (cls == TOPO)
            doc_both = changed & (cls == BOTH)
            doc_anat = changed & (cls == ANAT)
            # replay: apply only the anatomical decisions, then let the repo's own
            # converge_topology insert whatever buffer R1 then demands
            buf, unresolved = replay_topology(before, after, doc_anat | doc_both,
                                              geom[H]["faces"])
            replay_topo = buf & changed
            replay_extra = int((buf & ~changed).sum())
            req = required_by_r1(before, after, changed, geom[H]["A"])
            if spec.get("replay_topology"):
                # v3_clean->v4 has no reason column and its documented anatomical
                # clauses (allocortex membership, perirhinal demotion) cannot be
                # separated from the strict-gradient buffer in the changelog. The
                # revert-one test is the estimator for the buffer, and the count is
                # checked against the provenance. The in-place fragment cleanup that
                # followed does have a reason column, and absorbing a sub-20-vertex
                # belt component is a rule repair (R2 to R4) that the revert-one R1
                # test cannot see, so it is added from its changelog.
                cls[req & (cls == ANAT)] = TOPO
                extra = spec.get("extra_topology_csv")
                if extra:
                    fc = pd.read_csv(DER / extra)
                    ev = fc[fc["hemi"] == hemi]["vertex"].to_numpy().astype(int)
                    ev = ev[changed[ev]]
                    cls[ev] = TOPO
                doc_topo = changed & (cls == TOPO)
                doc_anat = changed & (cls == ANAT)
            for v in np.where(changed)[0]:
                rows.append({"hemi": hemi, "vertex": int(v), "step": f"{a}->{b}",
                             "old_type": NAME[int(before[v])], "new_type": NAME[int(after[v])],
                             "reason_class": cls[v],
                             "replayed_topology": bool(replay_topo[v]),
                             "required_by_r1": bool(req[v])})
            step_rows.append({"step": f"{a}->{b}", "hemi": hemi, "n_changed": n_changed,
                              "documented_topology": int(doc_topo.sum()),
                              "documented_both": int(doc_both.sum()),
                              "documented_anatomy": int(doc_anat.sum()),
                              "replayed_topology": int(replay_topo.sum()),
                              "replay_moves_unchanged_vertices": replay_extra,
                              "replay_unresolved_skips": unresolved,
                              "required_by_r1": int(req.sum()),
                              "r1_skips_before": int(len(r1_violating_edges(before, geom[H]["edges"]))),
                              "r1_skips_after": int(len(r1_violating_edges(after, geom[H]["edges"]))),
                              "note": DOC_NOTES[(a, b)]})
            log(f"  {a}->{b} {hemi}: changed {n_changed}, documented topology "
                f"{int(doc_topo.sum())} + both {int(doc_both.sum())}, replay "
                f"{int(replay_topo.sum())} (extra {replay_extra}), required {int(req.sum())}")
    return pd.DataFrame(rows), pd.DataFrame(step_rows)


# --------------------------------------------------------------------------- #
# Step 2 - quantify
# --------------------------------------------------------------------------- #


def quantify(cl: pd.DataFrame, chain, geom, desikan, log=print):
    """Cumulative per-vertex classification, counted once by the final reason."""
    order = {f"{a}->{b}": i for i, (a, b) in enumerate(zip(CHAIN[:-1], CHAIN[1:]))}
    cl = cl.copy()
    cl["step_index"] = cl["step"].map(order)
    last = (cl.sort_values("step_index")
              .groupby(["hemi", "vertex"], as_index=False)
              .last()[["hemi", "vertex", "reason_class", "step", "old_type", "new_type",
                       "replayed_topology", "required_by_r1"]])
    v9 = chain["v9"]
    rows, by_type, by_parcel = [], [], []
    tot_labelled = sum(int((v9[H] > 0).sum()) for H in ("L", "R"))
    for H, hemi in (("L", "lh"), ("R", "rh")):
        va = geom[H]["va"]
        sub = last[last.hemi == hemi]
        dk_lab, dk_names = desikan[H]
        for cls in (TOPO, BOTH, ANAT):
            idx = sub[sub.reason_class == cls]["vertex"].to_numpy()
            rows.append({"hemi": hemi, "reason_class": cls, "n_vertices": len(idx),
                         "pct_labelled_hemi": 100 * len(idx) / max(1, int((v9[H] > 0).sum())),
                         "area_mm2": float(va[idx].sum()) if len(idx) else 0.0})
            for c in range(1, 8):
                sel = idx[v9[H][idx] == c] if len(idx) else idx
                if len(sel):
                    by_type.append({"hemi": hemi, "reason_class": cls,
                                    "v9_type": NAME[c], "n_vertices": len(sel),
                                    "area_mm2": round(float(va[sel].sum()), 1),
                                    "pct_of_type": round(100 * len(sel) / int((v9[H] == c).sum()), 3)})
            if cls in (TOPO, BOTH) and len(idx):
                for p in np.unique(dk_lab[idx]):
                    sel = idx[dk_lab[idx] == p]
                    nm = dk_names[p] if 0 <= p < len(dk_names) else str(p)
                    by_parcel.append({"hemi": hemi, "reason_class": cls, "desikan_parcel": nm,
                                      "n_vertices": len(sel),
                                      "area_mm2": round(float(va[sel].sum()), 1),
                                      "pct_of_parcel": round(
                                          100 * len(sel) / max(1, int((dk_lab == p).sum())), 3)})
    summary = pd.DataFrame(rows)
    tot = summary.groupby("reason_class", as_index=False).agg(
        n_vertices=("n_vertices", "sum"), area_mm2=("area_mm2", "sum"))
    tot["hemi"] = "both"
    tot["pct_labelled_hemi"] = 100 * tot["n_vertices"] / tot_labelled
    summary = pd.concat([summary, tot], ignore_index=True)
    summary["area_mm2"] = summary["area_mm2"].round(1)
    summary["pct_labelled_hemi"] = summary["pct_labelled_hemi"].round(4)
    log(f"  total labelled vertices (164k, v9): {tot_labelled}")
    log(summary.to_string(index=False))
    return last, summary, pd.DataFrame(by_type), pd.DataFrame(by_parcel), tot_labelled


# --------------------------------------------------------------------------- #
# Step 3 - leave-them-out robustness
# --------------------------------------------------------------------------- #


def leave_out(last: pd.DataFrame, geom, n_spin: int, log=print):
    """Primary structure-function correlations with topology+both vertices and a
    two-hop mesh margin removed, on 32k fs_LR with the released nulls restricted."""
    import reviewer_response as rr
    from summarise_functional_features import FEATURES, build_validity_mask, load_all_features

    drop164 = {}
    for H in ("L", "R"):
        hemi = "lh" if H == "L" else "rh"
        sub = last[(last.hemi == hemi) & last.reason_class.isin([TOPO, BOTH])]
        m = np.zeros(geom[H]["A"].shape[0], bool)
        m[sub["vertex"].to_numpy()] = True
        drop164[H] = rc.dilate(m, geom[H]["A"], 2)
        log(f"    {hemi}: {int(m.sum())} topology/both vertices, "
            f"{int(drop164[H].sum())} after a two-hop margin (164k)")
    drop32 = rr.resample_to_32k({H: drop164[H].astype(np.float32) for H in ("L", "R")},
                                "nearest", "rr1_topology_margin")
    drop = np.concatenate([drop32["L"].astype(bool), drop32["R"].astype(bool)])

    lab32 = rc.labels_32k()
    labc = rc.cat(lab32, int).astype(int)
    rank = labc.astype(float)
    nulls = rc.load_nulls(n_spin)
    feats = load_all_features(DATASET)
    base = rc.cat(build_validity_mask(lab32, feats), bool).astype(bool)
    gene = np.concatenate([np.load(NM_CACHE / f"abagen_genepc1_fsLR32k_hemi-{H}.npy")
                           for H in ("L", "R")]).astype(float)

    def one(vals, keep):
        rho = float(stats.spearmanr(vals[keep], rank[keep])[0])
        nr = np.empty(n_spin)
        for i in range(n_spin):
            spun = nulls[:, i][keep]
            ok = np.isfinite(spun)
            nr[i] = stats.spearmanr(vals[keep][ok], spun[ok])[0]
        return rho, float((np.sum(np.abs(nr) >= abs(rho)) + 1) / (n_spin + 1)), int(keep.sum())

    published = {"myelin": 0.5894, "thickness": -0.5018, "gradient": -0.5761, "genepc1": 0.6657}
    pub_q = {"myelin": 0.0045, "thickness": 0.0060, "gradient": 0.0045, "genepc1": 0.0030}
    rows = []
    # the nine-feature family, so BH is computed within it as published
    ps_all, res_all = [], []
    for f in FEATURES:
        v = rc.cat(feats[f.key])
        rho, p, n = one(v, base & ~drop)
        ps_all.append(p); res_all.append((f.key, f.label, rho, p, n))
    q_all = _bh(np.array(ps_all))
    for (key, label, rho, p, n), q in zip(res_all, q_all):
        if key not in published:
            continue
        rows.append({"measure": label, "key": key, "family": "structure-function (9)",
                     "n_retained": n, "rho_leaveout": round(rho, 4),
                     "spin_p_leaveout": round(p, 4), "q_leaveout": round(float(q), 4),
                     "rho_published": published[key], "q_published": pub_q[key],
                     "delta_rho": round(rho - published[key], 4)})
    keep_g = (labc > 0) & np.isfinite(gene) & ~drop
    rho, p, n = one(gene, keep_g)
    rows.append({"measure": "AHBA gene expression PC1", "key": "genepc1",
                 "family": "external reference (3)", "n_retained": n,
                 "rho_leaveout": round(rho, 4), "spin_p_leaveout": round(p, 4),
                 "q_leaveout": np.nan, "rho_published": published["genepc1"],
                 "q_published": pub_q["genepc1"],
                 "delta_rho": round(rho - published["genepc1"], 4)})
    df = pd.DataFrame(rows)
    log(df.to_string(index=False))
    return df, drop164, int(drop.sum())


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def figure(last: pd.DataFrame, chain, out_path, log=print):
    """Topology-repaired vertices on the inflated fsaverage surface over a grey cyto7 map."""
    from make_presentation_figures import lighting_normals, lit_panel, load_surface
    v9 = chain["v9"]
    fig = plt.figure(figsize=(13.5, 7.0))
    fig.patch.set_facecolor("white")
    panels = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
    for i, (hemi, view) in enumerate(panels):
        H = "L" if hemi == "lh" else "R"
        coords, faces = load_surface("Validation210", H, "inflated") \
            if False else _inflated(hemi)
        lab = v9[H]
        col = np.zeros((lab.shape[0], 3))
        greys = {0: 0.98, 1: 0.86, 2: 0.80, 3: 0.74, 4: 0.68, 5: 0.62, 6: 0.56, 7: 0.50}
        for c, g in greys.items():
            col[lab == c] = g
        sub = last[(last.hemi == hemi) & last.reason_class.isin([TOPO, BOTH])]
        topo = sub[sub.reason_class == TOPO]["vertex"].to_numpy()
        both = sub[sub.reason_class == BOTH]["vertex"].to_numpy()
        col[both] = (0.99, 0.68, 0.25)
        col[topo] = (0.78, 0.10, 0.12)
        ax = fig.add_subplot(2, 2, i + 1)
        _render(ax, coords, faces, col, hemi, view)
        ax.set_title(f"{hemi.upper()} {view}", fontsize=11, weight="bold")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=(0.78, 0.10, 0.12), label="changed for topology only"),
                        Patch(facecolor=(0.99, 0.68, 0.25), label="anatomical change also required by R1 (both)"),
                        Patch(facecolor="0.7", label="cyto7 v9 (greyscale)")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Vertices changed to satisfy the topology rules (v3_clean to v9)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(str(out_path), dpi=200, facecolor="white")
    plt.close(fig)
    log(f"  wrote {out_path}")


def _inflated(hemi):
    p = cfg.data_dir() / "fsaverage_surfaces" / f"{hemi}.inflated"
    coords, faces = nib.freesurfer.io.read_geometry(str(p))
    return np.asarray(coords, float), np.asarray(faces, int)


def _render(ax, coords, faces, col, hemi, view):
    from matplotlib.collections import PolyCollection
    azim = {"lateral": (-1 if hemi == "lh" else 1), "medial": (1 if hemi == "lh" else -1)}[view]
    keep = np.ones(len(faces), bool)
    # simple painter: project onto the y-z plane, sort by x so the near side wins
    pts = coords[:, [1, 2]]
    depth = coords[:, 0] * azim
    fd = depth[faces].mean(1)
    order = np.argsort(fd)
    tri = faces[order][keep[order]]
    verts = pts[tri]
    fc = col[tri].mean(axis=1)
    n = np.cross(coords[tri[:, 1]] - coords[tri[:, 0]], coords[tri[:, 2]] - coords[tri[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.55 + 0.45 * np.clip(n @ np.array([azim * 0.6, -0.5, 0.6]), 0, 1)
    fc = np.clip(fc * shade[:, None], 0, 1)
    ax.add_collection(PolyCollection(verts, facecolors=fc, edgecolors="face", linewidths=0.0))
    ax.set_xlim(pts[:, 0].min() - 3, pts[:, 0].max() + 3)
    ax.set_ylim(pts[:, 1].min() - 3, pts[:, 1].max() + 3)
    ax.set_aspect("equal")
    ax.axis("off")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=1000)
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frozen_before = rc.frozen_hashes()

    print("== loading the v3_clean to v9 chain (164k fsaverage) ==")
    chain = load_chain()
    geom = geometry_164k()
    desikan = desikan_164k()

    print("== step 1: classify every changed vertex ==")
    cl, steps = classify(chain, geom)
    cl.to_csv(OUTDIR / "repair_classification.csv", index=False)
    steps.to_csv(OUTDIR / "step_attribution.csv", index=False)

    print("== step 2: quantify ==")
    last, summary, by_type, by_parcel, tot = quantify(cl, chain, geom, desikan)
    last.to_csv(OUTDIR / "repair_classification_final.csv", index=False)
    summary.to_csv(OUTDIR / "repair_summary.csv", index=False)
    by_type.to_csv(OUTDIR / "repair_by_type.csv", index=False)
    by_parcel.to_csv(OUTDIR / "repair_by_desikan.csv", index=False)

    print("== step 2b: surface figure ==")
    figure(last, chain, OUTDIR / "topology_repair_map.png")

    print("== step 3: leave-them-out robustness ==")
    lo, drop164, n_drop32 = leave_out(last, geom, args.n_spin)
    lo.to_csv(OUTDIR / "leave_out_robustness.csv", index=False)

    # pre-chain step v3 -> v3_clean, for the record
    pre = pd.read_csv(DER / "v3_to_v3clean_speckle_changelog.csv")
    ok, detail = rc.check_frozen(frozen_before)
    (OUTDIR / "rr1_summary.json").write_text(json.dumps({
        "total_labelled_164k_v9": tot,
        "prechain_v3_to_v3clean_speckle_vertices": int(len(pre)),
        "topology_vertices": int(summary[(summary.hemi == "both") &
                                         (summary.reason_class == TOPO)]["n_vertices"].iloc[0]),
        "both_vertices": int(summary[(summary.hemi == "both") &
                                     (summary.reason_class == BOTH)]["n_vertices"].iloc[0]),
        "anatomy_vertices": int(summary[(summary.hemi == "both") &
                                        (summary.reason_class == ANAT)]["n_vertices"].iloc[0]),
        "n_dropped_with_margin_164k": int(sum(int(m.sum()) for m in drop164.values())),
        "n_dropped_with_margin_32k": n_drop32,
        "max_abs_delta_rho": float(lo["delta_rho"].abs().max()),
        "frozen_unchanged": ok, "frozen": detail, "n_spin": args.n_spin},
        indent=2), encoding="utf-8")
    print("frozen files unchanged:", ok)
    return cl, summary, lo


if __name__ == "__main__":
    main()
