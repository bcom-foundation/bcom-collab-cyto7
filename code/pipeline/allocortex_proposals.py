"""Allocortex vs periallocortex: proposals + consistency table (Layer 1C).

Implements the UPDATED ``docs/REFINE_allocortex.md`` section 1C. Proposals only -
nothing is ever written to an annot.

Periallocortex-belt reference = **weighted** consensus of three surface-native
sources (the first run used only the two gyral atlases; this fixes that):
  * **FreeSurfer ex-vivo histological** (entorhinal + perirhinal; perirhinal_exvivo
    is the perirhinal/BA35-36 histological label) - WEIGHT 3 (highest).
  * **Glasser HCP-MMP1.0** {EC, PreS, PeEc, Pir, PHA1, PHA2, PHA3}, resampled
    32k fs_LR -> 164k fsaverage by NEAREST-NEIGHBOUR (neuromaps/wb_command) - WEIGHT 2.
  * **Destrieux + Desikan** {entorhinal, parahippocampal} (gyral) - WEIGHT 1 each (lowest).

**Allocortex != periallocortex.** Periallocortex (EC/perirhinal/parahippocampal)
maps to agranular/dysgranular in the Structural Model, NOT to cyto7 Allocortex
(code 1 = archi/paleocortex core). So belt + cyto7 in {Allocortex, Agranular,
Dysgranular} is **concordant**, never an "under_allo" edit. A true discrepancy
is only flagged when cyto7 disagrees with the **histological + Glasser** (trusted)
consensus on the allocortex-vs-isocortex call - not when only the gyral atlases do.

Headline output = the per-area **periallocortex-consistency table**: cyto7 paints
HA/HB/HC (von Economo uncinate/parauncinate/rhinal) as Allocortex but
EC/parahippocampal as Agranular/Dysgranular - an inconsistency the experts should
resolve with one rule.

Topology guard: any proposal that would relabel a Dysgranular patch to Allocortex
is tagged ``r1_safe=False`` / ``requires_agranular_buffer`` (Allocortex ord 0
abutting Dysgranular ord 2 breaks R1) and excluded from auto-application.

Outputs (nothing committed to the annot):
  resources/cyto7_derived/allocortex_edit_proposals.csv
  resources/cyto7_derived/periallocortex_consistency.csv
  figures/refine/allocortex_adjudication.png

Run::
    conda activate cyto7
    python scripts/allocortex_proposals.py        # default annot = v2 if present else v1
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import csv
import os
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch
from scipy.sparse.csgraph import connected_components

from cyto7_surface_io import REPO_ROOT
from audit_topology import adjacency
from build_support_map import (
    ATLAS_DIR, DERIVED_DIR, REFINE_DIR, VE_DIR, geodesic_from_medial, label_mask,
)
from make_presentation_figures import (
    HEMIS, LIGHT_BLUE, lighting_normals, lit_panel, load_labels, load_surface,
    resolve_annot_paths, version_label,
)

# Source weights (cytoarchitectonic > Glasser > gyral).
W_FS, W_GLASSER, W_DESTRIEUX, W_DESIKAN = 3.0, 2.0, 1.0, 1.0
DEFAULT_WB = cfg.workbench_dir()
GLASSER_BELT = ["EC", "PreS", "PeEc", "Pir", "PHA1", "PHA2", "PHA3"]
DESTRIEUX_BELT = ("G_oc-temp_med-Parahip",)            # parahippocampal (gyral)
DESIKAN_BELT = ("entorhinal", "parahippocampal")
ECONOMO_PERIALLO = ("HA", "HB", "HC")                   # von Economo limbic areas painted Allocortex

CODE_NAME = {0: "medial", 1: "Allocortex", 2: "Agranular", 3: "Dysgranular",
             4: "Eulaminate I", 5: "Eulaminate II", 6: "Eulaminate III", 7: "Koniocortex"}
PERIALLO_COMPATIBLE = {1, 2, 3}     # Allocortex/Agranular/Dysgranular
ISOCORTEX = {4, 5, 6, 7}
ordc = lambda code: code - 1        # ordinal


# --------------------------------------------------------------------------- #
# Source masks
# --------------------------------------------------------------------------- #


def fs_exvivo_masks(hemi: str, n: int) -> dict[str, np.ndarray]:
    return {
        "EC": label_mask(ATLAS_DIR / f"{hemi}.entorhinal_exvivo.label", n),
        "Perirhinal/BA35-36": label_mask(ATLAS_DIR / f"{hemi}.perirhinal_exvivo.label", n),
    }


def destrieux_mask(hemi, n):
    labels, _c, names = nib.freesurfer.io.read_annot(str(ATLAS_DIR / f"{hemi}.aparc.a2009s.annot"))
    names = [x.decode() if isinstance(x, bytes) else x for x in names]
    idx = [i for i, nm in enumerate(names) if nm in DESTRIEUX_BELT]
    return np.isin(np.clip(labels, 0, len(names) - 1), idx)


def desikan_masks(hemi, n):
    labels, _c, names = nib.freesurfer.io.read_annot(str(VE_DIR / f"{hemi}.aparc.annot"))
    names = [x.decode() if isinstance(x, bytes) else x for x in names]
    out = {}
    for nm in DESIKAN_BELT:
        if nm in names:
            out[nm] = labels == names.index(nm)
    return out


def economo_masks(hemi, n):
    labels, _c, names = nib.freesurfer.io.read_annot(str(VE_DIR / f"{hemi}.economo.annot"))
    names = [x.decode() if isinstance(x, bytes) else x for x in names]
    out = {}
    for acr in ECONOMO_PERIALLO:
        if acr in names:
            out[acr] = np.asarray(labels) == names.index(acr)
    return out


def glasser_belt_masks(hemi: str, n: int, workbench_bin: str | None) -> dict[str, np.ndarray]:
    """Glasser belt parcels resampled 32k fs_LR -> 164k fsaverage (nearest), cached."""
    cache = ATLAS_DIR / f"glasser_belt_164k_{hemi}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    wb = workbench_bin or os.environ.get("WORKBENCH_BIN") or DEFAULT_WB
    if wb and Path(wb).is_dir() and wb not in os.environ["PATH"]:
        os.environ["PATH"] = wb + os.pathsep + os.environ["PATH"]
    from netneurotools import datasets as nd
    from neuromaps import transforms
    from nibabel.gifti import GiftiDataArray, GiftiImage
    b = nd.fetch_mmpall(verbose=0)
    src = {"lh": b.L, "rh": b.R}[hemi]
    H = {"lh": "L", "rh": "R"}[hemi]
    g = nib.load(str(src))
    data = np.asarray(g.darrays[0].data)
    key = {l.label: l.key for l in g.labeltable.labels}
    out = {}
    for short in GLASSER_BELT:
        name = f"{H}_{short}_ROI"
        m32 = (data == key[name]).astype(np.float32)
        gi = GiftiImage(); gi.add_gifti_data_array(GiftiDataArray(m32))
        res = transforms.fslr_to_fsaverage(gi, target_density="164k", hemi=H, method="nearest")
        out[short] = (np.asarray(res[0].agg_data()) > 0.5)
    np.savez(cache, **{k: v for k, v in out.items()})
    print(f"    cached Glasser belt -> {cache.name}")
    return out


# --------------------------------------------------------------------------- #
# Per-vertex consensus
# --------------------------------------------------------------------------- #


def build_consensus(hemi, n, workbench_bin):
    """Return per-vertex source memberships, weighted score, trusted/gyral belt,
    and the named area masks for the consistency table."""
    fs = fs_exvivo_masks(hemi, n)
    gl = glasser_belt_masks(hemi, n, workbench_bin)
    dx = destrieux_mask(hemi, n)
    dk = desikan_masks(hemi, n)
    eco = economo_masks(hemi, n)

    fs_belt = np.zeros(n, bool)
    for m in fs.values():
        fs_belt |= m
    gl_belt = np.zeros(n, bool)
    for m in gl.values():
        gl_belt |= m
    dk_belt = np.zeros(n, bool)
    for m in dk.values():
        dk_belt |= m
    dx_belt = dx

    weighted = (W_FS * fs_belt + W_GLASSER * gl_belt + W_DESTRIEUX * dx_belt +
                W_DESIKAN * dk_belt) / (W_FS + W_GLASSER + W_DESTRIEUX + W_DESIKAN)
    trusted_belt = fs_belt | gl_belt           # histological + Glasser call
    gyral_belt = dx_belt | dk_belt

    sources = {"FS_exvivo": fs_belt, "Glasser": gl_belt, "Destrieux": dx_belt, "Desikan": dk_belt}
    src_w = {"FS_exvivo": W_FS, "Glasser": W_GLASSER, "Destrieux": W_DESTRIEUX, "Desikan": W_DESIKAN}

    # named areas for the consistency table
    areas = {}
    for acr, m in eco.items():
        areas[f"vE_{acr}"] = m
    areas["EC"] = fs["EC"] | gl.get("EC", np.zeros(n, bool))
    areas["Perirhinal/BA35-36"] = fs["Perirhinal/BA35-36"] | gl.get("PeEc", np.zeros(n, bool))
    for p in ("PHA1", "PHA2", "PHA3", "Pir", "PreS"):
        areas[p] = gl.get(p, np.zeros(n, bool))
    return dict(weighted=weighted, trusted=trusted_belt, gyral=gyral_belt,
                sources=sources, src_w=src_w, areas=areas)


# --------------------------------------------------------------------------- #
# Proposals
# --------------------------------------------------------------------------- #


def proposals_hemi(hemi, lab, coords, faces, A, con, gdist, far_mm, min_verts):
    n = lab.shape[0]
    trusted = con["trusted"]
    weighted = con["weighted"]
    allo = lab == 1

    cat_mask = {
        # cyto7 calls isocortex where histology+Glasser say periallocortex
        "discrepant_iso_in_periallo": trusted & np.isin(lab, list(ISOCORTEX)),
        # cyto7 Allocortex out in isocortex, no trusted support, far from medial wall
        "discrepant_allo_in_iso": allo & (~trusted) & (~con["gyral"]) & (gdist > far_mm),
    }
    concordant = trusted & np.isin(lab, list(PERIALLO_COMPATIBLE))

    rows = []
    for cat, mask in cat_mask.items():
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        ncomp, comp = connected_components(A[idx][:, idx], directed=False)
        for k in range(ncomp):
            members = idx[comp == k]
            if members.size < min_verts:
                continue
            centroid = coords[members].mean(0)
            cvert = int(members[np.argmin(np.linalg.norm(coords[members] - centroid, axis=1))])
            dom = int(np.bincount(lab[members], minlength=8).argmax())
            agree = [s for s, m in con["sources"].items() if m[members].mean() > 0.5]
            wmean = float(weighted[members].mean())
            consensus = "periallo" if wmean >= 0.5 else ("periallo(weak)" if wmean > 0 else "isocortex")
            # boundary neighbour ordinals (for R1 check)
            bmask = (A[members].sum(0).A1 > 0)
            bmask[members] = False
            nb = lab[bmask & (lab >= 1)]
            nb_ord = (nb - 1) if nb.size else np.array([], int)
            if cat == "discrepant_iso_in_periallo":
                suggested = "periallocortex (Agranular/Dysgranular)"
                cand_ord = 2  # dysgranular, nearest periallo to eulaminate
            else:
                suggested = "isocortex / periallocortex (per reference)"
                cand_ord = int(np.median(nb_ord)) if nb_ord.size else 3
            # R1 safety
            requires_buffer = (CODE_NAME[dom] == "Dysgranular" and "Allocortex" in suggested)
            r1_safe = bool((nb_ord.size == 0 or np.all(np.abs(cand_ord - nb_ord) <= 1))
                           and not requires_buffer)
            topo = ("requires_agranular_buffer (would manufacture an R1 skip)"
                    if not r1_safe else "one-step relabel, R1-safe")
            rows.append({
                "hemi": hemi, "category": cat, "cluster_id": f"{hemi}_{cat}_{k}",
                "n_verts": int(members.size),
                "centroid": "(%.1f, %.1f, %.1f)" % tuple(centroid), "centroid_vertex": cvert,
                "cyto7_label": CODE_NAME[dom],
                "weighted_consensus": consensus,
                "weighted_score": round(wmean, 3),
                "sources_agreeing": "+".join(agree) if agree else "none",
                "source_weights": ";".join(f"{s}={con['src_w'][s]:g}" for s in agree) if agree else "",
                "dist_to_medial_wall_mm": round(float(np.median(gdist[members])), 2),
                "suggested_class": suggested,
                "topology_effect": topo,
                "r1_safe": r1_safe,
            })
    return rows, cat_mask, concordant


# --------------------------------------------------------------------------- #
# Consistency table
# --------------------------------------------------------------------------- #


def consistency_rows(hemi, lab, con, gdist):
    rows = []
    for area, m in con["areas"].items():
        m = m & (lab >= 1)  # restrict to labelled cortex
        if m.sum() == 0:
            continue
        codes = np.bincount(lab[m], minlength=8)
        dom = int(codes.argmax())
        current = CODE_NAME[dom]
        # which sources define this area's belt membership
        srcs = [s for s, sm in con["sources"].items() if (sm & m).sum() > 0.2 * m.sum()]
        candidate = "Agranular/Dysgranular (periallocortex)"
        if dom in (2, 3):
            verdict = "consistent (periallocortex as agranular/dysgranular)"
        elif dom == 1:
            verdict = "INCONSISTENT: currently Allocortex (vs periallo->agranular/dysgranular)"
        else:
            verdict = "INCONSISTENT: currently isocortex (eulaminate)"
        frac = {CODE_NAME[c]: round(float(codes[c] / m.sum()), 2) for c in range(1, 8) if codes[c] > 0}
        rows.append({
            "hemi": hemi, "area": area, "n_verts": int(m.sum()),
            "dominant_cyto7_label": current,
            "cyto7_composition": str(frac),
            "reference": "periallocortex",
            "reference_sources": "+".join(srcs) if srcs else "none",
            "dist_to_medial_wall_mm": round(float(np.median(gdist[m])), 2),
            "current_class": current, "candidate_class": candidate,
            "current_vs_candidate": verdict,
        })
    return rows


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def render_packet(version, per_hemi, out_path, dpi):
    geom = {h: load_surface(h, "inflated") for h in HEMIS}
    vn = {h: lighting_normals(h) for h in HEMIS}
    cols = [(h, v) for h in HEMIS for v in ("medial", "ventral")]
    rows = [("FS ex-vivo (histo, w3)", "FS_exvivo"),
            ("Glasser (w2)", "Glasser"),
            ("gyral Destrieux+Desikan (w1)", "gyral"),
            ("concordant / discrepant", "verdict")]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(len(cols) * 3.0, len(rows) * 2.9),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    teal = np.array([0.0, 0.55, 0.55])
    for ri, (rlabel, rkey) in enumerate(rows):
        for ci, (hemi, view) in enumerate(cols):
            ax = axes[ri, ci]
            coords, faces = geom[hemi]
            ph = per_hemi[hemi]
            lab = ph["lab"]; A = ph["A"]; con = ph["con"]
            allo = lab == 1
            outline = allo & (A @ (~allo).astype(np.int8) > 0)
            rgb = np.tile(np.array([0.86, 0.86, 0.86]), (coords.shape[0], 1))
            rgb[lab == 0] = LIGHT_BLUE
            if rkey == "FS_exvivo":
                rgb[con["sources"]["FS_exvivo"]] = teal
            elif rkey == "Glasser":
                rgb[con["sources"]["Glasser"]] = teal
            elif rkey == "gyral":
                rgb[con["gyral"]] = teal
            else:
                rgb[ph["concordant"]] = (0.20, 0.70, 0.30)             # green concordant
                rgb[ph["cat_mask"]["discrepant_iso_in_periallo"]] = (0.85, 0.10, 0.10)  # red
                rgb[ph["cat_mask"]["discrepant_allo_in_iso"]] = (0.95, 0.55, 0.0)       # orange
            rgb[outline] = (0, 0, 0)
            lit_panel(ax, coords, faces, rgb, hemi, view, vn[hemi])
            if ri == 0:
                ax.set_title(f"{hemi.upper()} {view}", fontsize=10)
        axes[ri, 0].text2D(-0.14, 0.5, rlabel, transform=axes[ri, 0].transAxes, rotation=90,
                           va="center", ha="center", fontsize=9.5, weight="bold")
    handles = [Patch(facecolor=teal, label="belt (this source)"),
               Patch(facecolor=(0, 0, 0), label="cyto7 allocortex outline"),
               Patch(facecolor=(0.20, 0.70, 0.30), label="concordant (periallo=agran/dysgran)"),
               Patch(facecolor=(0.85, 0.10, 0.10), label="discrepant: cyto7 isocortex in periallo"),
               Patch(facecolor=(0.95, 0.55, 0.0), label="discrepant: cyto7 allo in isocortex")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Allocortex adjudication ({version}) - weighted belt consensus; PROPOSALS ONLY",
                 fontsize=14, y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out_path}")


# --------------------------------------------------------------------------- #
# REPORT.md patch
# --------------------------------------------------------------------------- #


def update_report(report_path: Path, prop: list, cons: list):
    if not report_path.exists():
        return
    txt = report_path.read_text(encoding="utf-8")
    import re
    n_iso = sum(r["category"] == "discrepant_iso_in_periallo" for r in prop)
    n_allo = sum(r["category"] == "discrepant_allo_in_iso" for r in prop)
    n_unsafe = sum(not r["r1_safe"] for r in prop)
    inconsistent = [c for c in cons if c["current_vs_candidate"].startswith("INCONSISTENT")]
    sec = []
    sec.append("## 3. Allocortex / periallocortex proposals (flagged — NOT applied)\n")
    sec.append("Belt reference is now a **weighted consensus** of FreeSurfer ex-vivo histology "
               "(w3), Glasser HCP-MMP1.0 resampled to 164k (w2), and gyral Destrieux/Desikan (w1). "
               "Periallocortex maps to agranular/dysgranular in the Structural Model, so cyto7 "
               "Agranular/Dysgranular in the belt is **concordant**, not an under_allo edit.\n")
    sec.append(f"- **True discrepancies (proposals): {len(prop)}** — "
               f"{n_iso} `discrepant_iso_in_periallo` (cyto7 isocortex where histology+Glasser say "
               f"periallo), {n_allo} `discrepant_allo_in_iso` (cyto7 Allocortex out in isocortex). "
               f"{n_unsafe} tagged `r1_safe=False` (Dysgranular->Allocortex would break R1).")
    sec.append(f"- The first run's 4 'under_allo' clusters (cyto7 Agranular/Dysgranular in the "
               f"medial-temporal belt) are now correctly logged as **concordant** and dissolve "
               f"from the proposal list.\n")
    sec.append("**Headline: periallocortex-consistency table** "
               "(`periallocortex_consistency.csv`). cyto7 labels the periallocortical belt "
               f"inconsistently — {len(inconsistent)} of {len(cons)} area-rows are INCONSISTENT, "
               "chiefly the von Economo HA/HB/HC areas painted **Allocortex** while EC/perirhinal/"
               "PHA are painted **Agranular/Dysgranular**. The experts should pick one rule.\n")
    sec.append("| hemi | area | dominant cyto7 | reference sources | dist-medial (mm) | verdict |")
    sec.append("| --- | --- | --- | --- | --- | --- |")
    for c in sorted(cons, key=lambda r: (r["area"], r["hemi"])):
        sec.append(f"| {c['hemi']} | {c['area']} | {c['dominant_cyto7_label']} | "
                   f"{c['reference_sources']} | {c['dist_to_medial_wall_mm']} | "
                   f"{c['current_vs_candidate'].split(':')[0]} |")
    sec.append("")
    block = "\n".join(sec)
    new = re.sub(r"## 3\. Allocortex.*?(?=\n## 4\.)", block + "\n", txt, flags=re.S)
    report_path.write_text(new, encoding="utf-8")
    print(f"  updated {report_path} (section 3)")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def default_annot() -> str:
    v2 = DERIVED_DIR / "pial.lh.cyto7.v2.annot"
    if v2.exists():
        return str(DERIVED_DIR / "pial.{hemi}.cyto7.v2.annot")
    return str(cfg.atlas_dir("provenance/as_painted") /
               "pial.{hemi}.cyto7.annot")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annot", default=default_annot())
    p.add_argument("--far-mm", type=float, default=12.0)
    p.add_argument("--min-cluster-verts", type=int, default=10)
    p.add_argument("--workbench-bin", default=None)
    p.add_argument("--out-derived", type=Path, default=DERIVED_DIR)
    p.add_argument("--out-fig", type=Path, default=REFINE_DIR)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    annot_paths = resolve_annot_paths(args.annot)
    version = version_label(annot_paths)
    print(f"Allocortex 1C (weighted consensus) for cyto7 {version}: {annot_paths['lh']}")
    all_prop, all_cons, per_hemi = [], [], {}
    for hemi in HEMIS:
        lab = load_labels(annot_paths[hemi], hemi)
        coords, faces = load_surface(hemi, "pial")
        A = adjacency(faces, lab.shape[0])
        con = build_consensus(hemi, lab.shape[0], args.workbench_bin)
        gdist = geodesic_from_medial(coords, faces, lab == 0)
        prop, cat_mask, concordant = proposals_hemi(hemi, lab, coords, faces, A, con,
                                                    gdist, args.far_mm, args.min_cluster_verts)
        cons = consistency_rows(hemi, lab, con, gdist)
        all_prop += prop; all_cons += cons
        per_hemi[hemi] = {"lab": lab, "A": A, "con": con, "cat_mask": cat_mask,
                          "concordant": concordant}
        print(f"  {hemi}: {len(prop)} proposals "
              f"({sum(p['category']=='discrepant_iso_in_periallo' for p in prop)} iso-in-periallo, "
              f"{sum(p['category']=='discrepant_allo_in_iso' for p in prop)} allo-in-iso); "
              f"concordant belt verts={int(concordant.sum())}; {len(cons)} consistency rows")

    args.out_derived.mkdir(parents=True, exist_ok=True)
    # proposals csv
    pcols = ["hemi", "category", "cluster_id", "n_verts", "centroid", "centroid_vertex",
             "cyto7_label", "weighted_consensus", "weighted_score", "sources_agreeing",
             "source_weights", "dist_to_medial_wall_mm", "suggested_class", "topology_effect", "r1_safe"]
    all_prop.sort(key=lambda r: (-r["n_verts"]))
    with open(args.out_derived / "allocortex_edit_proposals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pcols); w.writeheader(); w.writerows(all_prop)
    print(f"  wrote allocortex_edit_proposals.csv ({len(all_prop)} proposals; PROPOSALS ONLY)")
    # consistency csv
    ccols = ["hemi", "area", "n_verts", "dominant_cyto7_label", "cyto7_composition", "reference",
             "reference_sources", "dist_to_medial_wall_mm", "current_class", "candidate_class",
             "current_vs_candidate"]
    with open(args.out_derived / "periallocortex_consistency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ccols); w.writeheader(); w.writerows(all_cons)
    print(f"  wrote periallocortex_consistency.csv ({len(all_cons)} area rows)")

    if not args.no_figure:
        render_packet(version, per_hemi, args.out_fig / "allocortex_adjudication.png", args.dpi)
    update_report(args.out_fig / "REPORT.md", all_prop, all_cons)
    print("Done.")


if __name__ == "__main__":
    main()
