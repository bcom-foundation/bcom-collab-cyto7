"""Build the cyto7 **v4** map: expand surface allocortex to the medial-temporal
periarchicortex, move perirhinal to mesocortex, and enforce a STRICT sequential
gradient (SPEC_v4_map_revision.md / GC_MEETING_v4_plan.md).

Starting from the released v3_clean map, per hemisphere:

  1. **Entorhinal (EC) -> allocortex (1).** ex-vivo EC label (Fischl 2009) UNION
     Desikan ``entorhinal``.
  2. **Presubiculum + parasubiculum -> allocortex (1).** No clean fsaverage
     surface label exists, so operationalised as the **parahippocampal band
     within ``--presub-hops`` graph-hops of the medial wall** (Destrieux
     ``G_oc-temp_med-Parahip``), i.e. the periarchicortical strip bordering the
     medial wall / hippocampal fissure, medial to EC, minus EC. DOCUMENTED
     APPROXIMATION (flagged in PROVENANCE_v4.txt).
  3. **Piriform (olfactory allocortex) -> allocortex (1).** Reuse the v3_clean
     piriform sliver (already code 1).
  4. **Perirhinal (BA35/36) -> mesocortex.** ex-vivo perirhinal MINUS the allo set;
     each vertex -> agranular (2) if graph-closer to allocortex, else dysgranular
     (3) toward isocortex (adjacency decides per vertex).
  5. **Strict R1 repair.** With the full allocortex set locked, grow a graded
     mesocortex buffer (``converge_topology``) until every edge has |Δτ| <= 1
     (zero skips) -- allocortex never abuts isocortex directly.

Outputs: ``resources/cyto7_derived/pial.{lh,rh}.cyto7.v4.annot`` (+ ctab preserved),
``PROVENANCE_v4.txt``, ``v3clean_to_v4_changelog.csv``.

Run::
    conda activate cyto7
    python scripts/build_v4_map.py
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import csv
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from scipy.sparse import csr_matrix

from cyto7_surface_io import REPO_ROOT
from audit_topology import adjacency, audit_hemi, ordinal
from apply_rule_v3 import converge_topology

DERIVED = cfg.atlas_dir("fsaverage")
VE_DIR = cfg.data_dir() / "voneconomo"
ATLAS_DIR = cfg.data_dir() / "refine_atlases"
SURF_DIR = cfg.data_dir() / "fsaverage_surfaces"
HEMIS = ("lh", "rh")
CODE_NAME = {0: "unknown", 1: "Allocortex", 2: "agranular", 3: "dysgranular",
             4: "eulaminate I", 5: "eulaminate II", 6: "eulaminate III", 7: "koniocortex"}
ISO_CODES = (4, 5, 6, 7)


def _annot(path):
    lab, ctab, names = nib.freesurfer.io.read_annot(str(path))
    return np.asarray(lab), ctab, [n.decode() if isinstance(n, bytes) else n for n in names]


def _label_mask(path, n):
    m = np.zeros(n, bool)
    m[nib.freesurfer.io.read_label(str(path))] = True
    return m


def _parcel_mask(annot_path, target_names, n):
    lab, _c, names = _annot(annot_path)
    idx = [i for i, nm in enumerate(names) if nm in target_names]
    return np.isin(lab, idx) if idx else np.zeros(n, bool)


def _hops_from(A: csr_matrix, seed: np.ndarray) -> np.ndarray:
    """Unweighted BFS graph-hop distance from any seed vertex (-1 if unreached)."""
    n = A.shape[0]
    dist = np.full(n, -1, int)
    dist[seed] = 0
    frontier = seed.copy()
    d = 0
    while frontier.any():
        d += 1
        nxt = (A @ frontier.astype(np.int8) > 0) & (dist < 0)
        dist[nxt] = d
        frontier = nxt
    return dist


def build_hemi(hemi: str, args) -> dict:
    v3, ctab, names = _annot(DERIVED / f"pial.{hemi}.cyto7.v3_clean.annot")
    n = v3.shape[0]
    coords, faces = nib.freesurfer.read_geometry(str(SURF_DIR / f"{hemi}.pial"))
    faces = np.asarray(faces, np.int64)
    A = adjacency(faces, n)
    lab = v3.copy()
    labelled = v3 >= 1

    # --- allocortex additions ---
    ec = (_label_mask(ATLAS_DIR / f"{hemi}.entorhinal_exvivo.label", n)
          | _parcel_mask(VE_DIR / f"{hemi}.aparc.annot", {"entorhinal"}, n)) & labelled
    medial = v3 == 0
    hops_med = _hops_from(A, medial)
    parahip = _parcel_mask(ATLAS_DIR / f"{hemi}.aparc.a2009s.annot", {"G_oc-temp_med-Parahip"}, n)
    presub = parahip & (hops_med >= 1) & (hops_med <= args.presub_hops) & labelled & ~ec
    piri = v3 == 1  # existing v3_clean piriform sliver
    allo = (ec | presub | piri) & labelled

    # --- perirhinal -> mesocortex (2/3 by adjacency), excluding the allo set ---
    peri = (_label_mask(ATLAS_DIR / f"{hemi}.perirhinal_exvivo.label", n)) & labelled & ~allo
    dist_allo = _hops_from(A, allo)
    dist_iso = _hops_from(A, np.isin(v3, ISO_CODES))
    peri_code = np.where(dist_allo <= dist_iso, 2, 3)  # allo-facing -> agranular

    # apply relabels
    lab[allo] = 1
    lab[peri] = peri_code[peri]

    # --- strict R1 repair: lock the whole allocortex, grow graded meso buffer ---
    allo_locked = lab == 1
    lab, buf_changed, unresolved = converge_topology(lab, faces, allo_locked, max_pass=args.max_pass)

    # audit
    res, _flags = audit_hemi(lab, faces, A, coords)
    r1 = res["R1_sequential_gradients"]["n_skip_edges"]

    changed = np.where((lab != v3) & labelled)[0]
    return {
        "hemi": hemi, "v3": v3, "v4": lab, "ctab": ctab, "names": names,
        "n_ec": int(ec.sum()), "n_presub": int(presub.sum()), "n_piri": int(piri.sum()),
        "n_allo_v4": int((lab == 1).sum()), "n_peri": int(peri.sum()),
        "n_buffer": int(buf_changed.sum()), "n_changed": int(changed.size),
        "r1_after": int(r1), "unresolved": int(len(unresolved)), "changed_idx": changed,
        "res": res,
    }


def write_annot(path, labels, ctab, names):
    nm = [n.encode() if isinstance(n, str) else n for n in names]
    nib.freesurfer.io.write_annot(str(path), labels.astype(np.int32), ctab, nm, fill_ctab=True)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--presub-hops", type=int, default=3,
                   help="parahippocampal band width (hops from medial wall) used as the "
                        "presubiculum/parasubiculum proxy (default 3).")
    p.add_argument("--max-pass", type=int, default=60)
    args = p.parse_args(argv)

    results = {}
    changelog_rows = []
    for hemi in HEMIS:
        print(f"== {hemi} ==")
        r = build_hemi(hemi, args)
        results[hemi] = r
        write_annot(DERIVED / f"pial.{hemi}.cyto7.v4.annot", r["v4"], r["ctab"], r["names"])
        print(f"  allo: EC={r['n_ec']} presub-proxy={r['n_presub']} piriform={r['n_piri']} "
              f"-> v4 allocortex={r['n_allo_v4']} (was {int((r['v3']==1).sum())})")
        print(f"  perirhinal->meso={r['n_peri']}  strict-buffer moved={r['n_buffer']}  "
              f"total changed={r['n_changed']}  R1 after={r['r1_after']}  unresolved={r['unresolved']}")
        for v in r["changed_idx"]:
            changelog_rows.append({"hemi": hemi, "vertex": int(v),
                                   "v3_clean": CODE_NAME[int(r["v3"][v])],
                                   "v4": CODE_NAME[int(r["v4"][v])]})

    with open(DERIVED / "v3clean_to_v4_changelog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["hemi", "vertex", "v3_clean", "v4"])
        w.writeheader(); w.writerows(changelog_rows)
    print(f"\nchangelog: {len(changelog_rows)} vertices -> {DERIVED/'v3clean_to_v4_changelog.csv'}")

    _write_provenance(results, args)
    ok = all(r["r1_after"] == 0 and r["unresolved"] == 0 for r in results.values())
    print(f"\nSTRICT R1 == 0 both hemis: {ok}")


def _write_provenance(results, args):
    L = ["cyto7 v4 — allocortex expansion + strict sequential gradient",
         "=" * 60,
         "Source: pial.{lh,rh}.cyto7.v3_clean.annot -> pial.{lh,rh}.cyto7.v4.annot",
         "Framework: Zaldívar-Díez & García-Cabezas 2026 (GC meeting, July 2026).",
         "Tiers: allocortex=1; mesocortex=2-3; isocortex=4-7. 7-type ordinal scale kept.",
         "",
         "Allocortex membership (surface):",
         "  - Entorhinal (EC): FreeSurfer ex-vivo EC label ∪ Desikan 'entorhinal'.",
         f"  - Presubiculum+parasubiculum: APPROXIMATION (no clean fsaverage label) = "
         f"Destrieux 'G_oc-temp_med-Parahip' within {args.presub_hops} graph-hops of the "
         "medial wall (periarchicortical strip bordering the hippocampal fissure, medial to "
         "EC), minus EC. *** FLAGGED: proxy, pending a dedicated presub/parasub surface label.",
         "  - Piriform (olfactory allocortex): retained from v3_clean.",
         "Perirhinal (BA35/36) -> mesocortex: ex-vivo perirhinal minus the allo set; per-vertex "
         "agranular(2) if graph-closer to allocortex else dysgranular(3) toward isocortex.",
         "",
         "Strict gradient (R1): converge_topology grows a graded mesocortex buffer (lower the "
         "higher endpoint of every |Δτ|>=2 skip by one ordinal per pass; locked allocortex never "
         "moved) until every edge has |Δτ|<=1. Allocortex therefore never abuts isocortex.",
         "",
         "Per-hemisphere summary (vertices):",
    ]
    for h, r in results.items():
        L.append(f"  {h}: allocortex {int((r['v3']==1).sum())} -> {r['n_allo_v4']} "
                 f"(EC {r['n_ec']}, presub-proxy {r['n_presub']}, piriform {r['n_piri']}); "
                 f"perirhinal->meso {r['n_peri']}; strict-buffer {r['n_buffer']}; "
                 f"total changed {r['n_changed']}; R1 skip-edges after = {r['r1_after']}.")
    L.append("")
    L.append("Changelog: v3clean_to_v4_changelog.csv (every moved vertex).")
    (DERIVED / "PROVENANCE_v4.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {DERIVED/'PROVENANCE_v4.txt'}")


if __name__ == "__main__":
    main()
