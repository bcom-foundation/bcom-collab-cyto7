"""Cingulate adjudication panel + candidate anterior->posterior correction.

Implements ``docs/SPEC_cingulate_adjudication.md``. Base map = **v4**
(``resources/cyto7_derived/pial.{lh,rh}.cyto7.v4.annot``). cyto7 stays 7 labels.
This produces an adjudication aid + a **candidate** correction for García-Cabezas
to approve/edit -- it does **not** overwrite v4.

Two parts, both on the 164k fsaverage surface:

**Part A -- cingulate-profile adjudication panel (both hemispheres).**
  Medial surface (inflated + pial), cyto7 types in greyscale underneath, the four
  Desikan cingulate subregions outlined (rostral/caudal anterior, posterior,
  isthmus); if present, the finer von Economo limbic parcellation is drawn as an
  alternative outline. Beside it, a table of cyto7-type composition per subregion
  with the canonical expectation, so the mismatch is obvious. Percentages are over
  the *total* Desikan-region vertices; medial-wall ("unknown") vertices are shown
  but ignored (never reassigned).
  Outputs ``figures/v9/change_history/cingulate_profile_{lh,rh}.png`` +
  ``cingulate_profile.csv``.

**Part B -- proposed correction (CANDIDATE, does not overwrite v4).**
  Implements the canonical gradient as a reviewable proposal:
    * Anterior (rostral + caudal anterior cingulate): keep **agranular**.
    * Posterior cingulate: reassign agranular -> **dysgranular**.
    * Isthmus / retrosplenial: reassign agranular -> **dysgranular** (first pass);
      leave existing eulaminate as is. The retrosplenial-granular subtlety
      (area 29) is FLAGGED for GC, never introduced automatically.
  Changes are restricted to the cingulate Desikan subregions. ``converge_topology``
  (strict R1) is re-run on the candidate so no |Δτ|>=2 skips are created, and
  ``audit_hemi`` confirms R1=0 and R2-R4 do not regress. Moved vertices are logged
  to ``v4_to_cingulate_candidate_changelog.csv``. A before/after figure and updated
  composition table are written.

Framing (stated in the outputs): this candidate encodes the mainstream
ACC-agranular / PCC-dysgranular / retrosplenial-dysgranular scheme. **Two calls are
explicitly García-Cabezas's to set:** (1) the exact anterior->posterior
(agranular->dysgranular) transition location along the cingulate; (2) how to treat
retrosplenial granular cortex (area 29). The candidate is a starting point to move,
not a final map.

Run::
    conda activate cyto7
    python scripts/cingulate_adjudication.py            # both parts
    python scripts/cingulate_adjudication.py --part A   # panel only
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cyto7_surface_io import REPO_ROOT
from audit_topology import adjacency, audit_hemi
from apply_rule_v3 import converge_topology
from make_presentation_figures import (
    CAMERA, HEMIS, LIGHT_BLUE, TYPE_GREY, TYPE_NAMES, lighting_normals,
    lit_panel, load_labels, load_surface, sulc_shading,
)

DERIVED = cfg.atlas_dir("fsaverage")
VE_DIR = cfg.data_dir() / "voneconomo"
OUT_DIR = cfg.results_dir("tables") / "change_history"

V4_TEMPLATE = str(DERIVED / "pial.{hemi}.cyto7.v4.annot")
CAND_TEMPLATE = str(DERIVED / "pial.{hemi}.cyto7.v4_cingulate_candidate.annot")

CODE_NAME = {0: "unknown", 1: "Allocortex", 2: "agranular", 3: "dysgranular",
             4: "eulaminate I", 5: "eulaminate II", 6: "eulaminate III",
             7: "koniocortex"}
TYPE_CODES = [1, 2, 3, 4, 5, 6, 7]

#: Desikan cingulate subregions (both hemispheres share these names).
ANT_SUBS = ("rostralanteriorcingulate", "caudalanteriorcingulate")
POST_SUBS = ("posteriorcingulate", "isthmuscingulate")
CING_SUBS = ANT_SUBS + POST_SUBS

#: Human-readable label + canonical cytoarchitectural expectation per subregion.
SUB_LABEL = {
    "rostralanteriorcingulate": "rostral-ant. (ACC)",
    "caudalanteriorcingulate": "caudal-ant. (MCC)",
    "posteriorcingulate": "posterior (PCC)",
    "isthmuscingulate": "isthmus (RSC)",
}
SUB_EXPECT = {
    "rostralanteriorcingulate": "agranular (24/25/33)",
    "caudalanteriorcingulate": "agranular→dysgranular (mid-cing.)",
    "posteriorcingulate": "dysgranular (23/31)",
    "isthmuscingulate": "dysgranular/granular (29/30)",
}
#: Outline colour per Desikan subregion.
SUB_COLOR = {
    "rostralanteriorcingulate": (0.90, 0.60, 0.00),   # orange
    "caudalanteriorcingulate": (0.85, 0.30, 0.10),    # red-orange
    "posteriorcingulate": (0.10, 0.35, 0.85),         # blue
    "isthmuscingulate": (0.55, 0.15, 0.75),           # purple
}

#: von Economo limbic (cingulate) areas -> alternative parcellation panel.
VE_LIMBIC = ("LA1", "LA2", "LC1", "LC2", "LC3", "LD", "LE")
VE_PALETTE = {
    "LA1": (0.90, 0.60, 0.00), "LA2": (0.95, 0.75, 0.30),
    "LC1": (0.10, 0.35, 0.85), "LC2": (0.20, 0.55, 0.90), "LC3": (0.40, 0.70, 0.95),
    "LD": (0.55, 0.15, 0.75), "LE": (0.75, 0.35, 0.55),
}

#: Highlight colour for changed vertices in the before/after figure.
C_CHANGED = (0.95, 0.05, 0.55)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def load_aparc(hemi: str, atlas: str = "aparc"):
    """Return (labels, names) for a FreeSurfer .annot on the 164k surface."""
    fname = f"{hemi}.aparc.annot" if atlas == "aparc" else f"{hemi}.economo.annot"
    lab, _ctab, names = nib.freesurfer.io.read_annot(str(VE_DIR / fname))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return np.asarray(lab), names


def sub_mask(alab: np.ndarray, anames: list[str], name: str) -> np.ndarray:
    """Boolean mask of one named parcel (empty if the name is absent)."""
    if name not in anames:
        return np.zeros(alab.shape[0], bool)
    return alab == anames.index(name)


def composition(lab: np.ndarray, mask: np.ndarray) -> dict:
    """cyto7 composition of *mask*. Percentages are over the TOTAL region
    vertices (matching the SPEC); the medial-wall 'unknown' count is reported
    separately but stays in the denominator."""
    total = int(mask.sum())
    counts = {c: int((mask & (lab == c)).sum()) for c in range(0, 8)}
    pct = {c: (100.0 * counts[c] / total if total else 0.0) for c in range(0, 8)}
    return {"total": total, "counts": counts, "pct": pct}


def boundary_band(mask: np.ndarray, A, width: int = 1) -> np.ndarray:
    """Vertices on the boundary of *mask*: the inner rim (inside, touching
    outside) plus the outer rim (outside, touching inside), optionally dilated by
    ``width-1`` hops for visibility. Baking the boundary into the surface vertex
    colours (rather than overlaying 3-D lines) avoids matplotlib's 3-D
    depth-sorting occlusion entirely -- the outline is part of the surface."""
    inside = mask
    outer = (A @ inside.astype(np.int8) > 0) & ~inside
    inner = (A @ (~inside).astype(np.int8) > 0) & inside
    band = outer | inner
    for _ in range(max(0, width - 1)):
        band = band | (A @ band.astype(np.int8) > 0)
    return band


def bake_outlines(rgb: np.ndarray, masks: dict, colors: dict, A,
                  width: int = 1) -> np.ndarray:
    """Overwrite *rgb* at each mask's boundary band with its outline colour.
    Later names win on overlap (Desikan subregions are disjoint, so order is
    immaterial there)."""
    for name, mask in masks.items():
        if mask.any():
            rgb[boundary_band(mask, A, width)] = np.array(colors[name])
    return rgb


def greyscale_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = np.tile(np.array(LIGHT_BLUE), (lab.shape[0], 1))
    for c in TYPE_CODES:
        rgb[lab == c] = np.array(TYPE_GREY[c])
    return rgb


# --------------------------------------------------------------------------- #
# Composition table (matplotlib) shared by Part A and Part B
# --------------------------------------------------------------------------- #

TABLE_TYPES = [1, 2, 3, 4, 5]  # allocortex..eulaminate II (koniocortex never in cingulate)


def _pct_cell(comp: dict, c: int) -> str:
    p = comp["pct"][c]
    return f"{p:.0f}%" if p >= 0.5 else "·"


def draw_composition_table(ax, lab: np.ndarray, alab, anames, hemi: str,
                           lab_after: np.ndarray | None = None):
    """Render a composition table for one hemisphere.

    If *lab_after* is given, each cell shows ``before -> after`` percentages.
    """
    ax.axis("off")
    header = ["subregion", "n", "unk"] + [CODE_NAME[c] for c in TABLE_TYPES] + \
             ["canonical expectation"]
    rows = []
    for name in CING_SUBS:
        m = sub_mask(alab, anames, name)
        comp = composition(lab, m)
        cells = [SUB_LABEL[name], str(comp["total"]), str(comp["counts"][0])]
        if lab_after is None:
            for c in TABLE_TYPES:
                cells.append(_pct_cell(comp, c))
        else:
            comp2 = composition(lab_after, m)
            for c in TABLE_TYPES:
                a, b = _pct_cell(comp, c), _pct_cell(comp2, c)
                cells.append(a if a == b else f"{a}→{b}")
        cells.append(SUB_EXPECT[name])
        rows.append(cells)
    n_types = len(TABLE_TYPES)
    col_widths = [0.155, 0.05, 0.05] + [0.072] * n_types + [0.265]
    tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center", loc="center",
                   colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.5)
    # tint the subregion name cell by its outline colour; left-align text columns
    for (r, cidx), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.92")
        if cidx == 0 and r > 0:
            name = CING_SUBS[r - 1]
            cell.set_facecolor(tuple(0.6 + 0.4 * np.array(SUB_COLOR[name])))
            cell.get_text().set_ha("left")
        if cidx == len(header) - 1 and r > 0:
            cell.get_text().set_ha("left")
            cell.set_text_props(fontsize=6.8)


# --------------------------------------------------------------------------- #
# Part A -- cingulate-profile panel
# --------------------------------------------------------------------------- #


def part_a(out_dir: Path, dpi: int, sulc_strength: float = 0.35) -> None:
    print("== Part A: cingulate-profile panel + CSV ==")
    # --- CSV (both hemispheres) ---
    csv_rows = []
    for hemi in HEMIS:
        lab = load_labels(V4_TEMPLATE, hemi)
        alab, anames = load_aparc(hemi)
        for name in CING_SUBS:
            comp = composition(lab, sub_mask(alab, anames, name))
            row = {"hemi": hemi, "subregion": name,
                   "canonical_expectation": SUB_EXPECT[name],
                   "total_vertices": comp["total"],
                   "unknown_vertices": comp["counts"][0]}
            for c in TYPE_CODES:
                row[f"{CODE_NAME[c]}_n"] = comp["counts"][c]
                row[f"{CODE_NAME[c]}_pct"] = round(comp["pct"][c], 1)
            csv_rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cingulate_profile.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  wrote {csv_path}")

    # --- per-hemisphere figure ---
    for hemi in HEMIS:
        lab = load_labels(V4_TEMPLATE, hemi)
        alab, anames = load_aparc(hemi)
        ve_lab, ve_names = load_aparc(hemi, atlas="economo")
        have_ve = ve_lab.shape[0] == lab.shape[0] and any(n in ve_names for n in VE_LIMBIC)
        vnorm = lighting_normals(hemi)
        _, faces_top = load_surface(hemi, "inflated")
        A = adjacency(faces_top, lab.shape[0])
        des_masks = {n: sub_mask(alab, anames, n) for n in CING_SUBS}
        ve_masks = ({n: sub_mask(ve_lab, ve_names, n) for n in VE_LIMBIC
                     if n in ve_names} if have_ve else {})

        panels = [("inflated", "Desikan"), ("pial", "Desikan")]
        if ve_masks:
            panels.append(("inflated", "vonEconomo"))
        ncol = len(panels)

        fig = plt.figure(figsize=(4.6 * ncol, 7.4))
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, ncol, height_ratios=[1.35, 1.0], hspace=0.08,
                              wspace=0.02)

        for ci, (surf, kind) in enumerate(panels):
            coords, faces = load_surface(hemi, surf)
            ss = sulc_shading(hemi, sulc_strength) if surf == "pial" else None
            ax = fig.add_subplot(gs[0, ci], projection="3d")
            rgb = greyscale_rgb(lab)
            if kind == "Desikan":
                bake_outlines(rgb, des_masks, SUB_COLOR, A, width=2)
                title = f"{hemi.upper()} medial — {surf}\nDesikan cingulate subregions"
            else:
                bake_outlines(rgb, ve_masks, VE_PALETTE, A, width=2)
                title = f"{hemi.upper()} medial — inflated\nvon Economo limbic (alt.)"
            lit_panel(ax, coords, faces, rgb, hemi, "medial", vnorm, sulc_shade=ss)
            ax.set_title(title, fontsize=11)

        ax_tbl = fig.add_subplot(gs[1, :])
        draw_composition_table(ax_tbl, lab, alab, anames, hemi)

        # legends
        des_handles = [Patch(facecolor=SUB_COLOR[n], edgecolor="0.3",
                             label=SUB_LABEL[n]) for n in CING_SUBS]
        if ve_masks:
            des_handles += [Patch(facecolor=VE_PALETTE[n], edgecolor="0.3", label=n)
                            for n in ve_masks]
        type_handles = [Patch(facecolor=TYPE_GREY[c], edgecolor="0.4",
                              label=TYPE_NAMES[c - 1]) for c in TYPE_CODES]
        leg1 = fig.legend(handles=des_handles, loc="upper center",
                          ncol=len(des_handles), fontsize=8, frameon=False,
                          bbox_to_anchor=(0.5, 0.06))
        fig.add_artist(leg1)
        fig.legend(handles=type_handles, loc="lower center", ncol=7, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(
            f"cyto7 v4 — cingulate profile ({hemi.upper()}): Desikan subregions vs "
            "canonical cytoarchitecture", fontsize=14, y=0.985)
        fig.text(0.5, 0.945,
                 "cyto7 greyscale underneath, subregion boundaries baked as coloured "
                 "bands; % over total Desikan-region vertices ('unk' = medial wall, "
                 "not reassigned). PCC over-assigns agranular vs the canonical "
                 "dysgranular expectation.",
                 ha="center", va="top", fontsize=8, color="0.35")
        out = out_dir / f"cingulate_profile_{hemi}.png"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# Part B -- candidate correction
# --------------------------------------------------------------------------- #


def _audit_scalars(lab, faces, A, coords) -> dict:
    res, _flags = audit_hemi(lab, faces, A, coords)
    r2_bad = sum(not v["single_annular_ring"]
                 for v in res["R2_core_continuity_rings"].values())
    return {
        "R1_skip_edges": int(res["R1_sequential_gradients"]["n_skip_edges"]),
        "R2_nonannular_rings": int(r2_bad),
        "R3_single_component": bool(res["R3_matrix_eulaminate_II"]["ok_single_component"]),
        "R4_annular_islands": int(sum(v["n_annular_violations"]
                                      for v in res["R4_islands"].values())),
    }


def build_candidate_hemi(hemi: str):
    """Return (v4, candidate, changelog_rows, info) for one hemisphere."""
    v4, ctab, names = nib.freesurfer.io.read_annot(str(Path(V4_TEMPLATE.format(hemi=hemi))))
    v4 = np.asarray(v4)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    coords, faces = load_surface(hemi, "pial")
    n = v4.shape[0]
    A = adjacency(faces, n)
    alab, anames = load_aparc(hemi)

    post_mask = np.zeros(n, bool)
    for name in POST_SUBS:
        post_mask |= sub_mask(alab, anames, name)
    cing_mask = post_mask.copy()
    for name in ANT_SUBS:
        cing_mask |= sub_mask(alab, anames, name)

    cand = v4.copy()
    # Posterior cingulate + isthmus: agranular (2) -> dysgranular (3). Anterior
    # kept agranular; existing eulaminate/allocortex untouched (retrosplenial
    # granular flagged for GC, not introduced here).
    reassign = post_mask & (v4 == 2)
    cand[reassign] = 3

    # Strict R1 repair: lock allocortex, grow a graded buffer until |Δτ|<=1.
    allo_locked = cand == 1
    cand, buf_changed, unresolved = converge_topology(cand, faces, allo_locked, max_pass=60)

    changed = np.where(cand != v4)[0]
    # verify the restriction: every changed vertex is inside the cingulate.
    outside = int((~cing_mask[changed]).sum())

    audit_v4 = _audit_scalars(v4, faces, A, coords)
    audit_cand = _audit_scalars(cand, faces, A, coords)

    # subregion lookup for the changelog
    sub_of = np.array(["outside-cingulate"] * n, dtype=object)
    for name in CING_SUBS:
        sub_of[sub_mask(alab, anames, name)] = name

    rows = []
    for v in changed:
        reason = ("posterior->dysgranular" if reassign[v]
                  else "topology_buffer" if buf_changed[v] else "topology")
        rows.append((hemi, int(v), CODE_NAME[int(v4[v])], CODE_NAME[int(cand[v])],
                     sub_of[v], reason))

    info = {
        "hemi": hemi, "reassigned": int(reassign.sum()),
        "buffer": int(buf_changed.sum()), "changed": int(changed.size),
        "outside_cingulate": outside, "unresolved_skips": int(len(unresolved)),
        "audit_v4": audit_v4, "audit_cand": audit_cand,
        "ctab": ctab, "names": names, "v4": v4, "cand": cand, "changed_idx": changed,
        "alab": alab, "anames": anames, "rows": rows,
    }
    return info


def part_b(out_dir: Path, dpi: int, write_annot: bool, sulc_strength: float = 0.35) -> None:
    print("== Part B: candidate correction (does NOT overwrite v4) ==")
    infos = {}
    all_rows = []
    for hemi in HEMIS:
        info = build_candidate_hemi(hemi)
        infos[hemi] = info
        all_rows += [dict(zip(
            ("hemi", "vertex", "old_label", "new_label", "subregion", "reason"), r))
            for r in info["rows"]]
        print(f"  {hemi}: reassigned(agranular->dysgranular)={info['reassigned']} "
              f"buffer={info['buffer']} total_changed={info['changed']} "
              f"outside_cingulate={info['outside_cingulate']} "
              f"unresolved_skips={info['unresolved_skips']}")
        print(f"       audit v4:  {info['audit_v4']}")
        print(f"       audit cand:{info['audit_cand']}")
        if info["outside_cingulate"] > 0:
            print(f"  !! {hemi}: {info['outside_cingulate']} changed vertices fell "
                  "OUTSIDE the cingulate — investigate before use.")
        if info["audit_cand"]["R1_skip_edges"] != 0 or info["unresolved_skips"] != 0:
            print(f"  !! {hemi}: strict R1 NOT clean (skip_edges="
                  f"{info['audit_cand']['R1_skip_edges']}, unresolved="
                  f"{info['unresolved_skips']}) — FLAGGED.")

    # --- changelog ---
    out_dir.mkdir(parents=True, exist_ok=True)
    clog = DERIVED / "v4_to_cingulate_candidate_changelog.csv"
    with open(clog, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["hemi", "vertex", "old_label",
                                          "new_label", "subregion", "reason"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"  wrote {clog} ({len(all_rows)} rows)")

    # --- candidate composition CSV (before/after) ---
    comp_rows = []
    for hemi in HEMIS:
        info = infos[hemi]
        for name in CING_SUBS:
            m = sub_mask(info["alab"], info["anames"], name)
            cb, ca = composition(info["v4"], m), composition(info["cand"], m)
            row = {"hemi": hemi, "subregion": name, "total_vertices": cb["total"]}
            for c in TYPE_CODES:
                row[f"{CODE_NAME[c]}_pct_v4"] = round(cb["pct"][c], 1)
                row[f"{CODE_NAME[c]}_pct_candidate"] = round(ca["pct"][c], 1)
            comp_rows.append(row)
    comp_path = out_dir / "cingulate_candidate_composition.csv"
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)
    print(f"  wrote {comp_path}")

    # --- write candidate annots (v4 untouched) ---
    if write_annot:
        for hemi in HEMIS:
            info = infos[hemi]
            out = Path(CAND_TEMPLATE.format(hemi=hemi))
            nm = [n.encode() if isinstance(n, str) else n for n in info["names"]]
            nib.freesurfer.io.write_annot(str(out), info["cand"].astype(np.int32),
                                          info["ctab"], nm, fill_ctab=True)
            print(f"  wrote {out.name} (candidate; v4 left untouched)")

    # --- framing / flags note ---
    write_framing_note(out_dir, infos)

    # --- before/after figure ---
    render_before_after(infos, out_dir, dpi, sulc_strength)


def write_framing_note(out_dir: Path, infos: dict) -> None:
    lines = [
        "cyto7 — cingulate adjudication: CANDIDATE anterior->posterior correction",
        "=" * 72,
        "Base map: v4 (pial.{lh,rh}.cyto7.v4.annot). v4 is NOT modified.",
        "Candidate: pial.{lh,rh}.cyto7.v4_cingulate_candidate.annot (reviewable proposal).",
        "",
        "What the candidate does (canonical gradient):",
        "  - Anterior cingulate (rostral + caudal anterior): kept AGRANULAR.",
        "  - Posterior cingulate: agranular -> DYSGRANULAR (recover PCC for dysgranular).",
        "  - Isthmus/retrosplenial: agranular -> DYSGRANULAR (first pass); existing",
        "    eulaminate left as is.",
        "  - Changes restricted to the four Desikan cingulate subregions.",
        "  - Strict R1 re-run (converge_topology): a graded agranular seam is auto-",
        "    inserted wherever new dysgranular would abut allocortex, so |Δτ|<=1 holds.",
        "",
        "This encodes the mainstream ACC-agranular / PCC-dysgranular / retrosplenial-",
        "dysgranular scheme (von Economo–Koskinas; Vogt; Structural Model).",
        "",
        "*** TWO CALLS ARE EXPLICITLY GARCÍA-CABEZAS'S TO SET ***",
        "  (1) The exact anterior->posterior (agranular->dysgranular) TRANSITION",
        "      LOCATION along the cingulate. The candidate places it at the Desikan",
        "      caudal-anterior | posterior boundary; this is a starting point to move.",
        "  (2) How to treat RETROSPLENIAL GRANULAR cortex (AREA 29). The candidate does",
        "      NOT introduce koniocortex/granular in the isthmus automatically; the",
        "      retrosplenial-granular subtlety is FLAGGED for GC to decide.",
        "",
        "Per-hemisphere summary:",
    ]
    for hemi in HEMIS:
        info = infos[hemi]
        lines.append(
            f"  {hemi}: agranular->dysgranular reassigned={info['reassigned']}, "
            f"topology buffer={info['buffer']}, total changed={info['changed']}, "
            f"outside-cingulate={info['outside_cingulate']}; "
            f"R1 v4={info['audit_v4']['R1_skip_edges']} -> "
            f"cand={info['audit_cand']['R1_skip_edges']}.")
    lines += [
        "",
        "R1-R4 audit (v4 vs candidate) — R1 must be 0 on the candidate and R2-R4 must",
        "not regress:",
    ]
    for hemi in HEMIS:
        info = infos[hemi]
        lines.append(f"  {hemi} v4:   {info['audit_v4']}")
        lines.append(f"  {hemi} cand: {info['audit_cand']}")
    lines += [
        "",
        "Changelog: v4_to_cingulate_candidate_changelog.csv (every moved vertex).",
        "Composition: cingulate_candidate_composition.csv (before/after, per subregion).",
        "The candidate is a starting point to move, not a final map.",
    ]
    note = out_dir / "cingulate_candidate_README.txt"
    note.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {note}")


def render_before_after(infos: dict, out_dir: Path, dpi: int,
                        sulc_strength: float = 0.35) -> None:
    print("  rendering before/after figure ...")
    vnorm = {h: lighting_normals(h) for h in HEMIS}
    geom = {(h, s): load_surface(h, s) for h in HEMIS for s in ("inflated", "pial")}
    sulc = {h: sulc_shading(h, sulc_strength) for h in HEMIS}

    fig = plt.figure(figsize=(11, 17.5))
    fig.patch.set_facecolor("white")
    sub = fig.subfigures(2, 1, height_ratios=[3.0, 1.15], hspace=0.02)

    # rows: (hemi, surf); cols: v4 (before), candidate (after)
    rows = [(h, s) for h in HEMIS for s in ("inflated", "pial")]
    axes = sub[0].subplots(len(rows), 2, subplot_kw={"projection": "3d"})
    A = {h: adjacency(geom[(h, "inflated")][1], infos[h]["v4"].shape[0]) for h in HEMIS}
    for r, (hemi, surf) in enumerate(rows):
        coords, faces = geom[(hemi, surf)]
        info = infos[hemi]
        alab, anames = info["alab"], info["anames"]
        des_masks = {n: sub_mask(alab, anames, n) for n in CING_SUBS}
        ss = sulc[hemi] if surf == "pial" else None
        changed_mask = np.zeros(info["v4"].shape[0], bool)
        changed_mask[info["changed_idx"]] = True
        for c, (lab, ttl) in enumerate(((info["v4"], "v4 (current)"),
                                        (info["cand"], "candidate"))):
            ax = axes[r, c]
            rgb = greyscale_rgb(lab)
            if c == 1:  # highlight changed vertices on the candidate panel
                rgb[changed_mask] = C_CHANGED
            bake_outlines(rgb, des_masks, SUB_COLOR, A[hemi], width=2)
            lit_panel(ax, coords, faces, rgb, hemi, "medial", vnorm[hemi], sulc_shade=ss)
            if r == 0:
                ax.set_title(ttl, fontsize=13, weight="bold")
            if c == 0:
                ax.text2D(-0.04, 0.5, f"{hemi.upper()} {surf}", transform=ax.transAxes,
                          rotation=90, va="center", ha="center", fontsize=11,
                          weight="bold")

    # tables (before -> after), one per hemi, stacked full-width so the
    # 'v4→cand' cells do not overflow.
    tbl_axes = sub[1].subplots(2, 1)
    sub[1].subplots_adjust(left=0.06, right=0.97, hspace=0.55, top=0.9, bottom=0.05)
    for ax, hemi in zip(tbl_axes, HEMIS):
        info = infos[hemi]
        draw_composition_table(ax, info["v4"], info["alab"], info["anames"], hemi,
                               lab_after=info["cand"])
        ax.set_title(f"{hemi.upper()} composition  (v4 → candidate)", fontsize=10,
                     weight="bold")

    handles = [Patch(facecolor=TYPE_GREY[c], edgecolor="0.4", label=TYPE_NAMES[c - 1])
               for c in TYPE_CODES]
    handles += [Patch(facecolor=C_CHANGED, edgecolor="0.4", label="changed vertices")]
    handles += [Line2D([0], [0], color=SUB_COLOR[n], lw=2.2, label=SUB_LABEL[n])
                for n in CING_SUBS]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.008))
    fig.suptitle("cyto7 cingulate — v4 (current) vs candidate correction "
                 "(agranular→dysgranular in PCC + isthmus)", fontsize=14, y=0.997)
    fig.text(0.5, 0.978,
             "CANDIDATE for García-Cabezas. Transition location and retrosplenial-"
             "granular (area 29) are GC's calls — see cingulate_candidate_README.txt.",
             ha="center", va="top", fontsize=8.5, color="0.35")
    out = out_dir / "cingulate_candidate_before_after.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["A", "B", "both"], default="both")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--sulc-shading", type=float, default=0.35)
    p.add_argument("--no-write-annot", action="store_true",
                   help="Part B: skip writing the candidate .annot files.")
    args = p.parse_args(argv)

    if args.part in ("A", "both"):
        part_a(args.out, args.dpi, args.sulc_shading)
    if args.part in ("B", "both"):
        part_b(args.out, args.dpi, not args.no_write_annot, args.sulc_shading)
    print("Done.")


if __name__ == "__main__":
    main()
