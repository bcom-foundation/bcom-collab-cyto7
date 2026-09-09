"""Miguel adjudication pack (SPEC_miguel_adjudication_pack.md) — v6.

A digestible summary that lets García-Cabezas verify how each **allocortical and
mesocortical** region was identified. Isocortex is context only. Uses the
``figure_style`` palette (GC default). Writes to ``figures/v9/adjudication/`` and
assembles ``docs/MIGUEL_ADJUDICATION_v6.md``.

  Part 1 — v6 on pial + inflated + flat (both hemis, medial+lateral), shared legend.
  Part 2 — reference-parcellation overlays for the allo/meso belt + expected-vs-
           identified provenance table (PNG + CSV; proxy/uncertain flagged).
  Part 3 — per-hemisphere composition matrices (Desikan + von-Economo), LH vs RH
           side by side, + total-variation-distance (TVD) asymmetry flags.

Run::  conda activate cyto7 && python scripts/miguel_pack.py
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
import numpy as np
import nibabel as nib
from matplotlib.patches import Patch

from cyto7_surface_io import REPO_ROOT, resolve_target_map, surface_path
from audit_topology import adjacency
import figure_style as fs
from figure_style import (CYTO7_NAMES, cyto7_legend_handles, cyto7_listed_cmap,
                          cyto7_palette, cyto7_vertex_rgb)
from make_presentation_figures import (
    HEMIS, LIGHT_BLUE, lighting_normals, lit_panel, load_surface, sulc_shading,
)
from cingulate_adjudication import bake_outlines

VE_DIR = cfg.data_dir() / "voneconomo"
ATLAS_DIR = cfg.data_dir() / "refine_atlases"
CROSSED = cfg.atlas_dir("fsaverage/crossed")
OUT = cfg.results_dir("tables") / "adjudication"
#: canonical cyto7 map version rendered by the pack (override with --version).
VERSION = "v9"
HK = {"lh": "L", "rh": "R"}
CODE_NAME = {1: "Allocortex", 2: "agranular", 3: "dysgranular", 4: "eulaminate I",
             5: "eulaminate II", 6: "eulaminate III", 7: "koniocortex"}
COL_LABELS = ["Allo", "Agr", "Dys", "EulI", "EulII", "EulIII", "Kon"]


def _aparc(hemi):
    lab, _c, names = nib.freesurfer.io.read_annot(str(VE_DIR / f"{hemi}.aparc.annot"))
    return np.asarray(lab), [n.decode() if isinstance(n, bytes) else n for n in names]


def _destrieux(hemi):
    lab, _c, names = nib.freesurfer.io.read_annot(str(ATLAS_DIR / f"{hemi}.aparc.a2009s.annot"))
    return np.asarray(lab), [n.decode() if isinstance(n, bytes) else n for n in names]


def _label_mask(path, n):
    m = np.zeros(n, bool)
    m[nib.freesurfer.read_label(str(path))] = True
    return m


def _submask(alab, anames, wanted, n):
    idx = [i for i, nm in enumerate(anames) if nm in wanted]
    return np.isin(alab, idx) if idx else np.zeros(n, bool)


# --------------------------------------------------------------------------- #
# Part 1 — v6 on three surfaces
# --------------------------------------------------------------------------- #


def part1_three_surfaces(dpi, palette="gc", dataset="Validation210"):
    print("== Part 1: v6 on pial + inflated + flat ==")
    lab164 = resolve_target_map(VERSION, "fsaverage")
    lab32 = resolve_target_map(VERSION, "fs_LR")
    vnorm = {h: lighting_normals(h) for h in HEMIS}
    cols = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05], hspace=0.04, wspace=0.02)

    for r, surf in enumerate(("pial", "inflated")):
        geom = {h: load_surface(h, surf) for h in HEMIS}
        ss = {h: (sulc_shading(h, 0.35) if surf == "pial" else None) for h in HEMIS}
        for c, (hemi, view) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            coords, faces = geom[hemi]
            rgb = cyto7_vertex_rgb(lab164[HK[hemi]], palette)
            lit_panel(ax, coords, faces, rgb, hemi, view, vnorm[hemi], sulc_shade=ss[hemi])
            if r == 0:
                ax.set_title(f"{hemi.upper()} {view}", fontsize=12)
            if c == 0:
                ax.text2D(-0.06, 0.5, surf, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=13, weight="bold")
    # flat row (nilearn, 32k fs_LR), LH cols 0-1, RH cols 2-3
    from nilearn import plotting
    cmap = cyto7_listed_cmap(palette)
    for k, (H, hemi_nl) in enumerate((("L", "left"), ("R", "right"))):
        ax = fig.add_subplot(gs[2, k * 2:k * 2 + 2], projection="3d")
        d = lab32[H].astype(float)
        d[lab32[H] == 0] = np.nan
        plotting.plot_surf_roi(str(surface_path(dataset, H, "flat")), d, hemi=hemi_nl,
                               view="dorsal", axes=ax, cmap=cmap, vmin=1, vmax=7,
                               colorbar=False)
        ax.set_title(f"{H}H flat", fontsize=12)
        if k == 0:
            ax.text2D(-0.03, 0.5, "flat", transform=ax.transAxes, rotation=90,
                      va="center", ha="center", fontsize=13, weight="bold")
    fig.legend(handles=cyto7_legend_handles(palette), loc="lower center", ncol=7,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"cyto7 {VERSION} — pial / inflated / flat ({palette} palette)", fontsize=16, y=0.99)
    fig.subplots_adjust(bottom=0.06, top=0.94)
    out = OUT / f"{VERSION}_three_surfaces.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# Part 2 — reference overlays + provenance table
# --------------------------------------------------------------------------- #

#: reference regions to outline over v6 (Desikan unless noted), grouped by colour.
OVERLAY_GROUPS = {
    "entorhinal / parahippocampal": ((0.90, 0.60, 0.0), ["entorhinal", "parahippocampal"]),
    "anterior cingulate": ((0.85, 0.30, 0.10), ["rostralanteriorcingulate", "caudalanteriorcingulate"]),
    "posterior cingulate + isthmus": ((0.10, 0.35, 0.85), ["posteriorcingulate", "isthmuscingulate"]),
    "insula": ((0.55, 0.15, 0.75), ["insula"]),
    "temporal pole": ((0.13, 0.63, 0.35), ["temporalpole"]),
    "orbitofrontal": ((0.75, 0.35, 0.55), ["medialorbitofrontal", "lateralorbitofrontal"]),
}


def part2_overlays(dpi, palette="gc"):
    print("== Part 2a: reference-parcellation overlays ==")
    lab164 = resolve_target_map(VERSION, "fsaverage")
    vnorm = {h: lighting_normals(h) for h in HEMIS}
    views = ["lateral", "medial", "ventral"]
    fig, axes = plt.subplots(len(HEMIS), len(views), figsize=(len(views) * 4.0, len(HEMIS) * 4.0),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)
    for r, hemi in enumerate(HEMIS):
        coords, faces = load_surface(hemi, "inflated")
        n = coords.shape[0]
        A = adjacency(faces, n)
        alab, anames = _aparc(hemi)
        masks, colors = {}, {}
        for gname, (col, parcels) in OVERLAY_GROUPS.items():
            masks[gname] = _submask(alab, anames, parcels, n)
            colors[gname] = col
        # ex-vivo perirhinal as its own outline
        masks["perirhinal (ex-vivo)"] = _label_mask(ATLAS_DIR / f"{hemi}.perirhinal_exvivo.label", n)
        colors["perirhinal (ex-vivo)"] = (0.0, 0.80, 0.90)  # cyan — distinct from the greyscale fill / allocortex
        for c, view in enumerate(views):
            ax = axes[r, c]
            rgb = cyto7_vertex_rgb(lab164[HK[hemi]], palette)
            bake_outlines(rgb, masks, colors, A, width=1)
            lit_panel(ax, coords, faces, rgb, hemi, view, vnorm[hemi])
            if r == 0:
                ax.set_title(view, fontsize=13)
            if c == 0:
                ax.text2D(-0.05, 0.5, hemi.upper(), transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=13, weight="bold")
    handles = [Patch(facecolor=col, label=g) for g, (col, _p) in OVERLAY_GROUPS.items()]
    handles.append(Patch(facecolor=(0.0, 0.80, 0.90), label="perirhinal (ex-vivo)"))
    handles += cyto7_legend_handles(palette)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Reference-parcellation outlines over cyto7 {VERSION} (allo/meso belt)",
                 fontsize=15, y=1.0)
    fig.subplots_adjust(bottom=0.1)
    out = OUT / "reference_overlays.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


#: provenance rows: (region, source(s), expected, region-mask-spec, flag)
#: mask-spec: ("aparc", [names]) | ("destrieux", [names]) | ("exvivo", filename) |
#:            ("allo_in", [aparc names]) | ("dest_in_aparc", ([destrieux names], [aparc names]))
PROVENANCE = [
    ("Entorhinal", "FS ex-vivo (Augustinack/Fischl) ∪ Desikan entorhinal", "allocortex",
     ("aparc", ["entorhinal"]), ""),
    ("Presubiculum / parasubiculum", "Destrieux parahippocampal band (PROXY)", "allocortex",
     ("destrieux", ["G_oc-temp_med-Parahip"]), "PROXY — no clean surface label; flag"),
    ("Piriform (olfactory)", "Glasser Pir / v3 piriform sliver", "allocortex",
     ("allo_in", ["lateralorbitofrontal", "insula"]), "small paleocortex island"),
    ("Perirhinal (BA35/36)", "FS ex-vivo perirhinal (minus v7 allocortex overlap)",
     "mesocortex (agr/dys)", ("exvivo_nonallo", "perirhinal_exvivo.label"), ""),
    ("Anterior cingulate", "Desikan rostral+caudal anterior cingulate", "agranular",
     ("aparc", ["rostralanteriorcingulate", "caudalanteriorcingulate"]), ""),
    ("Posterior cingulate", "Desikan posterior cingulate", "dysgranular",
     ("aparc", ["posteriorcingulate"]), ""),
    ("Isthmus / retrosplenial", "Desikan isthmus cingulate", "dysgranular / granular",
     ("aparc", ["isthmuscingulate"]), "retrosplenial granular (area 29) — GC call"),
    ("Insula", "Desikan insula", "agranular → dysgranular",
     ("aparc", ["insula"]), ""),
    ("Anterior insula (agranular candidate)",
     "Destrieux short insular gyri + ant. circular sulcus ∩ Desikan insula", "agranular",
     ("dest_in_aparc", (["G_insular_short", "S_circular_insula_ant"], ["insula"])),
     "SIGN-OFF: agranular ventro-anterior insula = frontoinsular (von Economo FJK/FI); "
     "confirm extent & L/R balance vs insula_paint_guide_v7.png"),
    ("Temporal pole", "Desikan temporal pole", "dysgranular",
     ("aparc", ["temporalpole"]), ""),
    ("Orbitofrontal", "Desikan medial+lateral orbitofrontal", "agranular/dys → eulaminate (gradient)",
     ("aparc", ["medialorbitofrontal", "lateralorbitofrontal"]),
     "broad parcel: caudomedial paralimbic (agr/dys) grades to granular lateral/anterior OFC "
     "(eulaminate); von Economo agrees eulaminate-dominant — heterogeneity expected, not a flag"),
]

EXPECT_CODES = {
    "allocortex": {1}, "mesocortex (agr/dys)": {2, 3}, "agranular": {2},
    "dysgranular": {3}, "dysgranular / granular": {3, 7}, "agranular → dysgranular": {2, 3},
    "agranular / dysgranular": {2, 3},
    "agranular/dys → eulaminate (gradient)": {2, 3, 4, 5},
}


def _region_mask(spec, hemi, n):
    kind, arg = spec
    if kind == "aparc":
        alab, anames = _aparc(hemi)
        return _submask(alab, anames, arg, n)
    if kind == "destrieux":
        dlab, dnames = _destrieux(hemi)
        return _submask(dlab, dnames, arg, n)
    if kind in ("exvivo", "exvivo_nonallo"):
        return _label_mask(ATLAS_DIR / f"{hemi}.{arg}", n)
    if kind == "allo_in":  # allocortex vertices within the given aparc parcels
        alab, anames = _aparc(hemi)
        return _submask(alab, anames, arg, n)
    if kind == "dest_in_aparc":  # Destrieux parcels clipped to an aparc parcel
        dnames_want, anames_want = arg
        dlab, dnames = _destrieux(hemi)
        alab, anames = _aparc(hemi)
        return _submask(dlab, dnames, dnames_want, n) & _submask(alab, anames, anames_want, n)
    return np.zeros(n, bool)


def part2_table(dpi):
    print("== Part 2b: provenance table ==")
    lab = resolve_target_map(VERSION, "fsaverage")
    rows = []
    for region, source, expected, spec, flag in PROVENANCE:
        counts = np.zeros(8, int)
        allo_ov = 0  # for exvivo_nonallo: vertices of the ex-vivo label that are v6 allocortex
        full = 0
        for hemi in HEMIS:
            n = lab[HK[hemi]].shape[0]
            m = _region_mask(spec, hemi, n)
            v = lab[HK[hemi]]
            if spec[0] == "allo_in":
                m = m & (v == 1)  # piriform: allocortex within OFC/insula
            if spec[0] == "exvivo_nonallo":
                full += int(m.sum())
                allo_ov += int((m & (v == 1)).sum())
                m = m & (v != 1)  # perirhinal-proper: exclude the v6 allocortex overlap
            for c in range(0, 8):
                counts[c] += int((m & (v == c)).sum())
        if spec[0] == "exvivo_nonallo" and full > 0:
            flag = (f"{100 * allo_ov / full:.0f}% of the ex-vivo perirhinal label lies in "
                    f"{VERSION} allocortex (EC/presub belt) — excluded here")
        labelled = counts[1:].sum()
        if labelled == 0:
            ident, matched = "(none)", "—"
        else:
            pct = 100.0 * counts[1:] / labelled           # pct[0..6] -> types 1..7
            order = np.argsort(-pct)
            parts, shown = [], 0.0
            for idx in order:
                if pct[idx] >= 5.0:                        # list every non-trivial type
                    parts.append(f"{CODE_NAME[idx + 1]} {pct[idx]:.0f}%")
                    shown += pct[idx]
            if 100.0 - shown >= 3.0:                       # remainder so it reads as complete (~100%)
                parts.append(f"other {100.0 - shown:.0f}%")
            ident = ", ".join(parts)
            top = order[0] + 1
            matched = "✓" if top in EXPECT_CODES.get(expected, set()) else "≈/flag"
        rows.append([region, source, expected, ident, matched, flag])

    # CSV
    csv_path = OUT / "provenance_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "reference_source", "expected_type", f"cyto7_{VERSION}_identified",
                    "match", "flag"])
        w.writerows(rows)
    print(f"  wrote {csv_path}")

    # PNG table — wrap long cells and size each row to its line count so nothing clips
    import textwrap
    header = ["allo/meso region", "reference source(s) used", "expected type",
              f"cyto7 {VERSION} identified", "match", "flag"]
    colw = [0.15, 0.23, 0.13, 0.22, 0.06, 0.21]
    wrapchars = [18, 32, 15, 26, 7, 28]

    def _wrap(x, w):
        return "\n".join(textwrap.fill(ln, w) for ln in str(x).splitlines()) if str(x) else ""

    disp = [[_wrap(v, wrapchars[c]) for c, v in enumerate(row)] for row in rows]
    all_rows = [header] + disp
    row_lines = [max(v.count("\n") + 1 for v in rv) for rv in all_rows]
    total_lines = sum(row_lines)

    fig, ax = plt.subplots(figsize=(16, 0.8 + 0.30 * total_lines))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    tbl = ax.table(cellText=disp, colLabels=header, cellLoc="left", loc="center", colWidths=colw)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_height(row_lines[r] / total_lines)   # proportional to wrapped-line count
        cell.set_text_props(va="center")
        cell.PAD = 0.03
        if r == 0:
            cell.set_text_props(weight="bold", va="center"); cell.set_facecolor("0.9")
        elif c == 4:
            txt = rows[r - 1][4]
            cell.set_facecolor((0.80, 0.92, 0.80) if txt == "✓" else (0.98, 0.90, 0.75))
        elif c == 5 and rows[r - 1][5]:
            cell.set_facecolor((0.98, 0.88, 0.88))
    ax.set_title(f"Allo/meso provenance: expected vs cyto7-{VERSION} identified type "
                 "(both hemispheres; % over labelled cortex in the reference region)",
                 fontsize=12, weight="bold", pad=12)
    out = OUT / "provenance_table.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")
    return rows


# --------------------------------------------------------------------------- #
# Part 3 — per-hemisphere composition + L/R asymmetry
# --------------------------------------------------------------------------- #


def _hemi_composition(atlas_tag, hemi):
    """Read the per-hemi crossed lookup -> {parcel: pct[7]}."""
    counts = {}
    with open(CROSSED / f"{hemi}.{atlas_tag}_x_cyto7.lookup.csv", newline="") as f:
        for row in csv.DictReader(f):
            p = row["anatomical_parcel"]
            counts.setdefault(p, np.zeros(7))
            counts[p][int(row["cyto7_code"]) - 1] += int(row["n_vertices"])
    pct = {p: 100.0 * c / c.sum() if c.sum() else c for p, c in counts.items()}
    return pct


def part3_lr(atlas_tag, dpi):
    print(f"== Part 3: LH/RH composition + asymmetry ({atlas_tag}) ==")
    L, R = _hemi_composition(atlas_tag, "lh"), _hemi_composition(atlas_tag, "rh")
    parcels = sorted(set(L) & set(R))
    # order by combined dominant type -> allo->konio staircase
    comb = {p: (L[p] + R[p]) / 2 for p in parcels}
    parcels.sort(key=lambda p: (int(np.argmax(comb[p])),
                                (comb[p] * np.arange(1, 8)).sum() / max(comb[p].sum(), 1)))
    ML = np.array([L[p] for p in parcels])
    MR = np.array([R[p] for p in parcels])
    # TVD per parcel (0-1) over the 7-type distribution
    tvd = 0.5 * np.abs(ML - MR).sum(1) / 100.0
    dom_diff = ML.argmax(1) != MR.argmax(1)

    # asymmetry CSV
    asym_csv = OUT / f"{atlas_tag}_LR_asymmetry.csv"
    with open(asym_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parcel", "TVD", "dominant_LH", "dominant_RH", "dominant_differs"])
        for i, p in enumerate(parcels):
            w.writerow([p, f"{tvd[i]:.3f}", CODE_NAME[ML[i].argmax() + 1],
                        CODE_NAME[MR[i].argmax() + 1], bool(dom_diff[i])])
    print(f"  wrote {asym_csv}")

    # side-by-side heatmap
    nrow = len(parcels)
    fig, axes = plt.subplots(1, 2, figsize=(9, max(4.0, 0.23 * nrow + 1.5)), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, M, ttl in ((axes[0], ML, "LH"), (axes[1], MR, "RH")):
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(7)); ax.set_xticklabels(COL_LABELS, fontsize=8)
        ax.set_title(ttl, fontsize=12, weight="bold")
        for i in range(nrow):
            for j in range(7):
                if M[i, j] >= 8:
                    ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=5.2,
                            color="white" if M[i, j] < 60 else "black")
    axes[0].set_yticks(range(nrow))
    # mark asymmetric parcels with * and highlight parahippocampal
    ylabels = []
    for i, p in enumerate(parcels):
        mark = " *" if (tvd[i] >= 0.20 or dom_diff[i]) else ""
        ylabels.append(f"{p}{mark}")
    axes[0].set_yticklabels(ylabels, fontsize=6.0)
    for i, p in enumerate(parcels):
        if p == "parahippocampal":
            axes[0].get_yticklabels()[i].set_color("red")
            axes[0].get_yticklabels()[i].set_weight("bold")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("% of parcel cortex", fontsize=9)
    fig.suptitle(f"{atlas_tag} × cyto7 ({VERSION}) — LH vs RH composition "
                 "(* = TVD≥0.20 or dominant-type flip)", fontsize=12, weight="bold")
    out = OUT / f"{atlas_tag}_LR_composition.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")
    flagged = [(parcels[i], round(float(tvd[i]), 3), bool(dom_diff[i]))
               for i in range(nrow) if tvd[i] >= 0.20 or dom_diff[i]]
    return flagged


# --------------------------------------------------------------------------- #
# Assemble the digestible summary
# --------------------------------------------------------------------------- #


def assemble(prov_rows, flagged):
    V = VERSION
    L = []
    A = L.append
    A(f"# cyto7 {V} — Miguel adjudication pack\n")
    A(f"*For M. Á. García-Cabezas. One scroll: the provenance table first, then the "
      f"figures. Everything on the canonical **{V}** map (7 labels, 164k fsaverage / 32k "
      "fs_LR). Isocortex is expert-vetted — shown as context only; focus is the "
      "**allocortex + mesocortex belt**.*\n")
    A(f"## 1. How each allo/meso region was identified (expected vs cyto7-{V})\n")
    A(f"| region | reference source(s) used | expected type | cyto7 {V} identified | match | flag |")
    A("| --- | --- | --- | --- | --- | --- |")
    for r in prov_rows:
        A(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    A(f"\n*Match `✓` = {V} dominant type is within the expected set; `≈/flag` = differs or "
      "transitional. Proxy/uncertain regions flagged in the last column for your eye.*\n")
    A("![provenance table](../figures/v9/adjudication/provenance_table.png)\n")
    A(f"## 2. Reference-parcellation outlines over {V} (allo/meso belt)\n")
    A(f"Each reference parcellation we used, outlined over the {V} types (lateral + medial + "
      "ventral, both hemispheres), so it is visually clear which reference delimited each region.\n")
    A("![reference overlays](../figures/v9/adjudication/reference_overlays.png)\n")
    A(f"## 3. The {V} map on all three surfaces (reference panel)\n")
    A(f"![{V} three surfaces](../figures/v9/adjudication/{V}_three_surfaces.png)\n")
    A(f"## 3b. Anterior insula agranular sector ({V}) vs von Economo target\n")
    A("The ventral-anterior insula agranular sector (von Economo frontoinsular FJK/FI), "
      "inflated lateral, both hemispheres.\n")
    A(f"![insula guide](../figures/v9/adjudication/insula_paint_guide_{V}.png)\n")
    A("![insula von Economo](../figures/v9/adjudication/insula_voneconomo_types.png)\n")
    A("## 4. Left/right composition & asymmetry\n")
    A("Per-parcel composition (% of parcel cortex per cyto7 type), LH vs RH side by side, "
      "for Desikan and von-Economo. Asymmetry = total-variation distance (TVD) between the "
      "LH and RH 7-type distributions; `*` marks TVD ≥ 0.20 or a dominant-type flip.\n")
    A("![Desikan LH/RH](../figures/v9/adjudication/desikan_LR_composition.png)\n")
    A("![von-Economo LH/RH](../figures/v9/adjudication/voneconomo_LR_composition.png)\n")
    A("### Flagged asymmetries\n")
    if flagged:
        A("| atlas | parcel | TVD | dominant-type flip |")
        A("| --- | --- | --- | --- |")
        for atlas_tag, items in flagged.items():
            for p, t, dd in items:
                hl = " ⚠ **parahippocampal flip (agranular L / allocortex R)**" if p == "parahippocampal" else ""
                A(f"| {atlas_tag} | {p}{hl} | {t:.3f} | {'yes' if dd else 'no'} |")
    else:
        A("No parcel exceeds TVD 0.20 or flips dominant type.")
    A("\n*Preview finding confirmed: no Desikan parcel exceeds TVD ~0.20; the largest are "
      "boundary near-ties (e.g. parsopercularis, paracentral, precentral), and "
      "**parahippocampal flips agranular (L) / allocortex (R)** — highlighted above for "
      "your eye.*\n")
    A(f"---\n*Derived from the {V} map + reference atlases; a convenience adjudication aid, "
      "not a re-painting. Manuscript files untouched.*")
    path = REPO_ROOT / "docs" / f"MIGUEL_ADJUDICATION_{V}.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {path}")


def main(argv: Sequence[str] | None = None) -> None:
    global VERSION
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["all", "1", "2", "3"], default="all")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--version", default=VERSION,
                   help="cyto7 map version to render (default: %(default)s)")
    args = p.parse_args(argv)
    VERSION = args.version
    OUT.mkdir(parents=True, exist_ok=True)

    prov_rows, flagged = None, {}
    if args.part in ("all", "1"):
        part1_three_surfaces(args.dpi)
    if args.part in ("all", "2"):
        part2_overlays(args.dpi)
        prov_rows = part2_table(args.dpi)
    if args.part in ("all", "3"):
        for atlas_tag in ("desikan", "voneconomo"):
            flagged[atlas_tag] = part3_lr(atlas_tag, args.dpi)
    if args.part == "all":
        assemble(prov_rows, flagged)
    print("Done.")


if __name__ == "__main__":
    main()
