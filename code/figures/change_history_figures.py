"""Per-type change-history figures (v1 -> v4) for expert adjudication.

Implements ``docs/SPEC_change_history_figures.md``.

For **allocortex, agranular, dysgranular** (one figure each) this shows the full
provenance of the released v4 map: what was **added** (green), **removed** (red),
and changed purely for **topology** (blue), from the v1 as-painted map to v4, with
the **source/rule** behind each change, so García-Cabezas can adjudicate the
painting at a glance. cyto7 stays 7 labels.

Chain (all on 164k fsaverage): v1 -> v2 (identical) -> v3 (changelog_v2_to_v3
rule_clause) -> v3_clean (speckle = topology) -> v4 (EC/presub-proxy/piriform/
perirhinal->meso from the build masks; strict-gradient buffer = topology) ->
v4 fragment-cleanup (topology). Each changed vertex is attributed to the
**last** step that set its final label (recency), and coloured blue if that step
was a topology correction, else green (net added to T) / red (net removed from T).

Outputs
    figures/v9/change_history/change_provenance_{lh,rh}.csv
    figures/v9/change_history/{allocortex,agranular,dysgranular}_history.png

Run::  conda activate cyto7 && python scripts/change_history_figures.py
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
from matplotlib.patches import Patch

from cyto7_surface_io import REPO_ROOT
from make_presentation_figures import (
    HEMIS, LIGHT_BLUE, lighting_normals, lit_panel, load_surface, sulc_shading,
)
from audit_topology import adjacency
# reuse the exact v4 build helpers so v4 source masks match the released map
from build_v4_map import (
    _annot, _label_mask, _parcel_mask, _hops_from, build_hemi,
    ATLAS_DIR, VE_DIR, SURF_DIR, ISO_CODES,
)

import nibabel as nib

DERIVED = cfg.atlas_dir("fsaverage")
V1_DIR = cfg.atlas_dir("provenance/as_painted")
OUT_DIR = cfg.results_dir("tables") / "change_history"

#: target types (code -> name/slug) that get a figure.
TARGETS = {1: ("Allocortex", "allocortex"), 2: ("Agranular", "agranular"),
           3: ("Dysgranular", "dysgranular")}
ALL_NAMES = {0: "medial", 1: "allocortex", 2: "agranular", 3: "dysgranular",
             4: "eulaminate I", 5: "eulaminate II", 6: "eulaminate III",
             7: "koniocortex"}

# colour semantics (identical across the three figures)
C_BASE = np.array(LIGHT_BLUE)
C_GREY = np.array((0.80, 0.80, 0.80))
C_GREEN = np.array((0.15, 0.65, 0.20))   # added (evidence)
C_RED = np.array((0.87, 0.15, 0.15))     # removed (evidence)
C_BLUE = np.array((0.15, 0.38, 0.88))    # topology (add or remove)


# --------------------------------------------------------------------------- #
# Provenance class mapping
# --------------------------------------------------------------------------- #

#: v2->v3 rule_clause prefix -> (provenance class, is_topology)
V3_PREFIX_CLASS = {
    "topology_buffer": ("topology (buffer, v3)", True),
    "allocortex_scope": ("allocortex scope (v3)", False),
    "Pir": ("allocortex scope (piriform, v3)", False),
    "Perirhinal": ("Structural-Model (perirhinal, v3)", False),
    "EC": ("Structural-Model (EC, v3)", False),
    "cingulate_posterior": ("gyral rule (cingulate, v3)", False),
    "cingulate_anterior": ("gyral rule (cingulate, v3)", False),
    "HABC": ("Structural-Model (HABC, v3)", False),
    "PHA1": ("Glasser HCP-MMP (PHA1, v3)", False),
}
V3_FALLBACK = ("Structural-Model (v3, unspecified)", False)

CL_EC_V4 = ("EC/subiculum (v4, ex-vivo∪Desikan)", False)
CL_PRESUB_V4 = ("presub/parasub PROXY (v4, Destrieux)", False)   # *** FLAGGED
CL_PERI_V4 = ("perirhinal→meso (v4)", False)
CL_STRICT_V4 = ("topology (strict-gradient buffer, v4)", True)
CL_SPECKLE = ("topology (speckle, v3_clean)", True)
CL_FRAG = ("topology (fragment cleanup, v4)", True)


def _v3_class(rule_clause: str):
    prefix = rule_clause.split("->")[0].split("(")[0].strip()
    return V3_PREFIX_CLASS.get(prefix, V3_FALLBACK)


# --------------------------------------------------------------------------- #
# Build the per-vertex provenance chain for one hemisphere
# --------------------------------------------------------------------------- #


def _read_changelog(path: Path, hemi: str):
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("hemi") == hemi:
                rows.append(row)
    return rows


def _v4_source_masks(hemi: str, v3c: np.ndarray, presub_hops: int = 3):
    """Recompute the v4 evidence masks (EC / presub-proxy / perirhinal->meso) on
    164k, matching build_v4_map exactly."""
    n = v3c.shape[0]
    coords, faces = nib.freesurfer.read_geometry(str(SURF_DIR / f"{hemi}.pial"))
    faces = np.asarray(faces, np.int64)
    A = adjacency(faces, n)
    labelled = v3c >= 1
    ec = (_label_mask(ATLAS_DIR / f"{hemi}.entorhinal_exvivo.label", n)
          | _parcel_mask(VE_DIR / f"{hemi}.aparc.annot", {"entorhinal"}, n)) & labelled
    medial = v3c == 0
    hops_med = _hops_from(A, medial)
    parahip = _parcel_mask(ATLAS_DIR / f"{hemi}.aparc.a2009s.annot",
                           {"G_oc-temp_med-Parahip"}, n)
    presub = parahip & (hops_med >= 1) & (hops_med <= presub_hops) & labelled & ~ec
    piri = v3c == 1
    allo = (ec | presub | piri) & labelled
    peri = _label_mask(ATLAS_DIR / f"{hemi}.perirhinal_exvivo.label", n) & labelled & ~allo
    return ec, presub, peri


def provenance_hemi(hemi: str, presub_hops: int = 3):
    """Return a dict of 164k arrays: v1, v4final, class (object), topo (bool)."""
    v1 = _annot(V1_DIR / f"pial.{hemi}.cyto7.annot")[0]
    v3 = _annot(DERIVED / f"pial.{hemi}.cyto7.v3.annot")[0]
    v3c = _annot(DERIVED / f"pial.{hemi}.cyto7.v3_clean.annot")[0]
    v4f = _annot(DERIVED / f"pial.{hemi}.cyto7.v4.annot")[0]
    n = v1.shape[0]

    # v4pre (pre fragment-cleanup) via the exact build, to know strict-buffer set
    args = argparse.Namespace(presub_hops=presub_hops, max_pass=60)
    v4pre = build_hemi(hemi, args)["v4"]
    ec, presub, peri = _v4_source_masks(hemi, v3c, presub_hops)

    cls = np.array([""] * n, dtype=object)
    topo = np.zeros(n, bool)

    # step 1: v1(=v2) -> v3, rule_clause per vertex
    changed13 = np.where(v1 != v3)[0]
    rule_by_v = {int(r["vertex"]): r["rule_clause"]
                 for r in _read_changelog(DERIVED / "changelog_v2_to_v3.csv", hemi)}
    for v in changed13:
        c, t = _v3_class(rule_by_v.get(int(v), ""))
        cls[v], topo[v] = c, t

    # step 2: v3 -> v3_clean (speckle = topology)
    for v in np.where(v3 != v3c)[0]:
        cls[v], topo[v] = CL_SPECKLE

    # step 3: v3_clean -> v4pre (build: EC / presub / perirhinal->meso / strict-buffer)
    ch3 = (v3c != v4pre)
    ec_ch = ch3 & ec
    presub_ch = ch3 & presub & ~ec
    peri_ch = ch3 & peri & ~ec & ~presub
    strict_ch = ch3 & ~ec_ch & ~presub_ch & ~peri_ch
    for v in np.where(ec_ch)[0]:
        cls[v], topo[v] = CL_EC_V4
    for v in np.where(presub_ch)[0]:
        cls[v], topo[v] = CL_PRESUB_V4
    for v in np.where(peri_ch)[0]:
        cls[v], topo[v] = CL_PERI_V4
    for v in np.where(strict_ch)[0]:
        cls[v], topo[v] = CL_STRICT_V4

    # step 4: v4pre -> v4final (fragment cleanup = topology), most recent -> overrides
    for r in _read_changelog(DERIVED / "v4_fragment_cleanup_changelog.csv", hemi):
        v = int(r["vertex"])
        cls[v], topo[v] = CL_FRAG

    return {"v1": v1, "v4": v4f, "cls": cls, "topo": topo}


# --------------------------------------------------------------------------- #
# Per-type colour field on 164k
# --------------------------------------------------------------------------- #

# code: 0 base-brain, 1 grey (unchanged T extent), 2 green added, 3 red removed,
#       4 blue topology (add or remove involving T)
CODE_RGB = {0: C_BASE, 1: C_GREY, 2: C_GREEN, 3: C_RED, 4: C_BLUE}


def type_code_field(prov: dict, T: int) -> np.ndarray:
    v1, v4, topo = prov["v1"], prov["v4"], prov["topo"]
    inT1, inT4 = (v1 == T), (v4 == T)
    added = inT4 & ~inT1
    removed = inT1 & ~inT4
    unchanged = inT4 & inT1
    involves = added | removed
    code = np.zeros(v1.shape[0], int)
    code[unchanged] = 1
    code[added] = 2
    code[removed] = 3
    code[involves & topo] = 4   # topology overrides add/remove colour
    return code


def code_to_rgb(code: np.ndarray) -> np.ndarray:
    rgb = np.tile(C_BASE, (code.shape[0], 1))
    for k, c in CODE_RGB.items():
        rgb[code == k] = c
    return rgb


def extent_rgb(lab: np.ndarray, T: int) -> np.ndarray:
    rgb = np.tile(C_BASE, (lab.shape[0], 1))
    rgb[lab == T] = C_GREY
    return rgb


# --------------------------------------------------------------------------- #
# 32k resample for the flat panel
# --------------------------------------------------------------------------- #


def resample_164k_to_32k_nearest(arr164: np.ndarray, H: str) -> np.ndarray:
    """Nearest-neighbour fsaverage(164k) -> fs_LR(32k), matching resolve_target_map."""
    import os
    wb = os.environ.get("WORKBENCH_BIN") or (
        cfg.workbench_dir())
    if wb and Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")
    from neuromaps import transforms
    from nibabel.gifti import GiftiDataArray, GiftiImage
    gi = GiftiImage()
    gi.add_gifti_data_array(GiftiDataArray(arr164.astype(np.float32)))
    res = transforms.fsaverage_to_fslr(gi, "32k", hemi=H, method="nearest")
    return np.rint(np.asarray(res[0].agg_data())).astype(np.int16)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

VIEWS_3D = ["lateral", "medial"]
LEGEND = [
    Patch(facecolor=C_GREY, edgecolor="0.4", label="v4 extent (unchanged)"),
    Patch(facecolor=C_GREEN, edgecolor="0.4", label="added (evidence)"),
    Patch(facecolor=C_RED, edgecolor="0.4", label="removed (evidence)"),
    Patch(facecolor=C_BLUE, edgecolor="0.4", label="topology (speckle / strict buffer / fragment)"),
]


def _bar_breakdown(ax, counts: dict, title: str, kind: str):
    """Horizontal bar of per-class vertex counts (added or removed)."""
    if not counts:
        ax.text(0.5, 0.5, "(none)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return
    items = sorted(counts.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [C_BLUE if "topology" in k else (C_GREEN if kind == "added" else C_RED)
              for k in labels]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="0.3")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v}", va="center", fontsize=7.5)
    ax.set_xlabel("vertices", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.margins(x=0.15)


def build_type_figure(T: int, provs: dict, geom, vnorm, sulc, flat_fields,
                      out_dir: Path, dpi: int):
    name, slug = TARGETS[T]
    fig = plt.figure(figsize=(16, 15))
    fig.patch.set_facecolor("white")
    sub = fig.subfigures(4, 1, height_ratios=[1.05, 1.7, 1.15, 1.0], hspace=0.03)

    # ---- section 1: narrative strip (inflated, medial view) ----
    sf = sub[0]
    sf.suptitle(f"{name} — provenance narrative:  v1 as-painted  →  changes  →  v4"
                f"   (inflated, medial view)", fontsize=13, y=0.88)
    for x, ttl in zip((0.24, 0.53, 0.82), ("v1 extent", "changes", "v4 extent")):
        sf.text(x, 0.80, ttl, ha="center", va="center", fontsize=11)
    ax = sf.subplots(2, 3, subplot_kw={"projection": "3d"})
    for r, hemi in enumerate(HEMIS):
        coords, faces = geom[(hemi, "inflated")]
        prov = provs[hemi]
        panels = [extent_rgb(prov["v1"], T),
                  code_to_rgb(type_code_field(prov, T)),
                  extent_rgb(prov["v4"], T)]
        for c, rgb in enumerate(panels):
            a = ax[r, c]
            lit_panel(a, coords, faces, rgb, hemi, "medial", vnorm[hemi])
            if c == 0:
                a.text2D(-0.05, 0.5, hemi.upper(), transform=a.transAxes, rotation=90,
                         va="center", ha="center", fontsize=11, weight="bold")

    # ---- section 2: change map on inflated + pial ----
    sf = sub[1]
    sf.suptitle(f"{name} — change map on inflated & pial (green add / red remove / blue topology)",
                fontsize=13, y=1.0)
    cols = [(h, v) for h in HEMIS for v in VIEWS_3D]  # LH lat, LH med, RH lat, RH med
    ax = sf.subplots(2, len(cols), subplot_kw={"projection": "3d"})
    for r, surf in enumerate(["inflated", "pial"]):
        for c, (hemi, view) in enumerate(cols):
            coords, faces = geom[(hemi, surf)]
            rgb = code_to_rgb(type_code_field(provs[hemi], T))
            ss = sulc[hemi] if surf == "pial" else None
            a = ax[r, c]
            lit_panel(a, coords, faces, rgb, hemi, view, vnorm[hemi], sulc_shade=ss)
            if r == 0:
                a.set_title(f"{hemi.upper()} {view}", fontsize=10)
            if c == 0:
                a.text2D(-0.06, 0.5, surf, transform=a.transAxes, rotation=90,
                         va="center", ha="center", fontsize=11, weight="bold")

    # ---- section 3: flat change map (2D PolyCollection, exact colours) ----
    sf = sub[2]
    sf.suptitle(f"{name} — change map on the flat surface (32k fs_LR)", fontsize=13, y=0.98)
    if flat_fields.get(T):
        from matplotlib.collections import PolyCollection
        ax = sf.subplots(1, 2)
        for a, H in zip(ax, ("L", "R")):
            coords2d, faces, code32 = flat_fields[T][H]
            # Colour each face by its MODAL code, not by averaging RGB across its three
            # vertices: these codes (added / removed / topology) are categorical, so a
            # blend lands on a colour that means nothing. This is the categorical analogue
            # of the median fix applied to the ordinal cyto7 flat panels
            # (SPEC_flat_panel_median_recolor); a median would be meaningless here because
            # the codes have no order, and the |Δtype| >= 2 metric does not apply.
            import flat_display
            fmode = flat_display.face_mode(faces, code32)
            fcol = code_to_rgb(fmode)
            fcol[fmode == 0] = (0.94, 0.94, 0.94)        # base/medial -> light-grey silhouette
            tri = coords2d[faces]                        # (F,3,2)
            pc = PolyCollection(tri, facecolors=fcol, edgecolors="none", antialiased=False)
            pc.set_rasterized(True)
            a.add_collection(pc)
            a.set_xlim(coords2d[:, 0].min(), coords2d[:, 0].max())
            a.set_ylim(coords2d[:, 1].min(), coords2d[:, 1].max())
            a.set_aspect("equal")
            a.axis("off")
            a.set_title(f"{'LH' if H == 'L' else 'RH'} flat", fontsize=11)
    else:
        a = sf.subplots(1, 1)
        a.text(0.5, 0.5, "(flat panels skipped: --no-flat)", ha="center", va="center",
               fontsize=11, color="0.5")
        a.axis("off")

    # ---- section 4: source breakdown ----
    sf = sub[3]
    bd_title = f"{name} — provenance breakdown (net v1→v4, both hemispheres)"
    if T == 1:
        bd_title += ("\nNB presub/parasub additions are a DOCUMENTED PROXY "
                     "(Destrieux parahippocampal band, PROVENANCE_v4) — flagged for GC.")
    sf.suptitle(bd_title, fontsize=12, y=1.14)
    added_counts, removed_counts = {}, {}
    for hemi in HEMIS:
        prov = provs[hemi]
        v1, v4, clsarr = prov["v1"], prov["v4"], prov["cls"]
        added = np.where((v4 == T) & (v1 != T))[0]
        removed = np.where((v1 == T) & (v4 != T))[0]
        for v in added:
            k = clsarr[v] or "(unattributed)"
            added_counts[k] = added_counts.get(k, 0) + 1
        for v in removed:
            k = clsarr[v] or "(unattributed)"
            removed_counts[k] = removed_counts.get(k, 0) + 1
    ax = sf.subplots(1, 2)
    _bar_breakdown(ax[0], added_counts, f"added to {name} — by source", "added")
    _bar_breakdown(ax[1], removed_counts, f"removed from {name} — by rule", "removed")
    sf.subplots_adjust(bottom=0.32, top=0.78, left=0.22, right=0.97)

    fig.legend(handles=LEGEND, loc="lower center", ncol=4, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.008))
    fig.suptitle(f"cyto7 change history — {name}  (v1 as-painted → released v4)",
                 fontsize=16, y=1.0)
    out = out_dir / f"{slug}_history.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def write_provenance_csv(provs: dict, out_dir: Path):
    for hemi in HEMIS:
        prov = provs[hemi]
        v1, v4, clsarr, topo = prov["v1"], prov["v4"], prov["cls"], prov["topo"]
        changed = np.where(v1 != v4)[0]
        out = out_dir / f"change_provenance_{hemi}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["hemi", "vertex", "v1_type", "v4_type", "source_class",
                        "is_topology", "colour",
                        "cat_allocortex", "cat_agranular", "cat_dysgranular"])
            for v in changed:
                a, b = int(v1[v]), int(v4[v])
                istopo = bool(topo[v])
                colour = "blue" if istopo else "green/red (per-type add/remove)"
                cats = []
                for T in (1, 2, 3):
                    if b == T and a != T:
                        cats.append("topology" if istopo else "added-to-T")
                    elif a == T and b != T:
                        cats.append("topology" if istopo else "removed-from-T")
                    elif a == T and b == T:
                        cats.append("unchanged-T")
                    else:
                        cats.append("na")
                w.writerow([hemi, int(v), ALL_NAMES[a], ALL_NAMES[b],
                            clsarr[v] or "(unattributed)", istopo, colour, *cats])
        print(f"  wrote {out} ({changed.size} changed vertices)")


def print_summary(provs: dict):
    print("\n== per-type net change summary (v1 -> v4) ==")
    print(f"{'type':<12}{'hemi':<5}{'added':>7}{'removed':>9}{'topo(add/rem)':>15}")
    for T in (1, 2, 3):
        for hemi in HEMIS:
            prov = provs[hemi]
            v1, v4, topo = prov["v1"], prov["v4"], prov["topo"]
            added = (v4 == T) & (v1 != T)
            removed = (v1 == T) & (v4 != T)
            ntopo = int(((added | removed) & topo).sum())
            print(f"{TARGETS[T][0]:<12}{hemi:<5}{int(added.sum()):>7}"
                  f"{int(removed.sum()):>9}{ntopo:>15}")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--presub-hops", type=int, default=3)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--dataset", default="Validation210")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-flat", action="store_true", help="skip the flat panels (fast).")
    args = p.parse_args(argv)

    print("Building provenance chain (v1 -> v2 -> v3 -> v3_clean -> v4) ...")
    provs = {hemi: provenance_hemi(hemi, args.presub_hops) for hemi in HEMIS}
    print_summary(provs)

    args.out.mkdir(parents=True, exist_ok=True)
    write_provenance_csv(provs, args.out)

    # geometry / lighting caches shared across the three figures
    print("\nLoading surfaces + lighting ...")
    geom = {(h, s): load_surface(h, s) for h in HEMIS for s in ("inflated", "pial")}
    vnorm = {h: lighting_normals(h) for h in HEMIS}
    sulc = {h: sulc_shading(h, 0.35) for h in HEMIS}

    # flat 32k fields per (type, hemi): (coords2d, faces, resampled code)
    from cyto7_surface_io import load_surface_geometry
    flat_fields = {T: {} for T in TARGETS}
    if not args.no_flat:
        print("Resampling change fields to 32k fs_LR for the flat panels ...")
        flat_geom = {H: load_surface_geometry(args.dataset, H, "flat") for H in ("L", "R")}
        for T in TARGETS:
            for hemi, H in (("lh", "L"), ("rh", "R")):
                code32 = resample_164k_to_32k_nearest(type_code_field(provs[hemi], T), H)
                coords, faces = flat_geom[H]
                flat_fields[T][H] = (np.asarray(coords)[:, :2], np.asarray(faces), code32)

    print("\nRendering per-type figures ...")
    for T in TARGETS:
        if args.no_flat:
            # provide a dummy flat field that renders base-only (skipped in figure)
            pass
        build_type_figure(T, provs, geom, vnorm, sulc, flat_fields, args.out, args.dpi)
    print("\nDone.")


if __name__ == "__main__":
    main()
