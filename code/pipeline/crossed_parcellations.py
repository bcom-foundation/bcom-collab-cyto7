"""Crossed parcellations: anatomical parcel x cyto7 type (SPEC_crossed_parcellations.md).

A derived convenience release that subdivides each anatomical parcel P by the cyto7
type T present within it, yielding sub-parcels ``P__T`` (e.g.
``posteriorcingulate__dysgranular``) so users can select "the dysgranular part of the
posterior cingulate". Built on the canonical **v6** map, cortex only
(``?h.cortex.label``); the medial wall stays ``unknown``. cyto7 stays 7 labels.

Ships Desikan x cyto7 and von-Economo x cyto7 (both hemispheres). Destrieux is an
easy add (see ``ATLASES``) but not shipped by default.

Step 2 cleanup: within each parcel P, any sub-parcel ``P__T`` smaller than
``--min-vertices`` (default 20) is absorbed into the majority spatially-adjacent
sub-parcel *of the same parcel P* (surface-graph majority-neighbour — the speckle rule,
scoped to P; never merges across anatomical parcels). Iterates until no under-size
sub-parcel remains except isolated ones with no same-P neighbour (reported).

Outputs under ``resources/crossed_parcellations/``:
  {lh,rh}.desikan_x_cyto7.annot     + .lookup.csv
  {lh,rh}.voneconomo_x_cyto7.annot  + .lookup.csv
  README.md

Run::  conda activate cyto7 && python scripts/crossed_parcellations.py --map v6
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

from cyto7_surface_io import REPO_ROOT, resolve_target_map
from audit_topology import adjacency
from figure_style import CYTO7_VIRIDIS

VE_DIR = cfg.data_dir() / "voneconomo"
ATLAS_DIR = cfg.data_dir() / "refine_atlases"
SURF_DIR = cfg.data_dir() / "fsaverage_surfaces"
OUT_DIR = cfg.atlas_dir("fsaverage/crossed")
HEMIS = ("lh", "rh")

#: cyto7 code -> short slug used in the P__T sub-parcel name.
TYPE_SLUG = {1: "allocortex", 2: "agranular", 3: "dysgranular", 4: "eulaminate1",
             5: "eulaminate2", 6: "eulaminate3", 7: "koniocortex"}

#: short column labels for the composition matrix (Step 4), allo -> konio.
COL_LABELS = ["Allo", "Agr", "Dys", "EulI", "EulII", "EulIII", "Kon"]

FIG_DIR = cfg.results_dir("tables") / "crossed"

#: anatomical parcel names that are NOT real cortex parcels (medial wall / background).
NON_PARCEL = {"unknown", "corpuscallosum", "medial_wall", "none", ""}

#: atlases to cross (label = output tag, file template per hemi).
ATLASES = {
    "desikan": lambda h: VE_DIR / f"{h}.aparc.annot",
    "voneconomo": lambda h: VE_DIR / f"{h}.economo.annot",
    # "destrieux": lambda h: ATLAS_DIR / f"{h}.aparc.a2009s.annot",  # optional add
}


def _read_annot(path):
    lab, ctab, names = nib.freesurfer.io.read_annot(str(path))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return np.asarray(lab), np.asarray(ctab), names


def _cortex_mask(hemi, n):
    idx = nib.freesurfer.read_label(str(ATLAS_DIR / f"{hemi}.cortex.label"))
    m = np.zeros(n, bool)
    m[idx] = True
    return m


def _parcel_base_rgb(anat_ctab, anat_names):
    """{parcel_name -> (r,g,b) in 0-1} from the anatomical atlas colortable."""
    out = {}
    for i, nm in enumerate(anat_names):
        out[nm] = tuple(float(v) / 255.0 for v in anat_ctab[i, :3])
    return out


# --------------------------------------------------------------------------- #
# Step 2 — within-parcel speckle cleanup
# --------------------------------------------------------------------------- #


def cleanup_within_parcel(anat_lab, anat_names, xtype, A, parcel_ids, min_v):
    """Absorb sub-parcels (P, T) < *min_v* into the majority same-P neighbour type.

    Mutates and returns *xtype* (per-vertex cyto7 type used for crossing). Returns
    ``(xtype, absorbed_per_parcel, isolated)`` where absorbed counts vertices moved
    and isolated lists (parcel, type, n) groups that had no same-P neighbour."""
    absorbed: dict[str, int] = {}
    isolated: list[tuple[str, int, int]] = []
    for pid in parcel_ids:
        pname = anat_names[pid]
        inP = anat_lab == pid
        if not inP.any():
            continue
        changed = True
        while changed:
            changed = False
            types = [int(t) for t in np.unique(xtype[inP]) if t >= 1]
            counts = {t: int(np.sum(inP & (xtype == t))) for t in types}
            small = sorted([t for t, c in counts.items() if c < min_v],
                           key=lambda t: counts[t])
            for t in small:
                grp = inP & (xtype == t)
                ncur = int(grp.sum())
                if ncur == 0 or ncur >= min_v:
                    continue
                # same-P boundary neighbours of a different type
                nb = (A @ grp.astype(np.int8) > 0) & inP & ~grp
                nb_types = xtype[nb]
                nb_types = nb_types[nb_types >= 1]
                if nb_types.size == 0:
                    isolated.append((pname, t, ncur))
                    continue
                vals, cnts = np.unique(nb_types, return_counts=True)
                tgt = int(vals[cnts.argmax()])
                xtype[grp] = tgt
                absorbed[pname] = absorbed.get(pname, 0) + ncur
                changed = True
                break  # recompute counts after each absorption
    return xtype, absorbed, isolated


# --------------------------------------------------------------------------- #
# Build one crossed annot for one hemisphere
# --------------------------------------------------------------------------- #


def build_hemi(hemi, atlas_tag, atlas_path, cyto_lab, coords, faces, min_v):
    n = cyto_lab.shape[0]
    anat_lab, anat_ctab, anat_names = _read_annot(atlas_path)
    cortex = _cortex_mask(hemi, n)
    A = adjacency(faces, n)
    base_rgb = _parcel_base_rgb(anat_ctab, anat_names)

    # which anatomical ids are real parcels (on cortex, real name)
    real = np.zeros(n, bool)
    parcel_ids = []
    for pid, nm in enumerate(anat_names):
        if nm.lower() in NON_PARCEL:
            continue
        m = (anat_lab == pid) & cortex
        if m.any():
            real |= m
            parcel_ids.append(pid)

    # crossing type (mutable copy of cyto7 restricted to real-parcel cortex)
    xtype = np.where(real & (cyto_lab >= 1), cyto_lab, 0).astype(int)

    xtype, absorbed, isolated = cleanup_within_parcel(
        anat_lab, anat_names, xtype, A, parcel_ids, min_v)

    # assign sub-parcel indices: 0 = unknown/background
    labels_out = np.zeros(n, np.int32)
    names_out = ["unknown"]
    rgb_out = [(0.88, 0.96, 0.96)]           # light background
    lookup = []  # (id, parcel, cyto_code, cyto_name, n)
    used_rgb = {(224, 245, 245)}
    for pid in parcel_ids:
        pname = anat_names[pid]
        inP = anat_lab == pid
        for t in [int(x) for x in np.unique(xtype[inP]) if x >= 1]:
            m = inP & (xtype == t)
            nvox = int(m.sum())
            if nvox == 0:
                continue
            idx = len(names_out)
            labels_out[m] = idx
            slug = TYPE_SLUG[t]
            names_out.append(f"{pname}__{slug}")
            # colour: blend parcel base with the cyto7 type hue; nudge to unique.
            base = np.array(base_rgb.get(pname, (0.5, 0.5, 0.5)))
            tcol = np.array(CYTO7_VIRIDIS[t])
            rgb = tuple(0.45 * base + 0.55 * tcol)
            r255 = _unique_rgb255(rgb, used_rgb)
            rgb_out.append(tuple(c / 255.0 for c in r255))
            lookup.append((idx, pname, t, slug, nvox))

    ctab = _build_ctab(rgb_out)
    return {
        "hemi": hemi, "atlas": atlas_tag, "labels": labels_out, "ctab": ctab,
        "names": names_out, "lookup": lookup, "absorbed": absorbed,
        "isolated": isolated, "n_subparcels": len(names_out) - 1,
    }


def _unique_rgb255(rgb01, used):
    r, g, b = (int(round(np.clip(c, 0, 1) * 255)) for c in rgb01)
    while (r, g, b) in used:
        b = (b + 1) % 256
        if b == 0:
            g = (g + 1) % 256
    used.add((r, g, b))
    return (r, g, b)


def _build_ctab(rgb_list):
    n = len(rgb_list)
    ctab = np.zeros((n, 5), int)
    for i, (r, g, b) in enumerate(rgb_list):
        R, G, B = int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
        ctab[i] = [R, G, B, 0, R + G * 256 + B * 65536]
    return ctab


# --------------------------------------------------------------------------- #
# Step 4 — parcel x cyto7 composition matrix + heatmap
# --------------------------------------------------------------------------- #


def composition_matrix(lookup_rows):
    """From combined (both-hemisphere) lookup rows [(parcel, code, n), ...] build a
    parcels x 7 **percent** matrix (each row sums to 100% over labelled cortex in the
    parcel). Rows sorted by dominant cyto7 type then weighted-mean type (allo->konio
    staircase). Returns (parcels, pct[P,7], counts[P,7])."""
    parcels = sorted({p for p, _c, _n in lookup_rows})
    pidx = {p: i for i, p in enumerate(parcels)}
    counts = np.zeros((len(parcels), 7), float)
    for p, code, n in lookup_rows:
        counts[pidx[p], int(code) - 1] += n
    totals = counts.sum(1, keepdims=True)
    pct = 100.0 * counts / np.where(totals > 0, totals, 1)
    # sort rows: dominant type, then weighted-mean type (both ascending -> staircase)
    dom = pct.argmax(1)
    wmean = (pct * np.arange(1, 8)).sum(1) / np.where(pct.sum(1) > 0, pct.sum(1), 1)
    order = sorted(range(len(parcels)), key=lambda i: (dom[i], wmean[i]))
    parcels = [parcels[i] for i in order]
    return parcels, pct[order], counts[order]


def plot_composition_matrix(atlas_tag, parcels, pct, out_png, dpi, annot_min=8.0):
    """Heatmap: rows = parcels (allo->konio staircase), cols = 7 cyto7 types,
    cell = % of the parcel's labelled cortex. Sequential (viridis) magnitude cmap
    from the figure_style convention; cells >= *annot_min* % annotated."""
    nrow = len(parcels)
    fig, ax = plt.subplots(figsize=(6.2, max(4.0, 0.23 * nrow + 1.5)))
    fig.patch.set_facecolor("white")
    im = ax.imshow(pct, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(7))
    ax.set_xticklabels(COL_LABELS, fontsize=9)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(parcels, fontsize=6.2)
    ax.set_xlabel("cyto7 type", fontsize=10)
    for i in range(nrow):
        for j in range(7):
            v = pct[i, j]
            if v >= annot_min:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v < 60 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("% of parcel cortex", fontsize=9)
    ax.set_title(f"{atlas_tag} × cyto7 (v6) — composition per parcel\n"
                 "(both hemispheres, post-cleanup; row-sorted allo→konio)",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_png}")


def write_composition_csv(parcels, pct, counts, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anatomical_parcel", "parcel_n_vertices"]
                   + [f"{s}_pct" for s in COL_LABELS]
                   + [f"{s}_n" for s in COL_LABELS])
        for i, p in enumerate(parcels):
            w.writerow([p, int(counts[i].sum())]
                       + [f"{v:.1f}" for v in pct[i]]
                       + [int(v) for v in counts[i]])
    print(f"  wrote {out_csv}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", default="v9")
    p.add_argument("--min-vertices", type=int, default=20)
    p.add_argument("--atlases", nargs="+", default=list(ATLASES),
                   choices=list(ATLASES))
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cyto = resolve_target_map(args.map, "fsaverage")     # 164k, {"L","R"}
    hkey = {"lh": "L", "rh": "R"}
    summary = []
    for atlas_tag in args.atlases:
        atlas_rows = []  # combined-hemisphere (parcel, code, n) for the Step-4 matrix
        for hemi in HEMIS:
            coords, faces = nib.freesurfer.read_geometry(str(SURF_DIR / f"{hemi}.pial"))
            faces = np.asarray(faces, np.int64)
            res = build_hemi(hemi, atlas_tag, ATLASES[atlas_tag](hemi),
                             cyto[hkey[hemi]], coords, faces, args.min_vertices)
            # write annot
            annot_path = args.out / f"{hemi}.{atlas_tag}_x_cyto7.annot"
            nm = [x.encode() for x in res["names"]]
            nib.freesurfer.io.write_annot(str(annot_path), res["labels"],
                                          res["ctab"], nm, fill_ctab=True)
            # write lookup CSV
            csv_path = args.out / f"{hemi}.{atlas_tag}_x_cyto7.lookup.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["sub_parcel_id", "anatomical_parcel", "cyto7_code",
                            "cyto7_type", "n_vertices"])
                w.writerows(res["lookup"])
            tot_abs = sum(res["absorbed"].values())
            print(f"{hemi} {atlas_tag}: {res['n_subparcels']} sub-parcels; "
                  f"absorbed {tot_abs} vtx across {len(res['absorbed'])} parcels; "
                  f"isolated<{args.min_vertices}={len(res['isolated'])}")
            if res["isolated"]:
                for pn, t, nn in res["isolated"][:12]:
                    print(f"    isolated: {pn}__{TYPE_SLUG[t]} ({nn} vtx, no same-parcel neighbour)")
            summary.append({"hemi": hemi, "atlas": atlas_tag,
                            "n_subparcels": res["n_subparcels"],
                            "absorbed_vtx": tot_abs,
                            "n_parcels_absorbed": len(res["absorbed"]),
                            "n_isolated": len(res["isolated"]),
                            "absorbed_per_parcel": res["absorbed"],
                            "isolated": res["isolated"]})
            atlas_rows += [(p, code, n) for (_id, p, code, _slug, n) in res["lookup"]]
            print(f"   wrote {annot_path.name} + {csv_path.name}")

        # Step 4 — composition matrix (both hemispheres combined, post-cleanup)
        parcels, pct, counts = composition_matrix(atlas_rows)
        write_composition_csv(parcels, pct, counts,
                              FIG_DIR / f"{atlas_tag}_x_cyto7_composition.csv")
        plot_composition_matrix(atlas_tag, parcels, pct,
                                FIG_DIR / f"{atlas_tag}_x_cyto7_matrix.png", args.dpi)

    _write_readme(args.out, args.map, args.min_vertices, args.atlases, summary)
    print("Done.")


def _write_readme(out_dir, map_version, min_v, atlases, summary):
    L = []
    A = L.append
    A("# Crossed parcellations — anatomical parcel × cyto7 type (derived release)\n")
    A(f"Each anatomical parcel **P** is subdivided by the cyto7 cytoarchitectural type "
      f"**T** present within it, giving sub-parcels named **`P__T`** (e.g. "
      f"`posteriorcingulate__dysgranular`). Built on the canonical cyto7 **{map_version}** "
      "map (7 labels), 164k fsaverage, cortex only.\n")
    A("## Files (per hemisphere `lh`/`rh`)")
    for atlas_tag in atlases:
        A(f"- `{{lh,rh}}.{atlas_tag}_x_cyto7.annot` — FreeSurfer annotation; each entry is a "
          f"`P__T` sub-parcel (index 0 = `unknown` = medial wall / non-cortex).")
        A(f"- `{{lh,rh}}.{atlas_tag}_x_cyto7.lookup.csv` — "
          "`sub_parcel_id, anatomical_parcel, cyto7_code, cyto7_type, n_vertices`.")
    A("\n## How it was built")
    A(f"1. **Cross:** for every cortex vertex (in `?h.cortex.label`), take its anatomical "
      f"parcel P and its cyto7 {map_version} type T (1–7) → sub-parcel `P__T`. Medial wall / "
      "non-cortex vertices stay `unknown` (index 0). Anatomical `unknown`/`corpuscallosum`/"
      "`medial_wall` are not crossed.")
    A(f"2. **Clean up tiny sub-parcels:** within each parcel P, any `P__T` with "
      f"**< {min_v} vertices** (`--min-vertices`) is absorbed into the majority "
      "spatially-adjacent sub-parcel *of the same parcel P* (surface-graph "
      "majority-neighbour; never merges across anatomical parcels). Iterated until only "
      "isolated under-size groups with no same-parcel neighbour remain (reported below).")
    A("\n## Composition matrices (parcel × cyto7 type)")
    A("`figures/v9/crossed/{desikan,voneconomo}_x_cyto7_matrix.png` + "
      "`*_composition.csv`: per anatomical parcel, the % of its labelled cortex in each "
      "cyto7 type (rows sorted allo→koniocortex; both hemispheres combined, post-cleanup; "
      "medial-wall/unknown excluded from the denominator).")
    A("\n## Cleanup summary")
    A("| hemi | atlas | sub-parcels | vtx absorbed | parcels touched | isolated (<min, kept) |")
    A("| --- | --- | ---: | ---: | ---: | ---: |")
    for s in summary:
        A(f"| {s['hemi']} | {s['atlas']} | {s['n_subparcels']} | {s['absorbed_vtx']} | "
          f"{s['n_parcels_absorbed']} | {s['n_isolated']} |")
    isolated_any = [(s, iso) for s in summary for iso in s["isolated"]]
    if isolated_any:
        A("\nIsolated under-size sub-parcels kept (no same-parcel neighbour of another type):")
        for s, (pn, t, nn) in isolated_any:
            A(f"- {s['hemi']} {s['atlas']}: `{pn}__{TYPE_SLUG[t]}` ({nn} vtx)")
    A("\n## Caveats / citation")
    A(f"- **Derived convenience product.** Cite the cyto7 **{map_version}** map *and* the "
      "anatomical atlas (Desikan–Killiany / von Economo–Koskinas) used for the crossing.")
    A("- Sub-parcels **inherit the anatomical atlas's boundaries** — the cross does not move "
      "any boundary; it only intersects the two labellings. A `P__T` sub-parcel is as good "
      "(or as coarse) as parcel P's definition.")
    A("- cyto7 stays **7 labels**; the crossing adds no new cytoarchitectural classes.")
    A("- Colours are a blend of the parcel colour and the cyto7 type hue (viridis), nudged to "
      "be unique per entry; identity is authoritative in the lookup CSV, not the colour.")
    A(f"- Min-size rule (**{min_v}**), the within-parcel majority-neighbour absorption target, "
      "the sub-parcel naming (`P__T`), and whether to also ship Destrieux are open choices "
      "for Ricardo/GC (see `docs/SPEC_crossed_parcellations.md`).")
    (out_dir / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
