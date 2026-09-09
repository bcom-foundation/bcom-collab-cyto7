"""RR11 - the resolution cascade figure: what each coarsening step throws away.

After RR10 the same cytoarchitectural information exists at three resolutions: the
released vertex map, the crossed parcellation (101 Desikan nodes, 115 von Economo)
and one majority cyto7 type per classical parcel. Section 4.7 argues the middle one
exists because the third discards about a third of the within-parcel variation while
the first cannot be consumed by node-based models.

Three categorical surfaces side by side look almost identical at a glance, which
would undercut that argument rather than make it, so the figure is built around the
loss: the misassignment map (panel b) and the misassignment fraction (panel d) are
what carry it, and the cascade (panel a) is context for them.

Type-misassignment fraction: the proportion of labelled vertices whose released v9
cyto7 type differs from the type its node is assigned. Zero at vertex level by
construction, the absorbed residue only for the crossed parcellation, and whatever
one type per parcel actually costs, which is the number the manuscript asserts.

Run::
    conda run -n cyto7 python scripts/rr11_resolution_cascade.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib.colors import ListedColormap

import rr_common as rc
from cyto7_surface_io import REPO_ROOT, resolve_target_map
from figure_style import CYTO7_VIRIDIS, CYTO7_NAMES, SIGNED_DELTA

OUTDIR = rc.OUT / "rr11_cascade"
FIGDIR = cfg.results_dir("tables") / "manuscript"
STAGED = REPO_ROOT / "manuscript" / "preprint" / "26th_August_2026" / "figures"
CROSSED = cfg.atlas_dir("fsaverage/crossed")
VE_DIR = cfg.data_dir() / "voneconomo"
AT_DIR = cfg.data_dir() / "refine_atlases"

HEMIS = ("lh", "rh")
HKEY = {"lh": "L", "rh": "R"}
NL = {"lh": "left", "rh": "right"}
NON_PARCEL = {"unknown", "corpuscallosum", "medial_wall", "none", "", "???"}

ATLASES = {
    "aparc_cyto7": ("Desikan-Killiany", lambda h: VE_DIR / f"{h}.aparc.annot"),
    "voneconomo_cyto7": ("von Economo", lambda h: VE_DIR / f"{h}.economo.annot"),
}

#: the RR10 files are inputs here and must come out unchanged.
FROZEN_EXTRA = sorted(CROSSED.glob("*.annot")) + sorted(CROSSED.glob("*.label.gii")) + \
    [CROSSED / "aparc_cyto7_nodes.tsv", CROSSED / "voneconomo_cyto7_nodes.tsv"]


def _log_factory(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")

    def log(*args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()
    return log, fh


def _read_annot(path: Path):
    lab, ctab, names = nib.freesurfer.io.read_annot(str(path))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return np.asarray(lab), np.asarray(ctab), names


def _cortex(hemi: str, n: int) -> np.ndarray:
    idx = nib.freesurfer.read_label(str(AT_DIR / f"{hemi}.cortex.label"))
    m = np.zeros(n, bool)
    m[idx] = True
    return m


# --------------------------------------------------------------------------- #
# The three resolutions
# --------------------------------------------------------------------------- #


def build_resolutions(tag: str, path_fn, cyto: dict, va: dict, geom: dict,
                      log=print) -> dict:
    """Per-vertex assigned type at each of the three resolutions, per hemisphere.

    ``vertex``  the released v9 type itself
    ``crossed`` the cyto7 type named by the RR10 node the vertex belongs to
    ``areal``   the majority cyto7 type of the vertex's whole anatomical parcel

    The areal majority is taken within hemisphere, which is what a node-based model
    using one node per parcel per hemisphere actually does; the bilateral variant is
    reported beside it because the row label counts parcel names once.
    """
    nodes = {}
    with open(CROSSED / f"{tag}_nodes.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            nodes[(r["hemi"], int(r["annot_index"]))] = int(r["cyto7_ordinal"])

    out = {h: {} for h in HEMIS}
    pooled: dict[tuple[str, int], float] = {}
    for h in HEMIS:
        anat_lab, _c, anat_names = _read_annot(path_fn(h))
        n = anat_lab.shape[0]
        cortex = _cortex(h, n)
        c9 = np.asarray(cyto[HKEY[h]], int)
        parcel = np.full(n, "", object)
        for pid, nm in enumerate(anat_names):
            if nm.lower() in NON_PARCEL:
                continue
            m = (anat_lab == pid) & cortex
            if m.any():
                parcel[m] = nm
        valid = (parcel != "") & (c9 >= 1)

        xlab, _xc, xnames = _read_annot(CROSSED / f"{h}.{tag}.annot")
        crossed = np.zeros(n, int)
        for i in range(1, len(xnames)):
            m = xlab == i
            if m.any():
                crossed[m] = nodes.get((h, i), 0)

        pidx = np.zeros(n, int)
        for k, nm in enumerate(sorted({str(x) for x in parcel[valid]}), start=1):
            pidx[valid & (parcel == nm)] = k
        out[h].update({"valid": valid, "vertex": c9, "crossed": crossed,
                       "parcel": parcel, "n": n, "node_index": xlab,
                       "parcel_index": pidx})
        for p, t, a in zip(parcel[valid], c9[valid], va[h][valid]):
            pooled[(p, int(t))] = pooled.get((p, int(t)), 0.0) + float(a)

    # majority type per parcel, within hemisphere and bilaterally
    bil: dict[str, dict[int, float]] = {}
    for (p, t), a in pooled.items():
        bil.setdefault(p, {})[t] = a
    maj_bil = {p: max(d, key=d.get) for p, d in bil.items()}
    for h in HEMIS:
        d = out[h]
        per: dict[str, dict[int, float]] = {}
        for p, t, a in zip(d["parcel"][d["valid"]], d["vertex"][d["valid"]],
                           va[h][d["valid"]]):
            per.setdefault(p, {})[int(t)] = per.setdefault(p, {}).get(int(t), 0.0) + float(a)
        maj_h = {p: max(v, key=v.get) for p, v in per.items()}
        areal = np.zeros(d["n"], int)
        areal_bil = np.zeros(d["n"], int)
        for p in maj_h:
            m = d["valid"] & (d["parcel"] == p)
            areal[m] = maj_h[p]
            areal_bil[m] = maj_bil[p]
        d["areal"] = areal
        d["areal_bilateral"] = areal_bil
        d["faces"] = geom[h]["faces"]
        d["n_parcels"] = len(maj_h)
    out["n_parcel_names"] = len(maj_bil)
    out["n_nodes"] = len({(k[0], k[1]) for k in nodes}) and len(
        {v for v in set(nodes.values())}) and len(set(
            (r for r in nodes))) and len({i for (_h, i) in nodes})
    return out


def misassignment(res: dict, va: dict, log=print) -> list[dict]:
    """Fraction of labelled vertices whose v9 type differs from their node's type."""
    rows = []
    for key, label in (("vertex", "vertex"), ("crossed", "crossed"),
                       ("areal", "areal"), ("areal_bilateral", "areal (bilateral majority)")):
        nv = nb = 0
        area_t = area_b = 0.0
        diffs, wdiffs = [], []
        for h in HEMIS:
            d = res[h]
            v = d["valid"]
            true_t = d["vertex"][v]
            got_t = d[key][v]
            bad = true_t != got_t
            nv += int(v.sum())
            nb += int(bad.sum())
            area_t += float(va[h][v].sum())
            area_b += float(va[h][v][bad].sum())
            if bad.any():
                diffs.append(np.abs(true_t[bad] - got_t[bad]).astype(float))
                wdiffs.append(va[h][v][bad])
        dd = np.concatenate(diffs) if diffs else np.zeros(0)
        ww = np.concatenate(wdiffs) if wdiffs else np.zeros(0)
        rows.append({
            "resolution": label,
            "n_labelled_vertices": nv, "n_misassigned_vertices": nb,
            "fraction_vertices": nb / nv,
            "area_mm2_labelled": round(area_t, 1), "area_mm2_misassigned": round(area_b, 1),
            "fraction_area": area_b / area_t,
            "mean_abs_ordinal_diff": float(dd.mean()) if dd.size else 0.0,
            "mean_abs_ordinal_diff_area_weighted":
                float(np.average(dd, weights=ww)) if dd.size else 0.0,
            "max_abs_ordinal_diff": int(dd.max()) if dd.size else 0})
    return rows


# --------------------------------------------------------------------------- #
# Panel c parcel choice
# --------------------------------------------------------------------------- #


def parcel_purity(res: dict, va: dict, log=print) -> dict:
    """Per-parcel purity, the statistic section 3.3 quotes, next to the total.

    Section 3.3's "the median classical parcel is about 67% a single type" is a median
    over parcels, each parcel counting once. The misassignment fraction is a total over
    vertices, so large parcels count for more. The two are different numbers whenever
    parcel size correlates with heterogeneity, which is why both are reported here.
    """
    per: dict[str, dict[int, float]] = {}
    for h in HEMIS:
        d = res[h]
        v = d["valid"]
        for p, t, a in zip(d["parcel"][v], d["vertex"][v], va[h][v]):
            key = f"{h}.{p}"
            per.setdefault(key, {})[int(t)] = per.setdefault(key, {}).get(int(t), 0.0) + float(a)
    pur, sizes = [], []
    for k, d in per.items():
        tot = sum(d.values())
        pur.append(max(d.values()) / tot)
        sizes.append(tot)
    pur = np.asarray(pur)
    sizes = np.asarray(sizes)
    r = float(np.corrcoef(sizes, pur)[0, 1])
    out = {"n_parcels_per_hemi_pooled": int(pur.size),
           "median_parcel_purity": float(np.median(pur)),
           "mean_parcel_purity": float(pur.mean()),
           "area_weighted_purity": float(np.average(pur, weights=sizes)),
           "corr_size_vs_purity": r}
    log(f"    per-parcel purity: median {100*np.median(pur):.1f}%, mean "
        f"{100*pur.mean():.1f}%, area-weighted {100*np.average(pur, weights=sizes):.1f}%; "
        f"size vs purity r = {r:+.2f}")
    return out


def choose_parcel(res: dict, va: dict, log=print) -> dict:
    """Most heterogeneous Desikan parcel: most types above 10%, ties on size."""
    comp: dict[str, dict[int, float]] = {}
    for h in HEMIS:
        d = res[h]
        v = d["valid"]
        for p, t, a in zip(d["parcel"][v], d["vertex"][v], va[h][v]):
            comp.setdefault(p, {})[int(t)] = comp.setdefault(p, {}).get(int(t), 0.0) + float(a)
    cand = []
    for p, d in comp.items():
        tot = sum(d.values())
        above = sorted(t for t, a in d.items() if a / tot > 0.10)
        cand.append({"parcel": p, "n_types_above_10pct": len(above), "types": above,
                     "area_mm2": round(tot, 1),
                     "pct": {int(t): round(100 * a / tot, 1) for t, a in sorted(d.items())}})
    cand.sort(key=lambda r: (-r["n_types_above_10pct"], -r["area_mm2"]))
    log(f"    most heterogeneous parcels: " + ", ".join(
        f"{c['parcel']} ({c['n_types_above_10pct']} types, {c['area_mm2']:.0f} mm2)"
        for c in cand[:4]))
    return cand[0] | {"runners_up": cand[1:4]}


# --------------------------------------------------------------------------- #
# QC: adjacent |delta type| >= 2 in the categorical panels
# --------------------------------------------------------------------------- #


def qc_skip_edges(res: dict, geom: dict, log=print) -> dict:
    """Count adjacent vertex pairs whose assigned types differ by 2 or more.

    The flat-panel work established that these are what a reader sees as a spurious
    seam. At vertex level the released map has none on its native mesh; a coarsened
    map can legitimately have them, since a parcel boundary is free to jump two
    types, so this is reported rather than repaired.
    """
    out = {}
    for key in ("vertex", "crossed", "areal"):
        tot = 0
        for h in HEMIS:
            d = res[h]
            faces = geom[h]["faces"]
            lab = d[key]
            v = d["valid"]
            e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
            ok = v[e[:, 0]] & v[e[:, 1]]
            e = e[ok]
            dif = np.abs(lab[e[:, 0]].astype(int) - lab[e[:, 1]].astype(int))
            tot += int((dif >= 2).sum())
        out[key] = tot
    log(f"    adjacent |delta type| >= 2 edges: " +
        ", ".join(f"{k} {v}" for k, v in out.items()))
    return out


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def _cat_cmap():
    return ListedColormap([CYTO7_VIRIDIS[c] for c in range(1, 8)])


def _panel(ax, infl, data, hemi, view, sulc, cmap, vmin, vmax):
    from nilearn import plotting
    plotting.plot_surf_roi(infl, roi_map=data, hemi=NL[hemi], view=view, axes=ax,
                           cmap=cmap, vmin=vmin, vmax=vmax, bg_map=sulc,
                           bg_on_data=True, colorbar=False, avg_method="median")


def _contours(ax, infl, labels, hemi, levels, colour="white", lw=0.35):
    from nilearn import plotting
    try:
        plotting.plot_surf_contours(infl, labels, axes=ax, levels=levels,
                                    colors=[colour] * len(levels), linewidths=lw)
    except Exception:
        pass


def boundary_mask(labels, faces, valid):
    """Vertices on a node or parcel boundary, for thin white node outlines.

    Folded into the categorical array as an extra colour code rather than drawn as a
    second surface: overlaying a second nilearn call paints an opaque mesh that hides
    the layer underneath, and per-level contour tracing needs one pass per node.
    """
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    ok = valid[e[:, 0]] & valid[e[:, 1]]
    e = e[ok]
    d = labels[e[:, 0]] != labels[e[:, 1]]
    b = np.zeros(labels.shape[0], bool)
    b[e[d, 0]] = True
    b[e[d, 1]] = True
    return b


def _cat_cmap_with_boundary():
    """Seven viridis type colours plus code 8 for a white node boundary."""
    return ListedColormap([CYTO7_VIRIDIS[c] for c in range(1, 8)] + [(1.0, 1.0, 1.0)])


def _cat_panel(ax, fa, h, view, data, valid, boundary=None):
    from nilearn import plotting as pl
    d = np.asarray(data, float).copy()
    if boundary is not None:
        d = d.copy()
        d[boundary & valid] = 8.0
    d[~valid] = np.nan
    pl.plot_surf_roi(fa[h]["infl"], roi_map=d, hemi=NL[h], view=view, axes=ax,
                     cmap=_cat_cmap_with_boundary(), vmin=0.5, vmax=8.5,
                     bg_map=fa[h]["sulc"], bg_on_data=True, colorbar=False,
                     avg_method="median")
    ax.set_axis_off()


def _delta_panel(ax, fa, h, view, assigned, truth, valid):
    """Signed ordinal difference on the diverging scale used by Figure 2b.

    Shifted into 0 to 6 so plot_surf_roi accepts it (it rejects negatives) and so the
    scale is pinned rather than autoscaled: an all-zero row, which is exactly what the
    vertex row is by construction, would otherwise collapse the colour normalisation.
    """
    from nilearn import plotting as pl
    # Clipped to the plus or minus 3 scale Figure 2b already uses. The observed range
    # runs to 5 ordinal steps, but those are rare and letting them set the scale would
    # flatten the plus or minus 1 and 2 differences that carry the panel. Stated in the
    # caption rather than left implicit.
    sd = np.clip(np.asarray(assigned, float) - np.asarray(truth, float), -3, 3) + 3.0
    sd[~valid] = np.nan
    pl.plot_surf_roi(fa[h]["infl"], roi_map=sd, hemi=NL[h], view=view, axes=ax,
                     cmap=SIGNED_DELTA.cmap, vmin=0.0, vmax=6.0,
                     bg_map=fa[h]["sulc"], bg_on_data=True, colorbar=False,
                     avg_method="median")
    ax.set_axis_off()


def _zoom_to(ax, coords, mask, pad=1.25):
    """Tighten a rendered 3D surface axes onto one parcel's bounding box."""
    c = coords[mask]
    if c.size == 0:
        return
    ctr = c.mean(0)
    half = float(np.abs(c - ctr).max()) * pad
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)


ROW_KEYS = ("vertex", "crossed", "areal")


def build_figure(res_all, allrows, chosen, fa, infl_coords, counts, out_png, dpi,
                 log=print):
    """Panels a to d at the house width of 190 mm, viridis type palette."""
    import matplotlib as mpl
    from matplotlib.patches import Patch
    from nilearn import plotting as pl

    res = res_all["aparc_cyto7"]
    W = 190 / 25.4
    fig = plt.figure(figsize=(W, W * 1.14))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 5, width_ratios=[0.02, 1, 1, 1, 1],
                          hspace=0.01, wspace=0.005,
                          left=0.158, right=0.995, top=0.905, bottom=0.435)
    views = [("lh", "lateral"), ("lh", "medial")]

    bmask = {}
    for h in HEMIS:
        f = res[h]["faces"]
        bmask[(h, "crossed")] = boundary_mask(res[h]["node_index"], f, res[h]["valid"])
        bmask[(h, "areal")] = boundary_mask(res[h]["parcel_index"], f, res[h]["valid"])

    row_label = {
        "vertex": "vertex\n{:,} labelled\nvertices".format(counts["vertex"]),
        "crossed": "crossed\n{} nodes".format(counts["crossed"]),
        "areal": "areal\n{} parcels".format(counts["areal"])}

    for r, key in enumerate(ROW_KEYS):
        # figure coordinates, not an axes: a nilearn 3D axes reserves wide internal
        # margins and paints over anything placed in the neighbouring grid cell.
        top, bot = 0.905, 0.435
        yc = top - (r + 0.5) * (top - bot) / 3.0
        fig.text(0.150, yc, row_label[key], ha="right", va="center", fontsize=8.0,
                 linespacing=1.5)
        for c, (h, view) in enumerate(views):
            ax = fig.add_subplot(gs[r, 1 + c], projection="3d")
            _cat_panel(ax, fa, h, view, res[h][key], res[h]["valid"],
                       boundary=bmask.get((h, key)))
            ax2 = fig.add_subplot(gs[r, 3 + c], projection="3d")
            _delta_panel(ax2, fa, h, view, res[h][key], res[h]["vertex"],
                         res[h]["valid"])
        log("    row {} rendered".format(key))

    fig.text(0.012, 0.930, "a   cascade: three resolutions of the same map",
             fontsize=9.0, weight="bold", ha="left")
    fig.text(0.560, 0.930, "b   where the loss falls",
             fontsize=9.0, weight="bold", ha="left")

    # ---- panel c: one parcel, three resolutions -------------------------- #
    pname = chosen["parcel"]
    h = "lh"
    pm = res[h]["valid"] & (res[h]["parcel"] == pname)
    gc = fig.add_gridspec(1, 3, left=0.045, right=0.560, top=0.375, bottom=0.130,
                          wspace=0.01)
    for i, key in enumerate(ROW_KEYS):
        ax = fig.add_subplot(gc[0, i], projection="3d")
        d = np.asarray(res[h][key], float).copy()
        if key in ("crossed", "areal"):
            d[bmask[(h, key)] & pm] = 8.0
        d[~pm] = np.nan
        pl.plot_surf_roi(fa[h]["infl"], roi_map=d, hemi=NL[h], view="lateral", axes=ax,
                         cmap=_cat_cmap_with_boundary(), vmin=0.5, vmax=8.5,
                         bg_map=fa[h]["sulc"], bg_on_data=True, colorbar=False,
                         avg_method="median")
        _zoom_to(ax, infl_coords[h], pm)
        ax.set_axis_off()
        ax.text2D(0.5, -0.02, key, transform=ax.transAxes, ha="center", va="top",
                  fontsize=7.6)
    fig.text(0.012, 0.400,
             "c   one parcel in close-up: {} (left hemisphere)".format(pname),
             fontsize=9.0, weight="bold", ha="left")

    # ---- panel d: the quantification ------------------------------------- #
    dax = fig.add_axes([0.680, 0.150, 0.290, 0.205])
    atlases = ["Desikan-Killiany", "von Economo"]
    vals = {}
    for a in atlases:
        vals[a] = [next(r["fraction_vertices"] for r in allrows
                        if r["atlas"] == a and r["resolution"] == k) * 100
                   for k in ROW_KEYS]
    x = np.arange(3)
    wdt = 0.36
    for j, a in enumerate(atlases):
        bars = dax.bar(x + (j - 0.5) * wdt, vals[a], wdt,
                       color=["#4c72b0", "#dd8452"][j], label=a, zorder=3)
        for b, v in zip(bars, vals[a]):
            dax.text(b.get_x() + b.get_width() / 2, v + 0.9,
                     ("{:.2f}".format(v) if v < 5 else "{:.1f}".format(v)),
                     ha="center", va="bottom", fontsize=6.6)
    dax.set_xticks(x)
    dax.set_xticklabels(ROW_KEYS, fontsize=7.4)
    dax.set_ylabel("type misassignment\n(% of labelled vertices)", fontsize=7.2)
    dax.set_ylim(0, 52)
    dax.set_xlim(-0.6, 2.6)
    dax.tick_params(labelsize=6.8)
    dax.grid(axis="y", alpha=0.25, zorder=0)
    for sp in ("top", "right"):
        dax.spines[sp].set_visible(False)
    dax.legend(fontsize=6.4, frameon=False, loc="upper left",
               bbox_to_anchor=(-0.02, 1.02), handlelength=1.0)
    dax.axhline(100 / 3, ls=":", lw=0.9, color="0.3", zorder=2)
    dax.text(-0.55, 100 / 3 + 0.9, "one third", fontsize=6.2, color="0.3",
             ha="left", va="bottom")
    fig.text(0.640, 0.400, "d   how much each step discards",
             fontsize=9.0, weight="bold", ha="left")

    # ---- legends ---------------------------------------------------------- #
    handles = [Patch(facecolor=CYTO7_VIRIDIS[c], edgecolor="0.4",
                     label=CYTO7_NAMES[c - 1]) for c in range(1, 8)]
    handles.append(Patch(facecolor="white", edgecolor="0.4", label="node boundary"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.020, 0.010),
               ncol=4, fontsize=6.8, frameon=False, handlelength=1.1,
               columnspacing=0.9, handletextpad=0.4)
    cax = fig.add_axes([0.700, 0.048, 0.240, 0.013])
    mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap(SIGNED_DELTA.cmap),
                              norm=mpl.colors.Normalize(-3, 3),
                              orientation="horizontal", ticks=[-3, 0, 3])
    cax.tick_params(labelsize=6.2, length=2, pad=1)
    cax.set_title("panel b: assigned minus true ordinal type", fontsize=6.2, pad=2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=dpi, facecolor="white")
    plt.close(fig)
    log("    wrote {}".format(out_png))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log, fh = _log_factory(OUTDIR / "run_rr11.log")
    t0 = time.time()

    frozen_before = rc.frozen_hashes()
    extra_before = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    log(f"frozen: {len(frozen_before)} released v9 files + {len(extra_before)} RR10 files")

    cyto = resolve_target_map("v9", "fsaverage")
    g = rc.fsaverage_geometry("white", "164k")
    va = {h: rc.vertex_areas(*g[HKEY[h]]) for h in HEMIS}
    geom = {h: {"faces": g[HKEY[h]][1]} for h in HEMIS}

    allrows, chosen, qc, purity = [], None, {}, {}
    res_all = {}
    for tag, (pretty, path_fn) in ATLASES.items():
        log(f"\n===== {tag} ({pretty}) =====")
        res = build_resolutions(tag, path_fn, cyto, va, geom, log=log)
        res_all[tag] = res
        rows = misassignment(res, va, log=log)
        for r in rows:
            r["atlas"] = pretty
            r["n_nodes"] = {"vertex": r["n_labelled_vertices"],
                            "crossed": None, "areal": res["n_parcel_names"],
                            "areal (bilateral majority)": res["n_parcel_names"]}[r["resolution"]]
            log(f"  {r['resolution']:28s} misassigned {r['n_misassigned_vertices']:>7,} / "
                f"{r['n_labelled_vertices']:,} vertices = {100*r['fraction_vertices']:5.2f}%  "
                f"(area {100*r['fraction_area']:5.2f}%)  mean |d ordinal| "
                f"{r['mean_abs_ordinal_diff']:.2f}")
        allrows += rows
        qc[tag] = qc_skip_edges(res, geom, log=log)
        purity[tag] = parcel_purity(res, va, log=log)
        if tag == "aparc_cyto7":
            chosen = choose_parcel(res, va, log=log)

    with open(OUTDIR / "misassignment_fractions.csv", "w", newline="",
              encoding="utf-8") as f:
        cols = ["atlas", "resolution", "n_nodes", "n_labelled_vertices",
                "n_misassigned_vertices", "fraction_vertices", "area_mm2_labelled",
                "area_mm2_misassigned", "fraction_area", "mean_abs_ordinal_diff",
                "mean_abs_ordinal_diff_area_weighted", "max_abs_ordinal_diff"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in allrows:
            w.writerow({k: r.get(k) for k in cols})
    log(f"\nwrote {OUTDIR / 'misassignment_fractions.csv'}")

    if not args.stats_only:
        log("\n== building the figure ==")
        from nilearn import datasets, surface
        fsa = datasets.fetch_surf_fsaverage(mesh="fsaverage")
        fa = {h: {"infl": fsa["infl_" + NL[h]], "sulc": fsa["sulc_" + NL[h]]}
              for h in HEMIS}
        infl_coords = {h: surface.load_surf_mesh(fsa["infl_" + NL[h]])[0] for h in HEMIS}
        counts = {"vertex": int(res_all["aparc_cyto7"]["lh"]["valid"].sum()
                                + res_all["aparc_cyto7"]["rh"]["valid"].sum()),
                  "crossed": 101, "areal": res_all["aparc_cyto7"]["n_parcel_names"]}
        fig_path = FIGDIR / "cyto7_supp_resolution_cascade.png"
        build_figure(res_all, allrows, chosen, fa, infl_coords, counts,
                     fig_path, args.dpi, log=log)
        STAGED.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, STAGED / "cyto7_supp_resolution_cascade.png")
        shutil.copy2(fig_path, OUTDIR / "cyto7_supp_resolution_cascade.png")
        log(f"  staged into {STAGED / 'cyto7_supp_resolution_cascade.png'}")

    ok, detail = rc.check_frozen(frozen_before)
    extra_after = {p.name: rc.sha256(p) for p in FROZEN_EXTRA if p.exists()}
    ok_extra = extra_before == extra_after
    log(f"released v9 unchanged: {ok}; RR10 crossed files unchanged: {ok_extra}")

    (OUTDIR / "rr11_summary.json").write_text(json.dumps({
        "misassignment": allrows, "panel_c_parcel": chosen, "qc_skip_edges": qc,
        "parcel_purity": purity,
        "frozen_v9_unchanged": ok, "frozen_rr10_unchanged": ok_extra,
        "runtime_seconds": round(time.time() - t0, 1)}, indent=2, default=float),
        encoding="utf-8")
    log(f"total runtime {(time.time()-t0)/60:.1f} min")
    fh.close()
    return allrows, chosen, res_all


if __name__ == "__main__":
    main()
