"""Orbitofrontal cortex (OFC) in the von Economo-Koskinas parcellation (inflated,
ventral view). Map-independent reference, mirrors ``insula_voneconomo.py``.

areas: von Economo OFC areas (FF, FE, FG, FH, FLMN, FJK …) labelled by acronym + GC type.
types: the same region coloured by the Garcia-Cabezas-derived cyto7 TYPE.

Purpose: show that von Economo places the AGRANULAR field at FLMN (parolfactory /
precommissural) + FJK (frontoinsular / frontal piriform), while the bulk of the OFC
(FF/FG/FH) is eulaminate I — i.e. the GC-consistent agranular target is the caudomedial
tip, and the areal atlas overstates its extent.

Usage:  python scripts/ofc_voneconomo.py [areas|types|both]   (nibabel + matplotlib only)
"""
import os, csv, sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURF = os.path.join(ROOT, "resources", "fsaverage_surfaces", "{h}.inflated")
ECON = os.path.join(ROOT, "resources", "voneconomo", "{h}.economo.annot")
APARC = os.path.join(ROOT, "resources", "voneconomo", "{h}.aparc.annot")
TYPES = os.path.join(ROOT, "resources", "voneconomo", "von_economo_cortical_types.csv")
OUT = os.path.join(ROOT, "figures", "v9", "adjudication", "ofc_voneconomo.png")

CYTO7 = {1: (0.55, 0.20, 0.65), 2: (0.20, 0.35, 0.85), 3: (0.10, 0.62, 0.70),
         4: (0.25, 0.70, 0.35), 5: (0.62, 0.78, 0.20), 6: (0.98, 0.62, 0.10),
         7: (0.98, 0.85, 0.15)}
CNAME = {0: "periallo/limbic", 1: "Allocortex", 2: "Agranular", 3: "Dysgranular",
         4: "Eulaminate I", 5: "Eulaminate II", 6: "Eulaminate III", 7: "Koniocortex"}
OFC = ["medialorbitofrontal", "lateralorbitofrontal"]   # Desikan OFC crop region

A2C, A2N = {}, {}
with open(TYPES, newline="") as f:
    for r in csv.DictReader(f):
        c = r["cyto7_code"].strip()
        A2C[r["economo_acronym"]] = int(c) if c else 0
        A2N[r["economo_acronym"]] = r["economo_area_name"]
QUAL = plt.get_cmap("tab20")


def am(h, w):   # aparc membership mask
    lab, _c, names = nib.freesurfer.io.read_annot(APARC.format(h=h))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    idx = [i for i, nm in enumerate(names) if nm in w]
    return np.isin(lab, idx)


def shade(coords, F, light=np.array([0.2, 0.3, -1.0])):
    t = coords[F]; n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9); light = light / np.linalg.norm(light)
    return 0.55 + 0.45 * np.clip(np.abs(n @ light), 0, 1)


def facemaj(vals, F):
    fl = vals[F]; out = np.zeros(len(F), int)
    for i in range(len(F)):
        v = fl[i]; v = v[v > 0]
        if len(v):
            out[i] = np.bincount(v).argmax()
    return out


def load(hemi):
    coords, faces = nib.freesurfer.read_geometry(SURF.format(h=hemi))
    elab, _c, en = nib.freesurfer.io.read_annot(ECON.format(h=hemi))
    en = [n.decode() if isinstance(n, bytes) else n for n in en]
    ofc = am(hemi, OFC); c = coords[ofc]; lo = c.min(0) - 10; hi = c.max(0) + 10
    inv = np.all((coords >= lo) & (coords <= hi), axis=1)
    F = faces[inv[faces].all(axis=1)]
    return coords, F, np.asarray(elab), en, lo, hi, ofc


def panel(ax, coords, F, facecol, lo, hi, ofc):
    ax.add_collection3d(Poly3DCollection(coords[F], facecolors=facecol, edgecolors="none",
                                         linewidths=0, shade=False))
    for s, a in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        s(lo[a], hi[a])
    ax.set_box_aspect(hi - lo); ax.view_init(elev=-89, azim=0); ax.set_axis_off()
    xm, zm = coords[ofc][:, 0].mean(), hi[2]
    ax.text(xm, hi[1] - 2, zm, "anterior", fontsize=9, style="italic", color="0.25")
    ax.text(xm, lo[1] + 2, zm, "posterior", fontsize=9, style="italic", color="0.25")


def main(which="both", dpi=150):
    data = {}; present = []
    for hemi in ("lh", "rh"):
        coords, F, elab, en, lo, hi, ofc = load(hemi)
        fa = facemaj(elab, F)
        data[hemi] = (coords, F, en, fa, lo, hi, ofc)
        for fid in np.unique(fa):
            if fid > 0 and en[fid] not in present:
                present.append(en[fid])
    present = sorted(present, key=lambda a: (A2C.get(a, 9), a))
    acol = {a: QUAL(i % 20)[:3] for i, a in enumerate(present)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def render_fig(mode):
        fig = plt.figure(figsize=(15, 7)); fig.patch.set_facecolor("white")
        for col, hemi in enumerate(("lh", "rh")):
            coords, F, en, fa, lo, hi, ofc = data[hemi]; sh = shade(coords, F)
            ax = fig.add_subplot(1, 2, col + 1, projection="3d")
            if mode == "areas":
                rgb = np.array([acol.get(en[fid], (0.9, 0.9, 0.9)) if fid > 0 else (0.9, 0.9, 0.9) for fid in fa])
                ax.set_title(f"{hemi.upper()} — von Economo areas", fontsize=12, weight="bold")
            else:
                ft = np.array([A2C.get(en[fid], 0) if fid > 0 else 0 for fid in fa])
                rgb = np.array([CYTO7.get(int(t), (0.88, 0.88, 0.88)) for t in ft])
                ax.set_title(f"{hemi.upper()} — von Economo → cyto7 type", fontsize=12, weight="bold")
            panel(ax, coords, F, np.clip(rgb * sh[:, None], 0, 1), lo, hi, ofc)
        if mode == "areas":
            leg = [Patch(facecolor=acol[a], label=f"{a}: {A2N.get(a, '')[:26]} ({CNAME[A2C.get(a, 0)]})") for a in present]
            fig.legend(handles=leg, loc="center left", bbox_to_anchor=(0.87, 0.5), fontsize=6.5,
                       frameon=False, title="von Economo areas (near OFC)")
            sub = "areas"
        else:
            leg = [Patch(facecolor=CYTO7[c], label=CNAME[c]) for c in range(1, 8)]
            leg.append(Patch(facecolor=(0.88, 0.88, 0.88), label=CNAME[0]))
            fig.legend(handles=leg, loc="center left", bbox_to_anchor=(0.87, 0.5), fontsize=8,
                       frameon=False, title="cyto7 type (GC mapping)")
            sub = "types"
        fig.suptitle("OFC in von Economo-Koskinas (inflated, ventral): FLMN/FJK = agranular OFC field "
                     "(areal → upper bound); FF/FG/FH = eulaminate I; FFα = dysgranular",
                     fontsize=11.5, weight="bold", y=0.99)
        fig.subplots_adjust(left=0.01, right=0.86, top=0.9, bottom=0.02, wspace=0.02)
        p = OUT.replace(".png", f"_{sub}.png")
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight"); plt.close(fig)
        print("saved", p)

    if which in ("areas", "both"):
        render_fig("areas")
    if which in ("types", "both"):
        render_fig("types")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "both")
