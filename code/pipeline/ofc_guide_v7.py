"""Orbitofrontal (OFC) painting / adjudication guide for a cyto7 version (default v7),
inflated + ventral view. Mirrors ``insula_guide_v7.py`` for GC to adjudicate the OFC.

Fill = current cyto7 type. Overlay A (magenta) = von-Economo agranular OFC field
(FLMN+FJK ∩ Desikan OFC; areal → an upper bound on extent). Overlay B (cyan) = the
finer Destrieux caudomedial agranular candidate (gyrus rectus / subcallosal / medial
+ suborbital orbital sulci — the OFC analog of the frontoinsular insula sector).

**Adjudication aid only — no map edits, 7 labels.** v7's OFC agranular (~10%) already
matches von Economo (~13%); v7 is if anything more conservative (a dysgranular band vE
lumps into eulaminate I). The guide poses the caudomedial-extent question to GC; it does
not assert v7 is wrong. nibabel + matplotlib only.

Usage:  python scripts/ofc_guide_v7.py [v7]
"""
import os, sys
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURF = ROOT + "/resources/fsaverage_surfaces/{h}.inflated"
ANN = ROOT + "/resources/cyto7_derived/pial.{h}.cyto7.{v}.annot"
APARC = ROOT + "/resources/voneconomo/{h}.aparc.annot"
DEST = ROOT + "/resources/refine_atlases/{h}.aparc.a2009s.annot"
ECON = ROOT + "/resources/voneconomo/{h}.economo.annot"
VER = sys.argv[1] if len(sys.argv) > 1 else "v7"
OUT = ROOT + f"/figures/v9/adjudication/ofc_paint_guide_{VER}.png"

CYTO7 = {1: (0.55, 0.20, 0.65), 2: (0.20, 0.35, 0.85), 3: (0.10, 0.62, 0.70),
         4: (0.25, 0.70, 0.35), 5: (0.62, 0.78, 0.20), 6: (0.98, 0.62, 0.10),
         7: (0.98, 0.85, 0.15)}
NAMES = {1: "Allocortex", 2: "Agranular", 3: "Dysgranular", 4: "Eulaminate I",
         5: "Eulaminate II", 6: "Eulaminate III", 7: "Koniocortex"}
OFC = ["medialorbitofrontal", "lateralorbitofrontal"]                       # Desikan
DEST_CM = ["G_rectus", "G_subcallosal", "S_orbital_med-olfact", "S_suborbital"]  # Destrieux
VE_AGR = ["FLMN", "FJK"]                                                    # von Economo
CODE_TO_SHORT = {1: "Allo", 2: "Agr", 3: "Dys", 4: "EulI", 5: "EulII", 6: "EulIII", 7: "Kon"}


def sub(path, h, w):
    lab, _c, nm = nib.freesurfer.io.read_annot(path.format(h=h))
    nm = [x.decode() if isinstance(x, bytes) else x for x in nm]
    idx = [i for i, x in enumerate(nm) if x in w]
    return np.isin(lab, idx)


def shade(coords, F, light=np.array([0.2, 0.3, -1.0])):   # light from below (ventral)
    t = coords[F]; n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = light / np.linalg.norm(light)
    return 0.55 + 0.45 * np.clip(np.abs(n @ light), 0, 1)


def _comp(lab, mask):
    t = lab[mask]; t = t[t > 0]
    if not len(t):
        return "(empty)", 0
    parts = [f"{CODE_TO_SHORT[c]} {100 * (t == c).sum() / len(t):.0f}%"
             for c in range(1, 8) if (t == c).sum()]
    return ", ".join(parts), int((t == 2).sum())


def render(ax, h, azim):
    coords, faces = nib.freesurfer.read_geometry(SURF.format(h=h))
    lab = np.asarray(nib.freesurfer.io.read_annot(ANN.format(h=h, v=VER))[0])
    ofc = sub(APARC, h, OFC)
    vetarg = sub(ECON, h, VE_AGR) & ofc          # areal agranular field ∩ OFC
    cm = sub(DEST, h, DEST_CM)                    # Destrieux caudomedial candidate
    c = coords[ofc]; lo = c.min(0) - 10; hi = c.max(0) + 10
    inv = np.all((coords >= lo) & (coords <= hi), axis=1)
    F = faces[inv[faces].all(axis=1)]
    sh = shade(coords, F)
    fmaj = np.array([np.bincount(lab[r][lab[r] > 0]).argmax() if (lab[r] > 0).any() else 0
                     for r in F])
    rgb = np.clip(np.array([CYTO7.get(int(t), (0.88, 0.88, 0.88)) for t in fmaj]) * sh[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(coords[F], facecolors=rgb, edgecolors="none",
                                         linewidths=0, shade=False))
    # Overlay A — vE agranular field (magenta); Overlay B — Destrieux candidate (white,
    # so it reads over the teal dysgranular fill that a cyan line would blend into)
    for mask, edge, lw in ((vetarg, (1.0, 0.0, 1.0, 0.85), 0.5),
                           (cm, (1.0, 1.0, 1.0, 0.9), 0.5)):
        fm = mask[F].sum(1) >= 2
        if fm.any():
            ax.add_collection3d(Poly3DCollection(coords[F[fm]], facecolors=(0, 0, 0, 0.0),
                                                 edgecolors=edge, linewidths=lw))
    for s, a in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        s(lo[a], hi[a])
    # ventral view: look up at the inferior surface (-z); same azim both hemis (RH is the
    # anatomical mirror of LH). anterior (+y) toward the top of the panel.
    ax.set_box_aspect(hi - lo); ax.view_init(elev=-89, azim=0); ax.set_axis_off()
    zm = hi[2]
    ax.text(coords[ofc][:, 0].mean(), hi[1] - 2, zm, "anterior", fontsize=9, style="italic", color="0.25")
    ax.text(coords[ofc][:, 0].mean(), lo[1] + 2, zm, "posterior", fontsize=9, style="italic", color="0.25")
    vc, va = _comp(lab, vetarg); cc, ca = _comp(lab, cm); oc, oa = _comp(lab, ofc)
    return {"ofc": (oc, oa, int(ofc.sum())), "vetarg": (vc, va, int(vetarg.sum())),
            "cm": (cc, ca, int(cm.sum()))}


fig = plt.figure(figsize=(15, 7)); fig.patch.set_facecolor("white"); info = {}
for i, (h, azim) in enumerate((("lh", 90), ("rh", -90))):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    info[h] = render(ax, h, azim)
    d = info[h]
    ax.set_title(f"{h.upper()} OFC ({VER}) — vE-target agranular {d['vetarg'][1]}/{d['vetarg'][2]}"
                 f" ({100 * d['vetarg'][1] / max(d['vetarg'][2], 1):.0f}%)",
                 fontsize=10.5, weight="bold")
    print(f"{h}: OFC n={d['ofc'][2]} [{d['ofc'][0]}]")
    print(f"    vE agranular field (FLMN+FJK in OFC) n={d['vetarg'][2]} [{d['vetarg'][0]}] agr_vtx={d['vetarg'][1]}")
    print(f"    Destrieux caudomedial candidate n={d['cm'][2]} [{d['cm'][0]}] agr_vtx={d['cm'][1]}")
handles = [Patch(facecolor=CYTO7[c], label=NAMES[c]) for c in range(1, 8)]
handles.append(Line2D([0], [0], color=(1, 0, 1), lw=2, label="von Economo FLMN+FJK agranular field ∩ OFC (areal → upper bound)"))
handles.append(Line2D([0], [0], color=(0.4, 0.4, 0.4), lw=2, label="Destrieux caudomedial agranular candidate (gyrus rectus / caudal orbital; white outline)"))
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.04))
fig.suptitle(f"cyto7 {VER} OFC (inflated, ventral): current type (fill) vs von-Economo agranular field "
             "(magenta) + finer caudomedial candidate (white)", fontsize=12.5, weight="bold", y=0.99)
fig.text(0.5, 0.005, "Adjudication aid — no map change. v7 OFC agranular ≈10% already matches von Economo (~13%); "
         "the dysgranular band is vE eulaminate I. Areal atlas overstates the agranular extent (upper bound).",
         ha="center", va="top", fontsize=8, color="0.35")
fig.subplots_adjust(bottom=0.2, top=0.9, wspace=0.02)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=155, facecolor="white", bbox_inches="tight")
print("saved", OUT)
