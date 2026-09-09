"""Structure-function feature gallery (SPEC_feature_gallery.md).

One row per neuromaps/HCP feature: the feature on the 32k fs_LR inflated surface
(LH/RH, lateral + medial) with cyto7 v9 type borders overlaid (magma), plus a
box-and-whisker-by-cyto7-type panel annotated with the spin-tested Spearman rho,
spin p and FDR q (from figures/functional_summary_table.csv).

  --set main  : FDR-significant features (myelin, thickness, functional gradient)  -> Fig 4
  --set supp  : intrinsic timescale + delta/theta/alpha/beta/gamma power           -> Supp fig

Colour scheme (kept to three roles): viridis = cyto7 label/support; red-blue =
signed difference (Fig 2); magma = every observed feature surface (here).

Run::  conda activate cyto7 && python scripts/feature_gallery.py --set main
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from cyto7_surface_io import (scatter_dscalar_to_surface, surface_path, thickness_dscalar_path,  # noqa: E402
                              myelin_dscalar_path, load_cached_feature)

CACHE = REPO / "resources" / "cyto7_derived" / "cache"
OUTDIR = REPO / "figures" / "v9" / "manuscript"
NAMES = {1: "Allo", 2: "Agr", 3: "Dys", 4: "EulI", 5: "EulII", 6: "EulIII", 7: "Konio"}
VIEWS = [("L", "lateral"), ("L", "medial"), ("R", "lateral"), ("R", "medial")]


def _meg(desc):
    return lambda: load_cached_feature("hcps1200", desc)


SETS = {
    "main": [
        ("myelin", "T1w/T2w myelin", "ratio", lambda: scatter_dscalar_to_surface(myelin_dscalar_path("Validation210"))),
        ("thickness", "Cortical thickness", "mm", lambda: scatter_dscalar_to_surface(thickness_dscalar_path("Validation210"))),
        ("gradient", "Principal functional gradient", "a.u.", lambda: load_cached_feature("margulies2016", "fcgradient01")),
    ],
    "supp": [
        ("timescale", "Intrinsic timescale", "a.u.", _meg("megtimescale")),
        ("delta", "Delta power (0.5--4 Hz)", "a.u.", _meg("megdelta")),
        ("theta", "Theta power (4--8 Hz)", "a.u.", _meg("megtheta")),
        ("alpha", "Alpha power (8--12 Hz)", "a.u.", _meg("megalpha")),
        ("beta", "Beta power (12--30 Hz)", "a.u.", _meg("megbeta")),
        ("gamma", "Gamma power (30--60 Hz)", "a.u.", _meg("meggamma1")),
    ],
}
DIVERGING = {"gradient"}  # rendered in magma anyway (uniform gallery); value can be negative


def proj(H, view, c):
    y, z = c[:, 1], c[:, 2]
    return (-y, z) if (H, view) in (("L", "lateral"), ("R", "medial")) else (y, z)


def precompute(lab, geo):
    G = {}
    for H, view in VIEWS:
        coords, faces = geo[H]; lb = lab[H]
        n = np.cross(coords[faces][:, 1] - coords[faces][:, 0], coords[faces][:, 2] - coords[faces][:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        fr = (n[:, 0] < 0) if (H, view) in (("L", "lateral"), ("R", "medial")) else (n[:, 0] > 0)
        F = faces[fr]; fx = coords[F].mean(1)[:, 0]
        order = np.argsort(-fx) if (H, view) in (("L", "lateral"), ("R", "medial")) else np.argsort(fx)
        F = F[order]; sx, sy = proj(H, view, coords); verts = np.stack([sx[F], sy[F]], axis=-1)
        sh = 0.62 + 0.38 * np.clip(np.abs(n[fr, 2][order]), 0, 1); medf = (lb[F] == 0).sum(1) >= 2
        E = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), axis=1), axis=0)
        frv = np.zeros(len(coords), bool); frv[F.ravel()] = True
        bd = (lb[E[:, 0]] != lb[E[:, 1]]) & (lb[E[:, 0]] > 0) & (lb[E[:, 1]] > 0) & frv[E[:, 0]] & frv[E[:, 1]]
        segs = np.stack([np.stack([sx[E[bd, 0]], sy[E[bd, 0]]], 1), np.stack([sx[E[bd, 1]], sy[E[bd, 1]]], 1)], 1)
        G[(H, view)] = dict(F=F, verts=verts, sh=sh, medf=medf, segs=segs, lim=(sx.min(), sx.max(), sy.min(), sy.max()))
    return G


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="setname", choices=["main", "supp"], default="main")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)
    feats = SETS[args.setname]
    stats = pd.read_csv(REPO / "figures" / "functional_summary_table.csv").groupby("FeatureKey").first()
    lab = {H: np.load(CACHE / f"v9_labels_fsLR32k_hemi-{H}.npy") for H in ("L", "R")}
    geo = {H: (lambda g: (np.asarray(g.darrays[0].data, float), np.asarray(g.darrays[1].data, int)))(
        nib.load(str(surface_path("Validation210", H, "inflated")))) for H in ("L", "R")}
    G = precompute(lab, geo)

    nrow = len(feats)
    fig = plt.figure(figsize=(19, 3.6 * nrow)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(nrow, 7, width_ratios=[1, 1, 1, 1, 0.06, 0.65, 1.45], wspace=0.04, hspace=0.55)
    cmap = plt.get_cmap("magma")
    for r, (key, name, unit, loader) in enumerate(feats):
        d = loader(); div = key in DIVERGING
        allv = np.concatenate([d[H][lab[H] > 0] for H in ("L", "R")]); allv = allv[np.isfinite(allv)]
        if not div:
            allv = allv[allv > 0]
        vmin, vmax = np.percentile(allv, 2), np.percentile(allv, 98); nm = Normalize(vmin, vmax)
        for ci, (H, view) in enumerate(VIEWS):
            ax = fig.add_subplot(gs[r, ci]); g = G[(H, view)]; val = d[H]
            fv = np.nanmean(val[g["F"]], axis=1); col = cmap(nm(fv))[:, :3]
            bad = g["medf"] | ~np.isfinite(fv)
            if not div:
                bad = bad | (fv <= 0)
            col[bad] = (0.68, 0.85, 0.90); col = np.clip(col * g["sh"][:, None], 0, 1)
            ax.add_collection(PolyCollection(g["verts"], facecolors=col, edgecolors="face", linewidths=0.3))
            ax.add_collection(LineCollection(g["segs"], colors="white", linewidths=0.3, alpha=0.6))
            x0, x1, y0, y1 = g["lim"]; ax.set_xlim(x0 - 2, x1 + 2); ax.set_ylim(y0 - 2, y1 + 2)
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                ax.set_title(f"{H}H {view}", fontsize=10)
            if ci == 0:
                ax.text(-0.06, 0.5, name.split(" (")[0].replace(" ", "\n", 1), transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10, weight="bold")
        cax = fig.add_subplot(gs[r, 4]); cb = fig.colorbar(cm.ScalarMappable(norm=nm, cmap=cmap), cax=cax)
        cb.set_label(f"{name} ({unit})", fontsize=9)
        axb = fig.add_subplot(gs[r, 6])
        for t in range(1, 8):
            for H, off, fc, mc in (("L", -0.18, "0.35", "w"), ("R", 0.18, "0.78", "0.2")):
                vals = d[H][(lab[H] == t) & np.isfinite(d[H]) & ((d[H] > 0) | div)]
                axb.boxplot(vals, positions=[t + off], widths=0.32, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor=fc, edgecolor="0.3"), medianprops=dict(color=mc),
                            whiskerprops=dict(color=fc), capprops=dict(color=fc))
        s = stats.loc[key]; sig = "*" if s.p_spin_fdr < 0.05 else ""
        axb.set_xticks(range(1, 8)); axb.set_xticklabels([NAMES[t] for t in range(1, 8)], rotation=30, ha="right", fontsize=8)
        axb.set_ylabel(unit, fontsize=9); axb.set_xlim(0.4, 7.6); axb.grid(axis="y", ls=":", alpha=0.4)
        axb.set_title(f"{name} by cyto7 type   (Spearman $\\rho$={s.spearman_rho:+.2f}, "
                      f"spin p={s.p_spin:.3f}, q={s.p_spin_fdr:.3f}{sig})", fontsize=9.5)
        if r == 0:
            axb.legend(handles=[Patch(facecolor="0.35", label="LH"), Patch(facecolor="0.78", label="RH")],
                       loc="upper right", fontsize=8, frameon=False)
    tag = "FDR-significant features" if args.setname == "main" else "remaining features (spin n.s.)"
    fig.suptitle(f"Structure--function feature gallery (surface + cyto7 borders, magma; box-by-type + spin-tested $\\rho$)  ---  {tag}",
                 fontsize=13, weight="bold", y=0.995)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"feature_gallery_{args.setname}.png"
    fig.savefig(str(out), dpi=args.dpi, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
