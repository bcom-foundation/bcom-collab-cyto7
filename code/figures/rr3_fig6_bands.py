"""RR3 - Figure 6 rebuilt as a declared headline panel with separated statistical units.

Invoked by ``fig5_structural_model_gradients.py --rr3``. Keeps that figure's width,
DPI, RdBu_r signed-bar encoding and filled/open survivor markers, and changes three
things review item 11 asks for:

  1. the twelve measures are grouped into three labelled bands by statistical unit
     (vertex, parcel, subject) with a light grey tint, so unlike statistics are not
     read off one continuous axis,
  2. RORB is included, at rho = +0.78 the strongest association in the paper,
     together with the three MEG measures that survive but were omitted,
  3. the inclusion rule is stated inside the panel, so the figure declares itself a
     headline selection rather than implying a census.

Every rho and q is read from figures/v9/review_response/rr2_table/outcome_table.csv.
Nothing is hard-coded, so the panel cannot disagree with the outcome table.
"""
from __future__ import annotations

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import rr_common as rc

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg  # noqa: F401  (puts sibling code dirs on sys.path)

OUTDIR = rc.OUT / "rr3_fig6"
TABLE = rc.OUT / "rr2_table" / "outcome_table.csv"
MANUSCRIPT_FIGS = (rc.REPO_ROOT / "manuscript" / "preprint" / "26th_August_2026" / "figures")
FIGNAME = "cyto7_structural_model_gradients.png"
DPI = 600

# band label -> [(display name, key in the outcome table)]
BANDS = [
    ("Vertex-level (32k fs_LR vertices)", [
        ("Gene expression PC1 (AHBA)", "genepc1"),
        ("T1w/T2w myelin", "myelin"),
        ("Ionotropic/metabotropic index", "composite_iono_minus_metabo_index"),
        # Evolutionary expansion removed by RR29. The comparison is no longer reported:
        # the available expansion estimates rest on cross-species alignments whose
        # phylogenetic assumptions differ from the cortical-type framework, so it moves
        # to future work as a direct human-to-macaque comparison in matched types.
        # It was a non-survivor, so the survivor count is unchanged at 10 of now 11.
        ("Receptor diversity", "diversity_shannon_entropy_H"),
        ("Cortical thickness", "thickness"),
        ("Functional gradient", "gradient"),
    ]),
    ("Parcel-level (Desikan-68 parcels)", [
        ("RORB, layer-IV marker", "layer_RORB"),
    ]),
    ("Subject-level (N = 89, group mean of per-subject $\\rho$)", [
        ("Spectral centroid (MEG)", "centroid"),
        ("Aperiodic-corrected $\\beta$ power (MEG)", "osc_beta"),
        ("Slow/fast band ratio (MEG)", "sf_ratio"),
        ("Intrinsic timescale (MEG)", "int_area"),
    ]),
]

# No count and no table number here, on purpose. Text baked into a rendered image cannot be
# grepped and does not recompile, so anything that can change underneath it silently falsifies
# the figure. That already happened twice: the test count went stale when the outcome table grew
# (RR16), and the table number went stale when the supplement was renumbered (RR19). Both live in
# the tex caption, which is one grep from the generator that produces them. The table is named,
# not numbered, because there is only one outcome table and its own caption identifies it.
RULE = ("Headline panel, not a census: one measure per construct, from data independent of the atlas.\n"
        "Individual receptor maps in Fig. S4, disease maps in Fig. S7, the complete outcome table in "
        "the supplement.")


def main(argv=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tbl = pd.read_csv(TABLE).set_index("key")

    rowsrc = []
    for band, members in BANDS:
        got = []
        for name, key in members:
            r = tbl.loc[key]
            got.append({"band": band, "measure": name, "key": key,
                        "rho": float(r.effect_value), "q": float(r.q),
                        "unit": r.unit, "family": r.family,
                        "survives": bool(float(r.q) < 0.05),
                        "source_file": r.source_file})
        got.sort(key=lambda d: d["rho"], reverse=True)
        rowsrc.extend(got)
    df = pd.DataFrame(rowsrc)
    df.to_csv(OUTDIR / "fig6_source_values.csv", index=False)

    # RR34 Part D. This figure has read the outcome table since RR3, which is why it never
    # drifted, but nothing checked that per run. Logging each value it is about to draw puts
    # it under the same guard as Figures 2, 4, S2, S6, S9, S10 and S11.
    import rr32_outcome_stats as rs
    rs.reset_ledger()
    for r in df.itertuples():
        rs.render(r.key, "rho", r.rho)
        rs.render(r.key, "q", r.q)
    rs.verify_renders("figure_5_structural_model_gradients")

    # ---- layout: one row per measure, a header row above each band ---- #
    ypos, band_extent = [], []
    y = 0.9                      # room for the rises/falls annotations at the top
    for band, _members in BANDS:
        sub = df[df.band == band]
        y += 0.95                # the band's header row
        start = y - 0.5
        for _ in range(len(sub)):
            ypos.append(y)
            y += 1.0
        band_extent.append((band, start, y - 0.5, start - 0.42))
        y += 0.35
    df["y"] = ypos
    top = y - 0.35

    cmap = plt.cm.RdBu_r
    norm = mc.Normalize(-0.7, 0.7)
    fig = plt.figure(figsize=(190 / 25.4, 150 / 25.4))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.30, 0.175, 0.675, 0.805])
    ax.set_ylim(top + 0.1, -0.35)
    ax.set_xlim(-0.80, 0.92)

    for i, (band, lo, hi, ylab) in enumerate(band_extent):
        ax.add_patch(Rectangle((-0.80, lo), 1.72, hi - lo,
                               facecolor="0.93" if i % 2 == 0 else "0.965",
                               edgecolor="none", zorder=0))
        ax.text(-0.78, ylab, band, fontsize=7.8, va="center", ha="left", color="0.3",
                style="italic", zorder=4)
        ax.plot([-0.80, 0.92], [lo, lo], color="0.72", lw=0.7, zorder=1)

    ax.axvline(0, color="0.6", lw=1.0, zorder=1)
    for _, r in df.iterrows():
        c = cmap(norm(r.rho))
        ax.plot([0, r.rho], [r.y, r.y], color=c, lw=3.2, solid_capstyle="round", zorder=2)
        ax.plot([r.rho], [r.y], marker="o", ms=11, mfc=c if r.survives else "white",
                mec=c, mew=2.0, zorder=3)
        ha = "left" if r.rho > 0 else "right"
        off = 0.03 if r.rho > 0 else -0.03
        ax.text(r.rho + off, r.y, f"{r.rho:+.2f}{'*' if r.survives else ''}", va="center",
                ha=ha, fontsize=8, color="0.2")

    ax.set_yticks(df.y.to_numpy())
    ax.set_yticklabels(df.measure.tolist(), fontsize=8.4)
    ax.set_xlabel("Spearman $\\rho$ with cyto7 type (allocortex $\\to$ koniocortex)", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # "increases"/"decreases", not "rises"/"falls": the co-author asked for those verbs
    # throughout, and these two are baked into the image where no .tex grep can find them.
    ax.text(0.44, 0.05, "increases with type", fontsize=8, style="italic", color="#b2182b",
            ha="center", va="center")
    ax.text(-0.44, 0.05, "decreases with type", fontsize=8, style="italic", color="#2166ac",
            ha="center", va="center")
    leg = [Line2D([0], [0], marker="o", color="0.35", lw=0, mfc="0.35", mec="0.35", ms=9,
                  label="survives spin + FDR ($q<0.05$)"),
           Line2D([0], [0], marker="o", color="0.35", lw=0, mfc="white", mec="0.35", mew=2,
                  ms=9, label="directional (n.s.)")]
    # the legend sits in the empty left half of the two positive subject-level rows
    y_anchor = float(df[df.rho > 0].y.max()) - 1.45
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(-0.785, y_anchor),
              bbox_transform=ax.transData, fontsize=7.2, frameon=False,
              handletextpad=0.4, borderpad=0.3, labelspacing=0.7)
    fig.text(0.30, 0.018, RULE, fontsize=6.6, color="0.35", ha="left", va="bottom",
             linespacing=1.6)

    out = OUTDIR / FIGNAME
    fig.savefig(str(out), dpi=DPI, facecolor="white")
    plt.close(fig)
    # Write the shipped copy in the same pass. Staging it by hand is how the manuscript copy
    # and the generator's copy drift apart; the same failure produced the stale table in RR16.
    shutil.copy2(out, MANUSCRIPT_FIGS / FIGNAME)
    print(f"staged   {MANUSCRIPT_FIGS / FIGNAME}")
    n_surv = int(df.survives.sum())
    print(f"saved {out} ({len(df)} measures, {n_surv} survivors)")
    for _, r in df.iterrows():
        print(f"  {r.band.splitlines()[0]:16s} {r.measure[:34]:34s} rho={r.rho:+.2f} "
              f"q={r.q:.3f} {'*' if r.survives else ''}")
    (OUTDIR / "survivor_count.txt").write_text(
        f"measures plotted: {len(df)}\nsurvivors (q<0.05): {n_surv}\n"
        f"non-survivors: {', '.join(df[~df.survives].measure)}\n", encoding="utf-8")
    return df


if __name__ == "__main__":
    main()
