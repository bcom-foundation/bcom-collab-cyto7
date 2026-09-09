"""Rule-based periallocortex reclassification -> candidate v3 (Layer 1D).

Implements ``docs/REFINE_allocortex.md`` section 1D. Applies a CHOSEN rule (the
default reviewer proposal, CLI-overridable) to the periallocortex belt produced
by Layer 1C, yielding a **candidate v3** annot. Reversible (delete v3 to revert);
v1/v2 are never touched. Off unless ``--apply-rule`` is given.

Default rule (Barbas convention):
  * Allocortex (code 1) shrinks to its true core (piriform/periamygdaloid +
    on-surface archicortex); everything else currently code 1 is regraded.
  * von Economo HA, HB, HC -> Agranular.
  * EC (entorhinal) -> Agranular  (``--ec-grade {agranular,dysgranular}``).
  * Perirhinal BA35 -> Agranular, BA36 -> Dysgranular (FS ex-vivo split);
    PHA1 -> Dysgranular; Pir -> Agranular.
  * PHA2/PHA3 -> left Eulaminate I by default (``--pha-posterior``); flagged.
  * PreS -> left Allocortex by default (``--pres-grade``); flagged.

Tensions resolved by AUTHORITY, not optimisation:
  * von Economo TF stays Eulaminate I (GC2020) — it is not in the belt set, so
    it is never touched.
  * No support-maximisation: support is recomputed/reported afterwards,
    never used as the objective.

Topology: after relabelling, iterate (<=10 passes) inserting an Agranular buffer
wherever Allocortex would abut Dysgranular (R1); abort + flag if convergence
needs a semantic choice.

Outputs: ``resources/cyto7_derived/pial.{lh,rh}.cyto7.v3.annot``,
``changelog_v2_to_v3.csv``, and v1-vs-v3 difference figures. The downstream
audit / support / presentation re-runs are separate CLI calls (see the
commands printed at the end / the REPORT).

Run::
    conda activate cyto7
    python scripts/apply_rule_v3.py --apply-rule
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
from matplotlib.patches import Patch

from cyto7_surface_io import REPO_ROOT
from audit_topology import adjacency, audit_hemi, ordinal, unique_edges
from allocortex_proposals import build_consensus
from build_support_map import DERIVED_DIR, REFINE_DIR
from make_presentation_figures import (
    HEMIS, LIGHT_BLUE, TYPE_GREY, TYPE_NAMES, lighting_normals, lit_panel,
    load_labels, load_surface, resolve_annot_paths, version_label,
)

V2_TEMPLATE = str(DERIVED_DIR / "pial.{hemi}.cyto7.v2.annot")
CODE = {"Allocortex": 1, "Agranular": 2, "Dysgranular": 3, "Eulaminate I": 4}
CODE_NAME = {0: "medial", 1: "Allocortex", 2: "Agranular", 3: "Dysgranular",
             4: "Eulaminate I", 5: "Eulaminate II", 6: "Eulaminate III", 7: "Koniocortex"}
REGRADE_ELIGIBLE = {1, 2, 3}      # periallo zone only; never regrade eulaminate or medial wall
RULE_ID = "scoped-paleocortex-v2 (June-2026 decision)"

ANT_CINGULATE = ("rostralanteriorcingulate", "caudalanteriorcingulate")
POST_CINGULATE = ("posteriorcingulate", "isthmuscingulate")

#: Colours for the difference figure, keyed by rule clause.
CLAUSE_COLOR = {
    "allocortex_scope->Agranular": (0.60, 0.60, 0.60),
    "cingulate_anterior->Agranular": (0.99, 0.75, 0.20),
    "cingulate_posterior->Dysgranular": (0.20, 0.30, 0.70),
    "HABC->Agranular": (0.84, 0.19, 0.15),
    "EC->Agranular": (0.99, 0.50, 0.0),
    "Perirhinal->Agranular": (0.42, 0.24, 0.60),
    "PHA1->Dysgranular": (0.12, 0.47, 0.71),
    "Pir->Agranular": (0.55, 0.63, 0.80),
    "topology_buffer": (0.0, 0.0, 0.0),
}


def desikan_mask(hemi, names_wanted):
    labels, _c, names = nib.freesurfer.io.read_annot(
        str(cfg.data_dir() / "voneconomo" / f"{hemi}.aparc.annot"))
    names = [x.decode() if isinstance(x, bytes) else x for x in names]
    idx = [i for i, nm in enumerate(names) if nm in names_wanted]
    return np.isin(np.clip(labels, 0, len(names) - 1), idx)


def converge_topology(lab, faces, allo_core_locked, max_pass=40):
    """Restore R1 by growing a GRADED agranular/eulaminate buffer (one ordinal
    step per pass) wherever |dordinal| >= 2.

    Per skip edge, step the endpoint that is free to move one ordinal toward the
    other, preferring to PROTECT eulaminate (raise the lower, periallo side)
    unless the lower side is a *locked* allocortex vertex (piriform/PreS) — in
    which case the higher (eulaminate) side is stepped DOWN, inserting the graded
    periallo->isocortex buffer the piriform core requires. Locked allocortex is
    never relabelled. Iterating reduces every gap to <=1 (a true R1 fixpoint);
    only genuinely impossible cases (both endpoints locked) are returned as
    unresolved.

    Returns (new_lab, buffer_changed_mask, unresolved_skip_edges)."""
    edges = unique_edges(faces)
    buffer_changed = np.zeros(lab.shape[0], bool)
    for _ in range(max_pass):
        ordv = ordinal(lab)
        both = (lab[edges[:, 0]] >= 1) & (lab[edges[:, 1]] >= 1)
        de = np.abs(ordv[edges[:, 0]] - ordv[edges[:, 1]])
        skip = both & (de >= 2)
        if not skip.any():
            return lab, buffer_changed, np.empty((0, 2), int)
        se = edges[skip]
        hi_ord = np.maximum(ordv[se[:, 0]], ordv[se[:, 1]])
        hi_vert = np.where(ordv[se[:, 0]] >= ordv[se[:, 1]], se[:, 0], se[:, 1])
        # Step every skip's higher endpoint DOWN one ordinal toward the lower
        # (the graded periallo/eulaminate buffer). The higher endpoint is never
        # the locked allocortex (that is the minimum), so progress is guaranteed
        # and gaps shrink monotonically until R1 holds.
        sh = ~allo_core_locked[hi_vert]
        if not sh.any():
            return lab, buffer_changed, se   # genuinely impossible (both locked)
        down_ord = np.full(lab.shape[0], np.inf)
        np.minimum.at(down_ord, hi_vert[sh], hi_ord[sh] - 1)
        dm = np.isfinite(down_ord)
        lab[dm] = (down_ord[dm] + 1).astype(lab.dtype)
        buffer_changed[dm] = True
    ordv = ordinal(lab)
    both = (lab[edges[:, 0]] >= 1) & (lab[edges[:, 1]] >= 1)
    se = edges[both & (np.abs(ordv[edges[:, 0]] - ordv[edges[:, 1]]) >= 2)]
    return lab, buffer_changed, se


def apply_rule_hemi(hemi, v2, faces, A, con, args):
    """Scoped-paleocortex rule (June-2026). Return (v3_labels, rows, info).

    Allocortex is retained ONLY for genuine piriform/periamygdaloid paleocortex
    (current allocortex within the Glasser-Pir zone) + the flagged PreS (if
    kept); EVERYTHING else currently code 1 is regraded. Archicortex is
    off-surface (medial wall) and never labelled here.
    """
    n = v2.shape[0]
    areas = con["areas"]
    eligible = np.isin(v2, list(REGRADE_ELIGIBLE))
    clause_of = np.empty(n, object)

    def pic(name):
        return areas.get(name, np.zeros(n, bool))

    # Genuine paleocortex core to KEEP as Allocortex: current allo within the
    # piriform/periamygdaloid (Glasser Pir, +1-hop periamygdaloid) zone.
    # CORRECTED allocortex clause (June-2026): the genuine on-surface paleocortex
    # = the Glasser piriform (Pir) parcel itself. LABEL those vertices Allocortex
    # (do NOT regrade Pir), regardless of their current cyto7 label.
    core_keep = pic("Pir")
    pres = pic("PreS")
    pres_keep = (v2 == 1) & pres if args.pres_grade == "allocortex" else np.zeros(n, bool)
    cing_ant = desikan_mask(hemi, ANT_CINGULATE)
    cing_post = desikan_mask(hemi, POST_CINGULATE)
    cing_keep = (v2 == 1) & (cing_ant | cing_post) if args.cingulate_allo == "keep" \
        else np.zeros(n, bool)

    v3 = v2.copy()

    # (1) Scope: every current Allocortex vertex -> Agranular by default.
    allo = v2 == 1
    v3[allo] = 2
    clause_of[allo] = "allocortex_scope->Agranular"
    # (2) Cingulate split (overrides default on the regraded cingulate allo).
    sel = allo & cing_ant
    clause_of[sel] = "cingulate_anterior->Agranular"           # stays Agranular
    sel = allo & cing_post
    v3[sel] = 3; clause_of[sel] = "cingulate_posterior->Dysgranular"
    # (3) Named periallo-area targets (mask & eligible). Later wins on overlap.
    ec_code = 2 if args.ec_grade == "agranular" else 3
    for name, code, clause in (
        ("vE_HA", 2, "HABC->Agranular"), ("vE_HB", 2, "HABC->Agranular"),
        ("vE_HC", 2, "HABC->Agranular"),
        ("PHA1", 3, "PHA1->Dysgranular"),
        ("Perirhinal/BA35-36", 2, "Perirhinal->Agranular"),
        ("EC", ec_code, "EC->Agranular"),
    ):
        sel = pic(name) & eligible & (v3 != code)
        v3[sel] = code
        clause_of[sel] = clause
    if args.pha_posterior == "dysgranular":
        for name in ("PHA2", "PHA3"):
            sel = pic(name) & eligible & (v3 != 3)
            v3[sel] = 3; clause_of[sel] = "PHA2/3->Dysgranular"
    # (4) Label/keep genuine allocortex: the piriform paleocortex core (+ any
    # flagged retentions). These win over every regrade above.
    for keep, cl in ((core_keep, "Pir->Allocortex(paleocortex)"),
                     (pres_keep, "PreS->Allocortex(kept)"),
                     (cing_keep, "cingulate->Allocortex(kept)")):
        v3[keep] = 1
        clause_of[keep] = cl

    # (5) Topology convergence (locked allocortex never relabelled).
    allo_locked = v3 == 1
    v3, buffer_changed, unresolved = converge_topology(v3, faces, allo_locked)
    clause_of[buffer_changed] = "topology_buffer"

    rows = []
    for v in np.where(v3 != v2)[0]:
        rows.append((hemi, int(v), int(v2[v]), int(v3[v]),
                     clause_of[v] if clause_of[v] else "rule"))
    info = {
        "changed": int((v3 != v2).sum()),
        "allo_before": int((v2 == 1).sum()), "allo_after": int((v3 == 1).sum()),
        "core_keep": int(core_keep.sum()), "pres_keep": int(pres_keep.sum()),
        "buffer": int(buffer_changed.sum()),
        "unresolved_skip_edges": int(unresolved.shape[0]),
    }
    return v3, rows, info


# --------------------------------------------------------------------------- #
# v1-vs-v3 difference figure
# --------------------------------------------------------------------------- #


def render_diff(v1_paths, v3_paths, changelog_by_hemi, surface, out_path, dpi):
    geom = {h: load_surface(h, surface) for h in HEMIS}
    vn = {h: lighting_normals(h) for h in HEMIS}
    grey = {c: np.array(TYPE_GREY[c]) for c in range(1, 8)}
    cols = [(h, v) for h in HEMIS for v in ("medial", "ventral")]
    rows = ["v1 (greyscale)", "v3 (greyscale)", "changed (by rule clause)"]
    fig, axes = plt.subplots(3, len(cols), figsize=(len(cols) * 3.0, 3 * 2.9),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    v1 = {h: load_labels(v1_paths[h], h) for h in HEMIS}
    v3 = {h: load_labels(v3_paths[h], h) for h in HEMIS}
    for ci, (hemi, view) in enumerate(cols):
        coords, faces = geom[hemi]
        # row 0: v1 greyscale
        r0 = np.tile(np.array(LIGHT_BLUE), (coords.shape[0], 1))
        for c in range(1, 8):
            r0[v1[hemi] == c] = grey[c]
        lit_panel(axes[0, ci], coords, faces, r0, hemi, view, vn[hemi])
        axes[0, ci].set_title(f"{hemi.upper()} {view}", fontsize=10)
        # row 1: v3 greyscale
        r1 = np.tile(np.array(LIGHT_BLUE), (coords.shape[0], 1))
        for c in range(1, 8):
            r1[v3[hemi] == c] = grey[c]
        lit_panel(axes[1, ci], coords, faces, r1, hemi, view, vn[hemi])
        # row 2: changed by clause
        r2 = np.tile(np.array([0.86, 0.86, 0.86]), (coords.shape[0], 1))
        r2[v1[hemi] == 0] = LIGHT_BLUE
        for v, clause in changelog_by_hemi[hemi].items():
            r2[v] = CLAUSE_COLOR.get(clause, (0.5, 0.5, 0.5))
        lit_panel(axes[2, ci], coords, faces, r2, hemi, view, vn[hemi])
    for r, lbl in enumerate(rows):
        axes[r, 0].text2D(-0.12, 0.5, lbl, transform=axes[r, 0].transAxes, rotation=90,
                          va="center", ha="center", fontsize=10, weight="bold")
    handles = [Patch(facecolor=c, label=k) for k, c in CLAUSE_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"cyto7 v1 -> candidate v3 ({RULE_ID}) on {surface} — periallocortical belt",
                 fontsize=13, y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out_path}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annot", default=V2_TEMPLATE, help="Source annot (default: v2).")
    p.add_argument("--apply-rule", action="store_true", help="Write the candidate v3 (else preview-report only).")
    p.add_argument("--ec-grade", choices=["agranular", "dysgranular"], default="agranular")
    p.add_argument("--pha-posterior", choices=["eulaminate1", "dysgranular"], default="eulaminate1")
    p.add_argument("--pres-grade", choices=["allocortex", "agranular"], default="agranular",
                   help="Committed default: PreS -> Agranular (was a flagged Allocortex exception).")
    p.add_argument("--cingulate-allo", choices=["regrade", "keep"], default="regrade",
                   help="regrade (default): anterior cingulate allo->Agranular, posterior->Dysgranular; "
                        "keep: leave cingulate allocortex as-is.")
    p.add_argument("--workbench-bin", default=None)
    p.add_argument("--out-derived", type=Path, default=DERIVED_DIR)
    p.add_argument("--out-fig", type=Path, default=REFINE_DIR)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args(argv)


def _audit_totals(v_paths):
    tot = {"R1": 0, "R2_open_rings": 0, "R3_bad": 0, "R4_annular": 0}
    for hemi in HEMIS:
        lab = load_labels(v_paths[hemi], hemi)
        coords, faces = load_surface(hemi, "pial")
        A = adjacency(faces, lab.shape[0])
        res, _ = audit_hemi(lab, faces, A, coords)
        tot["R1"] += res["R1_sequential_gradients"]["n_skip_edges"]
        tot["R2_open_rings"] += sum(not v["single_annular_ring"]
                                    for v in res["R2_core_continuity_rings"].values())
        tot["R3_bad"] += int(not res["R3_matrix_eulaminate_II"]["ok_single_component"])
        tot["R4_annular"] += sum(v["n_annular_violations"] for v in res["R4_islands"].values())
    return tot


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    v2_paths = resolve_annot_paths(args.annot)
    print(f"Rule '{RULE_ID}' (cingulate={args.cingulate_allo}, ec={args.ec_grade}, "
          f"pha_posterior={args.pha_posterior}, pres={args.pres_grade}); apply={args.apply_rule}")

    v3_paths, changelog_by_hemi, all_rows, infos = {}, {}, [], {}
    for hemi in HEMIS:
        v2 = load_labels(v2_paths[hemi], hemi)
        coords, faces = load_surface(hemi, "pial")
        A = adjacency(faces, v2.shape[0])
        con = build_consensus(hemi, v2.shape[0], args.workbench_bin)
        v3, rows, info = apply_rule_hemi(hemi, v2, faces, A, con, args)
        infos[hemi] = info
        changelog_by_hemi[hemi] = {r[1]: r[4] for r in rows}
        all_rows += rows
        print(f"  {hemi}: changed={info['changed']} "
              f"(allocortex {info['allo_before']}->{info['allo_after']}; "
              f"core_keep={info['core_keep']}, pres_keep={info['pres_keep']}, "
              f"buffer={info['buffer']}, unresolved_skips={info['unresolved_skip_edges']})")
        if info["unresolved_skip_edges"] > 0:
            print(f"  !! {hemi}: topology did NOT converge without a semantic choice "
                  f"({info['unresolved_skip_edges']} skip edges) — FLAGGED.")
        if args.apply_rule:
            out = args.out_derived / f"pial.{hemi}.cyto7.v3.annot"
            _, ctab, names = nib.freesurfer.io.read_annot(str(v2_paths[hemi]))
            nib.freesurfer.io.write_annot(str(out), v3.astype(np.int32), ctab, names)
            v3_paths[hemi] = out
            print(f"     wrote {out.name}")

    if args.apply_rule:
        clog = args.out_derived / "changelog_v2_to_v3.csv"
        with open(clog, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["hemi", "vertex", "old_label", "new_label", "rule_clause"])
            w.writerows(all_rows)
        print(f"  wrote {clog} ({len(all_rows)} rows)")

        # per-clause counts
        from collections import Counter
        cc = Counter(r[4] for r in all_rows)
        print("  vertices moved per rule clause:", dict(cc))

        # before/after audit
        print("  R1-R4 v2:", _audit_totals(v2_paths))
        print("  R1-R4 v3:", _audit_totals(v3_paths))

        if not args.no_figure:
            for surf in ("inflated", "pial"):
                render_diff(v2_paths, v3_paths, changelog_by_hemi, surf,
                            args.out_fig / f"v1_vs_v3_{surf}.png", args.dpi)
        print("\nNext (downstream re-runs):")
        print("  python scripts/audit_topology.py --annot 'resources/cyto7_derived/pial.{hemi}.cyto7.v3.annot'")
        print("  python scripts/build_support_map.py --annot 'resources/cyto7_derived/pial.{hemi}.cyto7.v3.annot'")
        print("  python scripts/refresh_figures.py --annot-version v3   (and --annot-version v1)")
    print("Done.")


if __name__ == "__main__":
    main()
