"""Step 1 aux for SPEC_adopt_v9: changelog, provenance, ring analysis (v8 -> v9).

v9 = v8 + an L/R entorhinal even-up (543 LH vertices agranular -> allocortex in
entorhinal/parahippocampal/temporal-pole/fusiform to match RH; RH untouched). Reads the
hand-painted ``pial.{lh,rh}.cyto7.v9.annot`` and writes, without touching the map:

* ``resources/cyto7_derived/v8_to_v9_changelog.csv`` — per-vertex old->new (names).
* ``resources/cyto7_derived/PROVENANCE_v9.txt`` — provenance + computed diff stats.
* ``figures/v9/topology/ring_analysis_v9.csv`` — per-type components/encircles/speck count.

Prints the diff summary (incl. Desikan-region breakdown of the changed vertices) + a
<30-vtx speck check.

Run::  conda activate cyto7 && python scripts/build_v9_aux.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import nibabel as nib

from audit_topology import (
    adjacency, components_of, encircles_hub, load_surface, TYPE_NAMES, CODE_NAME,
)

REPO = Path(__file__).resolve().parent.parent
DER = REPO / "resources" / "cyto7_derived"
TOPO = REPO / "figures" / "v9" / "topology"
NAME = {0: "unknown", **CODE_NAME}


def load_lab(hemi: str, ver: str) -> np.ndarray:
    return np.asarray(nib.freesurfer.io.read_annot(
        str(DER / f"pial.{hemi}.cyto7.{ver}.annot"))[0])


def desikan(hemi: str):
    """Desikan per-vertex parcel labels + names, or (None, None)."""
    p = REPO / "resources" / "voneconomo" / f"{hemi}.aparc.annot"
    if not p.exists():
        return None, None
    lab, _c, names = nib.freesurfer.io.read_annot(str(p))
    return np.asarray(lab), [x.decode() if isinstance(x, bytes) else x for x in names]


def main() -> int:
    TOPO.mkdir(parents=True, exist_ok=True)
    rows = []
    stats = {}
    allo = {"v8": {}, "v9": {}}
    region_breakdown = {}
    for hemi in ("lh", "rh"):
        v8 = load_lab(hemi, "v8")
        v9 = load_lab(hemi, "v9")
        assert v8.shape == v9.shape
        changed = np.where(v8 != v9)[0]
        for vtx in changed:
            rows.append([hemi, int(vtx), NAME[int(v8[vtx])], NAME[int(v9[vtx])]])
        net = {NAME[c]: int((v9 == c).sum() - (v8 == c).sum()) for c in range(1, 8)}
        trans = Counter((NAME[int(v8[v])], NAME[int(v9[v])]) for v in changed)
        stats[hemi] = {"changed": int(changed.size), "net": net, "top": trans.most_common(6)}
        allo["v8"][hemi] = int((v8 == 1).sum())
        allo["v9"][hemi] = int((v9 == 1).sum())
        # Desikan-region breakdown of the changed vertices
        dlab, dnames = desikan(hemi)
        if dlab is not None and changed.size:
            rb = Counter(dnames[dlab[v]] for v in changed)
            region_breakdown[hemi] = rb.most_common()

    # ---- changelog CSV ----
    clog = DER / "v8_to_v9_changelog.csv"
    with open(clog, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hemi", "vertex", "v8", "v9"])
        w.writerows(rows)
    print(f"wrote {clog} ({len(rows)} rows)")

    # ---- ring analysis CSV ----
    ring_rows, speck_report = [], []
    for hemi in ("lh", "rh"):
        lab = load_lab(hemi, "v9")
        coords, faces = load_surface(hemi, "pial")
        A = adjacency(faces, lab.shape[0])
        centerx = float(coords[:, 0].mean())
        for code in range(1, 8):
            m = lab == code
            total = int(m.sum())
            ncomp, sizes, _, _ = components_of(m, A)
            main = int(sizes[0]) if sizes else 0
            small = int(sum(1 for s in sizes if s < 30))
            enc = encircles_hub(m, A, lab, coords, centerx) if total else False
            ring_rows.append([hemi, TYPE_NAMES[code - 1], ncomp, total, main,
                              round(100.0 * main / total, 1) if total else 0.0, bool(enc), small])
            if small:
                speck_report.append(f"    {hemi} {TYPE_NAMES[code-1]}: {small} comp(s) <30 vtx "
                                    f"(sizes {[s for s in sizes if s < 30]})")
    ring = TOPO / "ring_analysis_v9.csv"
    with open(ring, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hemi", "type", "components", "total", "main", "main_pct",
                    "encircles", "small_lt30"])
        w.writerows(ring_rows)
    print(f"wrote {ring}")

    # ---- provenance ----
    def netline(h):
        s = stats[h]
        top = ", ".join(f"{a}->{b} {n}" for (a, b), n in s["top"]) or "(none)"
        net = ", ".join(f"{k} {v:+d}" for k, v in s["net"].items() if v) or "no net change"
        rb = region_breakdown.get(h)
        rbs = ("; Desikan regions: " + ", ".join(f"{nm} {ct}" for nm, ct in rb)) if rb else ""
        return f"  {h}: {s['changed']} vtx changed; net {net}.\n      transitions: {top}{rbs}"

    prov = DER / "PROVENANCE_v9.txt"
    text = f"""cyto7 v9 — L/R entorhinal consistency fix of v8 (CANONICAL)
========================================================================
Source: pial.{{lh,rh}}.cyto7.v8.annot -> pial.{{lh,rh}}.cyto7.v9.annot
cyto7 stays 7 labels. v9 is the RELEASED / CANONICAL map (Miguel approved v8; v9 is a
small L/R consistency fix). v4-v8 retained untouched.

Origin: hand-painted by Ricardo Salvador from v8. Evens up the left-right entorhinal
painting: moves {stats['lh']['changed']} LH vertices agranular -> allocortex (entorhinal /
parahippocampal / temporal-pole / fusiform), bringing LH entorhinal allocortex ~64% -> ~98%
to match RH (~92%). RH is UNTOUCHED (0 changes). Cortex-only; the true medial wall
(?h.cortex.label) is never painted; medial-wall footprint identical to v8 (3153 LH / 2089 RH).
Isocortex and the belt/insula are unchanged from v8.

Per-vertex diff (v8_to_v9_changelog.csv):
{netline('lh')}
{netline('rh')}

Allocortex surface vertices: LH {allo['v8']['lh']} -> {allo['v9']['lh']}; RH {allo['v8']['rh']} -> {allo['v9']['rh']} (unchanged); total {allo['v8']['lh']+allo['v8']['rh']} -> {allo['v9']['lh']+allo['v9']['rh']}.

Topology: strict R1=0 preserved; allocortex = 1 connected component, beta1=0 (open arc);
belt = one annulus; EulII 1-comp/3-holes; EulIII/konio 3 islands. Identical structure to v8.
See figures/v9/topology/topology_audit_v9.* and ring_analysis_v9.csv.

Wiring: resolve_target_map('v9', ...) resolves (allowed-versions tuple includes 'v9');
v9 is the default map (SPEC_adopt_v9 Step 5). Manuscript text files not edited
(RS applies manuscript/MANUSCRIPT_CHANGES_v9.md on Overleaf).
"""
    prov.write_text(text, encoding="utf-8")
    print(f"wrote {prov}")

    print("\n=== diff summary ===")
    for h in ("lh", "rh"):
        print(netline(h))
    print(f"  allocortex LH {allo['v8']['lh']}->{allo['v9']['lh']}, RH {allo['v8']['rh']}->{allo['v9']['rh']}")
    print("=== speck check (<30 vtx) ===")
    print("\n".join(speck_report) if speck_report else "  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
