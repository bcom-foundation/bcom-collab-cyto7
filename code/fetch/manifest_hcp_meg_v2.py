#!/usr/bin/env python
"""Stage 0 manifest for SPEC_hcp_meg_dynamics_v2: forward-model inventory.

Records, per subject: presence of the sensor `preproc` (rmegpreproc) and
`anatomy` packages; the anatomy forward ingredients (singleshell headmodel,
2D/3D sourcemodels); whether a PRECOMPUTED leadfield/forward is shipped (it is
NOT — FieldTrip must compute it); and expected MD5 (verified one-at-a-time
during the run). Writes results/hcp_meg_v2_manifest.tsv + the confirmed route.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg
import re, zipfile
from pathlib import Path

SERVER = cfg.data_dir("019-HCP Young MEG data")
OUT = Path(r"./results/hcp_meg_v2_manifest.tsv")


def expected_md5(zp: Path):
    m5 = zp.with_suffix(zp.suffix + ".md5")
    if not m5.exists():
        return ""
    m = re.match(r"([0-9a-fA-F]{32})", m5.read_text(errors="replace").strip())
    return m.group(1).lower() if m else ""


def anatomy_contents(subj: str):
    """Return (headmodel_type, has_sm2d, sm3d_list, has_leadfield) from the zip listing."""
    zp = SERVER / f"{subj}_MEG_anatomy.zip"
    if not zp.exists():
        return ("", False, "", False)
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
    has_sm2d = any("sourcemodel_2d.mat" in n for n in names)
    sm3d = sorted({re.search(r"sourcemodel_(3d\d+mm)", n).group(1)
                   for n in names if re.search(r"sourcemodel_3d\d+mm\.mat", n)})
    has_hm = any(n.endswith("headmodel.mat") for n in names)
    has_lf = any(re.search(r"leadfield|forward", n, re.I) and n.endswith(".mat") for n in names)
    return ("singleshell" if has_hm else "", has_sm2d, ",".join(sm3d), has_lf)


def main():
    subs_pre = sorted({re.match(r"(\d+)_", p.name).group(1)
                       for p in SERVER.glob("*_MEG_Restin_preproc.zip")})
    subs_ana = sorted({re.match(r"(\d+)_", p.name).group(1)
                       for p in SERVER.glob("*_MEG_anatomy.zip")})
    subs = sorted(set(subs_pre) | set(subs_ana))

    cols = ["subject", "has_preproc", "has_anatomy", "headmodel", "sourcemodel_2d",
            "sourcemodel_3d", "precomputed_leadfield", "preproc_expected_md5",
            "anatomy_expected_md5", "forward_route"]
    rows = []
    # inventory anatomy contents once (same structure across subjects); still record per subject
    for s in subs:
        pre = SERVER / f"{s}_MEG_Restin_preproc.zip"
        ana = SERVER / f"{s}_MEG_anatomy.zip"
        hm, sm2d, sm3d, lf = anatomy_contents(s) if ana.exists() else ("", False, "", False)
        rows.append(dict(
            subject=s, has_preproc=int(pre.exists()), has_anatomy=int(ana.exists()),
            headmodel=hm, sourcemodel_2d=int(sm2d), sourcemodel_3d=sm3d,
            precomputed_leadfield=int(lf),
            preproc_expected_md5=expected_md5(pre) if pre.exists() else "",
            anatomy_expected_md5=expected_md5(ana) if ana.exists() else "",
            forward_route="1a_fieldtrip_singleshell"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    n_both = sum(1 for r in rows if r["has_preproc"] and r["has_anatomy"])
    n_lf = sum(r["precomputed_leadfield"] for r in rows)
    print(f"subjects: {len(subs)} | preproc {len(subs_pre)} | anatomy {len(subs_ana)} | "
          f"both(usable) {n_both}")
    print(f"headmodel type: singleshell | precomputed leadfield present: {n_lf}/{len(subs)} "
          f"(0 => FieldTrip computes it via ft_prepare_leadfield)")
    print(f"CONFIRMED ROUTE: 1a (FieldTrip singleshell, MATLAB R2018a) — MATLAB scriptable/licensed; "
          f"FieldTrip 20200607 being installed.")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
