#!/usr/bin/env python
"""Stage 0 audit + integrity manifest for the HCP Young-Adult MEG data.

Part of SPEC_hcp_meg_aperiodic.md. Inventories the HCP-MEG archives on the
server, records their expected MD5 (from the shipped .md5 sidecar files),
recomputes MD5 for the small packages (anatomy, parcel_yeo) + a dtseries
spot-check, and writes results/hcp_meg_manifest.tsv.

Why not verify every dtseries by default: each dtseries archive is ~20 GB
(89 subjects = ~1.8 TB). Stage-0 established these packages hold only
band-power *envelopes*, not the broadband time series the pre-registered
specparam test needs, so the dtseries data is not used in this stop-and-log.
Full re-hashing of 1.8 TB is deferred; pass --full to force it (e.g. before a
future beamform run once the sensor rmegpreproc package is obtained).

Usage:
    conda run -n cyto7 python scripts/manifest_hcp_meg.py
    conda run -n cyto7 python scripts/manifest_hcp_meg.py --full
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg
import argparse, hashlib, re, sys
from pathlib import Path

SERVER = cfg.data_dir("019-HCP Young MEG data")
OUT = Path(r"./results/hcp_meg_manifest.tsv")

# package suffix -> short name
PKG_SUFFIX = {
    "_MEG_Restin_dtseries.zip": "dtseries",
    "_MEG_Restin_parcel_yeo.zip": "parcel_yeo",
    "_MEG_anatomy.zip": "anatomy",
}
# dtseries subjects to spot-check by default (spread across the ID range)
SPOTCHECK = {"100307", "512835", "990366"}


def md5_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def expected_md5(md5_path: Path) -> str | None:
    if not md5_path.exists():
        return None
    txt = md5_path.read_text(encoding="utf-8", errors="replace").strip()
    m = re.match(r"([0-9a-fA-F]{32})", txt)
    return m.group(1).lower() if m else None


def subject_of(fname: str) -> str:
    m = re.match(r"(\d+)_", fname)
    return m.group(1) if m else "?"


def package_of(fname: str) -> str | None:
    for suffix, name in PKG_SUFFIX.items():
        if fname.endswith(suffix):
            return name
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="recompute MD5 for ALL archives incl. every dtseries (~1.8 TB read)")
    args = ap.parse_args()

    if not SERVER.exists():
        print(f"STOP: server path not reachable: {SERVER}", file=sys.stderr)
        return 2

    archives = sorted(p for p in SERVER.glob("*.zip"))
    if not archives:
        print(f"STOP: no .zip archives found under {SERVER}", file=sys.stderr)
        return 2

    rows = []
    n_pass = n_fail = n_deferred = n_nomd5 = 0
    for zp in archives:
        fname = zp.name
        pkg = package_of(fname)
        if pkg is None:
            continue
        subj = subject_of(fname)
        size = zp.stat().st_size
        exp = expected_md5(zp.with_suffix(zp.suffix + ".md5"))

        do_hash = args.full or pkg in ("anatomy", "parcel_yeo") or \
            (pkg == "dtseries" and subj in SPOTCHECK)

        if exp is None:
            status = "no_md5_file"
            actual = ""
            n_nomd5 += 1
        elif do_hash:
            actual = md5_of(zp)
            if actual == exp:
                status = "pass"; n_pass += 1
            else:
                status = "FAIL"; n_fail += 1
        else:
            actual = ""
            status = "deferred"; n_deferred += 1

        rows.append(dict(subject=subj, package=pkg, file=fname, size_bytes=size,
                         megconnectome_version="3.0", expected_md5=exp or "",
                         actual_md5=actual, status=status))
        print(f"{status:10s} {pkg:11s} {fname}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["subject", "package", "file", "size_bytes", "megconnectome_version",
            "expected_md5", "actual_md5", "status"]
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    # subject x package coverage summary
    subs = sorted({r["subject"] for r in rows})
    have = {(r["subject"], r["package"]) for r in rows}
    print("\n=== coverage (subjects x package) ===")
    print(f"total subjects with >=1 package: {len(subs)}")
    for pkg in ("dtseries", "parcel_yeo", "anatomy"):
        present = sorted(s for s in subs if (s, pkg) in have)
        print(f"  {pkg:11s}: {len(present)} subjects")
    print(f"\nchecksums: pass={n_pass} FAIL={n_fail} deferred={n_deferred} no_md5={n_nomd5}")
    print(f"manifest written -> {OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
