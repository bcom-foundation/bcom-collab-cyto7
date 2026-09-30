"""Rebuild dist/cyto7-atlas-v9.zip, the standalone atlas bundle.

WHY THIS SCRIPT EXISTS
----------------------
The zip shipped alongside the first release was built by hand and then never rebuilt,
and it went stale in ways nothing checked. It carried a ``LICENSE`` whose copyright
line still read ``Copyright (c) 2026 TODO(AUTHORS)`` long after the repository's own
licences had been filled in, and READMEs predating later edits. A hand-built artifact
sitting next to a generated manifest is the classic two-paths failure: you fix one and
miss the other, and nothing compares them.

WHAT GOES IN
------------
The atlas only. This is the convenience download for people who want the maps without
the analysis; the full repository snapshot is what goes to Zenodo. Contents come from
``git ls-files``, never from the working tree, for the same reason ``make_manifest.sh``
does it -- an untracked scratch file must never end up inside a release.

WHY PYTHON AND NOT ``zip``
--------------------------
``zip`` is not present on a stock Windows box (here it resolves only through MiKTeX's
bin directory), and the stdlib gives us something the command-line tool does not: a
**deterministic** archive. Entries are written in sorted order with a fixed timestamp,
so rebuilding from the same commit produces the same bytes, and the manifest hash of
this file only changes when its contents really change.

ORDER MATTERS
-------------
Run this BEFORE ``make_manifest.sh``. The manifest hashes this zip, so regenerating the
manifest first leaves a manifest that fails ``sha256sum -c``.

    python make_dist.py && sh make_manifest.sh
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "cyto7-atlas-v9"
OUT = ROOT / "dist" / f"{NAME}.zip"

# Any fixed date works; it only has to be constant across rebuilds.
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def tracked(prefix: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", prefix], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.split("\0") if p)


def main() -> int:
    files = tracked("atlas")
    if not files:
        print("no tracked files under atlas/ -- wrong directory?", file=sys.stderr)
        return 1

    # arcname -> bytes. The atlas README stands in as the bundle's top-level README,
    # and the data licence is the applicable one for an atlas-only download.
    payload: dict[str, bytes] = {f: (ROOT / f).read_bytes() for f in files}
    payload["README.md"] = (ROOT / "atlas" / "README.md").read_bytes()
    payload["LICENSE"] = (ROOT / "LICENSE-DATA").read_bytes()

    # An internal manifest, so the bundle can be verified without the repository.
    lines = [f"{hashlib.sha256(payload[k]).hexdigest()}  {k}\n"
             for k in sorted(payload)]
    payload["SHA256SUMS.txt"] = "".join(lines).encode()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc in sorted(payload):
            info = zipfile.ZipInfo(f"{NAME}/{arc}", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, payload[arc])

    print(f"  dist/{NAME}.zip: {len(payload)} files, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")
    print("  now run: sh make_manifest.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
