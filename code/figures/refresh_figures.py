"""One-command refresh of the cyto7 presentation deck (Layer-1 figures).

Thin wrapper around ``make_presentation_figures.py``: loops Figure A + Figure B
over ``{pial, inflated}`` (and ``flat`` if asked) for a chosen annot version,
writing everything to ``figures/refine/presentation/``. With ``--all`` it also
re-runs the topology audit (Layer 1) and support map (Layer 1.5) so a single
command brings the whole derived set in sync with the current annot.

Examples::

    conda activate cyto7
    python scripts/refresh_figures.py                      # latest annot (v2 if present, else v1)
    python scripts/refresh_figures.py --annot-version v1   # the original hand-drawn map
    python scripts/refresh_figures.py --annot /path/to/pial.{hemi}.cyto7.v3.annot
    python scripts/refresh_figures.py --flat               # also emit 32k flat panels (when available)
    python scripts/refresh_figures.py --all                # also refresh audit + support

Cross-platform: pure Python, uses the ``cyto7`` env, runs on Windows and
WSL/Linux. A ``make refresh-figures`` target calls the same wrapper.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
from pathlib import Path
from typing import Sequence

from cyto7_surface_io import REPO_ROOT

import make_presentation_figures as mpf

V1_TEMPLATE = str(cfg.atlas_dir("provenance/as_painted")
                  / "pial.{hemi}.cyto7.annot")
DERIVED_DIR = cfg.atlas_dir("fsaverage")


def resolve_annot(version: str | None, explicit: str | None) -> str:
    """Resolve --annot / --annot-version to a {hemi}-templated annot path."""
    if explicit:
        return explicit
    if version in (None, "latest"):
        # canonical = highest available derived version (v6 is the released map).
        for ver in ("v9", "v8", "v7", "v6", "v5", "v4", "v3_clean", "v3", "v2"):
            if (DERIVED_DIR / f"pial.lh.cyto7.{ver}.annot").exists():
                return str(DERIVED_DIR / f"pial.{{hemi}}.cyto7.{ver}.annot")
        return V1_TEMPLATE
    if version == "v1":
        return V1_TEMPLATE
    # v2, v3, ...: look under the derived dir.
    cand = DERIVED_DIR / f"pial.lh.cyto7.{version}.annot"
    if not cand.exists():
        raise SystemExit(f"No annot for version '{version}' (expected {cand}).")
    return str(DERIVED_DIR / f"pial.{{hemi}}.cyto7.{version}.annot")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--annot-version", default="latest",
                   help="v1 | v2 | ... | latest (default). Resolves to a file under resources/.")
    g.add_argument("--annot", default=None, help="Explicit annot path or {hemi} template (overrides --annot-version).")
    p.add_argument("--flat", action="store_true", help="Also emit the 32k fs_LR flat panels (when available).")
    p.add_argument("--all", action="store_true",
                   help="Also re-run the topology audit (Layer 1) and support map (Layer 1.5).")
    p.add_argument("--no-ventral", action="store_true")
    p.add_argument("--reference-img", default=None)
    p.add_argument("--out", type=Path, default=mpf.DEFAULT_OUT)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    annot = resolve_annot(args.annot_version, args.annot)
    surfaces = ["pial", "inflated"] + (["flat"] if args.flat else [])
    print(f"Refreshing presentation figures for annot: {annot}")
    print(f"Surfaces: {surfaces}\n")

    for surface in surfaces:
        mpf_argv = ["--annot", annot, "--surface", surface, "--out", str(args.out),
                    "--dpi", str(args.dpi), "--which", "both"]
        if args.no_ventral:
            mpf_argv.append("--no-ventral")
        if args.reference_img:
            mpf_argv += ["--reference-img", args.reference_img]
        try:
            mpf.main(mpf_argv)
        except SystemExit as exc:
            # flat (not implemented on 164k) and similar -> warn, keep going.
            print(f"  [skip {surface}] {exc}")

    if args.all:
        _refresh_derived(annot)
    print("\nrefresh-figures: done.")


def _refresh_derived(annot: str) -> None:
    """Re-run audit (Layer 1) + support (Layer 1.5) if those scripts exist."""
    scripts_dir = REPO_ROOT / "scripts"
    for modname, label in (("audit_topology", "topology audit (Layer 1)"),
                           ("build_support_map", "support map (Layer 1.5)")):
        if (scripts_dir / f"{modname}.py").exists():
            print(f"\n[--all] running {label} ...")
            import importlib
            mod = importlib.import_module(modname)
            mod.main(["--annot", annot])
        else:
            print(f"[--all] {label}: scripts/{modname}.py not present yet — skipped.")


if __name__ == "__main__":
    main()
