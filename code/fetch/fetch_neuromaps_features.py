"""Fetch and cache the neuromaps functional feature maps on the 32k fs_LR mesh.

This is a **one-time** acquisition step. It downloads the normative MEG and
fMRI feature maps via :mod:`neuromaps`, resamples them onto the 32k ``fs_LR``
mesh used throughout this repository, and caches the per-hemisphere arrays as
``.npy`` files under ``resources/neuromaps_cache/``. After running it once, the
analysis script (``summarise_functional_features.py``) runs entirely offline and
deterministically, and needs neither an internet connection nor Connectome
Workbench.

Feature maps cached
-------------------
* **hcps1200** (Shafiei et al., 2022) -- MEG group maps, native ``fsLR den-4k``,
  resampled to 32k:
  intrinsic timescale (``megtimescale``) and band power
  ``megdelta``/``megtheta``/``megalpha``/``megbeta``/``meggamma1``.
* **margulies2016** -- the principal functional connectivity gradient
  (``fcgradient01``), native ``fsLR den-32k`` (used as-is).

Why Connectome Workbench is needed *here only*
----------------------------------------------
The MEG maps are released at ``fsLR den-4k`` and must be resampled to 32k.
:mod:`neuromaps` performs surface resampling by calling ``wb_command``
(Connectome Workbench). This script therefore needs Workbench on the ``PATH``
*at fetch time*; it locates it via ``--workbench-bin`` or the ``WORKBENCH_BIN``
environment variable, falling back to a default install location. The resampled
result is cached, so no downstream script needs Workbench.

Usage
-----
    conda activate cyto7
    python scripts/fetch_neuromaps_features.py
    # or, if Workbench lives elsewhere:
    python scripts/fetch_neuromaps_features.py --workbench-bin /path/to/bin
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from cyto7_surface_io import REPO_ROOT

#: Default Connectome Workbench binary directory (Windows install used in this
#: project). Override with --workbench-bin or the WORKBENCH_BIN env var.
DEFAULT_WORKBENCH_BIN = (
    cfg.workbench_dir()
)

#: Cache directory for the resampled 32k feature arrays.
CACHE_DIR = cfg.data_dir() / "neuromaps_cache"

#: Features to fetch: (source, desc, native_density, human-readable name).
FEATURES: list[tuple[str, str, str, str]] = [
    ("hcps1200", "megtimescale", "4k", "Intrinsic timescale (tau)"),
    ("hcps1200", "megdelta", "4k", "Delta power (0.5-4 Hz)"),
    ("hcps1200", "megtheta", "4k", "Theta power (4-8 Hz)"),
    ("hcps1200", "megalpha", "4k", "Alpha power (8-12 Hz)"),
    ("hcps1200", "megbeta", "4k", "Beta power (12-30 Hz)"),
    ("hcps1200", "meggamma1", "4k", "Gamma power (30-60 Hz)"),
    ("margulies2016", "fcgradient01", "32k", "Principal functional gradient"),
]


def _ensure_workbench_on_path(workbench_bin: str | None) -> None:
    """Prepend the Workbench binary directory to ``PATH`` if it exists."""
    candidate = (
        workbench_bin
        or os.environ.get("WORKBENCH_BIN")
        or DEFAULT_WORKBENCH_BIN
    )
    if candidate and Path(candidate).exists():
        os.environ["PATH"] = candidate + os.pathsep + os.environ["PATH"]
        print(f"Using Connectome Workbench at: {candidate}")
    else:
        print(
            "WARNING: Connectome Workbench not found "
            f"(looked at '{candidate}'). Resampling of the 4k MEG maps will "
            "fail; pass --workbench-bin with the correct path."
        )


def _cache_name(source: str, desc: str, hemi: str) -> Path:
    """Return the cache file path for one feature/hemisphere."""
    return CACHE_DIR / f"{source}_{desc}_fsLR32k_hemi-{hemi}.npy"


def fetch_and_cache(features: Sequence[tuple[str, str, str, str]]) -> None:
    """Download, resample to 32k, and cache each requested feature map."""
    # Imported here so the module import does not require neuromaps unless
    # this acquisition step is actually run.
    from neuromaps import datasets, transforms

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for source, desc, native_den, pretty in features:
        print(f"\nFetching {source}/{desc} ({pretty})...")
        fetched = datasets.fetch_annotation(source=source, desc=desc)

        if native_den != "32k":
            print(f"  resampling fsLR den-{native_den} -> den-32k (wb_command)...")
            fetched = transforms.fslr_to_fslr(
                fetched, target_density="32k", method="linear"
            )

        # fetched is a (L, R) tuple of GIFTI images/paths.
        import nibabel as nib

        for hemi, gii in zip(("L", "R"), fetched):
            data = np.asarray(
                gii.agg_data() if hasattr(gii, "agg_data") else nib.load(gii).agg_data(),
                dtype=float,
            )
            if data.shape[0] != 32492:
                raise RuntimeError(
                    f"{source}/{desc} hemi {hemi}: expected 32492 vertices, "
                    f"got {data.shape[0]}"
                )
            out = _cache_name(source, desc, hemi)
            np.save(out, data)
            print(f"  cached {out.name}  (n={data.shape[0]}, "
                  f"finite={np.isfinite(data).sum()})")

        manifest.append(
            {
                "source": source,
                "desc": desc,
                "name": pretty,
                "native_density": native_den,
                "cached_density": "32k",
                "space": "fsLR",
            }
        )

    manifest_path = CACHE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}")
    print("Done. The functional summary can now run offline.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workbench-bin",
        default=None,
        help="Connectome Workbench binary directory (for 4k->32k resampling). "
        "Defaults to $WORKBENCH_BIN or the project default path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _ensure_workbench_on_path(args.workbench_bin)
    fetch_and_cache(FEATURES)


if __name__ == "__main__":
    main()
