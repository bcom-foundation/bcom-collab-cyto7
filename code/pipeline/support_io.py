"""Shared loader for the anatomy-only support on the 32k fs_LR mesh.

Used by the structure-function analyses (EXTEND task) to optionally mask/weight
low-support vertices. Kept in its own module (not in cyto7_surface_io) to
avoid a circular import: it depends on build_support_map, which itself
imports cyto7_surface_io.

ANATOMY-ONLY: the support here is the atlas+topology+geometry+prior combined
score; the T1w/T2w data overlay is never used (it would be circular against the
myelin/functional analyses).
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import os
from pathlib import Path

import numpy as np

from cyto7_surface_io import REPO_ROOT

_WB_DEFAULT = cfg.workbench_dir()
_DERIVED = cfg.atlas_dir("fsaverage")


def _annot_template(version: str) -> str:
    if version == "v1":
        return str(cfg.atlas_dir("provenance/as_painted")
                   / "pial.{hemi}.cyto7.annot")
    return str(_DERIVED / f"pial.{{hemi}}.cyto7.{version}.annot")


def anatomy_support_32k(version: str, workbench_bin: str | None = None) -> dict[str, np.ndarray]:
    """Per-version anatomy-only combined support resampled to 32k fs_LR.

    Builds the support map for *version* into a per-version cache dir (no
    figure), then resamples the 164k combined support -> 32k (linear).
    Returns ``{"L": arr, "R": arr}`` (32492 each).
    """
    import nibabel as nib
    import build_support_map as bcm
    from neuromaps import transforms

    cache = _DERIVED / f"cache_conf_{version}"
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "pial.lh.cyto7.confidence.shape.gii").exists():
        bcm.main(["--annot", _annot_template(version), "--no-figure",
                  "--out-derived", str(cache), "--out-fig", str(cache)])
    wb = workbench_bin or os.environ.get("WORKBENCH_BIN") or _WB_DEFAULT
    if Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
        os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")
    out = {}
    for H, hemi in (("L", "lh"), ("R", "rh")):
        g = nib.load(str(cache / f"pial.{hemi}.cyto7.confidence.shape.gii"))
        res = transforms.fsaverage_to_fslr(g, "32k", hemi=H, method="linear")
        out[H] = np.asarray(res[0].agg_data())
    return out
