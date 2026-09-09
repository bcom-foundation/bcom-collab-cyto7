"""Machine-specific configuration for the cyto7 pipeline.

Every path that used to be hard-coded to one workstation is read from an
environment variable here, so the code runs anywhere. Nothing in this file is
required to use the atlas itself: it matters only when re-running analyses that
need external tools or third-party data.

Variables, all optional until a script actually needs one:

  CYTO7_DATA_DIR        root for third-party data you have fetched yourself
                        (HCP, AHBA, ENIGMA, BigBrain). No default.
  CYTO7_WORKBENCH_DIR   directory containing wb_command. Needed only for
                        fsaverage to fs_LR resampling.
  CYTO7_FREESURFER_HOME FreeSurfer installation, for mri_surf2surf.
  CYTO7_FIELDTRIP_DIR   FieldTrip, for the MEG source reconstruction only.
  CYTO7_TOOLS_DIR       fallback parent for the two above.
  CYTO7_MATLAB_BIN      MATLAB executable, for the MEG source reconstruction
                        only. Optional; a documented default is used if unset.
  CYTO7_SCRATCH_DIR     working space for large intermediates. Optional;
                        defaults to a temp directory.

Each accessor raises a MissingConfig naming the variable and what it is for,
rather than failing later with a stack trace about a path that does not exist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The pipeline, figure and fetch scripts import each other by bare module name.
# Running "python code/pipeline/foo.py" puts only code/pipeline on sys.path, so the
# remaining code directories are added here, once, by the module they all import.
_CODE = Path(__file__).resolve().parent
for _d in (_CODE, _CODE / "pipeline", _CODE / "figures", _CODE / "fetch"):
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))


class MissingConfig(RuntimeError):
    """A required path is not configured. The message names the variable."""


class _Unconfigured:
    """Stands in for a path whose environment variable is not set.

    Scripts build module-level constants such as
    ``NEUROMAPS_CACHE = data_dir() / "neuromaps_cache"`` at import time. Raising
    there would stop a script that never touches that constant, so the failure is
    deferred: joining and printing work, and the MissingConfig is raised only when
    something actually tries to open the path.
    """

    def __init__(self, var, what, parts=()):
        self._var, self._what, self._parts = var, what, tuple(parts)

    def __truediv__(self, other):
        return _Unconfigured(self._var, self._what, self._parts + (str(other),))

    def _fail(self):
        raise MissingConfig(chr(10).join([
            f"{self._var} is not set, and something tried to use it.",
            f"  It must point to {self._what}.",
            "  Path requested: " + "/".join(("$" + self._var, *self._parts)),
            "  Set it in your shell, or copy .env.example to .env and edit it.",
            "  See docs/REPRODUCING.md for what each dataset needs.",
        ]))

    # readable in help text and log lines, so argparse defaults still render
    def __str__(self):
        return "/".join(("$" + self._var, *self._parts))

    __repr__ = __str__

    # anything that actually touches the filesystem fails, by name
    def __fspath__(self):
        self._fail()

    def exists(self):
        return False

    def is_dir(self):
        return False

    def is_file(self):
        return False

    @property
    def name(self):
        return self._parts[-1] if self._parts else ""

    def __getattr__(self, item):
        self._fail()


def _get(var: str, what: str, required: bool = True) -> Path | None:
    v = os.environ.get(var)
    if not v:
        if not required:
            return None
        raise MissingConfig(chr(10).join([
            f"{var} is not set.",
            f"  It must point to {what}.",
            "  Set it in your shell, or copy .env.example to .env and edit it.",
            "  See docs/REPRODUCING.md for what each dataset needs.",
        ]))
    p = Path(v)
    if not p.exists():
        raise MissingConfig(f"{var} is set to {p}, which does not exist.")
    return p


_DATA_WHAT = ("the root of third-party data you have fetched yourself "
              "(HCP, AHBA, ENIGMA, BigBrain)")


def data_dir(sub: str = ""):
    """Root of your own copy of third-party data. Never redistributed here.

    Returns an _Unconfigured stand-in when CYTO7_DATA_DIR is unset, so that
    importing a script that merely mentions a third-party path still works.
    """
    v = os.environ.get("CYTO7_DATA_DIR")
    if not v:
        u = _Unconfigured("CYTO7_DATA_DIR", _DATA_WHAT)
        return u / sub if sub else u
    root = Path(v)
    return root / sub if sub else root


def _lazy(var: str, what: str):
    """Configured Path if the variable is set, otherwise the lazy stand-in.

    Same reasoning as data_dir: these are commonly assigned to module-level
    constants, so the failure belongs at first use rather than at import.
    """
    v = os.environ.get(var)
    return Path(v) if v else _Unconfigured(var, what)


def workbench_dir():
    return _lazy("CYTO7_WORKBENCH_DIR",
                 "the Connectome Workbench bin directory containing wb_command")


def freesurfer_home():
    return _lazy("CYTO7_FREESURFER_HOME", "your FreeSurfer installation")


def fieldtrip_dir():
    return _lazy("CYTO7_FIELDTRIP_DIR",
                 "your FieldTrip installation (MEG source reconstruction only)")


def tools_dir():
    return _lazy("CYTO7_TOOLS_DIR", "a parent directory holding external tools")


def scratch_dir(sub: str = "") -> Path:
    """Working space for large intermediates. Defaults to a temp directory.

    Unlike the other accessors this never raises: a scratch location always has a
    reasonable default, and the caller only needs it to be writable.
    """
    import tempfile
    root = os.environ.get("CYTO7_SCRATCH_DIR")
    p = Path(root) if root else Path(tempfile.gettempdir()) / "cyto7_scratch"
    p = p / sub if sub else p
    p.mkdir(parents=True, exist_ok=True)
    return p


def optional(var: str) -> Path | None:
    return _get(var, "", required=False)


# --------------------------------------------------------------------------- #
# Release layout
# --------------------------------------------------------------------------- #
# The working repository kept everything under resources/ and figures/. The public
# release is organised by what a file IS (atlas, results, figures), so the handful
# of path constants that pointed into the old tree resolve through here instead.

ATLAS = REPO_ROOT / "atlas"
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

#: where a 164k annot may live, in search order
_ANNOT_DIRS = [ATLAS / "fsaverage",
               ATLAS / "fsaverage" / "support",
               ATLAS / "provenance" / "versions",
               ATLAS / "provenance" / "as_painted"]


def atlas_dir(sub: str = "") -> Path:
    return ATLAS / sub if sub else ATLAS


def figures_dir(sub: str = "") -> Path:
    return FIGURES / sub if sub else FIGURES


def results_dir(sub: str = "") -> Path:
    return RESULTS / sub if sub else RESULTS


def find_annot(filename: str) -> Path:
    """Locate a released annot by file name, searching the atlas tree.

    Raises with the name and the directories searched, so a missing superseded
    version is a sentence rather than a FileNotFoundError on a path nobody
    recognises.
    """
    for d in _ANNOT_DIRS:
        c = d / filename
        if c.exists():
            return c
    raise MissingConfig(chr(10).join([
        f"{filename} is not in this release.",
        "  Searched: " + ", ".join(str(d.relative_to(REPO_ROOT)) for d in _ANNOT_DIRS),
        "  The released map is atlas/fsaverage/pial.{lh,rh}.cyto7.v9.annot.",
    ]))
