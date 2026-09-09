"""Shared I/O helpers for the cyto7 / Glasser surface analyses.

This module centralises the file-path resolution and data loading used by the
figure-generating scripts in this repository, so that every script reads the
cyto7 labels, the HCP myelin map, the curvature map and the surface geometries
in exactly the same way.

All data live on the 32k ``fs_LR`` mesh (32,492 vertices per hemisphere):

* **cyto7 labels** -- per-vertex cytoarchitectural class (0 = unlabelled/medial
  wall, 1-7 = cyto7 classes), stored as GIFTI label files in ``resources/``.
* **Myelin (T1w/T2w) and curvature** -- stored as CIFTI dense-scalar
  (``.dscalar.nii``) files. CIFTI only stores the ~29.7k cortical grayordinates
  (no medial wall), so values are scattered back onto the full 32,492-vertex
  mesh, leaving medial-wall vertices at 0.0. A myelin value of exactly 0.0
  therefore means "no data" (medial wall), not "zero myelin".

Paths are resolved relative to the repository root, so the scripts run from a
clean checkout on either Windows or Linux/WSL without editing paths.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Repository root (this file lives in ``<root>/scripts/``).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: cyto7 class integer labels in increasing order of laminar differentiation.
LABEL_LEVELS: list[int] = [1, 2, 3, 4, 5, 6, 7]

#: Human-readable names for each cyto7 class (index 0 == label 1).
LABEL_NAMES: list[str] = [
    "Allocortex",
    "agranular",
    "dysgranular",
    "eulaminate I",
    "eulaminate II",
    "eulaminate III",
    "koniocortex",
]

#: The two HCP group-average datasets shipped under ``resources``.
DATASETS = ("Validation210", "Parcellation210")


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #


def glasser_fs_lr_dir(dataset: str) -> Path:
    """Return the ``fsaverage_LR32k`` directory for a Glasser HCP dataset.

    Parameters
    ----------
    dataset:
        ``"Validation210"`` or ``"Parcellation210"``. The cyto7 labels were
        resampled against the Validation210 mesh, so that is the default
        elsewhere.
    """
    name = f"Q1-Q6_Related{dataset}"
    return (
        REPO_ROOT
        / "resources"
        / "glasser_resources"
        / "Glasser_et_al_2016_HCP_MMP1.0_v6_RVVG"
        / "HCP_PhaseTwo"
        / name
        / "MNINonLinear"
        / "fsaverage_LR32k"
    )


def surface_path(dataset: str, hemi: str, geometry: str) -> Path:
    """Build the path to a 32k ``fs_LR`` surface GIFTI file.

    Parameters
    ----------
    dataset:
        ``"Validation210"`` or ``"Parcellation210"``.
    hemi:
        ``"L"`` or ``"R"``.
    geometry:
        ``"pial"``, ``"inflated"``, ``"flat"``, ``"midthickness"`` or
        ``"white"``.
    """
    prefix = f"Q1-Q6_Related{dataset}"
    if geometry == "flat":
        # The flat surface has no MSMAll/DeDrift qualifier in its filename.
        filename = f"{prefix}.{hemi}.flat.32k_fs_LR.surf.gii"
    else:
        suffix = "MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii"
        filename = f"{prefix}.{hemi}.{geometry}_{suffix}"
    return glasser_fs_lr_dir(dataset) / filename


def label_path(hemi: str) -> Path:
    """Path to the cyto7 label GIFTI for a hemisphere (``"lh"`` / ``"rh"``)."""
    return (
        REPO_ROOT
        / "resources"
        / "cyto7_annot_files_32k_fsaverage"
        / f"pial.{hemi}.cyto7.32k_fs_LR.label.gii"
    )


def myelin_dscalar_path(dataset: str) -> Path:
    """Path to the HCP myelin (T1w/T2w) CIFTI dscalar for a dataset."""
    prefix = f"Q1-Q6_Related{dataset}"
    return (
        glasser_fs_lr_dir(dataset)
        / f"{prefix}.MyelinMap_BC_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii"
    )


def curvature_dscalar_path(dataset: str) -> Path:
    """Path to the HCP curvature CIFTI dscalar for a dataset."""
    prefix = f"Q1-Q6_Related{dataset}"
    return (
        glasser_fs_lr_dir(dataset)
        / f"{prefix}.curvature_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii"
    )


def thickness_dscalar_path(dataset: str) -> Path:
    """Path to the HCP cortical-thickness CIFTI dscalar for a dataset."""
    prefix = f"Q1-Q6_Related{dataset}"
    return (
        glasser_fs_lr_dir(dataset)
        / f"{prefix}.thickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii"
    )


#: Cache directory for the resampled 32k neuromaps feature arrays (populated by
#: ``fetch_neuromaps_features.py``).
NEUROMAPS_CACHE: Path = cfg.data_dir() / "neuromaps_cache"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

#: CIFTI structure names for the two cortical hemispheres, keyed by "L"/"R".
_CORTEX_STRUCTURES = {
    "L": "CIFTI_STRUCTURE_CORTEX_LEFT",
    "R": "CIFTI_STRUCTURE_CORTEX_RIGHT",
}


def load_cyto7_labels(hemi: str) -> np.ndarray:
    """Load the cyto7 per-vertex class labels for one hemisphere.

    Parameters
    ----------
    hemi:
        ``"lh"`` or ``"rh"``.

    Returns
    -------
    numpy.ndarray
        A length-32492 integer array; 0 marks unlabelled vertices (medial
        wall) and 1-7 are the cyto7 classes.
    """
    gifti = nib.load(str(label_path(hemi)))
    return np.asarray(gifti.darrays[0].data)


def scatter_dscalar_to_surface(dscalar_path: Path) -> dict[str, np.ndarray]:
    """Read a CIFTI dscalar and scatter each hemisphere onto its full surface.

    CIFTI dense-scalar files store only the cortical grayordinates (the ~29.7k
    vertices that are not medial wall). This reads the first data row, then
    scatters each hemisphere's values back onto the full 32,492-vertex ``fs_LR``
    mesh, leaving medial-wall vertices at 0.0.

    Parameters
    ----------
    dscalar_path:
        Path to a ``.dscalar.nii`` CIFTI-2 file.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping ``{"L": <32492 array>, "R": <32492 array>}``.
    """
    img = nib.load(str(dscalar_path))
    data = np.asarray(img.get_fdata())[0]  # first map; shape (59412,)
    brain_model = img.header.get_axis(1)  # BrainModelAxis

    out: dict[str, np.ndarray] = {}
    for hemi, structure_name in _CORTEX_STRUCTURES.items():
        data_slice, model = next(
            (sl, mdl)
            for name, sl, mdl in brain_model.iter_structures()
            if name == structure_name
        )
        full = np.zeros(brain_model.nvertices[structure_name], dtype=float)
        full[model.vertex] = data[data_slice]
        out[hemi] = full
    return out


def load_myelin_on_surface(dataset: str) -> dict[str, np.ndarray]:
    """Load the HCP myelin (T1w/T2w) map scattered onto the full surfaces."""
    return scatter_dscalar_to_surface(myelin_dscalar_path(dataset))


def load_curvature_on_surface(dataset: str) -> dict[str, np.ndarray]:
    """Load the HCP curvature map scattered onto the full surfaces.

    Sign convention (HCP ``fs_LR``): negative on gyral crowns/walls, positive
    in sulcal fundi -- so ``curv <= 0`` selects gyral domes and walls.
    """
    return scatter_dscalar_to_surface(curvature_dscalar_path(dataset))


def load_thickness_on_surface(dataset: str) -> dict[str, np.ndarray]:
    """Load the HCP cortical-thickness map scattered onto the full surfaces."""
    return scatter_dscalar_to_surface(thickness_dscalar_path(dataset))


def load_cached_feature(source: str, desc: str) -> dict[str, np.ndarray]:
    """Load a cached neuromaps feature (32k ``fs_LR``) for both hemispheres.

    The arrays are produced by ``fetch_neuromaps_features.py`` and stored under
    :data:`NEUROMAPS_CACHE`.

    Parameters
    ----------
    source:
        neuromaps source, e.g. ``"hcps1200"`` or ``"margulies2016"``.
    desc:
        neuromaps description, e.g. ``"megtimescale"`` or ``"fcgradient01"``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping ``{"L": <32492 array>, "R": <32492 array>}``.

    Raises
    ------
    FileNotFoundError
        If the cache is missing (run ``fetch_neuromaps_features.py`` first).
    """
    out: dict[str, np.ndarray] = {}
    for hemi in ("L", "R"):
        path = NEUROMAPS_CACHE / f"{source}_{desc}_fsLR32k_hemi-{hemi}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached feature {path}. Run "
                "`python scripts/fetch_neuromaps_features.py` first."
            )
        out[hemi] = np.load(path)
    return out


def load_surface_geometry(
    dataset: str, hemi: str, geometry: str = "midthickness"
) -> tuple[np.ndarray, np.ndarray]:
    """Load vertex coordinates and triangle faces of a surface GIFTI.

    Parameters
    ----------
    dataset:
        ``"Validation210"`` or ``"Parcellation210"``.
    hemi:
        ``"L"`` or ``"R"``.
    geometry:
        Surface to load (default ``"midthickness"``, appropriate for geodesic
        distance computations).

    Returns
    -------
    (coords, faces):
        ``coords`` has shape (n_vertices, 3); ``faces`` has shape
        (n_faces, 3) of vertex indices.
    """
    gifti = nib.load(str(surface_path(dataset, hemi, geometry)))
    coords = np.asarray(gifti.darrays[0].data, dtype=float)
    faces = np.asarray(gifti.darrays[1].data, dtype=np.int64)
    return coords, faces


def medial_wall_buffer_mask(
    coords: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    buffer_mm: float = 3.0,
) -> np.ndarray:
    """Boolean mask of vertices farther than *buffer_mm* from the medial wall.

    The medial wall / unlabelled zone is taken to be every vertex with cyto7
    label 0. A graph is built over the mesh edges (weighted by Euclidean edge
    length), and the geodesic distance from the nearest label-0 vertex is
    computed with Dijkstra's algorithm. Vertices whose distance exceeds
    *buffer_mm* are kept (``True``). This excludes a rim of cortex adjacent to
    the medial wall, where surface metrics are least reliable.

    Parameters
    ----------
    coords:
        Vertex coordinates, shape (n_vertices, 3).
    faces:
        Triangle faces, shape (n_faces, 3).
    labels:
        Per-vertex cyto7 labels (0 = medial wall / unlabelled).
    buffer_mm:
        Buffer radius in millimetres.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length n_vertices (``True`` == keep).
    """
    n_vertices = coords.shape[0]
    # Undirected edge list from all three triangle sides.
    src = faces[:, [0, 1, 2]].ravel()
    dst = faces[:, [1, 2, 0]].ravel()
    edge_len = np.linalg.norm(coords[src] - coords[dst], axis=1)
    graph = csr_matrix((edge_len, (src, dst)), shape=(n_vertices, n_vertices))

    seed_indices = np.where(labels == 0)[0]
    if seed_indices.size == 0:
        # No medial wall on this hemisphere -> keep everything.
        return np.ones(n_vertices, dtype=bool)

    geodesic = dijkstra(
        csgraph=graph, directed=False, indices=seed_indices, min_only=True
    )
    return geodesic > buffer_mm


# --------------------------------------------------------------------------- #
# Target-map resolver (shared by the structure-function analyses; EXTEND task)
# --------------------------------------------------------------------------- #

#: 164k fsaverage annot templates per version (``{hemi}`` -> lh/rh).
_V1_164K = (cfg.atlas_dir("provenance/as_painted")
            / "pial.{hemi}.cyto7.annot")
_DERIVED_164K = "pial.{hemi}.cyto7.{ver}.annot"   # resolved by cfg.find_annot
_TARGET_CACHE = cfg.scratch_dir("resample_cache")
_WORKBENCH_DEFAULT = (
    cfg.workbench_dir()
)


def _target_paths(version_or_path: str) -> tuple[str, dict[str, Path]]:
    """Resolve a version/path to (tag, {"L": lh_path, "R": rh_path}) of 164k annots."""
    s = str(version_or_path)
    if "{hemi}" in s or "{h}" in s:
        tag = "custom"
        return tag, {"L": Path(s.format(hemi="lh", h="lh")),
                     "R": Path(s.format(hemi="rh", h="rh"))}
    if s in ("v1", "v2", "v3", "v3_clean", "v4", "v5", "v6", "v7", "v8", "v9"):
        tmpl = str(_DERIVED_164K).replace("{ver}", s)
        return s, {"L": cfg.find_annot(tmpl.format(hemi="lh")),
                   "R": cfg.find_annot(tmpl.format(hemi="rh"))}
    raise ValueError(f"Unrecognised target-map version/path: {version_or_path!r}")


def resolve_target_map(
    version_or_path: str, mesh: str = "fs_LR", workbench_bin: str | None = None
) -> dict[str, np.ndarray]:
    """Return per-vertex cyto7 labels for a target-map *version* on a *mesh*.

    Parameters
    ----------
    version_or_path:
        ``"v1"`` | ``"v3"`` | ... or a ``{hemi}``-templated path to a 164k annot.
    mesh:
        ``"fsaverage"`` / ``"164k"`` returns the native 163,842-vertex labels;
        ``"fs_LR"`` / ``"32k"`` resamples 164k -> 32k by **nearest-neighbour**
        (sphere registration via neuromaps/``wb_command``) and caches the result
        under ``resources/cyto7_derived/cache/``. Both versions go through the
        *identical* resampling so they are treated the same.

    Returns
    -------
    dict
        ``{"L": <labels>, "R": <labels>}`` (int per-vertex cyto7 codes 0-7).
    """
    import os

    tag, paths = _target_paths(version_or_path)
    labels164 = {
        H: np.asarray(nib.freesurfer.io.read_annot(str(paths[H]))[0])
        for H in ("L", "R")
    }
    if mesh in ("fsaverage", "164k"):
        return labels164
    if mesh not in ("fs_LR", "32k", "fslr"):
        raise ValueError(f"Unknown mesh {mesh!r}")

    _TARGET_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    need_resample = []
    for H in ("L", "R"):
        cpath = _TARGET_CACHE / f"{tag}_labels_fsLR32k_hemi-{H}.npy"
        if cpath.exists():
            out[H] = np.load(cpath)
        else:
            need_resample.append((H, cpath))
    if need_resample:
        wb = workbench_bin or os.environ.get("WORKBENCH_BIN") or _WORKBENCH_DEFAULT
        if wb and Path(wb).is_dir() and wb not in os.environ.get("PATH", ""):
            os.environ["PATH"] = wb + os.pathsep + os.environ.get("PATH", "")
        from neuromaps import transforms
        from nibabel.gifti import GiftiDataArray, GiftiImage

        for H, cpath in need_resample:
            gi = GiftiImage()
            gi.add_gifti_data_array(GiftiDataArray(labels164[H].astype(np.float32)))
            res = transforms.fsaverage_to_fslr(gi, "32k", hemi=H, method="nearest")
            arr = np.rint(np.asarray(res[0].agg_data())).astype(np.int16)
            np.save(cpath, arr)
            out[H] = arr
    return out
