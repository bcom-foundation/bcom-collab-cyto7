"""Volume export and connectivity helpers (UPSTREAM PROVENANCE — see note).

.. note::

   This module is the **upstream generator** for the tractography figures: it
   produced the cached tables in ``resources/tractography/`` that
   ``plot_tractography_analysis.py`` turns into figures. It is included for
   provenance and reproducibility of *that generation step*, and is **not run
   as part of the figure pipeline**.

   Re-running it requires external resources that are NOT shipped in this
   repository:

   * **FreeSurfer 6.0 inside Docker** (``mri_aparc2aseg``,
     ``mri_robust_register``, ``mri_vol2vol``) and FreeSurfer's ``fsaverage``
     (incl. ``mri/T1.mgz``), to warp the cyto7 annot into a label *volume* on
     the tractography reference grid (the result, ``cyto7_in_reference.nii.gz``,
     *is* cached in ``resources/tractography/``).
   * A **whole-brain tractogram** in the reference space (e.g. the HCP-1065
     population-averaged ``.trk`` atlas bundles) plus the reference volume
     ``reference.nii.gz`` (~23 MB) — both external and not in this repo.
   * **DIPY** (``pip install dipy``), used by the connectivity / geometry
     functions below.

   The functions are also intentionally Qt/VTK-free (they were called from the
   interactive "Painter" app); there is no CLI ``main`` here. To regenerate the
   tables, call ``compute_connectivity_per_bundle`` / ``compute_bundle_geometry_
   per_bundle`` / ``compute_connectivity_matrix`` with the tractogram + label
   volume, as the app did.

Original module docstring follows.

Volume export and connectivity helpers.

This module wraps FreeSurfer 6.0 commands (``mri_aparc2aseg``,
``mri_robust_register``, ``mri_vol2vol``) executed inside Docker, plus DIPY
helpers that turn a cyto label volume on the tractography reference grid and
an HCP tractogram into a 7x7 connectivity matrix and per-bundle geometry
tables.

The module is intentionally Qt/VTK-free so the pure logic can be unit-tested
without an OpenGL context.

Key design choice
-----------------
``mri_aparc2aseg`` writes cortical labels as ``1000 + ctab_value`` (left
hemisphere) and ``2000 + ctab_value`` (right hemisphere). To make the
downstream ``matrix[1:8, 1:8]`` slicing work directly, this module:

1. Re-saves the annot with a CTAB whose ``structure_id`` column equals the
   internal label (0..7).
2. Post-processes the volume produced by ``mri_aparc2aseg`` to remap
   ``1000+i`` and ``2000+i`` back to ``i``, with everything else set to 0.

The resulting volume is a clean 0..7 cytoarchitectonic label map.

Pipeline
--------
``mri_aparc2aseg`` produces fsaverage-space cyto labels (``.mgz``).
``mri_robust_register`` computes a linear map from fsaverage's ``mri/T1.mgz``
to the user-specified tractography reference (e.g. HCP-1065
``reference.nii.gz`` in ICBM2009 nonlinear asymmetric space); the transform is
saved as an LTA. ``mri_vol2vol --lta`` warps the label volume onto that
reference grid. The output (e.g. ``cyto7_in_reference.nii.gz``) lives in the
same voxel grid and world frame as the tractography reference and streamlines.
:func:`resample_labels_volume_to_trk_reference` is a no-op when the source is
already on that grid; otherwise it resamples onto the TRK header grid.

Connectivity matrices (``dipy.tracking.utils.connectivity_matrix``) support
two counting modes: **endpoints only** (``inclusive=False``) counts
streamlines whose two ends land in two regions; **any pass-through**
(``inclusive=True``) counts streamlines that visit both regions at any point
along the tract. These have different interpretations; the GUI exposes the
choice for the saved matrix. Because symmetric matrices zero the diagonal in
DIPY, within-region streamline counts (e.g. U-fibers) are filled separately
using endpoint-only, asymmetric counting.

Bundle geometry uses endpoint-only grouping so bundle membership reflects
region-to-region connections, not pass-throughs.

All frames are RAS+mm world space; no orientation flags or per-bundle
geometric corrections are applied in this module.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Filename written under ``FreeSurferConfig.output_dir`` by
# :func:`compute_fsaverage_to_reference_registration`.
FSAVERAGE_TO_REFERENCE_LTA_FILENAME = "fsaverage_to_reference.lta"

LogCallback = Callable[[str], None]


# --- Names used by both the matrix and the geometry CSVs ---
CYTO_NAMES_FULL = [
    "Medial Wall",
    "Allocortex",
    "Agranular",
    "Dysgranular",
    "Eulaminate-I",
    "Eulaminate-II",
    "Eulaminate-III",
    "Koniocortex",
]
CYTO_NAMES = CYTO_NAMES_FULL[1:]  # 7 names, label ids 1..7


def make_freesurfer_friendly_ctab(base_ctab: np.ndarray) -> np.ndarray:
    """Return a CTAB copy whose ``structure_id`` (column 4) equals the row index.

    The painter's normal CTAB stores packed RGB values as the structure_id so
    annot files round-trip nicely with other viewers. ``mri_aparc2aseg`` adds
    the structure_id to ``1000`` (LH) or ``2000`` (RH) when projecting to the
    volume, so for a clean 0..7 label map we want structure_id == internal
    label. This helper produces such a CTAB without mutating the input.
    """
    ctab = np.asarray(base_ctab).copy()
    if ctab.ndim != 2 or ctab.shape[1] < 5:
        raise ValueError("CTAB must be a 2D array with at least 5 columns.")
    for i in range(ctab.shape[0]):
        ctab[i, 4] = int(i)
    return ctab


def remap_aparc2aseg_volume(
    data: np.ndarray, n_labels: int = 7
) -> np.ndarray:
    """Remap an mri_aparc2aseg volume so cortical labels are 1..n_labels.

    Args:
        data: 3D integer label volume produced by ``mri_aparc2aseg``.
        n_labels: number of cortical labels (default 7 for cyto7).

    Returns:
        A new int32 volume of the same shape with values in [0, n_labels].
        Voxels outside the cortical labels are set to 0.
    """
    data = np.asarray(data).astype(np.int32, copy=False)
    out = np.zeros_like(data, dtype=np.int32)
    for i in range(1, n_labels + 1):
        out[data == 1000 + i] = i
        out[data == 2000 + i] = i
    return out


@dataclass
class FreeSurferConfig:
    """Configuration for running FreeSurfer 6.0 inside Docker."""

    docker_image: str = "freesurfer/freesurfer:6.0"
    license_path: str = ""
    license_mount: str = "/opt/freesurfer/license.txt"
    subjects_dir: str = ""
    subject: str = "fsaverage"
    output_dir: str = ""
    extra_docker_args: List[str] = field(default_factory=list)

    def validate(self, require_subject: bool = True) -> None:
        if not self.docker_image:
            raise ValueError("Docker image is required.")
        if not self.license_path or not Path(self.license_path).is_file():
            raise FileNotFoundError(
                f"FreeSurfer license file not found: {self.license_path!r}"
            )
        if not self.subjects_dir or not Path(self.subjects_dir).is_dir():
            raise FileNotFoundError(
                f"SUBJECTS_DIR not found: {self.subjects_dir!r}"
            )
        if require_subject:
            if not self.subject:
                raise ValueError("Subject name is required (e.g. 'fsaverage').")
            subj_path = Path(self.subjects_dir) / self.subject
            if not subj_path.is_dir():
                raise FileNotFoundError(
                    f"Subject directory not found: {subj_path}"
                )
        if not self.output_dir:
            raise ValueError("Output directory is required.")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def mount_host_volume_as_tractography_reference(
    host_reference_path: str,
) -> Tuple[str, str, Dict[str, str]]:
    """Mount a tractography reference volume read-only at ``/template/reference<ext>``.

    The file extension is preserved so FreeSurfer dispatches the correct
    reader (NIfTI vs MGZ).

    Returns
    -------
    container_path, log_label, extra_mounts
        ``extra_mounts`` maps resolved host path -> container path for
        :func:`build_docker_command`.
    """
    host_p = Path(host_reference_path)
    if not host_p.is_file():
        raise FileNotFoundError(
            f"Tractography reference volume not found: {host_reference_path!r}"
        )
    resolved = str(host_p.resolve())
    suffixes = [s.lower() for s in host_p.suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] == [".nii", ".gz"]:
        ext = ".nii.gz"
    elif suffixes and suffixes[-1] in (".mgz", ".mgh", ".nii"):
        ext = suffixes[-1]
    else:
        raise ValueError(
            f"Unsupported reference extension on {host_p.name!r}; "
            "expected .mgz, .mgh, .nii, or .nii.gz"
        )
    container_path = f"/template/reference{ext}"
    return container_path, f"host:{host_p.name}", {resolved: container_path}


def default_warped_labels_nifti_basename(
    input_volume_host_path: str, subject: str
) -> str:
    """Default ``*_in_reference.nii.gz`` basename for :func:`mri_vol2vol` output."""
    inp = Path(input_volume_host_path)
    stem = inp.name
    for ext in (".mgz", ".nii.gz", ".nii"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    sub = (subject or "").strip()
    suffix = f"_{sub}" if sub else ""
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return f"{stem}_in_reference.nii.gz"


def _norm_host_path(p: str) -> str:
    """Normalize a host path for Docker (forward slashes, absolute)."""
    s = os.fspath(Path(p).resolve())
    return s.replace("\\", "/")


def build_docker_command(
    cfg: FreeSurferConfig,
    inner_cmd: str,
    extra_mounts: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return the argv list for ``docker run`` that executes ``inner_cmd``.

    ``inner_cmd`` is a single bash string executed inside the container.
    The license, SUBJECTS_DIR and output dir are mounted at fixed paths
    (``cfg.license_mount``, ``/subjects``, ``/output``).
    """
    cfg.validate()

    cmd: List[str] = [
        "docker", "run", "--rm",
        "-v", f"{_norm_host_path(cfg.license_path)}:{cfg.license_mount}:ro",
        "-v", f"{_norm_host_path(cfg.subjects_dir)}:/subjects",
        "-v", f"{_norm_host_path(cfg.output_dir)}:/output",
        "-e", "SUBJECTS_DIR=/subjects",
    ]
    if extra_mounts:
        for host, inside in extra_mounts.items():
            cmd.extend(["-v", f"{_norm_host_path(host)}:{inside}:ro"])
    cmd.extend(list(cfg.extra_docker_args))
    cmd.extend([
        "--entrypoint", "/bin/bash",
        cfg.docker_image,
        "-lc", inner_cmd,
    ])
    return cmd


def run_streamed(
    cmd: List[str],
    log_cb: Optional[LogCallback] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Run ``cmd`` and stream stdout/stderr line-by-line to ``log_cb``.

    Returns the process return code.
    """
    if log_cb:
        log_cb("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if log_cb:
                log_cb(line.rstrip())
    finally:
        rc = proc.wait()
    if log_cb:
        log_cb(f"[exit code: {rc}]")
    return rc


def docker_available(log_cb: Optional[LogCallback] = None) -> bool:
    """Return True if ``docker`` is on PATH and the daemon answers."""
    if shutil.which("docker") is None:
        if log_cb:
            log_cb("docker executable not found on PATH.")
        return False
    try:
        rc = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).returncode
    except (subprocess.SubprocessError, OSError) as exc:
        if log_cb:
            log_cb(f"`docker info` failed: {exc}")
        return False
    return rc == 0


def ensure_image_pulled(
    image: str, log_cb: Optional[LogCallback] = None
) -> None:
    """Pull ``image`` if it is not already present locally."""
    if not image:
        raise ValueError("Image name is required.")
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode == 0:
        if log_cb:
            log_cb(f"Docker image already present: {image}")
        return
    if log_cb:
        log_cb(f"Pulling docker image: {image}")
    rc = run_streamed(["docker", "pull", image], log_cb)
    if rc != 0:
        raise RuntimeError(f"Failed to pull docker image: {image}")


def _save_remapped_volume(
    in_path: str, out_path: str, n_labels: int = 7
) -> None:
    """Read ``in_path``, remap to 0..n_labels, and write to ``out_path``."""
    import nibabel as nib

    img = nib.load(in_path)
    data = np.asanyarray(img.dataobj).astype(np.int32)
    clean = remap_aparc2aseg_volume(data, n_labels=n_labels)
    if isinstance(img, nib.Nifti1Image):
        new = nib.Nifti1Image(clean, img.affine, img.header)
    else:
        new = nib.MGHImage(clean, img.affine, img.header)
    nib.save(new, out_path)


def convert_annot_to_volume(
    cfg: FreeSurferConfig,
    annot_tag: str,
    output_name: Optional[str] = None,
    n_labels: int = 7,
    post_process: bool = True,
    log_cb: Optional[LogCallback] = None,
) -> str:
    """Run ``mri_aparc2aseg`` for ``annot_tag`` and remap to 0..n_labels.

    The annot files at ``$SUBJECTS_DIR/<subject>/label/{lh,rh}.<annot_tag>.annot``
    must already exist (write them with :func:`save_annot_to_subject` from
    inside the painter). The output is written under ``cfg.output_dir``.
    """
    if output_name is None:
        output_name = f"{annot_tag}_{cfg.subject}.mgz"
    output_host = str(Path(cfg.output_dir) / output_name)

    raw_name = output_name
    if post_process:
        raw_name = output_name.replace(".mgz", "_raw.mgz")
        if raw_name == output_name:
            raw_name = output_name + ".raw"
    raw_host = str(Path(cfg.output_dir) / raw_name)

    inner = (
        f"mri_aparc2aseg --s {cfg.subject} "
        f"--annot {annot_tag} "
        f"--o /output/{raw_name}"
    )
    cmd = build_docker_command(cfg, inner)
    rc = run_streamed(cmd, log_cb)
    if rc != 0:
        raise RuntimeError(f"mri_aparc2aseg failed (exit {rc}).")
    if not Path(raw_host).is_file():
        raise FileNotFoundError(
            f"Expected mri_aparc2aseg output missing: {raw_host}"
        )

    if post_process:
        if log_cb:
            log_cb(
                f"Post-processing: remapping 1xxx/2xxx -> 1..{n_labels} "
                f"into {output_host}"
            )
        _save_remapped_volume(raw_host, output_host, n_labels=n_labels)
    return output_host


def compute_fsaverage_to_reference_registration(
    cfg: FreeSurferConfig,
    reference_host_path: str,
    output_lta_host_path: str,
    log_cb: Optional[LogCallback] = None,
) -> str:
    """Compute fsaverage → reference linear registration via mri_robust_register.

    Runs inside the FreeSurfer Docker container. The reference is mounted
    read-only at ``/template/reference.<ext>`` (extension preserved). Output LTA
    is written on the host at ``output_lta_host_path``.

    Returns
    -------
    str
        Resolved host path to the LTA file.
    """
    ref_p = Path(reference_host_path)
    if not ref_p.is_file():
        raise FileNotFoundError(
            f"Reference volume not found: {reference_host_path!r}")

    t1_host = Path(cfg.subjects_dir) / cfg.subject / "mri" / "T1.mgz"
    if not t1_host.is_file():
        raise FileNotFoundError(
            f"Subject T1 required for mri_robust_register not found: {t1_host}"
        )

    cfg.validate()
    out_lta = Path(output_lta_host_path)
    out_lta.parent.mkdir(parents=True, exist_ok=True)
    lta_name = out_lta.name

    targ_path, targ_label, extra_mounts = mount_host_volume_as_tractography_reference(
        str(ref_p)
    )
    mov = f"$SUBJECTS_DIR/{cfg.subject}/mri/T1.mgz"
    inner = (
        "mri_robust_register "
        f"--mov {mov} "
        f"--dst {targ_path} "
        f"--lta /output/{lta_name} "
        "--satit"
    )
    if log_cb:
        log_cb(
            f"[register] mri_robust_register --mov {mov} --dst {targ_path} "
            f"({targ_label}) --lta /output/{lta_name} --satit"
        )
    cmd = build_docker_command(cfg, inner, extra_mounts=extra_mounts)
    rc = run_streamed(cmd, log_cb)
    if rc != 0:
        raise RuntimeError(f"mri_robust_register failed (exit {rc}).")
    if not out_lta.is_file():
        raise FileNotFoundError(
            f"Expected LTA missing after registration: {out_lta}"
        )
    if log_cb:
        log_cb(f"[register] Wrote {out_lta}")
    return str(out_lta.resolve())


def build_mri_vol2vol_lta_shell_command(
    mov_basename: str,
    output_nifti_basename: str,
    targ_path: str,
    lta_basename: str,
    interp: str,
) -> str:
    """Shell command for ``mri_vol2vol`` with ``--lta`` in Docker.

    ``mov_basename`` / ``output_nifti_basename`` / ``lta_basename`` are
    filenames only under ``/output``.
    """
    return (
        "mri_vol2vol "
        f"--mov /output/{mov_basename} "
        f"--targ {targ_path} "
        f"--lta /output/{lta_basename} "
        f"--interp {interp} "
        f"--o /output/{output_nifti_basename}"
    )


def warp_volume_to_reference(
    cfg: FreeSurferConfig,
    input_volume: str,
    reference_host_path: str,
    lta_host_path: str,
    output_name: Optional[str] = None,
    interp: str = "nearest",
    log_cb: Optional[LogCallback] = None,
) -> str:
    """Warp a label volume onto the tractography reference grid using ``--lta``.

    ``input_volume`` is a host path; if it does not already live inside
    ``cfg.output_dir`` it will be copied there so the container can see it.

    ``lta_host_path`` must be a file under ``cfg.output_dir`` (mounted at
    ``/output``) so the inner command can pass ``--lta /output/<name>``.
    ``reference_host_path`` must be the same file used when computing the LTA
    (same path on disk as for :func:`compute_fsaverage_to_reference_registration`).
    """
    inp = Path(input_volume)
    if not inp.is_file():
        raise FileNotFoundError(f"Input volume missing: {inp}")
    lta_p = Path(lta_host_path)
    if not lta_p.is_file():
        raise FileNotFoundError(f"LTA not found: {lta_host_path!r}")

    cfg.validate()
    out_dir = Path(cfg.output_dir).resolve()
    if lta_p.resolve().parent != out_dir:
        raise ValueError(
            "lta_host_path must be inside cfg.output_dir so Docker sees it "
            f"at /output/ (got {lta_host_path!r}, output_dir={cfg.output_dir!r})"
        )

    if inp.parent.resolve() != out_dir:
        target = out_dir / inp.name
        shutil.copyfile(inp, target)
        if log_cb:
            log_cb(f"Copied input volume into output dir: {target}")
        inp = target

    if output_name is None:
        output_name = default_warped_labels_nifti_basename(
            str(inp), cfg.subject
        )
    output_host = str(out_dir / output_name)
    lta_basename = lta_p.name

    targ_path, targ_label, extra_mounts = mount_host_volume_as_tractography_reference(
        reference_host_path
    )
    if log_cb:
        log_cb(
            f"[warp] mri_vol2vol --targ {targ_label} --lta /output/{lta_basename}"
        )

    inner = build_mri_vol2vol_lta_shell_command(
        inp.name,
        output_name,
        targ_path,
        lta_basename,
        interp,
    )
    cmd = build_docker_command(cfg, inner, extra_mounts=extra_mounts)
    rc = run_streamed(cmd, log_cb)
    if rc != 0:
        raise RuntimeError(f"mri_vol2vol failed (exit {rc}).")
    if not Path(output_host).is_file():
        raise FileNotFoundError(
            f"Expected warped volume missing: {output_host}"
        )

    if log_cb:
        try:
            import nibabel as nib

            img = nib.load(output_host)
            sh = img.shape[:3]
            log_cb(f"[warp] Output NIfTI shape={sh} path={output_host}")
        except Exception as exc:  # noqa: BLE001
            log_cb(f"[warp] could not read output shape: {exc}")
    return output_host


def canonicalize_nifti_for_viewers(
    nifti_path: str,
    log_cb: Optional[LogCallback] = None,
) -> str:
    """Re-save a NIfTI in nibabel \"closest canonical\" RAS storage order.

    ``mri_vol2vol`` and some other FreeSurfer writers emit NIfTI whose
    **qform/sform correctly map voxels to RAS+mm world space**, but whose
    **IJK axis ordering** is permuted relative to what many viewers expect.
    In 3D Slicer this often appears as wrong slice presets (e.g. \"Axial\"
    showing a coronal cut). Native ``fsaverage`` ``.mgz`` can look fine
    because the FreeSurfer reader applies different defaults.

    :func:`nibabel.as_closest_canonical` permutes/flips the array and updates
    the affine so world coordinates of every voxel are **unchanged** — safe
    for label maps and for subsequent ``resample_from_to`` / tractography.

    Parameters
    ----------
    nifti_path:
        Path to ``.nii`` or ``.nii.gz`` (overwritten in place).

    Returns
    -------
    str
        Resolved path to ``nifti_path``.
    """
    import nibabel as nib

    p = Path(nifti_path)
    if not p.is_file():
        raise FileNotFoundError(f"NIfTI not found: {nifti_path!r}")
    low = p.name.lower()
    if not (low.endswith(".nii") or low.endswith(".nii.gz")):
        raise ValueError(
            f"Expected .nii / .nii.gz for canonicalization: {nifti_path!r}")

    img = nib.load(str(p))
    canon = nib.as_closest_canonical(img)

    unchanged = (
        img.shape == canon.shape
        and np.allclose(img.affine, canon.affine, rtol=1e-5, atol=1e-5)
    )
    if unchanged:
        if log_cb:
            log_cb(
                f"[orientation] NIfTI already in canonical RAS storage order: "
                f"{p.name}"
            )
        return str(p.resolve())

    if log_cb:
        log_cb(
            f"[orientation] Re-saving {p.name}: canonical RAS axis order for "
            f"viewers (array shape {img.shape[:3]} → {canon.shape[:3]}). "
            "World locations of labels are unchanged."
        )
    nib.save(canon, str(p))
    return str(p.resolve())


# ---------- DIPY analyses (lazy imports) ----------

def _import_dipy_stack():
    try:
        import nibabel as nib  # noqa: F401
        import pandas as pd  # noqa: F401
        from dipy.io.streamline import load_tractogram  # noqa: F401
        from dipy.tracking import utils  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DIPY/pandas/nibabel are required for connectivity analysis.\n"
            "Install them with: pip install dipy pandas nibabel"
        ) from exc


def log_trk_nifti_reference_agreement(
    trk_path: str,
    reference_img: Any,
    log_cb: Optional[LogCallback] = None,
) -> bool:
    """Compare TRK spatial metadata to a label ``Nifti1Image``; log a clear verdict.

    DIPY ``connectivity_matrix`` and our VTK mesh both assume streamline
    endpoints (in RAS+ millimetres) live in the **same** world space as
    ``reference_img.affine`` maps voxel indices to. That is only guaranteed
    when the TRK header's ``voxel_to_rasmm``, dimensions, voxel sizes, and
    orientation match the NIfTI — the check ``is_header_compatible`` enforces
    when ``trk_header_check=True`` in ``load_tractogram``.

    Parameters
    ----------
    trk_path:
        ``.trk`` or ``.trk.gz`` path (nibabel reads either).
    reference_img:
        Loaded label volume from ``nibabel.load(...)``.
    log_cb:
        Optional logger; multi-line explanation on mismatch.

    Returns
    -------
    bool
        ``True`` when spatial attributes match within tolerance.
    """
    _import_dipy_stack()
    import nibabel as nib
    from dipy.io.utils import get_reference_info

    try:
        trk_file = nib.streamlines.load(str(trk_path), lazy_load=True)
        aff_t, dim_t, vs_t, vo_trk = get_reference_info(trk_file)
        aff_n, dim_n, vs_n, vo_nii = get_reference_info(reference_img)
    except Exception as exc:
        if log_cb:
            log_cb(
                f"[spatial] Could not compare TRK / NIfTI headers "
                f"({type(exc).__name__}: {exc}).")
        return False

    def _vo_str(vo: Any) -> str:
        if isinstance(vo, bytes):
            return vo.decode("latin1", errors="replace").upper()
        return str(vo).upper()

    vo_trk_s, vo_nii_s = _vo_str(vo_trk), _vo_str(vo_nii)

    aff_t_arr = np.asarray(aff_t, dtype=np.float64)
    aff_n_arr = np.asarray(aff_n, dtype=np.float64)
    aff_t_nifti_center = trk_voxel_affine_as_nifti_center(aff_t_arr)
    ok_affine_raw = np.allclose(aff_t_arr, aff_n_arr, rtol=1e-3, atol=1e-3)
    ok_affine_centered = np.allclose(
        aff_t_nifti_center, aff_n_arr, rtol=1e-3, atol=1e-3)
    ok_affine = ok_affine_raw or ok_affine_centered
    ok_shape = np.array_equal(dim_t, dim_n)
    ok_vs = np.allclose(vs_t, vs_n, rtol=1e-3, atol=1e-3)
    ok_vo = vo_trk_s == vo_nii_s
    linear_trk_nii = np.allclose(
        aff_t_arr[:3, :3], aff_n_arr[:3, :3], rtol=1e-3, atol=1e-3)
    linear_center_nii = np.allclose(
        aff_t_nifti_center[:3, :3], aff_n_arr[:3, :3], rtol=1e-3, atol=1e-3)
    same_voxel_orientation = linear_trk_nii or linear_center_nii
    ok = ok_affine and ok_shape and ok_vs and ok_vo
    relaxed_world_match = (
        same_voxel_orientation and ok_vs and ok_vo and not ok)

    if not log_cb:
        return bool(ok or relaxed_world_match)

    if ok:
        msg = (
            "[spatial] TRK header matches the label NIfTI (voxel↔RAS affine, "
            "grid shape, voxel sizes, orientation code). Streamlines and "
            "labels share one consistent RAS+mm world space."
        )
        if ok_affine_centered and not ok_affine_raw:
            msg += (
                " (NIfTI uses standard voxel-**center** indexing; TRK header "
                "matrix is TrackVis **corner** convention — treated as match.)"
            )
        log_cb(msg)
        return True

    if relaxed_world_match:
        dim_t_l = np.asarray(dim_t).tolist()
        dim_n_l = np.asarray(dim_n).tolist()
        log_cb(
            "[spatial] OK: TRK and label volume share world frame; bounding "
            f"boxes differ (TRK {dim_t_l} vs volume {dim_n_l}) — fine for "
            "visualization."
        )
        return True

    log_cb(
        "[spatial] **Mismatch** between the tractography TRK reference and "
        "your label NIfTI (see details below). Streamline points are still "
        "read in the TRK file's RAS+mm space; endpoint voxel labels use "
        "**this** NIfTI's affine. If the two templates differ, both "
        "**visualization** and **connectivity_matrix** assign endpoints to the "
        "wrong voxels — matrices can still look structured without being "
        "correct. Resample the cyto labels onto the TRK template grid (or "
        "use the reference MNI volume bundled with the tractography atlas).")
    if not ok_affine:
        log_cb(
            "[spatial]   · Affine mismatch (TRK voxel_to_rasmm vs NIfTI sform, "
            "including NIfTI voxel-center vs TRK corner adjustment).")
    if not ok_shape:
        log_cb(
            f"[spatial]   · Grid shape TRK {np.asarray(dim_t).tolist()} vs "
            f"volume {np.asarray(dim_n).tolist()}.")
    if not ok_vs:
        log_cb("[spatial]   · Voxel size mismatch.")
    if not ok_vo:
        log_cb(f"[spatial]   · Voxel order TRK {vo_trk_s!r} vs NIfTI {vo_nii_s!r}.")
    return ok


def resolve_trk_path_for_spatial_reference(
    trk_path_or_dir: str,
    log_cb: Optional[LogCallback] = None,
) -> str:
    """Return a ``.trk`` or ``.trk.gz`` path whose header defines the atlas grid.

    If ``trk_path_or_dir`` is a file, it is returned as-is. If it is a
    directory (HCP-style atlas), the first bundle from :func:`find_trk_bundles`
    is used (deterministic sort).
    """
    p = Path(trk_path_or_dir)
    if p.is_file():
        low = p.name.lower()
        if not (low.endswith(".trk") or low.endswith(".trk.gz")):
            raise ValueError(
                f"Not a tractogram file: {trk_path_or_dir!r} "
                "(expected .trk or .trk.gz)")
        return str(p.resolve())
    if p.is_dir():
        bundles = find_trk_bundles(str(p), log_cb=log_cb)
        if not bundles:
            raise FileNotFoundError(
                f"No .trk / .trk.gz files under directory: {trk_path_or_dir!r}")
        _cat, _stem, fpath = bundles[0]
        if log_cb:
            log_cb(
                f"[resample] Reference TRK (first sorted bundle): {fpath}")
        return str(Path(fpath).resolve())
    raise FileNotFoundError(
        f"Tractogram path is not a file or directory: {trk_path_or_dir!r}")


# Basenames (case-insensitive) searched in the atlas folder next to .trk files.
# Drop in the tractography template / reference T1 the atlas documentation
# points to (must match the .trk header grid — the log checks this).
_ATLAS_REFERENCE_CANDIDATES: Tuple[str, ...] = (
    "tractography_reference.nii.gz",
    "tractography_reference.nii",
    "atlas_reference.nii.gz",
    "atlas_reference.nii",
    "reference.nii.gz",
    "reference.nii",
    "template.nii.gz",
    "template.nii",
    "mni152_template.nii.gz",
    "mni152_template.nii",
    "MNI152_T1_1mm_brain.nii.gz",
    "MNI152_T1_1mm_brain.nii",
)


def find_atlas_reference_nifti(atlas_dir: str) -> Optional[str]:
    """Return a path to an optional atlas reference NIfTI, or ``None``.

    Looks only at **files in the root** of ``atlas_dir`` (not subfolders),
    for common filenames such as ``reference.nii.gz``. If you obtain the
    MNI / diffusion template that the tractography atlas was built in, place
    it here; :func:`resolve_resample_target_volume` will prefer it when its
    header matches the first bundle's ``.trk`` header.

    Parameters
    ----------
    atlas_dir:
        Same folder you pass as the HCP-style atlas directory (the parent of
        ``association/``, ``cerebellum/``, etc.).
    """
    root = Path(atlas_dir)
    if not root.is_dir():
        return None
    lower_map = {p.name.lower(): p for p in root.iterdir() if p.is_file()}
    for cand in _ATLAS_REFERENCE_CANDIDATES:
        p = lower_map.get(cand.lower())
        if p is not None:
            return str(p.resolve())
    return None


def resolve_resample_target_volume(
    trk_path_or_dir: str,
    log_cb: Optional[LogCallback] = None,
    *,
    trk_one: Optional[str] = None,
    explicit_template_path: Optional[str] = None,
    accept_explicit_template_trk_mismatch: bool = False,
    synthetic_trk_affine: str = "nifti_center",
) -> Any:
    """NIfTI grid to use as the target for ``resample_from_to`` (tractography).

    Resolution order:

    1. **explicit_template_path** — ICBM 2009c asymmetric (197×233×189), or any
       NIfTI that shares the tractography template grid. Must pass
       :func:`log_trk_nifti_reference_agreement` against ``trk_one`` unless
       ``accept_explicit_template_trk_mismatch`` is true.
    2. Optional ``reference.nii.gz`` etc. in the atlas folder root (see
       :data:`_ATLAS_REFERENCE_CANDIDATES`).
    3. Synthetic volume from :func:`reference_nifti_image_from_trk` (``.trk``
       header only).

    Parameters
    ----------
    trk_path_or_dir:
        File or folder passed from the UI (used to locate optional reference
        NIfTI and, if ``trk_one`` is omitted, to pick the reference bundle).
    trk_one:
        If already resolved (e.g. same path used for post-save checks), pass it
        to avoid duplicate logging from :func:`resolve_trk_path_for_spatial_reference`.
    explicit_template_path:
        Optional path to the tractography template (e.g. ICBM 2009 asymmetric
        NIfTI). Use when atlas docs name a template that is not auto-discovered.
    accept_explicit_template_trk_mismatch:
        If true, use ``explicit_template_path`` even when DIPY reports header
        mismatch vs the ``.trk``. **Streamlines will mis-register** if the file
        is wrong; only for expert debugging.
    synthetic_trk_affine:
        When building the synthetic grid from the ``.trk`` header only:
        ``\"nifti_center\"`` (default) applies :func:`trk_voxel_affine_as_nifti_center`;
        ``\"raw_header\"`` uses DIPY's ``voxel_to_rasmm`` unchanged.
    """
    import nibabel as nib

    affine_mode = _coerce_synthetic_trk_affine(synthetic_trk_affine)
    resolved_trk = trk_one or resolve_trk_path_for_spatial_reference(
        trk_path_or_dir, log_cb=log_cb)
    fallback = reference_nifti_image_from_trk(
        resolved_trk, synthetic_affine=affine_mode)

    if explicit_template_path:
        p = Path(explicit_template_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"Tractography grid template not found: {explicit_template_path!r}")
        ref_img = nib.load(str(p))
        if log_cb:
            log_cb(f"[resample] Explicit tractography grid template: {p}")
        ok = log_trk_nifti_reference_agreement(resolved_trk, ref_img, log_cb=log_cb)
        if ok:
            return ref_img
        if accept_explicit_template_trk_mismatch:
            if log_cb:
                log_cb(
                    "[resample] **WARNING:** Using explicit template despite "
                    "[spatial] mismatch — tractography coordinates may not "
                    "match this volume.")
            return ref_img
        raise ValueError(
            "The explicit tractography grid template does not match the .trk "
            "header (see [spatial] lines in the log). Pick the template that "
            "was used to generate these streamlines, or enable "
            "'accept mismatch' only if you are sure the file is correct."
        )

    base = Path(trk_path_or_dir)
    search_dirs: List[Path] = []
    if base.is_dir():
        search_dirs.append(base)
    elif base.is_file():
        search_dirs.append(base.parent)

    ref_str: Optional[str] = None
    for d in search_dirs:
        ref_str = find_atlas_reference_nifti(str(d))
        if ref_str:
            break

    if not ref_str:
        if log_cb:
            if affine_mode == "raw_header":
                detail = (
                    "raw TRK voxel_to_rasmm (no +0.5 voxel-center shift)"
                )
            else:
                detail = (
                    "NIfTI voxel-center affine (+0.5 shift from TrackVis "
                    "corner indexing)"
                )
            log_cb(
                "[resample] No optional atlas reference NIfTI in folder root "
                f"(looked for e.g. reference.nii.gz); using .trk header grid "
                f"with {detail}."
            )
        return fallback

    ref_img = nib.load(ref_str)
    if log_cb:
        log_cb(f"[resample] Optional atlas reference volume: {ref_str}")
    ok = log_trk_nifti_reference_agreement(resolved_trk, ref_img, log_cb=log_cb)
    if ok:
        return ref_img

    if log_cb:
        detail = (
            "raw TRK affine (no +0.5 shift)"
            if affine_mode == "raw_header"
            else "NIfTI-center-adjusted TRK affine"
        )
        log_cb(
            "[resample] Optional reference file does **not** match the .trk "
            "header — ignoring it and using the synthesised grid from the "
            f"tractogram ({detail}; otherwise labels would land on wrong voxels)."
        )
    return fallback


def log_resample_grid_diagnosis(
    trk_path: str,
    target_img: Any,
    source_on_disk: Any,
    *,
    log_cb: Optional[LogCallback] = None,
) -> None:
    """Log .trk grid vs chosen target vs source; explain ITK-SNAP overlay cases."""
    if log_cb is None:
        return
    _import_dipy_stack()
    import nibabel as nib
    from dipy.io.utils import get_reference_info

    tf = nib.streamlines.load(str(trk_path), lazy_load=True)
    _a, dim_t, _vs, _vo = get_reference_info(tf)
    trk_shape = tuple(int(x) for x in np.asarray(dim_t).ravel()[:3])
    tgt_shape = tuple(int(x) for x in target_img.shape[:3])
    src_shape = tuple(int(x) for x in source_on_disk.shape[:3])

    log_cb(f"[resample] TRK header grid (reference bundle): {list(trk_shape)}")
    log_cb(f"[resample] Resampling target volume shape: {list(tgt_shape)}")
    log_cb(f"[resample] Source label file shape (on disk): {list(src_shape)}")
    if trk_shape != tgt_shape:
        log_cb(
            "[resample] **Warning:** Target NIfTI shape differs from TRK "
            "header dims — unusual; check target selection.")

    s_aff = np.asarray(source_on_disk.affine, dtype=float)
    t_aff = np.asarray(target_img.affine, dtype=float)
    same_grid = (
        src_shape == tgt_shape
        and np.allclose(s_aff, t_aff, rtol=1e-3, atol=1e-3)
    )
    if same_grid:
        log_cb(
            "[resample] **Interpretation:** The tractography template grid "
            "matches your cyto label volume (same shape + affine on "
            "disk). Overlays in ITK-SNAP with ``*_in_reference.nii.gz`` are "
            "expected when the warp target matches the tractography reference."
        )
    elif src_shape == tgt_shape:
        log_cb(
            "[resample] **Note:** Source and target shapes match but affines "
            "differ — overlay in ITK-SNAP may differ from the source file.")


def trk_voxel_affine_as_nifti_center(trk_voxel_to_rasmm: np.ndarray) -> np.ndarray:
    """Convert TRK ``voxel_to_rasmm`` to a NIfTI-style affine for ``resample_from_to``.

    DIPY's :func:`dipy.io.utils.get_reference_info` returns the raw TrackVis
    matrix: integer grid indices follow the **corner** of each voxel. Standard
    NIfTI / :func:`nibabel.processing.resample_from_to` map **integer** indices
    ``(i, j, k)`` to the **center** of that voxel in world space.

    Using the raw TRK matrix as the resampling target affine therefore shifts
    sampling by half a voxel along each axis — labels miss the brain and the
    ``*_trkref.nii.gz`` looks cropped or nearly empty.

    Parameters
    ----------
    trk_voxel_to_rasmm
        4×4 ``voxel_to_rasmm`` from the ``.trk`` header (DIPY/nibabel).

    Returns
    -------
    np.ndarray
        4×4 affine whose linear part matches ``trk_voxel_to_rasmm`` and whose
        translation is shifted by ``L @ (0.5, 0.5, 0.5)`` with ``L`` the 3×3
        linear block (voxel index offset in **voxel units**, not mm).
    """
    M = np.asarray(trk_voxel_to_rasmm, dtype=np.float64).copy()
    L = M[:3, :3]
    half = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    M[:3, 3] = M[:3, 3] + (L @ half)
    return M


def _coerce_synthetic_trk_affine(mode: str) -> str:
    """Return ``\"nifti_center\"`` or ``\"raw_header\"`` for TRK-derived resample targets."""
    m = (mode or "nifti_center").strip().lower().replace("-", "_")
    if m in ("nifti_center", "center"):
        return "nifti_center"
    if m in ("raw_header", "raw"):
        return "raw_header"
    raise ValueError(
        f"synthetic_trk_affine must be 'nifti_center' or 'raw_header', not {mode!r}"
    )


def reference_nifti_image_from_trk(
    trk_path: str,
    *,
    synthetic_affine: str = "nifti_center",
) -> Any:
    """Build a dummy ``Nifti1Image`` with the TRK header's shape and VOX→RAS affine.

    By default, TrackVis ``voxel_to_rasmm`` is converted with
    :func:`trk_voxel_affine_as_nifti_center` so integer voxel indices match
    NIfTI **center** convention in ``resample_from_to``. Some atlases already
    store a NIfTI-compatible matrix — using ``synthetic_affine=\"raw_header\"``
    skips the half-voxel translation (try when ``*_trkref.nii.gz`` looks
    cropped but ICBM / reference NIfTI overlay is correct).

    Headers (``dim``, ``pixdim``, qform/sform) are filled from TRK metadata.
    """
    mode = _coerce_synthetic_trk_affine(synthetic_affine)
    _import_dipy_stack()
    import nibabel as nib
    from dipy.io.utils import get_reference_info

    trk_file = nib.streamlines.load(str(trk_path), lazy_load=True)
    affine, dimensions, voxel_sizes, _vo = get_reference_info(trk_file)
    dims = np.asarray(dimensions, dtype=int).ravel()
    if dims.size < 3:
        raise ValueError(f"Invalid TRK dimensions: {dimensions!r}")
    shape = (int(dims[0]), int(dims[1]), int(dims[2]))
    data = np.zeros(shape, dtype=np.float32)
    A_raw = np.asarray(affine, dtype=np.float64)
    if mode == "raw_header":
        A = A_raw.copy()
    else:
        A = trk_voxel_affine_as_nifti_center(A_raw)
    img = nib.Nifti1Image(data, A)
    vs = np.asarray(voxel_sizes, dtype=np.float64).ravel()[:3]
    if vs.size >= 3 and np.all(np.isfinite(vs)) and np.all(vs > 0):
        img.header.set_zooms(tuple(float(v) for v in vs))
    img.set_sform(A, code=1)
    img.set_qform(A, code=1)
    return img


def resample_label_volume_to_target(
    source_img: Any,
    target_img: Any,
    *,
    log_cb: Optional[LogCallback] = None,
) -> Any:
    """Resample discrete labels from ``source_img`` onto ``target_img``'s grid.

    Uses nearest-neighbour (``order=0``) and rounds to ``int32``. Background
    uses fill value 0.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    out = resample_from_to(
        source_img,
        target_img,
        order=0,
        mode="constant",
        cval=0.0,
    )
    data = np.rint(np.asanyarray(out.dataobj, dtype=np.float64)).astype(np.int32)
    data = np.clip(data, 0, None)
    # Use the same affine as target_img (resample_from_to should match, but
    # avoid any header/qform drift by rebuilding from target explicitly when
    # the caller passes TRK-derived targets).
    A = np.asarray(target_img.affine, dtype=np.float64)
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.int32)
    hdr.set_qform(A, code=1)
    hdr.set_sform(A, code=1)
    new_img = nib.Nifti1Image(data, A, hdr)
    if log_cb:
        mx = int(data.max()) if data.size else 0
        log_cb(
            f"[resample] Output shape {data.shape}, label max {mx} "
            f"(was {int(np.max(np.asanyarray(source_img.dataobj))):d} in source).")
    return new_img


def resample_labels_volume_to_trk_reference(
    source_nifti_path: str,
    trk_path_or_dir: str,
    output_nifti_path: str,
    *,
    grid_template_nifti: Optional[str] = None,
    accept_grid_template_trk_mismatch: bool = False,
    synthetic_trk_affine: str = "nifti_center",
    log_cb: Optional[LogCallback] = None,
) -> str:
    """Resample a label volume onto the voxel grid encoded in a TRK header.

    Parameters
    ----------
    source_nifti_path:
        Cyto (or other) label map (e.g. ``*_in_reference.nii.gz`` from
        :func:`warp_volume_to_reference`, on the tractography reference grid).
    trk_path_or_dir:
        One ``.trk`` / ``.trk.gz`` **or** an HCP-style atlas folder; the
        reference grid is read from that TRK (or the first sorted bundle).
        Optional: place ``reference.nii.gz`` (or see
        :func:`find_atlas_reference_nifti`) in the **atlas root** — if its
        header matches the bundles, it is used as the resampling target.
    output_nifti_path:
        Path for the new ``.nii`` / ``.nii.gz`` (parent dirs created as needed).
    grid_template_nifti:
        Optional explicit template (e.g. ICBM 2009 asymmetric NIfTI). Must
        match the ``.trk`` header unless ``accept_grid_template_trk_mismatch``.
    accept_grid_template_trk_mismatch:
        Allow explicit template even when DIPY reports TRK/header mismatch.
    synthetic_trk_affine:
        Passed to :func:`resolve_resample_target_volume` when the fallback
        synthetic volume is built from the ``.trk`` header.

    Returns
    -------
    str
        ``output_nifti_path`` after a successful save.
    """
    import nibabel as nib

    if not Path(source_nifti_path).is_file():
        raise FileNotFoundError(
            f"Source volume not found: {source_nifti_path!r}")

    trk_one = resolve_trk_path_for_spatial_reference(
        trk_path_or_dir, log_cb=log_cb)
    tmpl = ((grid_template_nifti or "").strip() or None)
    target_img = resolve_resample_target_volume(
        trk_path_or_dir,
        log_cb=log_cb,
        trk_one=trk_one,
        explicit_template_path=tmpl,
        accept_explicit_template_trk_mismatch=(
            accept_grid_template_trk_mismatch
        ),
        synthetic_trk_affine=synthetic_trk_affine,
    )
    source_raw = nib.load(source_nifti_path)
    src_aff = np.asarray(source_raw.affine, dtype=np.float64)
    tgt_aff = np.asarray(target_img.affine, dtype=np.float64)
    if (
        source_raw.shape[:3] == target_img.shape[:3]
        and np.allclose(src_aff, tgt_aff, atol=1e-3)
    ):
        out_path = Path(output_nifti_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_nifti_path, str(out_path))
        if log_cb:
            log_cb(
                "[resample] Source and reference already share grid; "
                "copying file unchanged."
            )
            log_cb("[resample] Post-save spatial check (new volume vs same TRK):")
            saved = nib.load(str(out_path))
            log_trk_nifti_reference_agreement(trk_one, saved, log_cb=log_cb)
        return str(out_path.resolve())

    source_img = source_raw

    log_resample_grid_diagnosis(
        trk_one, target_img, source_raw, log_cb=log_cb)

    if log_cb:
        log_cb(
            f"[resample] Moving labels from shape {source_img.shape[:3]} onto "
            f"TRK grid {target_img.shape[:3]} "
            f"(from {Path(trk_one).name}).")

    out_img = resample_label_volume_to_target(
        source_img, target_img, log_cb=log_cb)

    out_path = Path(output_nifti_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    if log_cb:
        log_cb(f"[resample] Wrote {out_path}")
        res_data = np.asarray(out_img.dataobj, dtype=np.int32)
        nnz = int(np.count_nonzero(res_data))
        ntot = int(res_data.size)
        pct = 100.0 * nnz / max(ntot, 1)
        log_cb(
            f"[resample] Nonzero label voxels: {nnz} / {ntot} ({pct:.4f}%).")
        if nnz > 0 and pct < 0.01:
            log_cb(
                "[resample] **Note:** labeled voxels cover a tiny fraction of "
                "the tractography grid — if the brain mesh looks like a small "
                "chunk or streamlines look 90° off, the cyto volume may still "
                "live in a different MNI variant than this TRK atlas "
                "(or its qform/sform may be wrong on disk).")

        log_cb("[resample] Post-save spatial check (new volume vs same TRK):")
        saved = nib.load(str(out_path))
        log_trk_nifti_reference_agreement(trk_one, saved, log_cb=log_cb)

    return str(out_path.resolve())


def _log_trk_voxel_order_and_label_axes(
    trk_path: str,
    labels_img: Any,
    log_cb: Optional[LogCallback],
) -> None:
    """Log ``.trk`` ``voxel_order`` vs label NIfTI axis codes for debugging."""
    if log_cb is None:
        return
    import nibabel as nib
    from nibabel.orientations import aff2axcodes

    try:
        tf = nib.streamlines.load(str(trk_path), lazy_load=True)
        vo = tf.header.get("voxel_order", b"")
        if isinstance(vo, bytes):
            vo = vo.decode("latin1", errors="replace").strip().upper()
        ax = "".join(aff2axcodes(labels_img.affine))
        log_cb(
            f"[tractography] TRK voxel_order={vo!r}, "
            f"label volume aff2axcodes={ax!r}."
        )
    except Exception as exc:  # noqa: BLE001
        log_cb(f"[tractography] Could not read TRK voxel_order: {exc}")


def _streamline_subbundle(streamlines: Any, indices: Any) -> Sequence[Any]:
    """Index into DIPY ArraySequence or a plain list of streamline arrays."""
    if isinstance(streamlines, list):
        idx = np.asarray(indices, dtype=int).ravel()
        return [streamlines[int(i)] for i in idx]
    return streamlines[indices]


def load_streamlines_aligned_to_volume(
    trk_path: str,
    nifti_path: str,
    log_cb: Optional[LogCallback] = None,
    *,
    labels_img: Optional[Any] = None,
) -> List[np.ndarray]:
    """Return streamline polylines in **RAS+mm** aligned with ``labels_img``.

    Loads the tractogram with ``reference="same"`` so DIPY's internal spatial
    metadata matches the ``.trk`` header (integer grid, ``voxel_to_rasmm``).
    Nibabel maps stored points to world mm; **do not** pass the canonical
    label :class:`~nibabel.Nifti1Image` as the tractography reference — that
    does not re-orient points and can disagree with SFT bookkeeping when
    headers use LPS voxel order.

    Parameters
    ----------
    trk_path, nifti_path:
        Tractogram path and label NIfTI path (``nifti_path`` used if
        ``labels_img`` is omitted).
    labels_img:
        Optional pre-loaded image; still canonicalized for logging / axis codes.
    """
    _import_dipy_stack()
    import nibabel as nib

    if labels_img is None:
        if not Path(nifti_path).is_file():
            raise FileNotFoundError(
                f"MNI / label volume not found: {nifti_path!r}")
        labels_img = nib.load(nifti_path)
    labels_img = nib.as_closest_canonical(labels_img)
    _log_trk_voxel_order_and_label_axes(trk_path, labels_img, log_cb)

    sft = _load_tractogram_safe(trk_path, "same", log_cb=log_cb)
    if sft is None:
        raise RuntimeError(
            f"Could not load tractogram {trk_path!r}; see log above.")
    return [np.asarray(s, dtype=np.float32) for s in sft.streamlines]


def _load_tractogram_safe(
    trk_path: str,
    reference: Any,
    log_cb: Optional[LogCallback] = None,
):
    """Load a tractogram with permissive checks; return ``None`` on failure.

    ``reference`` is passed to DIPY/nibabel ``load_tractogram``: a NIfTI image,
    or the string ``\"same\"`` to use the ``.trk`` file's own header (recommended
    for visualization and connectivity so SFT grid metadata matches the file).

    DIPY's ``load_tractogram`` does **not** support ``.trk.gz`` - it
    switches on the file extension and ``.gz`` isn't in its allowlist, so
    it silently returns ``False``. Since the HCP1065 atlas ships gzipped,
    every bundle fails that way. To handle this, ``.trk.gz`` files are
    transparently decompressed to a temporary ``.trk`` file before being
    handed to DIPY.

    Both ``bbox_valid_check`` and ``trk_header_check`` are disabled so
    files still load when the TRK header does not exactly match the
    reference NIfTI (common when the tractography atlas template and the
    cyto MNI volume come from different pipelines). In that situation,
    streamline coordinates remain in the TRK file's world-mm frame while
    ``connectivity_matrix`` maps endpoints into voxels using the **label
    volume's** affine — fix grid alignment via
    :func:`resample_labels_volume_to_trk_reference`.

    Streamlines are **not** pruned here: dropping out-of-grid tracks only
    hides alignment bugs. If ``connectivity_matrix(..., inclusive=True)``
    raises on negative voxel indices, fix the label/TRK spatial agreement
    (see :func:`resample_labels_volume_to_trk_reference`).

    If anything still goes wrong, the helper logs the reason and returns
    ``None`` so callers can skip the file gracefully.
    """
    from dipy.io.streamline import load_tractogram

    src = Path(trk_path)
    actual_path = str(src)
    tmp_dir: Optional[Path] = None

    if src.name.lower().endswith(".trk.gz"):
        import gzip
        import tempfile
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="painter_trk_"))
            # ``X.trk.gz`` -> ``X.trk``
            tmp_trk = tmp_dir / src.name[: -len(".gz")]
            with gzip.open(str(src), "rb") as fin, open(tmp_trk, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            actual_path = str(tmp_trk)
        except Exception as exc:
            if log_cb:
                log_cb(
                    f"     gzip decompression failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

    try:
        try:
            result = load_tractogram(
                actual_path,
                reference,
                bbox_valid_check=False,
                trk_header_check=False,
            )
        except Exception as exc:
            if log_cb:
                log_cb(
                    f"     load_tractogram raised: "
                    f"{type(exc).__name__}: {exc}"
                )
            return None

        if not hasattr(result, "streamlines"):
            if log_cb:
                log_cb(
                    f"     load_tractogram returned {result!r}; "
                    "check the file extension (.trk/.tck/.fib/.vtk/.dpy/"
                    ".trx) and that the reference volume is readable."
                )
            return None
        return result
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_volume_and_streamlines(
    nifti_path: str,
    trk_path: str,
    log_cb: Optional[LogCallback] = None,
) -> Tuple[Any, np.ndarray, np.ndarray, Any, Any]:
    import nibabel as nib

    if log_cb:
        log_cb(f"Loading label volume: {nifti_path}")
    labels_img = nib.as_closest_canonical(nib.load(nifti_path))
    if log_cb:
        log_cb(
            "[orientation] Labels in canonical RAS storage order ( VTK / DIPY )."
        )
    labels_data = np.asanyarray(labels_img.dataobj).astype(np.int32)
    affine = np.asarray(labels_img.affine, dtype=np.float64)

    log_trk_nifti_reference_agreement(trk_path, labels_img, log_cb=log_cb)
    _log_trk_voxel_order_and_label_axes(trk_path, labels_img, log_cb)

    if log_cb:
        log_cb(f"Loading tractogram (this may take a minute): {trk_path}")
    tractogram = _load_tractogram_safe(trk_path, "same", log_cb=log_cb)
    if tractogram is None:
        raise RuntimeError(
            f"Could not load tractogram {trk_path!r}. "
            "Check the log above for the reason."
        )
    streamlines = tractogram.streamlines
    if log_cb:
        log_cb(f"Total streamlines loaded: {len(streamlines)}")
    return labels_img, labels_data, affine, streamlines, tractogram


def connectivity_matrix_with_inclusive_fallback(
    streamlines: Any,
    affine: np.ndarray,
    label_volume: np.ndarray,
    *,
    log_cb: Optional[LogCallback] = None,
    context: str = "",
    tractogram: Any = None,
    prefer_inclusive: bool = True,
) -> Tuple[Any, Any]:
    """Run DIPY ``connectivity_matrix``, with progressive fallbacks.

    Parameters
    ----------
    prefer_inclusive
        If True (default), try ``inclusive=True`` first (counts a connection
        if a streamline passes through both regions at any point along its
        length). If False, try ``inclusive=False`` first (counts only
        streamlines whose endpoints land in the two regions). The other
        mode is used as a fallback if the preferred mode raises.

    Falls back to ``remove_invalid_streamlines`` + endpoint-only retry
    if both modes fail.
    """
    _import_dipy_stack()
    from dipy.tracking import utils

    def _call(sl: Any, inclusive: bool):
        return utils.connectivity_matrix(
            sl,
            affine,
            label_volume,
            inclusive=inclusive,
            symmetric=True,
            return_mapping=True,
            mapping_as_streamlines=False,
        )

    last_exc: Optional[BaseException] = None
    suffix = f" ({context})" if context else ""
    order = (True, False) if prefer_inclusive else (False, True)
    for inclusive in order:
        try:
            return _call(streamlines, inclusive)
        except Exception as exc:
            last_exc = exc
            if log_cb:
                mode = "inclusive=True" if inclusive else "inclusive=False"
                log_cb(
                    f"[connectivity] {mode} failed{suffix}: "
                    f"{type(exc).__name__}: {exc}"
                )

    if tractogram is not None and hasattr(
            tractogram, "remove_invalid_streamlines"):
        try:
            rem = tractogram.remove_invalid_streamlines()
        except Exception as exc:
            if log_cb:
                log_cb(
                    f"[connectivity] remove_invalid_streamlines raised: {exc}"
                )
            if last_exc is not None:
                raise last_exc from exc
            raise
        if rem is not None and log_cb:
            removed, kept = rem
            if removed:
                log_cb(
                    f"[connectivity] Dropped {len(removed)} streamlines outside "
                    f"the label FOV (kept {len(kept)}); retrying endpoints-only."
                )
        try:
            return _call(tractogram.streamlines, False)
        except Exception as exc2:
            if log_cb:
                log_cb(
                    f"[connectivity] Retry after pruning failed{suffix}: "
                    f"{type(exc2).__name__}: {exc2}"
                )
            if last_exc is not None:
                raise last_exc from exc2
            raise exc2

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(
        "connectivity_matrix_with_inclusive_fallback: internal error")


def _fill_within_region_diagonal_from_endpoints(
    cyto_matrix: np.ndarray,
    streamlines: Any,
    affine: np.ndarray,
    labels_data: np.ndarray,
    log_cb: Optional[LogCallback] = None,
    *,
    context: str = "",
) -> None:
    """Set main diagonal of ``cyto_matrix`` from endpoint-only asymmetric counts.

    Mutates ``cyto_matrix`` in place (must be an ``(L, L)`` slice of region
    labels ``1..L``). ``symmetric=True`` in DIPY zeros the diagonal; this
    restores within-region streamline counts (both ends in the same parcel).
    """
    _import_dipy_stack()
    from dipy.tracking import utils

    n_l = int(cyto_matrix.shape[0])
    try:
        m_asym, _ = utils.connectivity_matrix(
            streamlines,
            affine,
            labels_data,
            inclusive=False,
            symmetric=False,
            return_mapping=True,
            mapping_as_streamlines=False,
        )
        within = np.diag(m_asym[1 : n_l + 1, 1 : n_l + 1])
        np.fill_diagonal(cyto_matrix, within)
    except Exception as exc:
        if log_cb:
            log_cb(
                f"[connectivity] Within-region count failed{context}: "
                f"{type(exc).__name__}: {exc}. Diagonal left at zero."
            )


def compute_connectivity_matrix(
    nifti_path: str,
    trk_path: str,
    output_csv: str,
    n_labels: int = 7,
    log_cb: Optional[LogCallback] = None,
    *,
    prefer_inclusive: bool = False,
) -> Any:
    """Compute and save the n_labels x n_labels streamline-count matrix.

    The off-diagonal entries use ``connectivity_matrix_with_inclusive_fallback``
    with the ``prefer_inclusive`` mode. The diagonal entries are the count of
    streamlines whose start and end both fall in the same region
    (endpoint-only, asymmetric DIPY call), representing within-region
    connections such as U-fibers, regardless of the off-diagonal mode.
    """
    _import_dipy_stack()
    import pandas as pd

    _, labels_data, affine, streamlines, sft = _load_volume_and_streamlines(
        nifti_path,
        trk_path,
        log_cb,
    )
    if log_cb:
        mode = (
            "inclusive=True (any point)"
            if prefer_inclusive
            else "inclusive=False (endpoints only)"
        )
        log_cb(f"Computing {n_labels}x{n_labels} connectivity matrix [{mode}]...")
    matrix, _grouping = connectivity_matrix_with_inclusive_fallback(
        streamlines,
        affine,
        labels_data,
        log_cb=log_cb,
        tractogram=sft,
        prefer_inclusive=prefer_inclusive,
    )
    cyto_matrix = matrix[1 : n_labels + 1, 1 : n_labels + 1].copy()
    if log_cb:
        log_cb(
            "Computing within-region streamline counts "
            "(diagonal, endpoint-only)..."
        )
    _fill_within_region_diagonal_from_endpoints(
        cyto_matrix,
        streamlines,
        affine,
        labels_data,
        log_cb=log_cb,
    )

    df = pd.DataFrame(cyto_matrix, index=CYTO_NAMES, columns=CYTO_NAMES)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv)
    if log_cb:
        log_cb("--- Cytoarchitectural streamline counts ---")
        log_cb(df.to_string())
        log_cb(f"Saved matrix to {output_csv}")
    return df


def find_trk_bundles(
    atlas_dir: str,
    log_cb: Optional[LogCallback] = None,
) -> List[Tuple[str, str, str]]:
    """Recursively find every ``*.trk`` and ``*.trk.gz`` under ``atlas_dir``.

    The HCP1065 atlas (and similar) is organised as
    ``<atlas_dir>/<category>/<bundle>.trk.gz`` (e.g. ``association``,
    ``cerebellum``, ``commissural``, ``cranial nerve``, ``projection``).

    Returns a list of ``(category, bundle_name, file_path)`` tuples, sorted
    by category then bundle name. ``category`` is the immediate parent
    folder name of each TRK file.

    If ``log_cb`` is provided and the walk yields zero matches, this
    function logs a small diagnostic (how many dirs/files were walked and
    the first few top-level entries) so users can see whether the path is
    wrong, the archive is still zipped, etc.
    """
    atlas = Path(atlas_dir)
    if not atlas.is_dir():
        raise NotADirectoryError(f"Atlas directory not found: {atlas_dir!r}")

    found: Dict[str, Tuple[str, str, str]] = {}
    n_dirs = 0
    n_files = 0
    for dirpath, _dirnames, filenames in os.walk(atlas):
        n_dirs += 1
        for fname in filenames:
            n_files += 1
            lower = fname.lower()
            if lower.endswith(".trk.gz"):
                stem = fname[: -len(".trk.gz")]
            elif lower.endswith(".trk"):
                stem = fname[: -len(".trk")]
            else:
                continue
            category = Path(dirpath).name
            full = str(Path(dirpath) / fname)
            key = os.path.normcase(os.path.abspath(full))
            if key in found:
                continue
            found[key] = (category, stem, full)

    if not found and log_cb is not None:
        try:
            kids = sorted(p.name for p in atlas.iterdir())
        except OSError as exc:
            kids = [f"<iterdir failed: {exc}>"]
        log_cb(
            f"[find_trk_bundles] No .trk or .trk.gz files under "
            f"{atlas_dir!r}. Walked {n_dirs} dirs and {n_files} files."
        )
        log_cb(
            f"[find_trk_bundles] Top-level entries (up to 20): "
            f"{kids[:20]}"
        )
        log_cb(
            "[find_trk_bundles] Hint: point at the folder that *contains* "
            "the category subfolders (e.g. .../HCP1065_Atlas/), and make "
            "sure the .trk.gz files are extracted (not still inside an "
            "archive)."
        )

    return sorted(found.values(), key=lambda t: (t[0], t[1]))


def compute_connectivity_per_bundle(
    nifti_path: str,
    atlas_dir: str,
    output_csv: str,
    n_labels: int = 7,
    min_streamlines: int = 1,
    log_cb: Optional[LogCallback] = None,
    *,
    prefer_inclusive: bool = False,
) -> Any:
    """Compute per-tract streamline counts between cyto-label pairs.

    Walks ``atlas_dir`` for ``*.trk(.gz)`` files (see :func:`find_trk_bundles`),
    runs ``connectivity_matrix`` on each bundle, and writes a long-form CSV
    with one row per (Category, Tract_Name, Source_Cyto, Target_Cyto). Only
    the upper triangle (``i <= j``) is recorded since the matrix is symmetric.

    Diagonal counts use the same endpoint-only within-region rule as
    :func:`compute_connectivity_matrix`. Off-diagonals follow ``prefer_inclusive``.

    Aggregated counts (summed across all bundles) are saved alongside as
    ``<output_csv>_aggregate.csv``.
    """
    _import_dipy_stack()
    import nibabel as nib
    import pandas as pd

    if log_cb:
        log_cb(f"Loading label volume: {nifti_path}")
    labels_img = nib.as_closest_canonical(nib.load(nifti_path))
    labels_data = np.asanyarray(labels_img.dataobj).astype(np.int32)
    affine = labels_img.affine

    bundles = find_trk_bundles(atlas_dir, log_cb=log_cb)
    if log_cb:
        log_cb(f"Found {len(bundles)} TRK files under {atlas_dir}")
    if bundles:
        log_trk_nifti_reference_agreement(
            bundles[0][2], labels_img, log_cb=log_cb)
        _log_trk_voxel_order_and_label_axes(
            bundles[0][2], labels_img, log_cb)

    rows: List[Dict[str, Any]] = []
    aggregate = np.zeros((n_labels, n_labels), dtype=np.int64)

    for cat, bname, fpath in bundles:
        if log_cb:
            log_cb(f"  -> {cat}/{bname}")
        tractogram = _load_tractogram_safe(fpath, "same", log_cb=log_cb)
        if tractogram is None:
            continue
        streamlines = tractogram.streamlines
        sft = tractogram
        if len(streamlines) == 0:
            if log_cb:
                log_cb("     (empty tractogram, skipping)")
            continue

        matrix, _grouping = connectivity_matrix_with_inclusive_fallback(
            streamlines,
            affine,
            labels_data,
            log_cb=log_cb,
            context=f"{cat}/{bname}",
            tractogram=sft,
            prefer_inclusive=prefer_inclusive,
        )
        cyto_block = matrix[1 : n_labels + 1, 1 : n_labels + 1].copy()
        ctx = f" [{cat}/{bname}]"
        _fill_within_region_diagonal_from_endpoints(
            cyto_block,
            streamlines,
            affine,
            labels_data,
            log_cb=log_cb,
            context=ctx,
        )

        for i in range(1, n_labels + 1):
            for j in range(i, n_labels + 1):
                count = int(cyto_block[i - 1, j - 1])
                # update aggregate (symmetric, so mirror off-diagonal)
                aggregate[i - 1, j - 1] += count
                if i != j:
                    aggregate[j - 1, i - 1] += count
                if count < min_streamlines:
                    continue
                rows.append({
                    "Category": cat,
                    "Tract_Name": bname,
                    "Source_Cyto": CYTO_NAMES_FULL[i],
                    "Target_Cyto": CYTO_NAMES_FULL[j],
                    "Streamline_Count": count,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["Category", "Tract_Name", "Streamline_Count"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    out = Path(output_csv)
    agg_path = str(out.with_name(out.stem + "_aggregate" + out.suffix))
    agg_df = pd.DataFrame(
        aggregate, index=CYTO_NAMES, columns=CYTO_NAMES)
    agg_df.to_csv(agg_path)

    if log_cb:
        log_cb(f"Saved per-tract counts to {output_csv}")
        log_cb(f"Saved aggregate matrix to {agg_path}")
        if not df.empty:
            log_cb("--- Top 10 rows ---")
            log_cb(df.head(10).to_string(index=False))
    return df


def compute_bundle_geometry_per_bundle(
    nifti_path: str,
    atlas_dir: str,
    output_csv: str,
    n_labels: int = 7,
    min_streamlines: int = 5,
    log_cb: Optional[LogCallback] = None,
) -> Any:
    """Compute per-tract bundle geometry per cyto-label pair.

    For each ``*.trk(.gz)`` file under ``atlas_dir``, compute mean length,
    tortuosity and absolute mean direction of the streamlines whose
    endpoints fall in each pair of cyto labels. ``min_streamlines`` skips
    sparse pairs (default matches the user's reference script: 5).

    Bundle membership is determined by streamline endpoints (endpoint-only
    mode). Streamlines that merely pass through a region pair without their
    endpoints landing there are excluded from that bundle's statistics.

    Output CSV columns: Category, Tract_Name, Source_Cyto, Target_Cyto,
    Streamline_Count, Mean_Length_mm, Mean_Tortuosity,
    Dir_X_LR_axis, Dir_Y_AP_axis, Dir_Z_IS_axis
    (axis magnitudes along RAS anatomical axes, not signed directions;
    see :func:`compute_bundle_geometry`).
    """
    _import_dipy_stack()
    import nibabel as nib
    import pandas as pd

    if log_cb:
        log_cb(f"Loading label volume: {nifti_path}")
    labels_img = nib.as_closest_canonical(nib.load(nifti_path))
    labels_data = np.asanyarray(labels_img.dataobj).astype(np.int32)
    affine = labels_img.affine

    bundles = find_trk_bundles(atlas_dir, log_cb=log_cb)
    if log_cb:
        log_cb(f"Found {len(bundles)} TRK files under {atlas_dir}")
    if bundles:
        log_trk_nifti_reference_agreement(
            bundles[0][2], labels_img, log_cb=log_cb)
        _log_trk_voxel_order_and_label_axes(
            bundles[0][2], labels_img, log_cb)

    rows: List[Dict[str, Any]] = []

    for cat, bname, fpath in bundles:
        if log_cb:
            log_cb(f"  -> {cat}/{bname}")
        tractogram = _load_tractogram_safe(fpath, "same", log_cb=log_cb)
        if tractogram is None:
            continue
        streamlines = tractogram.streamlines
        sft = tractogram
        if len(streamlines) == 0:
            if log_cb:
                log_cb("     (empty tractogram, skipping)")
            continue

        _matrix, grouping = connectivity_matrix_with_inclusive_fallback(
            streamlines,
            affine,
            labels_data,
            log_cb=log_cb,
            context=f"{cat}/{bname}",
            tractogram=sft,
            prefer_inclusive=False,
        )

        for (node_i, node_j), indices in grouping.items():
            if node_i == 0 or node_j == 0:
                continue
            if node_i > n_labels or node_j > n_labels:
                continue
            if node_i > node_j:
                continue

            sub_bundle = _streamline_subbundle(streamlines, indices)
            if len(sub_bundle) < min_streamlines:
                continue

            lengths: List[float] = []
            tortuosities: List[float] = []
            directions: List[np.ndarray] = []

            for s in sub_bundle:
                if len(s) < 2:
                    continue
                diffs = np.diff(s, axis=0)
                length = float(np.sum(np.linalg.norm(diffs, axis=1)))
                end_to_end = float(np.linalg.norm(s[-1] - s[0]))
                tortuosity = (length / end_to_end) if end_to_end > 0 else 1.0
                vec = np.abs(s[-1] - s[0])
                n = float(np.linalg.norm(vec))
                norm_vec = (vec / n) if n > 0 else np.zeros(3)
                lengths.append(length)
                tortuosities.append(tortuosity)
                directions.append(np.asarray(norm_vec, dtype=float))

            if not lengths:
                continue

            mean_dir = np.mean(np.stack(directions), axis=0)
            rows.append({
                "Category": cat,
                "Tract_Name": bname,
                "Source_Cyto": CYTO_NAMES_FULL[node_i],
                "Target_Cyto": CYTO_NAMES_FULL[node_j],
                "Streamline_Count": int(len(sub_bundle)),
                "Mean_Length_mm": round(float(np.mean(lengths)), 2),
                "Mean_Tortuosity": round(float(np.mean(tortuosities)), 3),
                "Dir_X_LR_axis": round(float(mean_dir[0]), 3),
                "Dir_Y_AP_axis": round(float(mean_dir[1]), 3),
                "Dir_Z_IS_axis": round(float(mean_dir[2]), 3),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["Category", "Tract_Name", "Streamline_Count"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if log_cb:
        log_cb(f"Saved per-tract geometry to {output_csv}")
        if not df.empty:
            log_cb("--- Top 10 longest tract-pair connections ---")
            top = df.sort_values("Mean_Length_mm", ascending=False).head(10)
            log_cb(
                top[
                    [
                        "Category",
                        "Tract_Name",
                        "Source_Cyto",
                        "Target_Cyto",
                        "Mean_Length_mm",
                        "Mean_Tortuosity",
                    ]
                ].to_string(index=False)
            )
    return df


def compute_bundle_geometry(
    nifti_path: str,
    trk_path: str,
    output_csv: str,
    n_labels: int = 7,
    log_cb: Optional[LogCallback] = None,
) -> Any:
    """Compute per-bundle geometric properties and save as CSV.

    Bundle membership is determined by streamline endpoints (endpoint-only
    mode). Streamlines that merely pass through a region pair without their
    endpoints landing there are excluded from that bundle's statistics.

    Direction components (Dir_*_axis) are bundle-averaged absolute values
    of the normalized end-to-end vector along each anatomical axis. They
    represent how aligned the bundle is with each axis (0 = orthogonal,
    1 = fully aligned), not a signed direction.
    """
    _import_dipy_stack()
    import pandas as pd

    _, labels_data, affine, streamlines, sft = _load_volume_and_streamlines(
        nifti_path,
        trk_path,
        log_cb,
    )
    if log_cb:
        log_cb("Grouping streamlines by region pair...")
    _matrix, grouping = connectivity_matrix_with_inclusive_fallback(
        streamlines,
        affine,
        labels_data,
        log_cb=log_cb,
        tractogram=sft,
        prefer_inclusive=False,
    )

    results: List[Dict[str, Any]] = []
    if log_cb:
        log_cb("Calculating geometric properties for each connection...")
    for (node_i, node_j), indices in grouping.items():
        if node_i == 0 or node_j == 0:
            continue
        if node_i > n_labels or node_j > n_labels:
            continue
        if node_i > node_j:
            continue

        bundle = _streamline_subbundle(streamlines, indices)
        if len(bundle) == 0:
            continue

        lengths: List[float] = []
        tortuosities: List[float] = []
        directions: List[np.ndarray] = []

        for s in bundle:
            if len(s) < 2:
                continue
            diffs = np.diff(s, axis=0)
            length = float(np.sum(np.linalg.norm(diffs, axis=1)))
            end_to_end = float(np.linalg.norm(s[-1] - s[0]))
            tortuosity = (length / end_to_end) if end_to_end > 0 else 1.0
            vec = np.abs(s[-1] - s[0])
            n = float(np.linalg.norm(vec))
            norm_vec = (vec / n) if n > 0 else np.zeros(3)
            lengths.append(length)
            tortuosities.append(tortuosity)
            directions.append(np.asarray(norm_vec, dtype=float))

        if not lengths:
            continue

        mean_dir = np.mean(np.stack(directions), axis=0)
        results.append({
            "Source": CYTO_NAMES_FULL[node_i],
            "Target": CYTO_NAMES_FULL[node_j],
            "Streamline_Count": int(len(bundle)),
            "Mean_Length_mm": round(float(np.mean(lengths)), 2),
            "Mean_Tortuosity": round(float(np.mean(tortuosities)), 3),
            "Dir_X_LR_axis": round(float(mean_dir[0]), 3),
            "Dir_Y_AP_axis": round(float(mean_dir[1]), 3),
            "Dir_Z_IS_axis": round(float(mean_dir[2]), 3),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["Source", "Target"]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    if log_cb:
        if not df.empty:
            log_cb("--- Top 5 longest connections ---")
            top = df.sort_values("Mean_Length_mm", ascending=False).head(5)
            log_cb(
                top[
                    [
                        "Source",
                        "Target",
                        "Mean_Length_mm",
                        "Mean_Tortuosity",
                    ]
                ].to_string(index=False)
            )
        log_cb(f"Saved geometric tract data to {output_csv}")
    return df
