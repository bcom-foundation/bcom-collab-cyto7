#!/usr/bin/env python
"""SPEC_hcp_meg_dynamics_v2 driver — realistic forward (FieldTrip singleshell) + true INT.

Two phases:
  * PHASE 1 (beamform): per subject, checksum-gate -> unzip 3 rmegpreproc + anatomy
    (headmodel + sourcemodel_2d) -> call MATLAB `beamform_subject.m` (FieldTrip LCMV ->
    per-run PSD + ACF intrinsic-timescale maps) -> persist `sub-<id>_dynamics.mat` ->
    delete that subject's unpacked raw. tqdm over 89 subjects + a status file with
    done/total, elapsed, ETA, per-subject stage/pass-fail. Resumable (skips persisted).
  * PHASE 2 (analysis): load persisted PSDs/INT maps -> specparam (fixed 2-40 primary,
    knee 2-40, broad 1-100) + knee-timescale + aperiodic-corrected band power; INT (3
    defs); split-half/across-run reliability; validity gates (source alpha vs megalpha,
    INT vs megtimescale); align to cyto7 v9 (reuse meg_subjectlevel.build_4k_alignment)
    and test with the released spin + BH-FDR (dynamics family), allocortex-excluded;
    side-by-side with v1 single-sphere + released single-map values. Writes CSV, figures,
    report. Persisted PSD/map scratch is NOT deleted.

Constraints honored: server-only, one-subject raw at a time, checksum-gated, cyto7 v9,
released rotations (seed 0, n=1000), dynamics FDR family, released tables untouched.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import hashlib
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyto7_surface_io import REPO_ROOT  # noqa: E402

SERVER = cfg.data_dir("019-HCP Young MEG data")
VSC = SERVER / "_scratch" / "v2"
RAW = VSC / "raw"
PERSIST = VSC / "persist"           # kept at the end (per SPEC)
OUT_SF = cfg.results_dir("tables") / "structure_function"
RESULTS = REPO_ROOT / "results"
CACHE = cfg.data_dir() / "neuromaps_cache"
STATUS = RESULTS / "hcp_meg_v2_progress.txt"
RUNLOG = RESULTS / "hcp_meg_v2_run.log"

# Set CYTO7_MATLAB_BIN to your own MATLAB executable. The fallback is the default
# Windows install location for the version this analysis was run with, kept so the
# original run stays documented; it names no particular machine or user.
MATLAB = str(cfg.optional("CYTO7_MATLAB_BIN")
             or r"C:/Program Files/MATLAB/R2018a/bin/matlab.exe")
FT = cfg.fieldtrip_dir()
SCRIPTS = str(Path(__file__).resolve().parent)
QC_SUBJECTS = 3                     # keep qc source time series for the first N subjects
SEED = 0

ALL_BANDS = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30),
             "gamma1": (30, 60)}


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNLOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# subjects / integrity
# --------------------------------------------------------------------------- #
def list_subjects() -> list[str]:
    pre = {re.match(r"(\d+)_", p.name).group(1) for p in SERVER.glob("*_MEG_Restin_preproc.zip")}
    ana = {re.match(r"(\d+)_", p.name).group(1) for p in SERVER.glob("*_MEG_anatomy.zip")}
    return sorted(pre & ana)


def _md5(path: Path, chunk=16 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _expected(zp: Path):
    m5 = zp.with_suffix(zp.suffix + ".md5")
    if not m5.exists():
        return None
    m = re.match(r"([0-9a-fA-F]{32})", m5.read_text(errors="replace").strip())
    return m.group(1).lower() if m else None


def verify(subj: str, zp: Path) -> bool:
    marker = PERSIST / f"sub-{subj}.{zp.name}.md5ok"
    if marker.exists():
        return True
    exp = _expected(zp)
    if exp is None:
        marker.write_text("no_sidecar"); return True
    if _md5(zp) != exp:
        log(f"  [FAIL] {subj}: MD5 mismatch {zp.name}")
        return False
    marker.write_text(exp); return True


def persist_path(subj: str) -> Path:
    return PERSIST / f"sub-{subj}_dynamics.mat"


# --------------------------------------------------------------------------- #
# status file
# --------------------------------------------------------------------------- #
def write_status(done: int, total: int, t0: float, subj: str, stage: str,
                 npass: int, nfail: int):
    el = time.time() - t0
    rate = el / done if done else 0
    eta = rate * (total - done)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        f"HCP-MEG dynamics v2 (FieldTrip singleshell) — phase 1 (beamform)\n"
        f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"progress: {done}/{total} subjects done ({100*done/total:.1f}%)\n"
        f"pass: {npass}  fail: {nfail}\n"
        f"current: {subj}  stage: {stage}\n"
        f"elapsed: {el/3600:.2f} h   rate: {rate/60:.1f} min/subj   ETA: {eta/3600:.2f} h\n",
        encoding="utf-8")


# --------------------------------------------------------------------------- #
# PHASE 1 — beamform loop
# --------------------------------------------------------------------------- #
def unpack_subject(subj: str) -> dict | None:
    d = RAW / subj
    d.mkdir(parents=True, exist_ok=True)
    pre = SERVER / f"{subj}_MEG_Restin_preproc.zip"
    ana = SERVER / f"{subj}_MEG_anatomy.zip"
    with zipfile.ZipFile(pre) as zf:
        runs = sorted(n for n in zf.namelist()
                      if re.search(rf"{subj}_MEG_\d+-Restin_rmegpreproc\.mat$", n))
        for n in runs:
            (d / Path(n).name).write_bytes(zf.read(n))
    with zipfile.ZipFile(ana) as zf:
        for key in ("headmodel", "sourcemodel_2d"):
            nm = next((n for n in zf.namelist() if n.endswith(f"anatomy_{key}.mat")), None)
            if nm is None:
                log(f"  [FAIL] {subj}: anatomy missing {key}")
                return None
            (d / Path(nm).name).write_bytes(zf.read(nm))
    run_files = sorted(str(d / Path(n).name) for n in runs)
    return {"dir": d,
            "hm": str(d / f"{subj}_MEG_anatomy_headmodel.mat"),
            "sm": str(d / f"{subj}_MEG_anatomy_sourcemodel_2d.mat"),
            "runs": run_files}


def beamform_one(subj: str, paths: dict, qc: bool) -> bool:
    out = persist_path(subj)
    runs_csv = ";".join(paths["runs"])
    mlog = VSC / f"beamform_{subj}.log"
    cmd = [MATLAB, "-nosplash", "-nodesktop", "-wait", "-logfile", str(mlog),
           "-r", (f"addpath('{SCRIPTS}'); try; beamform_subject('{paths['hm']}','{paths['sm']}',"
                  f"'{runs_csv}','{str(out)}','{FT}','{1 if qc else 0}'); catch e; "
                  f"disp(getReport(e)); end; exit")]
    subprocess.run(cmd, check=False)
    return out.exists()


def phase1(subjects: list[str], t0: float):
    PERSIST.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    total = len(subjects)
    try:
        from tqdm import tqdm
        bar = tqdm(subjects, desc="beamform", unit="subj")
    except Exception:
        bar = subjects
    npass = nfail = done = 0
    for i, subj in enumerate(subjects):
        if persist_path(subj).exists():
            done += 1; npass += 1
            write_status(done, total, t0, subj, "cached", npass, nfail)
            if hasattr(bar, "update"):
                bar.update(1)
            continue
        write_status(done, total, t0, subj, "verify", npass, nfail)
        pre = SERVER / f"{subj}_MEG_Restin_preproc.zip"
        ana = SERVER / f"{subj}_MEG_anatomy.zip"
        if not (verify(subj, pre) and verify(subj, ana)):
            nfail += 1; log(f"[{i+1}/{total}] {subj}: checksum FAIL -> skip"); continue
        ts = time.time()
        log(f"[{i+1}/{total}] {subj}: unpack")
        write_status(done, total, t0, subj, "unpack", npass, nfail)
        paths = unpack_subject(subj)
        if paths is None:
            nfail += 1; continue
        write_status(done, total, t0, subj, "beamform(matlab)", npass, nfail)
        log(f"[{i+1}/{total}] {subj}: beamform ({len(paths['runs'])} runs)")
        ok = beamform_one(subj, paths, qc=(i < QC_SUBJECTS))
        import shutil
        shutil.rmtree(paths["dir"], ignore_errors=True)   # delete raw immediately
        if ok:
            npass += 1; done += 1
            log(f"[{i+1}/{total}] {subj}: DONE ({time.time()-ts:.0f}s) -> {persist_path(subj).name}")
        else:
            nfail += 1; log(f"[{i+1}/{total}] {subj}: beamform FAILED (see {VSC}/beamform_{subj}.log)")
        write_status(done, total, t0, subj, "done" if ok else "failed", npass, nfail)
        if hasattr(bar, "update"):
            bar.update(1)
    log(f"PHASE 1 complete: pass={npass} fail={nfail} / {total}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-spin", type=int, default=1000)
    args = ap.parse_args(argv)

    if not SERVER.exists():
        log(f"STOP: server not reachable {SERVER}"); return 2
    subjects = args.subjects or list_subjects()
    if args.limit:
        subjects = subjects[: args.limit]
    log(f"subjects: {len(subjects)}  (persist dir: {PERSIST})")

    t0 = time.time()
    if args.phase in ("1", "all"):
        phase1(subjects, t0)
    if args.phase in ("2", "all"):
        import meg_dynamics_v2_analyse as an   # phase-2 module (written separately)
        an.run(subjects, args.n_spin, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
