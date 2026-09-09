"""Acquire the von Economo-Koskinas atlas and build the cortical-type lookup.

This script implements section 2 of ``COMPARE_cyto7_vs_voneconomo.md``. It does
two things:

1. **Fetch the von Economo-Koskinas areal atlas** (Scholtens et al., 2018) via
   :func:`netneurotools.datasets.fetch_voneconomo` and copy the classifier
   (``.gcs``), colour table (``.ctab``) and area ``info.csv`` into
   ``resources/voneconomo/`` for provenance.

   The areal *annotation* on the standard ``fsaverage`` mesh is produced by
   ``scripts/make_economo_annot_fsaverage.sh`` (FreeSurfer ``mris_ca_label``),
   which writes ``resources/voneconomo/{lh,rh}.economo.annot``. That step needs
   FreeSurfer; on this Windows host it is run through WSL. This Python script
   only *checks* for those annotations and prints the command to regenerate
   them if missing -- it does not require FreeSurfer itself.

2. **Materialise the von-Economo-area -> Garcia-Cabezas 6-type lookup** as
   ``resources/voneconomo/von_economo_cortical_types.csv``. The assignment is
   taken from Garcia-Cabezas, Hacker & Zikopoulos (2020), *A Protocol for
   Cortical Type Analysis ... Applied on ... the Atlas of Von Economo and
   Koskinas* (Front. Neuroanat. 14:576015), Tables 4-7.

   The Scholtens MRI atlas uses the 43 top-level von Economo area letters
   (FA, FB, ... TG); Garcia-Cabezas subdivide several of these into sub-areas
   with their own types. The *headline* type for each Scholtens area is the
   type of the Garcia-Cabezas row bearing that exact von Economo letter (the
   principal/eponymous sub-area). Where the area's sub-areas span more than one
   type we additionally record the full spread and set ``ambiguous=TRUE`` (a
   mean of the sub-area type scores, Agranular=1 ... Koniocortex=6, is reported
   for context but does NOT override the principal type, because the sub-areas
   are unweighted here and small opercular/limbic strips would skew it).

   These rows -- and the ambiguous ones in particular -- must be checked by the
   human authors against the histology and the Garcia-Cabezas protocol before
   the comparison is quoted; the whole table is written with ``verified=FALSE``.

Run::

    conda activate cyto7
    python scripts/fetch_voneconomo_atlas.py            # fetch + build lookup
    python scripts/fetch_voneconomo_atlas.py --help

See ``COMPARE_cyto7_vs_voneconomo.md`` sections 2a/2b and 5.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import pandas as pd

from cyto7_surface_io import REPO_ROOT

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

VONECONOMO_DIR: Path = cfg.data_dir() / "voneconomo"

#: Garcia-Cabezas cortical types in ascending laminar differentiation, with the
#: numeric score used by Garcia-Cabezas et al. (2020) for averaging multi-type
#: areas (Agranular=1 ... Koniocortex=6) and the matching cyto7 integer code
#: (§1 of the task: agranular->2 ... koniocortex->7).
TYPE_ORDER: list[str] = [
    "agranular",
    "dysgranular",
    "eulaminate I",
    "eulaminate II",
    "eulaminate III",
    "koniocortex",
]
#: Garcia-Cabezas score (1-based) for each type.
TYPE_SCORE: dict[str, int] = {t: i + 1 for i, t in enumerate(TYPE_ORDER)}
#: cyto7 integer code for each type (agranular=2 ... koniocortex=7).
TYPE_TO_CYTO7: dict[str, int] = {t: i + 2 for i, t in enumerate(TYPE_ORDER)}
#: Short tokens used in the assignment table below.
TOKEN_TO_TYPE: dict[str, str] = {
    "Ag": "agranular",
    "Dys": "dysgranular",
    "Eu-I": "eulaminate I",
    "Eu-II": "eulaminate II",
    "Eu-III": "eulaminate III",
    "Konio": "koniocortex",
}

# --------------------------------------------------------------------------- #
# The Garcia-Cabezas (2020) assignment, keyed by Scholtens area acronym.
# --------------------------------------------------------------------------- #
# For each Scholtens (top-level) von Economo area:
#   "principal" = the cortical type of the Garcia-Cabezas row bearing that exact
#                 von Economo letter (the area's principal/eponymous sub-area).
#                 This is the headline assignment (-> cyto7_code). ``None`` marks
#                 limbic/periallocortical areas Garcia-Cabezas give no
#                 isocortical 6-type -- left blank and excluded.
#   "spread"    = all Garcia-Cabezas type(s) found across this area's sub-areas
#                 (Tables 4-7), including the principal. Used only to flag
#                 ``ambiguous`` (spread spans >1 type) and report a mean score.
#   "src"/"note"= provenance and rationale.
# "Dys-Ag" border rows are entered with both component tokens in the spread.
GC_ASSIGNMENT: dict[str, dict] = {
    # ---- Frontal: motor / premotor (Table 4) ----
    "FA":      {"principal": "Eu-III", "spread": ["Eu-III"], "src": "GC2020 Table 4",
                "note": "Precentral / primary motor (FA, FAg, FAop): eulaminate III."},
    "FB":      {"principal": "Eu-II", "spread": ["Eu-II", "Eu-I"], "src": "GC2020 Table 4",
                "note": "Agranular frontal premotor FB=Eu-II; opercular FBop/FB(C)op/FBC=Eu-I."},
    "FC":      {"principal": "Eu-I", "spread": ["Eu-I", "Dys"], "src": "GC2020 Table 4",
                "note": "Intermediate frontal FC=Eu-I; intermedio-limbic FCL=Dys."},
    "FCBm":    {"principal": "Eu-II", "spread": ["Eu-II"], "src": "GC2020 Table 4",
                "note": "Broca's area (magnocellular agranular intermediate frontal)."},
    # ---- Frontal: prefrontal (Table 5) ----
    "FD":      {"principal": "Eu-II", "spread": ["Eu-II", "Eu-I"], "src": "GC2020 Table 5",
                "note": "Granular frontal FD/FDC/FDm=Eu-II; FDop/FDL=Eu-I."},
    "FDdelta": {"principal": "Eu-I", "spread": ["Eu-I"], "src": "GC2020 Table 5",
                "note": "Middle granular frontal area (FD-delta / FD-Delta)."},
    "FDT":     {"principal": "Eu-III", "spread": ["Eu-III"], "src": "GC2020 Table 5",
                "note": "Triangular frontal area (FD-Gamma)."},
    "FE":      {"principal": "Eu-II", "spread": ["Eu-II", "Eu-I"], "src": "GC2020 Table 5",
                "note": "Frontopolar FE=Eu-II; limbic frontopolar FEL=Eu-I."},
    "FF":      {"principal": "Eu-I", "spread": ["Eu-I", "Dys"], "src": "GC2020 Table 5",
                "note": "Orbital: granular FFphi/pretriangular FFpsi=Eu-I; agranular FFalpha=Dys."},
    "FG":      {"principal": "Eu-I", "spread": ["Eu-I"], "src": "GC2020 Table 5",
                "note": "Area of straight gyrus (FG, FGi)."},
    "FH":      {"principal": "Eu-I", "spread": ["Eu-I", "Dys"], "src": "GC2020 Table 5",
                "note": "Prefrontal FH=Eu-I; paralimbic/limbic prefrontal FHL'/FHL=Dys."},
    "FJK":     {"principal": "Ag", "spread": ["Ag", "Dys"], "src": "GC2020 Table 5",
                "note": "Frontoinsular FI=Ag; frontal piriform FK=Dys-Ag (periallocortex)."},
    "FLMN":    {"principal": "Ag", "spread": ["Ag", "Dys"], "src": "GC2020 Table 5",
                "note": "Paraolfactory/geniculate (FL2/FL3/FM/FMi=Ag, FL1=Dys); olfactory periallocortex."},
    # ---- Limbic 'H' temporal areas: periallocortex, no isocortical 6-type ----
    "HA":      {"principal": None, "spread": [], "src": "GC2020 (n/a)",
                "note": "Uncinate area (temporopolar limbic periallocortex); no isocortical type in GC -> excluded."},
    "HB":      {"principal": None, "spread": [], "src": "GC2020 (n/a)",
                "note": "Parauncinate area (limbic periallocortex); no isocortical type -> excluded."},
    "HC":      {"principal": None, "spread": [], "src": "GC2020 (n/a)",
                "note": "Rhinal area limitans (borders allocortex/entorhinal); no isocortical type -> excluded."},
    # ---- Insula (Table 6) ----
    "IA":      {"principal": "Dys", "spread": ["Dys", "Eu-I"], "src": "GC2020 Table 6",
                "note": "Precentral insula IA1/IA2=Dys; transitional IA2(B)=Eu-I."},
    "IB":      {"principal": "Eu-I", "spread": ["Eu-I"], "src": "GC2020 Table 6",
                "note": "Postcentral insula (IB, IBT)."},
    # ---- Cingulate / limbic lobe (Table 6) ----
    "LA1":     {"principal": "Ag", "spread": ["Ag"], "src": "GC2020 Table 6",
                "note": "Precingulate agranular anterior limbic area."},
    "LA2":     {"principal": "Ag", "spread": ["Ag"], "src": "GC2020 Table 6",
                "note": "Anterior cingulate agranular anterior limbic area."},
    "LC1":     {"principal": "Eu-I", "spread": ["Eu-I"], "src": "GC2020 Table 6",
                "note": "Dorsal posterior cingulate area."},
    "LC2":     {"principal": "Dys", "spread": ["Dys"], "src": "GC2020 Table 6",
                "note": "Ventral posterior cingulate area."},
    "LC3":     {"principal": "Ag", "spread": ["Ag"], "src": "GC2020 Table 6",
                "note": "Posterior cingulate area limitans."},
    "LD":      {"principal": "Ag", "spread": ["Ag"], "src": "GC2020 Table 6",
                "note": "Agranular retrosplenial area."},
    "LE":      {"principal": "Dys", "spread": ["Dys", "Ag"], "src": "GC2020 Table 6",
                "note": "Retrosplenial granulosa: superior LE1=Dys, inferior LE2=Ag."},
    # ---- Occipital (Table 7) ----
    "OA":      {"principal": "Eu-III", "spread": ["Eu-III"], "src": "GC2020 Table 7",
                "note": "Peristriate (OA1/OA2/OAm), Brodmann 19."},
    "OB":      {"principal": "Eu-III", "spread": ["Eu-III"], "src": "GC2020 Table 7",
                "note": "Parastriate area (V2 / Brodmann 18)."},
    "OC":      {"principal": "Konio", "spread": ["Konio"], "src": "GC2020 Table 7",
                "note": "Striate area / V1 (Brodmann 17): koniocortex."},
    # ---- Parietal (Table 7) ----
    "PA":      {"principal": "Eu-II", "spread": ["Eu-II", "Eu-III"], "src": "GC2020 Table 7",
                "note": "Giant pyramidal post/para-central area (PA1/PA2)."},
    "PB":      {"principal": "Konio", "spread": ["Konio"], "src": "GC2020 Table 7",
                "note": "Oral post-central area (S1 proper, PB1/PB2): koniocortex."},
    "PC":      {"principal": "Eu-II", "spread": ["Eu-II"], "src": "GC2020 Table 7",
                "note": "Intermediate post-central area (PC, PCgamma)."},
    "PD":      {"principal": "Eu-II", "spread": ["Eu-II"], "src": "GC2020 Table 7",
                "note": "Caudal post-central / superior parietal transition (PD, P(D)E)."},
    "PE":      {"principal": "Eu-II", "spread": ["Eu-II", "Eu-III"], "src": "GC2020 Table 7",
                "note": "Superior parietal (PEm/PEp=Eu-II, PEgamma=Eu-III)."},
    "PF":      {"principal": "Eu-II", "spread": ["Eu-II", "Eu-I"], "src": "GC2020 Table 7",
                "note": "Supramarginal / inferior parietal (PF=Eu-II; PF1/PFop/PFcm=Eu-I)."},
    "PG":      {"principal": "Eu-I", "spread": ["Eu-I"], "src": "GC2020 Table 7",
                "note": "Angular area."},
    "PH":      {"principal": "Eu-I", "spread": ["Eu-I"], "src": "GC2020 Table 7",
                "note": "Basal (temporooccipital) parietal area (PHP/PHT/PHO)."},
    # ---- Temporal (Table 6) ----
    "TA":      {"principal": "Eu-II", "spread": ["Eu-II", "Eu-III", "Eu-I"], "src": "GC2020 Table 6",
                "note": "Superior temporal (TA1=Eu-II/Eu-III, TA2=Eu-I); auditory association."},
    "TB":      {"principal": "Eu-III", "spread": ["Eu-III"], "src": "GC2020 Table 6",
                "note": "Magnocellular supratemporal area simplex."},
    "TC":      {"principal": "Konio", "spread": ["Konio"], "src": "GC2020 Table 6",
                "note": "Supratemporal area granulosa (primary auditory koniocortex)."},
    "TD":      {"principal": "Konio", "spread": ["Konio"], "src": "GC2020 Table 6",
                "note": "Intercalated supratemporal area: koniocortex."},
    "TE":      {"principal": "Eu-II", "spread": ["Eu-III", "Eu-I"], "src": "GC2020 Table 6",
                "note": "Middle (TE1=Eu-III) and inferior (TE2=Eu-I) temporal proper; spans types -> Eu-II midpoint."},
    "TF":      {"principal": "Eu-I", "spread": ["Eu-I"], "src": "GC2020 Table 6",
                "note": "Fusiform area."},
    "TG":      {"principal": "Dys", "spread": ["Dys", "Ag"], "src": "GC2020 Table 6",
                "note": "Temporopolar TG=Dys; agranular temporopolar TGa=Ag."},
}


# --------------------------------------------------------------------------- #
# Atlas fetch
# --------------------------------------------------------------------------- #


def fetch_atlas_files(force: bool = False) -> None:
    """Fetch the Scholtens atlas via netneurotools and copy into resources.

    Copies the ``.gcs`` classifier, ``.ctab`` colour table, the area
    ``info.csv`` and the dataset ``LICENSE`` into ``resources/voneconomo/``.
    """
    try:
        from netneurotools import datasets as nnt_datasets
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "netneurotools is required to fetch the von Economo atlas. "
            "Install it into the cyto7 env: `pip install netneurotools`."
        ) from exc

    VONECONOMO_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching von Economo-Koskinas atlas (Scholtens et al., 2018)...")
    bunch = nnt_datasets.fetch_voneconomo(force=force, verbose=1)

    src_dir = Path(bunch["info"]).parent
    copied = []
    for pattern in ("*.gcs", "*.ctab", "*info*.csv", "LICENSE"):
        for src in src_dir.glob(pattern):
            dst = VONECONOMO_DIR / src.name
            if force or not dst.exists():
                shutil.copy2(src, dst)
            copied.append(dst.name)
    print(f"  copied to {VONECONOMO_DIR}: {sorted(set(copied))}")


def check_annotations() -> bool:
    """Check that the fsaverage economo annotations exist; instruct if not.

    Returns ``True`` if both ``{lh,rh}.economo.annot`` are present.
    """
    lh = VONECONOMO_DIR / "lh.economo.annot"
    rh = VONECONOMO_DIR / "rh.economo.annot"
    if lh.exists() and rh.exists():
        print(f"  von Economo annotations present: {lh.name}, {rh.name}")
        return True
    print(
        "\n  von Economo fsaverage annotations are MISSING. Generate them with "
        "FreeSurfer's mris_ca_label (the primary route, exact fsaverage mesh):\n"
        "    wsl.exe -d Ubuntu-20.04 bash "
        "scripts/make_economo_annot_fsaverage.sh\n"
        "  (requires FreeSurfer; on this host it lives in WSL Ubuntu-20.04).\n"
    )
    return False


# --------------------------------------------------------------------------- #
# Lookup CSV
# --------------------------------------------------------------------------- #


def _mean_score(spread: list[str]) -> float | None:
    """Mean Garcia-Cabezas type score (Ag=1 ... Konio=6) over a sub-area spread."""
    if not spread:
        return None
    scores = [TYPE_SCORE[TOKEN_TO_TYPE[t]] for t in spread]
    return sum(scores) / len(scores)


def build_lookup_csv(info_csv: Path | None = None) -> pd.DataFrame:
    """Build and write ``von_economo_cortical_types.csv``.

    Joins the Scholtens area ``info.csv`` (area id + acronym + name) with the
    Garcia-Cabezas assignment above. Prints any unmatched / excluded areas.
    """
    if info_csv is None:
        info_csv = next(VONECONOMO_DIR.glob("*info*.csv"))
    info = pd.read_csv(info_csv)
    # Cortical areas only: drop 'unknown' (id 0) and 'corpuscallosum' (id 1).
    cortical = info[~info["acronym"].isin(["unknown", "corpuscallosum"])].copy()

    rows = []
    unmatched, excluded = [], []
    for _, r in cortical.iterrows():
        acr = str(r["acronym"])
        spec = GC_ASSIGNMENT.get(acr)
        if spec is None:
            unmatched.append(acr)
            rep, spread, src, note = None, [], "", "NO Garcia-Cabezas match"
        else:
            rep = TOKEN_TO_TYPE[spec["principal"]] if spec["principal"] else None
            spread = spec["spread"]
            src, note = spec["src"], spec["note"]
            if rep is None:
                excluded.append(acr)
        spread_types = sorted({TOKEN_TO_TYPE[t] for t in spread}, key=TYPE_SCORE.get)
        ambiguous = len(spread_types) > 1
        mean = _mean_score(spread)
        rows.append(
            {
                "economo_area_id": int(r["id"]),
                "economo_area_name": r["region"] if pd.notna(r["region"]) else "",
                "economo_acronym": acr,
                "garcia_cabezas_type": rep if rep is not None else "",
                "cyto7_code": TYPE_TO_CYTO7[rep] if rep is not None else "",
                "gc_subarea_types": "|".join(spread),
                "gc_mean_score": round(mean, 3) if mean is not None else "",
                "ambiguous": "TRUE" if ambiguous else "FALSE",
                "source": src,
                "verified": "FALSE",
                "note": note,
            }
        )

    df = pd.DataFrame(rows).sort_values("economo_area_id").reset_index(drop=True)
    out = VONECONOMO_DIR / "von_economo_cortical_types.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote lookup -> {out}  ({len(df)} cortical areas)")

    # Reporting required by the task (§2b).
    n_assigned = (df["garcia_cabezas_type"] != "").sum()
    n_ambig = (df["ambiguous"] == "TRUE").sum()
    print(f"  assigned an isocortical 6-type: {n_assigned}/{len(df)}")
    print(f"  flagged ambiguous (sub-areas span >1 type): {n_ambig}")
    if unmatched:
        print(f"  !! UNMATCHED areas (no Garcia-Cabezas entry): {unmatched}")
    if excluded:
        print(
            "  Excluded (limbic/periallocortical, no isocortical 6-type, "
            f"garcia_cabezas_type blank): {excluded}"
        )
    print(
        "  NOTE: verified=FALSE for all rows. This lookup is a VERIFICATION "
        "CHECKPOINT -- the human authors must confirm it (especially the "
        "ambiguous rows) against the histology and the Garcia-Cabezas protocol "
        "before the comparison is quoted in the manuscript."
    )
    return df


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download the atlas and overwrite copied files / cache.",
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Do not fetch the atlas; only (re)build the lookup CSV from the "
        "info.csv already in resources/voneconomo/.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.skip_fetch:
        fetch_atlas_files(force=args.force)
    check_annotations()
    build_lookup_csv()
    print("Done.")


if __name__ == "__main__":
    main()
