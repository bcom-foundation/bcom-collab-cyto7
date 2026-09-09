#!/usr/bin/env bash
# Apply the Scholtens et al. (2018) von Economo-Koskinas probabilistic GCS
# classifier to the standard FreeSurfer ``fsaverage`` subject, producing
# per-vertex areal annotations on the 163,842-vertex mesh -- an exact mesh
# match with the cyto7 labels (no resampling).
#
# This is the PRIMARY acquisition route from COMPARE_cyto7_vs_voneconomo.md
# (section 2a). It must run inside a Linux environment with FreeSurfer on the
# machine; on this Windows host it is invoked through WSL (Ubuntu-20.04), e.g.:
#
#     wsl.exe -d Ubuntu-20.04 bash /mnt/e/.../scripts/make_economo_annot_fsaverage.sh
#
# Inputs (already checked into the repo by fetch_voneconomo_atlas.py):
#   resources/voneconomo/atl-vonEconomoKoskinas_hemi-{L,R}_probabilistic.gcs
#   resources/voneconomo/atl-vonEconomoKoskinas_hemi-{L,R}_probabilistic.ctab
# Outputs:
#   resources/voneconomo/{lh,rh}.economo.annot   (on standard fsaverage)
set -euo pipefail

FREESURFER_HOME="${FREESURFER_HOME:-/usr/local/freesurfer}"
# Use a clean PATH: the Windows PATH leaks into WSL and its "(x86)" entries
# break shell parsing, so we deliberately do not inherit it here.
export FREESURFER_HOME
export PATH="$FREESURFER_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export SUBJECTS_DIR="$FREESURFER_HOME/subjects"

# Repo root: directory containing this script's parent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_DIR="$SCRIPT_DIR/../resources/voneconomo"
cd "$VE_DIR"

echo "FREESURFER_HOME=$FREESURFER_HOME"
echo "SUBJECTS_DIR=$SUBJECTS_DIR"
"$FREESURFER_HOME/bin/mris_ca_label" --version 2>/dev/null || true

for hemi in lh rh; do
    H=$([ "$hemi" = "lh" ] && echo L || echo R)
    gcs="$VE_DIR/atl-vonEconomoKoskinas_hemi-${H}_probabilistic.gcs"
    ctab="$VE_DIR/atl-vonEconomoKoskinas_hemi-${H}_probabilistic.ctab"
    out="$VE_DIR/${hemi}.economo.annot"
    echo "=== ${hemi}: applying ${gcs} -> ${out} ==="
    "$FREESURFER_HOME/bin/mris_ca_label" -t "$ctab" \
        fsaverage "$hemi" \
        "$SUBJECTS_DIR/fsaverage/surf/${hemi}.sphere.reg" \
        "$gcs" "$out"
    echo "    wrote $VE_DIR/$out"
done
echo "Done."
