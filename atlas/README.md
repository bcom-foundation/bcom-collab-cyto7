# cyto7 atlas files

TODO(DOI): Zenodo DOI for this atlas release.

Cite the paper and the data DOI together; see the repository `README.md`.

## What each file is

### `fsaverage/` (164k vertices per hemisphere, FreeSurfer `fsaverage`)

| file | contents |
|---|---|
| `pial.{lh,rh}.cyto7.v9.annot` | **the released map.** Seven labels, 1 to 7, allocortex to koniocortex, plus `unknown` on the medial wall |
| `support/pial.{lh,rh}.cyto7.support*.shape.gii` | per-vertex support and its components (anatomical prior, topology, geometry, data overlay) |
| `support/pial.{lh,rh}.cyto7.support_categorical.annot` | the support map binned into categories |
| `crossed/` | the crossed parcellation: a classical atlas intersected with cyto7. See its own `README_crossed.md` |

### `fs_LR_32k/` (32,492 vertices per hemisphere, HCP fs_LR)

Nearest-neighbour resamples of the same map, for HCP-style pipelines. Use directly;
no further resampling is needed.

### `provenance/`

The as-painted map, every superseded version, the per-vertex change logs between
versions, and a `PROVENANCE_*.txt` for each version stating what changed and why.
This is what lets a reader audit how the released map came to be.

## Mapping the atlas onto your own subject

### FreeSurfer

For any subject with a completed `recon-all`, the standard spherical registration
carries the annotation across:

```bash
export SUBJECTS_DIR=/path/to/your/subjects
for hemi in lh rh; do
  mri_surf2surf     --srcsubject fsaverage     --trgsubject YOUR_SUBJECT     --hemi $hemi     --sval-annot fsaverage/pial.$hemi.cyto7.v9.annot     --tval $SUBJECTS_DIR/YOUR_SUBJECT/label/$hemi.cyto7.v9.annot
done
```

Then the usual tools work unchanged, for example per-type cortical thickness:

```bash
mri_segstats --annot YOUR_SUBJECT lh cyto7.v9   --i $SUBJECTS_DIR/YOUR_SUBJECT/surf/lh.thickness --sum lh.cyto7.thickness.txt
```

### A volumetric map for one subject

The atlas is released on the surface only. To get a volumetric parcellation —
for head modelling, an EEG/MEG or tES source space, or volume-space fMRI —
build one per subject, after the `mri_surf2surf` step above:

```bash
mri_aparc2aseg --s YOUR_SUBJECT --annot cyto7.v9 --o cyto7+aseg.mgz
```

This writes a volume in which cortical voxels carry the cyto7 type and
subcortical structures come from the standard `aseg`.

**Read this before using the output.** `mri_aparc2aseg` adds the annotation's
`structure_id` to 1000 (left) or 2000 (right). The released colour tables store
a *packed RGB value* in that field, because that is what makes the `.annot`
files round-trip correctly in other viewers. So the command above does **not**
give labels 1001–1007 and 2001–2007; it gives 1000-plus-packed-RGB, which is
not a usable label map.

Two ways to clean labels, both in `code/pipeline/compute_tractography_connectivity.py`:

- `make_freesurfer_friendly_ctab()` rewrites the colour table so `structure_id`
  equals the cyto7 code, giving 1001–1007 and 2001–2007 directly; or
- `remap_aparc2aseg_volume()` post-processes the volume the naive command
  produces into labels 1–7, with 0 elsewhere.

This is the path the tractography analysis uses (Annex G of the paper), so it is
exercised code rather than an untested recipe.

**No fsaverage volume is shipped**, deliberately. fsaverage's volume is a
template average, so filling its ribbon with cyto7 labels produces something
that looks usable but is not appropriate for individual analysis; the
per-subject recipe above gives a better product for free. A volumetric release
in MNI152 space is under consideration for a future version. It needs its own
provenance record and a check that the seven types and the map's topology
survive the ribbon fill and the nonlinear warp, so it is not included here.

### Connectome Workbench (fs_LR 32k)

The 32k label GIFTIs are already in fs_LR 32k, so subject data in that space needs
no resampling. To build a CIFTI label file and parcellate:

```bash
wb_command -cifti-create-label cyto7.v9.32k_fs_LR.dlabel.nii   -left-label  fs_LR_32k/pial.lh.cyto7.32k_fs_LR.label.gii   -right-label fs_LR_32k/pial.rh.cyto7.32k_fs_LR.label.gii

wb_command -cifti-parcellate YOUR_DATA.dtseries.nii cyto7.v9.32k_fs_LR.dlabel.nii   COLUMN YOUR_DATA.cyto7.ptseries.nii
```

To move the map to a different mesh density, use `wb_command -label-resample` with
`ADAP_BARY_AREA` and the appropriate sphere and midthickness surfaces.

## Caveats

- The map is hand-painted by a single rater and expert-revised. The support map
  quantifies uncertainty in the template boundary, not disagreement between raters.
- A node inherits the local boundary uncertainty of the map. See the paper's
  limitations section.
- The crossed parcellation's retention threshold is evaluated across both
  hemispheres, so a node may fall below it in one hemisphere. Per-hemisphere vertex
  counts are in the node tables.
