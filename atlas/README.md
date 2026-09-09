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
