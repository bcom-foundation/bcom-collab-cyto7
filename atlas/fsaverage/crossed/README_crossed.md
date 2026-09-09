# Crossed parcellation: classical atlas x cyto7 type

A node-based parcellation for region-based whole-brain models. Each classical parcel is
intersected with the released cyto7 v9 map, so a cytoarchitecturally heterogeneous parcel
becomes several type-homogeneous nodes: `superiortemporal_eul2`, `parahippocampal_agr`,
`pericalcarine_kon`.

Desikan-Killiany gives 101 nodes from 34 parcels; von Economo gives 115 from
43. Node names split on the **last** underscore into the base parcel, spelled exactly
as FreeSurfer spells it, and the cyto7 type abbreviation
(`allo agr dys eul1 eul2 eul3 kon` = ordinals 1 to 7).

## Files

| file | what it is |
|---|---|
| `?h.aparc_cyto7.annot` | FreeSurfer annotation on fsaverage 164k, embedded colour table |
| `?h.voneconomo_cyto7.annot` | same, von Economo base |
| `?h.*.32k_fs_LR.label.gii` | nearest-neighbour resample to 32k fs_LR, same convention as the cyto7 release |
| `aparc_cyto7.ctab` | FreeSurfer colour table, so `mri_annotation2label` works standalone |
| `*_nodes.tsv` | one row per node per hemisphere: id, name, hemi, base parcel, cyto7 ordinal and name, vertices, area, components, largest-component fraction, whether it absorbed residue |
| `?h.*_cc.annot` | **secondary** contiguity-split variant, components suffixed `_1`, `_2`, ... |
| `PROVENANCE_crossed.txt` | inputs with SHA-256, build rules, id scheme, counts |

The colour-table index is the same in both hemispheres. Node-table ids are `index` for lh
and `1000 + index` for rh, so a bilateral table has no collisions. Index 0 is `unknown`
(medial wall).

## Map it to your own subject

The whole point of shipping a `.annot` on fsaverage is that it transfers through the
standard spherical registration. For a subject with a completed `recon-all`:

```bash
export SUBJECTS_DIR=/path/to/your/subjects
for hemi in lh rh; do
  mri_surf2surf \
    --srcsubject fsaverage \
    --trgsubject YOUR_SUBJECT \
    --hemi $hemi \
    --sval-annot $hemi.aparc_cyto7.annot \
    --tval $SUBJECTS_DIR/YOUR_SUBJECT/label/$hemi.aparc_cyto7.annot
done
```

Then the usual tools work unchanged, for example per-node thickness:

```bash
mri_segstats --annot YOUR_SUBJECT lh aparc_cyto7 \
  --i $SUBJECTS_DIR/YOUR_SUBJECT/surf/lh.thickness --sum lh.aparc_cyto7.thickness.txt
```

This was verified, not assumed: see `PROVENANCE_crossed.txt` and the RR10 hand-back for the
round-trip test to a real `recon-all` subject, in which every node reappeared on the subject.

## 32k fs_LR (Workbench)

The `.32k_fs_LR.label.gii` files are already in fs_LR 32k, the space HCP-style subject data
use, so no resampling is needed. Parcellate a CIFTI directly:

```bash
wb_command -cifti-parcellate YOUR_DATA.dtseries.nii aparc_cyto7.32k_fs_LR.dlabel.nii \
  COLUMN YOUR_DATA.aparc_cyto7.ptseries.nii
```

To build that dlabel from the two hemisphere label files, or to move the parcellation to a
different mesh density:

```bash
wb_command -cifti-create-label aparc_cyto7.32k_fs_LR.dlabel.nii \
  -left-label lh.aparc_cyto7.32k_fs_LR.label.gii \
  -right-label rh.aparc_cyto7.32k_fs_LR.label.gii

wb_command -label-resample lh.aparc_cyto7.32k_fs_LR.label.gii \
  fs_LR.L.sphere.32k_fs_LR.surf.gii TARGET.L.sphere.surf.gii ADAP_BARY_AREA \
  lh.aparc_cyto7.TARGET.label.gii -area-surfs SRC.L.midthickness.surf.gii \
  TARGET.L.midthickness.surf.gii
```

## Read it in Python

```python
import nibabel as nib, pandas as pd
labels, ctab, names = nib.freesurfer.io.read_annot("lh.aparc_cyto7.annot")
nodes = pd.read_csv("aparc_cyto7_nodes.tsv", sep="\t").query("hemi == 'lh'").set_index("annot_index")
# labels[v] is the annot index of vertex v; nodes.loc[labels[v]] gives its name, base parcel and cyto7 type
```

## Caveats

- The node inventory is **bilateral**: a (parcel, type) cell is a node when its vertex count
  over both hemispheres exceeds 200. A node can therefore be much smaller in one
  hemisphere than the floor suggests, and two nodes are present in the left hemisphere only.
  Per-hemisphere sizes are in the node table; filter on `n_vertices` if your model needs a
  per-hemisphere minimum.
- Nodes that absorbed sub-threshold residue are not 100% type-pure. The node table flags
  them (`absorbed_residue`) and the hand-back reports the purity distribution.
- A node may span several patches. `n_components` and `frac_largest_component` are in the
  node table, and `?h.*_cc.annot` is the split variant if you need contiguous nodes.
- The cross moves no boundary. A node is only as good as its base parcel's definition, and
  inherits the local boundary uncertainty of the cyto7 map.
