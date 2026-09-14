# cyto7

**cyto7 is an open, vertex-level, seven-type cytoarchitectural atlas of the human
cerebral cortex**, hand-painted on the FreeSurfer `fsaverage` surface following the
García-Cabezas protocol and revised by the protocol's authors. Existing digital
cortical-type maps are area-level: they assign one type per classical area and
cannot represent a type boundary that runs through an area. cyto7 can, because it
is defined per vertex.

The seven types, in order of increasing laminar differentiation, are allocortex,
agranular, dysgranular, eulaminate I, eulaminate II, eulaminate III and
koniocortex.

## Citing this work

Please cite both the paper and the data.

**Paper.** Salvador R., Mercadal B., Castaldo F., García-Cabezas M. Á., Ruffini G.
*cyto7: an open, vertex-level cortical type atlas of the human cortex.*

- Published version: DOI pending.
- Preprint, openly downloadable: DOI pending.

**Data.** The atlas release is archived on Zenodo: DOI pending.

Both DOIs are minted after the paper is final and are shown as pending until then.
`CITATION.cff` carries the machine-readable form and will be updated in the same pass.

```bibtex
@article{cyto7,
  author  = {Salvador, Ricardo and Mercadal, Borja and Castaldo, Francesca and
             Garc{\'\i}a-Cabezas, Miguel {\'A}ngel and Ruffini, Giulio},
  title   = {cyto7: an open, vertex-level cortical type atlas of the
             human cortex},
  year    = {2026},
  doi     = {TODO},
  note    = {Atlas release archived on Zenodo, DOI pending}
}
```

## Getting the atlas in one paragraph

Everything you need is in `atlas/`. For FreeSurfer, take
`atlas/fsaverage/pial.lh.cyto7.v9.annot` and its right-hemisphere twin and map them
onto any subject with a completed `recon-all` using one `mri_surf2surf --sval-annot`
call, given in `atlas/README.md`. For fs_LR 32k pipelines (HCP-style), use the label
GIFTIs in `atlas/fs_LR_32k/` directly, no resampling needed. If your model wants
nodes rather than vertices, `atlas/fsaverage/crossed/` holds the crossed
parcellation, a classical atlas intersected with cyto7 so that a heterogeneous
parcel becomes several type-homogeneous nodes, with a per-node table.

The atlas files are standard FreeSurfer `.annot` and GIFTI. Reading them needs no
code from this repository and none of the dependencies below.

## What's in the repo

| path | what it holds |
|---|---|
| `atlas/fsaverage/` | the released 164k `.annot` for both hemispheres, the canonical product |
| `atlas/fsaverage/support/` | the per-vertex anatomical support score and its four components |
| `atlas/fsaverage/crossed/` | the crossed parcellation: classical atlases intersected with cyto7, plus per-node tables |
| `atlas/fs_LR_32k/` | label GIFTIs for HCP-style pipelines, no resampling needed |
| `atlas/provenance/` | how each map version was made, the change logs, and every superseded version |
| `results/tables/` | the aggregate tables behind every number in the paper |
| `results/tables/review_response/` | the follow-up analyses added after first submission |
| `results/subjects_used.txt` | the HCP subject identifiers used, values not included |
| `figures/` | the final manuscript and supplementary figures |
| `code/pipeline/` | the analyses, one script per step |
| `code/figures/` | the figure generators |
| `code/fetch/` | one fetcher per third-party dataset |
| `code/cyto7_config.py` | every machine-specific path, read from environment variables |
| `docs/` | methods notes and the reproduction guide |
| `dist/SHA256SUMS.txt` | a checksum for every file in the repository |
| `ANALYSIS_PROVENANCE.md` | every analysis mapped to its script, null model, seed and output file |
| `AUTHORS` | the copyright holders |

## Checking a number in the paper

`ANALYSIS_PROVENANCE.md` maps every analysis to the script that ran it, the null model and
seed it used, and the file under `results/` that holds its numbers. Start there if you want
to verify a specific figure or table. This is the path we recommend, and it needs nothing
installed: the numbers are in the tables.

## No third-party data is redistributed here, and why

This repository ships **the atlas, all of the code, and the aggregate result tables
that underpin the paper's numbers**. It ships **no third-party data**: no HCP
per-subject or group maps, no AHBA expression matrices, no ENIGMA effect-size maps,
no BigBrain profiles, no neuromaps downloads, no third-party PDFs.

Those datasets each carry their own data-use terms, and redistributing them here
would breach them. Instead, `code/fetch/` holds one fetch script per dataset, and
`docs/REPRODUCING.md` documents what to register for and where to put the result.
The aggregate tables in `results/` are the reported results of this study, not
redistributed data.

## What the code is, and what it will and will not do from a clone

The code is released so that every analysis can be inspected and audited: what was
computed, against which null, with which seed. It is the record of how the published
numbers were produced. Please read it that way rather than as a turnkey pipeline.

Being specific, because a vague claim here would be worse than none:

- **The atlas and the result tables are complete and verifiable from a clone.** Every
  file has a checksum in `dist/SHA256SUMS.txt`; `sha256sum -c dist/SHA256SUMS.txt`
  from the repository root verifies all of them.
- **Most analyses need third-party data that is not here.** They read it through
  `CYTO7_DATA_DIR` and stop with a message naming the variable and the dataset rather
  than a stack trace. That is expected behaviour, not a fault.
- **Some scripts do not yet run from a clone even where their inputs are present.**
  Of the modules under `code/`, ten still address the directory layout of the working
  repository this release was extracted from, rather than the layout here. They are
  listed with their corrected paths in `ANALYSIS_PROVENANCE.md`. We would rather say
  so than have you discover it.
- **We do not claim an end-to-end reproduction from a clone.** Reproducing the full
  set of figures requires the third-party datasets, Connectome Workbench, FreeSurfer,
  and for the MEG analyses FieldTrip and MATLAB. `docs/REPRODUCING.md` documents that
  path; it is a guide, not a guarantee.

## Requirements

`requirements.txt` pins every Python dependency, derived from what the scripts
actually import. `code/environment.yml` is the conda equivalent. Neither is needed to
use the atlas files themselves.

Non-Python tools, each needed only by specific steps, are configured through
environment variables documented in `code/.env.example`: Connectome Workbench,
FreeSurfer, and FieldTrip with MATLAB for the MEG source reconstruction.

## Licence

The split is deliberate.

- **Code** (`code/`): MIT, see `LICENSE-CODE`.
- **Atlas, result tables and figures** (`atlas/`, `results/`, `figures/`):
  CC BY 4.0, see `LICENSE-DATA`.

Copyright is held by The CYTO7 Authors; see `AUTHORS` for the individuals and
institutions.

The atlas is defined on the FreeSurfer `fsaverage` mesh. No FreeSurfer geometry is
redistributed here, only per-vertex labels on that mesh.

## Quick start

```bash
# verify what you received
sha256sum -c dist/SHA256SUMS.txt

# set up the analysis environment (not needed to use the atlas)
conda env create -f code/environment.yml
conda activate cyto7

# an analysis that needs no third-party data
python code/pipeline/audit_topology.py --version v9
```

Windows note: some paths in this repository are long enough that a clone into an
already-deep directory can fail with "Filename too long". If that happens, either
clone nearer the drive root or enable long paths with
`git config --global core.longpaths true`.
