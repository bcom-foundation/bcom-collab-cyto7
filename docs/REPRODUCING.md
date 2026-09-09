# Reproducing this work from an empty machine

## 1. Environment

```bash
conda env create -f code/environment.yml
conda activate cyto7
```

The environment pins an OpenBLAS build of numpy on purpose: the MKL build
hard-crashes inside `numpy.linalg` on some Windows CPUs during matplotlib's 3D axis
setup.

## 2. Configuration

Nothing needs configuring to use the atlas. To re-run analyses, copy
`code/.env.example` to `code/.env` and fill in what you have:

| variable | needed for |
|---|---|
| `CYTO7_DATA_DIR` | any analysis using third-party data (HCP, AHBA, ENIGMA, BigBrain) |
| `CYTO7_WORKBENCH_DIR` | fsaverage to fs_LR 32k resampling |
| `CYTO7_FREESURFER_HOME` | `mri_surf2surf`, `mri_aparc2aseg` |
| `CYTO7_FIELDTRIP_DIR` | MEG source reconstruction only |

Any script that needs one of these and cannot find it stops with a message naming
the variable and the dataset. It does not fail with a stack trace.

## 3. Using the code from your own script

`code/cyto7_config.py` puts `code/`, `code/pipeline/`, `code/figures/` and
`code/fetch/` on `sys.path` when it is imported, so import it first and the rest of
the pipeline is then importable by bare name:

```python
import cyto7_config as cfg          # must come first
import cyto7_surface_io as io
labels = io.resolve_target_map("v9", "fsaverage")   # {"L": ..., "R": ...}
```

The shipped scripts carry their own bootstrap, so `python code/pipeline/<name>.py`
works from a clone with nothing on `PYTHONPATH`.

## 3b. What runs with no third-party data at all

These depend only on files shipped in this repository:

```bash
python code/pipeline/audit_topology.py --version v9
python code/pipeline/rr10_crossed_parcellation.py --skip-roundtrip
```

## 4. What needs data you must obtain yourself

| dataset | how to obtain | fetch script |
|---|---|---|
| HCP S1200 (myelin, MEG) | register at db.humanconnectome.org, accept the Open Access Data Use Terms | `code/fetch/fetch_hcp.py` |
| AHBA microarray | downloaded by `abagen` on first use, no registration | `code/fetch/fetch_ahba.py` |
| neuromaps annotations | no registration | `code/fetch/fetch_neuromaps_features.py` |
| ENIGMA effect sizes | via the ENIGMA toolbox, no registration | `code/fetch/fetch_enigma.py` |
| BigBrain profiles | via BigBrainWarp | `code/fetch/fetch_bigbrain.py` |
| von Economo atlas | third-party classifiers | `code/fetch/fetch_voneconomo_atlas.py` |

Point `CYTO7_DATA_DIR` at wherever you put them; each fetch script prints the layout
it expects.

## 5. Order

Most analyses are independent of each other and depend only on the released map.
Where order matters it is stated in the script docstring.
