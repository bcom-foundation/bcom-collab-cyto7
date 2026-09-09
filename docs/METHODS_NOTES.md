# Methods notes

Pointers into the paper, not a duplicate of it. Where a number is quoted here it is
to help you find the code that produced it.

## What lives where

| paper section | code |
|---|---|
| map construction and topology rules | `code/pipeline/apply_rule_v3.py`, `code/pipeline/audit_topology.py` |
| support map | `code/pipeline/build_support_map.py` |
| benchmark against von Economo | `code/pipeline/compare_cyto7_vs_voneconomo.py` |
| structure-function features | `code/pipeline/summarise_functional_features.py`, `code/pipeline/extend_structure_function.py` |
| MEG dynamics | `code/pipeline/meg_dynamics_v2*.py`, `code/pipeline/meg_subjectlevel.py` |
| tractography | `code/pipeline/compute_tractography_connectivity.py`, `code/figures/plot_tractography_analysis.py` |
| crossed parcellation | `code/pipeline/rr10_crossed_parcellation.py` |
| external validation | `code/pipeline/bigbrain_*.py`, `code/pipeline/fig9_predictions.py` |

## Review-response analyses

The `rr1` to `rr11` scripts are the analyses added in response to peer review. Their
outputs are the tables under `results/tables/`, and each carries a hand-back
document recording what was found. `code/pipeline/rr_common.py` holds the shared
machinery: the released-file integrity check, the spin-null loader, mesh geometry
helpers and the decision log.

## Statistical conventions

- Spatial null: Alexander-Bloch spin test, seed 0, N = 1000 rotations, on the 32k
  fs_LR mesh, unless a script states otherwise.
- Where a null changes the scale of the predictor, the test statistic is scale-free
  (a correlation, or a coefficient on a standardised predictor). See
  `results/tables/rr9_proximity/` for why this matters.
- FDR control is Benjamini-Hochberg within declared families; the families are
  enumerated in the outcome table under `results/tables/rr2_table/`.
