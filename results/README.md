# Aggregate result tables

The tables that underpin the numbers reported in the paper. Everything here is a result of
this study, not redistributed third-party data.

- `tables/` mirrors the analysis directories of the working repository, one subdirectory per
  analysis, each with the hand-back document that records what was found.
- `report_*.md` are the per-analysis reports.
- `subjects_used.txt` lists the 200 Human Connectome Project subject identifiers used in the
  individual-generalisation analysis.

## What is here and what is not, on subject-level data

**Naming the subjects is fine; redistributing their derived data is not.** That line is why
`subjects_used.txt` ships and the following do not:

| kept out | why |
|---|---|
| `hcp_meg_manifest.tsv`, `hcp_meg_v2_manifest.tsv` | subject identifiers alongside per-file checksums of HCP archives |
| `persubject_rho.csv`, `persubject_type_medians.csv` | a subject identifier column beside that subject's derived values |
| `meg_*_persubject_rho.csv`, `partial_persubject_rho.csv` | the same, for the MEG analyses |

A bare list of which subjects were analysed is standard practice and is necessary for anyone
to reproduce the selection at all. A table pairing an identifier with values derived from that
subject's data is a redistribution of HCP-derived per-subject data, which the HCP Open Access
Data Use Terms govern, so those tables stay with the authors. Group-level and per-type
summaries derived from the same subjects are published here in full.

If you need the per-subject values, obtain HCP access yourself and re-run the pipeline: the
subject list here plus `code/pipeline/` reproduces them.
