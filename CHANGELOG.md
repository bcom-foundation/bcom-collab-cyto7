# Changelog

Notable changes to the released atlas.

## Two version numbers, and which is which

This project carries two independent version axes, and they will both appear as "v1".

- **Map versions — `v1` … `v9`.** These number the *atlas itself*: successive states of the
  hand-painted map. `v9` is the released map. `v1`–`v8` are superseded and ship under
  `atlas/provenance/` as an audit trail, not as products. This changelog is organised by
  these, and they are what `PROVENANCE_v*.txt` and the per-vertex change logs refer to.
- **Repository releases — `v1.0.0`, …** These number the *published snapshot*: a Git tag and
  the corresponding Zenodo deposition. `v1.0.0` is the first public release and contains map
  v9. A later release might ship the same map with corrected documentation, or a new map
  without changing the major version.

So "v1" alone is ambiguous: `atlas/provenance/` v1 is the earliest painting, while release
v1.0.0 is the current public snapshot. Say "map v9" or "release v1.0.0" and the ambiguity
disappears.

The Zenodo deposition is **not** created from a GitHub release webhook — it is a manual
deposition. Enabling the GitHub–Zenodo integration and cutting a release would mint a second,
separate DOI for the same work. Do not enable it; see `README.md` for the two DOIs in use.

## v9 (current release)

The released map. v9 is v8 plus a 543-vertex left/right entorhinal evening-up; the
right hemisphere is unchanged from v8. See `atlas/provenance/PROVENANCE_v9.txt`.

Shipped alongside the map in this release:

- the crossed parcellation (classical atlas intersected with cyto7) with per-node
  tables, colour tables and a contiguity-split variant
- the support map and its components
- the full per-vertex change log from the as-painted map to v9

## v1 to v8

Superseded working versions, shipped under `atlas/provenance/` so that the path
from the initial hand-painting to the released map can be audited. Each has a
`PROVENANCE_*.txt` stating what changed and why. They are provenance, not products:
use v9.

## Unreleased

### Added

- The bioRxiv preprint DOI, [10.64898/2026.10.01.755866](https://doi.org/10.64898/2026.10.01.755866), in `README.md` (text and BibTeX), `CITATION.cff` and `.zenodo.json`. Documentation only: no atlas file changed, so the Zenodo snapshot of release v1.0.0 still holds the map shipped here.

### Fixed

- The per-vertex anatomical support products shipped in `atlas/fsaverage/support/` were the v3 build. They had never been refreshed when the atlas moved to v9, because the release pipeline wrote each version into its own cache directory and no step promoted the current one to the published location. All fourteen files are now the v9 build. The v9 per-type medians they reproduce are unchanged from those already reported, since every analysis read the cache directly; only the published copies were stale.

### Changed

- The score is renamed from "support" to **anatomical support score** (short form `support`), to stop it being read as a statistical support measure. Files are now `pial.<hemi>.cyto7.support*`, and the results table moved from `results/tables/support/` to `results/tables/support/`.
