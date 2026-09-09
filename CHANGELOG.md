# Changelog

Notable changes to the released atlas.

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

### Fixed

- The per-vertex anatomical support products shipped in `atlas/fsaverage/support/` were the v3 build. They had never been refreshed when the atlas moved to v9, because the release pipeline wrote each version into its own cache directory and no step promoted the current one to the published location. All fourteen files are now the v9 build. The v9 per-type medians they reproduce are unchanged from those already reported, since every analysis read the cache directly; only the published copies were stale.

### Changed

- The score is renamed from "support" to **anatomical support score** (short form `support`), to stop it being read as a statistical support measure. Files are now `pial.<hemi>.cyto7.support*`, and the results table moved from `results/tables/support/` to `results/tables/support/`.
