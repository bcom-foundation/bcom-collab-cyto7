# Provenance notes: two bookkeeping discrepancies in intermediate versions

**Neither affects the released v9 map.** Both concern the internal bookkeeping of intermediate
versions (v4 and v5) that ship here as provenance. The released map, the per-vertex change
logs and every number reported in the paper are unaffected. They are recorded here because
the audit found them and a provenance record that hides its own inconsistencies is worth less
than one that states them.

Both were found by the topology-repair audit; its full output is in
`results/tables/review_response/rr1_topology/`.

## 1. The v5 record reports a 29-vertex topology buffer that the change log does not contain

`PROVENANCE_v5.txt` reports, for the right hemisphere, `buffer 29`, referring to a
`topology_buffer` edit class applied by the strict-R1 pass.

`v4_to_v5_changelog.csv` contains no `topology_buffer` rows at all. Its three reason classes
account for every changed vertex exactly, with no room for a fourth:

| hemisphere | cingulate_gradient | ring_closure | boundary_smooth | sum | v5 record reports |
|---|---:|---:|---:|---:|---:|
| lh | 1,719 | 636 | 137 | **2,492** | changed 2,492, buffer 0 |
| rh | 1,357 | 688 | 147 | **2,192** | changed 2,192, buffer 29 |

4,684 rows in total, matching 2,492 + 2,192.

Two readings are consistent with the evidence, and the record does not distinguish them: the
29 buffer vertices may have been relabelled again by the later `boundary_smooth` pass and so
lost their original reason, or the count may be bookkeeping that was never written to the log.
Either way the change log is complete with respect to which vertices moved, which is what
matters for auditing the map; only the attribution of 29 of them is uncertain.

## 2. The v4 strict-gradient buffer of 16 plus 9 vertices recovers as 14 plus 6

`PROVENANCE_v4.txt` reports a strict-gradient buffer of 16 vertices (lh) and 9 (rh), grown by
`converge_topology` to remove every remaining two-step edge.

`v3clean_to_v4_changelog.csv` records which vertices moved and their before and after labels
(1,712 lh, 1,471 rh, matching the record's totals) but carries no reason column, so the buffer
cannot be recovered from it by class. Recovering it instead by a revert-one test, which asks
which vertices R1 strictly requires, yields **14 lh and 6 rh**.

The five-vertex shortfall is explained rather than unexplained. The fragment cleanup described
in the same record ran **in place on v4** and moved 50 vertices (37 lh, 13 rh), absorbing every
belt-type connected component below 20 vertices into its majority adjacent type. Some buffer
vertices were among those overwritten, so their final label is no longer the one the buffer
set, and a test that reads final labels cannot see them.

## What this does and does not affect

- **The released v9 map: unaffected.** v9 is reached from v8, and neither discrepancy touches
  any v6-to-v9 step.
- **The per-vertex change logs: unaffected.** Every changed vertex is recorded in every log.
  What is uncertain in case 1 is the reason attached to 29 of them.
- **The topology-repair audit: unaffected.** The 284 `boundary_smooth` vertices are counted as
  topology there, so no vertex is lost from the total either way, and the audit reports its
  attribution as a bracket rather than a single number for exactly this reason.
- **Any number in the paper: unaffected.** Nothing reported depends on the internal edit-class
  bookkeeping of v4 or v5.

## 2026-09-02 — anatomical support products refreshed to v9, and renamed

The fourteen per-vertex products in `atlas/fsaverage/support/` were the **v3** build. They
had been byte-identical to `resources/cyto7_derived/cache_conf_v3/` in the analysis repo
since 2026-06-24 and were never refreshed as the atlas moved v4 → v9.

**Cause.** `rerun_all_v9.py` invokes the builder with `--out-derived <cache_conf_v9>`, so
each version's build lands in its own cache directory. No step ever promoted the current
version's cache to the top level of `resources/cyto7_derived/`, which is what the release
copies. It went unnoticed because **no analysis reads the top-level copies**: every reader
is version-parameterised onto `cache_conf_<version>`. Only the public release used them.
A promotion step is now part of `rerun_all_v9.py`, so the release location cannot fall
behind the atlas again.

**Effect on published numbers: none.** Every result was computed from `cache_conf_v9`
directly. The promoted files reproduce the released per-type medians in
`results/tables/support/support_summary.csv` exactly, for all seven types in both
hemispheres, and rebuilding the atlas-free variant of the score from the promoted
components reproduces the cache-built variant to zero difference at every vertex.

**Rename.** The score is now the **anatomical support score** (short form `support`), so it
is not read as a statistical support measure. Files are `pial.<hemi>.cyto7.support*`;
the results table moved to `results/tables/support/`. The `cache_conf_*` directories in the
analysis repo keep the historical `support` spelling so older caches stay readable, and
the code that reads them keeps that spelling deliberately.

Old and new SHA-256 for all fourteen files are recorded in the analysis repo at
`figures/v9/review_response/rr17_confidence_promotion/bundle_sha_before_after.csv`.
