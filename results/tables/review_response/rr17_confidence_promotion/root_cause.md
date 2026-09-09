# RR17 Step 1: why the promotion never happened

## The mechanism

`scripts/rerun_all_v9.py`, before this task, at what were lines 77-79:

```python
conf_cache = DER / "cache_conf_v9"
bcm.main(["--annot", ANNOT, "--out-derived", str(conf_cache),
          "--out-fig", str(FIG / "confidence")])
```

`build_confidence_map.py` writes its fourteen products to `--out-derived`. That argument
defaults to `DERIVED_DIR`, which is the top level of `resources/cyto7_derived/`, but the
version re-run scripts override it to point at that version's own cache directory. So the
builder never writes the top level when it is run the way the pipeline runs it, and **no
step anywhere copies a cache back up to the top level**.

The top-level files are therefore not stale in the sense of having been updated and then
gone out of date. They were **written once, by the last run that used the default
`--out-derived`**, which was the v3 build on 2026-06-24, and never touched again through
v4, v5, v6, v7, v8 and v9. All fourteen were byte-identical to `cache_conf_v3` when this
task began; `sha_before_after.csv` records that for each file individually.

## Why it went unnoticed for six atlas versions

This is the part that matters more than the mechanism, and it is the reason the fix has to
be in the pipeline rather than a one-off copy.

**No analysis in the repository reads the top-level copies.** Every reader is
version-parameterised onto `cache_conf_<version>`:

| reader | path it builds |
|---|---|
| `support_io.py` (was `confidence_io.py`) line 44 | `cache_conf_{version}` |
| `extend_structure_function.py` line 176 | `cache_conf_{version}` |
| `reviewer_response.py` line 145 | `cache_conf_{version}` |
| `render_support_myelin.py` line 9 | `cache_conf_v9`, hard-coded |
| `individual_myelin_variability.py`, `individual_myelin_signalcontrol.py`, `summarise_functional_features.py`, `plot_myelin_progression_regression.py` | via `anatomy_support_32k(version)` |

So the staleness was invisible to every figure, table and statistic in the paper. The one
consumer that did read the top level was the **public release**: RR12 copied those fourteen
files into `atlas/fsaverage/support/` in the bundle. The defect was silent inside the
repository and visible only to whoever downloaded the atlas.

That is also why it could not have corrupted a published number, and Step 3's acceptance
test confirms it did not.

## The fix

`rerun_all_v9.py` now calls `_promote_to_top_level(conf_cache)` immediately after the
build. It copies the fourteen products from the version's cache to the top level under the
new `support` names and returns what it wrote, so the run report can state what the release
actually contains. The comment at the call site records why the step exists, so it does not
get removed as redundant by someone reading only `--out-derived`.

The cache keeps the historical `confidence` spelling. That is deliberate: renaming files
inside `cache_conf_v3` through `cache_conf_v9` would break every older cache and is
forbidden by the task's guardrails. Only the promoted copies carry the new name.
