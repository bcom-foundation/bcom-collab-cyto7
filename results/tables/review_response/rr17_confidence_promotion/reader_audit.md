# RR17 Step 2: every reader of a support (formerly confidence) file

## Summary

Nine scripts read the per-vertex score. **All nine read a `cache_conf_<version>` directory;
none reads the top-level copies.** Two questions the spec asked are answered below.

## The readers

| script | how it resolves the path | version it gets |
|---|---|---|
| `support_io.py` (was `confidence_io.py`) | `_DERIVED / f"cache_conf_{version}"` | caller's argument |
| `extend_structure_function.py` | `cyto7_derived / f"cache_conf_{version}"` | caller's argument |
| `reviewer_response.py` `support_components_32k` | `cyto7_derived / f"cache_conf_{version}"` | argument, default `"v3"` |
| `render_support_myelin.py` | literal `cache_conf_v9` | v9, hard-coded |
| `individual_myelin_variability.py` | `anatomy_support_32k(ver)` | caller's argument |
| `individual_myelin_signalcontrol.py` | `anatomy_support_32k(ver)` | caller's argument |
| `summarise_functional_features.py` | `anatomy_support_32k(ver)` | caller's argument |
| `plot_myelin_progression_regression.py` | `anatomy_support_32k(ver)` | caller's argument |
| `rr15_concordance.py` | builds into a scratch cache | its own |

The defaults of `"v3"` on `support_components_32k(version="v3")` and `load_core(dataset,
version="v3")` are the historical value from when v3 was current. They are not what the
released figures used: `rerun_all_v9.py` line 73 calls `rr.load_core(args.dataset, V)` with
`V = "v9"`, and `comps = support_components_32k(core["version"])` therefore resolves to
`cache_conf_v9` throughout. The defaults are a latent trap for anyone calling these
functions directly, but they did not affect any released result.

## `reviewer_response.py` line 58: legacy-intentional, or a bug?

```python
CONF_CACHE = REPO_ROOT / "resources" / "cyto7_derived" / "cache_conf_v3"
```

**Neither: it is dead code.** Grepping the whole of `scripts/` for `CONF_CACHE` returns
exactly this one line, its definition. The constant is never referenced anywhere, in this
module or any other. It is a leftover from before the module was parameterised by version,
and the parameterised `conf_cache` local at line 145 replaced it without the constant being
removed.

So it pinned nothing to v3 and caused no wrong result. It is worth deleting purely so the
next person auditing this does not have to repeat the grep, but it has no behavioural
effect, and I have left it in place rather than make an unrequested edit.

## Why `cache_conf_v8`'s right hemisphere is byte-identical to v9's

Confirmed rather than assumed, by hashing the inputs:

| file | SHA-256 (first 20) |
|---|---|
| `pial.rh.cyto7.v8.annot` | `504941c21952414b65e0` |
| `pial.rh.cyto7.v9.annot` | `504941c21952414b65e0` |
| `pial.lh.cyto7.v8.annot` | `d3cef05ca91a3a39253e` |
| `pial.lh.cyto7.v9.annot` | `89d5de08dc395b5bd4a5` |

The v8 and v9 **right-hemisphere annots are the same file**. `v8_to_v9_changelog.csv` has
543 rows and every one of them is `lh`: v9 is v8 plus the 543-vertex left entorhinal
even-up, with the right hemisphere untouched.

The score is a deterministic function of the annot and the fixed surface geometry, so an
identical input annot gives a byte-identical output map. The right-hemisphere score files
match for the same reason the right-hemisphere annots do, and the left-hemisphere ones
differ (`31209e643020aac479b7` for v8 against `84be914a79c6b03a0155` for v9) exactly where
the input differs. There is no caching artefact or copy error here.
