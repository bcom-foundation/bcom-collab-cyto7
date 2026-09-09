# The two source sets, as found in the code

Quoted verbatim. Line numbers are as of commit `9fb1007`.

## Set 1: C_atlas, the confidence component

`scripts/build_confidence_map.py`, lines 189 to 203. Two branches, isocortex and allocortex:

```python
    # ---- C_atlas ----
    C_atlas = np.full(n, np.nan)
    w_atlas = np.zeros(n)
    ve = voneconomo_code(hemi, n, lookup)
    iso = labelled & (lab >= 2) & (ve >= 1)            # isocortex with a vE type
    C_atlas[iso] = 1.0 - np.abs(ordc[iso] - ordinal(ve)[iso]) / MAXD
    w_atlas[iso] = args.w_atlas
    # allocortex: belt consensus (FS ex-vivo + Destrieux)
    fs_belt = (label_mask(ATLAS_DIR / f"{hemi}.entorhinal_exvivo.label", n) |
               label_mask(ATLAS_DIR / f"{hemi}.perirhinal_exvivo.label", n))
    dx_belt = destrieux_belt_mask(hemi, n)
    belt_votes = fs_belt.astype(float) + dx_belt.astype(float)
    allo = lab == 1
    C_atlas[allo] = belt_votes[allo] / 2.0
    w_atlas[allo] = args.w_atlas
```

| branch | applies to | sources | weights | disagreement rule |
|---|---|---|---|---|
| isocortex | `lab >= 2` with a von Economo type | **von Economo only** (1 source) | n/a, single source | ordinal closeness `1 - abs(d)/6`, graded |
| allocortex | `lab == 1` | **FreeSurfer ex-vivo** (entorhinal OR perirhinal) and **Destrieux** (parahippocampal) | **unweighted, 1 each** | fraction of the 2 sources placing the vertex in the belt: 0, 0.5 or 1 |

Where no branch applies, the per-vertex weight is 0 and the component does not contribute.

## Set 2: the Annex B weighted belt reference

`scripts/allocortex_proposals.py`, line 69 and lines 177 to 178:

```python
# Source weights (cytoarchitectonic > Glasser > gyral).
W_FS, W_GLASSER, W_DESTRIEUX, W_DESIKAN = 3.0, 2.0, 1.0, 1.0
```

```python
    weighted = (W_FS * fs_belt + W_GLASSER * gl_belt + W_DESTRIEUX * dx_belt +
                W_DESIKAN * dk_belt) / (W_FS + W_GLASSER + W_DESTRIEUX + W_DESIKAN)
    trusted_belt = fs_belt | gl_belt           # histological + Glasser call
    gyral_belt = dx_belt | dk_belt
```

with the member areas fixed at lines 71 to 74:

```python
GLASSER_BELT = ["EC", "PreS", "PeEc", "Pir", "PHA1", "PHA2", "PHA3"]
DESTRIEUX_BELT = ("G_oc-temp_med-Parahip",)            # parahippocampal (gyral)
DESIKAN_BELT = ("entorhinal", "parahippocampal")
```

| applies to | sources | weights | disagreement rule |
|---|---|---|---|
| the limbic belt only | FS ex-vivo, Glasser (7 areas), Destrieux (1 area), Desikan (2 areas) | **3 / 2 / 1 / 1**, sum 7 | weighted fraction of sources placing the vertex in the belt, 8 possible values from 0 to 1 |

## What Annex C says, and whether it is right

Annex C's implementation note, verbatim:

> *In the released build this general form is realised with a single reference for isocortex (the
> ordinal agreement 1 − |τ(v) − τ_vE(v)|/6 against the von-Economo-derived type map) and, for the
> allocortex sliver, the fraction of two belt sources (FreeSurfer ex-vivo + Destrieux) marking the
> vertex as paleocortical belt. This uses fewer sources than the Annex B benchmark reference (which
> adds Glasser and Desikan, weights 3/2/1/1); harmonising the two source sets is a minor planned
> refinement.*

**Annex C's description of the code is accurate.** Both branches are described correctly, the two
missing sources are named correctly, and the 3/2/1/1 weights match line 69 exactly. Nothing in it is
stale.

**One word in it is wrong, and Annex B is the place that says so.** Annex C calls the second set "the
Annex B benchmark reference". Annex B itself, line 143, says the opposite:

> *This reliability-weighted consensus was used during map construction to adjudicate limbic-belt
> labels where the areal atlas is unreliable; **it is a construction reference, not a benchmark that
> yields a reported agreement score**.*

So the phrase "benchmark reference" in Annex C should be "construction reference", to agree with
Annex B.

## The structural point that decides the scope of any harmonisation

**The two sets are not two versions of the same object.** The Annex B reference is a **belt-membership
consensus defined only over the limbic belt**. It has no isocortical counterpart: there is no
four-source ordinal reference for eulaminate or koniocortex anywhere in the codebase, because Glasser,
Destrieux and Desikan do not carry cytoarchitectural type.

Therefore harmonising can only touch **the allocortex branch of C_atlas**, which covers
6,421 of 304,972 labelled vertices, **2.1% of the map**. The isocortex branch, 97.9% of the map, stays
on von Economo whatever is decided, and Annex C's general formula already describes that correctly as
the single-source realisation of a weighted sum.

This is the reason the harmonised score correlates with the released one at r = 0.992 while the
allocortex median moves by 0.24: almost nothing changes, and what does change is concentrated in the
one type whose median the manuscript quotes.
