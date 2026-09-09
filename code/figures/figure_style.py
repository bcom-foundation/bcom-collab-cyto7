"""Central figure-style module — one source of truth for colour across all cyto7 figures.

Principle (SPEC_figure_style_guide.md): **sequential for magnitudes, diverging for
signed/centred quantities, ordered-categorical for the cyto7 types; one colormap per
quantity, reused everywhere; colorblind-safe.**

Categorical cyto7 palettes (ordered allocortex -> koniocortex, codes 1..7):

* :data:`CYTO7_GC` -- the **default**. The García-Cabezas convention used throughout this
  project and encoded in the cyto7 annot colortable: an ordered **greyscale-by-type ramp**
  (darker = less differentiated -> lighter = more differentiated; GC Fig. 8). Field-familiar
  and colorblind-safe. *(If a coloured GC palette is preferred, replace the RGB values in
  this one constant — every figure follows.)*
* :data:`CYTO7_VIRIDIS` -- the alternate: viridis sampled at 7 ordered levels.
* :data:`CYTO7_DISTINCT` -- a high-contrast qualitative palette (for the "isolated single
  type" montages where one type must pop against a plain brain).
* :data:`TIER3` -- a 3-colour ordered palette for the allocortex / mesocortex / isocortex
  tiers.

Code 0 (medial wall / unlabelled) is always :data:`LIGHT_BLUE`.

Continuous quantities are :class:`ContinuousSpec` (cmap + fixed vmin/vmax + label), so the
same scale is reused everywhere a quantity appears: :data:`MYELIN` (HCP T1w/T2w scale,
1.0-2.2), :data:`CONFIDENCE` (viridis, 0-1), :data:`SIGNED_DELTA` (RdBu_r, -3..+3),
:data:`GRADIENT` (RdBu_r diverging, symmetric), :data:`TIMESCALE` (magma), :data:`BANDPOWER`
(viridis).

All figure scripts import from here rather than defining their own colours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# cyto7 categorical
# --------------------------------------------------------------------------- #

#: cyto7 integer codes (1..7) and their canonical names (ascending differentiation).
CYTO7_CODES: list[int] = [1, 2, 3, 4, 5, 6, 7]
CYTO7_NAMES: list[str] = [
    "Allocortex", "Agranular", "Dysgranular", "Eulaminate I",
    "Eulaminate II", "Eulaminate III", "Koniocortex",
]

#: Medial wall / unlabelled base colour (PowerPoint light blue).
LIGHT_BLUE: tuple[float, float, float] = (0.62, 0.78, 0.92)

#: DEFAULT — García-Cabezas greyscale-by-type ramp (darker = less differentiated).
#: Re-spaced for perceptual contrast: the dark end (allocortex vs agranular) is
#: widened so the two least-differentiated types stay distinguishable after surface
#: lighting darkens them. Ordered categorical, colorblind-safe.
CYTO7_GC: dict[int, tuple[float, float, float]] = {
    1: (0.050, 0.050, 0.050),   # Allocortex  (near-black)
    2: (0.250, 0.250, 0.250),   # Agranular   (+0.20 gap vs allocortex)
    3: (0.420, 0.420, 0.420),   # Dysgranular
    4: (0.560, 0.560, 0.560),   # Eulaminate I
    5: (0.700, 0.700, 0.700),   # Eulaminate II
    6: (0.840, 0.840, 0.840),   # Eulaminate III
    7: (0.980, 0.980, 0.980),   # Koniocortex
}


def _sample_cmap(name: str, n: int) -> dict[int, tuple]:
    import matplotlib as mpl
    cmap = mpl.colormaps[name]
    xs = np.linspace(0.0, 1.0, n)
    return {i + 1: tuple(float(v) for v in cmap(x)[:3]) for i, x in enumerate(xs)}


#: ALTERNATE — viridis sampled at 7 ordered levels (allo dark-purple -> konio yellow).
CYTO7_VIRIDIS: dict[int, tuple[float, float, float]] = _sample_cmap("viridis", 7)

#: Ordinal grayscale ramp (type 1 dark -> type 7 light) for the labeled reference
#: figure (SPEC_labeled_areas_figure). Distinct from CYTO7_GC (which is re-spaced for
#: post-lighting contrast): this is an even dark->light ramp; medial wall is white.
CYTO7_GRAYSCALE: dict[int, tuple[float, float, float]] = {
    t: (v, v, v) for t, v in zip(range(1, 8), (0.16, 0.30, 0.42, 0.55, 0.68, 0.80, 0.92))
}

#: Qualitative high-contrast palette for isolated-single-type montages.
CYTO7_DISTINCT: dict[int, tuple[float, float, float]] = {
    1: (0.85, 0.10, 0.55), 2: (0.27, 0.00, 0.33), 3: (0.22, 0.34, 0.55),
    4: (0.13, 0.57, 0.55), 5: (0.36, 0.78, 0.39), 6: (0.74, 0.86, 0.20),
    7: (0.99, 0.91, 0.15),
}

#: 3-tier ordered palette: allocortex / mesocortex / isocortex.
TIER3: dict[str, tuple[float, float, float]] = {
    "allocortex": (0.20, 0.10, 0.36),   # viridis-dark
    "mesocortex": (0.13, 0.57, 0.55),   # viridis-teal
    "isocortex":  (0.86, 0.86, 0.20),   # viridis-yellow-green
}

#: Registry so a CLI ``--palette`` maps to a palette dict.
CYTO7_PALETTES: dict[str, dict[int, tuple]] = {
    "gc": CYTO7_GC,
    "viridis": CYTO7_VIRIDIS,
    "distinct": CYTO7_DISTINCT,
    "grayscale": CYTO7_GRAYSCALE,
}
DEFAULT_PALETTE = "gc"


def cyto7_palette(name: str = DEFAULT_PALETTE) -> dict[int, tuple]:
    """Return the {1..7 -> rgb} palette for *name* ('gc' | 'viridis' | 'distinct')."""
    try:
        return CYTO7_PALETTES[name]
    except KeyError:
        raise ValueError(f"Unknown cyto7 palette {name!r}; "
                         f"choose from {sorted(CYTO7_PALETTES)}")


def cyto7_vertex_rgb(labels, palette: str = DEFAULT_PALETTE, base=LIGHT_BLUE):
    """Per-vertex RGB array for cyto7 *labels* (0 -> base colour, 1..7 -> palette)."""
    pal = cyto7_palette(palette)
    rgb = np.tile(np.asarray(base, float), (np.asarray(labels).shape[0], 1))
    for c in CYTO7_CODES:
        rgb[labels == c] = np.asarray(pal[c], float)
    return rgb


def cyto7_listed_cmap(palette: str = DEFAULT_PALETTE) -> ListedColormap:
    """ListedColormap of the 7 type colours (for imshow/nilearn with vmin=1, vmax=7)."""
    pal = cyto7_palette(palette)
    return ListedColormap([pal[c] for c in CYTO7_CODES])


def cyto7_legend_handles(palette: str = DEFAULT_PALETTE, edgecolor="0.4") -> list[Patch]:
    """Shared cyto7 type legend (7 Patches, allo -> konio)."""
    pal = cyto7_palette(palette)
    return [Patch(facecolor=pal[c], edgecolor=edgecolor, label=CYTO7_NAMES[c - 1])
            for c in CYTO7_CODES]


def tier3_legend_handles(edgecolor="0.4") -> list[Patch]:
    return [Patch(facecolor=TIER3[k], edgecolor=edgecolor, label=k.capitalize())
            for k in ("allocortex", "mesocortex", "isocortex")]


# --------------------------------------------------------------------------- #
# Continuous quantities (one colormap + fixed range per quantity)
# --------------------------------------------------------------------------- #

#: HCP 2016 "myelin" (T1w/T2w) colour scale: black -> purple -> blue -> green -> yellow -> red.
_HCP_COLORS = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.5), (0.0, 0.0, 1.0),
               (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
HCP_CMAP = LinearSegmentedColormap.from_list("hcp_myelin", _HCP_COLORS)


@dataclass(frozen=True)
class ContinuousSpec:
    """A continuous quantity's fixed display style: colormap + range + label."""
    cmap: object
    vmin: float
    vmax: float
    label: str

    def norm(self) -> Normalize:
        return Normalize(self.vmin, self.vmax)


MYELIN = ContinuousSpec(HCP_CMAP, 1.0, 2.2, "T1w/T2w (myelin)")
CONFIDENCE = ContinuousSpec("viridis", 0.0, 1.0, "support")
SIGNED_DELTA = ContinuousSpec("RdBu_r", -3.0, 3.0, "Δ ordinal type (signed)")
GRADIENT = ContinuousSpec("RdBu_r", -1.5, 1.5, "functional gradient (z)")
TIMESCALE = ContinuousSpec("magma", None, None, "intrinsic timescale")   # data-scaled
BANDPOWER = ContinuousSpec("viridis", None, None, "band power")           # data-scaled

#: Back-compat aliases (older scripts referenced these names).
MYELIN_VMIN, MYELIN_VMAX = MYELIN.vmin, MYELIN.vmax


def add_colorbar(fig, spec: ContinuousSpec, ax=None, orientation="vertical",
                 fraction=0.046, pad=0.04, label=None):
    """Attach a colorbar for a :class:`ContinuousSpec` (fixed range + label)."""
    from matplotlib.cm import ScalarMappable
    vmin = spec.vmin if spec.vmin is not None else 0.0
    vmax = spec.vmax if spec.vmax is not None else 1.0
    sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap=spec.cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation=orientation, fraction=fraction, pad=pad)
    cb.set_label(label or spec.label, fontsize=9)
    return cb


# Convenience list of every continuous spec (for docs / tests).
CONTINUOUS = {
    "myelin": MYELIN, "support": CONFIDENCE, "signed_delta": SIGNED_DELTA,
    "gradient": GRADIENT, "timescale": TIMESCALE, "bandpower": BANDPOWER,
}
