"""Manuscript figure generation for the Nature Methods XL-MS benchmarking paper.

Run:
    pipenv run python -m src.analysis.manuscript_plots

Outputs PNG + SVG to NormalisedData/plots/manuscript/.
"""
from __future__ import annotations

import itertools
import math
import weakref
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.transforms import ScaledTranslation, offset_copy

from src.analysis.fdr import add_decoy_category
from src.analysis.entrapment import add_origin_flags, load_organism_sets
from src.analysis.ibaq_copresence import add_impossible_flag_ecoli_human, load_ibaq_presence
from src.analysis.overlap import (
    _pair_key,
    _protein_pair_key,
    compute_peptide_pair_overlap,
    compute_protein_pair_overlap,
)
from src.analysis.score_hist import (EXCLUDED_ENGINES, SP_ECOLI, SP_HUMAN,
                                     SP_MYCPN, DS_EH, DS_EM)
from src.analysis.spectrum_origin import (
    ENGINE_SCAN_TYPE,
    add_impossible_flag,
    add_spectrum_origin,
    load_merged_lookup,
)

mpl.rcParams["svg.fonttype"] = "none"   # embed text as SVG <text> elements, not paths
# Keep every raster its own <image> in vector output. Matplotlib defaults this
# to True, which fuses all the images in an axes into one before writing — so
# the faded backgrounds arrive as a handful of merged blobs and cannot be
# moved or deleted individually when a figure is touched up by hand.
mpl.rcParams["image.composite_image"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
NORMALISED_DIR = REPO_ROOT / "NormalisedData"
OUT_DIR = NORMALISED_DIR / "plots" / "manuscript"

# Whether to also write the individual panel crops of the multi-panel figures.
# Off by default: fig_multi1 and fig_multi2 emit 13 extra names (26 files with
# both extensions) between them, which buries the main figures in OUT_DIR.
# Turned on by the --subpannel CLI flag; main() leaves it off.
SAVE_PANELS = False

# Consistent engine display names and order
ENGINE_ORDER = [
    "kojak", "scout_fl", "scout_db", "xisearch", "proteinprospector",
    "mango", "msannika", "merox",
    "plink2", "plink3", "plink3_filtered",
    "xlinkx25", "xlinkx_mgf",
]
ENGINE_LABELS = {
    "kojak": "Kojak",
    "scout_fl": "Scout (FL)",
    "scout_db": "Scout (DB)",
    "xisearch": "xiSEARCH",
    "proteinprospector": "ProteinProspector",
    "mango": "Mango",
    "msannika": "MS Annika",
    "merox": "MeroX",
    "plink2": "pLink2",
    "plink3": "pLink3 (FDR)",
    "plink3_filtered": "pLink3 (post-filter)",
    "xlinkx25": "XlinkX PD2.5",
    "xlinkx_mgf": "XlinkX PD3.3",
    # Alternative PPI roll-ups: keys that exist only at PPI level, carrying a
    # second derivation of the same engine's protein pairs. Not in ENGINE_ORDER,
    # but fig2 plots every engine present in the summaries, so they need labels.
    # The derivation itself is appended by engine_label() on PPI panels only.
    "plink2_csm": "pLink2",
    "xlinkx_mgf_csm": "XlinkX PD3.3",
}

# How each entry's PPI level was derived. Appended by engine_label() on PPI
# panels only: the same engine key at CSM level is a direct export and must not
# carry the note. "Res.Pair" is the engine's own residue-pair report, the level
# closest to PPI but shipping validated targets only; "CSMs" is aggregated from
# the CSM export including its decoys, so a PPI-level decoy FDR is computable.
PPI_DERIVATION = {
    # pLink2 has no residue-pair export either; both of its PPI entries are
    # CSM-derived and differ only in the aggregation used, so only the shared
    # csm_to_ppi one is marked.
    "plink2_csm": "CSM",
    "msannika": "ResPair",
    "xlinkx_mgf": "ResPair",
    "xlinkx_mgf_csm": "CSM",
    "xlinkx25": "CSM",
}


def engine_label(engine: str, level: str | None = None) -> str:
    """Display name, with the PPI derivation appended on PPI panels only."""
    base = ENGINE_LABELS.get(engine, engine)
    note = PPI_DERIVATION.get(engine) if level == "PPI" else None
    return f"{base} ({note})" if note else base


# Name of each engine's own CSM score column, as it appears in its native
# export, before the loader renames it to the canonical "score" — see each
# loader's rename()/assignment of "score" in src/loaders/. Used as the
# x-axis label on the per-engine score-distribution panels (FigS1/FigS2,
# plink3_scores) instead of the generic "Score", since the raw scale and
# distribution shape differ by engine and are not directly comparable.
# Mango has no native score column: it plots 1 - q-value (see src/loaders/
# mango.py), so it gets a description rather than a column name.
NATIVE_SCORE_COLUMN = {
    "kojak": "Sum_score",
    "scout_fl": "Score",
    "scout_db": "Score",
    "xisearch": "Score",
    "proteinprospector": "SVM.score",
    "mango": "1 − q-value",
    "msannika": "Combined Score",
    "merox": "Score",
    "plink2": "Score",
    "plink3": "Score",
    "plink3_filtered": "Score",
    "xlinkx25": "XlinkX Score",
    "xlinkx_mgf": "XlinkX Score",
}

C_DECOY   = "#1f77b4"   # blue
C_ENTRAP  = "#f17000"   # orange
C_IMPOSS  = "#d62728"   # red
C_COFRAC  = "#7b2ff2"   # purple — never-co-fractionating, as in fig_s5 / fig5

# Overlap-figure segments, shared by fig7_overlap, fig7_overlap_ppi (which is
# also fig_multi1 panel g), fig_s1_overlap_ecoli_mycpn and fig_s5, so the
# figures cannot drift apart: green family = possible, orange/red = impossible.
# All of them pool the two impossible segments into one (C_IMPOSS, the
# project's general impossible-red) via _draw_overlap_bars(collapse_impossible=
# True) — except fig_s5's cofractionation-split variant, which exists
# specifically to show the full shared/unique x possible/impossible/
# not-co-fractionating breakdown below and keeps all six segments.
C_OVL_SH_POS = "#50cf50"   # dark green:  shared + possible
C_OVL_SH_IMP = "#f17000"   # orange:      shared + impossible
C_OVL_UN_POS = "#b6efb0"   # light green: unique + possible
C_OVL_UN_IMP = "#d62728"   # red:         unique + impossible
# Optional third impossible class: pairs impossible only because the two E. coli
# proteins never co-fractionated, as opposed to a human entrapment match. Purple
# because the project already uses #eb88ff for "not co-present" elsewhere; the
# darker shade marks the shared half, matching the dark/light convention above.
# Worst CVD separation involving either purple is dE 57.8 (protan), well clear of
# the orange/red pair already in this palette at 18.8.
C_OVL_SH_NOCP = "#7b2ff2"  # dark purple:  shared + impossible (not co-fractionating)
C_OVL_UN_NOCP = "#eb88ff"  # light purple: unique + impossible (not co-fractionating)
REF_FDR   = 2.0         # declared 2% FDR reference line

# Engines with no dedicated PPI-level FDR control: their protein-pair output is
# aggregated/extrapolated from a lower level (CSM or residue-pair report), so a
# declared 2% at PPI is not an independent measurement. Shared by figS3's "*"
# tick-label convention and the PPI overlap panels (fig7_overlap_ppi / fig_multi1
# panel g), which inherit the same caveat for the same reason.
_PPI_FROM_CSM = {"xlinkx25", "xlinkx_mgf", "plink2"}


def load_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ecoli_human, ecoli_mycpn) summary DataFrames, engine rows only.

    Both DataFrames contain rows for all levels (CSM and PPI). Figure functions
    must filter on the ``level`` column themselves, e.g.::

        csm = eh[eh["level"] == "CSM"]
    """
    eh = pd.read_csv(NORMALISED_DIR / "summary_ecoli_human.csv")
    em = pd.read_csv(NORMALISED_DIR / "summary_ecoli_mycpn.csv")
    eh = eh[eh["search_engine"] != "ALL"].copy()
    em = em[em["search_engine"] != "ALL"].copy()
    eh = eh[~eh["search_engine"].isin(EXCLUDED_ENGINES)].copy()
    em = em[~em["search_engine"].isin(EXCLUDED_ENGINES)].copy()
    return eh, em


def _save(fig: plt.Figure, name: str) -> None:
    # House style, applied at the choke point so no figure can opt out: one body
    # size everywhere, panel letters larger. Runs before the faded-background
    # sync, which measures text extents.
    _set_uniform_text_size(fig, fontsize=_SCATTER_LABEL_FS)
    _sync_faded_backgrounds(fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .svg")


# Label placement rounds for _add_labels_dynamic. Round 3 keeps each offset but
# re-anchors the text along it: "above ha=left" still sits above the point, with
# the text starting there and running right instead of being centred on it. That
# is what lets a wide label beside a spine stay inside the axes.
_LABEL_CARDINALS = ("above", "left", "right", "below")
_LABEL_DIAGONALS = ("above right", "above left", "below right", "below left")
def _label_distance_ladder(base=1.0, outward=2.0, inward=1.0, step=0.25):
    """Multiples of `offset` to try, in order.

    *base* first, then alternately a step further out and a step nearer,
    widening each time: 1, 1.25, 0.75, 1.5, 0.5, … The alternation matters
    because the two
    directions fail differently. Moving out buys clearance but strands the
    label from its point; moving in keeps it attached but risks the marker.
    Trying them in step lets each label settle at whichever is nearest to
    *base* in either direction, rather than always drifting outwards.

    The inward arm stops short of the outward one (*inward* < *outward*), which
    keeps the ladder from reaching distances where the text would sit inside its
    own marker.
    """
    out = np.arange(0.0, outward, step)
    back = -np.arange(step, inward, step)
    merged = sorted(list(enumerate(out)) + list(enumerate(back, 2)))
    return tuple(float(round(base + delta, 4)) for _, delta in merged)


# Every round is retried at each of these, so a label only moves away from the
# preferred distance once nothing at it is free anywhere.
_LABEL_DISTANCES = _label_distance_ladder()
_LABEL_REANCHORED = (
    "above ha=left", "above ha=right", "below ha=left", "below ha=right",
    "left va=top", "left va=bottom", "right va=top", "right va=bottom",
    "above right va=center", "above right va=top",
    "above left va=center", "above left va=top",
    "below right va=center", "below right va=bottom",
    "below left va=center", "below left va=bottom",
)


def _add_labels_dynamic(
    ax,
    xs,
    ys,
    labels,
    *,
    offset=6,
    fontsize=None,
    colors=None,
    styles=None,
    preferred=(_LABEL_CARDINALS, _LABEL_DIAGONALS, _LABEL_REANCHORED),
    distances=_LABEL_DISTANCES,
    avoid_points=True,
    avoid_crossings=True,
    avoid_bboxes=(),
    avoid_xy=None,
    orient_neighbours=0,
    orient_radius=None,
    point_padding=3,
    adjust=True,
    at_force_text=(0.5, 0.5),
    at_force_pull=(0.4, 0.4),
    at_force_static=(0.5, 0.5),
    at_max_move=(40, 40),
    at_only_move=None,
    arrowprops=None,
    **text_kwargs,
):
    """
    Add non-overlapping labels to a Matplotlib Axes.

    Labels are placed using this priority:
        1. above the data point
        2. left of the data point
        3. right of the data point
        4. below the data point

    If none of those positions is available, the label is handed to
    adjustText, which is allowed to move it freely and an arrow is drawn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add labels to.

    xs, ys : array-like
        Data-point coordinates.

    labels : array-like
        Text labels.

    offset : float, default=6
        Distance from the data point in display points.

    fontsize : float, optional
        Font size.

    colors : sequence, optional
        One colour per label. Always per-label, never a single colour for all —
        a lone RGB tuple could not be told apart from a sequence of three
        colours. For one colour throughout, pass ``color=`` instead, which goes
        to ``**text_kwargs``.

    styles : sequence of dict, optional
        One dict of text properties per label, merged over ``**text_kwargs``
        and over ``colors``, so it wins where they overlap. Use it when labels
        need to differ in more than colour.

    preferred : tuple of tuples, default=(_LABEL_CARDINALS, _LABEL_DIAGONALS,
                                          _LABEL_REANCHORED)
        Position preference, as rounds. Every label is offered the first round's
        positions before any label is offered the second's, so a label does not
        take a diagonal that a later label needs as its only cardinal. Whatever
        is still unplaced after the last round goes to adjustText.

        Round 3 re-anchors: same offsets, but the point may sit at the text's
        left end, middle or right end (for "above"/"below") or at its top,
        middle or bottom (for the others). A centred label whose point is 30 px
        from the spine but which is 81 px wide has nowhere to go in rounds 1-2;
        re-anchored, it can sit beside its point and stay on the axes.

        A flat sequence of names is accepted and treated as a single round.

    distances : sequence of float, default=_LABEL_DISTANCES
        Multiples of *offset* at which to retry the rounds, in the order given —
        see :func:`_label_distance_ladder`. The schedule is distance-major:
        every round at the first distance for all labels, then every round at
        the second, and so on, so no label takes a far position while another
        still has nothing at a nearer one. This is what lets a label reach past
        a neighbour that has already claimed the space next to it, instead of
        falling through to adjustText.

    avoid_points : bool, default=True
        Whether labels should avoid other data points.

    avoid_crossings : bool, default=True
        Reject a position whose label would be separated from its own point by
        another label *or another marker*. Without it a label can end up on the
        far side of a neighbour, so the eye has to jump over something to
        connect a point to its own label — the effect looks like crossed leader
        lines even though none are drawn. Markers count for the same reason
        labels do: a label reached only by passing over someone else's point
        reads as belonging to that point.

    avoid_bboxes : sequence of Bbox, default=()
        Display-space boxes that are already occupied. Lets a caller place
        labels in several calls without the later ones landing on the earlier —
        see :func:`_add_labels_by_density`.

    avoid_xy : tuple of array-like, optional
        ``(x, y)`` of every point on the axes, when that is more than the points
        being labelled here. Defaults to ``xs, ys``. A label must dodge all the
        markers, not just those belonging to its own call.

    orient_neighbours : int, default=0
        If non-zero, reorder each round's positions per label to point away from
        the crowd: take the mean unit vector towards this many nearest points
        and try the candidates in descending order of how squarely they face the
        other way. 0 keeps the declared order for every label. Ties keep the
        declared order, so it refines ``preferred`` rather than replacing it.

    orient_radius : float, optional
        Only neighbours within this many display points steer the order. None
        considers the nearest ``orient_neighbours`` at any distance — which for
        an isolated point means being nudged away from something far off that
        was never going to collide. A radius of roughly a label width keeps the
        rule local.

    point_padding : float, default=3
        Extra padding around data points, in display points.

    adjust : bool, default=True
        Whether to run adjustText on labels that cannot use a preferred
        position.

    at_force_text : tuple, default=(1.0, 1.0)
        adjustText's ``force_text``: how hard the fallback labels push each
        other apart. The ``at_`` prefix marks arguments passed straight through
        to adjustText, so they cannot be confused with this function's own.
        Well above adjustText's (0.1, 0.2) default — the fallback labels are by
        definition the ones in the crowded corner of the plot.

    at_force_pull : tuple, default=(0.5, 0.5)
        adjustText's ``force_pull``: how hard a label is drawn back towards the
        point it belongs to. Far above adjustText's (0.01, 0.01) default, which
        keeps fallback labels tethered near their points — lower it to let them
        escape a crowded corner, at the cost of longer leader lines.

    at_force_static : tuple, default=(0.5, 0.5)
        adjustText's ``force_static``: how hard the data points and the objects
        in ``objects`` push labels away.

    at_max_move : tuple, default=(10, 10)
        adjustText's ``max_move``: the furthest a label may travel per
        iteration, in display points.

    at_only_move : dict, optional
        adjustText's ``only_move``. Defaults to ``{"text": "xy"}`` — a label
        that reached the fallback has no good fixed position left, so it is
        allowed both axes.

    arrowprops : dict, optional
        Arrow properties passed to adjustText. If omitted, a subtle default
        is used.

    **text_kwargs
        Additional keyword arguments passed to ax.annotate().

    Returns
    -------
    texts : list
        All created Matplotlib Text/Annotation objects.

    fallback : list
        Labels that were passed to adjustText.
    """

    xs = list(xs)
    ys = list(ys)
    labels = list(labels)

    if not (len(xs) == len(ys) == len(labels)):
        raise ValueError("xs, ys, and labels must have the same length")

    if colors is not None:
        colors = list(colors)
        if len(colors) != len(labels):
            raise ValueError("colors must have the same length as labels")

    if styles is not None:
        styles = list(styles)
        if len(styles) != len(labels):
            raise ValueError("styles must have the same length as labels")

    def style_for(i):
        """Text properties for label *i*: shared first, per-label on top."""
        out = {}
        if fontsize is not None:
            out["fontsize"] = fontsize
        out.update(text_kwargs)
        if colors is not None:
            out["color"] = colors[i]
        if styles is not None:
            out.update(styles[i])
        return out

    # We need a renderer to obtain actual text bounding boxes.
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)

    # Downward placements are anchored va="top", i.e. on the text's layout top,
    # which sits barely above the capitals. Upward ones are anchored va="bottom"
    # on the layout bottom, which includes the whole font descent — so "above"
    # gets up to a descent of free clearance that "below" does not, and the
    # below labels crowd their markers (measured 2.2 pt against 5.0 pt for a
    # label with no descenders). Push the downward ones out by that descent so
    # every position clears by the same amount.
    _fp = mpl.font_manager.FontProperties(
        size=fontsize if fontsize is not None else plt.rcParams["font.size"])
    _descent = renderer.get_text_width_height_descent("Ag", _fp, False)[2]
    _descent *= 72.0 / fig.dpi          # display pixels -> points
    _down = offset + _descent

    def _positions_for(scale):
        """Position -> (dx, dy, ha, va), at *scale* times the nominal offset."""
        o = offset * scale
        down = o + _descent           # downward placements still clear the descent
        base = {
            "above":       (0,   o,     "center", "bottom"),
            "left":        (-o,  0,     "right",  "center"),
            "right":       (o,   0,     "left",   "center"),
            "below":       (0,   -down, "center", "top"),
            "above right": (o,   o,     "left",   "bottom"),
            "above left":  (-o,  o,     "right",  "bottom"),
            "below right": (o,   -down, "left",   "top"),
            "below left":  (-o,  -down, "right",  "top"),
        }

        # Each base plus its re-anchorings: same offset, anchor slid along the
        # text. Above/below vary the horizontal anchor and keep the vertical
        # one, so the label stays above (or below) the point; the rest reverse.
        out = {}
        for name, (dx, dy, ha, va) in base.items():
            out[name] = (dx, dy, ha, va)
            if name in ("above", "below"):
                for alt in ("left", "right"):
                    out[f"{name} ha={alt}"] = (dx, dy, alt, va)
            else:
                for alt in ("top", "center", "bottom"):
                    if alt != va:
                        out[f"{name} va={alt}"] = (dx, dy, ha, alt)
        return out

    scaled_positions = {sc: _positions_for(sc) for sc in distances}
    positions = _positions_for(1.0)      # for validation and the direction rule

    # A flat sequence of names is one round; anything else is already rounds.
    rounds = ((tuple(preferred),) if preferred and isinstance(preferred[0], str)
              else tuple(tuple(r) for r in preferred))

    invalid = {name for r in rounds for name in r} - set(positions)
    if invalid:
        raise ValueError(
            f"Invalid preferred positions: {invalid}. "
            f"Use any of: {', '.join(sorted(positions))}."
        )

    placed_bboxes = list(avoid_bboxes)
    avoid_x, avoid_y = (xs, ys) if avoid_xy is None else (list(avoid_xy[0]),
                                                          list(avoid_xy[1]))
    texts = [None] * len(labels)   # filled by index, so input order is kept
    fallback = []
    fallback_idx = []

    # Transform data coordinates -> display coordinates.
    transform = ax.transData

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def point_bbox(x, y):
        """Return a small display-space bbox around a data point."""

        px, py = transform.transform((x, y))

        return plt.matplotlib.transforms.Bbox.from_extents(
            px - point_padding,
            py - point_padding,
            px + point_padding,
            py + point_padding,
        )

    def overlaps(bbox, others):
        """Check whether bbox overlaps any existing bbox."""

        return any(bbox.overlaps(other) for other in others)

    def segment_hits(p0, p1, bb):
        """Whether the segment p0->p1 enters the axis-aligned box *bb*.

        Liang-Barsky clipping: walk the four slabs, narrowing the parameter
        range the segment may occupy inside the box; an empty range means it
        misses.
        """
        (x0, y0), (x1, y1) = p0, p1
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for num, den in ((x0 - bb.x0, -dx), (bb.x1 - x0, dx),
                         (y0 - bb.y0, -dy), (bb.y1 - y0, dy)):
            if den == 0:
                if num < 0:
                    return False          # parallel to this slab and outside it
                continue
            t = num / den
            if den < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
        return t0 <= t1

    # Marker boxes, built once: the crossing test walks them for every
    # surviving candidate, and the transform does not change mid-call.
    _point_boxes = [(px, py, point_bbox(px, py))
                    for px, py in zip(avoid_x, avoid_y)]

    def crosses_between(x, y, bbox, label_boxes):
        """Whether another label or marker lies between the point and a candidate."""
        here = ax.transData.transform((x, y))
        mid = ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)

        if any(segment_hits(here, mid, other) for other in label_boxes):
            return True

        # Skip this label's own marker, which the segment starts inside.
        return any(segment_hits(here, mid, box)
                   for px, py, box in _point_boxes
                   if not (px == x and py == y))

    # Unit direction of each candidate, for the orientation rule below. Taken
    # from the offsets themselves, so "below" stays consistent with the descent
    # correction applied to it.
    _dir = {name: np.array([dx, dy], float) / (np.hypot(dx, dy) or 1.0)
            for name, (dx, dy, _, _) in positions.items()}
    _avoid_px = (ax.transData.transform(np.column_stack([avoid_x, avoid_y]))
                 if orient_neighbours else None)
    _radius_px = (None if orient_radius is None
                  else orient_radius * fig.dpi / 72.0)

    def ordered_positions(x, y, names):
        """*names*, best-facing first: away from the mean direction of the crowd."""
        if not orient_neighbours:
            return names

        here = np.asarray(ax.transData.transform((x, y)), float)
        delta = _avoid_px - here
        dist = np.hypot(delta[:, 0], delta[:, 1])

        near = dist > 1e-9                      # drop this label's own point
        if _radius_px is not None:
            near &= dist <= _radius_px
        if not near.any():
            return names

        pick = np.argsort(dist[near])[:orient_neighbours]
        crowd = (delta[near][pick] / dist[near][pick][:, None]).mean(axis=0)

        norm = float(np.hypot(*crowd))
        if norm < 1e-9:                         # neighbours cancel out
            return names
        away = -crowd / norm

        # sorted() is stable, so equally good candidates keep declared order.
        return sorted(names, key=lambda nm: -float(_dir[nm] @ away))

    # ------------------------------------------------------------------
    # First passes: preferred positions, one round at a time
    # ------------------------------------------------------------------

    pending = list(range(len(labels)))

    # (distance, round) pairs, distance-major: all rounds at the closest
    # distance for every label before any label is offered the next distance.
    schedule = [(sc, names) for sc in distances for names in rounds]

    for _scale, round_positions in schedule:
        at_scale = scaled_positions[_scale]
        still_pending = []
        for i in pending:
            x, y, label = xs[i], ys[i], labels[i]

            placed = False

            for position in ordered_positions(x, y, round_positions):

                dx, dy, ha, va = at_scale[position]

                kwargs = dict(
                    xy=(x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                )

                kwargs.update(style_for(i))

                text = ax.annotate(label, **kwargs)

                bbox = text.get_window_extent(renderer=renderer)

                # Slightly expand the bbox so labels don't sit directly
                # against each other.
                bbox = bbox.expanded(1.02, 1.05)

                # Leaving the axes counts as a collision. Without this a "left"
                # position against the left spine reads as free and the label is
                # drawn off-plot — and, because nothing registers as a collision,
                # the adjustText fallback never fires.
                inside = (bbox.x0 >= ax_bbox.x0 and bbox.x1 <= ax_bbox.x1
                          and bbox.y0 >= ax_bbox.y0 and bbox.y1 <= ax_bbox.y1)

                collision = not inside or overlaps(bbox, placed_bboxes)

                if avoid_crossings and not collision:
                    collision = crosses_between(x, y, bbox, placed_bboxes)

                if avoid_points and not collision:
                    # Avoid every data point except this label's own, which is
                    # skipped because the preferred positions are already offset
                    # from it. Compared by coordinate rather than index: with
                    # avoid_xy the two lists need not line up.
                    for px, py in zip(avoid_x, avoid_y):
                        if px == x and py == y:
                            continue

                        if bbox.overlaps(point_bbox(px, py)):
                            collision = True
                            break

                if not collision:
                    placed_bboxes.append(bbox)
                    texts[i] = text
                    placed = True
                    break

                # Candidate didn't work.
                text.remove()

            if not placed:
                still_pending.append(i)

        pending = still_pending

    # ------------------------------------------------------------------
    # Whatever no round could place -> adjustText.
    # ------------------------------------------------------------------

    for i in pending:

        kwargs = dict(ha="center", va="bottom")
        kwargs.update(style_for(i))

        # A plain data-coordinate Text, not an offset annotation like the
        # preferred placements above. adjustText moves labels with
        # set_position(), which an annotation created with
        # textcoords="offset points" interprets as a *point offset* rather
        # than a data coordinate — so the label lands somewhere unrelated,
        # and ensure_inside_axes cannot keep it on the plot either.
        text = ax.text(xs[i], ys[i], labels[i], **kwargs)

        texts[i] = text
        fallback.append(text)
        fallback_idx.append(i)

    # ------------------------------------------------------------------
    # Last pass: let adjustText solve only the difficult labels.
    # ------------------------------------------------------------------

    if adjust and fallback:
        _adjust_fallback_labels(
            ax, fallback,
            x=avoid_x, y=avoid_y,
            target_x=[xs[i] for i in fallback_idx],
            target_y=[ys[i] for i in fallback_idx],
            objects=placed_bboxes,
            arrowprops=arrowprops,
            at_force_text=at_force_text, at_force_pull=at_force_pull,
            at_force_static=at_force_static, at_max_move=at_max_move,
            at_only_move=at_only_move,
        )

    return texts, fallback


def _adjust_fallback_labels(ax, fallback, *, x, y, target_x, target_y, objects,
                            arrowprops=None,
                            at_force_text=(1.0, 1.0), at_force_pull=(0.5, 0.5),
                            at_force_static=(0.5, 0.5), at_max_move=(10, 10),
                            at_only_move=None):
    """Hand the labels no preferred position could take to adjustText.

    Split out of :func:`_add_labels_dynamic` so a caller that places in several
    passes can run this once over every leftover, rather than once per pass —
    adjustText can only separate labels it sees together.
    """
    if not fallback:
        return

    if arrowprops is None:
        arrowprops = dict(arrowstyle="-", color="0.5", lw=0.8,
                          shrinkA=2, shrinkB=2)

    adjust_text(
        fallback,
        ax=ax,
        x=x,
        y=y,

        # Labels already placed will not move, so they are static obstacles
        # here rather than part of the problem.
        objects=objects,
        target_x=target_x,
        target_y=target_y,

        # Difficult labels may now move freely.
        only_move=({"text": "xy"} if at_only_move is None else at_only_move),

        # Keep labels reasonably close to their points.
        force_pull=at_force_pull,

        # Push difficult labels apart.
        force_text=at_force_text,

        max_move=at_max_move,

        # Don't aggressively push against the data points.
        force_static=at_force_static,

        arrowprops=arrowprops,
    )


def _density_elbow(sorted_scores) -> int:
    """Where a sorted crowding curve bends: the size of the dense group.

    The knee construction — the point furthest from the chord joining the first
    and last of the curve, after scaling both axes to [0, 1] so the result does
    not depend on the units. For a run of tightly packed points followed by a
    jump to the roomy ones, that lands in the jump.
    """
    y = np.asarray(sorted_scores, dtype=float)
    n = len(y)
    if n < 3:
        return n

    span = y[-1] - y[0]
    if span <= 0:                       # every point equally crowded
        return n

    x = np.linspace(0.0, 1.0, n)
    return int(np.argmax(np.abs((y - y[0]) / span - x))) + 1


def _add_labels_by_density(ax, xs, ys, labels, *, colors=None, styles=None,
                           neighbours=2, arrowprops=None,
                           at_force_text=(0.5, 0.5),
                           at_force_pull=(0.4, 0.4),
                           at_force_static=(0.5, 0.5),
                           at_max_move=(40, 40),
                           at_only_move=None, **kwargs):
    """Place labels crowded-first, then the rest — a wrapper on _add_labels_dynamic.

    :func:`_add_labels_dynamic` is greedy in input order, so whichever engine
    happens to come first in ENGINE_ORDER gets first refusal on the good
    positions. Inside a cluster that is backwards: a point with room to spare
    will find a slot whenever it is placed, while a crowded one may have nothing
    left by the time its turn comes.

    So score every point by how crowded it is, split the scores at their elbow,
    and run the full placement — every preferred round — for the dense group
    before the roomy group is offered anything. Crowding is measured in
    **display** space: these axes are sqrt- or log-scaled, so equal distances in
    data units are not equal distances on the page.

    ``neighbours`` sets how many nearest points the score averages over.

    Leftovers from both groups go to adjustText together, in one call, because
    adjustText can only separate labels it sees at the same time.

    Returns (texts, fallback) like the function it wraps, with *texts* in input
    order.
    """
    xs, ys, labels = list(xs), list(ys), list(labels)
    n = len(labels)
    if n == 0:
        return [], []

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Mean distance to the nearest `neighbours` points, in display space.
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    k = min(neighbours, n - 1)
    score = (np.sort(d, axis=1)[:, :k].mean(axis=1) if k > 0
             else np.zeros(n))

    order = list(np.argsort(score, kind="stable"))
    split = _density_elbow(score[order])
    groups = [order[:split], order[split:]]

    texts = [None] * n
    fallback, fallback_idx = [], []
    obstacles = list(kwargs.pop("avoid_bboxes", ()))

    for group in groups:
        if not group:
            continue
        sub_texts, sub_fallback = _add_labels_dynamic(
            ax,
            [xs[i] for i in group], [ys[i] for i in group],
            [labels[i] for i in group],
            colors=None if colors is None else [colors[i] for i in group],
            styles=None if styles is None else [styles[i] for i in group],
            avoid_bboxes=obstacles,
            # dodge every marker, not just this group's; a caller may widen it
            # further (paired markers, say) by passing its own avoid_xy.
            avoid_xy=kwargs.pop("avoid_xy", None) or (xs, ys),
            adjust=False,               # one adjustText run at the end instead
            **kwargs,
        )
        deferred = {id(t) for t in sub_fallback}
        for slot, i in enumerate(group):
            text = sub_texts[slot]
            texts[i] = text
            if id(text) in deferred:
                fallback.append(text)
                fallback_idx.append(i)
            else:
                # Placed, so it is an obstacle for the next group.
                obstacles.append(
                    text.get_window_extent(renderer=renderer).expanded(1.02, 1.05))

    _adjust_fallback_labels(
        ax, fallback,
        x=xs, y=ys,
        target_x=[xs[i] for i in fallback_idx],
        target_y=[ys[i] for i in fallback_idx],
        objects=obstacles,
        arrowprops=arrowprops,
        at_force_text=at_force_text, at_force_pull=at_force_pull,
        at_force_static=at_force_static, at_max_move=at_max_move,
        at_only_move=at_only_move,
    )

    return texts, fallback


def _save_panel(fig: plt.Figure, axes, name: str, margin: float = 0.12) -> None:
    """Crop-save one or more axes from an already-drawn figure.

    Computes the union of the tight bounding boxes of *axes*, pads by
    *margin* inches, and writes PNG + SVG without closing the figure.
    Call this before the final ``_save()`` that closes the figure.

    No-op unless :data:`SAVE_PANELS` is set (``--subpannel``). Gating here
    rather than at the call sites means the composite figures need no guards
    of their own.
    """
    from matplotlib.transforms import Bbox

    if not SAVE_PANELS:
        return

    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    _set_uniform_text_size(fig, fontsize=_SCATTER_LABEL_FS)
    _sync_faded_backgrounds(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [ax.get_tightbbox(renderer) for ax in axes if ax is not None]
    union = Bbox.union(bboxes)
    inch = union.transformed(fig.dpi_scale_trans.inverted())
    padded = Bbox([[inch.x0 - margin, inch.y0 - margin],
                   [inch.x1 + margin, inch.y1 + margin]])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=150, bbox_inches=padded)
    print(f"  saved {name}.png / .svg")


_PANEL_LABEL_GID = "panel-label"
_PANEL_LABEL_FS: float = 16   # panel letters, the one exception to the uniform size


def _draw_panel_label(ax: plt.Axes, label: str, x: float = -0.08,
                      y: float = 1.03) -> None:
    """Draw a panel letter (a, b, c, …) in the axes' upper-left corner.

    Tagged with a gid so :func:`_set_uniform_text_size` can leave it at
    ``_PANEL_LABEL_FS`` while flattening every other text to the body size.
    """
    txt = ax.text(x, y, label, fontsize=_PANEL_LABEL_FS, fontweight="bold",
                  transform=ax.transAxes, va="top")
    txt.set_gid(_PANEL_LABEL_GID)


def _set_uniform_text_size(fig: plt.Figure, fontsize: float) -> None:
    """Force a uniform font size for all text objects in a figure.

    Panel letters are the single exception: they carry ``_PANEL_LABEL_GID`` and
    keep ``_PANEL_LABEL_FS`` so the panel hierarchy survives the harmonisation.
    Everything else — including hand-tuned schematic labels — is flattened, so
    schematics have to earn their legibility from geometry, not type size.
    """
    for txt in fig.findobj(match=mpl.text.Text):
        txt.set_fontsize(_PANEL_LABEL_FS
                         if txt.get_gid() == _PANEL_LABEL_GID else fontsize)


# Faded backgrounds are placed from a one-off measurement of the artist they sit
# behind, so they have to be re-measured once the figure layout is final. Keyed
# weakly so closed figures do not pile up. See _sync_faded_backgrounds().
_FADED_BG_SYNCS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _sync_faded_backgrounds(fig: plt.Figure) -> None:
    """Re-align every faded background in *fig* with the artist it sits behind.

    :func:`add_faded_background` can only measure an artist where it currently
    is, but the gradient is anchored in axes coordinates while text is sized in
    points and legends are anchored to the axes corner. Any later layout change
    — ``tight_layout``, a ``suptitle``, a font-size sweep — moves the artist
    relative to the frozen gradient and leaves the halo visibly offset and
    mis-sized. One draw settles the final layout (axes boxes and legend offsets
    alike); re-measuring after it puts each gradient back under its artist.
    """
    entries = _FADED_BG_SYNCS.get(fig)
    if not entries:
        return

    fig.canvas.draw()
    for _artist, _bg, sync in entries:
        sync()

    behind = _pixels_behind(fig, entries)
    for (artist, background, _sync), has_ink in zip(entries, behind):
        # A halo over blank paper is just a pale box on the page. Only keep the
        # ones that are actually masking something.
        background.set_visible(has_ink)


def _pixels_behind(fig: plt.Figure, entries: list) -> list[bool]:
    """For each entry, whether anything is drawn under its artist.

    Hides the halos *and* the artists they belong to, renders once, and asks
    whether each artist's footprint is still uniform in colour. Uniform means
    nothing but background was there, whatever that background happens to be —
    no assumption that the page is white.

    Falls back to "yes" for every entry if the canvas cannot be read, since a
    halo that is not needed costs less than one that is missing.
    """
    was_visible = [(a, a.get_visible()) for a, _, _ in entries]
    was_visible += [(b, b.get_visible()) for _, b, _ in entries]
    for art, _ in was_visible:
        art.set_visible(False)
    try:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    except Exception:
        for art, vis in was_visible:
            art.set_visible(vis)
        return [True] * len(entries)
    for art, vis in was_visible:
        art.set_visible(vis)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    height = buf.shape[0]

    out = []
    for artist, _background, _sync in entries:
        bb = artist.get_window_extent(renderer=renderer)
        # window extents count y upward from the bottom; the buffer counts down.
        x0, x1 = int(np.floor(bb.x0)), int(np.ceil(bb.x1))
        y0, y1 = int(np.floor(height - bb.y1)), int(np.ceil(height - bb.y0))
        x0, x1 = max(x0, 0), min(x1, buf.shape[1])
        y0, y1 = max(y0, 0), min(y1, height)
        region = buf[y0:y1, x0:x1]
        out.append(bool(region.size) and not bool((region == region[0, 0]).all()))
    return out


def _halo_alpha(w_px: float, h_px: float, border_px: float, max_alpha: float,
                max_side: int = 400) -> np.ndarray:
    """Alpha field for a *w_px* x *h_px* box wearing a *border_px* halo.

    The returned array covers the box grown by ``2 * border_px`` on every side.
    Alpha is ``max_alpha`` on the box itself and reaches 0 at the outer edge,
    easing with a smoothstep so it passes half strength exactly ``border_px``
    out — which is where a Gaussian blur of a hard edge sits.

    Distance to an axis-aligned rectangle has rounded-rectangle iso-contours, so
    this reproduces the geometry Scientific-Inkscape's ``text_ghoster.py`` uses
    (bounding box grown by ``EXTENT * font_size``, corner radius equal to the
    growth, blurred with ``sigma = STDDEV * growth``) without an actual
    convolution. A radial gradient — what this used to be — instead inscribes an
    ellipse in the box, so with the wide flat boxes text produces it left the
    ends and all four corners of the label uncovered.
    """
    reach = 2.0 * border_px
    box_w, box_h = max(w_px, 1.0), max(h_px, 1.0)
    total_w, total_h = box_w + 2 * reach, box_h + 2 * reach

    # Grid resolution tracks the box's pixel aspect, so stretching it over the
    # extent keeps the corner arcs circular instead of shearing them.
    scale = max(total_w, total_h) / max_side
    nx = int(np.clip(round(total_w / scale), 8, max_side))
    ny = int(np.clip(round(total_h / scale), 8, max_side))

    # Pixel centres, in a frame whose origin is the padded box's lower-left.
    x = (np.arange(nx) + 0.5) * total_w / nx
    y = (np.arange(ny) + 0.5) * total_h / ny
    dx = np.maximum(0.0, np.maximum(reach - x, x - (reach + box_w)))
    dy = np.maximum(0.0, np.maximum(reach - y, y - (reach + box_h)))
    d = np.hypot(dx[None, :], dy[:, None])

    t = np.clip(d / reach, 0.0, 1.0)
    return max_alpha * (1.0 - t * t * (3.0 - 2.0 * t))


def add_faded_background(ax, artist, max_alpha=0.75, extent=0.5,
                         color=(1.0, 1.0, 1.0)):
    """Add a soft rounded-rectangle halo behind an existing Matplotlib artist.

    Geometry follows Scientific-Inkscape's ``text_ghoster.py``: the artist's
    bounding box grown by ``border = extent * font_size``, corners rounded at
    that same radius, edge feathered as if Gaussian-blurred. See
    :func:`_halo_alpha`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the artist.
    artist : matplotlib.artist.Artist
        Artist to place the background behind.
    max_alpha : float
        Opacity over the artist itself. 0.75 matches Inkscape's ``OPACITY``;
        below ~0.7 a saturated line crossing the label still reads through.
    extent : float
        Halo thickness as a multiple of the artist's font size, so a long label
        and a short one get the same margin. Artists without a font size (legend
        handles) fall back to the figure's body size. 0.5 matches Inkscape's
        ``EXTENT``; the halo fades out by ``2 * extent`` font sizes.
    color : tuple
        RGB colour of the halo (default white).
    """
    fig = ax.figure

    # The artist has to have been laid out before it can be measured.
    fig.canvas.draw()

    def _geometry():
        bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
        fs_pt = (artist.get_fontsize() if hasattr(artist, "get_fontsize")
                 else _SCATTER_LABEL_FS)
        border = max(extent * fs_pt * fig.dpi / 72.0, 1.0)
        reach = 2.0 * border

        alpha = _halo_alpha(bb.width, bb.height, border, max_alpha)
        rgba = np.empty((*alpha.shape, 4))
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
        rgba[..., 3] = alpha

        inv = ax.transAxes.inverted()
        (x0, y0) = inv.transform((bb.x0 - reach, bb.y0 - reach))
        (x1, y1) = inv.transform((bb.x1 + reach, bb.y1 + reach))
        return rgba, (x0, x1, y0, y1)

    rgba, img_extent = _geometry()

    # imshow updates the data limits even under a transAxes transform, which
    # would rescale the surrounding plot; save and restore around it.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    background = ax.imshow(
        rgba,
        extent=img_extent,
        transform=ax.transAxes,
        origin="lower",
        interpolation="bilinear",
        aspect="auto",
        zorder=artist.get_zorder() - 1,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    def _sync() -> None:
        """Rebuild the halo around the artist at its current position and size."""
        new_rgba, new_extent = _geometry()
        keep_x, keep_y = ax.get_xlim(), ax.get_ylim()
        background.set_data(new_rgba)
        background.set_extent(new_extent)
        ax.set_xlim(keep_x)
        ax.set_ylim(keep_y)

    # (artist, halo, reposition) — the sync pass needs all three: the artist to
    # measure what is behind it, the halo to move and to show or hide.
    _FADED_BG_SYNCS.setdefault(fig, []).append((artist, background, _sync))

    return background


def _fit_left_xlim(ax: plt.Axes, pad_pts: float = 2.0) -> None:
    """Tighten the left x-limit to just clear the leftmost data-space text.

    Renders the figure once to measure actual glyph extents (accounting for
    font size, rotation, and number of characters), then pads by *pad_pts*
    display points. Only considers Text objects whose transform is
    ax.transData and whose x-position is left of x = 0.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    min_x_disp: float | None = None
    for txt in ax.texts:
        if txt.get_transform() is not ax.transData:
            continue
        if txt.get_position()[0] >= 0:
            continue
        try:
            bb = txt.get_window_extent(renderer)
            if min_x_disp is None or bb.x0 < min_x_disp:
                min_x_disp = bb.x0
        except Exception:
            pass
    if min_x_disp is None:
        return
    pad_px = pad_pts * renderer.dpi / 72.0
    new_left = float(inv.transform((min_x_disp - pad_px, 0))[0])
    ax.set_xlim(new_left, ax.get_xlim()[1])


def _fit_right_xlim(ax: plt.Axes, pad_pts: float = 2.0) -> None:
    """Tighten the right x-limit to just clear the rightmost data-space text.

    Mirror of _fit_left_xlim for the right side. Only considers Text objects
    whose transform is ax.transData and whose x-position is right of x = 1.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    max_x_disp: float | None = None
    for txt in ax.texts:
        if txt.get_transform() is not ax.transData:
            continue
        if txt.get_position()[0] <= 1:
            continue
        try:
            bb = txt.get_window_extent(renderer)
            if max_x_disp is None or bb.x1 > max_x_disp:
                max_x_disp = bb.x1
        except Exception:
            pass
    if max_x_disp is None:
        return
    pad_px = pad_pts * renderer.dpi / 72.0
    new_right = float(inv.transform((max_x_disp + pad_px, 0))[0])
    ax.set_xlim(ax.get_xlim()[0], new_right)


def plot_fig1_schematic() -> None:
    """Figure 1: three-panel conceptual overview of the validation resource."""
    fig = plt.figure(figsize=(17, 6))
    axes = fig.subplots(1, 3)

    # ── Panel A: Experimental design ──────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A   Experimental design", loc="left", fontweight="bold", fontsize=_SCATTER_LABEL_FS)

    def box(ax, x, y, w, h, color, label, fontsize=_SCATTER_LABEL_FS):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="grey", linewidth=0.8, zorder=2))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=_SCATTER_LABEL_FS, zorder=3)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    box(ax, 0.5, 7.0, 4.0, 1.5, "#aec7e8", f"{DS_EH}\n(entrapment dataset)")
    box(ax, 0.5, 4.0, 4.0, 1.5, "#ffbb78", f"{DS_EM}\n(mixed-species dataset)")

    box(ax, 6.0, 8.5, 3.5, 0.8, "#c5e1a5", "Decoy FDR", fontsize=_SCATTER_LABEL_FS)
    box(ax, 6.0, 7.2, 3.5, 0.8, "#fff176", "Entrapment FDR\n(r ≈ 1)", fontsize=_SCATTER_LABEL_FS)
    box(ax, 6.0, 5.9, 3.5, 0.8, "#ef9a9a", f"Impossible FPR\n({SP_HUMAN} proteins)", fontsize=_SCATTER_LABEL_FS)

    box(ax, 6.0, 4.5, 3.5, 0.8, "#c5e1a5", "Decoy FDR", fontsize=_SCATTER_LABEL_FS)
    box(ax, 6.0, 3.2, 3.5, 0.8, "#fff176", "Entrapment FDR\n(TW+WW model)", fontsize=_SCATTER_LABEL_FS)
    box(ax, 6.0, 1.9, 3.5, 0.8, "#ef9a9a", "Impossible FPR\n(cross-species)", fontsize=_SCATTER_LABEL_FS)

    for y in (8.9, 7.6, 6.3):
        arrow(ax, 4.5, 7.75, 6.0, y)
    for y in (4.9, 3.6, 2.3):
        arrow(ax, 4.5, 4.75, 6.0, y)

    # ── Panel B: TW+WW model ──────────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("B   TW+WW entrapment model", loc="left", fontweight="bold", fontsize=_SCATTER_LABEL_FS)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    labels = ["SS\n(same species)", "SM\n(one wrong)", "MM\n(both wrong)"]
    colors_tt = ["#c5e1a5", "#fff176", "#ef9a9a"]
    widths    = [0.55, 0.30, 0.15]
    x = 0.02
    for label, color, w in zip(labels, colors_tt, widths):
        ax.add_patch(mpatches.Rectangle((x, 0.55), w - 0.02, 0.30,
                                         facecolor=color, edgecolor="grey", lw=0.8))
        ax.text(x + (w - 0.02)/2, 0.70, label, ha="center", va="center", fontsize=_SCATTER_LABEL_FS)
        x += w

    ax.text(0.5, 0.50, "← only WW false positives reach MM →",
            ha="center", va="top", fontsize=_SCATTER_LABEL_FS, style="italic", color="grey")
    ax.text(0.5, 0.42,
            "FP in SS  =  SM/r  −  MM/r²\n"
            "TotalFP  =  FP_SS  +  SM  +  MM",
            ha="center", va="top", fontsize=_SCATTER_LABEL_FS, color="grey",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="grey"))
    ax.text(0.5, 0.18,
            "r  =  N(wrong-species) / N(right-species)\nfrom FASTA (per direction)",
            ha="center", va="top", fontsize=_SCATTER_LABEL_FS, color="#555555")

    # ── Panel C: Pipeline flow ────────────────────────────────────────────────
    ax = axes[2]
    ax.set_title("C   Normalisation pipeline", loc="left", fontweight="bold", fontsize=_SCATTER_LABEL_FS)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    steps = [
        (5.0, 9.0, "#e3f2fd", "11 search engines\n(raw output)"),
        (5.0, 7.0, "#e8f5e9", "Common schema\n(loaders + pipeline.py)"),
        (5.0, 5.0, "#fff8e1", "combined.parquet\n(per dataset/level)"),
        (5.0, 3.0, "#fce4ec", "Three estimators\n(summarise.py)"),
        (5.0, 1.0, "#f3e5f5", "Figures + benchmark\n(manuscript_plots.py)"),
    ]
    # reuses box/arrow helpers defined in panel A above
    for x, y, color, label in steps:
        box(ax, x - 3.5, y - 0.6, 7.0, 1.2, color, label, fontsize=_SCATTER_LABEL_FS)
    for i in range(len(steps) - 1):
        arrow(ax, 5.0, steps[i][1] - 0.6, 5.0, steps[i+1][1] + 0.6)

    fig.tight_layout()
    _save(fig, "fig1_resource_schematic")


# Figure 2 grid: rows = datasets, cols = level (CSM / PPI). Every panel plots
# impossible FPR (y) against decoy FDR (x); two axis treatments are produced.
_FIG2_DATASETS = ((DS_EH, "eh"), (DS_EM, "em"))
_FIG2_XCOL, _FIG2_YCOL = "decoy_fdr_pct", "tt_impossible_fpr_pct"
_FIG2_XLABEL, _FIG2_YLABEL = ("Observed Decoy FDR (%)",
                              "Observable false-target rate (%)")
# Axis floor for the sqrt variant, matching fig_multi2 panel a. Points at or
# below this (notably a reported decoy FDR of 0) move to the "undef." column.
_FIG2_SQRT_FLOOR = 0.5
# log2 cannot represent 0, so both axes need a floor. x=1.0 clears the smallest
# non-zero decoy FDR (1.445%); y=0.25 clears the smallest impossible FPR
# (0.468%, Scout at E.coli–Human PPI).
_FIG2_LOG2_XFLOOR = 1.0
_FIG2_LOG2_YFLOOR = 0.25
# Share of the x-axis given to the "undef." column. Applied in transformed
# space so the column is the same width in every panel and on either scale.
_FIG2_UNDEF_FRAC = 0.15
# Every point is drawn at this size. Marker area is deliberately not a data
# channel here: the two axes already carry the comparison, and sizing by TT
# count made large-n engines dominate visually without being any more reliable.
# Shared by fig2_impossible_vs_decoy* and fig_multi2 panel a so the two match.
_FIG2_MARKER_SIZE = 60


# Engine colours for fig2_impossible_vs_decoy* and fig_multi2 panel a.
_FIG2_PALETTE_L  = 65.0   # CIE L* — one lightness for every engine
_FIG2_PALETTE_C  = 38.0   # CIE C* — one chroma for every engine
_FIG2_PALETTE_H0 = 20.0   # hue of the first engine, degrees


def _lch_to_hex(L: float, C: float, h_deg: float) -> str:
    """CIE LCh(ab) -> sRGB hex, clipped into gamut."""
    h = np.deg2rad(h_deg)
    a, b = C * np.cos(h), C * np.sin(h)
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    unf = lambda t: t ** 3 if t ** 3 > 0.008856 else (t - 16.0 / 116.0) / 7.787
    xyz = np.array([unf(fx), unf(fy), unf(fz)]) * np.array([0.95047, 1.0, 1.08883])
    lin = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689,  1.8758,  0.0415],
                    [0.0557,  -0.2040,  1.0570]]) @ xyz
    lin = np.clip(lin, 0.0, 1.0)   # clip before the transfer function, not after
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return mpl.colors.to_hex(np.clip(srgb, 0.0, 1.0))


def _fig2_palette() -> dict:
    """One colour per engine: evenly spaced in hue at a single L* and C*.

    This used to sample ``cm.tab20`` at ``i / len(ENGINE_ORDER)``. tab20 is ten
    hues in two tiers — one saturated, one pale — so those steps landed on both
    tiers arbitrarily: lightness spanned L* 42-83 and chroma 0-79, which left
    ProteinProspector (pure red, C* 79) shouting and pLink3 (pure grey, C* 0)
    invisible for reasons unrelated to the data. Holding L* and C* fixed removes
    that unintended emphasis and leaves hue as the only difference.

    L* 65 rather than a lighter, more pastel value: engine labels are drawn in
    the engine's own colour, so lightness is bounded below by text legibility.
    Colour only has to be distinguishable here, not memorable — every point
    carries an inline label. Note that no 12-colour categorical palette is
    colour-vision-deficiency safe (minimum pairwise ΔE under deuteranopia is
    ~4 for this one and for tab20 alike); the labels carry identity.
    """
    n = len(ENGINE_ORDER)
    return {e: _lch_to_hex(_FIG2_PALETTE_L, _FIG2_PALETTE_C,
                           _FIG2_PALETTE_H0 + 360.0 * i / n)
            for i, e in enumerate(ENGINE_ORDER)}


def _fig2_scatter(ax: plt.Axes, df: pd.DataFrame, level: str, palette: dict,
                  *, annotate: bool = True, marker_size: int = _FIG2_MARKER_SIZE,
                  zero_is_undefined: bool = False, pair_df: pd.DataFrame | None = None,
                  defer_zero_y: bool = False,
                  ) -> tuple[list[tuple], list, list[tuple], list[tuple]]:
    """Scatter one (dataset, level) subset onto ``ax``.

    Returns (pts, texts, undef_pts, specs, zero_pts). pts is a list of (x, y); texts is
    the list of annotation objects (empty when annotate=False); specs is a list
    With ``defer_zero_y`` a y of exactly 0 — which a log axis cannot show — is
    not drawn at all; it is returned in *zero_pts* for the caller to place once
    the axis limits are known. Such points take no part in those limits, so a
    single zero cannot drag the floor down.

    ``pair_df`` adds a second y per engine from another frame, drawn as an open
    marker joined to the filled one by a line in the engine's colour. Both ends
    land in *pts*, so a caller can keep labels clear of either.

    specs is a list
    of (x, y, label, colour) for callers that place labels themselves — e.g. via
    :func:`_add_labels_dynamic`, which wants coordinates rather than ready-made
    artists, and is collected regardless of *annotate*.
    undef_pts is a list of (eng, y) for engines whose x is undefined; the caller
    places these in a dedicated right-side column. Mango is always undefined
    (decoy-free). With ``zero_is_undefined`` a reported decoy FDR of exactly 0
    joins them — required on a sqrt axis floored above 0, where x=0 is off-scale.
    """
    sub = df[df["level"] == level]
    pts: list[tuple] = []
    texts: list = []
    undef_pts: list[tuple] = []
    specs: list[tuple] = []
    zero_pts: list[tuple] = []
    rows = []
    for _, r in sub.iterrows():
        eng = r["search_engine"]
        x, y = r.get(_FIG2_XCOL, float("nan")), r.get(_FIG2_YCOL, float("nan"))
        if pd.isna(x) or pd.isna(y):
            continue
        rows.append((eng, float(x), float(y)))

    for eng, x, y in rows:
        label = engine_label(eng, level)
        if defer_zero_y and y == 0:
            zero_pts.append((eng, x))
            continue
        if eng == "mango" or (zero_is_undefined and x <= 0):
            undef_pts.append((eng, y))
            continue
        color = palette.get(eng, "grey")

        if pair_df is not None:
            alt = pair_df[(pair_df["level"] == level)
                          & (pair_df["search_engine"] == eng)]
            y2 = (float(alt.iloc[0][_FIG2_YCOL])
                  if not alt.empty and not pd.isna(alt.iloc[0].get(_FIG2_YCOL))
                  else None)
            if y2 is not None and abs(y2 - float(y)) > 1e-9:
                ax.plot([x, x], [y, y2], color=color, lw=1.0, alpha=0.75, zorder=2)
                ax.scatter(x, y2, marker="o", facecolors="none", edgecolors=color,
                           s=marker_size, linewidths=1.2, zorder=3)
                pts.append((x, y2))

        ax.scatter(x, y, marker="o", color=color, s=marker_size, zorder=3,
                   edgecolors="white", linewidths=0.4)
        if annotate:
            t = ax.text(x, y, label, fontsize=_SCATTER_LABEL_FS,
                        ha="left", va="bottom", color=color)
            texts.append(t)
        specs.append((x, y, label, color))
        pts.append((x, y))
    return pts, texts, undef_pts, specs, zero_pts


def _fig2_refs(ax: plt.Axes, lim: float, lo: float = 0.0) -> None:
    """Draw the y=x diagonal, the 2% reference lines, and their axis labels.

    ``lo`` is where the diagonal starts — required on log axes, which cannot
    represent the default 0.
    """
    # One style for every reference line in the panel: the diagonal and the two
    # 2% guides carry the same weight of meaning, so they get the same weight of
    # ink. Defined once so they cannot drift apart.
    ref = dict(color="grey", lw=0.8, ls="--", alpha=0.6, zorder=1)
    ax.plot([lo, lim], [lo, lim], **ref)
    ax.axhline(REF_FDR, **ref)
    ax.axvline(REF_FDR, **ref)
    # Label the 2% lines at the axis edges using blended transforms, but only
    # where no tick already sits at 2 — on log2 axes one does, and the two
    # labels would overprint. Call this after the ticks are set.
    # get_yaxis_transform: x in axes fraction [0,1], y in data coords
    if not any(np.isclose(t, REF_FDR) for t in ax.get_yticks()):
        ax.text(-0.01, REF_FDR, "2%", ha="right", va="center", fontsize=_SCATTER_LABEL_FS,
                color="grey", alpha=0.9, transform=ax.get_yaxis_transform(),
                clip_on=False)
    # get_xaxis_transform: x in data coords, y in axes fraction [0,1]
    if not any(np.isclose(t, REF_FDR) for t in ax.get_xticks()):
        ax.text(REF_FDR, -0.04, "2%", ha="center", va="top", fontsize=_SCATTER_LABEL_FS,
                color="grey", alpha=0.9, transform=ax.get_xaxis_transform(),
                clip_on=False)


def _draw_axis_break(ax: plt.Axes, pos: float, *, axis: str = "x",
                     size: float = 13.0, offset_pt: float = 2.0) -> None:
    """Mark a scale break in the x- or y-spine at *pos* with a "//" symbol.

    Three short diagonal strokes layered at the same spot: a wide white one
    first, which paints over the spine directly behind it, then two
    narrower ones — in the spine's own colour and linewidth, so the break
    reads as a cut in that line rather than a separate black mark — offset
    a couple of points apart along the spine. The usual typographic break
    mark for "this column/row is not a continuation of the numeric axis"
    (here, the ``undef.`` column or row next to it). Drawn with a custom
    two-point marker path rather than a scaled line so the stroke stays a
    fixed size in points regardless of the axes' data range, log scale, or
    aspect ratio.

    ``axis="x"`` pins the mark to the bottom spine at data x=*pos*
    (``ax.get_xaxis_transform()``: x in data, y in axes fraction).
    ``axis="y"`` pins it to the left spine at data y=*pos*, with the whole
    mark rotated 90°.
    """
    spine = ax.spines["bottom" if axis == "x" else "left"]
    spine_color = spine.get_edgecolor()
    spine_lw = spine.get_linewidth()
    if axis == "x":
        diag = [(-0.35, -1.0), (0.35, 1.0)]
        base = ax.get_xaxis_transform()
        anchor = ([pos], [0])
        offsets = [ScaledTranslation(sign * offset_pt / 72.0, 0, ax.figure.dpi_scale_trans)
                   for sign in (-1, 1)]
    else:
        diag = [(-1.0, -0.35), (1.0, 0.35)]
        base = ax.get_yaxis_transform()
        anchor = ([0], [pos])
        offsets = [ScaledTranslation(0, sign * offset_pt / 72.0, ax.figure.dpi_scale_trans)
                   for sign in (-1, 1)]
    common = dict(marker=diag, linestyle="none", clip_on=False, zorder=10)
    ax.plot(*anchor, transform=base, markersize=size,
            markeredgewidth=spine_lw * 3.5, markeredgecolor="white", **common)
    for offset in offsets:
        ax.plot(*anchor, transform=base + offset, markersize=size,
                markeredgewidth=spine_lw, markeredgecolor=spine_color, **common)


def _fig2_legend(fig: plt.Figure, palette: dict) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color=palette.get(e, "grey"),
                   linestyle="None", markersize=6, label=ENGINE_LABELS.get(e, e))
        for e in ENGINE_ORDER
    ]
    fig.legend(handles=handles, loc="center right", fontsize=_SCATTER_LABEL_FS, ncol=1,
               bbox_to_anchor=(1.01, 0.5), frameon=True, title="Search engine")


def plot_fig2_crossvalidation(eh: pd.DataFrame, em: pd.DataFrame) -> None:
    """Figure 2: impossible FPR vs decoy FDR — sqrt, log2 and inset variants."""
    #plot_fig2_impossible_sqrt(eh, em)
    plot_fig2_impossible_log2(eh, em)
    #plot_fig2_impossible_pair(eh, em)
    #plot_fig2_impossible_pair(eh, em, scale="sqrt")
    # plot_fig2_impossible_inset(eh, em)


def plot_fig2_impossible_sqrt(eh: pd.DataFrame, em: pd.DataFrame) -> None:
    """Figure 2 (sqrt): square-root axes — see :func:`_plot_fig2_impossible`."""
    _plot_fig2_impossible(_eh_impossible_entrapment_only(eh), em, scale="sqrt")


def plot_fig2_impossible_pair(eh: pd.DataFrame, em: pd.DataFrame, scale="log2") -> None:
    """Figure 2 showing both impossible definitions per engine, joined by a line.

    Filled marker: the search-internal criterion alone — a human entrapment
    match, or a wrong-organism spectrum. Open marker: the same plus the iBAQ
    co-fractionation test. Decoy FDR does not depend on the definition, so the
    pair is a vertical segment, and its length is exactly what the
    co-fractionation criterion adds for that engine.
    """
    _plot_fig2_impossible(
        _eh_impossible_entrapment_only(eh), em, scale=scale,
        name_suffix="_pair", pair={"eh": eh, "em": _em_impossible_with_copresence(em)})


def plot_fig2_impossible_log2(eh: pd.DataFrame, em: pd.DataFrame) -> None:
    """Figure S4 (log2 axes) — see :func:`_plot_fig2_impossible`."""
    _plot_fig2_impossible(_eh_impossible_entrapment_only(eh), em, scale="log2",
                          save_name="figS4_Observable_error_vs_observed_decoy")


def _plot_fig2_impossible(eh: pd.DataFrame, em: pd.DataFrame, *, scale: str,
                          name_suffix: str = "", pair: dict | None = None,
                          save_name: str | None = None) -> None:
    """Impossible FPR vs decoy FDR, 2×2 (dataset × level), on compressed axes.

    Engines are labelled inline instead of via a colour legend. The y=x line
    marks where the observable error equals the reported FDR; points above it
    under-report. Mango (no decoys) and any
    engine reporting exactly 0% decoy FDR go in a dedicated column right of a
    visual separator — both are undefined on x rather than meaningfully at 0.

    ``scale`` selects the axis compression, both of which expand the dense
    low-value cluster near the 2% reference while keeping the high-FPR outliers
    on-scale:

    - ``"sqrt"`` — handles y = 0 natively, so the y-axis still reaches 0.
    - ``"log2"`` — cannot represent 0, so y is floored at 0.25. That is below
      the smallest observed value (Scout, 0.468% at E.coli–Human PPI); panel a's
      0.5 floor would clip it, as panel a only ever plots CSM data.
    """
    sns.set_theme(style="white", font_scale=0.85)
    frames = {"eh": eh, "em": em}
    palette = _fig2_palette()
    # Optional second reading per engine, drawn as an open marker on a stem from
    # the filled one. Keyed like `frames`, so each row picks up its own.
    pair_frames = pair or {}

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for row, (ds_label, key) in enumerate(_FIG2_DATASETS):
        df = frames[key]
        for col, level in enumerate(("CSM", "PPI")):
            ax = axes[row][col]
            pts, texts, undef_pts, specs, zero_pts = _fig2_scatter(
                ax, df, level, palette, annotate=False, zero_is_undefined=True,
                pair_df=pair_frames.get(key), defer_zero_y=(scale == "log2"))

            # Every y that will be drawn, including the open end of a pair and
            # the undefined column, so the floor below can clear all of them.
            y_seen = [q[1] for q in pts] + [u[1] for u in undef_pts]
            if pair_frames.get(key) is not None:
                alt_all = pair_frames[key]
                for _eng, _ in undef_pts:
                    hit = alt_all[(alt_all["level"] == level)
                                  & (alt_all["search_engine"] == _eng)]
                    if not hit.empty and not pd.isna(hit.iloc[0].get(_FIG2_YCOL)):
                        y_seen.append(float(hit.iloc[0][_FIG2_YCOL]))

            lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05

            if scale == "log2":
                x_lo, y_lo = _FIG2_LOG2_XFLOOR, _FIG2_LOG2_YFLOOR
                # Drop the floor to the power of two below the smallest value
                # rather than clipping it: on a log axis a fixed floor silently
                # hides anything under it.
                _pos = [v for v in y_seen if v > 0]
                if _pos and min(_pos) < y_lo:
                    y_lo = float(2.0 ** np.floor(np.log2(min(_pos))))
                    # A value sitting exactly on a power of two would land on
                    # the floor, where the zeros go; drop one more octave so
                    # the two cannot be confused.
                    if y_lo >= min(_pos):
                        y_lo /= 2.0
                ax.set_xscale("log", base=2)
                ax.set_yscale("log", base=2)
                all_ticks = tuple(
                    2.0 ** k for k in range(int(np.floor(np.log2(min(x_lo, y_lo)))), 7))
                fwd, inv = np.log2, np.exp2
            else:
                # sqrt scale: forward=sqrt, inverse=square (both valid on [0, ∞)).
                # Only x is floored; y still reaches 0, so engines sitting near
                # the floor (Scout, 0.47% at E.coli–Human PPI) are not clipped.
                x_lo, y_lo = _FIG2_SQRT_FLOOR, 0.0
                ax.set_xscale("function", functions=(np.sqrt, np.square))
                ax.set_yscale("function", functions=(np.sqrt, np.square))
                all_ticks = (0, 1, 4, 9, 16, 25, 36, 49, 64)
                fwd, inv = np.sqrt, np.square

            # Undefined column, sized in transformed (screen) space rather than
            # as a multiple of lim, so it occupies the same fraction of the
            # x-axis in every panel whatever that panel's data range or scale.
            if undef_pts:
                t_lo, t_lim = fwd(x_lo), fwd(lim)
                t_total = (t_lim - _FIG2_UNDEF_FRAC * t_lo) / (1 - _FIG2_UNDEF_FRAC)
                gap = t_total - t_lim
                sep_x   = float(inv(t_lim + 0.30 * gap))
                mango_x = float(inv(t_lim + 0.62 * gap))
                x_total = float(inv(t_total))
            else:
                sep_x = mango_x = x_total = lim

            ax.set_xlim(x_lo, x_total)
            ax.set_ylim(y_lo, lim)
            ax.set_xticks([t for t in all_ticks if x_lo <= t <= lim])
            ax.set_yticks([t for t in all_ticks if y_lo <= t <= lim])

            # Zeros sit exactly on the floor, and the floor's tick then reads
            # "undef." — a 0 %% observable FP rate is not a measurement of
            # zero error, it is the absence of any observable case. Placing them
            # on the axis line rather than just above it keeps that distinction
            # visible, and they were kept out of the limit above so one zero
            # cannot pull the whole axis down.
            if scale == "log2":
                # "%g" rather than ScalarFormatter: the default needs set_locs()
                # to have run or it yields empty strings, and it renders these
                # powers of two as 0.06 / 32.00 where 0.0625 / 32 is both
                # shorter and exact.
                has_zero = bool(zero_pts) or any(y == 0 for _, y in undef_pts)
                ax.xaxis.set_major_formatter(
                    mpl.ticker.FuncFormatter(lambda v, pos: f"{v:g}"))
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                    lambda v, pos, _lo=y_lo, _z=has_zero:
                        "undef." if (_z and abs(v - _lo) < 1e-12) else f"{v:g}"))
                if has_zero:
                    _draw_axis_break(ax, y_lo+0.05, axis="y")

            for eng, zx in zero_pts:
                color = palette.get(eng, "grey")
                ax.scatter(zx, y_lo, marker="o", color=color, s=_FIG2_MARKER_SIZE,
                           zorder=3, edgecolors="white", linewidths=0.4,
                           clip_on=False)
                specs.append((zx, y_lo, engine_label(eng, level), color))
                pts.append((zx, y_lo))

            # After the ticks: _fig2_refs suppresses its 2% labels where a tick
            # already sits at 2. y=x is only drawn where both axes are in range.
            diag_lo = max(x_lo, y_lo)
            _fig2_refs(ax, lim, lo=diag_lo)

            # No quadrant shading: the y=x line already marks the boundary,
            # and the fills read as a verdict on the whole region rather than
            # on the points in it.
            ax.set_xlabel(_FIG2_XLABEL, fontsize=_SCATTER_LABEL_FS)
            ax.set_ylabel(_FIG2_YLABEL, fontsize=_SCATTER_LABEL_FS)
            if row == 0:
                ax.set_title(level, fontsize=_SCATTER_LABEL_FS)

            # Undefined-x engines in their own column — added to `specs` so the
            # label pass below sees every point at once.
            if undef_pts:
                ax.axvline(sep_x, color="grey", ls=":", lw=0.5, alpha=0.4)
                _draw_axis_break(ax, sep_x)
                alt_df = pair_frames.get(key)
                for eng, y in undef_pts:
                    if y == 0:
                        y = y_lo
                    color = palette.get(eng, "grey")
                    if alt_df is not None:
                        alt = alt_df[(alt_df["level"] == level)
                                     & (alt_df["search_engine"] == eng)]
                        if not alt.empty and not pd.isna(alt.iloc[0].get(_FIG2_YCOL)):
                            y2 = float(alt.iloc[0][_FIG2_YCOL])
                            if abs(y2 - float(y)) > 1e-9:
                                ax.plot([mango_x, mango_x], [y, y2], color=color,
                                        lw=1.0, alpha=0.75, zorder=2)
                                ax.scatter(mango_x, y2, marker="o", facecolors="none",
                                           edgecolors=color, s=_FIG2_MARKER_SIZE,
                                           linewidths=1.2, zorder=3)
                                pts.append((mango_x, y2))
                    ax.scatter(mango_x, y, marker="o", color=color,
                               s=_FIG2_MARKER_SIZE,
                               zorder=3, edgecolors="white", linewidths=0.4)
                    specs.append((mango_x, y, engine_label(eng, level), color))
                ax.text(mango_x, -0.04, "undef.", ha="center", va="top",
                        fontsize=_SCATTER_LABEL_FS, transform=ax.get_xaxis_transform(),
                        clip_on=False)

            if specs:
                # Preferred positions first, adjustText only for what will not
                # fit. Most engines are isolated enough for a deterministic
                # placement, which also makes the figure reproducible run to run
                # — bare adjustText gave a different layout on every render.
                lx, ly, llabels, lcolors = zip(*specs)
                # at_force_pull well below the default: at 0.5 the fallback
                # labels stay tethered so tightly that Scout (FL) and Scout (DB)
                # print on top of each other in the E. coli-H. sap. PPI panel.
                # 0.05 separates them; lower again changes nothing visible.
                _add_labels_by_density(
                    ax, lx, ly, llabels,
                    offset=5, fontsize=_SCATTER_LABEL_FS, colors=lcolors,
                    # `pts` carries both ends of every pair, so labels keep
                    # clear of the open markers too.
                    avoid_xy=([q[0] for q in pts], [q[1] for q in pts]),
                    at_force_pull=(0.05, 0.05),
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
            sns.despine(ax=ax)
        axes[row][0].set_ylabel(f"{ds_label}\n{_FIG2_YLABEL}", fontsize=_SCATTER_LABEL_FS)

    if pair_frames:
        axes[0][0].legend(
            handles=[
                plt.Line2D([], [], marker="o", color="#333333", ls="none",
                           markersize=7, label="entrapment / wrong species only"),
                plt.Line2D([], [], marker="o", color="#333333", ls="none",
                           markersize=7, markerfacecolor="none",
                           label="+ never co-fractionating"),
            ],
            loc="lower right", fontsize=_SCATTER_LABEL_FS, frameon=True)

    # No legend at all: every point is labelled inline (as in panel a), and the
    # quadrant background colours are explained in the figure caption.
    fig.tight_layout()
    _save(fig, save_name or f"fig2_impossible_vs_decoy_{scale}{name_suffix}")


def _eh_impossible_entrapment_only(eh: pd.DataFrame) -> pd.DataFrame:
    """Return *eh* with ``tt_impossible_fpr_pct`` restricted to the entrapment criterion.

    The E.coli-Human impossible rate normally combines two criteria: a TT match
    involving a human protein (impossible because no human protein is in the
    sample) and a match between two E. coli proteins that never co-eluted above
    the iBAQ threshold in any SEC fraction. Only the first is an impossibility;
    the second is an abundance argument, and a protein that was never quantified
    at all is indistinguishable from one that was quantified but never
    co-present.

    The main figures therefore use the entrapment criterion alone, so the
    lower-bound claim they carry holds without further assumption. The
    co-fractionation criterion is retained in the supplementary figures, which
    report the two criteria separately.

    Matches that were impossible only by co-fractionation are counted as
    possible here; the denominator (all TT matches) is unchanged.
    """
    out = eh.copy()
    tt = out["TT"].replace(0, float("nan"))
    out["tt_impossible_fpr_pct"] = 100 * out["TT_impossible_human"] / tt
    return out


def _em_copresence_detail(em: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Per (engine, level) breakdown of what adding the co-presence term to
    the standard E.coli-Mycoplasma species-origin rule changes.

    The standard rule counts a TT match impossible only when its species
    assignment conflicts with the spectrum's organism of origin. It therefore
    treats every E. coli-E. coli match on an E. coli spectrum as possible —
    including pairs whose two proteins never co-occur above the iBAQ
    threshold in any SEC fraction, which the E.coli-Human dataset already
    counts as impossible (``impossible_reason = 'not_copresent'``). Both
    datasets use the same E. coli lysate, so the co-presence table applies to
    both.

    Adding that term makes the two datasets use one definition of
    "impossible", at the cost of leaning on quantitative data external to the
    search — reported as a separate figure/export rather than folded into the
    main summaries, which keep the stricter, search-internal definition.

    Mycoplasma-Mycoplasma pairs are left alone: the co-presence table covers
    the E. coli experiment only.

    Returns ``{(engine, level): {"extra": int, "impossible_total": int,
    "denom": int}}`` — ``extra`` is the count of matches newly flagged
    impossible by the co-presence term alone (disjoint from the standard
    impossible set by construction, since it's only tested among rows the
    standard rule already called possible). Shared by
    :func:`_em_impossible_with_copresence` (which only needs the resulting
    percentage) and the OFT-table CSV export (which needs the raw count too).
    """
    from src.analysis.ibaq_copresence import _check_copresence, load_ibaq_presence

    ibaq = load_ibaq_presence(REPO_ROOT / "input" / "proteinGroups.txt")
    lookup = load_merged_lookup(REPO_ROOT / "input" / "merged_files.json")

    def _not_copresent(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Rows within *mask* whose two proteins never share a fraction."""
        if not mask.any():
            return pd.Series(False, index=frame.index)
        hit = frame.loc[mask].apply(
            lambda r: not _check_copresence(r["Protein1"], r["Protein2"], ibaq), axis=1)
        out = pd.Series(False, index=frame.index)
        out.loc[hit[hit].index] = True
        return out

    detail: dict[tuple[str, str], dict] = {}
    for level in ("CSM", "PPI"):
        parquet = NORMALISED_DIR / "ecoli_mycpn" / level / "combined.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        df = df[~df["search_engine"].isin(EXCLUDED_ENGINES)].copy()
        df = add_decoy_category(df)

        for eng, g in df.groupby("search_engine"):
            if level == "CSM":
                scan_type = ENGINE_SCAN_TYPE.get(eng)
                if scan_type is None:
                    continue
                g = add_spectrum_origin(g.copy(), lookup, scan_type)
                g = g[g["decoy_cat"] == "TT"]
                from_ecoli = g["spectrum_origin"] == "ecoli"
                from_mycpn = g["spectrum_origin"] == "mycoplasma"
                # Class in the direction of the spectrum's own organism; rows of
                # unknown origin are counted in neither, as in fdr.py.
                cls = pd.Series(pd.NA, index=g.index, dtype=object)
                cls[from_ecoli] = g.loc[from_ecoli, "class_ecoli_spec"]
                cls[from_mycpn] = g.loc[from_mycpn, "class_mycpn_spec"]
                impossible = cls.isin(["SM", "MM"]).fillna(False)
                possible = (cls == "SS").fillna(False)
                extra = _not_copresent(g, possible & from_ecoli)
            else:
                g = g[g["decoy_cat"] == "TT"]
                impossible = g["match_origin"] == "mixed"
                possible = ~impossible
                extra = _not_copresent(g, possible & (g["match_origin"] == "ecoli"))

            impossible_total = impossible | extra
            possible_final = possible & ~extra
            denom = int(impossible_total.sum()) + int(possible_final.sum())
            detail[(eng, level)] = {
                "extra": int(extra.sum()),
                "impossible_total": int(impossible_total.sum()),
                "denom": denom,
            }
    return detail


def _em_impossible_with_copresence(em: pd.DataFrame) -> pd.DataFrame:
    """Return *em* with ``tt_impossible_fpr_pct`` recomputed to include
    E. coli-E. coli pairs that were never co-present — see
    :func:`_em_copresence_detail` for the definition.
    """
    detail = _em_copresence_detail(em)
    new_pct = []
    for e, l, v in zip(em["search_engine"], em["level"], em["tt_impossible_fpr_pct"]):
        d = detail.get((e, l))
        if d is None:
            new_pct.append(v)
        elif d["denom"]:
            new_pct.append(100 * d["impossible_total"] / d["denom"])
        else:
            new_pct.append(float("nan"))
    out = em.copy()
    out["tt_impossible_fpr_pct"] = new_pct
    return out


def export_oft_tables(eh: pd.DataFrame, em: pd.DataFrame,
                      out_dir: Path = NORMALISED_DIR) -> None:
    """Write 4 CSVs (one per dataset x level) with each pipeline's raw
    TT/TD/DD counts, observed decoy FDR, and the OFT breakdown its OFTR is
    built from: the search-internal criterion alone, and that criterion plus
    the (E. coli-only) co-presence term combined.

    Columns: search_pipeline, TT, TD, DD, decoy_fdr, OFT_species,
    OFTR_species, OFT_not_copresent, OFTR_combined. The four rate columns
    (``decoy_fdr``, ``OFTR_species``, ``OFTR_combined``) are fractions (0-1),
    not percentages — unlike every ``*_pct`` column elsewhere in this project.

    ``OFT_species``/``OFTR_species`` are the search-internal criterion alone
    — a human entrapment match for ecoli_human, a wrong-organism
    spectrum/protein for ecoli_mycpn — matching what every main figure plots
    as the OFTR. ``OFT_not_copresent`` is the additional count the
    co-presence term alone contributes (E. coli pairs that never co-eluted
    above the iBAQ threshold in any SEC fraction); ``OFTR_combined`` is the
    resulting rate with both criteria applied together. For ecoli_human this
    combined rate is already what the summary CSV calls
    ``tt_impossible_fpr``; for ecoli_mycpn it requires re-deriving the
    co-presence term via :func:`_em_copresence_detail`, since the main
    summary only carries the species-only count.

    Rows follow ``ENGINE_ORDER`` (the same engine set and order as every main
    figure); PPI-only alternate derivations (``*_csm`` roll-ups) are not
    separate rows here, matching that same convention.
    """
    em_detail = _em_copresence_detail(em)

    for ds_label, df, is_eh in (("ecoli_human", eh, True), ("ecoli_mycpn", em, False)):
        for level in ("CSM", "PPI"):
            sub = df[df["level"] == level]
            rows = []
            for eng in ENGINE_ORDER:
                r = sub[sub["search_engine"] == eng]
                if r.empty:
                    continue
                r = r.iloc[0]
                tt = int(r["TT"])
                if is_eh:
                    oft_species = int(r["TT_impossible_human"])
                    oft_species_frac = oft_species / tt if tt else float("nan")
                    oft_not_copresent = int(r["TT_impossible_not_copresent"])
                    oftr_combined = float(r["tt_impossible_fpr"])
                else:
                    oft_species = int(r["TT_impossible"])
                    oft_species_frac = float(r["tt_impossible_fpr"])
                    d = em_detail.get((eng, level))
                    if d is None:
                        oft_not_copresent = 0
                        oftr_combined = oft_species_frac
                    else:
                        oft_not_copresent = d["extra"]
                        oftr_combined = (d["impossible_total"] / d["denom"]
                                         if d["denom"] else float("nan"))
                rows.append({
                    "search_pipeline": ENGINE_LABELS.get(eng, eng),
                    "TT": tt,
                    "TD": int(r["TD"]),
                    "DD": int(r["DD"]),
                    "decoy_fdr": float(r["decoy_fdr"]),
                    "OFT_species": oft_species,
                    "OFTR_species": oft_species_frac,
                    "OFT_not_copresent": oft_not_copresent,
                    "OFTR_combined": oftr_combined,
                })
            table = pd.DataFrame(rows)
            path = out_dir / f"oft_table_{ds_label}_{level}.csv"
            table.to_csv(path, index=False)
            print(f"  saved {path.relative_to(REPO_ROOT)}")


_PAIR_SHARING_CATEGORIES = [
    "all", "target_target", "decoy",
    "tt_possible", "oft_species", "oft_not_copresent", "oft_total",
]


def _pair_sharing_not_copresent(frame: pd.DataFrame, mask: pd.Series, ibaq) -> pd.Series:
    """Row mask (aligned to *frame*) of rows in *mask* that fail co-presence."""
    from src.analysis.ibaq_copresence import _check_copresence
    out = pd.Series(False, index=frame.index)
    if not mask.any():
        return out
    hit = frame.loc[mask].apply(
        lambda r: not _check_copresence(r["Protein1"], r["Protein2"], ibaq), axis=1)
    out.loc[hit[hit].index] = True
    return out


def _em_oft_reason(df: pd.DataFrame, level: str, ibaq, lookup) -> pd.Series:
    """Per-row OFT reason for ecoli_mycpn, aligned to df's own index:
    ``''`` (possible / not evaluated), ``'species'``, or ``'not_copresent'``.
    Non-TT rows get ``''``. Mirrors :func:`_em_copresence_detail`'s per-engine
    logic but keeps the per-row classification instead of aggregating it.
    """
    reason = pd.Series("", index=df.index, dtype=object)
    for eng, g in df.groupby("search_engine"):
        if level == "CSM":
            scan_type = ENGINE_SCAN_TYPE.get(eng)
            if scan_type is None:
                continue
            g2 = add_spectrum_origin(g.copy(), lookup, scan_type)
            g2 = g2[g2["decoy_cat"] == "TT"]
            from_ecoli = g2["spectrum_origin"] == "ecoli"
            from_mycpn = g2["spectrum_origin"] == "mycoplasma"
            cls = pd.Series(pd.NA, index=g2.index, dtype=object)
            cls[from_ecoli] = g2.loc[from_ecoli, "class_ecoli_spec"]
            cls[from_mycpn] = g2.loc[from_mycpn, "class_mycpn_spec"]
            species_bad = cls.isin(["SM", "MM"]).fillna(False)
            possible = (cls == "SS").fillna(False)
            extra = _pair_sharing_not_copresent(g2, possible & from_ecoli, ibaq)
        else:
            g2 = g[g["decoy_cat"] == "TT"]
            species_bad = g2["match_origin"] == "mixed"
            possible = ~species_bad
            extra = _pair_sharing_not_copresent(g2, possible & (g2["match_origin"] == "ecoli"), ibaq)
        reason.loc[g2.index[species_bad]] = "species"
        reason.loc[g2.index[extra]] = "not_copresent"
    return reason


def _pair_sharing_key_sets(df: pd.DataFrame, level: str,
                           reason: pd.Series) -> dict[str, dict[str, set]]:
    """engine -> category -> set(pair_key), for every category in
    :data:`_PAIR_SHARING_CATEGORIES`.
    """
    key_col1, key_col2 = ("Peptide1", "Peptide2") if level == "CSM" else ("Protein1", "Protein2")
    has_pos = level == "CSM" and "PepPos1" in df.columns and "PepPos2" in df.columns
    cols = [key_col1, key_col2, "decoy_cat"]
    if has_pos:
        cols += ["PepPos1", "PepPos2"]
    sub = df[["search_engine"] + cols].copy()
    sub["_reason"] = reason.reindex(sub.index).fillna("")

    out: dict[str, dict[str, set]] = {}
    for eng, g in sub.groupby("search_engine"):
        cats = {c: set() for c in _PAIR_SHARING_CATEGORIES}
        for row in g.itertuples(index=False):
            p1, p2, dc = row[1], row[2], row[3]
            pos1, pos2 = (row[4], row[5]) if has_pos else (None, None)
            reas = row[-1]
            key = (_pair_key(p1, p2, pos1=pos1, pos2=pos2) if level == "CSM"
                   else _protein_pair_key(p1, p2))
            cats["all"].add(key)
            if dc == "TT":
                cats["target_target"].add(key)
                if reas == "species":
                    cats["oft_species"].add(key)
                    cats["oft_total"].add(key)
                elif reas == "not_copresent":
                    cats["oft_not_copresent"].add(key)
                    cats["oft_total"].add(key)
                else:
                    cats["tt_possible"].add(key)
            else:
                cats["decoy"].add(key)
        out[eng] = cats
    return out


def _pair_sharing_signature_counts(key_sets: dict[str, dict[str, set]],
                                   category: str) -> dict[frozenset, int]:
    """frozenset(engines reporting a pair) -> number of distinct pairs with
    exactly that signature, for one category.
    """
    key_to_engines: dict = {}
    for eng, cats in key_sets.items():
        for key in cats[category]:
            key_to_engines.setdefault(key, set()).add(eng)
    sig_count: dict[frozenset, int] = {}
    for engines in key_to_engines.values():
        sig = frozenset(engines)
        sig_count[sig] = sig_count.get(sig, 0) + 1
    return sig_count


def _pair_sharing_stats_table(key_sets: dict[str, dict[str, set]],
                              engines: list[str]) -> pd.DataFrame:
    """One row per sharing level n = 1..len(engines). For each category and
    each n, stats (min/max/mean/median/stderr) are taken across the counts
    of every specific n-sized combination of pipelines (including
    combinations with zero shared pairs) — i.e. across all C(len(engines), n)
    combinations, not just the ones that happen to occur. stderr is the
    sample standard deviation (ddof=1) over sqrt(n_combinations); it is NaN
    when n == len(engines), since there is then only one combination.
    """
    n_total = len(engines)
    sig_counts_by_cat = {cat: _pair_sharing_signature_counts(key_sets, cat)
                         for cat in _PAIR_SHARING_CATEGORIES}
    rows = []
    for n in range(1, n_total + 1):
        row = {"shared_among_n": n}
        for cat in _PAIR_SHARING_CATEGORIES:
            sig_counts = sig_counts_by_cat[cat]
            vals = [sig_counts.get(frozenset(combo), 0)
                    for combo in itertools.combinations(engines, n)]
            arr = np.array(vals, dtype=float)
            row[f"{cat}_min"] = arr.min()
            row[f"{cat}_max"] = arr.max()
            row[f"{cat}_mean"] = round(float(arr.mean()), 3)
            row[f"{cat}_median"] = float(np.median(arr))
            row[f"{cat}_stderr"] = (round(float(arr.std(ddof=1) / math.sqrt(len(arr))), 3)
                                    if len(arr) > 1 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_pair_sharing_tables(
        exclude_engines: tuple[str, ...] = ("plink3_filtered",),
) -> dict[tuple[str, str], pd.DataFrame]:
    """(dataset, level) -> the pair-sharing stats table (see
    :func:`_pair_sharing_stats_table`). Shared by the CSV export and the
    bar-chart figure so both read off the identical, freshly computed table.
    """
    keep_engines = [e for e in ENGINE_ORDER if e not in exclude_engines]
    ibaq = load_ibaq_presence(REPO_ROOT / "input" / "proteinGroups.txt")
    lookup = load_merged_lookup(REPO_ROOT / "input" / "merged_files.json")

    tables: dict[tuple[str, str], pd.DataFrame] = {}
    for dataset in ("ecoli_human", "ecoli_mycpn"):
        for level in ("CSM", "PPI"):
            parquet = NORMALISED_DIR / dataset / level / "combined.parquet"
            if not parquet.exists():
                continue
            df = pd.read_parquet(parquet)
            df = df[df["search_engine"].isin(keep_engines)].copy()
            df = add_decoy_category(df)
            engines_here = sorted(df["search_engine"].unique())
            if not engines_here:
                continue
            if dataset == "ecoli_human":
                reason = df["impossible_reason"].replace({"human": "species"})
            else:
                reason = _em_oft_reason(df, level, ibaq, lookup)
            key_sets = _pair_sharing_key_sets(df, level, reason)
            tables[(dataset, level)] = _pair_sharing_stats_table(key_sets, engines_here)
    return tables


def export_pair_sharing_table(out_dir: Path = NORMALISED_DIR,
                              exclude_engines: tuple[str, ...] = ("plink3_filtered",)) -> None:
    """Write one CSV per (dataset, level) with a row per pipeline-sharing
    level n = 1 (unique to one pipeline) through every pipeline present, and
    columns ``{category}_{min,max,mean,median,stderr}`` for seven categories:
    ``all``, ``target_target``, ``decoy``, ``tt_possible``, ``oft_species``,
    ``oft_not_copresent``, ``oft_total``.

    For a given n and category, the stats are taken across every specific
    combination of n pipelines (there are C(len(engines), n) of them),
    counting distinct pairs shared by exactly that combination (not a
    superset) — see :func:`_pair_sharing_stats_table`.

    *exclude_engines* drops those pipelines entirely before counting, not
    just from the denominator: a pair shared only with an excluded engine is
    dropped too, as if that engine had never run. Defaults to pLink3's
    post-filter variant, a derived re-run of pLink3 rather than an
    independent pipeline.
    """
    tables = _compute_pair_sharing_tables(exclude_engines)
    for (dataset, level), table in tables.items():
        path = out_dir / f"pair_sharing_{dataset}_{level}.csv"
        table.to_csv(path, index=False)
        print(f"  saved {path.relative_to(REPO_ROOT)}")


def plot_fig_s6_total_overlap() -> None:
    """Figure S6: 4-panel bar chart of the pair-sharing-by-pipeline-combination
    stats from :func:`_compute_pair_sharing_tables`. Rows are datasets
    (E.coli-Human top, E.coli-Mycoplasma bottom); columns are the pair level
    (CSM left, PPI right) — the same row/column convention as
    :func:`plot_fig_s5_overlap_grid`, with the dataset given once as the left
    column's y-axis label and the level once as the top row's title, and the
    panel letter in the corner via :func:`_draw_panel_label`.

    Within a panel, x is the sharing level n and each x carries 4 grouped
    bars: TT possible, OFT (wrong species), OFT (not co-present), OFT total.
    Bar height is the across-combinations mean; error bars are the
    across-combinations standard error (zero, i.e. no visible error, at
    n == n_total where there is only one combination).
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    tables = _compute_pair_sharing_tables()

    # DS_EH/DS_EM italicise the species via mathtext `\it{...}`; swapping that
    # for `\boldsymbol{...}` bolds the row label without touching the shared
    # constant (plain fontweight="bold" does not reach mathtext spans).
    def _bold_species(label: str) -> str:
        return label.replace(r"\it{", r"\boldsymbol{")

    rows = (("ecoli_human", _bold_species(DS_EH)), ("ecoli_mycpn", _bold_species(DS_EM)))
    cols = (("CSM", "CSM level"), ("PPI", "PPI level"))
    letters = (("a", "b"), ("c", "d"))
    cats = ["tt_possible", "oft_species", "oft_not_copresent", "oft_total"]
    colors = [C_OVL_SH_POS, C_ENTRAP, C_COFRAC, C_IMPOSS]
    labels = ["TT possible", "OFT (wrong species)", "OFT (not co-present)", "OFT total"]
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    for r, (dataset, ds_label) in enumerate(rows):
        for c, (level, col_title) in enumerate(cols):
            ax = axes[r][c]
            table = tables.get((dataset, level))
            if table is None:
                ax.set_visible(False)
                continue
            n_vals = table["shared_among_n"].values
            x = np.arange(len(n_vals))
            for offset, cat, color, label in zip(offsets, cats, colors, labels):
                means = table[f"{cat}_mean"].values
                stderr = np.nan_to_num(table[f"{cat}_stderr"].values, nan=0.0)
                ax.bar(x + offset, means, width=width, yerr=stderr, capsize=2,
                       color=color, label=label, alpha=0.85,
                       edgecolor="white", linewidth=0.4,
                       error_kw=dict(elinewidth=0.8, ecolor="black"))
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.1)
            ax.set_xticks(x)
            tick_labels = [str(nv) for nv in n_vals]
            tick_labels[0] = "1\nunique"
            tick_labels[-1] = f"{n_vals[-1]}\nall"
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("Pipelines sharing the pair (n)", fontsize=_SCATTER_LABEL_FS)
            ax.set_ylabel("Mean distinct pairs per combination", fontsize=_SCATTER_LABEL_FS)
            if r == 0:
                ax.set_title(col_title, fontsize=_SCATTER_LABEL_FS, fontweight="bold")
            _draw_panel_label(ax, letters[r][c])
        axes[r][0].set_ylabel(f"{ds_label}\nMean distinct pairs per combination",
                              fontsize=_SCATTER_LABEL_FS)

    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4,
               fontsize=_SCATTER_LABEL_FS, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "fig_s6_total_overlap")


def plot_fig2_impossible_copresence(eh: pd.DataFrame, em: pd.DataFrame) -> None:
    """fig2 variant counting non-co-present E. coli pairs as impossible.

    Only the E.coli-Mycoplasma row changes: *eh* is passed through untouched, so
    its ``tt_impossible_fpr_pct`` keeps the co-presence term the summaries
    already carry. The standard variants strip that with
    :func:`_eh_impossible_entrapment_only`; here both rows keep it, which is the
    point of the figure.
    """
    em_ext = _em_impossible_with_copresence(em)
    for scale in ("log2",): # "sqrt", 
        _plot_fig2_impossible(eh, em_ext, scale=scale, name_suffix="_copresence")


def plot_multi_fig2_copresence(eh: pd.DataFrame, em: pd.DataFrame) -> None:
    """fig_multi2 built on the extended E.coli-Mycoplasma impossible definition.

    Panels a and b change, since both read ``tt_impossible_fpr_pct``; panel c is
    E.coli-Human only and is identical to the standard figure. See
    :func:`_em_impossible_with_copresence` for what the extension is and why it
    is kept out of the summaries.
    """
    em_ext = _em_impossible_with_copresence(em)
    for parallel in (False, True):
        plot_multi_fig2(eh, em_ext, parallel, name_suffix="_copresence")


def plot_fig2_impossible_inset(eh: pd.DataFrame, em: pd.DataFrame) -> None:
    """Figure 2 (inset): linear axes with a zoomed inset on the 0–~10% cluster.

    Each panel keeps a full-range linear view (so outliers stay visible) plus a
    corner inset zoomed to the origin cluster, connected by ``indicate_inset``.
    Mango (no decoys → undefined decoy FDR) is placed in a dedicated column
    to the right of a visual separator rather than at the misleading x=0.
    """
    sns.set_theme(style="white", font_scale=0.85)
    frames = {"eh": eh, "em": em}
    palette = _fig2_palette()

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for row, (ds_label, key) in enumerate(_FIG2_DATASETS):
        df = frames[key]
        for col, level in enumerate(("CSM", "PPI")):
            ax = axes[row][col]
            pts, texts, mango_pts, _, _ = _fig2_scatter(ax, df, level, palette)

            lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05
            sep_x   = lim * 1.08
            mango_x = lim * 1.18
            x_total = lim * 1.30 if mango_pts else lim

            _fig2_refs(ax, lim)
            ax.set_xlim(0, x_total)
            ax.set_ylim(0, lim)
            ax.set_xlabel(_FIG2_XLABEL, fontsize=_SCATTER_LABEL_FS)
            ax.set_ylabel(_FIG2_YLABEL, fontsize=_SCATTER_LABEL_FS)
            if row == 0:
                ax.set_title(level, fontsize=_SCATTER_LABEL_FS)

            # Mango in "undefined" column — add text to `texts` so adjust_text
            # handles all labels together in one pass below.
            if mango_pts:
                ax.axvline(sep_x, color="grey", ls=":", lw=0.5, alpha=0.4)
                for eng, y in mango_pts:
                    color = palette.get(eng, "grey")
                    ax.scatter(mango_x, y, marker="o", color=color,
                               s=_FIG2_MARKER_SIZE,
                               zorder=3, edgecolors="white", linewidths=0.4)
                    t = ax.text(mango_x, y, engine_label(eng, level),
                                fontsize=_SCATTER_LABEL_FS, ha="left", va="bottom", color=color)
                    texts.append(t)
                ax.text(mango_x, -0.04, "undef.", ha="center", va="top",
                        fontsize=_SCATTER_LABEL_FS, transform=ax.get_xaxis_transform(),
                        clip_on=False)

            if texts:
                # Tuned rather than left on defaults: the PPI panels now carry two
                # long labels for the same engine (one per PPI derivation), which
                # collide at default clearance.
                adjust_text(texts, ax=ax, expand=(1.25, 1.45),
                            force_text=(0.5, 1.0), force_static=(0.4, 0.8),
                            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
            sns.despine(ax=ax)

            if pts and any(x > 15 or y > 15 for x, y in pts):
                axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
                _, ins_texts, _, _, _ = _fig2_scatter(axins, df, level, palette,
                                                annotate=True, marker_size=28)
                _fig2_refs(axins, 5)
                axins.set_xlim(0, 5)
                axins.set_ylim(0, 5)
                axins.tick_params(labelsize=_SCATTER_LABEL_FS)
                if ins_texts:
                    adjust_text(ins_texts, ax=axins,
                                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
                ax.indicate_inset_zoom(axins, edgecolor="grey", alpha=0.6)
                sns.despine(ax=axins)
        axes[row][0].set_ylabel(f"{ds_label}\n{_FIG2_YLABEL}", fontsize=_SCATTER_LABEL_FS)

    _fig2_legend(fig, palette)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    _save(fig, "fig2_impossible_vs_decoy_inset")
def plot_fig3_csm_benchmarking(eh: pd.DataFrame, em: pd.DataFrame) -> None:
    """Figure 3: two-panel CSM-level grouped bar comparison."""
    sns.set_theme(style="whitegrid", font_scale=0.9)

    def _panel(ax: plt.Axes, df: pd.DataFrame, title: str,
                entrap_col: str, entrap_label: str) -> None:
        csm = df[df["level"] == "CSM"].copy()
        csm = csm.set_index("search_engine").reindex(ENGINE_ORDER).reset_index()
        csm = csm.dropna(subset=["decoy_fdr_pct"])

        x = range(len(csm))
        width = 0.25
        offsets = [-width, 0, width]
        cols = ["decoy_fdr_pct", entrap_col, "tt_impossible_fpr_pct"]
        colors = [C_DECOY, C_ENTRAP, C_IMPOSS]
        labels = ["Decoy FDR", entrap_label, "Impossible FPR"]

        for offset, col, color, label in zip(offsets, cols, colors, labels):
            vals = csm[col].fillna(0).values
            ax.bar([xi + offset for xi in x], vals,
                   width=width, color=color, label=label, alpha=0.85,
                   edgecolor="white", linewidth=0.4)

        ax.axhline(REF_FDR, color="black", lw=0.8, ls="--", alpha=0.7, label="2% reference")
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [ENGINE_LABELS.get(e, e) for e in csm["search_engine"]],
            rotation=40, ha="right", fontsize=_SCATTER_LABEL_FS)
        ax.set_ylabel("Rate (%)", fontsize=_SCATTER_LABEL_FS)
        ax.set_title(title, fontsize=_SCATTER_LABEL_FS, fontweight="bold")
        ax.set_ylim(0, max(55, csm[cols].fillna(0).max().max() * 1.1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    _panel(axes[0], eh, f"A   {DS_EH} (CSM level)",
           "tt_entrap_fpr_pct", "Entrapment FPR")
    _panel(axes[1], em, f"B   {DS_EM} (CSM level)",
           "tt_entrap_fpr_pct", "Entrapment FPR (TW+WW)")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4,
               fontsize=_SCATTER_LABEL_FS, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig4_csm_benchmarking")


def plot_fig4_score_hist_selected() -> None:
    """Figure 4: 2×2 score histogram composite for selected engines (ecoli_human CSM).

    Reads NormalisedData/ecoli_human/CSM/combined.parquet and classifies matches
    using the same logic as score_hist.py. Shows four engines in a 2×2 grid:
    Kojak and Scout (well-calibrated), pLink3 and MeroX (miscalibrated).
    Panels where a dominant TT class dwarfs the others automatically use a
    broken y-axis (same _auto_break threshold as score_hist.py).
    """
    from src.analysis.score_hist import (
        _auto_break,
        _bin_edges,
        _classify_ecoli_human,
        _draw_hist,
        _draw_hist_broken,
        _EH_CLASS_ORDER,
        _engine_axes,
        _engine_axes_broken,
        load_organism_sets,
        C_DECOY as _SH_DECOY,
        C_EE_TT as _SH_EE_TT,
        C_ENTRAP as _SH_ENTRAP,
    )

    INPUT_DIR = REPO_ROOT / "input"
    ecoli_set, human_set = load_organism_sets(
        INPUT_DIR / "ecoli_protein_accessions.txt",
        INPUT_DIR / "human_protein_accessions.txt",
    )

    df = pd.read_parquet(NORMALISED_DIR / "ecoli_human" / "CSM" / "combined.parquet")
    df = _classify_ecoli_human(df, ecoli_set, human_set)

    selected = [
        ("kojak",  "A   Kojak (well-calibrated)"),
        ("scout_fl", "B   Scout (FL) (well-calibrated)"),
        ("plink3", "C   pLink3 (FDR) (decoy distortion)"),
        ("merox",  "D   MeroX (pre-filter distortion)"),
    ]

    fig = plt.figure(figsize=(14, 9))
    gs = mgridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (eng, title) in enumerate(selected):
        row, col = divmod(idx, 2)
        cell = gs[row, col]
        eng_df = df[df["search_engine"] == eng]

        if eng_df.empty:
            ax = fig.add_subplot(cell)
            ax.text(0.5, 0.5, f"{ENGINE_LABELS.get(eng, eng)}\n(no data)",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.axis("off")
            continue

        bins = _bin_edges(eng_df["score"])
        break_info = _auto_break(eng_df, bins, _EH_CLASS_ORDER, ["TT Ecoli-Ecoli"])

        if break_info is not None:
            break_at, top_max = break_info
            ax_top, ax_bot, _ = _engine_axes_broken(fig, cell, with_bar=False)
            _draw_hist_broken(ax_top, ax_bot, eng_df, _EH_CLASS_ORDER,
                              title, break_at, top_max, line=True)
        else:
            ax_h, _ = _engine_axes(fig, cell, with_bar=False)
            _draw_hist(ax_h, eng_df, _EH_CLASS_ORDER, title, line=True)

    # Shared legend. The histograms are drawn with score_hist._EH_CLASS_ORDER, so the
    # swatches must come from score_hist's constants — hard-coding them here lets the
    # legend drift out of sync with the curves it labels.
    legend_handles = [
        mpatches.Patch(color=_SH_EE_TT,  alpha=0.7, label=f"TT {SP_ECOLI}"),
        mpatches.Patch(color=_SH_ENTRAP, alpha=0.7, label=f"TT Entrapment ({SP_HUMAN})"),
        mpatches.Patch(color=_SH_DECOY,  alpha=0.7, label="Decoy (TD+DD)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=_SCATTER_LABEL_FS, frameon=False, bbox_to_anchor=(0.5, -0.01))
    _save(fig, "fig3_score_hist_selected")


def plot_figS3_impossible_breakdown(eh: pd.DataFrame, em: pd.DataFrame, *,
                                    cofractionation: bool = True,
                                    name_suffix: str = "") -> None:
    """Supplement Figure S3: impossible-match breakdown as % of TT, 2×2 (dataset × level).

    Segment order within every bar is possible, then the co-fractionation call,
    then the entrapment / wrong-species call, so the definite error sits at the
    outer edge and the weaker abundance-based criterion between the two.

    With *cofractionation* the co-fractionation term is shown for **both**
    datasets: E.coli–Human takes it from ``TT_impossible_not_copresent``, and
    E.coli–Mycoplasma from :func:`_em_impossible_with_copresence`, whose extra
    over the cross-species rate is exactly the co-fractionation share. Without
    it, each dataset shows only its intrinsic criterion — a human entrapment
    match, or a wrong-organism spectrum.

    Each dataset is ranked independently on its own CSM total impossible rate,
    under whichever definition the figure is drawing: panel b takes panel a's
    order and panel d takes panel c's, so a pipeline sits at the same height in
    both panels of its dataset and CSM and PPI read against each other. The two
    datasets are ordered separately because they do not cover the same
    pipelines.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)

    C_POSSIBLE  = "#50cf50"
    # The dark purple of fig_s5_overlap_grid_cofractionation_split, so the
    # never-co-fractionating class reads the same across the two figures.
    # Not just for consistency: simulated under protan/deutan/tritan CVD,
    # this shade's worst-case separation from the other colors on this bar
    # (red especially, which it sits directly next to) is dE76 56.8 vs the
    # light-purple alternative's 46.5.
    C_IMP_NOCP  = C_OVL_SH_NOCP
    C_IMP_HUMAN = "#d62728"

    # Co-fractionation-extended E.coli–Mycoplasma rates; the excess over the
    # plain cross-species rate is that dataset's co-fractionation share.
    em_ext = _em_impossible_with_copresence(em) if cofractionation else em

    # Engine order: mean CSM impossible rate across datasets, under the same
    # definition the figure draws — the totals printed beside each bar.
    eh_csm = eh[eh["level"] == "CSM"].set_index("search_engine")
    if cofractionation:
        eh_csm_fpr = (100 * (eh_csm["TT_impossible_human"]
                             + eh_csm["TT_impossible_not_copresent"]) / eh_csm["TT"])
    else:
        eh_csm_fpr = 100 * eh_csm["TT_impossible_human"] / eh_csm["TT"]
    em_csm_fpr = em_ext[em_ext["level"] == "CSM"].set_index("search_engine")["tt_impossible_fpr_pct"]

    def _rank(fpr: pd.Series) -> list:
        """Engines present in *fpr*, worst first (bottom of chart), best last."""
        avail = [e for e in ENGINE_ORDER
                 if e in fpr.index and not pd.isna(fpr.get(e))]
        return sorted(avail, key=lambda e: float(fpr.get(e)), reverse=True)

    eh_engine_order = _rank(eh_csm_fpr)
    em_engine_order = _rank(em_csm_fpr)
    n = max(len(eh_engine_order), len(em_engine_order))
    bar_h = 0.7

    # _PPI_FROM_CSM (module level): PPI derived by aggregating CSM results — no
    # native PPI output from the engine. NB the * marks the FDR *level* (no
    # dedicated PPI-level control), not the derivation, which engine_label()
    # spells out separately on PPI panels.

    # ‡ — FDR controlled pooled over link types rather than separately for
    # heteromeric matches, so the declared 2 % is not what heteromeric links
    # actually got. Per (engine, level), because it can differ between the two:
    # Scout (DB) reports 10.5 % decoy FDR at CSM but 1.94 % at PPI, so only its
    # CSM list is pooled. MS Annika is pooled at both (8.7 % CSM, 14.4 % PPI).
    _NO_HET_FDR = {("msannika", "CSM"), ("msannika", "PPI"),
                   ("scout_db", "CSM")}

    def _tick_label(eng: str, level: str) -> str:
        base = engine_label(eng, level)
        if (eng, level) in _NO_HET_FDR:
            return base + " ‡"
        # * flags a PPI extrapolated from CSM, so it is meaningless on a CSM panel.
        if level == "PPI" and eng in _PPI_FROM_CSM:
            return base + " *"
        return base

    fig, axes = plt.subplots(2, 2, figsize=(14, max(6, n * 0.55 + 2)), squeeze=False)

    def _panel_label(ax: plt.Axes, label: str) -> None:
        _draw_panel_label(ax, label)

    def _panel_eh(ax: plt.Axes, df: pd.DataFrame, level: str, engine_order: list,
                  label: str) -> None:
        _panel_label(ax, label)
        y = np.arange(len(engine_order))
        sub = df[df["level"] == level].set_index("search_engine")
        poss, nocp, human, fpr_ann = [], [], [], []
        for eng in engine_order:
            tt = sub["TT"].get(eng, float("nan"))
            if pd.isna(tt) or tt == 0:
                poss.append(0.0); nocp.append(0.0); human.append(0.0)
                fpr_ann.append(float("nan"))
            else:
                hum = 100 * sub["TT_impossible_human"].get(eng, 0) / tt
                ncp = (100 * sub["TT_impossible_not_copresent"].get(eng, 0) / tt
                       if cofractionation else 0.0)
                human.append(hum); nocp.append(ncp)
                poss.append(max(0.0, 100.0 - hum - ncp))
                fpr_ann.append(hum + ncp)
        pa, na, ha = map(np.array, (poss, nocp, human))
        ax.barh(y, pa, height=bar_h, color=C_POSSIBLE,  zorder=2)
        ax.barh(y, na, height=bar_h, left=pa,      color=C_IMP_NOCP,  zorder=2)
        ax.barh(y, ha, height=bar_h, left=pa + na, color=C_IMP_HUMAN, zorder=2)
        for i, v in enumerate(fpr_ann):
            if not pd.isna(v):
                ax.text(101, i, f"{v:.1f}%", va="center", fontsize=_SCATTER_LABEL_FS)
        ax.axvline(100, color="black", lw=0.5, alpha=0.5)
        ax.axvline(100 - REF_FDR, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([_tick_label(e, level) for e in engine_order],
                           fontsize=_SCATTER_LABEL_FS)
        # Dataset + level carried by the x-label instead of a panel title.
        ax.set_xlabel(f"{DS_EH}  % of TT {level}s", fontsize=_SCATTER_LABEL_FS)
        ax.set_xlim(0, 115)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

    def _panel_em(ax: plt.Axes, df: pd.DataFrame, level: str, eng_list: list,
                  label: str) -> None:
        _panel_label(ax, label)
        sub = df[df["level"] == level].set_index("search_engine")
        ext = em_ext[em_ext["level"] == level].set_index("search_engine")
        poss, nocp, imp, fpr_ann = [], [], [], []
        for eng in eng_list:
            base = sub["tt_impossible_fpr_pct"].get(eng, float("nan"))
            if pd.isna(base):
                poss.append(0.0); nocp.append(0.0); imp.append(0.0)
                fpr_ann.append(float("nan"))
                continue
            total = ext["tt_impossible_fpr_pct"].get(eng, base) if cofractionation else base
            imp.append(base)
            nocp.append(max(0.0, total - base))
            poss.append(max(0.0, 100.0 - total))
            fpr_ann.append(total)
        ye = np.arange(len(eng_list))
        pa, na, ia = map(np.array, (poss, nocp, imp))
        ax.barh(ye, pa, height=bar_h, color=C_POSSIBLE,  zorder=2)
        ax.barh(ye, na, height=bar_h, left=pa,      color=C_IMP_NOCP,  zorder=2)
        ax.barh(ye, ia, height=bar_h, left=pa + na, color=C_IMP_HUMAN, zorder=2)
        for i, v in enumerate(fpr_ann):
            if not pd.isna(v):
                ax.text(101, i, f"{v:.1f}%", va="center", fontsize=_SCATTER_LABEL_FS)
        ax.axvline(100, color="black", lw=0.5, alpha=0.5)
        ax.axvline(100 - REF_FDR, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_yticks(ye)
        ax.set_yticklabels([_tick_label(e, level) for e in eng_list],
                           fontsize=_SCATTER_LABEL_FS)
        ax.set_xlabel(f"{DS_EM}  % of TT {level}s", fontsize=_SCATTER_LABEL_FS)
        ax.set_xlim(0, 115)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

    # b takes a's order, d takes c's.
    _panel_eh(axes[0][0], eh, "CSM", eh_engine_order, "a")
    _panel_eh(axes[0][1], eh, "PPI", eh_engine_order, "b")
    _panel_em(axes[1][0], em, "CSM", em_engine_order, "c")
    _panel_em(axes[1][1], em, "PPI", em_engine_order, "d")

    legend_handles = [mpatches.Patch(color=C_POSSIBLE, label="Possible")]
    if cofractionation:
        legend_handles.append(
            mpatches.Patch(color=C_IMP_NOCP,
                           label=f"OFT: {SP_ECOLI} pair never co-fractionating"))
    legend_handles.append(
        mpatches.Patch(color=C_IMP_HUMAN,
                       label="OFT: entrapment match / wrong-organism spectrum"))
    legend_handles.append(
        Line2D([0], [0], color="black", lw=0.8, ls="--", alpha=0.6, label="2% FDR"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=_SCATTER_LABEL_FS, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.text(0.5, -0.025,
             "* No dedicated PPI FDR; PPI error extrapolated from lower level."
             "  ‡ no heteromeric specific FDR.",
             ha="center", fontsize=_SCATTER_LABEL_FS, style="italic")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, f"figS3_impossible_breakdown{name_suffix}")


def _draw_eh_csm_impossible_panel(
    ax: plt.Axes, eh: pd.DataFrame, *, panel_label: str | None = None
) -> None:
    """E. coli–Human CSM impossible-match breakdown (mirrors figS3 top-left panel).

    Sorted by EH CSM impossible FPR descending (worst at bottom, best at top).
    """
    if panel_label:
        _draw_panel_label(ax, panel_label)
    C_POSSIBLE  = "#50cf50"
    C_IMP_HUMAN = "#d62728"
    bar_h = 0.7

    sub = eh[eh["level"] == "CSM"].set_index("search_engine")
    fpr_col = "tt_impossible_fpr_pct"

    eng_present = [e for e in ENGINE_ORDER
                   if e in sub.index and not pd.isna(sub.loc[e, fpr_col])]
    ranked = sorted(eng_present,
                    key=lambda e: float(sub.loc[e, fpr_col]),
                    reverse=True)
    y = np.arange(len(ranked))

    # Two categories only: matches impossible by co-fractionation count as
    # possible here, so the bars still sum to 100% of TT.
    poss, human, fpr_ann = [], [], []
    for eng in ranked:
        tt = float(sub.loc[eng, "TT"])
        if tt == 0:
            poss.append(0.); human.append(0.)
            fpr_ann.append(float("nan"))
        else:
            human.append(100 * float(sub.loc[eng, "TT_impossible_human"]) / tt)
            poss.append(100 - human[-1])
            fpr_ann.append(float(sub.loc[eng, fpr_col]))

    pa, ha = map(np.array, (poss, human))
    ax.barh(y, pa, height=bar_h, color=C_POSSIBLE,  edgecolor="none", zorder=2)
    ax.barh(y, ha, height=bar_h, left=pa, color=C_IMP_HUMAN, edgecolor="none", zorder=2)
    for i, v in enumerate(fpr_ann):
        if not pd.isna(v):
            ax.text(101, i, f"{v:.1f}%", va="center", fontsize=_SCATTER_LABEL_FS)
    ax.axvline(100, color="black", lw=0.5, alpha=0.5)
    # Expectation at the declared 2% FDR: a well-calibrated engine's bar should
    # stop at 98% possible, so the gap past this line is the observable FP.
    # The dashed line stops at the top edge of the best-ranked (topmost) bar
    # instead of running all the way to the frame; a short diagonal dashed
    # segment continues from there to the "2% FDR" label, which sits in the
    # top-left corner (near the frame) away from the line rather than
    # straddling it.
    ref_x = 100 - REF_FDR
    top_bar_y = (len(ranked) - 1) + bar_h / 2
    y0, y1 = ax.get_ylim()
    top_bar_frac = (top_bar_y - y0) / (y1 - y0)
    ax.axvline(ref_x, ymin=0, ymax=top_bar_frac, color="black", lw=0.8, ls="--", alpha=0.6)

    label_shift = ScaledTranslation(-12 / ax.figure.dpi, -1 / ax.figure.dpi, ax.figure.dpi_scale_trans)
    label_transform = ax.get_xaxis_transform() + label_shift
    ax.text(ref_x, 1.0, "2% FDR", transform=label_transform,
            ha="right", va="top", fontsize=_SCATTER_LABEL_FS, alpha=0.7)

    # Connects the top of the shortened vertical line to the label, landing on
    # the label's vertical center rather than the top of its (va="top")
    # anchor — half the font size down, in points converted to inches (dpi
    # cancels against dpi_scale_trans so this is resolution-independent).
    connector_text_target = label_transform + ScaledTranslation(
        0, -(_SCATTER_LABEL_FS / 2) / 72, ax.figure.dpi_scale_trans)

    # Built with annotate (rather than baking the two endpoints down to
    # display pixels and drawing with IdentityTransform) so it stays
    # correctly placed after _save()'s bbox_inches="tight" crop, which shifts
    # the canvas origin after this function runs.
    ax.annotate(
        "", xy=(ref_x, top_bar_frac), xycoords=ax.get_xaxis_transform(),
        xytext=(ref_x, 1.0), textcoords=connector_text_target,
        arrowprops=dict(arrowstyle="-", linestyle="--", linewidth=0.8,
                        color="black", alpha=0.6, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([ENGINE_LABELS.get(e, e) for e in ranked], fontsize=_SCATTER_LABEL_FS)
    ax.set_xlabel("% of Target CSM matches", fontsize=_SCATTER_LABEL_FS)
    ax.set_xlim(0, 115)
    # title removed per manuscript layout
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.legend(handles=[
        mpatches.Patch(color=C_POSSIBLE,  label="Possible"),
        mpatches.Patch(color=C_IMP_HUMAN, label="Observable False Target (OFT)"),
    ], fontsize=_SCATTER_LABEL_FS, frameon=True, loc="center")


def plot_fig6_xlinkx_before_after(eh: pd.DataFrame, em: pd.DataFrame) -> None:
    """Figure 6: XlinkX PD2.5 vs XlinkX PD3.3 — ecoli_human, decoy FDR + impossible FPR."""
    sns.set_theme(style="whitegrid", font_scale=0.9)

    xlinkx_engines = ["xlinkx25", "xlinkx_mgf"]
    before_color   = "#d62728"
    after_color    = "#50cf50"
    engine_colors  = [before_color, after_color]

    metrics = [
        ("decoy_fdr_pct",         "Decoy FDR (%)"),
        ("tt_impossible_fpr_pct", "Impossible FPR (%)"),
    ]
    levels = ["CSM", "PPI"]
    level_hatches = {"CSM": "", "PPI": "///"}

    engine_display = ["XlinkX PD2.5", "XlinkX PD3.3"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(9, 4.5), sharey=True)

    bar_width = 0.28

    # Pre-compute bar positions/values per (panel, engine, level) and collect all values.
    all_vals: list[float] = []
    # panel_bars[col][eng_idx] = [(xpos, val, color, hatch, level), ...]
    panel_bars: list[list[list[tuple[float, float, str, str, str]]]] = []

    for col, (metric, _) in enumerate(metrics):
        eng_groups: list[list[tuple[float, float, str, str, str]]] = []
        x_cursor = 0.0
        for eng, color in zip(xlinkx_engines, engine_colors):
            level_bars: list[tuple[float, float, str, str, str]] = []
            for li, (level, hatch) in enumerate(level_hatches.items()):
                sub = eh[(eh["search_engine"] == eng) & (eh["level"] == level)]
                val = float(sub[metric].values[0]) if not sub.empty and not pd.isna(sub[metric].values[0]) else 0.0
                xpos = x_cursor + li * (bar_width + 0.02)
                level_bars.append((xpos, val, color, hatch, level))
                all_vals.append(val)
            eng_groups.append(level_bars)
            x_cursor += len(levels) * (bar_width + 0.02) + 0.25
        panel_bars.append(eng_groups)

    y_max = max(all_vals) * 1.1

    for col, (metric, ylabel) in enumerate(metrics):
        ax = axes[col]
        tick_positions: list[float] = []
        tick_labels: list[str] = []

        for eng_idx, (eng_name, level_bars) in enumerate(zip(engine_display, panel_bars[col])):
            eng_xpos: list[float] = []
            for xpos, val, color, hatch, level in level_bars:
                ax.bar(xpos, val, bar_width, color=color,
                       hatch=hatch, edgecolor="white", alpha=0.85)
                tick_positions.append(xpos)
                tick_labels.append(level)
                eng_xpos.append(xpos)
            # Group label centred below the engine's pair of bars.
            group_center = sum(eng_xpos) / len(eng_xpos)
            ax.text(group_center, -0.22, eng_name, ha="center", va="top",
                    fontsize=_SCATTER_LABEL_FS, fontweight="bold",
                    transform=ax.get_xaxis_transform())

        ax.axhline(REF_FDR, color="black", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(0, y_max)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=_SCATTER_LABEL_FS)
        ax.set_ylabel(ylabel if col == 0 else "", fontsize=_SCATTER_LABEL_FS)
        ax.set_title(ylabel.split(" (")[0], fontsize=_SCATTER_LABEL_FS, fontweight="bold")

    handles = [
        mpatches.Patch(facecolor="grey", hatch="",    label="CSM"),
        mpatches.Patch(facecolor="grey", hatch="///", label="PPI"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=_SCATTER_LABEL_FS,
               frameon=True, bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.tight_layout(rect=[0, 0.07, 1.0, 1.0])
    _save(fig, "fig6_xlinkx_before_after")


def plot_fig7_overlap() -> None:
    """Figure 7: cross-engine peptide-pair overlap for ecoli_human CSM.

    Horizontal stacked bars showing, for each engine, how many of its unique
    peptide pairs are shared with other engines vs. unique to itself.
    XlinkX PD2.5 is annotated with its fraction of impossible pairs.
    """
    # Load raw CSM parquet and annotate.
    parquet = NORMALISED_DIR / "ecoli_human" / "CSM" / "combined.parquet"
    if not parquet.exists():
        print(f"  [overlap] {parquet} not found — skipping fig7")
        return

    df = pd.read_parquet(parquet)
    df = add_decoy_category(df)

    ecoli_set, human_set = load_organism_sets(
        ecoli_path=REPO_ROOT / "input" / "ecoli_protein_accessions.txt",
        human_path=REPO_ROOT / "input" / "human_protein_accessions.txt",
    )
    df = add_origin_flags(df, ecoli_set=ecoli_set, human_set=human_set)

    ibaq_presence = load_ibaq_presence(REPO_ROOT / "input" / "proteinGroups.txt", cutoff=1e6)
    df = add_impossible_flag_ecoli_human(df, ibaq_presence=ibaq_presence)

    summaries, _pair_table = compute_peptide_pair_overlap(df)

    # Sort engines: highest total-pair count at top.
    summaries = sorted(summaries, key=lambda s: s.n_pairs, reverse=True)

    engine_names = [s.engine for s in summaries]
    labels = [ENGINE_LABELS.get(e, e) for e in engine_names]
    n_unique  = [s.n_unique_to_engine for s in summaries]
    n_shared  = [s.n_shared_any for s in summaries]
    pct_imp   = {s.engine: s.pct_impossible for s in summaries}

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(summaries))))
    _draw_overlap_bars(ax, summaries, labels,
                       xlabel="Number of distinct peptide pairs",
                       legend_loc="upper right", collapse_impossible=True)
    fig.tight_layout()
    _save(fig, "fig7_overlap")


def _load_overlap_ppi_data(entrapment_only: bool = False) -> tuple[list, list[str]]:
    """Load and annotate the EH PPI parquet; return (summaries, labels).

    Restricted to canonical ``ENGINE_ORDER`` engines to avoid double-counting
    the ``*_csm`` alternative PPI roll-ups.

    ``entrapment_only`` restricts ``impossible`` to the human-entrapment
    criterion, dropping matches that are impossible only because their two
    E. coli proteins never co-eluted. The main figures use it so that every
    panel of fig_multi1 rests on the same definition; the supplementary
    figures keep both criteria.
    """
    parquet = NORMALISED_DIR / "ecoli_human" / "PPI" / "combined.parquet"
    if not parquet.exists():
        return [], []
    df = pd.read_parquet(parquet)
    df = df[df["search_engine"].isin(ENGINE_ORDER)]
    df = add_decoy_category(df)
    ecoli_set, human_set = load_organism_sets(
        ecoli_path=REPO_ROOT / "input" / "ecoli_protein_accessions.txt",
        human_path=REPO_ROOT / "input" / "human_protein_accessions.txt",
    )
    df = add_origin_flags(df, ecoli_set=ecoli_set, human_set=human_set)
    ibaq_presence = load_ibaq_presence(REPO_ROOT / "input" / "proteinGroups.txt", cutoff=1e6)
    df = add_impossible_flag_ecoli_human(df, ibaq_presence=ibaq_presence)
    if entrapment_only:
        df["impossible"] = df["impossible_reason"] == "human"
    summaries, _ = compute_protein_pair_overlap(df)
    summaries = sorted(summaries, key=lambda s: s.n_pairs, reverse=True)
    labels = [ENGINE_LABELS.get(s.engine, s.engine) for s in summaries]
    return summaries, labels


def _draw_overlap_bars(
    ax: plt.Axes, summaries: list, labels: list[str], *,
    xlabel: str, panel_label: str | None = None,
    legend_loc: str | None = None,
    entrapment_summaries: list | None = None,
    collapse_impossible: bool = False,
    impossible_label: str = "Observable FP",
    extra_legend_label: str | None = None,
    extra_legend_label_last: bool = False,
) -> None:
    """Stacked horizontal bars for cross-engine pair overlap.

    Shared by every overlap figure so segment order, colours and stacking stay
    identical across them. Pass ``legend_loc=None`` for panels that rely on a
    figure-level legend.

    Four segments by default. Passing *entrapment_summaries* — the same panel
    recomputed with the entrapment criterion alone — draws Shared, Pipeline
    exclusive, and the impossible pairs split by reason instead of by
    shared/pipeline-exclusive membership: pairs impossible because a human
    protein is involved, and pairs impossible only because their two E. coli
    proteins never co-fractionated. Shared/exclusive membership is derived
    from how many pipelines report a pair and so does not depend on the
    impossible flag, which makes the co-fractionation counts an exact
    subtraction. For E. coli-Mycoplasma the difference is zero and no purple
    segment appears.

    ``collapse_impossible=True`` draws three segments instead: Shared,
    Pipeline exclusive, and a single pooled-impossible segment (labelled
    *impossible_label*, "Observable FP" by default) combining the shared- and
    exclusive-impossible counts — that membership stops mattering once a pair
    is impossible. Mutually exclusive with *entrapment_summaries*, which keeps
    the impossible reason visible instead of collapsing it away.

    *extra_legend_label*, if given, is added to the legend as a plain-text
    entry (an invisible swatch, left-aligned with the segment patches like
    any other entry) — used for the "@ 2% FDR" caveat. It is prepended ahead
    of the segment patches by default; pass *extra_legend_label_last=True* to
    append it below them instead.
    """
    if panel_label:
        _draw_panel_label(ax, panel_label)

    y = list(range(len(summaries)))
    if collapse_impossible:
        segments = [
            ([s.n_shared_possible for s in summaries], C_OVL_SH_POS, "Shared"),
            ([s.n_unique_possible for s in summaries], C_OVL_UN_POS, "Pipeline exclusive"),
            ([s.n_shared_impossible + s.n_unique_impossible for s in summaries],
             C_IMPOSS, impossible_label),
        ]
    elif entrapment_summaries is None:
        segments = [
            ([s.n_shared_possible   for s in summaries], C_OVL_SH_POS, "Shared, possible"),
            ([s.n_shared_impossible for s in summaries], C_OVL_SH_IMP, "Shared, observable FP"),
            ([s.n_unique_possible   for s in summaries], C_OVL_UN_POS, "Pipeline exclusive, possible"),
            ([s.n_unique_impossible for s in summaries], C_OVL_UN_IMP, "Pipeline exclusive, observable FP"),
        ]
    else:
        ent = {s.engine: s for s in entrapment_summaries}
        def split(full_attr):
            hum, nocp = [], []
            for s in summaries:
                e = ent.get(s.engine)
                h = getattr(e, full_attr) if e is not None else 0
                hum.append(h)
                nocp.append(max(0, getattr(s, full_attr) - h))
            return hum, nocp
        sh_hum, sh_nocp = split("n_shared_impossible")
        un_hum, un_nocp = split("n_unique_impossible")
        # Shared/exclusive still gets its own segment for the possible pairs,
        # but the two impossible reasons are pooled across shared and
        # pipeline-exclusive — this panel exists to show the co-fractionation
        # share, not to re-litigate that split for it. The co-fractionation
        # call sits inside the entrapment / wrong-species one, the weaker
        # abundance-based criterion before the more definite one, so the most
        # certain error still lands at the outer edge of the bar.
        nocp = [a + b for a, b in zip(sh_nocp, un_nocp)]
        hum  = [a + b for a, b in zip(sh_hum, un_hum)]
        segments = [
            ([s.n_shared_possible for s in summaries], C_OVL_SH_POS,  "Shared"),
            ([s.n_unique_possible for s in summaries], C_OVL_UN_POS,  "Pipeline exclusive"),
            (nocp, C_OVL_SH_NOCP, "Observable FP: not co-fractionating"),
            (hum,  C_IMPOSS,      "Observable FP: entrapment / wrong species"),
        ]

    left = [0] * len(y)
    for vals, colour, label in segments:
        ax.barh(y, vals, left=left, color=colour, edgecolor="none", label=label)
        left = [a + b for a, b in zip(left, vals)]

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=_SCATTER_LABEL_FS)
    ax.set_xlabel(xlabel, fontsize=_SCATTER_LABEL_FS)
    max_pairs = max((s.n_pairs for s in summaries), default=1)
    ax.set_xlim(0, max_pairs * 1.05)
    if legend_loc:
        handles, leg_labels = ax.get_legend_handles_labels()
        extra_text_index = None
        if extra_legend_label:
            extra_handle = mpatches.Patch(color="none", label=extra_legend_label)
            if extra_legend_label_last:
                handles = handles + [extra_handle]
                leg_labels = leg_labels + [extra_legend_label]
                extra_text_index = -1
            else:
                handles = [extra_handle] + handles
                leg_labels = [extra_legend_label] + leg_labels
                extra_text_index = 0
        leg = ax.legend(handles=handles, labels=leg_labels, loc=legend_loc,
                         fontsize=_SCATTER_LABEL_FS, frameon=True)
        if extra_text_index is not None:
            # extra_legend_label has no visible patch, so its text renders
            # where the other entries' label text starts (handle width +
            # handle-text gap to the right of the patches). Shift it left by
            # that same gap so it starts flush with the patches instead.
            text = leg.get_texts()[extra_text_index]
            shift_pts = -(mpl.rcParams["legend.handlelength"]
                          + mpl.rcParams["legend.handletextpad"]) * _SCATTER_LABEL_FS
            text.set_transform(
                offset_copy(text.get_transform(), fig=ax.figure, x=shift_pts, y=0, units="points")
            )
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)


def _draw_overlap_ppi_bars(
    ax: plt.Axes, summaries: list, labels: list[str],
    *, panel_label: str | None = None, collapse_impossible: bool = True,
    impossible_label: str = "Observable FP",
    fdr_label_in_legend: bool = True,
    fdr_label_last: bool = False,
) -> None:
    """Stacked horizontal bars for PPI protein-pair overlap.

    Three segments (Shared / Pipeline exclusive / Impossible) by default —
    see :func:`_draw_overlap_bars`'s ``collapse_impossible``.

    Engines in :data:`_PPI_FROM_CSM` have no dedicated PPI-level FDR — their
    protein pairs are aggregated from a lower level — so their tick label gets
    the same " *" fig5 uses, with a matching footnote under the panel.

    ``fdr_label_in_legend=False`` moves the "@ 2% FDR" caveat out of the
    legend and into a plain text label above the top-right corner of the
    axes instead — used where the legend sits inside a cramped composite
    panel. ``fdr_label_last=True`` keeps it in the legend but appends it
    below the segment patches instead of prepending it above them.
    """
    marked = [lbl + " *" if s.engine in _PPI_FROM_CSM else lbl
              for s, lbl in zip(summaries, labels)]
    _draw_overlap_bars(ax, summaries, marked,
                       xlabel="Number of distinct protein pairs",
                       panel_label=panel_label, legend_loc="center",
                       collapse_impossible=collapse_impossible,
                       impossible_label=impossible_label,
                       extra_legend_label="@ 2% FDR" if fdr_label_in_legend else None,
                       extra_legend_label_last=fdr_label_last)
    if not fdr_label_in_legend:
        ax.text(0.98, 0.97, "@ 2% FDR", ha="right", va="top",
                fontsize=_SCATTER_LABEL_FS, transform=ax.transAxes)
    if any(s.engine in _PPI_FROM_CSM for s in summaries):
        ax.text(1.0, -0.1, "* PPI extrapolated from a lower level (no dedicated PPI FDR)",
                ha="right", va="top", fontsize=_SCATTER_LABEL_FS * 0.85,
                style="italic", alpha=0.8, transform=ax.transAxes, clip_on=False)


def plot_fig7_overlap_ppi() -> None:
    """Figure 7 (PPI): cross-engine protein-pair overlap for ecoli_human PPI."""
    summaries, labels = _load_overlap_ppi_data()
    if not summaries:
        print("  [overlap_ppi] data not found — skipping fig7_ppi")
        return
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(summaries))))
    _draw_overlap_ppi_bars(ax, summaries, labels)
    fig.tight_layout()
    _save(fig, "fig7_overlap_ppi")


def plot_fig_s1_overlap_em() -> None:
    """Supplement: cross-engine peptide-pair overlap for ecoli_mycpn CSM.

    Impossible = match_origin == 'mixed' (cross-species link, always impossible
    regardless of spectrum origin). Three-segment stacked bars identical to fig7.
    """
    parquet = NORMALISED_DIR / "ecoli_mycpn" / "CSM" / "combined.parquet"
    if not parquet.exists():
        print(f"  [overlap_em] {parquet} not found — skipping fig_s1")
        return

    df = pd.read_parquet(parquet)
    df = df[~df["search_engine"].isin(EXCLUDED_ENGINES)]
    df = add_decoy_category(df)

    # For ecoli_mycpn: impossible = cross-species link (match_origin == 'mixed').
    # This is unconditional (no per-scan lookup required).
    if "match_origin" not in df.columns:
        print("  [overlap_em] match_origin column missing — skipping fig_s1")
        return
    df["impossible"] = df["match_origin"] == "mixed"

    summaries, _pair_table = compute_peptide_pair_overlap(df)
    summaries = sorted(summaries, key=lambda s: s.n_pairs, reverse=True)

    labels = [ENGINE_LABELS.get(e, e) for e in (s.engine for s in summaries)]

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(summaries))))
    _draw_overlap_bars(ax, summaries, labels,
                       xlabel="Number of distinct peptide pairs",
                       legend_loc="upper right", collapse_impossible=True)
    fig.tight_layout()
    _save(fig, "fig_s1_overlap_ecoli_mycpn")


def _overlap_summaries(dataset: str, level: str, entrapment_only: bool = False) -> list:
    """Cross-engine pair-overlap summaries for one (dataset, level).

    CSM yields peptide pairs, PPI protein pairs. Restricted to ``ENGINE_ORDER``
    so the ``*_csm`` alternative PPI roll-ups are not double-counted.

    Each dataset has one criterion that is intrinsic to its design — a human
    entrapment match for E.coli–Human, a cross-species link for
    E.coli–Mycoplasma — and both additionally get the iBAQ co-fractionation
    test applied to their E. coli–E. coli pairs, unless *entrapment_only* is
    set. The two datasets share the same E. coli lysate, so the same
    co-fractionation table applies to both; restricting it to one of them
    would make the panels answer different questions.
    """
    parquet = NORMALISED_DIR / dataset / level / "combined.parquet"
    if not parquet.exists():
        print(f"  [overlap] {parquet} not found — skipping")
        return []

    df = pd.read_parquet(parquet)
    df = df[df["search_engine"].isin(ENGINE_ORDER)].copy()
    df = add_decoy_category(df)

    if dataset == "ecoli_human":
        ecoli_set, human_set = load_organism_sets(
            ecoli_path=REPO_ROOT / "input" / "ecoli_protein_accessions.txt",
            human_path=REPO_ROOT / "input" / "human_protein_accessions.txt",
        )
        df = add_origin_flags(df, ecoli_set=ecoli_set, human_set=human_set)
        ibaq_presence = load_ibaq_presence(REPO_ROOT / "input" / "proteinGroups.txt", cutoff=1e6)
        df = add_impossible_flag_ecoli_human(df, ibaq_presence=ibaq_presence)
        if entrapment_only:
            df["impossible"] = df["impossible_reason"] == "human"
    else:
        if "match_origin" not in df.columns:
            print(f"  [overlap] match_origin missing for {dataset} {level} — skipping")
            return []
        df["impossible"] = df["match_origin"] == "mixed"
        if not entrapment_only:
            from src.analysis.ibaq_copresence import _check_copresence
            ibaq = load_ibaq_presence(REPO_ROOT / "input" / "proteinGroups.txt", cutoff=1e6)
            ecoli_pairs = df["match_origin"] == "ecoli"
            if ecoli_pairs.any():
                nocp = df.loc[ecoli_pairs].apply(
                    lambda r: not _check_copresence(r["Protein1"], r["Protein2"], ibaq),
                    axis=1)
                df.loc[nocp[nocp].index, "impossible"] = True

    compute = compute_peptide_pair_overlap if level == "CSM" else compute_protein_pair_overlap
    summaries, _ = compute(df)
    return summaries


def plot_fig_s5_overlap_grid(entrapment_only: bool = False,
                             split_cofractionation: bool = False,
                             name_suffix: str = "") -> None:
    """Supplement: 2x2 cross-engine overlap grid.

    Rows are datasets (E.coli–Human top, E.coli–Mycoplasma bottom); columns are
    the pair level (unique peptide pairs left, unique protein pairs right).

    Each row shares one engine order, taken from that row's peptide-pair panel,
    so an engine sits at the same y position in both of its panels and the two
    levels read against each other directly. The rows are ordered independently
    because the two datasets do not cover the same engine set.
    """
    rows = (("ecoli_human", DS_EH), ("ecoli_mycpn", DS_EM))
    cols = (("CSM", "peptide"), ("PPI", "protein"))
    letters = (("a", "b"), ("c", "d"))

    data = {(ds, lvl): _overlap_summaries(ds, lvl, entrapment_only=entrapment_only)
            for ds, _ in rows for lvl, _ in cols}
    # For the split view the same panels are recomputed with the entrapment
    # criterion alone; the difference is the co-fractionation-only share.
    ent = ({(ds, lvl): _overlap_summaries(ds, lvl, entrapment_only=True)
            for ds, _ in rows for lvl, _ in cols}
           if split_cofractionation else None)
    if not any(data.values()):
        print("  [overlap_grid] no data — skipping fig_s5")
        return

    n_max = max((len(v) for v in data.values()), default=1)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(2, 2, figsize=(14, max(6.0, n_max * 0.55 + 2.0)),
                             squeeze=False)

    for r, (ds, ds_label) in enumerate(rows):
        base = data[(ds, "CSM")] or data[(ds, "PPI")]
        order = [s.engine for s in sorted(base, key=lambda s: s.n_pairs, reverse=True)]
        for c, (lvl, unit) in enumerate(cols):
            by_engine = {s.engine: s for s in data[(ds, lvl)]}
            ordered = [by_engine[e] for e in order if e in by_engine]
            _draw_overlap_bars(
                axes[r][c], ordered,
                [ENGINE_LABELS.get(s.engine, s.engine) for s in ordered],
                xlabel=f"{ds_label}  distinct {unit} pairs",
                panel_label=letters[r][c],
                entrapment_summaries=ent[(ds, lvl)] if ent else None,
                collapse_impossible=not split_cofractionation,
            )

    # Leading plain-text entry: every count in this figure is each pipeline's
    # own declared-2%-FDR result set, stated once here rather than per panel.
    if split_cofractionation:
        # The one variant that keeps the impossible reason visible: its
        # reason to exist is showing the co-fractionation share separately,
        # so shared/exclusive collapses here instead (see _draw_overlap_bars).
        # "@ 2% FDR" sits vertically centered beside the four segment
        # patches, which form a 2x2 grid (Shared/Pipeline exclusive on top,
        # the two Observable-FP reasons below) rather than a single row — a
        # plain ncol-based fig.legend() can't mix a row-spanning entry with a
        # grid, so this is built from offsetbox primitives instead.
        fs = _SCATTER_LABEL_FS

        def _swatch_row(color: str, text: str) -> HPacker:
            da = DrawingArea(fs * 1.4, fs * 0.6, 0, 0)
            da.add_artist(mpatches.Rectangle((0, 0), fs * 1.4, fs * 0.6,
                                             facecolor=color, edgecolor="none"))
            return HPacker(children=[da, TextArea(text, textprops=dict(fontsize=fs))],
                           align="center", pad=0, sep=4)

        col1 = VPacker(children=[
            _swatch_row(C_OVL_SH_POS,  "Shared"),
            _swatch_row(C_OVL_SH_NOCP, "OFT: not co-fractionating"),
        ], align="left", pad=0, sep=6)
        col2 = VPacker(children=[
            _swatch_row(C_OVL_UN_POS, "Pipeline exclusive"),
            _swatch_row(C_IMPOSS,     "OFT: entrapment / wrong species"),
        ], align="left", pad=0, sep=6)
        grid = HPacker(children=[col1, col2], align="center", pad=0, sep=16)
        content = HPacker(children=[TextArea("@ 2% FDR", textprops=dict(fontsize=fs)), grid],
                          align="center", pad=0, sep=16)
        box = AnchoredOffsetbox(loc="lower center", child=content, pad=0.5, borderpad=0.6,
                                bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
                                frameon=True)
        fig.add_artist(box)
    else:
        handles = [
            mpatches.Patch(color="none", label="@ 2% FDR"),
            mpatches.Patch(color=C_OVL_SH_POS, label="Shared"),
            mpatches.Patch(color=C_OVL_UN_POS, label="Pipeline exclusive"),
            mpatches.Patch(color=C_IMPOSS,     label="OFT"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4,
                  fontsize=_SCATTER_LABEL_FS, frameon=True, bbox_to_anchor=(0.5, 0.0))
    # State the criterion on the figure: the two variants differ only in the
    # E. coli–Human rows and are otherwise identical.
    if split_cofractionation:
        note = f"impossible split by cause: {SP_HUMAN} entrapment match vs never co-fractionating"
    elif entrapment_only:
        note = f"impossible = {SP_HUMAN} entrapment match only"
    else:
        note = f"impossible = {SP_HUMAN} entrapment match, or {SP_ECOLI} pair never co-fractionating"
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, f"fig_s5_overlap_grid{name_suffix}")



# ── helpers shared by the two multi-panel figures ─────────────────────────────

def _load_eh_annotated_csm(entrapment_only: bool = True) -> pd.DataFrame:
    """Load ecoli_human CSM parquet with decoy_cat, match_origin, and impossible flags.

    ``entrapment_only`` restricts ``impossible`` to the human-entrapment
    criterion, dropping matches that are impossible only because their two
    E. coli proteins never co-eluted.

    It defaults to True because that is the only criterion a decoy count can be
    compared against: the 2:1 decoy:entrapment expectation comes from the
    database (human is ~49 % of the target tryptic peptide space, decoys are a
    1:1 reversed copy), and co-fractionation carries no such expectation. It
    also over-flags — most E. coli accessions never clear the iBAQ cutoff — so
    mixing it in roughly halves every observed ratio. Every current caller wants
    entrapment only and says so explicitly; pass ``False`` to opt into the
    broader definition, as the co-fractionation variants of the overlap figures
    do through their own loaders.
    """
    df = pd.read_parquet(NORMALISED_DIR / "ecoli_human" / "CSM" / "combined.parquet")
    df = add_decoy_category(df)
    ecoli_set, human_set = load_organism_sets(
        REPO_ROOT / "input" / "ecoli_protein_accessions.txt",
        REPO_ROOT / "input" / "human_protein_accessions.txt",
    )
    df = add_origin_flags(df, ecoli_set, human_set)
    ibaq = load_ibaq_presence(REPO_ROOT / "input" / "proteinGroups.txt")
    df = add_impossible_flag_ecoli_human(df, ibaq)
    # Kept separate from ``impossible`` so a figure can show the two criteria
    # side by side. add_impossible_flag_ecoli_human only sets this reason where
    # the entrapment test did *not* fire, so the two classes are disjoint.
    df["cofrac_fp"] = ((df["decoy_cat"] == "TT")
                       & (df["impossible_reason"] == "not_copresent"))
    if entrapment_only:
        df["impossible"] = df["impossible_reason"] == "human"
    return df


def _load_em_annotated_csm() -> pd.DataFrame:
    """Load ecoli_mycpn CSM parquet with decoy_cat, spectrum_origin and impossible flags.

    The ecoli_mycpn counterpart of :func:`_load_eh_annotated_csm`. "Impossible"
    is the spectrum-origin test — an E. coli spectrum matched to Mycoplasma
    proteins or the reverse — which is the definition behind
    ``tt_impossible_fpr_pct`` in the summaries, so this and the impossible-FPR
    figures count the same false positives.

    The test is applied per engine because the scan convention differs between
    them (``ENGINE_SCAN_TYPE``). An engine with no known convention falls back
    to the unconditional cross-species test the overlap figures use; that is
    weaker — blind to a same-organism pair matched to the wrong organism — but
    keeps the engine in the figure rather than dropping it silently.
    """
    df = pd.read_parquet(NORMALISED_DIR / "ecoli_mycpn" / "CSM" / "combined.parquet")
    df = df[~df["search_engine"].isin(EXCLUDED_ENGINES)].copy()
    df = add_decoy_category(df)
    lookup = load_merged_lookup(REPO_ROOT / "input" / "merged_files.json")

    parts: list[pd.DataFrame] = []
    for eng, g in df.groupby("search_engine", sort=False):
        scan_type = ENGINE_SCAN_TYPE.get(eng)
        if scan_type is None:
            g = g.copy()
            g["spectrum_origin"] = "unknown"
            g["impossible"] = g["match_origin"] == "mixed"
        else:
            g = add_spectrum_origin(g.copy(), lookup, scan_type)
            g = add_impossible_flag(g)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["cofrac_fp"] = _never_cofractionating(out)
    return out


def _never_cofractionating(df: pd.DataFrame) -> pd.Series:
    """TT E. coli-E. coli matches whose two proteins never share an SEC fraction.

    The ecoli_mycpn parquet carries no ``impossible_reason``, so the iBAQ test
    that :func:`_load_eh_annotated_csm` inherits has to be run here. Both
    datasets come from the same E. coli lysate, so the same co-presence table
    applies — the argument :func:`_em_impossible_with_copresence` already makes.

    Restricted to pairs the spectrum-origin test calls *possible*, so this class
    and ``impossible`` stay disjoint. Unlike the CSM branch of
    :func:`_em_impossible_with_copresence` this does not also require
    ``spectrum_origin == "ecoli"``: engines with no known scan convention have
    origin "unknown" and would otherwise drop out of the figure entirely, and
    ``match_origin == "ecoli"`` on a possible match already implies the spectrum
    was not Mycoplasma.
    """
    from src.analysis.ibaq_copresence import _check_copresence, load_ibaq_presence

    ibaq = load_ibaq_presence(REPO_ROOT / "input" / "proteinGroups.txt")
    mask = ((df["decoy_cat"] == "TT") & ~df["impossible"]
            & (df["match_origin"] == "ecoli"))
    out = pd.Series(False, index=df.index)
    if not mask.any():
        return out
    hit = df.loc[mask].apply(
        lambda r: not _check_copresence(r["Protein1"], r["Protein2"], ibaq), axis=1)
    out.loc[hit[hit].index] = True
    return out


def _ks_decoy_vs_fp(df: pd.DataFrame) -> pd.DataFrame:
    """Per-engine KS distance between the decoy and known-FP score distributions.

    The dataset-agnostic half of :func:`_compute_ks_and_ratio`: it needs only
    ``decoy_cat``, ``impossible`` and ``score``, so it serves both datasets.
    Returns a DataFrame indexed by search_engine with ks_d, n_decoy, n_false_tgt.
    """
    from scipy.stats import ks_2samp

    rows = []
    for eng, g in df.groupby("search_engine"):
        decoys    = g[g["decoy_cat"].isin(["TD", "DD"])]["score"].dropna().values
        false_tgt = g[(g["decoy_cat"] == "TT") & g["impossible"]]["score"].dropna().values
        n_d, n_f  = len(decoys), len(false_tgt)
        ks_d      = ks_2samp(decoys, false_tgt).statistic if (n_d > 5 and n_f > 5) else float("nan")
        rows.append({"search_engine": eng, "ks_d": ks_d,
                     "n_decoy": n_d, "n_false_tgt": n_f})
    return pd.DataFrame(rows).set_index("search_engine")


def _compute_ks_and_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Per-engine KS D (decoy vs impossible score distributions) and log2 count-ratio deviation.

    Expected decoy:entrapment ratio = 2 for ecoli_human (decoy DB ≈ 2× entrapment
    DB), so the ratio half is ecoli_human-only; :func:`_ks_decoy_vs_fp` is the
    part that generalises. Returns DataFrame indexed by search_engine.

    The ratio is taken against ``n_entrap`` — TT matches involving a non-E. coli
    protein — and not against ``n_false_tgt``, which follows whichever
    ``impossible`` definition the caller loaded. Only the entrapment count has
    the database-composition expectation behind it; with a co-fractionation-
    inclusive frame the two differ and the deviation would be measured against
    the wrong denominator. ``n_entrap`` is also what
    :func:`_draw_ks_ratio_scatter` puts on its x-axis, so the two agree.
    """
    import math

    EXPECTED = 2.0
    stats = _ks_decoy_vs_fp(df)
    entrap = (df[(df["decoy_cat"] == "TT") & (df["match_origin"] != "ecoli")]
              .groupby("search_engine")["score"].count())
    stats["n_entrap"] = entrap.reindex(stats.index).fillna(0).astype(int)

    def _dev(n_d: int, n_f: int) -> float:
        obs_ratio = n_d / n_f if n_f > 0 else float("nan")
        return math.log2(obs_ratio / EXPECTED) if obs_ratio > 0 else float("nan")

    stats["log2_ratio_dev"] = [
        _dev(d, e) for d, e in zip(stats["n_decoy"], stats["n_entrap"])
    ]
    return stats[["ks_d", "log2_ratio_dev", "n_decoy", "n_false_tgt", "n_entrap"]]


def _draw_norm_score_hist(
    ax: plt.Axes,
    eng_df: pd.DataFrame,
    body_label: str,
    ks_d: float,
    xlabel: str = "Score",
    chart_title: str | None = None,
    show_ylabel: bool = True,
    show_cofrac: bool = False,
    fp_label: str = "Obs. FP",
) -> None:
    """Score-distribution histograms (absolute counts, raw score axis) for one engine panel.

    ``fp_label`` names the observable-false-target curve/legend entry —
    "Obs. FP" by default, overridden to "OFT" by fig_multi1 panel e alone so
    FigS1/FigS2 (this function's other caller) keep their own wording.

    ``show_cofrac`` adds a third series for matches the iBAQ co-fractionation
    test rejects but the entrapment / spectrum-origin test accepts. Off by
    default: fig_multi1 is deliberately entrapment-only, and it shares this
    helper with the supplementary grids that do want the extra class.
    """
    decoys    = eng_df[eng_df["decoy_cat"].isin(["TD", "DD"])]["score"].dropna().values
    false_tgt = eng_df[(eng_df["decoy_cat"] == "TT") & eng_df["impossible"]]["score"].dropna().values
    # Third, disjoint class: matches the co-fractionation test rejects but the
    # entrapment / spectrum-origin test accepts. Drawn only where the caller
    # supplied it, so panels fed a frame without the column are unchanged.
    cofrac = (eng_df[eng_df["cofrac_fp"]]["score"].dropna().values
              if show_cofrac and "cofrac_fp" in eng_df.columns else np.array([]))

    no_decoys   = len(decoys) < 5
    show_cofrac = show_cofrac and len(cofrac) >= 5

    if len(false_tgt) < 5:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=_SCATTER_LABEL_FS)
        title_text = chart_title if chart_title else body_label
        if title_text:
            ax.set_title(title_text, fontsize=_SCATTER_LABEL_FS, fontweight="bold")
        return

    # The range has to span every series drawn, or the extra class gets clipped.
    ref = np.concatenate(([false_tgt] if no_decoys else [decoys, false_tgt])
                         + ([cofrac] if show_cofrac else []))
    lo = np.nanpercentile(ref, 1)
    hi = np.nanpercentile(ref, 99)
    bins = np.linspace(lo, hi, 30)
    cx   = 0.5 * (bins[:-1] + bins[1:])

    f_h, _ = np.histogram(false_tgt, bins=bins)
    ax.step(cx, f_h, where="mid", color=C_IMPOSS, lw=1.8, label=f"{fp_label} (n={len(false_tgt)})")
    ax.fill_between(cx, f_h, step="mid", alpha=0.30, color=C_IMPOSS)
    # Hatch as its own layer, facecolor="none": a single fill_between with
    # both alpha and hatch set renders fine in PNG (Agg draws hatch lines at
    # full opacity regardless of the patch's alpha) but is a known matplotlib
    # SVG-backend bug — the pattern tile's background rect comes out opaque
    # instead of transparent, so the hatch lines all but disappear into it.
    # Splitting the hatch onto a separate, alpha-free, facecolor="none" layer
    # sidesteps that: its pattern tile has nothing to fill but the lines.
    fp_hatch = ax.fill_between(cx, f_h, step="mid", facecolor="none", hatch="////", edgecolor=C_IMPOSS)
    fp_hatch.set_hatch_linewidth(0.5)  # half the 1.0 default — the full-width lines read too heavy

    if show_cofrac:
        c_h, _ = np.histogram(cofrac, bins=bins)
        ax.step(cx, c_h, where="mid", color=C_COFRAC, lw=1.8,
                label=f"No co-fract. (n={len(cofrac)})")
        ax.fill_between(cx, c_h, step="mid", alpha=0.18, color=C_COFRAC)

    if not no_decoys:
        d_h, _ = np.histogram(decoys, bins=bins)
        ax.step(cx, d_h, where="mid", color=C_DECOY,  lw=1.8, label=f"Decoy (n={len(decoys)})")
        ax.fill_between(cx, d_h, step="mid", alpha=0.22, color=C_DECOY)

    # KS distance as first (invisible-marker) legend entry
    ks_str = f"Distance = {ks_d:.2f}" if not np.isnan(ks_d) else "Distance = N/A"
    # ks_handle    = mpatches.Patch(color="none", label=ks_str)
    fp_handle    = plt.Line2D([0], [0], color=C_IMPOSS, lw=1.8, label=f"{fp_label} (n={len(false_tgt)})")
    if no_decoys:
        handles = [mpatches.Patch(color="none", label="Decoys ND"), fp_handle]
    else:
        handles = [plt.Line2D([0], [0], color=C_DECOY, lw=1.8,
                              label=f"Decoy (n={len(decoys)})"), fp_handle]
    if show_cofrac:
        handles.append(plt.Line2D([0], [0], color=C_COFRAC, lw=1.8,
                                  label=f"No co-fract. (n={len(cofrac)})"))
    leg = ax.legend(handles=handles, fontsize=_SCATTER_LABEL_FS,
                    frameon=False, loc="upper left")

    for txt in leg.get_texts():
        add_faded_background(ax, txt)
    for handle in leg.legend_handles:
        add_faded_background(ax, handle)

    # Body label (reliable / unreliable) as bold text inside the chart
    # The co-fractionation series makes the legend three rows tall, which runs
    # past the middle of the panel on wide engine names — so that variant puts
    # the label in the far corner. Without it the original placement stands.
    bl = (ax.text(0.97, 0.97, body_label, ha="right", va="top", transform=ax.transAxes,
                  fontsize=_SCATTER_LABEL_FS, fontweight="bold")
          if show_cofrac else
          ax.text(0.6, 0.80, body_label, ha="center", va="center", transform=ax.transAxes,
                  fontsize=_SCATTER_LABEL_FS, fontweight="bold"))
    add_faded_background(ax, bl)

    ax.set_xlabel(xlabel, fontsize=_SCATTER_LABEL_FS)
    if show_ylabel:
        ax.set_ylabel("Count", fontsize=_SCATTER_LABEL_FS)
    if chart_title:
        ax.set_title(chart_title, fontsize=_SCATTER_LABEL_FS, fontweight="bold")
    ax.tick_params(labelsize=_SCATTER_LABEL_FS, labelbottom=False, labelleft=False)
    ax.grid(False)


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def plot_supp_fig2_ks_scatter() -> None:
    """Supplement Figure 2: KS D vs entrapment/decoy count ratio (standalone).

    Entrapment criterion only, matching FigS1. Both axes then describe the same
    false-positive set: KS D is the shape agreement between the decoy and
    entrapment score distributions, the x-axis their count ratio. Loading the
    co-fractionation-inclusive frame would measure shape against one set and
    counts against another, since the x-axis reads ``n_entrap``.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    df    = _load_eh_annotated_csm(entrapment_only=True)
    stats = _compute_ks_and_ratio(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_ks_ratio_scatter(ax, stats, panel_label=None)
    fig.tight_layout()
    _save(fig, "supp_fig2_ks_scatter")


def _draw_fp_d_histogram_grid(
    df: pd.DataFrame, stats: pd.DataFrame, name: str, n_cols: int = 4,
) -> None:
    """One decoy-vs-known-FP score histogram per engine, laid out in a grid.

    Shared by FigS1 (E. coli–Human) and FigS2 (E. coli–Mycoplasma) so the two
    stay directly comparable. Only engines actually present in *df* get a cell —
    ecoli_mycpn has no pLink2 or XlinkX PD2.5 run — and spare cells are hidden.
    pLink3 (post-filter) is excluded here: it has no decoys, so its panel is
    just "Decoys ND" against the Obs. FP curve with nothing to hold up
    against; its comparison against pLink3 (FDR) lives in :func:`plot_plink3_scores`
    instead.

    Panels run alphabetically by displayed name so a reader can find an engine
    without knowing ENGINE_ORDER. The sort is case-insensitive: a plain sort
    would push pLink2/pLink3/xiSEARCH past the capitalised names. ENGINE_ORDER
    still decides *membership*, keeping the alternative PPI roll-ups out.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    present = set(df["search_engine"].unique())
    engines = sorted(
        (e for e in ENGINE_ORDER if e in present and e != "plink3_filtered"),
        key=lambda e: ENGINE_LABELS.get(e, e).lower(),
    )

    n_rows = int(np.ceil(len(engines) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False)
    flat_axes = list(axes.ravel())

    for i, eng in enumerate(engines):
        _draw_norm_score_hist(
            flat_axes[i],
            df[df["search_engine"] == eng],
            ENGINE_LABELS.get(eng, eng),
            float(stats.loc[eng, "ks_d"]) if eng in stats.index else float("nan"),
            show_cofrac=True,
            xlabel=NATIVE_SCORE_COLUMN.get(eng, "Score"),
            show_ylabel=(i % n_cols == 0),
        )

    for ax in flat_axes[len(engines):]:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, name)


def plot_supp_figs1_fp_d_histogram() -> None:
    """Supplement Figure S1: decoy vs known-FP score histograms for all engines.

    Uses E. coli–Human CSM data and shows one panel per search engine.
    "Known FP" is a TT match to a human entrapment protein — the entrapment
    criterion only.

    The iBAQ co-fractionation criterion is deliberately excluded. This figure
    exists to be read against the 2:1 decoy:entrapment expectation, and that
    number comes purely from the database (human is ~49 % of the target tryptic
    peptide space, decoys are a 1:1 reversed copy). Co-fractionation is an
    orthogonal test with no decoy-count expectation attached, and it over-flags
    — most E. coli accessions never clear the iBAQ cutoff — so folding it in
    roughly halves every ratio and breaks the comparison. The co-fractionation
    view lives in fig_s5_overlap_grid instead.
    """
    df = _load_eh_annotated_csm(entrapment_only=True)
    _draw_fp_d_histogram_grid(
        df,
        _compute_ks_and_ratio(df),
        "FigS1_FP_D_histogram",
    )


def plot_supp_figs2_fp_d_histogram() -> None:
    """Supplement Figure S2: the FigS1 comparison on E. coli–Mycoplasma CSMs.

    Same panels and definitions as FigS1, with "known FP" supplied by the other
    reference dataset: a CSM whose spectrum came from one organism but was
    matched to the other. Reading the two side by side separates engine
    behaviour from any quirk of a single ground-truth construction.
    """
    df = _load_em_annotated_csm()
    _draw_fp_d_histogram_grid(
        df,
        _ks_decoy_vs_fp(df),
        "FigS2_FP_D_histogram",
    )


def plot_plink3_scores() -> None:
    """Supplement: pLink3 (FDR) vs pLink3 (post-filter) score distributions.

    pLink3 (post-filter) reports validated targets only — no decoys — so its
    FigS1/FigS2 panel is just "Decoys ND" against the Obs. FP curve, with
    nothing to hold up against. This figure puts the two pLink3 derivations
    side by side instead, on both reference datasets: top row E. coli–Human,
    bottom row E. coli–Mycoplasma; left column pLink3 (FDR, the full
    re-exported report at the declared 2 % FDR), right column pLink3
    (post-filter, pLink3's own final validated-targets report). Same drawing
    function and statistics as FigS1/FigS2 (entrapment-only for E. coli–Human,
    matching :func:`plot_supp_figs1_fp_d_histogram`), so the numbers agree.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    eh = _load_eh_annotated_csm(entrapment_only=True)
    em = _load_em_annotated_csm()
    eh_stats = _compute_ks_and_ratio(eh)
    em_stats = _ks_decoy_vs_fp(em)

    engines = ["plink3", "plink3_filtered"]
    rows = [(eh, eh_stats, DS_EH, ("a", "b")), (em, em_stats, DS_EM, ("c", "d"))]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), squeeze=False)
    for r, (df, stats, ds_label, panel_labels) in enumerate(rows):
        for c, (eng, lbl) in enumerate(zip(engines, panel_labels)):
            ax = axes[r][c]
            _draw_panel_label(ax, lbl)
            sub = df[df["search_engine"] == eng]
            _draw_norm_score_hist(
                ax, sub, ENGINE_LABELS.get(eng, eng),
                float(stats.loc[eng, "ks_d"]) if eng in stats.index else float("nan"),
                xlabel=NATIVE_SCORE_COLUMN.get(eng, "Score"),
                chart_title=ds_label,
                show_cofrac=True,
                show_ylabel=(c == 0),
            )
    fig.tight_layout()
    _save(fig, "plink3_scores")


def _draw_ks_ratio_scatter(
    ax: plt.Axes,
    stats: pd.DataFrame,
    *,
    panel_label: str | None = "d",
) -> None:
    """KS D vs entrapment/decoy count ratio scatter (panel 1d).

    Parameters
    ----------
    ax:          Axes to draw on.
    stats:       DataFrame from ``_compute_ks_and_ratio()``.
    panel_label: Bold letter placed in the upper-left corner; pass ``None``
                 to suppress it (e.g. when embedding in a standalone figure).
    """
    if panel_label:
        _draw_panel_label(ax, panel_label, x=-0.03, y=1.06)

    ks_max  = max(float(stats.loc[e, "ks_d"]) for e in stats.index
                  if not np.isnan(float(stats.loc[e, "ks_d"])))
    norm_ks = plt.Normalize(0, ks_max)
    cmap_ks = cm.RdYlGn_r

    scatter_texts = []
    for eng in ENGINE_ORDER:
        if eng not in stats.index:
            continue
        ks_d = float(stats.loc[eng, "ks_d"])
        n_d  = float(stats.loc[eng, "n_decoy"])
        n_e  = float(stats.loc[eng, "n_entrap"])
        if np.isnan(ks_d) or n_d == 0:
            continue  # Mango: no decoys
        ratio = n_e / n_d
        color = cmap_ks(norm_ks(ks_d))
        ax.scatter(ratio, ks_d, s=90, color=color, zorder=4)
        scatter_texts.append(ax.text(ratio, ks_d, ENGINE_LABELS.get(eng, eng), fontsize=_SCATTER_LABEL_FS,))

    adjust_text(scatter_texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

    ax.axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.65,
               label="expected ratio = 0.5\n(decoy DB ≈ 2× entrapment DB)")
    ax.set_xlabel(f"Entrapment count / Decoy count  ({DS_EH}, CSM)", fontsize=_SCATTER_LABEL_FS)
    ax.set_ylabel("KS D", fontsize=_SCATTER_LABEL_FS)
    ax.legend(fontsize=_SCATTER_LABEL_FS, frameon=False, loc="upper left")
    ax.set_xlim(left=0)
    ax.set_ylim(0, ks_max * 1.2)
    sns.despine(ax=ax)
    ax.text(1.0, -0.12,
            "Mango omitted: decoy-free engine — KS D not applicable.",
            ha="right", va="top", fontsize=_SCATTER_LABEL_FS, style="italic", color="grey",
            transform=ax.transAxes)


def _draw_assumption_schematic(ax: plt.Axes, *, panel_label: str | None = "a") -> None:
    """Upper conceptual schematic for fig_multi1 (the assumption being tested)."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    if panel_label:
        _draw_panel_label(ax, panel_label, x=-0.08, y=1.01)

    # Row 1: Accepted TT bar — green (true targets) | gap | red (false target) | hatched red (Obs. FP)
    bar_y0, bar_h = 7.0, 2.0           # 10 % taller, to clear the 3-line FP label
    bar_yc = bar_y0 + bar_h / 2
    # Labels sit slightly below the bar's centre. va="center" centres the text
    # *layout* box, which reserves descent under the last line that "matches" /
    # "FP" do not use, so the ink reads high: measured 1 px of clearance above
    # "False" against 5 px below "matches". Half that difference re-centres the
    # visible block on the bar.
    txt_yc = bar_yc - 0.066
    gap = 0.20                          # visual gap between green and red section
    green_end = 6.7                     # green bar right edge
    fp_x = green_end + gap             # red section starts here (6.90)
    # Widened at green's expense (same right edge, ~9.92, as before) so
    # "Decoy matches" — same width as the red/FP block below — fits inside
    # its bar instead of overflowing it.
    fp_total_w = 3.02
    fp_left_w  = fp_total_w * 0.65     # False target matches portion
    fp_right_w = fp_total_w * 0.35     # Obs. FP hatched portion

    ax.add_patch(mpatches.FancyBboxPatch((0.3, bar_y0), green_end - 0.3, bar_h,
        boxstyle="round,pad=0.08", fc="#50cf50", ec="none", alpha=0.75, zorder=2))
    ax.text((0.3 + green_end) / 2, txt_yc, "True target matches",
            ha="center", va="center", fontsize=_SCATTER_LABEL_FS, color="white", fontweight="bold")

    ax.add_patch(mpatches.FancyBboxPatch((fp_x, bar_y0), fp_total_w, bar_h,
        boxstyle="round,pad=0.08", fc="#d62728", ec="none", alpha=0.65, zorder=2))
    ax.text(fp_x + fp_left_w / 2, txt_yc, "False\ntarget\nmatches",
            ha="center", va="center", fontsize=_SCATTER_LABEL_FS, color="white", fontweight="bold",
            linespacing=0.85, zorder=5)

    obs_x = fp_x + fp_left_w
    # Two patches, not one translucent one: this block overlaps the red "false
    # target matches" block, so an alpha fill would composite over *that* and
    # come out darker than panel e. The opaque underlay is panel e's effective
    # fill — C_IMPOSS at alpha 0.30 over white — so both panels land on the
    # same background whatever sits behind them.
    ax.add_patch(mpatches.FancyBboxPatch((obs_x, bar_y0), fp_right_w, bar_h,
        boxstyle="round,pad=0.05", fc="#f2bdbe", ec="none", zorder=3))
    ax.add_patch(mpatches.FancyBboxPatch((obs_x, bar_y0), fp_right_w, bar_h,
        boxstyle="round,pad=0.05", fc="none", ec=C_IMPOSS, lw=0.8,
        hatch="////", zorder=3.1))
    # Red glyphs on the hatch, lifted off it with two white strokes: a solid
    # inner one for contrast against the hatch lines themselves, and a wider,
    # half-transparent outer one as a soft halo against the lighter gaps
    # between them. withStroke draws its own stroke-then-fill each time, so
    # the two layer in the order given — wider/fainter first, solid on top.
    ax.text(obs_x + fp_right_w / 2, txt_yc, "OFT",
            ha="center", va="center", fontsize=_SCATTER_LABEL_FS, color="#bc2223ff",
            fontweight="bold", linespacing=0.85, zorder=5,
            path_effects=[mpatheffects.withStroke(linewidth=3.0, foreground="white", alpha=0.5),
                          mpatheffects.withStroke(linewidth=1.5, foreground="white")])

    # Row 2: Question box (left, aligned with green) + Decoy bar (right, aligned with red/FP)
    # Decoy bar: same height as other bars; question box: 20 % taller to fit 3-line text
    # The 1.10 gap (was 0.50) makes room for the "≈" linking the two rows below.
    bar_h2   = bar_h                     # decoy bar height (same as row 1)
    row2_yc  = bar_y0 - 1.10 - bar_h2 / 2  # centre, leaving ~1.10 gap below row 1
    box_h    = 2.30                        # question box height (larger to fit 3-line text)
    row2_y0_bar = row2_yc - bar_h2 / 2
    row2_y0_box = row2_yc - box_h  / 2

    # Question box: explicit patch so its width never overflows into the decoy bar.
    q_x0 = 0.3
    q_x1 = green_end - 0.05
    # ax.add_patch(mpatches.FancyBboxPatch(
    #     (q_x0, row2_y0_box), q_x1 - q_x0, box_h,
    #     boxstyle="round,pad=0.08", fc="lightyellow", ec="#aaaaaa", lw=1.0, zorder=2))
    # ax.text((q_x0 + q_x1) / 2, row2_yc,
    #         "Do decoys and observable false targets\n"
    #         "behave alike after the complete\nsearch pipeline?",
    #         ha="center", va="center", fontsize=_SCATTER_LABEL_FS - 4, style="italic", zorder=3)

    ax.add_patch(mpatches.FancyBboxPatch((fp_x, row2_y0_bar), fp_total_w, bar_h2,
        boxstyle="round,pad=0.08", fc="#1f77b4", ec="none", alpha=0.80, zorder=2))
    ax.text(fp_x + fp_total_w / 2, row2_yc, "Decoy matches",
            ha="center", va="center", fontsize=_SCATTER_LABEL_FS, color="white", fontweight="bold")

    # The assumption the whole benchmark rests on. The decoy bar stands in for
    # the *whole* red block — every false target match — not just the observable
    # slice, so the dashed guides carry the red block's outer edges down to the
    # decoy bar and the "≈" reads as a claim about counts rather than position.
    link_top = bar_y0                       # underside of the red block
    link_bot = row2_y0_bar + bar_h2         # top of the decoy bar
    link_yc  = (link_top + link_bot) / 2
    _pad = 0.08   # the boxstyle pad of the two bars these guides connect
    for x in (fp_x - _pad, fp_x + fp_total_w + _pad):
        ax.plot([x, x], [link_bot, link_top], color="#999999", lw=0.9,
                ls=(0, (3, 3)), zorder=1)
    ax.text(fp_x + fp_total_w / 2, link_yc, "\u2248", ha="center", va="center",
            fontsize=_SCATTER_LABEL_FS, color="#444444", fontweight="bold",
            zorder=6, bbox=dict(boxstyle="square,pad=0.10", fc="white", ec="none"))
    # ax.text(green_end - 0.15, link_yc, "assumed to model", ha="right", va="center",
    #         fontsize=_SCATTER_LABEL_FS - 1, style="italic", color="#444444")


def _draw_concordant_discordant(ax: plt.Axes, *, panel_label: str | None = "b") -> None:
    """Mini concordant/discordant score-distribution schematic for fig_multi1."""
    ax.set_xlim(0, 10); ax.set_ylim(4.4, 7.0); ax.axis("off")
    if panel_label:
        _draw_panel_label(ax, panel_label, x=-0.08, y=1.01)

    scl    = 0.88
    y_base = 5.0
    ax.text(2.5, y_base + scl * 0.90 + 0.15, "Concordant\n→ FDR reliable",
            ha="center", fontsize=_SCATTER_LABEL_FS, fontweight="bold", va="bottom")
    ax.text(7.5, y_base + scl * 1.50 + 0.15, "Discordant\n→ FDR unreliable",
            ha="center", fontsize=_SCATTER_LABEL_FS, fontweight="bold", va="bottom")

    x_s = np.linspace(0, 1, 60)
    for ox, col_dec, col_fp, offset, fp_scale in [
        (1.0, C_DECOY, C_IMPOSS,  0.00, 0.90),
        (6.0, C_DECOY, C_IMPOSS, -0.35, 1.50),
    ]:
        y_d  = np.exp(-((x_s - 0.35) ** 2) / 0.04)
        y_fp = np.exp(-((x_s - 0.35 + offset) ** 2) / 0.04) * fp_scale
        ax.plot(ox + x_s * 3.0, y_base + scl * y_d,  color=col_dec,  lw=2.0)
        ax.plot(ox + x_s * 3.0, y_base + scl * y_fp, color=col_fp, lw=2.0, ls="--")

    ax.plot([], [], color=C_DECOY,  lw=2.0, label="Decoy")
    ax.plot([], [], color=C_IMPOSS, lw=2.0, ls="--", label="Observable False Target (OFT)")
    ax.legend(loc="lower center", fontsize=_SCATTER_LABEL_FS, frameon=False, ncol=2)


# Geometry shared by the two design schematics (fig_multi1 panels c and d), so
# the database cylinder and the outcome stacks land in the same place in both.
# Spacing is uniform at 1.75; the three 12 pt labels in d need ~1.70 minimum to
# clear each other, and the rightmost stack has to stay inside xlim=10.
_DESIGN_CYL_X = 3.80
_DESIGN_STACK_X = (5.68, 7.43, 9.18)
# "+" between adjacent stacks: the midpoint, nudged by half the sheet stagger so
# it sits between the drawn shapes rather than between their nominal centres.
_DESIGN_PLUS_X = tuple(round((a + b) / 2 + 0.045, 3)
                       for a, b in zip(_DESIGN_STACK_X, _DESIGN_STACK_X[1:]))
# Outcome stacks sit below the arrow line, so the arrow tip meets the top edge
# of the middle sheet rather than passing through the bottom one. _sheets draws
# n sheets of height 0.46*ys stepping up by 0.13*ys, putting that edge
# 0.46*ys/2 + 0.13*ys = 0.36*ys above the stack's nominal centre.
_DESIGN_STACK_DY = -0.36 * 1.5
# Decoy-variant cylinder (c/d): sized to leave headroom below ylim's top
# edge, so the top cap doesn't read as cut off once the figure-level tight
# crop removes any spare margin above it. Shared by both panels so they
# stay the same size as each other.
_DESIGN_CYL_H_DECOY = 3.0 * 0.90
# Panel c's decoy variant shares this wider xlim with panel d (which needs it
# for its fourth outcome stack) purely so the two panels render at the same
# px-per-data-unit scale — both occupy the same physical column width, so a
# narrower xlim in c alone would make its identically-sized-in-data-units
# shapes (cylinder, spectra, stacks) look larger than d's.
_DESIGN_XLIM_DECOY = 12.5
# Decoy-match stack's x position, one more _DESIGN_STACK_X step past the last
# slot — shared by both decoy-variant panels so "OFT"/"decoy match" land at
# the same x in c and d regardless of how many outcome stacks precede them
# (c has one fewer than d).
_DESIGN_DECOY_STACK_X = _DESIGN_STACK_X[2] + (_DESIGN_STACK_X[2] - _DESIGN_STACK_X[1])


def _design_helpers(ax: plt.Axes, ys: float = 1.0):
    """Return (_sp, _cyl, _sheets, _arr) closures over ax for design schematics.

    ys scales all shape heights and y-offsets (but not text font sizes).
    """
    def _sp(cx, cy, col, w=1.50, h=0.62, n=3):
        h = h * ys
        _pks = [(0.10, .30), (0.25, .82), (0.40, .55), (0.54, 1.0),
                (0.67, .42), (0.80, .63), (0.92, .28)]
        for i in range(n):
            x0 = cx - w/2 + i*0.10
            y0 = cy - h/2 + i*0.17*ys
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0, y0), w, h, boxstyle="square,pad=0",
                fc="white", ec=col, lw=0.9, alpha=0.88, zorder=3+i))
            for px, ph in _pks:
                ax.plot([x0+px*w, x0+px*w],
                        [y0+0.06*h, y0+0.06*h+ph*h*0.84],
                        color=col, lw=0.7, zorder=4+i)

    def _cyl(cx, cy, w, h, cols, lbls=None, bottom=True, top=True, heights=None,
             bottom_full=False):
        """*heights*: optional relative weights (same length as *cols*) for
        unequal segment sizes — e.g. a decoy segment drawn the same total
        size as the combined target segments above it. Defaults to equal
        division, same as every pre-existing call site.

        *bottom_full*: draw the base as a complete ellipse (like the joint
        and top caps) instead of the default near-side-only half ellipse —
        for a cylinder whose bottom is the true base of the whole stack
        (nothing below it to imply a hidden far half), e.g. the decoy segment
        in the decoy-variant design schematics.
        """
        h = h * ys
        n = len(cols)
        weights = list(heights) if heights is not None else [1.0] * n
        total_w = sum(weights)
        seg_h = [h * wgt / total_w for wgt in weights]
        # Cumulative segment boundaries from the cylinder's own bottom edge —
        # segments can differ in height (see *heights*), so each boundary
        # accumulates rather than stepping by a fixed amount.
        y_edges = [cy - h / 2]
        for sh_k in seg_h:
            y_edges.append(y_edges[-1] + sh_k)
        ery = 0.13 * ys
        lbls = lbls or [""] * n

        # Rectangles: upper segments at higher zorder so they sit in front.
        for k, (c, lbl) in enumerate(zip(cols, lbls)):
            y0, sh_k = y_edges[k], seg_h[k]
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx-w/2, y0), w, sh_k,
                boxstyle="square,pad=0", fc=c, ec="white", lw=0.5,
                alpha=0.72, zorder=3+k))
            if lbl:
                ax.text(cx, y0+sh_k/2, lbl, ha="center", va="center",
                        fontsize=_SCATTER_LABEL_FS, color="white", fontweight="bold",
                        zorder=5+k)

        # Bottom cap — the base of the cylinder stack. Default: half ellipse
        # (near side only), drawn as a PathPatch so it never overlaps the
        # rectangle above it. bottom_full: a complete Ellipse instead, same
        # as the joint/top caps.
        def _bottom_half_patch(zorder, ec):
            theta = np.linspace(np.pi, 2 * np.pi, 60)
            rx, ry = w / 2, ery
            by = y_edges[0]
            xs = cx + rx * np.cos(theta)
            ys_pts = by + ry * np.sin(theta)
            verts = np.column_stack([xs, ys_pts])
            verts = np.vstack([verts, verts[0]])
            codes = ([mpl.path.Path.MOVETO]
                     + [mpl.path.Path.LINETO] * (len(theta) - 1)
                     + [mpl.path.Path.CLOSEPOLY])
            ax.add_patch(mpl.patches.PathPatch(
                mpl.path.Path(verts, codes),
                fc=cols[0], ec=ec, lw=0.5, alpha=0.55, zorder=zorder))

        if bottom and bottom_full:
            # zorder above the bottom segment's own rectangle (3+0=3): a full
            # ellipse at the half-ellipse's old zorder=2 would have its top
            # half hidden behind that rectangle, drawing as the same clipped
            # near-side arc it was meant to replace. That top half then reads
            # darker than the bottom half regardless — it's alpha=0.55 over
            # the rectangle's own 0.72 fill, vs. the bottom half's 0.55 over
            # plain white — so a second half-ellipse patch stacks the bottom
            # half's alpha the same way, to match.
            ax.add_patch(mpatches.Ellipse(
                (cx, y_edges[0]), w, ery*2,
                fc=cols[0], ec="white", lw=0.5, alpha=0.55, zorder=3.5))
            _bottom_half_patch(zorder=3.6, ec="none")
        elif bottom:
            _bottom_half_patch(zorder=2, ec="white")

        # Joint ellipses between segments: use the UPPER segment's colour so the
        # upper cylinder appears to sit on top of the lower one.
        for k in range(1, n):
            y_joint = y_edges[k]
            ax.add_patch(mpatches.Ellipse(
                (cx, y_joint), w, ery*2,
                fc=cols[k], ec="white", lw=0.5, alpha=0.90, zorder=3+k+1))

        # Top ellipse — cap of the topmost cylinder.
        if top:
            ax.add_patch(mpatches.Ellipse(
                (cx, y_edges[-1]), w, ery*2,
                fc=cols[-1], ec="white", lw=0.5, alpha=0.90, zorder=3+n+1))

    def _sheets(cx, cy, fc, lbl="", lbl_col=None, n=3, w=1.0, h=0.46, lbl_size=5.2):
        h = h * ys
        for i in range(n):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx-w/2+i*0.09, cy-h/2+i*0.13*ys), w, h,
                boxstyle="round,pad=0.03",
                fc=fc, ec="white", lw=0.6,
                alpha=0.45+i*0.18, zorder=3+i))
        if lbl:
            top_y = cy + h/2 + (n-1)*0.13*ys
            ax.text(cx+(n-1)*0.045, top_y+0.07*ys, lbl,
                    ha="center", va="bottom", fontsize=lbl_size,
                    color=lbl_col or fc, fontweight="bold")

    def _arr(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                   lw=0.9, mutation_scale=9), zorder=6)

    def _plus(cx, cy):
        """A "+" between two outcome stacks, drawn at *cy*.

        Called with the arrow's y, not the stacks': the "+" reads as part of the
        same connective line as the arrow, so it sits on that line rather than
        on the (lower) centre of the stacks it separates.
        """
        ax.text(cx, cy, "+", ha="center", va="center",
                fontsize=_SCATTER_LABEL_FS, color="#666666", fontweight="bold",
                zorder=6)

    return _sp, _cyl, _sheets, _arr, _plus


def _draw_design1(ax: plt.Axes, *, panel_label: str | None = "c") -> None:
    """Single-entrapment design schematic (E. coli–Human) for fig_multi1.

    Shares c/d geometry with :func:`_draw_design2`: same ylim span, same
    cylinder, same outcome-stack positions and spacing. c and d sit in
    equal-height gridspec cells, so an unequal ylim made identical shapes render
    at different aspect ratios — the cylinders were both 103 px wide but 88 vs
    76 px tall, which reads as "different size".
    """
    ax.set_xlim(0, 10); ax.set_ylim(1.0, 7.0); ax.axis("off")
    if panel_label:
        _draw_panel_label(ax, panel_label, x=-0.05, y=1.01)

    _CE = "#50cf50"
    _CH = "#d62728"
    _sp, _cyl, _sheets, _arr, _plus = _design_helpers(ax, ys=1.5)

    CY = 4.8
    _sp(1.30, CY, _CE)
    ax.text(1.30, CY - 0.57*1.5, f"{SP_ECOLI}\nspectra",
            ha="center", va="top", fontsize=_SCATTER_LABEL_FS, color=_CE)
    _arr(2.30, 2.78, CY)
    ax.text(2.54, CY+0.20*1.5, "DB", ha="center", fontsize=_SCATTER_LABEL_FS, color="#666666")
    # w=1.95, not 1.50: "M. pneum." in panel d needs ~1.7 data units at the
    # uniform 12 pt this figure forces, and c matches d so the two designs
    # stay visually parallel.
    _cyl(_DESIGN_CYL_X, CY, w=1.95, h=1.50, cols=[_CE, _CH], lbls=[SP_ECOLI, SP_HUMAN])
    _arr(4.85, 5.15, CY)
    _SY = CY + _DESIGN_STACK_DY
    _sheets(_DESIGN_STACK_X[0], _SY, "#a5d6a7", lbl="possible",     lbl_col=_CE)
    _sheets(_DESIGN_STACK_X[1], _SY, "#ef9a9a", lbl="not\npresent", lbl_col=_CH)
    _plus(_DESIGN_PLUS_X[0], CY)


def _draw_design2(ax: plt.Axes, *, panel_label: str | None = "d") -> None:
    """Dual-entrapment design schematic (E. coli–Mycoplasma) for fig_multi1."""
    ax.set_xlim(0, 10); ax.set_ylim(1.0, 7.0); ax.axis("off")
    if panel_label:
        _draw_panel_label(ax, panel_label, x=-0.05, y=1.01)

    # Both species are genuinely present, so both stay in the green family
    # (green = possible); only the mis-matched class gets a foreign hue.
    # Sea green rather than a lighter/darker #50cf50: it separates on hue as
    # well as lightness, which survives the thin spectra outlines and the
    # small bold labels. Orange is avoided here — it already means
    # "entrapment" (C_ENTRAP) elsewhere in this figure.
    # Mis-matched reuses panel c's "not present" red so both design
    # schematics mark the impossible class with the same colour.
    _CE = "#50cf50"
    _CM = "#0f8b6e"
    _CX = "#d62728"
    _sp, _cyl, _sheets, _arr, _plus = _design_helpers(ax, ys=1.5)

    CY = 4.8
    _sp(1.30, CY+0.62*1.5, _CE, w=1.40, h=0.56, n=3)
    ax.text(1.30, CY+0.26*1.5, SP_ECOLI,
            ha="center", va="top", fontsize=_SCATTER_LABEL_FS, color=_CE, fontweight="bold")
    _sp(1.30, CY-0.72*1.5, _CM, w=1.40, h=0.56, n=3)
    ax.text(1.30, CY-1.07*1.5, SP_MYCPN,
            ha="center", va="top", fontsize=_SCATTER_LABEL_FS, color=_CM, fontweight="bold")
    _arr(2.30, 2.78, CY)
    ax.text(2.54, CY+0.20*1.5, "DB", ha="center", fontsize=_SCATTER_LABEL_FS, color="#666666")
    _cyl(_DESIGN_CYL_X, CY, w=1.95, h=1.50, cols=[_CE, _CM], lbls=[SP_ECOLI, SP_MYCPN])
    _arr(4.85, 5.15, CY)
    # c and d share a ylim, so the sheets need no scale correction — both
    # panels take the same defaults.
    _SY = CY + _DESIGN_STACK_DY
    _sheets(_DESIGN_STACK_X[0], _SY, "#a5d6a7", lbl=f"{SP_ECOLI}\npos.", lbl_col=_CE)
    _sheets(_DESIGN_STACK_X[1], _SY, "#6fc9b2", lbl=f"{SP_MYCPN}\npos.", lbl_col=_CM)
    _sheets(_DESIGN_STACK_X[2], _SY, "#ef9a9a", lbl="mis-\nmatched",     lbl_col=_CX)
    for _px in _DESIGN_PLUS_X:
        _plus(_px, CY)


def _draw_design1_decoy(ax: plt.Axes, *, panel_label: str | None = "c") -> None:
    """Single-entrapment design schematic (E. coli-Human), extended with the
    decoy half of the search database and a decoy-match outcome stack.

    Same layout as :func:`_draw_design1`, plus: a C_DECOY segment at the
    bottom of the DB cylinder, sized equal to the two target segments
    combined (the decoy database is a reversed 1:1 copy of the target
    database everywhere else in this project — see
    _eh_impossible_entrapment_only's docstring); and a third outcome stack,
    in the slot _draw_design1 leaves unused, for matches landing on that
    decoy segment, "+"-connected to the target-outcome stacks like the rest.
    """
    ax.set_xlim(0, _DESIGN_XLIM_DECOY); ax.set_ylim(1.0, 7.0); ax.axis("off")
    if panel_label:
        _draw_panel_label(ax, panel_label, x=-0.05, y=1.01)

    _CE = "#50cf50"
    _CH = "#d62728"
    _sp, _cyl, _sheets, _arr, _plus = _design_helpers(ax, ys=1.5)

    CY = 4.8
    _sp(1.30, CY, _CE)
    ax.text(1.30, CY - 0.57*1.5, f"{SP_ECOLI}\nspectra",
            ha="center", va="top", fontsize=_SCATTER_LABEL_FS, color=_CE)
    _arr(2.30, 2.78, CY)
    ax.text(2.54, CY+0.20*1.5, "DB", ha="center", fontsize=_SCATTER_LABEL_FS, color="#666666")
    _cyl(_DESIGN_CYL_X, CY, w=1.95, h=_DESIGN_CYL_H_DECOY, cols=[C_DECOY, _CE, _CH],
         lbls=["Decoy", SP_ECOLI, SP_HUMAN], heights=[2, 1, 1], bottom_full=True)
    _arr(4.85, 5.15, CY)
    _SY = CY + _DESIGN_STACK_DY
    # "OFT" and "decoy match" sit at panel d's positions (_DESIGN_STACK_X[2]
    # and _DESIGN_DECOY_STACK_X) rather than c's own [1]/[2] slots, so the two
    # stacks they share with panel d line up across both panels — c just has
    # one fewer stack before them, leaving a gap where d's "M. pneum. possible" is.
    _sheets(_DESIGN_STACK_X[0], _SY, "#a5d6a7", lbl=f"{SP_ECOLI}\npossible", lbl_col=_CE)
    _sheets(_DESIGN_STACK_X[2], _SY, "#ef9a9a", lbl="OFT", lbl_col=_CH)
    _plus((_DESIGN_STACK_X[0] + _DESIGN_STACK_X[2]) / 2 + 0.045, CY)
    _plus((_DESIGN_STACK_X[2] + _DESIGN_DECOY_STACK_X) / 2 + 0.045, CY)
    _sheets(_DESIGN_DECOY_STACK_X, _SY, C_DECOY, lbl="decoy", lbl_col=C_DECOY)


def _draw_design2_decoy(ax: plt.Axes, *, panel_label: str | None = "d") -> None:
    """Dual-entrapment design schematic (E. coli-Mycoplasma), extended with
    the decoy half of the search database and a decoy-match outcome stack.

    Same extension as :func:`_draw_design1_decoy`; needs a fourth outcome-
    stack slot (the three-stack version already uses all of
    _DESIGN_STACK_X), same spacing as the existing three, so xlim widens to
    fit it.
    """
    ax.set_xlim(0, _DESIGN_XLIM_DECOY); ax.set_ylim(1.0, 7.0); ax.axis("off")
    if panel_label:
        _draw_panel_label(ax, panel_label, x=-0.05, y=1.01)

    _CE = "#50cf50"
    _CM = "#0f8b6e"
    _CX = "#d62728"
    _sp, _cyl, _sheets, _arr, _plus = _design_helpers(ax, ys=1.5)

    CY = 4.8
    _sp(1.30, CY+0.62*1.5, _CE, w=1.40, h=0.56, n=3)
    ax.text(1.30, CY+0.26*1.5, SP_ECOLI,
            ha="center", va="top", fontsize=_SCATTER_LABEL_FS, color=_CE, fontweight="bold")
    _sp(1.30, CY-0.72*1.5, _CM, w=1.40, h=0.56, n=3)
    ax.text(1.30, CY-1.07*1.5, SP_MYCPN,
            ha="center", va="top", fontsize=_SCATTER_LABEL_FS, color=_CM, fontweight="bold")
    _arr(2.30, 2.78, CY)
    ax.text(2.54, CY+0.20*1.5, "DB", ha="center", fontsize=_SCATTER_LABEL_FS, color="#666666")
    _cyl(_DESIGN_CYL_X, CY, w=1.95, h=_DESIGN_CYL_H_DECOY, cols=[C_DECOY, _CE, _CM],
         lbls=["Decoy", SP_ECOLI, SP_MYCPN], heights=[2, 1, 1], bottom_full=True)
    _arr(4.85, 5.15, CY)
    _SY = CY + _DESIGN_STACK_DY
    _sheets(_DESIGN_STACK_X[0], _SY, "#a5d6a7", lbl=f"{SP_ECOLI}\npossible", lbl_col=_CE)
    _sheets(_DESIGN_STACK_X[1], _SY, "#6fc9b2", lbl=f"{SP_MYCPN}\npossible", lbl_col=_CM)
    _sheets(_DESIGN_STACK_X[2], _SY, "#ef9a9a", lbl="OFT",                   lbl_col=_CX)
    for _px in _DESIGN_PLUS_X:
        _plus(_px, CY)
    # Same spacing as the existing three stacks (all "+"-connected, so a
    # wider gap before this one would read as a break in the chain that isn't
    # there) — _DESIGN_DECOY_STACK_X, shared with panel c so both panels'
    # "decoy" stack lands at the same x.
    _plus((_DESIGN_STACK_X[2] + _DESIGN_DECOY_STACK_X) / 2 + 0.045, CY)
    _sheets(_DESIGN_DECOY_STACK_X, _SY, C_DECOY, lbl="decoy", lbl_col=C_DECOY)


def plot_multi_fig1_design_with_decoy() -> None:
    """Exploratory variant of fig_multi1 panels c/d, side by side, with the
    decoy half of the search database and a decoy-match outcome stack added.

    Not wired into plot_multi_fig1 or main()/CLI yet.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(14, 5))
    _draw_design1_decoy(ax_c, panel_label="c")
    _draw_design2_decoy(ax_d, panel_label="d")
    fig.tight_layout()
    _save(fig, "fig_multi1_design_with_decoy")


def plot_multi_fig1(eh: pd.DataFrame, decoy_design: bool = False) -> None:
    """Figure 1 (multi-panel): Do decoys represent false target matches?

    Layout (4 rows):
      Row 0:  a (left)  |  c (right)
      Row 1:  b (left)  |  d (right)
      Row 2:  e0 | e1 | e2  (full width)
      Row 3:  f (left)  |  g (right)

    a – Conceptual schematic (bar diagram)
    b – Concordant / discordant score-distribution mini-graphs
    c – Single-entrapment design (E. coli–Human)
    d – Dual-entrapment design (E. coli–Mycoplasma)
    e – Three representative score distributions
    f – E. coli–Human CSM impossible breakdown
    g – Protein-pair overlap (PPI)

    ``decoy_design=True`` swaps panels c/d for :func:`_draw_design1_decoy`/
    :func:`_draw_design2_decoy` (decoy segment + decoy-match outcome stack)
    and widens their column accordingly, saving to a separate
    ``_with_decoy``-suffixed file rather than overwriting the standard one —
    exploratory, like the standalone :func:`plot_multi_fig1_design_with_decoy`.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)

    # Main figures use the entrapment criterion only; the co-fractionation
    # criterion is a supplementary analysis (see _eh_impossible_entrapment_only).
    eh = _eh_impossible_entrapment_only(eh)

    df    = _load_eh_annotated_csm(entrapment_only=True)
    stats = _compute_ks_and_ratio(df)

    # ── layout ──
    # Row 0: a (left)  | c (right)
    # Row 1: b (left)  | d (right)
    # Row 2: e0 | e1 | e2  (full width)
    # Row 3: f (left)  | g (right)
    fig = plt.figure(figsize=(14, 18))
    gs = mgridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[4.3, 4.5, 5], hspace=0.10)

    # Row 0: 2×2 block — (a, b) on left, (c, d) on right — tight inner hspace
    # decoy_design's panel d needs xlim 0-12.5 instead of 0-10 (a fourth
    # outcome stack) — scale the right column's share up by that same 1.25x
    # so its design elements render at the same size as the standard layout,
    # rather than shrinking to fit the old column width.
    c_ratio = 1.07 * 1.25 if decoy_design else 1.07
    gs_top = mgridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0],
                                               height_ratios=[1, 1],
                                               # The design schematics (c, d) carry
                                               # three 12 pt binomials side by
                                               # side and need ~5.3 in. The extra
                                               # width comes out of the gutter,
                                               # not out of panels a/b, which keep
                                               # the ~4.95 in they had under the
                                               # old 1.1/1.0, wspace=0.30 split.
                                               width_ratios=[1.0, c_ratio],
                                               hspace=0.0, wspace=0.117)
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[1, 0])
    ax_c = fig.add_subplot(gs_top[0, 1])
    ax_d = fig.add_subplot(gs_top[1, 1])

    # Row 1: four score-distribution panels
    gs2 = mgridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.38)
    ax_e = [fig.add_subplot(gs2[i]) for i in range(4)]

    # Row 2: impossible breakdown | PPI overlap
    gs3 = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.35)
    ax_f = fig.add_subplot(gs3[0])
    ax_g = fig.add_subplot(gs3[1])

    # ── panels a, b, c, d ─────────────────────────────────────────────────────
    _draw_assumption_schematic(ax_a, panel_label="a")
    _draw_concordant_discordant(ax_b, panel_label="b")
    if decoy_design:
        _draw_design1_decoy(ax_c, panel_label="c")
        _draw_design2_decoy(ax_d, panel_label="d")
    else:
        _draw_design1(ax_c, panel_label="c")
        _draw_design2(ax_d, panel_label="d")

    # ── panel e: four score distribution panels ───────────────────────────────
    # (eng, display_name, body_label, chart_title)
    examples_e = [
        ("kojak",    "Kojak",        "reliable",    None),
        ("xisearch", "xiSEARCH",     "reliable",    None),
        ("plink3",   "pLink3 (FDR)", "unreliable",  None),
        ("xlinkx25", "XlinkX PD2.5", "unreliable",  None),
    ]
    for ax, (eng, disp, body_lbl, chart_title) in zip(ax_e, examples_e):
        ks_d = float(stats.loc[eng, "ks_d"]) if eng in stats.index else float("nan")
        _draw_norm_score_hist(ax, df[df["search_engine"] == eng], body_lbl, ks_d,
                              xlabel=f"{disp} Score", chart_title=chart_title,
                              fp_label="OFT")
    _draw_panel_label(ax_e[0], "e", x=-0.18, y=1.08)
    ax_e[0].set_title(ax_e[0].get_title(), fontsize=_SCATTER_LABEL_FS, fontweight="bold")

    if decoy_design:
        # Shift panels c/d right as a unit so the decoy-match stack in panel
        # d right-aligns with panel e's rightmost (XlinkX PD2.5) chart frame.
        # c/d's own column width is driven by their own content, not panel
        # e's, so without this the decoy stack lands wherever that column
        # happens to end.
        _stack_step = _DESIGN_STACK_X[2] - _DESIGN_STACK_X[1]
        decoy_cx = _DESIGN_STACK_X[2] + _stack_step
        # _sheets' own geometry (w=1.0, n=3, each layer offset +i*0.09): the
        # rightmost of the 3 stacked sheets has its right edge at
        # cx - w/2 + (n-1)*0.09 + w = cx + w/2 + (n-1)*0.09.
        decoy_right_data = decoy_cx + 1.0 / 2 + 2 * 0.09
        decoy_right_disp = ax_d.transData.transform((decoy_right_data, 0))
        decoy_right_fig = fig.transFigure.inverted().transform(decoy_right_disp)[0]
        dx = ax_e[3].get_position().x1 - decoy_right_fig
        for ax in (ax_c, ax_d):
            pos = ax.get_position()
            ax.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])

    # ── panel f: E. coli–Human CSM impossible breakdown ──────────────────────
    _draw_eh_csm_impossible_panel(ax_f, eh, panel_label="f")

    # ── panel g: protein-pair overlap PPI ────────────────────────────────────
    ppi_summaries, ppi_labels = _load_overlap_ppi_data(entrapment_only=True)
    _draw_overlap_ppi_bars(ax_g, ppi_summaries, ppi_labels, panel_label="g",
                           fdr_label_last=True,
                           impossible_label="Observable False Target (OFT)")

    # Save individual panels before closing the figure
    _save_panel(fig, ax_a,  "fig_multi1_panel_a")
    _save_panel(fig, ax_b,  "fig_multi1_panel_b")
    _save_panel(fig, ax_c,  "fig_multi1_panel_c")
    _save_panel(fig, ax_d,  "fig_multi1_panel_d")
    _save_panel(fig, ax_e,  "fig_multi1_panel_e")
    _save_panel(fig, ax_f,  "fig_multi1_panel_f")
    _save_panel(fig, ax_g,  "fig_multi1_panel_g")

    _save(fig, "fig_multi1_decoy_question" if decoy_design
          else "fig_multi1_decoy_question_no_decoy_design")


# ── Figure 2 ──────────────────────────────────────────────────────────────────

DEDICATED_PPI_FDR = {"kojak", "xisearch"}  # iProphet / xiFDR applied at PPI level

_MARKER_EH = "o"
_MARKER_EM = "^"
_DS_COLORS = {"ecoli_human": "#1f77b4", "ecoli_mycpn": "#f17000"}
_SCATTER_LABEL_FS: float = 12   # font size for scatter annotations; also set in rcParams before adjust_text

# Species spelled out in full, unlike the abbreviated DS_EH/DS_EM used as the
# short form everywhere else — for _plot_multi_fig2_panel_a_single's in-panel
# title only, keyed by the same DS_EH/DS_EM value so a == DS_EH check still
# works on the caller-supplied dataset_label.
_DS_TITLE_FULL = {
    DS_EH: f"{SP_ECOLI}–$\\it{{H.\\,sapiens}}$",
    DS_EM: f"{SP_ECOLI}–$\\it{{M.\\,pneumoniae}}$",
}


def _plot_multi_fig2_panel_a(
    ax: plt.Axes,
    eh_csm: pd.DataFrame,
    em_csm: pd.DataFrame,
    scale: str = "log2",
) -> None:
    """Draw panel 2a on the provided axis."""
    ax.text(-0.12, 1.03, "a", fontsize=_SCATTER_LABEL_FS, fontweight="bold",
            transform=ax.transAxes, va="top")

    palette = _fig2_palette()
    # Panel-a-only override: Scout (FL) reads too similarly to its neighbours
    # here. Local to this panel rather than _fig2_palette() itself, which the
    # standalone fig2_impossible_vs_decoy* figures also share.
    palette["scout_fl"] = "#b55a32e5"
    max_x = 0.0
    max_y = 0.0
    no_x: list[tuple] = []
    specs2a: list[tuple] = []   # (x, y, label, color) — labelled once, after the axis is final
    pts2a: list[tuple] = []     # every marker position, so labels dodge all of them
    # Index into specs2a of Scout (FL)'s two points, keyed by marker, so its
    # labels can be nudged left afterwards without disturbing how the rest of
    # this tight cluster (ProteinProspector/Kojak/xiSEARCH) got placed.
    scout_fl_idx: dict[str, int] = {}
    for df_csm, marker in [
        (eh_csm, _MARKER_EH),
        (em_csm, _MARKER_EM),
    ]:
        for _, row in df_csm.iterrows():
            eng = row["search_engine"]
            if eng not in ENGINE_ORDER:
                continue
            x = row["decoy_fdr_pct"]
            y = row["tt_impossible_fpr_pct"]
            if pd.isna(y):
                continue
            if pd.isna(x) or x == 0:
                no_x.append((float(y), marker, eng))
                continue
            col = palette.get(eng, "#333333")
            ax.scatter(x, y, s=_FIG2_MARKER_SIZE, marker=marker, color=col,
                       alpha=0.80, zorder=4, edgecolors="white", linewidths=0.4)
            max_x = max(max_x, float(x))
            max_y = max(max_y, float(y))
            pts2a.append((float(x), float(y)))
            if eng == "scout_fl":
                scout_fl_idx[marker] = len(specs2a)
            specs2a.append((float(x), float(y), ENGINE_LABELS.get(eng, eng), col))

    lim_x = max_x * 1.08
    lim_y = max_y * 1.08
    diag_lim = min(lim_x, lim_y)
    sep_x = lim_x * 1.08
    mango_x = lim_x * 1.18
    x_total = lim_x * 1.30 if no_x else lim_x

    if scale == "log2":
        x_floor = 1.0
        y_floor = 0.5
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlim(x_floor, x_total)
        ax.set_ylim(y_floor, lim_y)
        log2_ticks = [t for t in (0.5, 1, 2, 4, 8, 16, 32, 64)
                      if t <= max(lim_x, lim_y)]
        ax.set_xticks([t for t in log2_ticks if x_floor <= t <= lim_x])
        ax.set_yticks([t for t in log2_ticks if y_floor <= t <= lim_y])
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        x_lo = x_floor
        y_lo = y_floor
    else:  # sqrt
        sqrt_floor = 0.5
        ax.set_xscale("function", functions=(np.sqrt, np.square))
        ax.set_yscale("function", functions=(np.sqrt, np.square))
        ax.set_xlim(sqrt_floor, x_total)
        ax.set_ylim(sqrt_floor, lim_y)
        ax.set_xticks([t for t in (0, 1, 4, 9, 16, 25, 36, 49) if t >= sqrt_floor and t <= lim_x])
        ax.set_yticks([t for t in (0, 1, 4, 9, 16, 25, 36, 49) if t >= sqrt_floor and t <= lim_y])
        x_lo = sqrt_floor
        y_lo = sqrt_floor

    # ax.fill_between([x_lo, diag_lim], [x_lo, diag_lim], [lim_y, lim_y],
    #                 color="#ffcccc", alpha=0.30, zorder=0)
    # ax.fill_between([x_lo, diag_lim], [y_lo, y_lo], [x_lo, diag_lim],
    #                 color="#ccffcc", alpha=0.30, zorder=0)
    # if lim_x > diag_lim:
    #     ax.fill_between([diag_lim, lim_x], [y_lo, y_lo], [lim_y, lim_y],
    #                     color="#ccffcc", alpha=0.30, zorder=0)

    ax.plot([x_lo, lim_x], [REF_FDR, REF_FDR], color="red", lw=0.8, ls="--", alpha=0.6)
    ax.plot([REF_FDR, REF_FDR], [y_lo, lim_y], color="black", lw=0.8, ls="--", alpha=0.6)
    ax.text(x_lo, REF_FDR, "Limit", ha="left", va="bottom", fontsize=_SCATTER_LABEL_FS,
            color="red", alpha=0.9)
    ax.text(REF_FDR, 0.97, "target FDR", rotation=90, ha="right", va="top",
            transform=ax.get_xaxis_transform(), fontsize=_SCATTER_LABEL_FS,
            color="black", alpha=0.9)

    if not any(np.isclose(t, REF_FDR) for t in ax.get_yticks()):
        ax.text(-0.01, REF_FDR, "2%", ha="right", va="center", fontsize=_SCATTER_LABEL_FS,
                color="grey", alpha=0.9, transform=ax.get_yaxis_transform(),
                clip_on=False)
    if not any(np.isclose(t, REF_FDR) for t in ax.get_xticks()):
        ax.text(REF_FDR, -0.03, "2%", ha="center", va="top", fontsize=_SCATTER_LABEL_FS,
                color="grey", alpha=0.9, transform=ax.get_xaxis_transform(),
                clip_on=False)

    if no_x:
        _draw_axis_break(ax, sep_x)
        for y_m, marker, eng in no_x:
            col_m = palette.get(eng, "grey")
            ax.scatter(mango_x, y_m, s=_FIG2_MARKER_SIZE, marker=marker, color=col_m,
                       alpha=0.8, zorder=4, edgecolors="white", linewidths=0.4)
            pts2a.append((mango_x, y_m))
            specs2a.append((mango_x, y_m, ENGINE_LABELS.get(eng, eng), col_m))
        ax.text(mango_x, -0.03, "undef.", ha="center", va="top", fontsize=_SCATTER_LABEL_FS,
                transform=ax.get_xaxis_transform(), clip_on=False)

    mpl.rcParams["font.size"] = _SCATTER_LABEL_FS

    texts2a: list = []
    if specs2a:
        lx, ly, llabels, lcolors = zip(*specs2a)
        apx, apy = zip(*pts2a)
        # Same deterministic placement as fig2_impossible_vs_decoy*: offer
        # each label a preferred cardinal/diagonal slot right next to its own
        # point before falling back to adjustText, instead of adjustText's
        # free-floating spread (which drifted labels away from their points
        # in the crowded low-FDR corner).
        texts2a, _ = _add_labels_by_density(
            ax, lx, ly, llabels,
            offset=5, fontsize=_SCATTER_LABEL_FS, colors=lcolors, zorder=5,
            avoid_xy=(apx, apy),
            at_force_pull=(0.05, 0.05),
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

    # Scout (FL)'s pair sits in a corner so tight (its own EH point, after the
    # entrapment-only adjustment, lands within a couple of pixels of three
    # other engines' EM points) that the automatic pass above lands it
    # wherever has room rather than where it reads best. Nudge each label
    # left of its marker — circle slightly up, triangle slightly down, per
    # the reference layout — trying progressively smaller nudges and falling
    # back to a plain "left" if even that collides, before finally giving up
    # and leaving the automatic position, which is at least guaranteed not to
    # overlap anything.
    if scout_fl_idx and texts2a:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        other_bboxes = [t.get_window_extent(renderer=renderer)
                        for i, t in enumerate(texts2a) if i not in scout_fl_idx.values()]
        pt_to_px = ax.figure.dpi / 72.0
        inv = ax.transData.inverted()
        for marker, candidates in (
            (_MARKER_EH, [((-7, 4), "bottom"), ((-7, 2), "bottom"), ((-7, 0), "center")]),
            (_MARKER_EM, [((-7, -4), "top"), ((-7, -2), "top"), ((-7, 0), "center")]),
        ):
            idx = scout_fl_idx.get(marker)
            if idx is None:
                continue
            t = texts2a[idx]
            is_annotation = isinstance(t, mpl.text.Annotation)
            # The deterministic pass above either places this via ax.annotate
            # (an Annotation, offset in points from its xy) or, if every
            # preferred slot collided, hands it to adjustText as a plain Text
            # already moved to some absolute data position. Both need their
            # own marker's pixel location as the origin for "left"/"up"/"down".
            mx, my, _, _ = specs2a[idx]
            marker_px = ax.transData.transform((mx, my))
            original = t.xyann if is_annotation else t.get_position()
            original_ha, original_va = t.get_ha(), t.get_va()
            placed = False
            for offset_xy, va in candidates:
                if is_annotation:
                    t.xyann = offset_xy
                else:
                    target_px = marker_px + np.array(offset_xy) * pt_to_px
                    t.set_position(tuple(inv.transform(target_px)))
                t.set_ha("right")
                t.set_va(va)
                new_bbox = t.get_window_extent(renderer=renderer)
                if not any(new_bbox.overlaps(b) for b in other_bboxes):
                    placed = True
                    break
            if not placed:
                if is_annotation:
                    t.xyann = original
                else:
                    t.set_position(original)
                t.set_ha(original_ha)
                t.set_va(original_va)

    ax.set_xlabel("Reported decoy FDR (%)", fontsize=_SCATTER_LABEL_FS)
    ax.set_ylabel("Observable false-target CSM rate (%)", fontsize=_SCATTER_LABEL_FS)
    # ax.set_title("CSM-level: reported vs observed error", fontsize=_SCATTER_LABEL_FS, fontweight="bold")
    # ax.text(1.1, 33, "Observable error\nexceeds reported FDR",
    #         ha="left", va="top", fontsize=_SCATTER_LABEL_FS, color="#c62828",
    #         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    # ax.text(lim_x * 0.8, y_lo + 0.05, "No contradiction\ndetected",
    #         ha="right", va="bottom", fontsize=_SCATTER_LABEL_FS, color="#28c630",
    #         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    # ax.text(lim_x * 0.1, 0.12, "No contradiction\ndetected", ha="left", va="bottom",
    #         fontsize=_SCATTER_LABEL_FS, color="#444444", transform=ax.transAxes)
    handles2a = [
        plt.scatter([], [], marker=_MARKER_EH, color="#333333", s=50, label=DS_EH),
        plt.scatter([], [], marker=_MARKER_EM, color="#333333", s=50, label=DS_EM),
    ]
    ax.legend(handles=handles2a, fontsize=_SCATTER_LABEL_FS, frameon=True,
              loc="upper right",
              bbox_to_anchor=(lim_x, 32),
              bbox_transform=ax.transData)
    sns.despine(ax=ax)


def _fig2_scatter_bounds(df_csm: pd.DataFrame) -> tuple[float, float]:
    """Max (decoy_fdr_pct, tt_impossible_fpr_pct) among ENGINE_ORDER rows.

    Same point filter :func:`_plot_multi_fig2_panel_a` (and its split
    counterpart below) apply per marker — a defined y and, for the x maximum,
    a defined and nonzero x. Used to compute one shared scale across the two
    split panels before either is drawn.
    """
    max_x = 0.0
    max_y = 0.0
    for _, row in df_csm.iterrows():
        if row["search_engine"] not in ENGINE_ORDER:
            continue
        y = row["tt_impossible_fpr_pct"]
        if pd.isna(y):
            continue
        max_y = max(max_y, float(y))
        x = row["decoy_fdr_pct"]
        if pd.isna(x) or x == 0:
            continue
        max_x = max(max_x, float(x))
    return max_x, max_y


def _plot_multi_fig2_panel_a_single(
    ax: plt.Axes,
    df_csm: pd.DataFrame,
    marker: str,
    dataset_label: str,
    panel_label: str,
    lim_x: float,
    lim_y: float,
    scale: str = "log2",
) -> None:
    """One dataset's CSM decoy-FDR-vs-observable-FPR scatter — the split
    variant of panel 2a, which overlays both datasets on one axes instead.

    *lim_x*/*lim_y* are shared across both split panels (the caller computes
    them from both datasets' own maxima via :func:`_fig2_scatter_bounds`) so
    the two panels stay on the same scale and are directly comparable. Unlike
    :func:`_plot_multi_fig2_panel_a`, there is no cross-dataset Scout (FL)
    label nudge: that existed only to separate the E. coli-Human and
    E. coli-Mycoplasma points when they shared one axes, which cannot happen
    once each dataset has its own panel.
    """
    _draw_panel_label(ax, panel_label, x=-0.12, y=1.03)

    palette = _fig2_palette()
    palette["scout_fl"] = "#b55a32e5"
    no_x: list[tuple] = []
    specs: list[tuple] = []
    pts: list[tuple] = []
    # XlinkX PD2.5 (E. coli-Human only) sits just right of the target-FDR
    # line near the panel's top, i.e. right where "target FDR" is anchored —
    # tracked so its label can be nudged clear afterwards, the same way
    # Scout (FL) is nudged in _plot_multi_fig2_panel_a.
    xlinkx25_idx: int | None = None

    for _, row in df_csm.iterrows():
        eng = row["search_engine"]
        if eng not in ENGINE_ORDER:
            continue
        x = row["decoy_fdr_pct"]
        y = row["tt_impossible_fpr_pct"]
        if pd.isna(y):
            continue
        if pd.isna(x) or x == 0:
            no_x.append((float(y), eng))
            continue
        col = palette.get(eng, "#333333")
        ax.scatter(x, y, s=_FIG2_MARKER_SIZE, marker=marker, color=col,
                   alpha=0.80, zorder=4, edgecolors="white", linewidths=0.4)
        pts.append((float(x), float(y)))
        if eng == "xlinkx25":
            xlinkx25_idx = len(specs)
        specs.append((float(x), float(y), ENGINE_LABELS.get(eng, eng), col))

    sep_x = lim_x * 1.08
    mango_x = lim_x * 1.18
    x_total = lim_x * 1.30 if no_x else lim_x

    if scale == "log2":
        x_floor = 1.0
        y_floor = 0.5
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlim(x_floor, x_total)
        ax.set_ylim(y_floor, lim_y)
        log2_ticks = [t for t in (0.5, 1, 2, 4, 8, 16, 32, 64)
                      if t <= max(lim_x, lim_y)]
        ax.set_xticks([t for t in log2_ticks if x_floor <= t <= lim_x])
        ax.set_yticks([t for t in log2_ticks if y_floor <= t <= lim_y])
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        x_lo = x_floor
        y_lo = y_floor
    else:  # sqrt
        sqrt_floor = 0.5
        ax.set_xscale("function", functions=(np.sqrt, np.square))
        ax.set_yscale("function", functions=(np.sqrt, np.square))
        ax.set_xlim(sqrt_floor, x_total)
        ax.set_ylim(sqrt_floor, lim_y)
        ax.set_xticks([t for t in (0, 1, 4, 9, 16, 25, 36, 49) if sqrt_floor <= t <= lim_x])
        ax.set_yticks([t for t in (0, 1, 4, 9, 16, 25, 36, 49) if sqrt_floor <= t <= lim_y])
        x_lo = sqrt_floor
        y_lo = sqrt_floor

    ax.plot([x_lo, lim_x], [REF_FDR, REF_FDR], color="red", lw=0.8, ls="--", alpha=0.6)
    ax.plot([REF_FDR, REF_FDR], [y_lo, lim_y], color="black", lw=0.8, ls="--", alpha=0.6)
    # Reference-line labels, both on the left. Kept as Text objects (not just
    # drawn) so their rendered extents can be fed to _add_labels_by_density
    # below as obstacles, instead of relying on a fixed side to dodge
    # whichever engine label happens to land nearby.
    # "target FDR" is a short two-line label hugging the top of the line
    # rather than a tall string rotated along its whole length: the rotated
    # form reached down into the pLink2/MeroX/pLink3 (FDR) cluster, and even
    # as a registered obstacle that only pushes the engine label placed there
    # through adjustText's soft force, not a hard constraint, so it did not
    # reliably clear.
    # Nudged a couple points right of the y-axis spine (in display space, so
    # it holds regardless of the log-scale data range) rather than sitting
    # exactly at x_lo, where ha="left" text would visually touch the spine.
    left_edge_trans = offset_copy(ax.transData, fig=ax.figure, x=2, y=0, units="points")
    ref_texts = [
        ax.text(x_lo, REF_FDR, "limit", ha="left", va="bottom", fontsize=_SCATTER_LABEL_FS,
                color="red", alpha=0.9, transform=left_edge_trans),
        # Right-anchored at the line (text sits to its left) rather than
        # centred on it: XlinkX PD2.5's own label routes rightward from its
        # point, which sits close to this line near the top of the panel, so
        # centring let the two meet in a wide panel even though they cleared
        # in a narrower one.
        ax.text(REF_FDR, 0.99, "target\nFDR", ha="right", va="top", linespacing=0.9,
                transform=ax.get_xaxis_transform(), fontsize=_SCATTER_LABEL_FS,
                color="black", alpha=0.9),
        # In-panel title, top right, replacing the axes title below the
        # figure-level E. coli-H. sap./M. pneum. header used elsewhere — here
        # spelled out in full since it is the only species label in the panel.
        ax.text(0.97, 0.97, _DS_TITLE_FULL.get(dataset_label, dataset_label),
                ha="right", va="top", transform=ax.transAxes,
                fontsize=_SCATTER_LABEL_FS, fontweight="bold"),
    ]

    # E. coli-Human only: the declared-2%-FDR "limit" line above is a ceiling
    # every engine is claiming to meet; this is the rate a well-calibrated
    # engine should actually land at, given the human-entrapment database is
    # built 1:1 with the decoy database so "observed" and "reported" should
    # agree at half the declared limit.
    if dataset_label == DS_EH:
        expectation = 1.0
        ax.plot([x_lo, lim_x], [expectation, expectation], color="darkgreen",
                lw=0.8, ls="--", alpha=0.7)
        ref_texts.append(
            ax.text(x_lo, expectation, "expectation", ha="left", va="bottom",
                    fontsize=_SCATTER_LABEL_FS, color="darkgreen", alpha=0.9,
                    transform=left_edge_trans))

    if not any(np.isclose(t, REF_FDR) for t in ax.get_yticks()):
        ax.text(-0.01, REF_FDR, "2%", ha="right", va="center", fontsize=_SCATTER_LABEL_FS,
                color="grey", alpha=0.9, transform=ax.get_yaxis_transform(),
                clip_on=False)
    if not any(np.isclose(t, REF_FDR) for t in ax.get_xticks()):
        ax.text(REF_FDR, -0.03, "2%", ha="center", va="top", fontsize=_SCATTER_LABEL_FS,
                color="grey", alpha=0.9, transform=ax.get_xaxis_transform(),
                clip_on=False)

    if no_x:
        _draw_axis_break(ax, sep_x)
        for y_m, eng in no_x:
            col_m = palette.get(eng, "grey")
            ax.scatter(mango_x, y_m, s=_FIG2_MARKER_SIZE, marker=marker, color=col_m,
                       alpha=0.8, zorder=4, edgecolors="white", linewidths=0.4)
            pts.append((mango_x, y_m))
            specs.append((mango_x, y_m, ENGINE_LABELS.get(eng, eng), col_m))
        ax.text(mango_x, -0.03, "undef.", ha="center", va="top", fontsize=_SCATTER_LABEL_FS,
                transform=ax.get_xaxis_transform(), clip_on=False)

    mpl.rcParams["font.size"] = _SCATTER_LABEL_FS

    if specs:
        lx, ly, llabels, lcolors = zip(*specs)
        apx, apy = zip(*pts)
        # Limit / target FDR / expectation are drawn above, before this call,
        # specifically so their rendered extents are available here to treat
        # as fixed obstacles alongside the scatter points.
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        # Expanded well beyond _add_labels_by_density's own 1.02/1.05 margin
        # between placed-label obstacles: a borderline-clearing candidate
        # (e.g. XlinkX PD2.5's own label, whose point sits close to the
        # target-FDR line) still reads as touching at normal viewing size, so
        # give these reference labels a wide keep-out margin instead of a
        # tight one.
        ref_bboxes = [t.get_window_extent(renderer=renderer).expanded(1.5, 2.0)
                     for t in ref_texts]
        texts, _ = _add_labels_by_density(
            ax, lx, ly, llabels,
            offset=5, fontsize=_SCATTER_LABEL_FS, colors=lcolors, zorder=5,
            avoid_xy=(apx, apy),
            avoid_bboxes=ref_bboxes,
            at_force_pull=(0.05, 0.05),
            at_force_static=(1.0, 1.0),
            at_max_move=(80, 80),
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

        # XlinkX PD2.5's point sits close enough to the target-FDR line that
        # the automatic pass above sometimes lands its label right where
        # "target FDR" is anchored, directly above the line — an expanded
        # obstacle bbox alone proved insufficient (it only informs the hard
        # structured rounds; this label can still fall to the soft adjustText
        # fallback, which doesn't guarantee clearance). So, as with Scout
        # (FL) in _plot_multi_fig2_panel_a, nudge it explicitly afterwards:
        # try pushing right/below-right of its own point, away from the line,
        # before falling back to the automatic position.
        if xlinkx25_idx is not None and texts:
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()
            other_bboxes = (ref_bboxes
                            + [t.get_window_extent(renderer=renderer)
                               for i, t in enumerate(texts) if i != xlinkx25_idx])
            pt_to_px = ax.figure.dpi / 72.0
            inv = ax.transData.inverted()
            t = texts[xlinkx25_idx]
            is_annotation = isinstance(t, mpl.text.Annotation)
            mx, my, _, _ = specs[xlinkx25_idx]
            marker_px = ax.transData.transform((mx, my))
            original = t.xyann if is_annotation else t.get_position()
            original_ha, original_va = t.get_ha(), t.get_va()
            for offset_xy, ha, va in [((8, 4), "left", "bottom"),
                                       ((8, 0), "left", "center"),
                                       ((8, -6), "left", "top")]:
                if is_annotation:
                    t.xyann = offset_xy
                else:
                    target_px = marker_px + np.array(offset_xy) * pt_to_px
                    t.set_position(tuple(inv.transform(target_px)))
                t.set_ha(ha)
                t.set_va(va)
                new_bbox = t.get_window_extent(renderer=renderer)
                if not any(new_bbox.overlaps(b) for b in other_bboxes):
                    break
            else:
                if is_annotation:
                    t.xyann = original
                else:
                    t.set_position(original)
                t.set_ha(original_ha)
                t.set_va(original_va)

    ax.set_xlabel("Reported decoy FDR (%)", fontsize=_SCATTER_LABEL_FS)
    ax.set_ylabel("Observable false-target CSM rate (%)", fontsize=_SCATTER_LABEL_FS)
    sns.despine(ax=ax)


def _plot_multi_fig2_panel_b(
    ax: plt.Axes,
    eh_csm: pd.DataFrame,
    eh_ppi: pd.DataFrame,
    em_csm: pd.DataFrame,
    em_ppi: pd.DataFrame,
    panel_label: str = "b",
) -> None:
    """Draw panel 2b on the provided axis."""
    ax.text(-0.12, 1.03, panel_label, fontsize=_SCATTER_LABEL_FS, fontweight="bold",
            transform=ax.transAxes, va="top")

    # (csm_val, ppi_val, label, color) — collected for fan layout
    label_items: list[tuple[float, float, str, str]] = []
    for eng in ENGINE_ORDER:
        for df_c, df_p, ds_key, marker in [
            (eh_csm, eh_ppi, "ecoli_human", _MARKER_EH),
            (em_csm, em_ppi, "ecoli_mycpn", _MARKER_EM),
        ]:
            c_row = df_c[df_c["search_engine"] == eng]
            p_row = df_p[df_p["search_engine"] == eng]
            if c_row.empty or p_row.empty:
                continue
            csm_val = float(c_row.iloc[0]["tt_impossible_fpr_pct"])
            ppi_val = float(p_row.iloc[0]["tt_impossible_fpr_pct"])
            color = _DS_COLORS[ds_key]
            ax.plot([0, 1], [csm_val, ppi_val], color=color, lw=1.4,
                    ls="-", alpha=0.8, zorder=3)
            ax.scatter([0, 1], [csm_val, ppi_val], s=35, color=color,
                       marker=marker, zorder=4, alpha=0.9)
            label_items.append((csm_val, ppi_val, ENGINE_LABELS.get(eng, eng), color))

    # All labels on the right, sorted descending by PPI value
    label_items.sort(key=lambda t: t[1], reverse=True)
    all_vals = [t[0] for t in label_items] + [t[1] for t in label_items]
    y_max = max(all_vals) * 1.08
    ppi_min = min(t[1] for t in label_items)

    label_x = 1.50
    n = len(label_items)
    y_positions = np.linspace(y_max * 0.97, max(ppi_min * 0.5, y_max * 0.02), n)

    for (csm_val, ppi_val, label, color), y_out in zip(label_items, y_positions):
        ax.plot([1.02, label_x - 0.02], [ppi_val, y_out],
                color=color, lw=0.6, alpha=0.7, zorder=2)
        ax.text(label_x, y_out, label, fontsize=_SCATTER_LABEL_FS, va="center", ha="left", color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CSM", "PPI"], fontsize=_SCATTER_LABEL_FS)
    ax.set_xlim(-0.15, 2.05)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Observable false-target rate (%)", fontsize=_SCATTER_LABEL_FS)
    ax.set_title("CSM → PPI progression", fontsize=_SCATTER_LABEL_FS, fontweight="bold")
    ax.axhline(REF_FDR, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.text(1.02, REF_FDR, "2%", ha="left", va="center", fontsize=_SCATTER_LABEL_FS,
            color="black", alpha=0.7, transform=ax.get_yaxis_transform(), clip_on=False)
    handles2b = [
        plt.Line2D([0], [0], color=_DS_COLORS["ecoli_human"], lw=1.5, label=DS_EH),
        plt.Line2D([0], [0], color=_DS_COLORS["ecoli_mycpn"], lw=1.5, label=DS_EM),
    ]
    ax.legend(handles=handles2b, fontsize=_SCATTER_LABEL_FS, frameon=True, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.15))
    sns.despine(ax=ax)


def _plot_multi_fig2_panel_b_parallel(
    ax: plt.Axes,
    eh_csm: pd.DataFrame,
    eh_ppi: pd.DataFrame,
    em_csm: pd.DataFrame,
    em_ppi: pd.DataFrame,
    panel_label: str = "b",
) -> None:
    """Draw an alternative panel 2b as a parallel-coordinates plot (CSM → PPI)."""
    ax.text(-0.12, 1.03, panel_label, fontsize=_SCATTER_LABEL_FS, fontweight="bold",
            transform=ax.transAxes, va="top")

    entries: list[dict] = []
    for eng in ENGINE_ORDER:
        for df_c, df_p, ds_key, marker in [
            (eh_csm, eh_ppi, "ecoli_human", _MARKER_EH),
            (em_csm, em_ppi, "ecoli_mycpn", _MARKER_EM),
        ]:
            c_row = df_c[df_c["search_engine"] == eng]
            p_row = df_p[df_p["search_engine"] == eng]
            if c_row.empty or p_row.empty:
                continue
            entries.append({
                "engine": eng,
                "label": ENGINE_LABELS.get(eng, eng),
                "csm": float(c_row.iloc[0]["tt_impossible_fpr_pct"]),
                "ppi": float(p_row.iloc[0]["tt_impossible_fpr_pct"]),
                "dataset": ds_key,
                "marker": marker,
                "color": _DS_COLORS[ds_key],
                "linestyle": "-",
            })

    # All labels on the right of the PPI axis, sorted descending by PPI value.
    label_items: list[tuple] = []   # (ppi, label, color)
    _SHORT = {"ProteinProspector": "Prot.Prosp."}

    axis_top = 65.0
    y_max = max((max(e["csm"], e["ppi"]) for e in entries), default=REF_FDR) * 1.1
    for e in entries:
        ax.plot([0, 1], [e["csm"], e["ppi"]], color=e["color"], lw=1.2,
                ls=e["linestyle"], alpha=0.8, zorder=3)
        ax.scatter([0, 1], [e["csm"], e["ppi"]], color=e["color"],
                   marker=e["marker"], s=32, alpha=0.9, zorder=4)
        label_items.append((e["ppi"], _SHORT.get(e["label"], e["label"]), e["color"]))

    # Fan: evenly spaced across full y range, highest PPI at top.
    label_items.sort(key=lambda t: t[0], reverse=True)
    n = len(label_items)
    ys_out = [(axis_top - 1) - (axis_top - 2) * i / max(n - 1, 1) for i in range(n)]
    for (orig_y, lbl, col), y_out in zip(label_items, ys_out):
        ax.plot([1.0, 1.20], [orig_y, y_out],
                color=col, lw=0.6, alpha=0.7, zorder=2, clip_on=False)
        ax.text(1.21, y_out, lbl, fontsize=_SCATTER_LABEL_FS, va="center", ha="left",
                color=col, clip_on=False)

    ax.axvline(0, color="grey", lw=1.0, alpha=0.7, zorder=1)
    ax.axvline(1, color="grey", lw=1.0, alpha=0.7, zorder=1)
    ax.axhline(REF_FDR, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.text(-0.06, REF_FDR, "2%", ha="right", va="bottom", fontsize=_SCATTER_LABEL_FS,
            color="black", alpha=0.7, clip_on=False)
    ax.set_xlim(-0.35, 1.85)   # right limit fixed; left tightened below
    ax.set_ylim(0, axis_top)

    # Remove default plot axes; keep only custom parallel-coordinate guides.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Coordinate-axis labels: rotated 90°, beside each vertical line at 2/3 height.
    _lbl_kw = dict(ha="center", va="center", fontsize=_SCATTER_LABEL_FS, rotation=90, zorder=5)
    y_label = axis_top * 2 / 3
    _t_csm = ax.text(-0.05, y_label, "CSM", **_lbl_kw)
    _t_ppi = ax.text(1.05,  y_label, "PPI", **_lbl_kw)

    # 0 and 65% markers: outside each axis line to avoid connector overlap.
    ax.text(-0.04, 0,        "0",   ha="right", va="center", fontsize=_SCATTER_LABEL_FS, color="grey")
    ax.text(-0.04, axis_top, "65%", ha="right", va="center", fontsize=_SCATTER_LABEL_FS, color="grey")
    ax.text(1.04,  0,        "0",   ha="left",  va="center", fontsize=_SCATTER_LABEL_FS, color="grey")
    ax.text(1.04,  axis_top, "65%", ha="left",  va="center", fontsize=_SCATTER_LABEL_FS, color="grey")

    _fit_left_xlim(ax)    # tighten left limit to actual glyph extents
    _fit_right_xlim(ax)   # tighten right limit to actual glyph extents

    # Apply faded backgrounds after xlim is finalised so axes-fraction coords are correct.
    for _t in (_t_csm, _t_ppi):
        add_faded_background(ax, _t, max_alpha=0.95, extent=0.8,
                             color=(1.0, 1.0, 1.0))

    handles = [
        plt.Line2D([0], [0], color=_DS_COLORS["ecoli_human"], lw=1.5, label=DS_EH),
        plt.Line2D([0], [0], color=_DS_COLORS["ecoli_mycpn"], lw=1.5, label=DS_EM),
    ]
    ax.legend(handles=handles, fontsize=_SCATTER_LABEL_FS, frameon=True, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.05))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    #sns.despine(ax=ax)


def _plot_multi_fig2_panel_c(
    ax: plt.Axes,
    vals: list[float],
    ylabel: str,
    panel_label: str | None = None,
) -> None:
    """Draw one XlinkX before/after subpanel onto the provided axis."""
    if panel_label is not None:
        _draw_panel_label(ax, panel_label, x=-0.35, y=1.03)

    ver_labels = ["XlinkX\nPD2.5", "XlinkX\nPre"]
    ver_colors = ["#d62728", "#50cf50"]
    x_pos = [0, 1]

    # Panel c is E. coli–Human throughout, so the connector takes panel b's
    # E. coli–Human colour instead of a neutral grey. Sourced from _DS_COLORS so
    # the two panels cannot drift apart. The markers keep their own red->green,
    # which encodes before/after rather than dataset.
    ax.plot(x_pos, vals, color=_DS_COLORS["ecoli_human"], lw=1.2, zorder=2)
    for xi, (val, col) in enumerate(zip(vals, ver_colors)):
        ax.scatter([xi], [val], s=90, color=col, zorder=3)
        ax.text(xi, val + (max(vals) - min(vals)) * 0.06 + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=_SCATTER_LABEL_FS)
    ax.axhline(REF_FDR, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ver_labels, fontsize=_SCATTER_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_SCATTER_LABEL_FS)
    ax.set_xlim(-0.3, 1.3)
    y_lo = min(v for v in vals if not np.isnan(v))
    y_hi = max(v for v in vals if not np.isnan(v))
    pad = (y_hi - y_lo) * 0.25 + 0.01
    ax.set_ylim(max(0, y_lo - pad), y_hi + pad * 2.5)
    ax.tick_params(labelsize=7)
    sns.despine(ax=ax)


def _plot_multi_fig2_panel_c_parallel(
    ax: plt.Axes,
    csm_vals: list[float],
    ppi_vals: list[float],
    panel_label: str | None = None,
) -> None:
    """Draw one combined panel-c as parallel coordinates (PD2.5 -> PD3.3)."""
    if panel_label is not None:
        _draw_panel_label(ax, panel_label, x=-0.12, y=1.03)

    x_pos = [0, 1]
    x_labels = ["XlinkX PD2.5", "XlinkX PD3.3"]
    # Both series are E. coli–Human, so both take panel b's colour for that
    # dataset rather than encoding CSM vs PPI by hue; the two stay apart on
    # marker shape and the italic midpoint label. Matches the non-parallel
    # panel c, which colours its connectors the same way.
    series_color = _DS_COLORS["ecoli_human"]
    line_specs = [
        ("CSM", csm_vals, series_color, "o"),
        ("PPI", ppi_vals, series_color, "^"),
    ]

    y_lo = 0.0
    y_hi = 65.0   # match panel b axis_top
    label_offset = (y_hi - y_lo) * 0.02

    # Explicit vertical coordinate axes for the two software versions.
    ax.plot([0, 0], [y_lo, y_hi], color="grey", lw=1.0, alpha=0.7, zorder=1)
    ax.plot([1, 1], [y_lo, y_hi], color="grey", lw=1.0, alpha=0.7, zorder=1)

    for name, vals, color, marker in line_specs:
        if any(np.isnan(v) for v in vals):
            continue
        ax.plot(x_pos, vals, color=color, lw=1.5, alpha=0.9, label=name, zorder=2)
        ax.scatter(x_pos, vals, s=85, color=color, marker=marker, zorder=3)
        for xi, yi in zip(x_pos, vals):
            ax.text(xi - 0.04, yi, f"{yi:.2f}", ha="right", va="center",
                    fontsize=_SCATTER_LABEL_FS, color=color)
        # Line label at midpoint
        mid_y = (vals[0] + vals[1]) / 2
        ax.text(0.5, mid_y + label_offset + 2, name, ha="center", va="bottom",
                fontsize=_SCATTER_LABEL_FS, color=color, fontstyle="italic")

    ax.axhline(REF_FDR, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.text(-0.06, REF_FDR, "2%", ha="right", va="bottom", fontsize=_SCATTER_LABEL_FS,
            color="black", alpha=0.7, clip_on=False)
    ax.set_xlim(-0.35, 1.25)
    ax.set_ylim(y_lo, y_hi)

    # Hide default axes and place custom labels for the two vertical coordinates.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    y_label = 24
    ax.text(-0.05, y_label, x_labels[0], ha="center", va="center", fontsize=_SCATTER_LABEL_FS, rotation=90)
    ax.text(1.05, y_label, x_labels[1], ha="center", va="center", fontsize=_SCATTER_LABEL_FS, rotation=90)
    ax.text(-0.22, y_hi / 2, "Observable false-target rate (%)", ha="center", va="center",
            fontsize=_SCATTER_LABEL_FS, rotation=90, clip_on=False)

    _fit_left_xlim(ax)    # tighten left limit to actual glyph extents
    _fit_right_xlim(ax)   # tighten right limit to actual glyph extents



def plot_multi_fig2(eh: pd.DataFrame, em: pd.DataFrame, b_as_parallel_coordinates: bool,
                    name_suffix: str = "") -> None:
    """Figure 2 (multi-panel): Consequences and practical use.

    2a – CSM-level scatter (reported decoy FDR vs observable false-target rate)
    2b – CSM-to-PPI progression per engine
    2c – XlinkX before/after trajectory (KS D, CSM lb, PPI lb)

    ``name_suffix`` is appended to every file this writes, so a variant built on
    a differently-defined *em* does not overwrite the standard figure.
    """
    sns.set_theme(style="white", font_scale=0.9)

    # Main figures use the entrapment criterion only (see
    # _eh_impossible_entrapment_only); co-fractionation is supplementary.
    eh = _eh_impossible_entrapment_only(eh)

    eh_csm = eh[eh["level"] == "CSM"].copy()
    eh_ppi = eh[eh["level"] == "PPI"].copy()
    em_csm = em[em["level"] == "CSM"].copy()
    em_ppi = em[em["level"] == "PPI"].copy()

    # ── layout ──
    fig = plt.figure(figsize=(19, 6))
    gs = mgridspec.GridSpec(1, 3, figure=fig, width_ratios=[2.2, 2.2, 1.6], wspace=0.15)
    ax2a = fig.add_subplot(gs[0])
    ax2b = fig.add_subplot(gs[1])
    if b_as_parallel_coordinates:
        ax2c = fig.add_subplot(gs[2])
    else:
        gs2c = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.9)
        ax2c = [fig.add_subplot(gs2c[i]) for i in range(2)]

    versions   = ["xlinkx25", "xlinkx_mgf"]
    csm_vals = [float(eh_csm[eh_csm["search_engine"] == v]["tt_impossible_fpr_pct"].values[0])
                if not eh_csm[eh_csm["search_engine"] == v].empty else float("nan")
                for v in versions]
    ppi_vals = [float(eh_ppi[eh_ppi["search_engine"] == v]["tt_impossible_fpr_pct"].values[0])
                if not eh_ppi[eh_ppi["search_engine"] == v].empty else float("nan")
                for v in versions]

    # Standalone panel_a in both scale variants (before composite is drawn).
    # Skipped wholesale when panels are off: this figure exists only to be
    # panel-saved, so drawing it would be pure waste.
    if SAVE_PANELS:
        for _scale in ("sqrt", "log2"):
            _fig_sa, _ax_sa = plt.subplots(1, 1, figsize=(6, 5))
            _plot_multi_fig2_panel_a(_ax_sa, eh_csm, em_csm, scale=_scale)
            _save_panel(_fig_sa, _ax_sa, f"fig_multi2_panel_a_{_scale}{name_suffix}")
            plt.close(_fig_sa)

    # Composite figure uses log2
    _plot_multi_fig2_panel_a(ax2a, eh_csm, em_csm, scale="log2")
    if b_as_parallel_coordinates:
        _plot_multi_fig2_panel_b_parallel(ax2b, eh_csm, eh_ppi, em_csm, em_ppi)
        _plot_multi_fig2_panel_c_parallel(ax2c, csm_vals, ppi_vals, panel_label="c")
    else:
        _plot_multi_fig2_panel_b(ax2b, eh_csm, eh_ppi, em_csm, em_ppi)
        _plot_multi_fig2_panel_c(ax2c[0], csm_vals, "CSM\nimpossible FPR (%)", panel_label="c")
        _plot_multi_fig2_panel_c(ax2c[1], ppi_vals, "PPI\nimpossible FPR (%)")
        ax2c[0].set_title(f"XlinkX: before/after trajectory\n({DS_EH})", fontsize=_SCATTER_LABEL_FS,
                          fontweight="bold")
        for ax in ax2c[1:]:
            ax.tick_params(labelleft=False)

    # Save individual panels before closing the figure
    if b_as_parallel_coordinates:
        _save_panel(fig, ax2b,  f"fig_multi2_panel_b_parallel{name_suffix}")
    else:
        _save_panel(fig, ax2b,  f"fig_multi2_panel_b{name_suffix}")
    _save_panel(fig, ax2c,  f"fig_multi2_panel_c{name_suffix}")

    if b_as_parallel_coordinates:
        _save(fig, f"fig_multi2_consequences_parallel{name_suffix}")
    else:
        _save(fig, f"fig_multi2_consequences{name_suffix}")


def plot_multi_fig2_split(eh: pd.DataFrame, em: pd.DataFrame, b_as_parallel_coordinates: bool,
                          name_suffix: str = "") -> None:
    """Figure 2 (multi-panel), panel-2a split variant.

    Same as :func:`plot_multi_fig2` except panel 2a's two-dataset overlay
    becomes two panels, one per dataset, so every point gets its own axes
    instead of sharing one with a different marker per dataset:

    a – CSM-level scatter, E. coli-Human only
    b – CSM-level scatter, E. coli-Mycoplasma only
    c – CSM-to-PPI progression per engine (2b in the combined figure)
    d – XlinkX before/after trajectory (2c in the combined figure)

    The two scatters share one x/y scale, computed from both datasets' own
    maxima, so they stay directly comparable despite being separate axes.
    """
    sns.set_theme(style="white", font_scale=0.9)

    eh = _eh_impossible_entrapment_only(eh)

    eh_csm = eh[eh["level"] == "CSM"].copy()
    eh_ppi = eh[eh["level"] == "PPI"].copy()
    em_csm = em[em["level"] == "CSM"].copy()
    em_ppi = em[em["level"] == "PPI"].copy()

    max_x_eh, max_y_eh = _fig2_scatter_bounds(eh_csm)
    max_x_em, max_y_em = _fig2_scatter_bounds(em_csm)
    lim_x = max(max_x_eh, max_x_em) * 1.08
    lim_y = max(max_y_eh, max_y_em) * 1.08

    # ── layout ── one extra column versus plot_multi_fig2's 1x3 grid: panel a
    # here is two narrower panels instead of one.
    fig = plt.figure(figsize=(22, 6))
    gs = mgridspec.GridSpec(1, 4, figure=fig, width_ratios=[1.8, 1.8, 2.2, 1.6], wspace=0.25)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])
    if b_as_parallel_coordinates:
        ax_d = fig.add_subplot(gs[3])
    else:
        gs_d = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], wspace=0.9)
        ax_d = [fig.add_subplot(gs_d[i]) for i in range(2)]

    versions = ["xlinkx25", "xlinkx_mgf"]
    csm_vals = [float(eh_csm[eh_csm["search_engine"] == v]["tt_impossible_fpr_pct"].values[0])
                if not eh_csm[eh_csm["search_engine"] == v].empty else float("nan")
                for v in versions]
    ppi_vals = [float(eh_ppi[eh_ppi["search_engine"] == v]["tt_impossible_fpr_pct"].values[0])
                if not eh_ppi[eh_ppi["search_engine"] == v].empty else float("nan")
                for v in versions]

    _plot_multi_fig2_panel_a_single(ax_a, eh_csm, _MARKER_EH, DS_EH, "a", lim_x, lim_y)
    _plot_multi_fig2_panel_a_single(ax_b, em_csm, _MARKER_EM, DS_EM, "b", lim_x, lim_y)

    if b_as_parallel_coordinates:
        _plot_multi_fig2_panel_b_parallel(ax_c, eh_csm, eh_ppi, em_csm, em_ppi, panel_label="c")
        _plot_multi_fig2_panel_c_parallel(ax_d, csm_vals, ppi_vals, panel_label="d")
    else:
        _plot_multi_fig2_panel_b(ax_c, eh_csm, eh_ppi, em_csm, em_ppi, panel_label="c")
        _plot_multi_fig2_panel_c(ax_d[0], csm_vals, "CSM\nimpossible FPR (%)", panel_label="d")
        _plot_multi_fig2_panel_c(ax_d[1], ppi_vals, "PPI\nimpossible FPR (%)")
        ax_d[0].set_title(f"XlinkX: before/after trajectory\n({DS_EH})", fontsize=_SCATTER_LABEL_FS,
                          fontweight="bold")
        for ax in ax_d[1:]:
            ax.tick_params(labelleft=False)

    if b_as_parallel_coordinates:
        _save(fig, f"fig_multi2_consequences_parallel_split{name_suffix}")
    else:
        _save(fig, f"fig_multi2_consequences_split{name_suffix}")


def _dumbbell_engines(
    eh_csm: pd.DataFrame, eh_ppi: pd.DataFrame,
    em_csm: pd.DataFrame, em_ppi: pd.DataFrame,
) -> list[str]:
    """Engines with both a CSM and a PPI value in at least one dataset."""
    def _has(df_c, df_p, eng):
        return (not df_c[df_c["search_engine"] == eng].empty
                and not df_p[df_p["search_engine"] == eng].empty)
    return [e for e in ENGINE_ORDER
            if _has(eh_csm, eh_ppi, e) or _has(em_csm, em_ppi, e)]


def _draw_csm_ppi_dumbbell(
    ax: plt.Axes,
    eh_csm: pd.DataFrame,
    eh_ppi: pd.DataFrame,
    em_csm: pd.DataFrame,
    em_ppi: pd.DataFrame,
    panel_label: str | None = None,
) -> None:
    """Alternative layout for fig_multi2 panel b/c (CSM-to-PPI progression),
    drawn on the given axes.

    The original panel puts CSM and PPI on the x-axis (two columns) and fans
    every engine's connecting line out to a label on the right — with every
    engine and both datasets sharing those two columns, the lines cross
    heavily. Here each engine gets its own evenly-spaced x position instead,
    and a vertical line connects its CSM value (circle) up or down to its PPI
    value (triangle), so no two engines' lines can cross. Colour still marks
    dataset (blue = E. coli-Human, orange = E. coli-Mycoplasma, matching
    every other figure in this project); marker shape marks stage instead of
    dataset, which this layout no longer needs shape for.
    """
    if panel_label:
        # Same x/y offset as _plot_multi_fig2_panel_a_single and
        # _plot_multi_fig2_panel_c_parallel (panels a/b and d), so the four
        # panel letters line up: c sits under a, d under b.
        _draw_panel_label(ax, panel_label, x=-0.12, y=1.03)

    engines = _dumbbell_engines(eh_csm, eh_ppi, em_csm, em_ppi)
    x = {eng: i for i, eng in enumerate(engines)}
    offset = 0.12
    CSM_MARKER, PPI_MARKER = "o", "^"

    # log2, matching panel a/b's scale: the values span two decades (Scout
    # ~0.2% to XlinkX PD2.5 ~58%), so a linear axis crushes most engines into
    # the bottom few pixels. y_floor=0.5 matches _plot_multi_fig2_panel_a_single;
    # the one true-zero value (Scout FL, E. coli-Human PPI) is clamped to it
    # for display rather than dropped, since log(0) can't be plotted at all.
    y_floor = 0.5
    y_max = y_floor
    all_vals: list[float] = []

    for eng in engines:
        xi = x[eng]
        for df_c, df_p, ds_key, dx in [
            (eh_csm, eh_ppi, "ecoli_human", -offset),
            (em_csm, em_ppi, "ecoli_mycpn",  offset),
        ]:
            c_row = df_c[df_c["search_engine"] == eng]
            p_row = df_p[df_p["search_engine"] == eng]
            if c_row.empty or p_row.empty:
                # No CSM and/or PPI value for this engine in this dataset —
                # mark it rather than leaving a silent gap, in the missing
                # dataset's own colour so it reads as "that series has
                # nothing here" rather than a third data class. Axes-fraction
                # y so it sits just above the bottom spine regardless of
                # y_floor's actual data value.
                ax.text(xi + dx, 0.02, "nd", ha="center", va="bottom",
                        fontsize=_SCATTER_LABEL_FS, color=_DS_COLORS[ds_key],
                        alpha=0.9, transform=ax.get_xaxis_transform())
                continue
            csm_val = float(c_row.iloc[0]["tt_impossible_fpr_pct"])
            ppi_val = float(p_row.iloc[0]["tt_impossible_fpr_pct"])
            all_vals += [csm_val, ppi_val]
            csm_plot = max(csm_val, y_floor)
            ppi_plot = max(ppi_val, y_floor)
            color = _DS_COLORS[ds_key]
            xi_d = xi + dx
            ax.plot([xi_d, xi_d], [csm_plot, ppi_plot], color=color, lw=1.4,
                    alpha=0.8, zorder=3)
            ax.scatter([xi_d], [csm_plot], s=45, marker=CSM_MARKER, color=color,
                       zorder=4, edgecolors="white", linewidths=0.4)
            ax.scatter([xi_d], [ppi_plot], s=45, marker=PPI_MARKER, color=color,
                       zorder=4, edgecolors="white", linewidths=0.4)

    y_max = max(all_vals) * 1.15 if all_vals else y_floor

    ax.set_yscale("log", base=2)
    ax.set_ylim(y_floor, y_max)
    log2_ticks = [t for t in (0.5, 1, 2, 4, 8, 16, 32, 64) if y_floor <= t <= y_max]
    ax.set_yticks(log2_ticks)
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    ax.set_xticks(list(x.values()))
    ax.set_xticklabels([ENGINE_LABELS.get(e, e) for e in engines],
                       fontsize=_SCATTER_LABEL_FS, rotation=45, ha="right")
    ax.set_xlim(-0.5, len(engines) - 0.5)
    ax.set_ylabel("Observable false-target rate (%)", fontsize=_SCATTER_LABEL_FS)
    ax.axhline(REF_FDR, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.text(-0.5, REF_FDR, "2%", ha="left", va="bottom",
            fontsize=_SCATTER_LABEL_FS, color="grey", alpha=0.9)

    handles = [
        plt.Line2D([0], [0], color=_DS_COLORS["ecoli_human"], lw=1.5, label=DS_EH),
        plt.Line2D([0], [0], color=_DS_COLORS["ecoli_mycpn"], lw=1.5, label=DS_EM),
        plt.scatter([], [], marker=CSM_MARKER, color="#333333", s=45, label="CSM"),
        plt.scatter([], [], marker=PPI_MARKER, color="#333333", s=45, label="PPI"),
    ]
    # Inside the axes, upper-left: every engine there (Kojak..Mango) stays
    # under 10%, so this corner is empty up to the axes top — unlike an
    # external bbox_to_anchor below the rotated x-tick labels, which
    # bbox_inches="tight" kept cropping back into collision with them.
    ax.legend(handles=handles, fontsize=_SCATTER_LABEL_FS, frameon=True, ncol=2,
             loc="upper left")
    sns.despine(ax=ax)


def plot_csm_ppi_dumbbell(eh: pd.DataFrame, em: pd.DataFrame) -> None:
    """Standalone figure wrapping :func:`_draw_csm_ppi_dumbbell`.

    Exploratory: not wired into plot_multi_fig2 itself, though it is reused
    (without its own panel label) as row 2 of :func:`plot_multi_fig2_two_row`.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    eh = _eh_impossible_entrapment_only(eh)

    eh_csm = eh[eh["level"] == "CSM"].copy()
    eh_ppi = eh[eh["level"] == "PPI"].copy()
    em_csm = em[em["level"] == "CSM"].copy()
    em_ppi = em[em["level"] == "PPI"].copy()

    engines = _dumbbell_engines(eh_csm, eh_ppi, em_csm, em_ppi)
    fig, ax = plt.subplots(figsize=(max(10, len(engines) * 0.9), 6))
    _draw_csm_ppi_dumbbell(ax, eh_csm, eh_ppi, em_csm, em_ppi)
    fig.tight_layout()
    _save(fig, "csm_ppi_dumbbell")


def plot_multi_fig2_two_row(eh: pd.DataFrame, em: pd.DataFrame, name_suffix: str = "") -> None:
    """Figure 2 (multi-panel), two-row layout.

    Row 1: panel-2a split into E. coli-Human (a) / E. coli-Mycoplasma (b)
    scatters, as in :func:`plot_multi_fig2_split`.
    Row 2: the CSM-to-PPI dumbbell plot (c) beside the XlinkX before/after
    trajectory (d), using the parallel-coordinates version of the latter
    (:func:`_plot_multi_fig2_panel_c_parallel` — CSM and PPI as one panel's
    two coordinate axes, not split into two separate charts). The dumbbell
    takes two-thirds of the row's width — it has one category per engine to
    lay out, versus the trajectory panel's two fixed points.
    """
    sns.set_theme(style="white", font_scale=0.9)
    eh = _eh_impossible_entrapment_only(eh)

    eh_csm = eh[eh["level"] == "CSM"].copy()
    eh_ppi = eh[eh["level"] == "PPI"].copy()
    em_csm = em[em["level"] == "CSM"].copy()
    em_ppi = em[em["level"] == "PPI"].copy()

    max_x_eh, max_y_eh = _fig2_scatter_bounds(eh_csm)
    max_x_em, max_y_em = _fig2_scatter_bounds(em_csm)
    lim_x = max(max_x_eh, max_x_em) * 1.08
    lim_y = max(max_y_eh, max_y_em) * 1.08

    versions = ["xlinkx25", "xlinkx_mgf"]
    csm_vals = [float(eh_csm[eh_csm["search_engine"] == v]["tt_impossible_fpr_pct"].values[0])
                if not eh_csm[eh_csm["search_engine"] == v].empty else float("nan")
                for v in versions]
    ppi_vals = [float(eh_ppi[eh_ppi["search_engine"] == v]["tt_impossible_fpr_pct"].values[0])
                if not eh_ppi[eh_ppi["search_engine"] == v].empty else float("nan")
                for v in versions]

    # ── layout ──
    fig = plt.figure(figsize=(16, 11))
    gs = mgridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.6)

    gs_top = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.25)
    ax_a = fig.add_subplot(gs_top[0])
    ax_b = fig.add_subplot(gs_top[1])

    gs_bot = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],
                                               width_ratios=[1, 1], wspace=0.3)
    ax_c = fig.add_subplot(gs_bot[0])
    ax_d = fig.add_subplot(gs_bot[1])

    _plot_multi_fig2_panel_a_single(ax_a, eh_csm, _MARKER_EH, DS_EH, "a", lim_x, lim_y)
    _plot_multi_fig2_panel_a_single(ax_b, em_csm, _MARKER_EM, DS_EM, "b", lim_x, lim_y)

    _draw_csm_ppi_dumbbell(ax_c, eh_csm, eh_ppi, em_csm, em_ppi, panel_label="c")

    # Parallel-coordinates version (CSM and PPI as one panel's two coordinate
    # axes, not split into two separate charts), matching the parallel
    # variant of plot_multi_fig2's own panel c.
    _plot_multi_fig2_panel_c_parallel(ax_d, csm_vals, ppi_vals, panel_label="d")

    _save(fig, f"fig_multi2_consequences_tworow{name_suffix}")


def main() -> None:
    # legacy no-args entry for programmatic use: run everything
    eh, em = load_summaries()
    plot_fig1_schematic()
    plot_fig2_crossvalidation(eh, em)
    plot_fig2_impossible_copresence(eh, em)
    plot_fig4_score_hist_selected()
    plot_fig3_csm_benchmarking(eh, em)
    plot_figS3_impossible_breakdown(eh, em)
    plot_figS3_impossible_breakdown(eh, em, cofractionation=False,
                                    name_suffix="_no_cofractionation")
    plot_fig6_xlinkx_before_after(eh, em)
    plot_fig7_overlap()
    plot_fig7_overlap_ppi()
    plot_supp_fig2_ks_scatter()
    plot_supp_figs1_fp_d_histogram()
    plot_supp_figs2_fp_d_histogram()
    plot_plink3_scores()
    # Non-decoy fig_multi1 is kept in the module but commented out here — the
    # decoy-design variant (panels c/d with the decoy cylinder segment and
    # decoy-match outcome stack) is the current one.
    # plot_multi_fig1(eh)
    plot_multi_fig1(eh, decoy_design=True)
    # Other fig_multi2 variants (combined, split, copresence) are kept in the
    # module but commented out here — the two-row layout is the current one.
    # plot_multi_fig2(eh, em, False)
    # plot_multi_fig2(eh, em, True)
    # plot_multi_fig2_split(eh, em, False)
    # plot_multi_fig2_split(eh, em, True)
    # plot_multi_fig2_copresence(eh, em)
    plot_multi_fig2_two_row(eh, em)
    export_oft_tables(eh, em)
    export_pair_sharing_table()
    print("Done. Figures written to", OUT_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate manuscript figures (selectively)")
    # per-figure "only" flags; if any provided the script runs only the requested figures
    parser.add_argument("--fig1-only", action="store_true", help="Run fig1 only")
    parser.add_argument("--fig2-only", action="store_true", help="Run fig2 only")
    parser.add_argument("--fig3-only", action="store_true", help="Run fig3 only")
    parser.add_argument("--fig4-only", action="store_true", help="Run fig4 only")
    parser.add_argument("--fig6-only", action="store_true", help="Run fig6 only")
    parser.add_argument("--fig7-only", action="store_true", help="Run fig7 only")
    parser.add_argument("--fig-s1-only", action="store_true", help="Run supplementary ecoli_mycpn overlap only")
    parser.add_argument("--fig-s2-only", action="store_true", help="Run supplementary FP/KS histogram only")
    parser.add_argument("--fig-s3-only", action="store_true", help="Run supplementary impossible-match breakdown only")
    parser.add_argument("--fig-s5-only", action="store_true", help="Run supplementary 2x2 overlap grid only")
    parser.add_argument("--fig-multi1-only", action="store_true", help="Run multi fig1 only")
    parser.add_argument("--fig-multi2-only", action="store_true", help="Run multi fig2 only (combined + split panel-2a variants)")
    parser.add_argument("--oft-tables-only", action="store_true", help="Run the 4 OFT-breakdown CSV exports only")
    parser.add_argument("--pair-sharing-only", action="store_true", help="Run the per-sharing-level pair-count CSV exports only")
    parser.add_argument("--fig-s6-only", action="store_true", help="Run fig_s6_total_overlap (4-panel pair-sharing bar chart) only")
    parser.add_argument("--subpannel", action="store_true",
                        help="Also write the individual sub-panel crops of the multi-panel figures")

    args = parser.parse_args()

    SAVE_PANELS = args.subpannel

    # Decide which figures to run. Only the --*-only flags select figures:
    # --subpannel is a modifier, so passing it alone must still run everything
    # rather than selecting nothing.
    any_only = any(v for k, v in vars(args).items() if k.endswith("_only"))

    eh, em = load_summaries()

    def run_fig1():
        plot_fig1_schematic()

    def run_fig2():
        plot_fig2_crossvalidation(eh, em)
        #plot_fig2_impossible_copresence(eh, em)

    def run_fig3():
        plot_fig3_csm_benchmarking(eh, em)

    def run_fig4():
        plot_fig4_score_hist_selected()

    def run_fig6():
        plot_fig6_xlinkx_before_after(eh, em)

    def run_fig7():
        plot_fig7_overlap()
        plot_fig7_overlap_ppi()

    def run_s1():
        plot_fig_s1_overlap_em()

    def run_s3():
        plot_figS3_impossible_breakdown(eh, em)
        plot_figS3_impossible_breakdown(eh, em, cofractionation=False,
                                        name_suffix="_no_cofractionation")

    def run_s5():
        # both criteria, plus entrapment-only to match the main figures
        plot_fig_s5_overlap_grid()
        plot_fig_s5_overlap_grid(entrapment_only=True,
                                 name_suffix="_entrapment_only")
        plot_fig_s5_overlap_grid(split_cofractionation=True,
                                 name_suffix="_cofractionation_split")

    def run_multi1():
        # plot_multi_fig1(eh)
        plot_multi_fig1(eh, decoy_design=True)

    def run_multi2():
        # Other fig_multi2 variants (combined, split, copresence) are kept in
        # the module but commented out here — the two-row layout is current.
        # plot_multi_fig2(eh, em, False)
        # plot_multi_fig2(eh, em, True)
        # plot_multi_fig2_split(eh, em, False)
        # plot_multi_fig2_split(eh, em, True)
        # plot_multi_fig2_copresence(eh, em)
        plot_multi_fig2_two_row(eh, em)

    def run_s2():
        plot_supp_fig2_ks_scatter()
        plot_supp_figs1_fp_d_histogram()
        plot_supp_figs2_fp_d_histogram()
        plot_plink3_scores()

    def run_oft_tables():
        export_oft_tables(eh, em)

    def run_pair_sharing():
        export_pair_sharing_table()

    def run_s6():
        plot_fig_s6_total_overlap()

    # If any --*-only specified, run only those; otherwise run all (legacy behaviour)
    if any_only:
        if args.fig1_only:
            run_fig1()
        if args.fig2_only:
            run_fig2()
        if args.fig3_only:
            run_fig3()
        if args.fig4_only:
            run_fig4()
        if args.fig6_only:
            run_fig6()
        if args.fig7_only:
            run_fig7()
        if args.fig_s1_only:
            run_s1()
        if args.fig_s3_only:
            run_s3()
        if args.fig_s5_only:
            run_s5()
        if args.fig_s2_only:
            run_s2()
        if args.fig_multi1_only:
            run_multi1()
        if args.fig_multi2_only:
            run_multi2()
        if args.oft_tables_only:
            run_oft_tables()
        if args.pair_sharing_only:
            run_pair_sharing()
        if args.fig_s6_only:
            run_s6()
    else:
        # run everything (original script behaviour)
        run_fig1(); run_fig2(); run_fig4(); run_fig3(); run_fig6(); run_fig7();
        run_s1(); run_s2(); run_s3(); run_s5(); run_multi1(); run_multi2()
        run_oft_tables()
        run_pair_sharing()
        run_s6()

    print("Done. Figures written to", OUT_DIR)
