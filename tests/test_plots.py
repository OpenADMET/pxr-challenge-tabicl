"""What every figure has to be true of, whichever rows it happens to hold.

These are contract tests rather than rendering tests. Nothing here checks that
a chart looks a particular way; they check the things a reader relies on to
carry a reading from one figure to the next, and which are easy to break
without noticing: two identities landing on colours that read as the same, a
tooltip whose lines arrive in a different order, a label that says one thing in
one figure and another elsewhere.
"""

from __future__ import annotations

import colorsys
import itertools
import math

import pytest

import plots

# Below this two colours read as one at marker size, whatever they measure.
# The figures draw plotly's default qualitative palette, whose closest pair is
# its blue and its purple, so 21.8 is the best any assignment of it can do and
# the bar is set just under. This guards against a regression below what the
# palette allows rather than asserting a comfortable margin.
MIN_DISTANCE = 21.0


def lab(colour: str) -> tuple[float, float, float]:
    """Return a hex colour in CIELAB, for distances that match the eye."""
    r, g, b = (int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = linear(r), linear(g), linear(b)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def family(colour: str) -> str:
    """Return the name a reader would give a colour.

    Distance is not enough on its own. Two greens can measure far apart and
    still read as a pair, which is how the graph networks came to share a hue
    and why this is checked separately.
    """
    r, g, b = (int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hue, _, saturation = colorsys.rgb_to_hls(r, g, b)
    if saturation < 0.15:
        return "grey"
    degrees = hue * 360
    bands = [
        (0, 18, "red"),
        (18, 45, "orange"),
        (45, 70, "yellow"),
        (70, 150, "green"),
        (150, 195, "cyan"),
        (195, 250, "blue"),
        (250, 290, "purple"),
        (290, 335, "magenta"),
        (335, 361, "red"),
    ]
    return next(name for low, high, name in bands if low <= degrees < high)


def test_every_identity_takes_a_colour_of_its_own():
    colours = list(plots.IDENTITY_COLOUR.values())

    assert len(set(colours)) == len(colours)


def test_no_two_identities_read_as_the_same_colour():
    both = {**plots.IDENTITY_COLOUR, **plots.VERDICT_COLOUR}
    for (one, first), (other, second) in itertools.combinations(both.items(), 2):
        if {one, other} == set(plots.VERDICT_COLOUR):
            # the two greys are a deliberate pair, dark against light
            continue
        distance = math.dist(lab(first), lab(second))
        assert distance >= MIN_DISTANCE, f"{one} and {other} are {distance:.1f} apart"


def test_every_colour_comes_from_the_palette():
    # a colour invented outside it would be the one a reader cannot place
    for name, colour in plots.IDENTITY_COLOUR.items():
        assert colour in plots.PALETTE, f"{name} is not a plotly default colour"


def test_the_verdict_colours_are_not_in_any_identity_family():
    # a reader reads blue as "not separated" and grey as "separated" wherever
    # they appear, so no identity may borrow either reading
    # a grey that is faintly blue reads as an identity, which is what this
    # caught when the verdicts moved off blue and an identity moved onto it
    reserved = {family(colour) for colour in plots.VERDICT_COLOUR.values()}
    assert reserved == {"grey"}
    for name, colour in plots.IDENTITY_COLOUR.items():
        assert family(colour) not in reserved, f"{name} reads as a verdict colour"


@pytest.mark.parametrize("label", ["CheMeleon embedding", "RDKit + Mordred 256"])
def test_a_short_name_is_left_on_one_line(label):
    assert plots.wrap_label(label) == label


def test_a_long_name_breaks_at_a_join():
    label = "CheMeleon log2FC embedding + Chemprop log2FC readout + RDKit 128"

    wrapped = plots.wrap_label(label)

    assert wrapped.count("<br>") == 1
    # the break falls after a plus, so each line is a whole ingredient
    head, tail = wrapped.split("<br>")
    assert head.endswith("+")
    assert " + " not in tail or tail.count(" + ") <= 1


def test_a_wrapped_name_reads_back_as_one_line():
    wrapped = plots.wrap_label("CheMeleon log2FC embedding + Chemprop log2FC readout + RDKit 128")

    # the tooltip's title is the same name unwrapped, not two words run together
    assert "+Chemprop" not in plots._visible(wrapped)


def _frame():
    """One panel's rows, built from a small stand-in rather than from runs."""
    ranking = [
        {
            "slug": "a",
            "config": {"embedding": "chemeleon", "readout": "none", "descriptors": "none"},
            "mae": 0.40,
            "seed_spread": 0.01,
            "ensemble": {"mae": 0.39},
            "label": "A",
            "detail": "model: TabPFN v3",
        },
        {
            "slug": "b",
            "config": {"embedding": "none", "readout": "chemprop_log2fc", "descriptors": "none"},
            "mae": 0.50,
            "seed_spread": 0.02,
            "ensemble": {"mae": 0.49},
            "label": "B",
            "detail": "model: TabPFN v3",
        },
    ]
    evidence = {
        "gate": "best_featureset",
        "leader_slug": "a",
        "indistinguishable_from_leader": [],
        "ranking": ranking,
        "significance": {
            "comparison_intervals": {"a": (0.0, 0.0), "b": (0.04, 0.16)},
            "leader_halfwidth": 0.02,
            "against_leader": [{"slug": "b", "p_value": 0.01, "separated": True}],
        },
    }
    return evidence


def test_the_leader_is_given_no_verdict_line():
    frame = plots.comparison_frame(_frame())

    leader = frame.iloc[0]
    assert leader["verdict"] == ""
    assert "separated" not in leader["hover"]


def test_a_tooltip_opens_with_the_name_and_the_score():
    frame = plots.comparison_frame(_frame())

    lines = frame.iloc[1]["hover"].split("<br>")
    assert lines[0] == "<b>B</b>"
    assert lines[1].startswith("MAE 0.5000 ")
    assert "over seeds" in lines[1]
    assert lines[2].startswith("seed ensemble MAE:")
    assert lines[3] == f"{plots.SEPARATED}: p=0.0100"


def test_a_carried_row_says_where_it_came_from():
    frame = plots.comparison_frame(
        _frame(),
        named={"b": "best_gnn"},
        carried={"b"},
        origins={"best_gnn": "established in figure 1"},
    )

    carried = frame[frame["slug"] == "b"].iloc[0]
    assert carried["role"] == plots.CARRIED
    assert "established in figure 1" in carried["hover"]


def test_a_row_this_figure_swept_is_not_drawn_as_carried():
    # it keeps its identity colour and its own shape: colour says what a row
    # is, shape says where it came from
    frame = plots.comparison_frame(_frame(), named={"b": "best_gnn"}, carried=set())

    row = frame[frame["slug"] == "b"].iloc[0]
    assert row["role"] == plots.SCORED
    assert row["colour"] == plots.IDENTITY_COLOUR["best_gnn"]


def test_a_separated_row_takes_a_grey_whisker_whatever_its_colour():
    frame = plots.comparison_frame(_frame(), named={"b": "best_gnn"})

    row = frame[frame["slug"] == "b"].iloc[0]
    assert row["colour"] == plots.IDENTITY_COLOUR["best_gnn"]
    assert row["whisker"] == plots.VERDICT_COLOUR[plots.SEPARATED]
