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

import pandas as pd
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


def test_the_leader_says_it_is_the_best_rather_than_saying_nothing():
    frame = plots.comparison_frame(_frame())

    leader = frame.iloc[0]
    # every row answers the same question in the same slot; a leader with no
    # verdict reads as a row whose standing was never checked
    assert leader["verdict"] == "best"
    assert "p=" not in leader["hover"]


def test_a_tooltip_opens_with_the_name_and_the_score():
    frame = plots.comparison_frame(_frame())

    lines = frame.iloc[1]["hover"].split("<br>")
    assert lines[0] == "<b>B</b>"
    # how it stands comes first, then what it scored
    assert lines[1] == f"{plots.SEPARATED}: p=0.0100"
    assert lines[2].startswith("MAE 0.5000 ")
    assert "over seeds" in lines[2]
    assert lines[3].startswith("seed ensemble MAE:")


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


def test_a_decisive_p_value_is_not_rounded_to_zero():
    # four decimals print every decisive comparison as 0.0000, which reads as
    # certainty rather than as a number too small to write
    assert plots.p_text(2.0e-07) == ": p=2.0e-07"
    assert plots.p_text(3.1e-15) == ": p=3.1e-15"


def test_an_ordinary_p_value_keeps_its_decimals():
    assert plots.p_text(0.0234) == ": p=0.0234"
    assert plots.p_text(plots.P_FLOOR) == ": p=0.0010"


def test_an_untested_row_is_given_no_p_value():
    # a reference is drawn where its score puts it and tested against nothing
    assert plots.p_text(None) == ""


def test_an_underflowed_p_value_says_so_rather_than_claiming_zero():
    assert plots.p_text(0.0) == ": p≈0"


def test_the_band_spans_the_best_row_interval():
    # the reach a row has to fall outside of to be separated from the best,
    # carried down the column so it can be read against every other row
    frame = plots.comparison_frame(_frame())
    best = frame[frame["verdict"] == "best"].iloc[0]

    band = plots.leader_band(frame)
    assert band == (best["mae"] - best["err_minus"], best["mae"] + best["err_plus"])


def test_a_panel_with_no_tested_best_row_gets_no_band():
    # nothing was measured here, so there is no reach to anchor on
    frame = plots.comparison_frame(_frame())
    frame.loc[frame["verdict"] == "best", ["err_minus", "err_plus"]] = 0.0

    assert plots.leader_band(frame) is None


def test_the_band_is_drawn_below_the_rows_on_every_panel():
    frame = plots.comparison_frame(_frame())

    figure = plots.comparison_figure(frame, frame, subtitles=("left", "right"))
    shapes = figure.layout.shapes
    assert len(shapes) == 2
    assert [shape.xref for shape in shapes] == ["x", "x2"]
    for shape in shapes:
        assert shape.layer == "below"
        # full height, so the anchor reaches every row rather than the top one
        assert (shape.y0, shape.y1) == (0, 1)


def _ensemble_frame():
    """Return a sweep over member counts, with both errors and both spread sources."""
    return pd.DataFrame(
        {
            "n_estimators": [1, 2, 4, 8],
            "mae": [0.4443, 0.4374, 0.4395, 0.4358],
            "seed_spread": [0.0069, 0.0055, 0.0060, 0.0052],
            "ensemble_mae": [0.4301, 0.4288, 0.4290, 0.4285],
            "spearman_model": [0.21, 0.24, 0.26, 0.27],
            "spearman_ensemble": [0.18, 0.19, 0.20, 0.21],
        }
    )


def test_the_sweep_draws_both_errors_against_member_count():
    # the gap between one model and the seeds ensembled is the comparison the
    # figure exists to make, so both have to be on the same panel
    figure = plots.ensemble_figure(_ensemble_frame())

    on_error_panel = [t for t in figure.data if t.xaxis in (None, "x")]
    names = {t.name for t in on_error_panel}
    assert plots.ENSEMBLE_LABEL["single"] in names
    assert plots.ENSEMBLE_LABEL["ensemble"] in names


def test_the_member_axis_is_logarithmic_and_named_by_what_was_run():
    # the sweep doubles, so the ticks are the counts rather than powers of ten
    figure = plots.ensemble_figure(_ensemble_frame())

    assert figure.layout.xaxis.type == "log"
    assert list(figure.layout.xaxis.tickvals) == [1, 2, 4, 8]
    assert list(figure.layout.xaxis.ticktext) == ["1", "2", "4", "8"]


def test_a_spread_source_the_sweep_did_not_record_is_left_off():
    # a run without a predictive spread has no model curve to draw, and an
    # empty one would read as a measured zero
    frame = _ensemble_frame().drop(columns=["spearman_model"])

    figure = plots.ensemble_figure(frame)
    assert "model spread" not in {t.name for t in figure.data}
    assert "ensemble spread" in {t.name for t in figure.data}


def test_a_reference_is_drawn_as_a_rule_rather_than_a_row():
    # nothing in the sweep is a comparison to the reference; it is there to say
    # whether one member already clears what the rest of the work achieved
    figure = plots.ensemble_figure(_ensemble_frame(), references={"best gnn": 0.4902})

    rules = [s for s in figure.layout.shapes if s.type == "line"]
    assert len(rules) == 1
    assert rules[0].y0 == rules[0].y1 == 0.4902


def test_an_empty_sweep_is_refused():
    with pytest.raises(plots.PlotError, match="at least one member count"):
        plots.ensemble_figure(pd.DataFrame(columns=["n_estimators", "mae"]))
