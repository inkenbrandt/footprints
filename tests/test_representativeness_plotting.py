"""
Plotting tests for :mod:`fluxfootprints.representativeness_plotting`.

Covers the four figures of Chu et al. (2021) the module draws -- the land-cover
composition of Fig. 1e, the footprint-against-target scatter of Fig. 1f and
Fig. 6, the sensor location bias densities of Fig. 7, and the stacked
three-level index of Fig. 5 and Fig. 8.

A plot has no return value to assert on beyond its artists, so these tests
check the three things that actually break:

* the contract -- every helper returns ``(fig, ax)``, draws on an axes it was
  handed rather than opening its own, and shows nothing;
* the data-to-artist mapping -- the marks carry the numbers that went in, the
  radius ramp runs dark-to-light with increasing radius, and a series with
  nothing to draw is skipped rather than faked;
* the frames -- both the column-shaped frame of
  :func:`sensor_location_bias_series` and the index-shaped frame of
  :func:`assess_representativeness` are accepted, since the two are the
  module's own outputs and a caller will reach for either.

Figures are rendered through the Agg backend and closed as they are made, so
nothing here needs a display.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fluxfootprints.representativeness import (
    BIAS_THRESHOLD,
    CategoricalResult,
    Level,
)
from fluxfootprints.representativeness_plotting import (
    level_colors,
    plot_bias_density,
    plot_footprint_target_scatter,
    plot_landcover_composition,
    plot_level_bars,
    radius_colors,
)

RADII = (250.0, 500.0, 1000.0)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opened, whichever way it exits."""
    yield
    plt.close("all")


@pytest.fixture
def categorical() -> list[CategoricalResult]:
    """
    Three radii over a footprint that is pure forest, in thinning surroundings.

    The dominant class holds the whole footprint and a falling share of each
    wider disc, which is the Fig. 1e story in miniature.
    """
    targets = {
        250.0: {41: 90.0, 71: 10.0},
        500.0: {41: 60.0, 71: 30.0, 82: 10.0},
        1000.0: {41: 30.0, 71: 45.0, 82: 25.0},
    }
    return [
        CategoricalResult(
            radius=radius,
            dominant_class=41,
            p_footprint=100.0,
            p_target=share[41],
            chi2=1.0,
            p_value=0.5,
            dof=2,
            level=Level.HIGH,
            footprint_composition={41: 100.0},
            target_composition=share,
        )
        for radius, share in targets.items()
    ]


@pytest.fixture
def bias() -> pd.DataFrame:
    """
    A ``sensor_location_bias_series``-shaped frame over three radii.

    The footprint sees a fixed field; the target-area mean falls with radius,
    so the bias is positive and grows -- the sign the paper reports.
    """
    rng = np.random.default_rng(0)
    rows = []
    for month in range(12):
        footprint = 0.55 + 0.05 * np.sin(month)
        for radius in RADII:
            target = footprint * (1.0 - radius / 5000.0) + 0.01 * rng.standard_normal()
            rows.append(
                {
                    "time": pd.Timestamp(f"2018-{month + 1:02d}-01"),
                    "radius": radius,
                    "value_footprint": footprint,
                    "value_target": target,
                    "delta": (footprint - target) / target,
                    "within_threshold": abs((footprint - target) / target)
                    <= BIAS_THRESHOLD,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def levels() -> pd.DataFrame:
    """A summary-shaped table of 20 sites, verdicts worsening with radius."""
    verdicts = {
        250.0: ["high"] * 14 + ["medium"] * 4 + ["low"] * 2,
        500.0: ["high"] * 8 + ["medium"] * 8 + ["low"] * 4,
        1000.0: ["high"] * 2 + ["medium"] * 8 + ["low"] * 10,
    }
    return pd.DataFrame(
        [
            {"site_id": f"S{index:02d}", "radius": radius, "landcover_level": level}
            for radius, column in verdicts.items()
            for index, level in enumerate(column)
        ]
    )


# ------------------------------
# Ramps
# ------------------------------


def _luminance(rgba) -> float:
    """Rough perceived lightness of an RGBA tuple, for ordering checks."""
    red, green, blue = rgba[:3]
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def test_radius_ramp_runs_dark_to_light():
    """The smallest radius is darkest, as the paper's legends read."""
    colors = radius_colors([1000, 250, 500])
    assert list(colors) == [250.0, 500.0, 1000.0]

    lightness = [_luminance(colors[radius]) for radius in colors]
    assert lightness == sorted(lightness)


def test_level_ramp_runs_high_to_low():
    """HIGH is darkest, LOW lightest, and all three are distinct."""
    colors = level_colors()
    assert list(colors) == [Level.HIGH, Level.MEDIUM, Level.LOW]

    lightness = [_luminance(colors[level]) for level in colors]
    assert lightness == sorted(lightness)
    assert len(set(lightness)) == 3


# ------------------------------
# Fig. 1e -- land-cover composition
# ------------------------------


def test_composition_returns_fig_ax(categorical):
    """The contract: a Figure, its Axes, and no window."""
    fig, ax = plot_landcover_composition(categorical)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.figure is fig


def test_composition_draws_footprint_and_every_radius(categorical):
    """One series per radius plus the footprint reference, all in the legend."""
    _, ax = plot_landcover_composition(categorical)

    labels = [line.get_label() for line in ax.lines]
    assert labels == ["250 m", "500 m", "1000 m", "Footprint-weighted"]

    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend[0] == "Footprint-weighted"
    assert legend[1:] == ["250 m", "500 m", "1000 m"]


def test_composition_marks_carry_the_percentages(categorical):
    """The x of each mark is the share the result carried, class by class."""
    _, ax = plot_landcover_composition(categorical)

    classes = [text.get_text() for text in ax.get_yticklabels()]
    assert classes[0] == "41", "the dominant class leads the ordering"

    footprint = ax.lines[-1]
    assert footprint.get_xdata()[0] == pytest.approx(100.0)

    disc_250 = ax.lines[0]
    assert disc_250.get_xdata()[0] == pytest.approx(90.0)
    # A class the footprint never saw is drawn at zero, not dropped.
    assert min(disc_250.get_xdata()) == pytest.approx(0.0)


def test_composition_honours_class_labels_and_max_classes(categorical):
    """Labels are substituted, and the tail of small classes can be cut."""
    _, ax = plot_landcover_composition(
        categorical, class_labels={41: "Deciduous forest"}, max_classes=2
    )

    labels = [text.get_text() for text in ax.get_yticklabels()]
    assert labels[0] == "Deciduous forest"
    assert len(labels) == 2


def test_composition_draws_on_a_given_axes(categorical):
    """A supplied axes is drawn on rather than a fresh figure opened."""
    fig, ax = plt.subplots()
    out_fig, out_ax = plot_landcover_composition(categorical, ax=ax)

    assert out_ax is ax
    assert out_fig is fig


def test_composition_rejects_empty_results():
    """Nothing to draw is an error, not a blank panel."""
    with pytest.raises(ValueError, match="results is empty"):
        plot_landcover_composition([])


def test_composition_rejects_unknown_radius(categorical):
    """A radius the results never held is a caller error."""
    with pytest.raises(ValueError, match="are not in results"):
        plot_landcover_composition(categorical, radii=[750.0])


def test_composition_rejects_nonpositive_max_classes(categorical):
    """Keeping zero classes would draw an empty panel."""
    with pytest.raises(ValueError, match="max_classes must be positive"):
        plot_landcover_composition(categorical, max_classes=0)


# ------------------------------
# Fig. 1f, Fig. 6 -- footprint against target area
# ------------------------------


def test_scatter_draws_a_series_and_a_fit_per_radius(bias):
    """Each radius gets its points, its RMA line, and one shared 1:1."""
    _, ax = plot_footprint_target_scatter(bias)

    labelled = [
        line.get_label()
        for line in ax.lines
        if not str(line.get_label()).startswith("_")
    ]
    assert labelled == ["250 m", "500 m", "1000 m", "1:1"]
    # Three series, three fits, one reference.
    assert len(ax.lines) == 7


def test_scatter_shares_one_range_on_both_axes(bias):
    """Equal aspect and one range, so 1:1 runs corner to corner."""
    _, ax = plot_footprint_target_scatter(bias)

    assert ax.get_xlim() == pytest.approx(ax.get_ylim())
    assert ax.get_aspect() == 1.0


def test_scatter_single_series_annotates_and_drops_the_legend_entry(bias):
    """
    One series is named by the axis labels, so only 1:1 is worth a legend
    entry; the fit statistics go in the corner instead.
    """
    _, ax = plot_footprint_target_scatter(
        bias[bias["radius"] == 250.0], radius_column=None
    )

    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend == ["1:1"]

    annotation = "\n".join(text.get_text() for text in ax.texts)
    assert "slope" in annotation and "n = 12" in annotation


def test_scatter_skips_the_fit_when_asked(bias):
    """``fit=False`` leaves the points and the reference line alone."""
    _, ax = plot_footprint_target_scatter(bias, fit=False)

    assert len(ax.lines) == 4
    assert not ax.texts


def test_scatter_accepts_an_indexed_frame(bias):
    """
    ``assess_representativeness`` carries radius in the index; the same figure
    comes out either way.
    """
    indexed = bias.set_index(["time", "radius"])
    _, ax = plot_footprint_target_scatter(indexed)

    labelled = [
        line.get_label()
        for line in ax.lines
        if not str(line.get_label()).startswith("_")
    ]
    assert labelled == ["250 m", "500 m", "1000 m", "1:1"]


def test_scatter_labels_name_the_variable(bias):
    """The field's name reaches both axis labels."""
    _, ax = plot_footprint_target_scatter(bias, variable="NDVI")

    assert ax.get_xlabel() == "Footprint-weighted NDVI"
    assert ax.get_ylabel() == "Target-area NDVI"


def test_scatter_rejects_a_non_frame():
    """The frames are the input; an array is a mistake worth naming."""
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        plot_footprint_target_scatter(np.zeros((4, 2)))


def test_scatter_reports_a_missing_column(bias):
    """A renamed column is named in the error, with what was found."""
    with pytest.raises(ValueError, match="missing the column"):
        plot_footprint_target_scatter(bias.drop(columns="value_target"))


def test_scatter_rejects_an_all_missing_frame(bias):
    """No finite pair is an error rather than an empty panel."""
    empty = bias.assign(value_target=np.nan)
    with pytest.raises(ValueError, match="No finite"):
        plot_footprint_target_scatter(empty)


# ------------------------------
# Fig. 7 -- sensor location bias
# ------------------------------


def test_density_draws_one_curve_per_radius(bias):
    """A kernel density per radius, over the paper's +/-100 % window."""
    _, ax = plot_bias_density(bias)

    labelled = [
        line.get_label()
        for line in ax.lines
        if not str(line.get_label()).startswith("_")
    ]
    assert labelled == ["250 m", "500 m", "1000 m"]
    assert ax.get_xlim() == pytest.approx((-100.0, 100.0))
    assert ax.get_ylim()[0] == pytest.approx(0.0)


def test_density_scales_to_percent_by_default(bias):
    """
    The default axis is a percentage, as Fig. 7 is; ``percent=False`` keeps the
    fractions the analysis returns, which moves the curve by two decades.
    """
    _, percent_ax = plot_bias_density(bias)
    _, fraction_ax = plot_bias_density(bias, percent=False, clip=(-1.0, 1.0))

    assert percent_ax.get_xlabel().endswith("(%)")
    assert fraction_ax.get_xlabel().endswith("(-)")

    def peak(ax):
        line = ax.lines[0]
        return line.get_xdata()[np.argmax(line.get_ydata())]

    assert peak(percent_ax) == pytest.approx(100.0 * peak(fraction_ax), rel=0.05)


def test_density_draws_threshold_guides_only_when_asked(bias):
    """The +/-10 % guides are opt-in, as the published figure has none."""
    _, bare = plot_bias_density(bias)
    _, guided = plot_bias_density(bias, threshold=BIAS_THRESHOLD)

    # One zero rule either way, plus the two guides when asked for.
    assert len(guided.lines) - len(bare.lines) == 2


def test_density_reads_the_assess_bias_column(bias):
    """``assess_representativeness`` names the same quantity ``bias``."""
    renamed = bias.rename(columns={"delta": "bias"})
    _, ax = plot_bias_density(renamed)

    assert len(ax.get_legend().get_texts()) == 3


def test_density_skips_a_radius_with_no_spread(bias):
    """
    A radius whose biases are identical has no bandwidth to estimate, so it is
    left out rather than drawn as a spike.
    """
    flat = bias.copy()
    flat.loc[flat["radius"] == 500.0, "delta"] = 0.2
    _, ax = plot_bias_density(flat)

    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend == ["250 m", "1000 m"]


def test_density_rejects_a_frame_with_nothing_to_estimate(bias):
    """Every radius degenerate is an error, not a blank panel."""
    flat = bias.assign(delta=0.0)
    with pytest.raises(ValueError, match="no kernel density"):
        plot_bias_density(flat)


def test_density_rejects_a_backwards_clip(bias):
    """A window that does not increase would draw nothing."""
    with pytest.raises(ValueError, match="clip must increase"):
        plot_bias_density(bias, clip=(50.0, -50.0))


def test_density_reports_a_missing_bias_column(bias):
    """With neither conventional name present, both are named."""
    with pytest.raises(ValueError, match="none of the bias columns"):
        plot_bias_density(bias.drop(columns="delta"))


# ------------------------------
# Fig. 5, Fig. 8 -- the three-level index
# ------------------------------


def test_level_bars_stack_to_a_hundred_percent(levels):
    """Three stacked series per radius, each bar filling the axis."""
    _, ax = plot_level_bars(levels, level_column="landcover_level")

    assert len(ax.containers) == 3
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "High",
        "Medium",
        "Low",
    ]

    totals = np.zeros(len(RADII))
    for container in ax.containers:
        totals += np.array([patch.get_height() for patch in container])
    assert totals == pytest.approx(100.0)


def test_level_bars_carry_the_shares(levels):
    """HIGH falls from 70 % to 10 % as the disc widens, as the table says."""
    _, ax = plot_level_bars(levels, level_column="landcover_level")

    high = [patch.get_height() for patch in ax.containers[0]]
    assert high == pytest.approx([70.0, 40.0, 10.0])

    assert [text.get_text() for text in ax.get_xticklabels()] == ["250", "500", "1000"]
    assert ax.get_ylabel() == "Percentage of sites (%)"


def test_level_bars_can_stack_counts(levels):
    """``percent=False`` shows how many sites each bar rests on."""
    _, ax = plot_level_bars(levels, level_column="landcover_level", percent=False)

    high = [patch.get_height() for patch in ax.containers[0]]
    assert high == pytest.approx([14.0, 8.0, 2.0])
    assert ax.get_ylabel() == "Number of sites"


def test_level_bars_drop_unclassified_sites(levels):
    """
    A site with no verdict at a radius leaves the count, as the paper's own
    34 misclassified sites leave its land-cover panels.
    """
    dropped = levels.copy()
    dropped.loc[
        (dropped["radius"] == 250.0) & (dropped["landcover_level"] == "low"),
        "landcover_level",
    ] = np.nan
    _, ax = plot_level_bars(dropped, level_column="landcover_level", percent=False)

    totals = np.zeros(len(RADII))
    for container in ax.containers:
        totals += np.array([patch.get_height() for patch in container])
    assert totals == pytest.approx([18.0, 20.0, 20.0])


def test_level_bars_accept_an_indexed_frame(levels):
    """Radius in the index, as ``assess_representativeness`` leaves it."""
    indexed = levels.set_index(["site_id", "radius"]).rename(
        columns={"landcover_level": "level"}
    )
    _, ax = plot_level_bars(indexed)

    high = [patch.get_height() for patch in ax.containers[0]]
    assert high == pytest.approx([70.0, 40.0, 10.0])


def test_level_bars_reject_an_ambiguous_table(levels):
    """
    A summary carries both halves of the analysis, so the column to draw has
    to be named -- but the detection still has to find one when it is alone.
    """
    both = levels.assign(continuous_level="medium")
    _, ax = plot_level_bars(both, level_column="continuous_level")
    medium = [patch.get_height() for patch in ax.containers[1]]
    assert medium == pytest.approx([100.0] * len(RADII))

    with pytest.raises(ValueError, match="has no column"):
        plot_level_bars(levels, level_column="continuous_level")


def test_level_bars_reject_a_value_that_is_not_a_level(levels):
    """A typo in the level column is named rather than silently dropped."""
    typo = levels.replace({"landcover_level": {"medium": "moderate"}})
    with pytest.raises(ValueError, match="which are not"):
        plot_level_bars(typo, level_column="landcover_level")


def test_level_bars_reject_an_unclassified_table(levels):
    """Every verdict missing leaves nothing to count."""
    blank = levels.assign(landcover_level=np.nan)
    with pytest.raises(ValueError, match="missing"):
        plot_level_bars(blank, level_column="landcover_level")
