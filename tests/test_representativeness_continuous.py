"""
Continuous-field representativeness tests for :mod:`fluxfootprints.representativeness`.

Sect. 2.4 of Chu et al. (2021) regresses target-area against footprint-weighted
values with R's ``lmodel2`` -- a reduced major axis fit, not ordinary least
squares -- and turns the slope, intercept, and R-squared into a three-level
index. The fixtures here are built to pin down each step without copying
numbers back out of the implementation:

* the RMA slope has three independent characterisations -- the OLS slope
  divided by ``r``, the ratio of standard deviations signed by ``r``, and the
  reciprocal of the fit with the variables swapped -- so each is checked
  against the others rather than against a stored constant. The third is what
  separates RMA from OLS most sharply: OLS is not symmetric in its variables;
* points on an exact line fix the slope, the intercept, and R-squared at
  values that are exact in binary, and collapse the confidence interval onto
  the point estimate, which is the degenerate end of McArdle's (1988) formula;
* the classification thresholds are exercised exactly on 0.8, 0.6, 0.9, 1.1,
  and +/-0.1, because every one of them is inclusive on one side in the paper
  and the boundary is where a three-level index is decided;
* the matched-period frames are written out by hand so that ``n``, RMSE, and
  MAE can be computed on paper, and so the six-match rule of the paper can be
  crossed one period at a time.

The analytical and bootstrap intervals are checked against each other on a
clean sample rather than either being checked against a hard-coded width: they
rest on different assumptions and agreeing to a few thousandths is the claim
worth making about them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxfootprints.representativeness import (
    DEFAULT_ALPHA,
    MIN_MATCHES,
    TARGET_RADII,
    Level,
    classify_continuous,
    continuous_representativeness,
    model2_regression,
    rma_regression,
    sensor_location_bias_series,
)

#: A scattered but well-correlated pair of series, in the 0-1 range of EVI.
FOOTPRINT = np.array([0.21, 0.35, 0.52, 0.66, 0.71, 0.44, 0.29, 0.61])
TARGET = np.array([0.19, 0.36, 0.47, 0.62, 0.70, 0.41, 0.31, 0.57])

PAIR_COLUMNS = ["time", "radius", "value_footprint", "value_target"]

RESULT_COLUMNS = [
    "radius",
    "n",
    "intercept",
    "slope",
    "r_squared",
    "p_value",
    "rmse",
    "mae",
    "intercept_lower",
    "intercept_upper",
    "slope_lower",
    "slope_upper",
    "sufficient",
    "level",
]


# ----------------------------
# Fixtures
# ----------------------------
def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """The least-squares slope, fitted independently of the module."""
    return float(np.polyfit(x, y, 1)[0])


def matched_frame(
    footprint: np.ndarray = FOOTPRINT,
    target: np.ndarray = TARGET,
    radius: float = 250.0,
) -> pd.DataFrame:
    """One radius of the tidy frame :func:`sensor_location_bias_series` returns."""
    return pd.DataFrame(
        {
            "time": pd.date_range("2016-01-01", periods=len(footprint), freq="MS"),
            "radius": float(radius),
            "value_footprint": footprint,
            "value_target": target,
        }
    )


def two_radii_frame() -> pd.DataFrame:
    """Two radii over the same periods, the second the more biased."""
    return pd.concat(
        [
            matched_frame(radius=250.0),
            matched_frame(target=0.7 * TARGET + 0.02, radius=3000.0),
        ],
        ignore_index=True,
    )


# ----------------------------
# rma_regression, Eq. 7
# ----------------------------
def test_the_slope_is_the_least_squares_slope_divided_by_the_correlation():
    """The defining relation between the RMA and OLS slopes."""
    r = float(np.corrcoef(FOOTPRINT, TARGET)[0, 1])

    fit = rma_regression(FOOTPRINT, TARGET)

    assert fit.slope == pytest.approx(ols_slope(FOOTPRINT, TARGET) / r)


def test_the_slope_is_the_ratio_of_standard_deviations_signed_by_the_correlation():
    expected = np.std(TARGET) / np.std(FOOTPRINT)

    fit = rma_regression(FOOTPRINT, TARGET)

    assert fit.slope == pytest.approx(expected)


def test_least_squares_is_the_shallower_fit_and_must_not_be_substituted():
    """
    The bias the docstring warns about, in the direction it warns about.

    With ``r < 1`` the OLS slope is a strict attenuation of the RMA slope, so
    a site fitted by OLS drifts towards the bottom of the paper's
    ``0.9 <= slope <= 1.1`` window for no reason but the scatter.
    """
    fit = rma_regression(FOOTPRINT, TARGET)

    assert fit.r_squared < 1.0
    assert ols_slope(FOOTPRINT, TARGET) < fit.slope


def test_the_fit_is_symmetric_in_its_two_variables():
    """
    Swapping the variables inverts the RMA slope exactly; OLS does not.

    Neither series is the error-free one here, so a fit that depended on which
    was called the predictor would not be a model II fit at all.
    """
    forward = rma_regression(FOOTPRINT, TARGET)
    reverse = rma_regression(TARGET, FOOTPRINT)

    assert reverse.slope == pytest.approx(1.0 / forward.slope)
    assert ols_slope(TARGET, FOOTPRINT) != pytest.approx(1.0 / forward.slope)


def test_the_line_passes_through_the_centroid():
    fit = rma_regression(FOOTPRINT, TARGET)

    assert fit.intercept + fit.slope * FOOTPRINT.mean() == pytest.approx(TARGET.mean())


def test_r_squared_and_the_p_value_are_the_correlations():
    r = float(np.corrcoef(FOOTPRINT, TARGET)[0, 1])

    fit = rma_regression(FOOTPRINT, TARGET)

    assert fit.r_squared == pytest.approx(r**2)
    assert 0.0 < fit.p_value < DEFAULT_ALPHA


def test_a_perfect_line_is_recovered_exactly():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    fit = rma_regression(x, 2.0 * x + 1.0)

    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.r_squared == pytest.approx(1.0)


def test_a_perfect_line_leaves_no_room_in_the_interval():
    """B collapses to zero with no scatter, and the limits onto the estimate."""
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    fit = rma_regression(x, 2.0 * x + 1.0)

    assert fit.slope_ci == pytest.approx((2.0, 2.0))
    assert fit.intercept_ci == pytest.approx((1.0, 1.0))


def test_the_interval_brackets_the_estimate_and_is_ordered():
    fit = rma_regression(FOOTPRINT, TARGET)

    assert fit.slope_ci[0] < fit.slope < fit.slope_ci[1]
    assert fit.intercept_ci[0] < fit.intercept < fit.intercept_ci[1]


def test_more_scatter_widens_the_interval():
    noise = np.array([0.04, -0.05, 0.06, -0.04, 0.05, -0.06, 0.04, -0.04])

    tight = rma_regression(FOOTPRINT, TARGET)
    loose = rma_regression(FOOTPRINT, TARGET + noise)

    tight_width = tight.slope_ci[1] - tight.slope_ci[0]
    loose_width = loose.slope_ci[1] - loose.slope_ci[0]
    assert loose_width > tight_width


def test_the_intercept_limits_follow_the_slope_limits_through_the_centroid():
    fit = rma_regression(FOOTPRINT, TARGET)

    expected = sorted(TARGET.mean() - np.array(fit.slope_ci) * FOOTPRINT.mean())
    assert fit.intercept_ci == pytest.approx(tuple(expected))


def test_a_negative_relation_gives_a_negative_slope_with_ordered_limits():
    fit = rma_regression(FOOTPRINT, -TARGET)

    assert fit.slope < 0.0
    assert fit.slope_ci[0] < fit.slope < fit.slope_ci[1]


def test_the_two_interval_methods_agree_on_a_clean_sample():
    rng = np.random.default_rng(20210214)
    x = rng.uniform(0.1, 0.9, 200)
    y = 0.95 * x + 0.02 + rng.normal(0.0, 0.02, 200)

    analytical = rma_regression(x, y)
    bootstrap = rma_regression(x, y, ci_method="bootstrap", n_boot=4000, random_state=0)

    assert bootstrap.slope == pytest.approx(analytical.slope)
    assert bootstrap.slope_ci == pytest.approx(analytical.slope_ci, abs=0.01)
    assert bootstrap.intercept_ci == pytest.approx(analytical.intercept_ci, abs=0.01)


def test_the_bootstrap_repeats_under_the_same_seed_and_moves_without_one():
    first = rma_regression(FOOTPRINT, TARGET, ci_method="bootstrap", random_state=7)
    same = rma_regression(FOOTPRINT, TARGET, ci_method="bootstrap", random_state=7)
    other = rma_regression(FOOTPRINT, TARGET, ci_method="bootstrap", random_state=8)

    assert first.slope_ci == same.slope_ci
    assert first.slope_ci != other.slope_ci


def test_the_method_and_level_are_recorded_on_the_fit():
    fit = rma_regression(FOOTPRINT, TARGET, ci_level=0.9)

    assert fit.ci_level == 0.9
    assert fit.ci_method == "analytical"
    assert fit.n == len(FOOTPRINT)


def test_a_narrower_level_gives_a_narrower_interval():
    ninety = rma_regression(FOOTPRINT, TARGET, ci_level=0.90)
    ninety_nine = rma_regression(FOOTPRINT, TARGET, ci_level=0.99)

    assert ninety.slope_ci[0] > ninety_nine.slope_ci[0]
    assert ninety.slope_ci[1] < ninety_nine.slope_ci[1]


def test_periods_missing_either_value_are_dropped():
    footprint = np.append(FOOTPRINT, [np.nan, 0.5])
    target = np.append(TARGET, [0.5, np.nan])

    fit = rma_regression(footprint, target)

    assert fit.n == len(FOOTPRINT)
    assert fit.slope == pytest.approx(rma_regression(FOOTPRINT, TARGET).slope)


def test_a_constant_variable_leaves_the_fit_undefined_without_warning():
    """No correlation exists against a constant, and scipy is never asked for one."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = rma_regression(FOOTPRINT, np.full_like(FOOTPRINT, 0.4))

    assert np.isnan(fit.slope)
    assert np.isnan(fit.r_squared)
    assert np.isnan(fit.slope_ci).all()
    assert fit.n == len(FOOTPRINT)


def test_an_exactly_zero_correlation_leaves_the_slope_unoriented():
    """
    The magnitude of the RMA slope survives an uncorrelated pair; its sign
    does not, so ``lmodel2`` refuses the fit and so does this.
    """
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    y = np.array([1.0, -1.0, -1.0, 1.0])
    assert np.corrcoef(x, y)[0, 1] == 0.0

    fit = rma_regression(x, y)

    assert np.isnan(fit.slope)
    assert np.isnan(fit.intercept)
    assert fit.r_squared == pytest.approx(0.0)
    assert fit.p_value == pytest.approx(1.0)


def test_mismatched_series_raise():
    with pytest.raises(ValueError, match="pair up one for one"):
        rma_regression(FOOTPRINT, TARGET[:-1])


def test_too_few_pairs_to_fit_raise():
    with pytest.raises(ValueError, match="at least three finite pairs"):
        rma_regression([0.1, 0.2], [0.1, 0.2])


def test_pairs_lost_to_missing_values_are_counted_before_refusing():
    with pytest.raises(ValueError, match="at least three finite pairs"):
        rma_regression([0.1, 0.2, 0.3, np.nan], [0.1, np.nan, 0.3, 0.4])


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 95.0])
def test_a_confidence_level_off_the_unit_interval_raises(level):
    with pytest.raises(ValueError, match="ci_level"):
        rma_regression(FOOTPRINT, TARGET, ci_level=level)


def test_an_unknown_interval_method_raises():
    with pytest.raises(ValueError, match="ci_method"):
        rma_regression(FOOTPRINT, TARGET, ci_method="jackknife")


def test_a_bootstrap_of_no_resamples_raises():
    with pytest.raises(ValueError, match="n_boot"):
        rma_regression(FOOTPRINT, TARGET, ci_method="bootstrap", n_boot=0)


# ----------------------------
# model2_regression
# ----------------------------
def test_the_four_value_form_is_the_same_fit():
    intercept, slope, r_squared, p_value = model2_regression(FOOTPRINT, TARGET)
    fit = rma_regression(FOOTPRINT, TARGET)

    assert (intercept, slope, r_squared, p_value) == (
        fit.intercept,
        fit.slope,
        fit.r_squared,
        fit.p_value,
    )


# ----------------------------
# classify_continuous, Sect. 2.4
# ----------------------------
def test_a_tight_one_to_one_fit_is_high():
    assert classify_continuous(0.94, 0.96, 0.02, 1e-8) is Level.HIGH


@pytest.mark.parametrize(
    ("r_squared", "slope", "intercept"),
    [
        (0.8, 1.0, 0.0),
        (0.95, 0.9, 0.0),
        (0.95, 1.1, 0.0),
        (0.95, 1.0, 0.1),
        (0.95, 1.0, -0.1),
    ],
)
def test_the_high_criteria_are_inclusive_at_their_edges(r_squared, slope, intercept):
    assert classify_continuous(r_squared, slope, intercept, 1e-8) is Level.HIGH


def test_a_fit_just_inside_r_squared_but_outside_the_slope_window_is_medium():
    assert classify_continuous(0.95, 1.11, 0.0, 1e-8) is Level.MEDIUM


def test_a_fit_just_inside_r_squared_but_outside_the_intercept_window_is_medium():
    assert classify_continuous(0.95, 1.0, 0.11, 1e-8) is Level.MEDIUM


def test_a_significant_but_scattered_fit_is_medium():
    assert classify_continuous(0.6, 0.7, 0.2, 0.049) is Level.MEDIUM


def test_a_fit_below_the_medium_r_squared_is_low():
    assert classify_continuous(0.599, 1.0, 0.0, 1e-8) is Level.LOW


def test_a_fit_at_the_significance_level_is_low():
    """``p < alpha`` is strict in the paper."""
    assert classify_continuous(0.7, 1.0, 0.0, DEFAULT_ALPHA) is Level.LOW


def test_the_significance_level_is_configurable():
    assert classify_continuous(0.7, 0.5, 0.3, 0.02, alpha=0.01) is Level.LOW
    assert classify_continuous(0.7, 0.5, 0.3, 0.02, alpha=0.10) is Level.MEDIUM


def test_a_degenerate_fit_falls_through_to_low():
    nan = float("nan")
    assert classify_continuous(nan, nan, nan, nan) is Level.LOW


# ----------------------------
# continuous_representativeness, Sect. 2.4
# ----------------------------
def test_one_row_per_radius_in_the_order_given():
    result = continuous_representativeness(two_radii_frame(), radii=(3000.0, 250.0))

    assert list(result.columns) == RESULT_COLUMNS
    assert result["radius"].tolist() == [3000.0, 250.0]
    assert result.index.tolist() == [0, 1]


def test_the_row_of_a_radius_is_the_fit_of_that_radius_alone():
    frame = two_radii_frame()
    rows = frame[frame["radius"] == 250.0]
    fit = rma_regression(rows["value_footprint"], rows["value_target"])

    result = continuous_representativeness(frame, radii=(250.0, 3000.0))
    row = result.iloc[0]

    assert row["n"] == fit.n
    assert row["slope"] == pytest.approx(fit.slope)
    assert row["intercept"] == pytest.approx(fit.intercept)
    assert row["r_squared"] == pytest.approx(fit.r_squared)
    assert row["p_value"] == pytest.approx(fit.p_value)
    assert (row["slope_lower"], row["slope_upper"]) == pytest.approx(fit.slope_ci)
    assert (row["intercept_lower"], row["intercept_upper"]) == pytest.approx(
        fit.intercept_ci
    )


def test_the_errors_are_taken_against_the_one_to_one_line():
    """
    Table 1's RMSE and MAE compare the two series, not the fitted residuals.
    """
    frame = matched_frame()
    difference = TARGET - FOOTPRINT

    row = continuous_representativeness(frame, radii=(250.0,)).iloc[0]

    assert row["rmse"] == pytest.approx(np.sqrt(np.mean(difference**2)))
    assert row["mae"] == pytest.approx(np.mean(np.abs(difference)))


def test_the_bias_grows_with_the_target_area():
    """
    The paper's central result: the slope falls and the errors rise as the
    disc extends past the footprint.
    """
    result = continuous_representativeness(two_radii_frame(), radii=(250.0, 3000.0))

    assert result.loc[0, "slope"] > result.loc[1, "slope"]
    assert result.loc[0, "rmse"] < result.loc[1, "rmse"]


def test_a_radius_short_of_the_match_count_is_not_fitted():
    frame = matched_frame(FOOTPRINT[:4], TARGET[:4])

    row = continuous_representativeness(frame, radii=(250.0,)).iloc[0]

    assert row["n"] == 4
    assert not row["sufficient"]
    assert pd.isna(row["level"])
    assert np.isnan(row["slope"])
    assert np.isnan(row["r_squared"])
    assert np.isnan(row["slope_lower"])


def test_an_unfitted_radius_still_reports_its_errors():
    """``n`` and the two errors need no regression, so they are not withheld."""
    frame = matched_frame(FOOTPRINT[:4], TARGET[:4])

    row = continuous_representativeness(frame, radii=(250.0,)).iloc[0]

    difference = TARGET[:4] - FOOTPRINT[:4]
    assert row["rmse"] == pytest.approx(np.sqrt(np.mean(difference**2)))
    assert row["mae"] == pytest.approx(np.mean(np.abs(difference)))


def test_insufficient_data_is_not_reported_as_low_representativeness():
    """
    The distinction the paper draws by fitting only 166 of its 214 sites: too
    little evidence to judge is not the same as a poor fit.
    """
    frame = matched_frame(FOOTPRINT[:4], np.array([0.9, 0.1, 0.8, 0.2]))

    row = continuous_representativeness(frame, radii=(250.0,)).iloc[0]

    assert row["level"] is not Level.LOW
    assert pd.isna(row["level"])


def test_the_six_matches_of_the_paper_are_the_default_bar():
    five = continuous_representativeness(
        matched_frame(FOOTPRINT[:5], TARGET[:5]), radii=(250.0,)
    )
    six = continuous_representativeness(
        matched_frame(FOOTPRINT[:6], TARGET[:6]), radii=(250.0,)
    )

    assert MIN_MATCHES == 6
    assert not five.loc[0, "sufficient"]
    assert six.loc[0, "sufficient"]
    assert six.loc[0, "level"] in tuple(Level)


def test_the_bar_is_configurable():
    frame = matched_frame(FOOTPRINT[:4], TARGET[:4])

    result = continuous_representativeness(frame, radii=(250.0,), min_matches=3)

    assert result.loc[0, "sufficient"]
    assert not pd.isna(result.loc[0, "level"])


def test_periods_missing_either_value_do_not_count_towards_the_bar():
    footprint = np.append(FOOTPRINT[:5], np.nan)
    target = np.append(TARGET[:5], 0.6)

    row = continuous_representativeness(
        matched_frame(footprint, target), radii=(250.0,)
    ).iloc[0]

    assert row["n"] == 5
    assert not row["sufficient"]


def test_a_radius_absent_from_the_frame_is_reported_empty_rather_than_raising():
    row = continuous_representativeness(matched_frame(), radii=(750.0,)).iloc[0]

    assert row["n"] == 0
    assert not row["sufficient"]
    assert np.isnan(row["rmse"])
    assert pd.isna(row["level"])


def test_an_integer_radius_finds_the_float_in_the_frame():
    result = continuous_representativeness(matched_frame(radius=250.0), radii=(250,))

    assert result.loc[0, "n"] == len(FOOTPRINT)


def test_radii_of_the_frame_that_were_not_asked_for_are_ignored():
    result = continuous_representativeness(two_radii_frame(), radii=(250.0,))

    assert len(result) == 1
    assert result.loc[0, "n"] == len(FOOTPRINT)


def test_the_settings_are_recorded_on_the_frame():
    result = continuous_representativeness(
        matched_frame(), radii=(250.0,), min_matches=4, alpha=0.01, ci_level=0.9
    )

    assert result.attrs == {
        "min_matches": 4,
        "alpha": 0.01,
        "ci_level": 0.9,
        "ci_method": "analytical",
    }


def test_the_bootstrap_carries_through_to_the_rows():
    frame = matched_frame()

    result = continuous_representativeness(
        frame, radii=(250.0,), ci_method="bootstrap", n_boot=500, random_state=3
    )
    fit = rma_regression(
        FOOTPRINT, TARGET, ci_method="bootstrap", n_boot=500, random_state=3
    )

    assert result.attrs["ci_method"] == "bootstrap"
    limits = (result.loc[0, "slope_lower"], result.loc[0, "slope_upper"])
    assert limits == pytest.approx(fit.slope_ci)


def test_the_series_of_a_bias_run_feeds_straight_in():
    """The two halves of Sect. 2.4 join without reshaping in between."""
    grid = np.arange(-45.0, 50.0, 10.0)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")

    def field(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, coords={"x": grid, "y": grid}, dims=("x", "y"))

    w = field(np.where((xx == 5.0) & (yy == 5.0), 1.0, 0.0))
    periods = {
        month: (w, field(np.full(xx.shape, 0.2 + 0.05 * index) + 0.01 * (xx > 0)))
        for index, month in enumerate(pd.date_range("2016-04-01", periods=7, freq="MS"))
    }

    bias = sensor_location_bias_series(periods, grid, grid, radii=(20.0, 40.0))
    result = continuous_representativeness(bias, radii=(20.0, 40.0))

    assert result["n"].tolist() == [7, 7]
    assert result["sufficient"].all()
    assert set(result["level"]) <= set(Level)


def test_something_other_than_a_frame_of_pairs_is_refused():
    with pytest.raises(ValueError, match="tidy DataFrame"):
        continuous_representativeness({"radius": [250.0]})


def test_a_frame_missing_a_required_column_is_refused():
    frame = matched_frame().drop(columns="value_target")

    with pytest.raises(KeyError, match="value_target"):
        continuous_representativeness(frame)


def test_empty_radii_raise():
    with pytest.raises(ValueError, match="radii holds no target areas"):
        continuous_representativeness(matched_frame(), radii=())


@pytest.mark.parametrize("radius", [0.0, -250.0, float("nan"), float("inf")])
def test_a_radius_that_is_not_a_disc_raises(radius):
    with pytest.raises(ValueError, match="positive and finite"):
        continuous_representativeness(matched_frame(), radii=(radius,))


def test_a_bar_below_a_fittable_regression_raises():
    with pytest.raises(ValueError, match="min_matches"):
        continuous_representativeness(matched_frame(), min_matches=2)


def test_the_default_radii_are_the_papers():
    result = continuous_representativeness(matched_frame())

    assert result["radius"].tolist() == [float(r) for r in TARGET_RADII]
