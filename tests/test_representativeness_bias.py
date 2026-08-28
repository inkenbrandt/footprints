"""
Sensor location bias tests for :mod:`fluxfootprints.representativeness`.

Eq. 6 of Chu et al. (2021) is a ratio of two averages this package computes
elsewhere, so the fixtures here are chosen so that both averages, and hence
the bias, can be written down by hand:

* a constant field averages to that constant under any weighting and over any
  disc, so its bias is exactly zero at every radius;
* the grid skips ``x = y = 0``, so the disc of radius 10 m holds exactly the
  four cells at ``(+/-5, +/-5)``. A footprint carrying all its weight on one
  of them, over a field whose four values average to exactly 1, puts the bias
  wherever the fixture wants it -- 1/16 and 1/8 here, both exact in binary, so
  the inclusive ``|delta| <= 0.10`` criterion is tested without floating-point
  luck at the boundary;
* a field that is zero over the disc leaves the ratio undefined, which must
  read as a missing flag rather than as a failed threshold, or the percentages
  of the paper's Fig. 7 would count gaps as biased site-months.

The series tests then check that mapping over periods only stacks these rows,
and that averaging ``within_threshold`` per radius -- the Fig. 7 reduction --
skips the undefined periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxfootprints.representativeness import (
    BIAS_THRESHOLD,
    TARGET_RADII,
    footprint_weighted_value,
    sensor_location_bias,
    sensor_location_bias_series,
    target_area_mask,
    target_area_value,
)

#: 20 x 20 cells of 10 m, centred on the tower and skipping x = y = 0.
GRID = np.arange(-95.0, 100.0, 10.0)
SIGMA = 50.0

XX, YY = np.meshgrid(GRID, GRID, indexing="ij")

#: Radius of the disc holding exactly the four cells at (+/-5, +/-5).
FOUR_CELL_RADIUS = 10.0

COLUMNS = ["radius", "value_footprint", "value_target", "delta", "within_threshold"]


# ----------------------------
# Fixtures
# ----------------------------
def field(values: np.ndarray, name: str = "field") -> xr.DataArray:
    """Wrap a (x, y) array as a raster on the tower-centred grid."""
    return xr.DataArray(
        np.asarray(values, dtype=float),
        coords={"x": GRID, "y": GRID},
        dims=("x", "y"),
        name=name,
    )


def weights(sigma_x: float = SIGMA, sigma_y: float = SIGMA, x0: float = 0.0):
    """A Gaussian footprint on the grid, renormalised to sum to 1."""
    w = np.exp(-0.5 * (((XX - x0) / sigma_x) ** 2 + (YY / sigma_y) ** 2))
    return field(w / w.sum(), name="fclim")


def point_weights(x0: float = 5.0, y0: float = 5.0) -> xr.DataArray:
    """All of the footprint weight on the single cell centred on (x0, y0)."""
    w = np.where((XX == x0) & (YY == y0), 1.0, 0.0)
    return field(w, name="fclim")


def constant(value: float = 7.5) -> xr.DataArray:
    return field(np.full(XX.shape, value))


def four_cell_field(offset: float) -> xr.DataArray:
    """
    A field averaging to exactly 1 over the four-cell disc.

    ``(5, 5)`` carries ``1 + offset`` and ``(-5, -5)`` carries ``1 - offset``,
    so a footprint sitting on the first sees a bias of exactly `offset`.
    """
    values = np.ones(XX.shape)
    values[(XX == 5.0) & (YY == 5.0)] = 1.0 + offset
    values[(XX == -5.0) & (YY == -5.0)] = 1.0 - offset
    return field(values)


# ----------------------------
# sensor_location_bias, Eq. 6
# ----------------------------
def test_constant_field_has_no_bias_at_any_radius():
    result = sensor_location_bias(weights(), constant(7.5), GRID, GRID)

    assert list(result.columns) == COLUMNS
    assert result["radius"].tolist() == [float(r) for r in TARGET_RADII]
    assert result["value_footprint"].to_numpy() == pytest.approx(7.5)
    assert result["value_target"].to_numpy() == pytest.approx(7.5)
    assert result["delta"].to_numpy() == pytest.approx(0.0)
    assert result["within_threshold"].all()


def test_delta_matches_equation_six_computed_by_hand():
    """A footprint and a field with no symmetry to lean on."""
    w = weights(sigma_x=35.0, sigma_y=70.0, x0=25.0)
    ramp = field(2.0 + XX / 100.0 + YY / 200.0)

    result = sensor_location_bias(w, ramp, GRID, GRID, radii=(50.0, 150.0))

    expected_footprint = float((w * ramp).sum())
    for row in result.itertuples():
        inside = target_area_mask(GRID, GRID, row.radius).values
        expected_target = float(ramp.values[inside].mean())

        assert row.value_footprint == pytest.approx(expected_footprint)
        assert row.value_target == pytest.approx(expected_target)
        assert row.delta == pytest.approx(
            (expected_footprint - expected_target) / expected_target
        )


def test_the_two_averages_are_those_of_the_building_blocks():
    w = weights(x0=15.0)
    ramp = field(2.0 + XX / 100.0 + YY / 200.0)

    result = sensor_location_bias(w, ramp, GRID, GRID, radii=(75.0,))
    row = result.iloc[0]

    assert row["value_footprint"] == pytest.approx(
        footprint_weighted_value(w, ramp).value
    )
    assert row["value_target"] == pytest.approx(
        target_area_value(ramp, GRID, GRID, 75.0).value
    )


def test_radii_are_reported_in_the_order_given():
    """The frame follows the caller's order rather than sorting itself."""
    result = sensor_location_bias(
        weights(), constant(), GRID, GRID, radii=(150.0, 50.0, 100.0)
    )

    assert result["radius"].tolist() == [150.0, 50.0, 100.0]
    assert result.index.tolist() == [0, 1, 2]


def test_the_footprint_value_does_not_move_with_the_radius():
    """Only the target area changes with the radius, so Eq. 5 is computed once."""
    half = field(np.where(XX < 0.0, 0.0, 1.0))

    result = sensor_location_bias(weights(x0=20.0), half, GRID, GRID)

    assert result["value_footprint"].nunique() == 1
    assert result["value_footprint"].iloc[0] == pytest.approx(
        footprint_weighted_value(weights(x0=20.0), half).value
    )


def test_a_footprint_on_higher_ground_than_its_surroundings_biases_positive():
    """The sign of the paper's result: EVI_footprint > EVI_target."""
    core = field(np.where(np.hypot(XX, YY) <= 40.0, 2.0, 1.0))

    result = sensor_location_bias(weights(sigma_x=20.0, sigma_y=20.0), core, GRID, GRID)

    assert (result["delta"] > 0.0).all()


# ----------------------------
# The |delta| <= 0.10 criterion
# ----------------------------
def test_the_threshold_is_the_papers_ten_percent():
    assert BIAS_THRESHOLD == 0.10


def test_a_bias_inside_the_threshold_passes():
    """1/16, exact in binary, against a target area averaging to exactly 1."""
    result = sensor_location_bias(
        point_weights(),
        four_cell_field(0.0625),
        GRID,
        GRID,
        radii=(FOUR_CELL_RADIUS,),
    )

    assert result["value_footprint"].iloc[0] == 1.0625
    assert result["value_target"].iloc[0] == 1.0
    assert result["delta"].iloc[0] == 0.0625
    assert bool(result["within_threshold"].iloc[0]) is True


def test_a_bias_outside_the_threshold_fails():
    result = sensor_location_bias(
        point_weights(),
        four_cell_field(0.125),
        GRID,
        GRID,
        radii=(FOUR_CELL_RADIUS,),
    )

    assert result["delta"].iloc[0] == 0.125
    assert bool(result["within_threshold"].iloc[0]) is False


def test_a_negative_bias_is_judged_on_its_magnitude():
    """The criterion is |delta|, so the footprint may sit either side."""
    result = sensor_location_bias(
        point_weights(x0=-5.0, y0=-5.0),
        four_cell_field(0.0625),
        GRID,
        GRID,
        radii=(FOUR_CELL_RADIUS,),
    )

    assert result["delta"].iloc[0] == -0.0625
    assert bool(result["within_threshold"].iloc[0]) is True


def test_the_flag_follows_the_reported_delta():
    """A near target area the footprint matches, and a far one it does not."""
    w = weights(sigma_x=10.0, sigma_y=10.0)
    core = field(np.where(np.hypot(XX, YY) <= 40.0, 3.0, 1.0))

    result = sensor_location_bias(w, core, GRID, GRID, radii=(20.0, 40.0, 95.0))

    expected = result["delta"].abs() <= BIAS_THRESHOLD
    assert result["within_threshold"].astype(bool).tolist() == expected.tolist()
    assert result["within_threshold"].any()
    assert not result["within_threshold"].all()


def test_the_threshold_is_recorded_on_the_frame():
    result = sensor_location_bias(weights(), constant(), GRID, GRID)

    assert result.attrs["bias_threshold"] == BIAS_THRESHOLD


# ----------------------------
# Undefined biases
# ----------------------------
def test_a_zero_target_area_leaves_the_bias_undefined():
    """Eq. 6 divides by EVI_target; a flag of False would claim to know more."""
    result = sensor_location_bias(weights(), constant(0.0), GRID, GRID, radii=(50.0,))

    assert np.isnan(result["delta"].iloc[0])
    assert result["within_threshold"].isna().all()


def test_a_field_of_nodata_leaves_the_bias_undefined():
    result = sensor_location_bias(
        weights(), constant(np.nan), GRID, GRID, radii=(50.0,)
    )

    assert np.isnan(result["value_footprint"].iloc[0])
    assert np.isnan(result["value_target"].iloc[0])
    assert result["within_threshold"].isna().all()


def test_empty_radii_raise():
    with pytest.raises(ValueError, match="no target areas"):
        sensor_location_bias(weights(), constant(), GRID, GRID, radii=())


def test_a_raster_off_the_grid_raises():
    with pytest.raises(ValueError, match="different grids"):
        sensor_location_bias(
            weights(), constant().isel(x=slice(0, 5)), GRID, GRID, radii=(50.0,)
        )


# ----------------------------
# sensor_location_bias_series
# ----------------------------
def periods() -> dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]:
    """Three site-months: one within the threshold, one outside, one undefined."""
    w = point_weights()
    return {
        pd.Timestamp("2020-07-01"): (w, four_cell_field(0.0625)),
        pd.Timestamp("2020-06-01"): (w, four_cell_field(0.125)),
        pd.Timestamp("2020-08-01"): (w, constant(0.0)),
    }


def test_the_series_stacks_one_frame_per_period_and_radius():
    radii = (FOUR_CELL_RADIUS, 50.0)

    result = sensor_location_bias_series(periods(), GRID, GRID, radii=radii)

    assert list(result.columns) == ["time", *COLUMNS]
    assert len(result) == len(periods()) * len(radii)
    assert result.index.tolist() == list(range(len(result)))
    assert result.attrs["bias_threshold"] == BIAS_THRESHOLD


def test_the_rows_of_a_period_are_the_single_period_frame():
    radii = (FOUR_CELL_RADIUS, 50.0)
    month = pd.Timestamp("2020-07-01")
    w, raster = periods()[month]

    result = sensor_location_bias_series(periods(), GRID, GRID, radii=radii)
    rows = result[result["time"] == month].drop(columns="time").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        rows, sensor_location_bias(w, raster, GRID, GRID, radii=radii)
    )


def test_periods_keep_the_order_they_were_given():
    """Site-months are evaluated as presented, not sorted."""
    result = sensor_location_bias_series(
        periods(), GRID, GRID, radii=(FOUR_CELL_RADIUS,)
    )

    assert result["time"].tolist() == list(periods())


def test_the_fraction_within_the_threshold_skips_undefined_periods():
    """The Fig. 7 reduction: one of the two defined site-months is within."""
    result = sensor_location_bias_series(
        periods(), GRID, GRID, radii=(FOUR_CELL_RADIUS,)
    )

    within = result.groupby("radius")["within_threshold"]

    assert within.count().loc[FOUR_CELL_RADIUS] == 2
    assert within.mean().loc[FOUR_CELL_RADIUS] == pytest.approx(0.5)


def test_the_three_input_forms_agree():
    mapping = periods()
    as_series = pd.Series(list(mapping.values()), index=list(mapping))
    as_triples = [(time, w, raster) for time, (w, raster) in mapping.items()]

    from_mapping = sensor_location_bias_series(mapping, GRID, GRID, radii=(50.0,))
    from_series = sensor_location_bias_series(as_series, GRID, GRID, radii=(50.0,))
    from_triples = sensor_location_bias_series(as_triples, GRID, GRID, radii=(50.0,))

    pd.testing.assert_frame_equal(from_mapping, from_series)
    pd.testing.assert_frame_equal(from_mapping, from_triples)


def test_labels_need_not_be_timestamps():
    result = sensor_location_bias_series(
        {"Jan": (weights(), constant()), "Feb": (weights(), constant())},
        GRID,
        GRID,
        radii=(50.0,),
    )

    assert result["time"].tolist() == ["Jan", "Feb"]


def test_empty_pairs_raise():
    with pytest.raises(ValueError, match="no periods"):
        sensor_location_bias_series({}, GRID, GRID)


def test_a_bare_pair_without_a_time_label_is_refused():
    with pytest.raises(TypeError, match="carries no time label"):
        sensor_location_bias_series([(weights(), constant())], GRID, GRID)


def test_a_mapping_value_that_is_not_a_pair_is_refused():
    with pytest.raises(TypeError, match="climatology, raster"):
        sensor_location_bias_series({"Jan": weights()}, GRID, GRID)


def test_something_that_is_not_a_collection_of_periods_is_refused():
    with pytest.raises(TypeError, match="mapping of time"):
        sensor_location_bias_series(7, GRID, GRID)


def test_a_failing_period_names_the_time_it_failed_at():
    bad = {pd.Timestamp("2020-06-01"): (weights(), constant().isel(x=slice(0, 5)))}

    with pytest.raises(ValueError, match="At time Timestamp"):
        sensor_location_bias_series(bad, GRID, GRID, radii=(50.0,))
