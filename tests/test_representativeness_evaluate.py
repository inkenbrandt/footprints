"""
Site-level evaluation tests for :mod:`fluxfootprints.representativeness`.

Covers the three functions that summarise one site: :func:`evaluate_landcover`,
:func:`evaluate_vegetation_index`, and :func:`representativeness_summary`.
They sit between the per-comparison primitives and the whole-record driver,
:func:`assess_representativeness`, so the tests check three things beyond the
obvious shapes:

* the dataclasses are filled in the paper's units -- ``CategoricalResult``
  reports percentages where ``categorical_representativeness`` reports
  fractions, and ``ContinuousResult.bias`` is as long as the ``n`` it was
  fitted on;
* a climatology that has already been truncated is rescaled rather than cut a
  second time, so a slice of ``monthly_climatologies`` can be passed straight
  in and gives the same answer as the raw climatology it came from;
* the numbers agree with what the driver reports for the same inputs, since
  the two paths are meant to be different front doors onto one method.

The synthetic site is the one the driver tests use: a footprint sitting inside
a 300 m disc of a single land-cover class, and a vegetation index peaking on
the tower, so the expected verdicts can be reasoned about rather than recorded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxfootprints.representativeness import (
    SUMMARY_COLUMNS,
    CategoricalResult,
    ClimatologyMetrics,
    ContinuousResult,
    Level,
    assess_representativeness,
    categorical_representativeness,
    climatology_metrics,
    continuous_representativeness,
    evaluate_landcover,
    evaluate_vegetation_index,
    monthly_climatologies,
    representativeness_summary,
    sensor_location_bias,
    truncate_to_contour,
)

STATION_LAT = 40.0
STATION_LON = -111.9
TZ = -7

GRID = np.arange(-495.0, 500.0, 10.0)
XX, YY = np.meshgrid(GRID, GRID, indexing="ij")
RADIUS = np.hypot(XX, YY)

RADII = (250.0, 500.0, 1000.0)

NEAR_CLASS = 41
FAR_CLASS = 81
N_PAIRS = 6


def _field(values: np.ndarray, name: str) -> xr.DataArray:
    """Wrap an (x, y) array as a raster on the tower-centred grid."""
    return xr.DataArray(
        np.asarray(values, dtype=float),
        coords={"x": GRID, "y": GRID},
        dims=("x", "y"),
        name=name,
    )


def _density(sigma_x: float, sigma_y: float, y0: float) -> xr.DataArray:
    """A Gaussian climatology as a source-weight density [m-2] on 10 m cells."""
    w = np.exp(-0.5 * ((XX / sigma_x) ** 2 + ((YY - y0) / sigma_y) ** 2))
    return _field(w / (w.sum() * 100.0), "fclim")


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture(scope="module")
def fclim() -> xr.DataArray:
    """One raw climatology, its source area well inside the near class."""
    return _density(90.0, 70.0, 60.0)


@pytest.fixture(scope="module")
def landcover() -> xr.DataArray:
    """One class within 300 m of the tower, another beyond it."""
    return _field(
        np.where(RADIUS <= 300.0, float(NEAR_CLASS), float(FAR_CLASS)), "landcover"
    )


@pytest.fixture(scope="module")
def pairs() -> tuple[list[xr.DataArray], list[xr.DataArray]]:
    """Six matched climatology / vegetation-index periods."""
    climatologies, fields = [], []
    for index in range(N_PAIRS):
        climatologies.append(_density(85.0 + 5.0 * index, 70.0, 50.0 + 8.0 * index))
        amplitude = 0.20 + 0.03 * index
        fields.append(
            _field(0.2 + amplitude * np.exp(-0.5 * (RADIUS / 400.0) ** 2), "EVI")
        )
    return climatologies, fields


@pytest.fixture(scope="module")
def categorical(fclim, landcover) -> list[CategoricalResult]:
    return evaluate_landcover(fclim, landcover, radii=RADII)


@pytest.fixture(scope="module")
def continuous(pairs) -> list[ContinuousResult]:
    return evaluate_vegetation_index(*pairs, radii=RADII)


@pytest.fixture(scope="module")
def metrics(fclim) -> ClimatologyMetrics:
    return climatology_metrics(fclim, seasonal_overlap=0.72, daynight_overlap=0.95)


# ============================
# evaluate_landcover
# ============================
def test_one_result_per_radius_in_order(categorical):
    assert [entry.radius for entry in categorical] == list(RADII)
    assert all(isinstance(entry, CategoricalResult) for entry in categorical)


def test_shares_are_percentages_not_fractions(fclim, landcover, categorical):
    frame = categorical_representativeness(
        truncate_to_contour(fclim), landcover, GRID, GRID, radii=RADII
    )
    assert [entry.p_footprint for entry in categorical] == pytest.approx(
        100.0 * frame["p_footprint"].to_numpy()
    )
    assert [entry.p_target for entry in categorical] == pytest.approx(
        100.0 * frame["p_target"].to_numpy()
    )


def test_compositions_are_percentages_summing_to_a_hundred(categorical):
    for entry in categorical:
        assert sum(entry.footprint_composition.values()) == pytest.approx(100.0)
        assert sum(entry.target_composition.values()) == pytest.approx(100.0)


def test_the_footprint_composition_is_the_same_at_every_radius(categorical):
    first = categorical[0].footprint_composition
    for entry in categorical[1:]:
        assert entry.footprint_composition == first
        assert entry.dominant_class == categorical[0].dominant_class
        assert entry.p_footprint == pytest.approx(categorical[0].p_footprint)


def test_the_target_composition_picks_up_the_far_class(categorical):
    # The 250 m disc is wholly inside the near class; the wider ones are not.
    assert set(categorical[0].target_composition) == {NEAR_CLASS}
    assert set(categorical[-1].target_composition) == {NEAR_CLASS, FAR_CLASS}
    assert categorical[-1].target_composition[NEAR_CLASS] == pytest.approx(
        categorical[-1].p_target
    )


def test_the_level_is_an_enum_member(categorical):
    assert all(isinstance(entry.level, Level) for entry in categorical)
    assert categorical[0].level is Level.HIGH
    assert categorical[-1].level is Level.LOW


def test_a_truncated_climatology_is_not_cut_twice(fclim, landcover, categorical):
    already = truncate_to_contour(fclim)
    again = evaluate_landcover(already, landcover, radii=RADII)

    assert [entry.p_footprint for entry in again] == pytest.approx(
        [entry.p_footprint for entry in categorical]
    )
    assert [entry.chi2 for entry in again] == pytest.approx(
        [entry.chi2 for entry in categorical]
    )


def test_a_monthly_climatology_slice_is_accepted(landcover):
    times = pd.date_range("2020-06-01", "2020-06-30 23:00", freq="6h")
    stack = xr.DataArray(
        np.repeat(_density(90.0, 70.0, 60.0).values[None], len(times), axis=0),
        dims=("time", "x", "y"),
        coords={"time": times, "x": GRID, "y": GRID},
    )
    clim = monthly_climatologies(
        stack, latitude=STATION_LAT, longitude=STATION_LON, tz=TZ
    )
    weights = clim.footprint_climatology.isel(month=0, period=0)

    results = evaluate_landcover(weights, landcover, radii=RADII)
    assert [entry.radius for entry in results] == list(RADII)
    assert results[0].p_footprint == pytest.approx(100.0)


def test_the_fraction_argument_changes_the_source_area(fclim):
    # A tight patch of the near class, so that widening the contour pushes the
    # source area out of it and the footprint composition actually moves.
    patchy = _field(
        np.where(RADIUS <= 120.0, float(NEAR_CLASS), float(FAR_CLASS)), "landcover"
    )
    wide = evaluate_landcover(fclim, patchy, radii=RADII, fraction=0.95)
    narrow = evaluate_landcover(fclim, patchy, radii=RADII, fraction=0.5)

    assert narrow[0].p_footprint > wide[0].p_footprint
    assert narrow[0].footprint_composition != wide[0].footprint_composition


def test_a_climatology_without_coordinates_raises(fclim, landcover):
    bare = xr.DataArray(fclim.values, dims=("x", "y"))
    with pytest.raises(ValueError, match="no 'x' coordinate"):
        evaluate_landcover(bare, landcover, radii=RADII)


def test_a_mismatched_grid_raises(fclim):
    coarse = xr.DataArray(
        np.zeros((10, 10)),
        coords={"x": np.arange(10.0), "y": np.arange(10.0)},
        dims=("x", "y"),
    )
    with pytest.raises(ValueError):
        evaluate_landcover(fclim, coarse, radii=RADII)


def test_empty_radii_raise(fclim, landcover):
    with pytest.raises(ValueError, match="no target areas"):
        evaluate_landcover(fclim, landcover, radii=())


def test_a_raster_of_only_nodata_raises(fclim):
    empty = _field(np.full(XX.shape, np.nan), "landcover")
    with pytest.raises(ValueError, match="No cell carries both"):
        evaluate_landcover(fclim, empty, radii=RADII)


def test_it_agrees_with_the_driver(fclim, landcover):
    times = pd.date_range("2020-06-01", "2020-06-30 23:00", freq="6h")
    stack = xr.DataArray(
        np.repeat(fclim.values[None], len(times), axis=0),
        dims=("time", "x", "y"),
        coords={"time": times, "x": GRID, "y": GRID},
    )
    results = assess_representativeness(
        stack,
        station_lat=STATION_LAT,
        station_lon=STATION_LON,
        landcover=landcover,
        radii=RADII,
        tz=TZ,
    )
    driver = (
        results[(results["scope"] == "site") & (results["kind"] == "categorical")]
        .reset_index()
        .query("period == 'daytime'")
        .sort_values("radius")
    )
    direct = evaluate_landcover(fclim, landcover, radii=RADII)

    # The driver reports fractions, the dataclass percentages.
    assert driver["value_footprint"].to_numpy() == pytest.approx(
        [entry.p_footprint / 100.0 for entry in direct]
    )
    assert driver["value_target"].to_numpy() == pytest.approx(
        [entry.p_target / 100.0 for entry in direct]
    )
    assert driver["chi2"].to_numpy() == pytest.approx(
        [entry.chi2 for entry in direct]
    )
    assert driver["level"].tolist() == [entry.level for entry in direct]


# ============================
# evaluate_vegetation_index
# ============================
def test_one_fit_per_radius_in_order(continuous):
    assert [entry.radius for entry in continuous] == list(RADII)
    assert all(isinstance(entry, ContinuousResult) for entry in continuous)


def test_every_pair_enters_every_radius(continuous):
    assert all(entry.n == N_PAIRS for entry in continuous)
    assert all(len(entry.bias) == entry.n for entry in continuous)


def test_the_bias_array_holds_the_per_period_deltas(pairs, continuous):
    climatologies, fields = pairs
    expected = [
        sensor_location_bias(
            truncate_to_contour(climatology), field, GRID, GRID, radii=RADII
        )
        for climatology, field in zip(climatologies, fields)
    ]
    for index, entry in enumerate(continuous):
        assert entry.bias == pytest.approx(
            [frame["delta"].to_numpy()[index] for frame in expected]
        )


def test_the_regression_matches_continuous_representativeness(pairs, continuous):
    climatologies, fields = pairs
    frames = []
    for index, (climatology, field) in enumerate(zip(climatologies, fields)):
        frame = sensor_location_bias(
            truncate_to_contour(climatology), field, GRID, GRID, radii=RADII
        )
        frame.insert(0, "time", index)
        frames.append(frame)
    expected = continuous_representativeness(
        pd.concat(frames, ignore_index=True), radii=RADII, min_matches=3
    )

    for column, attribute in (
        ("slope", "slope"),
        ("intercept", "intercept"),
        ("r_squared", "r_squared"),
        ("rmse", "rmse"),
        ("mae", "mae"),
        ("p_value", "p_value"),
    ):
        assert [getattr(entry, attribute) for entry in continuous] == pytest.approx(
            expected[column].to_numpy()
        )


def test_within_threshold_is_the_share_of_periods_inside_the_band(continuous):
    for entry in continuous:
        expected = np.mean(np.abs(entry.bias) <= 0.10)
        assert entry.within_threshold == pytest.approx(expected)


def test_the_bias_threshold_is_honoured(pairs, continuous):
    lenient = evaluate_vegetation_index(*pairs, radii=RADII, bias_threshold=0.5)
    assert continuous[-1].within_threshold < 1.0
    assert lenient[-1].within_threshold == pytest.approx(1.0)
    # The threshold changes only the flag count, not the fit.
    assert [entry.slope for entry in lenient] == pytest.approx(
        [entry.slope for entry in continuous]
    )


def test_the_footprint_sees_higher_values_so_the_slope_falls(continuous):
    assert all(entry.slope < 1.0 for entry in continuous)
    slopes = [entry.slope for entry in continuous]
    assert slopes == sorted(slopes, reverse=True)
    assert all((entry.bias > 0.0).all() for entry in continuous)


def test_mismatched_sequence_lengths_raise(pairs):
    climatologies, fields = pairs
    with pytest.raises(ValueError, match="matched element for element"):
        evaluate_vegetation_index(climatologies, fields[:-1], radii=RADII)


def test_too_few_pairs_raise(pairs):
    climatologies, fields = pairs
    with pytest.raises(ValueError, match="at least three matched pairs"):
        evaluate_vegetation_index(climatologies[:2], fields[:2], radii=RADII)


def test_a_bad_pair_names_its_position(pairs):
    climatologies, fields = pairs
    broken = list(fields)
    broken[2] = xr.DataArray(
        np.zeros((10, 10)),
        coords={"x": np.arange(10.0), "y": np.arange(10.0)},
        dims=("x", "y"),
    )
    with pytest.raises(ValueError, match="At pair 2"):
        evaluate_vegetation_index(climatologies, broken, radii=RADII)


def test_truncated_climatologies_are_accepted(pairs, continuous):
    climatologies, fields = pairs
    already = [truncate_to_contour(entry) for entry in climatologies]
    again = evaluate_vegetation_index(already, fields, radii=RADII)
    assert [entry.slope for entry in again] == pytest.approx(
        [entry.slope for entry in continuous]
    )


def test_a_radius_with_too_few_finite_pairs_gets_no_level():
    # A hole in the scene swallows the 250 m disc for half the periods, so
    # that radius keeps only two matches while the wider ones keep four.
    climatologies, fields = [], []
    for index in range(4):
        climatologies.append(_density(90.0, 70.0, 420.0 + 10.0 * index))
        values = 0.2 + (0.20 + 0.03 * index) * np.exp(-0.5 * (RADIUS / 400.0) ** 2)
        if index >= 2:
            values = np.where(RADIUS <= 260.0, np.nan, values)
        fields.append(_field(values, "EVI"))

    results = evaluate_vegetation_index(
        climatologies, fields, radii=(250.0, 1000.0)
    )
    near, far = results

    assert near.n == 2
    assert near.level is None
    assert np.isnan(near.slope)
    assert len(near.bias) == 2

    assert far.n == 4
    assert isinstance(far.level, Level)


# ============================
# representativeness_summary
# ============================
def test_the_summary_carries_its_documented_columns(categorical, continuous, metrics):
    table = representativeness_summary(
        categorical, continuous, metrics, site_id="US-Tst"
    )
    assert tuple(table.columns) == SUMMARY_COLUMNS
    assert isinstance(table.index, pd.RangeIndex)
    assert len(table) == len(RADII)
    assert table["radius"].tolist() == list(RADII)


def test_both_halves_land_on_the_same_row(categorical, continuous, metrics):
    table = representativeness_summary(categorical, continuous, metrics).set_index(
        "radius"
    )
    for entry in categorical:
        assert table.loc[entry.radius, "p_footprint"] == pytest.approx(
            entry.p_footprint
        )
        assert table.loc[entry.radius, "chi2"] == pytest.approx(entry.chi2)
    for entry in continuous:
        assert table.loc[entry.radius, "slope"] == pytest.approx(entry.slope)
        assert table.loc[entry.radius, "n"] == entry.n


def test_the_bias_columns_summarise_the_per_period_deltas(continuous):
    table = representativeness_summary(continuous=continuous).set_index("radius")
    for entry in continuous:
        assert table.loc[entry.radius, "mean_bias"] == pytest.approx(
            np.mean(entry.bias)
        )
        assert table.loc[entry.radius, "median_bias"] == pytest.approx(
            np.median(entry.bias)
        )
        assert table.loc[entry.radius, "bias_within_threshold"] == pytest.approx(
            entry.within_threshold
        )


def test_the_levels_are_plain_strings_that_survive_a_csv(
    categorical, continuous, tmp_path
):
    table = representativeness_summary(categorical, continuous)
    assert table["landcover_level"].tolist() == [
        entry.level.value for entry in categorical
    ]
    assert table["continuous_level"].tolist() == [
        entry.level.value for entry in continuous
    ]
    # str(Level.HIGH) is "Level.HIGH"; the column must not carry that.
    path = tmp_path / "summary.csv"
    table.to_csv(path, index=False)
    reread = pd.read_csv(path)
    assert reread["landcover_level"].isin(["high", "medium", "low"]).all()
    assert reread["continuous_level"].isin(["high", "medium", "low"]).all()
    # They still compare equal to the members.
    assert (table["landcover_level"] == Level.HIGH).any()


def test_the_metrics_are_repeated_down_the_frame(categorical, continuous, metrics):
    table = representativeness_summary(categorical, continuous, metrics)
    assert table["fetch"].to_numpy() == pytest.approx(metrics.fetch)
    assert table["symmetry"].to_numpy() == pytest.approx(metrics.symmetry)
    assert table["seasonal_overlap"].to_numpy() == pytest.approx(0.72)
    assert table["daynight_overlap"].to_numpy() == pytest.approx(0.95)
    assert (table["n_cells"] == metrics.n_cells).all()


def test_absent_overlaps_come_through_as_missing(categorical, fclim):
    table = representativeness_summary(categorical, metrics=climatology_metrics(fclim))
    assert table["seasonal_overlap"].isna().all()
    assert table["daynight_overlap"].isna().all()
    assert table["fetch"].notna().all()


def test_omitted_metrics_leave_those_columns_missing(categorical):
    table = representativeness_summary(categorical)
    for column in ("fetch", "area", "symmetry", "contour_level"):
        assert table[column].isna().all()


def test_each_half_alone_leaves_the_other_missing(categorical, continuous):
    only_landcover = representativeness_summary(categorical=categorical)
    assert only_landcover["chi2"].notna().all()
    assert only_landcover["slope"].isna().all()
    assert only_landcover["continuous_level"].isna().all()

    only_continuous = representativeness_summary(continuous=continuous)
    assert only_continuous["slope"].notna().all()
    assert only_continuous["chi2"].isna().all()
    assert only_continuous["landcover_level"].isna().all()


def test_the_site_column_is_always_present(categorical):
    labelled = representativeness_summary(categorical, site_id="US-Tst")
    assert (labelled["site_id"] == "US-Tst").all()

    unlabelled = representativeness_summary(categorical)
    assert "site_id" in unlabelled.columns
    assert unlabelled["site_id"].isna().all()


def test_radii_present_in_only_one_half_still_get_a_row(pairs, categorical):
    other = evaluate_vegetation_index(*pairs, radii=(250.0, 3000.0))
    table = representativeness_summary(categorical, other)

    assert table["radius"].tolist() == [250.0, 500.0, 1000.0, 3000.0]
    assert table["slope"].isna().tolist() == [False, True, True, False]
    assert table["chi2"].isna().tolist() == [False, False, False, True]


def test_counts_stay_nullable_integers(categorical, continuous, metrics):
    table = representativeness_summary(categorical, continuous, metrics)
    for column in ("dof", "n", "n_cells"):
        assert table[column].dtype == pd.Int64Dtype()

    partial = representativeness_summary(categorical=categorical)
    assert partial["n"].isna().all()
    assert partial["n"].dtype == pd.Int64Dtype()


def test_two_empty_halves_raise():
    with pytest.raises(ValueError, match="no radius to build"):
        representativeness_summary()
    with pytest.raises(ValueError, match="no radius to build"):
        representativeness_summary(categorical=[], continuous=[])


def test_a_repeated_radius_raises(categorical, continuous):
    with pytest.raises(ValueError, match="two results for the 250 m"):
        representativeness_summary(categorical=list(categorical) + [categorical[0]])
    with pytest.raises(ValueError, match="two results for the 250 m"):
        representativeness_summary(continuous=list(continuous) + [continuous[0]])


def test_summaries_concatenate_across_sites(categorical, continuous, metrics):
    pooled = pd.concat(
        [
            representativeness_summary(categorical, continuous, metrics, site_id=site)
            for site in ("US-Tst", "US-Oth")
        ],
        ignore_index=True,
    )
    assert len(pooled) == 2 * len(RADII)
    assert set(pooled["site_id"]) == {"US-Tst", "US-Oth"}
    assert tuple(pooled.columns) == SUMMARY_COLUMNS


# ============================
# The removed driver
# ============================
def test_the_old_stub_driver_is_gone():
    import fluxfootprints
    from fluxfootprints import representativeness

    assert not hasattr(fluxfootprints, "evaluate_representativeness")
    assert not hasattr(representativeness, "evaluate_representativeness")
    assert "evaluate_representativeness" not in representativeness.__all__
    assert "evaluate_representativeness" not in fluxfootprints.__all__


def test_the_replacement_driver_is_exported():
    import fluxfootprints

    assert "assess_representativeness" in fluxfootprints.__all__
    assert callable(fluxfootprints.assess_representativeness)
