"""
Driver tests for :func:`fluxfootprints.assess_representativeness`.

The driver's job is composition, not arithmetic: every number it reports is
produced by a function tested elsewhere in this suite. These tests therefore
check that

* the tidy frame has the shape and index the schema promises, and that the row
  counts follow from the months, periods, radii, and scenes supplied;
* each block of rows reproduces, value for value, what calling the underlying
  function directly would give -- ``sensor_location_bias`` for the per-period
  continuous rows, ``categorical_representativeness`` for the land-cover ones,
  ``seasonal_overlap`` and ``daynight_overlap`` for the site-year ones, and
  ``continuous_representativeness`` for the site-level regressions;
* each analysis is genuinely optional, and the four accepted forms of the
  continuous input agree;
* the published-schema tables carry the documented columns and the paper's
  units, and the writer puts them on disk.

The synthetic site is built so the expected classifications can be reasoned
about rather than recorded: the footprint sits inside a 300 m disc of one
land-cover class, so it is homogeneous at 250 m and increasingly unrepresentative
beyond it, and the vegetation index peaks on the tower, so the footprint-weighted
value exceeds the target-area mean at every radius, as it did at every site in
the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxfootprints.representativeness import (
    CATEGORICAL_KIND,
    CHU_TABLE_COLUMNS,
    CLIMATOLOGY_KIND,
    CONTINUOUS_KIND,
    PERIOD_SCOPE,
    RESULT_COLUMNS,
    RESULT_INDEX,
    SITE_SCOPE,
    SITE_YEAR_SCOPE,
    Level,
    assess_representativeness,
    categorical_representativeness,
    continuous_representativeness,
    daynight_overlap,
    export_representativeness_tables,
    monthly_climatologies,
    representativeness_table,
    seasonal_overlap,
    sensor_location_bias,
)

STATION_LAT = 40.0
STATION_LON = -111.9
TZ = -7

#: 100 x 100 cells of 10 m, centred on the tower.
GRID = np.arange(-495.0, 500.0, 10.0)
XX, YY = np.meshgrid(GRID, GRID, indexing="ij")
RADIUS = np.hypot(XX, YY)

RADII = (250.0, 500.0, 1000.0)

#: Scene months, one Landsat-like retrieval in each.
SCENE_MONTHS = (2, 5, 8, 11)

NEAR_CLASS = 41
FAR_CLASS = 81


# ----------------------------
# Fixtures
# ----------------------------
def _grid_field(values: np.ndarray, name: str) -> xr.DataArray:
    """Wrap an (x, y) array as a raster on the tower-centred grid."""
    return xr.DataArray(
        np.asarray(values, dtype=float),
        coords={"x": GRID, "y": GRID},
        dims=("x", "y"),
        name=name,
    )


@pytest.fixture(scope="module")
def footprints() -> xr.DataArray:
    """A year of six-hourly footprints whose centroid drifts with the season."""
    times = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="6h")
    rng = np.random.default_rng(0)

    stack = []
    for stamp in times:
        y0 = 60.0 + 40.0 * np.sin(2.0 * np.pi * stamp.dayofyear / 365.0)
        sigma_x = 80.0 + 20.0 * rng.random()
        sigma_y = 60.0 + 20.0 * rng.random()
        w = np.exp(-0.5 * ((XX / sigma_x) ** 2 + ((YY - y0) / sigma_y) ** 2))
        # A density [m-2] on 10 m cells, as the models produce.
        stack.append(w / (w.sum() * 100.0))

    return xr.DataArray(
        np.stack(stack),
        dims=("time", "x", "y"),
        coords={"time": times, "x": GRID, "y": GRID},
        name="f_2d",
    )


@pytest.fixture(scope="module")
def landcover() -> xr.DataArray:
    """One class within 300 m of the tower, another beyond it."""
    return _grid_field(
        np.where(RADIUS <= 300.0, float(NEAR_CLASS), float(FAR_CLASS)), "landcover"
    )


@pytest.fixture(scope="module")
def scenes() -> dict[pd.Timestamp, xr.DataArray]:
    """Four vegetation-index scenes, each peaking on the tower."""
    fields = {}
    for month in SCENE_MONTHS:
        # Positive at every scene, so the tower always sits on a local maximum
        # and the footprint-weighted value exceeds the target-area mean.
        amplitude = 0.20 + 0.05 * month
        fields[pd.Timestamp(f"2020-{month:02d}-15")] = _grid_field(
            0.2 + amplitude * np.exp(-0.5 * (RADIUS / 400.0) ** 2), "EVI"
        )
    return fields


@pytest.fixture(scope="module")
def climatology(footprints: xr.DataArray) -> xr.Dataset:
    """The monthly climatologies the driver would otherwise build itself."""
    return monthly_climatologies(
        footprints, latitude=STATION_LAT, longitude=STATION_LON, tz=TZ
    )


@pytest.fixture(scope="module")
def results(
    footprints: xr.DataArray,
    landcover: xr.DataArray,
    scenes: dict[pd.Timestamp, xr.DataArray],
) -> pd.DataFrame:
    """A full analysis: both fields, three radii, one site-year."""
    return assess_representativeness(
        footprints,
        station_lat=STATION_LAT,
        station_lon=STATION_LON,
        site_id="US-Tst",
        landcover=landcover,
        continuous=scenes,
        radii=RADII,
        tz=TZ,
        min_matches=3,
    )


def _rows(frame: pd.DataFrame, scope: str, kind: str) -> pd.DataFrame:
    """Select one block of the tidy frame, index reset for easy comparison."""
    selected = frame[(frame["scope"] == scope) & (frame["kind"] == kind)]
    return selected.reset_index()


# ----------------------------
# Schema and shape
# ----------------------------
def test_index_and_columns_follow_the_schema(results):
    assert tuple(results.index.names) == RESULT_INDEX
    assert tuple(results.columns) == RESULT_COLUMNS


def test_index_is_unique_with_one_scene_per_month(results):
    assert results.index.is_unique


def test_row_counts_follow_the_months_periods_and_radii(results):
    months, periods, radii = 12, 2, len(RADII)

    counts = results.groupby(["scope", "kind"]).size()
    assert counts[(PERIOD_SCOPE, CLIMATOLOGY_KIND)] == months * periods
    assert counts[(PERIOD_SCOPE, CATEGORICAL_KIND)] == months * periods * radii
    assert counts[(PERIOD_SCOPE, CONTINUOUS_KIND)] == (
        len(SCENE_MONTHS) * periods * radii
    )
    assert counts[(SITE_YEAR_SCOPE, CLIMATOLOGY_KIND)] == periods
    assert counts[(SITE_SCOPE, CATEGORICAL_KIND)] == periods * radii
    assert counts[(SITE_SCOPE, CONTINUOUS_KIND)] == periods * radii


def test_aggregate_rows_leave_the_finer_index_levels_missing(results):
    site_year = _rows(results, SITE_YEAR_SCOPE, CLIMATOLOGY_KIND)
    assert site_year["month"].isna().all()
    assert site_year["radius"].isna().all()
    assert (site_year["year"] == 2020).all()

    for kind in (CATEGORICAL_KIND, CONTINUOUS_KIND):
        site = _rows(results, SITE_SCOPE, kind)
        assert site["year"].isna().all()
        assert site["month"].isna().all()
        assert site["radius"].notna().all()


def test_site_label_is_carried_into_the_index(results):
    assert set(results.index.get_level_values("site")) == {"US-Tst"}


def test_run_settings_are_recorded_in_attrs(results):
    assert results.attrs["site_id"] == "US-Tst"
    assert results.attrs["radii"] == RADII
    assert results.attrs["contour_fraction"] == pytest.approx(0.8)
    assert results.attrs["continuous_variable"] == "EVI"
    assert results.attrs["landcover_variable"] == "land_cover"
    assert results.attrs["unmatched_fields"] == ()


# ----------------------------
# The rows reproduce the underlying functions
# ----------------------------
def test_continuous_rows_match_sensor_location_bias(results, climatology, scenes):
    stamp = pd.Timestamp("2020-08-15")
    weights = climatology.footprint_climatology.sel(
        month=pd.Timestamp("2020-08-01"), period="daytime"
    )
    expected = sensor_location_bias(weights, scenes[stamp], GRID, GRID, radii=RADII)

    got = _rows(results, PERIOD_SCOPE, CONTINUOUS_KIND)
    got = got[
        (got["period"] == "daytime") & (got["month"] == pd.Timestamp("2020-08-01"))
    ].sort_values("radius")

    assert got["radius"].tolist() == list(RADII)
    assert got["value_footprint"].to_numpy() == pytest.approx(
        expected["value_footprint"].to_numpy()
    )
    assert got["value_target"].to_numpy() == pytest.approx(
        expected["value_target"].to_numpy()
    )
    assert got["bias"].to_numpy() == pytest.approx(expected["delta"].to_numpy())
    assert got["time"].eq(stamp).all()


def test_categorical_rows_match_categorical_representativeness(
    results, climatology, landcover
):
    weights = climatology.footprint_climatology.sel(
        month=pd.Timestamp("2020-05-01"), period="nighttime"
    )
    expected = categorical_representativeness(
        weights, landcover, GRID, GRID, radii=RADII
    )

    got = _rows(results, PERIOD_SCOPE, CATEGORICAL_KIND)
    got = got[
        (got["period"] == "nighttime") & (got["month"] == pd.Timestamp("2020-05-01"))
    ].sort_values("radius")

    assert got["dominant_class"].tolist() == [NEAR_CLASS] * len(RADII)
    assert got["value_footprint"].to_numpy() == pytest.approx(
        expected["p_footprint"].to_numpy()
    )
    assert got["value_target"].to_numpy() == pytest.approx(
        expected["p_target"].to_numpy()
    )
    assert got["chi2"].to_numpy() == pytest.approx(expected["chi2"].to_numpy())
    assert got["level"].tolist() == expected["level"].tolist()


def test_site_year_overlaps_match_the_overlap_indices(results, climatology):
    weights = climatology.footprint_climatology
    got = _rows(results, SITE_YEAR_SCOPE, CLIMATOLOGY_KIND).set_index("period")

    for period in ("daytime", "nighttime"):
        expected = seasonal_overlap(weights.sel(period=period))
        assert got.loc[period, "seasonal_overlap"] == pytest.approx(expected)

    expected_daynight = daynight_overlap(
        weights.sel(period="daytime"), weights.sel(period="nighttime")
    )
    # Eq. 3 belongs to neither period, so it is written to both rows.
    assert got["daynight_overlap"].to_numpy() == pytest.approx(expected_daynight)


def test_site_regression_matches_continuous_representativeness(results):
    period_rows = _rows(results, PERIOD_SCOPE, CONTINUOUS_KIND)
    pairs = period_rows[period_rows["period"] == "daytime"][
        ["time", "radius", "value_footprint", "value_target"]
    ].rename(columns={"time": "time"})
    expected = continuous_representativeness(pairs, radii=RADII, min_matches=3)

    got = _rows(results, SITE_SCOPE, CONTINUOUS_KIND)
    got = got[got["period"] == "daytime"].sort_values("radius")

    for column in ("slope", "intercept", "r_squared", "rmse", "mae", "p_value"):
        assert got[column].to_numpy() == pytest.approx(expected[column].to_numpy())
    assert got["n"].tolist() == expected["n"].tolist()
    assert got["level"].tolist() == expected["level"].tolist()


def test_site_year_metrics_average_the_monthly_ones(results):
    monthly = _rows(results, PERIOD_SCOPE, CLIMATOLOGY_KIND)
    daytime = monthly[monthly["period"] == "daytime"]
    site_year = _rows(results, SITE_YEAR_SCOPE, CLIMATOLOGY_KIND).set_index("period")

    for column in ("fetch", "area", "symmetry"):
        assert site_year.loc["daytime", column] == pytest.approx(
            daytime[column].mean()
        )


# ----------------------------
# What the numbers should say about this landscape
# ----------------------------
def test_a_footprint_inside_one_class_is_representative_only_nearby(results):
    site = _rows(results, SITE_SCOPE, CATEGORICAL_KIND)
    site = site[site["period"] == "daytime"].set_index("radius")

    # The whole 80 % source area sits in the near class.
    assert site["value_footprint"].to_numpy() == pytest.approx(1.0)
    # The disc picks up more of the far class as it grows.
    assert site["value_target"].is_monotonic_decreasing
    # Level is a str enum, so the column compares equal to the plain strings.
    assert site.loc[250.0, "level"] == Level.HIGH
    assert site.loc[1000.0, "level"] == Level.LOW


def test_the_footprint_sees_higher_values_than_its_surroundings(results):
    rows = _rows(results, PERIOD_SCOPE, CONTINUOUS_KIND)
    assert (rows["value_footprint"] > rows["value_target"]).all()
    assert (rows["bias"] > 0.0).all()


def test_regression_slope_falls_away_from_the_tower(results):
    site = _rows(results, SITE_SCOPE, CONTINUOUS_KIND)
    site = site[site["period"] == "daytime"].sort_values("radius")
    # Table 1: 0.96 at 250 m falling to 0.80 at 3000 m.
    assert site["slope"].is_monotonic_decreasing
    assert site["slope"].iloc[0] < 1.0


def test_within_threshold_averages_into_the_percentages_of_fig_7(results):
    rows = _rows(results, PERIOD_SCOPE, CONTINUOUS_KIND)
    share = rows.groupby("radius")["within_threshold"].mean()
    assert share.loc[250.0] >= share.loc[1000.0]
    assert ((share >= 0.0) & (share <= 1.0)).all()


def test_retained_weights_are_one_over_a_fully_covering_raster(results):
    rows = _rows(results, PERIOD_SCOPE, CONTINUOUS_KIND)
    assert rows["retained_footprint"].to_numpy() == pytest.approx(1.0)
    assert rows["retained_target"].to_numpy() == pytest.approx(1.0)


def test_a_partly_covering_scene_lowers_the_retained_weight(
    footprints, scenes
):
    stamp = pd.Timestamp("2020-08-15")
    gapped = scenes[stamp].where(XX < 0.0)
    results = assess_representativeness(
        footprints,
        station_lat=STATION_LAT,
        station_lon=STATION_LON,
        continuous={stamp: gapped},
        radii=(250.0,),
        tz=TZ,
        min_matches=3,
    )
    rows = _rows(results, PERIOD_SCOPE, CONTINUOUS_KIND)
    assert (rows["retained_footprint"] < 1.0).all()
    assert rows["retained_target"].to_numpy() == pytest.approx(0.5, abs=0.02)


# ----------------------------
# Optional analyses and input forms
# ----------------------------
def test_land_cover_alone_produces_no_continuous_rows(footprints, landcover):
    results = assess_representativeness(
        footprints,
        station_lat=STATION_LAT,
        station_lon=STATION_LON,
        landcover=landcover,
        radii=(250.0,),
        tz=TZ,
    )
    kinds = set(results["kind"])
    assert kinds == {CLIMATOLOGY_KIND, CATEGORICAL_KIND}
    assert representativeness_table(results, "S5").empty
    assert not representativeness_table(results, "S4").empty


def test_a_vegetation_index_alone_produces_no_categorical_rows(climatology, scenes):
    results = assess_representativeness(
        climatology, continuous=scenes, radii=(250.0,), min_matches=3
    )
    kinds = set(results["kind"])
    assert kinds == {CLIMATOLOGY_KIND, CONTINUOUS_KIND}
    assert representativeness_table(results, "S4").empty
    assert not representativeness_table(results, "S6").empty


def test_neither_field_still_reports_the_climatology_metrics(climatology):
    results = assess_representativeness(climatology, radii=(250.0,))
    assert set(results["kind"]) == {CLIMATOLOGY_KIND}
    assert results["fetch"].notna().any()
    assert results["seasonal_overlap"].notna().any()


def test_a_prebuilt_climatology_gives_the_same_answer(
    footprints, climatology, landcover, scenes
):
    from_footprints = assess_representativeness(
        footprints,
        station_lat=STATION_LAT,
        station_lon=STATION_LON,
        landcover=landcover,
        continuous=scenes,
        radii=RADII,
        tz=TZ,
        min_matches=3,
    )
    from_climatology = assess_representativeness(
        climatology,
        landcover=landcover,
        continuous=scenes,
        radii=RADII,
        min_matches=3,
    )
    pd.testing.assert_frame_equal(from_footprints, from_climatology)


def test_the_continuous_input_forms_agree(climatology, scenes):
    stamps = sorted(scenes)
    stacked = xr.concat(
        [scenes[stamp] for stamp in stamps], dim=pd.Index(stamps, name="time")
    ).transpose("time", "x", "y")

    forms = {
        "mapping": scenes,
        "series": pd.Series(list(scenes.values()), index=list(scenes)),
        "pairs": [(stamp, scenes[stamp]) for stamp in stamps],
        "dataarray": stacked,
        "dataset": stacked.to_dataset(name="EVI"),
    }

    reference = assess_representativeness(
        climatology, continuous=forms["mapping"], radii=RADII, min_matches=3
    )
    for name, form in forms.items():
        got = assess_representativeness(
            climatology, continuous=form, radii=RADII, min_matches=3
        )
        pd.testing.assert_frame_equal(
            got.sort_index(), reference.sort_index(), obj=name
        )


def test_a_scene_outside_the_record_is_reported_as_unmatched(climatology, scenes):
    stray = pd.Timestamp("2019-06-15")
    fields = dict(scenes)
    fields[stray] = scenes[pd.Timestamp("2020-08-15")]

    results = assess_representativeness(
        climatology, continuous=fields, radii=(250.0,), min_matches=3
    )
    assert results.attrs["unmatched_fields"] == (str(stray),)
    assert _rows(results, PERIOD_SCOPE, CONTINUOUS_KIND)["month"].min() == pd.Timestamp(
        "2020-02-01"
    )


def test_month_partition_gives_one_period_and_no_daynight_overlap(footprints):
    results = assess_representativeness(
        footprints, partition="month", radii=(250.0,)
    )
    assert set(results.index.get_level_values("period")) == {"all"}
    site_year = _rows(results, SITE_YEAR_SCOPE, CLIMATOLOGY_KIND)
    assert site_year["daynight_overlap"].isna().all()
    assert site_year["seasonal_overlap"].notna().all()


def test_min_times_holds_back_thin_months(footprints):
    # Six-hourly footprints give at most 124 timesteps in a 31-day month, so a
    # bar of 200 keeps the full months and drops the partial ones.
    dense = assess_representativeness(footprints, partition="month", radii=(250.0,))
    thinned = assess_representativeness(
        footprints,
        partition="month",
        radii=(250.0,),
        min_times=120,
    )
    dense_months = set(dense.index.get_level_values("month").dropna())
    thin_months = set(thinned.index.get_level_values("month").dropna())
    assert thin_months < dense_months


def test_holding_back_every_month_refuses_the_analysis(footprints):
    with pytest.raises(ValueError, match="nothing to evaluate"):
        assess_representativeness(
            footprints, partition="month", radii=(250.0,), min_times=10_000
        )


# ----------------------------
# Validation
# ----------------------------
def test_empty_radii_raise(climatology):
    with pytest.raises(ValueError, match="no target areas"):
        assess_representativeness(climatology, radii=())


@pytest.mark.parametrize("radius", [0.0, -250.0, float("nan"), float("inf")])
def test_an_unusable_radius_raises(climatology, radius):
    with pytest.raises(ValueError, match="positive and finite"):
        assess_representativeness(climatology, radii=(radius,))


def test_a_raster_off_the_grid_raises(climatology):
    coarse = xr.DataArray(
        np.zeros((10, 10)),
        coords={"x": np.arange(10.0), "y": np.arange(10.0)},
        dims=("x", "y"),
    )
    with pytest.raises(ValueError, match="footprint grid"):
        assess_representativeness(climatology, landcover=coarse, radii=(250.0,))


def test_a_raster_path_without_a_tower_position_raises(climatology, tmp_path):
    with pytest.raises(ValueError, match="no tower position"):
        assess_representativeness(
            climatology, landcover=tmp_path / "nlcd.tif", radii=(250.0,)
        )


def test_an_untimed_continuous_array_raises(climatology, scenes):
    with pytest.raises(ValueError, match="no time series"):
        assess_representativeness(
            climatology,
            continuous=scenes[pd.Timestamp("2020-08-15")],
            radii=(250.0,),
        )


def test_a_climatology_dataset_without_coords_raises(climatology):
    stripped = climatology.drop_vars("x")
    with pytest.raises(ValueError, match="no 'x' coordinate"):
        assess_representativeness(stripped, radii=(250.0,))


# ----------------------------
# Published-schema tables
# ----------------------------
@pytest.mark.parametrize("dataset", ["S4", "S5", "S6"])
def test_each_table_carries_its_documented_columns(results, dataset):
    table = representativeness_table(results, dataset)
    assert tuple(table.columns) == CHU_TABLE_COLUMNS[dataset]
    assert not table.empty
    assert isinstance(table.index, pd.RangeIndex)


def test_table_lookup_is_case_insensitive(results):
    pd.testing.assert_frame_equal(
        representativeness_table(results, "s6"),
        representativeness_table(results, "S6"),
    )


def test_an_unknown_dataset_raises(results):
    with pytest.raises(KeyError, match="S4"):
        representativeness_table(results, "S7")


def test_s4_reports_shares_as_percentages(results):
    table = representativeness_table(results, "S4")
    source = _rows(results, SITE_SCOPE, CATEGORICAL_KIND)
    assert table["P_FOOTPRINT"].to_numpy() == pytest.approx(
        100.0 * source["value_footprint"].to_numpy()
    )
    assert table["P_TARGET"].to_numpy() == pytest.approx(
        100.0 * source["value_target"].to_numpy()
    )
    assert table["P_DIFF"].to_numpy() == pytest.approx(
        (table["P_FOOTPRINT"] - table["P_TARGET"]).to_numpy()
    )
    assert table["REP_LC"].isin(["high", "medium", "low"]).all()


def test_s5_reports_the_bias_as_a_percentage_and_the_month_as_a_number(results):
    table = representativeness_table(results, "S5")
    source = _rows(results, PERIOD_SCOPE, CONTINUOUS_KIND)
    assert table["DELTA"].to_numpy() == pytest.approx(
        100.0 * source["bias"].to_numpy()
    )
    assert set(table["MONTH"]) == set(SCENE_MONTHS)
    assert (table["YEAR"] == 2020).all()


def test_s6_follows_the_layout_of_table_1(results):
    table = representativeness_table(results, "S6")
    source = _rows(results, SITE_SCOPE, CONTINUOUS_KIND)
    assert table["SLOPE"].to_numpy() == pytest.approx(source["slope"].to_numpy())
    assert table["R2"].to_numpy() == pytest.approx(source["r_squared"].to_numpy())
    assert (table["SLOPE_LCL"] <= table["SLOPE"]).all()
    assert (table["SLOPE"] <= table["SLOPE_UCL"]).all()


def test_a_reset_frame_is_accepted(results):
    pd.testing.assert_frame_equal(
        representativeness_table(results.reset_index(), "S5"),
        representativeness_table(results, "S5"),
    )


def test_a_frame_that_is_not_a_result_raises():
    with pytest.raises(ValueError, match="missing the column"):
        representativeness_table(pd.DataFrame({"a": [1]}), "S4")


# ----------------------------
# Writer
# ----------------------------
def test_csv_export_writes_the_three_tables(results, tmp_path):
    written = export_representativeness_tables(results, tmp_path, prefix="US-Tst")

    assert sorted(written) == ["S4", "S5", "S6"]
    for dataset, path in written.items():
        assert path.name == f"US-Tst_{dataset}.csv"
        reread = pd.read_csv(path)
        assert tuple(reread.columns) == CHU_TABLE_COLUMNS[dataset]
        assert len(reread) == len(representativeness_table(results, dataset))


def test_the_output_directory_is_created(results, tmp_path):
    target = tmp_path / "nested" / "out"
    export_representativeness_tables(results, target)
    assert target.is_dir()


def test_parquet_export_preserves_dtypes(results, tmp_path):
    pytest.importorskip("pyarrow")
    written = export_representativeness_tables(results, tmp_path, fmt="parquet")

    reread = pd.read_parquet(written["S6"])
    assert tuple(reread.columns) == CHU_TABLE_COLUMNS["S6"]
    assert reread["N"].dtype == pd.Int64Dtype()
    assert reread["SLOPE"].to_numpy() == pytest.approx(
        representativeness_table(results, "S6")["SLOPE"].to_numpy()
    )


def test_an_empty_table_is_skipped_unless_asked_for(climatology, scenes, tmp_path):
    results = assess_representativeness(
        climatology, continuous=scenes, radii=(250.0,), min_matches=3
    )

    skipped = export_representativeness_tables(results, tmp_path / "skip")
    assert "S4" not in skipped

    kept = export_representativeness_tables(
        results, tmp_path / "keep", write_empty=True
    )
    assert "S4" in kept
    assert pd.read_csv(kept["S4"]).empty


def test_a_subset_of_datasets_can_be_requested(results, tmp_path):
    written = export_representativeness_tables(results, tmp_path, datasets=("S6",))
    assert sorted(written) == ["S6"]
    assert not (tmp_path / "representativeness_S4.csv").exists()


def test_an_unknown_format_raises(results, tmp_path):
    with pytest.raises(ValueError, match="fmt must be one of"):
        export_representativeness_tables(results, tmp_path, fmt="xlsx")


def test_frames_from_several_sites_concatenate(results):
    other = results.rename(index={"US-Tst": "US-Oth"}, level="site")
    pooled = pd.concat([results, other])
    table = representativeness_table(pooled, "S6")
    assert set(table["SITE_ID"]) == {"US-Tst", "US-Oth"}
    assert len(table) == 2 * len(representativeness_table(results, "S6"))
