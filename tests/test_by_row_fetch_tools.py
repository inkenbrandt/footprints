# tests/test_by_row_fetch_tools.py
import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point
from rasterio.transform import Affine

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints.by_row_fetch_tools import (
    polar_to_cartesian_dataframe,
    aggregate_to_daily_centroid,
    generate_density_raster,
    concat_fetch_gdf,
    impute_evapotranspiration,
)

FETCH_COLUMNS = [
    "X_FETCH_90",
    "Y_FETCH_90",
    "X_FETCH_55",
    "Y_FETCH_55",
    "X_FETCH_40",
    "Y_FETCH_40",
]


def _fetch_frame():
    """Two half-hourly rows on 2020-01-01 and one on 2020-01-03 (01-02 is a gap)."""
    idx = pd.to_datetime(
        ["2020-01-01 00:00", "2020-01-01 12:00", "2020-01-03 06:00"],
    )
    return pd.DataFrame(
        {
            "X_FETCH_90": [100.0, 200.0, 300.0],
            "Y_FETCH_90": [10.0, 20.0, 30.0],
            "X_FETCH_55": [50.0, 60.0, 70.0],
            "Y_FETCH_55": [5.0, 6.0, 7.0],
            "X_FETCH_40": [10.0, 20.0, 30.0],
            "Y_FETCH_40": [1.0, 2.0, 3.0],
        },
        index=idx,
    )


def _centroid_frame():
    """Two sub-daily rows on 2020-01-01, one on 2020-01-02, with ET weights."""
    return pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 12:00", "2020-01-02 06:00"]
            ),
            "X": [0.0, 10.0, 5.0],
            "Y": [0.0, 20.0, 7.0],
            "ET": [1.0, 3.0, 2.0],
        }
    )


# ---------------------------------------------------------------------------
# polar_to_cartesian_dataframe
# ---------------------------------------------------------------------------
def test_polar_to_cartesian_dataframe_basic():
    """Known-angle sanity check + invalid-value handling."""
    df = pd.DataFrame(
        {
            "WD": [0, 90, 180, 270, -9999, np.nan],
            "Dist": [1, 1, 1, 1, 1, 1],
        }
    )

    out = polar_to_cartesian_dataframe(df)

    # Expected Cartesian coordinates for the first four valid rows
    exp_x = np.array([0, 1, 0, -1], dtype=float)
    exp_y = np.array([1, 0, -1, 0], dtype=float)

    np.testing.assert_allclose(out["X_Dist"].iloc[:4], exp_x, atol=1e-12)
    np.testing.assert_allclose(out["Y_Dist"].iloc[:4], exp_y, atol=1e-12)

    # Invalid entries should propagate to NaN
    assert out["X_Dist"].iloc[4:].isna().all()
    assert out["Y_Dist"].iloc[4:].isna().all()


def test_polar_to_cartesian_dataframe_returns_only_new_columns():
    """The result holds only the new X_/Y_ columns and the input is unchanged."""
    df = pd.DataFrame({"WD": [0, 90], "Dist": [1, 1]})
    original_columns = list(df.columns)

    out = polar_to_cartesian_dataframe(df)

    # Only the freshly computed coordinate columns are returned
    assert list(out.columns) == ["X_Dist", "Y_Dist"]

    # The input DataFrame is left untouched (no in-place mutation)
    assert list(df.columns) == original_columns

    # Index is preserved
    assert out.index.equals(df.index)


def test_polar_to_cartesian_dataframe_list_of_columns():
    """A list of distance columns yields an X_/Y_ pair for each."""
    df = pd.DataFrame(
        {
            "WD": [0, 90, 180],
            "FETCH_90": [1, 1, 1],
            "FETCH_40": [2, 2, 2],
        }
    )

    out = polar_to_cartesian_dataframe(df, dist_column=["FETCH_90", "FETCH_40"])

    assert list(out.columns) == [
        "X_FETCH_90",
        "Y_FETCH_90",
        "X_FETCH_40",
        "Y_FETCH_40",
    ]

    # Spot-check known angles for both distance columns
    np.testing.assert_allclose(
        out["X_FETCH_90"], np.array([0, 1, 0], dtype=float), atol=1e-12
    )
    np.testing.assert_allclose(
        out["Y_FETCH_90"], np.array([1, 0, -1], dtype=float), atol=1e-12
    )
    np.testing.assert_allclose(
        out["X_FETCH_40"], np.array([0, 2, 0], dtype=float), atol=1e-12
    )
    np.testing.assert_allclose(
        out["Y_FETCH_40"], np.array([2, 0, -2], dtype=float), atol=1e-12
    )


# ---------------------------------------------------------------------------
# aggregate_to_daily_centroid
# ---------------------------------------------------------------------------
def test_aggregate_to_daily_centroid_unweighted():
    """One row per calendar day holding the plain mean of X and Y."""
    out = aggregate_to_daily_centroid(_centroid_frame(), weighted=False)

    assert list(out.columns) == ["Date", "X", "Y"]
    assert len(out) == 2

    # 2020-01-01: mean of (0, 10) and (0, 20); 2020-01-02: the lone row
    np.testing.assert_allclose(out["X"], [5.0, 5.0])
    np.testing.assert_allclose(out["Y"], [10.0, 7.0])


def test_aggregate_to_daily_centroid_weighted_by_et():
    """The weighted centroid uses ET as the weight for each sub-daily row."""
    out = aggregate_to_daily_centroid(_centroid_frame(), weighted=True)

    # 2020-01-01: (0*1 + 10*3) / (1 + 3) = 7.5 and (0*1 + 20*3) / 4 = 15.0
    # 2020-01-02: a single row, so the weight cancels out
    np.testing.assert_allclose(out["X"], [7.5, 5.0])
    np.testing.assert_allclose(out["Y"], [15.0, 7.0])


def test_aggregate_to_daily_centroid_equal_weights_match_unweighted():
    """With a constant ET column the weighted result collapses to the mean."""
    df = _centroid_frame()
    df["ET"] = 2.0

    weighted = aggregate_to_daily_centroid(df.copy(), weighted=True)
    unweighted = aggregate_to_daily_centroid(df.copy(), weighted=False)

    np.testing.assert_allclose(weighted["X"], unweighted["X"])
    np.testing.assert_allclose(weighted["Y"], unweighted["Y"])


def test_aggregate_to_daily_centroid_custom_column_names():
    """Alternate timestamp and coordinate column names are honoured."""
    df = _centroid_frame().rename(
        columns={"Timestamp": "datetime_start", "X": "easting", "Y": "northing"}
    )

    out = aggregate_to_daily_centroid(
        df,
        date_column="datetime_start",
        x_column="easting",
        y_column="northing",
        weighted=True,
    )

    assert list(out.columns) == ["Date", "easting", "northing"]
    np.testing.assert_allclose(out["easting"], [7.5, 5.0])
    np.testing.assert_allclose(out["northing"], [15.0, 7.0])


def test_aggregate_to_daily_centroid_dates_are_sorted_python_dates():
    """The Date column holds datetime.date objects, one per day, in order."""
    # Feed the rows in reverse order to confirm the groupby sorts them
    out = aggregate_to_daily_centroid(_centroid_frame().iloc[::-1].copy())

    assert list(out["Date"]) == [
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 2),
    ]


def test_aggregate_to_daily_centroid_parses_string_timestamps():
    """A string timestamp column is coerced with pd.to_datetime."""
    df = _centroid_frame()
    df["Timestamp"] = df["Timestamp"].astype(str)

    out = aggregate_to_daily_centroid(df, weighted=False)

    assert len(out) == 2
    np.testing.assert_allclose(out["X"], [5.0, 5.0])


@pytest.mark.parametrize("weighted", [True, False])
def test_aggregate_to_daily_centroid_does_not_mutate_input(weighted):
    """No 'Date' helper column is added and the timestamp column is not recast."""
    df = _centroid_frame()
    df["Timestamp"] = df["Timestamp"].astype(str)  # would be coerced if mutated
    before = df.copy()

    aggregate_to_daily_centroid(df, weighted=weighted)

    assert "Date" not in df.columns
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# generate_density_raster
# ---------------------------------------------------------------------------
def test_generate_density_raster_properties():
    """Minimal smoke test: array dims, transform, bounds, and non-negativity."""
    # Create ten weighted points jittered off a diagonal line so the
    # coordinates aren't collinear (gaussian_kde requires a non-singular
    # covariance matrix, which a perfectly straight line can't provide).
    rng = np.random.default_rng(0)
    base = np.linspace(0, 100, 10)
    x = base + rng.normal(scale=5, size=10)
    y = base + rng.normal(scale=5, size=10)
    pts = [Point(px, py) for px, py in zip(x, y)]
    weights = rng.uniform(0.5, 1.5, size=10)
    gdf = gpd.GeoDataFrame({"ET": weights}, geometry=pts, crs="EPSG:5070")

    density, transform, bounds = generate_density_raster(gdf, resolution=50)

    # Basic array checks
    assert density.ndim == 2 and density.size > 0
    assert np.all(density >= 0)

    # Affine transform should reflect the chosen resolution
    assert isinstance(transform, Affine)
    assert transform.a == pytest.approx(50)  # pixel width
    assert transform.e == pytest.approx(-50)  # pixel height (negative y-scale)

    # Bounds tuple sanity
    assert isinstance(bounds, tuple) and len(bounds) == 4


# ---------------------------------------------------------------------------
# concat_fetch_gdf
# ---------------------------------------------------------------------------
def test_concat_fetch_gdf_stacks_three_fetch_rings():
    """Every input row becomes three points, tagged with weights 90/55/40."""
    _, gdf = concat_fetch_gdf(_fetch_frame())

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 9  # 3 rows x 3 fetch distances
    assert sorted(gdf["weights"].unique().tolist()) == [40, 55, 90]
    assert (gdf["weights"].value_counts() == 3).all()
    assert gdf.index.name == "datetime_start"

    # Geometry must agree with the x/y columns it was built from
    np.testing.assert_allclose(gdf.geometry.x, gdf["x"])
    np.testing.assert_allclose(gdf.geometry.y, gdf["y"])

    # The 90% ring keeps its own coordinates
    ring90 = gdf[gdf["weights"] == 90]
    np.testing.assert_allclose(sorted(ring90["x"]), [100.0, 200.0, 300.0])


def test_concat_fetch_gdf_daily_weighted_centroid():
    """Daily rows are the fetch-distance-weighted mean of that day's points."""
    gdf_day, _ = concat_fetch_gdf(_fetch_frame())

    assert list(gdf_day.index) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-03"),
    ]

    # 2020-01-01 has six points: (100,90) (200,90) (50,55) (60,55) (10,40) (20,40)
    day1_x = (100 * 90 + 200 * 90 + 50 * 55 + 60 * 55 + 10 * 40 + 20 * 40) / 370
    day1_y = (10 * 90 + 20 * 90 + 5 * 55 + 6 * 55 + 1 * 40 + 2 * 40) / 370
    # 2020-01-03 has the three rings of a single row
    day3_x = (300 * 90 + 70 * 55 + 30 * 40) / 185
    day3_y = (30 * 90 + 7 * 55 + 3 * 40) / 185

    np.testing.assert_allclose(gdf_day["x"], [day1_x, day3_x])
    np.testing.assert_allclose(gdf_day["y"], [day1_y, day3_y])
    np.testing.assert_allclose(gdf_day.geometry.x, [day1_x, day3_x])
    np.testing.assert_allclose(gdf_day.geometry.y, [day1_y, day3_y])

    # weights is the mean of the three ring labels, not their sum
    np.testing.assert_allclose(gdf_day["weights"], (90 + 55 + 40) / 3)


def test_concat_fetch_gdf_drops_empty_days():
    """2020-01-02 has no observations and is absent from the daily frame."""
    gdf_day, _ = concat_fetch_gdf(_fetch_frame())

    assert pd.Timestamp("2020-01-02") not in gdf_day.index
    assert not gdf_day.drop(columns="geometry").isna().any().any()


@pytest.mark.parametrize("nan_column", FETCH_COLUMNS)
def test_concat_fetch_gdf_drops_rows_missing_any_fetch(nan_column):
    """A NaN in any one fetch column removes that row from all three rings."""
    data = _fetch_frame()
    data.loc[data.index[1], nan_column] = np.nan

    gdf_day, gdf = concat_fetch_gdf(data)

    assert len(gdf) == 6  # 2 surviving rows x 3 rings
    assert pd.Timestamp("2020-01-01 12:00") not in gdf.index

    # Only the 00:00 row is left on 2020-01-01
    day1 = gdf_day.loc[pd.Timestamp("2020-01-01")]
    assert day1["x"] == pytest.approx((100 * 90 + 50 * 55 + 10 * 40) / 185)


def test_concat_fetch_gdf_sets_crs():
    """Both outputs carry the requested CRS; the default is EPSG:5070."""
    gdf_day, gdf = concat_fetch_gdf(_fetch_frame())
    assert gdf_day.crs.to_epsg() == 5070
    assert gdf.crs.to_epsg() == 5070

    gdf_day_utm, gdf_utm = concat_fetch_gdf(_fetch_frame(), epsg=32612)
    assert gdf_day_utm.crs.to_epsg() == 32612
    assert gdf_utm.crs.to_epsg() == 32612


def test_concat_fetch_gdf_does_not_mutate_input():
    """The caller's frame is untouched by the dropna and reshape."""
    data = _fetch_frame()
    before = data.copy()

    concat_fetch_gdf(data)

    pd.testing.assert_frame_equal(data, before)


# ---------------------------------------------------------------------------
# impute_evapotranspiration
# ---------------------------------------------------------------------------
def test_impute_evapotranspiration_fills_short_gap_by_interpolation():
    """A single missing half hour is filled from its neighbours."""
    idx = pd.date_range("2020-05-01", periods=4, freq="30min")
    df = pd.DataFrame({"ET": [1.0, np.nan, 3.0, 4.0]}, index=idx)

    out = impute_evapotranspiration(df)

    assert not out["ET"].isna().any()
    # Interpolation gives 2.0 for the gap; the centred 6-wide rolling mean then
    # smooths the resulting (1, 2, 3, 4) series.
    np.testing.assert_allclose(out["ET"], [2.0, 2.5, 2.5, 2.5])


def test_impute_evapotranspiration_uses_seasonal_median_for_long_gaps():
    """A gap too long to interpolate is filled from the same day-of-year and
    time of day in another year."""
    gap_year = pd.date_range("2019-01-10", periods=48, freq="30min")
    obs_year = pd.date_range("2020-01-10", periods=48, freq="30min")
    assert gap_year[0].dayofyear == obs_year[0].dayofyear

    values = 1.0 + np.arange(48) * 0.3
    df = pd.DataFrame(
        {"ET": np.concatenate([np.full(48, np.nan), values])},
        index=gap_year.append(obs_year),
    )

    out = impute_evapotranspiration(df)

    assert not out["ET"].isna().any()

    # Away from the block edges (where the rolling window straddles the two
    # years) the imputed 2019 day reproduces the observed 2020 day exactly.
    imputed = out["ET"].values[:48]
    observed = out["ET"].values[48:]
    np.testing.assert_allclose(imputed[4:44], observed[4:44])


def test_impute_evapotranspiration_backfills_when_no_median_exists():
    """With only one year of data a long gap falls back to bfill/ffill."""
    idx = pd.date_range("2020-05-01", periods=20, freq="30min")
    values = np.full(20, np.nan)
    values[:2] = 1.0
    values[-2:] = 100.0
    df = pd.DataFrame({"ET": values}, index=idx)

    out = impute_evapotranspiration(df)

    assert not out["ET"].isna().any()
    # Every value is a mean of observed/filled points bounded by the two levels
    assert out["ET"].min() >= 1.0
    assert out["ET"].max() <= 100.0
    assert out["ET"].is_monotonic_increasing


def test_impute_evapotranspiration_smooths_complete_series():
    """A gap-free constant series survives the rolling smoother unchanged."""
    idx = pd.date_range("2020-01-10", periods=48, freq="30min")
    df = pd.DataFrame({"ET": np.full(48, 5.0)}, index=idx)

    out = impute_evapotranspiration(df)

    np.testing.assert_allclose(out["ET"], 5.0)


def test_impute_evapotranspiration_writes_to_separate_out_field():
    """in_field is left as-is when out_field names a different column."""
    idx = pd.date_range("2020-05-01", periods=4, freq="30min")
    df = pd.DataFrame({"raw": [1.0, np.nan, 3.0, 4.0]}, index=idx)

    out = impute_evapotranspiration(df, in_field="raw", out_field="ET_filled")

    assert "ET_filled" in out.columns
    assert out["raw"].isna().sum() == 1  # source column untouched
    assert not out["ET_filled"].isna().any()


def test_impute_evapotranspiration_drops_helpers_and_preserves_input():
    """Helper columns are removed and the caller's frame is not modified."""
    idx = pd.date_range("2020-05-01", periods=6, freq="30min")
    df = pd.DataFrame(
        {"ET": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0], "other": list("abcdef")},
        index=idx,
    )
    before = df.copy()

    out = impute_evapotranspiration(df)

    assert not {"hour", "minute", "day_of_year"} & set(out.columns)
    assert list(out.columns) == ["ET", "other"]
    assert out.index.equals(df.index)
    pd.testing.assert_frame_equal(df, before)


def test_impute_evapotranspiration_coerces_string_index():
    """A string index is converted to a DatetimeIndex."""
    df = pd.DataFrame(
        {"ET": [1.0, np.nan, 3.0, 4.0]},
        index=[
            "2020-05-01 00:00",
            "2020-05-01 00:30",
            "2020-05-01 01:00",
            "2020-05-01 01:30",
        ],
    )

    out = impute_evapotranspiration(df)

    assert isinstance(out.index, pd.DatetimeIndex)
    assert not out["ET"].isna().any()


def test_impute_evapotranspiration_all_missing_stays_missing():
    """Nothing to impute from: the column comes back all NaN rather than raising."""
    idx = pd.date_range("2020-05-01", periods=6, freq="30min")
    df = pd.DataFrame({"ET": np.full(6, np.nan)}, index=idx)

    out = impute_evapotranspiration(df)

    assert out["ET"].isna().all()
