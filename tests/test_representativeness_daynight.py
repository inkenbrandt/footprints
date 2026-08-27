"""
Day-night partitioning and monthly-climatology tests for
:mod:`fluxfootprints.representativeness`.

Chu et al. (2021), Sect. 2.2, split each month's footprints into a daytime and
a nighttime climatology by the potential (top-of-atmosphere) incoming shortwave
radiation, calling a record daytime wherever that radiation exceeds 0 W m-2.
The solar geometry behind that threshold is checked here against published
quantities rather than against itself: NOAA sunrise and sunset times for a
real site, the equation of time at four dates spanning its swing, the 12-hour
equinox day at the equator, and the polar day and night above the Arctic
Circle. Published rise and set times include atmospheric refraction and the
solar disc radius (0.833 degrees below the true horizon), which the geometric
crossing used here does not, so the two differ by the few minutes the Sun takes
to cross that angle -- late at sunrise, early at sunset.

The climatology fixtures are Gaussian footprints whose width and downwind
offset switch with the day-night flag, reproducing the paper's finding that
nighttime footprints reach farther and cover more area.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxfootprints.representativeness import (
    ALL_HOURS,
    DAYTIME,
    NIGHTTIME,
    SOLAR_CONSTANT,
    SW_IN_POT_COLUMN,
    climatology_metrics,
    daynight_overlap,
    monthly_climatologies,
    partition_daynight,
    potential_radiation,
    seasonal_overlap,
    truncate_to_contour,
)

# Salt Lake City, in local standard time year-round as AmeriFlux records are.
SLC_LAT = 40.7608
SLC_LON = -111.8910
SLC_TZ = -7

STEP = 20.0
EXTENT = 400.0


# ------------------------------
# Fixtures
# ------------------------------


def minute_of(times: pd.DatetimeIndex, index: int) -> str:
    """Format one timestamp as HH:MM."""
    return times[index].strftime("%H:%M")


def horizon_crossings(
    latitude: float,
    longitude: float,
    tz: float,
    date: str,
) -> tuple[str | None, str | None]:
    """Find the day's geometric sunrise and sunset to the minute, by scan."""
    times = pd.date_range(f"{date} 00:00", f"{date} 23:59", freq="1min")
    daytime = partition_daynight(times, latitude, longitude, tz=tz).to_numpy()
    rises = np.flatnonzero(~daytime[:-1] & daytime[1:])
    sets = np.flatnonzero(daytime[:-1] & ~daytime[1:])
    return (
        minute_of(times, rises[0] + 1) if rises.size else None,
        minute_of(times, sets[0]) if sets.size else None,
    )


def minutes_apart(first: str, second: str) -> float:
    """Signed difference in minutes between two HH:MM clock readings."""

    def to_minutes(hhmm: str) -> int:
        return int(hhmm[:2]) * 60 + int(hhmm[3:])

    return to_minutes(first) - to_minutes(second)


def footprint_series(
    times: pd.DatetimeIndex,
    daytime: np.ndarray | None = None,
    day_sigma: float = 60.0,
    night_sigma: float = 110.0,
) -> xr.DataArray:
    """
    Build time-resolved Gaussian footprints [m-2] with dims ``(time, x, y)``.

    Nighttime footprints are wider and pushed farther downwind, as the paper
    reports; passing `daytime` explicitly keeps the fixture independent of the
    partitioning under test.
    """
    if daytime is None:
        daytime = partition_daynight(
            times, SLC_LAT, SLC_LON, tz=SLC_TZ
        ).to_numpy()
    x = np.arange(-EXTENT, EXTENT + STEP / 2, STEP)
    y = np.arange(-EXTENT, EXTENT + STEP / 2, STEP)
    sigma = np.where(daytime, day_sigma, night_sigma)[:, None, None]
    offset = sigma

    xx, yy = np.meshgrid(x, y, indexing="ij")
    f = np.exp(
        -0.5 * (((xx[None] - offset) / sigma) ** 2 + (yy[None] / sigma) ** 2)
    ) / (2 * np.pi * sigma**2)
    return xr.DataArray(
        f,
        coords={"time": times, "x": x, "y": y},
        dims=("time", "x", "y"),
        name="footprint_2d",
        attrs={"units": "m-2"},
    )


class FakeModel:
    """Stand-in for a run footprint model: time-resolved array plus its frame."""

    def __init__(self, f_2d: xr.DataArray | None, df: pd.DataFrame | None = None):
        self.f_2d = f_2d
        if df is None and f_2d is not None:
            df = pd.DataFrame(index=pd.DatetimeIndex(f_2d["time"].values))
        self.df = df


@pytest.fixture
def quarter() -> pd.DatetimeIndex:
    """Three-hourly timestamps over a four-month site-period."""
    return pd.date_range("2020-01-01", "2020-04-30 23:00", freq="3h")


# ------------------------------
# Solar geometry
# ------------------------------


@pytest.mark.parametrize(
    ("date", "noaa_rise", "noaa_set"),
    [
        # NOAA published times, converted to the standard-time clock the data
        # are on: 2020-06-21 is 05:58/21:01 MDT, 2020-12-21 is 07:48/17:03 MST.
        ("2020-06-21", "04:58", "20:01"),
        ("2020-12-21", "07:48", "17:03"),
    ],
)
def test_horizon_crossings_match_noaa_sunrise_and_sunset(date, noaa_rise, noaa_set):
    rise, sunset = horizon_crossings(SLC_LAT, SLC_LON, SLC_TZ, date)
    # Refraction and the solar disc put the published times a few minutes
    # early at sunrise and late at sunset; the geometric crossing lags both.
    assert 2 <= minutes_apart(rise, noaa_rise) <= 10
    assert 2 <= minutes_apart(noaa_set, sunset) <= 10


def test_equinox_day_is_twelve_hours_at_the_equator():
    rise, sunset = horizon_crossings(0.0, 0.0, 0, "2020-03-20")
    assert minutes_apart(sunset, rise) == pytest.approx(720.0, abs=5.0)


def test_the_sun_never_sets_in_the_arctic_summer_and_never_rises_in_winter():
    # Utqiagvik, Alaska (71.32 N), well inside the Arctic Circle.
    summer = pd.date_range("2020-06-21", periods=48, freq="30min")
    winter = pd.date_range("2020-12-21", periods=48, freq="30min")

    assert bool(partition_daynight(summer, 71.3230, -156.6114, tz=-9).all())
    assert not bool(partition_daynight(winter, 71.3230, -156.6114, tz=-9).any())


@pytest.mark.parametrize(
    ("date", "reference"),
    # Equation of time at its two maxima and two minima, in minutes.
    [
        ("2020-02-11", -14.2),
        ("2020-05-14", 3.7),
        ("2020-07-26", -6.5),
        ("2020-11-03", 16.4),
    ],
)
def test_equation_of_time_matches_reference_values(date, reference):
    # Solar noon on the prime meridian lags clock noon by the equation of time.
    times = pd.date_range(f"{date} 00:00", f"{date} 23:59", freq="1min", tz="UTC")
    solar_noon = times[int(np.argmax(potential_radiation(times, 0.0, 0.0).to_numpy()))]
    equation_of_time = 720.0 - (solar_noon.hour * 60 + solar_noon.minute)
    assert equation_of_time == pytest.approx(reference, abs=1.0)


def test_solar_noon_shifts_four_minutes_per_degree_of_longitude():
    times = pd.date_range("2020-03-20 00:00", periods=1440, freq="1min", tz="UTC")
    noons = [
        int(np.argmax(potential_radiation(times, 0.0, longitude).to_numpy()))
        for longitude in (0.0, -15.0)
    ]
    assert noons[1] - noons[0] == pytest.approx(60.0, abs=1.0)


def test_flux_tracks_the_earth_sun_distance_over_the_year():
    # The orbit is eccentric: perihelion in early January brings the Earth
    # about 3.4 % more irradiance than aphelion in early July.
    noons = pd.DatetimeIndex(
        ["2020-01-03 12:00", "2020-07-04 12:00"]
    ).tz_localize("UTC")
    perihelion, aphelion = potential_radiation(noons, 0.0, 0.0).to_numpy()
    assert perihelion / aphelion == pytest.approx(1.069, abs=0.005)


def test_overhead_sun_receives_about_the_solar_constant():
    # Equinox noon on the equator puts the Sun within a degree of the zenith.
    noon = pd.DatetimeIndex(["2020-03-20 12:07"]).tz_localize("UTC")
    flux = float(potential_radiation(noon, 0.0, 0.0).iloc[0])
    assert flux == pytest.approx(SOLAR_CONSTANT, rel=0.02)


def test_radiation_is_never_negative_and_is_zero_all_night():
    times = pd.date_range("2020-09-01", periods=48, freq="30min")
    flux = potential_radiation(times, SLC_LAT, SLC_LON, tz=SLC_TZ)
    daytime = partition_daynight(times, SLC_LAT, SLC_LON, tz=SLC_TZ)

    assert bool((flux >= 0.0).all())
    assert bool((flux[~daytime] == 0.0).all())
    assert bool((flux[daytime] > 0.0).all())


def test_radiation_keeps_the_index_and_is_named_for_the_ameriflux_variable():
    times = pd.date_range("2020-09-01", periods=4, freq="6h")
    flux = potential_radiation(times, SLC_LAT, SLC_LON, tz=SLC_TZ)

    assert flux.name == SW_IN_POT_COLUMN == "SW_IN_POT"
    pd.testing.assert_index_equal(flux.index, times)


def test_a_frame_and_its_index_give_the_same_radiation():
    times = pd.date_range("2020-09-01", periods=8, freq="3h")
    frame = pd.DataFrame({"ustar": np.full(8, 0.3)}, index=times)
    pd.testing.assert_series_equal(
        potential_radiation(frame, SLC_LAT, SLC_LON, tz=SLC_TZ),
        potential_radiation(times, SLC_LAT, SLC_LON, tz=SLC_TZ),
    )


# ------------------------------
# Time zones
# ------------------------------


def test_a_fixed_offset_and_an_aware_index_agree():
    naive = pd.date_range("2020-09-01", periods=48, freq="30min")
    aware = naive.tz_localize(dt.timezone(dt.timedelta(hours=SLC_TZ)))

    np.testing.assert_array_equal(
        partition_daynight(naive, SLC_LAT, SLC_LON, tz=SLC_TZ).to_numpy(),
        partition_daynight(aware, SLC_LAT, SLC_LON).to_numpy(),
    )


def test_a_zone_name_applies_daylight_saving_and_a_fixed_offset_does_not():
    # The trap the docstring warns about: AmeriFlux clocks never spring
    # forward, so reading a summer timestamp through a zone name shifts it an
    # hour and can move a record across the horizon. Sunrise on this day is
    # 05:02 standard time, so 05:30 is daytime on the standard clock and
    # 04:30 -- still night -- when read as daylight time.
    summer = pd.date_range("2020-06-21 05:30", periods=1, freq="30min")
    assert bool(partition_daynight(summer, SLC_LAT, SLC_LON, tz=-7).iloc[0])
    assert not bool(
        partition_daynight(summer, SLC_LAT, SLC_LON, tz="America/Denver").iloc[0]
    )

    # In winter the two agree, because Denver is then on standard time.
    winter = pd.date_range("2020-12-21", periods=48, freq="30min")
    np.testing.assert_array_equal(
        partition_daynight(winter, SLC_LAT, SLC_LON, tz=-7).to_numpy(),
        partition_daynight(winter, SLC_LAT, SLC_LON, tz="America/Denver").to_numpy(),
    )


def test_a_naive_index_needs_a_time_zone():
    times = pd.date_range("2020-09-01", periods=4, freq="6h")
    with pytest.raises(ValueError, match="time-zone naive"):
        partition_daynight(times, SLC_LAT, SLC_LON)


def test_an_aware_index_refuses_a_second_time_zone():
    times = pd.date_range("2020-09-01", periods=4, freq="6h", tz="UTC")
    with pytest.raises(ValueError, match="already carries the time zone"):
        partition_daynight(times, SLC_LAT, SLC_LON, tz=-7)


def test_a_missing_timestamp_is_refused():
    times = pd.DatetimeIndex(["2020-09-01 00:00", pd.NaT, "2020-09-01 12:00"])
    with pytest.raises(ValueError, match="NaT"):
        partition_daynight(times, SLC_LAT, SLC_LON, tz=SLC_TZ)


@pytest.mark.parametrize("tz", ["Mars/Olympus_Mons", "MST7MDT-nonsense"])
def test_an_unknown_zone_name_is_refused(tz):
    times = pd.date_range("2020-09-01", periods=4, freq="6h")
    with pytest.raises(ValueError, match="time-zone database"):
        partition_daynight(times, SLC_LAT, SLC_LON, tz=tz)


@pytest.mark.parametrize("tz", [True, None, [1]])
def test_an_unusable_time_zone_type_is_refused(tz):
    times = pd.date_range("2020-09-01", periods=4, freq="6h")
    with pytest.raises((TypeError, ValueError)):
        partition_daynight(times, SLC_LAT, SLC_LON, tz=tz)


def test_an_absurd_offset_is_refused():
    times = pd.date_range("2020-09-01", periods=4, freq="6h")
    with pytest.raises(ValueError, match="within \\+/- 24"):
        partition_daynight(times, SLC_LAT, SLC_LON, tz=99)


def test_the_timestamps_must_be_a_datetime_index():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        partition_daynight(
            pd.DataFrame({"a": [1]}, index=[0]), SLC_LAT, SLC_LON, tz=SLC_TZ
        )
    with pytest.raises(TypeError, match="DataFrame"):
        partition_daynight([1, 2, 3], SLC_LAT, SLC_LON, tz=SLC_TZ)


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(91.0, 0.0), (np.nan, 0.0), (0.0, 400.0), (0.0, np.inf)],
)
def test_an_out_of_range_geolocation_is_refused(latitude, longitude):
    times = pd.date_range("2020-09-01", periods=4, freq="6h")
    with pytest.raises(ValueError, match="latitude|longitude"):
        partition_daynight(times, latitude, longitude, tz=SLC_TZ)


def test_the_geolocation_is_required_without_a_radiation_column():
    times = pd.date_range("2020-09-01", periods=4, freq="6h")
    with pytest.raises(ValueError, match="geolocation"):
        partition_daynight(times, tz=SLC_TZ)


# ------------------------------
# The day-night flag
# ------------------------------


def test_the_flag_is_a_named_boolean_series_on_the_input_index():
    times = pd.date_range("2020-09-01", periods=48, freq="30min")
    daytime = partition_daynight(times, SLC_LAT, SLC_LON, tz=SLC_TZ)

    assert daytime.dtype == bool
    assert daytime.name == DAYTIME == "daytime"
    pd.testing.assert_index_equal(daytime.index, times)


def test_the_threshold_is_strictly_positive_radiation():
    # Chu et al. (2021), Sect. 2.2: potential incoming radiation > 0 W m-2.
    times = pd.date_range("2020-09-01", periods=3, freq="1h")
    frame = pd.DataFrame({SW_IN_POT_COLUMN: [-1.0, 0.0, 1e-9]}, index=times)
    assert partition_daynight(frame).tolist() == [False, False, True]


def test_a_precomputed_column_is_preferred_over_the_geometry():
    times = pd.date_range("2020-06-21", periods=4, freq="6h")
    # Deliberately at odds with the Sun, to show which one wins.
    frame = pd.DataFrame({SW_IN_POT_COLUMN: [100.0, 0.0, 0.0, 100.0]}, index=times)

    assert partition_daynight(frame, SLC_LAT, SLC_LON, tz=SLC_TZ).tolist() == [
        True,
        False,
        False,
        True,
    ]
    # ...and the geometry disagrees, so the column really did decide it.
    assert partition_daynight(
        frame, SLC_LAT, SLC_LON, tz=SLC_TZ, sw_in_pot=None
    ).tolist() == [False, True, True, True]


def test_the_column_is_matched_ignoring_case():
    times = pd.date_range("2020-09-01", periods=2, freq="12h")
    frame = pd.DataFrame({"sw_in_pot": [0.0, 500.0]}, index=times)
    assert partition_daynight(frame).tolist() == [False, True]


def test_a_column_needs_no_geolocation_or_time_zone():
    times = pd.date_range("2020-09-01", periods=2, freq="12h")
    frame = pd.DataFrame({SW_IN_POT_COLUMN: [0.0, 500.0]}, index=times)
    assert partition_daynight(frame).tolist() == [False, True]


def test_gaps_in_the_column_are_filled_from_the_geometry():
    times = pd.date_range("2020-06-21", periods=8, freq="3h")
    geometry = partition_daynight(times, SLC_LAT, SLC_LON, tz=SLC_TZ)
    radiation = potential_radiation(times, SLC_LAT, SLC_LON, tz=SLC_TZ)
    gapped = radiation.copy()
    gapped.iloc[2:5] = np.nan

    filled = partition_daynight(
        pd.DataFrame({SW_IN_POT_COLUMN: gapped}), SLC_LAT, SLC_LON, tz=SLC_TZ
    )
    np.testing.assert_array_equal(filled.to_numpy(), geometry.to_numpy())


def test_gaps_without_a_geolocation_are_refused():
    times = pd.date_range("2020-09-01", periods=3, freq="6h")
    frame = pd.DataFrame({SW_IN_POT_COLUMN: [0.0, np.nan, 500.0]}, index=times)
    with pytest.raises(ValueError, match="missing values"):
        partition_daynight(frame)


def test_day_and_night_partition_the_record():
    times = pd.date_range("2020-01-01", periods=24 * 30, freq="1h")
    daytime = partition_daynight(times, SLC_LAT, SLC_LON, tz=SLC_TZ)
    assert int(daytime.sum()) + int((~daytime).sum()) == len(times)
    # A January month at 41 N: short days, so nights dominate.
    assert 0.3 < daytime.mean() < 0.45


# ------------------------------
# Monthly climatologies
# ------------------------------


def test_climatologies_carry_the_documented_dims_and_coords(quarter):
    clim = monthly_climatologies(
        footprint_series(quarter), latitude=SLC_LAT, longitude=SLC_LON, tz=SLC_TZ
    )

    assert clim.footprint_climatology.dims == ("month", "period", "x", "y")
    assert clim.period.values.tolist() == [DAYTIME, NIGHTTIME]
    np.testing.assert_array_equal(
        clim.month.values,
        pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
    )
    np.testing.assert_array_equal(
        clim.x.values, np.arange(-EXTENT, EXTENT + STEP / 2, STEP)
    )


def test_every_populated_group_is_renormalised_to_unit_sum(quarter):
    clim = monthly_climatologies(
        footprint_series(quarter), latitude=SLC_LAT, longitude=SLC_LON, tz=SLC_TZ
    )
    sums = clim.footprint_climatology.sum(dim=("x", "y"))
    np.testing.assert_allclose(sums.values, 1.0, atol=1e-12)


def test_a_group_is_the_truncated_mean_of_its_contributing_footprints(quarter):
    f_2d = footprint_series(quarter)
    daytime = partition_daynight(quarter, SLC_LAT, SLC_LON, tz=SLC_TZ).to_numpy()
    clim = monthly_climatologies(
        f_2d, latitude=SLC_LAT, longitude=SLC_LON, tz=SLC_TZ
    )

    january = (quarter.month == 1) & (quarter.year == 2020)
    chosen = np.flatnonzero(january & daytime)
    expected = truncate_to_contour(
        f_2d.isel(time=chosen).sum(dim="time") / chosen.size, fraction=0.8
    )
    np.testing.assert_allclose(
        clim.footprint_climatology.isel(month=0).sel(period=DAYTIME).values,
        expected.values,
        atol=1e-12,
    )


def test_the_group_counts_add_up_to_the_record(quarter):
    clim = monthly_climatologies(
        footprint_series(quarter), latitude=SLC_LAT, longitude=SLC_LON, tz=SLC_TZ
    )
    assert int(clim.n_times.sum()) == len(quarter)
    assert bool((clim.n_times > 0).all())


def test_nighttime_climatologies_reach_farther_and_cover_more(quarter):
    # Chu et al. (2021), Sect. 3.1: in over 95 % of site-years, the nighttime
    # climatology extends about 45 % farther and covers about 90 % more area.
    clim = monthly_climatologies(
        footprint_series(quarter), latitude=SLC_LAT, longitude=SLC_LON, tz=SLC_TZ
    )
    weights = clim.footprint_climatology.isel(month=0)
    day = climatology_metrics(weights.sel(period=DAYTIME))
    night = climatology_metrics(weights.sel(period=NIGHTTIME))

    assert night.fetch > day.fetch
    assert night.area > day.area


def test_the_groups_feed_the_overlap_indices_directly(quarter):
    clim = monthly_climatologies(
        footprint_series(quarter), latitude=SLC_LAT, longitude=SLC_LON, tz=SLC_TZ
    )
    day = clim.footprint_climatology.sel(period=DAYTIME)
    night = clim.footprint_climatology.sel(period=NIGHTTIME)

    # The fixture's shape depends only on the flag, so every month's daytime
    # climatology is identical and Eq. 2 returns exactly one.
    assert seasonal_overlap(day) == pytest.approx(1.0)
    assert seasonal_overlap(night) == pytest.approx(1.0)
    # Day and night differ in width and offset, so Eq. 3 falls short of one.
    assert 0.0 < daynight_overlap(day, night) < 1.0


def test_the_contour_is_recorded_for_every_group(quarter):
    f_2d = footprint_series(quarter)
    clim = monthly_climatologies(
        f_2d, latitude=SLC_LAT, longitude=SLC_LON, tz=SLC_TZ
    )
    weights = clim.footprint_climatology.isel(month=0).sel(period=DAYTIME)

    assert clim.attrs["contour_fraction"] == 0.8
    assert clim.attrs["partition"] == "month+daynight"
    assert int(clim.contour_n_cells.isel(month=0).sel(period=DAYTIME)) == int(
        (weights > 0).sum()
    )
    assert float(clim.contour_level.isel(month=0).sel(period=DAYTIME)) > 0.0


def test_a_narrower_contour_keeps_fewer_cells(quarter):
    f_2d = footprint_series(quarter)
    counts = [
        int(
            monthly_climatologies(
                f_2d,
                latitude=SLC_LAT,
                longitude=SLC_LON,
                tz=SLC_TZ,
                fraction=fraction,
            ).contour_n_cells.isel(month=0).sel(period=DAYTIME)
        )
        for fraction in (0.5, 0.8, 0.95)
    ]
    assert counts[0] < counts[1] < counts[2]


def test_months_are_calendar_months_of_a_year_not_months_of_the_year():
    times = pd.DatetimeIndex(["2019-01-15 12:00", "2020-01-15 12:00"])
    clim = monthly_climatologies(
        footprint_series(times, daytime=np.array([True, True])),
        partition="month",
    )
    np.testing.assert_array_equal(
        clim.month.values, pd.DatetimeIndex(["2019-01-01", "2020-01-01"])
    )


def test_pooling_every_hour_gives_a_single_period(quarter):
    clim = monthly_climatologies(footprint_series(quarter), partition="month")

    assert clim.period.values.tolist() == [ALL_HOURS]
    assert clim.footprint_climatology.dims == ("month", "period", "x", "y")
    assert int(clim.n_times.sum()) == len(quarter)
    assert clim.attrs["partition"] == "month"


def test_pooling_needs_no_geolocation(quarter):
    # Without a day-night split there is no Sun to place, so a bare array works.
    clim = monthly_climatologies(footprint_series(quarter), partition="month")
    assert bool(np.isfinite(clim.footprint_climatology).any())


def test_an_empty_group_is_nan_rather_than_zero():
    # A single all-daytime record leaves the nighttime group empty.
    times = pd.date_range("2020-06-21 12:00", periods=4, freq="1h")
    clim = monthly_climatologies(
        footprint_series(times, daytime=np.ones(4, dtype=bool)),
        latitude=SLC_LAT,
        longitude=SLC_LON,
        tz=SLC_TZ,
        daytime=np.ones(4, dtype=bool),
    )
    night = clim.footprint_climatology.sel(period=NIGHTTIME)

    assert int(clim.n_times.isel(month=0).sel(period=NIGHTTIME)) == 0
    assert bool(np.isnan(night).all())
    assert bool(np.isfinite(clim.footprint_climatology.sel(period=DAYTIME)).all())


def test_a_thin_group_is_held_back_by_min_times():
    times = pd.date_range("2020-06-21 00:00", periods=7, freq="3h")
    flag = np.array([False, False, True, True, True, False, False])
    f_2d = footprint_series(times, daytime=flag)

    # Three daytime records fall short of the floor; four nighttime ones clear it.
    clim = monthly_climatologies(f_2d, daytime=flag, min_times=4)
    assert int(clim.n_times.isel(month=0).sel(period=DAYTIME)) == 3
    assert int(clim.n_times.isel(month=0).sel(period=NIGHTTIME)) == 4
    assert bool(np.isnan(clim.footprint_climatology.sel(period=DAYTIME)).all())
    assert bool(np.isfinite(clim.footprint_climatology.sel(period=NIGHTTIME)).all())


def test_empty_groups_can_be_dropped_before_the_overlap_indices():
    times = pd.date_range("2020-06-21 12:00", periods=4, freq="1h")
    clim = monthly_climatologies(
        footprint_series(times, daytime=np.ones(4, dtype=bool)),
        daytime=np.ones(4, dtype=bool),
    )
    kept = clim.where(clim.n_times > 0, drop=True)
    assert kept.period.values.tolist() == [DAYTIME]


def test_timesteps_carrying_no_weight_are_left_out_of_the_denominator():
    times = pd.date_range("2020-06-21 00:00", periods=4, freq="6h")
    flag = np.ones(4, dtype=bool)
    f_2d = footprint_series(times, daytime=flag)
    blanked = f_2d.copy()
    blanked[2] = 0.0  # e.g. a timestep outside the model's validity bounds

    clim = monthly_climatologies(blanked, daytime=flag, partition="month")
    assert int(clim.n_times.isel(month=0).sel(period=ALL_HOURS)) == 3
    # The three live timesteps are identical, so the climatology is one of them.
    expected = truncate_to_contour(f_2d.isel(time=0), fraction=0.8)
    np.testing.assert_allclose(
        clim.footprint_climatology.isel(month=0).sel(period=ALL_HOURS).values,
        expected.values,
        atol=1e-12,
    )


# ------------------------------
# Accepted inputs
# ------------------------------


def test_a_model_its_dataset_and_its_array_all_work(quarter):
    f_2d = footprint_series(quarter)
    expected = monthly_climatologies(
        f_2d, latitude=SLC_LAT, longitude=SLC_LON, tz=SLC_TZ
    )

    for source in (
        FakeModel(f_2d),
        xr.Dataset({"footprint_2d": f_2d}),
    ):
        xr.testing.assert_allclose(
            monthly_climatologies(
                source, latitude=SLC_LAT, longitude=SLC_LON, tz=SLC_TZ
            ),
            expected,
        )


def test_a_models_radiation_column_is_picked_up(quarter):
    f_2d = footprint_series(quarter)
    # A column that calls everything daytime, so it cannot be confused with
    # the geometry it overrides.
    frame = pd.DataFrame(
        {SW_IN_POT_COLUMN: np.full(len(quarter), 500.0)}, index=quarter
    )
    clim = monthly_climatologies(FakeModel(f_2d, frame), partition="month+daynight")

    assert int(clim.n_times.sel(period=NIGHTTIME).sum()) == 0
    assert int(clim.n_times.sel(period=DAYTIME).sum()) == len(quarter)


def test_a_precomputed_flag_is_used_as_given(quarter):
    f_2d = footprint_series(quarter)
    flag = pd.Series(np.arange(len(quarter)) % 2 == 0, index=quarter)
    clim = monthly_climatologies(f_2d, daytime=flag)

    assert int(clim.n_times.sel(period=DAYTIME).sum()) == int(flag.sum())


def test_a_flag_that_misses_timestamps_is_refused(quarter):
    f_2d = footprint_series(quarter)
    flag = pd.Series(True, index=quarter[:-3])
    with pytest.raises(ValueError, match="does not cover"):
        monthly_climatologies(f_2d, daytime=flag)


def test_a_flag_of_the_wrong_length_is_refused(quarter):
    with pytest.raises(ValueError, match="flags for"):
        monthly_climatologies(footprint_series(quarter), daytime=np.ones(3, dtype=bool))


def test_an_unrecognised_partition_is_refused(quarter):
    with pytest.raises(ValueError, match="partition must be"):
        monthly_climatologies(footprint_series(quarter), partition="week")


def test_a_nonpositive_min_times_is_refused(quarter):
    with pytest.raises(ValueError, match="min_times"):
        monthly_climatologies(footprint_series(quarter), partition="month", min_times=0)


def test_a_model_that_has_not_been_run_is_refused():
    with pytest.raises(ValueError, match="has not been run"):
        monthly_climatologies(FakeModel(None), partition="month")


def test_a_dataset_without_time_resolved_footprints_is_refused():
    empty = xr.Dataset({"footprint_climatology": (("x", "y"), np.ones((3, 3)))})
    with pytest.raises(ValueError, match="no time-resolved footprint"):
        monthly_climatologies(empty, partition="month")


def test_the_wrong_dims_are_refused(quarter):
    collapsed = footprint_series(quarter).sum(dim="time")
    with pytest.raises(ValueError, match=r"dims \(time, x, y\)"):
        monthly_climatologies(collapsed, partition="month")


def test_an_unusable_input_type_is_refused():
    with pytest.raises(TypeError, match="footprint model"):
        monthly_climatologies(np.ones((2, 3, 3)), partition="month")


def test_footprints_without_a_time_coordinate_are_refused(quarter):
    f_2d = footprint_series(quarter).drop_vars("time")
    with pytest.raises(ValueError, match="no 'time' coordinate"):
        monthly_climatologies(f_2d, partition="month")
