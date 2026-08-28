"""
Weighted-statistics tests for :mod:`fluxfootprints.representativeness`.

The fixtures are chosen so that every expected value can be written down by
hand rather than read off a previous run:

* a constant field must come back as that constant under any weighting, since
  the weights are renormalised to sum to 1;
* a field that is 0 west of the tower and 1 east of it must come back as 0.5
  under any footprint symmetric about ``x = 0``, and over any target disc,
  because the two halves carry equal weight;
* a field that is nodata over one of those halves must come back as the other
  half's value with half the weight retained.

The grid deliberately has no cell centred on ``x = 0``, so the half-and-half
split falls on a cell boundary and neither half gets an extra column.

The last section runs the same four functions on a raster brought in through
:func:`~fluxfootprints.sample_raster_on_grid`, the alignment path they are
meant to consume, and is skipped without rasterio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxfootprints.representativeness import (
    WeightedValue,
    footprint_weighted_composition,
    footprint_weighted_value,
    sample_raster_on_grid,
    target_area_composition,
    target_area_mask,
    target_area_value,
)

STATION_LAT = 40.0
STATION_LON = -111.9

#: 20 x 20 cells of 10 m, centred on the tower and skipping x = y = 0.
GRID = np.arange(-95.0, 100.0, 10.0)
STEP = 10.0
SIGMA = 50.0

XX, YY = np.meshgrid(GRID, GRID, indexing="ij")


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


def constant(value: float = 7.5) -> xr.DataArray:
    return field(np.full(XX.shape, value))


def half_and_half(west: float = 0.0, east: float = 1.0) -> xr.DataArray:
    """0 west of the tower, 1 east of it; the split is on a cell boundary."""
    return field(np.where(XX < 0.0, west, east))


def two_classes(west: int = 11, east: int = 42) -> xr.DataArray:
    return field(np.where(XX < 0.0, west, east), name="landcover")


# ----------------------------
# footprint_weighted_value, Eq. 5
# ----------------------------
def test_constant_field_returns_the_constant():
    result = footprint_weighted_value(weights(), constant(7.5))

    assert isinstance(result, WeightedValue)
    assert result.value == pytest.approx(7.5)
    assert result.retained_weight == pytest.approx(1.0)
    assert result.n_cells == GRID.size**2


def test_symmetric_footprint_over_a_half_and_half_field_gives_one_half():
    result = footprint_weighted_value(weights(), half_and_half())

    assert result.value == pytest.approx(0.5)
    assert result.retained_weight == pytest.approx(1.0)


def test_value_is_the_sum_of_weight_times_field():
    """Eq. 5 itself, on a footprint and a field with no symmetry to lean on."""
    w = weights(sigma_x=35.0, sigma_y=70.0, x0=25.0)
    ramp = field(XX + 2.0 * YY)

    result = footprint_weighted_value(w, ramp)

    assert result.value == pytest.approx(float((w * ramp).sum()))


def test_unnormalised_weights_are_renormalised():
    """A climatology of raw densities gives the same value as a unit-sum one."""
    w = weights()
    scaled = w * 0.37

    assert footprint_weighted_value(scaled, half_and_half()).value == pytest.approx(
        footprint_weighted_value(w, half_and_half()).value
    )
    assert footprint_weighted_value(scaled, constant()).retained_weight == 1.0


def test_nodata_cells_are_dropped_and_the_surviving_weights_renormalised():
    """The eastern half alone, carrying half of a symmetric footprint."""
    east_only = half_and_half(west=np.nan, east=3.0)

    result = footprint_weighted_value(weights(), east_only)

    assert result.value == pytest.approx(3.0)
    assert result.retained_weight == pytest.approx(0.5)
    assert result.n_cells == GRID.size**2 // 2


def test_nodata_outside_the_footprint_does_not_reduce_the_retained_weight():
    """Zero-weight cells carry no weight to lose, so coverage there is moot."""
    w = weights()
    truncated = w.where(np.hypot(XX, YY) <= 60.0, 0.0)
    patchy = constant(2.0).where(truncated > 0.0)

    result = footprint_weighted_value(truncated, patchy)

    assert result.value == pytest.approx(2.0)
    assert result.retained_weight == pytest.approx(1.0)
    assert result.n_cells == int((truncated > 0.0).sum())


def test_a_field_that_is_nodata_everywhere_gives_nan():
    result = footprint_weighted_value(weights(), field(np.full(XX.shape, np.nan)))

    assert np.isnan(result.value)
    assert result.retained_weight == 0.0
    assert result.n_cells == 0


def test_numpy_arrays_are_accepted_on_a_matching_shape():
    result = footprint_weighted_value(weights().values, half_and_half().values)

    assert result.value == pytest.approx(0.5)


def test_a_field_on_another_grid_raises():
    other = xr.DataArray(
        np.zeros((GRID.size, GRID.size)),
        coords={"x": GRID + 1000.0, "y": GRID},
        dims=("x", "y"),
    )

    with pytest.raises(ValueError, match="different grids"):
        footprint_weighted_value(weights(), other)


def test_a_field_of_another_shape_raises():
    with pytest.raises(ValueError, match="different grids"):
        footprint_weighted_value(weights(), constant().isel(x=slice(1, None)))


def test_weights_carrying_no_source_weight_raise():
    with pytest.raises(ValueError, match="no source weight"):
        footprint_weighted_value(field(np.zeros(XX.shape)), constant())


def test_negative_weights_raise():
    with pytest.raises(ValueError, match="negative weights"):
        footprint_weighted_value(field(XX), constant())


# ----------------------------
# footprint_weighted_composition
# ----------------------------
def test_composition_splits_a_half_and_half_raster_evenly():
    composition = footprint_weighted_composition(weights(), two_classes())

    assert isinstance(composition, pd.Series)
    assert composition.sum() == pytest.approx(1.0)
    assert composition.loc[11] == pytest.approx(0.5)
    assert composition.loc[42] == pytest.approx(0.5)


def test_composition_is_indexed_by_integer_class_code_in_order():
    composition = footprint_weighted_composition(weights(), two_classes(81, 41))

    assert composition.index.name == "class"
    assert composition.index.dtype == np.int64
    assert list(composition.index) == [41, 81]


def test_composition_of_a_single_class_is_all_of_it():
    composition = footprint_weighted_composition(
        weights(), field(np.full(XX.shape, 90))
    )

    assert list(composition.index) == [90]
    assert composition.loc[90] == pytest.approx(1.0)


def test_composition_records_the_retained_weight_in_attrs():
    landcover = two_classes().where(XX > 0.0)

    composition = footprint_weighted_composition(weights(), landcover)

    assert composition.sum() == pytest.approx(1.0)
    assert composition.loc[42] == pytest.approx(1.0)
    assert composition.attrs["retained_weight"] == pytest.approx(0.5)
    assert composition.attrs["n_cells"] == GRID.size**2 // 2


def test_composition_weights_by_source_strength_not_by_area():
    """A patch at the tower takes a share far above the area it covers."""
    landcover = field(np.where(np.hypot(XX, YY) <= 30.0, 2, 1))
    w = weights(sigma_x=20.0, sigma_y=20.0)

    weighted = footprint_weighted_composition(w, landcover)
    areal = target_area_composition(landcover, GRID, GRID, 95.0)

    assert weighted.loc[2] > 3.0 * areal.loc[2]


def test_composition_of_an_entirely_nodata_raster_is_empty():
    composition = footprint_weighted_composition(
        weights(), field(np.full(XX.shape, np.nan))
    )

    assert composition.empty
    assert composition.attrs["retained_weight"] == 0.0


def test_non_integral_class_codes_stay_floats():
    composition = footprint_weighted_composition(
        weights(), field(np.where(XX < 0.0, 1.5, 2.5))
    )

    assert composition.index.dtype == np.float64
    assert list(composition.index) == [1.5, 2.5]


# ----------------------------
# target_area_value
# ----------------------------
def test_target_area_of_a_constant_field_returns_the_constant():
    result = target_area_value(constant(7.5), GRID, GRID, 50.0)

    assert result.value == pytest.approx(7.5)
    assert result.retained_weight == pytest.approx(1.0)
    assert result.n_cells == int(target_area_mask(GRID, GRID, 50.0).sum())


def test_target_area_over_a_half_and_half_field_gives_one_half():
    """The disc is symmetric about x = 0, so the two halves cancel exactly."""
    result = target_area_value(half_and_half(), GRID, GRID, 50.0)

    assert result.value == pytest.approx(0.5)


def test_target_area_uses_only_the_cells_inside_the_disc():
    inner = field(np.where(np.hypot(XX, YY) <= 40.0, 1.0, 0.0))

    near = target_area_value(inner, GRID, GRID, 40.0)
    far = target_area_value(inner, GRID, GRID, 90.0)

    assert near.value == pytest.approx(1.0)
    assert far.value < 0.5
    assert far.n_cells > near.n_cells


def test_target_area_is_unweighted():
    """Unlike Eq. 5, every cell of the disc counts the same."""
    ramp = field(np.hypot(XX, YY))
    mask = target_area_mask(GRID, GRID, 70.0)

    result = target_area_value(ramp, GRID, GRID, 70.0)

    assert result.value == pytest.approx(float(ramp.where(mask).mean()))


def test_target_area_ignores_nodata_and_reports_the_coverage():
    east_only = half_and_half(west=np.nan, east=3.0)

    result = target_area_value(east_only, GRID, GRID, 50.0)

    assert result.value == pytest.approx(3.0)
    assert result.retained_weight == pytest.approx(0.5)


def test_target_area_of_an_entirely_nodata_raster_is_nan():
    result = target_area_value(field(np.full(XX.shape, np.nan)), GRID, GRID, 50.0)

    assert np.isnan(result.value)
    assert result.retained_weight == 0.0
    assert result.n_cells == 0


def test_a_disc_larger_than_the_domain_is_clipped_to_it():
    result = target_area_value(constant(1.0), GRID, GRID, 3000.0)

    assert result.n_cells == GRID.size**2
    assert result.retained_weight == pytest.approx(1.0)


def test_a_disc_smaller_than_a_cell_raises():
    with pytest.raises(ValueError, match="No cell centre"):
        target_area_value(constant(), GRID, GRID, 3.0)


@pytest.mark.parametrize("radius", [0.0, -100.0, np.nan])
def test_a_radius_that_is_not_positive_raises(radius):
    with pytest.raises(ValueError, match="radius must be positive"):
        target_area_value(constant(), GRID, GRID, radius)


def test_a_raster_on_another_grid_raises():
    with pytest.raises(ValueError, match="different grids"):
        target_area_value(constant(), GRID + 1000.0, GRID, 50.0)


# ----------------------------
# target_area_composition
# ----------------------------
def test_target_composition_splits_a_half_and_half_raster_evenly():
    composition = target_area_composition(two_classes(), GRID, GRID, 50.0)

    assert composition.sum() == pytest.approx(1.0)
    assert composition.loc[11] == pytest.approx(0.5)
    assert composition.loc[42] == pytest.approx(0.5)


def test_target_composition_counts_cells_not_weight():
    landcover = field(np.where(np.hypot(XX, YY) <= 40.0, 2, 1))
    mask = target_area_mask(GRID, GRID, 90.0)

    composition = target_area_composition(landcover, GRID, GRID, 90.0)

    inside = int(mask.sum())
    assert composition.loc[2] == pytest.approx(
        int((landcover.where(mask) == 2).sum()) / inside
    )


def test_target_composition_records_the_classified_fraction_of_the_disc():
    landcover = two_classes().where(XX > 0.0)

    composition = target_area_composition(landcover, GRID, GRID, 50.0)

    assert composition.loc[42] == pytest.approx(1.0)
    assert composition.attrs["retained_weight"] == pytest.approx(0.5)


def test_target_composition_of_an_entirely_nodata_raster_is_empty():
    composition = target_area_composition(
        field(np.full(XX.shape, np.nan)), GRID, GRID, 50.0
    )

    assert composition.empty
    assert composition.attrs["retained_weight"] == 0.0


# ----------------------------
# target_area_mask
# ----------------------------
def test_the_mask_covers_about_pi_r_squared():
    radius = 70.0
    mask = target_area_mask(GRID, GRID, radius)

    area = float(mask.sum()) * STEP**2
    assert area == pytest.approx(np.pi * radius**2, rel=0.05)


def test_the_mask_lands_on_the_grid_it_was_given():
    mask = target_area_mask(GRID, GRID, 50.0)

    assert mask.dims == ("x", "y")
    assert np.array_equal(mask["x"].values, GRID)
    assert mask.dtype == np.bool_
    assert mask.attrs["radius"] == 50.0


def test_a_meshgrid_instead_of_axes_raises():
    with pytest.raises(ValueError, match="1-D array"):
        target_area_mask(XX, YY, 50.0)


# ----------------------------
# The alignment path of sample_raster_on_grid
# ----------------------------
@pytest.fixture
def write_raster(tmp_path):
    """Write a synthetic GeoTIFF over the tower domain, in the grid's own CRS."""
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import rasterio
    from rasterio.transform import from_origin

    from fluxfootprints.openet_masking import footprint_grid_geometry

    geometry = footprint_grid_geometry(GRID, GRID, STATION_LAT, STATION_LON)

    def _write(name, values, left_offset=-200.0, dtype="float32", nodata=None):
        values = np.asarray(values)
        height, width = values.shape
        path = tmp_path / name
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dtype,
            "crs": geometry.crs,
            "transform": from_origin(
                geometry.x_origin + left_offset, geometry.y_origin + 200.0, 10.0, 10.0
            ),
        }
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(values.astype(dtype), 1)
        return path

    return _write


def aligned(path, categorical=False):
    return sample_raster_on_grid(
        path, GRID, GRID, STATION_LAT, STATION_LON, categorical=categorical
    )


def test_an_aligned_constant_raster_feeds_both_value_functions(write_raster):
    path = write_raster("const.tif", np.full((40, 40), 7.5))
    raster = aligned(path)

    assert raster.dims == ("x", "y")
    assert footprint_weighted_value(weights(), raster).value == pytest.approx(7.5)
    assert target_area_value(raster, GRID, GRID, 500.0).value == pytest.approx(7.5)


def test_an_aligned_class_raster_feeds_both_composition_functions(write_raster):
    """Vertical stripes of NLCD-like codes, two of them over the grid."""
    codes = np.where(np.arange(40) < 20, 41, 81)
    path = write_raster("codes.tif", np.tile(codes, (40, 1)), dtype="int16")
    raster = aligned(path, categorical=True)

    weighted = footprint_weighted_composition(weights(), raster)
    areal = target_area_composition(raster, GRID, GRID, 95.0)

    assert list(weighted.index) == [41, 81]
    assert weighted.sum() == pytest.approx(1.0)
    assert weighted.loc[41] == pytest.approx(0.5)
    assert areal.loc[41] == pytest.approx(0.5)


def test_a_raster_covering_half_the_grid_halves_the_retained_weight(write_raster):
    """The source starts at the tower, so only the eastern half is covered."""
    path = write_raster("east.tif", np.full((40, 40), 3.0), left_offset=0.0)
    raster = aligned(path)

    value = footprint_weighted_value(weights(), raster)
    target = target_area_value(raster, GRID, GRID, 500.0)

    assert value.value == pytest.approx(3.0)
    assert value.retained_weight == pytest.approx(0.5)
    assert target.value == pytest.approx(3.0)
    assert target.retained_weight == pytest.approx(0.5)


def test_the_retained_weight_matches_the_aligned_rasters_own_coverage(write_raster):
    path = write_raster("east.tif", np.full((40, 40), 3.0), left_offset=0.0)
    raster = aligned(path)
    w = weights(sigma_x=40.0, sigma_y=80.0, x0=-15.0)

    result = footprint_weighted_value(w, raster)

    assert result.retained_weight == pytest.approx(
        float(w.where(raster.notnull()).sum())
    )
    assert result.n_cells == int(raster.notnull().sum())


def test_declared_nodata_in_the_source_is_honoured(write_raster):
    values = np.full((40, 40), 3.0)
    values[:20, :] = -9999.0  # the northern half of the source
    path = write_raster("north_nodata.tif", values, nodata=-9999.0)
    raster = aligned(path)

    result = target_area_value(raster, GRID, GRID, 500.0)

    assert result.value == pytest.approx(3.0)
    assert result.retained_weight == pytest.approx(0.5)
