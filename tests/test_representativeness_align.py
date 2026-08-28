"""
Raster-alignment tests for :mod:`fluxfootprints.representativeness`.

The fixtures are small synthetic GeoTIFFs written to ``tmp_path`` in the
tower's own UTM zone, so what ``_align_raster`` has to do is a pure resample
rather than a datum shift, and the expected values can be written down by hand:

* a constant field must survive any resampling unchanged;
* a field that is nodata over the western half of the domain must come back
  valid over exactly the eastern half of the footprint grid;
* class codes must survive nearest-neighbour resampling as the same codes,
  while a linear ramp must come back interpolated under bilinear.

The source rasters are offset by half a cell from the footprint grid so that no
footprint cell centre ever lands on a source cell boundary, which keeps the
nearest-neighbour result exact and free of tie-breaking; the same trick as in
``test_openet_masking.py``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import xarray as xr

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("rioxarray")

import rioxarray  # also registers the .rio accessor used below
from rasterio.transform import from_origin

from fluxfootprints.openet_masking import footprint_grid_geometry
from fluxfootprints.representativeness import _align_raster

STATION_LAT = 40.0
STATION_LON = -111.9

#: Footprint grid: 21 x 21 cells of 10 m, centred on the tower.
GRID = np.arange(-100.0, 100.0 + 10.0, 10.0)

#: Source rasters: 40 x 40 cells of 10 m, so they overhang the footprint grid.
SOURCE_RES = 10.0
SOURCE_HALF = 200.0
SOURCE_N = int(2 * SOURCE_HALF / SOURCE_RES)

NODATA = -9999.0


# ----------------------------
# Fixtures
# ----------------------------
def _geometry(x=GRID, y=GRID, crs="auto"):
    return footprint_grid_geometry(x, y, STATION_LAT, STATION_LON, crs=crs)


def footprint_grid(x=GRID, y=GRID, crs="auto", write_crs=True):
    """A georeferenced footprint grid, in projected metres, ready for .rio."""
    geom = _geometry(x, y, crs)
    xs = geom.x_origin + np.asarray(x, dtype=float)
    ys = geom.y_origin + np.asarray(y, dtype=float)
    grid = xr.DataArray(
        np.zeros((ys.size, xs.size)),
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
        name="footprint",
    )
    return grid.rio.write_crs(geom.crs) if write_crs else grid


def write_raster(
    path,
    values,
    res=SOURCE_RES,
    half=SOURCE_HALF,
    nodata=NODATA,
    dtype="float32",
    crs=None,
    count=1,
):
    """Write a synthetic GeoTIFF covering the tower domain, half a cell off."""
    geom = _geometry()
    values = np.asarray(values)
    height, width = values.shape[-2:]
    transform = from_origin(
        geom.x_origin - half - res / 2.0,
        geom.y_origin + half + res / 2.0,
        res,
        res,
    )
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": geom.crs if crs is None else crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        if count == 1:
            dst.write(values.astype(dtype), 1)
        else:
            dst.write(values.astype(dtype))
    return path


def constant_raster(path, value=7.5, **kwargs):
    return write_raster(path, np.full((SOURCE_N, SOURCE_N), value), **kwargs)


def west_nodata_raster(path, value=3.0, nodata=NODATA):
    """Valid over the eastern half of the domain, nodata over the western."""
    values = np.full((SOURCE_N, SOURCE_N), value, dtype="float32")
    values[:, : SOURCE_N // 2] = nodata
    return write_raster(path, values, nodata=nodata)


def ramp_raster(path):
    """Column index as the value: a west-to-east linear ramp."""
    ramp = np.tile(np.arange(SOURCE_N, dtype="float32"), (SOURCE_N, 1))
    return write_raster(path, ramp, nodata=None)


def code_raster(path, codes=(11, 42, 81, 90)):
    """Vertical stripes of NLCD-like class codes, one stripe per code."""
    stripe = SOURCE_N // len(codes)
    values = np.repeat(np.asarray(codes, dtype="int16"), stripe)
    return write_raster(
        path,
        np.tile(values, (SOURCE_N, 1)),
        nodata=0,
        dtype="int16",
    )


# ----------------------------
# Alignment onto the grid
# ----------------------------
def test_aligned_raster_lands_on_the_footprint_grid(tmp_path):
    grid = footprint_grid()
    aligned, valid = _align_raster(constant_raster(tmp_path / "const.tif"), grid)

    assert aligned.dims == grid.dims
    assert aligned.shape == grid.shape
    assert np.array_equal(aligned["x"].values, grid["x"].values)
    assert np.array_equal(aligned["y"].values, grid["y"].values)
    assert valid.dims == grid.dims
    assert valid.shape == grid.shape


def test_aligned_raster_multiplies_against_the_footprint_without_realignment(tmp_path):
    """The coords must match exactly, or xarray's inner join would empty this."""
    grid = footprint_grid()
    aligned, _ = _align_raster(constant_raster(tmp_path / "const.tif"), grid)

    assert (aligned * (grid + 1.0)).shape == grid.shape


def test_constant_field_survives_resampling(tmp_path):
    grid = footprint_grid()
    aligned, valid = _align_raster(constant_raster(tmp_path / "const.tif"), grid)

    assert np.allclose(aligned.values, 7.5)
    assert bool(valid.all())


def test_aligned_raster_is_float_whatever_the_source_dtype(tmp_path):
    grid = footprint_grid()
    aligned, valid = _align_raster(
        code_raster(tmp_path / "codes.tif"), grid, categorical=True
    )

    assert aligned.dtype == np.float64
    assert valid.dtype == np.bool_


def test_accepts_an_already_opened_dataarray(tmp_path):
    grid = footprint_grid()
    path = constant_raster(tmp_path / "const.tif")
    opened = rioxarray.open_rasterio(path, masked=True)

    from_path, _ = _align_raster(path, grid)
    from_array, _ = _align_raster(opened, grid)

    assert np.array_equal(from_path.values, from_array.values)


def test_reprojects_a_source_in_another_crs(tmp_path):
    """A geographic source is fine; it is the target grid that must be metric."""
    grid = footprint_grid()
    path = constant_raster(tmp_path / "geographic.tif")
    lonlat = rioxarray.open_rasterio(path, masked=True).rio.reproject("EPSG:4326")

    aligned, valid = _align_raster(lonlat, grid)

    assert bool(valid.all())
    assert np.allclose(aligned.values, 7.5)


# ----------------------------
# Resampling choice
# ----------------------------
def test_categorical_resampling_preserves_class_codes(tmp_path):
    grid = footprint_grid()
    aligned, valid = _align_raster(
        code_raster(tmp_path / "codes.tif"), grid, categorical=True
    )

    present = np.unique(aligned.values[valid.values])
    assert set(present.tolist()) <= {11.0, 42.0, 81.0, 90.0}


def test_continuous_resampling_interpolates_between_source_cells(tmp_path):
    """Bilinear on a ramp gives values the nearest-neighbour result never has."""
    grid = footprint_grid(x=GRID + 5.0, y=GRID + 5.0)
    path = ramp_raster(tmp_path / "ramp.tif")

    linear, _ = _align_raster(path, grid)
    nearest, _ = _align_raster(path, grid, categorical=True)

    assert np.allclose(nearest.values, np.round(nearest.values))
    assert not np.allclose(linear.values, nearest.values)
    assert np.any(np.abs(linear.values - np.round(linear.values)) > 0.1)


def test_categorical_resampling_invents_no_intermediate_codes(tmp_path):
    """Bilinear would blend 11 and 42 into codes that mean nothing."""
    grid = footprint_grid()
    path = code_raster(tmp_path / "codes.tif")

    nearest, near_valid = _align_raster(path, grid, categorical=True)
    linear, lin_valid = _align_raster(path, grid, categorical=False)

    assert set(np.unique(nearest.values[near_valid.values]).tolist()) <= {
        11.0,
        42.0,
        81.0,
        90.0,
    }
    blended = np.unique(linear.values[lin_valid.values])
    assert not set(blended.tolist()) <= {11.0, 42.0, 81.0, 90.0}


# ----------------------------
# Nodata
# ----------------------------
def test_nodata_marks_the_uncovered_half_invalid(tmp_path):
    grid = footprint_grid()
    aligned, valid = _align_raster(
        west_nodata_raster(tmp_path / "half.tif"), grid, categorical=True
    )

    west = valid.isel(x=slice(0, 10))
    east = valid.isel(x=slice(11, None))
    assert not bool(west.any())
    assert bool(east.all())
    assert np.all(np.isnan(aligned.isel(x=slice(0, 10)).values))
    assert np.allclose(aligned.isel(x=slice(11, None)).values, 3.0)


def test_valid_mask_is_exactly_where_the_aligned_values_are_finite(tmp_path):
    grid = footprint_grid()
    aligned, valid = _align_raster(
        west_nodata_raster(tmp_path / "half.tif"), grid, categorical=True
    )

    assert np.array_equal(valid.values, np.isfinite(aligned.values))


def test_grid_outside_the_source_comes_back_invalid(tmp_path):
    """A footprint grid the raster does not reach is all nodata, not an error."""
    far = GRID + 10_000.0
    grid = footprint_grid(x=far, y=far)

    aligned, valid = _align_raster(constant_raster(tmp_path / "const.tif"), grid)

    assert not bool(valid.any())
    assert np.all(np.isnan(aligned.values))


def test_undeclared_nodata_can_be_supplied_by_the_caller(tmp_path):
    grid = footprint_grid()
    values = np.full((SOURCE_N, SOURCE_N), 2.0, dtype="float32")
    values[:, : SOURCE_N // 2] = -1.0
    path = write_raster(tmp_path / "undeclared.tif", values, nodata=None)

    kept, kept_valid = _align_raster(path, grid, categorical=True)
    _, dropped_valid = _align_raster(path, grid, categorical=True, nodata=-1.0)

    assert bool(kept_valid.all())
    assert np.allclose(kept.isel(x=slice(0, 10)).values, -1.0)
    assert not bool(dropped_valid.isel(x=slice(0, 10)).any())
    assert bool(dropped_valid.isel(x=slice(11, None)).all())


def test_supplied_nodata_adds_to_the_declared_one(tmp_path):
    """Masking an extra value must not un-mask the raster's own nodata."""
    grid = footprint_grid()
    path = west_nodata_raster(tmp_path / "half.tif", value=3.0)

    _, valid = _align_raster(path, grid, categorical=True, nodata=3.0)

    assert not bool(valid.any())


def test_nodata_is_resolved_alike_for_a_path_and_an_opened_array(tmp_path):
    """open_rasterio has already applied the declared nodata; the argument
    must still mean the same thing on both routes."""
    grid = footprint_grid()
    path = west_nodata_raster(tmp_path / "half.tif", value=3.0)
    opened = rioxarray.open_rasterio(path, masked=False)

    from_path, path_valid = _align_raster(path, grid, categorical=True)
    from_array, array_valid = _align_raster(opened, grid, categorical=True)

    assert np.array_equal(path_valid.values, array_valid.values)
    assert np.array_equal(np.isnan(from_path.values), np.isnan(from_array.values))
    assert np.allclose(
        from_path.values[path_valid.values], from_array.values[array_valid.values]
    )


def test_bilinear_interpolates_from_the_valid_side_of_a_nodata_edge(tmp_path):
    """A boundary cell must not be dragged towards the nodata sentinel."""
    grid = footprint_grid()
    aligned, valid = _align_raster(west_nodata_raster(tmp_path / "half.tif"), grid)

    kept = aligned.values[valid.values]
    assert kept.size > 0
    assert np.allclose(kept, 3.0)


# ----------------------------
# Band selection
# ----------------------------
def test_band_selects_from_a_multiband_source(tmp_path):
    grid = footprint_grid()
    stack = np.stack(
        [np.full((SOURCE_N, SOURCE_N), value) for value in (1.0, 2.0, 3.0)]
    )
    path = write_raster(tmp_path / "stack.tif", stack, nodata=None, count=3)

    for band, expected in ((1, 1.0), (2, 2.0), (3, 3.0)):
        aligned, _ = _align_raster(path, grid, band=band)
        assert np.allclose(aligned.values, expected)


def test_band_out_of_range_is_an_error(tmp_path):
    grid = footprint_grid()
    path = constant_raster(tmp_path / "const.tif")

    with pytest.raises(ValueError, match="band 4 is out of range"):
        _align_raster(path, grid, band=4)
    with pytest.raises(ValueError, match="numbered from 1"):
        _align_raster(path, grid, band=0)


# ----------------------------
# Validation
# ----------------------------
def test_footprint_grid_without_a_crs_is_an_error(tmp_path):
    grid = footprint_grid(write_crs=False)

    with pytest.raises(ValueError, match="footprint grid: it carries no CRS"):
        _align_raster(constant_raster(tmp_path / "const.tif"), grid)


def test_source_raster_without_a_crs_is_an_error(tmp_path):
    grid = footprint_grid()
    bare = xr.DataArray(
        np.ones((SOURCE_N, SOURCE_N)),
        dims=("y", "x"),
        coords={
            "y": np.arange(SOURCE_N, dtype=float),
            "x": np.arange(SOURCE_N, dtype=float),
        },
    )

    with pytest.raises(ValueError, match="source raster: it carries no CRS"):
        _align_raster(bare, grid)


def test_geographic_footprint_grid_is_an_error(tmp_path):
    """Fetch, area, and the target-area radii are metres; degrees are not."""
    lonlat = xr.DataArray(
        np.zeros((5, 5)),
        dims=("y", "x"),
        coords={
            "y": np.linspace(STATION_LAT - 0.01, STATION_LAT + 0.01, 5),
            "x": np.linspace(STATION_LON - 0.01, STATION_LON + 0.01, 5),
        },
    ).rio.write_crs("EPSG:4326")

    with pytest.raises(ValueError, match="geographic CRS"):
        _align_raster(constant_raster(tmp_path / "const.tif"), lonlat)


def test_geographic_error_names_the_units_and_the_way_out(tmp_path):
    lonlat = footprint_grid().rio.write_crs("EPSG:4326", inplace=False)

    with pytest.raises(ValueError) as excinfo:
        _align_raster(constant_raster(tmp_path / "const.tif"), lonlat)

    message = str(excinfo.value)
    assert "degrees" in message
    assert "footprint_grid_geometry" in message


def test_footprint_grid_without_spatial_dims_is_an_error(tmp_path):
    timeline = xr.DataArray(np.zeros(4), dims=("time",), coords={"time": np.arange(4)})

    with pytest.raises(ValueError, match="spatial dimensions"):
        _align_raster(constant_raster(tmp_path / "const.tif"), timeline)


def test_a_dataset_is_rejected_on_either_side(tmp_path):
    grid = footprint_grid()
    path = constant_raster(tmp_path / "const.tif")

    with pytest.raises(TypeError, match="not a Dataset"):
        _align_raster(path, grid.to_dataset(name="footprint"))
    with pytest.raises(TypeError, match="single DataArray"):
        _align_raster(grid.to_dataset(name="footprint"), grid)


def test_a_non_dataarray_footprint_is_rejected(tmp_path):
    with pytest.raises(TypeError, match="must be an xarray.DataArray"):
        _align_raster(constant_raster(tmp_path / "const.tif"), [1, 2, 3])


def test_missing_rioxarray_is_reported_with_install_instructions(tmp_path, monkeypatch):
    """A None entry in sys.modules makes the import fail the way an absent
    install would, without disturbing anything else."""
    grid = footprint_grid()
    path = constant_raster(tmp_path / "const.tif")
    monkeypatch.setitem(sys.modules, "rioxarray", None)

    with pytest.raises(ImportError, match="pip install rioxarray"):
        _align_raster(path, grid)
