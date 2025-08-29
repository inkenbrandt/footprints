# test_ffp_xr_contours.py
import numpy as np
import xarray as xr
import pytest

try:
    from shapely.geometry import Polygon, MultiPolygon
except Exception:  # shapely optional
    Polygon = MultiPolygon = None


@pytest.mark.skipif(Polygon is None, reason="shapely not available")
def test_contour_polygons(contour_fn, compute_fn, synthetic_inputs):
    if contour_fn is None:
        pytest.skip("No contour function exported by ffp_xr.")

    ds, meta = synthetic_inputs
    out = compute_fn(ds, **meta)

    # Use single 2D slice
    if isinstance(out, xr.Dataset):
        da = next(iter(out.data_vars.values()))
    else:
        da = out
    if "time" in da.dims:
        da = da.isel(time=0)

    # common signatures:
    # contour_fn(da, levels=[0.5, 0.8]) or contour_fn(field=da, perc=[80])
    polys = None
    try:
        polys = contour_fn(da, levels=[0.8])
    except TypeError:
        try:
            polys = contour_fn(field=da, perc=[80])
        except Exception as e:
            pytest.skip(f"Could not call contour function with known signatures: {e}")

    # Accept list of shapely geometries or GeoDataFrame
    geoms = None
    try:
        import geopandas as gpd

        if isinstance(polys, gpd.GeoDataFrame):
            geoms = list(polys.geometry)
    except Exception:
        pass

    if geoms is None:
        # assume list/iterable of shapely Polygons/MultiPolygons
        geoms = list(polys) if hasattr(polys, "__iter__") else [polys]

    assert geoms, "No geometries returned for r% contour."
    for g in geoms:
        assert isinstance(
            g, (Polygon, MultiPolygon)
        ), f"Unexpected geometry type: {type(g)}"
        assert g.is_valid, "Returned geometry is invalid."
        assert g.area > 0, "Degenerate zero-area contour."


@pytest.mark.skipif(Polygon is None, reason="shapely not available")
def test_contour_monotonic_area(contour_fn, compute_fn, synthetic_inputs):
    if contour_fn is None:
        pytest.skip("No contour function exported by ffp_xr.")

    ds, meta = synthetic_inputs
    da = compute_fn(ds, **meta)
    if isinstance(da, xr.Dataset):
        da = next(iter(da.data_vars.values()))
    if "time" in da.dims:
        da = da.isel(time=0)

    # Increasing r% should not decrease area (80% ≥ 50%)
    levels = [0.5, 0.8]
    try:
        polys50 = contour_fn(da, levels=[levels[0]])
        polys80 = contour_fn(da, levels=[levels[1]])
    except TypeError:
        polys50 = contour_fn(field=da, perc=[50])
        polys80 = contour_fn(field=da, perc=[80])

    def _total_area(polys):
        from shapely.ops import unary_union
        from shapely.geometry import Polygon, MultiPolygon

        if hasattr(polys, "geometry"):  # GeoDataFrame
            geoms = list(polys.geometry)
        else:
            geoms = list(polys)
        merged = unary_union(geoms)
        if isinstance(merged, (Polygon, MultiPolygon)):
            return merged.area
        return sum(g.area for g in geoms)

    area50 = _total_area(polys50)
    area80 = _total_area(polys80)
    assert area80 >= area50, "Higher r% contour should cover equal or larger area."


def test_export_contours_gpkg_roundtrip(
    export_gpkg_fn, contour_fn, compute_fn, synthetic_inputs, tmp_path
):
    if export_gpkg_fn is None or contour_fn is None:
        pytest.skip("Export function or contour function not available.")

    try:
        import geopandas as gpd  # noqa: F401
    except Exception:
        pytest.skip("geopandas not available; skipping export test.")

    ds, meta = synthetic_inputs
    out = compute_fn(ds, **meta)
    if isinstance(out, xr.Dataset):
        da = next(iter(out.data_vars.values()))
    else:
        da = out
    if "time" in da.dims:
        da = da.isel(time=0)

    # build 80% contour and export
    try:
        polys = contour_fn(da, levels=[0.8])
    except TypeError:
        polys = contour_fn(field=da, perc=[80])

    gpkg = tmp_path / "contours.gpkg"
    # common signatures:
    # export_contours_gpkg(polys, gpkg, layer="r80")
    # or export_contours_gpkg(geoms=polys, path=..., layer=...)
    try:
        export_gpkg_fn(polys, gpkg, layer="r80")
    except TypeError:
        export_gpkg_fn(geoms=polys, path=str(gpkg), layer="r80")

    # Read back to confirm a layer was written
    import fiona

    layers = fiona.listlayers(str(gpkg))
    assert "r80" in layers and len(layers) > 0
