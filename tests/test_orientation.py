"""Geographic orientation checks: x is east, y is north, wind comes FROM WD."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import gaussian_kde
from shapely.geometry import Point, Polygon

from fluxfootprints import FFPModel, KormannMeixnerModel, WangFootprintModel
from fluxfootprints import compare, ffp_daily_monthly_helper as helper
from fluxfootprints.by_row_fetch_tools import generate_density_raster
from fluxfootprints.ffp_xr import ffp_climatology_new
from fluxfootprints.footprint_plotting import FootprintPlotter, add_plotting_to_footprint
from fluxfootprints.ls_footprint_model import LSFootprintConfig
from fluxfootprints.ls_footprint_adapter import LSFootprintModelAdapter


def met_frame(direction):
    return pd.DataFrame(
        dict(zm=[2.0], z0=[0.05], ustar=[0.4], ol=[-100.0],
             h=[1000.0], sigmav=[0.5], umean=[3.0], wind_dir=[direction]),
        index=pd.date_range("2024-06-01", periods=1, freq="h"),
    )


@pytest.mark.parametrize("direction", np.arange(0.0, 361.0, 45.0))
@pytest.mark.parametrize("kind", ["ffp", "ffp_rsl", "ffp_in_rsl", "ffp_xr", "km", "wang", "comparison_km"])
def test_sources_follow_meteorological_bearing(kind, direction):
    df = met_frame(direction)
    if kind == "ffp_in_rsl":
        df["zm"] = 1.2  # Above 20*z0, within z_star = 27.5*z0 = 1.375 m.
    params = dict(domain=[-200, 200, -200, 200], dx=5, dy=5,
                  smooth_data=False, verbosity=0)
    if kind == "comparison_km":
        f, (x, y) = compare.run_km(df, domain=params["domain"], dx=5)
    else:
        if kind in ("ffp", "ffp_rsl", "ffp_in_rsl"):
            model = FFPModel(df, rslayer=kind != "ffp", **params)
            model.run()
        elif kind == "ffp_xr":
            model = ffp_climatology_new(df, **params)
            model.calc_xr_footprint()
        else:
            model = {"km": KormannMeixnerModel, "wang": WangFootprintModel}[kind](df, **params)
            model.run()
        f = model.fclim_2d.transpose("x", "y").values
        x, y = np.meshgrid(model.x, model.y, indexing="ij")
    assert np.isfinite(f).all() and f.sum() > 0
    theta = np.deg2rad(direction)
    along = x * np.sin(theta) + y * np.cos(theta)
    across = x * np.cos(theta) - y * np.sin(theta)
    assert (f * along).sum() / f.sum() > 5
    assert abs((f * across).sum()) < 0.01 * (f * along).sum()
    assert f[along < -1e-6].sum() < 1e-10 * f.sum()


def test_enabling_rsl_above_sublayer_preserves_footprint():
    params = dict(domain=[-100, 100, -60, 60], dx=5, dy=5,
                  smooth_data=False, verbosity=0)
    base = FFPModel(met_frame(30), **params)
    corrected = FFPModel(met_frame(30), rslayer=True, **params)
    base.run()
    corrected.run()
    xr.testing.assert_allclose(base.fclim_2d, corrected.fclim_2d)


def test_invalid_timesteps_do_not_invent_sources_in_all_directions():
    df = met_frame(90)
    df["ol"] = -0.01  # Outside the stability validity bounds.
    model = FFPModel(df, domain=[-100, 100, -100, 100], dx=5, dy=5,
                     smooth_data=False, verbosity=0)
    model.run()
    assert float(model.fclim_2d.sum()) == 0


@pytest.mark.parametrize("direction", [0, 45, 90, 135, 180, 225, 270, 315, 360])
def test_ls_adapter_bearing_and_asymmetric_domain(monkeypatch, direction):
    # Fix the random seed while exercising the adapter's actual config and model.
    def seeded_config(**kwargs):
        return LSFootprintConfig(**kwargs, seed=3, t_max=100)
    monkeypatch.setattr("fluxfootprints.ls_footprint_adapter.LSFootprintConfig", seeded_config)
    model = LSFootprintModelAdapter(met_frame(direction),
        domain=[-200, 300, -150, 250], dx=10, dy=10, n_particles=1000)
    model.run()
    assert model.x[[0, -1]] == pytest.approx([-195, 295])
    assert model.y[[0, -1]] == pytest.approx([-145, 245])
    f = model.fclim_2d
    cx, cy = float((f * f.x).sum() / f.sum()), float((f * f.y).sum() / f.sum())
    theta = np.deg2rad(direction)
    assert cx * np.sin(theta) + cy * np.cos(theta) > 5
    assert abs(cx * np.cos(theta) - cy * np.sin(theta)) < 3


@pytest.mark.parametrize("method", ["rasterio", "skimage"])
@pytest.mark.parametrize("descending", [False, True])
def test_contour_keeps_off_center_peak(method, descending):
    pytest.importorskip(method)
    x, y = np.arange(-60, 61, 10), np.arange(-40, 41, 10)
    z = np.zeros((len(y), len(x)))
    z[6, 9] = 1  # east 30 m, north 20 m, on a rectangular grid
    if descending:
        x, y, z = x[::-1], y[::-1], z[::-1, ::-1]
    vertices = helper._make_contour_polygon_from_field_alt(z, x, y, method=method)
    centroid = Polygon(vertices).centroid
    assert (centroid.x, centroid.y) == pytest.approx((30, 20))


def source_field():
    x, y = np.arange(-60, 61, 10), np.arange(-40, 41, 10)
    z = np.zeros((len(x), len(y)))
    z[9, 6] = 1
    return xr.DataArray(z[None], dims=("time", "x", "y"),
                        coords=dict(time=[pd.Timestamp("2024-06-01")], x=x, y=y))


def test_contour_stats_preserve_named_dimensions(tmp_path):
    da = source_field()
    model = SimpleNamespace(x=da.x.values, y=da.y.values)
    out = helper.export_contour_stats_csv(pd.DataFrame(), model,
        helper.SummaryResult(f_daily_mean=da), 40, -111,
        tmp_path / "orientation.csv", method="rasterio")
    row = pd.read_csv(out).iloc[0]
    # Tower is on the central meridian: positive offsets must land NE of it.
    assert row.centroid_lon > -111
    assert row.centroid_lat > 40


@pytest.mark.parametrize("reverse_x, reverse_y", [(False, False), (True, False), (False, True), (True, True)])
def test_geotiff_peak_has_correct_world_coordinates(tmp_path, reverse_x, reverse_y):
    import rasterio
    from pyproj import Transformer
    da = source_field()
    if reverse_x:
        da = da.isel(x=slice(None, None, -1))
    if reverse_y:
        da = da.isel(y=slice(None, None, -1))
    model = SimpleNamespace(x=da.x.values, y=da.y.values)
    helper.export_rasters_geotiff(model, helper.SummaryResult(f_daily_mean=da),
        40, -111, tmp_path, which=("daily_mean",))
    with rasterio.open(next(tmp_path.glob("*.tif"))) as src:
        row, col = np.unravel_index(src.read(1).argmax(), src.shape)
        x, y = src.xy(row, col)
        x0, y0 = Transformer.from_crs(4326, src.crs, always_xy=True).transform(-111, 40)
        assert (x - x0, y - y0) == pytest.approx((30, 20))


def test_kde_values_match_affine_pixel_centers():
    points = np.array([[0, 10], [10, 30], [30, 80], [70, 90], [80, 70]], dtype=float)
    weights = np.arange(1.0, 6.0)
    gdf = gpd.GeoDataFrame(dict(ET=weights), geometry=[Point(p) for p in points], crs=5070)
    density, transform, bounds = generate_density_raster(gdf, resolution=20, buffer_distance=30)
    rows, cols = np.indices(density.shape)
    x, y = transform * (cols + 0.5, rows + 0.5)
    expected = gaussian_kde(points.T, weights=weights)(np.vstack([x.ravel(), y.ravel()]))
    np.testing.assert_allclose(density, expected.reshape(density.shape))
    assert transform * (0, 0) == pytest.approx((bounds[0], bounds[3]))
    assert transform * (density.shape[1], density.shape[0]) == pytest.approx((bounds[2], bounds[1]))


@pytest.mark.parametrize("grid_kind", ["1d", "ij", "xy"])
def test_plot_heatmap_and_contours_share_geographic_peak(grid_kind):
    da = source_field().isel(time=0)
    x, y, f = da.x.values, da.y.values, da.values
    if grid_kind != "1d":
        x, y = np.meshgrid(x, y, indexing=grid_kind)
        if grid_kind == "xy":
            f = f.T
    plotter = FootprintPlotter()
    fig, ax = plotter.plot_footprint(x, y, f, clevs=[0.5])
    mesh = ax.collections[-1]
    peak = np.unravel_index(np.asarray(mesh.get_array()).argmax(), f.shape)
    corners = mesh.get_coordinates()
    center = corners[peak[0]:peak[0]+2, peak[1]:peak[1]+2].mean(axis=(0, 1))
    np.testing.assert_allclose(center, (30, 20))
    vx, vy = plotter.get_contour_vertices(x, y, f, 0.5)
    centroid = Polygon(np.column_stack([vx, vy])).centroid
    assert (centroid.x, centroid.y) == pytest.approx((30, 20))
    plt.close(fig)


def test_plot_snapshot_uses_named_dimensions():
    class Example:
        pass
    add_plotting_to_footprint(Example)
    ds = source_field().transpose("time", "y", "x").to_dataset(name="footprint")
    ds.attrs["grid_spacing"] = 10
    fig, ax = Example().plot_footprint_snapshot(ds, 0)
    assert ax.collections[-1].get_array().shape == (13, 9)
    plt.close(fig)


def test_comparison_ffp_wrapper_imports_and_keeps_grid_orientation():
    f, (x, y) = compare.run_ffp_xr(met_frame(90), domain=[-100, 100, -60, 60], dx=5)
    assert f.shape == x.shape == y.shape
    assert (f * x).sum() > 0
    assert abs((f * y).sum()) < 1e-10
