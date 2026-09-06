# test_bootstrap_footprint_uncertainty.py
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import rasterio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints.bootstrap_footprint_uncertainty import (
    BlockStack,
    BootstrapResult,
    GridSpec,
    accumulate_blocks,
    block_bootstrap,
    block_labels,
    ffp_period_adapter,
    rotate_climatology,
    source_area_mask,
    write_geotiff,
)

TOWER_LAT, TOWER_LON = 37.7353, -111.5708


# ----------------------------
# helpers
# ----------------------------
def make_grid(half_width=50.0, dx=10.0):
    return GridSpec.square(half_width, dx, TOWER_LAT, TOWER_LON)


def make_rect_grid():
    """Deliberately non-square so a lost transpose cannot pass silently."""
    return GridSpec(
        x=np.array([-10.0, 0.0, 10.0, 20.0]),   # nx = 4
        y=np.array([-5.0, 5.0, 15.0]),          # ny = 3
        tower_lat=TOWER_LAT,
        tower_lon=TOWER_LON,
    )


def gaussian_xy(grid, x0=0.0, y0=0.0, sx=20.0, sy=20.0):
    """Unit-mass 2-D Gaussian density (m^-2) in the model's (x, y) order."""
    gx = np.exp(-0.5 * ((grid.x - x0) / sx) ** 2)
    gy = np.exp(-0.5 * ((grid.y - y0) / sy) ** 2)
    fp = np.outer(gx, gy)                       # (nx, ny) -> "xy" order
    return fp / (fp.sum() * grid.cell_area)


def make_stack(grid, n_blocks=30, periods_per_block=4, capture=1.0, seed=1):
    """A BlockStack of unit-mass Gaussians with jittered centres."""
    rng = np.random.default_rng(seed)
    stack = np.empty((n_blocks, *grid.shape), dtype=np.float32)
    for b in range(n_blocks):
        fp = gaussian_xy(grid, x0=rng.normal(0, 8), y0=rng.normal(0, 8))
        stack[b] = grid.to_raster_order(fp) * periods_per_block
    return BlockStack(
        stack=stack,
        n_periods=np.full(n_blocks, periods_per_block, dtype=np.int64),
        captured=np.full(n_blocks, capture * periods_per_block, dtype=np.float64),
        labels=list(range(n_blocks)),
        grid=grid,
    )


def make_df(n=8, freq="30min", start="2024-06-24 14:30"):
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"value": np.arange(n, dtype=float)}, index=idx)


# ==========================================================================
# GridSpec
# ==========================================================================
def test_square_is_centred_on_the_tower():
    g = make_grid(half_width=50.0, dx=10.0)
    assert g.shape == (10, 10)
    assert g.x.size == 10 and g.y.size == 10
    # cell centres straddle zero; the receptor sits on a cell edge, not a centre
    assert g.x.mean() == pytest.approx(0.0)
    assert g.y.mean() == pytest.approx(0.0)
    np.testing.assert_allclose(g.x, g.y)


def test_square_spacing_and_area():
    g = make_grid(half_width=60.0, dx=5.0)
    assert g.dx == pytest.approx(5.0)
    assert g.dy == pytest.approx(5.0)
    assert g.cell_area == pytest.approx(25.0)
    assert g.crs == "EPSG:32612"


def test_square_honours_a_custom_crs():
    g = GridSpec.square(50.0, 10.0, TOWER_LAT, TOWER_LON, crs="EPSG:26912")
    assert g.crs == "EPSG:26912"


def test_shape_is_row_col_not_x_y():
    g = make_rect_grid()
    assert g.shape == (3, 4)          # (ny, nx)


def test_gridspec_is_frozen():
    g = make_grid()
    with pytest.raises(Exception):
        g.tower_lat = 0.0


# --- to_raster_order -------------------------------------------------------
def test_to_raster_order_xy_transposes_and_flips_north_up():
    g = make_rect_grid()
    arr = np.zeros((4, 3))            # (nx, ny)
    arr[3, 2] = 1.0                   # easternmost x, northernmost y
    out = g.to_raster_order(arr, "xy")
    assert out.shape == (3, 4)
    assert out[0, 3] == 1.0           # row 0 is north, last column is east
    assert out.sum() == pytest.approx(arr.sum())


def test_to_raster_order_yx_only_flips():
    g = make_rect_grid()
    arr = np.zeros((3, 4))            # already (ny, nx)
    arr[2, 0] = 1.0                   # northernmost y, westernmost x
    out = g.to_raster_order(arr, "yx")
    assert out.shape == (3, 4)
    assert out[0, 0] == 1.0


def test_to_raster_order_is_an_involution_on_square_grids():
    g = make_grid()
    rng = np.random.default_rng(0)
    arr = rng.random(g.shape)
    # yx twice = flip twice = identity
    np.testing.assert_allclose(
        g.to_raster_order(g.to_raster_order(arr, "yx"), "yx"), arr
    )


def test_to_raster_order_rejects_unknown_model_order():
    g = make_grid()
    with pytest.raises(ValueError, match="model_order must be"):
        g.to_raster_order(np.zeros(g.shape), "rc")


def test_to_raster_order_rejects_mismatched_shape():
    g = make_rect_grid()
    with pytest.raises(ValueError, match="does not match grid"):
        g.to_raster_order(np.zeros((3, 4)), "xy")   # right shape, wrong order


def test_to_raster_order_accepts_a_nested_list():
    g = make_rect_grid()
    out = g.to_raster_order([[0.0] * 3 for _ in range(4)], "xy")
    assert out.shape == (3, 4)


# ==========================================================================
# block_labels
# ==========================================================================
def test_block_labels_are_calendar_days():
    idx = pd.date_range("2024-06-24 14:30", periods=5, freq="12h")
    np.testing.assert_array_equal(block_labels(idx, 1), [0, 1, 1, 2, 2])


def test_block_labels_group_multiple_days():
    idx = pd.date_range("2024-06-24 14:30", periods=5, freq="12h")
    np.testing.assert_array_equal(block_labels(idx, 2), [0, 0, 0, 1, 1])


def test_block_labels_origin_is_the_first_day_not_the_epoch():
    idx = pd.date_range("2030-01-05 00:00", periods=3, freq="1D")
    np.testing.assert_array_equal(block_labels(idx, 1), [0, 1, 2])


def test_block_labels_leave_gaps_for_missing_days():
    idx = pd.DatetimeIndex(["2024-06-24 01:00", "2024-06-29 01:00"])
    np.testing.assert_array_equal(block_labels(idx, 1), [0, 5])


# ==========================================================================
# BlockStack
# ==========================================================================
def test_blockstack_drops_empty_blocks_and_keeps_labels_aligned():
    g = make_grid()
    stack = np.arange(3 * g.shape[0] * g.shape[1], dtype=np.float32).reshape(3, *g.shape)
    bs = BlockStack(
        stack=stack.copy(),
        n_periods=np.array([2, 0, 5]),
        captured=np.array([1.0, 0.0, 4.0]),
        labels=["a", "b", "c"],
        grid=g,
    )
    assert bs.n_blocks == 2
    assert list(bs.labels) == ["a", "c"]
    np.testing.assert_array_equal(bs.n_periods, [2, 5])
    np.testing.assert_allclose(bs.captured, [1.0, 4.0])
    np.testing.assert_allclose(bs.stack[1], stack[2])


def test_blockstack_keeps_everything_when_all_blocks_are_populated():
    g = make_grid()
    bs = BlockStack(
        stack=np.zeros((2, *g.shape), dtype=np.float32),
        n_periods=np.array([1, 1]),
        captured=np.array([0.9, 0.8]),
        labels=[0, 1],
        grid=g,
    )
    assert bs.n_blocks == 2
    assert list(bs.labels) == [0, 1]


def test_mean_capture_is_weighted_by_period_count():
    g = make_grid()
    bs = BlockStack(
        stack=np.zeros((2, *g.shape), dtype=np.float32),
        n_periods=np.array([1, 3]),
        captured=np.array([0.9, 2.4]),     # 0.9 and 0.8 per period
        labels=[0, 1],
        grid=g,
    )
    assert bs.mean_capture == pytest.approx((0.9 + 2.4) / 4)


# ==========================================================================
# accumulate_blocks
# ==========================================================================
def test_accumulate_requires_a_datetime_index():
    g = make_grid()
    df = pd.DataFrame({"value": [1.0, 2.0]})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        accumulate_blocks(df, g, lambda row: None, progress=False)


def test_accumulate_sums_raw_footprints_within_a_block():
    g = make_grid()
    df = make_df(n=4, freq="30min")            # all on one day
    one = np.ones((g.x.size, g.y.size))
    bs = accumulate_blocks(df, g, lambda row: one, progress=False)

    assert bs.n_blocks == 1
    assert bs.n_periods.tolist() == [4]
    np.testing.assert_allclose(bs.stack[0], 4.0)
    # captured = summed mass * cell area, accumulated raw (never renormalised)
    assert bs.captured[0] == pytest.approx(4 * one.sum() * g.cell_area)


def test_accumulate_splits_across_days():
    g = make_grid()
    df = make_df(n=6, freq="12h", start="2024-06-24 00:00")   # 3 calendar days
    one = np.ones((g.x.size, g.y.size))
    bs = accumulate_blocks(df, g, lambda row: one, block_days=1, progress=False)
    assert bs.n_blocks == 3
    assert bs.n_periods.tolist() == [2, 2, 2]
    assert bs.n_periods.sum() == 6


def test_accumulate_honours_block_days():
    g = make_grid()
    df = make_df(n=6, freq="12h", start="2024-06-24 00:00")
    one = np.ones((g.x.size, g.y.size))
    bs = accumulate_blocks(df, g, lambda row: one, block_days=3, progress=False)
    assert bs.n_blocks == 1
    assert bs.n_periods.tolist() == [6]


def test_accumulate_blocks_follow_the_calendar_not_the_first_timestamp():
    """A run starting mid-afternoon still breaks on midnight boundaries."""
    g = make_grid()
    df = make_df(n=6, freq="12h", start="2024-06-24 14:30")
    one = np.ones((g.x.size, g.y.size))
    bs = accumulate_blocks(df, g, lambda row: one, block_days=1, progress=False)
    assert bs.n_periods.tolist() == [1, 2, 2, 1]


def test_accumulate_skips_none_periods():
    g = make_grid()
    df = make_df(n=4)
    one = np.ones((g.x.size, g.y.size))
    fn = lambda row: None if row.value % 2 else one
    bs = accumulate_blocks(df, g, fn, progress=False)
    assert bs.n_periods.tolist() == [2]


def test_accumulate_warns_and_skips_when_the_model_raises():
    g = make_grid()
    df = make_df(n=2)

    def boom(row):
        raise RuntimeError("model blew up")

    with pytest.warns(UserWarning, match="footprint failed"):
        bs = accumulate_blocks(df, g, boom, progress=False)
    assert bs.n_blocks == 0                    # every block emptied out
    assert bs.n_periods.size == 0


def test_accumulate_skips_non_finite_footprints():
    g = make_grid()
    df = make_df(n=3)
    one = np.ones((g.x.size, g.y.size))

    def fn(row):
        if row.value == 1:
            bad = one.copy()
            bad[0, 0] = np.nan
            return bad
        if row.value == 2:
            bad = one.copy()
            bad[0, 0] = np.inf
            return bad
        return one

    bs = accumulate_blocks(df, g, fn, progress=False)
    assert bs.n_periods.tolist() == [1]
    np.testing.assert_allclose(bs.stack[0], 1.0)


def test_accumulate_propagates_model_order():
    g = make_rect_grid()
    df = make_df(n=1)
    arr = np.zeros((3, 4))                     # (ny, nx)
    arr[2, 0] = 1.0
    bs = accumulate_blocks(df, g, lambda row: arr, model_order="yx", progress=False)
    assert bs.stack[0][0, 0] == pytest.approx(1.0)


def test_accumulate_raises_on_a_grid_shape_mismatch():
    """A wrong-shaped footprint is a configuration error, not a bad period:
    the reshape happens outside the per-period try, so it aborts the run."""
    g = make_grid()
    df = make_df(n=1)
    with pytest.raises(ValueError, match="does not match grid"):
        accumulate_blocks(df, g, lambda row: np.ones((3, 3)), progress=False)


def test_accumulate_stack_is_float32():
    g = make_grid()
    bs = accumulate_blocks(
        make_df(n=2), g, lambda row: np.ones((g.x.size, g.y.size)), progress=False
    )
    assert bs.stack.dtype == np.float32


def test_accumulate_prints_progress_only_when_asked(capsys):
    g = make_grid()
    df = make_df(n=2)
    one = np.ones((g.x.size, g.y.size))

    accumulate_blocks(df, g, lambda row: one, progress=False)
    assert capsys.readouterr().out == ""

    accumulate_blocks(df, g, lambda row: one, progress=True)
    out = capsys.readouterr().out
    assert "0/2 periods" in out
    assert "2 usable, 0 skipped, 1 blocks" in out


def test_accumulate_labels_are_the_block_ids():
    g = make_grid()
    df = make_df(n=4, freq="1D")
    one = np.ones((g.x.size, g.y.size))
    bs = accumulate_blocks(df, g, lambda row: one, progress=False)
    assert list(bs.labels) == [0, 1, 2, 3]


# ==========================================================================
# source_area_mask
# ==========================================================================
def test_source_area_mask_is_the_smallest_set_reaching_R():
    w = np.array([[4.0, 3.0], [2.0, 1.0]])
    m = source_area_mask(w, 0.5)
    assert m.sum() == 2                        # 4 + 3 = 0.7 >= 0.5; 4 alone is 0.4
    assert w[m].sum() / w.sum() >= 0.5
    assert np.array_equal(m, np.array([[True, True], [False, False]]))


def test_source_area_mask_takes_the_largest_pixels_first():
    rng = np.random.default_rng(3)
    w = rng.random((12, 12))
    m = source_area_mask(w, 0.6)
    assert w[m].min() >= w[~m].max()


def test_source_area_mask_is_minimal():
    rng = np.random.default_rng(4)
    w = rng.random((10, 10))
    for R in (0.25, 0.5, 0.8, 0.95):
        m = source_area_mask(w, R)
        inside = np.sort(w[m])[::-1]
        assert inside.sum() / w.sum() >= R - 1e-12
        # dropping the smallest included pixel must fall short
        assert inside[:-1].sum() / w.sum() < R


def test_source_area_mask_is_monotone_in_R():
    rng = np.random.default_rng(5)
    w = rng.random((8, 8))
    prev = source_area_mask(w, 0.2)
    for R in (0.4, 0.6, 0.8, 0.99):
        cur = source_area_mask(w, R)
        assert cur.sum() >= prev.sum()
        assert np.all(cur[prev])               # nested
        prev = cur


def test_source_area_mask_of_an_empty_raster_is_empty():
    m = source_area_mask(np.zeros((4, 4)), 0.9)
    assert m.dtype == bool
    assert m.shape == (4, 4)
    assert not m.any()


def test_source_area_mask_handles_a_non_positive_total():
    m = source_area_mask(np.full((3, 3), -1.0), 0.5)
    assert not m.any()


def test_source_area_mask_at_R_one_takes_everything():
    w = np.array([[4.0, 3.0], [2.0, 1.0]])
    assert source_area_mask(w, 1.0).all()


def test_source_area_mask_preserves_shape_on_non_square_rasters():
    rng = np.random.default_rng(6)
    w = rng.random((3, 7))
    assert source_area_mask(w, 0.5).shape == (3, 7)


def test_source_area_mask_never_overruns_the_raster():
    """searchsorted can land past the end when R == 1 and floats round up."""
    w = np.full((5, 5), 1.0)
    m = source_area_mask(w, 1.0)
    assert m.sum() == 25


# ==========================================================================
# block_bootstrap
# ==========================================================================
def test_bootstrap_shapes_and_bookkeeping():
    g = make_grid()
    bs = make_stack(g, n_blocks=30)
    res = block_bootstrap(bs, n_members=40, levels=(0.5, 0.8), seed=0)

    assert isinstance(res, BootstrapResult)
    assert res.levels == (0.5, 0.8)
    for arr in (res.w_mean, res.w_p05, res.w_p50, res.w_p95, res.w_cv):
        assert arr.shape == g.shape
        assert arr.dtype == np.float32
    for R in res.levels:
        assert res.p_include[R].shape == g.shape
    assert res.n_members + res.n_dropped == 40
    assert res.member_capture.shape == (res.n_members,)


def test_bootstrap_normalises_each_member_to_unit_mass():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=30, seed=0)
    assert res.w_mean.sum() * g.cell_area == pytest.approx(1.0, rel=1e-4)
    assert res.w_p50.sum() * g.cell_area == pytest.approx(1.0, rel=0.05)


def test_bootstrap_percentiles_are_ordered():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=60, seed=0)
    assert np.all(res.w_p05 <= res.w_p50 + 1e-12)
    assert np.all(res.w_p50 <= res.w_p95 + 1e-12)


def test_bootstrap_p_include_is_a_probability():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=50, levels=(0.5, 0.9), seed=0)
    for R, p in res.p_include.items():
        assert p.min() >= 0.0 and p.max() <= 1.0
        # the wider source area must cover at least as much, pixel by pixel
        assert R in (0.5, 0.9)
    assert np.all(res.p_include[0.9] >= res.p_include[0.5] - 1e-6)


def test_bootstrap_p_include_is_one_at_the_footprint_peak():
    g = make_grid()
    res = block_bootstrap(make_stack(g, seed=2), n_members=50, levels=(0.9,), seed=0)
    peak = np.unravel_index(np.argmax(res.w_mean), res.w_mean.shape)
    assert res.p_include[0.9][peak] == pytest.approx(1.0)


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    g = make_grid()
    bs = make_stack(g)
    a = block_bootstrap(bs, n_members=25, seed=7)
    b = block_bootstrap(bs, n_members=25, seed=7)
    np.testing.assert_array_equal(a.w_mean, b.w_mean)
    np.testing.assert_array_equal(a.p_include[0.5], b.p_include[0.5])
    np.testing.assert_allclose(a.member_capture, b.member_capture)


def test_bootstrap_differs_across_seeds():
    g = make_grid()
    bs = make_stack(g)
    a = block_bootstrap(bs, n_members=25, seed=7)
    b = block_bootstrap(bs, n_members=25, seed=8)
    assert not np.array_equal(a.w_mean, b.w_mean)


def test_bootstrap_sorts_levels():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=20, levels=(0.9, 0.5, 0.8), seed=0)
    assert res.levels == (0.5, 0.8, 0.9)


def test_bootstrap_warns_when_there_are_too_few_blocks():
    g = make_grid()
    bs = make_stack(g, n_blocks=5)
    with pytest.warns(UserWarning, match="block-bootstrap intervals are unreliable"):
        block_bootstrap(bs, n_members=10, seed=0)


def test_bootstrap_is_quiet_with_enough_blocks():
    g = make_grid()
    bs = make_stack(g, n_blocks=25)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        block_bootstrap(bs, n_members=10, seed=0)


def test_bootstrap_default_min_capture_tracks_the_widest_level():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=10, levels=(0.5, 0.8), seed=0)
    assert res.meta["min_capture"] == pytest.approx(0.85)


def test_bootstrap_default_min_capture_is_clipped_at_0_99():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=10, levels=(0.99,), seed=0)
    assert res.meta["min_capture"] == pytest.approx(0.99)


def test_bootstrap_drops_members_below_min_capture():
    g = make_grid()
    bs = make_stack(g, n_blocks=30, capture=1.0)
    # three badly truncated blocks; a member that draws them falls short
    bs.captured[:3] = 0.0
    res = block_bootstrap(bs, n_members=60, levels=(0.8,), min_capture=0.95, seed=0)
    assert 0 < res.n_members < 60
    assert res.n_members + res.n_dropped == 60
    assert np.all(res.member_capture >= 0.95)


def test_bootstrap_keeps_everything_when_min_capture_is_zero():
    g = make_grid()
    bs = make_stack(g, n_blocks=30, capture=0.1)
    res = block_bootstrap(bs, n_members=20, min_capture=0.0, seed=0)
    assert res.n_dropped == 0
    assert res.n_members == 20


def test_bootstrap_member_capture_is_a_per_period_mean():
    g = make_grid()
    bs = make_stack(g, n_blocks=30, periods_per_block=4, capture=0.77)
    res = block_bootstrap(bs, n_members=15, min_capture=0.0, seed=0)
    np.testing.assert_allclose(res.member_capture, 0.77, rtol=1e-6)


def test_bootstrap_raises_when_every_member_fails_the_capture_test():
    g = make_grid()
    bs = make_stack(g, n_blocks=30, capture=0.05)
    with pytest.raises(RuntimeError, match="domain is far\n?\\s*too small|capture test"):
        block_bootstrap(bs, n_members=10, min_capture=0.9, seed=0)


def test_bootstrap_raises_when_every_member_has_zero_mass():
    """Zero-mass members normalise to NaN and must be rejected, not propagated."""
    g = make_grid()
    bs = BlockStack(
        stack=np.zeros((25, *g.shape), dtype=np.float32),
        n_periods=np.full(25, 2, dtype=np.int64),
        captured=np.full(25, 2.0),      # capture passes; mass does not
        labels=list(range(25)),
        grid=g,
    )
    with pytest.raises(RuntimeError):
        block_bootstrap(bs, n_members=10, min_capture=0.5, seed=0)


def test_bootstrap_cv_is_non_negative_where_it_is_defined():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=40, seed=0)
    finite = res.w_cv[np.isfinite(res.w_cv)]
    assert finite.size > 0
    assert np.all(finite >= 0.0)


def test_bootstrap_cv_floor_masks_the_faint_tail():
    g = make_grid()
    bs = make_stack(g)
    low = block_bootstrap(bs, n_members=40, cv_floor_quantile=0.10, seed=0)
    high = block_bootstrap(bs, n_members=40, cv_floor_quantile=0.90, seed=0)
    n_masked_low = int(np.isnan(low.w_cv).sum())
    n_masked_high = int(np.isnan(high.w_cv).sum())
    assert n_masked_high > n_masked_low
    assert high.meta["cv_floor"] > low.meta["cv_floor"]


def test_bootstrap_cv_masking_follows_the_reported_floor():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=40, cv_floor_quantile=0.5, seed=0)
    floor = res.meta["cv_floor"]
    assert np.all(np.isnan(res.w_cv[res.w_mean < floor]))


def test_bootstrap_cv_is_zero_when_every_member_is_identical():
    """One block means every resample is the same climatology."""
    g = make_grid()
    bs = make_stack(g, n_blocks=1)
    with pytest.warns(UserWarning):
        res = block_bootstrap(bs, n_members=5, min_capture=0.0,
                              cv_floor_quantile=0.0, seed=0)
    finite = res.w_cv[np.isfinite(res.w_cv)]
    np.testing.assert_allclose(finite, 0.0, atol=1e-6)


def test_bootstrap_meta_records_the_run():
    g = make_grid()
    bs = make_stack(g, n_blocks=30, periods_per_block=4, capture=0.95)
    res = block_bootstrap(bs, n_members=12, levels=(0.8,), seed=42)
    assert res.meta["n_blocks"] == 30
    assert res.meta["n_periods"] == 120
    assert res.meta["mean_capture"] == pytest.approx(0.95)
    assert res.meta["seed"] == 42
    assert "cv_floor" in res.meta and "min_capture" in res.meta


def test_bootstrap_accepts_an_unseeded_rng():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=8, seed=None)
    assert res.meta["seed"] is None
    assert res.n_members > 0


@pytest.mark.filterwarnings("ignore:Degrees of freedom:RuntimeWarning")
def test_bootstrap_with_a_single_member():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=1, min_capture=0.0, seed=0)
    assert res.n_members == 1
    np.testing.assert_allclose(res.w_mean, res.w_p50, rtol=1e-5)
    # std with ddof=1 over one sample is undefined
    assert np.all(np.isnan(res.w_cv) | ~np.isfinite(res.w_cv))


def test_bootstrap_p_include_uses_the_kept_members_as_denominator():
    """A dropped member must not dilute the inclusion probability."""
    g = make_grid()
    bs = make_stack(g, n_blocks=30)
    bs.captured[:15] = 0.1 * bs.n_periods[:15]
    res = block_bootstrap(bs, n_members=40, levels=(0.9,), min_capture=0.5, seed=0)
    assert 0 < res.n_members < 40
    assert res.p_include[0.9].max() == pytest.approx(1.0)


# ==========================================================================
# BootstrapResult.bands
# ==========================================================================
def test_bands_names_and_order():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=10, levels=(0.5, 0.8, 0.9), seed=0)
    names = [n for n, _ in res.bands()]
    assert names == [
        "p_include_50", "p_include_80", "p_include_90",
        "w_mean", "w_p05", "w_p50", "w_p95", "w_cv",
    ]


def test_bands_returns_the_actual_arrays():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=10, levels=(0.5,), seed=0)
    bands = dict(res.bands())
    assert bands["w_mean"] is res.w_mean
    assert bands["p_include_50"] is res.p_include[0.5]
    assert all(a.shape == g.shape for a in bands.values())


def test_bands_rounds_the_level_label():
    g = make_grid()
    res = block_bootstrap(make_stack(g), n_members=10, levels=(0.333,),
                          min_capture=0.0, seed=0)
    assert [n for n, _ in res.bands()][0] == "p_include_33"


# ==========================================================================
# rotate_climatology
# ==========================================================================
def test_rotate_by_zero_is_a_no_op():
    rng = np.random.default_rng(0)
    w = rng.random((21, 21))
    np.testing.assert_allclose(rotate_climatology(w, 0.0), w, atol=1e-5)


def test_rotate_positive_is_clockwise_on_a_north_up_raster():
    w = np.zeros((21, 21), dtype=np.float32)
    w[5, 10] = 1.0                                    # due north of the tower
    east = rotate_climatology(w, 90.0)
    assert np.unravel_index(np.argmax(east), east.shape) == (10, 15)
    west = rotate_climatology(w, -90.0)
    assert np.unravel_index(np.argmax(west), west.shape) == (10, 5)
    south = rotate_climatology(w, 180.0)
    assert np.unravel_index(np.argmax(south), south.shape) == (15, 10)


def test_rotate_preserves_total_mass():
    rng = np.random.default_rng(1)
    w = np.zeros((41, 41))
    w[15:26, 15:26] = rng.random((11, 11))            # kept clear of the corners
    for deg in (-10.0, -5.0, 5.0, 10.0, 37.0):
        assert rotate_climatology(w, deg).sum() == pytest.approx(w.sum(), rel=1e-5)


def test_rotate_never_returns_negative_weights():
    w = np.zeros((31, 31))
    w[10:20, 10:20] = 1.0                             # sharp edges ring under interp
    out = rotate_climatology(w, 17.0)
    assert out.min() >= 0.0


def test_rotate_returns_float32():
    w = np.ones((11, 11), dtype=np.float64)
    assert rotate_climatology(w, 5.0).dtype == np.float32


def test_rotate_an_empty_raster_stays_empty():
    out = rotate_climatology(np.zeros((9, 9)), 30.0)
    assert out.dtype == np.float32
    assert not out.any()


def test_rotate_keeps_the_shape():
    w = np.zeros((13, 17))
    w[6, 8] = 1.0
    assert rotate_climatology(w, 45.0).shape == (13, 17)


def test_rotate_composes_approximately():
    rng = np.random.default_rng(2)
    w = np.zeros((51, 51))
    w[20:31, 20:31] = rng.random((11, 11))
    once = rotate_climatology(rotate_climatology(w, 45.0), 45.0)
    twice = rotate_climatology(w, 90.0)
    # bilinear interpolation smooths, so compare centroids not pixels
    def centroid(a):
        r, c = np.indices(a.shape)
        return (r * a).sum() / a.sum(), (c * a).sum() / a.sum()
    np.testing.assert_allclose(centroid(once), centroid(twice), atol=0.5)


# ==========================================================================
# write_geotiff
# ==========================================================================
@pytest.fixture
def result_and_grid():
    g = make_grid(half_width=50.0, dx=10.0)
    return block_bootstrap(make_stack(g), n_members=20, levels=(0.5, 0.9), seed=0), g


def test_write_geotiff_returns_the_path(tmp_path, result_and_grid):
    res, _ = result_and_grid
    p = tmp_path / "out.tif"
    assert write_geotiff(res, str(p)) == str(p)
    assert p.exists()


def test_write_geotiff_profile(tmp_path, result_and_grid):
    res, g = result_and_grid
    p = tmp_path / "out.tif"
    write_geotiff(res, str(p))
    with rasterio.open(p) as src:
        assert src.count == len(res.levels) + 5
        assert (src.height, src.width) == g.shape
        assert src.dtypes[0] == "float32"
        assert src.crs.to_string() == g.crs
        assert np.isnan(src.nodata)
        assert src.profile["compress"] == "deflate"


def test_write_geotiff_band_descriptions_match_bands(tmp_path, result_and_grid):
    res, _ = result_and_grid
    p = tmp_path / "out.tif"
    write_geotiff(res, str(p))
    with rasterio.open(p) as src:
        assert list(src.descriptions) == [n for n, _ in res.bands()]


def test_write_geotiff_roundtrips_the_values(tmp_path, result_and_grid):
    res, _ = result_and_grid
    p = tmp_path / "out.tif"
    write_geotiff(res, str(p))
    with rasterio.open(p) as src:
        for i, (_, arr) in enumerate(res.bands(), start=1):
            np.testing.assert_allclose(
                src.read(i), np.asarray(arr, dtype=np.float32), equal_nan=True
            )


def test_write_geotiff_georeferences_the_tower(tmp_path, result_and_grid):
    from pyproj import Transformer

    res, g = result_and_grid
    p = tmp_path / "out.tif"
    write_geotiff(res, str(p))

    tx = Transformer.from_crs("EPSG:4326", g.crs, always_xy=True)
    e0, n0 = tx.transform(g.tower_lon, g.tower_lat)
    with rasterio.open(p) as src:
        t = src.transform
        assert t.a == pytest.approx(g.dx)
        assert t.e == pytest.approx(-g.dy)
        # upper-left corner of the north-west pixel
        assert t.c == pytest.approx(e0 + g.x.min() - g.dx / 2.0)
        assert t.f == pytest.approx(n0 + g.y.max() + g.dy / 2.0)
        # the tower sits at the centre of the domain
        cx, cy = t * (g.shape[1] / 2.0, g.shape[0] / 2.0)
        assert cx == pytest.approx(e0, abs=1e-6)
        assert cy == pytest.approx(n0, abs=1e-6)


def test_write_geotiff_writes_metadata_tags(tmp_path, result_and_grid):
    res, _ = result_and_grid
    p = tmp_path / "out.tif"
    write_geotiff(res, str(p))
    with rasterio.open(p) as src:
        tags = src.tags()
    assert tags["n_members"] == str(res.n_members)
    assert tags["n_dropped"] == str(res.n_dropped)
    assert tags["levels"] == "0.5,0.9"
    assert tags["n_blocks"] == str(res.meta["n_blocks"])
    assert tags["seed"] == str(res.meta["seed"])
    assert "m^-2" in tags["note"]


def test_write_geotiff_honours_the_compression_choice(tmp_path, result_and_grid):
    res, _ = result_and_grid
    p = tmp_path / "lzw.tif"
    write_geotiff(res, str(p), compress="lzw")
    with rasterio.open(p) as src:
        assert src.profile["compress"] == "lzw"


def test_write_geotiff_north_up_orientation(tmp_path):
    """Band values must land in the same row/col the arrays hold them in."""
    g = make_grid(half_width=50.0, dx=10.0)
    res = block_bootstrap(make_stack(g, seed=9), n_members=20, levels=(0.5,), seed=0)
    p = tmp_path / "o.tif"
    write_geotiff(res, str(p))
    with rasterio.open(p) as src:
        band = src.read(src.descriptions.index("w_mean") + 1)
    assert np.unravel_index(np.argmax(band), band.shape) == np.unravel_index(
        np.argmax(res.w_mean), res.w_mean.shape
    )


# ==========================================================================
# ffp_period_adapter
# ==========================================================================
class StubModel:
    """Records construction kwargs and returns a flat unit footprint."""

    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        StubModel.calls.append(kwargs)

    def run(self):
        return {"fclim_2d": np.ones((4, 4))}


def good_row(**over):
    row = dict(WS=3.0, WD=270.0, USTAR=0.4, MO_LENGTH=-50.0,
               V_SIGMA=0.4, PBLH_F=1000.0)
    row.update(over)
    return pd.Series(row)


@pytest.fixture(autouse=True)
def _reset_stub():
    StubModel.calls = []
    yield
    StubModel.calls = []


def test_adapter_returns_a_callable():
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=StubModel)
    assert callable(fn)


def test_adapter_runs_the_model_for_a_valid_row():
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=StubModel)
    out = fn(good_row())
    np.testing.assert_allclose(out, 1.0)
    assert len(StubModel.calls) == 1


def test_adapter_passes_the_grid_and_state_through():
    g = make_grid(half_width=50.0, dx=10.0)
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=StubModel)
    fn(good_row())
    kw = StubModel.calls[0]
    assert kw["zm"] == 2.0 and kw["z0"] == 0.05
    assert kw["h"] == 1000.0 and kw["ol"] == -50.0
    assert kw["ustar"] == 0.4 and kw["sigmav"] == 0.4
    assert kw["umean"] == 3.0 and kw["wind_dir"] == 270.0
    assert kw["dx"] == g.dx and kw["dy"] == g.dy
    assert kw["domain"] == (g.x.min(), g.x.max(), g.y.min(), g.y.max())


@pytest.mark.parametrize(
    "col", ["WS", "WD", "USTAR", "MO_LENGTH", "V_SIGMA", "PBLH_F"]
)
def test_adapter_rejects_a_nan_in_any_required_column(col):
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=StubModel)
    assert fn(good_row(**{col: np.nan})) is None
    assert StubModel.calls == []


def test_adapter_rejects_a_missing_required_column():
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=StubModel)
    row = good_row().drop("USTAR")
    assert fn(row) is None


@pytest.mark.parametrize(
    "over, why",
    [
        (dict(PBLH_F=10.0), "boundary layer at or below 10 m"),
        (dict(PBLH_F=5.0), "collapsed boundary layer"),
        (dict(USTAR=0.1), "friction velocity at the 0.1 threshold"),
        (dict(USTAR=0.05), "friction velocity below threshold"),
        (dict(V_SIGMA=0.0), "no crosswind variance"),
        (dict(V_SIGMA=-1.0), "negative crosswind variance"),
    ],
)
def test_adapter_rejects_unusable_turbulence(over, why):
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=StubModel)
    assert fn(good_row(**over)) is None, why


def test_adapter_enforces_the_kljun_roughness_bound():
    g = make_grid()
    # z_m must exceed 20 * z_0; 20 * 0.2 = 4.0 > 2.0
    fn = ffp_period_adapter(g, 2.0, 0.2, model_cls=StubModel)
    assert fn(good_row()) is None


def test_adapter_enforces_the_kljun_boundary_layer_bound():
    g = make_grid()
    # z_m must stay below 0.8 * h; 0.8 * 12 = 9.6 < 20
    fn = ffp_period_adapter(g, 20.0, 0.05, model_cls=StubModel)
    assert fn(good_row(PBLH_F=12.0)) is None


def test_adapter_accepts_a_row_just_inside_the_kljun_bounds():
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=StubModel)
    assert fn(good_row(PBLH_F=11.0)) is not None      # 0.8 * 11 = 8.8 > 2.0


def test_adapter_rejects_extreme_instability():
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=StubModel)
    assert fn(good_row(MO_LENGTH=-0.1)) is None       # z_m / L = -20 < -15.5
    assert fn(good_row(MO_LENGTH=-1.0)) is not None   # z_m / L = -2


def test_adapter_accepts_stable_stratification():
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=StubModel)
    assert fn(good_row(MO_LENGTH=50.0)) is not None


def test_adapter_unwraps_a_bare_array_result():
    class Bare(StubModel):
        def run(self):
            return np.full((2, 2), 3.0)

    g = make_grid()
    out = ffp_period_adapter(g, 2.0, 0.05, model_cls=Bare)(good_row())
    np.testing.assert_allclose(out, 3.0)


def test_adapter_passes_through_a_none_result():
    class Empty(StubModel):
        def run(self):
            return {"fclim_2d": None}

    g = make_grid()
    assert ffp_period_adapter(g, 2.0, 0.05, model_cls=Empty)(good_row()) is None


def test_adapter_returns_an_ndarray_not_a_dataarray():
    class Listy(StubModel):
        def run(self):
            return [[1.0, 2.0], [3.0, 4.0]]

    g = make_grid()
    out = ffp_period_adapter(g, 2.0, 0.05, model_cls=Listy)(good_row())
    assert isinstance(out, np.ndarray)


def test_adapter_does_not_swallow_model_errors():
    class Broken(StubModel):
        def run(self):
            raise RuntimeError("solver diverged")

    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05, model_cls=Broken)
    with pytest.raises(RuntimeError, match="solver diverged"):
        fn(good_row())
    # accumulate_blocks is the layer that turns this into a warning
    df = pd.DataFrame(
        [good_row()], index=pd.DatetimeIndex(["2024-06-24 14:30"])
    )
    with pytest.warns(UserWarning, match="footprint failed"):
        bs = accumulate_blocks(df, g, fn, progress=False)
    assert bs.n_blocks == 0


def test_adapter_default_model_cls_resolves_to_the_ffp_model():
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05)        # imports FFPModel internally
    assert callable(fn)
    assert fn(good_row(USTAR=np.nan)) is None    # screening runs before the model


@pytest.mark.xfail(
    reason="ffp_period_adapter calls model_cls(zm=..., z0=..., ...) but "
           "FFPModel.__init__ requires a `df` positional argument; the default "
           "adapter path cannot run the bundled model",
    strict=True,
)
def test_adapter_default_model_cls_is_call_compatible():
    g = make_grid()
    fn = ffp_period_adapter(g, 2.0, 0.05)
    fn(good_row())


# ==========================================================================
# end to end
# ==========================================================================
def test_full_pipeline_accumulate_bootstrap_write(tmp_path):
    g = make_grid(half_width=100.0, dx=10.0)
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-06-24 00:00", periods=48 * 25, freq="30min")
    df = pd.DataFrame({"x0": rng.normal(0, 15, idx.size),
                       "y0": rng.normal(0, 15, idx.size)}, index=idx)

    def period(row):
        if rng.random() < 0.1:                   # a realistic fraction of gaps
            return None
        return gaussian_xy(g, x0=row.x0, y0=row.y0, sx=25.0, sy=25.0)

    blocks = accumulate_blocks(df, g, period, block_days=1, progress=False)
    assert blocks.n_blocks == 25
    assert 0.0 < blocks.mean_capture <= 1.0

    res = block_bootstrap(blocks, n_members=50, levels=(0.5, 0.8, 0.9), seed=0)
    assert res.n_members > 0
    assert res.w_mean.sum() * g.cell_area == pytest.approx(1.0, rel=1e-4)
    assert np.all(res.p_include[0.9] >= res.p_include[0.5] - 1e-6)

    p = write_geotiff(res, str(tmp_path / "pipeline.tif"))
    with rasterio.open(p) as src:
        assert src.count == 8
        assert src.read(1).shape == g.shape


def test_pipeline_recovers_a_known_off_centre_source_area(tmp_path):
    """A footprint pinned east of the tower must put its source area east."""
    g = make_grid(half_width=100.0, dx=10.0)
    idx = pd.date_range("2024-07-01", periods=48 * 30, freq="30min")
    df = pd.DataFrame({"v": np.zeros(idx.size)}, index=idx)

    fp = gaussian_xy(g, x0=40.0, y0=0.0, sx=15.0, sy=15.0)
    blocks = accumulate_blocks(df, g, lambda row: fp, block_days=1, progress=False)
    res = block_bootstrap(blocks, n_members=30, levels=(0.8,), seed=0)

    peak_row, peak_col = np.unravel_index(np.argmax(res.w_mean), res.w_mean.shape)
    assert g.x[peak_col] == pytest.approx(40.0, abs=g.dx)
    assert g.y[g.shape[0] - 1 - peak_row] == pytest.approx(0.0, abs=g.dy)
    # deterministic input -> zero spread, so every member shares one source area
    assert set(np.unique(res.p_include[0.8])) <= {0.0, 1.0}
