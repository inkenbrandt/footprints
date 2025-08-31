# tests/test_ls_footprint_model.py
import os
import sys
import math
import numpy as np
import pytest

# Use the requested path shim to import from ../src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints import (  # noqa: E402
    KAPPA,
    LSFootprintConfig,
    BackwardLSModel,
    log_wind_profile,
    sigma_w,
    sigma_v,
    lagrangian_timescale,
)


# -------------------------
# Utility/math sanity tests
# -------------------------

def test_log_wind_profile_exact_value():
    z, ustar, z0 = 10.0, 0.4, 0.1
    expected = (ustar / KAPPA) * math.log(z / z0)
    assert np.isclose(log_wind_profile(z, ustar, z0), expected, rtol=0, atol=1e-12)

def test_log_wind_profile_monotonic_in_z_and_z0():
    ustar = 0.5
    # Increasing z increases U(z)
    U1 = log_wind_profile(5.0, ustar, 0.1)
    U2 = log_wind_profile(10.0, ustar, 0.1)
    assert U2 > U1
    # Increasing z0 decreases U(z)
    U3 = log_wind_profile(10.0, ustar, 0.2)
    assert U3 < U2

def test_sigma_scaling():
    for ustar in [0.2, 0.4, 0.8]:
        assert np.isclose(sigma_w(ustar), 1.3 * ustar)
        assert np.isclose(sigma_v(ustar), 2.0 * ustar)

def test_lagrangian_timescale_positive_and_increasing():
    ustar = 0.3
    zm = 20.0
    z_vals = np.array([1.0, 5.0, 10.0, 20.0])
    tl = [lagrangian_timescale(z, zm, ustar) for z in z_vals]
    assert all(t > 0 for t in tl)
    # Roughly increasing with z
    assert tl[0] < tl[-1]


# -------------------------
# Config and RNG
# -------------------------

def test_config_rng_reproducible():
    cfg1 = LSFootprintConfig(zm=10, ustar=0.4, L=-50, h=800, wind_dir_deg=270, seed=123)
    cfg2 = LSFootprintConfig(zm=10, ustar=0.4, L=-50, h=800, wind_dir_deg=270, seed=123)
    r1 = cfg1.rng().normal(size=5)
    r2 = cfg2.rng().normal(size=5)
    assert np.allclose(r1, r2)


# -------------------------
# Model/grid construction
# -------------------------

def test_model_grid_shapes():
    cfg = LSFootprintConfig(
        zm=20.0, ustar=0.4, L=-50.0, h=1000.0, wind_dir_deg=270.0,
        domain=(2000.0, 2000.0), dx=20.0, dy=20.0, n_particles=100, seed=1
    )
    m = BackwardLSModel(cfg)
    # Expected number of bins:
    nx = int(cfg.domain[0] / cfg.dx)         # x in [-x_max, 0] → x_max/dx bins
    ny = int((2 * cfg.domain[1]) / cfg.dy)   # y in [-y_max, y_max] → 2*y_max/dy bins
    assert m.footprint_2d.shape == (nx, ny)
    # Bin centers lengths also match
    x_c, y_c, _ = m.footprint()
    assert x_c.shape[0] == nx
    assert y_c.shape[0] == ny


# -------------------------
# Accumulation helper
# -------------------------

def test_accumulate_counts_increase_sum():
    cfg = LSFootprintConfig(
        zm=20.0, ustar=0.4, L=-50.0, h=1000.0, wind_dir_deg=270.0,
        domain=(200.0, 200.0), dx=10.0, dy=10.0, n_particles=10, seed=2
    )
    m = BackwardLSModel(cfg)
    before = m.footprint_2d.sum()
    # Two touchdown points inside domain
    xs = np.array([-5.0, -15.0])
    ys = np.array([0.0, 0.0])
    m._accumulate(xs, ys)
    after = m.footprint_2d.sum()
    # histogram2d counts each point once
    assert np.isclose(after - before, 2.0)


# -------------------------
# Simulation fixtures
# -------------------------

@pytest.fixture(scope="module")
def model_result():
    """Run a moderate simulation once and share results."""
    cfg = LSFootprintConfig(
        zm=20.0,
        ustar=0.4,
        L=-50.0,
        h=1000.0,
        wind_dir_deg=270.0,
        z0=0.1,
        n_particles=4000,   # keep tests fast
        dt=0.25,
        t_max=600.0,
        domain=(2000.0, 2000.0),
        dx=20.0,
        dy=20.0,
        seed=42,
    )
    m = BackwardLSModel(cfg)
    m.run()
    x_c, y_c, F = m.footprint()
    return cfg, m, x_c, y_c, F


# -------------------------
# Simulation behavior
# -------------------------

def test_run_normalization_and_nonneg(model_result):
    cfg, m, x_c, y_c, F = model_result
    # All non-negative
    assert np.all(F >= 0)
    # Normalization: ∑ F * dx * dy ≈ 1
    mass = (F * cfg.dx * cfg.dy).sum()
    assert np.isfinite(F).all()
    assert np.isclose(mass, 1.0, rtol=1e-3, atol=5e-4)

def test_crosswind_integrated_consistency(model_result):
    cfg, m, x_c, y_c, F = model_result
    xx, fx = m.crosswind_integrated()
    # Same x coordinate array
    assert np.allclose(xx, x_c)
    # f(x) equals integral over y
    fx_from_F = (F.sum(axis=1) * cfg.dy)
    assert np.allclose(fx, fx_from_F, rtol=1e-12, atol=1e-12)
    # And ∑ f(x) dx ≈ 1
    assert np.isclose(np.sum(fx) * cfg.dx, 1.0, rtol=1e-3, atol=5e-4)

def test_update_velocity_shapes_and_finiteness():
    cfg = LSFootprintConfig(zm=15, ustar=0.5, L=-100, h=600, wind_dir_deg=270, seed=7)
    m = BackwardLSModel(cfg)
    n = 128
    z = np.full(n, cfg.zm)
    u = np.zeros(n)
    v = np.zeros(n)
    w = np.zeros(n)
    u2, v2, w2 = m._update_velocity(z, u, v, w)
    assert u2.shape == (n,) and v2.shape == (n,) and w2.shape == (n,)
    assert np.isfinite(u2).all() and np.isfinite(v2).all() and np.isfinite(w2).all()
