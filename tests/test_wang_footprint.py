# test_wang_footprint.py
import os
import sys
import numpy as np
import pytest

# Import path shim (as requested)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints.wang_footprint import wang2006_fy, reconstruct_gaussian_2d

# -------------------------
# Helpers for test scenarios
# -------------------------
def make_valid_params():
    """
    Choose a parameter set within Wang et al. (2006) stated validity:
      - convective: L < 0 and 0.01 <= (-L/h) <= 0.1
      - 0.1 h <= zm <= 0.6 h
    """
    h = 1000.0          # m
    zm = 200.0          # 0.2 h (within [0.1, 0.6] h)
    L = -100.0          # -L/h = 0.1 (upper end of stated range)
    return zm, h, L


def monotone_increasing(a):
    return np.all(np.diff(a) >= -1e-12)  # allow tiny numerical noise


def monotone_decreasing(a):
    return np.all(np.diff(a) <= 1e-12)


def column_std(Fcol, y):
    """Return std dev of a non-negative column over y (normalized internally)."""
    area = np.trapz(Fcol, y)
    if area == 0 or not np.isfinite(area):
        return np.nan
    w = Fcol / area
    mu = np.trapz(y * w, y)
    var = np.trapz((y - mu) ** 2 * w, y)
    return np.sqrt(var)


# ---------------
# Unit test suite
# ---------------
def test_wang2006_fy_raises_on_negative_x():
    zm, h, L = make_valid_params()
    x = np.array([-1.0, 0.0, 1.0])  # contains negative
    with pytest.raises(ValueError):
        _ = wang2006_fy(x, zm=zm, h=h, L=L)


def test_wang2006_fy_raises_on_nonconvective_L():
    zm, h, _ = make_valid_params()
    x = np.linspace(0.0, 5000.0, 501)
    with pytest.raises(ValueError):
        _ = wang2006_fy(x, zm=zm, h=h, L=1.0)  # not convective


@pytest.mark.parametrize("num, span", [(2001, 6000.0), (1001, 6000.0)])
def test_wang2006_fy_normalizes_to_one_uniform_grid(num, span):
    zm, h, L = make_valid_params()
    x = np.linspace(0.0, span, num)
    fy = wang2006_fy(x, zm=zm, h=h, L=L)
    integ = np.trapz(fy, x)
    assert np.isfinite(integ)
    assert np.isclose(integ, 1.0, rtol=0, atol=2e-3)


def test_wang2006_fy_normalizes_to_one_nonuniform_grid():
    zm, h, L = make_valid_params()
    # Nonuniform spacing: denser near the origin
    x = np.unique(
        np.concatenate([np.linspace(0, 500, 400), np.linspace(500, 6000, 200)])
    )
    fy = wang2006_fy(x, zm=zm, h=h, L=L)
    # Use general integral (trapz handles nonuniform axis)
    integ = np.trapz(fy, x)
    assert np.isfinite(integ)
    assert np.isclose(integ, 1.0, rtol=0, atol=3e-3)


def test_wang2006_fy_has_single_peak_and_is_unimodal():
    zm, h, L = make_valid_params()
    x = np.linspace(0.0, 6000.0, 3001)  # fine resolution for shape test
    fy = wang2006_fy(x, zm=zm, h=h, L=L)

    # location of maximum
    k = np.argmax(fy)
    left = fy[:k+1]
    right = fy[k:]
    assert monotone_increasing(left)
    assert monotone_decreasing(right)

    # Peak must not be at domain edges
    assert 0 < k < len(x) - 1


def test_reconstruct_gaussian_2d_shapes_and_normalization_default_alpha():
    zm, h, L = make_valid_params()
    x = np.linspace(0.0, 6000.0, 601)
    fy = wang2006_fy(x, zm=zm, h=h, L=L)

    ny = 301
    X, Y, F = reconstruct_gaussian_2d(x, fy, ny=ny, alpha=0.3)

    # Shape expectations for indexing="xy": (ny, nx)
    assert X.shape == (ny, len(x))
    assert Y.shape == (ny, len(x))
    assert F.shape == (ny, len(x))

    # Global normalization: ∬ F dx dy = 1
    dx = np.gradient(x)
    y = Y[:, 0]  # Y grid is same across columns
    dy = y[1] - y[0]
    total = np.sum(F * dx[None, :] * dy)
    assert np.isfinite(total)
    assert np.isclose(total, 1.0, rtol=0, atol=3e-3)

    # Symmetry about y=0
    col = len(x) // 2
    col_vals = F[:, col]
    assert np.allclose(col_vals, col_vals[::-1], rtol=1e-10, atol=1e-10)


def test_reconstruct_gaussian_2d_column_integrates_back_to_fy():
    zm, h, L = make_valid_params()
    x = np.linspace(0.0, 6000.0, 801)
    fy = wang2006_fy(x, zm=zm, h=h, L=L)

    ny = 401
    X, Y, F = reconstruct_gaussian_2d(x, fy, ny=ny, alpha=0.25)

    y = Y[:, 0]
    # Recover fy by integrating over y for each column
    fy_recovered = np.trapz(F, y, axis=0)

    # Allow a small tolerance on the per-column agreement.
    mask = fy > (0.01 * fy.max())
    rel_err = np.abs(fy_recovered[mask] - fy[mask]) / fy[mask]
    assert np.nanmax(rel_err) < 0.05  # within 5% for meaningful columns


def test_reconstruct_gaussian_2d_sigma_source_options_affect_spread():
    zm, h, L = make_valid_params()
    x = np.linspace(0.0, 6000.0, 601)
    fy = wang2006_fy(x, zm=zm, h=h, L=L)

    # Compare alpha-law vs. sigma_v/U ballistic law where sigma_v/U is larger.
    ny = 301
    _, Y1, F1 = reconstruct_gaussian_2d(x, fy, ny=ny, alpha=0.1)
    _, Y2, F2 = reconstruct_gaussian_2d(x, fy, ny=ny, sigma_v=1.0, U=2.0)  # sigma_y = 0.5 * x

    y = Y1[:, 0]  # same y grid shape

    # Pick a column sufficiently downwind so spreads are measurable
    i = int(0.6 * (len(x) - 1))

    std_alpha = column_std(F1[:, i], y)
    std_ballistic = column_std(F2[:, i], y)

    # Numerical tie-breaking tolerance (can differ at 1e-13 level)
    assert std_ballistic >= std_alpha - 1e-9


def test_reconstruct_gaussian_2d_mismatched_shapes_raises():
    zm, h, L = make_valid_params()
    x = np.linspace(0.0, 6000.0, 501)
    fy = wang2006_fy(x, zm=zm, h=h, L=L)
    # Chop one element to cause mismatch
    with pytest.raises(ValueError):
        _ = reconstruct_gaussian_2d(x[:-1], fy, ny=201)


def test_end_to_end_basic_consistency():
    """
    Smoke test running both stages over a modest grid.
    Ensures outputs are finite and positive (no NaNs/Infs and no negative densities).
    Also verifies the expected relationship between column std and centerline amplitude.
    """
    zm, h, L = make_valid_params()
    x = np.linspace(0.0, 4000.0, 401)
    fy = wang2006_fy(x, zm=zm, h=h, L=L)
    assert np.isfinite(fy).all()
    assert (fy >= 0).all()
    assert fy.max() > 0

    X, Y, F = reconstruct_gaussian_2d(x, fy, ny=201, alpha=0.3)
    assert np.isfinite(X).all() and np.isfinite(Y).all() and np.isfinite(F).all()
    assert (F >= 0).all()

    # Centerline row (y=0)
    midrow = F.shape[0] // 2
    y = Y[:, 0]

    # Estimate per-column sigma from the reconstructed column shape
    sigmas = np.array([column_std(F[:, i], y) for i in range(F.shape[1])])

    # Mask out columns with tiny mass where numerical noise dominates
    mask = fy > (0.01 * fy.max())
    Fmid = F[midrow, :]

    # For a Gaussian column with std sigma, centerline amplitude A ~ fy / (sqrt(2π) * sigma)
    # => A * sigma is proportional to fy with an (almost) constant factor across x.
    lhs = Fmid[mask] * sigmas[mask]
    rhs = fy[mask]

    # High rank correlation (monotonic relationship), more robust than raw Pearson here
    from scipy.stats import spearmanr
    rho, _ = spearmanr(lhs, rhs)
    assert rho > 0.97

    # And the proportionality constant should be close to 1/sqrt(2π) (~0.3989)
    k_med = np.median(lhs / rhs)
    assert np.isfinite(k_med)
    assert np.isclose(k_med, 1.0 / np.sqrt(2.0 * np.pi), rtol=0.25, atol=0.02)