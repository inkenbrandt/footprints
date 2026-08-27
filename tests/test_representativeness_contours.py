"""
Contour truncation tests for :mod:`fluxfootprints.representativeness`.

The fixtures are isotropic and anisotropic 2-D Gaussian footprints, for which
the enclosed source fraction is known in closed form. For a bivariate normal
density with standard deviations ``sx``, ``sy``, the level sets are ellipses of
constant Mahalanobis radius ``R``, and the mass they enclose is

    P(R) = 1 - exp(-R**2 / 2),

so the isoline holding fraction ``r`` sits at ``R = sqrt(-2 ln(1 - r))`` and
carries the density

    f(R) = (1 - r) / (2 pi sx sy).

Both the contour level and the area it encloses are therefore checkable against
theory rather than against a previous run of the code.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from fluxfootprints.base_footprint_model import _source_weight_threshold
from fluxfootprints.representativeness import (
    DEFAULT_CONTOUR_FRACTION,
    contour_level_for_fraction,
    footprint_contour_mask,
    truncate_to_contour,
)

SIGMA = 50.0
EXTENT = 400.0  # 8 sigma, so the grid holds essentially the whole footprint
STEP = 5.0


def gaussian_footprint(
    sigma_x: float = SIGMA,
    sigma_y: float = SIGMA,
    extent: float = EXTENT,
    dx: float = STEP,
    dy: float | None = None,
) -> xr.DataArray:
    """Build a 2-D Gaussian footprint density [m-2] on an (x, y) grid."""
    dy = dx if dy is None else dy
    x = np.arange(-extent, extent + dx / 2, dx)
    y = np.arange(-extent, extent + dy / 2, dy)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    f = np.exp(-0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2)) / (
        2 * np.pi * sigma_x * sigma_y
    )
    return xr.DataArray(
        f,
        coords={"x": x, "y": y},
        dims=("x", "y"),
        name="fclim",
        attrs={"units": "m-2", "long_name": "footprint climatology"},
    )


def analytic_level(fraction: float, sigma_x: float = SIGMA, sigma_y: float = SIGMA):
    """Density on the isoline enclosing `fraction` of a Gaussian footprint."""
    return (1.0 - fraction) / (2 * np.pi * sigma_x * sigma_y)


def analytic_area(fraction: float, sigma_x: float = SIGMA, sigma_y: float = SIGMA):
    """Area of the ellipse enclosing `fraction` of a Gaussian footprint [m2]."""
    return -2.0 * np.log(1.0 - fraction) * np.pi * sigma_x * sigma_y


def enclosed_fraction(fclim: xr.DataArray, mask: xr.DataArray) -> float:
    """Share of the grid's total source weight that falls inside `mask`."""
    return float(fclim.where(mask, 0.0).sum() / fclim.sum())


# ------------------------------
# Contour level
# ------------------------------


@pytest.mark.parametrize("fraction", [0.5, 0.8, 0.9])
def test_contour_level_matches_gaussian_theory(fraction):
    fclim = gaussian_footprint()
    level = contour_level_for_fraction(fclim, fraction=fraction)
    assert level == pytest.approx(analytic_level(fraction), rel=0.02)


def test_contour_level_matches_theory_on_an_anisotropic_grid():
    # Elongated footprint on rectangular cells: exercises dx != dy.
    fclim = gaussian_footprint(sigma_x=80.0, sigma_y=40.0, dx=10.0, dy=5.0)
    level = contour_level_for_fraction(fclim, fraction=0.8)
    assert level == pytest.approx(analytic_level(0.8, 80.0, 40.0), rel=0.02)


def test_contour_level_reuses_the_shared_kernel():
    # The public helper must be the model's threshold logic, not a second copy.
    fclim = gaussian_footprint()
    assert contour_level_for_fraction(fclim, fraction=0.8) == _source_weight_threshold(
        fclim.values, STEP * STEP, 0.8
    )


def test_contour_level_falls_with_fraction():
    fclim = gaussian_footprint()
    levels = [contour_level_for_fraction(fclim, fraction=f) for f in (0.5, 0.8, 0.95)]
    assert levels[0] > levels[1] > levels[2] > 0


def test_explicit_spacing_matches_inferred_spacing():
    fclim = gaussian_footprint()
    assert contour_level_for_fraction(
        fclim, dx=STEP, dy=STEP, fraction=0.8
    ) == contour_level_for_fraction(fclim, fraction=0.8)


def test_spacing_must_be_given_when_coordinates_are_absent():
    fclim = gaussian_footprint().drop_vars(["x", "y"])
    with pytest.raises(ValueError, match="no 'x' coordinate"):
        contour_level_for_fraction(fclim, fraction=0.8)
    # ...and is then usable with the spacing supplied.
    assert contour_level_for_fraction(
        fclim, dx=STEP, dy=STEP, fraction=0.8
    ) == pytest.approx(analytic_level(0.8), rel=0.02)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5, np.nan])
def test_fraction_outside_the_unit_interval_is_rejected(fraction):
    fclim = gaussian_footprint()
    with pytest.raises(ValueError, match="fraction"):
        contour_level_for_fraction(fclim, fraction=fraction)


def test_empty_footprint_is_rejected():
    fclim = gaussian_footprint() * 0.0
    with pytest.raises(ValueError, match="no positive source weight"):
        contour_level_for_fraction(fclim, fraction=0.8)


def test_fraction_beyond_the_captured_mass_saturates():
    # A domain clipped at 0.8 sigma holds far less than 80 % of the footprint,
    # so the contour degrades to the smallest positive density on the grid.
    fclim = gaussian_footprint(extent=40.0)
    assert float(fclim.sum()) * STEP * STEP < 0.8
    level = contour_level_for_fraction(fclim, fraction=0.8)
    assert level == pytest.approx(float(fclim.min()))
    assert bool(footprint_contour_mask(fclim, fraction=0.8).all())


def test_non_finite_cells_are_ignored():
    # NaNs sort ahead of every real weight once the order is reversed, so they
    # must be dropped before the cumulative sum rather than poisoning it.
    fclim = gaussian_footprint()
    padded = fclim.where(np.hypot(fclim.x, fclim.y) < 300.0)
    zeroed = padded.fillna(0.0)
    assert contour_level_for_fraction(padded, fraction=0.8) == pytest.approx(
        contour_level_for_fraction(zeroed, fraction=0.8)
    )


# ------------------------------
# Contour mask
# ------------------------------


def test_contour_mask_is_the_analytic_disc():
    fraction = 0.8
    fclim = gaussian_footprint()
    mask = footprint_contour_mask(fclim, fraction=fraction)

    radius = SIGMA * np.sqrt(-2.0 * np.log(1.0 - fraction))
    cell_radius = np.hypot(*np.meshgrid(fclim.x, fclim.y, indexing="ij"))
    diagonal = STEP * np.sqrt(2.0)

    assert bool(mask.where(cell_radius <= radius - diagonal, True).all())
    assert not bool(mask.where(cell_radius >= radius + diagonal, False).any())


@pytest.mark.parametrize("fraction", [0.5, 0.8, 0.9])
def test_contour_mask_encloses_the_requested_fraction(fraction):
    fclim = gaussian_footprint()
    mask = footprint_contour_mask(fclim, fraction=fraction)
    kept = enclosed_fraction(fclim, mask)
    # Cells tied at the threshold ride along, so the mask can only overshoot,
    # and then only by the mass of that ring.
    assert kept >= fraction - 1e-9
    assert kept == pytest.approx(fraction, abs=0.01)


@pytest.mark.parametrize("fraction", [0.5, 0.8, 0.9])
def test_contour_mask_area_matches_gaussian_theory(fraction):
    fclim = gaussian_footprint()
    mask = footprint_contour_mask(fclim, fraction=fraction)
    area = float(mask.sum()) * STEP * STEP
    assert area == pytest.approx(analytic_area(fraction), rel=0.05)


def test_contour_mask_is_boolean_and_grid_aligned():
    fclim = gaussian_footprint()
    mask = footprint_contour_mask(fclim, fraction=0.8)
    assert mask.dtype == bool
    assert mask.dims == fclim.dims
    np.testing.assert_array_equal(mask.x.values, fclim.x.values)
    np.testing.assert_array_equal(mask.y.values, fclim.y.values)
    assert mask.attrs["contour_fraction"] == 0.8
    assert mask.attrs["contour_level"] == contour_level_for_fraction(
        fclim, fraction=0.8
    )


def test_contour_mask_grows_with_fraction():
    fclim = gaussian_footprint()
    counts = [
        int(footprint_contour_mask(fclim, fraction=f).sum()) for f in (0.5, 0.8, 0.95)
    ]
    assert counts[0] < counts[1] < counts[2]


# ------------------------------
# Truncation and renormalisation
# ------------------------------


@pytest.mark.parametrize("fraction", [0.5, 0.8, 0.9])
def test_truncated_weights_sum_to_unity(fraction):
    # Chu et al. (2021), Sect. 2.2: "the footprint weights were rescaled to sum
    # up to unity within the 80 % contours".
    truncated = truncate_to_contour(gaussian_footprint(), fraction=fraction)
    assert float(truncated.sum()) == pytest.approx(1.0, abs=1e-12)


def test_truncation_zeroes_the_cells_outside_the_contour():
    fclim = gaussian_footprint()
    mask = footprint_contour_mask(fclim, fraction=0.8)
    truncated = truncate_to_contour(fclim, fraction=0.8)

    assert bool((truncated.where(~mask, 0.0) == 0.0).all())
    assert bool((truncated.where(mask, 1.0) > 0.0).all())
    assert int((truncated > 0).sum()) == int(mask.sum())


def test_truncation_preserves_the_relative_weights():
    # Renormalising rescales every retained cell by one constant, so the ratio
    # between any two of them is untouched.
    fclim = gaussian_footprint()
    truncated = truncate_to_contour(fclim, fraction=0.8)
    ratio = (truncated / fclim).where(truncated > 0.0)
    assert float(ratio.max() - ratio.min()) == pytest.approx(0.0, abs=1e-12)
    assert float(ratio.max()) == pytest.approx(
        1.0 / float(fclim.where(footprint_contour_mask(fclim, fraction=0.8), 0.0).sum())
    )


def test_truncation_without_renormalisation_keeps_the_densities():
    fclim = gaussian_footprint()
    mask = footprint_contour_mask(fclim, fraction=0.8)
    truncated = truncate_to_contour(fclim, fraction=0.8, renormalize=False)

    xr.testing.assert_allclose(truncated.where(mask), fclim.where(mask))
    assert float(truncated.sum()) * STEP * STEP == pytest.approx(0.8, abs=0.01)
    assert truncated.attrs["contour_renormalized"] == "false"
    assert truncated.attrs["units"] == "m-2"


def test_truncation_records_the_contour_in_attrs():
    fclim = gaussian_footprint()
    truncated = truncate_to_contour(fclim, fraction=0.8)
    mask = footprint_contour_mask(fclim, fraction=0.8)

    assert truncated.attrs["contour_fraction"] == 0.8
    assert truncated.attrs["contour_level"] == contour_level_for_fraction(
        fclim, fraction=0.8
    )
    assert truncated.attrs["contour_n_cells"] == int(mask.sum())
    assert truncated.attrs["contour_renormalized"] == "true"
    assert truncated.attrs["units"] == "1"
    # Metadata carried by the input survives alongside the new keys.
    assert truncated.attrs["long_name"] == "footprint climatology"
    assert truncated.name == fclim.name
    assert fclim.attrs["units"] == "m-2", "the input must not be mutated"


def test_truncation_defaults_to_the_eighty_percent_contour():
    fclim = gaussian_footprint()
    assert DEFAULT_CONTOUR_FRACTION == 0.8
    xr.testing.assert_identical(
        truncate_to_contour(fclim), truncate_to_contour(fclim, fraction=0.8)
    )


def test_truncation_leaves_no_nan_behind():
    fclim = gaussian_footprint()
    padded = fclim.where(np.hypot(fclim.x, fclim.y) < 300.0)
    truncated = truncate_to_contour(padded, fraction=0.8)
    assert bool(np.isfinite(truncated).all())
    assert float(truncated.sum()) == pytest.approx(1.0, abs=1e-12)


def test_truncated_weights_reproduce_a_known_weighted_mean():
    # A field that is 1 inside the 80 % contour and 0 outside must average to 1
    # under the truncated weights, whatever the underlying densities were.
    fclim = gaussian_footprint()
    weights = truncate_to_contour(fclim, fraction=0.8)
    field = footprint_contour_mask(fclim, fraction=0.8).astype(float)
    assert float((weights * field).sum()) == pytest.approx(1.0, abs=1e-12)
