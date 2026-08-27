"""
Climatology-metric tests for :mod:`fluxfootprints.representativeness`.

The fixtures are 2-D Gaussian footprints, whose 80 % contour is an ellipse of
known size and position, so fetch, area, and symmetry (Chu et al., 2021,
Sect. 2.2) are all checkable against theory. For a bivariate normal density
with standard deviations ``sx``, ``sy``, the isoline holding fraction ``r``
sits at Mahalanobis radius ``R = sqrt(-2 ln(1 - r))``, which encloses the
ellipse with semi-axes ``R sx`` and ``R sy``. For a footprint centred on the
tower that gives

    X = R max(sx, sy),   A = pi R**2 sx sy,   S = min(sx, sy) / max(sx, sy),

and for one displaced downwind by ``x0`` along an isotropic sigma, the contour
is a disc of radius ``R sigma`` offset from the tower, so

    X = x0 + R sigma,    A = pi (R sigma)**2,   S = (R sigma / X)**2.

Tolerances are set by the grid: whole cells are counted for area, and the fetch
reaches only to cell centres, so both are resolved to about one cell.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from fluxfootprints.representativeness import (
    ASYMMETRY_THRESHOLD,
    ClimatologyMetrics,
    climatology_metrics,
    contour_level_for_fraction,
    footprint_area,
    footprint_contour_mask,
    footprint_fetch,
    footprint_symmetry,
    symmetry_index,
    truncate_to_contour,
)

SIGMA = 50.0
EXTENT = 400.0  # 8 sigma, so the grid holds essentially the whole footprint
STEP = 5.0
DIAGONAL = STEP * np.sqrt(2.0)


def gaussian_footprint(
    sigma_x: float = SIGMA,
    sigma_y: float = SIGMA,
    extent: float = EXTENT,
    dx: float = STEP,
    dy: float | None = None,
    x0: float = 0.0,
) -> xr.DataArray:
    """Build a 2-D Gaussian footprint density [m-2], optionally offset in x."""
    dy = dx if dy is None else dy
    x = np.arange(-extent, extent + dx / 2, dx)
    y = np.arange(-extent, extent + dy / 2, dy)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    f = np.exp(-0.5 * (((xx - x0) / sigma_x) ** 2 + (yy / sigma_y) ** 2)) / (
        2 * np.pi * sigma_x * sigma_y
    )
    return xr.DataArray(
        f,
        coords={"x": x, "y": y},
        dims=("x", "y"),
        name="fclim",
        attrs={"units": "m-2", "long_name": "footprint climatology"},
    )


def mahalanobis_radius(fraction: float) -> float:
    """Radius, in standard deviations, of the isoline enclosing `fraction`."""
    return float(np.sqrt(-2.0 * np.log(1.0 - fraction)))


def analytic_area(fraction: float, sigma_x: float = SIGMA, sigma_y: float = SIGMA):
    """Area of the ellipse enclosing `fraction` of a Gaussian footprint [m2]."""
    return np.pi * mahalanobis_radius(fraction) ** 2 * sigma_x * sigma_y


# ------------------------------
# Fetch
# ------------------------------


@pytest.mark.parametrize("fraction", [0.5, 0.8, 0.9])
def test_fetch_matches_gaussian_theory(fraction):
    mask = footprint_contour_mask(gaussian_footprint(), fraction=fraction)
    radius = SIGMA * mahalanobis_radius(fraction)
    # Measured to cell centres, so the fetch lands within a cell of the isoline.
    assert footprint_fetch(mask) == pytest.approx(radius, abs=DIAGONAL)


def test_fetch_is_the_farthest_cell_not_the_farthest_along_an_axis():
    # The corner of a square contour is farther out than its edge; the fetch is
    # a radial distance, so it must find the corner.
    x = np.array([-10.0, 0.0, 10.0])
    mask = xr.DataArray(
        np.ones((3, 3), dtype=bool), coords={"x": x, "y": x}, dims=("x", "y")
    )
    assert footprint_fetch(mask) == pytest.approx(np.hypot(10.0, 10.0))


def test_fetch_reads_the_same_contour_from_a_mask_or_a_truncation():
    # A truncated climatology carries its contour in the cells it kept, so the
    # three spellings of "inside the 80 % contour" must agree.
    fclim = gaussian_footprint()
    mask = footprint_contour_mask(fclim, fraction=0.8)
    weights = truncate_to_contour(fclim, fraction=0.8)
    densities = truncate_to_contour(fclim, fraction=0.8, renormalize=False)

    assert footprint_fetch(weights) == footprint_fetch(mask)
    assert footprint_fetch(densities) == footprint_fetch(mask)


def test_fetch_grows_with_fraction():
    fclim = gaussian_footprint()
    fetches = [
        footprint_fetch(truncate_to_contour(fclim, fraction=f))
        for f in (0.5, 0.8, 0.95)
    ]
    assert fetches[0] < fetches[1] < fetches[2]


def test_fetch_is_measured_from_the_tower_not_the_footprint_peak():
    # A footprint displaced downwind reaches farther from the tower, even though
    # its contour is the same size.
    offset = 200.0
    weights = truncate_to_contour(gaussian_footprint(x0=offset), fraction=0.8)
    radius = SIGMA * mahalanobis_radius(0.8)

    assert footprint_fetch(weights) == pytest.approx(offset + radius, abs=DIAGONAL)
    # ...and re-centring the origin on the peak recovers the undisplaced fetch.
    assert footprint_fetch(weights, origin=(offset, 0.0)) == pytest.approx(
        radius, abs=DIAGONAL
    )


def test_fetch_of_an_empty_contour_is_nan():
    empty = xr.zeros_like(gaussian_footprint())
    assert np.isnan(footprint_fetch(empty))
    assert np.isnan(footprint_fetch(empty.astype(bool)))


def test_fetch_requires_coordinates_to_measure_from():
    mask = footprint_contour_mask(gaussian_footprint(), fraction=0.8)
    with pytest.raises(ValueError, match="no 'x' coordinate"):
        footprint_fetch(mask.drop_vars(["x", "y"]))


# ------------------------------
# Area
# ------------------------------


@pytest.mark.parametrize("fraction", [0.5, 0.8, 0.9])
def test_area_matches_gaussian_theory(fraction):
    mask = footprint_contour_mask(gaussian_footprint(), fraction=fraction)
    assert footprint_area(mask) == pytest.approx(analytic_area(fraction), rel=0.05)


def test_area_is_the_cell_count_times_the_cell_area():
    mask = footprint_contour_mask(gaussian_footprint(), fraction=0.8)
    assert footprint_area(mask) == float(mask.sum()) * STEP * STEP


def test_area_matches_theory_on_an_anisotropic_grid():
    fclim = gaussian_footprint(sigma_x=80.0, sigma_y=40.0, dx=10.0, dy=5.0)
    weights = truncate_to_contour(fclim, fraction=0.8)
    assert footprint_area(weights) == pytest.approx(
        analytic_area(0.8, 80.0, 40.0), rel=0.05
    )


def test_area_counts_the_cells_a_truncation_kept():
    fclim = gaussian_footprint()
    mask = footprint_contour_mask(fclim, fraction=0.8)
    weights = truncate_to_contour(fclim, fraction=0.8)
    assert footprint_area(weights) == footprint_area(mask)
    assert footprint_area(weights) == float((weights > 0).sum()) * STEP * STEP


def test_area_accepts_explicit_spacing_when_coordinates_are_absent():
    mask = footprint_contour_mask(gaussian_footprint(), fraction=0.8)
    assert footprint_area(mask.drop_vars(["x", "y"]), dx=STEP, dy=STEP) == (
        footprint_area(mask)
    )


def test_area_of_an_empty_contour_is_zero():
    assert footprint_area(xr.zeros_like(gaussian_footprint())) == 0.0


# ------------------------------
# Symmetry, Eq. 1
# ------------------------------


def test_symmetry_index_is_eq_1():
    assert symmetry_index(area=100.0, fetch=10.0) == pytest.approx(
        100.0 / (np.pi * 100.0)
    )


def test_symmetry_index_is_bounded_above_by_one():
    # Whole cells inside a fetch measured to cell centres can push the raw ratio
    # past 1; the index is a fraction of a disc and must not exceed it.
    assert symmetry_index(area=1000.0, fetch=10.0) == 1.0


@pytest.mark.parametrize("fetch", [0.0, -1.0, np.nan, np.inf])
def test_symmetry_index_without_a_fetch_is_nan(fetch):
    assert np.isnan(symmetry_index(area=100.0, fetch=fetch))


def test_symmetry_of_a_centred_isotropic_footprint_is_one():
    weights = truncate_to_contour(gaussian_footprint(), fraction=0.8)
    assert footprint_symmetry(weights) == pytest.approx(1.0, abs=0.01)


def test_symmetry_of_an_elongated_footprint_is_the_axis_ratio():
    # An ellipse of semi-axes a, b has A / (pi a**2) = b / a.
    fclim = gaussian_footprint(sigma_x=80.0, sigma_y=40.0, dx=10.0, dy=5.0)
    weights = truncate_to_contour(fclim, fraction=0.8)
    assert footprint_symmetry(weights) == pytest.approx(40.0 / 80.0, abs=0.05)


def test_symmetry_falls_as_the_footprint_is_displaced_from_the_tower():
    symmetries = [
        footprint_symmetry(truncate_to_contour(gaussian_footprint(x0=x0), fraction=0.8))
        for x0 in (0.0, 100.0, 200.0)
    ]
    assert symmetries[0] > symmetries[1] > symmetries[2]


def test_a_displaced_footprint_is_asymmetric_by_the_papers_threshold():
    # A disc of radius R sigma centred x0 downwind fills (R sigma / X)**2 of the
    # disc of radius X = x0 + R sigma that the index compares it against.
    offset = 200.0
    radius = SIGMA * mahalanobis_radius(0.8)
    weights = truncate_to_contour(gaussian_footprint(x0=offset), fraction=0.8)

    expected = (radius / (offset + radius)) ** 2
    assert footprint_symmetry(weights) == pytest.approx(expected, abs=0.02)
    assert footprint_symmetry(weights) < ASYMMETRY_THRESHOLD


def test_asymmetry_threshold_is_the_papers_value():
    assert ASYMMETRY_THRESHOLD == 0.30


def test_footprint_symmetry_is_the_index_of_the_measured_area_and_fetch():
    weights = truncate_to_contour(gaussian_footprint(x0=120.0), fraction=0.8)
    assert footprint_symmetry(weights) == symmetry_index(
        footprint_area(weights), footprint_fetch(weights)
    )


def test_symmetry_of_an_empty_contour_is_nan():
    assert np.isnan(footprint_symmetry(xr.zeros_like(gaussian_footprint())))


# ------------------------------
# climatology_metrics
# ------------------------------


def test_metrics_gather_the_three_measures_and_the_cell_count():
    fclim = gaussian_footprint()
    mask = footprint_contour_mask(fclim, fraction=0.8)
    metrics = climatology_metrics(fclim, fraction=0.8)

    assert isinstance(metrics, ClimatologyMetrics)
    assert metrics.fetch == footprint_fetch(mask)
    assert metrics.area == footprint_area(mask)
    assert metrics.symmetry == footprint_symmetry(mask)
    assert metrics.n_cells == int(mask.sum())
    assert metrics.fraction == 0.8
    assert metrics.contour_level == contour_level_for_fraction(fclim, fraction=0.8)


def test_metrics_report_the_mass_the_contour_encloses():
    fclim = gaussian_footprint()
    metrics = climatology_metrics(fclim, fraction=0.8)
    # Cells tied at the threshold ride along, so the contour can only overshoot.
    assert metrics.enclosed_fraction >= 0.8 - 1e-9
    assert metrics.enclosed_fraction == pytest.approx(0.8, abs=0.01)


def test_metrics_match_gaussian_theory():
    metrics = climatology_metrics(gaussian_footprint(), fraction=0.8)
    assert metrics.fetch == pytest.approx(SIGMA * mahalanobis_radius(0.8), abs=DIAGONAL)
    assert metrics.area == pytest.approx(analytic_area(0.8), rel=0.05)
    assert metrics.symmetry == pytest.approx(1.0, abs=0.01)


def test_metrics_do_not_truncate_an_already_truncated_climatology_again():
    # Re-thresholding renormalised weights at 80 % would cut the source area
    # down to the inner 64 %; the truncation's own contour must be reused.
    fclim = gaussian_footprint()
    raw = climatology_metrics(fclim, fraction=0.8)
    truncated = climatology_metrics(truncate_to_contour(fclim, fraction=0.8))

    assert truncated.n_cells == raw.n_cells
    assert truncated.fetch == raw.fetch
    assert truncated.area == raw.area
    assert truncated.symmetry == raw.symmetry
    assert truncated.fraction == raw.fraction
    assert truncated.contour_level == raw.contour_level


def test_metrics_of_a_truncated_climatology_cannot_report_enclosed_mass():
    # The mass outside the contour has been zeroed, so the share retained is
    # gone with it.
    weights = truncate_to_contour(gaussian_footprint(), fraction=0.8)
    assert np.isnan(climatology_metrics(weights).enclosed_fraction)


def test_metrics_honour_a_non_default_fraction():
    fclim = gaussian_footprint()
    half = climatology_metrics(fclim, fraction=0.5)
    most = climatology_metrics(fclim, fraction=0.9)

    assert half.fraction == 0.5
    assert half.n_cells < most.n_cells
    assert half.area < most.area
    assert half.fetch < most.fetch
    assert half.contour_level > most.contour_level


def test_metrics_carry_the_overlap_indices_through():
    fclim = gaussian_footprint()
    assert climatology_metrics(fclim).seasonal_overlap is None
    assert climatology_metrics(fclim).daynight_overlap is None

    metrics = climatology_metrics(fclim, seasonal_overlap=0.62, daynight_overlap=0.71)
    assert metrics.seasonal_overlap == pytest.approx(0.62)
    assert metrics.daynight_overlap == pytest.approx(0.71)


def test_metrics_ignore_non_finite_cells():
    fclim = gaussian_footprint()
    padded = fclim.where(np.hypot(fclim.x, fclim.y) < 300.0)
    zeroed = padded.fillna(0.0)
    assert climatology_metrics(padded) == climatology_metrics(zeroed)


def test_metrics_need_coordinates_even_when_the_spacing_is_supplied():
    # Spacing alone sizes the cells but does not say where they sit, and the
    # fetch is a distance from the tower: better to say so than to hand back a
    # nan fetch inside an otherwise complete result.
    bare = gaussian_footprint().drop_vars(["x", "y"])
    with pytest.raises(ValueError, match="Cannot infer dx"):
        climatology_metrics(bare)
    with pytest.raises(ValueError, match="Cannot measure fetch"):
        climatology_metrics(bare, dx=STEP, dy=STEP)
