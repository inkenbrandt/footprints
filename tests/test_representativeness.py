"""
End-to-end coverage for :mod:`fluxfootprints.representativeness`.

Where the topic-specific suites in this directory each drill into one part of
the module, this file walks the whole of Chu et al. (2021) once: the geometry
of Sect. 2.2, the overlap indices of Eqs. 2-3, the weighted statistics of
Eqs. 5-6, the three-level indices of Sect. 2.4, and the driver that assembles
them. Every check is anchored to something knowable independently of the code.

**Geometry from the Gaussian.** For the bivariate normal density

    f(x, y) = exp(-((x - x0)^2 / sx^2 + y^2 / sy^2) / 2) / (2 pi sx sy),

the isoline enclosing a fraction ``r`` of the mass sits at Mahalanobis radius
``R = sqrt(-2 ln(1 - r))`` and carries the density ``(1 - r) / (2 pi sx sy)``.
It bounds the ellipse with semi-axes ``R sx`` and ``R sy``, so a climatology
centred on the tower has

    X80 = R max(sx, sy),  A80 = pi R^2 sx sy,  S80 = min(sx, sy) / max(sx, sy),

and one displaced downwind by ``x0`` along an isotropic sigma bounds a disc of
radius ``R sigma`` offset from the tower, giving

    X80 = x0 + R sigma,  A80 = pi (R sigma)^2,  S80 = (R sigma / X80)^2.

The grid sets the tolerances: areas count whole cells and the fetch reaches
only to cell centres, so both resolve to about one cell either way.

**Overlap from arithmetic.** Eqs. 2-3 are checked on two- and four-cell
footprints whose geometric means close in exact form -- three months holding
``(1/2, 1/4, 1/4)`` in rotation give ``O80_season = 3 * 2^(-5/3)``, and a month
pairing ``(1/4, 3/4)`` against ``(3/4, 1/4)`` gives ``O80_daynight = sqrt(3)/2``.

**Weighted statistics from a 5x5 raster.** On a five-by-five grid of 10 m cells
the 15 m disc holds exactly the nine central cells and the 25 m disc exactly
twenty-one, so target-area means and compositions are ratios of small integers
and the sensor location bias of Eq. 6 comes out at 4/3 by hand.

**Properties over a concentric landscape.** The two property tests use a site
whose heterogeneity grows with distance: a footprint confined to the innermost
ring, land-cover rings around it, and vegetation-index scenes whose far field
varies independently of their core. Representativeness there can only get worse
as the target area grows, and every index the module reports must stay in
[0, 1] whatever footprint it is handed.

**Integration.** One slow-marked test drives the whole pipeline from the
bundled AmeriFlux US-CRT record in ``data/``, and checks that the tidy frame
carries the documented index, columns, dtypes, and scope/kind blocks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxfootprints.representativeness import (
    ASYMMETRY_THRESHOLD,
    BIAS_THRESHOLD,
    CATEGORICAL_KIND,
    CLIMATOLOGY_KIND,
    CONTINUOUS_KIND,
    MIN_MATCHES,
    PERIOD_SCOPE,
    RESULT_COLUMNS,
    RESULT_INDEX,
    SITE_SCOPE,
    SITE_YEAR_SCOPE,
    TARGET_RADII,
    CategoricalResult,
    ClimatologyMetrics,
    ContinuousResult,
    Level,
    assess_representativeness,
    categorical_representativeness,
    classify_categorical,
    classify_continuous,
    climatology_metrics,
    continuous_representativeness,
    contour_level_for_fraction,
    daynight_overlap,
    daynight_overlap_index,
    evaluate_landcover,
    evaluate_vegetation_index,
    footprint_area,
    footprint_contour_mask,
    footprint_fetch,
    footprint_symmetry,
    footprint_weighted_composition,
    footprint_weighted_value,
    model2_regression,
    overlap,
    representativeness_summary,
    rma_regression,
    sample_raster_on_grid,
    seasonal_overlap,
    seasonal_overlap_index,
    sensor_location_bias,
    sensor_location_bias_series,
    symmetry_index,
    target_area_composition,
    target_area_mask,
    target_area_value,
    truncate_to_contour,
)

# ------------------------------
# Analytic constants and helpers
# ------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

#: The fraction the paper truncates at, and the Mahalanobis radius of its
#: isoline on a bivariate normal.
FRACTION = 0.8
MAHALANOBIS = math.sqrt(-2.0 * math.log(1.0 - FRACTION))

#: Gaussian fixture grid: 5 m cells out to 8 sigma, so essentially the whole
#: footprint sits on the domain and the enclosed mass reaches the fraction.
STEP = 5.0
EXTENT = 400.0
DIAGONAL = STEP * math.sqrt(2.0)

#: (sigma_x, sigma_y, x0) of the Gaussian climatologies the geometry tests use:
#: circular, elongated crosswind, elongated along wind, and displaced downwind.
GAUSSIAN_CASES = (
    (50.0, 50.0, 0.0),
    (80.0, 40.0, 0.0),
    (40.0, 80.0, 0.0),
    (50.0, 50.0, 100.0),
)

#: Small raster grid: five 10 m cells per axis, centred on the tower.
SMALL_AXIS = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])

#: Radii whose discs hold exactly 9 and 21 of the 25 cells of that grid.
SMALL_INNER_RADIUS = 15.0
SMALL_OUTER_RADIUS = 25.0
SMALL_INNER_CELLS = 9
SMALL_OUTER_CELLS = 21

#: Concentric-heterogeneity site: 10 m cells out to 1500 m from the tower.
SITE_AXIS = np.arange(-1495.0, 1500.0, 10.0)

#: Land-cover ring radii [m] and the class code each ring carries.
RING_EDGES = (400.0, 900.0, 1600.0)
RING_CLASSES = (11, 22, 33, 44)

#: Ordering of the three-level index, so "no better than" is expressible.
LEVEL_RANK: dict[Level, int] = {Level.LOW: 0, Level.MEDIUM: 1, Level.HIGH: 2}


def gaussian_footprint(
    sigma_x: float = 50.0,
    sigma_y: float = 50.0,
    x0: float = 0.0,
    extent: float = EXTENT,
    step: float = STEP,
) -> xr.DataArray:
    """Build a 2-D Gaussian footprint density [m-2], optionally offset in x."""
    axis = np.arange(-extent, extent + step / 2.0, step)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    density = np.exp(-0.5 * (((xx - x0) / sigma_x) ** 2 + (yy / sigma_y) ** 2)) / (
        2.0 * np.pi * sigma_x * sigma_y
    )
    return xr.DataArray(
        density,
        coords={"x": axis, "y": axis},
        dims=("x", "y"),
        name="fclim",
        attrs={"units": "m-2", "long_name": "footprint climatology"},
    )


def field_on(axis: np.ndarray, values: np.ndarray, name: str) -> xr.DataArray:
    """Wrap an ``(x, y)`` array as a raster on a tower-centred grid."""
    return xr.DataArray(
        np.asarray(values, dtype=float),
        coords={"x": axis, "y": axis},
        dims=("x", "y"),
        name=name,
    )


def small_field(values: Any, name: str = "field") -> xr.DataArray:
    """Wrap a 5x5 array as a raster on the small tower-centred grid."""
    return field_on(SMALL_AXIS, np.asarray(values, dtype=float), name)


def two_by_two(values: Any, name: str = "w") -> xr.DataArray:
    """Wrap four weights as a footprint on a 2x2 grid, for the overlap kernels."""
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(2, 2),
        coords={"x": np.array([0.0, 10.0]), "y": np.array([0.0, 10.0])},
        dims=("x", "y"),
        name=name,
    )


def stack_months(months: list[xr.DataArray]) -> xr.DataArray:
    """Stack monthly climatologies over the ``month`` dimension."""
    return xr.concat(months, dim="month")


def as_level(value: Any) -> Level:
    """Coerce a reported level -- a ``str`` in a DataFrame -- back to a member."""
    return Level(value)


# ------------------------------
# Shared fixtures
# ------------------------------


@pytest.fixture(scope="module")
def site_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Meshed offsets and radius of the concentric-heterogeneity site."""
    xx, yy = np.meshgrid(SITE_AXIS, SITE_AXIS, indexing="ij")
    return xx, yy, np.hypot(xx, yy)


@pytest.fixture(scope="module")
def site_footprint(
    site_mesh: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> xr.DataArray:
    """
    A truncated climatology confined well inside the innermost land-cover ring.

    A 90 m Gaussian displaced 60 m downwind, so its 80 % contour stops short of
    the 400 m ring edge. That is what makes the site representative at the
    smallest target radius and steadily less so beyond it.
    """
    xx, yy, _ = site_mesh
    density = np.exp(-0.5 * ((xx / 90.0) ** 2 + ((yy - 60.0) / 90.0) ** 2))
    # A density [m-2] on 10 m cells, as the footprint models produce.
    return truncate_to_contour(
        field_on(SITE_AXIS, density / (density.sum() * 100.0), "fclim")
    )


@pytest.fixture(scope="module")
def ring_landcover(
    site_mesh: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> xr.DataArray:
    """Concentric land-cover rings: one class per annulus around the tower."""
    _, _, radius = site_mesh
    codes = np.select(
        [radius <= edge for edge in RING_EDGES],
        [float(code) for code in RING_CLASSES[:-1]],
        default=float(RING_CLASSES[-1]),
    )
    return field_on(SITE_AXIS, codes, "land_cover")


@pytest.fixture(scope="module")
def scene_pairs(
    site_mesh: tuple[np.ndarray, np.ndarray, np.ndarray],
    site_footprint: xr.DataArray,
) -> dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]:
    """
    Ten monthly climatology / vegetation-index pairs over the concentric site.

    Each scene is a radially decaying core whose amplitude swings through the
    year, so the footprint and the inner discs track one another closely, plus
    a far field beyond 1200 m redrawn at random every month. The far field is
    uncorrelated with the core, so it degrades the regression only once the
    target area reaches out into it.
    """
    _, _, radius = site_mesh
    rng = np.random.default_rng(3)
    far = radius > 1200.0

    pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]] = {}
    for index, stamp in enumerate(pd.date_range("2020-01-15", periods=10, freq="MS")):
        core = 0.15 + 0.5 * np.exp(-0.5 * (radius / 600.0) ** 2) * (
            1.0 + 0.3 * np.sin(2.0 * np.pi * index / 10.0)
        )
        field = core + np.where(far, 0.6 * rng.random(), 0.0)
        pairs[stamp] = (site_footprint, field_on(SITE_AXIS, field, "EVI"))
    return pairs


@pytest.fixture(scope="module")
def smooth_scene_pairs(
    site_mesh: tuple[np.ndarray, np.ndarray, np.ndarray],
    site_footprint: xr.DataArray,
) -> dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]:
    """
    The same ten periods with the far field left out: a clean radial decay.

    :func:`scene_pairs` deliberately decorrelates its far field so that the
    site-level regression degrades all the way to LOW. That same far field
    makes the target-area mean rise again at the widest discs, which is not the
    behaviour the sensor location bias of Eq. 6 is being checked for. These
    scenes decay monotonically outward from the tower instead, so every disc a
    step wider can only dilute the mean further.
    """
    _, _, radius = site_mesh

    pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]] = {}
    for index, stamp in enumerate(pd.date_range("2020-01-15", periods=10, freq="MS")):
        field = 0.15 + 0.5 * np.exp(-0.5 * (radius / 600.0) ** 2) * (
            1.0 + 0.3 * np.sin(2.0 * np.pi * index / 10.0)
        )
        pairs[stamp] = (site_footprint, field_on(SITE_AXIS, field, "EVI"))
    return pairs


# ------------------------------
# 1. Geometry of the truncation contour (Sect. 2.2, Eq. 1)
# ------------------------------


class TestContourGeometry:
    """Fetch, area, symmetry, and contour level against the Gaussian isoline."""

    @pytest.mark.parametrize(("sigma_x", "sigma_y", "x0"), GAUSSIAN_CASES)
    @pytest.mark.parametrize("fraction", [0.5, 0.8, 0.9])
    def test_contour_level_is_the_analytic_isoline(
        self, sigma_x: float, sigma_y: float, x0: float, fraction: float
    ) -> None:
        """The level enclosing r of the mass is (1 - r) / (2 pi sx sy)."""
        fclim = gaussian_footprint(sigma_x, sigma_y, x0)
        expected = (1.0 - fraction) / (2.0 * np.pi * sigma_x * sigma_y)

        level = contour_level_for_fraction(fclim, fraction=fraction)

        assert level == pytest.approx(expected, rel=0.03)

    @pytest.mark.parametrize(("sigma_x", "sigma_y", "x0"), GAUSSIAN_CASES)
    def test_fetch_reaches_the_analytic_contour(
        self, sigma_x: float, sigma_y: float, x0: float
    ) -> None:
        """X80 is the far edge of the isoline, to within a cell diagonal."""
        mask = footprint_contour_mask(gaussian_footprint(sigma_x, sigma_y, x0))
        expected = (
            x0 + MAHALANOBIS * sigma_x if x0 else MAHALANOBIS * max(sigma_x, sigma_y)
        )

        fetch = footprint_fetch(mask)

        # Distances run to cell centres, so the fetch can only fall short of
        # the true contour, and by less than one cell diagonal.
        assert expected - DIAGONAL <= fetch <= expected + DIAGONAL

    @pytest.mark.parametrize(("sigma_x", "sigma_y", "x0"), GAUSSIAN_CASES)
    def test_area_matches_the_analytic_ellipse(
        self, sigma_x: float, sigma_y: float, x0: float
    ) -> None:
        """A80 is the area of the isoline ellipse, pi R^2 sx sy."""
        mask = footprint_contour_mask(gaussian_footprint(sigma_x, sigma_y, x0))
        expected = np.pi * MAHALANOBIS**2 * sigma_x * sigma_y

        assert footprint_area(mask) == pytest.approx(expected, rel=0.03)

    @pytest.mark.parametrize(("sigma_x", "sigma_y", "x0"), GAUSSIAN_CASES)
    def test_symmetry_matches_the_analytic_axis_ratio(
        self, sigma_x: float, sigma_y: float, x0: float
    ) -> None:
        """S80 is the axis ratio when centred, and shrinks when displaced."""
        mask = footprint_contour_mask(gaussian_footprint(sigma_x, sigma_y, x0))
        if x0:
            reach = MAHALANOBIS * sigma_x
            expected = (reach / (x0 + reach)) ** 2
        else:
            expected = min(sigma_x, sigma_y) / max(sigma_x, sigma_y)

        assert footprint_symmetry(mask) == pytest.approx(expected, rel=0.06)

    @pytest.mark.parametrize(("sigma_x", "sigma_y", "x0"), GAUSSIAN_CASES)
    def test_symmetry_is_area_over_the_bounding_disc(
        self, sigma_x: float, sigma_y: float, x0: float
    ) -> None:
        """Eq. 1 exactly: S80 = A80 / (pi X80^2), whatever the grid resolves."""
        mask = footprint_contour_mask(gaussian_footprint(sigma_x, sigma_y, x0))
        area = footprint_area(mask)
        fetch = footprint_fetch(mask)

        assert footprint_symmetry(mask) == pytest.approx(
            area / (np.pi * fetch**2), rel=1e-12
        )

    def test_displaced_footprint_is_flagged_asymmetric(self) -> None:
        """The 100 m-displaced case is the paper's asymmetric climatology."""
        mask = footprint_contour_mask(gaussian_footprint(50.0, 50.0, 100.0))

        assert footprint_symmetry(mask) < ASYMMETRY_THRESHOLD

    def test_centred_circular_footprint_is_not_asymmetric(self) -> None:
        """A circular climatology on the tower sits at the top of the range."""
        mask = footprint_contour_mask(gaussian_footprint(50.0, 50.0))

        assert footprint_symmetry(mask) > 0.95

    @pytest.mark.parametrize(
        ("area", "fetch", "expected"),
        [
            (np.pi * 100.0**2, 100.0, 1.0),
            (0.25 * np.pi * 100.0**2, 100.0, 0.25),
            (0.0, 100.0, 0.0),
            (10.0 * np.pi * 100.0**2, 100.0, 1.0),  # clipped, never above one
        ],
    )
    def test_symmetry_index_closed_form(
        self, area: float, fetch: float, expected: float
    ) -> None:
        """The bare index is A / (pi X^2), clipped into [0, 1]."""
        assert symmetry_index(area, fetch) == pytest.approx(expected)

    @pytest.mark.parametrize(("area", "fetch"), [(1.0, 0.0), (float("nan"), 10.0)])
    def test_symmetry_index_undefined(self, area: float, fetch: float) -> None:
        """A zero fetch or a non-finite area leaves the index undefined."""
        assert math.isnan(symmetry_index(area, fetch))

    def test_climatology_metrics_reproduces_its_parts(self) -> None:
        """The summary agrees with the individual routines and the Gaussian."""
        fclim = gaussian_footprint(80.0, 40.0)
        mask = footprint_contour_mask(fclim)

        metrics = climatology_metrics(fclim)

        assert isinstance(metrics, ClimatologyMetrics)
        assert metrics.fraction == FRACTION
        assert metrics.fetch == pytest.approx(footprint_fetch(mask))
        assert metrics.area == pytest.approx(footprint_area(mask))
        assert metrics.symmetry == pytest.approx(footprint_symmetry(mask))
        assert metrics.n_cells == int(mask.sum())
        assert metrics.contour_level == pytest.approx(contour_level_for_fraction(fclim))
        # The 8-sigma domain holds essentially the whole footprint, so the
        # contour encloses the fraction it was cut at.
        assert metrics.enclosed_fraction == pytest.approx(FRACTION, abs=0.01)
        assert metrics.seasonal_overlap is None
        assert metrics.daynight_overlap is None

    def test_metrics_carry_supplied_overlap_indices(self) -> None:
        """Both overlaps are site-year properties, so they are passed in."""
        metrics = climatology_metrics(
            gaussian_footprint(), seasonal_overlap=0.83, daynight_overlap=0.91
        )

        assert metrics.seasonal_overlap == pytest.approx(0.83)
        assert metrics.daynight_overlap == pytest.approx(0.91)

    def test_metrics_of_a_truncated_climatology_read_back_its_fraction(self) -> None:
        """A truncated input is summarised on the contour it was already cut at."""
        truncated = truncate_to_contour(gaussian_footprint(60.0, 60.0), fraction=0.5)

        metrics = climatology_metrics(truncated, fraction=FRACTION)

        assert metrics.fraction == 0.5
        assert math.isnan(metrics.enclosed_fraction)
        assert metrics.area == pytest.approx(
            np.pi * (-2.0 * math.log(0.5)) * 60.0 * 60.0, rel=0.03
        )

    def test_truncation_renormalises_the_retained_cells(self) -> None:
        """Retained weights sum to one; everything outside comes back zero."""
        fclim = gaussian_footprint(50.0, 50.0)
        mask = footprint_contour_mask(fclim)

        weights = truncate_to_contour(fclim)

        assert float(weights.sum()) == pytest.approx(1.0)
        assert bool(((weights > 0.0) == mask).all())
        assert weights.attrs["contour_fraction"] == FRACTION
        assert weights.attrs["units"] == "1"

    def test_truncation_can_keep_the_original_densities(self) -> None:
        """Without renormalisation the surviving cells keep their [m-2] values."""
        fclim = gaussian_footprint(50.0, 50.0)
        mask = footprint_contour_mask(fclim)

        weights = truncate_to_contour(fclim, renormalize=False)

        assert float(weights.sum()) == pytest.approx(
            float(fclim.where(mask, 0.0).sum())
        )

    @pytest.mark.parametrize("radius", [50.0, 100.0, 200.0, 380.0])
    def test_target_area_mask_counts_the_disc(self, radius: float) -> None:
        """Cell centres inside the disc approach pi r^2 / (dx dy)."""
        axis = np.arange(-EXTENT, EXTENT + STEP / 2.0, STEP)

        mask = target_area_mask(axis, axis, radius)

        assert int(mask.sum()) == pytest.approx(np.pi * radius**2 / STEP**2, rel=0.02)
        assert mask.attrs["radius"] == radius

    @pytest.mark.parametrize(
        ("radius", "expected"),
        [
            (SMALL_INNER_RADIUS, SMALL_INNER_CELLS),
            (SMALL_OUTER_RADIUS, SMALL_OUTER_CELLS),
        ],
    )
    def test_small_grid_discs_hold_the_cells_the_hand_values_assume(
        self, radius: float, expected: int
    ) -> None:
        """The 15 m and 25 m discs of the 5x5 grid hold 9 and 21 cells."""
        mask = target_area_mask(SMALL_AXIS, SMALL_AXIS, radius)

        assert int(mask.sum()) == expected

    def test_target_area_mask_rejects_an_unusable_radius(self) -> None:
        with pytest.raises(ValueError, match="positive and finite"):
            target_area_mask(SMALL_AXIS, SMALL_AXIS, 0.0)


# ------------------------------
# 2. Overlap indices (Sect. 2.2, Eqs. 2-3)
# ------------------------------


class TestOverlapIndices:
    """
    The overlap kernels against values worked out by hand.

    Every fixture here is a two- or four-cell footprint, so the sums of square
    roots close in exact form and the assertions do not depend on the module
    to say what the answer is.
    """

    def test_overlap_of_identical_footprints_is_one(self) -> None:
        weights = two_by_two([0.4, 0.3, 0.2, 0.1])

        assert overlap(weights, weights) == pytest.approx(1.0)

    def test_overlap_of_disjoint_supports_is_zero(self) -> None:
        assert overlap(
            two_by_two([0.5, 0.5, 0.0, 0.0]), two_by_two([0.0, 0.0, 0.5, 0.5])
        ) == pytest.approx(0.0)

    def test_overlap_of_half_shared_support(self) -> None:
        """Two halves sharing one of two cells: sqrt(0.5 * 0.5) = 0.5."""
        assert overlap(
            two_by_two([0.5, 0.5, 0.0, 0.0]), two_by_two([0.0, 0.5, 0.5, 0.0])
        ) == pytest.approx(0.5)

    def test_overlap_of_a_quarter_three_quarter_pair(self) -> None:
        """2 * sqrt(1/4 * 3/4) = sqrt(3) / 2."""
        assert overlap(
            two_by_two([0.25, 0.75, 0.0, 0.0]), two_by_two([0.75, 0.25, 0.0, 0.0])
        ) == pytest.approx(math.sqrt(3.0) / 2.0)

    def test_overlap_refuses_a_mismatched_grid(self) -> None:
        other = xr.DataArray(
            np.full((2, 2), 0.25),
            coords={"x": np.array([0.0, 20.0]), "y": np.array([0.0, 10.0])},
            dims=("x", "y"),
        )

        with pytest.raises(ValueError):
            overlap(two_by_two([0.25, 0.25, 0.25, 0.25]), other)

    def test_seasonal_overlap_of_three_rotated_months(self) -> None:
        """
        Three months holding (1/2, 1/4, 1/4) in rotation over three shared cells.

        Every shared cell has the same geometric mean, (1/2 * 1/4 * 1/4)^(1/3)
        = 2^(-5/3), and the fourth cell is empty in all three months, so
        O80_season = 3 * 2^(-5/3).
        """
        months = stack_months(
            [
                two_by_two([0.5, 0.25, 0.25, 0.0]),
                two_by_two([0.25, 0.5, 0.25, 0.0]),
                two_by_two([0.25, 0.25, 0.5, 0.0]),
            ]
        )

        assert seasonal_overlap(months) == pytest.approx(3.0 * 2.0 ** (-5.0 / 3.0))

    def test_seasonal_overlap_of_identical_months_is_one(self) -> None:
        month = two_by_two([0.4, 0.3, 0.2, 0.1])

        assert seasonal_overlap(stack_months([month, month, month])) == pytest.approx(
            1.0
        )

    def test_seasonal_overlap_reduces_to_the_kernel_at_two_months(self) -> None:
        """Eq. 2 with K = 2 is exactly the pairwise overlap of Eq. 3's kernel."""
        first = two_by_two([0.5, 0.5, 0.0, 0.0])
        second = two_by_two([0.0, 0.5, 0.5, 0.0])

        assert seasonal_overlap(stack_months([first, second])) == pytest.approx(
            overlap(first, second)
        )

    def test_one_month_pointing_elsewhere_zeroes_the_season(self) -> None:
        """The geometric mean is zero wherever any single month is."""
        months = stack_months(
            [
                two_by_two([0.5, 0.5, 0.0, 0.0]),
                two_by_two([0.5, 0.5, 0.0, 0.0]),
                two_by_two([0.0, 0.0, 0.5, 0.5]),
            ]
        )

        assert seasonal_overlap(months) == pytest.approx(0.0)

    def test_seasonal_overlap_needs_two_months(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            seasonal_overlap(stack_months([two_by_two([0.25, 0.25, 0.25, 0.25])]))

    def test_seasonal_overlap_refuses_unnormalised_months(self) -> None:
        months = stack_months(
            [two_by_two([0.5, 0.5, 0.0, 0.0]), two_by_two([0.5, 0.5, 0.5, 0.0])]
        )

        with pytest.raises(ValueError):
            seasonal_overlap(months)

    def test_daynight_overlap_averages_over_months(self) -> None:
        """One disjoint month and one identical month average to 1/2."""
        day = stack_months(
            [two_by_two([1.0, 0.0, 0.0, 0.0]), two_by_two([0.5, 0.5, 0.0, 0.0])]
        )
        night = stack_months(
            [two_by_two([0.0, 1.0, 0.0, 0.0]), two_by_two([0.5, 0.5, 0.0, 0.0])]
        )

        assert daynight_overlap(day, night) == pytest.approx(0.5)

    def test_daynight_overlap_of_a_single_month_is_the_kernel(self) -> None:
        """A month pairing (1/4, 3/4) against (3/4, 1/4) gives sqrt(3) / 2."""
        day = stack_months([two_by_two([0.25, 0.75, 0.0, 0.0])])
        night = stack_months([two_by_two([0.75, 0.25, 0.0, 0.0])])

        assert daynight_overlap(day, night) == pytest.approx(math.sqrt(3.0) / 2.0)

    def test_daynight_overlap_needs_matching_month_counts(self) -> None:
        month = two_by_two([0.5, 0.5, 0.0, 0.0])

        with pytest.raises(ValueError, match="different numbers of months"):
            daynight_overlap(stack_months([month, month]), stack_months([month]))

    def test_sequence_adapters_agree_with_the_stacked_form(self) -> None:
        """The ``*_index`` adapters are the same indices from sequences."""
        months = [
            two_by_two([0.5, 0.25, 0.25, 0.0]),
            two_by_two([0.25, 0.5, 0.25, 0.0]),
        ]
        nights = [
            two_by_two([0.25, 0.5, 0.25, 0.0]),
            two_by_two([0.5, 0.25, 0.25, 0.0]),
        ]

        assert seasonal_overlap_index(months) == pytest.approx(
            seasonal_overlap(stack_months(months))
        )
        assert daynight_overlap_index(months, nights) == pytest.approx(
            daynight_overlap(stack_months(months), stack_months(nights))
        )

    def test_overlap_indices_agree_on_gaussian_climatologies(self) -> None:
        """
        The hand values carry over to real climatologies.

        Two Gaussians offset by a quarter sigma overlap almost completely; one
        offset by four sigma barely at all. Both stay inside [0, 1].
        """
        near = [
            truncate_to_contour(gaussian_footprint(50.0, 50.0, offset))
            for offset in (0.0, 12.5)
        ]
        far = [
            truncate_to_contour(gaussian_footprint(50.0, 50.0, offset))
            for offset in (-200.0, 200.0)
        ]

        assert 0.8 < seasonal_overlap_index(near) <= 1.0
        assert 0.0 <= seasonal_overlap_index(far) < 0.05
        assert seasonal_overlap_index(near) > seasonal_overlap_index(far)


# ------------------------------
# 3. Weighted statistics on synthetic rasters (Sect. 2.4, Eq. 5)
# ------------------------------


class TestWeightedSampling:
    """
    Footprint-weighted and target-area statistics on a 5x5 synthetic raster.

    Weights sit on three named cells and the discs hold 9 and 21 of the 25
    cells, so every expected value is a ratio of small integers.
    """

    @staticmethod
    def three_cell_weights() -> xr.DataArray:
        """0.5 on the tower cell, 0.3 one cell east, 0.2 one cell west."""
        weights = np.zeros((5, 5))
        weights[2, 2] = 0.5  # (x, y) = (0, 0)
        weights[3, 2] = 0.3  # (x, y) = (10, 0)
        weights[1, 2] = 0.2  # (x, y) = (-10, 0)
        return small_field(weights, "w")

    @staticmethod
    def three_cell_raster(west: float = 3.0) -> xr.DataArray:
        """1, 2, and `west` under the three weighted cells; 5 everywhere else."""
        values = np.full((5, 5), 5.0)
        values[2, 2] = 1.0
        values[3, 2] = 2.0
        values[1, 2] = west
        return small_field(values, "EVI")

    def test_footprint_weighted_value_is_the_weighted_sum(self) -> None:
        """Eq. 5 by hand: 0.5 * 1 + 0.3 * 2 + 0.2 * 3 = 1.7."""
        result = footprint_weighted_value(
            self.three_cell_weights(), self.three_cell_raster()
        )

        assert result.value == pytest.approx(1.7)
        assert result.retained_weight == pytest.approx(1.0)
        assert result.n_cells == 3

    def test_footprint_weighted_value_renormalises_over_nodata(self) -> None:
        """Dropping the 0.2 cell leaves (0.5 + 0.6) / 0.8 = 1.375 on 0.8 weight."""
        result = footprint_weighted_value(
            self.three_cell_weights(), self.three_cell_raster(west=float("nan"))
        )

        assert result.value == pytest.approx(1.375)
        assert result.retained_weight == pytest.approx(0.8)
        assert result.n_cells == 2

    def test_footprint_weighted_value_of_an_empty_raster(self) -> None:
        """No cell carries both weight and data, so there is nothing to report."""
        result = footprint_weighted_value(
            self.three_cell_weights(), small_field(np.full((5, 5), np.nan))
        )

        assert math.isnan(result.value)
        assert result.retained_weight == 0.0
        assert result.n_cells == 0

    def test_footprint_weighted_value_accepts_unnormalised_weights(self) -> None:
        """Weights summing to something else are rescaled, not rejected."""
        weights = self.three_cell_weights() * 7.0

        assert footprint_weighted_value(
            weights, self.three_cell_raster()
        ).value == pytest.approx(1.7)

    def test_footprint_weighted_value_refuses_a_mismatched_grid(self) -> None:
        other = field_on(SMALL_AXIS * 2.0, np.zeros((5, 5)), "EVI")

        with pytest.raises(ValueError):
            footprint_weighted_value(self.three_cell_weights(), other)

    def test_footprint_weighted_composition_splits_the_weight(self) -> None:
        """0.5 + 0.3 of the weight on class 41, 0.2 on class 81."""
        codes = np.full((5, 5), 90.0)
        codes[2, 2] = 41.0
        codes[3, 2] = 41.0
        codes[1, 2] = 81.0

        composition = footprint_weighted_composition(
            self.three_cell_weights(), small_field(codes, "land_cover")
        )

        assert list(composition.index) == [41, 81]
        assert composition.loc[41] == pytest.approx(0.8)
        assert composition.loc[81] == pytest.approx(0.2)
        assert float(composition.sum()) == pytest.approx(1.0)
        assert composition.attrs["retained_weight"] == pytest.approx(1.0)
        assert composition.attrs["n_cells"] == 3

    def test_target_area_value_averages_the_inner_disc(self) -> None:
        """The 15 m disc is exactly the nine core cells, all holding 1.0."""
        values = np.zeros((5, 5))
        values[1:4, 1:4] = 1.0

        result = target_area_value(
            small_field(values), SMALL_AXIS, SMALL_AXIS, SMALL_INNER_RADIUS
        )

        assert result.value == pytest.approx(1.0)
        assert result.retained_weight == pytest.approx(1.0)
        assert result.n_cells == SMALL_INNER_CELLS

    def test_target_area_value_dilutes_over_the_outer_disc(self) -> None:
        """The 25 m disc holds 21 cells, nine of them the core: 9 / 21."""
        values = np.zeros((5, 5))
        values[1:4, 1:4] = 1.0

        result = target_area_value(
            small_field(values), SMALL_AXIS, SMALL_AXIS, SMALL_OUTER_RADIUS
        )

        assert result.value == pytest.approx(SMALL_INNER_CELLS / SMALL_OUTER_CELLS)
        assert result.n_cells == SMALL_OUTER_CELLS

    def test_target_area_value_reports_partial_coverage(self) -> None:
        """Nodata over the outer ring lowers the retained fraction, not the mean."""
        values = np.full((5, 5), np.nan)
        values[1:4, 1:4] = 1.0

        result = target_area_value(
            small_field(values), SMALL_AXIS, SMALL_AXIS, SMALL_OUTER_RADIUS
        )

        assert result.value == pytest.approx(1.0)
        assert result.retained_weight == pytest.approx(
            SMALL_INNER_CELLS / SMALL_OUTER_CELLS
        )
        assert result.n_cells == SMALL_INNER_CELLS

    def test_target_area_composition_counts_cells(self) -> None:
        """Nine core cells of one class against twelve of another, out of 21."""
        codes = np.full((5, 5), 81.0)
        codes[1:4, 1:4] = 41.0

        composition = target_area_composition(
            small_field(codes, "land_cover"),
            SMALL_AXIS,
            SMALL_AXIS,
            SMALL_OUTER_RADIUS,
        )

        assert composition.loc[41] == pytest.approx(
            SMALL_INNER_CELLS / SMALL_OUTER_CELLS
        )
        assert composition.loc[81] == pytest.approx(
            (SMALL_OUTER_CELLS - SMALL_INNER_CELLS) / SMALL_OUTER_CELLS
        )
        assert composition.attrs["n_cells"] == SMALL_OUTER_CELLS

    def test_target_area_composition_of_a_homogeneous_disc(self) -> None:
        codes = np.full((5, 5), 81.0)
        codes[1:4, 1:4] = 41.0

        composition = target_area_composition(
            small_field(codes, "land_cover"),
            SMALL_AXIS,
            SMALL_AXIS,
            SMALL_INNER_RADIUS,
        )

        assert list(composition.index) == [41]
        assert composition.loc[41] == pytest.approx(1.0)

    def test_target_area_value_needs_a_disc_holding_a_cell(self) -> None:
        """A disc smaller than the grid can miss every cell centre."""
        # Offset axes, so no cell sits on the tower: the nearest centre is
        # sqrt(50) m away and a 5 m disc catches none of them.
        offset = SMALL_AXIS + 5.0
        raster = field_on(offset, np.zeros((5, 5)), "EVI")

        with pytest.raises(ValueError, match="No cell centre"):
            target_area_value(raster, offset, offset, 5.0)


class TestRasterSampling:
    """Bringing an external raster onto the tower-centred grid."""

    def test_sample_raster_on_grid_returns_the_footprint_grid(
        self, tmp_path: Path
    ) -> None:
        """
        A GeoTIFF written on the grid's own georeferencing comes back unchanged.

        The raster is built from :func:`footprint_grid_geometry`, so the warp
        inside :func:`sample_raster_on_grid` is the identity and the values can
        be checked cell for cell. Its values depend only on easting, which
        keeps the check independent of the north-up row order of the file.
        """
        pytest.importorskip("rioxarray")
        rasterio = pytest.importorskip("rasterio")
        from fluxfootprints.openet_masking import footprint_grid_geometry

        axis = np.arange(-100.0, 101.0, 10.0)
        geometry = footprint_grid_geometry(axis, axis, 41.6285, -83.3471)
        # Row-invariant ramp: value = the cell's x offset from the tower.
        values = np.tile(axis, (axis.size, 1)).astype("float32")

        path = tmp_path / "ramp.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=geometry.height,
            width=geometry.width,
            count=1,
            dtype="float32",
            crs=geometry.crs,
            transform=geometry.transform,
        ) as dst:
            dst.write(values, 1)

        sampled = sample_raster_on_grid(path, axis, axis, 41.6285, -83.3471)

        assert sampled.dims == ("x", "y")
        assert sampled.shape == (axis.size, axis.size)
        np.testing.assert_allclose(sampled.coords["x"].values, axis)
        np.testing.assert_allclose(sampled.coords["y"].values, axis)
        assert np.isfinite(sampled.values).all()
        # Every column holds its own x offset, whatever the row.
        np.testing.assert_allclose(
            sampled.values, np.tile(axis[:, None], (1, axis.size)), atol=0.5
        )

    def test_sample_raster_on_grid_preserves_class_codes(self, tmp_path: Path) -> None:
        """Nearest-neighbour resampling keeps a categorical raster categorical."""
        pytest.importorskip("rioxarray")
        rasterio = pytest.importorskip("rasterio")
        from fluxfootprints.openet_masking import footprint_grid_geometry

        axis = np.arange(-100.0, 101.0, 10.0)
        geometry = footprint_grid_geometry(axis, axis, 41.6285, -83.3471)
        codes = np.where(axis < 0.0, 41, 81).astype("int16")
        values = np.tile(codes, (axis.size, 1))

        path = tmp_path / "classes.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=geometry.height,
            width=geometry.width,
            count=1,
            dtype="int16",
            crs=geometry.crs,
            transform=geometry.transform,
        ) as dst:
            dst.write(values, 1)

        sampled = sample_raster_on_grid(
            path, axis, axis, 41.6285, -83.3471, categorical=True
        )

        assert set(np.unique(sampled.values)) <= {41.0, 81.0}
        # The split follows the sign of the x offset it was written on.
        np.testing.assert_allclose(
            sampled.values,
            np.tile(np.where(axis < 0.0, 41.0, 81.0)[:, None], (1, axis.size)),
        )


# ------------------------------
# 4. Sensor location bias and the model II regression (Eqs. 6-7)
# ------------------------------


def core_weights() -> xr.DataArray:
    """Uniform weight over the nine cells of the 5x5 grid's core."""
    weights = np.zeros((5, 5))
    weights[1:4, 1:4] = 1.0 / SMALL_INNER_CELLS
    return small_field(weights, "w")


def core_raster() -> xr.DataArray:
    """One over the core, zero outside it."""
    values = np.zeros((5, 5))
    values[1:4, 1:4] = 1.0
    return small_field(values, "EVI")


class TestSensorLocationBias:
    """Eq. 6 on the 5x5 raster, where both averages are exact."""

    def test_bias_is_zero_when_the_disc_matches_the_footprint(self) -> None:
        """Footprint and 15 m disc are the same nine cells, so Delta = 0."""
        frame = sensor_location_bias(
            core_weights(),
            core_raster(),
            SMALL_AXIS,
            SMALL_AXIS,
            radii=[SMALL_INNER_RADIUS],
        )

        row = frame.iloc[0]
        assert row["value_footprint"] == pytest.approx(1.0)
        assert row["value_target"] == pytest.approx(1.0)
        assert row["delta"] == pytest.approx(0.0)
        assert bool(row["within_threshold"]) is True

    def test_bias_over_the_outer_disc_is_four_thirds(self) -> None:
        """(1 - 9/21) / (9/21) = 4/3, the paper's positive-bias case."""
        frame = sensor_location_bias(
            core_weights(),
            core_raster(),
            SMALL_AXIS,
            SMALL_AXIS,
            radii=[SMALL_OUTER_RADIUS],
        )

        row = frame.iloc[0]
        assert row["value_target"] == pytest.approx(
            SMALL_INNER_CELLS / SMALL_OUTER_CELLS
        )
        assert row["delta"] == pytest.approx(4.0 / 3.0)
        assert bool(row["within_threshold"]) is False

    def test_footprint_value_is_constant_down_the_frame(self) -> None:
        """Eq. 5 does not depend on the radius, so only the target moves."""
        frame = sensor_location_bias(
            core_weights(),
            core_raster(),
            SMALL_AXIS,
            SMALL_AXIS,
            radii=[SMALL_INNER_RADIUS, SMALL_OUTER_RADIUS],
        )

        assert frame["value_footprint"].nunique() == 1
        assert list(frame["radius"]) == [SMALL_INNER_RADIUS, SMALL_OUTER_RADIUS]
        assert frame.attrs["bias_threshold"] == BIAS_THRESHOLD

    @pytest.mark.parametrize(
        ("delta", "expected"),
        [
            (0.05, True),
            (0.99 * BIAS_THRESHOLD, True),
            (1.01 * BIAS_THRESHOLD, False),
            (0.2, False),
        ],
    )
    def test_threshold_separates_representative_periods(
        self, delta: float, expected: bool
    ) -> None:
        """
        The +/-10 % threshold of Sect. 2.4 splits the periods either side of it.

        The two inner cases sit 1 % either side of the threshold rather than on
        it: a bias built to land exactly on 0.10 arrives as 0.10000000000000003
        after the division of Eq. 6, so an exact-boundary assertion would be a
        test of floating-point luck rather than of the comparison.
        """
        # The footprint is uniform over the nine core cells, so its value is
        # whatever the core holds. Give the twelve remaining cells of the 25 m
        # disc the value that pulls the disc mean down to core / (1 + delta):
        #   (9 * core + 12 * ring) / 21 = core / (1 + delta).
        core = 1.0
        ring_value = (
            SMALL_OUTER_CELLS * core / (1.0 + delta) - SMALL_INNER_CELLS * core
        ) / (SMALL_OUTER_CELLS - SMALL_INNER_CELLS)

        disc = np.asarray(
            target_area_mask(SMALL_AXIS, SMALL_AXIS, SMALL_OUTER_RADIUS).values
        )
        ring = disc.copy()
        ring[1:4, 1:4] = False

        values = np.zeros((5, 5))
        values[1:4, 1:4] = core
        values[ring] = ring_value

        frame = sensor_location_bias(
            core_weights(),
            small_field(values),
            SMALL_AXIS,
            SMALL_AXIS,
            radii=[SMALL_OUTER_RADIUS],
        )

        assert frame.iloc[0]["delta"] == pytest.approx(delta)
        assert bool(frame.iloc[0]["within_threshold"]) is expected

    def test_series_stacks_the_periods_it_is_given(self) -> None:
        """One block per period and radius, under a leading ``time`` column."""
        pairs = {
            pd.Timestamp("2020-06-15"): (core_weights(), core_raster()),
            pd.Timestamp("2020-07-15"): (core_weights(), core_raster() * 2.0),
        }
        radii = [SMALL_INNER_RADIUS, SMALL_OUTER_RADIUS]

        series = sensor_location_bias_series(pairs, SMALL_AXIS, SMALL_AXIS, radii=radii)

        assert list(series.columns) == [
            "time",
            "radius",
            "value_footprint",
            "value_target",
            "delta",
            "within_threshold",
        ]
        assert len(series) == len(pairs) * len(radii)
        # Scaling the whole scene leaves the relative bias untouched.
        assert series.groupby("radius")["delta"].nunique().eq(1).all()


class TestModelTwoRegression:
    """The reduced major axis of Eq. 7 against closed-form values."""

    def test_slope_is_the_ratio_of_standard_deviations(self) -> None:
        """RMA slope = sign(r) * sd(y) / sd(x), exactly, for any correlated pair."""
        rng = np.random.default_rng(11)
        x = rng.normal(0.5, 0.1, 40)
        y = 0.7 * x + 0.05 + rng.normal(0.0, 0.01, 40)

        fit = rma_regression(x, y)

        assert fit.slope == pytest.approx(np.std(y, ddof=1) / np.std(x, ddof=1))
        assert fit.intercept == pytest.approx(y.mean() - fit.slope * x.mean())
        assert fit.n == 40

    def test_a_perfect_line_is_recovered_exactly(self) -> None:
        """On noiseless data the RMA line is the generating line."""
        x = np.linspace(0.1, 0.9, 12)
        y = 0.85 * x + 0.04

        fit = rma_regression(x, y)

        assert fit.slope == pytest.approx(0.85)
        assert fit.intercept == pytest.approx(0.04)
        assert fit.r_squared == pytest.approx(1.0)
        assert fit.ci_method == "analytical"
        assert fit.ci_level == 0.95

    def test_the_one_to_one_line_gives_a_unit_slope(self) -> None:
        x = np.linspace(0.2, 0.8, 15)

        fit = rma_regression(x, x.copy())

        assert fit.slope == pytest.approx(1.0)
        assert fit.intercept == pytest.approx(0.0, abs=1e-12)

    def test_rma_slope_exceeds_the_least_squares_slope(self) -> None:
        """The RMA slope is the OLS slope divided by r, so never shallower."""
        rng = np.random.default_rng(5)
        x = rng.normal(0.5, 0.1, 60)
        y = 0.9 * x + rng.normal(0.0, 0.05, 60)

        fit = rma_regression(x, y)
        ols = np.polyfit(x, y, 1)[0]

        assert fit.slope > ols
        assert fit.slope == pytest.approx(ols / math.sqrt(fit.r_squared))

    def test_confidence_limits_bracket_the_estimates(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(0.5, 0.1, 30)
        y = x + rng.normal(0.0, 0.02, 30)

        fit = rma_regression(x, y)

        assert fit.slope_ci[0] < fit.slope < fit.slope_ci[1]
        assert fit.intercept_ci[0] < fit.intercept < fit.intercept_ci[1]

    def test_bootstrap_limits_are_reproducible(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.normal(0.5, 0.1, 40)
        y = x + rng.normal(0.0, 0.03, 40)

        first = rma_regression(x, y, ci_method="bootstrap", n_boot=200, random_state=1)
        second = rma_regression(x, y, ci_method="bootstrap", n_boot=200, random_state=1)

        assert first.slope_ci == pytest.approx(second.slope_ci)
        assert first.ci_method == "bootstrap"

    def test_model2_regression_is_the_four_value_form(self) -> None:
        x = np.linspace(0.1, 0.9, 12)
        y = 0.85 * x + 0.04

        intercept, slope, r_squared, p_value = model2_regression(x, y)
        fit = rma_regression(x, y)

        assert (intercept, slope, r_squared) == pytest.approx(
            (fit.intercept, fit.slope, fit.r_squared)
        )
        assert p_value == pytest.approx(fit.p_value)

    def test_regression_needs_three_finite_pairs(self) -> None:
        with pytest.raises(ValueError):
            rma_regression([0.1, 0.2], [0.1, 0.2])


# ------------------------------
# 5. The three-level indices (Sect. 2.4)
# ------------------------------


class TestClassification:
    """Every branch of the two classifiers, at and around its thresholds."""

    @pytest.mark.parametrize(
        ("p_footprint", "p_target", "p_value", "expected"),
        [
            (92.0, 88.0, 0.42, Level.HIGH),
            (80.0, 80.0, 0.05, Level.HIGH),  # thresholds are inclusive
            (92.0, 61.0, 0.31, Level.MEDIUM),
            (50.0, 50.0, 0.90, Level.MEDIUM),
            (79.9, 88.0, 0.42, Level.MEDIUM),
            (92.0, 49.9, 0.42, Level.LOW),  # target below half
            (49.9, 92.0, 0.42, Level.LOW),  # footprint below half
            (92.0, 88.0, 0.04, Level.LOW),  # compositions differ
            (float("nan"), 88.0, 0.42, Level.LOW),  # undefined share
            (92.0, 88.0, float("nan"), Level.LOW),  # undefined test
        ],
    )
    def test_classify_categorical(
        self, p_footprint: float, p_target: float, p_value: float, expected: Level
    ) -> None:
        assert classify_categorical(p_footprint, p_target, p_value) == expected

    def test_classify_categorical_honours_alpha(self) -> None:
        assert classify_categorical(92.0, 88.0, 0.02) is Level.LOW
        assert classify_categorical(92.0, 88.0, 0.02, alpha=0.01) is Level.HIGH

    @pytest.mark.parametrize(
        ("r_squared", "slope", "intercept", "p_value", "expected"),
        [
            (0.94, 0.96, 0.02, 1e-9, Level.HIGH),
            (0.80, 0.90, -0.10, 1e-9, Level.HIGH),  # every bound inclusive
            (0.94, 0.85, 0.02, 1e-9, Level.MEDIUM),  # slope out of tolerance
            (0.94, 0.96, 0.15, 1e-9, Level.MEDIUM),  # intercept out of tolerance
            (0.71, 0.80, 0.06, 1e-9, Level.MEDIUM),
            (0.60, 0.80, 0.06, 0.049, Level.MEDIUM),
            (0.59, 0.96, 0.02, 1e-9, Level.LOW),  # too little variance explained
            (0.94, 0.85, 0.02, 0.20, Level.LOW),  # not significant
            (float("nan"), float("nan"), float("nan"), float("nan"), Level.LOW),
        ],
    )
    def test_classify_continuous(
        self,
        r_squared: float,
        slope: float,
        intercept: float,
        p_value: float,
        expected: Level,
    ) -> None:
        assert classify_continuous(r_squared, slope, intercept, p_value) == expected

    def test_levels_compare_as_strings(self) -> None:
        """``Level`` is a ``str`` enum, so it survives a DataFrame round trip."""
        assert Level.HIGH == "high"
        assert as_level("medium") is Level.MEDIUM
        assert sorted(LEVEL_RANK, key=LEVEL_RANK.get) == [
            Level.LOW,
            Level.MEDIUM,
            Level.HIGH,
        ]


# ------------------------------
# 6. Property: every reported index stays in [0, 1]
# ------------------------------


def random_climatology(rng: np.random.Generator) -> xr.DataArray:
    """A Gaussian footprint with random widths, orientation, and displacement."""
    sigma_x = float(rng.uniform(20.0, 120.0))
    sigma_y = float(rng.uniform(20.0, 120.0))
    x0 = float(rng.uniform(-150.0, 150.0))
    return gaussian_footprint(sigma_x, sigma_y, x0)


class TestIndicesStayInUnitInterval:
    """
    Every index the module reports is a fraction, and must read as one.

    The paper's figures put fetches on a log axis and areas in square metres,
    but each of its *indices* -- symmetry (Eq. 1), the two overlaps (Eqs. 2-3),
    the retained weights, the class shares, and the regression's R-squared and
    p-value -- is bounded by 0 and 1. These tests hold that bound over
    randomised footprints rather than over one hand-picked case.
    """

    SEEDS = tuple(range(12))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_symmetry_and_enclosed_mass_are_fractions(self, seed: int) -> None:
        fclim = random_climatology(np.random.default_rng(seed))

        metrics = climatology_metrics(fclim)

        assert 0.0 <= metrics.symmetry <= 1.0
        assert 0.0 <= metrics.enclosed_fraction <= 1.0
        assert metrics.area > 0.0
        assert metrics.fetch > 0.0
        assert metrics.n_cells > 0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_overlap_indices_are_fractions(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        days = [truncate_to_contour(random_climatology(rng)) for _ in range(4)]
        nights = [truncate_to_contour(random_climatology(rng)) for _ in range(4)]

        seasonal = seasonal_overlap_index(days)
        pairwise = daynight_overlap_index(days, nights)

        assert 0.0 <= seasonal <= 1.0
        assert 0.0 <= pairwise <= 1.0
        # The kernel of Eq. 3 is bounded by 1 for every individual month too.
        for day, night in zip(days, nights, strict=True):
            assert 0.0 <= overlap(day, night) <= 1.0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_weighted_statistics_report_fractions(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        weights = truncate_to_contour(random_climatology(rng))
        axis = weights.coords["x"].values
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        radius = np.hypot(xx, yy)

        # A field with holes in it, so the retained fractions have work to do.
        field = np.where(rng.random(radius.shape) < 0.1, np.nan, radius / 1000.0)
        codes = np.where(radius <= 150.0, 41.0, 81.0)

        value = footprint_weighted_value(weights, field_on(axis, field, "EVI"))
        composition = footprint_weighted_composition(
            weights, field_on(axis, codes, "land_cover")
        )
        disc = target_area_value(field_on(axis, field, "EVI"), axis, axis, 250.0)
        target = target_area_composition(
            field_on(axis, codes, "land_cover"), axis, axis, 250.0
        )

        assert 0.0 <= value.retained_weight <= 1.0
        assert 0.0 <= disc.retained_weight <= 1.0
        assert bool(((composition >= 0.0) & (composition <= 1.0)).all())
        assert bool(((target >= 0.0) & (target <= 1.0)).all())
        assert float(composition.sum()) == pytest.approx(1.0)
        assert float(target.sum()) == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_regression_statistics_are_fractions(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.1, 0.9, 20)
        y = rng.uniform(0.0, 1.2) * x + rng.normal(0.0, rng.uniform(0.0, 0.2), 20)

        fit = rma_regression(x, y)

        assert 0.0 <= fit.r_squared <= 1.0
        assert 0.0 <= fit.p_value <= 1.0

    def test_categorical_shares_and_p_values_are_fractions(
        self, site_footprint: xr.DataArray, ring_landcover: xr.DataArray
    ) -> None:
        frame = categorical_representativeness(
            site_footprint, ring_landcover, SITE_AXIS, SITE_AXIS, radii=TARGET_RADII
        )

        for column in ("p_footprint", "p_target"):
            assert bool(frame[column].between(0.0, 1.0).all())
        finite = frame["p_value"].replace([np.inf, -np.inf], np.nan).dropna()
        assert bool(finite.between(0.0, 1.0).all())
        assert set(frame["level"]).issubset({level.value for level in Level})

    def test_continuous_within_threshold_share_is_a_fraction(
        self, scene_pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]
    ) -> None:
        series = sensor_location_bias_series(
            scene_pairs, SITE_AXIS, SITE_AXIS, radii=TARGET_RADII
        )

        share = series.groupby("radius")["within_threshold"].mean()

        assert bool(((share >= 0.0) & (share <= 1.0)).all())


# ------------------------------
# 7. Property: representativeness degrades with the target radius
# ------------------------------


class TestMonotoneDegradation:
    """
    On a concentric-heterogeneity site, a wider disc can only look worse.

    The footprint sits inside the innermost land-cover ring and inside the core
    of every vegetation-index scene, so widening the target area can only pull
    in surface the footprint never saw. Neither three-level index may therefore
    improve as the radius grows, and the sensor location bias may only grow.
    """

    def test_land_cover_levels_never_improve_with_radius(
        self, site_footprint: xr.DataArray, ring_landcover: xr.DataArray
    ) -> None:
        frame = categorical_representativeness(
            site_footprint, ring_landcover, SITE_AXIS, SITE_AXIS, radii=TARGET_RADII
        )

        ranks = [LEVEL_RANK[as_level(level)] for level in frame["level"]]

        assert ranks == sorted(ranks, reverse=True)
        # The test has teeth: the smallest disc is representative and the
        # largest is not.
        assert as_level(frame["level"].iloc[0]) is Level.HIGH
        assert as_level(frame["level"].iloc[-1]) is Level.LOW

    def test_the_dominant_class_share_of_the_disc_falls_with_radius(
        self, site_footprint: xr.DataArray, ring_landcover: xr.DataArray
    ) -> None:
        """P_target thins as the outer rings enter; P_footprint cannot move."""
        frame = categorical_representativeness(
            site_footprint, ring_landcover, SITE_AXIS, SITE_AXIS, radii=TARGET_RADII
        )

        shares = list(frame["p_target"])

        assert shares == sorted(shares, reverse=True)
        assert shares[0] > shares[-1]
        assert frame["p_footprint"].nunique() == 1
        assert frame["dominant_class"].nunique() == 1
        assert int(frame["dominant_class"].iloc[0]) == RING_CLASSES[0]

    def test_sensor_location_bias_grows_with_radius(
        self, smooth_scene_pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]
    ) -> None:
        """
        Every scene peaks on the tower, so Delta is positive and non-decreasing.

        This is the sign the paper found at every one of its target radii: the
        footprint covers higher values than its surroundings (Sect. 3.3).
        """
        series = sensor_location_bias_series(
            smooth_scene_pairs, SITE_AXIS, SITE_AXIS, radii=TARGET_RADII
        )

        for _, period in series.groupby("time"):
            deltas = list(period.sort_values("radius")["delta"])
            assert all(delta > 0.0 for delta in deltas)
            assert deltas == sorted(deltas)

    def test_within_threshold_share_falls_with_radius(
        self, smooth_scene_pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]
    ) -> None:
        """Fig. 7: fewer periods clear +/-10 % as the disc widens."""
        series = sensor_location_bias_series(
            smooth_scene_pairs, SITE_AXIS, SITE_AXIS, radii=TARGET_RADII
        )

        share = series.groupby("radius")["within_threshold"].mean().astype(float)

        assert list(share) == sorted(share, reverse=True)
        assert share.iloc[0] > share.iloc[-1]

    def test_continuous_levels_never_improve_with_radius(
        self, scene_pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]
    ) -> None:
        """The regression of Eq. 7 walks HIGH to LOW as the disc widens."""
        series = sensor_location_bias_series(
            scene_pairs, SITE_AXIS, SITE_AXIS, radii=TARGET_RADII
        )

        frame = continuous_representativeness(series, radii=TARGET_RADII)

        assert bool(frame["sufficient"].all())
        assert bool((frame["n"] >= MIN_MATCHES).all())
        ranks = [LEVEL_RANK[as_level(level)] for level in frame["level"]]
        assert ranks == sorted(ranks, reverse=True)
        # All three levels are exercised on the way down.
        assert {as_level(level) for level in frame["level"]} == set(Level)

    def test_regression_slope_falls_away_from_one_with_radius(
        self, scene_pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]
    ) -> None:
        """Table 1: the slope drops below 1 and keeps dropping, as at every site."""
        series = sensor_location_bias_series(
            scene_pairs, SITE_AXIS, SITE_AXIS, radii=[250.0, 500.0, 1000.0]
        )

        frame = continuous_representativeness(series, radii=[250.0, 500.0, 1000.0])

        slopes = list(frame["slope"])
        assert slopes[0] < 1.0
        assert slopes == sorted(slopes, reverse=True)

    def test_too_few_matches_leaves_the_level_missing(
        self, scene_pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]
    ) -> None:
        """Below `min_matches` a radius is reported unfitted, as in Sect. 2.4."""
        few = dict(list(scene_pairs.items())[:4])
        series = sensor_location_bias_series(few, SITE_AXIS, SITE_AXIS, radii=[250.0])

        frame = continuous_representativeness(series, radii=[250.0])

        assert bool((~frame["sufficient"]).all())
        assert frame["level"].isna().all()
        assert int(frame["n"].iloc[0]) == 4


# ------------------------------
# 8. The dataclass-returning evaluators and the summary table
# ------------------------------


class TestEvaluatorsAndSummary:
    """The wrappers that report percentages and flatten them into a table."""

    def test_evaluate_landcover_reports_percentages(
        self, site_footprint: xr.DataArray, ring_landcover: xr.DataArray
    ) -> None:
        radii = [250.0, 1000.0, 3000.0]

        results = evaluate_landcover(site_footprint, ring_landcover, radii=radii)
        frame = categorical_representativeness(
            site_footprint, ring_landcover, SITE_AXIS, SITE_AXIS, radii=radii
        )

        assert [entry.radius for entry in results] == radii
        assert all(isinstance(entry, CategoricalResult) for entry in results)
        for entry, (_, row) in zip(results, frame.iterrows(), strict=True):
            # The frame reports fractions; the dataclass reports the paper's
            # percentages.
            assert entry.p_footprint == pytest.approx(100.0 * row["p_footprint"])
            assert entry.p_target == pytest.approx(100.0 * row["p_target"])
            assert entry.level is as_level(row["level"])
        assert sum(results[0].footprint_composition.values()) == pytest.approx(100.0)
        assert sum(results[-1].target_composition.values()) == pytest.approx(100.0)
        # The widest disc has reached the outermost rings.
        assert len(results[-1].target_composition) > len(results[0].target_composition)

    def test_evaluate_vegetation_index_matches_the_regression(
        self, scene_pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]]
    ) -> None:
        radii = [250.0, 1000.0]
        climatologies = [pair[0] for pair in scene_pairs.values()]
        fields = [pair[1] for pair in scene_pairs.values()]

        results = evaluate_vegetation_index(climatologies, fields, radii=radii)
        series = sensor_location_bias_series(
            scene_pairs, SITE_AXIS, SITE_AXIS, radii=radii
        )
        frame = continuous_representativeness(series, radii=radii)

        assert [entry.radius for entry in results] == radii
        assert all(isinstance(entry, ContinuousResult) for entry in results)
        for entry, (_, row) in zip(results, frame.iterrows(), strict=True):
            assert entry.slope == pytest.approx(row["slope"])
            assert entry.intercept == pytest.approx(row["intercept"])
            assert entry.r_squared == pytest.approx(row["r_squared"])
            assert entry.n == int(row["n"])
            assert entry.level is as_level(row["level"])
            assert entry.bias.size == entry.n
            assert 0.0 <= entry.within_threshold <= 1.0

    def test_summary_joins_the_two_halves_on_the_radius(
        self,
        site_footprint: xr.DataArray,
        ring_landcover: xr.DataArray,
        scene_pairs: dict[pd.Timestamp, tuple[xr.DataArray, xr.DataArray]],
    ) -> None:
        radii = [250.0, 1000.0]
        categorical = evaluate_landcover(site_footprint, ring_landcover, radii=radii)
        continuous = evaluate_vegetation_index(
            [pair[0] for pair in scene_pairs.values()],
            [pair[1] for pair in scene_pairs.values()],
            radii=radii,
        )
        metrics = climatology_metrics(site_footprint)

        table = representativeness_summary(
            categorical, continuous, metrics, site_id="US-Syn"
        )

        assert list(table["radius"]) == radii
        assert set(table["site_id"]) == {"US-Syn"}
        assert list(table["landcover_level"]) == [
            entry.level.value for entry in categorical
        ]
        assert list(table["continuous_level"]) == [
            entry.level.value for entry in continuous
        ]
        # Climatology metrics describe the footprint, so they repeat.
        assert table["symmetry"].nunique() == 1
        assert table["fetch"].iloc[0] == pytest.approx(metrics.fetch)

    def test_summary_needs_at_least_one_half(self) -> None:
        with pytest.raises(ValueError):
            representativeness_summary()


# ------------------------------
# 9. End-to-end run on the bundled AmeriFlux tower (slow)
# ------------------------------

#: Bundled US-CRT record: an alfalfa cropland on Lake Erie's south shore, whose
#: config and half-hourly BASE file ship with the repository.
CRT_CONFIG = DATA_DIR / "US-CRT_config.ini"
CRT_BASE = DATA_DIR / "AMF_US-CRT_BASE_HH_3-5.csv"

#: The growing season the integration test drives, one Landsat-like scene per
#: month, which clears the paper's six-match floor for a site-level regression.
CRT_START = "2011-04-01"
CRT_END = "2011-11-30"
CRT_MONTHS = 8
CRT_TZ = -5

#: Grid the climatology is built on. A 2.5 m tower over a 0.4 m canopy has a
#: source area of a few hundred metres, so 10 m cells out to 300 m hold about
#: 92 % of the flux -- enough that the 80 % contour is a real contour rather
#: than the whole domain.
CRT_STEP = 10.0
CRT_HALF_WIDTH = 300.0
CRT_RADII = (100.0, 200.0, 300.0)

#: The tower's own land-cover class, held within 120 m of it, and the
#: surrounding one -- a synthetic stand-in for the NLCD tile of the paper, so
#: that the test needs no raster download and no optional dependency.
CRT_NEAR_CLASS = 82
CRT_FAR_CLASS = 141

#: Skip reason when the repository is checked out without its data files.
MISSING_RECORD = "the bundled AmeriFlux US-CRT BASE record is not present"


@pytest.fixture(scope="module")
def crt_model():
    """
    A footprint model run on the bundled US-CRT half-hourly record.

    Builds real footprints from the shipped AmeriFlux BASE file: the config
    supplies the tower position, the BASE columns are renamed to the solver's
    names, weak-turbulence half-hours are dropped, and the aerodynamic
    parameters come from the alfalfa preset the site's own metadata describes.
    """
    pytest.importorskip("fluxfootprints.improved_ffp")
    if not (CRT_CONFIG.exists() and CRT_BASE.exists()):
        pytest.skip(MISSING_RECORD)

    from fluxfootprints import (
        build_climatology,
        compute_aerodynamic_params,
        load_amf_df,
        load_config,
    )

    config = load_config(CRT_CONFIG)
    frame = load_amf_df(CRT_BASE, config).rename(
        columns={
            "WD": "wind_dir",
            "WS": "umean",
            "USTAR": "ustar",
            "MO_LENGTH": "ol",
            "V_SIGMA": "sigmav",
        }
    )
    season = frame.loc[
        CRT_START:CRT_END, ["wind_dir", "umean", "ustar", "ol", "sigmav"]
    ].dropna()
    # Kljun et al. (2015) do not apply below ustar ~ 0.1 m/s; the paper filters
    # such half-hours out before aggregating (Table S2).
    season = season[season["ustar"] >= 0.15]
    season = compute_aerodynamic_params(
        season, inst_height=2.5, crop_height=0.4, veg_type="alfalfa"
    )

    model = build_climatology(
        season,
        model_type="ffp",
        ustar="ustar",
        ol="ol",
        umean="umean",
        sigmav="sigmav",
        wind_dir="wind_dir",
        zm="zm",
        z0="z0",
        h=2000.0,
        dx=CRT_STEP,
        dy=CRT_STEP,
        domain=(-CRT_HALF_WIDTH, CRT_HALF_WIDTH, -CRT_HALF_WIDTH, CRT_HALF_WIDTH),
        verbosity=0,
    )
    return model, config


@pytest.fixture(scope="module")
def crt_fields(crt_model) -> tuple[xr.DataArray, dict[pd.Timestamp, xr.DataArray]]:
    """Synthetic land-cover and vegetation-index fields on the model's own grid."""
    model, _ = crt_model
    axis_x = np.asarray(model.x, dtype=float)
    axis_y = np.asarray(model.y, dtype=float)
    xx, yy = np.meshgrid(axis_x, axis_y, indexing="ij")
    radius = np.hypot(xx, yy)

    def grid(values: np.ndarray, name: str) -> xr.DataArray:
        return xr.DataArray(
            values, coords={"x": axis_x, "y": axis_y}, dims=("x", "y"), name=name
        )

    landcover = grid(
        np.where(radius <= 120.0, float(CRT_NEAR_CLASS), float(CRT_FAR_CLASS)),
        "land_cover",
    )
    scenes = {
        pd.Timestamp(f"2011-{month:02d}-15"): grid(
            0.2
            + 0.5 * np.exp(-0.5 * (radius / 200.0) ** 2) * (1.0 + 0.2 * np.sin(index)),
            "EVI",
        )
        for index, month in enumerate(range(4, 4 + CRT_MONTHS))
    }
    return landcover, scenes


@pytest.fixture(scope="module")
def results(crt_model, crt_fields) -> pd.DataFrame:
    """The whole analysis of Sects. 2.2-2.4, run once for the class below."""
    model, config = crt_model
    landcover, scenes = crt_fields
    return assess_representativeness(
        model,
        station_lat=config["station_latitude"],
        station_lon=config["station_longitude"],
        site_id="US-CRT",
        landcover=landcover,
        continuous=scenes,
        radii=CRT_RADII,
        tz=CRT_TZ,
    )


@pytest.mark.slow
class TestBundledTowerPipeline:
    """
    :func:`assess_representativeness` end to end on real half-hourly data.

    Nothing here re-checks arithmetic the sections above pin down. What it
    checks is that the driver survives a real record -- eight months of
    half-hourly AmeriFlux data, aggregated into daytime and nighttime monthly
    climatologies and compared against three target discs -- and hands back the
    frame its schema documents.
    """

    def test_frame_carries_the_documented_index_and_columns(
        self, results: pd.DataFrame
    ) -> None:
        assert tuple(results.index.names) == RESULT_INDEX
        assert tuple(results.columns) == RESULT_COLUMNS
        assert not results.empty

    def test_columns_have_the_documented_dtypes(self, results: pd.DataFrame) -> None:
        """Counts and flags stay nullable, so a missing one does not become 0.0."""
        for column in ("dof", "n", "n_cells", "n_times"):
            assert results[column].dtype == "Int64"
        for column in ("within_threshold", "sufficient"):
            assert results[column].dtype == "boolean"
        for column in ("value_footprint", "fetch", "area", "symmetry", "r_squared"):
            assert results[column].dtype == np.dtype("float64")

    def test_index_levels_carry_the_run(self, results: pd.DataFrame) -> None:
        index = results.reset_index()

        assert set(index["site"].dropna()) == {"US-CRT"}
        assert set(index["period"].dropna()) == {"daytime", "nighttime"}
        assert set(index["variable"].dropna()) == {"footprint", "land_cover", "EVI"}
        assert set(index["radius"].dropna()) == set(CRT_RADII)
        assert set(index["year"].dropna()) == {2011}

    def test_every_documented_scope_and_kind_block_is_present(
        self, results: pd.DataFrame
    ) -> None:
        blocks = set(
            map(tuple, results.reset_index()[["scope", "kind"]].dropna().to_numpy())
        )

        assert blocks == {
            (PERIOD_SCOPE, CLIMATOLOGY_KIND),
            (PERIOD_SCOPE, CATEGORICAL_KIND),
            (PERIOD_SCOPE, CONTINUOUS_KIND),
            (SITE_YEAR_SCOPE, CLIMATOLOGY_KIND),
            (SITE_SCOPE, CATEGORICAL_KIND),
            (SITE_SCOPE, CONTINUOUS_KIND),
        }

    def test_row_counts_follow_from_the_months_periods_and_radii(
        self, results: pd.DataFrame
    ) -> None:
        counts = results.reset_index().groupby(["scope", "kind"]).size()
        periods = 2  # daytime and nighttime
        radii = len(CRT_RADII)

        assert counts[(PERIOD_SCOPE, CLIMATOLOGY_KIND)] == CRT_MONTHS * periods
        assert counts[(PERIOD_SCOPE, CATEGORICAL_KIND)] == CRT_MONTHS * periods * radii
        assert counts[(PERIOD_SCOPE, CONTINUOUS_KIND)] == CRT_MONTHS * periods * radii
        assert counts[(SITE_YEAR_SCOPE, CLIMATOLOGY_KIND)] == periods
        assert counts[(SITE_SCOPE, CATEGORICAL_KIND)] == periods * radii
        assert counts[(SITE_SCOPE, CONTINUOUS_KIND)] == periods * radii

    def test_per_period_climatology_rows_are_physical(
        self, results: pd.DataFrame
    ) -> None:
        """Fetch, area, and symmetry of Sect. 2.2, on a real source area."""
        rows = results.query("scope == @PERIOD_SCOPE and kind == @CLIMATOLOGY_KIND")

        assert bool(rows["fetch"].between(0.0, CRT_HALF_WIDTH * math.sqrt(2.0)).all())
        assert bool((rows["area"] > 0.0).all())
        assert bool(rows["symmetry"].between(0.0, 1.0).all())
        assert bool((rows["contour_level"] > 0.0).all())
        assert bool((rows["n_cells"] > 0).all())
        # Each aggregated month-period stands on real half-hours.
        assert bool((rows["n_times"] > 0).all())
        # The radius level belongs to the discs, not to the footprint itself.
        assert rows.reset_index()["radius"].isna().all()

    def test_nighttime_footprints_reach_farther_than_daytime(
        self, results: pd.DataFrame
    ) -> None:
        """
        Sect. 3.1: nighttime source areas extended farther at >95 % of site-years.

        Read off the site-year rows, whose fetch and area are the month means
        of Fig. 3.
        """
        rows = results.query(
            "scope == @SITE_YEAR_SCOPE and kind == @CLIMATOLOGY_KIND"
        ).reset_index()
        by_period = rows.set_index("period")

        assert by_period.loc["nighttime", "fetch"] > by_period.loc["daytime", "fetch"]

    def test_site_year_rows_carry_the_overlap_indices(
        self, results: pd.DataFrame
    ) -> None:
        """Eqs. 2-3 are site-year properties, so they live only on these rows."""
        rows = results.query("scope == @SITE_YEAR_SCOPE and kind == @CLIMATOLOGY_KIND")

        assert bool(rows["seasonal_overlap"].between(0.0, 1.0).all())
        assert bool(rows["daynight_overlap"].between(0.0, 1.0).all())
        # Day and night share the high-weight core, so they overlap more than
        # the months of a season do -- the paper's Fig. 3d.
        assert bool((rows["daynight_overlap"] > rows["seasonal_overlap"]).all())

    def test_per_period_continuous_rows_hold_the_matched_scenes(
        self, results: pd.DataFrame
    ) -> None:
        """One row per matched scene, period, and radius -- the paper's Dataset S5."""
        rows = results.query("scope == @PERIOD_SCOPE and kind == @CONTINUOUS_KIND")

        assert rows["time"].notna().all()
        assert bool(rows["value_footprint"].between(0.0, 1.0).all())
        assert bool(rows["value_target"].between(0.0, 1.0).all())
        assert rows["bias"].notna().all()
        assert rows["within_threshold"].notna().all()
        assert bool(rows["retained_footprint"].between(0.0, 1.0).all())
        assert bool(rows["retained_target"].between(0.0, 1.0).all())
        # The scenes peak on the tower, so the footprint sees higher values
        # than its surroundings at every radius, as at every site in Sect. 3.3.
        assert bool((rows["bias"] > 0.0).all())

    def test_categorical_rows_report_a_class_and_a_level(
        self, results: pd.DataFrame
    ) -> None:
        rows = results.query("kind == @CATEGORICAL_KIND")

        assert set(rows["dominant_class"].dropna()) == {CRT_NEAR_CLASS}
        assert bool(rows["value_footprint"].between(0.0, 1.0).all())
        assert bool(rows["value_target"].between(0.0, 1.0).all())
        assert set(rows["level"].dropna()) <= {level.value for level in Level}
        assert rows["level"].notna().all()
        # Sect. 2.4 applies the +/-10 % threshold only to the continuous field.
        assert rows["within_threshold"].isna().all()

    def test_site_level_regression_is_fitted_and_classified(
        self, results: pd.DataFrame
    ) -> None:
        """The paper's Dataset S6 and Table 1, per period and radius."""
        rows = results.query("scope == @SITE_SCOPE and kind == @CONTINUOUS_KIND")

        assert bool(rows["sufficient"].all())
        assert bool((rows["n"] == CRT_MONTHS).all())
        assert bool((rows["n"] >= MIN_MATCHES).all())
        assert rows["level"].notna().all()
        assert bool(rows["r_squared"].between(0.0, 1.0).all())
        assert bool(rows["p_value"].between(0.0, 1.0).all())
        assert bool((rows["rmse"] >= rows["mae"]).all())
        assert bool((rows["slope_lower"] <= rows["slope"]).all())
        assert bool((rows["slope"] <= rows["slope_upper"]).all())
        assert bool((rows["intercept_lower"] <= rows["intercept"]).all())
        assert bool((rows["intercept"] <= rows["intercept_upper"]).all())

    def test_representativeness_falls_off_with_the_target_radius(
        self, results: pd.DataFrame
    ) -> None:
        """
        The synthetic fields are concentric, so the real site behaves like the
        property fixture: no disc a step wider looks better than the one inside
        it.
        """
        rows = (
            results.query("scope == @SITE_SCOPE and kind == @CONTINUOUS_KIND")
            .reset_index()
            .sort_values(["period", "radius"])
        )

        for _, period in rows.groupby("period"):
            ranks = [LEVEL_RANK[as_level(level)] for level in period["level"]]
            assert ranks == sorted(ranks, reverse=True)
            assert as_level(period["level"].iloc[0]) is Level.HIGH

    def test_attrs_record_the_settings_the_run_used(
        self, results: pd.DataFrame
    ) -> None:
        attrs = results.attrs

        assert attrs["site_id"] == "US-CRT"
        assert attrs["radii"] == tuple(CRT_RADII)
        assert attrs["contour_fraction"] == FRACTION
        assert attrs["bias_threshold"] == BIAS_THRESHOLD
        assert attrs["min_matches"] == MIN_MATCHES
        assert attrs["partition"] == "month+daynight"
        assert attrs["landcover_variable"] == "land_cover"
        assert attrs["continuous_variable"] == "EVI"
        # Every scene fell in a month that was aggregated.
        assert attrs["unmatched_fields"] == ()
        assert "108350" in attrs["reference"]

    def test_each_analysis_is_optional(self, crt_model, crt_fields) -> None:
        """A land-cover-only run is the same driver with the scenes left out."""
        model, config = crt_model
        landcover, _ = crt_fields

        results = assess_representativeness(
            model,
            station_lat=config["station_latitude"],
            station_lon=config["station_longitude"],
            site_id="US-CRT",
            landcover=landcover,
            radii=CRT_RADII,
            tz=CRT_TZ,
        )

        kinds = set(results["kind"].dropna())
        assert CONTINUOUS_KIND not in kinds
        assert {CLIMATOLOGY_KIND, CATEGORICAL_KIND} <= kinds
        assert tuple(results.columns) == RESULT_COLUMNS
