"""
Land-cover representativeness tests for :mod:`fluxfootprints.representativeness`.

Sect. 2.4 of Chu et al. (2021) combines three quantities per target radius --
the dominant class's footprint-weighted share, its share of the disc, and a
chi-square test between the two full compositions -- so the fixtures here are
built to pin each of them down by hand:

* the grid skips ``x = y = 0``, so the disc of radius 10 m holds exactly the
  four cells at ``(+/-5, +/-5)``. Painting classes onto those four cells puts
  P_target on an exact quarter, which lands the 0.50 and 0.80 criteria of
  Goeckede et al. (2008) on values that are exact in binary rather than on
  floating-point luck at the boundary;
* a footprint carrying all its weight on cells of one class fixes P_footprint
  at exactly 1 whatever the surrounding landscape does, which separates the
  dominant-class step from the target-area step;
* a landscape of a single class makes both compositions identical and the test
  degenerate -- one class, no degrees of freedom -- which is the homogeneous
  site the paper's HIGH level is meant to catch, and which ``scipy`` alone
  reports as a ``nan`` p-value.

The chi-square rows then check the two documented conventions rather than a
statistic copied back out of the implementation: that the pseudo-counts scale
with the target area's classified cell count, and that classes absent from
both compositions are dropped so they cost no degrees of freedom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxfootprints.representativeness import (
    DEFAULT_ALPHA,
    TARGET_RADII,
    Level,
    categorical_representativeness,
    classify_categorical,
    footprint_weighted_composition,
    target_area_composition,
)

#: 20 x 20 cells of 10 m, centred on the tower and skipping x = y = 0.
GRID = np.arange(-95.0, 100.0, 10.0)

XX, YY = np.meshgrid(GRID, GRID, indexing="ij")

#: Radius of the disc holding exactly the four cells at (+/-5, +/-5).
FOUR_CELL_RADIUS = 10.0

#: Consolidated land-cover codes of Table S6, in the style of NLCD.
FOREST, CROP, WATER = 41, 82, 11

COLUMNS = [
    "radius",
    "dominant_class",
    "p_footprint",
    "p_target",
    "chi2",
    "p_value",
    "dof",
    "level",
]


# ----------------------------
# Fixtures
# ----------------------------
def raster(values: np.ndarray, name: str = "landcover") -> xr.DataArray:
    """Wrap a (x, y) array as a raster on the tower-centred grid."""
    return xr.DataArray(
        np.asarray(values, dtype=float),
        coords={"x": GRID, "y": GRID},
        dims=("x", "y"),
        name=name,
    )


def uniform(code: int) -> xr.DataArray:
    """A landscape of a single class."""
    return raster(np.full(XX.shape, float(code)))


def split(west: int, east: int) -> xr.DataArray:
    """A landscape halved along x, `west` for x < 0 and `east` for x > 0."""
    return raster(np.where(XX < 0, float(west), float(east)))


def four_cells(codes: tuple[int, int, int, int]) -> xr.DataArray:
    """
    Paint `codes` onto the four cells of the 10 m disc, WATER elsewhere.

    The cells are taken in ``(x, y)`` order ``(-5, -5), (-5, 5), (5, -5),
    (5, 5)``, so a caller can put P_target on an exact quarter.
    """
    values = np.full(XX.shape, float(WATER))
    inner = (np.abs(XX) == 5.0) & (np.abs(YY) == 5.0)
    values[inner] = np.asarray(codes, dtype=float)
    return raster(values)


def point_footprint(x0: float, y0: float) -> xr.DataArray:
    """A footprint carrying all its weight on the single cell at (x0, y0)."""
    values = np.where((XX == x0) & (YY == y0), 1.0, 0.0)
    return raster(values, name="w")


def disc_footprint(radius: float) -> xr.DataArray:
    """A footprint spread evenly over the cells within `radius` of the tower."""
    values = np.where(np.hypot(XX, YY) <= radius, 1.0, 0.0)
    return raster(values / values.sum(), name="w")


# ----------------------------
# Frame shape and invariants
# ----------------------------
def test_returns_one_row_per_radius_in_the_order_given() -> None:
    """The frame follows `radii`, as the caller ordered them."""
    radii = (90.0, FOUR_CELL_RADIUS, 50.0)
    frame = categorical_representativeness(
        disc_footprint(30.0), uniform(FOREST), GRID, GRID, radii=radii
    )

    assert list(frame.columns) == COLUMNS
    assert frame["radius"].tolist() == list(radii)


def test_defaults_to_the_paper_radii() -> None:
    """Sect. 2.1's six target areas are the default."""
    frame = categorical_representativeness(
        disc_footprint(30.0), uniform(FOREST), GRID, GRID
    )

    assert frame["radius"].tolist() == [float(r) for r in TARGET_RADII]


def test_dominant_class_and_footprint_share_are_constant_down_the_frame() -> None:
    """Neither depends on the target area, so both repeat unchanged."""
    frame = categorical_representativeness(
        point_footprint(-5.0, -5.0),
        split(FOREST, CROP),
        GRID,
        GRID,
        radii=(FOUR_CELL_RADIUS, 50.0, 90.0),
    )

    assert frame["dominant_class"].nunique() == 1
    assert frame["p_footprint"].nunique() == 1


def test_attrs_record_alpha_and_the_footprint_composition() -> None:
    """The composition the dominant class came from travels with the frame."""
    w, landcover = disc_footprint(30.0), split(FOREST, CROP)
    frame = categorical_representativeness(
        w, landcover, GRID, GRID, radii=(90.0,), alpha=0.01
    )

    assert frame.attrs["alpha"] == pytest.approx(0.01)
    pd.testing.assert_series_equal(
        frame.attrs["footprint_composition"],
        footprint_weighted_composition(w, landcover),
    )


# ----------------------------
# The dominant class (P_footprint)
# ----------------------------
def test_dominant_class_is_the_largest_footprint_share_not_the_largest_area() -> None:
    """
    A footprint on one cell of a class the landscape barely holds still names
    that class dominant -- the mismatch the index exists to report.
    """
    landcover = four_cells((FOREST, WATER, WATER, WATER))
    frame = categorical_representativeness(
        point_footprint(-5.0, -5.0), landcover, GRID, GRID, radii=(90.0,)
    )

    assert frame.loc[0, "dominant_class"] == FOREST
    assert frame.loc[0, "p_footprint"] == pytest.approx(1.0)
    # FOREST is one cell of the whole domain, so it dominates nothing else.
    assert frame.loc[0, "p_target"] < 0.01


def test_ties_for_the_largest_share_go_to_the_lowest_class_code() -> None:
    """Documented tie-breaking, so a tied frame is reproducible."""
    values = np.zeros(XX.shape)
    values[(XX == -5.0) & (YY == 5.0)] = 1.0
    values[(XX == 5.0) & (YY == 5.0)] = 1.0
    w = raster(values / 2.0, name="w")

    frame = categorical_representativeness(
        w, split(CROP, FOREST), GRID, GRID, radii=(FOUR_CELL_RADIUS,)
    )

    assert frame.loc[0, "p_footprint"] == pytest.approx(0.5)
    assert frame.loc[0, "dominant_class"] == FOREST  # 41 < 82


def test_class_codes_come_back_as_integers() -> None:
    """
    The alignment path returns float64, so an NLCD code arrives as ``41.0``;
    a caller holding a class lookup table will index it with ``41``.
    """
    frame = categorical_representativeness(
        disc_footprint(30.0), uniform(FOREST), GRID, GRID, radii=(90.0,)
    )

    dominant = frame.loc[0, "dominant_class"]
    assert dominant == FOREST
    assert isinstance(dominant, (int, np.integer))


# ----------------------------
# The target-area share (P_target)
# ----------------------------
def test_target_share_matches_the_composition_over_the_same_disc() -> None:
    """P_target is the dominant class's row of :func:`target_area_composition`."""
    landcover = split(FOREST, CROP)
    frame = categorical_representativeness(
        point_footprint(-5.0, -5.0),
        landcover,
        GRID,
        GRID,
        radii=(FOUR_CELL_RADIUS, 90.0),
    )

    for row in frame.itertuples():
        composition = target_area_composition(landcover, GRID, GRID, row.radius)
        assert row.p_target == pytest.approx(composition[row.dominant_class])


def test_target_share_falls_to_zero_when_the_disc_holds_none_of_the_class() -> None:
    """A dominant class the disc lacks reads as 0, not as a missing value."""
    # FOREST sits at (-5, -5) only; the 10 m disc holds it, a wider ring of
    # CROP does not, so shrinking the disc to that one cell is not the test --
    # instead put the footprint on FOREST and read a disc that excludes it.
    values = np.full(XX.shape, float(CROP))
    values[(XX == -85.0) & (YY == -85.0)] = float(FOREST)
    landcover = raster(values)

    frame = categorical_representativeness(
        point_footprint(-85.0, -85.0), landcover, GRID, GRID, radii=(FOUR_CELL_RADIUS,)
    )

    assert frame.loc[0, "dominant_class"] == FOREST
    assert frame.loc[0, "p_target"] == 0.0


def test_target_share_is_an_exact_quarter_on_the_four_cell_disc() -> None:
    """One of four cells is 0.25 exactly, so the criteria sit off the boundary."""
    landcover = four_cells((FOREST, CROP, CROP, CROP))
    frame = categorical_representativeness(
        point_footprint(-5.0, -5.0), landcover, GRID, GRID, radii=(FOUR_CELL_RADIUS,)
    )

    assert frame.loc[0, "p_target"] == 0.25


# ----------------------------
# The chi-square test
# ----------------------------
def test_identical_single_class_compositions_are_not_significantly_different() -> None:
    """
    A homogeneous landscape leaves one class and no degrees of freedom, which
    ``scipy`` reports as a ``nan`` p-value; that would misclassify the very
    sites the paper's HIGH level is for.
    """
    frame = categorical_representativeness(
        disc_footprint(30.0), uniform(FOREST), GRID, GRID, radii=(FOUR_CELL_RADIUS,)
    )

    assert frame.loc[0, "dof"] == 0
    assert frame.loc[0, "chi2"] == 0.0
    assert frame.loc[0, "p_value"] == 1.0
    assert frame.loc[0, "level"] == Level.HIGH


def test_matching_compositions_are_not_significantly_different() -> None:
    """Two classes the footprint and the disc hold in the same shares agree."""
    landcover = split(FOREST, CROP)
    frame = categorical_representativeness(
        disc_footprint(90.0), landcover, GRID, GRID, radii=(90.0,)
    )

    assert frame.loc[0, "dof"] == 1
    assert frame.loc[0, "chi2"] == pytest.approx(0.0, abs=1e-9)
    assert frame.loc[0, "p_value"] == pytest.approx(1.0)


def test_a_class_the_disc_lacks_makes_the_statistic_unbounded() -> None:
    """
    Zero expected pseudo-counts leave a term that cannot be finite, so the
    compositions are irreconcilable rather than merely unlikely.
    """
    # The footprint spans FOREST and CROP; the four-cell disc holds only CROP.
    landcover = four_cells((CROP, CROP, CROP, CROP))
    values = np.zeros(XX.shape)
    values[(XX == -5.0) & (YY == -5.0)] = 1.0  # CROP
    values[(XX == -95.0) & (YY == -95.0)] = 1.0  # WATER
    w = raster(values / 2.0, name="w")

    frame = categorical_representativeness(
        w, landcover, GRID, GRID, radii=(FOUR_CELL_RADIUS,)
    )

    assert np.isinf(frame.loc[0, "chi2"])
    assert frame.loc[0, "p_value"] == 0.0
    assert frame.loc[0, "level"] == Level.LOW


def test_classes_absent_from_both_cost_no_degrees_of_freedom() -> None:
    """
    A product carrying classes neither the footprint nor the disc holds must
    not spend degrees of freedom on them, or the test weakens with the legend
    rather than with the landscape.
    """
    # Two classes inside the 10 m disc; a third and fourth far outside it.
    values = np.full(XX.shape, float(WATER))
    values[XX > 50.0] = 21.0
    values[(np.abs(XX) == 5.0) & (np.abs(YY) == 5.0)] = np.asarray(
        [FOREST, FOREST, CROP, CROP], dtype=float
    )
    landcover = raster(values)

    frame = categorical_representativeness(
        disc_footprint(FOUR_CELL_RADIUS),
        landcover,
        GRID,
        GRID,
        radii=(FOUR_CELL_RADIUS,),
    )

    assert frame.loc[0, "dof"] == 1  # FOREST and CROP only


def test_pseudo_counts_scale_with_the_target_areas_classified_cells() -> None:
    """
    The documented sample size is the disc's classified cell count, so the
    same pair of compositions gives a statistic that grows with the disc.
    """
    landcover = split(FOREST, CROP)
    # A footprint 3:1 on FOREST against discs that are 1:1, at two sizes.
    values = np.zeros(XX.shape)
    values[(XX == -5.0) & (np.abs(YY) == 5.0)] = 1.5
    values[(XX == 5.0) & (np.abs(YY) == 5.0)] = 0.5
    w = raster(values / values.sum(), name="w")

    frame = categorical_representativeness(
        w, landcover, GRID, GRID, radii=(FOUR_CELL_RADIUS, 90.0)
    )

    small, large = frame.loc[0], frame.loc[1]
    n_small = target_area_composition(
        landcover, GRID, GRID, FOUR_CELL_RADIUS
    ).attrs["n_cells"]
    n_large = target_area_composition(landcover, GRID, GRID, 90.0).attrs["n_cells"]

    # Compositions are identical at both radii; only the sample size moves.
    assert small.p_footprint == pytest.approx(large.p_footprint)
    assert small.p_target == pytest.approx(large.p_target)
    assert large.chi2 == pytest.approx(small.chi2 * n_large / n_small)


def test_an_unclassified_disc_leaves_the_test_undefined() -> None:
    """
    No classified cell inside the disc means no sample size, which must read
    as an undefined test rather than as agreement.
    """
    landcover = raster(np.where(np.hypot(XX, YY) < 15.0, np.nan, float(FOREST)))
    frame = categorical_representativeness(
        disc_footprint(60.0), landcover, GRID, GRID, radii=(FOUR_CELL_RADIUS, 90.0)
    )

    assert np.isnan(frame.loc[0, "chi2"])
    assert np.isnan(frame.loc[0, "p_value"])
    assert frame.loc[0, "level"] == Level.LOW
    # The wider disc is classified, and matches the footprint exactly.
    assert frame.loc[1, "level"] == Level.HIGH


# ----------------------------
# Classification (Sect. 2.4)
# ----------------------------
@pytest.mark.parametrize(
    ("p_footprint", "p_target", "p_value", "expected"),
    [
        (92.0, 88.0, 0.42, Level.HIGH),
        (80.0, 80.0, 0.05, Level.HIGH),  # every criterion inclusive
        (92.0, 61.0, 0.31, Level.MEDIUM),
        (79.9, 99.0, 0.90, Level.MEDIUM),  # the footprint alone falls short
        (50.0, 50.0, 0.05, Level.MEDIUM),
        (92.0, 88.0, 0.049, Level.LOW),  # agreeing shares, differing wholes
        (92.0, 49.9, 0.90, Level.LOW),
        (49.9, 92.0, 0.90, Level.LOW),
        (92.0, 88.0, float("nan"), Level.LOW),  # an undefined test
    ],
)
def test_classify_categorical(
    p_footprint: float, p_target: float, p_value: float, expected: Level
) -> None:
    """The three levels of Sect. 2.4, on and around each criterion."""
    assert classify_categorical(p_footprint, p_target, p_value) == expected


def test_classify_categorical_honours_alpha() -> None:
    """A stricter alpha lets compositions the default calls different pass."""
    assert classify_categorical(92.0, 88.0, 0.02) == Level.LOW
    assert classify_categorical(92.0, 88.0, 0.02, alpha=0.01) == Level.HIGH


def test_default_alpha_is_the_papers_five_percent() -> None:
    """Sect. 2.4 tests at p >= 0.05."""
    assert DEFAULT_ALPHA == 0.05


def test_levels_compare_and_serialise_as_strings() -> None:
    """``Level`` is a ``str`` enum, so the column goes to CSV as it reads."""
    frame = categorical_representativeness(
        disc_footprint(30.0), uniform(FOREST), GRID, GRID, radii=(90.0,)
    )

    assert frame.loc[0, "level"] == "high"
    assert (frame["level"] == Level.HIGH).all()


def test_representativeness_declines_as_the_target_area_widens() -> None:
    """
    The paper's central result: a site homogeneous within its footprint loses
    representativeness as the disc reaches into a different landscape.
    """
    # FOREST within 40 m of the tower, CROP beyond it.
    landcover = raster(
        np.where(np.hypot(XX, YY) <= 40.0, float(FOREST), float(CROP))
    )
    frame = categorical_representativeness(
        disc_footprint(20.0), landcover, GRID, GRID, radii=(FOUR_CELL_RADIUS, 90.0)
    )

    assert frame.loc[0, "level"] == Level.HIGH
    assert frame.loc[1, "level"] == Level.LOW
    assert frame.loc[1, "p_target"] < frame.loc[0, "p_target"]


# ----------------------------
# Input validation
# ----------------------------
def test_empty_radii_is_rejected() -> None:
    """Nothing to compare the footprint against."""
    with pytest.raises(ValueError, match="radii holds no target areas"):
        categorical_representativeness(
            disc_footprint(30.0), uniform(FOREST), GRID, GRID, radii=()
        )


def test_a_footprint_on_no_classified_cell_is_rejected() -> None:
    """Without a composition there is no dominant class to report."""
    landcover = raster(np.where(np.hypot(XX, YY) <= 40.0, np.nan, float(FOREST)))

    with pytest.raises(ValueError, match="no composition and no dominant class"):
        categorical_representativeness(
            disc_footprint(20.0), landcover, GRID, GRID, radii=(90.0,)
        )


def test_a_landcover_raster_off_the_grid_is_rejected() -> None:
    """Mismatched grids would be silently aligned by xarray otherwise."""
    coarse = xr.DataArray(
        np.full((10, 10), float(FOREST)),
        coords={"x": GRID[::2], "y": GRID[::2]},
        dims=("x", "y"),
    )

    with pytest.raises(ValueError, match="different grids"):
        categorical_representativeness(
            disc_footprint(30.0), coarse, GRID, GRID, radii=(90.0,)
        )


def test_a_radius_smaller_than_the_grid_is_rejected() -> None:
    """A disc holding no cell centre has no composition to compare."""
    with pytest.raises(ValueError, match="No cell centre lies within"):
        categorical_representativeness(
            disc_footprint(30.0), uniform(FOREST), GRID, GRID, radii=(1.0,)
        )


@pytest.mark.parametrize("radius", [0.0, -250.0, float("nan"), float("inf")])
def test_a_radius_that_is_not_positive_and_finite_is_rejected(radius: float) -> None:
    """Delegated to :func:`target_area_mask`, but part of this contract."""
    with pytest.raises(ValueError, match="radius must be positive and finite"):
        categorical_representativeness(
            disc_footprint(30.0), uniform(FOREST), GRID, GRID, radii=(radius,)
        )
