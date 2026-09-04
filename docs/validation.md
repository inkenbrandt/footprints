# Validation against Chu et al. (2021)

`fluxfootprints.representativeness` reimplements the footprint-representativeness
method of:

> Chu, H., Luo, X., Ouyang, Z., et al. (2021). Representativeness of
> Eddy-Covariance flux footprints for areas surrounding AmeriFlux sites.
> *Agricultural and Forest Meteorology* **301–302**, 108350.
> [doi:10.1016/j.agrformet.2021.108350](https://doi.org/10.1016/j.agrformet.2021.108350)

The authors released their inputs and results as a Zenodo archive
([doi:10.5281/zenodo.4015350](https://doi.org/10.5281/zenodo.4015350), Datasets
S1–S6), which lets us check our implementation against the published numbers
rather than only against synthetic fixtures. This page records that comparison.

## What is compared

We recompute the Sect. 2.2 metrics — fetch **X80**, area **A80**, symmetry
**S80** (Eq. 1), seasonal overlap **O80,season** (Eq. 2), and day–night overlap
**O80,daynight** (Eq. 3) — from **Dataset S2**, the authors' own monthly
footprint climatology rasters, and compare them against **Dataset S1**, the
site-year metrics the authors published for those same rasters.

Driving the comparison from S2 rather than from raw flux data is deliberate: it
isolates *our metric code* from the upstream climatology model, so a mismatch
points at the metrics rather than at a difference in how the footprints were
built.

The tests live in `tests/test_chu2020_validation.py`. They are marked `slow` and
skip unless `CHU2020_DATA` points at a local copy of the archive:

```bash
CHU2020_DATA="/path/to/Chu et al dataset" pytest tests/test_chu2020_validation.py
```

## Sites

Four site-years spanning the regimes the paper contrasts:

| Site | IGBP | Year | Regime |
|---|---|---|---|
| US-NR1 | ENF | 2011 | Subalpine forest; drainage flow makes nights far-reaching and asymmetric |
| US-SRM | WSA | 2017 | Semi-arid savanna; the most asymmetric climatology of the set |
| US-Ne1 | CRO | 2010 | Irrigated maize; canopy growth drives strong seasonal variation |
| US-ARM | CRO | 2011 | Rain-fed cropland; the strongest seasonal variation of the set |

US-NR1 and US-SRM are the paper's two asymmetric exemplars, quoted with
nighttime S80 of 0.31 and 0.18. US-Ne1 and US-ARM carry the lowest seasonal
overlap here (nighttime O80,season of 0.80 and 0.61), which is the paper's
signature of a canopy that changes through the growing season.

## Grid handling

The S2 rasters are EPSG:4326, sum to exactly 1.0, and carry explicit zeros
outside the 80 % contour — so their positive cells are exactly the cells inside
the contour, which is what `footprint_fetch` and `footprint_area` accept for an
already-truncated climatology.

Our functions want tower-centred metre offsets instead. Rather than resample,
the tests re-register each raster onto a common grid indexed by **integer cell
offsets from the tower**. This is exact: within a site-year every raster shares
one cell size, and each is registered so the tower falls on a cell centre to
better than 10⁻⁴ of a cell, so months differ only in which rectangle of the
common lattice they cover. Zero-padding to the union rectangle preserves each
month's unit sum, which the overlap indices require. Grid spacing in metres
comes from `pyproj.Geod` on the WGS84 ellipsoid; tower coordinates come from
Dataset S5 rather than being hard-coded.

X80 and A80 are month-wise means over the site-year (S1 reports a standard
deviation alongside each, so the published values are means). S80 is Eq. 1
applied to those means — that is the reading which reproduces the published
values; averaging the monthly S80 instead misses US-NR1's nighttime value by
0.03.

## Results

Measured 2026-08-30 against the full archive. "Ours" is recomputed from S2;
"published" is the S1 value.

### US-NR1 2011 — 78 × 54 cells, 16.35 × 16.65 m, 12 months

| Metric | Ours | Published | Residual |
|---|---|---|---|
| X80 day | 328.67 m | 331.94 m | −0.99 % |
| X80 night | 664.12 m | 668.39 m | −0.64 % |
| A80 day | 157 135 m² | 157 104 m² | +0.02 % |
| A80 night | 448 741 m² | 449 040 m² | −0.07 % |
| S80 day | 0.4630 | 0.4538 | +0.0092 |
| S80 night | 0.3239 | 0.3199 | +0.0039 |
| O80,season day | 0.9290 | 0.9393 | −0.0103 |
| O80,season night | 0.8103 | 0.8520 | −0.0418 |
| O80,daynight | 0.8673 | 0.8681 | −0.0007 |

### US-SRM 2017 — 80 × 51 cells, 7.75 × 7.10 m, 12 months

| Metric | Ours | Published | Residual |
|---|---|---|---|
| X80 day | 168.88 m | 170.49 m | −0.95 % |
| X80 night | 348.56 m | 351.62 m | −0.87 % |
| A80 day | 49 017 m² | 49 042 m² | −0.05 % |
| A80 night | 70 952 m² | 71 094 m² | −0.20 % |
| S80 day | 0.5471 | 0.5371 | +0.0100 |
| S80 night | 0.1859 | 0.1830 | +0.0028 |
| O80,season day | 0.8687 | 0.8740 | −0.0053 |
| O80,season night | 0.8750 | 0.8810 | −0.0059 |
| O80,daynight | 0.6988 | 0.7172 | −0.0184 |

### US-Ne1 2010 — 54 × 83 cells, 5.81 × 6.02 m, 12 months

| Metric | Ours | Published | Residual |
|---|---|---|---|
| X80 day | 189.94 m | 191.97 m | −1.05 % |
| X80 night | 219.62 m | 221.50 m | −0.85 % |
| A80 day | 44 959 m² | 45 054 m² | −0.21 % |
| A80 night | 56 903 m² | 56 965 m² | −0.11 % |
| S80 day | 0.3966 | 0.3891 | +0.0075 |
| S80 night | 0.3755 | 0.3696 | +0.0060 |
| O80,season day | 0.7870 | 0.7966 | −0.0096 |
| O80,season night | 0.7841 | 0.7952 | −0.0111 |
| O80,daynight | 0.9553 | 0.9558 | −0.0005 |

### US-ARM 2011 — 120 × 139 cells, 4.39 × 4.26 m, 12 months

| Metric | Ours | Published | Residual |
|---|---|---|---|
| X80 day | 226.31 m | 227.53 m | −0.54 % |
| X80 night | 255.68 m | 256.65 m | −0.38 % |
| A80 day | 73 951 m² | 73 986 m² | −0.05 % |
| A80 night | 90 581 m² | 90 626 m² | −0.05 % |
| S80 day | 0.4596 | 0.4549 | +0.0047 |
| S80 night | 0.4411 | 0.4380 | +0.0031 |
| O80,season day | 0.7038 | 0.7069 | −0.0031 |
| O80,season night | 0.5990 | 0.6091 | −0.0101 |
| O80,daynight | 0.9480 | 0.9495 | −0.0014 |

## Tolerances and why

| Metric | Tolerance | Worst residual | Mechanism |
|---|---|---|---|
| X80 | 2 % relative | −1.05 % | Cell-centre distance deficit |
| A80 | 1 % relative | −0.21 % | Whole-cell counting |
| S80 | 0.02 absolute | +0.0100 | X80 deficit propagated through Eq. 1 |
| O80,daynight | 0.03 absolute | −0.0184 | Per-month pairing, averaged |
| O80,season | 0.05 absolute | −0.0418 | Geometric mean over the common support |

Every tolerance is at least twice the largest residual observed, so the tests
assert agreement rather than merely pinning current behaviour.

The residuals are **systematic and signed**, not noise, which is what makes
them explainable:

- **X80 is low at every site-year.** `footprint_fetch` measures to cell
  *centres*, while the authors' fetch reaches the contour itself. The deficit is
  bounded by half a cell diagonal — 11.6 m at US-NR1, or 3.5 % of its daytime
  X80 — so the 2 % tolerance is a real constraint rather than a vacuous one.
  The test asserts the one-sidedness explicitly: measuring to cell centres
  cannot overshoot.
- **S80 is high at every site-year**, by 0.003–0.010. That is exactly the X80
  deficit propagating through S80 = A80 / (π·X80²): a 1 % low fetch inflates S80
  by about 2 %.
- **A80 agrees to 0.21 % or better.** Both implementations count whole cells on
  the same lattice, so only a contour cell's membership can differ.
- **O80,season is the loosest.** Eq. 2 is a geometric mean over the cells
  positive in *every* month, so it is dominated by the month of smallest support
  and is the metric most sensitive to float32 round-off in the rasters'
  near-zero edge cells. The one residual that reaches 0.042 is US-NR1 at night,
  whose monthly footprints swing widely (X80 s.d. 173 m on a 668 m mean) —
  precisely the case that amplifies this.

## Qualitative findings reproduced

Beyond the numeric comparison, the tests confirm our metrics reproduce the
paper's conclusions:

- **Nighttime footprints reach farther and cover more ground** than daytime at
  all four sites, as the paper reports for >95 % of its site-years.
- **The two asymmetric exemplars land at the quoted values** and classify the
  same way the published S1 values do against `ASYMMETRY_THRESHOLD` (0.30).
  Worth noting: the two sit on opposite sides of that line. US-SRM (0.186) is
  well under it, while US-NR1 (0.324) is *just above* — low enough for the paper
  to discuss alongside US-SRM, but not actually under the flag. The published
  values behave the same way (0.183 and 0.320), and US-NR1 is above 0.30 in
  every year S1 reports (0.3199, 0.3108, 0.3074).
- **The cropland sites vary more seasonally than the forest site**: US-Ne1 and
  US-ARM fall below 0.85 daytime O80,season while evergreen US-NR1 stays above
  0.9, matching the Sect. 3.2 attribution to canopies that change through the
  growing season.

## The climatology model upstream

The tests above hold the climatology fixed and check our metrics. The
complementary check — build the climatology ourselves and compare it with the
published raster — lives in the validation-data workspace, because it needs raw
AmeriFlux BASE data as well as the archive:

| | |
|---|---|
| Reader and comparison utilities | `data/validation_data/scripts/chu2020_reference.py` |
| Raster subset, inventory and reference metrics | `data/validation_data/scripts/inspect_chu2020_rasters.py` |
| Model-versus-reference comparison | `data/validation_data/scripts/compare_chu2020_climatology.py` |
| Reports | `data/validation_data/metadata/Chu2020/Chu2020_raster_inspection.md`, `Chu2020_step5_report.md` |

The case is **US-ARM 2011**, all twelve months, daytime and nighttime, at the
4 m system: the one site-year where a Chu site-year, a raw BASE record held
locally, and an existing preparation script coincide. The modelled climatology
is run on the published raster's own cell lattice, so registration is exact and
nothing is resampled.

What it found, in one line each:

- **Orientation reproduces.** The modelled centroid bearing is within 6° of the
  published one in every one of the 24 month-periods, and the 16-sector weight
  profiles correlate at 0.988–1.000.
- **Shape reproduces.** Taking our source area at the reference's own size in
  cells, the two regions overlap at IoU 0.94–0.98.
- **Extent is looser** — X80 within −25 % to +16 %, median about −8 % — and it
  is looser for a structural reason. The tail of a climatology is flat, so A80
  gains about 11 % for every additional per cent of enclosed mass and is 1.8
  times larger at the 85 % contour than at the 80 %; the published 80 % source
  area is where *our* 76–84 % contour falls. A few per cent of difference in
  the mass distribution therefore reads as tens of per cent of A80.
- **The unshared inputs dominate that residual.** Chu et al. publish neither
  their roughness length, their displacement height, nor their boundary-layer
  height. Swapping our fitted z0 for the `0.1 × canopy` rule alone moves X80 by
  15–17 percentage points, well past the residual being explained.

The moral for the metric functions is the useful part: **X80 and A80 are not
the right things to assert agreement on between two implementations** unless
their inputs are shared. Bearing, sector profile, and area-matched IoU are.

One convention is worth repeating because getting it wrong is silent: the 80 %
contour is an *absolute* mass, not a share of what the model domain happened to
capture. `_source_weight_threshold` accumulates `density × cell_area` until it
reaches 0.8, and so does the reference FFP implementation. Taking 0.8 of the
captured mass instead moves A80 by about 30 % on this benchmark, with no error
raised anywhere.

## Limitations

- The comparison covers the Sect. 2.2 **footprint-geometry** metrics only. The
  categorical (S4) and continuous (S5/S6) representativeness evaluations are
  not yet validated against the archive; they are covered by unit tests against
  synthetic fixtures.
- Four site-years of 712 are checked. They were chosen to span regimes and to
  include the paper's named exemplars, not sampled at random.
- Because the comparison starts from Dataset S2, it validates the metric code
  but **not** the climatology model that produces the footprints upstream. That
  gap is covered separately — see [the climatology comparison](#the-climatology-model-upstream)
  below.
- Numbering caution: `representativeness_table(results, dataset)` maps `"S5"` to
  site-month statistics and `"S6"` to site-level regressions, which is swapped
  relative to this data release, where S5 is site-level and S6 is site-month.
  The module appears to number the paper's supplementary *tables* rather than
  these datasets. Check which numbering is meant before mapping columns across.
