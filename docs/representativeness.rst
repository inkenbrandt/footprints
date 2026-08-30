Footprint representativeness
============================

A footprint climatology answers *where the flux came from*. Synthesis work
usually needs the complementary answer: how well that source area stands in for
the **target area** it is being used to represent — the model grid cell, the
remote-sensing pixel window, or the fixed-radius disc around the tower that a
network product assigns to the site.

:mod:`fluxfootprints.representativeness` implements that comparison following

    Chu, H., Luo, X., Ouyang, Z., et al. (2021). Representativeness of
    Eddy-Covariance flux footprints for areas surrounding AmeriFlux sites.
    *Agricultural and Forest Meteorology*, **301–302**, 108350.
    `doi:10.1016/j.agrformet.2021.108350
    <https://doi.org/10.1016/j.agrformet.2021.108350>`__

The module reproduces the paper's method end to end: monthly day/night
climatologies truncated at the 80 % source-weight contour, the geometry metrics
of Sect. 2.2, and the categorical and continuous evaluations of Sect. 2.4, each
reduced to a three-level ``HIGH`` / ``MEDIUM`` / ``LOW`` index.

.. seealso::

   :doc:`validation` records how these metrics compare against the authors'
   own published values, recomputed from their Zenodo archive.

   :doc:`notebooks/representativeness_example` runs the whole workflow on the
   bundled US-CRT tower record.

Grid conventions
----------------

Everything here works on the tower-centred grid the rest of the package uses:
``x`` and ``y`` are cell-centre offsets in metres from the tower, which sits at
the origin. Raw footprint weights are densities [m\ :sup:`-2`] and must be
multiplied by the cell area to give a source fraction; the *truncated*
climatologies the analysis functions consume are already renormalised to sum to
one over the retained cells.

The target area is a disc of radius :math:`r` about the tower. Chu et al. (2021)
evaluate six of them, which is what :data:`~fluxfootprints.TARGET_RADII` holds:

.. math:: r \in \{250,\ 500,\ 1000,\ 1500,\ 2000,\ 3000\}\ \mathrm{m}

The seven equations
-------------------

.. list-table::
   :header-rows: 1
   :widths: 8 34 58

   * - Eq.
     - Quantity
     - Implemented by
   * - 1
     - Symmetry index :math:`S_{80}`
     - :func:`~fluxfootprints.symmetry_index`,
       :func:`~fluxfootprints.footprint_symmetry`
   * - 2
     - Seasonal overlap :math:`O_{80,\mathrm{season}}`
     - :func:`~fluxfootprints.seasonal_overlap`
   * - 3
     - Day–night overlap :math:`O_{80,\mathrm{daynight}}`
     - :func:`~fluxfootprints.daynight_overlap`
   * - 4
     - Enhanced vegetation index
     - :func:`~fluxfootprints.fetch_landsat_evi`
   * - 5
     - Footprint-weighted value :math:`EVI_\mathrm{footprint}`
     - :func:`~fluxfootprints.footprint_weighted_value`
   * - 6
     - Sensor location bias :math:`\Delta`
     - :func:`~fluxfootprints.sensor_location_bias`
   * - 7
     - Site-level model II regression
     - :func:`~fluxfootprints.rma_regression`,
       :func:`~fluxfootprints.model2_regression`

Climatology metrics (Sect. 2.2)
-------------------------------

Truncation and the day–night split
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every half-hourly footprint of a calendar month is aggregated into a **daytime**
and a **nighttime** climatology, each truncated at the contour enclosing 80 % of
the source weight and rescaled so the retained cells sum to one. Day and night
are separated at :math:`SW_{IN,POT} > 0`, the potential incoming shortwave
radiation of a horizontal surface:

.. math::

   SW_{IN,POT} = \frac{S_0}{R^2}\,\max\!\left(\cos\theta_z,\ 0\right)

with :math:`S_0` the solar constant, :math:`R` the Earth–Sun distance in
astronomical units, and :math:`\theta_z` the solar zenith angle.
:func:`~fluxfootprints.partition_daynight` prefers a precomputed ``SW_IN_POT``
column when the record carries one — AmeriFlux BASE files do — and falls back to
solar geometry from :func:`~fluxfootprints.potential_radiation` otherwise.

:func:`~fluxfootprints.monthly_climatologies` does the aggregation and returns a
Dataset with dims ``(month, period, x, y)``.

Fetch, area, and symmetry (Eq. 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**X80** is the greatest distance from the tower to the truncation contour, and
**A80** the area the contour encloses. Their ratio against the circle of radius
X80 is the symmetry index:

.. math::

   S_{80} = \frac{A_{80}}{\pi X_{80}^{2}}

:math:`S_{80}` runs from 0 to 1. A value of 1 is a perfectly circular
climatology centred on the tower; below
:data:`~fluxfootprints.ASYMMETRY_THRESHOLD` (0.30) the paper calls the
climatology *relatively asymmetric* — a site whose wind rose concentrates the
source area into one or two lobes, so that a disc drawn around the tower holds a
great deal of ground the tower never sees.

Seasonal overlap (Eq. 2)
^^^^^^^^^^^^^^^^^^^^^^^^

How far the source area migrates through a site-year is measured by the
cell-wise geometric mean of the :math:`K` monthly climatologies, summed over the
:math:`I` grid cells:

.. math::

   O_{80,\mathrm{season}}
   = \sum_{i=1}^{I} \left( \prod_{k=1}^{K} \varphi_{ik} \right)^{1/K}

.. admonition:: Correction to the printed equation
   :class: important

   Equation 2 as printed in Chu et al. (2021) carries the exponent
   :math:`1/k` — the *bound* index of the product — rather than :math:`1/K`.
   That is a typo: :math:`k` has no value outside the product it is bound to,
   and only the :math:`1/K` reading returns 1.0 when every month is identical,
   which is the defining property of an overlap index.
   :func:`~fluxfootprints.seasonal_overlap` therefore implements :math:`1/K`, a
   true geometric mean over all :math:`K` months, and the :doc:`validation`
   page shows that this reading reproduces the authors' published values.

The geometric mean is evaluated as ``exp(mean(log(w)))`` over the cells that
*every* month covers and taken as zero elsewhere, so a zero in any one month
propagates to a zero cell rather than a NaN. The index consequently measures the
source area common to all months, and a single month pointing elsewhere is
enough to drive it towards zero. The paper reads values below 0.8 as
"noticeable monthly variability", which it found in 32–44 % of site-years,
concentrated in the cropland, grassland, and wetland sites whose canopy height
swings through the growing season.

Day–night overlap (Eq. 3)
^^^^^^^^^^^^^^^^^^^^^^^^^

The daytime and nighttime climatologies of each month are compared pairwise and
the result averaged over the :math:`K` months:

.. math::

   O_{80,\mathrm{daynight}}
   = \frac{1}{K} \sum_{k=1}^{K} \sum_{i=1}^{I}
     \left( \varphi^{\mathrm{day}}_{ik}\, \varphi^{\mathrm{night}}_{ik}
     \right)^{1/2}

The inner sum is the Bhattacharyya coefficient of the two weight fields read as
discrete distributions — :func:`~fluxfootprints.overlap`, which is also Eq. 2
for the special case :math:`K = 2`. Unlike Eq. 2, this index pairs month by
month *before* averaging, so a site whose day and night footprints migrate
together through the season is not penalised for the migration.

The surface fields
------------------

The evaluations of Sect. 2.4 compare a climatology against two external products
sampled onto the same grid: a categorical land-cover map, and a continuous
vegetation index.

Enhanced vegetation index (Eq. 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   EVI = 2.5\,\frac{\rho_\mathrm{NIR} - \rho_\mathrm{RED}}
                   {\rho_\mathrm{NIR} + 6\,\rho_\mathrm{RED}
                    - 7.5\,\rho_\mathrm{BLUE} + 1}

computed on surface *reflectance* — not on stored digital numbers, or the
``+ 1`` in the denominator would be negligible against them and the index
meaningless. :func:`~fluxfootprints.fetch_landsat_evi` pulls Landsat 5/TM and
8/OLI Collection 2 Level-2 scenes from Earth Engine, computes Eq. 4, and keeps
only the scenes that see the target disc essentially clear (< 1 % obscured, the
paper's criterion). :func:`~fluxfootprints.fetch_nlcd` fetches the matching
land-cover tile. Both are optional: any raster on disc can be warped onto the
footprint grid with :func:`~fluxfootprints.sample_raster_on_grid` instead.

Categorical evaluation (Sect. 2.4)
----------------------------------

Each land-cover class takes the share of footprint weight that falls on it,
:math:`P_\mathrm{footprint}`; the same class takes an unweighted share of the
target disc, :math:`P_\mathrm{target}`. The two compositions are compared with a
chi-square test, and the site is classified on the **dominant** class — the one
holding the largest footprint-weighted share.

.. list-table:: Land-cover representativeness index
   :header-rows: 1
   :widths: 14 86

   * - Level
     - Criteria
   * - ``HIGH``
     - :math:`P_\mathrm{footprint} \ge 80\,\%` **and**
       :math:`P_\mathrm{target} \ge 80\,\%`, **and** the compositions do not
       differ significantly (:math:`p \ge \alpha`, default
       :math:`\alpha = 0.05`)
   * - ``MEDIUM``
     - :math:`P_\mathrm{footprint} \ge 50\,\%` **and**
       :math:`P_\mathrm{target} \ge 50\,\%`, **and** :math:`p \ge \alpha`, but
       the ``HIGH`` criteria are not met
   * - ``LOW``
     - Otherwise — no class reaches 50 % in the footprint or in the target area,
       or the compositions differ significantly

The 50 % and 80 % thresholds follow Göckede et al. (2008).
:func:`~fluxfootprints.classify_categorical` applies them, and
:func:`~fluxfootprints.evaluate_landcover` runs the whole comparison across a
series of radii.

.. note::

   A chi-square test needs counts, but a footprint-weighted composition is a
   continuous share of source weight with no natural sample size. Both
   compositions are therefore scaled to **pseudo-counts** by the same sample
   size — the number of classified cells inside the target area. That puts the
   two on one footing, but it also fixes the power of the test by the grid: a
   finer grid raises the cell count, hence the counts, hence the statistic, for
   compositions that have not changed. Read the p-value as a comparison against
   the paper's fixed 30 m Landsat grid, not as an absolute significance.

Continuous evaluation (Sect. 2.4)
---------------------------------

Footprint-weighted value (Eq. 5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   EVI_\mathrm{footprint} = \sum_{j=1}^{J} \varphi_j\, EVI_j

the vegetation index averaged under the truncated, renormalised footprint
weights. Its counterpart :math:`EVI_\mathrm{target}` is the plain, unweighted
mean of the same raster over the target disc
(:func:`~fluxfootprints.target_area_value`).

Sensor location bias (Eq. 6)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \Delta = \frac{EVI_\mathrm{footprint} - EVI_\mathrm{target}}
                 {EVI_\mathrm{target}}

the relative difference between what the tower saw and what the disc contains,
after Schmid and Lloyd (1999). It is evaluated once per matched scene and target
radius by :func:`~fluxfootprints.sensor_location_bias_series`. Following Chen et
al. (2011) and Kim et al. (2006), Chu et al. (2021) count a period as
representative when :math:`|\Delta| \le 10\,\%`, which is
:data:`~fluxfootprints.BIAS_THRESHOLD`.

Site-level regression (Eq. 7)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pooling every matched scene of a site gives a regression of the target-area mean
on the footprint-weighted value:

.. math::

   EVI_\mathrm{target} \sim \beta_0 + \beta_1\, EVI_\mathrm{footprint}

Both variables are spatial averages of one noisy raster and both carry error, so
this is a **model II** fit — the reduced major axis, equivalently the standard
major axis, the ``"SMA"`` row of R's ``lmodel2``, which the paper used. Its
slope is the ordinary least-squares slope divided by the Pearson correlation
coefficient, :math:`\mathrm{sign}(r)\,s_y / s_x`; an OLS slope would sit
systematically shallower than the values the paper's Table 1 reports. At least
:data:`~fluxfootprints.MIN_MATCHES` (6) matched scenes are required before a
site-level fit is attempted — 166 of the paper's 214 sites cleared that bar.

.. list-table:: Continuous-field representativeness index
   :header-rows: 1
   :widths: 14 86

   * - Level
     - Criteria
   * - ``HIGH``
     - :math:`R^2 \ge 0.8` **and** :math:`0.9 \le \beta_1 \le 1.1` **and**
       :math:`-0.1 \le \beta_0 \le 0.1`
   * - ``MEDIUM``
     - :math:`R^2 \ge 0.6` **and** :math:`p < \alpha` (default
       :math:`\alpha = 0.05`), but the ``HIGH`` criteria are not met
   * - ``LOW``
     - Otherwise — :math:`R^2 < 0.6`, or :math:`p \ge \alpha`

.. warning::

   The intercept tolerance is *absolute* and calibrated to EVI's 0–1 range. A
   field on a different scale — land surface temperature in kelvin, say — needs
   a rescaled criterion before :func:`~fluxfootprints.classify_continuous` means
   anything.

Thresholds at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 36 14 50

   * - Constant
     - Value
     - Meaning
   * - :data:`~fluxfootprints.TARGET_RADII`
     - 250 … 3000 m
     - Target-area radii evaluated (Sect. 2.1)
   * - ``DEFAULT_CONTOUR_FRACTION``
     - 0.8
     - Source-weight fraction each climatology is truncated at
   * - :data:`~fluxfootprints.ASYMMETRY_THRESHOLD`
     - 0.30
     - :math:`S_{80}` below which a climatology is called asymmetric
   * - ``DEFAULT_ALPHA``
     - 0.05
     - Significance level for the chi-square test and the regression
   * - :data:`~fluxfootprints.BIAS_THRESHOLD`
     - 0.10
     - :math:`|\Delta|` at or below which a period counts as representative
   * - :data:`~fluxfootprints.MIN_MATCHES`
     - 6
     - Fewest matched scenes a site-level regression may be fitted on

Running the analysis
--------------------

:func:`~fluxfootprints.assess_representativeness` drives everything above and
returns one tidy frame::

    from fluxfootprints import assess_representativeness, build_climatology

    model = build_climatology(df, model_type="ffp", ...)

    results = assess_representativeness(
        model,
        station_lat=41.6285,
        station_lon=-83.3471,
        site_id="US-CRT",
        landcover="nlcd_2016.tif",
        continuous={"2011-07-04": "evi_20110704.tif"},
        tz=-5,
    )

The frame is indexed by ``(site, year, month, period, radius, variable)`` and
carries rows of three scopes, told apart by the ``scope`` and ``kind`` columns:

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - ``scope``
     - ``kind``
     - Rows
   * - ``period``
     - ``climatology``
     - One per aggregated month and period: ``fetch``, ``area``, ``symmetry``,
       ``contour_level``, ``n_cells``, ``n_times``
   * - ``period``
     - ``categorical``
     - One per month, period, and radius: ``dominant_class``,
       ``value_footprint``, ``value_target``, ``chi2``, ``dof``, ``p_value``,
       ``level``
   * - ``period``
     - ``continuous``
     - One per matched scene, period, and radius: ``value_footprint`` (Eq. 5),
       ``value_target``, ``bias`` (Eq. 6), ``within_threshold`` — the paper's
       Dataset S5
   * - ``site_year``
     - ``climatology``
     - One per year and period: ``fetch``, ``area``, and ``symmetry`` averaged
       over that year's months, plus ``seasonal_overlap`` (Eq. 2) and
       ``daynight_overlap`` (Eq. 3)
   * - ``site``
     - ``categorical``
     - One per period and radius, over every month pooled — the paper's Dataset
       S4, and the level reported in its Fig. 5
   * - ``site``
     - ``continuous``
     - One per period and radius, holding the Eq. 7 regression: ``slope``,
       ``intercept``, their confidence limits, ``r_squared``, ``p_value``,
       ``rmse``, ``mae``, ``n``, ``sufficient``, ``level`` — the paper's Dataset
       S6 and Table 1

Columns an analysis does not produce are missing, so the frame is sparse by
construction; select a slice with ``scope`` and ``kind`` before reading values
off it. :func:`~fluxfootprints.representativeness_table` reshapes those slices
into the published column layouts of Datasets S4–S6, and
:func:`~fluxfootprints.export_representativeness_tables` writes them out.

Figures
-------

:mod:`fluxfootprints.representativeness_plotting` draws the four diagnostics in
the form the paper publishes them. Each takes what the analysis functions
already return and hands back ``(fig, ax)`` without showing or saving anything,
so the caller keeps control of the title, the layout, and the output.

.. list-table::
   :header-rows: 1
   :widths: 44 14 42

   * - Function
     - Figure
     - Shows
   * - :func:`~fluxfootprints.plot_landcover_composition`
     - Fig. 1e
     - Footprint-weighted against target-area share, class by class
   * - :func:`~fluxfootprints.plot_footprint_target_scatter`
     - Figs. 1f, 6
     - Matched Eq. 5 pairs under the Eq. 7 fit and the 1:1 line
   * - :func:`~fluxfootprints.plot_bias_density`
     - Fig. 7
     - Kernel densities of the Eq. 6 bias, one per radius
   * - :func:`~fluxfootprints.plot_level_bars`
     - Figs. 5, 8
     - The three-level index stacked across target areas

Target-area radius is an *ordered* quantity, so it is drawn on a single-hue
sequential ramp, dark at the smallest radius and light at the largest — as the
paper's own legends read, "from dark to light, indicating an increasing distance
from the tower". The footprint-weighted series is not one step of that sequence
but the reference the target areas are read against, so it wears a single
contrasting accent.

API reference
-------------

fluxfootprints.representativeness module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Climatology metrics, the categorical and continuous evaluations, and the
:func:`~fluxfootprints.assess_representativeness` driver.

.. automodule:: fluxfootprints.representativeness
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.representativeness\_data module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Google Earth Engine retrieval of the NLCD land-cover tile and the Landsat EVI
scenes the evaluations consume. Needs the ``gee`` extra
(``pip install 'fluxfootprints[gee]'``) and an authenticated Earth Engine
project; nothing else in the package requires either.

.. automodule:: fluxfootprints.representativeness_data
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.representativeness\_plotting module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The four diagnostic figures.

.. automodule:: fluxfootprints.representativeness_plotting
   :members:
   :show-inheritance:
   :undoc-members:

References
----------

Chu, H., Luo, X., Ouyang, Z., et al. (2021). Representativeness of
Eddy-Covariance flux footprints for areas surrounding AmeriFlux sites.
*Agricultural and Forest Meteorology*, **301–302**, 108350.
`doi:10.1016/j.agrformet.2021.108350
<https://doi.org/10.1016/j.agrformet.2021.108350>`__

Göckede, M., Foken, T., Aubinet, M., et al. (2008). Quality control of
CarboEurope flux data — Part 1: Coupling footprint analyses with flux data
quality assessment to evaluate sites in forest ecosystems. *Biogeosciences*,
**5**\ (2), 433–450. `doi:10.5194/bg-5-433-2008
<https://doi.org/10.5194/bg-5-433-2008>`__

Schmid, H. P., & Lloyd, C. R. (1999). Spatial representativeness and the
location bias of flux footprints over inhomogeneous areas. *Agricultural and
Forest Meteorology*, **93**\ (3), 195–209.
`doi:10.1016/S0168-1923(98)00119-1
<https://doi.org/10.1016/S0168-1923(98)00119-1>`__
