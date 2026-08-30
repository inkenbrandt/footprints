fluxfootprints documentation
============================

**fluxfootprints** is a Python package for micrometeorological flux-footprint
analysis. It provides several footprint models behind a single, consistent
interface, together with tooling to build footprint climatologies from
eddy-covariance tower data, summarise them by day and month, and export the
results as GeoPackages, GeoTIFFs, and CSV statistics.

Available models
----------------

All models subclass :class:`~fluxfootprints.BaseFootprintModel` and share the
same ``run()`` / ``get_footprint_climatology()`` API:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Class
     - Type
     - Reference
   * - :class:`~fluxfootprints.FFPModel`
     - Parameterisation
     - Kljun et al. (2015)
   * - :class:`~fluxfootprints.ffp_climatology_new`
     - Parameterisation
     - Kljun et al. (2015), xarray implementation
   * - :class:`~fluxfootprints.KormannMeixnerModel`
     - Analytical
     - Kormann & Meixner (2001)
   * - :class:`~fluxfootprints.WangFootprintModel`
     - Analytical (convective BL only)
     - Wang et al. (2006)
   * - :class:`~fluxfootprints.LSFootprintModelAdapter`
     - Lagrangian stochastic
     - Backward particle model

.. toctree::
   :maxdepth: 2
   :caption: Background

   exp
   modeltypes
   validation

.. toctree::
   :maxdepth: 2
   :caption: Representativeness

   representativeness
   Worked example <notebooks/representativeness_example.ipynb>

.. toctree::
   :maxdepth: 2
   :caption: API reference

   fluxfootprints
   modules

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   Kljun FFP quick-start <notebooks/ffp_getting_started.ipynb>
   Kljun FFP (xarray) quick-start <notebooks/ffp_xr_getting_started.ipynb>
   Package-level climatology calculation <notebooks/footprint_package_calc.ipynb>
   End-to-end workflow example <notebooks/footprint_workflow_example.ipynb>
   Lagrangian stochastic model <notebooks/Getting_Started_LS_Footprint_Model.ipynb>
   Wang footprint model <notebooks/Getting_Started_Wang_Footprint.ipynb>
   Comparing footprint models <notebooks/footprint_model_comparison.ipynb>
   Animating footprint time series <notebooks/footprint_animation_example.ipynb>

.. toctree::
   :maxdepth: 1
   :caption: Data and post-processing

   NLDAS download and ET <notebooks/nldas_et_download_example.ipynb>

.. note::
   ``docs/notebooks/multiply_rasters.ipynb`` and
   ``docs/notebooks/openet_comparison.ipynb`` are not listed above: they are
   built on the separate ``micromet`` package rather than on
   :mod:`fluxfootprints`, so they are excluded from this documentation.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
