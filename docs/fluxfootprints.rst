fluxfootprints package
======================

API reference for every module in the :mod:`fluxfootprints` package.

Model interface
---------------

fluxfootprints.base\_footprint\_model module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fluxfootprints.base_footprint_model
   :members:
   :show-inheritance:
   :undoc-members:

Footprint models
----------------

fluxfootprints.improved\_ffp module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Kljun et al. (2015) FFP parameterisation.

.. automodule:: fluxfootprints.improved_ffp
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.ffp\_xr module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

xarray-based rewrite of the Kljun et al. (2015) parameterisation.

.. automodule:: fluxfootprints.ffp_xr
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.kormannmeixner module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Kormann & Meixner (2001) analytical model.

.. automodule:: fluxfootprints.kormannmeixner
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.kormannmeixner\_adapter module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fluxfootprints.kormannmeixner_adapter
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.wang\_footprint module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wang et al. (2006) convective boundary-layer analytical model.
Valid for daytime convective conditions only.

.. automodule:: fluxfootprints.wang_footprint
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.wang\_footprint\_adapter module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fluxfootprints.wang_footprint_adapter
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.ls\_footprint\_model module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Backward Lagrangian stochastic particle model.

.. automodule:: fluxfootprints.ls_footprint_model
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.ls\_footprint\_adapter module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fluxfootprints.ls_footprint_adapter
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.ep\_footprint module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One-dimensional EddyPro-style footprint estimates (Kljun 2004,
Kormann & Meixner 2001, Hsieh et al. 2000).

.. automodule:: fluxfootprints.ep_footprint
   :members:
   :show-inheritance:
   :undoc-members:

Workflow and export helpers
---------------------------

fluxfootprints.ffp\_daily\_monthly\_helper module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Config loading, climatology construction, daily/monthly summaries, and
GeoPackage / GeoTIFF / CSV export.

.. automodule:: fluxfootprints.ffp_daily_monthly_helper
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.openet\_masking module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Masking of footprint fields and exported GeoTIFFs by the valid-data masks
carried in daily OpenET rasters.

.. automodule:: fluxfootprints.openet_masking
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.by\_row\_fetch\_tools module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per-row fetch geometry, daily centroids, and density rasters.

.. automodule:: fluxfootprints.by_row_fetch_tools
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.compare module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Side-by-side model comparison and diagnostic metrics.

.. automodule:: fluxfootprints.compare
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.footprint\_plotting module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fluxfootprints.footprint_plotting
   :members:
   :show-inheritance:
   :undoc-members:

fluxfootprints.footprint\_animation module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hourly / daily / monthly videos of footprint raster time series, with
timestamp overlays and optional georeferenced basemaps.

.. automodule:: fluxfootprints.footprint_animation
   :members:
   :show-inheritance:
   :undoc-members:

Meteorological data retrieval
-----------------------------

fluxfootprints.nldas\_read\_functions module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fluxfootprints.nldas_read_functions
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: fluxfootprints
   :members:
   :show-inheritance:
   :undoc-members:
