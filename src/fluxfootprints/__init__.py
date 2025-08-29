# footprints/__init__.py

from .ffp_xr import ffp_climatology_new  # type: ignore[import]
from .footprint_daily_summary import daily_source_area_summary
from .weighted_summaries import et_weighted_summaries

from .ffp_daily_monthly_helper import (
    load_config,
    load_amf_df,
    build_climatology,
    summarize_periods,
    export_contours_gpkg,
    export_rasters_geotiff,
)

from .tools import (
    polar_to_cartesian_dataframe,
    aggregate_to_daily_centroid,
    generate_density_raster,
    concat_fetch_gdf,
)


__version__ = "0.1.13"
