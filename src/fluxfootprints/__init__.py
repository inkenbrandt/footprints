# footprints/__init__.py
from .base_footprint_model import BaseFootprintModel
from .improved_ffp import FFPModel
from .kormannmeixner_adapter import KormannMeixnerModel
from .ls_footprint_adapter import LSFootprintModelAdapter
from .wang_footprint_adapter import WangFootprintModel


from .ffp_xr import ffp_climatology_new  # type: ignore[import]

from .ffp_daily_monthly_helper import (
    load_config,
    load_amf_df,
    build_climatology,
    summarize_periods,
    export_contours_gpkg,
    export_rasters_geotiff,
    export_contour_stats_csv,
)

from .by_row_fetch_tools import (
    polar_to_cartesian_dataframe,
    aggregate_to_daily_centroid,
    generate_density_raster,
    concat_fetch_gdf,
)

from .improved_ffp import FFPModel

from .kormannmeixner import (  # type: ignore
    analytical_power_law_parameters,
    length_scale_xi,
    crosswind_integrated_footprint,
    footprint_2d,
    cumulative_fetch,
    effective_fetch,
    KAPPA,
)

from .ls_footprint_model import (  # noqa: E402
    KAPPA,
    LSFootprintConfig,
    BackwardLSModel,
    log_wind_profile,
    sigma_w,
    sigma_v,
    lagrangian_timescale,
)

from .wang_footprint import wang2006_fy, reconstruct_gaussian_2d

from .nldas_download_tools import (
    download_nldas,
    multiply_directories_rast,
    clip_to_utah_merge,
    mask_fp_cutoff,
    read_compiled_input,
    snap_centroid,
    load_stat_configs,
    extract_nldas_xr_to_df,
    norm_minmax_dly_et,
    norm_dly_et,
    normalize_eto_df,
    calc_nldas_refet,
    outline_valid_cells,
    fetch_and_preprocess_data,
)

__all__ = [
    # Base interface
    "BaseFootprintModel",
    
    # Model implementations
    "FFPModel",
    "KormannMeixnerModel", 
    "LSFootprintModelAdapter",
    "WangFootprintModel",
    
    # Helper functions
    "build_climatology",
    # ... rest ...
]

__version__ = "0.2.4"
