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
    compute_aerodynamic_params,
)

from .openet_masking import (
    apply_openet_mask,
    mask_footprint_dataarray,
    mask_summaries,
    mask_rasters_geotiff,
    index_openet_rasters,
    openet_mask_on_grid,
    footprint_grid_geometry,
    parse_raster_date,
    MaskedFootprint,
    GridGeometry,
)
from .openet_raster_diagnostics import mask_rasters_geotiff_with_diagnostics

from .footprint_animation import (
    FootprintAnimator,
    animate_footprint,
    resample_footprints,
    resolve_freq,
)

from .by_row_fetch_tools import (
    polar_to_cartesian_dataframe,
    aggregate_to_daily_centroid,
    generate_density_raster,
    concat_fetch_gdf,
)

from .nldas_read_functions import (
    call_nldas_time_series,
    parse_nldas_csv,
    fetch_nldas_forcing_dataset,
)

from .kormannmeixner import (  # type: ignore
    analytical_power_law_parameters,
    length_scale_xi,
    crosswind_integrated_footprint,
    footprint_2d,
    footprint_at_points,
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

from .representativeness import (
    TARGET_RADII,
    ASYMMETRY_THRESHOLD,
    BIAS_THRESHOLD,
    MIN_MATCHES,
    Level,
    ClimatologyMetrics,
    WeightedValue,
    CategoricalResult,
    ContinuousResult,
    RMAFit,
    contour_level_for_fraction,
    footprint_contour_mask,
    truncate_to_contour,
    target_area_mask,
    potential_radiation,
    partition_daynight,
    monthly_climatologies,
    footprint_fetch,
    footprint_area,
    symmetry_index,
    footprint_symmetry,
    climatology_metrics,
    overlap,
    seasonal_overlap,
    daynight_overlap,
    seasonal_overlap_index,
    daynight_overlap_index,
    footprint_weighted_value,
    footprint_weighted_composition,
    target_area_value,
    target_area_composition,
    sensor_location_bias,
    sensor_location_bias_series,
    rma_regression,
    model2_regression,
    classify_categorical,
    classify_continuous,
    categorical_representativeness,
    continuous_representativeness,
    evaluate_landcover,
    evaluate_vegetation_index,
    evaluate_representativeness,
    representativeness_summary,
    sample_raster_on_grid,
    predict_sigmav,
    export_representativeness_gpkg,
)

__all__ = [
    # Base interface
    "BaseFootprintModel",
    # Model implementations
    "FFPModel",
    "KormannMeixnerModel",
    "LSFootprintModelAdapter",
    "WangFootprintModel",
    "ffp_climatology_new",
    # Config / data loading
    "load_config",
    "load_amf_df",
    "compute_aerodynamic_params",
    # Climatology workflow
    "build_climatology",
    "summarize_periods",
    # Export helpers
    "export_contours_gpkg",
    "export_rasters_geotiff",
    "export_contour_stats_csv",
    # OpenET masking
    "apply_openet_mask",
    "mask_footprint_dataarray",
    "mask_summaries",
    "mask_rasters_geotiff",
    "mask_rasters_geotiff_with_diagnostics",
    "index_openet_rasters",
    "openet_mask_on_grid",
    "footprint_grid_geometry",
    "parse_raster_date",
    "MaskedFootprint",
    "GridGeometry",
    # Animation
    "FootprintAnimator",
    "animate_footprint",
    "resample_footprints",
    "resolve_freq",
    # Fetch / geospatial tools
    "polar_to_cartesian_dataframe",
    "aggregate_to_daily_centroid",
    "generate_density_raster",
    "concat_fetch_gdf",
    # NLDAS retrieval
    "call_nldas_time_series",
    "parse_nldas_csv",
    "fetch_nldas_forcing_dataset",
    # Kormann-Meixner analytical model
    "analytical_power_law_parameters",
    "length_scale_xi",
    "crosswind_integrated_footprint",
    "footprint_2d",
    "footprint_at_points",
    "cumulative_fetch",
    "effective_fetch",
    # Lagrangian stochastic model
    "LSFootprintConfig",
    "BackwardLSModel",
    "log_wind_profile",
    "sigma_w",
    "sigma_v",
    "lagrangian_timescale",
    # Wang & Davis analytical model
    "wang2006_fy",
    "reconstruct_gaussian_2d",
    # Representativeness (Chu et al., 2021)
    "TARGET_RADII",
    "ASYMMETRY_THRESHOLD",
    "BIAS_THRESHOLD",
    "MIN_MATCHES",
    "Level",
    "ClimatologyMetrics",
    "WeightedValue",
    "CategoricalResult",
    "ContinuousResult",
    "RMAFit",
    "contour_level_for_fraction",
    "footprint_contour_mask",
    "truncate_to_contour",
    "target_area_mask",
    "potential_radiation",
    "partition_daynight",
    "monthly_climatologies",
    "footprint_fetch",
    "footprint_area",
    "symmetry_index",
    "footprint_symmetry",
    "climatology_metrics",
    "overlap",
    "seasonal_overlap",
    "daynight_overlap",
    "seasonal_overlap_index",
    "daynight_overlap_index",
    "footprint_weighted_value",
    "footprint_weighted_composition",
    "target_area_value",
    "target_area_composition",
    "sensor_location_bias",
    "sensor_location_bias_series",
    "rma_regression",
    "model2_regression",
    "classify_categorical",
    "classify_continuous",
    "categorical_representativeness",
    "continuous_representativeness",
    "evaluate_landcover",
    "evaluate_vegetation_index",
    "evaluate_representativeness",
    "representativeness_summary",
    "sample_raster_on_grid",
    "predict_sigmav",
    "export_representativeness_gpkg",
    # Constants
    "KAPPA",
    "__version__",
]

__version__ = "0.4.0"
