import pandas as pd
import numpy as np

import geopandas as gpd

from rasterio.transform import from_origin
from scipy.stats import gaussian_kde


def polar_to_cartesian_dataframe(df, wd_column="WD", dist_column="Dist"):
    """
    Convert polar coordinates from a DataFrame to Cartesian coordinates.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing polar coordinates.
        wd_column (str): Column name for degrees from north.
        dist_column (str): Column name for distance from origin.

    Returns:
        pd.DataFrame: A DataFrame with added 'X' and 'Y' columns.
    """
    # Create copies of the input columns to avoid modifying original data
    wd = df[wd_column].copy()
    dist = df[dist_column].copy()

    # Identify invalid values (-9999 or NaN)
    invalid_mask = (wd == -9999) | (dist == -9999) | wd.isna() | dist.isna()

    # Convert degrees from north to standard polar angle (radians) where valid
    theta_radians = np.radians(90 - wd)

    # Calculate Cartesian coordinates, setting invalid values to NaN
    df[f"X_{dist_column}"] = np.where(
        invalid_mask, np.nan, dist * np.cos(theta_radians)
    )
    df[f"Y_{dist_column}"] = np.where(
        invalid_mask, np.nan, dist * np.sin(theta_radians)
    )

    return df


def aggregate_to_daily_centroid(
    df,
    date_column="Timestamp",
    x_column="X",
    y_column="Y",
    weighted=True,
):
    """
    Aggregate half-hourly coordinate data to daily centroids.

    Parameters:
        df (pd.DataFrame): DataFrame containing timestamp and coordinates.
        date_column (str): Column containing datetime values.
        x_column (str): Column name for X coordinate.
        y_column (str): Column name for Y coordinate.
        weighted (bool): Weighted by ET column or not (default: True).

    Returns:
        pd.DataFrame: Aggregated daily centroids.
    """
    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=df.loc[x.index, "ET"])

    # Ensure datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Group by date (ignoring time component)
    df["Date"] = df[date_column].dt.date

    # Calculate centroid (mean of X and Y)
    if weighted:

        # Compute weighted average using ET as weights
        daily_centroids = (
            df.groupby("Date")
            .apply(
                lambda g: pd.Series(
                    {
                        x_column: (g[x_column] * g["ET"]).sum() / g["ET"].sum(),
                        y_column: (g[y_column] * g["ET"]).sum() / g["ET"].sum(),
                    }
                )
            )
            .reset_index()
        )
    else:
        daily_centroids = (
            df.groupby("Date").agg({x_column: "mean", y_column: "mean"}).reset_index()
        )
    # Groupby and aggregate with namedAgg [1]:
    return daily_centroids


def generate_density_raster(
    gdf,
    resolution=50,  # Cell size in meters
    buffer_distance=200,  # Buffer beyond extent in meters
    epsg=5070,  # Default coordinate system
    weight_field="ET",
):
    """
    Generate a density raster from a point GeoDataFrame, weighted by the ET field.

    Parameters:
        gdf (GeoDataFrame): Input point GeoDataFrame with an 'ET' field.
        resolution (float): Raster cell size in meters (default: 50m).
        buffer_distance (float): Buffer beyond point extent (default: 200m).
        epsg (int): Coordinate system EPSG code (default: 5070).
        weight_field (str): Weight field name (default: ET).

    Returns:
        raster (numpy.ndarray): Normalized density raster.
        transform (Affine): Affine transformation for georeferencing.
        bounds (tuple): (xmin, ymin, xmax, ymax) of the raster extent.
    """

    # Ensure correct CRS
    gdf = gdf.to_crs(epsg=epsg)

    # Extract point coordinates and ET values
    x = gdf.geometry.x
    y = gdf.geometry.y
    weights = gdf[weight_field].values

    # Define raster extent with buffer
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xmin, xmax = xmin - buffer_distance, xmax + buffer_distance
    ymin, ymax = ymin - buffer_distance, ymax + buffer_distance

    # Create a mesh grid
    xgrid = np.arange(xmin, xmax, resolution)
    ygrid = np.arange(ymin, ymax, resolution)
    xmesh, ymesh = np.meshgrid(xgrid, ygrid)

    # Perform KDE with weights
    kde = gaussian_kde(np.vstack([x, y]), weights=weights)
    density = kde(np.vstack([xmesh.ravel(), ymesh.ravel()])).reshape(xmesh.shape)

    # Normalize to ensure sum of cell values is 1
    print(np.sum(density))
    # density /= np.sum(density)

    # Define raster transform
    transform = from_origin(xmin, ymax, resolution, resolution)

    return density, transform, (xmin, ymin, xmax, ymax)
