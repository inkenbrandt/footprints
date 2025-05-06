import pandas as pd
import numpy as np

import geopandas as gpd

from rasterio.transform import from_origin
from scipy.stats import gaussian_kde


def polar_to_cartesian_dataframe(df, wd_column="WD", dist_column="Dist"):
    """
    Convert polar coordinates in a DataFrame to Cartesian coordinates.

    This function adds Cartesian coordinate columns (`X_<dist_column>` and
    `Y_<dist_column>`) to the input DataFrame based on polar inputs defined
    by wind direction (in degrees from North) and radial distance.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing polar coordinate columns.
    wd_column : str, optional
        Name of the column representing wind direction in degrees from North,
        by default "WD".
    dist_column : str, optional
        Name of the column representing distance from origin,
        by default "Dist".

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with two new columns:
        - `'X_<dist_column>'`: Cartesian X coordinate
        - `'Y_<dist_column>'`: Cartesian Y coordinate

    Notes
    -----
    - The function treats `-9999` and `NaN` as missing values.
    - Wind direction is converted such that 0° = North, 90° = East, etc.
    - Cartesian conversion is performed using:
      `X = dist * cos(θ)` and `Y = dist * sin(θ)` where θ = (90 - WD) in degrees.
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
    Aggregate sub-daily coordinate data to daily centroids.

    This function groups coordinate data by calendar date and computes the
    centroid of X and Y values. If `weighted=True`, centroids are computed
    using a weighted average based on the 'ET' (evapotranspiration) column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing timestamps and coordinate columns.
    date_column : str, optional
        Name of the column with datetime objects, by default "Timestamp".
    x_column : str, optional
        Name of the column representing X coordinates, by default "X".
    y_column : str, optional
        Name of the column representing Y coordinates, by default "Y".
    weighted : bool, optional
        If True, uses the 'ET' column as weights for centroid calculation;
        otherwise computes an unweighted mean. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per date and columns for:
        - `'Date'`: datetime.date object
        - `x_column`: daily centroid X coordinate
        - `y_column`: daily centroid Y coordinate

    Notes
    -----
    - Requires a column named `'ET'` if `weighted=True`.
    - Assumes the timestamp column can be parsed with `pd.to_datetime`.
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
    Generate a weighted kernel density raster from a point GeoDataFrame.

    This function uses a Gaussian kernel density estimation (KDE) to create a raster
    from spatial point data, where each point is weighted by a specified attribute
    (e.g., evapotranspiration). The output raster is georeferenced with an affine
    transform and buffered beyond the point extent.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing point geometries and a weight field.
    resolution : float, optional
        Size of each raster cell in projection units (typically meters),
        by default 50.
    buffer_distance : float, optional
        Distance to extend the raster beyond the extent of the points, in the same
        units as the CRS (e.g., meters), by default 200.
    epsg : int, optional
        EPSG code for the output coordinate reference system, by default 5070
        (NAD83 / Conus Albers).
    weight_field : str, optional
        Name of the column in `gdf` to use as weights in the KDE calculation,
        by default "ET".

    Returns
    -------
    density : numpy.ndarray
        2D array of KDE density values (not normalized by default).
    transform : affine.Affine
        Affine transformation mapping array coordinates to geographic space.
    bounds : tuple of float
        Bounding box of the raster extent as (xmin, ymin, xmax, ymax).

    Notes
    -----
    - This function assumes that the input geometries are point features.
    - The returned density array is not normalized unless the line
      `density /= np.sum(density)` is uncommented.
    - The output CRS is set using `gdf.to_crs(epsg=...)`.
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
