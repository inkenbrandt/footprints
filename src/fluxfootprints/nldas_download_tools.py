# -*- coding: utf-8 -*-
import configparser
import pandas as pd
import numpy as np
import pathlib
import rasterio

from rasterio.warp import calculate_default_transform, reproject, Resampling
import datetime
from affine import Affine

import os
import glob
import re

import requests
from pathlib import Path

import rasterio.features
from shapely.geometry import shape
import geopandas as gpd

import xarray
import refet


def fetch_and_preprocess_data(url, station, startdate):
    """
    Retrieve and preprocess AmeriFlux eddy covariance data for a given station.

    Queries a remote database for flux data starting from a specified date and
    performs basic preprocessing, including timestamp parsing, resampling, and
    handling of missing values.

    Parameters
    ----------
    url : str
        Base URL of the database API.
    station : str
        Station identifier string (e.g., site code or station ID).
    startdate : str
        Starting date (ISO format: 'YYYY-MM-DD') for data retrieval.

    Returns
    -------
    df : pandas.DataFrame
        Preprocessed DataFrame indexed by timestamp and containing hourly-averaged
        flux variables. Empty if retrieval fails.

    Notes
    -----
    - Replaces `-9999` with `NaN` for missing values.
    - Drops rows where critical variables ('h2o', 'wd', 'ustar', 'v_sigma') are missing.
    - Resamples to hourly means using `numeric_only=True`.
    - Assumes the API conforms to a PostgREST-style filtering syntax.
    """
    headers = {"Accept-Profile": "groundwater", "Content-Type": "application/json"}
    params = {"stationid": f"eq.{station}", "datetime_start": f"gte.{startdate}"}

    try:
        response = requests.get(f"{url}/amfluxeddy", headers=headers, params=params)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        df["datetime_start"] = pd.to_datetime(df["datetime_start"])
        df = df.set_index("datetime_start")
        df.replace(-9999, np.nan, inplace=True)
        df = df.resample("1h").mean(numeric_only=True)
        df.dropna(subset=["h2o", "wd", "ustar", "v_sigma"], inplace=True)
        return df
    except requests.RequestException as e:
        print(f"Failed to fetch data for station {station}: {e}")
        return pd.DataFrame()


def multiply_directories_rast(dir1=None, dir2=None, out_dir=None, model="ensemble"):
    """
    Multiply matching GeoTIFF rasters from two directories based on dates in their
    filenames.

    This function searches for GeoTIFF files in two directories (``dir1`` and
    ``dir2``) that share a common date string in the format ``"YYYY_MM_DD"``. It
    multiplies pairs of rasters with :func:`multiply_geotiffs` and saves the
    results to ``out_dir``.

    Parameters
    ----------
    dir1 : pathlib.Path or str, optional
        Directory containing the first set of GeoTIFF files, typically with
        filenames ending in ``"_weighted.tif"``.  Defaults to
        ``"./output/usutw/"``.
    dir2 : pathlib.Path or str, optional
        Directory containing the second set of GeoTIFF files, typically with
        filenames starting with ``"ensemble_et_"``.  Defaults to
        ``"G:/My Drive/OpenET Exports/"``.
    out_dir : pathlib.Path or str, optional
        Output directory where the resulting multiplied rasters will be saved.
        If the directory does not exist, it is created.  Defaults to
        ``"./output/usutw_mult/"``.
    model : str, optional
        Name of the model used to identify files in ``dir2``.  Must match the
        file‑naming pattern (e.g., ``"ensemble"``, ``"eemetric"``, or
        ``"ssebop"``).  Default is ``"ensemble"``.

    Returns
    -------
    dict
        Mapping of :class:`pandas.Timestamp` objects (derived from the matched
        date strings) to the result returned by :func:`multiply_geotiffs` for
        each raster pair.

    Notes
    -----
    * Only files with matching ``"YYYY_MM_DD"`` date strings in both
      directories are processed.
    * Filenames in ``dir2`` must start with ``"{model}_et_"`` followed by the
      date.
    * Files without a matching pair in the other directory are skipped.
    * Output filenames are formatted
      ``"weighted_{model}_openet_{YYYY_MM_DD}.tif"``.

    Examples
    --------
    >>> from pathlib import Path
    >>> results = multiply_directories_rast(
    ...     dir1=Path("./output/usutw/"),
    ...     dir2=Path("G:/My Drive/OpenET Exports/"),
    ...     out_dir=Path("./output/usutw_mult/"),
    ...     model="ensemble",
    ... )
    >>> list(results.keys())[:2]
    [Timestamp('2021-03-05 00:00:00'), Timestamp('2021-03-06 00:00:00')]
    """

    # Set the paths to your two directories
    if dir1 is None:
        dir1 = pathlib.Path("./output/usutw/")  # e.g., contains '...20210305.tif', etc.
    if dir2 is None:
        dir2 = pathlib.Path("G:/My Drive/OpenET Exports/")
    if out_dir is None:
        out_dir = pathlib.Path("./output/usutw_mult/")

    # Check if it exists
    if not out_dir.exists():
        # Create the directory (including any necessary parent directories)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Directory {out_dir} created.")
    else:
        print(f"Directory {out_dir} already exists.")

    # Regex pattern for an 8-digit date (adjust if your date format is different)
    date_pattern = re.compile(r"\d{4}_\d{2}_\d{2}")

    # 1) Build a dictionary of {date_string: full_path} for files in dir2
    date_to_file_dir2 = {}
    for filename in dir2.glob(f"{model}_et_*.tif"):
        match = date_pattern.search(filename.stem)
        if match:
            date_str = match.group(0)
            date_to_file_dir2[date_str] = filename

    tsum = {}

    # 2) Iterate over the files in dir1, extract date, and check if we have a match in dir2
    for filename in dir1.glob("*_weighted.tif"):
        dt_str = filename.stem.split("_")[0].replace("-", "_")
        match = date_pattern.search(dt_str)
        if match:
            date_str = match.group(0)
            # Check if this date exists in dir2
            if date_str in date_to_file_dir2:
                date = pd.to_datetime(date_str, format="%Y_%m_%d")
                file1 = filename
                file2 = date_to_file_dir2[date_str]
                output_raster = out_dir / f"weighted_{model}_openet_{date_str}.tif"
                tsum[date] = multiply_geotiffs(file1, file2, output_raster)
    return tsum


def reproject_raster_dir(input_folder, output_folder, target_epsg="EPSG:5070"):
    """
    Reproject all GeoTIFF files in a directory to a specified EPSG coordinate system.

    This function reads all `.tif` files in the input directory, reprojects them to the
    specified coordinate reference system (CRS), and saves the reprojected rasters to
    the output directory using the same filenames.

    Parameters
    ----------
    input_folder : str or pathlib.Path
        Path to the directory containing input GeoTIFF files.
    output_folder : str or pathlib.Path
        Path to the directory where reprojected GeoTIFF files will be saved.
        If the directory does not exist, it will be created.
    target_epsg : str, optional
        EPSG code of the target CRS (e.g., "EPSG:5070"). Default is "EPSG:5070".

    Returns
    -------
    None
        The function performs file I/O but does not return any value.

    Notes
    -----
    - Uses Rasterio for reading, writing, and reprojecting GeoTIFF files.
    - Reprojection uses `Resampling.nearest` for band resampling.
    - Target transform, width, and height are calculated using `calculate_default_transform`.
    - The output raster preserves the number of bands and data type from the input.

    Examples
    --------
    >>> reproject_raster_dir(
    ...     input_folder="./input_rasters",
    ...     output_folder="./output_rasters",
    ...     target_epsg="EPSG:4326"
    ... )
    Reprojected ./input_rasters/file1.tif → ./output_rasters/file1.tif
    Reprojected ./input_rasters/file2.tif → ./output_rasters/file2.tif
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through every .tif file in the input folder
    for input_path in glob.glob(os.path.join(input_folder, "*.tif")):
        with rasterio.open(input_path) as src:
            # Calculate the transform, width, and height in the new CRS
            transform, width, height = calculate_default_transform(
                src.crs, target_epsg, src.width, src.height, *src.bounds
            )

            # Copy the metadata, then update with new CRS, transform, size
            meta = src.meta.copy()
            meta.update(
                {
                    "crs": target_epsg,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )

            # Build output file path
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_folder, filename)

            # Reproject and save
            with rasterio.open(output_path, "w", **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_epsg,
                        resampling=Resampling.nearest,
                    )

        print(f"Reprojected {input_path} → {output_path}")


def weighted_rasters(
    stationid="US-UTW",
    start_hr=6,
    normed_NLDAS_stats_path="nldas_all_normed.parquet",
    out_dir=None,
):
    """
    Generate daily weighted footprint rasters based on hourly fetch and normalized ETo data.
    This function reads a Parquet file containing daily normalized ETo values for multiple
    stations, filters it by the specified station ID, and applies hourly weighting to existing
    footprint rasters. For each unweighted TIFF file in `out_dir`, the function:
    1. Parses the date from the filename.
    2. Reads hourly bands from the raster, normalizes them by their global sum, and multiplies by the normalized ETo value for that hour.
    3. Sums these hourly weighted rasters into a single daily footprint raster.
    4. Writes the result to a new file named `<YYYY-MM-DD>_weighted.tif` in `out_dir`.

    Parameters
    ----------
    stationid : str, optional
        Station identifier used to look up normalized ETo values (default: 'US-UTW').
    start_hr : int, optional
        Starting hour for the data slice (default: 6, i.e. 7 AM).
    end_hr : int, optional
        Ending hour for the data slice (default: 18, i.e. 8 PM).
    normed_NLDAS_stats_path : str or pathlib.Path, optional
        Path to the Parquet file containing normalized ETo data (default: 'nldas_all_normed.parquet').
    out_dir : str or pathlib.Path, optional
        Output directory containing unweighted footprint rasters (default: current directory).

    Notes
    -----
    - Any TIFF files in `out_dir` with filenames starting with '20' (e.g., '2022-01-01.tif') are processed, unless they already contain the substring 'weighted' in their filename.
    - The function expects the TIFF filename to be in the form 'YYYY-MM-DD.tif' so it can parse out the date.
    - Only generates a weighted TIFF file if the total sum of the final footprint is within 0.15 of 1.0.
    - Hourly rasters with all NaN values are replaced with zeros.
    - Written rasters preserve the same georeferencing, resolution, and coordinate reference system as the input rasters.

    Returns
    -------
    None
        This function does not return anything. It writes a single-band, daily-weighted footprint
        raster to `out_dir` for each processed date.

    Example
    -------
    >>> weighted_rasters(
    ...     stationid='US-UTW',
    ...     start_hr=6,
    ...     end_hr=18,
    ...     normed_NLDAS_stats_path='nldas_all_normed.parquet',
    ...     out_dir='/path/to/tif/files'
    ... )
    """
    # Ensure out_dir is a Path
    if out_dir is None:
        out_dir = pathlib.Path("./output/")
    else:
        out_dir = pathlib.Path(out_dir)

    # Read the Parquet file and filter data for the specified station
    eto_df = pd.read_parquet(normed_NLDAS_stats_path)
    eto_df["daily_ETo_normed"] = eto_df["daily_ETo_normed"].fillna(
        0
    )  # Fill missing ETo with 0
    nldas_df = eto_df.loc[stationid]

    # Iterate over all TIFF files in out_dir that begin with '20'
    for out_f in out_dir.glob("20*.tif"):
        # Skip if the file is already weighted
        if "weighted" in out_f.stem:
            continue

        print(f"Processing {out_f.name}")

        # Parse the date from the file name
        try:
            date = datetime.datetime.strptime(out_f.stem, "%Y_%m_%d")
        except ValueError:
            print(
                f"Skipping {out_f.name} because its filename is not in 'YYYY-MM-DD' format."
            )
            continue

        # Prepare output file name
        final_outf = (
            out_dir / f"{date.year:04d}-{date.month:02d}-{date.day:02d}_weighted.tif"
        )

        # Skip if output already exists
        if final_outf.is_file():
            print(f"Weighted file already exists for {date.date()}. Skipping.")
            continue

        # Open the source raster once
        with rasterio.open(out_f) as src:
            band_indexes = src.indexes  # e.g. [1, 2, 3, ...] for each hour
            # We'll accumulate the weighted footprint across all hours
            normed_fetch_rasters = []

            for band_idx in band_indexes:
                # The hour we are processing: band 1 corresponds to (start_hr), band 2 -> (start_hr+1), etc.
                hour = band_idx + start_hr - 1
                dtindex = pd.to_datetime(f"{date:%Y-%m-%d} {hour:02d}:00:00")

                # Attempt to read the normalized ETo from nldas_df
                try:
                    norm_eto = nldas_df.loc[dtindex, "daily_ETo_normed"]
                except KeyError:
                    print(f"No NLDAS record for {dtindex}; using 0 as fallback.")
                    norm_eto = 0.0

                arr = src.read(band_idx)
                band_sum = np.nansum(arr)

                # Avoid division by zero
                if band_sum == 0 or np.isnan(band_sum):
                    # If everything is NaN or zero, use a zeros array
                    tmp = np.zeros_like(arr)
                else:
                    # Normalize by band sum
                    tmp = arr / band_sum

                # Multiply by normalized ETo
                weighted_arr = tmp * norm_eto
                normed_fetch_rasters.append(weighted_arr)

            # Sum the weighted hourly rasters into a single daily footprint
            final_footprint = sum(normed_fetch_rasters)

            # Only proceed if the daily sum is close to 1.0
            footprint_sum = final_footprint.sum()  # type: ignore
            if np.isclose(footprint_sum, 1.0, atol=0.15):
                # Write output raster
                print(f"Writing weighted footprint to {final_outf}")
                with rasterio.open(
                    final_outf,
                    "w",
                    driver="GTiff",
                    dtype=rasterio.float64,
                    count=1,
                    height=final_footprint.shape[0],  # type: ignore
                    width=final_footprint.shape[1],  # type: ignore
                    transform=src.transform,
                    crs=src.crs,
                    nodata=0.0,
                ) as out_raster:
                    out_raster.write(final_footprint, 1)
            else:
                print(
                    f"Final footprint sum check failed for {date.date()}: sum={footprint_sum:.3f}"
                )


def clip_to_utah_merge(file_dir="./NLDAS_data/", years=None, output_dir="./"):
    """
    Clip NLDAS NetCDF files to Utah boundaries and merge them by year.

    This function scans a directory for NetCDF files corresponding to the specified `years`,
    extracts data within the geographic bounds of Utah, and merges the resulting subsets
    along the time dimension. Outputs are saved in both NetCDF and Parquet formats.

    Parameters
    ----------
    file_dir : str or pathlib.Path, optional
        Directory containing NLDAS NetCDF files. Defaults to "./NLDAS_data/".
    years : list of int, optional
        List of years to process. If None, defaults to [2022, 2023, 2024].
    output_dir : str or pathlib.Path, optional
        Directory to store the output files. Defaults to "./".

    Returns
    -------
    None
        This function writes files to disk but does not return anything.

    Notes
    -----
    - Latitude bounds for Utah: 37.0 to 42.0
    - Longitude bounds for Utah: -114.0 to -109.0
    - Files are merged by year using `xarray.concat` on the `time` dimension.
    - Output filenames:
        - NetCDF:  `<year>_utah_merged.nc`
        - Parquet: `<year>_utah_merged.parquet`

    Examples
    --------
    >>> clip_to_utah_merge(file_dir="./NLDAS_data/", years=[2021, 2022], output_dir="./outputs/")
    """
    # Define Utah's latitude and longitude boundaries
    utah_lat_min, utah_lat_max = 37.0, 42.0
    utah_lon_min, utah_lon_max = -114.0, -109.0

    if isinstance(file_dir, Path):
        netcdf_files = file_dir
    else:
        # List of uploaded NetCDF files
        netcdf_files = pathlib.Path(file_dir)

    if isinstance(years, list):
        pass
    else:
        years = [2021, 2022, 2023, 2024]

    if isinstance(output_dir, Path):
        output_dir = output_dir
    else:
        # List of uploaded NetCDF files
        output_dir = pathlib.Path(output_dir)

    for year in years:
        print(year)
        # Extract Utah-specific data from each file and store datasets
        utah_datasets = []
        for file in netcdf_files.glob(f"{year}*.nc"):
            print(file)
            ds_temp = xarray.open_dataset(file)
            ds_utah_temp = ds_temp.sel(
                lat=slice(utah_lat_min, utah_lat_max),
                lon=slice(utah_lon_min, utah_lon_max),
            )
            utah_datasets.append(ds_utah_temp)

        # Merge all extracted datasets along the time dimension
        ds_merged = xarray.concat(utah_datasets, dim="time")

        # Save as NetCDF using a compatible format (default for xarray in this environment)
        netcdf_output_path = output_dir / f"{year}_utah_merged.nc"
        ds_merged.to_netcdf(netcdf_output_path)

        # Convert to Pandas DataFrame for Parquet format
        df_parquet = ds_merged.to_dataframe().reset_index()

        # Save as Parquet
        parquet_output_path = output_dir / f"{year}_utah_merged.parquet"
        df_parquet.to_parquet(parquet_output_path, engine="pyarrow")

        # Provide download links
        print(netcdf_output_path, parquet_output_path)


def multiply_geotiffs(input_a, input_b, output_path):
    """
    Multiply two GeoTIFF rasters after aligning them to the same CRS, extent, and resolution.

    This function opens two input rasters, reprojects the second raster to match the spatial
    properties of the first, multiplies them element-wise (handling NoData values gracefully),
    and writes the result to a new output raster file. It also prints and returns the sum of the
    resulting raster values.

    Parameters
    ----------
    input_a : str or pathlib.Path
        Path to the first GeoTIFF file. This raster is used as the spatial reference.
    input_b : str or pathlib.Path
        Path to the second GeoTIFF file. This raster is reprojected to match `input_a`.
    output_path : str or pathlib.Path
        Path where the output multiplied raster will be saved.

    Returns
    -------
    float
        The sum of the values in the output raster, excluding masked (NoData) values.

    Notes
    -----
    - The CRS, resolution, extent, and transform of `input_b` are matched to those of `input_a`
      using bilinear resampling.
    - The multiplication respects the NoData mask of `input_a`.
    - The resulting raster uses the data type of `input_a` and writes NoData areas as zeros.
    - The output raster is written as a single-band GeoTIFF.

    Examples
    --------
    >>> total = multiply_geotiffs("a.tif", "b.tif", "output_mult.tif")
    Output saved to: output_mult.tif
    Sum of raster values: 123456.78
    """

    # --- Open the first raster (this will be our "reference" grid) ---
    with rasterio.open(input_a) as src_a:
        profile_a = src_a.profile.copy()
        # Read the full data array for A
        data_a = src_a.read(1, masked=True)  # returns a MaskedArray if there's nodata

        # We'll store the relevant spatial info to guide reprojecting raster B
        ref_crs = src_a.crs
        ref_transform = src_a.transform
        ref_width = src_a.width
        ref_height = src_a.height

        # --- Open the second raster ---
        with rasterio.open(input_b) as src_b:
            # 1) We need both rasters in the same CRS. If different, we'll reproject B.
            # 2) We also want B to match A's resolution and extent exactly.

            # Create an empty array to hold the reprojected data from B
            data_b_aligned = np.zeros((ref_height, ref_width), dtype=src_a.dtypes[0])

            # Reproject (and resample) B to match A's grid
            reproject(
                source=rasterio.band(src_b, 1),
                destination=data_b_aligned,
                src_transform=src_b.transform,
                src_crs=src_b.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )

    # --- Perform the multiplication (masked arrays handle NoData gracefully) ---
    # Convert data_b_aligned to a masked array if you want to respect NoData from A
    data_b_masked = np.ma.array(data_b_aligned, mask=data_a.mask)
    data_mult = data_a * data_b_masked

    # --- Update the profile for the output ---
    # We'll keep the same data type as A. If needed, you can change this (e.g., float32).
    profile_out = profile_a.copy()
    profile_out.update(dtype=str(data_mult.dtype), count=1, nodata=None)

    # --- Write the result ---
    with rasterio.open(output_path, "w", **profile_out) as dst:
        dst.write(
            data_mult.filled(0).astype(profile_out["dtype"]), 1
        )  # fill masked with NaN or a NoData value

    print(f"Output saved to: {output_path}")
    with rasterio.open(output_path) as src:
        # Read the first band into a NumPy array
        band_data = src.read(1)

        # If you have "NoData" values and you'd like to exclude them, you can
        # read the band as a masked array:
        band_data = src.read(1, masked=True)
        print(band_data)
        # Then compute the sum of all values
        total_sum = np.sum(band_data)

        print("Sum of raster values:", total_sum)
    return total_sum


def mask_fp_cutoff(f_array, cutoff=0.9):
    """
    Mask all values in a footprint array outside the cumulative contribution cutoff.

    This function applies a cumulative contribution threshold to a footprint array,
    keeping only the highest values that together sum to the specified `cutoff`
    fraction of the total. All other values are set to a small constant near zero.

    Parameters
    ----------
    f_array : numpy.ndarray
        2D array of footprint contribution values (unitless).
    cutoff : float, optional
        Fraction of the cumulative footprint to retain (default is 0.9 for 90%).

    Returns
    -------
    numpy.ndarray
        2D array where values below the cutoff threshold are replaced with a small
        near-zero constant (`0.0`), and values above the threshold are retained.

    Notes
    -----
    - The cutoff is applied based on the sorted cumulative sum of all values.
    - Values below the determined threshold are set to `0.0`, not NaN, to avoid
      issues in downstream raster calculations.
    - This function is useful for visualizing or focusing on the core area of a
      footprint (e.g., 90% contribution area).
    - Uses a logarithmic debug message to record the computed threshold.

    Examples
    --------
    >>> masked = mask_fp_cutoff(footprint_array, cutoff=0.8)
    >>> np.sum(masked > 0)
    150  # Number of grid cells contributing to 80% of the flux
    """
    val_array = f_array.flatten()
    sort_df = pd.DataFrame({"f": val_array}).sort_values(by="f").iloc[::-1]
    sort_df["cumsum_f"] = sort_df["f"].cumsum()

    sort_group = sort_df.groupby("f", as_index=True).mean()
    diff = abs(sort_group["cumsum_f"] - cutoff)
    sum_cutoff = diff.idxmin()
    f_array = np.where(f_array >= sum_cutoff, f_array, np.nan)
    f_array[~np.isfinite(f_array)] = 0.00000000e000

    print(f"mask_fp_cutoff: applied cutoff={cutoff}, sum_cutoff={sum_cutoff}")
    return f_array


def download_nldas(
    date: datetime.date | datetime.datetime,
    hour: int,
    ed_user: str,
    ed_pass: str,
):
    """
    Download a single-hour NLDAS forcing dataset in NetCDF format.

    This function downloads a specific hourly NLDAS (North American Land Data Assimilation System)
    forcing product file from NASA GES DISC for a given date and hour using Earthdata credentials.
    The file is saved in a local `NLDAS_data/` directory and is skipped if it already exists.

    Parameters
    ----------
    date : datetime.date or datetime.datetime
        Date of the file to download.
    hour : int
        Hour of the file to download (0 to 23).
    ed_user : str
        Earthdata username for NASA GES DISC authentication.
    ed_pass : str
        Earthdata password for authentication.

    Returns
    -------
    None
        This function saves a NetCDF file to disk and does not return a value.

    Notes
    -----
    - Files are saved to the `NLDAS_data/` directory as: `YYYY_MM_DD_HH.nc`.
    - The function uses a two-step request process to handle Earthdata login redirection.
    - If the file already exists locally, it will not be re-downloaded.
    - File format: NetCDF (`.nc`) from the NLDAS_FORA0125_H.2.0 collection.

    Example
    -------
    >>> from datetime import datetime
    >>> download_nldas(datetime(2024, 7, 15), hour=14, ed_user="myuser", ed_pass="mypassword")
    File NLDAS_data/2024_07_15_14.nc downloaded successfully.
    """
    # NLDAS version 2, primary forcing set (a), DOY must be 3 digit zero padded, HH 2-digit between 00-23, MM and DD also 2 digit
    YYYY = date.year
    DOY = date.timetuple().tm_yday
    MM = date.month
    DD = date.day
    HH = hour

    nldas_out_dir = Path("NLDAS_data")
    if not nldas_out_dir.is_dir():
        nldas_out_dir.mkdir(parents=True, exist_ok=True)

    nldas_outf_path = nldas_out_dir / f"{YYYY}_{MM:02}_{DD:02}_{HH:02}.nc"
    if nldas_outf_path.is_file():
        print(f"{nldas_outf_path} already exists, not overwriting.")
        pass
        # do not overwrite!
    else:
        # data_url = f'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.002/{YYYY}/{DOY:03}/NLDAS_FORA0125_H.A{YYYY}{MM:02}{DD:02}.{HH:02}00.002.grb'
        data_url = f"https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/{YYYY}/{DOY:03}/NLDAS_FORA0125_H.A{YYYY}{MM:02}{DD:02}.{HH:02}00.020.nc"
        session = requests.Session()
        r1 = session.request("get", data_url)
        r = session.get(r1.url, stream=True, auth=(ed_user, ed_pass))

        # write grib file temporarily
        with open(nldas_outf_path, "wb") as outf:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:  # filter out keep-alive new chunks
                    outf.write(chunk)


def read_compiled_input(path: str | Path) -> tuple | None:
    r"""
    Load a CSV file that contains pre‑compiled inputs for flux‑footprint
    modelling, validate its contents, and return the cleaned data.

    The routine

    1. reads the file into a :class:`pandas.DataFrame`;
    2. parses the index as ``datetime`` and resamples to **hourly** means;
    3. verifies that every required variable is present;
    4. drops rows in which any required value is missing; and
    5. extracts site latitude/longitude from the first row.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file containing the pre‑processed input data.

    Returns
    -------
    tuple[pandas.DataFrame, float, float] or None
    *If* the file is valid, a tuple ``(df, latitude, longitude)``

    If required variables are missing or the file cannot be parsed,
    the function returns ``None``.

    Notes
    -----
    **Variables that *must* be present**

    - ``latitude``, ``longitude``
    - ``ET_corr``                    – corrected evapotranspiration
    - ``wind_dir``, ``u_star``, ``sigma_v``
    - ``zm`` (measurement height), ``hc`` (canopy height),
      ``d`` (displacement height), ``L`` (Obukhov length)
    - *Either* ``u_mean`` (mean wind speed) *or* ``z0`` (roughness length)

    **Optional columns that are retained if present**

    - ``IGBP_land_classification``
    - ``secondary_veg_type``

    The data are resampled with::

        df = df.resample("H").mean()

    and any hourly row still missing a required value is dropped.

    Examples
    --------
    >>> df, lat, lon = read_compiled_input("inputs/station_data.csv")
    >>> df.columns
    Index(['latitude', 'longitude', 'ET_corr', 'wind_dir', 'u_star', 'sigma_v',
           'zm', 'hc', 'd', 'L', 'u_mean'],
          dtype='object')
    """

    ret = None
    need_vars = {
        "latitude",
        "longitude",
        "ET_corr",
        "wind_dir",
        "u_star",
        "sigma_v",
        "zm",
        "hc",
        "d",
        "L",
    }
    # don't parse dates first check if required inputs exist to save processing time
    df = pd.read_csv(path, index_col="date", parse_dates=False)
    cols = df.columns
    check_1 = need_vars.issubset(cols)
    check_2 = len({"u_mean", "z0"}.intersection(cols)) >= 1  # need one or the other
    # if either test failed then insufficient input data for footprint, abort
    if not check_1 or not check_2:
        return ret
    ret = df
    ret.index = pd.to_datetime(df.index)
    ret = ret.resample("H").mean()
    lat, lon = ret[["latitude", "longitude"]].values[0]
    keep_vars = need_vars.union(
        {"u_mean", "z0", "IGBP_land_classification", "secondary_veg_type"}
    )
    drop_vars = list(set(cols).difference(keep_vars))
    ret.drop(drop_vars, 1, inplace=True)  # type: ignore
    ret.dropna(
        subset=["wind_dir", "u_star", "sigma_v", "d", "zm", "L", "ET_corr"],
        how="any",
        inplace=True,
    )
    return ret, lat, lon


def snap_centroid(station_x: float, station_y: float) -> tuple[float, float]:
    """
    Snap station coordinates to the nearest odd multiple of 15 within a 30-meter grid.

    This function adjusts a UTM coordinate pair `(station_x, station_y)` such that the
    resulting point aligns with a 30-meter grid cell and lands on an odd multiple of 15.
    This helps ensure symmetry and minimizes distortion in gridded analysis (e.g., for
    raster-based footprint modeling).

    Parameters
    ----------
    station_x : float
        Original x-coordinate (e.g., UTM Easting) of the station.
    station_y : float
        Original y-coordinate (e.g., UTM Northing) of the station.

    Returns
    -------
    tuple of float
        The adjusted `(x, y)` coordinates snapped to the appropriate grid centroid.

    Notes
    -----
    - The adjustment ensures that both `x` and `y` are centered on a 30-meter grid.
    - Coordinates are snapped to the nearest **odd multiple of 15** to avoid aligning
      exactly with grid cell edges.
    - This function is particularly useful for aligning station locations to raster grids
      for flux footprint or land surface model analysis.

    Examples
    --------
    >>> x_adj, y_adj = snap_centroid(435627.3, 4512322.7)
    >>> print(x_adj, y_adj)
    435630.0 4512322.5
    """
    # move coord to snap centroid to 30m grid, minimal distortion
    rx = station_x % 15
    if rx > 7.5:
        station_x += 15 - rx
        # final coords should be odd factors of 15
        if (station_x / 15) % 2 == 0:
            station_x -= 15
    else:
        station_x -= rx
        if (station_x / 15) % 2 == 0:
            station_x += 15

    ry = station_y % 15
    if ry > 7.5:
        print("ry > 7.5")
        station_y += 15 - ry
        if (station_y / 15) % 2 == 0:
            station_y -= 15
    else:
        print("ry <= 7.5")
        station_y -= ry
        if (station_y / 15) % 2 == 0:
            station_y += 15
    print("adjusted coordinates:", station_x, station_y)
    return station_x, station_y


def load_stat_configs(
    station,
    config_path="../../station_config/",
    secrets_path="../../secrets/config.ini",
):
    """
    Load station metadata and secrets from configuration files.

    Parameters:
    -----------
    station : str
        Station identifier.
    config_path : str
        Path to station configuration file.
    secrets_path : str
        Path to secrets configuration file.

    Returns:
    --------
    dict
        A dictionary containing station metadata and database URL.
    """

    if isinstance(config_path, Path):
        pass
    else:
        config_path = Path(config_path)

    config_path_loc = config_path / f"{station}.ini"
    config = configparser.ConfigParser()
    config.read(config_path_loc)

    if isinstance(secrets_path, Path):
        pass
    else:
        secrets_path = Path(secrets_path)

    secrets_config = configparser.ConfigParser()
    secrets_config.read(secrets_path)

    return {
        "url": secrets_config["DEFAULT"]["url"],
        "latitude": float(config["METADATA"]["station_latitude"]),
        "longitude": float(config["METADATA"]["station_longitude"]),
        "elevation": float(config["METADATA"]["station_elevation"]),
    }


def extract_nldas_xr_to_df(
    years,
    input_path=".",
    config_path="../../station_config/",
    secrets_path="../../secrets/config.ini",
    output_path="./output/",
):
    """
    Extract and compile NLDAS-derived timeseries for multiple stations into a single DataFrame.

    This function processes NLDAS datasets (in NetCDF format with ETo, ETr, and related variables)
    for specified years, extracts values for multiple stations based on their configuration `.ini`
    files, and writes the result to a Parquet file.

    Parameters
    ----------
    years : list of int
        List of years for which NLDAS NetCDF files (with ETo included) will be processed.
    input_path : str or pathlib.Path, optional
        Directory containing NetCDF input files named like `<year>_with_eto.nc`.
    config_path : str or pathlib.Path, optional
        Directory containing station `.ini` configuration files (e.g., `US-UTW.ini`) with keys
        like latitude, longitude, and elevation.
    secrets_path : str or pathlib.Path, optional
        Path to the secrets/config file used by the `load_configs` function.
    output_path : str or pathlib.Path, optional
        Directory to save the output Parquet file. Default is `./output/`.

    Returns
    -------
    None
        Writes a single Parquet file (`nldas_all.parquet`) containing hourly time series data
        indexed by station ID and datetime.

    Notes
    -----
    - The function uses `xarray` to open NetCDF files and `pandas` to store time series data.
    - It finds the nearest spatial point in the dataset for each station’s lat/lon/elevation.
    - Extracted variables include: ETo, ETr, PotEvap, LWdown, SWdown, Tair, Qair, PSurf,
      Wind_E, and Wind_N (converted to total wind speed).
    - Output columns: ['eto', 'etr', 'pet', 'lwdown', 'swdown', 'temperature',
      'rh', 'pressure', 'wind'].
    - Time is converted to local time by subtracting 7 hours from UTC.

    Example
    -------
    >>> extract_nldas_xr_to_df(
    ...     years=[2022, 2023],
    ...     input_path="./NLDAS_data",
    ...     config_path="./station_config",
    ...     output_path="./output"
    ... )
    """
    if isinstance(input_path, Path):
        pass
    else:
        input_path = Path(input_path)

    if isinstance(config_path, Path):
        pass
    else:
        config_path = Path(config_path)

    if isinstance(output_path, Path):
        pass
    else:
        output_path = Path(output_path)

    dataset = {}
    for year in years:
        ds = xarray.open_dataset(
            input_path / f"{year}_with_eto.nc",
        )
        dfs = {}
        for file in config_path.glob("US*.ini"):
            name = file.stem
            print(name)
            config = load_stat_configs(
                name, config_path=config_path, secrets_path=secrets_path  # type: ignore
            )

            # Define the target latitude, longitude, and elevation (adjust as needed)
            target_lat = pd.to_numeric(config["latitude"])
            target_lon = pd.to_numeric(config["longitude"])
            target_elev = int(pd.to_numeric(config["elevation"]))

            # Find the nearest latitude, longitude, and elevation in the dataset
            nearest_lat = ds["lat"].sel(lat=target_lat, method="nearest").values
            nearest_lon = ds["lon"].sel(lon=target_lon, method="nearest").values
            nearest_elev = (
                ds["elevation"].sel(elevation=target_elev, method="nearest").values
            )

            # Extract ETo time series at the nearest matching location
            eto_timeseries = ds["ETo"].sel(
                elevation=nearest_elev, lat=nearest_lat, lon=nearest_lon
            )
            etr_timeseries = ds["ETr"].sel(
                elevation=nearest_elev, lat=nearest_lat, lon=nearest_lon
            )

            # Extract PotEvap time series at the same location
            pet_ts = ds["PotEvap"].sel(lat=nearest_lat, lon=nearest_lon)
            lwd_ts = ds["LWdown"].sel(lat=nearest_lat, lon=nearest_lon)
            swd_ts = ds["SWdown"].sel(lat=nearest_lat, lon=nearest_lon)
            temp_ts = ds["Tair"].sel(lat=nearest_lat, lon=nearest_lon)
            rh_ts = ds["Qair"].sel(lat=nearest_lat, lon=nearest_lon)
            pres_ts = ds["PSurf"].sel(lat=nearest_lat, lon=nearest_lon)
            wind_u_ts = ds["Wind_E"].sel(lat=nearest_lat, lon=nearest_lon)
            wind_v_ts = ds["Wind_N"].sel(lat=nearest_lat, lon=nearest_lon)
            wind_ts = np.sqrt(wind_u_ts**2 + wind_v_ts**2)

            dfs[name] = pd.DataFrame(
                {
                    "datetime": ds["time"],
                    "eto": eto_timeseries,
                    "etr": etr_timeseries,
                    "pet": pet_ts,
                    "lwdown": lwd_ts,
                    "swdown": swd_ts,
                    "temperature": temp_ts,
                    "rh": rh_ts,
                    "pressure": pres_ts,
                    "wind": wind_ts,
                }
            ).round(4)
        dataset[year] = pd.concat(dfs)
    alldata = pd.concat(dataset)
    alldata["datetime"] = pd.to_datetime(alldata["datetime"])
    eto_df = alldata.reset_index().rename(
        columns={"level_0": "year", "level_1": "stationid"}
    )
    eto_df["datetime"] = eto_df["datetime"] - pd.Timedelta(hours=7)
    eto_df = eto_df.set_index(["stationid", "datetime"])

    # Save DataFrame to Parquet
    eto_df.to_parquet(output_path / "nldas_all.parquet")


def norm_minmax_dly_et(x):
    """
    Apply min-max normalization to a series of daily evapotranspiration values.

    This function scales input values linearly to the [0, 1] range using min-max normalization,
    then rounds the results to 4 decimal places.

    Parameters
    ----------
    x : pandas.Series or numpy.ndarray
        One-dimensional array or series of daily ET values.

    Returns
    -------
    numpy.ndarray
        Normalized array of values between 0 and 1, rounded to 4 decimals.

    Notes
    -----
    - If all values in `x` are equal, the result will be `NaN` due to division by zero.
    - This function does not handle missing values explicitly; ensure `x` is clean beforehand.

    Examples
    --------
    >>> import pandas as pd
    >>> x = pd.Series([2.1, 2.5, 3.0, 3.5])
    >>> norm_minmax_dly_et(x)
    array([0.   , 0.267, 0.6  , 1.   ])
    """
    # Normalize using min-max scaling and then divide by the sum,
    # rounding to 4 decimal places.
    return np.round(((x - x.min()) / (x.max() - x.min())), 4)


def norm_dly_et(x):
    """
    Normalize daily evapotranspiration values by their sum.

    This function scales input values so that their sum equals 1, useful for generating
    relative daily weights. The result is rounded to 4 decimal places.

    Parameters
    ----------
    x : pandas.Series or numpy.ndarray
        One-dimensional array or series of daily ET values.

    Returns
    -------
    numpy.ndarray
        Array of normalized values summing to 1, rounded to 4 decimal places.

    Notes
    -----
    - If the sum of `x` is zero, the result will contain `NaN` or `inf` values.
    - Ensure input is non-negative and non-empty for meaningful results.

    Examples
    --------
    >>> import numpy as np
    >>> norm_dly_et(np.array([1.0, 2.0, 3.0]))
    array([0.167, 0.333, 0.5  ])
    """
    return np.round(x / x.sum(), 4)


def normalize_eto_df(eto_df, eto_field="eto"):
    # Assuming eto_df has a datetime index or a 'datetime' column:
    df = eto_df.reset_index()
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    # Define the desired hour range (inclusive start, exclusive end)
    start_hour = 6
    end_hour = 18

    # Filter the DataFrame for only the hours within the specified range
    mask = (df["hour"] >= start_hour) & (df["hour"] <= end_hour)
    df_subset = df.loc[mask].copy()

    # Apply the normalization for each station and each day
    df_subset["daily_min_max_ETo"] = df_subset.groupby(["stationid", "date"])[
        eto_field
    ].transform(norm_minmax_dly_et)
    df_subset["daily_ETo_normed"] = df_subset.groupby(["stationid", "date"])[
        "daily_min_max_ETo"
    ].transform(norm_dly_et)
    df_subset = df_subset.set_index(["stationid", "datetime"])

    df_final = pd.merge(
        eto_df,
        df_subset[["daily_min_max_ETo", "daily_ETo_normed"]],
        how="left",
        left_index=True,
        right_index=True,
    )
    # df_final = df.merge(df_subset[['datetime', 'daily_ETo_normed']], on='datetime', how='left')
    df_final["daily_ETo_normed"] = df_final["daily_ETo_normed"].fillna(0)
    return df_final


def calc_nldas_refet(
    date, hour, nldas_out_dir, latitude, longitude, elevation, zm, station
):
    """
    Normalize hourly ETo values by day and station using min-max and sum-based scaling.

    This function adds two new normalized columns to an input ETo DataFrame:
    - `daily_min_max_ETo`: Min-max normalized hourly ETo (0–1 range within each day).
    - `daily_ETo_normed`: Sum-normalized values such that daily total equals 1 (unitless weights).

    The normalization is applied only to hours between 6 AM and 6 PM (inclusive), and the
    results are merged back into the original DataFrame. Missing values are filled with 0.

    Parameters
    ----------
    eto_df : pandas.DataFrame
        DataFrame containing at least a datetime index or column, an ETo column, and a
        'stationid' index or column. Typically produced from NLDAS extraction routines.
    eto_field : str, optional
        Column name in `eto_df` containing the ETo values to normalize. Default is "eto".

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with two new columns added:
        - 'daily_min_max_ETo'
        - 'daily_ETo_normed'

    Notes
    -----
    - Time window: normalization is applied only between 6:00 and 18:00 (local time).
    - Grouping is performed on both 'stationid' and calendar date.
    - `daily_ETo_normed` values are set to 0 outside the time window or when insufficient data exists.

    Examples
    --------
    >>> norm_df = normalize_eto_df(eto_df)
    >>> norm_df[["eto", "daily_ETo_normed"]].head()
    """
    YYYY = date.year
    DOY = date.timetuple().tm_yday
    MM = date.month
    DD = date.day
    HH = hour
    # already ensured to exist above loop
    nldas_outf_path = nldas_out_dir / f"{YYYY}_{MM:02}_{DD:02}_{HH:02}.grb"
    # open grib and extract needed data at nearest gridcell, calc ETr/ETo anf append to time series
    ds = xarray.open_dataset(nldas_outf_path, engine="pynio").sel(
        lat_110=latitude, lon_110=longitude, method="nearest"
    )
    # calculate hourly ea from specific humidity
    pair = ds.get("PRES_110_SFC").data / 1000  # type: ignore # nldas air pres in Pa convert to kPa
    sph = ds.get("SPF_H_110_HTGL").data  # kg/kg # type: ignore
    ea = refet.calcs._actual_vapor_pressure(  # type: ignore
        q=sph, pair=pair
    )  # ea in kPa # type: ignore
    # calculate hourly wind
    wind_u = ds.get("U_GRD_110_HTGL").data  # type: ignore
    wind_v = ds.get("V_GRD_110_HTGL").data  # type: ignore
    wind = np.sqrt(wind_u**2 + wind_v**2)
    # get temp convert to C
    temp = ds.get("TMP_110_HTGL").data - 273.15  # type: ignore
    # get rs
    rs = ds.get("DSWRF_110_SFC").data  # type: ignore
    unit_dict = {"rs": "w/m2"}
    # create refet object for calculating

    refet_obj = refet.Hourly(
        tmean=temp,
        ea=ea,
        rs=rs,
        uz=wind,
        zw=zm,
        elev=elevation,
        lat=latitude,
        lon=longitude,
        doy=DOY,
        time=HH,
        method="asce",
        input_units=unit_dict,
    )  # HH must be int

    out_dir = Path("All_output") / "AMF" / f"{station}"

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    # this one is saved under the site_ID subdir
    nldas_ts_outf = out_dir / f"nldas_ETr.csv"
    # save/append time series of point data
    dt = pd.to_datetime(f"{YYYY}-{MM}-{DD} {HH}:00")
    ETr_df = pd.DataFrame(
        columns=["ETr", "ETo", "ea", "sph", "wind", "pair", "temp", "rs"]
    )
    ETr_df.loc[dt, "ETr"] = refet_obj.etr()[0]
    ETr_df.loc[dt, "ETo"] = refet_obj.eto()[0]
    ETr_df.loc[dt, "ea"] = ea[0]
    ETr_df.loc[dt, "sph"] = sph
    ETr_df.loc[dt, "wind"] = wind
    ETr_df.loc[dt, "pair"] = pair
    ETr_df.loc[dt, "temp"] = temp
    ETr_df.loc[dt, "rs"] = rs
    ETr_df.index.name = "date"

    # if first run save file with individual datetime (hour data) else open and overwrite hour
    if not nldas_ts_outf.is_file():
        ETr_df.round(4).to_csv(nldas_ts_outf)
        nldas_df = ETr_df.round(4)
    else:
        curr_df = pd.read_csv(nldas_ts_outf, index_col="date", parse_dates=True)
        curr_df.loc[dt] = ETr_df.loc[dt]  # type: ignore
        curr_df.round(4).to_csv(nldas_ts_outf)
        nldas_df = curr_df.round(4)

    return nldas_df


def outline_valid_cells(raster_path, out_file=None):
    """
    Extract and outline the valid data extent of a raster as polygons.

    This function reads a raster file and identifies all non-nodata or non-NaN cells
    as valid. It then constructs vector polygons outlining the spatial extent of
    these valid regions. The result is returned as a GeoDataFrame and can optionally
    be saved to a GeoJSON file.

    Parameters
    ----------
    raster_path : str or pathlib.Path
        Path to the input raster file (e.g., GeoTIFF).
    out_file : str or pathlib.Path, optional
        Path to save the resulting polygon geometry as a GeoJSON file.
        If None (default), the result is not saved to disk.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing one or more polygons representing the extent
        of valid cells. If multiple regions exist, they are dissolved into a single
        geometry.

    Notes
    -----
    - If the raster has a defined `nodata` value, cells with this value are treated as invalid.
    - If no `nodata` is defined, the function uses NaN to identify invalid cells.
    - The output geometry is in the same CRS as the input raster.
    - Output is dissolved into one polygon using `union_all()` to form a contiguous boundary.

    Examples
    --------
    >>> gdf = outline_valid_cells("elevation.tif")
    >>> gdf.plot()

    >>> outline_valid_cells("et_2022.tif", out_file="valid_extent.geojson")
    Saved valid extent polygons to 'valid_extent.geojson'.
    """
    # 1. Open the raster
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # read the first band
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # 2. Build a "valid data" mask
    #    - If nodata is defined, use that
    #    - Otherwise, mask out NaNs
    if nodata is not None:
        valid_mask = data != nodata
    else:
        # If no nodata is declared, fall back to ignoring NaNs
        valid_mask = ~np.isnan(data)

    # Convert to 8-bit integer (1 for valid, 0 for invalid)
    valid_mask = valid_mask.astype(np.uint8)

    # 3. Polygonize the valid mask using rasterio.features.shapes
    shapes_and_vals = rasterio.features.shapes(valid_mask, transform=transform)

    polygons = []
    for geom, val in shapes_and_vals:
        # val == 1 means it's a valid area
        if val == 1:
            polygons.append(shape(geom))

    # 4. Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    # 5. (Optional) dissolve all polygons into one geometry if desired
    #    (useful if you only want a single boundary)
    if not gdf.empty:
        unioned_geom = gdf.geometry.union_all()
        gdf = gpd.GeoDataFrame(geometry=[unioned_geom], crs=gdf.crs)

    # 6. (Optional) save to file
    if out_file:
        gdf.to_file(out_file, driver="GeoJSON")
        print(f"Saved valid extent polygons to '{out_file}'.")

    return gdf
