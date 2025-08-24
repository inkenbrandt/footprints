from typing import Self

import pandas as pd
import geopandas as gpd
import numpy as np

from pyproj import Transformer
from rasterio import features
from rasterio.features import shapes
from rasterio.transform import from_origin, Affine
from shapely import GeometryCollection, MultiPolygon, convex_hull
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapelysmooth import taubin_smooth
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from .adapters import (
    KM2001Params,
    WangParams,
    km2001_isopleths,
    wang2006_isopleths,
    kljun2015_isopleths,
)


BOUNDARY_LAYER_HEIGHT = 2000
CONTOUR_SRC_PCT = [90]


class GIS_Files:
    _req_columns = [
        "date_time",
        "WS",
        "USTAR",
        "WD",
        "V_SIGMA",
        "MO_LENGTH",
        "instr_height_m",
        "canopy_height_m",
        "Z0_roughness",
    ]

    def __init__(self, tower_location: tuple[float, float]):
        """
        Initialize the Footprint object.

        Parameters
        ----------
        tower_location : tuple[float, float]
            The latitude and longitude of the tower location.
        tower_spec : dict
            A dictionary with the following required keys:
                - zm: The height of the measurement in meters.
                - z0 or umean: The surface roughness length in meters or the mean wind speed in meters per second.

        Notes
        -----
        The Footprint object is initialized with the tower location and tower specification.
        The UTM coordinates of the tower location are calculated and stored as easting and northing.
        """
        self.latitude = tower_location[0]
        self.longitude = tower_location[1]
        self._ws_limit_quantile = 0.95

        self.utm_crs: str = ""
        self.easting: float = 0.0
        self.northing: float = 0.0
        self.raw_raster: np.ndarray = np.ndarray(shape=(0, 0), dtype=np.uint32)
        self.raster: np.ndarray | None = None
        self.data: pd.DataFrame | None = None
        self.reference_eto: pd.DataFrame | None = None
        self.geometry: gpd.GeoDataFrame | None = None
        self.transform: Affine | None = None
        self.data_points: dict[str, int] = {}
        self.daily_timeseries: gpd.GeoDataFrame | None = None

        self._to_utm()

    @property
    def ws_limit_quantile(self) -> float:
        return self._ws_limit_quantile

    @ws_limit_quantile.setter
    def ws_limit_quantile(self, value: float):
        self._ws_limit_quantile = value

    @property
    def boundary_layer_height(self) -> float:
        return self.__dict__.get("_boundary_layer_height") or BOUNDARY_LAYER_HEIGHT

    @boundary_layer_height.setter
    def boundary_layer_height(self, value: float):
        self._boundary_layer_height = value

    @property
    def contour_src_pct(self):
        return self.__dict__.get("_contour_src_pct") or CONTOUR_SRC_PCT

    @contour_src_pct.setter
    def contour_src_pct(self, value: float | list[float | int]):
        self._contour_src_pct = value

    def __repr__(self):
        if self.geometry is not None:
            return f"Footprint with {len(self.geometry)} polygons.\n{self.geometry.head(1)}"

        return f"Footprint for latitude: {self.latitude}, longitude: {self.longitude}"

    def _to_utm(self):
        utm_zone = (
            int((self.longitude + 180) / 6) + 1
        )  # calculate UTM zone based on longitude
        self.utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"  # create UTM CRS

        transformer = Transformer.from_crs("epsg:4326", self.utm_crs)
        self.easting, self.northing = transformer.transform(self.latitude, self.longitude)  # type: ignore

    def _row_to_shapes(self, row, rs, model_name, boundary_layer_height):
        # Build model params
        if model_name == "km2001":
            params = KM2001Params(
                zm=row["zm"],
                z0=row["z0"],
                L=row["L"],
                ustar=row["u_star"],
                umean=row["u_mean"],
                sigma_v=row["sigma_v"],
            )
            res = km2001_isopleths(params, rs)
        elif model_name == "wang2006":
            params = WangParams(
                zm=row["zm"],
                h=boundary_layer_height,
                L=row["L"],
                sigma_v=row["sigma_v"],
                U=row["u_mean"],
            )
            res = wang2006_isopleths(params, rs)
        else:  # "kljun"
            res = kljun2015_isopleths(row, rs)

        # return one polygon per requested r (take first / only r by default)
        rkey = int(rs[0])
        return row["date_time"], res[rkey][0], res[rkey][1]

    def draw(self, max_rows: int, model: str = "kljun") -> "Self":
        if self.data is None:
            raise ValueError("data must be set before drawing.")
        df = self.data.copy()
        if max_rows > 0:
            df = df.iloc[:max_rows]

        # Parallelize time steps (safe because each row is independent)
        worker = partial(
            self._row_to_shapes,
            rs=self.contour_src_pct,
            model_name=model,
            boundary_layer_height=self.boundary_layer_height,
        )
        rows = df.to_dict(orient="records")

        results = []
        with ProcessPoolExecutor() as ex:
            for out in tqdm(
                ex.map(worker, rows), total=len(rows), desc="Half-hourly Footprints"
            ):
                if out is not None:
                    results.append(out)

        # Build polygons once at the end (fast path)

        times, polys = [], []
        for t, xr, yr in results:
            # translate to UTM absolute meters just once
            polys.append(Polygon(np.c_[xr + self.easting, yr + self.northing]))
            times.append(t)

        self.geometry = gpd.GeoDataFrame(
            {"times": times, "geometry": polys}, crs=self.utm_crs
        )
        if not self.geometry.empty:
            minx, miny, maxx, maxy = self.geometry.union_all().bounds
            print(
                f"Overall extent: minx = {minx}, miny = {miny}, maxx = {maxx}, maxy = {maxy}"
            )
        else:
            print(
                "No polygons were processed. Input data did not return any valid results."
            )

        # keep your daily counting logic if you want, unchanged…
        self.rows_per_day = (
            self.data.groupby(["yyyy", "mm", "day"]).size().reset_index(name="count")
        )
        self.data_points = {}
        for dt in times:
            k = str(pd.to_datetime(dt).date())
            self.data_points[k] = self.data_points.get(k, 0) + 1
        return self

    def attach(
        self, data: pd.DataFrame, reference_eto: pd.DataFrame | None = None
    ) -> Self:
        """
        Attach a pandas DataFrame to the Footprint object.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas DataFrame containing the required columns.

        Returns
        -------
        Self
            The Footprint object with the attached data.

        Notes
        -----
        The required columns are:
        - date_time: A datetime column.
        - instr_height_m: The instrument height in meters.
        - canopy_height_m: The canopy height in meters (used as displacement height).
        - WS: The mean wind speed in meters per second.
        - USTAR: The friction velocity in meters per second.
        - WD: The wind direction in degrees.
        - V_SIGMA: The lateral velocity in meters per second.
        - MO_LENGTH: The Obukhov length in meters.

        The `attach` method will subset the data to only include data between 9 AM and 3PM.
        """
        if False in [col in data.columns for col in self._req_columns]:
            raise ValueError(
                f"Data does not contain at least one of the required columns: {self._req_columns}"
            )

        self.data = data[self._req_columns].copy()

        # # Filter WS (wind speed) so outliers are removed.
        # wind_lim = self.data["WS"].quantile(self._ws_limit_quantile)
        # self.data["WS"] = self.data["WS"].clip(upper=wind_lim)

        # Convert to datetime.
        self.data["date_time"] = pd.to_datetime(self.data["date_time"])

        if reference_eto is not None:
            start_date = self.data["date_time"].dt.date.min()
            end_date = self.data["date_time"].dt.date.max()
            reference_eto["date"] = pd.to_datetime(reference_eto["date"])
            self.reference_eto = reference_eto[
                reference_eto["date"].dt.date.between(start_date, end_date)
            ][["date", "gridMET_ETo"]]

        # Separate year month day etc.
        self.data["yyyy"] = self.data["date_time"].dt.year
        self.data["mm"] = self.data["date_time"].dt.month
        self.data["day"] = self.data["date_time"].dt.day
        self.data["HH"] = self.data["date_time"].dt.hour
        self.data["MM"] = self.data["date_time"].dt.minute

        # Rearrange columns.
        self.data = self.data[
            [
                "date_time",
                "yyyy",
                "mm",
                "day",
                "HH",
                "MM",
                "instr_height_m",
                "canopy_height_m",
                "Z0_roughness",
                "WS",
                "MO_LENGTH",
                "V_SIGMA",
                "USTAR",
                "WD",
            ]
        ]

        # Rename columns.
        self.data = self.data.rename(
            columns={
                "instr_height_m": "zm",
                "canopy_height_m": "d",
                "Z0_roughness": "z0",
                "WS": "u_mean",
                "MO_LENGTH": "L",
                "V_SIGMA": "sigma_v",
                "USTAR": "u_star",
                "WD": "wind_dir",
            }
        )

        # Subset to only include data between 9 AM and 3PM.
        self.data = self.data[(self.data["HH"] >= 9) & (self.data["HH"] <= 15)]

        self.data = self.data.reset_index()

        return self

    def rasterize(self, resolution: int = 1, threshold: float = 0.0) -> Self:
        """
        Rasterize the footprint polygons to a numpy array.

        Parameters
        ----------
        resolution : int
            The desired resolution of the output raster in meters per pixel.
            Default is 1.

        Returns
        -------
        Self
            The Footprint object with the rasterized footprint polygons.

        Raises
        ------
        ValueError
            If the geometry is not set before calling the method.

        Notes
        -----
        The raster is created by accumulating the rasterized footprint polygons
        from the GeoDataFrame `geometry`. The output raster has a shape based
        on the bounds of the geometry and is stored in the `raster` attribute.
        The transformation matrix is stored in the `transform` attribute.
        """
        if self.geometry is None:
            raise ValueError("geometry must be set before rasterizing.")
        poly_data = self.geometry.reset_index().copy()

        minx, miny, maxx, maxy = poly_data.union_all().bounds
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        poly_data["date"] = pd.to_datetime(poly_data["times"]).dt.date
        self.transform = from_origin(minx, maxy, resolution, resolution)
        raster = np.zeros((height, width, poly_data["date"].nunique()), dtype=np.uint8)
        self.raw_raster = np.zeros((height, width), dtype=np.uint32)

        i = 0
        skipped = 0
        daily_polygon = []
        daily_timestamps = []

        def calc_daily_overlaps(group):
            nonlocal i
            nonlocal skipped
            nonlocal raster
            nonlocal daily_polygon
            nonlocal daily_timestamps

            assert self.transform
            group_date = group["date"].iloc[0]
            valid_rows_count = self.data_points[str(group_date)]

            daily_raster = np.zeros((height, width), dtype=np.uint8)
            for index, row in group.iterrows():
                try:
                    polygon = [row["geometry"]]
                    row_raster = features.rasterize(
                        polygon,
                        out_shape=(height, width),
                        transform=self.transform,
                        all_touched=True,
                        fill=0,
                    )

                    daily_raster += row_raster.astype(daily_raster.dtype)
                    self.raw_raster += row_raster.astype(self.raw_raster.dtype)
                except Exception as e:
                    print(f"Error in row {index}: {e}")

            # Weigh the overlap counts by the total number of valid rows for the day.
            # First, get the date of the current row from the datapoints dictionary.
            # Then, divide the overlap count by the valid row count.
            daily_raster = np.divide(daily_raster, valid_rows_count, dtype=np.float16)
            # If reference eto is provided, weigh the overlap counts by the fraction of eto data from total eto data.
            if self.reference_eto is not None:
                date_mask = self.reference_eto["date"].dt.date == group_date
                eto = self.reference_eto[date_mask]["gridMET_ETo"].values[0]

                daily_raster = daily_raster.astype(np.float32)
                raster = raster.astype(np.float32)
                daily_raster = daily_raster * (
                    eto / self.reference_eto["gridMET_ETo"].sum()
                )

            # Timeseries construction #
            # First, normalize a copy of the daily raster so the effect of overlap_threshold can be applied.
            daily_raster_copy = daily_raster.copy()
            daily_raster_copy = np.divide(
                daily_raster_copy, daily_raster_copy.max(), dtype=np.float32
            )
            # Create shapes from raster data.
            shapes_gen = shapes(
                daily_raster_copy,
                mask=daily_raster_copy > threshold,
                transform=self.transform,
            )
            # Create polygons from shapes.
            polygons = unary_union([shape(geom) for geom, _ in shapes_gen])
            # Append to the daily polygon list.
            daily_polygon.append(polygons)
            daily_timestamps.append(group_date)

            raster[:, :, i] = daily_raster
            i += 1

        tqdm.pandas(desc="Rasterizing Daily Polygons", position=0, leave=True)
        poly_data.groupby("date").progress_apply(calc_daily_overlaps)  # type: ignore
        self.daily_timeseries = gpd.GeoDataFrame(
            {"time": daily_timestamps, "geometry": daily_polygon}, crs=self.utm_crs
        )

        # Sum the raster along the third axis to get the total overlap count.
        raster = raster.sum(axis=2, dtype=raster.dtype)
        # Then normalize based on max weighed overlaps.
        self.raster = np.divide(raster, raster.max(), dtype=np.float32)

        return self

    def polygonize(
        self,
        threshold: float = 0.0,
        smoothing_factor: int = 50,
        merge_disjointed: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Create a single polygon from rasters that meet overlap threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold for the overlap count to be considered a valid polygon.
            Defaults to 0.0.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing a single polygon that meets the overlap
            threshold, with a single row and column named 'geometry'.

        Raises
        ------
        ValueError
            If the raster is not set before calling the method.

        Notes
        -----
        The polygon is created by masking the raster with the given threshold,
        and then creating a single polygon from the resulting shapes. The
        polygon is then smoothed to reduce the number of vertices. The
        GeoDataFrame is created with a single row and column named 'geometry'.
        """
        if self.raster is None:
            raise ValueError("raster must be set before polygonizing.")
        assert self.transform

        mask = self.raster > threshold

        shapes_gen = shapes(self.raster, mask=mask, transform=self.transform)
        polygons = [shape(geom) for geom, _ in shapes_gen]

        combined_polygon = unary_union(polygons)

        if isinstance(combined_polygon, MultiPolygon):
            footprint_segments = []
            for geo in combined_polygon.geoms:
                footprint_segments.append(unary_union(geo))
            combined_polygon = footprint_segments

        combined_polygon = (
            combined_polygon
            if isinstance(combined_polygon, list)
            else [combined_polygon]
        )

        if merge_disjointed and len(combined_polygon) > 1:
            print("Performing convex hull union...")
            # First, create a multipolygon containing all the polygons.
            mp = MultiPolygon(GeometryCollection(combined_polygon))
            # Then perform convex hull on it.
            combined_polygon = [convex_hull(mp)]

        gdf = gpd.GeoDataFrame(geometry=combined_polygon, crs=self.utm_crs)

        try:
            tqdm.pandas(desc="Smoothing Polygon(s)", position=0, leave=True)
            # Smoothen each polygon within the footprint.
            gdf["geometry"] = gdf["geometry"].progress_apply(
                lambda x: taubin_smooth(x, steps=smoothing_factor)
            )
        except Exception as e:
            print(f"Error in smoothing polygon: {e}")

        return gdf
