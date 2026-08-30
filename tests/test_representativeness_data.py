"""
Earth Engine retrieval tests for :mod:`fluxfootprints.representativeness_data`.

The module's whole job is to turn Earth Engine into rasters that
``representativeness._align_raster`` will accept, so nothing here talks to
Earth Engine. Instead ``sys.modules["ee"]`` is replaced by a small numpy-backed
fake that implements the handful of operations the module actually calls --
band selection, scale and offset, ``expression``, the QA bit tests, masking,
and ``computePixels``. That is enough to check the three things that would
otherwise only break against the live service:

* the arithmetic -- Eq. 4 evaluated on *reflectance*, i.e. after the
  Collection 2 scale factor and offset, and the ``QA_PIXEL`` bits read as the
  product guide numbers them;
* the grid -- the ``PixelGrid`` handed to Earth Engine and the coordinates
  handed back describe the same square, centred on the tower, in the CRS asked
  for;
* the contract with ``_align_raster`` -- dims, CRS, and ``nan`` in place of the
  zeros Earth Engine returns for masked pixels.

The sensor set is checked the same way: Landsat 7 must never be queried, since
Chu et al. (2021) exclude it.

The one thing asserted without the fake is the import guard: importing
:mod:`fluxfootprints` must not pull in ``earthengine-api``.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("rioxarray")

import rioxarray  # noqa: F401  - registers the .rio accessor asserted on below

from fluxfootprints import representativeness_data as rd

TOWER_X = 425000.0
TOWER_Y = 4428000.0
UTM = "EPSG:32612"

#: Digital numbers chosen so Eq. 4 comes out at exactly 0.8 once the
#: Collection 2 scale factor and offset are applied: reflectances of
#: NIR 0.625, RED 0.075, BLUE 0.0475 give 2.5 * 0.55 / 1.71875.
NIR_DN, RED_DN, BLUE_DN = 30000, 10000, 9000
EXPECTED_EVI = 0.8

#: ``QA_PIXEL`` values, by the bit each sets.
QA_CLEAR = 0
QA_FILL = 1 << 0
QA_DILATED = 1 << 1
QA_CIRRUS = 1 << 2
QA_CLOUD = 1 << 3
QA_SHADOW = 1 << 4
QA_SNOW = 1 << 5


# ----------------------------
# A numpy-backed stand-in for ee
# ----------------------------

#: Every reduceRegion call the fake sees, so a test can assert on how the
#: cloud fraction was asked for. Cleared by the ``fake_ee`` fixture.
REDUCTIONS: list = []


class FakeImage:
    """An ee.Image over numpy arrays: one values array and one mask per band."""

    def __init__(self, bands, masks=None, properties=None):
        self.bands = {name: np.asarray(values) for name, values in bands.items()}
        self.masks = {
            name: (
                np.ones_like(values, dtype=bool)
                if masks is None or name not in masks
                else np.asarray(masks[name], dtype=bool)
            )
            for name, values in self.bands.items()
        }
        self.properties = dict(properties or {})

    # -- structure --------------------------------------------------
    def _names(self):
        return list(self.bands)

    def select(self, names):
        wanted = [names] if isinstance(names, str) else list(names)
        return FakeImage(
            {name: self.bands[name] for name in wanted},
            {name: self.masks[name] for name in wanted},
            self.properties,
        )

    def rename(self, names):
        wanted = [names] if isinstance(names, str) else list(names)
        assert len(wanted) == len(self.bands)
        return FakeImage(
            dict(zip(wanted, self.bands.values())),
            dict(zip(wanted, self.masks.values())),
            self.properties,
        )

    def addBands(self, other):
        return FakeImage(
            {**self.bands, **other.bands},
            {**self.masks, **other.masks},
            self.properties,
        )

    def set(self, key, value):
        return FakeImage(self.bands, self.masks, {**self.properties, key: value})

    def get(self, key):
        return self.properties.get(key)

    # -- arithmetic -------------------------------------------------
    def _elementwise(self, function):
        return FakeImage(
            {name: function(values) for name, values in self.bands.items()},
            self.masks,
            self.properties,
        )

    def multiply(self, factor):
        return self._elementwise(lambda values: values * factor)

    def add(self, term):
        return self._elementwise(lambda values: values + term)

    def rightShift(self, bits):
        return self._elementwise(lambda values: values.astype(np.int64) >> bits)

    def bitwiseAnd(self, other):
        return self._elementwise(lambda values: values.astype(np.int64) & other)

    def Or(self, other):
        (mine,) = self.bands.values()
        (theirs,) = other.bands.values()
        return FakeImage(
            {"or": (mine.astype(bool) | theirs.astype(bool)).astype(np.int64)},
            {"or": next(iter(self.masks.values())) & next(iter(other.masks.values()))},
        )

    def Not(self):
        return self._elementwise(lambda values: (~values.astype(bool)).astype(np.int64))

    def expression(self, expression, inputs):
        namespace = {
            key: next(iter(image.bands.values())) for key, image in inputs.items()
        }
        masks = [next(iter(image.masks.values())) for image in inputs.values()]
        combined = masks[0]
        for mask in masks[1:]:
            combined = combined & mask
        with np.errstate(divide="ignore", invalid="ignore"):
            values = eval(expression, {"__builtins__": {}}, namespace)
        return FakeImage({"expression": np.asarray(values)}, {"expression": combined})

    # -- masking ----------------------------------------------------
    def mask(self):
        return FakeImage(
            {name: mask.astype(np.float64) for name, mask in self.masks.items()},
            {name: np.ones_like(mask, dtype=bool) for name, mask in self.masks.items()},
        )

    def updateMask(self, other):
        (keep,) = other.bands.values()
        keep = keep.astype(bool)
        return FakeImage(
            self.bands,
            {name: mask & keep for name, mask in self.masks.items()},
            self.properties,
        )

    def unmask(self, fill):
        return FakeImage(
            {
                name: np.where(self.masks[name], values, fill)
                for name, values in self.bands.items()
            },
            {name: np.ones_like(mask, dtype=bool) for name, mask in self.masks.items()},
            self.properties,
        )

    def reduceRegion(self, **kwargs):
        # The fake array *is* the target disc, so the region argument is
        # honoured by construction and the reduction is over everything valid.
        REDUCTIONS.append(kwargs)
        means = {}
        for name, values in self.bands.items():
            mask = self.masks[name]
            means[name] = float(values[mask].mean()) if mask.any() else None
        return SimpleNamespace(get=means.get)


class FakeCollection:
    """An ee.ImageCollection: an ordered list of FakeImage, filtered in python."""

    def __init__(self, items, collection_id=None, log=None):
        self.items = list(items)
        self.collection_id = collection_id
        self.log = log if log is not None else []

    def filterDate(self, start, end):
        kept = [
            image
            for image in self.items
            if start <= pd.Timestamp(image.get("date")).strftime("%Y-%m-%d") < end
        ]
        return FakeCollection(kept, self.collection_id, self.log)

    def filterBounds(self, geometry):
        return self

    def map(self, function):
        return FakeCollection(
            [function(image) for image in self.items], self.collection_id, self.log
        )

    def filter(self, predicate):
        return FakeCollection(
            [image for image in self.items if predicate(image)],
            self.collection_id,
            self.log,
        )

    def size(self):
        return SimpleNamespace(getInfo=lambda: len(self.items))

    def first(self):
        return self.items[0]

    def aggregate_array(self, key):
        return SimpleNamespace(getInfo=lambda: [item.get(key) for item in self.items])


def _fit(values, shape):
    """Tile or crop a fake band onto the grid the request asked for."""
    height, width = shape
    rows = -(-height // values.shape[0])
    columns = -(-width // values.shape[1])
    return np.tile(values, (rows, columns))[:height, :width]


def make_ee(catalog, collections, requests, queried):
    """
    Build a stand-in ``ee`` module over the given fake data.

    Parameters
    ----------
    catalog : dict
        Full Earth Engine asset id -> FakeImage, for id-based ``ee.Image``.
    collections : dict
        Collection id -> list of FakeImage.
    requests : list
        Appended to with every ``computePixels`` request, for inspection.
    queried : list
        Appended to with every collection id opened, so a test can assert that
        Landsat 7 was never asked for.
    """
    module = ModuleType("ee")

    def image(source):
        if isinstance(source, FakeImage):
            return source
        return catalog[source]

    def image_collection(collection_id):
        queried.append(collection_id)
        return FakeCollection(collections.get(collection_id, []), collection_id)

    def compute_pixels(request):
        requests.append(request)
        grid = request["grid"]
        height = grid["dimensions"]["height"]
        width = grid["dimensions"]["width"]
        source = request["expression"]
        dtype = [(name, "f8") for name in source.bands]
        out = np.zeros((height, width), dtype=dtype)
        for name, values in source.bands.items():
            # Earth Engine returns a masked pixel as zero; so does the fake.
            filled = np.where(source.masks[name], values, 0.0)
            out[name] = _fit(filled, (height, width))
        return out

    module.Image = image
    module.ImageCollection = image_collection
    module.Filter = SimpleNamespace(
        eq=lambda key, value: (lambda item: item.get(key) == value),
        lt=lambda key, value: (
            lambda item: item.get(key) is not None and item.get(key) < value
        ),
    )
    module.Reducer = SimpleNamespace(mean=lambda: "mean")
    module.Geometry = SimpleNamespace(
        Point=lambda coords, proj=None: SimpleNamespace(
            buffer=lambda radius: SimpleNamespace(coords=coords, radius=radius)
        )
    )
    module.Feature = lambda geometry, properties: {"properties": dict(properties)}
    module.FeatureCollection = lambda collection: SimpleNamespace(
        getInfo=lambda: {"features": list(collection.items)}
    )
    module.data = SimpleNamespace(_credentials=object(), computePixels=compute_pixels)
    return module


@pytest.fixture
def fake_ee(monkeypatch):
    """Install a stand-in ``ee``; tests fill its catalog through the handle."""
    REDUCTIONS.clear()
    state = SimpleNamespace(
        catalog={}, collections={}, requests=[], queried=[], module=None
    )
    state.module = make_ee(
        state.catalog, state.collections, state.requests, state.queried
    )
    monkeypatch.setitem(sys.modules, "ee", state.module)
    monkeypatch.setattr(rd, "_INITIALIZED", False)
    return state


def landsat_scene(bands, qa, date, shape=(4, 4)):
    """A fake Collection 2 scene: three reflectance bands and QA_PIXEL."""
    blue, red, nir = bands
    qa_values = (
        np.full(shape, qa, dtype=np.int64) if np.isscalar(qa) else np.asarray(qa)
    )
    return FakeImage(
        {
            "SR_B2": np.full(shape, blue, dtype=np.float64),
            "SR_B4": np.full(shape, red, dtype=np.float64),
            "SR_B5": np.full(shape, nir, dtype=np.float64),
            "QA_PIXEL": qa_values,
        },
        properties={
            "system:index": f"LC08_{date.replace('-', '')}",
            "system:time_start": int(pd.Timestamp(date).value // 10**6),
            "date": date,
        },
    )


# ----------------------------
# Import guard
# ----------------------------
def test_package_import_does_not_require_earthengine():
    """The core package must import with earthengine-api absent."""
    assert "ee" not in sys.modules
    import fluxfootprints  # noqa: F401

    assert "ee" not in sys.modules


def test_missing_earthengine_names_the_extra(monkeypatch):
    """The ImportError has to say how to install the dependency."""
    monkeypatch.setitem(sys.modules, "ee", None)
    monkeypatch.setattr(rd, "_INITIALIZED", False)
    with pytest.raises(ImportError, match="earthengine-api"):
        rd.initialize()


def test_initialize_leaves_an_existing_session_alone(fake_ee, monkeypatch):
    """Credentials already in place must not be re-initialised over."""
    calls = []
    fake_ee.module.Initialize = lambda **kwargs: calls.append(kwargs)
    assert rd.initialize() is fake_ee.module
    assert calls == []


# ----------------------------
# CRS handling
# ----------------------------
def test_crs_spec_prefers_the_authority_code():
    parsed, spec = rd._crs_spec(UTM)
    assert spec == {"crsCode": "EPSG:32612"}
    assert parsed.to_epsg() == 32612


def test_crs_spec_falls_back_to_wkt():
    """A CRS with no EPSG code still has to reach Earth Engine."""
    from pyproj import CRS

    bespoke = CRS.from_proj4("+proj=tmerc +lat_0=0 +lon_0=-111 +k=0.9996 +units=m")
    _, spec = rd._crs_spec(bespoke)
    assert set(spec) == {"crsWkt"}
    assert "PROJCRS" in spec["crsWkt"] or "PROJCS" in spec["crsWkt"]


def test_crs_spec_rejects_degrees():
    with pytest.raises(ValueError, match="geographic"):
        rd._crs_spec("EPSG:4326")


def test_crs_spec_rejects_nonsense():
    with pytest.raises(ValueError, match="coordinate reference system"):
        rd._crs_spec("not a crs")


# ----------------------------
# Grid construction
# ----------------------------
def test_pixel_grid_is_centred_on_the_tower():
    grid, xs, ys = rd._pixel_grid(
        TOWER_X, TOWER_Y, 300.0, 30.0, {"crsCode": UTM}, bands=2
    )
    assert grid["dimensions"] == {"width": 20, "height": 20}
    assert grid["affineTransform"]["translateX"] == TOWER_X - 300.0
    assert grid["affineTransform"]["translateY"] == TOWER_Y + 300.0
    assert grid["affineTransform"]["scaleY"] == -30.0
    assert grid["crsCode"] == UTM
    # Cell centres, so the outermost is half a cell inside the edge.
    assert xs[0] == pytest.approx(TOWER_X - 300.0 + 15.0)
    assert ys[0] == pytest.approx(TOWER_Y + 300.0 - 15.0)
    # Symmetric about the tower, and descending northwards.
    assert xs.mean() == pytest.approx(TOWER_X)
    assert ys.mean() == pytest.approx(TOWER_Y)
    assert np.all(np.diff(ys) < 0)


def test_pixel_grid_rounds_the_cell_count_up():
    """A radius that is not a whole number of cells must still cover it."""
    grid, xs, _ = rd._pixel_grid(0.0, 0.0, 100.0, 30.0, {"crsCode": UTM}, bands=2)
    assert grid["dimensions"]["width"] == 7  # ceil(200 / 30)
    assert xs[0] - 15.0 <= -100.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"radius": 0.0}, "radius"),
        ({"radius": -10.0}, "radius"),
        ({"scale": 0.0}, "scale"),
        ({"scale": np.nan}, "scale"),
        ({"x": np.nan}, "finite"),
    ],
)
def test_pixel_grid_rejects_bad_geometry(kwargs, match):
    call = {"x": 0.0, "y": 0.0, "radius": 300.0, "scale": 30.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        rd._pixel_grid(spec={"crsCode": UTM}, bands=2, **call)


def test_pixel_grid_refuses_an_oversized_request():
    """Over the response cap, the message has to name the knobs."""
    with pytest.raises(ValueError, match="response limit"):
        rd._pixel_grid(0.0, 0.0, 50_000.0, 1.0, {"crsCode": UTM}, bands=2)


# ----------------------------
# DataArray contract
# ----------------------------
def test_as_dataarray_is_ready_for_align_raster():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    valid = np.array([[1, 1], [0, 1]])
    array = rd._as_dataarray(
        values,
        valid,
        np.array([0.0, 30.0]),
        np.array([30.0, 0.0]),
        rd.CRS.from_user_input(UTM),
        name="evi",
        attrs={"source": "test"},
    )
    assert array.dims == ("y", "x")
    assert array.name == "evi"
    assert array.dtype == np.float64
    assert array.rio.crs.to_epsg() == 32612
    assert np.isnan(array.rio.nodata)
    assert np.isnan(array.values[1, 0])
    assert array.values[0, 0] == 1.0
    assert array.attrs["source"] == "test"


def test_as_dataarray_survives_align_raster():
    """The end of the contract: what comes back must actually warp."""
    from fluxfootprints.representativeness import _align_raster

    xs = TOWER_X + np.arange(-90.0, 91.0, 30.0)
    ys = TOWER_Y - np.arange(-90.0, 91.0, 30.0)
    source = rd._as_dataarray(
        np.full((ys.size, xs.size), 42.0),
        np.ones((ys.size, xs.size)),
        xs,
        ys,
        rd.CRS.from_user_input(UTM),
        name="nlcd",
        attrs={},
    )
    footprint = xr.DataArray(
        np.zeros((3, 3)),
        dims=("y", "x"),
        coords={
            "y": TOWER_Y + np.array([30.0, 0.0, -30.0]),
            "x": TOWER_X + np.array([-30.0, 0.0, 30.0]),
        },
        name="footprint",
    ).rio.write_crs(UTM)

    aligned, valid = _align_raster(source, footprint, categorical=True)
    assert aligned.dims == ("y", "x")
    assert bool(valid.all())
    assert np.allclose(aligned.values, 42.0)


# ----------------------------
# QA bits and Eq. 4
# ----------------------------
@pytest.mark.parametrize(
    "qa, obscured",
    [
        (QA_CLEAR, 0),
        (QA_FILL, 1),
        (QA_DILATED, 1),
        (QA_CIRRUS, 1),
        (QA_CLOUD, 1),
        (QA_SHADOW, 1),
        (QA_SNOW, 0),  # snow is a surface observation, not an obstruction
        (QA_CLOUD | QA_SNOW, 1),
    ],
)
def test_obscured_reads_the_qa_bits(qa, obscured):
    flagged = rd._obscured(FakeImage({"QA_PIXEL": np.array([[qa]], dtype=np.int64)}))
    assert int(next(iter(flagged.bands.values()))[0, 0]) == obscured


def test_obscured_counts_off_swath_pixels():
    """A pixel outside the scene is as unusable as a clouded one."""
    qa = FakeImage(
        {"QA_PIXEL": np.array([[QA_CLEAR, QA_CLEAR]], dtype=np.int64)},
        {"QA_PIXEL": np.array([[True, False]])},
    )
    flagged = next(iter(rd._obscured(qa).bands.values()))
    assert flagged.tolist() == [[0, 1]]


def test_evi_matches_equation_4_on_reflectance():
    """The scale factor and offset must be applied before Eq. 4."""
    scene = landsat_scene((BLUE_DN, RED_DN, NIR_DN), QA_CLEAR, "2015-06-01")
    result = rd._evi(scene, rd.LANDSAT_COLLECTIONS[1])
    assert set(result.bands) == {"evi", "valid"}
    assert result.bands["evi"] == pytest.approx(EXPECTED_EVI)
    assert result.bands["valid"].all()


def test_evi_masks_the_cloudy_pixels():
    qa = np.array([[QA_CLEAR, QA_CLOUD], [QA_SHADOW, QA_CLEAR]], dtype=np.int64)
    scene = landsat_scene((BLUE_DN, RED_DN, NIR_DN), qa, "2015-06-01", shape=(2, 2))
    result = rd._evi(scene, rd.LANDSAT_COLLECTIONS[1])
    assert result.bands["valid"].tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_landsat_seven_is_never_listed():
    """Chu et al. (2021) exclude ETM+; so must the collection table."""
    ids = [bands.collection for bands in rd.LANDSAT_COLLECTIONS]
    assert ids == ["LANDSAT/LT05/C02/T1_L2", "LANDSAT/LC08/C02/T1_L2"]
    assert not any("LE07" in identifier for identifier in ids)


# ----------------------------
# fetch_nlcd
# ----------------------------
@pytest.fixture
def nlcd_ee(fake_ee):
    """A three-epoch NLCD collection, 2019 holding a masked corner."""
    codes = np.array([[41.0, 42.0], [82.0, 90.0]])
    mask = np.array([[True, True], [True, False]])
    fake_ee.collections[rd.NLCD_COLLECTION] = [
        FakeImage({"landcover": codes}, {"landcover": mask}, {"system:index": year})
        for year in ("2013", "2016", "2019")
    ]
    return fake_ee


def test_fetch_nlcd_returns_class_codes_on_the_tower_grid(nlcd_ee):
    nlcd = rd.fetch_nlcd(TOWER_X, TOWER_Y, UTM, 2016, radius=30.0, scale=30.0)

    assert nlcd.name == "nlcd"
    assert nlcd.dims == ("y", "x")
    assert nlcd.shape == (2, 2)
    assert nlcd.rio.crs.to_epsg() == 32612
    # Class codes survive float64 exactly, and the masked corner is nan.
    assert nlcd.values[0].tolist() == [41.0, 42.0]
    assert nlcd.values[1, 0] == 82.0
    assert np.isnan(nlcd.values[1, 1])
    assert nlcd.attrs["year"] == 2016
    assert nlcd.attrs["source"].endswith("/2016")


def test_fetch_nlcd_requests_the_grid_it_returns(nlcd_ee):
    rd.fetch_nlcd(TOWER_X, TOWER_Y, UTM, 2016, radius=300.0, scale=30.0)
    (request,) = nlcd_ee.requests
    assert request["fileFormat"] == "NUMPY_NDARRAY"
    assert request["grid"]["crsCode"] == UTM
    assert request["grid"]["dimensions"] == {"width": 20, "height": 20}
    assert request["grid"]["affineTransform"]["translateX"] == TOWER_X - 300.0


def test_fetch_nlcd_asks_for_the_mask_alongside_the_codes(nlcd_ee):
    """Without the companion band a masked zero reads as a real class code."""
    rd.fetch_nlcd(TOWER_X, TOWER_Y, UTM, 2016, radius=30.0)
    (request,) = nlcd_ee.requests
    assert set(request["expression"].bands) == {"nlcd", "valid"}


def test_fetch_nlcd_lists_the_epochs_it_has(nlcd_ee):
    with pytest.raises(ValueError, match="2013, 2016, 2019"):
        rd.fetch_nlcd(TOWER_X, TOWER_Y, UTM, 2015)


def test_fetch_nlcd_rejects_a_geographic_crs(nlcd_ee):
    with pytest.raises(ValueError, match="geographic"):
        rd.fetch_nlcd(-111.9, 40.0, "EPSG:4326", 2016)


# ----------------------------
# fetch_landsat_evi
# ----------------------------
@pytest.fixture
def landsat_ee(fake_ee):
    """Two clear scenes out of order, one clouded, across both sensors."""
    # The July scene is half clouded, so it fails the paper's 1 % screening but
    # passes a threshold above its own obscured fraction.
    half_clouded = np.where(np.arange(16).reshape(4, 4) < 8, QA_CLOUD, QA_CLEAR).astype(
        np.int64
    )
    scenes = {
        "LANDSAT/LC08/C02/T1_L2": [
            landsat_scene((BLUE_DN, RED_DN, NIR_DN), QA_CLEAR, "2015-08-01"),
            landsat_scene((BLUE_DN, RED_DN, NIR_DN), half_clouded, "2015-07-01"),
        ],
        "LANDSAT/LT05/C02/T1_L2": [],
    }
    early = landsat_scene((BLUE_DN, RED_DN, NIR_DN), QA_CLEAR, "2011-05-01")
    early.bands["SR_B1"] = early.bands.pop("SR_B2")
    early.bands["SR_B3"] = early.bands.pop("SR_B4")
    early.bands["SR_B4"] = early.bands.pop("SR_B5")
    early.masks = {name: np.ones_like(v, dtype=bool) for name, v in early.bands.items()}
    early.properties["system:index"] = "LT05_20110501"
    scenes["LANDSAT/LT05/C02/T1_L2"].append(early)

    fake_ee.collections.update(scenes)
    for collection_id, images in scenes.items():
        for image in images:
            fake_ee.catalog[f"{collection_id}/{image.get('system:index')}"] = image
    return fake_ee


def test_fetch_landsat_evi_returns_a_scene_per_clear_retrieval(landsat_ee):
    scenes = rd.fetch_landsat_evi(
        TOWER_X, TOWER_Y, UTM, "2010-01-01", "2016-01-01", radius=60.0, scale=30.0
    )
    assert len(scenes) == 2  # the clouded July scene is screened out
    assert [scene.attrs["date"] for scene in scenes] == ["2011-05-01", "2015-08-01"]
    assert [scene.attrs["spacecraft"] for scene in scenes] == [
        "LANDSAT_5",
        "LANDSAT_8",
    ]
    for scene in scenes:
        assert scene.name == "evi"
        assert scene.dims == ("y", "x")
        assert scene.rio.crs.to_epsg() == 32612
        assert np.allclose(scene.values, EXPECTED_EVI)
        assert scene.attrs["obscured_fraction"] == pytest.approx(0.0)


def test_fetch_landsat_evi_carries_a_time_coordinate(landsat_ee):
    """The month a scene was retrieved in is what pairs it to a climatology."""
    scenes = rd.fetch_landsat_evi(
        TOWER_X, TOWER_Y, UTM, "2010-01-01", "2016-01-01", radius=60.0
    )
    times = [pd.Timestamp(scene.coords["time"].values) for scene in scenes]
    assert times == [pd.Timestamp("2011-05-01"), pd.Timestamp("2015-08-01")]
    assert times == sorted(times)


def test_fetch_landsat_evi_never_queries_landsat_seven(landsat_ee):
    rd.fetch_landsat_evi(TOWER_X, TOWER_Y, UTM, "2010-01-01", "2016-01-01", radius=60.0)
    assert landsat_ee.queried == [
        "LANDSAT/LT05/C02/T1_L2",
        "LANDSAT/LC08/C02/T1_L2",
    ]


def test_fetch_landsat_evi_screens_in_the_requested_projection(landsat_ee):
    """
    The obscured fraction must be reduced in the CRS and at the scale asked for.

    ``unmask`` composites against a constant image, which can leave the default
    projection as degrees; without an explicit ``crs`` the fraction would then
    be reduced at a resolution nothing in the module chose.
    """
    rd.fetch_landsat_evi(
        TOWER_X, TOWER_Y, UTM, "2010-01-01", "2016-01-01", radius=60.0, scale=30.0
    )
    assert REDUCTIONS
    for call in REDUCTIONS:
        assert call["crs"] == UTM
        assert call["scale"] == 30.0
        assert call["geometry"].radius == 60.0


def test_fetch_landsat_evi_honours_a_looser_cloud_threshold(landsat_ee):
    """The clouded scene comes back once the screening allows it."""
    scenes = rd.fetch_landsat_evi(
        TOWER_X, TOWER_Y, UTM, "2010-01-01", "2016-01-01", radius=60.0, max_cloud=0.6
    )
    assert len(scenes) == 3
    clouded = next(scene for scene in scenes if scene.attrs["date"] == "2015-07-01")
    assert clouded.attrs["obscured_fraction"] == pytest.approx(0.5)
    # Cloudy pixels come back as nan rather than as a plausible EVI.
    assert np.isnan(clouded.values).sum() == 8
    assert np.allclose(clouded.values[~np.isnan(clouded.values)], EXPECTED_EVI)


def test_fetch_landsat_evi_filters_by_date(landsat_ee):
    scenes = rd.fetch_landsat_evi(
        TOWER_X, TOWER_Y, UTM, "2015-01-01", "2016-01-01", radius=60.0
    )
    assert [scene.attrs["date"] for scene in scenes] == ["2015-08-01"]


def test_fetch_landsat_evi_returns_empty_when_nothing_is_clear(landsat_ee):
    scenes = rd.fetch_landsat_evi(
        TOWER_X, TOWER_Y, UTM, "2018-01-01", "2019-01-01", radius=60.0
    )
    assert scenes == []
    assert landsat_ee.requests == []


def test_fetch_landsat_evi_caps_the_number_of_requests(landsat_ee):
    with pytest.raises(ValueError, match="max_scenes"):
        rd.fetch_landsat_evi(
            TOWER_X,
            TOWER_Y,
            UTM,
            "2010-01-01",
            "2016-01-01",
            radius=60.0,
            max_scenes=1,
        )


@pytest.mark.parametrize("max_cloud", [-0.1, 1.5])
def test_fetch_landsat_evi_rejects_a_bad_threshold(landsat_ee, max_cloud):
    with pytest.raises(ValueError, match="max_cloud"):
        rd.fetch_landsat_evi(
            TOWER_X, TOWER_Y, UTM, "2010-01-01", "2016-01-01", max_cloud=max_cloud
        )


def test_fetch_landsat_evi_rejects_a_reversed_window(landsat_ee):
    with pytest.raises(ValueError, match="must not fall after"):
        rd.fetch_landsat_evi(TOWER_X, TOWER_Y, UTM, "2016-01-01", "2010-01-01")


def test_fetch_landsat_evi_accepts_timestamps(landsat_ee):
    """Dates may arrive as strings, dates, or pandas Timestamps."""
    scenes = rd.fetch_landsat_evi(
        TOWER_X,
        TOWER_Y,
        UTM,
        pd.Timestamp("2015-01-01"),
        pd.Timestamp("2016-01-01").date(),
        radius=60.0,
    )
    assert [scene.attrs["date"] for scene in scenes] == ["2015-08-01"]


def test_timestamp_rejects_nonsense():
    with pytest.raises(ValueError, match="not a date"):
        rd._timestamp("the day before yesterday", "start")


# ----------------------------
# Error surfacing
# ----------------------------
def test_compute_pixels_failure_names_the_causes(nlcd_ee):
    def explode(request):
        raise RuntimeError("Total request size must be less than 50331648 bytes.")

    nlcd_ee.module.data.computePixels = explode
    with pytest.raises(RuntimeError, match="response size limit"):
        rd.fetch_nlcd(TOWER_X, TOWER_Y, UTM, 2016, radius=30.0)
