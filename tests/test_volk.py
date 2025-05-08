import pytest
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append("../src")
from fluxfootprints.volk import (
    load_configs,
    fetch_and_preprocess_data,
    multiply_geotiffs,
    mask_fp_cutoff,
    snap_centroid,
)

@pytest.fixture
def dummy_dataframe():
    index = pd.date_range("2023-01-01 00:00", periods=5, freq="H")
    data = {
        "h2o": [1, 2, 3, 4, 5],
        "wd": [180, 190, 200, 210, 220],
        "ustar": [0.2, 0.3, 0.4, 0.5, 0.6],
        "v_sigma": [0.1, 0.1, 0.2, 0.2, 0.3],
    }
    return pd.DataFrame(data, index=index)

def test_mask_fp_cutoff_behavior():
    array = np.random.rand(10, 10)
    masked = mask_fp_cutoff(array, cutoff=0.9)
    assert masked.shape == array.shape
    assert np.all(masked >= 0)

def test_snap_centroid_odd_alignment():
    x, y = snap_centroid(435627.3, 4512322.7)
    assert (x % 30) == 15
    assert (y % 30) == 15

def create_test_raster(path, data, transform, crs="EPSG:4326"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

def test_multiply_geotiffs_creates_output(tmp_path):
    transform = from_origin(0, 10, 1, 1)
    data_a = np.ones((10, 10), dtype=np.float32)
    data_b = np.full((10, 10), 2, dtype=np.float32)

    file_a = tmp_path / "a.tif"
    file_b = tmp_path / "b.tif"
    output = tmp_path / "out.tif"

    create_test_raster(file_a, data_a, transform)
    create_test_raster(file_b, data_b, transform)

    total_sum = multiply_geotiffs(file_a, file_b, output)

    with rasterio.open(output) as src:
        result = src.read(1)
        assert np.allclose(result, 2)
        assert np.isclose(total_sum, 200.0)
        assert src.count == 1


def test_load_configs_reads_data(tmp_path):
    config_file = tmp_path / "station.ini"
    secrets_file = tmp_path / "config.ini"
    config_file.write_text("[METADATA]\nstation_latitude=40.0\nstation_longitude=-111.9\nstation_elevation=1500")
    secrets_file.write_text("[DEFAULT]\nurl=http://example.com")
    result = load_configs("station", config_path=config_file.parent, secrets_path=secrets_file)
    assert isinstance(result, dict)
    assert result["latitude"] == 40.0
