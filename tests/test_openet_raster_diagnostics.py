from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from fluxfootprints.openet_raster_diagnostics import (
    mask_rasters_geotiff_with_diagnostics,
)


CRS = "EPSG:32612"
TRANSFORM = from_origin(500000.0, 4500000.0, 30.0, 30.0)


def _write_raster(path: Path, data: np.ndarray, nodata=-9999.0):
    profile = {
        "driver": "GTiff",
        "height": data.shape[-2],
        "width": data.shape[-1],
        "count": 1 if data.ndim == 2 else data.shape[0],
        "dtype": "float32",
        "crs": CRS,
        "transform": TRANSFORM,
        "nodata": nodata,
    }
    arr = data[None, ...] if data.ndim == 2 else data
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"))
    return path


def test_mask_raster_returns_diagnostics_and_preserves_mass(tmp_path):
    fp_dir = tmp_path / "footprints"
    oe_dir = tmp_path / "openet"
    out_dir = tmp_path / "masked"
    fp_dir.mkdir()
    oe_dir.mkdir()

    footprint = np.ones((10, 10), dtype="float32")
    _write_raster(fp_dir / "ffp_daily_etw_20240201.tif", footprint, nodata=0.0)

    openet = np.full((10, 10), 5.0, dtype="float32")
    openet[:, :5] = -9999.0
    _write_raster(oe_dir / "ensemble_et_20240201.tif", openet)

    diag = mask_rasters_geotiff_with_diagnostics(
        fp_dir,
        oe_dir,
        out_dir=out_dir,
        renormalize=True,
    )

    assert isinstance(diag, pd.DataFrame)
    assert len(diag) == 1
    row = diag.iloc[0]
    assert row["retained_fraction"] == pytest.approx(0.5)
    assert row["masked_fraction"] == pytest.approx(0.5)
    assert row["valid_pixel_fraction"] == pytest.approx(0.5)
    assert row["original_sum"] == pytest.approx(100.0)
    assert row["retained_sum"] == pytest.approx(50.0)
    assert row["renormalized_sum"] == pytest.approx(100.0)
    assert row["scale_factor"] == pytest.approx(2.0)
    assert row["missing_openet"] == False

    with rasterio.open(row["output_path"]) as src:
        masked = src.read(1)

    assert masked.sum() == pytest.approx(100.0)
    assert np.all(masked[:, :5] == 0.0)
    assert np.all(masked[:, 5:] == pytest.approx(2.0))


def test_diagnostics_csv_is_written(tmp_path):
    fp = _write_raster(
        tmp_path / "ffp_daily_mean_20240201.tif",
        np.ones((4, 4), dtype="float32"),
        nodata=0.0,
    )
    openet = _write_raster(
        tmp_path / "openet_20240201.tif",
        np.ones((4, 4), dtype="float32"),
    )
    csv_path = tmp_path / "diagnostics.csv"

    diag = mask_rasters_geotiff_with_diagnostics(
        fp,
        openet,
        diagnostics_csv=csv_path,
    )

    assert csv_path.exists()
    from_csv = pd.read_csv(csv_path)
    assert len(from_csv) == len(diag) == 1
    assert from_csv.loc[0, "retained_fraction"] == pytest.approx(1.0)


def test_missing_openet_mask_records_zero_retention(tmp_path):
    fp = _write_raster(
        tmp_path / "ffp_daily_mean_20240202.tif",
        np.ones((4, 4), dtype="float32"),
        nodata=0.0,
    )
    openet = _write_raster(
        tmp_path / "openet_20240201.tif",
        np.ones((4, 4), dtype="float32"),
    )

    diag = mask_rasters_geotiff_with_diagnostics(
        fp,
        openet,
        on_missing="mask",
        renormalize=True,
    )

    row = diag.iloc[0]
    assert row["missing_openet"] == True
    assert row["retained_fraction"] == pytest.approx(0.0)
    assert row["retained_sum"] == pytest.approx(0.0)
    assert row["renormalized_sum"] == pytest.approx(0.0)


def test_invalid_retained_fraction_threshold_raises(tmp_path):
    with pytest.raises(ValueError, match="between 0 and 1"):
        mask_rasters_geotiff_with_diagnostics(
            tmp_path,
            tmp_path,
            min_retained_fraction=1.1,
        )
