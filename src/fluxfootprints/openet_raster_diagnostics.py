"""OpenET-aware masking of exported footprint rasters with diagnostics.

This module complements :mod:`fluxfootprints.openet_masking` with a disk-raster
workflow that preserves the existing masking behavior while returning a tidy
per-raster diagnostics table.  It is intended for footprint-weighted OpenET
comparisons, where footprint weight falling on missing OpenET pixels is removed
and the surviving weights may be renormalized before extracting ET.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling

from .openet_masking import (
    OpenETSource,
    _select_dates,
    index_openet_rasters,
    openet_mask_on_grid,
    parse_raster_date,
)


__all__ = ["mask_rasters_geotiff_with_diagnostics"]


def mask_rasters_geotiff_with_diagnostics(
    raster_dir: str | Path,
    openet: OpenETSource,
    out_dir: str | Path | None = None,
    pattern: str = "*.tif",
    fill_value: float | None = None,
    band: int | str = 1,
    nodata: float | None = None,
    valid_range: tuple[float | None, float | None] | None = None,
    treat_zero_as_nodata: bool = False,
    resampling: str | Resampling = "nearest",
    coverage_threshold: float = 0.5,
    combine: str = "any",
    on_missing: str = "skip",
    max_gap_days: int = 8,
    renormalize: bool = True,
    openet_pattern: str = "*.tif",
    recursive: bool = False,
    date_regex: str | Any | None = None,
    diagnostics_csv: str | Path | None = None,
    min_retained_fraction: float | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Mask footprint GeoTIFFs with OpenET validity and return diagnostics.

    Footprint cells that do not overlap valid OpenET data are replaced with
    ``fill_value``.  When ``renormalize=True`` (the default for this OpenET
    comparison helper), surviving footprint weights are multiplied by a scalar
    so each raster band retains the same total mass it had before masking.

    Unlike :func:`fluxfootprints.openet_masking.mask_rasters_geotiff`, this
    function returns a :class:`pandas.DataFrame` with one record per raster band
    and optionally writes that table to CSV.

    Parameters
    ----------
    raster_dir : path
        Directory containing footprint GeoTIFFs, or one footprint GeoTIFF.
    openet : path, sequence, or mapping
        Daily OpenET rasters accepted by ``index_openet_rasters``.
    out_dir : path, optional
        Output directory. Defaults to ``<raster_dir>/openet_masked``.
    pattern : str, default "*.tif"
        Glob for footprint rasters when ``raster_dir`` is a directory.
    fill_value : float, optional
        Value written where OpenET is missing. Defaults to footprint nodata, or
        0.0 when the footprint has no nodata value.
    band, nodata, valid_range, treat_zero_as_nodata, resampling,
    coverage_threshold, combine
        Options describing the OpenET valid-data mask.
    on_missing : {"skip", "mask", "nearest", "error"}, default "skip"
        Behavior when no OpenET raster matches the footprint date.
    max_gap_days : int, default 8
        Maximum nearest-date search distance when ``on_missing="nearest"``.
    renormalize : bool, default True
        Preserve pre-mask footprint mass by rescaling surviving weights.
    diagnostics_csv : path, optional
        Write the returned diagnostics table to this CSV.
    min_retained_fraction : float, optional
        If provided, emit a warning whenever the fraction of footprint mass
        overlapping valid OpenET falls below this value.

    Returns
    -------
    pandas.DataFrame
        One row per raster band with columns including ``date``, ``frequency``,
        ``source_path``, ``output_path``, ``openet_dates``, ``missing_openet``,
        ``band``, ``original_sum``, ``retained_sum``, ``retained_fraction``,
        ``masked_fraction``, ``valid_pixel_fraction``, ``renormalized_sum``, and
        ``scale_factor``.

    Notes
    -----
    ``retained_fraction`` is footprint-weighted coverage, not simple pixel
    coverage.  For footprint-weighted OpenET extraction this is generally the
    more useful quality-control metric because missing pixels near the footprint
    peak matter more than missing pixels in the low-weight tail.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    on_missing = on_missing.lower()
    if on_missing not in {"skip", "mask", "nearest", "error"}:
        raise ValueError(
            f"Unknown on_missing '{on_missing}'. "
            "Use 'skip', 'mask', 'nearest', or 'error'."
        )

    if min_retained_fraction is not None and not 0 <= min_retained_fraction <= 1:
        raise ValueError("min_retained_fraction must be between 0 and 1")

    raster_dir = Path(raster_dir)
    if raster_dir.is_dir():
        sources = sorted(p for p in raster_dir.glob(pattern) if p.is_file())
        default_out = raster_dir / "openet_masked"
    elif raster_dir.is_file():
        sources = [raster_dir]
        default_out = raster_dir.parent / "openet_masked"
    else:
        raise FileNotFoundError(f"No such file or directory: {raster_dir}")

    if not sources:
        raise ValueError(f"No rasters matching '{pattern}' in {raster_dir}")

    out_dir = Path(out_dir) if out_dir is not None else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    index = index_openet_rasters(
        openet,
        pattern=openet_pattern,
        recursive=recursive,
        date_regex=date_regex,
        logger=logger,
    )
    available = list(index.keys())

    records: list[dict[str, Any]] = []

    for source in sources:
        parsed = parse_raster_date(source)
        if parsed is None:
            logger.warning("No date in raster name %s; skipping.", source.name)
            continue

        date, kind = parsed
        freq = "monthly" if kind == "month" else "daily"
        dates = _select_dates(
            pd.Timestamp(date), freq, available, on_missing, max_gap_days
        )

        missing_openet = len(dates) == 0
        if missing_openet and on_missing == "error":
            raise FileNotFoundError(
                f"No OpenET raster for {date.isoformat()} (from {source.name})."
            )
        if missing_openet and on_missing == "skip":
            logger.warning(
                "No OpenET raster for %s; copying %s through unmasked.",
                date.isoformat(),
                source.name,
            )

        with rasterio.open(source) as src:
            profile = src.profile.copy()
            data = src.read().astype("float64")
            src_nodata = src.nodata

        if dates:
            paths = tuple(p for d in dates for p in index[d])
            mask = openet_mask_on_grid(
                paths,
                profile["crs"],
                profile["transform"],
                (profile["height"], profile["width"]),
                band=band,
                nodata=nodata,
                valid_range=valid_range,
                treat_zero_as_nodata=treat_zero_as_nodata,
                resampling=resampling,
                coverage_threshold=coverage_threshold,
                combine=combine,
            )
        elif on_missing == "mask":
            mask = np.zeros((profile["height"], profile["width"]), dtype=bool)
            paths = tuple()
        else:
            mask = np.ones((profile["height"], profile["width"]), dtype=bool)
            paths = tuple()

        fill = fill_value
        if fill is None:
            fill = float(src_nodata) if src_nodata is not None else 0.0

        data_valid = np.isfinite(data)
        if src_nodata is not None and np.isfinite(src_nodata):
            data_valid &= data != src_nodata

        keep = mask[None, :, :] & data_valid
        before = np.where(data_valid, data, 0.0).sum(axis=(1, 2))
        after = np.where(keep, data, 0.0).sum(axis=(1, 2))
        retained = np.divide(
            after,
            before,
            out=np.full_like(after, np.nan, dtype="float64"),
            where=before != 0,
        )

        # Simple pixel coverage is useful alongside the footprint-weighted
        # retained fraction, but it should not be substituted for it.
        valid_pixel_count = keep.sum(axis=(1, 2)).astype("float64")
        footprint_pixel_count = data_valid.sum(axis=(1, 2)).astype("float64")
        valid_pixel_fraction = np.divide(
            valid_pixel_count,
            footprint_pixel_count,
            out=np.full_like(valid_pixel_count, np.nan),
            where=footprint_pixel_count != 0,
        )

        out_data = np.where(mask[None, :, :], data, fill)
        scale = np.ones_like(before, dtype="float64")
        if renormalize:
            scale = np.divide(
                before,
                after,
                out=np.ones_like(before, dtype="float64"),
                where=after > 0,
            )
            out_data = np.where(keep, data * scale[:, None, None], out_data)

        # Sum only real output footprint values, excluding fill/nodata pixels.
        output_valid = keep if dates or on_missing == "mask" else data_valid
        renormalized_sum = np.where(
            output_valid,
            out_data,
            0.0,
        ).sum(axis=(1, 2))

        dest = out_dir / source.name
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(out_data.astype(profile["dtype"]))
            dst.update_tags(
                openet_masked="true",
                openet_renormalized=str(bool(renormalize)).lower(),
                openet_retained_fraction=(
                    float(retained[0]) if len(retained) == 1 else "per-band"
                ),
            )

        for i in range(data.shape[0]):
            retained_i = float(retained[i]) if np.isfinite(retained[i]) else np.nan
            record = {
                "date": pd.Timestamp(date),
                "frequency": freq,
                "source_path": str(source),
                "output_path": str(dest),
                "openet_dates": ";".join(d.isoformat() for d in dates),
                "openet_paths": ";".join(str(p) for p in paths),
                "missing_openet": bool(missing_openet),
                "band": i + 1,
                "original_sum": float(before[i]),
                "retained_sum": float(after[i]),
                "retained_fraction": retained_i,
                "masked_fraction": (
                    float(1.0 - retained_i) if np.isfinite(retained_i) else np.nan
                ),
                "valid_pixel_fraction": (
                    float(valid_pixel_fraction[i])
                    if np.isfinite(valid_pixel_fraction[i])
                    else np.nan
                ),
                "renormalized_sum": float(renormalized_sum[i]),
                "scale_factor": float(scale[i]),
                "renormalized": bool(renormalize),
            }
            records.append(record)

            if (
                min_retained_fraction is not None
                and np.isfinite(retained_i)
                and retained_i < min_retained_fraction
            ):
                logger.warning(
                    "%s band %d retains only %.1f%% of footprint mass over "
                    "valid OpenET pixels (threshold %.1f%%).",
                    source.name,
                    i + 1,
                    retained_i * 100.0,
                    min_retained_fraction * 100.0,
                )

    diagnostics = pd.DataFrame.from_records(records)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(["date", "source_path", "band"])
        diagnostics = diagnostics.reset_index(drop=True)

    if diagnostics_csv is not None:
        diagnostics_csv = Path(diagnostics_csv)
        diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(diagnostics_csv, index=False)

    return diagnostics
