"""Monthly footprint workflow with file logging and daily export audits.

Notebook usage (uses your existing station_data, station_metadata, FP_DIR,
DX, DY and DOMAIN):
    from footprint_workflow import run_footprint_workflow
    results = run_footprint_workflow(
        station_data, station_metadata, FP_DIR, DX, DY, DOMAIN,
        domain_folder='D:/footprints/domain_outputs', make_gifs=True,
    )

Dependencies: your working fluxfootprints environment, pandas, numpy,
xarray, rasterio, pyarrow; optional animation dependencies/contextily.
Input: FP_DIR/<station>/<station>_input_data.parquet (DatetimeIndex).
Metadata: station_metadata[station][1] = latitude; [2] = longitude.
ET weights assume equally spaced records and ET rates in consistent units.
No interpolation, timezone conversion or duplicate removal is performed.
Existing output filenames are overwritten on successful export. File existence
alone does not prove an output was regenerated in this run; see export_ok.
"""
from pathlib import Path
import gc
import logging
import numpy as np
import pandas as pd

SPATIAL = ('x', 'y')
REQUIRED = [
    'mo_length_1_1_1', 'v_sigma_1_1_1', 'ustar_1_1_1',
    'wd_1_1_1', 'ws_1_1_1', 'zm', 'z0',
]
LAYERS = {
    'daily_mean': 'f_daily_mean',
    'daily_etw': 'f_daily_et_weighted',
    'monthly_etw': 'f_monthly_et_weighted',
}


def setup_logger(folder, console=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('footprint_workflow')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    handler = logging.FileHandler(folder / 'footprint_processing.log',
                                  mode='a', encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    if console:
        handler = logging.StreamHandler()
        handler.setLevel(logging.WARNING)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def usable_frames(da):
    """Strict validation: finite, nonnegative, positive-mass raster."""
    return (
        np.isfinite(da).all(SPATIAL)
        & (da >= 0).all(SPATIAL)
        & (da.sum(SPATIAL, skipna=True) > 0)
    ).compute().to_series()


def audit_daily(df, footprints, summaries, period):
    days = pd.date_range(period.start_time, period.end_time.normalize(), freq='D')
    audit = pd.DataFrame(index=days)
    audit.index.name = 'date'
    valid_fp = usable_frames(footprints)
    times = footprints.time.to_index()
    et = df['et_for_weighting'].reindex(times)
    finite_et = np.isfinite(et)
    audit['input_records'] = df.resample('D').size()
    audit['valid_footprints'] = valid_fp.resample('D').sum()
    audit['matched_et_records'] = finite_et.resample('D').sum()
    audit['positive_et_with_footprint'] = (
        finite_et & et.gt(0) & valid_fp
    ).resample('D').sum()
    count_cols = list(audit.columns)
    audit[count_cols] = audit[count_cols].fillna(0).astype(int)
    for layer in ('daily_mean', 'daily_etw'):
        da = getattr(summaries, LAYERS[layer], None)
        audit[f'{layer}_usable'] = False if da is None else (
            usable_frames(da).reindex(days, fill_value=False).astype(bool)
        )
    audit['reason'] = np.select(
        [audit.input_records.eq(0), audit.valid_footprints.eq(0),
         audit.positive_et_with_footprint.eq(0), ~audit.daily_etw_usable],
        ['No input records', 'No valid modeled footprint',
         'No positive ET matched to valid footprints',
         'ET-weighted summary invalid; inspect arrays'],
        default='Ready for ET-weighted export',
    )
    return audit


def inspect_raster(path):
    """Check actual pixels, not just the existence of a TIFF."""
    import rasterio
    if not path.is_file():
        return False, False
    with rasterio.open(path) as src:
        pixels = src.read(1, masked=True).compressed()
    usable = bool(pixels.size and np.isfinite(pixels).all()
                  and (pixels >= 0).all() and pixels.sum() > 0)
    return True, usable


def save_coverage(frames, destination, logger):
    if not frames:
        return
    try:
        pd.concat(frames).sort_index().to_parquet(destination)
    except Exception:
        logger.exception('Failed to save domain coverage: %s', destination)


def run_footprint_workflow(
    station_data, station_metadata, fp_dir, dx, dy, domain,
    domain_folder=None, make_gifs=False, basemap=False, console=True,
    min_month_records=10, missing_values=(-9999,),
):
    """Run all stations, continuing after station/month/export/GIF failures.

    station_data is used for station IDs; parquet files supply the actual data.
    Timezone-aware indexes are rejected: convert deliberately to the intended
    daily time basis before running. Naive indexes are used unchanged.
    missing_values applies only to model/ET columns; customize for your data.
    min_month_records is only a monthly screening count, not a daily QC rule.
    """
    from fluxfootprints import (
        build_climatology, summarize_periods, export_rasters_geotiff,
    )
    fp_dir = Path(fp_dir)
    logger = setup_logger(fp_dir / 'logs', console=console)
    domain_folder = Path(domain_folder) if domain_folder else fp_dir / 'domain_outputs'
    if domain_folder.is_file():
        raise RuntimeError(f'{domain_folder} is a file. Rename it before running.')
    domain_folder.mkdir(parents=True, exist_ok=True)
    logger.info('Starting workflow: %d stations', len(station_data))
    all_results = []

    for station in station_data:
        out_dir = fp_dir / str(station)
        results, daily_coverage, monthly_coverage = [], [], []
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            lat, lon = station_metadata[station][1:3]
            df_all = pd.read_parquet(out_dir / f'{station}_input_data.parquet')
            if df_all.empty:
                raise ValueError('No input records')
            if not isinstance(df_all.index, pd.DatetimeIndex):
                raise ValueError('Input requires a DatetimeIndex; parse timestamps upstream')
            if df_all.index.tz is not None:
                raise ValueError('Timezone-aware index: choose the daily time basis upstream')
            if df_all.index.hasnans or df_all.index.has_duplicates:
                raise ValueError('Missing or duplicate timestamps; resolve upstream')
            df_all = df_all.sort_index()
            missing = set(REQUIRED + ['nldas_et']) - set(df_all.columns)
            if missing:
                raise ValueError(f'Missing required columns: {sorted(missing)}')
            for col in REQUIRED + ['nldas_et']:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
                df_all[col] = df_all[col].replace(list(missing_values), np.nan)
            months = pd.period_range(df_all.index.min(), df_all.index.max(), freq='M')
        except Exception as exc:
            logger.exception('[%s] Station initialization failed', station)
            all_results.append({'station': station, 'status': 'input_error', 'error': str(exc)})
            continue

        logger.info('[%s] %d records; %d months', station, len(df_all), len(months))
        for period in months:
            label, yyyymm = str(period), period.strftime('%Y%m')
            clim = summaries = footprints = None
            stage = 'input screening'
            row = {'station': station, 'month': label, 'status': 'pending',
                   'n_req_fields': 0, 'ts_len': None, 'error': ''}
            audit_path = out_dir / f'{station}_{yyyymm}_daily_audit.csv'
            try:
                df = df_all.loc[df_all.index.to_period('M') == period].copy()
                # Permit infinite Monin-Obukhov length (neutral limit), leaving
                # the model to enforce its own physical admissibility rules.
                finite_cols = [c for c in REQUIRED if c != 'mo_length_1_1_1']
                present = (df[REQUIRED].notna().all(axis=1)
                           & np.isfinite(df[finite_cols]).all(axis=1))
                row['n_req_fields'] = int(present.sum())
                logger.info('[%s %s] %d inputs; %d with required values',
                            station, label, len(df), row['n_req_fields'])
                if df.empty or row['n_req_fields'] < min_month_records:
                    row['status'] = 'skipped'
                    row['error'] = 'No data or insufficient required inputs'
                    logger.warning('[%s %s] %s', station, label, row['error'])
                    continue

                et = df['nldas_et']
                df['et_for_weighting'] = et.where(np.isfinite(et) & et.ge(0))
                logger.info('[%s %s] ET excluded: %d negative; %d nonfinite',
                            station, label, int(et.lt(0).sum()),
                            int((~np.isfinite(et)).sum()))
                stage = 'model construction'
                clim = build_climatology(
                    df, model_type='ffp_xr',
                    ustar='ustar_1_1_1', ol='mo_length_1_1_1',
                    umean='ws_1_1_1', sigmav='v_sigma_1_1_1',
                    wind_dir='wd_1_1_1', zm='zm', z0='z0', h=2000.0,
                    dx=dx, dy=dy, domain=domain, rslayer=True,
                    logger=logger, verbosity=0,
                )
                footprints = clim.get_footprint_timeseries()
                if footprints is None or footprints.sizes.get('time', 0) == 0:
                    raise ValueError('Model produced no time-resolved footprints')
                row['ts_len'] = footprints.sizes['time']
                logger.debug('[%s %s] Footprint sizes: %s', station, label,
                             dict(footprints.sizes))
                stage = 'summarization'
                summaries = summarize_periods(
                    clim, df, et_source='et_for_weighting', is_le=False,
                    calc_et_weighted=True, daily=True, monthly=True,
                    normalize_each_time=True,
                )
                # Collect the library's monthly statistics directly; never sum
                # daily percentages. This survives later export/GIF failures.
                for attr, collection in (
                    ('daily_domain_coverage', daily_coverage),
                    ('monthly_domain_coverage', monthly_coverage),
                ):
                    coverage = getattr(summaries, attr, None)
                    if coverage is not None and not coverage.empty:
                        collection.append(coverage.copy())
                stage = 'daily audit'
                audit = audit_daily(df, footprints, summaries, period)
                audit.to_csv(audit_path)
                issues = audit.loc[~audit.daily_etw_usable]
                if not issues.empty:
                    logger.warning('[%s %s] %d unusable ET-weighted days: %s',
                                   station, label, len(issues),
                                   issues.reason.value_counts().to_dict())
                    logger.debug('[%s %s] Daily issues:\n%s',
                                 station, label, issues.to_string())

                # Independent exports: failure in one product does not block others.
                stage = 'export'
                export_errors = []
                for layer, attr in LAYERS.items():
                    row[f'{layer}_export_ok'] = False
                    try:
                        da = getattr(summaries, attr, None)
                        if da is None:
                            raise ValueError(f'{attr} is unavailable')
                        if bool((da < 0).any().compute()):
                            raise ValueError(f'Negative footprint weights in {attr}')
                        if bool(np.isinf(da).any().compute()):
                            raise ValueError(f'Infinite footprint weights in {attr}')
                        # NaN-only days stay explicit in the audit; do not fabricate
                        # ET weights or substitute daily means for them.
                        export_rasters_geotiff(
                            clim, summaries, station_lat=lat, station_lon=lon,
                            out_dir=out_dir, crs_out='auto', which=(layer,),
                            prefix=f'ffp_{station}', dtype='float32', nodata=np.nan,
                        )
                        row[f'{layer}_export_ok'] = True
                    except Exception as exc:
                        export_errors.append(f'{layer}: {exc}')
                        logger.exception('[%s %s] %s export failed', station, label, layer)

                stage = 'export verification'
                for layer in ('daily_mean', 'daily_etw'):
                    exists, usable = [], []
                    for day in audit.index:
                        path = out_dir / f'ffp_{station}_{layer}_{day:%Y%m%d}.tif'
                        try:
                            found, good = inspect_raster(path)
                        except Exception:
                            logger.exception('Unreadable raster: %s', path)
                            found, good = path.is_file(), False
                        exists.append(found)
                        usable.append(good)
                    audit[f'{layer}_file_exists'] = exists
                    audit[f'{layer}_file_usable'] = usable
                    audit[f'{layer}_export_ok'] = row[f'{layer}_export_ok']
                    row[f'n_{layer}_files'] = sum(exists)
                    row[f'n_{layer}_usable_files'] = sum(usable)
                    mismatch = audit[f'{layer}_usable'] & ~audit[f'{layer}_file_usable']
                    if mismatch.any():
                        export_errors.append(f'{layer}: {int(mismatch.sum())} expected usable files missing/invalid')
                        logger.warning('[%s %s] %s expected usable files missing/invalid: %s',
                                       station, label, layer,
                                       audit.index[mismatch].strftime('%Y-%m-%d').tolist())
                audit.to_csv(audit_path)
                monthly_path = out_dir / f'ffp_{station}_monthly_etw_{yyyymm}.tif'
                found, good = inspect_raster(monthly_path)
                row['n_monthly_tifs'] = int(found)
                row['n_monthly_usable_tifs'] = int(good)
                row['status'] = 'partial_export' if export_errors else 'ok'
                row['error'] = '; '.join(export_errors)
                row['days_without_usable_etw'] = int((~audit.daily_etw_usable).sum())
                logger.info('[%s %s] %s; %d daily ETW files, %d usable',
                            station, label, row['status'], row['n_daily_etw_files'],
                            row['n_daily_etw_usable_files'])

                # Animation cannot prevent export or change export status.
                if make_gifs:
                    try:
                        from fluxfootprints.footprint_animation import animate_footprint
                        da = summaries.f_daily_mean
                        keep = usable_frames(da).to_numpy()
                        da = da.isel(time=np.flatnonzero(keep))
                        if da.sizes['time']:
                            gif_dir = fp_dir / 'gif_outputs'
                            gif_dir.mkdir(parents=True, exist_ok=True)
                            options = {}
                            if basemap:
                                import contextily as cx
                                options['basemap_source'] = cx.providers.Esri.WorldImagery
                            animate_footprint(
                                da, gif_dir / f'{station}_{label}_ffp_daily.gif',
                                station_lat=lat, station_lon=lon, basemap=basemap,
                                norm='log', cmap='inferno', mask_quantile=0.6,
                                alpha=0.7, contour_fractions=(0.8,),
                                title=f'Daily mean footprint, {station}, {label}',
                                fps=3, dpi=72, figsize=(5.0, 4.2), **options,
                            )
                            row['gif_status'] = 'ok'
                        else:
                            row['gif_status'] = 'skipped: no usable daily means'
                    except Exception as exc:
                        row['gif_status'] = f'error: {exc}'
                        logger.exception('[%s %s] GIF failed; export already completed',
                                         station, label)
            except Exception as exc:
                row['status'] = 'error'
                row['error'] = f'{stage}: {exc}'
                logger.exception('[%s %s] Failed during %s', station, label, stage)
            finally:
                results.append(row)
                all_results.append(row)
                # Checkpoint the summary even when a month was skipped or failed.
                try:
                    pd.DataFrame(results).to_csv(
                        domain_folder / f'{station}_processing_summary.csv', index=False)
                except Exception:
                    logger.exception('[%s] Failed to save processing summary', station)
                clim = summaries = footprints = None
                gc.collect()

        save_coverage(daily_coverage,
                      domain_folder / f'{station}_daily_domain_results.parquet', logger)
        save_coverage(monthly_coverage,
                      domain_folder / f'{station}_monthly_domain_results.parquet', logger)
        logger.info('[%s] Finished', station)

    result = pd.DataFrame(all_results)
    result.to_csv(domain_folder / 'all_stations_processing_summary.csv', index=False)
    logger.info('Workflow finished')
    return result
