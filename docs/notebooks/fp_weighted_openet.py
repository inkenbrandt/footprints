"""
Footprint-weighted OpenET extraction (faster version).

Changes from the original loop, roughly in order of how much time they save:

1. The OpenET tree is walked ONCE and indexed by (station, date) -> {model: path},
   instead of one recursive rglob() over the whole Drive folder per footprint-day.
2. OpenET files are read concurrently with a thread pool. The work is I/O-bound
   (especially on a Google Drive mount) and rasterio releases the GIL while reading.
3. GDAL is told not to list the parent directory on every open
   (GDAL_DISABLE_READDIR_ON_OPEN), which is slow in folders with many files.
4. Each footprint is read once per day instead of once per model.
5. The EPSG:4326 -> footprint-grid bilinear resampling is precomputed as a sparse
   matrix for each unique (OpenET grid, footprint grid) pair and reused, so each
   OpenET file costs a read plus one sparse mat-vec instead of a full GDAL warp.
6. Rows are collected in a list and written as one DataFrame per station.

It also fixes these bugs in the original:
- `openet_path` was never set inside the loop (the loop variable is `oetfile`).
- `oet` was never reset, so each station's output also held every previous station's rows.
- 'ensemble' was listed twice in `models`, so those files were processed twice.
- `pd.concat` of a dict of Series returns a Series, which has no `.to_parquet()`.
- `np.sum(phi_norm * openet_reproj * valid)` returns NaN whenever ANY pixel of the
  101 x 101 grid is NaN, because 0 * NaN is NaN.

Usage (works in a notebook or a script):

    from fp_weighted_openet import run
    results = run(station_locations.index,
                  ffp_dir=r"D:/footprints",
                  openet_dir=r"M:/My Drive/OpenET_Daily_Tiffs",
                  out_dir=".")
"""
from __future__ import annotations

import os
import re
import threading
import time
import pathlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from scipy import sparse

MODELS = ("ensemble", "disalexi", "eemetric", "geesebal", "ptjpl", "ssebop", "sims")
OPENET_DATE_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})")
GDAL_OPTS = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}


# ── 1. Index the OpenET folder once ───────────────────────────────────────
def index_openet(openet_dir, station_ids, models=MODELS):
    """Walk ``openet_dir`` once and return {(station_id, Timestamp): {model: Path}}.

    Matching uses the same rules as the original loop (station id and model name
    appear somewhere in the file name, date written as YYYY_MM_DD).
    """
    ids = sorted(map(str, station_ids), key=len, reverse=True)  # longest id wins
    index = defaultdict(dict)
    n_dupes = 0
    for root, _, files in os.walk(openet_dir):
        for name in files:
            if not name.lower().endswith(".tif"):
                continue
            m = OPENET_DATE_RE.search(name)
            station = next((s for s in ids if s in name), None)
            model = next((mdl for mdl in models if mdl in name), None)
            if m is None or station is None or model is None:
                continue
            key = (station, pd.Timestamp(int(m[1]), int(m[2]), int(m[3])))
            if model in index[key]:
                n_dupes += 1  # same station/date/model in two places; keep first
                continue
            index[key][model] = pathlib.Path(root, name)
    if n_dupes:
        print(f"index_openet: skipped {n_dupes} duplicate station/date/model files")
    return dict(index)


# ── 2. Precomputed bilinear resampling ────────────────────────────────────
_W_CACHE: dict = {}
_W_LOCK = threading.Lock()


def _grid_key(transform, crs, shape):
    return tuple(transform)[:6], crs.to_wkt(), tuple(shape)


def build_bilinear_matrix(src_transform, src_crs, src_shape,
                          dst_transform, dst_crs, dst_shape):
    """Return (W, nearest) describing bilinear resampling from a source grid
    onto a destination grid, following GDAL's conventions:

    W       sparse (n_dst x n_src) bilinear weights, pixel-centre based.
    nearest flat index of the source pixel containing each destination pixel
            centre, or -1 if that centre is outside the source raster.

    GDAL returns nodata when that containing pixel is nodata, and otherwise
    renormalises the weights over the valid neighbours; resample() does the same.
    Intended for upsampling (destination pixels smaller than source pixels,
    e.g. 30 m -> 10 m); when downsampling, GDAL widens its bilinear kernel and
    this matrix would no longer match it.
    """
    dh, dw = dst_shape
    sh, sw = src_shape
    rr, cc = np.mgrid[0:dh, 0:dw]
    x, y = dst_transform * (cc.ravel() + 0.5, rr.ravel() + 0.5)
    x, y = warp_transform(dst_crs, src_crs, x, y)
    fc, fr = ~src_transform * (np.asarray(x), np.asarray(y))   # source pixel coords

    inside = (fc >= 0) & (fc < sw) & (fr >= 0) & (fr < sh)
    nearest = np.full(dh * dw, -1, dtype=np.int64)
    nearest[inside] = (np.floor(fr[inside]).astype(np.int64) * sw
                       + np.floor(fc[inside]).astype(np.int64))
    fc, fr = fc - 0.5, fr - 0.5                                # centre-based
    c0, r0 = np.floor(fc).astype(np.int64), np.floor(fr).astype(np.int64)
    dc, dr = fc - c0, fr - r0

    dst_idx = np.arange(dh * dw)
    rows, cols, vals = [], [], []
    for oc, orr, w in ((0, 0, (1 - dc) * (1 - dr)), (1, 0, dc * (1 - dr)),
                       (0, 1, (1 - dc) * dr),       (1, 1, dc * dr)):
        c, r = c0 + oc, r0 + orr
        ok = inside & (c >= 0) & (c < sw) & (r >= 0) & (r < sh) & (w > 0)
        rows.append(dst_idx[ok])
        cols.append(r[ok] * sw + c[ok])
        vals.append(w[ok])
    W = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dh * dw, sh * sw),
    )
    return W, nearest


def get_bilinear_matrix(src_transform, src_crs, src_shape,
                        dst_transform, dst_crs, dst_shape):
    """Cached wrapper: built once per unique (source grid, destination grid)."""
    key = (_grid_key(src_transform, src_crs, src_shape),
           _grid_key(dst_transform, dst_crs, dst_shape))
    op = _W_CACHE.get(key)
    if op is None:
        op = build_bilinear_matrix(src_transform, src_crs, src_shape,
                                   dst_transform, dst_crs, dst_shape)
        with _W_LOCK:
            op = _W_CACHE.setdefault(key, op)
    return op


def resample(op, values, dst_shape):
    """Resample a source array with a (W, nearest) pair from
    get_bilinear_matrix(); NaN is treated as nodata, as in the original."""
    W, nearest = op
    v = values.ravel()
    ok = np.isfinite(v)
    num = W @ np.where(ok, v, 0.0)
    den = W @ ok.astype(np.float64)
    has_src = nearest >= 0
    good = np.zeros(W.shape[0], dtype=bool)
    good[has_src] = ok[nearest[has_src]]
    good &= den > 0
    out = np.full(W.shape[0], np.nan)
    out[good] = num[good] / den[good]
    return out.reshape(dst_shape)


# ── 3. Footprint-weighted ET (NaN-safe) ───────────────────────────────────
def compute_footprint_weighted_et(phi, et):
    """Return (ET_fp, ET_mean, fp_coverage).

    fp_coverage is the fraction of the footprint weight that landed on valid
    OpenET pixels, i.e. how much renormalisation happened. 1.0 means none.
    """
    valid = (phi > 0) & np.isfinite(et)
    w = phi[valid]
    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan, np.nan, 0.0
    e = et[valid]
    total = phi[phi > 0].sum()
    return float(w @ e / w_sum), float(e.mean()), float(w_sum / total)


# ── 4. One station-day: read footprint once, loop over models ─────────────
def process_day(station_id, fp_path, fp_date, model_files):
    rows = []
    with rasterio.Env(**GDAL_OPTS):
        with rasterio.open(fp_path) as src:
            phi = src.read(1, masked=True).astype(np.float64).filled(0.0)
            fp_transform, fp_crs = src.transform, src.crs
        unity = float(phi.sum())

        for model, oet_path in model_files.items():
            with rasterio.open(oet_path) as src:
                et = src.read(1, masked=True).astype(np.float64).filled(np.nan)
                op = get_bilinear_matrix(src.transform, src.crs, et.shape,
                                         fp_transform, fp_crs, phi.shape)
            et_on_fp = resample(op, et, phi.shape)
            ET_fp, ET_mean, cov = compute_footprint_weighted_et(phi, et_on_fp)
            rows.append({
                "station_id": station_id,
                "model": model,
                "footprint_file": fp_path.stem,
                "openet_file": oet_path.name,
                "footprint_date": fp_date,
                "ET_fp": ET_fp,
                "ET_mean": ET_mean,
                "fp_coverage": cov,
                "unity_check": unity,
            })
    return rows


# ── 5. Driver ─────────────────────────────────────────────────────────────
def run(station_ids, ffp_dir, openet_dir, out_dir=".", models=MODELS,
        max_workers=8, skip_existing=False):
    """Process every station; write station_oet_<id>.parquet for each.

    max_workers: threads reading files at once. 8-16 is reasonable for a
    network/Drive mount; for a local SSD, try about the number of CPU cores.
    skip_existing: leave stations that already have an output file alone,
    so an interrupted run can resume.
    """
    ffp_dir, out_dir = pathlib.Path(ffp_dir), pathlib.Path(out_dir)
    station_ids = [str(s) for s in station_ids]

    t0 = time.perf_counter()
    index = index_openet(openet_dir, station_ids, models)
    print(f"Indexed {sum(len(v) for v in index.values())} OpenET files "
          f"in {time.perf_counter() - t0:.1f} s")

    results = {}
    for sid in station_ids:
        out_path = out_dir / f"station_oet_{sid}.parquet"
        if skip_existing and out_path.exists():
            print(f"Station {sid}: output exists, skipping")
            continue

        tasks = []
        for fp in sorted((ffp_dir / sid).glob("*daily*.tif")):
            fp_date = pd.to_datetime(fp.stem.split("_")[-1], format="%Y%m%d")
            model_files = index.get((sid, fp_date))
            if model_files:
                tasks.append((sid, fp, fp_date, model_files))

        t1 = time.perf_counter()
        rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_day, *t): t for t in tasks}
            for fut in as_completed(futures):
                _, fp, fp_date, _ = futures[fut]
                try:
                    rows.extend(fut.result())
                except Exception as exc:  # keep going; report the bad day
                    print(f"  FAILED {sid} {fp_date:%Y-%m-%d} ({fp.name}): {exc!r}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["footprint_date", "model"], ignore_index=True)
        df.to_parquet(out_path, index=False)
        results[sid] = df
        print(f"Station {sid}: {len(tasks)} days, {len(df)} rows "
              f"in {time.perf_counter() - t1:.1f} s")
    return results


if __name__ == "__main__":
    # Example; replace with your own station list
    # station_locations = pd.read_csv(...).set_index("station_id")
    # run(station_locations.index, "D:/footprints", "M:/My Drive/OpenET_Daily_Tiffs")
    pass
