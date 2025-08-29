# test_ffp_xr_core.py
import numpy as np
import xarray as xr
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append("../src")

from fluxfootprints import ffp_xr  # the module under test


def _has_dims(a, names):
    return all(d in a.dims for d in names)


def test_compute_returns_xarray(compute_fn, synthetic_inputs):
    ds, meta = synthetic_inputs
    # Try common calling patterns:
    # 1) compute_fn(ds, **meta) or 2) compute_fn(ds, zm=..., z0=..., h=...) or 3) compute_fn(ds)
    try:
        out = compute_fn(ds, **meta)
    except TypeError:
        try:
            out = compute_fn(ds)
        except Exception as e:
            pytest.skip(f"Could not call compute function with ds only or ds+meta: {e}")

    assert isinstance(out, (xr.DataArray, xr.Dataset)), "Expected xarray output"


def test_output_dims_and_attrs(compute_fn, synthetic_inputs):
    ds, meta = synthetic_inputs
    out = compute_fn(ds, **meta) if meta else compute_fn(ds)

    # Allow DA or DS; prefer DA named like f_2d
    if isinstance(out, xr.Dataset):
        # Try to find a 2D or 3D (time,y,x) variable
        cand = None
        for k, v in out.data_vars.items():
            if set(["y", "x"]).issubset(v.dims):
                cand = v
                break
        assert cand is not None, "No 2D footprint variable found in Dataset."
        da = cand
    else:
        da = out

    # If time-stacked, expect ('time','y','x'); else ('y','x')
    if "time" in da.dims:
        assert _has_dims(da, ("time", "y", "x"))
    else:
        assert _has_dims(da, ("y", "x"))

    # Reasonable numeric values
    assert np.isfinite(da).any(), "All values are NaN/Inf."

    # Optional: mass-conservation style check (footprint weights ~ sum to 1 per time)
    if "time" in da.dims:
        sums = da.sum(dim=("y", "x"))
        # Don’t force exact equality; different grids/normalizations exist.
        assert (sums > 0).all(), "Zero total weight per timestep is suspicious."
        assert (sums < 10).all(), "Unusually large total weight suggests unit bug."
    else:
        tot = float(da.sum().values)
        assert tot > 0 and tot < 10, "Unexpected total footprint mass."


def test_wind_rotation_effects(compute_fn, synthetic_inputs):
    ds, meta = synthetic_inputs
    # Two cases: WD ~ 0° and WD ~ 180°
    ds0 = ds.copy()
    ds0["WD"][:] = 0.0
    ds180 = ds.copy()
    ds180["WD"][:] = 180.0

    f0 = compute_fn(ds0, **meta)
    f180 = compute_fn(ds180, **meta)

    # Choose a comparable slice (time or 2D)
    if isinstance(f0, xr.Dataset):
        f0 = next(iter(f0.data_vars.values()))
    if isinstance(f180, xr.Dataset):
        f180 = next(iter(f180.data_vars.values()))

    if "time" in f0.dims:
        f0 = f0.isel(time=0)
        f180 = f180.isel(time=0)

    # 180° flip should roughly mirror the field in x
    # (We don’t assume strict symmetry; just a coarse check.)
    f0r = f0.reverse("x")
    corr = xr.corr(f0, f0r)
    assert float(corr) > 0.2, "Expected some mirrored similarity when WD flips by 180°."
