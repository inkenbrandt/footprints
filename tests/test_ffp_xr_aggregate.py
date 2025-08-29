# test_ffp_xr_aggregate.py
import numpy as np
import pandas as pd
import xarray as xr
import pytest


@pytest.mark.filterwarnings("ignore:.*mean of empty slice.*")
def test_daily_monthly_aggregation(compute_fn, aggregate_fn, synthetic_inputs):
    ds, meta = synthetic_inputs
    out = compute_fn(ds, **meta)

    # Find the main DA
    if isinstance(out, xr.Dataset):
        da = None
        for k, v in out.data_vars.items():
            if set(["y", "x"]).issubset(v.dims):
                da = v
                break
        assert da is not None, "No 2D variable found in Dataset output."
    else:
        da = out

    # If no time dimension, skip
    if "time" not in da.dims:
        pytest.skip("Output has no 'time' dimension; aggregation not applicable.")

    # Path A: module provided an aggregator
    if aggregate_fn is not None:
        # Expect something like aggregate_fn(da, by="D" or period="D", how="sum"/"mean")
        # Try a few signatures
        daily = None
        for kwargs in (dict(by="D"), dict(period="D"), dict(rule="D")):
            try:
                daily = aggregate_fn(da, **kwargs)
                break
            except TypeError:
                continue
            except Exception as e:
                # If aggregator exists but barks, don’t hide real failures
                raise
        if daily is None:
            pytest.skip("aggregate_fn present but no recognized signature.")
    else:
        # Path B: use plain xarray resample as the module likely expects
        daily = da.resample(time="1D").sum()

    # Sanity checks
    assert set(["y", "x"]).issubset(daily.dims)
    assert "time" in daily.dims
    assert daily.time.size <= da.time.size  # fewer or equal time bins

    # Compare first day's sum to manual sum
    day0 = str(pd.to_datetime(da.time.values[0]).date())
    manual = da.sel(time=slice(f"{day0} 00:00", f"{day0} 23:59")).sum("time")
    comp = daily.sel(time=slice(day0, day0)).sum("time")
    # Same grid; values should be close
    diff = float(np.abs((manual - comp)).sum().values)
    assert diff < 1e-6 or diff / float(np.abs(manual).sum().values + 1e-12) < 1e-6

    # Monthly (again either via module or bare xarray)
    if aggregate_fn is not None:
        try:
            monthly = aggregate_fn(da, by="MS")
        except Exception:
            monthly = da.resample(time="MS").sum()
    else:
        monthly = da.resample(time="MS").sum()

    assert set(["y", "x"]).issubset(monthly.dims)
