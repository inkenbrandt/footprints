# tests/test_nldas_read_functions.py
"""Tests for :mod:`fluxfootprints.nldas_read_functions`.

Every test here is offline: ``requests.get`` is always monkeypatched, so no
network call is ever made.
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints import nldas_read_functions as nldas
from fluxfootprints.nldas_read_functions import (
    call_nldas_time_series,
    parse_nldas_csv,
    fetch_nldas_forcing_dataset,
)

TIME_SERIES_URL = "https://api.giovanni.earthdata.nasa.gov/timeseries"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_nldas_csv(param_name="Tair", timestamps=None, values=None, param_id=None):
    """Build a CSV string shaped like the Giovanni time series service output.

    ``parse_nldas_csv`` consumes the first 13 lines as ``key,value`` metadata
    pairs, then hands the rest to ``pd.read_csv(..., header=1)``.  ``header=1``
    ignores the first remaining line and treats the second as the column-header
    row, so the payload below carries *two* lines before the data starts.
    """
    if timestamps is None:
        timestamps = ["2020-01-01T00:00:00", "2020-01-01T01:00:00"]
    if values is None:
        values = [280.5, 281.25]
    if param_id is None:
        param_id = f"NLDAS_FORA0125_H_2_0_{param_name}"

    metadata = [
        ("title", "Time Series"),
        ("param_id", param_id),
        ("param_name", param_name),
        ("param_units", "K"),
        ("param_short_name", param_name),
        ("start_date", timestamps[0] if timestamps else ""),
        ("end_date", timestamps[-1] if timestamps else ""),
        ("lat", "40.0"),
        ("lon", "-111.0"),
        ("temporal_resolution", "hourly"),
        ("dataset_id", "NLDAS_FORA0125_H"),
        ("version", "2.0"),
        ("provider", "GES DISC"),
    ]
    assert len(metadata) == 13, "parser hard-codes a 13-line metadata block"

    lines = [f"{k},{v}" for k, v in metadata]
    lines.append("Timestamp (UTC),Data")  # consumed by header=1
    lines.append("Timestamp,Data")  # the column-header row itself
    lines.extend(f"{t},{v}" for t, v in zip(timestamps, values))
    return "\n".join(lines) + "\n"


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, text="", status_code=200):
        self.text = text
        self.status_code = status_code


class RecordingGet:
    """Callable that records every ``requests.get`` invocation."""

    def __init__(self, response=None):
        self.response = response if response is not None else FakeResponse("ok")
        self.calls = []

    def __call__(self, url, params=None, headers=None, **kwargs):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.response


# ---------------------------------------------------------------------------
# call_nldas_time_series
# ---------------------------------------------------------------------------


class TestCallNldasTimeSeries:
    def test_returns_response_text(self, monkeypatch):
        monkeypatch.setattr(nldas.requests, "get", RecordingGet(FakeResponse("body")))

        out = call_nldas_time_series(
            40.0, -111.0, "2020-01-01T00:00:00", "2020-01-02T00:00:00", "PARAM", "tok"
        )

        assert out == "body"

    def test_builds_request_url_params_and_headers(self, monkeypatch):
        fake_get = RecordingGet(FakeResponse("body"))
        monkeypatch.setattr(nldas.requests, "get", fake_get)

        call_nldas_time_series(
            41.5,
            -112.25,
            "2020-06-01T00:00:00",
            "2020-06-02T00:00:00",
            "NLDAS_FORA0125_H_2_0_Tair",
            "secret-token",
        )

        assert len(fake_get.calls) == 1
        call = fake_get.calls[0]
        assert call["url"] == TIME_SERIES_URL
        assert call["params"] == {
            "data": "NLDAS_FORA0125_H_2_0_Tair",
            "location": "[41.5,-112.25]",
            "time": "2020-06-01T00:00:00/2020-06-02T00:00:00",
        }
        assert call["headers"] == {"Authorization": "Bearer secret-token"}

    def test_prints_status_code(self, monkeypatch, capsys):
        monkeypatch.setattr(
            nldas.requests, "get", RecordingGet(FakeResponse("body", status_code=404))
        )

        call_nldas_time_series(0, 0, "a", "b", "PARAM", "tok")

        assert "Status Code: 404" in capsys.readouterr().out

    def test_error_body_is_returned_not_raised(self, monkeypatch):
        """The function does not call ``raise_for_status``; the body comes back."""
        monkeypatch.setattr(
            nldas.requests,
            "get",
            RecordingGet(FakeResponse("Unauthorized", status_code=401)),
        )

        assert call_nldas_time_series(0, 0, "a", "b", "PARAM", "bad") == "Unauthorized"

    def test_request_exceptions_propagate(self, monkeypatch):
        def boom(*args, **kwargs):
            raise nldas.requests.exceptions.ConnectionError("no network")

        monkeypatch.setattr(nldas.requests, "get", boom)

        with pytest.raises(nldas.requests.exceptions.ConnectionError):
            call_nldas_time_series(0, 0, "a", "b", "PARAM", "tok")


# ---------------------------------------------------------------------------
# parse_nldas_csv
# ---------------------------------------------------------------------------


class TestParseNldasCsv:
    def test_returns_all_thirteen_header_entries(self):
        headers, _ = parse_nldas_csv(make_nldas_csv())

        assert len(headers) == 13
        assert headers["param_name"] == "Tair"
        assert headers["param_id"] == "NLDAS_FORA0125_H_2_0_Tair"
        assert headers["param_units"] == "K"

    def test_header_values_are_stripped(self):
        headers, _ = parse_nldas_csv(make_nldas_csv())

        assert all(v == v.strip() for v in headers.values())

    def test_dataframe_columns_named_from_param_name(self):
        _, df = parse_nldas_csv(make_nldas_csv(param_name="Qair"))

        assert list(df.columns) == ["Timestamp", "Qair"]

    def test_dataframe_values_and_dtypes(self):
        _, df = parse_nldas_csv(
            make_nldas_csv(
                timestamps=["2020-03-01T00:00:00", "2020-03-01T01:00:00"],
                values=[275.0, 276.5],
            )
        )

        assert len(df) == 2
        assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"])
        assert df["Timestamp"].tolist() == [
            pd.Timestamp("2020-03-01T00:00:00"),
            pd.Timestamp("2020-03-01T01:00:00"),
        ]
        assert df["Tair"].tolist() == [275.0, 276.5]

    def test_single_data_row(self):
        _, df = parse_nldas_csv(
            make_nldas_csv(timestamps=["2020-01-01T00:00:00"], values=[300.0])
        )

        assert len(df) == 1
        assert df["Tair"].iloc[0] == 300.0

    def test_no_data_rows_gives_empty_frame_with_headers(self):
        headers, df = parse_nldas_csv(make_nldas_csv(timestamps=[], values=[]))

        assert headers["param_name"] == "Tair"
        assert df.empty
        assert list(df.columns) == ["Timestamp", "Tair"]

    def test_empty_string_raises_informative_value_error(self):
        with pytest.raises(ValueError, match="The returned CSV is empty"):
            parse_nldas_csv("")

    def test_truncated_header_block_raises_value_error(self):
        truncated = "\n".join(f"key{i},value{i}" for i in range(5)) + "\n"

        with pytest.raises(ValueError, match="The returned CSV is empty"):
            parse_nldas_csv(truncated)

    def test_html_error_page_raises_value_error(self):
        """A non-CSV body (e.g. an auth error page) fails with the CSV message."""
        with pytest.raises(ValueError, match="The returned CSV is empty"):
            parse_nldas_csv("<html><body>401 Unauthorized</body></html>")

    def test_extra_comma_in_header_line_raises_value_error(self):
        lines = [f"key{i},value{i}" for i in range(12)]
        lines.append("param_name,Tair,unexpected")

        with pytest.raises(ValueError, match="The returned CSV is empty"):
            parse_nldas_csv("\n".join(lines) + "\n")

    def test_original_value_error_is_chained(self):
        with pytest.raises(ValueError) as excinfo:
            parse_nldas_csv("")

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_missing_param_name_raises_key_error(self):
        """Metadata without a ``param_name`` key cannot name the data column."""
        lines = [f"key{i},value{i}" for i in range(13)]
        lines += ["Timestamp (UTC),Data", "Timestamp,Data", "2020-01-01T00:00:00,1.0"]

        with pytest.raises(KeyError, match="param_name"):
            parse_nldas_csv("\n".join(lines) + "\n")

    def test_parsing_is_repeatable_for_the_same_string(self):
        """The function takes a string, so the caller can parse it twice."""
        ts = make_nldas_csv()

        first_headers, first_df = parse_nldas_csv(ts)
        second_headers, second_df = parse_nldas_csv(ts)

        assert first_headers == second_headers
        pd.testing.assert_frame_equal(first_df, second_df)


# ---------------------------------------------------------------------------
# fetch_nldas_forcing_dataset
# ---------------------------------------------------------------------------

EXPECTED_VARS = {
    "temp_K": "NLDAS_FORA0125_H_2_0_Tair",
    "spec_hum": "NLDAS_FORA0125_H_2_0_Qair",
    "pressure_pa": "NLDAS_FORA0125_H_2_0_PSurf",
    "wind_u10": "NLDAS_FORA0125_H_2_0_Wind_E",
    "wind_v10": "NLDAS_FORA0125_H_2_0_Wind_N",
    "solar_rad": "NLDAS_FORA0125_H_2_0_SWdown",
}

TIMESTAMPS = ["2020-01-01T00:00:00", "2020-01-01T01:00:00"]


def _param_name_for(param_id):
    """``NLDAS_FORA0125_H_2_0_Tair`` -> ``Tair`` (the CSV's ``param_name``)."""
    return param_id.rsplit("NLDAS_FORA0125_H_2_0_", 1)[-1]


class RecordingFetch:
    """Stands in for ``call_nldas_time_series`` inside the module namespace."""

    def __init__(self, values_by_param=None, timestamps_by_param=None):
        self.values_by_param = values_by_param or {}
        self.timestamps_by_param = timestamps_by_param or {}
        self.calls = []

    def __call__(self, lat, lon, time_start, time_end, data, token):
        self.calls.append(
            {
                "lat": lat,
                "lon": lon,
                "time_start": time_start,
                "time_end": time_end,
                "data": data,
                "token": token,
            }
        )
        # Each variable gets a distinct param_name so the rename is exercised.
        return make_nldas_csv(
            param_name=_param_name_for(data),
            timestamps=self.timestamps_by_param.get(data, TIMESTAMPS),
            values=self.values_by_param.get(data, [1.0, 2.0]),
            param_id=data,
        )


@pytest.fixture
def fake_fetch(monkeypatch):
    fetch = RecordingFetch()
    monkeypatch.setattr(nldas, "call_nldas_time_series", fetch)
    return fetch


class TestFetchNldasForcingDataset:
    def test_requests_every_forcing_variable_once(self, fake_fetch):
        fetch_nldas_forcing_dataset(40.0, -111.0, "start", "end", "tok")

        requested = [c["data"] for c in fake_fetch.calls]
        assert requested == list(EXPECTED_VARS.values())

    def test_forwards_location_time_and_token_unchanged(self, fake_fetch):
        fetch_nldas_forcing_dataset(41.25, -112.5, "t0", "t1", "secret")

        for call in fake_fetch.calls:
            assert call["lat"] == 41.25
            assert call["lon"] == -112.5
            assert call["time_start"] == "t0"
            assert call["time_end"] == "t1"
            assert call["token"] == "secret"

    def test_columns_use_friendly_names(self, fake_fetch):
        df = fetch_nldas_forcing_dataset(40.0, -111.0, "start", "end", "tok")

        assert list(df.columns) == list(EXPECTED_VARS)

    def test_indexed_by_timestamp(self, fake_fetch):
        df = fetch_nldas_forcing_dataset(40.0, -111.0, "start", "end", "tok")

        assert df.index.name == "Timestamp"
        assert pd.api.types.is_datetime64_any_dtype(df.index)
        assert df.index.tolist() == [pd.Timestamp(t) for t in TIMESTAMPS]

    def test_values_land_in_the_right_columns(self, monkeypatch):
        values = {
            param: [float(i), float(i) + 0.5]
            for i, param in enumerate(EXPECTED_VARS.values())
        }
        fetch = RecordingFetch(values_by_param=values)
        monkeypatch.setattr(nldas, "call_nldas_time_series", fetch)

        df = fetch_nldas_forcing_dataset(40.0, -111.0, "start", "end", "tok")

        for i, name in enumerate(EXPECTED_VARS):
            assert df[name].tolist() == [float(i), float(i) + 0.5]

    def test_misaligned_timestamps_are_unioned_with_nan(self, monkeypatch):
        """Variables on different clocks align on the index union, not by position."""
        shifted = ["2020-01-01T01:00:00", "2020-01-01T02:00:00"]
        fetch = RecordingFetch(timestamps_by_param={EXPECTED_VARS["solar_rad"]: shifted})
        monkeypatch.setattr(nldas, "call_nldas_time_series", fetch)

        df = fetch_nldas_forcing_dataset(40.0, -111.0, "start", "end", "tok")

        assert df.index.tolist() == [
            pd.Timestamp("2020-01-01T00:00:00"),
            pd.Timestamp("2020-01-01T01:00:00"),
            pd.Timestamp("2020-01-01T02:00:00"),
        ]
        assert np.isnan(df.loc[pd.Timestamp("2020-01-01T00:00:00"), "solar_rad"])
        assert np.isnan(df.loc[pd.Timestamp("2020-01-01T02:00:00"), "temp_K"])

    def test_reports_progress_for_each_variable(self, fake_fetch, capsys):
        fetch_nldas_forcing_dataset(40.0, -111.0, "start", "end", "tok")

        out = capsys.readouterr().out
        for name in EXPECTED_VARS:
            assert f"Fetching {name}..." in out

    def test_parse_failure_propagates(self, monkeypatch):
        monkeypatch.setattr(nldas, "call_nldas_time_series", lambda *a, **k: "")

        with pytest.raises(ValueError, match="The returned CSV is empty"):
            fetch_nldas_forcing_dataset(40.0, -111.0, "start", "end", "tok")

    def test_end_to_end_through_mocked_requests(self, monkeypatch):
        """Exercises the real call/parse/assemble chain with only HTTP faked."""
        calls = []

        def get(url, params=None, headers=None, **kwargs):
            calls.append({"url": url, "params": params, "headers": headers})
            return FakeResponse(
                make_nldas_csv(
                    param_name=_param_name_for(params["data"]),
                    param_id=params["data"],
                )
            )

        monkeypatch.setattr(nldas.requests, "get", get)

        df = fetch_nldas_forcing_dataset(40.0, -111.0, "t0", "t1", "tok")

        assert [c["params"]["data"] for c in calls] == list(EXPECTED_VARS.values())
        assert list(df.columns) == list(EXPECTED_VARS)
        assert df.shape == (len(TIMESTAMPS), len(EXPECTED_VARS))
        assert df.notna().all().all()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_functions_are_exported_from_package():
    import fluxfootprints

    for name in (
        "call_nldas_time_series",
        "parse_nldas_csv",
        "fetch_nldas_forcing_dataset",
    ):
        assert name in fluxfootprints.__all__
        assert getattr(fluxfootprints, name) is getattr(nldas, name)
