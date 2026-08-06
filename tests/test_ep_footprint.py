"""
Unit‑tests for ep_footprint.py

Covered items
-------------
* Footprint.error sentinel helper
* kljun_04               – happy‑path & error branch
* kormann_meixner_01     – basic sanity checks
* hsieh_00               – basic sanity checks
* handle_footprint       – routing logic (“kljun” → “km” fallback, neutral “none”, unknown model)
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append("../src")

from fluxfootprints import ep_footprint as ep  # the module under test


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
ERR_VAL = -9999.0
ORDERED_FIELDS = [
    "offset",
    "x10",
    "x30",
    "x50",
    "x70",
    "x80",
    "x90",
]


def is_error(fp: ep.Footprint) -> bool:
    """True if every field carries the sentinel value."""
    return all(getattr(fp, f) == ERR_VAL for f in fp.__dataclass_fields__)


def assert_monotonic(fp: ep.Footprint):
    """Percentile distances (x10 … x90) must increase strictly."""
    values = [getattr(fp, f) for f in ORDERED_FIELDS]
    assert all(
        a < b for a, b in zip(values, values[1:])
    ), "Footprint distances not monotonic"


# ---------------------------------------------------------------------------
# Footprint.error
# ---------------------------------------------------------------------------
def test_error_helper():
    err = ep.Footprint.error()
    # Every attribute should equal the sentinel value
    assert is_error(err)
    # Dataclass must expose exactly eight fields
    assert len(err.__dataclass_fields__) == 8


# ---------------------------------------------------------------------------
# kljun_04
# ---------------------------------------------------------------------------
def test_kljun04_valid_output():
    fp = ep.kljun_04(
        std_w=0.5,  # √variance of w
        ustar=0.3,
        zL=0.1,
        sonic_height=2.0,
        disp_height=0.0,
        rough_length=0.1,
    )
    # Should not be the error footprint
    assert not is_error(fp)
    # Expected qualitative behaviour
    assert fp.peak > 0
    assert fp.x10 > 0
    assert_monotonic(fp)


def test_kljun04_error_branch():
    # std_w flagged as error → returns sentinel footprint
    fp = ep.kljun_04(
        std_w=ep.ERROR,
        ustar=0.3,
        zL=0.1,
        sonic_height=2.0,
        disp_height=0.0,
        rough_length=0.1,
    )
    assert is_error(fp)


# ---------------------------------------------------------------------------
# kormann_meixner_01
# ---------------------------------------------------------------------------
def test_kormann_meixner_basic():
    fp = ep.kormann_meixner_01(
        ustar=0.3,
        zL=0.1,
        wind_speed=5.0,
        sonic_height=2.0,
        disp_height=0.0,
    )
    assert not is_error(fp)
    assert fp.peak > 0
    assert_monotonic(fp)


# ---------------------------------------------------------------------------
# hsieh_00
# ---------------------------------------------------------------------------
def test_hsieh00_basic():
    fp = ep.hsieh_00(
        MO_length=-50.0,
        sonic_height=2.0,
        disp_height=0.0,
        rough_length=0.1,
    )
    assert not is_error(fp)
    assert fp.peak > 0
    assert_monotonic(fp)


# ---------------------------------------------------------------------------
# handle_footprint routing logic
# ---------------------------------------------------------------------------
def test_handle_footprint_direct_match():
    """When conditions are valid, handle_footprint should call kljun_04."""
    params = dict(
        var_w=0.25,  # (std_w ≈ 0.5)
        ustar=0.3,
        zL=0.1,
        wind_speed=5.0,
        MO_length=-50.0,
        sonic_height=2.0,
        disp_height=0.0,
        rough_length=0.1,
    )
    fp_direct = ep.kljun_04(
        std_w=np.sqrt(params["var_w"]),
        ustar=params["ustar"],
        zL=params["zL"],
        sonic_height=params["sonic_height"],
        disp_height=params["disp_height"],
        rough_length=params["rough_length"],
    )
    fp_handle = ep.handle_footprint(**params, foot_model="kljun_04")
    # All fields should match exactly
    for name in fp_direct.__dataclass_fields__:
        assert getattr(fp_direct, name) == pytest.approx(
            getattr(fp_handle, name), rel=1e-12
        )


def test_handle_footprint_fallback_to_km():
    """If u* is below the Kljun threshold, expect KM model instead."""
    params = dict(
        var_w=0.25,
        ustar=0.1,  # < 0.2 triggers fallback
        zL=0.1,
        wind_speed=5.0,
        MO_length=-50.0,
        sonic_height=2.0,
        disp_height=0.0,
        rough_length=0.1,
    )
    fp_handle = ep.handle_footprint(**params, foot_model="kljun_04")

    # Compare to direct K&M call
    fp_km = ep.kormann_meixner_01(
        ustar=params["ustar"],
        zL=params["zL"],
        wind_speed=params["wind_speed"],
        sonic_height=params["sonic_height"],
        disp_height=params["disp_height"],
    )
    for name in fp_km.__dataclass_fields__:
        assert getattr(fp_handle, name) == pytest.approx(
            getattr(fp_km, name), rel=1e-12
        )


def test_handle_footprint_none_and_unknown():
    none_fp = ep.handle_footprint(
        var_w=0.25,
        ustar=0.3,
        zL=0.1,
        wind_speed=5.0,
        MO_length=-50.0,
        sonic_height=2.0,
        disp_height=0.0,
        rough_length=0.1,
        foot_model="none",
    )
    assert is_error(none_fp)

    unknown_fp = ep.handle_footprint(
        var_w=0.25,
        ustar=0.3,
        zL=0.1,
        wind_speed=5.0,
        MO_length=-50.0,
        sonic_height=2.0,
        disp_height=0.0,
        rough_length=0.1,
        foot_model="does_not_exist",
    )
    assert is_error(unknown_fp)
