import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from fluxfootprints import compute_aerodynamic_params

@pytest.fixture
def sample_df():
    """Fixture providing a dummy flux DataFrame with 30-minute time-series data."""
    dates = pd.date_range("2024-06-01", periods=5, freq="30min")
    return pd.DataFrame(
        {
            "crop_h": [1.0, 1.5, 2.0, 2.4, 2.5],
            "inst_h": [3.5, 3.5, 3.5, 3.5, 3.5],
            "custom_z0_r": [0.10] * 5,
            "custom_dh_r": [0.65] * 5,
        },
        index=dates,
    )


# -----------------------------------------------------------------------------
# 1. Preset Physics & Equations
# -----------------------------------------------------------------------------


def test_preset_corn_jacobs(sample_df):
    """Test Jacobs & Van Boxel (1988) maize parameterization: z0 = 0.26 * (h - d)."""
    res = compute_aerodynamic_params(
        sample_df,
        inst_height=3.0,
        crop_height=2.0,
        veg_type="corn_jacobs",
    )
    # Expected math for h_c = 2.0, inst_height = 3.0:
    # d_h = 0.75 * 2.0 = 1.5 -> zm = 3.0 - 1.5 = 1.5
    # z0  = 0.26 * (2.0 - 1.5) = 0.13
    assert np.allclose(res["zm"], 1.5)
    assert np.allclose(res["z0"], 0.13)


def test_preset_alfalfa(sample_df):
    """Test standard FAO/ASCE Alfalfa preset."""
    res = compute_aerodynamic_params(
        sample_df,
        inst_height=3.0,
        crop_height=1.0,
        veg_type="alfalfa",
    )
    # d_h = 0.67 * 1.0 = 0.67 -> zm = 3.0 - 0.67 = 2.33
    # z0  = 0.123 * 1.0 = 0.123
    assert np.allclose(res["zm"], 2.33)
    assert np.allclose(res["z0"], 0.123)


def test_preset_stanhill(sample_df):
    """Test Stanhill (1969) empirical non-linear displacement height model."""
    res = compute_aerodynamic_params(
        sample_df,
        inst_height=3.0,
        crop_height=2.0,
        veg_type="stanhill",
    )
    expected_dh = 10 ** (0.979 * np.log10(2.0) - 0.154)
    expected_zm = 3.0 - expected_dh
    expected_z0 = 0.123 * 2.0

    assert np.allclose(res["zm"], expected_zm)
    assert np.allclose(res["z0"], expected_z0)


# -----------------------------------------------------------------------------
# 2. Custom Ratios & Overrides
# -----------------------------------------------------------------------------


def test_custom_ratios_override(sample_df):
    """Test passing custom z0_ratio and d_h_ratio without veg_type."""
    res = compute_aerodynamic_params(
        sample_df,
        inst_height=3.0,
        crop_height=2.0,
        z0_ratio=0.12,
        d_h_ratio=0.67,
    )
    assert np.allclose(res["zm"], 3.0 - (2.0 * 0.67))
    assert np.allclose(res["z0"], 2.0 * 0.12)


def test_hybrid_override(sample_df):
    """Test providing veg_type with explicit z0_ratio override."""
    res = compute_aerodynamic_params(
        sample_df,
        inst_height=3.0,
        crop_height=2.0,
        veg_type="alfalfa",
        z0_ratio=0.15,  # Overrides alfalfa preset default (0.123)
    )
    # d_h derived from alfalfa preset (0.67 * 2.0 = 1.34)
    assert np.allclose(res["zm"], 3.0 - 1.34)
    # z0 derived from custom ratio override (0.15 * 2.0 = 0.30)
    assert np.allclose(res["z0"], 0.30)


# -----------------------------------------------------------------------------
# 3. Type Flexibility (Scalars, Column Strings, Series)
# -----------------------------------------------------------------------------


def test_column_name_inputs(sample_df):
    """Test referencing DataFrame column names as strings."""
    res = compute_aerodynamic_params(
        sample_df,
        inst_height="inst_h",
        crop_height="crop_h",
        veg_type="corn_jacobs",
    )
    expected_dh = sample_df["crop_h"] * 0.75
    expected_zm = sample_df["inst_h"] - expected_dh
    expected_z0 = 0.26 * (sample_df["crop_h"] - expected_dh)

    pd.testing.assert_series_equal(res["zm"], expected_zm, check_names=False)
    pd.testing.assert_series_equal(res["z0"], expected_z0, check_names=False)


def test_standalone_series_input(sample_df):
    """Test passing an external pandas Series with unaligned index order."""
    external_crop_h = pd.Series([1.0, 1.2, 1.5, 1.8, 2.0], index=sample_df.index[::-1])

    res = compute_aerodynamic_params(
        sample_df,
        inst_height=3.0,
        crop_height=external_crop_h,
        veg_type="alfalfa",
    )
    # Ensure reindexing prevents silent NaNs
    assert not res["zm"].isna().any()
    assert not res["z0"].isna().any()


# -----------------------------------------------------------------------------
# 4. Error Handling & Input Validation
# -----------------------------------------------------------------------------


def test_missing_all_rules_raises_error(sample_df):
    """Test raising error when neither veg_type nor custom ratios are supplied."""
    with pytest.raises(ValueError, match="Missing parameter rule!"):
        compute_aerodynamic_params(
            sample_df,
            inst_height=3.0,
            crop_height=1.0,
        )


def test_missing_dh_rule_raises_error(sample_df):
    """Test raising error when z0_ratio is provided but d_h_ratio and veg_type are omitted."""
    with pytest.raises(ValueError, match="Missing displacement height rule!"):
        compute_aerodynamic_params(
            sample_df,
            inst_height=3.0,
            crop_height=1.0,
            z0_ratio=0.12,
        )


def test_invalid_veg_type_raises_error(sample_df):
    """Test raising error when an unsupported crop preset string is passed."""
    with pytest.raises(ValueError, match="Unknown veg_type 'banana_tree'"):
        compute_aerodynamic_params(
            sample_df,
            inst_height=3.0,
            crop_height=1.0,
            veg_type="banana_tree",
        )


def test_missing_column_raises_keyerror(sample_df):
    """Test raising KeyError when a specified column name string is missing from df."""
    with pytest.raises(KeyError, match="non_existent_col"):
        compute_aerodynamic_params(
            sample_df,
            inst_height=3.0,
            crop_height="non_existent_col",
            veg_type="alfalfa",
        )