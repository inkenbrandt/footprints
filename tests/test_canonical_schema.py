"""
Contract tests for :mod:`fluxfootprints.canonical_schema`.

The module under test is itself a validator, so these tests are written the way
the contract reads: a *conforming* frame is built once by :func:`good_frame`
and every test perturbs exactly one thing about it, then asserts on the stable
``Issue.code`` values rather than on message wording. That keeps the tests
readable as a list of the violations the contract names, and lets the prose of
a message change without a test failing.

Three properties get checked repeatedly, because they are the ones a benchmark
cannot tolerate losing:

* the validator never mutates the frame it is given,
* a unit problem is *reported*, never silently rescaled, and
* severity is split -- contract violations are errors, rows that
  :class:`~fluxfootprints.FFPModel` merely drops or flags are warnings, so
  ``report.ok`` stays True for the latter.

The registry itself (:data:`CANONICAL_FIELDS`) is checked for internal
consistency too, since the documentation tables are generated from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fluxfootprints.canonical_schema import (
    CANONICAL_FIELDS,
    CARDINAL_SECTORS,
    FFP_INPUT_FIELDS,
    OPTIONAL_FIELDS,
    PROVENANCE_SCHEMA_VERSION,
    REQUIRED_FIELDS,
    WIND_DIR_CONVENTION,
    CanonicalSchemaError,
    FieldSpec,
    Issue,
    Provenance,
    ValidationReport,
    _check_field,  # exclusive-bound branch
    angular_difference,
    assert_canonical,
    canonical_field_table,
    cardinal_sector,
    format_variable_provenance,
    load_provenance_sidecar,
    normalize_units,
    parse_variable_provenance,
    provenance_sidecar_path,
    read_canonical,
    validate_canonical_file,
    validate_canonical_frame,
    wrap_degrees,
    write_provenance_sidecar,
)

N_ROWS = 24


def good_index(n: int = N_ROWS) -> pd.DatetimeIndex:
    """A tz-aware, unique, sorted half-hourly index."""
    return pd.date_range(
        "2024-06-01", periods=n, freq="30min", tz="UTC", name="timestamp"
    )


def good_frame(n: int = N_ROWS, **overrides) -> pd.DataFrame:
    """A frame that validates clean: no errors and no warnings.

    Values are chosen to clear every Kljun et al. (2015) bound as well as the
    contract ranges, so any warning a test sees is the one it introduced.
    """
    df = pd.DataFrame(
        {
            "site_id": "US-Var",
            "ustar": np.linspace(0.3, 0.6, n),
            "sigmav": np.linspace(0.4, 0.7, n),
            "ol": np.linspace(-200.0, -50.0, n),
            "wind_dir": np.linspace(10.0, 350.0, n),
            "umean": np.linspace(2.0, 4.0, n),
            "zm": 3.0,
            "z0": 0.05,
            "h": 1000.0,
        },
        index=good_index(n),
    )
    for name, value in overrides.items():
        df[name] = value
    return df


def good_sidecar(**field_overrides) -> dict:
    """A sidecar that validates clean against :func:`good_frame`."""
    fields: dict[str, object] = {
        "site_id": {"provenance": "fixed_metadata", "units": ""},
        "ustar": {
            "provenance": "observed",
            "units": "m s-1",
            "source_variable": "USTAR",
        },
        "sigmav": {
            "provenance": "prep_calculated",
            "units": "m s-1",
            "method": "predict_sigmav from ustar and zm/L",
        },
        "ol": {
            "provenance": "network_derived",
            "units": "m",
            "source_variable": "MO_LENGTH",
        },
        "wind_dir": {
            "provenance": "observed",
            "units": "degrees",
            "source_variable": "WD",
        },
        "umean": {"provenance": "observed", "units": "m s-1", "source_variable": "WS"},
        "zm": {"provenance": "fixed_metadata", "units": "m"},
        "z0": {"provenance": "fixed_metadata", "units": "m"},
        "h": {
            "provenance": "external_forcing",
            "units": "m",
            "method": "NLDAS-2 planetary boundary layer height",
        },
    }
    fields.update(field_overrides)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "site_id": "US-Var",
        "timestamp_convention": "UTC, start-of-interval, 30 min",
        "wind_dir_convention": WIND_DIR_CONVENTION,
        "data_file": "US-Var_HH.parquet",
        "fields": fields,
    }


# --------------------------------------------------------------------------
# The baseline itself
# --------------------------------------------------------------------------


def test_good_frame_is_clean():
    report = validate_canonical_frame(good_frame())
    assert report.ok
    assert report.issues == [], report.summary()
    assert report.n_rows == N_ROWS
    assert set(report.fields_present) == set(REQUIRED_FIELDS) - {"timestamp"}


def test_good_frame_with_sidecar_is_clean():
    report = validate_canonical_frame(
        good_frame(), provenance=good_sidecar(), require_provenance=True
    )
    assert report.issues == [], report.summary()


def test_validator_never_mutates_the_frame():
    df = good_frame()
    before = df.copy(deep=True)
    validate_canonical_frame(df, provenance=good_sidecar(), strict_columns=True)
    pd.testing.assert_frame_equal(df, before)


def test_non_frame_input_is_a_type_error():
    with pytest.raises(TypeError, match="expected a pandas DataFrame"):
        validate_canonical_frame({"ustar": [0.4]})


# --------------------------------------------------------------------------
# The field registry and its generated table
# --------------------------------------------------------------------------


def test_required_fields_are_the_nine_the_contract_names():
    assert REQUIRED_FIELDS == ("timestamp", "site_id", *FFP_INPUT_FIELDS)
    assert len(FFP_INPUT_FIELDS) == 8
    assert set(REQUIRED_FIELDS).isdisjoint(OPTIONAL_FIELDS)
    assert set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS) == set(CANONICAL_FIELDS)


def test_every_spec_declares_at_least_one_provenance_class():
    for spec in CANONICAL_FIELDS.values():
        assert spec.provenance, spec.name
        assert spec.kind in {"numeric", "string", "datetime"}


def test_canonical_units_are_already_canonically_spelled():
    # normalize_units is the validator's only unit gate, so a registry entry it
    # does not round-trip would make a conforming sidecar unvalidatable.
    for spec in CANONICAL_FIELDS.values():
        assert normalize_units(spec.units) == spec.units, spec.name


def test_canonical_field_table_matches_the_registry():
    table = canonical_field_table()
    assert list(table.columns) == [
        "field",
        "units",
        "meaning",
        "kind",
        "required",
        "allowed_range",
        "provenance",
        "note",
    ]
    assert list(table["field"]) == list(CANONICAL_FIELDS)
    assert int(table["required"].sum()) == len(REQUIRED_FIELDS)
    # Unitless fields render as "-" so the Markdown column is never blank.
    assert table.loc[table["field"] == "site_id", "units"].item() == "-"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (FieldSpec("a", "", "", minimum=0.0, maximum=5.0), "(0, 5]"),
        (
            FieldSpec("a", "", "", minimum=0.0, maximum=5.0, min_inclusive=True),
            "[0, 5]",
        ),
        (
            FieldSpec("a", "", "", minimum=0.0, maximum=5.0, max_inclusive=False),
            "(0, 5)",
        ),
        (FieldSpec("a", "", "", minimum=0.5, maximum=27.5), "(0.5, 27.5]"),
        (FieldSpec("a", "", "", maximum=1.0), "(-inf, 1]"),
        (FieldSpec("a", "", "", minimum=1.0), "(1, inf)"),
        (FieldSpec("a", "", ""), "unbounded"),
        (FieldSpec("a", "", "", kind="string"), "-"),
    ],
)
def test_range_text_interval_notation(spec, expected):
    assert spec.range_text == expected


def test_range_text_formats_large_magnitudes_with_g():
    spec = FieldSpec("a", "", "", minimum=-1e7, maximum=1e7)
    assert spec.range_text == "(-1e+07, 1e+07]"


def test_provenance_text_follows_declaration_order():
    spec = FieldSpec(
        "a",
        "",
        "",
        provenance=frozenset({Provenance.PREP_CALCULATED, Provenance.OBSERVED}),
    )
    assert spec.provenance_text == "observed, prep_calculated"


def test_provenance_member_stringifies_to_its_value():
    assert str(Provenance.OBSERVED) == "observed"
    assert Provenance("network_derived") is Provenance.NETWORK_DERIVED


# --------------------------------------------------------------------------
# Units: spellings only, never conversions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("m/s", "m s-1"),
        ("m s^-1", "m s-1"),
        ("ms-1", "m s-1"),
        ("m.s-1", "m s-1"),
        ("M/S", "m s-1"),
        ("  m/s  ", "m s-1"),
        ("deg", "degrees"),
        ("Degree", "degrees"),
        ("°", "degrees"),
        ("W/m2", "W m-2"),
        ("w m-2", "W m-2"),
        ("degC", "degC"),
        ("celsius", "degC"),
        ("°C", "degC"),
        ("umol/m2/s", "umol m-2 s-1"),
        ("mmol m^-2 s^-1", "mmol m-2 s-1"),
        ("KPA", "kPa"),
        ("meters", "m"),
        ("", ""),
        ("-", ""),
        ("1", ""),
        ("unitless", ""),
        (None, ""),
        ("DEGREES", "degrees"),
    ],
)
def test_normalize_units_spellings(raw, expected):
    assert normalize_units(raw) == expected


def test_normalize_units_never_crosses_physical_units():
    # A rescale here would be a silent benchmark corruption, so Pa stays Pa.
    assert normalize_units("Pa") == "Pa"
    assert normalize_units("hPa") == "hPa"
    assert normalize_units("K") == "K"


def test_normalize_units_collapses_whitespace_in_unknown_spellings():
    assert normalize_units("  furlongs  per   fortnight ") == "furlongs per fortnight"


# --------------------------------------------------------------------------
# Angles
# --------------------------------------------------------------------------


def test_wrap_degrees_scalar_and_array():
    assert wrap_degrees(370.0) == pytest.approx(10.0)
    assert wrap_degrees(-10.0) == pytest.approx(350.0)
    np.testing.assert_allclose(
        wrap_degrees([0.0, 360.0, 720.0, -90.0]), [0.0, 0.0, 0.0, 270.0]
    )


def test_wrap_degrees_preserves_series_type_and_nan():
    s = pd.Series([370.0, np.nan, -10.0], index=list("abc"))
    out = wrap_degrees(s)
    assert isinstance(out, pd.Series)
    pd.testing.assert_index_equal(out.index, s.index)
    assert out.iloc[0] == pytest.approx(10.0)
    assert np.isnan(out.iloc[1])
    assert np.isnan(wrap_degrees(np.nan))


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (10.0, 350.0, 20.0),  # across north, the short way
        (350.0, 10.0, -20.0),
        (90.0, 90.0, 0.0),
        (0.0, 180.0, -180.0),  # an exact reversal lands on the closed end
        (180.0, 0.0, -180.0),
        (450.0, 90.0, 0.0),
    ],
)
def test_angular_difference(a, b, expected):
    assert angular_difference(a, b) == pytest.approx(expected)


def test_angular_difference_is_elementwise():
    np.testing.assert_allclose(
        angular_difference([10.0, 350.0], [350.0, 10.0]), [20.0, -20.0]
    )


def test_angular_difference_never_leaves_the_interval():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-1080, 1080, 500), rng.uniform(-1080, 1080, 500)
    diff = angular_difference(a, b)
    assert np.all(diff >= -180.0) and np.all(diff < 180.0)


@pytest.mark.parametrize(
    ("wd", "expected"),
    [
        (0.0, "N"),
        (355.0, "N"),
        (10.0, "N"),  # the tolerance bound is inclusive
        (10.001, None),
        (90.0, "E"),
        (180.0, "S"),
        (270.0, "W"),
        (45.0, None),
        (-5.0, "N"),  # wrapped before sectoring
        (365.0, "N"),
    ],
)
def test_cardinal_sector_scalar(wd, expected):
    assert cardinal_sector(wd) == expected


def test_cardinal_sector_array_and_nan():
    out = cardinal_sector(np.array([0.0, 90.0, 45.0, np.nan]))
    assert isinstance(out, np.ndarray)
    assert list(out) == ["N", "E", None, None]


def test_cardinal_sector_tolerance_widens_the_window():
    assert cardinal_sector(30.0, tolerance=10.0) is None
    assert cardinal_sector(30.0, tolerance=30.0) == "N"


def test_cardinal_sector_centres_are_the_documented_four():
    assert CARDINAL_SECTORS == {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
    for name, centre in CARDINAL_SECTORS.items():
        assert cardinal_sector(centre) == name


# --------------------------------------------------------------------------
# Issues and reports
# --------------------------------------------------------------------------


def test_issue_str_with_and_without_rows():
    assert str(Issue("error", "out_of_range", "ustar", "too big", 3, ("a", "b"))) == (
        "ERROR out_of_range [ustar]: too big (3 rows, e.g. a, b)"
    )
    assert str(Issue("warning", "empty_frame", None, "no rows")) == (
        "WARNING empty_frame: no rows"
    )


def test_report_partitions_by_severity():
    issues = [
        Issue("error", "missing_column", "h", "gone"),
        Issue("warning", "model_drops_h", "h", "low"),
        Issue("error", "ol_zero", "ol", "zero"),
    ]
    report = ValidationReport(issues=issues, n_rows=5)
    assert [i.code for i in report.errors] == ["missing_column", "ol_zero"]
    assert [i.code for i in report.warnings] == ["model_drops_h"]
    assert report.codes == {"missing_column", "model_drops_h", "ol_zero"}
    assert report.codes_for("error") == {"missing_column", "ol_zero"}
    assert report.codes_for("warning") == {"model_drops_h"}
    assert report.fields_with("model_drops_h") == {"h"}
    assert report.fields_with("nonexistent") == set()
    assert not report.ok


def test_warnings_alone_leave_the_report_ok():
    report = ValidationReport(issues=[Issue("warning", "w", None, "m")])
    assert report.ok
    assert report.raise_for_status() is report


def test_report_to_frame_columns_survive_an_empty_report():
    empty = ValidationReport().to_frame()
    assert list(empty.columns) == [
        "severity",
        "code",
        "field",
        "message",
        "n_rows",
        "examples",
    ]
    assert len(empty) == 0

    frame = ValidationReport(
        issues=[Issue("error", "c", "f", "m", 2, ("x", "y"))]
    ).to_frame()
    assert frame.loc[0, "examples"] == "x, y"
    assert frame.loc[0, "n_rows"] == 2


def test_report_summary_leads_with_the_verdict():
    report = ValidationReport(
        issues=[
            Issue("error", "ol_zero", "ol", "zero"),
            Issue("warning", "w", None, "m"),
        ],
        n_rows=7,
    )
    lines = report.summary().splitlines()
    assert lines[0] == "canonical schema: 1 error(s), 1 warning(s) over 7 rows"
    assert len(lines) == 3
    assert str(report) == report.summary()


def test_raise_for_status_carries_the_report_on_the_exception():
    report = ValidationReport(
        issues=[Issue("error", "ol_zero", "ol", "zero")], n_rows=1
    )
    with pytest.raises(CanonicalSchemaError) as excinfo:
        report.raise_for_status()
    assert excinfo.value.report is report
    assert "1 error(s)" in str(excinfo.value)
    assert isinstance(excinfo.value, ValueError)


# --------------------------------------------------------------------------
# Structure: index and columns
# --------------------------------------------------------------------------


def test_non_datetime_index_is_an_error():
    df = good_frame().reset_index(drop=True)
    report = validate_canonical_frame(df)
    assert "index_not_datetime" in report.codes_for("error")
    # The early return must not suppress the value checks.
    assert report.fields_present


def test_naive_index_is_an_error_by_default_and_a_warning_when_documented():
    df = good_frame()
    df.index = df.index.tz_localize(None)
    assert "index_naive" in validate_canonical_frame(df).codes_for("error")
    lenient = validate_canonical_frame(df, require_tz=False)
    assert "index_naive" in lenient.codes_for("warning")
    assert lenient.ok


def test_nat_in_the_index_is_an_error():
    df = good_frame()
    index = df.index.to_list()
    index[2] = pd.NaT
    df.index = pd.DatetimeIndex(index)
    issue = next(
        i for i in validate_canonical_frame(df).issues if i.code == "index_null"
    )
    assert issue.severity == "error"
    assert issue.n_rows == 1


def test_duplicate_timestamps_are_an_error_with_capped_examples():
    df = good_frame()
    df.index = pd.DatetimeIndex([df.index[0]] * len(df))
    issue = next(
        i for i in validate_canonical_frame(df).issues if i.code == "index_duplicated"
    )
    assert issue.n_rows == len(df) - 1
    assert len(issue.examples) == 5


def test_unsorted_index_is_only_a_warning():
    df = good_frame().iloc[::-1]
    report = validate_canonical_frame(df)
    assert "index_unsorted" in report.codes_for("warning")
    assert report.ok


def test_missing_required_column_is_an_error_naming_the_field():
    df = good_frame().drop(columns=["z0", "h"])
    report = validate_canonical_frame(df)
    assert report.fields_with("missing_column") == {"z0", "h"}
    assert not report.ok


def test_duplicate_column_is_an_error_and_suppresses_value_checks():
    df = good_frame()
    df = pd.concat([df, df[["ustar"]]], axis=1)
    report = validate_canonical_frame(df)
    assert report.fields_with("duplicate_column") == {"ustar"}
    # Indexing by an ambiguous name would raise, so values are left unchecked
    # until the duplicate is fixed.
    assert report.codes == {"duplicate_column"}


def test_empty_frame_is_an_error_and_skips_value_checks():
    df = good_frame().iloc[:0]
    report = validate_canonical_frame(df)
    assert "empty_frame" in report.codes_for("error")
    assert report.n_rows == 0
    assert report.codes == {"empty_frame"}


def test_strict_columns_warns_about_extra_vocabulary_only_when_asked():
    df = good_frame()
    df["FC_1_1_1"] = 1.0
    assert "unrecognized_column" not in validate_canonical_frame(df).codes
    strict = validate_canonical_frame(df, strict_columns=True)
    issue = next(i for i in strict.issues if i.code == "unrecognized_column")
    assert issue.severity == "warning"
    assert "FC_1_1_1" in issue.message
    assert strict.ok


def test_strict_columns_is_silent_on_a_purely_canonical_frame():
    report = validate_canonical_frame(good_frame(), strict_columns=True)
    assert "unrecognized_column" not in report.codes


# --------------------------------------------------------------------------
# Per-field value checks
# --------------------------------------------------------------------------


def test_blank_required_identifier_is_an_error():
    df = good_frame()
    df.loc[df.index[0], "site_id"] = "   "
    df.loc[df.index[1], "site_id"] = None
    issue = next(
        i for i in validate_canonical_frame(df).issues if i.code == "value_missing"
    )
    assert issue.field == "site_id"
    assert issue.n_rows == 2


def test_blank_optional_string_is_tolerated():
    df = good_frame()
    df["source_qc"] = ""
    assert "value_missing" not in validate_canonical_frame(df).codes


def test_non_numeric_required_column_is_an_error():
    df = good_frame()
    df["ustar"] = ["0.4"] * len(df)
    report = validate_canonical_frame(df)
    issue = next(i for i in report.issues if i.code == "non_numeric")
    assert issue.field == "ustar"
    # Range checks are skipped once the dtype is wrong.
    assert "out_of_range" not in report.codes


def test_non_finite_is_an_error_in_a_required_field():
    df = good_frame()
    df.loc[df.index[0], "ustar"] = np.nan
    df.loc[df.index[1], "ustar"] = np.inf
    issue = next(
        i for i in validate_canonical_frame(df).issues if i.code == "non_finite"
    )
    assert issue.severity == "error"
    assert issue.n_rows == 2
    assert "1 NaN, 1 infinite" in issue.message


def test_non_finite_is_only_a_warning_in_an_optional_field():
    df = good_frame(sensible_heat=100.0)
    df.loc[df.index[0], "sensible_heat"] = np.nan
    report = validate_canonical_frame(df)
    issue = next(i for i in report.issues if i.code == "non_finite")
    assert issue.severity == "warning"
    assert "infinite" not in issue.message
    assert report.ok


@pytest.mark.parametrize(
    ("field", "value", "bad"),
    [
        ("ustar", 0.0, True),  # minimum exclusive
        ("ustar", 0.001, False),
        ("ustar", 5.0, False),  # maximum inclusive
        ("ustar", 5.001, True),
        ("umean", 0.0, False),  # minimum inclusive
        ("umean", -0.001, True),
        ("wind_dir", 360.0, False),
        ("wind_dir", 361.0, True),
        ("wind_dir", -1.0, True),
        ("ol", -1e5, False),
        ("ol", -1.1e5, True),
        ("h", 5000.0, False),
        ("h", 5001.0, True),
    ],
)
def test_range_bounds_are_enforced_with_the_declared_inclusivity(field, value, bad):
    df = good_frame()
    df.loc[df.index[0], field] = value
    out = [
        i
        for i in validate_canonical_frame(df).issues
        if i.code == "out_of_range" and i.field == field
    ]
    assert bool(out) is bad
    if bad:
        assert out[0].n_rows == 1


def test_out_of_range_message_quotes_the_interval_and_units():
    df = good_frame()
    df.loc[df.index[0], "ustar"] = 99.0
    issue = next(
        i for i in validate_canonical_frame(df).issues if i.code == "out_of_range"
    )
    assert "(0, 5]" in issue.message and "m s-1" in issue.message


def test_out_of_range_message_omits_a_trailing_space_for_unitless_fields():
    # A unitless field would otherwise render "... range (0, 5] " with a stray
    # trailing space; the rstrip in the message builder is what this pins.
    spec = FieldSpec("gadget", "", "unitless gadget", minimum=0.0, maximum=5.0)
    df = pd.DataFrame({"gadget": [9.0]}, index=good_index(1))
    issues: list[Issue] = []
    _check_field(df, spec, issues)
    assert issues[0].message.endswith("(0, 5]")


def test_one_sided_and_unbounded_numeric_specs():
    # Every numeric field in the registry happens to be bounded on both sides;
    # these keep the half-open and unbounded paths working for the next one.
    df = pd.DataFrame({"gadget": [-1.0, 0.5, 99.0]}, index=good_index(3))

    lower_only: list[Issue] = []
    _check_field(df, FieldSpec("gadget", "", "", minimum=0.0), lower_only)
    assert lower_only[0].n_rows == 1

    upper_only: list[Issue] = []
    _check_field(df, FieldSpec("gadget", "", "", maximum=1.0), upper_only)
    assert upper_only[0].n_rows == 1

    unbounded: list[Issue] = []
    _check_field(df, FieldSpec("gadget", "", ""), unbounded)
    assert unbounded == []


def test_exclusive_maximum_branch():
    # No registry field declares max_inclusive=False, so the branch is covered
    # here directly to keep the option honest for whatever field uses it next.
    spec = FieldSpec("gadget", "", "", minimum=0.0, maximum=5.0, max_inclusive=False)
    df = pd.DataFrame({"gadget": [4.9, 5.0, 5.1]}, index=good_index(3))
    issues: list[Issue] = []
    _check_field(df, spec, issues)
    assert issues[0].code == "out_of_range"
    assert issues[0].n_rows == 2


# --------------------------------------------------------------------------
# Cross-field checks: contract violations
# --------------------------------------------------------------------------


def test_zm_at_or_above_h_is_an_error():
    report = validate_canonical_frame(good_frame(zm=1000.0, h=1000.0))
    assert "zm_ge_h" in report.codes_for("error")
    assert not report.ok


def test_zero_obukhov_length_is_an_error():
    df = good_frame()
    df.loc[df.index[0], "ol"] = 0.0
    issue = next(i for i in validate_canonical_frame(df).issues if i.code == "ol_zero")
    assert issue.severity == "error"
    assert issue.n_rows == 1


def test_zm_must_equal_measurement_minus_displacement_height():
    df = good_frame(measurement_height=5.0, displacement_height=1.0, zm=3.0)
    assert "zm_inconsistent" in validate_canonical_frame(df).codes_for("error")

    ok = good_frame(measurement_height=5.0, displacement_height=1.0, zm=4.0)
    assert "zm_inconsistent" not in validate_canonical_frame(ok).codes

    # A 1 cm tolerance absorbs rounding in the published heights.
    rounded = good_frame(measurement_height=5.0, displacement_height=1.0, zm=4.005)
    assert "zm_inconsistent" not in validate_canonical_frame(rounded).codes


def test_cross_field_checks_skip_columns_that_are_absent_or_non_numeric():
    df = good_frame().drop(columns=["h"])
    assert "zm_ge_h" not in validate_canonical_frame(df).codes

    text = good_frame(h="1000")
    report = validate_canonical_frame(text)
    assert "non_numeric" in report.codes
    assert "zm_ge_h" not in report.codes and "model_drops_h" not in report.codes


# --------------------------------------------------------------------------
# Cross-field checks: rows the model drops or flags (warnings only)
# --------------------------------------------------------------------------


def test_low_ustar_rows_are_warned_not_rejected():
    df = good_frame()
    df.loc[df.index[:3], "ustar"] = 0.05
    report = validate_canonical_frame(df)
    issue = next(i for i in report.issues if i.code == "model_drops_ustar")
    assert issue.severity == "warning"
    assert issue.n_rows == 3
    assert report.ok


def test_shallow_boundary_layer_rows_are_warned():
    report = validate_canonical_frame(good_frame(zm=1.0, z0=0.01, h=8.0))
    assert "model_drops_h" in report.codes_for("warning")
    assert report.ok


def test_roughness_sublayer_rows_are_warned():
    df = good_frame(zm=3.0, z0=0.2)  # 27.5 * 0.2 = 5.5 > 3
    report = validate_canonical_frame(df)
    issue = next(i for i in report.issues if i.code == "model_drops_rsl")
    assert issue.severity == "warning" and issue.field == "zm"
    assert report.ok


def test_upper_kljun_height_bound_is_warned_below_h_only():
    flagged = validate_canonical_frame(good_frame(zm=100.0, h=120.0))
    assert "kljun_height_high" in flagged.codes_for("warning")
    assert flagged.ok

    # Once zm >= h the stronger zm_ge_h error takes over instead.
    above = validate_canonical_frame(good_frame(zm=130.0, h=120.0))
    assert "zm_ge_h" in above.codes_for("error")
    assert "kljun_height_high" not in above.codes


def test_very_unstable_rows_are_warned():
    df = good_frame(zm=3.0, ol=-0.1)  # zm / ol = -30
    report = validate_canonical_frame(df)
    issue = next(i for i in report.issues if i.code == "kljun_stability")
    assert issue.severity == "warning" and issue.field == "ol"
    assert report.ok


def test_stability_check_ignores_the_zero_obukhov_rows_it_cannot_divide():
    df = good_frame()
    df.loc[df.index[0], "ol"] = 0.0
    report = validate_canonical_frame(df)
    assert "ol_zero" in report.codes
    assert "kljun_stability" not in report.codes


# --------------------------------------------------------------------------
# Unit smells
# --------------------------------------------------------------------------


def test_wind_dir_in_radians_is_smelled():
    df = good_frame(n=24, wind_dir=np.linspace(0.0, 6.2, 24))
    issue = next(
        i for i in validate_canonical_frame(df).issues if i.code == "suspect_units"
    )
    assert issue.field == "wind_dir" and issue.severity == "warning"


def test_wind_dir_smell_needs_enough_rows_to_be_confident():
    df = good_frame(n=15, wind_dir=np.linspace(0.0, 6.2, 15))
    assert "suspect_units" not in validate_canonical_frame(df).codes


def test_boundary_layer_height_in_kilometres_is_smelled():
    df = good_frame(zm=0.5, z0=0.01, h=1.5)
    fields = {
        i.field
        for i in validate_canonical_frame(df).issues
        if i.code == "suspect_units"
    }
    assert fields == {"h"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("air_temperature", 293.0),
        ("sonic_temperature", 293.0),
        ("pressure", 101325.0),
    ],
)
def test_temperature_and_pressure_unit_smells(field, value):
    report = validate_canonical_frame(good_frame(**{field: value}))
    assert field in {i.field for i in report.issues if i.code == "suspect_units"}
    # The range check catches it too; the smell explains *why*.
    assert field in report.fields_with("out_of_range")


def test_median_based_smells_need_ten_finite_values():
    df = good_frame(n=8, air_temperature=293.0)
    assert "suspect_units" not in validate_canonical_frame(df).codes


def test_median_based_smells_ignore_non_numeric_columns():
    report = validate_canonical_frame(good_frame(pressure="101325"))
    assert "suspect_units" not in report.codes
    assert "pressure" in report.fields_with("non_numeric")


def test_roughness_above_the_canopy_is_smelled():
    df = good_frame(zm=20.0, z0=0.5, canopy_height=0.3)
    issue = next(
        i for i in validate_canonical_frame(df).issues if i.code == "z0_above_canopy"
    )
    assert issue.severity == "warning" and issue.n_rows == len(df)


def test_zero_canopy_height_does_not_trip_the_roughness_smell():
    df = good_frame(canopy_height=0.0)
    assert "z0_above_canopy" not in validate_canonical_frame(df).codes


# --------------------------------------------------------------------------
# variable_provenance inline spelling
# --------------------------------------------------------------------------


def test_parse_variable_provenance():
    assert parse_variable_provenance("ol=network_derived;z0=prep_calculated") == {
        "ol": "network_derived",
        "z0": "prep_calculated",
    }
    assert parse_variable_provenance(" ol = observed ; h = external_forcing ") == {
        "ol": "observed",
        "h": "external_forcing",
    }


def test_parse_variable_provenance_skips_what_it_cannot_read():
    assert parse_variable_provenance("ol=network_derived;garbage;=x;9bad=observed") == {
        "ol": "network_derived"
    }
    assert parse_variable_provenance("") == {}
    assert parse_variable_provenance(None) == {}
    assert parse_variable_provenance(3.5) == {}


def test_format_variable_provenance_uses_registry_order():
    text = format_variable_provenance(
        {"z0": Provenance.PREP_CALCULATED, "ol": "network_derived"}
    )
    assert text == "ol=network_derived;z0=prep_calculated"


def test_format_variable_provenance_puts_unknown_fields_last():
    text = format_variable_provenance({"mystery": "observed", "ustar": "observed"})
    assert text == "ustar=observed;mystery=observed"


def test_variable_provenance_round_trips():
    mapping = {
        "ol": "network_derived",
        "z0": "prep_calculated",
        "h": "external_forcing",
    }
    assert parse_variable_provenance(format_variable_provenance(mapping)) == mapping


def test_format_variable_provenance_rejects_an_unknown_class():
    with pytest.raises(ValueError):
        format_variable_provenance({"ol": "guessed"})


# --------------------------------------------------------------------------
# Provenance sidecar files
# --------------------------------------------------------------------------


def test_provenance_sidecar_path_replaces_the_data_suffix():
    assert provenance_sidecar_path(Path("processed/US-Var/US-Var_HH.parquet")) == Path(
        "processed/US-Var/US-Var_HH.provenance.json"
    )
    assert provenance_sidecar_path("a/b.csv").name == "b.provenance.json"


def test_write_and_load_sidecar_round_trip(tmp_path):
    data_path = tmp_path / "site" / "US-Var_HH.parquet"
    written = write_provenance_sidecar(
        data_path,
        site_id="US-Var",
        fields={
            "ustar": {
                "provenance": Provenance.OBSERVED,
                "units": "m s-1",
                "source_variable": "USTAR",
            }
        },
        timestamp_convention="UTC, start-of-interval, 30 min",
        source={"dataset": "AmeriFlux FLUXNET", "version": "26-5"},
        notes="h is a reanalysis field",
    )
    assert written == provenance_sidecar_path(data_path)
    assert written.exists()  # the parent directory is created for us

    payload = load_provenance_sidecar(data_path)
    assert payload["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert payload["site_id"] == "US-Var"
    assert payload["data_file"] == "US-Var_HH.parquet"
    assert payload["wind_dir_convention"] == WIND_DIR_CONVENTION
    assert payload["source"]["version"] == "26-5"
    assert payload["notes"] == "h is a reanalysis field"
    # Enum members are serialised as their contract strings, not repr().
    assert payload["fields"]["ustar"]["provenance"] == "observed"
    assert json.loads(written.read_text(encoding="utf-8")) == payload


def test_write_sidecar_omits_absent_optional_sections(tmp_path):
    path = write_provenance_sidecar(
        tmp_path / "x.csv",
        site_id="US-Var",
        fields={},
        timestamp_convention="UTC",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "notes" not in payload
    assert payload["source"] == {}


def test_load_sidecar_reports_the_path_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match="provenance.json"):
        load_provenance_sidecar(tmp_path / "missing.parquet")


# --------------------------------------------------------------------------
# Provenance validation
# --------------------------------------------------------------------------


def _report_for(sidecar, df=None):
    return validate_canonical_frame(
        good_frame() if df is None else df, provenance=sidecar
    )


def test_missing_provenance_is_an_error_only_when_required():
    assert "provenance_absent" not in validate_canonical_frame(good_frame()).codes
    strict = validate_canonical_frame(good_frame(), require_provenance=True)
    assert "provenance_absent" in strict.codes_for("error")


def test_absent_schema_version_is_an_error_and_a_wrong_one_a_warning():
    missing = good_sidecar()
    del missing["schema_version"]
    issue = next(
        i for i in _report_for(missing).issues if i.code == "provenance_schema_version"
    )
    assert issue.severity == "error"

    stale = good_sidecar()
    stale["schema_version"] = "0.9"
    issue = next(
        i for i in _report_for(stale).issues if i.code == "provenance_schema_version"
    )
    assert issue.severity == "warning"
    assert "0.9" in issue.message


def test_implicit_timestamp_convention_is_an_error():
    sidecar = good_sidecar()
    sidecar["timestamp_convention"] = "   "
    report = _report_for(sidecar)
    assert "timestamp_convention_missing" in report.codes_for("error")
    assert report.fields_with("timestamp_convention_missing") == {"timestamp"}


def test_fields_section_must_be_a_mapping():
    sidecar = good_sidecar()
    sidecar["fields"] = ["ustar", "ol"]
    report = _report_for(sidecar)
    assert "provenance_malformed" in report.codes_for("error")
    # It bails out rather than guessing, so no per-field findings follow.
    assert "provenance_missing" not in report.codes


def test_absent_fields_section_reports_every_required_column():
    sidecar = good_sidecar()
    sidecar["fields"] = {}
    report = _report_for(sidecar)
    assert report.fields_with("provenance_missing") == set(REQUIRED_FIELDS) - {
        "timestamp"
    }


def test_a_required_column_absent_from_the_frame_is_not_also_a_provenance_gap():
    df = good_frame().drop(columns=["h"])
    sidecar = good_sidecar()
    del sidecar["fields"]["h"]
    report = validate_canonical_frame(df, provenance=sidecar)
    assert report.fields_with("missing_column") == {"h"}
    assert "provenance_missing" not in report.codes


def test_sidecar_documenting_an_absent_column_is_a_warning():
    sidecar = good_sidecar(
        latent_heat={
            "provenance": "observed",
            "units": "W m-2",
            "source_variable": "LE",
        }
    )
    report = _report_for(sidecar)
    issue = next(i for i in report.issues if i.code == "provenance_orphan")
    assert issue.severity == "warning" and issue.field == "latent_heat"
    assert report.ok


def test_a_record_that_is_not_a_mapping_is_an_error():
    report = _report_for(good_sidecar(ustar="observed"))
    assert "ustar" in report.fields_with("provenance_malformed")
    assert not report.ok


def test_an_unknown_provenance_class_is_an_error():
    sidecar = good_sidecar(ustar={"provenance": "guessed", "units": "m s-1"})
    issue = next(
        i for i in _report_for(sidecar).issues if i.code == "provenance_invalid_class"
    )
    assert issue.field == "ustar"
    assert "observed" in issue.message  # the message lists the vocabulary


def test_a_class_the_field_may_not_carry_is_an_error():
    sidecar = good_sidecar(
        site_id={"provenance": "observed", "units": "", "source_variable": "SITE"}
    )
    issue = next(
        i
        for i in _report_for(sidecar).issues
        if i.code == "provenance_class_not_allowed"
    )
    assert issue.field == "site_id"
    assert "fixed_metadata" in issue.message


@pytest.mark.parametrize(
    ("record", "needs"),
    [
        ({"provenance": "prep_calculated", "units": "m s-1"}, "method"),
        ({"provenance": "prep_calculated", "units": "m s-1", "method": "  "}, "method"),
        ({"provenance": "observed", "units": "m s-1"}, "source_variable"),
        (
            {"provenance": "observed", "units": "m s-1", "source_variable": ""},
            "source_variable",
        ),
    ],
)
def test_nothing_may_be_fabricated(record, needs):
    issue = next(
        i
        for i in _report_for(good_sidecar(ustar=record)).issues
        if i.code == "provenance_undocumented"
    )
    assert issue.severity == "error"
    assert needs in issue.message


def test_external_forcing_also_needs_a_method():
    sidecar = good_sidecar()
    del sidecar["fields"]["h"]["method"]
    assert "provenance_undocumented" in _report_for(sidecar).codes_for("error")


def test_fixed_metadata_needs_neither_method_nor_source_variable():
    assert "provenance_undocumented" not in _report_for(good_sidecar()).codes


def test_declared_units_are_checked_never_converted():
    sidecar = good_sidecar(
        ustar={"provenance": "observed", "units": "cm/s", "source_variable": "USTAR"}
    )
    df = good_frame()
    issue = next(
        i
        for i in validate_canonical_frame(df, provenance=sidecar).issues
        if i.code == "unit_mismatch"
    )
    assert issue.severity == "error" and issue.field == "ustar"
    assert "never here" in issue.message
    # The values themselves are untouched, exactly as documented.
    assert df["ustar"].iloc[0] == pytest.approx(0.3)


def test_a_recognised_unit_spelling_variant_is_accepted():
    sidecar = good_sidecar(
        ustar={"provenance": "observed", "units": "m/s", "source_variable": "USTAR"},
        wind_dir={"provenance": "observed", "units": "deg", "source_variable": "WD"},
    )
    assert "unit_mismatch" not in _report_for(sidecar).codes


def test_units_are_not_checked_for_fields_outside_the_registry():
    sidecar = good_sidecar(
        WS_1_1_1={"provenance": "observed", "units": "m/s", "source_variable": "WS"}
    )
    df = good_frame(WS_1_1_1=3.0)
    report = validate_canonical_frame(df, provenance=sidecar)
    assert "unit_mismatch" not in report.codes
    assert "provenance_orphan" not in report.codes


def test_provenance_may_be_given_as_a_data_path(tmp_path):
    data_path = tmp_path / "US-Var_HH.parquet"
    provenance_sidecar_path(data_path).write_text(
        json.dumps(good_sidecar()), encoding="utf-8"
    )
    report = validate_canonical_frame(good_frame(), provenance=data_path)
    assert report.issues == [], report.summary()
    # A plain string path works the same way.
    assert validate_canonical_frame(good_frame(), provenance=str(data_path)).ok


# --------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------


def test_read_canonical_csv_promotes_the_timestamp_column(tmp_path):
    path = tmp_path / "US-Var_HH.csv"
    good_frame().to_csv(path)
    df = read_canonical(path)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "timestamp"
    assert "timestamp" not in df.columns
    assert df.index.tz is not None
    assert validate_canonical_frame(df).ok


def test_read_canonical_parquet_round_trips_the_index(tmp_path):
    path = tmp_path / "US-Var_HH.parquet"
    original = good_frame()
    original.to_parquet(path)
    # check_freq=False: the index frequency is a pandas attribute, not data,
    # and parquet does not carry it.
    pd.testing.assert_frame_equal(read_canonical(path), original, check_freq=False)


def test_read_canonical_parses_a_datetime_like_index(tmp_path):
    path = tmp_path / "US-Var_HH.parquet"
    df = good_frame(n=3)
    df.index = df.index.astype(str)
    df.to_parquet(path)
    assert isinstance(read_canonical(path).index, pd.DatetimeIndex)


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
def test_read_canonical_leaves_an_unparseable_index_alone(tmp_path):
    path = tmp_path / "odd.parquet"
    df = good_frame(n=3)
    df.index = pd.Index(["alpha", "beta", "gamma"])
    df.to_parquet(path)
    out = read_canonical(path)
    assert not isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "timestamp"
    # And the validator then says so, rather than the reader raising.
    assert "index_not_datetime" in validate_canonical_frame(out).codes


def test_read_canonical_rejects_an_unsupported_file_type(tmp_path):
    path = tmp_path / "US-Var_HH.nc"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported canonical file type"):
        read_canonical(path)


def test_read_canonical_treats_txt_as_delimited_text(tmp_path):
    path = tmp_path / "US-Var_HH.txt"
    good_frame(n=3).to_csv(path)
    assert isinstance(read_canonical(path).index, pd.DatetimeIndex)


@pytest.mark.parametrize("suffix", [".pq", ".parquet"])
def test_read_canonical_accepts_both_parquet_suffixes(tmp_path, suffix):
    path = tmp_path / f"US-Var_HH{suffix}"
    good_frame(n=3).to_parquet(path)
    assert len(read_canonical(path)) == 3


def test_validate_canonical_file_requires_a_sidecar_by_default(tmp_path):
    path = tmp_path / "US-Var_HH.csv"
    good_frame().to_csv(path)
    assert "provenance_absent" in validate_canonical_file(path).codes_for("error")
    assert validate_canonical_file(path, require_provenance=False).ok


def test_validate_canonical_file_picks_up_the_sidecar_beside_it(tmp_path):
    path = tmp_path / "US-Var_HH.csv"
    good_frame().to_csv(path)
    provenance_sidecar_path(path).write_text(
        json.dumps(good_sidecar()), encoding="utf-8"
    )
    report = validate_canonical_file(path)
    assert report.issues == [], report.summary()
    assert report.n_rows == N_ROWS


def test_validate_canonical_file_passes_its_flags_through(tmp_path):
    path = tmp_path / "US-Var_HH.csv"
    df = good_frame(FC_1_1_1=1.0)
    df.index = df.index.tz_localize(None)
    df.to_csv(path)
    report = validate_canonical_file(
        path, require_tz=False, require_provenance=False, strict_columns=True
    )
    assert report.codes_for("warning") == {"index_naive", "unrecognized_column"}
    assert report.ok


# --------------------------------------------------------------------------
# assert_canonical
# --------------------------------------------------------------------------


def test_assert_canonical_returns_the_report_when_clean():
    report = assert_canonical(good_frame(), provenance=good_sidecar())
    assert isinstance(report, ValidationReport)
    assert report.ok


def test_assert_canonical_tolerates_warnings():
    # A preparation script must not be blocked by rows the model merely drops.
    df = good_frame()
    df.loc[df.index[0], "ustar"] = 0.05
    assert assert_canonical(df).codes == {"model_drops_ustar"}


def test_assert_canonical_raises_on_a_contract_violation():
    df = good_frame().drop(columns=["h"])
    with pytest.raises(CanonicalSchemaError) as excinfo:
        assert_canonical(df)
    assert "missing_column" in excinfo.value.report.codes


def test_assert_canonical_forwards_keyword_arguments():
    with pytest.raises(CanonicalSchemaError) as excinfo:
        assert_canonical(good_frame(), require_provenance=True)
    assert excinfo.value.report.codes == {"provenance_absent"}
