# src/fluxfootprints/canonical_schema.py
"""
canonical_schema.py
===================
The canonical benchmark-input contract, and a validator for it.

Every prepared validation dataset under ``data/validation_data/processed/`` is
converted to one table format, so that tests, examples and model-comparison
scripts can consume any site without per-site column editing. This module is
the machine-readable half of that contract; the prose half is
``data/validation_data/schemas/canonical_fluxfootprints_input.md``, whose field
tables are generated from :func:`canonical_field_table` so the two cannot
drift.

The contract has three parts.

1. **Fields.** Nine required columns -- ``timestamp`` (carried as the frame's
   ``DatetimeIndex``), ``site_id``, and the eight
   :class:`~fluxfootprints.FFPModel` inputs ``ustar``, ``sigmav``, ``ol``,
   ``wind_dir``, ``umean``, ``zm``, ``z0``, ``h`` -- plus a fixed vocabulary of
   optional measurement, QC and provenance columns. Each field declares its
   units, meaning, allowed range and the provenance classes it may carry
   (:data:`CANONICAL_FIELDS`).
2. **Provenance.** Every value in a processed table is exactly one of
   :class:`Provenance`: ``observed``, ``network_derived``, ``prep_calculated``,
   ``fixed_metadata`` or ``external_forcing``. A JSON sidecar beside each
   processed file records the class, units, source variable and derivation
   method per column (:func:`write_provenance_sidecar`). Nothing may be
   fabricated: a derived field without a recorded method is a validation error,
   not a warning.
3. **Conventions.** ``wind_dir`` is *meteorological* -- the direction the wind
   comes **from**, degrees clockwise from true north, on ``[0, 360]``. The
   source area therefore lies at bearing ``wind_dir`` from the tower, so
   ``wind_dir = 90`` puts it to the east. See :data:`WIND_DIR_CONVENTION` and
   :func:`cardinal_sector`.

Units are checked, never converted. :func:`validate_canonical_frame` reports a
declared-unit mismatch as an error and returns; it does not rescale the column,
because a silent conversion is exactly the failure a benchmark cannot tolerate.
The validator never mutates the frame it is given.

Severity is split. **Errors** are contract violations: a missing column, a
non-finite value, a value outside its physical range, ``zm >= h``, an
undeclared derivation. **Warnings** are rows the contract permits but
:class:`~fluxfootprints.FFPModel` will drop or flag -- ``ustar <= 0.1``,
``h <= 10``, ``zm <= 27.5 * z0``, ``zm >= 0.8 * h``, ``zm / ol < -15.5`` -- so a
benchmark subset can be chosen to survive the model's own filtering
(``_apply_validity_masks`` and ``check_validity_ranges``) instead of being
silently thinned by it.

Examples
--------
>>> report = validate_canonical_frame(df)          # doctest: +SKIP
>>> report.ok                                      # doctest: +SKIP
True
>>> report.raise_for_status()                      # doctest: +SKIP
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as _dc_field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "WIND_DIR_CONVENTION",
    "CARDINAL_SECTORS",
    "Provenance",
    "FieldSpec",
    "Issue",
    "ValidationReport",
    "CanonicalSchemaError",
    "CANONICAL_FIELDS",
    "REQUIRED_FIELDS",
    "FFP_INPUT_FIELDS",
    "OPTIONAL_FIELDS",
    "canonical_field_table",
    "normalize_units",
    "wrap_degrees",
    "angular_difference",
    "cardinal_sector",
    "parse_variable_provenance",
    "format_variable_provenance",
    "provenance_sidecar_path",
    "write_provenance_sidecar",
    "load_provenance_sidecar",
    "validate_canonical_frame",
    "validate_canonical_file",
    "read_canonical",
    "assert_canonical",
]

#: Version of the provenance-sidecar layout written by
#: :func:`write_provenance_sidecar`. Bump on any breaking key change.
PROVENANCE_SCHEMA_VERSION = "1.0"

#: The wind-direction convention, stated once so scripts can quote it verbatim.
WIND_DIR_CONVENTION = (
    "meteorological: wind_dir is the direction the wind comes FROM, in degrees "
    "clockwise from true north, on [0, 360]; the footprint source area lies "
    "upwind, at bearing wind_dir from the tower"
)

#: Cardinal sector centres used by the benchmark subsets, in degrees.
CARDINAL_SECTORS: dict[str, float] = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}


class Provenance(str, Enum):
    """How a value in a canonical table was obtained.

    Exactly one class applies to each column of a processed table.
    """

    #: Measured and published by the source network.
    OBSERVED = "observed"
    #: Computed by the publisher from its own observations (AmeriFlux
    #: ``MO_LENGTH``, ``ZL``; NEON ``distObkv``).
    NETWORK_DERIVED = "network_derived"
    #: Computed during preparation, from documented inputs by a documented
    #: formula. Requires ``method`` in the sidecar.
    PREP_CALCULATED = "prep_calculated"
    #: A site constant from BADM/BIF or site documentation.
    FIXED_METADATA = "fixed_metadata"
    #: Taken from a different dataset, e.g. a reanalysis boundary-layer height.
    #: Requires ``method`` in the sidecar.
    EXTERNAL_FORCING = "external_forcing"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


_ALL_PROVENANCE = frozenset(Provenance)
_MEASURED = frozenset(
    {Provenance.OBSERVED, Provenance.NETWORK_DERIVED, Provenance.PREP_CALCULATED}
)
_METADATA = frozenset(
    {Provenance.FIXED_METADATA, Provenance.PREP_CALCULATED, Provenance.OBSERVED}
)


@dataclass(frozen=True)
class FieldSpec:
    """The contract for one canonical column.

    Parameters
    ----------
    name
        Canonical column name.
    units
        Canonical units in the ``"m s-1"`` spelling; ``""`` for unitless or
        non-numeric fields.
    meaning
        One-line definition.
    kind
        ``"numeric"``, ``"string"`` or ``"datetime"``.
    required
        Whether a processed table must carry the column.
    minimum, maximum
        Allowed range; ``None`` is unbounded on that side.
    min_inclusive, max_inclusive
        Whether the bounds themselves are allowed.
    provenance
        Provenance classes this field may legitimately carry.
    note
        Model behaviour or preparation caveat worth carrying into the docs.
    """

    name: str
    units: str
    meaning: str
    kind: str = "numeric"
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    min_inclusive: bool = False
    max_inclusive: bool = True
    provenance: frozenset[Provenance] = _ALL_PROVENANCE
    note: str = ""

    @property
    def range_text(self) -> str:
        """The allowed range in interval notation, e.g. ``"(0, 5]"``."""
        if self.minimum is None and self.maximum is None:
            return "unbounded" if self.kind == "numeric" else "-"
        lo = "-inf" if self.minimum is None else _fmt(self.minimum)
        hi = "inf" if self.maximum is None else _fmt(self.maximum)
        left = "[" if (self.min_inclusive and self.minimum is not None) else "("
        right = "]" if (self.max_inclusive and self.maximum is not None) else ")"
        return f"{left}{lo}, {hi}{right}"

    @property
    def provenance_text(self) -> str:
        """Allowed provenance classes, comma-separated in declaration order."""
        return ", ".join(p.value for p in Provenance if p in self.provenance)


def _fmt(value: float) -> str:
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{value:g}"


# --------------------------------------------------------------------------
# Field registry
# --------------------------------------------------------------------------
# Ranges are physical/instrumental limits, deliberately wider than any single
# site: a value outside one is a unit or sign error, not an unusual half hour.
# The narrower Kljun et al. (2015) validity bounds the model itself applies are
# checked separately, at warning severity.

_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="timestamp",
        units="",
        meaning=(
            "Interval timestamp, carried as the frame's DatetimeIndex, which is "
            "what FFPModel turns into the time coordinate"
        ),
        kind="datetime",
        required=True,
        provenance=frozenset({Provenance.OBSERVED, Provenance.PREP_CALCULATED}),
        note="Timezone-aware, or the convention documented per site under metadata/",
    ),
    FieldSpec(
        name="site_id",
        units="",
        meaning="AmeriFlux/NEON site identifier, e.g. US-Var",
        kind="string",
        required=True,
        provenance=frozenset({Provenance.FIXED_METADATA}),
    ),
    FieldSpec(
        name="ustar",
        units="m s-1",
        meaning="Friction velocity",
        required=True,
        minimum=0.0,
        maximum=5.0,
        provenance=_MEASURED,
        note="FFPModel NaNs and drops rows with ustar <= 0.1",
    ),
    FieldSpec(
        name="sigmav",
        units="m s-1",
        meaning="Standard deviation of the lateral wind velocity",
        required=True,
        minimum=0.0,
        maximum=10.0,
        provenance=_MEASURED,
        note="predict_sigmav output is prep_calculated, never observed",
    ),
    FieldSpec(
        name="ol",
        units="m",
        meaning="Obukhov length; negative unstable, positive stable",
        required=True,
        minimum=-1e5,
        maximum=1e5,
        min_inclusive=True,
        provenance=_MEASURED,
        note="Must be non-zero; ol = zm / ZL must reuse the same zm the model is given",
    ),
    FieldSpec(
        name="wind_dir",
        units="degrees",
        meaning="Meteorological wind direction, the direction the wind comes FROM",
        required=True,
        minimum=0.0,
        maximum=360.0,
        min_inclusive=True,
        provenance=_MEASURED,
        note="Wrap before writing; FFPModel NaNs anything outside [0, 360]",
    ),
    FieldSpec(
        name="umean",
        units="m s-1",
        meaning="Mean horizontal wind speed at the measurement height",
        required=True,
        minimum=0.0,
        maximum=60.0,
        min_inclusive=True,
        provenance=_MEASURED,
    ),
    FieldSpec(
        name="zm",
        units="m",
        meaning="Measurement height above the displacement height, z_EC - d",
        required=True,
        minimum=0.0,
        maximum=200.0,
        provenance=frozenset(
            {
                Provenance.PREP_CALCULATED,
                Provenance.FIXED_METADATA,
                Provenance.NETWORK_DERIVED,
            }
        ),
        note="Never a nominal tower height; must be < h",
    ),
    FieldSpec(
        name="z0",
        units="m",
        meaning="Aerodynamic roughness length",
        required=True,
        minimum=0.0,
        maximum=10.0,
        provenance=frozenset(
            {
                Provenance.PREP_CALCULATED,
                Provenance.FIXED_METADATA,
                Provenance.NETWORK_DERIVED,
            }
        ),
        note=(
            "z0 = 0.1 * canopy_height is a labelled fallback approximation, "
            "not an observation"
        ),
    ),
    FieldSpec(
        name="h",
        units="m",
        meaning="Boundary-layer height",
        required=True,
        minimum=0.0,
        maximum=5000.0,
        provenance=frozenset(
            {
                Provenance.EXTERNAL_FORCING,
                Provenance.NETWORK_DERIVED,
                Provenance.PREP_CALCULATED,
                Provenance.OBSERVED,
            }
        ),
        note=(
            "build_climatology's 2000 m default and NEON's constant 1000 m are "
            "placeholders, not observations"
        ),
    ),
    # ---------------- optional: timestamps and geometry ----------------
    FieldSpec(
        name="source_timestamp",
        units="",
        meaning="Timestamp exactly as published, unparsed",
        kind="string",
        provenance=frozenset({Provenance.OBSERVED}),
    ),
    FieldSpec(
        name="measurement_height",
        units="m",
        meaning="Sensor height above ground, z_EC",
        minimum=0.0,
        maximum=200.0,
        provenance=_METADATA,
    ),
    FieldSpec(
        name="displacement_height",
        units="m",
        meaning="Zero-plane displacement height, d",
        minimum=0.0,
        maximum=100.0,
        min_inclusive=True,
        provenance=_METADATA,
    ),
    FieldSpec(
        name="canopy_height",
        units="m",
        meaning="Canopy height, h_c",
        minimum=0.0,
        maximum=120.0,
        min_inclusive=True,
        provenance=_METADATA,
    ),
    # ---------------- optional: fluxes and state ----------------
    FieldSpec(
        name="sensible_heat",
        units="W m-2",
        meaning="Sensible heat flux, H",
        minimum=-500.0,
        maximum=1500.0,
        min_inclusive=True,
        provenance=_MEASURED,
    ),
    FieldSpec(
        name="latent_heat",
        units="W m-2",
        meaning="Latent heat flux, LE",
        minimum=-500.0,
        maximum=1500.0,
        min_inclusive=True,
        provenance=_MEASURED,
    ),
    FieldSpec(
        name="air_temperature",
        units="degC",
        meaning="Air temperature",
        minimum=-90.0,
        maximum=60.0,
        min_inclusive=True,
        provenance=_MEASURED,
        note="degC, not K: a median above 150 is a unit error",
    ),
    FieldSpec(
        name="sonic_temperature",
        units="degC",
        meaning="Sonic temperature",
        minimum=-90.0,
        maximum=60.0,
        min_inclusive=True,
        provenance=_MEASURED,
    ),
    FieldSpec(
        name="pressure",
        units="kPa",
        meaning="Atmospheric pressure",
        minimum=30.0,
        maximum=115.0,
        min_inclusive=True,
        provenance=_MEASURED,
        note="kPa, not Pa or hPa",
    ),
    FieldSpec(
        name="co2_flux",
        units="umol m-2 s-1",
        meaning="CO2 flux, FC",
        minimum=-100.0,
        maximum=100.0,
        min_inclusive=True,
        provenance=_MEASURED,
    ),
    FieldSpec(
        name="h2o_flux",
        units="mmol m-2 s-1",
        meaning="H2O flux, FH2O",
        minimum=-50.0,
        maximum=50.0,
        min_inclusive=True,
        provenance=_MEASURED,
    ),
    # ---------------- optional: QC and provenance ----------------
    FieldSpec(
        name="source_qc",
        units="",
        meaning="Source QA/QC flags, unmodified",
        kind="string",
        provenance=frozenset({Provenance.OBSERVED, Provenance.NETWORK_DERIVED}),
    ),
    FieldSpec(
        name="prep_qc",
        units="",
        meaning="QA decisions made during preparation",
        kind="string",
        provenance=frozenset({Provenance.PREP_CALCULATED}),
    ),
    FieldSpec(
        name="source_file",
        units="",
        meaning="File under raw/ each row came from",
        kind="string",
        provenance=frozenset({Provenance.FIXED_METADATA}),
    ),
    FieldSpec(
        name="source_dataset_version",
        units="",
        meaning="Publisher version string, e.g. AmeriFlux 26-5",
        kind="string",
        provenance=frozenset({Provenance.FIXED_METADATA}),
    ),
    FieldSpec(
        name="source_url",
        units="",
        meaning="Retrieval URL",
        kind="string",
        provenance=frozenset({Provenance.FIXED_METADATA}),
    ),
    FieldSpec(
        name="source_doi",
        units="",
        meaning="DOI of the source product",
        kind="string",
        provenance=frozenset({Provenance.FIXED_METADATA}),
    ),
    FieldSpec(
        name="license",
        units="",
        meaning="License or use policy of the source",
        kind="string",
        provenance=frozenset({Provenance.FIXED_METADATA}),
    ),
    FieldSpec(
        name="variable_provenance",
        units="",
        meaning=(
            "Per-variable provenance, inline as "
            "'ol=network_derived;z0=prep_calculated', or a pointer to the JSON "
            "sidecar"
        ),
        kind="string",
        provenance=frozenset({Provenance.PREP_CALCULATED}),
    ),
)

#: Every canonical field, keyed by name, in contract order.
CANONICAL_FIELDS: dict[str, FieldSpec] = {spec.name: spec for spec in _FIELDS}

#: The nine columns a processed table must carry.
REQUIRED_FIELDS: tuple[str, ...] = tuple(s.name for s in _FIELDS if s.required)

#: The eight FFPModel inputs, in the order the model lists them.
FFP_INPUT_FIELDS: tuple[str, ...] = (
    "ustar",
    "sigmav",
    "ol",
    "wind_dir",
    "umean",
    "zm",
    "z0",
    "h",
)

#: The recommended additional columns.
OPTIONAL_FIELDS: tuple[str, ...] = tuple(s.name for s in _FIELDS if not s.required)


def canonical_field_table() -> pd.DataFrame:
    """Return the field registry as a table, one row per canonical field.

    The Markdown tables in ``schemas/canonical_fluxfootprints_input.md`` are
    generated from this, so the documentation and the validator cannot
    disagree.

    Returns
    -------
    pandas.DataFrame
        Columns ``field, units, meaning, kind, required, allowed_range,
        provenance, note``.
    """
    return pd.DataFrame(
        [
            {
                "field": s.name,
                "units": s.units or "-",
                "meaning": s.meaning,
                "kind": s.kind,
                "required": s.required,
                "allowed_range": s.range_text,
                "provenance": s.provenance_text,
                "note": s.note,
            }
            for s in _FIELDS
        ]
    )


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
# Spellings only. Nothing here converts a value: a declared unit that is not a
# recognised spelling of the canonical unit is an error, never a rescale.

_UNIT_ALIASES: dict[str, str] = {
    "m/s": "m s-1",
    "m s^-1": "m s-1",
    "ms-1": "m s-1",
    "m.s-1": "m s-1",
    "deg": "degrees",
    "degree": "degrees",
    "°": "degrees",
    "w/m2": "W m-2",
    "w m^-2": "W m-2",
    "w m-2": "W m-2",
    "c": "degC",
    "°c": "degC",
    "deg_c": "degC",
    "degc": "degC",
    "celsius": "degC",
    "umol/m2/s": "umol m-2 s-1",
    "umol m^-2 s^-1": "umol m-2 s-1",
    "mmol/m2/s": "mmol m-2 s-1",
    "mmol m^-2 s^-1": "mmol m-2 s-1",
    "kpa": "kPa",
    "metre": "m",
    "meter": "m",
    "metres": "m",
    "meters": "m",
    "": "",
    "-": "",
    "1": "",
    "unitless": "",
}


def normalize_units(units: str | None) -> str:
    """Normalize a unit spelling to its canonical form.

    Recognizes spelling variants only (``"m/s"`` -> ``"m s-1"``). It never maps
    between physically different units: ``"Pa"`` does not become ``"kPa"``.

    Parameters
    ----------
    units
        Declared unit string, or ``None``.

    Returns
    -------
    str
        The canonical spelling, or the input with whitespace collapsed when the
        spelling is not recognized.
    """
    if units is None:
        return ""
    text = re.sub(r"\s+", " ", str(units)).strip()
    key = text.lower()
    if key in _UNIT_ALIASES:
        return _UNIT_ALIASES[key]
    # Canonical spellings pass through untouched, whatever their case.
    for spec in _FIELDS:
        if spec.units and key == spec.units.lower():
            return spec.units
    return text


# --------------------------------------------------------------------------
# Angles
# --------------------------------------------------------------------------


def wrap_degrees(values: Any) -> Any:
    """Wrap bearings onto ``[0, 360)``.

    Parameters
    ----------
    values
        Scalar, array or Series of degrees, possibly negative or above 360.

    Returns
    -------
    Same type as the input, wrapped. NaN is preserved.
    """
    if isinstance(values, pd.Series):
        return values.mod(360.0)
    return np.mod(np.asarray(values, dtype=float), 360.0)


def angular_difference(a: Any, b: Any) -> Any:
    """Signed smallest angle from ``b`` to ``a``, in degrees on ``[-180, 180)``.

    A circular difference, for comparing a modelled footprint bearing against
    an expected one without a wrap-around fault near 0/360.

    Parameters
    ----------
    a, b
        Bearings in degrees.

    Returns
    -------
    Signed difference ``a - b``, wrapped. An exact reversal maps to -180,
    the closed end of the interval; compare magnitudes when only the size of
    the error matters.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return (diff + 180.0) % 360.0 - 180.0


def cardinal_sector(wind_dir: Any, tolerance: float = 10.0) -> Any:
    """Label bearings that fall within ``tolerance`` of a cardinal direction.

    Used to build the N/E/S/W benchmark subsets. Bearings outside every sector
    give ``None``.

    Parameters
    ----------
    wind_dir
        Meteorological wind direction, degrees.
    tolerance
        Half-width of each sector, degrees. The default 10 gives the
        350-10 / 80-100 / 170-190 / 260-280 windows the validation context asks
        for.

    Returns
    -------
    ``"N"``, ``"E"``, ``"S"``, ``"W"`` or ``None``, elementwise.

    Examples
    --------
    >>> cardinal_sector(355.0)
    'N'
    >>> cardinal_sector(45.0) is None
    True
    """
    scalar = np.isscalar(wind_dir)
    wd = wrap_degrees(np.atleast_1d(np.asarray(wind_dir, dtype=float)))
    out = np.full(wd.shape, None, dtype=object)
    finite = np.isfinite(wd)
    for name, centre in CARDINAL_SECTORS.items():
        hit = finite & (np.abs(angular_difference(wd, centre)) <= tolerance)
        out[hit] = name
    return out[0] if scalar else out


# --------------------------------------------------------------------------
# Issues and reports
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Issue:
    """One validation finding.

    Attributes
    ----------
    severity
        ``"error"`` for a contract violation, ``"warning"`` for a row the
        contract permits but ``FFPModel`` will drop or flag.
    code
        Stable machine-readable identifier, e.g. ``"out_of_range"``.
    field
        Canonical field the finding concerns, or ``None`` when it is
        frame-level.
    message
        Human-readable explanation.
    n_rows
        Number of offending rows; 0 for structural findings.
    examples
        Up to five offending index labels, as strings.
    """

    severity: str
    code: str
    field: str | None
    message: str
    n_rows: int = 0
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        where = f" [{self.field}]" if self.field else ""
        rows = (
            f" ({self.n_rows} rows, e.g. {', '.join(self.examples)})"
            if self.n_rows
            else ""
        )
        return f"{self.severity.upper()} {self.code}{where}: {self.message}{rows}"


class CanonicalSchemaError(ValueError):
    """Raised when a frame or file violates the canonical contract."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.summary())


@dataclass
class ValidationReport:
    """The outcome of validating one frame.

    Attributes
    ----------
    issues
        Every finding, errors and warnings alike, in the order found.
    n_rows
        Rows in the validated frame.
    fields_present
        Canonical fields the frame actually carried.
    """

    issues: list[Issue] = _dc_field(default_factory=list)
    n_rows: int = 0
    fields_present: tuple[str, ...] = ()

    @property
    def errors(self) -> list[Issue]:
        """Findings of severity ``"error"``."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        """Findings of severity ``"warning"``."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when there are no errors. Warnings do not affect this."""
        return not self.errors

    @property
    def codes(self) -> set[str]:
        """The set of codes present, for concise assertions in tests."""
        return {i.code for i in self.issues}

    def codes_for(self, severity: str) -> set[str]:
        """The set of codes present at one severity."""
        return {i.code for i in self.issues if i.severity == severity}

    def fields_with(self, code: str) -> set[str | None]:
        """Fields carrying a given code."""
        return {i.field for i in self.issues if i.code == code}

    def to_frame(self) -> pd.DataFrame:
        """Findings as a table, one row per issue."""
        return pd.DataFrame(
            [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "field": i.field,
                    "message": i.message,
                    "n_rows": i.n_rows,
                    "examples": ", ".join(i.examples),
                }
                for i in self.issues
            ],
            columns=["severity", "code", "field", "message", "n_rows", "examples"],
        )

    def summary(self) -> str:
        """Multi-line summary: the verdict line, then one line per finding."""
        head = (
            f"canonical schema: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s) over {self.n_rows} rows"
        )
        return "\n".join([head, *(f"  {i}" for i in self.issues)])

    def raise_for_status(self) -> ValidationReport:
        """Raise :class:`CanonicalSchemaError` if there is any error.

        Returns
        -------
        ValidationReport
            ``self``, so calls can be chained.
        """
        if not self.ok:
            raise CanonicalSchemaError(self)
        return self

    def __str__(self) -> str:  # pragma: no cover - delegation
        return self.summary()


def _examples(index: pd.Index, mask: Any, limit: int = 5) -> tuple[str, ...]:
    labels = index[np.asarray(mask, dtype=bool)][:limit]
    return tuple(str(label) for label in labels)


# --------------------------------------------------------------------------
# Provenance sidecar
# --------------------------------------------------------------------------

_PROVENANCE_ITEM_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([a-z_]+)\s*$")


def parse_variable_provenance(text: str) -> dict[str, str]:
    """Parse the inline ``variable_provenance`` spelling into a mapping.

    Parameters
    ----------
    text
        e.g. ``"ol=network_derived;z0=prep_calculated;h=external_forcing"``.

    Returns
    -------
    dict
        Field name -> provenance class string. Unparseable items are skipped.

    Examples
    --------
    >>> parse_variable_provenance("ol=network_derived;z0=prep_calculated")
    {'ol': 'network_derived', 'z0': 'prep_calculated'}
    """
    out: dict[str, str] = {}
    if not isinstance(text, str):
        return out
    for item in text.split(";"):
        match = _PROVENANCE_ITEM_RE.match(item)
        if match:
            out[match.group(1)] = match.group(2)
    return out


def format_variable_provenance(mapping: Mapping[str, Any]) -> str:
    """Inverse of :func:`parse_variable_provenance`, in field-registry order.

    Parameters
    ----------
    mapping
        Field name -> :class:`Provenance` member or its string value.

    Returns
    -------
    str
        The inline spelling for a ``variable_provenance`` column.
    """
    order = {name: i for i, name in enumerate(CANONICAL_FIELDS)}
    items = sorted(mapping.items(), key=lambda kv: order.get(kv[0], len(order)))
    return ";".join(f"{k}={Provenance(v).value}" for k, v in items)


def provenance_sidecar_path(data_path: str | Path) -> Path:
    """The sidecar path for a processed data file.

    ``processed/US-Var/US-Var_HH.parquet`` ->
    ``processed/US-Var/US-Var_HH.provenance.json``.
    """
    path = Path(data_path)
    return path.with_name(f"{path.stem}.provenance.json")


def write_provenance_sidecar(
    data_path: str | Path,
    *,
    site_id: str,
    fields: Mapping[str, Mapping[str, Any]],
    timestamp_convention: str,
    source: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> Path:
    """Write the JSON provenance sidecar for a processed file.

    Parameters
    ----------
    data_path
        The processed data file the sidecar describes.
    site_id
        AmeriFlux/NEON site ID.
    fields
        Per-column records. Each needs ``provenance`` (a :class:`Provenance`
        member or its value) and ``units``; ``prep_calculated`` and
        ``external_forcing`` also need ``method``, and ``observed`` and
        ``network_derived`` need ``source_variable``. Extra keys
        (``source_file``, ``notes``) are kept as given.
    timestamp_convention
        e.g. ``"UTC, start-of-interval, 30 min"``. Never left implicit.
    source
        Dataset-level provenance: ``dataset``, ``url``, ``doi``, ``version``,
        ``license``, ``retrieved_utc``.
    notes
        Free-text caveats.

    Returns
    -------
    pathlib.Path
        The sidecar path written.
    """
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "site_id": site_id,
        "timestamp_convention": timestamp_convention,
        "wind_dir_convention": WIND_DIR_CONVENTION,
        "data_file": Path(data_path).name,
        "source": dict(source or {}),
        "fields": {
            name: {
                key: (value.value if isinstance(value, Provenance) else value)
                for key, value in record.items()
            }
            for name, record in fields.items()
        },
    }
    if notes:
        payload["notes"] = notes
    path = provenance_sidecar_path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_provenance_sidecar(data_path: str | Path) -> dict[str, Any]:
    """Read the JSON sidecar belonging to a processed file.

    Parameters
    ----------
    data_path
        The processed data file, not the sidecar itself.

    Raises
    ------
    FileNotFoundError
        If no sidecar exists beside ``data_path``.
    """
    path = provenance_sidecar_path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"no provenance sidecar at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_provenance(
    provenance: Mapping[str, Any],
    columns: Iterable[str],
    issues: list[Issue],
) -> None:
    """Check a sidecar against the columns actually present."""
    present = set(columns)

    version = str(provenance.get("schema_version", ""))
    if version != PROVENANCE_SCHEMA_VERSION:
        issues.append(
            Issue(
                "error" if not version else "warning",
                "provenance_schema_version",
                None,
                f"sidecar schema_version {version!r}, expected "
                f"{PROVENANCE_SCHEMA_VERSION!r}",
            )
        )
    if not str(provenance.get("timestamp_convention", "")).strip():
        issues.append(
            Issue(
                "error",
                "timestamp_convention_missing",
                "timestamp",
                "sidecar does not state the timestamp convention; local vs UTC "
                "must never be implicit",
            )
        )

    records = provenance.get("fields") or {}
    if not isinstance(records, Mapping):
        issues.append(
            Issue(
                "error",
                "provenance_malformed",
                None,
                "sidecar 'fields' is not a mapping",
            )
        )
        return

    for name in REQUIRED_FIELDS:
        if name == "timestamp" or name not in present:
            continue
        if name not in records:
            issues.append(
                Issue(
                    "error",
                    "provenance_missing",
                    name,
                    "required field has no provenance record in the sidecar",
                )
            )

    for name, record in records.items():
        if name not in present:
            issues.append(
                Issue(
                    "warning",
                    "provenance_orphan",
                    name,
                    "sidecar documents a column the table does not carry",
                )
            )
        if not isinstance(record, Mapping):
            issues.append(
                Issue(
                    "error",
                    "provenance_malformed",
                    name,
                    "provenance record is not a mapping",
                )
            )
            continue

        raw_class = record.get("provenance")
        try:
            cls = Provenance(str(raw_class))
        except ValueError:
            issues.append(
                Issue(
                    "error",
                    "provenance_invalid_class",
                    name,
                    f"{raw_class!r} is not one of "
                    f"{', '.join(p.value for p in Provenance)}",
                )
            )
            continue

        spec = CANONICAL_FIELDS.get(name)
        if spec is not None and cls not in spec.provenance:
            issues.append(
                Issue(
                    "error",
                    "provenance_class_not_allowed",
                    name,
                    f"provenance {cls.value!r} is not allowed for this field; "
                    f"allowed: {spec.provenance_text}",
                )
            )

        # Nothing may be fabricated: a derivation with no recorded method, or an
        # observation with no source variable, is an error rather than a note.
        derived = cls in (Provenance.PREP_CALCULATED, Provenance.EXTERNAL_FORCING)
        measured = cls in (Provenance.OBSERVED, Provenance.NETWORK_DERIVED)
        if derived and not str(record.get("method", "")).strip():
            issues.append(
                Issue(
                    "error",
                    "provenance_undocumented",
                    name,
                    f"{cls.value} requires a documented 'method'",
                )
            )
        elif measured and not str(record.get("source_variable", "")).strip():
            issues.append(
                Issue(
                    "error",
                    "provenance_undocumented",
                    name,
                    f"{cls.value} requires the publisher's 'source_variable' name",
                )
            )

        if spec is not None:
            declared = normalize_units(record.get("units"))
            if declared != normalize_units(spec.units):
                issues.append(
                    Issue(
                        "error",
                        "unit_mismatch",
                        name,
                        f"sidecar declares units {record.get('units')!r}, "
                        f"canonical units are {spec.units or '-'!r}; convert "
                        "explicitly in the preparation script, never here",
                    )
                )


# --------------------------------------------------------------------------
# Frame validation
# --------------------------------------------------------------------------


def _check_index(df: pd.DataFrame, require_tz: bool, issues: list[Issue]) -> None:
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        issues.append(
            Issue(
                "error",
                "index_not_datetime",
                "timestamp",
                f"index is {type(index).__name__}; the canonical table carries "
                "timestamp as a DatetimeIndex, because FFPModel uses the index "
                "as the time coordinate",
            )
        )
        return

    if index.tz is None:
        issues.append(
            Issue(
                "error" if require_tz else "warning",
                "index_naive",
                "timestamp",
                "index is timezone-naive; make it aware, or document the "
                "convention in metadata/<site>/ and pass require_tz=False",
            )
        )
    if index.hasnans:
        issues.append(
            Issue(
                "error",
                "index_null",
                "timestamp",
                "index contains NaT",
                int(index.isna().sum()),
            )
        )
    if not index.is_unique:
        dupes = index.duplicated()
        issues.append(
            Issue(
                "error",
                "index_duplicated",
                "timestamp",
                "index has duplicate timestamps",
                int(dupes.sum()),
                _examples(index, dupes),
            )
        )
    if not index.is_monotonic_increasing:
        issues.append(
            Issue(
                "warning",
                "index_unsorted",
                "timestamp",
                "index is not sorted ascending",
            )
        )


def _check_field(df: pd.DataFrame, spec: FieldSpec, issues: list[Issue]) -> None:
    series = df[spec.name]
    index = df.index

    if spec.kind == "string":
        blank = series.isna() | series.astype(str).str.strip().eq("")
        if spec.required and blank.any():
            issues.append(
                Issue(
                    "error",
                    "value_missing",
                    spec.name,
                    "required identifier is empty or null",
                    int(blank.sum()),
                    _examples(index, blank),
                )
            )
        return

    if not pd.api.types.is_numeric_dtype(series):
        issues.append(
            Issue(
                "error",
                "non_numeric",
                spec.name,
                f"dtype is {series.dtype}; canonical numeric fields must be "
                "numeric, with missing-value sentinels already converted to NA",
            )
        )
        return

    values = series.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    if not finite.all():
        bad = ~finite
        nan_count = int(np.isnan(values).sum())
        inf_count = int(bad.sum()) - nan_count
        detail = f"{nan_count} NaN" + (f", {inf_count} infinite" if inf_count else "")
        issues.append(
            Issue(
                "error" if spec.required else "warning",
                "non_finite",
                spec.name,
                f"non-finite values ({detail}); FFPModel drops any row with a "
                "NaN in a required field",
                int(bad.sum()),
                _examples(index, bad),
            )
        )

    lo_bad = np.zeros_like(finite)
    hi_bad = np.zeros_like(finite)
    if spec.minimum is not None:
        lo_bad = finite & (
            values < spec.minimum if spec.min_inclusive else values <= spec.minimum
        )
    if spec.maximum is not None:
        hi_bad = finite & (
            values > spec.maximum if spec.max_inclusive else values >= spec.maximum
        )
    out = lo_bad | hi_bad
    if out.any():
        issues.append(
            Issue(
                "error",
                "out_of_range",
                spec.name,
                f"values outside the allowed range {spec.range_text} "
                f"{spec.units}".rstrip(),
                int(out.sum()),
                _examples(index, out),
            )
        )


def _check_cross_field(df: pd.DataFrame, issues: list[Issue]) -> None:
    index = df.index

    def col(name: str) -> np.ndarray | None:
        if name not in df.columns or not pd.api.types.is_numeric_dtype(df[name]):
            return None
        return df[name].to_numpy(dtype=float, na_value=np.nan)

    zm, h, z0, ol, ustar = (col(n) for n in ("zm", "h", "z0", "ol", "ustar"))

    # --- contract violations -------------------------------------------------
    if zm is not None and h is not None:
        bad = np.isfinite(zm) & np.isfinite(h) & (zm >= h)
        if bad.any():
            issues.append(
                Issue(
                    "error",
                    "zm_ge_h",
                    "zm",
                    "zm >= h; the measurement height must sit inside the "
                    "boundary layer (FFPModel NaNs zm where zm > h)",
                    int(bad.sum()),
                    _examples(index, bad),
                )
            )
    if ol is not None:
        bad = np.isfinite(ol) & (ol == 0.0)
        if bad.any():
            issues.append(
                Issue(
                    "error",
                    "ol_zero",
                    "ol",
                    "Obukhov length is exactly zero, so zm/ol is undefined",
                    int(bad.sum()),
                    _examples(index, bad),
                )
            )

    mh, dh = col("measurement_height"), col("displacement_height")
    if zm is not None and mh is not None and dh is not None:
        both = np.isfinite(zm) & np.isfinite(mh) & np.isfinite(dh)
        bad = both & (np.abs(zm - (mh - dh)) > 0.01)
        if bad.any():
            issues.append(
                Issue(
                    "error",
                    "zm_inconsistent",
                    "zm",
                    "zm does not equal measurement_height - displacement_height "
                    "to within 1 cm",
                    int(bad.sum()),
                    _examples(index, bad),
                )
            )

    # --- rows the contract allows but the model drops or flags ---------------
    if ustar is not None:
        bad = np.isfinite(ustar) & (ustar <= 0.1)
        if bad.any():
            issues.append(
                Issue(
                    "warning",
                    "model_drops_ustar",
                    "ustar",
                    "ustar <= 0.1 m s-1; FFPModel NaNs and drops these rows",
                    int(bad.sum()),
                    _examples(index, bad),
                )
            )
    if h is not None:
        bad = np.isfinite(h) & (h <= 10.0)
        if bad.any():
            issues.append(
                Issue(
                    "warning",
                    "model_drops_h",
                    "h",
                    "h <= 10 m; FFPModel NaNs and drops these rows",
                    int(bad.sum()),
                    _examples(index, bad),
                )
            )
    if zm is not None and z0 is not None:
        # The roughness-sublayer drop (zm <= 27.5 * z0) is stricter than Eq. 27's
        # lower height bound (zm <= 20 * z0), so it subsumes it: one check.
        bad = np.isfinite(zm) & np.isfinite(z0) & (zm <= 27.5 * z0)
        if bad.any():
            issues.append(
                Issue(
                    "warning",
                    "model_drops_rsl",
                    "zm",
                    "zm <= 27.5 * z0; inside the roughness sublayer, so FFPModel "
                    "drops these rows (rsl_valid, z* = 2.75 * 10 * z0). This also "
                    "covers the zm > 20 * z0 lower bound of Kljun et al. (2015) "
                    "Eq. 27",
                    int(bad.sum()),
                    _examples(index, bad),
                )
            )
    if zm is not None and h is not None:
        bad = np.isfinite(zm) & np.isfinite(h) & (zm >= 0.8 * h) & (zm < h)
        if bad.any():
            issues.append(
                Issue(
                    "warning",
                    "kljun_height_high",
                    "zm",
                    "zm >= 0.8 * h; above the upper bound of Kljun et al. (2015) "
                    "Eq. 27, so the interval is flagged invalid even though it "
                    "survives the NaN masks",
                    int(bad.sum()),
                    _examples(index, bad),
                )
            )
    if zm is not None and ol is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            stability = np.divide(
                zm,
                ol,
                out=np.full_like(zm, np.nan),
                where=np.isfinite(ol) & (ol != 0.0),
            )
        bad = np.isfinite(stability) & (stability < -15.5)
        if bad.any():
            issues.append(
                Issue(
                    "warning",
                    "kljun_stability",
                    "ol",
                    "zm / ol < -15.5; outside the stability bound of Kljun et al. "
                    "(2015) Eq. 27",
                    int(bad.sum()),
                    _examples(index, bad),
                )
            )


def _check_unit_smells(df: pd.DataFrame, issues: list[Issue]) -> None:
    """Magnitude heuristics that catch a unit error a range check would pass."""

    def median_of(name: str) -> float | None:
        if name not in df.columns or not pd.api.types.is_numeric_dtype(df[name]):
            return None
        values = df[name].to_numpy(dtype=float, na_value=np.nan)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size >= 10 else None

    if "wind_dir" in df.columns and pd.api.types.is_numeric_dtype(df["wind_dir"]):
        values = df["wind_dir"].to_numpy(dtype=float, na_value=np.nan)
        values = values[np.isfinite(values)]
        if values.size >= 20 and values.max() <= 2 * np.pi:
            issues.append(
                Issue(
                    "warning",
                    "suspect_units",
                    "wind_dir",
                    f"every wind_dir is <= {2 * np.pi:.2f}; this looks like "
                    "radians rather than degrees. Convert in the preparation "
                    "script and record it as prep_calculated",
                )
            )

    median_h = median_of("h")
    if median_h is not None and median_h < 10.0:
        issues.append(
            Issue(
                "warning",
                "suspect_units",
                "h",
                f"median h is {median_h:g} m; boundary-layer height in km "
                "rather than m?",
            )
        )

    for name, limit, hint in (
        ("air_temperature", 150.0, "K rather than degC?"),
        ("sonic_temperature", 150.0, "K rather than degC?"),
        ("pressure", 200.0, "Pa or hPa rather than kPa?"),
    ):
        median = median_of(name)
        if median is not None and median > limit:
            issues.append(
                Issue(
                    "warning",
                    "suspect_units",
                    name,
                    f"median {name} is {median:g}; {hint}",
                )
            )

    if {"z0", "canopy_height"} <= set(df.columns):
        z0 = df["z0"].to_numpy(dtype=float, na_value=np.nan)
        hc = df["canopy_height"].to_numpy(dtype=float, na_value=np.nan)
        bad = np.isfinite(z0) & np.isfinite(hc) & (hc > 0.0) & (z0 > hc)
        if bad.any():
            issues.append(
                Issue(
                    "warning",
                    "z0_above_canopy",
                    "z0",
                    "roughness length exceeds the canopy height, which is "
                    "unphysical",
                    int(bad.sum()),
                    _examples(df.index, bad),
                )
            )


def validate_canonical_frame(
    df: pd.DataFrame,
    *,
    provenance: Mapping[str, Any] | str | Path | None = None,
    require_tz: bool = True,
    require_provenance: bool = False,
    strict_columns: bool = False,
) -> ValidationReport:
    """Validate a DataFrame against the canonical benchmark contract.

    The frame is never modified and no value is ever converted: a unit problem
    is reported, not repaired.

    Parameters
    ----------
    df
        Candidate processed table, timestamps carried in a ``DatetimeIndex``.
    provenance
        A loaded sidecar mapping, a path to a processed data file whose sidecar
        should be read, or ``None`` to skip the provenance checks.
    require_tz
        Treat a timezone-naive index as an error (the default). Set ``False``
        only when the timestamp convention is documented in
        ``metadata/<site>/``; the finding is then downgraded to a warning.
    require_provenance
        Error when no provenance is supplied at all. A file destined for
        ``processed/`` should be validated with this on.
    strict_columns
        Warn about columns outside the canonical vocabulary. Off by default,
        since retaining raw source columns is encouraged.

    Returns
    -------
    ValidationReport
        Findings, with :attr:`ValidationReport.ok` False if any is an error.

    Examples
    --------
    >>> validate_canonical_frame(df).raise_for_status()   # doctest: +SKIP
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")

    issues: list[Issue] = []

    duplicated = df.columns[df.columns.duplicated()].unique().tolist()
    for name in duplicated:
        issues.append(
            Issue(
                "error",
                "duplicate_column",
                str(name),
                "column appears more than once",
            )
        )

    _check_index(df, require_tz, issues)

    if len(df) == 0:
        issues.append(Issue("error", "empty_frame", None, "table has no rows"))

    present = [name for name in CANONICAL_FIELDS if name in df.columns]
    for name in REQUIRED_FIELDS:
        if name == "timestamp" or name in df.columns:
            continue
        spec = CANONICAL_FIELDS[name]
        issues.append(
            Issue(
                "error",
                "missing_column",
                name,
                f"required column absent ({spec.meaning}, "
                f"{spec.units or 'no units'})",
            )
        )

    if strict_columns:
        unknown = [str(c) for c in df.columns if c not in CANONICAL_FIELDS]
        if unknown:
            issues.append(
                Issue(
                    "warning",
                    "unrecognized_column",
                    None,
                    "columns outside the canonical vocabulary: "
                    f"{', '.join(sorted(unknown)[:10])}",
                )
            )

    # Value checks index columns by name, which is ambiguous while a name is
    # duplicated; the duplicate is reported above and must be fixed first.
    if len(df) and not duplicated:
        for name in present:
            _check_field(df, CANONICAL_FIELDS[name], issues)
        _check_cross_field(df, issues)
        _check_unit_smells(df, issues)

    if provenance is None:
        if require_provenance:
            issues.append(
                Issue(
                    "error",
                    "provenance_absent",
                    None,
                    "no provenance sidecar supplied; every processed table needs one",
                )
            )
    else:
        if isinstance(provenance, (str, Path)):
            provenance = load_provenance_sidecar(provenance)
        _validate_provenance(provenance, df.columns, issues)

    return ValidationReport(
        issues=issues, n_rows=len(df), fields_present=tuple(present)
    )


def read_canonical(path: str | Path) -> pd.DataFrame:
    """Read a processed canonical file into a frame with a ``DatetimeIndex``.

    Supports ``.parquet``/``.pq`` and ``.csv``. A ``timestamp`` column, when
    present, becomes the index; timestamps are parsed but never re-zoned.

    Parameters
    ----------
    path
        Processed file under ``processed/``.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"unsupported canonical file type {suffix!r}: use .parquet or .csv"
        )

    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"])).drop(columns=["timestamp"])
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError):
            pass
    df.index.name = "timestamp"
    return df


def validate_canonical_file(
    path: str | Path,
    *,
    require_tz: bool = True,
    require_provenance: bool = True,
    strict_columns: bool = False,
) -> ValidationReport:
    """Read a processed file with its sidecar, and validate both.

    Parameters
    ----------
    path
        Processed data file. Its sidecar is read from
        :func:`provenance_sidecar_path` when one exists.
    require_tz, require_provenance, strict_columns
        Passed through to :func:`validate_canonical_frame`. Provenance is
        required here by default, since a file in ``processed/`` is a
        deliverable rather than an intermediate frame.

    Returns
    -------
    ValidationReport
    """
    df = read_canonical(path)
    sidecar = provenance_sidecar_path(path)
    provenance = load_provenance_sidecar(path) if sidecar.exists() else None
    return validate_canonical_frame(
        df,
        provenance=provenance,
        require_tz=require_tz,
        require_provenance=require_provenance,
        strict_columns=strict_columns,
    )


def assert_canonical(df: pd.DataFrame, **kwargs: Any) -> ValidationReport:
    """Validate a frame and raise on any error.

    Convenience wrapper around
    ``validate_canonical_frame(...).raise_for_status()`` for preparation
    scripts, which should refuse to write a non-conforming table.

    Parameters
    ----------
    df
        Candidate processed table.
    **kwargs
        Passed to :func:`validate_canonical_frame`.

    Returns
    -------
    ValidationReport

    Raises
    ------
    CanonicalSchemaError
    """
    return validate_canonical_frame(df, **kwargs).raise_for_status()
