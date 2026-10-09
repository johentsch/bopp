"""Conversions between the columnar ``score_note`` payload and the ``ext`` encoding."""

from __future__ import annotations

from typing import Any

import msgspec

from bopp.core import validate_and_set_annotation_id
from bopp.exceptions import BoppArgumentError
from bopp.io import resolve_extensions
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.payload.ext import ExtensionPayload
from bopp.models.v1.payload.score_note import ScoreNotePayload

from .models import EXT_SCHEMA, ScoreNote

#: Optional ``score_note`` columns, in schema order (``midi``/``tpc`` are required).
_OPTIONAL_COLUMNS = (
    "name",
    "octave",
    "staff",
    "voice",
    "mc",
    "mn",
    "measure",
    "tied",
    "gracenote",
    "nominal_duration",
    "scalar",
    "chord_id",
    "tremolo",
    "volta",
    "tuning",
    "anchor",
)

#: Columns whose schema items allow ``null``; all other columns are non-nullable.
_NULLABLE_COLUMNS = frozenset({"tied", "gracenote", "tremolo", "volta", "tuning"})

#: Columns encoded as ``[numerator, denominator]`` pairs rather than scalars.
_FRACTION_COLUMNS = frozenset({"nominal_duration", "scalar"})


def to_ext(annotation: Annotation, *, resolve_ext: bool = True) -> Annotation:
    """
    Re-encode a columnar ``score_note`` annotation as an ``ext`` annotation.

    Parameters
    ----------
    annotation : Annotation
        An Annotation instance whose ``payload`` is a `ScoreNotePayload`.
    resolve_ext : bool, default True
        If True, resolve the new ``ext`` payload's rows into `ScoreNote`
        instances via `bopp.io.resolve_extensions`. If False, rows remain
        plain dicts.

    Returns
    -------
    Annotation
        A new Annotation instance with an `ExtensionPayload` carrying one
        `ScoreNote`-shaped dict per observation, and a freshly computed id.
        The extent, confidence, metadata, sandbox and parents facets are
        reused unchanged from `annotation`; no lineage to `annotation` is
        recorded in the result. `annotation` itself is not modified.

    Raises
    ------
    BoppArgumentError
        If ``annotation.payload`` is not a `ScoreNotePayload`.

    Notes
    -----
    A column that is present on `annotation.payload` but consists entirely of
    `None` values is dropped: emitting it would require writing explicit
    `None` entries, which this encoding never does. Such a column cannot be
    recovered by `from_ext`.
    """
    payload = annotation.payload
    if not isinstance(payload, ScoreNotePayload):
        raise BoppArgumentError(
            f"to_ext requires a 'score_note' payload, got {type(payload).__name__}"
        )

    rows: list[dict[str, Any]] = []
    for i, (midi, tpc) in enumerate(zip(payload.midi, payload.tpc, strict=True)):
        row: dict[str, Any] = {"midi": midi, "tpc": tpc}
        for name in _OPTIONAL_COLUMNS:
            column = getattr(payload, name)
            if column is msgspec.UNSET:
                continue
            value = column[i]
            if value is None:
                continue
            if name in _FRACTION_COLUMNS:
                value = list(value)
            elif name == "tuning":
                value = float(value)
            row[name] = value
        rows.append(row)

    new_payload = ExtensionPayload(ext_schema=EXT_SCHEMA, value=rows)
    new_annotation = msgspec.structs.replace(annotation, payload=new_payload, id=msgspec.UNSET)
    validate_and_set_annotation_id(new_annotation)

    if resolve_ext:
        resolve_extensions(new_annotation, allow_missing=False)

    return new_annotation


def from_ext(annotation: Annotation) -> Annotation:
    """
    Re-encode an ``ext`` annotation using the score note schema back to a columnar payload.

    Parameters
    ----------
    annotation : Annotation
        An Annotation instance whose ``payload`` is an `ExtensionPayload` with
        ``ext_schema == bopp_score_ext.EXT_SCHEMA``. ``payload.value`` rows may
        already be resolved `ScoreNote` instances or still be raw dicts.

    Returns
    -------
    Annotation
        A new Annotation instance with a `ScoreNotePayload`, and a freshly
        computed id. The extent, confidence, metadata, sandbox and parents
        facets are reused unchanged from `annotation`. `annotation` itself is
        not modified.

    Raises
    ------
    BoppArgumentError
        If ``annotation.payload`` is not an `ExtensionPayload` with the
        expected ``ext_schema``, or if a non-nullable column is present on
        some rows but not on others.
    msgspec.ValidationError
        If a row fails validation against `ScoreNote` (propagated from the
        strict conversion, the same error `resolve_extensions` would raise).

    Notes
    -----
    ``from_ext(to_ext(ann))`` reproduces ``ann`` with the same id whenever
    ``ann`` has no column consisting only of `None` values, since such a
    column is dropped by `to_ext` and cannot be reconstructed here. A row's
    absent optional field decodes to `msgspec.UNSET`, not `None`; it is
    `msgspec.UNSET`, not `None`, that marks a value as missing when
    rebuilding the columnar payload below.
    """
    payload = annotation.payload
    if not isinstance(payload, ExtensionPayload) or payload.ext_schema != EXT_SCHEMA:
        raise BoppArgumentError(
            f"from_ext requires an 'ext' payload with ext_schema '{EXT_SCHEMA}'"
        )

    rows = msgspec.convert(payload.value, list[ScoreNote], strict=True)

    columns: dict[str, Any] = {
        "midi": [row.midi for row in rows],
        "tpc": [row.tpc for row in rows],
    }
    for name in _OPTIONAL_COLUMNS:
        values = [getattr(row, name) for row in rows]
        if all(value is msgspec.UNSET for value in values):
            continue
        if name not in _NULLABLE_COLUMNS and any(value is msgspec.UNSET for value in values):
            raise BoppArgumentError(
                f"Column '{name}' is present on some rows but missing on others"
            )
        if name in _NULLABLE_COLUMNS:
            values = [None if value is msgspec.UNSET else value for value in values]
        if name in _FRACTION_COLUMNS:
            values = [msgspec.to_builtins(value) for value in values]
        columns[name] = values

    new_payload = ScoreNotePayload(**columns)
    new_annotation = msgspec.structs.replace(annotation, payload=new_payload, id=msgspec.UNSET)
    validate_and_set_annotation_id(new_annotation)

    return new_annotation
