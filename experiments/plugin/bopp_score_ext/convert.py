"""Table-driven conversions between columnar score payloads and extension rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import msgspec
from msgspec import inspect as mi

from bopp.core import validate_and_set_annotation_id
from bopp.exceptions import BoppArgumentError
from bopp.io import resolve_extensions
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.payload.ext import ExtensionPayload
from bopp.registries.v1 import PAYLOAD_TYPE_REGISTRY

from .events import EVENT_EXT_SCHEMAS, EVENT_KINDS, EVENT_ROW_TYPES
from .models import EXT_SCHEMA, ScoreNote


@dataclass(frozen=True)
class _Column:
    name: str
    required: bool
    nullable: bool
    fraction: bool
    floating: bool


def _unwrap(info: mi.Type) -> mi.Type:
    while isinstance(info, mi.Metadata):
        info = info.type
    return info


def _columns(payload_type: type[msgspec.Struct]) -> tuple[_Column, ...]:
    """Inspect once, retaining generated field order and item nullability."""
    info = mi.type_info(payload_type)
    assert isinstance(info, mi.StructType)
    columns = []
    for field in info.fields:
        column = _unwrap(field.type)
        assert isinstance(column, mi.ListType)
        item = _unwrap(column.item_type)
        items = item.types if isinstance(item, mi.UnionType) else (item,)
        columns.append(
            _Column(
                field.name,
                field.required,
                any(isinstance(t, mi.NoneType) for t in items),
                any(isinstance(t, mi.TupleType) for t in items),
                any(isinstance(t, mi.FloatType) for t in items),
            )
        )
    return tuple(columns)


@dataclass(frozen=True)
class _Encoding:
    payload_type: type[msgspec.Struct]
    row_type: type[msgspec.Struct]
    ext_schema: str
    columns: tuple[_Column, ...]


_SPECS: list[tuple[str, type[msgspec.Struct], str]] = [
    ("score_note", ScoreNote, EXT_SCHEMA),
    *(
        (
            f"score_control_event.{kind}",
            EVENT_ROW_TYPES[kind],
            EVENT_EXT_SCHEMAS[kind],
        )
        for kind in EVENT_KINDS
    ),
]
_ENCODINGS = tuple(
    _Encoding(payload_type, row_type, schema, _columns(payload_type))
    for kind, row_type, schema in _SPECS
    for payload_type in [cast(type[msgspec.Struct], PAYLOAD_TYPE_REGISTRY[kind])]
)
_BY_PAYLOAD = {encoding.payload_type: encoding for encoding in _ENCODINGS}
_BY_SCHEMA = {encoding.ext_schema: encoding for encoding in _ENCODINGS}


def to_ext(annotation: Annotation, *, resolve_ext: bool = True) -> Annotation:
    """Convert score notes or control events to rows, preserving all other facets.

    Optional null cells are omitted. Entirely null columns cannot be recovered
    by from_ext. Fractions and floats use canonical wire representations.
    """
    payload = annotation.payload
    encoding = _BY_PAYLOAD.get(type(payload))
    if encoding is None:
        raise BoppArgumentError(
            f"to_ext requires a score payload, got {type(payload).__name__}"
        )

    present = [
        (col, getattr(payload, col.name))
        for col in encoding.columns
        if getattr(payload, col.name) is not msgspec.UNSET
    ]
    rows: list[dict[str, Any]] = []
    for values in zip(*(values for _, values in present), strict=True):
        row: dict[str, Any] = {}
        for (col, _), value in zip(present, values, strict=True):
            if value is None:
                continue
            if col.fraction:
                value = list(value)
            elif col.floating:
                value = float(value)
            row[col.name] = value
        rows.append(row)

    new_payload = ExtensionPayload(ext_schema=encoding.ext_schema, value=rows)
    result = msgspec.structs.replace(annotation, payload=new_payload, id=msgspec.UNSET)
    validate_and_set_annotation_id(result)
    if resolve_ext:
        resolve_extensions(result, allow_missing=False)
    return result


def from_ext(annotation: Annotation) -> Annotation:
    """Restore score columns from strictly validated rows in generated field order.

    All-missing optional columns are omitted; nullable cells become None.
    Partially missing non-nullable columns raise BoppArgumentError. Resolution
    errors propagate as msgspec.ValidationError. Input facets remain unchanged.
    """
    payload = annotation.payload
    if (
        not isinstance(payload, ExtensionPayload)
        or payload.ext_schema not in _BY_SCHEMA
    ):
        raise BoppArgumentError("from_ext requires a supported score extension schema")
    encoding = _BY_SCHEMA[payload.ext_schema]
    row_type = encoding.row_type
    rows = msgspec.convert(payload.value, list[row_type], strict=True)  # type: ignore[valid-type]
    columns: dict[str, Any] = {}
    for col in encoding.columns:
        values = [getattr(row, col.name) for row in rows]
        if not col.required and all(value is msgspec.UNSET for value in values):
            continue
        if not col.nullable and any(value is msgspec.UNSET for value in values):
            raise BoppArgumentError(
                f"Column '{col.name}' is present on some rows but missing on others"
            )
        if col.nullable:
            values = [None if value is msgspec.UNSET else value for value in values]
        if col.fraction:
            values = [msgspec.to_builtins(value) for value in values]
        columns[col.name] = values

    new_payload = encoding.payload_type(**columns)
    result = msgspec.structs.replace(annotation, payload=new_payload, id=msgspec.UNSET)
    validate_and_set_annotation_id(result)
    return result
