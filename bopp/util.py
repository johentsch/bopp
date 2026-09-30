#!/usr/bin/env python

from fractions import Fraction
from typing import Annotated, Any, get_args, get_origin

import msgspec
import numpy as np
import pandas as pd
import pyarrow as pa

from .models.v1.annotation import Annotation
from .models.v1.core import FractionBuffer
from .models.v1.event_block import EventBlock
from .models.v1.object import AnyObject
from .models.v1.score_object import ScoreObject

PAYLOAD_TYPE_COLUMN = "payload_type"

# ==========================================
# 1. Struct <-> DataFrame Translators
# ==========================================

def _get_tag(struct: msgspec.Struct) -> str | None:
    """
    Retrieves the tag value from a msgspec Struct, if it has one.
    """
    config = getattr(struct, "__struct_config__", None)
    if config is not None:
        return getattr(config, "tag", None)
    return None


def _unwrap(type_hint: Any) -> Any:
    """Strip Annotated / Union[..., UnsetType] wrappers down to the core type."""
    while True:
        if hasattr(type_hint, "__value__"):  # PEP 695 `type X = ...` alias
            type_hint = type_hint.__value__
        elif get_origin(type_hint) is Annotated:
            type_hint = get_args(type_hint)[0]
        else:
            return type_hint


def _is_coordinate(value: Any) -> bool:
    config = getattr(value, "__struct_config__", None)
    return config is not None and getattr(config, "tag_field", None) == "unit"


def _to_pylist(value: Any) -> list:
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return value.to_pylist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def fractions_from_buffer(buffer: FractionBuffer) -> list[Fraction | None]:
    """Materialize a struct-of-arrays FractionBuffer as Python Fractions (None for nulls)."""
    return [
        None if n is None or d is None else Fraction(n, d)
        for n, d in zip(_to_pylist(buffer.numerator), _to_pylist(buffer.denominator))
    ]


def fractions_to_buffer(values: Any) -> dict[str, list[int | None]]:
    """Split Fractions (or things Fraction() accepts) into the numerator/denominator arrays."""
    numerators, denominators = [], []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            numerators.append(None)
            denominators.append(None)
            continue
        fraction = value if isinstance(value, Fraction) else Fraction(str(value))
        numerators.append(fraction.numerator)
        denominators.append(fraction.denominator)
    return {"numerator": numerators, "denominator": denominators}


def _column_values(value: Any) -> Any:
    """Flatten one struct field into a DataFrame-ready column."""
    if isinstance(value, FractionBuffer):
        return fractions_from_buffer(value)
    if _is_coordinate(value):
        return _column_values(value.values)
    return value


def extract_header(obj: Annotation | ScoreObject | EventBlock) -> dict[str, Any]:
    """
    Extracts all singleton fields from an object for YAML serialization,
    explicitly omitting the parallel array blocks.
    """
    excluded_fields = {"extent", "payload", "confidence", "events"}

    header_data = {}
    tag = _get_tag(obj)
    if tag is not None:
        header_data["object_type"] = tag

    for field in msgspec.structs.fields(obj):
        if field.name in excluded_fields:
            continue

        value = getattr(obj, field.name)

        if value is not None and value is not msgspec.UNSET:
            header_data[field.name] = msgspec.to_builtins(value)

    return header_data


def _block_columns(block: Annotation | EventBlock) -> dict[str, Any]:
    """Self-describing columns for one homogeneous block of parallel arrays."""
    data = {}

    # 1. Extents: coordinates get a fourth header segment carrying their unit
    extent = getattr(block, "extent", msgspec.UNSET)
    if extent is not None and extent is not msgspec.UNSET:
        ext_type = _get_tag(extent)
        for field in msgspec.structs.fields(type(extent)):
            if field.name == "extent_type":
                continue
            val = getattr(extent, field.name)
            name = f"extent:{ext_type}:{field.name}"
            if _is_coordinate(val):
                name = f"{name}:{_get_tag(val)}"
            data[name] = _column_values(val)

    # 2. Payload
    payload_type = _get_tag(block.payload)
    for field in msgspec.structs.fields(type(block.payload)):
        if field.name == "payload_type":
            continue
        data[f"payload:{payload_type}:{field.name}"] = _column_values(getattr(block.payload, field.name))

    # 3. Confidence
    confidence = getattr(block, "confidence", msgspec.UNSET)
    if confidence is not None and confidence is not msgspec.UNSET:
        conf_type = _get_tag(confidence)
        for field in msgspec.structs.fields(type(confidence)):
            if field.name == "confidence_type":
                continue
            data[f"confidence:{conf_type}:{field.name}"] = _column_values(getattr(confidence, field.name))

    return data


def _sort_key(df: pd.DataFrame) -> pd.Series | None:
    start_cols = [c for c in df.columns if c.startswith("extent:") and c.split(":")[2] in ("start", "time")]
    if not start_cols:
        return None
    return df[start_cols[0]].map(lambda v: float(v) if v is not None else float("inf"))


def to_dataframe(obj: Annotation | ScoreObject | EventBlock) -> pd.DataFrame:
    """
    Converts an Annotation (or one EventBlock) into a DataFrame with self-describing
    column headers. A ScoreObject becomes the outer union of its blocks: a sparse table,
    sorted by start position, with a `payload_type` column naming each row's block.
    Singleton metadata is preserved in df.attrs.
    """
    if isinstance(obj, ScoreObject):
        frames = []
        for block in obj.events:
            # Object columns of Python values keep ints as ints (and nulls as None)
            # once blocks with different column sets are unioned.
            columns = {name: pd.Series([_to_pylist(v) if isinstance(v, (pa.Array, pa.ChunkedArray)) else list(v)][0], dtype=object) for name, v in _block_columns(block).items()}
            frame = pd.DataFrame(columns)
            frame.insert(0, PAYLOAD_TYPE_COLUMN, _get_tag(block.payload))
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[PAYLOAD_TYPE_COLUMN])
        df = df.astype(object).where(df.notna(), None)
        key = _sort_key(df)
        if key is not None:
            df = df.iloc[np.argsort(key.to_numpy(), kind="stable")].reset_index(drop=True)
    else:
        df = pd.DataFrame(_block_columns(obj))

    # Stash the singleton data safely into the DataFrame's attributes
    df.attrs.update(extract_header(obj))
    return df


# ==========================================
# 2. DataFrame -> Struct
# ==========================================

def _field_types(struct_type: type) -> dict[str, Any]:
    return {field.name: _unwrap(field.type) for field in msgspec.structs.fields(struct_type)}


def _payload_struct(payload_type: str) -> type:
    from .models.v1 import payloads

    for member in get_args(_unwrap(payloads.AnyPayload)):
        if member.__struct_config__.tag == payload_type:
            return member
    raise ValueError(f"Unknown payload_type {payload_type!r}")


def _coordinate_struct(unit: str) -> type:
    from .models.v1 import coordinate

    for member in get_args(_unwrap(coordinate.Coordinate)):
        if member.__struct_config__.tag == unit:
            return member
    raise ValueError(f"Unknown coordinate unit {unit!r}")


def _restore_values(values: list, field_type: Any) -> Any:
    """Undo the CSV flattening for one column, given the target struct field type."""
    if field_type is FractionBuffer:
        return fractions_to_buffer(values)
    return values


def _block_from_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Rebuild the raw dict of one homogeneous block from its self-describing columns."""
    block: dict[str, Any] = {}

    coord_cols = [c for c in df.columns if c.startswith("extent:")]
    payload_cols = [c for c in df.columns if c.startswith("payload:")]
    conf_cols = [c for c in df.columns if c.startswith("confidence:")]

    if coord_cols:
        ext_type = coord_cols[0].split(":")[1]
        block["extent"] = {"extent_type": ext_type}
        for col in coord_cols:
            parts = col.split(":")
            field_name = parts[2] if len(parts) > 2 else "values"
            values = df[col].tolist()
            if len(parts) > 3:
                unit = parts[3]
                # The unit decides the number type; fractions were written as "n/d".
                values_type = _field_types(_coordinate_struct(unit))["values"]
                block["extent"][field_name] = {"unit": unit, "values": _restore_values(values, values_type)}
            else:
                block["extent"][field_name] = values

    if payload_cols:
        payload_type = payload_cols[0].split(":")[1]
        types = _field_types(_payload_struct(payload_type))
        block["payload"] = {"payload_type": payload_type}
        for col in payload_cols:
            parts = col.split(":")
            field_name = parts[2] if len(parts) > 2 else "values"
            block["payload"][field_name] = _restore_values(df[col].tolist(), types.get(field_name))

    if conf_cols:
        conf_type = conf_cols[0].split(":")[1]
        block["confidence"] = {"confidence_type": conf_type}
        for col in conf_cols:
            parts = col.split(":")
            field_name = parts[2] if len(parts) > 2 else "confidence"
            block["confidence"][field_name] = df[col].tolist()

    return block


def from_dataframe(df: pd.DataFrame, dec_hook=None) -> Annotation | ScoreObject:
    """
    Reconstitutes a strictly typed object from a DataFrame.
    Expects singletons (media_id, metadata, object_type) to be present in df.attrs.
    """
    header = {
        "bopp_version": df.attrs.get("bopp_version", "1.0.0"),
        "media_id": df.attrs.get("media_id", "unknown:media"),
    }
    if "metadata" in df.attrs:
        header["metadata"] = df.attrs["metadata"]

    if df.attrs.get("object_type") == "score_object" or PAYLOAD_TYPE_COLUMN in df.columns:
        events = []
        for payload_type, rows in df.groupby(PAYLOAD_TYPE_COLUMN, sort=False):
            prefix = f"payload:{payload_type}:"
            keep = [c for c in rows.columns if not c.startswith("payload:") or c.startswith(prefix)]
            events.append(_block_from_columns(rows[keep].drop(columns=[PAYLOAD_TYPE_COLUMN])))
        raw = {"object_type": "score_object", **header, "events": events}
        return msgspec.convert(raw, type=ScoreObject, dec_hook=dec_hook)

    raw = {"object_type": "annotation", **header, **_block_from_columns(df)}
    return msgspec.convert(raw, type=Annotation, dec_hook=dec_hook)
