#!/usr/bin/env python

import ast
import math
from fractions import Fraction
import msgspec
import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

from .util import extract_header, from_dataframe, to_dataframe
from pathlib import Path

from bopp.models.v1.annotation import Annotation
from bopp.models.v1.object import AnyObject
from bopp.models.v1.score_object import ScoreObject

from typing import Any, Annotated, get_args, get_origin


_TYPE_MAP = {
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
    "bool": pa.bool_(),
    "string": pa.string(),
}


def decode_arrow(type_hint, value):
    """
    Intercepts the msgspec parser to build native Arrow arrays
    instead of standard Python lists, respecting precision type hints.
    """
    target = type_hint
    arrow_type = None

    # Inspect metadata attached via Annotated types (handles typing and typing_extensions)
    while get_origin(target) is Annotated or hasattr(target, "__metadata__"):
        metadata_list = getattr(target, "__metadata__", None) or get_args(target)[1:]
        for metadata in metadata_list:
            if isinstance(metadata, str) and metadata in _TYPE_MAP:
                arrow_type = _TYPE_MAP[metadata]
            elif isinstance(metadata, pa.DataType):
                arrow_type = metadata

        if get_origin(target) is Annotated:
            target = get_args(target)[0]
        elif hasattr(target, "__origin__") and getattr(target, "__origin__") is not target:
            target = getattr(target, "__origin__")
        else:
            break

    if target is pa.Array or target == pa.Array:
        # Convert the raw parsed list into a contiguous Arrow buffer using the precision hint if present
        return pa.array(value, type=arrow_type)

    raise TypeError(f"Type {type_hint} is not supported")


def encode_arrow(obj: Any) -> Any:
    """
    Intercepts PyArrow arrays during serialization and converts
    them back to standard Python lists for Msgpack/JSON.
    """
    # Catch both contiguous Arrays and ChunkedArrays natively
    if isinstance(obj, (pa.Array, pa.ChunkedArray)):
        return obj.to_pylist()

    # msgspec requires you to raise a NotImplementedError if the hook
    # receives an object it doesn't know how to handle.
    raise NotImplementedError(f"Object of type {type(obj)} is not supported")


def _csv_cell(value: Any) -> Any:
    """Canonical CSV spelling of one cell: lists as Python literals, fractions as n/d."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Fraction):
        return str(value)
    return value


def to_csv(obj: Annotation | ScoreObject, filepath: str | Path) -> None:
    """Writes metadata as YAML frontmatter, followed by the DataFrame."""

    df = to_dataframe(obj)
    metadata = extract_header(obj)

    for col in df.columns:
        if df[col].dtype == object:
            # Series.map would re-infer int + None columns as float; keep Python values.
            df[col] = pd.Series([_csv_cell(v) for v in df[col]], dtype=object, index=df.index)

    # 1. Convert the metadata dictionary to a YAML string
    yaml_text = yaml.dump(metadata, sort_keys=False, default_flow_style=False)

    # 2. Prefix every line with a comment hash
    frontmatter = ["# ---\n"]
    for line in yaml_text.splitlines():
        frontmatter.append(f"# {line}\n")
    frontmatter.append("# ---\n")

    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        # 3. Write the header
        f.writelines(frontmatter)

        # 4. Hand the open file pointer to Pandas!
        df.to_csv(f, index=False, lineterminator="\n")


def load_bopp_json(filepath: str, type: Any = AnyObject) -> Annotation | ScoreObject:
    """Reads a BOPP JSON file directly into the Python model."""
    # msgspec operates fastest on raw bytes, so we read as "rb"
    with open(filepath, "rb") as f:
        data = f.read()

    return msgspec.json.decode(data, type=type, dec_hook=decode_arrow)


def save_bopp_json(obj: Annotation | ScoreObject, filepath: str) -> None:
    """Serializes an object directly into a JSON file."""
    # msgspec encodes structs natively without needing conversion dicts
    json_data = msgspec.json.encode(obj, enc_hook=encode_arrow)

    with open(filepath, "wb") as f:
        f.write(json_data)
    print(f"Successfully saved to {filepath} ({len(json_data)} bytes)")


# ==========================================
# 1. Saving (Encoding) to Msgpack
# ==========================================
def save_bopp_msgpack(obj: Annotation | ScoreObject, filepath: str) -> None:
    """
    Serializes an object directly into a binary msgpack file.
    """
    # msgspec encodes structs natively without needing conversion dicts
    binary_data = msgspec.msgpack.encode(obj, enc_hook=encode_arrow)

    with open(filepath, "wb") as f:
        f.write(binary_data)
    print(f"Successfully saved to {filepath} ({len(binary_data)} bytes)")


# ==========================================
# 2. Loading (Decoding) from Msgpack
# ==========================================
def load_bopp_msgpack(filepath: str, type: Any = AnyObject) -> Annotation | ScoreObject:
    """
    Reads a binary msgpack file and decodes/validates it into the tagged
    object type it declares (running the length checks via __post_init__).
    Pass type=Annotation for legacy files without an object_type tag.
    """
    with open(filepath, "rb") as f:
        binary_data = f.read()

    # Decode and instantly validate against the schema
    return msgspec.msgpack.decode(binary_data, type=type, dec_hook=decode_arrow)


def read_bopp_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Reads a BOPP CSV file, extracts the YAML frontmatter into df.attrs,
    and returns the tabular data as a Pandas DataFrame.
    """
    yaml_lines = []

    with open(filepath, 'r', encoding='utf-8') as f:
        # 1. Parse YAML Frontmatter
        first_line = f.readline().strip()

        if first_line == "# ---":
            while True:
                line = f.readline()
                if not line or line.strip() == "# ---":
                    break
                # Strip the comment hash and leading space
                yaml_lines.append(line.lstrip('#').lstrip(' '))

            metadata = yaml.safe_load("".join(yaml_lines))
        else:
            # No frontmatter found, reset the file pointer
            f.seek(0)
            metadata = {}

        # 2. Hand the open file pointer directly to Pandas
        df = pd.read_csv(f)

    # 3. Restore nulls: to_csv writes missing values as empty fields, which
    # Pandas reads back as NaN. Map them back to None so Arrow columns regain
    # proper nulls (this mirrors the write path, which maps every NaN to None
    # before constructing the Annotation). The astype(object) detour is needed
    # because float blocks cannot hold None.
    df = df.astype(object).where(df.notna(), None)

    # 4. Handle Polyphonic / List Data safely
    # If a payload contains lists (e.g., ["C", "E", "G"]), the CSV writer saves them
    # as literal strings. We evaluate them back to actual Python lists here.
    def _parse_cell(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str):
            return ast.literal_eval(v)
        return v

    payload_cols = [c for c in df.columns if c.startswith("payload:")]
    for col in payload_cols:
        if df[col].dtype == object and df[col].astype(str).str.startswith('[').any():
            df[col] = df[col].apply(_parse_cell)

    # Attach the singleton fields directly to the DataFrame attributes
    df.attrs = metadata
    return df


def load_bopp_csv(filepath: str | Path) -> Annotation | ScoreObject:
    """
    End-to-end wrapper: Reads a BOPP CSV file directly into a validated object.
    """
    df = read_bopp_csv(filepath)
    return from_dataframe(df, dec_hook=decode_arrow)
