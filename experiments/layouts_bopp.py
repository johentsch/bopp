"""Store canonical benchmark facets as individual or collected BOPP annotations.

Use a separate output directory for each facet/layout/format benchmark cell.
Returned paths enumerate every stored byte; no index or manifest is needed.
FORMATS lists the formats of each layout and FORMAT_NOTES states, per format,
any deviation from upstream's own writers and loaders.

Whole-file zstd containers have an eight-byte unsigned little-endian decoded
byte count followed by one zstd frame, and use an extra ``.zst`` suffix. The
codec is ``pyarrow.Codec('zstd')`` with its default compression level (1 in
PyArrow 26). Before decompressing, the reader requires the zstd frame magic
``28 b5 2f fd`` at offset 8 and a declared size of at most ``ZSTD_MAX_RATIO``
times the frame size plus ``ZSTD_SIZE_SLACK`` bytes; violations and
decompression failures raise BoppArgumentError (a ValueError) naming the file.
JSONL streams lines after whole-file decompression when compressed and skips
blank lines.

CSV uses upstream's Polars schema parsing, without whole-file compression.
Upstream's save_bopp_csv omits the annotation id. Format ``csv`` therefore
inserts the original id as the first TOML frontmatter line (keeping the file's
own line endings), so that validate=True includes load_bopp_csv's content-id
check like JSON and MessagePack. Format ``csv_upstream`` keeps upstream's bytes
exactly (same ``.csv`` file extension); having no id, its validate=True arm is
the typed decode only.
"""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import msgspec
import pandas as pd
import polars as pl
import pyarrow as pa
import tomllib

from bopp.core import ensure_annotation_id, validate_annotation_id
from bopp.exceptions import BoppArgumentError
from bopp.io import (
    _build_polars_csv_schema,
    _parse_complex_cell,
    _VersionHeader,
    load_bopp_csv,
    resolve_extensions,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.models.v1.annotation import Annotation
from bopp.registries import get_registry
from bopp.registries.v1 import EXTENT_TYPE_REGISTRY, PAYLOAD_TYPE_REGISTRY
from experiments import canonical

_EXTENT_KIND = "quarters_interval.fraction"
FORMATS: dict[str, tuple[str, ...]] = {
    "per_item": ("json", "msgpack", "csv", "csv_upstream"),
    "collection": ("json", "msgpack", "jsonl"),
}
"""Formats of each layout."""
FORMAT_NOTES: dict[str, str] = {
    "json": "",
    "msgpack": "",
    "jsonl": (
        "Not an upstream format: one msgspec JSON annotation per line, each line "
        "decoded and validated like bopp.io.load_bopp_json."
    ),
    "csv": (
        "Upstream save_bopp_csv bytes plus the annotation id inserted into the "
        "TOML frontmatter (46 bytes per file, 47 with CRLF), so validate=True "
        "includes load_bopp_csv's content-id check."
    ),
    "csv_upstream": (
        "Exactly upstream save_bopp_csv/load_bopp_csv; the file has no id, so "
        "validate=True is the typed decode only, without a content-id check."
    ),
}
"""One-sentence deviation from upstream's writers/loaders; empty for none."""
ZSTD_MAX_RATIO = 1024
"""Largest accepted ratio of declared decoded size to zstd frame size."""
ZSTD_SIZE_SLACK = 2**20
"""Declared decoded bytes accepted on top of the ratio bound."""
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_EXTENSIONS = {"csv_upstream": "csv"}
_CSV_FORMATS = {"csv", "csv_upstream"}
_IDENTITY_KEY = "layouts_bopp_identity"


def supported(
    layout: str, fmt: str, facet: str, *, compression: str = "none"
) -> bool | str:
    """
    Report whether a benchmark cell can preserve every valid canonical value.

    Parameters
    ----------
    layout, fmt : str
        Storage layout and serialization format.
    facet : str
        Key in canonical.FACETS.
    compression : {'none', 'zstd'}, default 'none'
        Whole-file compression; CSV only supports none.

    Returns
    -------
    bool or str
        True for supported cells, otherwise a nonempty reason. A reason string
        is truthy, so callers must compare the result with True explicitly.
    """
    if facet not in canonical.FACETS:
        return f"Unknown canonical facet: {facet!r}"
    reason = _format_reason(layout, fmt, compression)
    return True if reason is None else reason


def _format_reason(layout: str, fmt: str, compression: str) -> str | None:
    """Describe unsupported layouts and formats without requiring a facet."""
    if layout not in FORMATS:
        return f"Unknown BOPP layout: {layout!r}"
    if fmt not in FORMATS[layout]:
        return f"Format {fmt!r} is not supported by layout {layout!r}"
    if compression not in {"none", "zstd"}:
        return f"Unknown whole-file compression: {compression!r}"
    if fmt in _CSV_FORMATS and compression != "none":
        return "CSV does not support whole-file zstd compression"
    return None


def _component(value: str) -> str:
    """Require an unambiguous, safe corpus or piece path component."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(char in value for char in ("/", "\\", "\x00"))
    ):
        raise BoppArgumentError(f"Invalid corpus/piece component: {value!r}")
    return value


def media_id_to_identity(media_id: str) -> tuple[str, str]:
    """
    Recover a corpus and piece from a ``<dataset>:<corpus>/<piece>`` media id.

    Parameters
    ----------
    media_id : str
        Identifier in the form ``<dataset>:<corpus>/<piece>``, the dataset an
        alphanumeric prefix such as ``dlc``, ``bpsfh`` or ``vienna``.

    Returns
    -------
    tuple of str
        Corpus and piece, preserving punctuation and Unicode literally.

    Raises
    ------
    BoppArgumentError
        If the identifier is not an unambiguous dataset corpus/piece triple.
    """
    scheme, colon, rest = media_id.partition(":") if isinstance(media_id, str) else ("", "", "")
    if not colon or not scheme.isascii() or not scheme.isalnum():
        raise BoppArgumentError(f"Expected a <dataset>:<corpus>/<piece> media id, got {media_id!r}")
    parts = rest.split("/")
    if len(parts) != 2:
        raise BoppArgumentError(f"Expected <dataset>:<corpus>/<piece>, got {media_id!r}")
    return _component(parts[0]), _component(parts[1])


def _parsed_identity(media_id: str) -> tuple[str, str] | None:
    try:
        return media_id_to_identity(media_id)
    except BoppArgumentError:
        return None


def write(
    frame: pd.DataFrame,
    facet: str,
    out_dir: str | Path,
    *,
    layout: str,
    fmt: str,
    compression: str = "none",
    extent_kind: str = _EXTENT_KIND,
    media_id: Callable[[str, str], str] | None = None,
) -> list[Path]:
    """
    Write a canonical facet in lexical corpus/piece order.

    Parameters
    ----------
    frame : pandas.DataFrame
        Canonical facet table, with row order within each piece preserved.
    facet : str
        Key in canonical.FACETS, determining the payload kind.
    out_dir : str or pathlib.Path
        Dedicated output directory for this benchmark cell.
    layout : {'per_item', 'collection'}
        One annotation per file or one collection file per facet.
    fmt : {'json', 'msgpack', 'jsonl', 'csv', 'csv_upstream'}
        Format accepted by supported; JSONL is collection-only, the CSV
        formats are per-item only and both use the ``.csv`` extension.
    compression : {'none', 'zstd'}, default 'none'
        Whole-file compression; CSV only supports none.
    extent_kind : str, optional
        Registered extent kind; defaults to quarters_interval.fraction.
    media_id : callable or None, optional
        Function of (corpus, piece) returning a media id. Defaults to
        dlc:<corpus>/<piece>. Identifiers that do not parse back to the same
        corpus/piece with media_id_to_identity store the identity in annotation
        sandbox metadata, so arbitrary identifiers can be read without a callback.

    Returns
    -------
    list of pathlib.Path
        All written files. An empty per-item facet creates no files; an empty
        collection still creates one file.

    Raises
    ------
    BoppArgumentError
        If the cell is unsupported, identity is invalid, or an empty frame uses
        a nondefault extent (there is no annotation to carry that schema).
    """
    status = supported(layout, fmt, facet, compression=compression)
    if status is not True:
        raise BoppArgumentError(status)
    if frame.empty and extent_kind != _EXTENT_KIND:
        raise BoppArgumentError("An empty facet requires the default extent kind")
    for name in ("corpus", "piece"):
        for value in frame[name].unique():
            _component(value)
    annotations = []
    identities = []
    for (corpus, piece), group in frame.groupby(["corpus", "piece"], sort=True):
        ann = canonical.frame_to_annotation(
            group,
            payload_kind=canonical.FACETS[facet]["payload_kind"],
            extent_kind=extent_kind,
            media_id=f"dlc:{corpus}/{piece}"
            if media_id is None
            else media_id(corpus, piece),
        )
        if media_id is not None and _parsed_identity(ann.media_id) != (corpus, piece):
            ann.sandbox = {_IDENTITY_KEY: {"corpus": corpus, "piece": piece}}
            ann.id = msgspec.UNSET
            ensure_annotation_id(ann)
        annotations.append(ann)
        identities.append((corpus, piece))
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = _suffix(fmt, compression)
    if layout == "per_item":
        save = {
            "json": save_bopp_json,
            "msgpack": save_bopp_msgpack,
            "csv": save_bopp_csv,
            "csv_upstream": save_bopp_csv,
        }[fmt]
        paths = []
        for (corpus, piece), ann in zip(identities, annotations, strict=True):
            path = root / corpus / f"{piece}.{facet}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            save(ann, path)
            if fmt == "csv":
                # Preserve the original id omitted by upstream to_dataframe.
                path.write_bytes(_inject_csv_id(path.read_bytes(), ann.id, path))
            elif compression == "zstd":
                path.write_bytes(_compress(path.read_bytes()))
            paths.append(path)
        return paths
    path = root / f"{facet}{suffix}"
    for ann in annotations:
        ensure_annotation_id(ann)
    if fmt == "jsonl":
        with path.open("wb") as stream:
            for ann in annotations:
                stream.write(msgspec.json.encode(ann) + b"\n")
    else:
        codec = msgspec.json if fmt == "json" else msgspec.msgpack
        path.write_bytes(codec.encode(annotations))
    if compression == "zstd":
        path.write_bytes(_compress(path.read_bytes()))
    return [path]


def _suffix(fmt: str, compression: str) -> str:
    """Return the file suffix of a format, including any ``.zst``."""
    zst = ".zst" if compression == "zstd" else ""
    return f".{_EXTENSIONS.get(fmt, fmt)}{zst}"


def _inject_csv_id(data: bytes, ann_id: str, path: Path) -> bytes:
    """
    Insert an id line into upstream CSV frontmatter, keeping its line endings.

    Parameters
    ----------
    data : bytes
        File content written by bopp.io.save_bopp_csv.
    ann_id : str
        Annotation id to store.
    path : pathlib.Path
        File the data came from, for error messages.

    Returns
    -------
    bytes
        The content with ``# id = "<id>"`` as the first frontmatter line, or
        unchanged if the frontmatter already has an id.

    Raises
    ------
    BoppArgumentError
        If the content does not start with a closed ``# ---`` frontmatter.
    """
    lines = data.splitlines(keepends=True)
    if not lines or lines[0].strip() != b"# ---":
        raise BoppArgumentError(f"{path}: CSV frontmatter delimiter '# ---' not found")
    for close, line in enumerate(lines[1:], start=1):
        if line.strip() == b"# ---":
            break
    else:
        raise BoppArgumentError(f"{path}: CSV frontmatter has no closing '# ---'")
    metadata = tomllib.loads(
        "".join(line.decode().lstrip("#").lstrip(" ") for line in lines[1:close])
    )
    if "id" in metadata:
        return data
    newline = lines[0][len(lines[0].rstrip(b"\r\n")) :] or b"\n"
    id_line = f'# id = "{ann_id}"'.encode() + newline
    return b"".join([lines[0], id_line, *lines[1:]])


def _paths(
    path_or_dir: str | Path, layout: str, fmt: str, compression: str
) -> list[Path]:
    """Resolve a dedicated directory or a single file deterministically."""
    path = Path(path_or_dir)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    suffix = _suffix(fmt, compression)
    pattern = f"*/*{suffix}" if layout == "per_item" else f"*{suffix}"

    def item_key(item: Path) -> tuple[str, str]:
        stem = item.name.removesuffix(suffix)
        for facet in sorted(canonical.FACETS, key=len, reverse=True):
            if stem.endswith(f".{facet}"):
                return item.parent.name, stem.removesuffix(f".{facet}")
        return item.parent.name, stem

    paths = sorted(path.glob(pattern), key=item_key)
    if layout == "collection" and len(paths) != 1:
        raise BoppArgumentError("A collection directory must contain exactly one file")
    return paths


def _compress(data: bytes) -> bytes:
    """Prefix a whole-file zstd frame with its decoded byte count."""
    return len(data).to_bytes(8, "little") + pa.Codec("zstd").compress(
        data, asbytes=True
    )


def _read_bytes(path: Path, compression: str) -> bytes:
    """Read and optionally decompress one file, validating the container."""
    data = path.read_bytes()
    if compression == "none":
        return data
    if len(data) < 8 + len(_ZSTD_MAGIC):
        raise BoppArgumentError(f"{path}: truncated zstd size header")
    if data[8:12] != _ZSTD_MAGIC:
        raise BoppArgumentError(
            f"{path}: no zstd frame after the size header (not a zstd container)"
        )
    size = int.from_bytes(data[:8], "little")
    limit = ZSTD_MAX_RATIO * (len(data) - 8) + ZSTD_SIZE_SLACK
    if size > limit:
        raise BoppArgumentError(
            f"{path}: corrupt zstd size header, declared {size} bytes exceeds "
            f"the bound of {limit}"
        )
    try:
        decoded = pa.Codec("zstd").decompress(
            data[8:], decompressed_size=size, asbytes=True
        )
    except OSError as err:
        raise BoppArgumentError(f"{path}: zstd decompression failed: {err}") from err
    if len(decoded) != size:
        raise BoppArgumentError(f"{path}: incorrect zstd decoded size")
    return decoded


def _decode_bytes(data: bytes, fmt: str, validate: bool, *, collection: bool = False):
    """Share upstream's typed decode and id/extension ordering across containers."""
    codec = msgspec.msgpack if fmt == "msgpack" else msgspec.json
    if not validate:
        return codec.decode(data)
    if collection:
        # One typed call for the entire list; DLC annotations use the v1 schema.
        decoded = codec.decode(data, type=list[Annotation])
        annotations = decoded
    else:
        # Mirror bopp.io._load_msgspec_encoded, including version dispatch.
        header = codec.decode(data, type=_VersionHeader)
        annotation_cls = get_registry(header.bopp_version)["Annotation"]
        decoded = codec.decode(data, type=annotation_cls)
        annotations = [decoded]
    for ann in annotations:
        if collection:
            get_registry(ann.bopp_version)
        validate_annotation_id(ann)
        resolve_extensions(ann)
    return decoded


def _decode_csv(path: Path) -> dict[str, Any]:
    """Parse upstream CSV to builtins without constructing any Annotation."""
    with path.open("r", encoding="utf-8") as stream:
        lines = []
        if stream.readline().strip() == "# ---":
            while line := stream.readline():
                if line.strip() == "# ---":
                    break
                lines.append(line.lstrip("#").lstrip(" "))
        else:
            stream.seek(0)
        metadata = tomllib.loads("".join(lines))
        position = stream.tell()
        header = stream.readline()
        stream.seek(position)
        version = metadata.get("bopp_version", _VersionHeader().bopp_version)
        schema, complex_columns = _build_polars_csv_schema(header, version)
        columns = pl.read_csv(stream, schema=schema or None).to_dict(as_series=False)
    document = {"bopp_version": version, **metadata}
    for column, values in columns.items():
        facet, tag, field = column.split(":")
        target = document.setdefault(facet, {})
        target[f"{facet}_type"] = tag
        target[field] = (
            [_parse_complex_cell(value) for value in values]
            if column in complex_columns
            else values
        )
    return document


def decode_only(
    path_or_dir: str | Path,
    *,
    layout: str,
    fmt: str,
    validate: bool,
    compression: str = "none",
) -> object:
    """
    Read and decode stored annotations without building any canonical frames.

    Parameters
    ----------
    path_or_dir : str or pathlib.Path
        A single file or dedicated benchmark cell directory.
    layout, fmt : str
        Storage layout and format, as used by write.
    validate : bool
        True uses typed Annotation decoding, extension resolution and existing
        id validation, matching bopp.io. Missing ids remain permitted. False
        decodes only to builtins, without constructing Annotation objects.
        CSV parsing uses upstream's schema-directed Polars dtypes in both arms;
        csv_upstream files have no id, so their True arm checks types only.
    compression : {'none', 'zstd'}, default 'none'
        Whole-file compression. Decompression is included in this operation.

    Returns
    -------
    object
        List of annotations, or list of builtin dictionaries. JSONL is read
        line by line and fully decoded here so its work is included in timing.

    Raises
    ------
    BoppArgumentError
        If the layout/format is unsupported or the collection is ambiguous.
    """
    reason = _format_reason(layout, fmt, compression)
    if reason is not None:
        raise BoppArgumentError(reason)
    paths = _paths(path_or_dir, layout, fmt, compression)
    if fmt in _CSV_FORMATS:
        return [
            load_bopp_csv(path) if validate else _decode_csv(path) for path in paths
        ]
    if layout == "per_item":
        return [
            _decode_bytes(_read_bytes(path, compression), fmt, validate)
            for path in paths
        ]
    path = paths[0]
    if fmt == "jsonl":
        stream = (
            path.open("rb")
            if compression == "none"
            else BytesIO(_read_bytes(path, compression))
        )
        with stream:
            return [
                _decode_bytes(line, "json", validate) for line in stream if line.strip()
            ]
    return _decode_bytes(_read_bytes(path, compression), fmt, validate, collection=True)


def _identity(ann: Any) -> tuple[str, str]:
    """Read default media identity or explicitly stored override identity."""
    sandbox = ann.get("sandbox", {}) if isinstance(ann, dict) else ann.sandbox
    if isinstance(sandbox, dict) and _IDENTITY_KEY in sandbox:
        identity = sandbox[_IDENTITY_KEY]
        return _component(identity["corpus"]), _component(identity["piece"])
    media_id = ann["media_id"] if isinstance(ann, dict) else ann.media_id
    return media_id_to_identity(media_id)


def _builtin_frame(ann: dict[str, Any], corpus: str, piece: str) -> pd.DataFrame:
    """Convert wire dictionaries using canonical schema rules, without structs."""
    extent, payload = ann["extent"], ann["payload"]
    extent_struct = EXTENT_TYPE_REGISTRY[extent["extent_type"]]
    payload_struct = PAYLOAD_TYPE_REGISTRY[payload["payload_type"]]
    count = len(extent[canonical._columns(extent_struct)[0][0]])
    columns: dict[str, Any] = {"corpus": [corpus] * count, "piece": [piece] * count}
    for data, struct in ((extent, extent_struct), (payload, payload_struct)):
        for name, _, convert in canonical._columns(struct):
            values = data.get(name)
            columns[name] = (
                [None] * count
                if values is None
                else [None if v is None else convert(v) for v in values]
            )
    return pd.DataFrame(columns, dtype=object)


def load(
    path_or_dir: str | Path,
    facet: str,
    *,
    layout: str,
    fmt: str,
    validate: bool = True,
    compression: str = "none",
) -> pd.DataFrame:
    """
    Decode a stored facet and reproduce its canonical table.

    Parameters
    ----------
    path_or_dir : str or pathlib.Path
        A file or dedicated benchmark cell directory.
    facet : str
        Key in canonical.FACETS.
    compression : {'none', 'zstd'}, default 'none'
        Whole-file compression; CSV only supports none.
    layout, fmt : str
        Storage layout and format, as used by write.
    validate : bool, default True
        Whether decode_only performs typed decoding and id validation.

    Returns
    -------
    pandas.DataFrame
        Canonical table in lexical corpus/piece order, preserving piece rows.

    Raises
    ------
    BoppArgumentError
        If the cell is unsupported or contains a different payload kind.
    """
    status = supported(layout, fmt, facet, compression=compression)
    if status is not True:
        raise BoppArgumentError(status)
    annotations = decode_only(
        path_or_dir, layout=layout, fmt=fmt, validate=validate, compression=compression
    )
    frames = []
    for ann in sorted(annotations, key=_identity):
        corpus, piece = _identity(ann)
        kind = (
            ann.payload.__struct_config__.tag
            if validate
            else ann["payload"]["payload_type"]
        )
        if kind != canonical.FACETS[facet]["payload_kind"]:
            raise BoppArgumentError(
                f"Unexpected payload kind for facet {facet!r}: {kind!r}"
            )
        frames.append(
            canonical.annotation_to_frame(ann, corpus=corpus, piece=piece)
            if validate
            else _builtin_frame(ann, corpus, piece)
        )
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(
        columns=canonical._frame_columns(
            canonical.FACETS[facet]["payload_kind"], _EXTENT_KIND
        ),
        dtype=object,
    )
