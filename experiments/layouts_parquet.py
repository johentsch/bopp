"""Store canonical DLC facets in three Parquet benchmark layouts.

Partitioned output uses one ParquetWriter per corpus in a Hive directory,
retaining corpus in each file as a plain string. Single and partitioned files
have one row group per piece, in canonical lexical order. PyArrow's dictionary
encoding default (use_dictionary=True, with fallback to plain encoding) and
statistics default (write_statistics=True) are unchanged.

Each layout and kind of call gets the reader that measured fastest on the
notes of three DLC corpora (207 pieces, 188k rows; best of 7, interleaved, on
a loaded machine). Per-item paths and corpus directories are pruned by name
before any file is opened. Two or more per-item files are read by one
``pyarrow.parquet.read_table`` over the path list (``partitioning=None``):
its multithreaded dataset scan beats a Python loop over files by about 2x
on full reads and 2.5x on projections, since per-file overhead dominates
small files (from five files on; two files are a tie). A lone per-item file,
the one-piece random access, is read by ``ParquetFile.read`` (about 2 ms
against 3.5 ms for ``read_table``). Single and partitioned files are opened
once with ``ParquetFile``, which beats ``read_table`` there by 2-4x: full
reads and projections use ``read`` (slightly faster than ``read_row_groups``
over every group); piece selection picks the row groups whose corpus/piece
statistics equal a requested pair (min == max and no nulls, since the writer
stores one piece per row group) and reads only those. PyArrow's thread defaults
(``use_threads=True``) stay on every path. No Arrow schema is imposed at read
time, so stored types come back as stored and validation stays a separate
step (the multi-file per-item read takes its schema from the first file, as
any dataset scan does; write gives every file the same schema, so nothing is
cast, and a later file's nulls relax the nullability); the corpus column of the partitioned layout is read from the file, a
plain string, never a Hive dictionary. The I/O entry point performs neither
validation nor pandas conversion.

``frame_to_table`` builds every column with ``arrow_schema.column_to_arrow``,
which rejects booleans that Arrow inference would coerce to numbers and
reports every conversion error as ``ArrowValidationError``.

Several facets may share one output directory, and corpus and piece names may
contain dots (``a.b``, even ``x.notes``). Every file name ends in the facet key
(``<piece>.<facet>.parquet`` per item, ``<facet>.parquet`` otherwise), and
readers select files by that exact suffix, never by directory scans, so a
facet never reads a sibling facet's files or schema. Per-item piece names are
recovered by removing the suffix, which keeps the lexical corpus/piece order
of the writer. Corpus and piece must be nonempty single path components.
Different layouts of one facet must not share a directory.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bopp.registries.v1 import EXTENT_TYPE_REGISTRY, PAYLOAD_TYPE_REGISTRY
from experiments import arrow_schema as validation
from experiments.canonical import FACETS

_EXTENT_KIND = "quarters_interval.fraction"
_LAYOUTS = {"per_item", "single", "partitioned"}


def _payload(facet: str) -> str:
    """Resolve a DLC facet, reporting unknown names consistently."""
    if facet not in FACETS:
        raise ValueError(f"Unknown DLC facet: {facet!r}")
    return FACETS[facet]["payload_kind"]


def _wire(value: Any) -> Any:
    """Convert a canonical nested value without numeric or null coercion."""
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, tuple):
        return list(value)
    return value


def _schema(facet: str, extent: str = _EXTENT_KIND) -> pa.Schema:
    """Order JSON-derived fields like the canonical generated structs.

    Some control-event JSON documents interleave required and optional fields,
    whereas generated structs place required fields first.
    """
    payload = _payload(facet)
    schema = validation.arrow_schema(payload, extent)
    names = [
        "corpus",
        "piece",
        *EXTENT_TYPE_REGISTRY[extent].__struct_fields__,
        *PAYLOAD_TYPE_REGISTRY[payload].__struct_fields__,
    ]
    return pa.schema([schema.field(name) for name in names])


def frame_to_table(
    frame: pd.DataFrame, facet: str, *, extent_kind: str | None = None
) -> pa.Table:
    """
    Convert a canonical frame to its validated, schema-derived Arrow table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Object columns with Python scalars, Fraction, integer tuples and None.
        Every schema column, including wholly null optional columns, is required.
    facet : str
        Key in canonical.FACETS.
    extent_kind : str or None, optional
        Extent schema tag; None defaults to quarters_interval.fraction.

    Returns
    -------
    pyarrow.Table
        Typed columns in schema order, with Fraction encoded as integer pairs.

    Raises
    ------
    ValueError
        If columns are missing, duplicated or unexpected, or the facet is unknown.
    ArrowValidationError
        If a value violates the JSON schema constraints.
    """
    payload = _payload(facet)
    extent = _EXTENT_KIND if extent_kind is None else extent_kind
    schema = _schema(facet, extent)
    if not frame.columns.is_unique or set(frame.columns) != set(schema.names):
        raise ValueError("Expected every canonical schema column exactly once")
    # Infer nonempty columns before validation: pre-casting would silently turn
    # some invalid floats or booleans into integers. Explicitly type null columns.
    arrays = []
    for field in schema:
        values = [_wire(value) for value in frame[field.name]]
        arrays.append(
            pa.nulls(len(values), field.type)
            if all(value is None for value in values)
            else validation.column_to_arrow(field.name, values, payload, extent)
        )
    return validation.validate_table(
        pa.Table.from_arrays(arrays, names=schema.names), payload, extent
    ).select(schema.names)


def _number_union(field: pa.Field) -> bool:
    """Whether a float64 field stores a BOPP integer/number item union."""
    items = (field.metadata or {}).get(b"bopp:items")
    if not pa.types.is_floating(field.type) or items is None:
        return False
    kinds = json.loads(items).get("type", [])
    return isinstance(kinds, list) and {"integer", "number"} <= set(kinds)


def table_to_frame(table: pa.Table) -> pd.DataFrame:
    """
    Restore canonical Python cells without pandas numeric inference.

    Parameters
    ----------
    table : pyarrow.Table
        Full or projected table with schema-derived Arrow column types.

    Returns
    -------
    pandas.DataFrame
        Object columns in input order. Integer pairs become Fraction, integer
        lists become tuples, nulls become None, and scalars retain Python types.
        Integer/number union columns (float64 with ``bopp:items`` metadata)
        restore integral floats as ints, see
        ``arrow_schema.restore_number_union``.
    """
    columns = {}
    for field, column in zip(table.schema, table.columns, strict=True):
        values = column.to_pylist()
        if pa.types.is_fixed_size_list(field.type) and field.type.list_size == 2:
            values = [None if v is None else Fraction(*v) for v in values]
        elif pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            values = [None if v is None else tuple(v) for v in values]
        elif _number_union(field):
            values = validation.restore_number_union(values)
        columns[field.name] = pd.Series(values, dtype=object)
    return pd.DataFrame(columns, dtype=object)


def _paths(out: Path, facet: str, layout: str) -> list[Path]:
    """Find only this facet's files, in canonical corpus/piece order."""
    _payload(facet)
    if layout not in _LAYOUTS:
        raise ValueError(f"Unknown Parquet layout: {layout!r}")
    if layout == "single":
        path = out / f"{facet}.parquet"
        return [path] if path.is_file() else []
    if layout == "partitioned":
        return sorted(
            out.glob(f"corpus=*/{facet}.parquet"), key=lambda path: path.parent.name
        )
    return sorted(
        out.glob(f"*/*.{facet}.parquet"),
        key=lambda path: (
            path.parent.name,
            path.name.removesuffix(f".{facet}.parquet"),
        ),
    )


def write(
    frame: pd.DataFrame,
    facet: str,
    out: str | Path,
    *,
    layout: str,
    compression: str = "zstd",
    extent_kind: str | None = None,
) -> list[Path]:
    """
    Write a canonical facet with one row group per nonempty piece.

    Parameters
    ----------
    frame : pandas.DataFrame
        Canonical DLC facet; rows within each piece retain their input order.
    facet : str
        Key in canonical.FACETS.
    out : str or pathlib.Path
        Output directory, without existing files for this facet and layout.
    layout : {'per_item', 'single', 'partitioned'}
        Per-item files, one combined file, or one file per corpus in Hive paths.
    compression : {'zstd', 'none'}, optional
        Parquet compression; dictionary encoding remains at PyArrow defaults.
    extent_kind : str or None, optional
        Extent schema tag; None defaults to quarters_interval.fraction.

    Returns
    -------
    list of pathlib.Path
        Written Parquet files in lexical order. Empty per-item and partitioned
        facets create the output directory but no files; single creates an empty
        file with the full schema.

    Raises
    ------
    ValueError
        If the facet, layout, compression or canonical frame is invalid.
    FileExistsError
        If facet files already exist, preventing stale pieces after a rewrite.
    """
    out = Path(out)
    existing = _paths(out, facet, layout)
    if compression not in {"zstd", "none"}:
        raise ValueError(f"Unknown Parquet compression: {compression!r}")
    if existing:
        raise FileExistsError(f"Facet output already exists: {existing[0]}")
    table = frame_to_table(frame, facet, extent_kind=extent_kind)
    groups = frame.groupby(["corpus", "piece"], sort=True).indices
    for corpus, piece in groups:
        for value in (corpus, piece):
            if not value or value in {".", ".."} or "/" in value or "\\" in value:
                raise ValueError("Corpus and piece must be nonempty path components")
    out.mkdir(parents=True, exist_ok=True)
    codec = None if compression == "none" else compression
    paths = []
    writer = None
    current_path = None
    try:
        if layout == "single":
            current_path = out / f"{facet}.parquet"
            writer = pq.ParquetWriter(current_path, table.schema, compression=codec)
            paths.append(current_path)
        for (corpus, piece), indices in groups.items():
            if layout == "per_item":
                path = out / corpus / f"{piece}.{facet}.parquet"
            elif layout == "partitioned":
                path = out / f"corpus={corpus}" / f"{facet}.parquet"
            else:
                path = current_path
            if path != current_path:
                if writer is not None:
                    writer.close()
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(path, table.schema, compression=codec)
                current_path = path
                paths.append(path)
            piece_table = table.take(pa.array(indices, type=pa.int64()))
            writer.write_table(piece_table, row_group_size=len(indices))
    finally:
        if writer is not None:
            writer.close()
    return paths


def _projection(columns: Sequence[str] | None) -> list[str] | None:
    """Always prepend identities and retain requested projection order."""
    if columns is None:
        return None
    return list(dict.fromkeys(["corpus", "piece", *columns]))


def read_table(
    out: str | Path,
    facet: str,
    *,
    layout: str,
    columns: Sequence[str] | None = None,
    pieces: Sequence[tuple[str, str]] | None = None,
    extent_kind: str | None = None,
) -> pa.Table:
    """
    Read Parquet data with projection and piece pruning, without validation.

    Parameters
    ----------
    out : str or pathlib.Path
        Directory produced by write.
    facet : str
        Key in canonical.FACETS.
    layout : {'per_item', 'single', 'partitioned'}
        Stored layout.
    columns : sequence of str or None, optional
        Requested columns in order; corpus and piece are always prepended.
        None reads every stored column.
    pieces : sequence of (str, str) or None, optional
        Corpus/piece pairs, combined as an OR of exact pair filters. None reads
        all pieces; an empty sequence reads no rows. Selection retains storage
        order, independent of request order and duplicate pairs.
    extent_kind : str or None, optional
        Extent schema tag of the empty result when no file exists; None
        defaults to quarters_interval.fraction.

    Returns
    -------
    pyarrow.Table
        Physical Arrow data, including a plain string corpus column. Single
        and partitioned files, and a lone per-item file, are opened once each;
        two or more per-item files are read by one ``pyarrow.parquet.read_table``
        call over the pruned path list.

    Raises
    ------
    FileNotFoundError
        If the output directory or a required single-layout file is missing.
    ValueError
        If the layout, facet or projection is unknown.
    """
    out = Path(out)
    paths = _paths(out, facet, layout)
    if not out.is_dir() or (layout == "single" and not paths):
        raise FileNotFoundError(out / f"{facet}.parquet")
    projected = _projection(columns)
    selected = None if pieces is None else set(pieces)
    all_paths = paths
    if selected is not None:
        if layout == "per_item":
            wanted = {out / c / f"{p}.{facet}.parquet" for c, p in selected}
            paths = [path for path in paths if path in wanted]
        elif layout == "partitioned":
            corpora = {f"corpus={c}" for c, _ in selected}
            paths = [path for path in paths if path.parent.name in corpora]
    if selected == set():
        paths = []
    if layout == "per_item" and len(paths) > 1:
        return _read_files(paths, projected)
    tables = []
    schema = None
    for path in paths:
        file = pq.ParquetFile(path)
        schema = file.schema_arrow
        names = _check_projection(schema, projected)
        if layout == "per_item" or selected is None:
            tables.append(file.read(columns=names))
            continue
        groups, ambiguous = _row_groups(file.metadata, selected)
        if not groups:
            continue
        tables.append(
            _filter_pieces(file, groups, names, selected)
            if ambiguous
            else file.read_row_groups(groups, columns=names)
        )
    if tables:
        # Unify field nullability (e.g. a file rewritten with relaxed fields)
        # without casting values: differing stored types still raise.
        return pa.concat_tables(tables, promote_options="default")
    if schema is None:
        schema = (
            pq.read_schema(all_paths[0])
            if all_paths
            else _schema(facet, _EXTENT_KIND if extent_kind is None else extent_kind)
        )
    names = _check_projection(schema, projected)
    if names is not None:
        schema = pa.schema([schema.field(name) for name in names])
    return schema.empty_table()


def _read_files(paths: list[Path], projected: list[str] | None) -> pa.Table:
    """
    Read several per-item files with one multithreaded dataset read.

    Parameters
    ----------
    paths : list of pathlib.Path
        At least two files, already pruned by piece, in storage order.
    projected : list of str or None
        Projection including corpus/piece, or None for every column.

    Returns
    -------
    pyarrow.Table
        Rows in path order. The dataset takes its schema from the first file
        and leaves values uncast when every file stores the same types, as
        write guarantees. A field the first file declares non-nullable is
        relaxed when a later file holds nulls in it, as concatenation would.

    Raises
    ------
    ValueError
        If a projected column is not stored in the first file.
    """
    try:
        table = pq.read_table(
            [str(path) for path in paths], columns=projected, partitioning=None
        )
    except pa.ArrowInvalid:
        # A missing projected column surfaces as a FieldRef error; report it
        # like every other layout. The schema is read only on this error path.
        _check_projection(pq.read_schema(paths[0]), projected)
        raise
    fields = [
        field.with_nullable(True) if not field.nullable and column.null_count else field
        for field, column in zip(table.schema, table.columns, strict=True)
    ]
    if fields != list(table.schema):
        table = table.cast(pa.schema(fields, metadata=table.schema.metadata))
    return table


def _check_projection(
    schema: pa.Schema, projected: list[str] | None
) -> list[str] | None:
    """Reject projected names that the stored schema lacks."""
    if projected is not None and (missing := set(projected) - set(schema.names)):
        raise ValueError(f"Unknown projected columns: {sorted(missing)}")
    return projected


def _row_groups(
    metadata: pq.FileMetaData, selected: set[tuple[str, str]]
) -> tuple[list[int], bool]:
    """
    Choose row groups by their corpus/piece statistics, from the read footer.

    Parameters
    ----------
    metadata : pyarrow.parquet.FileMetaData
        Footer of an opened file.
    selected : set of (str, str)
        Requested corpus/piece pairs.

    Returns
    -------
    groups : list of int
        Row groups, in storage order, holding exactly one requested pair or
        lacking usable statistics. Statistics are usable when corpus and piece
        each have min == max and a null count of zero; a group with nulls in
        either, or without a null count, holds rows outside its min/max.
    ambiguous : bool
        Whether a group without usable statistics was included, so the rows
        read still need an exact pair filter (which drops null identities).
    """
    leaves = {metadata.schema.column(j).path: j for j in range(metadata.num_columns)}
    groups = []
    ambiguous = False
    for i in range(metadata.num_row_groups):
        group = metadata.row_group(i)
        stats = [group.column(leaves[name]).statistics for name in ("corpus", "piece")]
        if all(
            s is not None
            and s.has_min_max
            and s.min == s.max
            and s.has_null_count
            and s.null_count == 0
            for s in stats
        ):
            if (stats[0].min, stats[1].min) in selected:
                groups.append(i)
        elif group.num_rows:
            groups.append(i)
            ambiguous = True
    return groups, ambiguous


def _filter_pieces(
    file: pq.ParquetFile,
    groups: list[int],
    names: list[str] | None,
    selected: set[tuple[str, str]],
) -> pa.Table:
    """Read groups (projections always hold corpus/piece) and keep exact pairs.

    Only reached for row groups without single-pair statistics, which write
    never produces; the separator NUL cannot occur in a path component.
    """
    table = file.read_row_groups(groups, columns=names)
    keys = pa.array([f"{c}\0{p}" for c, p in selected], type=pa.string())
    joined = pc.binary_join_element_wise(table["corpus"], table["piece"], "\0")
    return table.filter(pc.is_in(joined, value_set=keys))


def _validate_projection(
    table: pa.Table, facet: str, extent: str = _EXTENT_KIND
) -> pa.Table:
    """Apply exactly the shared validator's checks to only the read columns.

    validate_table requires unprojected mandatory columns; validate_columns
    applies the same per-column checks and casts to the columns present. No
    unselected column is read or validated.
    """
    return validation.validate_columns(table, _payload(facet), extent)


def load(
    out: str | Path,
    facet: str,
    *,
    layout: str,
    validate: bool = True,
    columns: Sequence[str] | None = None,
    pieces: Sequence[tuple[str, str]] | None = None,
    extent_kind: str | None = None,
) -> pd.DataFrame:
    """
    Read, optionally validate, and restore a canonical facet frame.

    Parameters
    ----------
    out : str or pathlib.Path
        Directory produced by write.
    facet : str
        Key in canonical.FACETS.
    layout : {'per_item', 'single', 'partitioned'}
        Stored layout.
    validate : bool, optional
        Apply schema checks and casts after reading, default True. Projections
        use the same compiled checks, restricted to selected columns.
    columns : sequence of str or None, optional
        Projection as in read_table, always including corpus/piece.
    pieces : sequence of (str, str) or None, optional
        Exact corpus/piece selection as in read_table.
    extent_kind : str or None, optional
        Extent schema tag; None defaults to quarters_interval.fraction.

    Returns
    -------
    pandas.DataFrame
        Canonical object columns and Python values, including None for nulls.

    Raises
    ------
    ArrowValidationError
        If validation is enabled and a selected value violates its constraints.
    """
    extent = _EXTENT_KIND if extent_kind is None else extent_kind
    table = read_table(
        out,
        facet,
        layout=layout,
        columns=columns,
        pieces=pieces,
        extent_kind=extent,
    )
    if validate:
        table = (
            validation.validate_table(table, _payload(facet), extent).select(
                _schema(facet, extent).names
            )
            if columns is None
            else _validate_projection(table, facet, extent)
        )
    return table_to_frame(table)


def polars_load(
    out: str | Path, facet: str, *, layout: str, extent_kind: str | None = None
) -> pl.DataFrame:
    """
    Collect a lazy Polars Parquet scan for the additional benchmark arm.

    Parameters
    ----------
    out : str or pathlib.Path
        Directory produced by write.
    facet : str
        Key in canonical.FACETS.
    layout : {'per_item', 'single', 'partitioned'}
        Stored layout.
    extent_kind : str or None, optional
        Extent schema tag of the empty result when no file exists.

    Returns
    -------
    polars.DataFrame
        Native Polars values, without canonical conversion or validation. Empty
        layouts with no files use an empty schema-derived Arrow table instead.
    """
    paths = _paths(Path(out), facet, layout)
    if not paths:
        return pl.from_arrow(
            read_table(out, facet, layout=layout, extent_kind=extent_kind)
        )
    return pl.scan_parquet(paths, hive_partitioning=False).collect()
