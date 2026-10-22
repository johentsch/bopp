"""Map BOPP column schemas to Arrow and validate the Parquet baseline.

Kinds are the registry tags of ``bopp.registries.v1``, resolved through an
index of the ``payload_type``/``extent_type`` constants of the installed JSON
schemas. Field metadata retains the original JSON item schema. Validation uses
Arrow kernels, including parent indices to report nested failures in table row
units. An entirely null optional column represents an omitted BOPP column. The
derived schema and the per-column checks are compiled once per kind and reused,
so repeated validation of many small tables is not dominated by schema work.

An integer/number item union (optionally nullable) maps to float64 because
Parquet cannot store Arrow unions. The original union remains in ``bopp:items``
metadata. This mapping cannot distinguish an integral source float such as
480.0 from an int. ``restore_number_union`` restores integral floats as ints;
the benchmark's Vienna ticks are integers, so their round trip is exact.
Integers outside the exact float64 range (magnitude above 2**53) are rejected
in float64 columns, not rounded, and integers outside the int64 range in
integer columns, both with ``ArrowValidationError``. BOPP accepts some of
these (any int in a number column, up to 2**64 - 1 in an integer column), so
Parquet validation is stricter there; the benchmark data stays far inside.

Arrow's type inference from Python lists converts a boolean that follows a
float to 1.0 or 0.0 (and a NumPy boolean that follows an int to 1 or 0), so an
order-dependent coercion would accept values that BOPP rejects.
``reject_booleans`` rejects Python and NumPy booleans in every non-boolean
scalar position, nested items included, whether a row is a list, a tuple or
a NumPy array (boolean-dtype arrays inside list cells would otherwise be
inferred as integers), and ``column_to_arrow`` applies it
before inferring a column's array and reports every Arrow conversion error as
``ArrowValidationError``. ``annotation_to_arrow`` builds its columns with it;
callers that build tables from Python values themselves must do the same,
since validate_table only sees the already coerced Arrow values.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cache
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple

import msgspec
import pyarrow as pa
import pyarrow.compute as pc

import bopp
from bopp.models.v1.annotation import Annotation

_SCHEMA_ROOT = Path(bopp.__file__).resolve().parents[1] / "schemas" / "v1"
_SCALARS = {
    "integer": pa.int64(),
    "number": pa.float64(),
    "string": pa.string(),
    "boolean": pa.bool_(),
}
#: Numeric bounds in the order their diagnostics are reported.
_BOUNDS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("minimum", pc.greater_equal),
    ("maximum", pc.less_equal),
    ("exclusiveMinimum", pc.greater),
    ("exclusiveMaximum", pc.less),
)
#: Length bounds, as the comparison that detects a violation.
_LENGTHS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("minItems", pc.less),
    ("maxItems", pc.greater),
)
#: ECMA-262 ``.``, which excludes all line terminators.
_ECMA_DOT = r"[^\n\r\x{2028}\x{2029}]"
#: Python ``re`` ``.`` without ``DOTALL``, which excludes the line feed only.
_PYTHON_DOT = r"[^\n]"
_ITEM_KEYWORDS = frozenset(
    {
        "type",
        "description",
        "title",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "items",
        "prefixItems",
        "enum",
        "pattern",
    }
)


class ArrowValidationError(ValueError):
    """A table violates a BOPP column constraint."""


class _Checks(NamedTuple):
    """Resolved Arrow type and compiled constraints of one JSON item schema."""

    type: pa.DataType
    nullable: bool
    bounds: tuple[tuple[str, Callable[..., Any], Any], ...] = ()
    lengths: tuple[tuple[str, Callable[..., Any], int], ...] = ()
    enum: pa.Array | None = None
    enum_nullable: bool = False
    pattern: str | None = None
    prefix: tuple[_Checks, ...] | None = None
    items: _Checks | None = None


class _Column(NamedTuple):
    """One table column: its JSON item schema and compiled constraints."""

    name: str
    item: dict[str, Any]
    required: bool
    checks: _Checks


@cache
def _document(path: Path) -> dict[str, Any]:
    """Read an installed schema once."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as err:
        raise ValueError(f"Unknown BOPP schema: {path.name}") from err


@cache
def _tag_index(category: str) -> dict[str, Path]:
    """
    Index installed schemas by their registry tag, not by their file stem.

    Parameters
    ----------
    category : {'payload', 'extent'}
        Schema directory to scan recursively.

    Returns
    -------
    dict of str to pathlib.Path
        Mapping of the ``<category>_type`` constant to its schema file; base
        schemas without such a constant are absent.
    """
    discriminator = f"{category}_type"
    index = {}
    for path in sorted((_SCHEMA_ROOT / category).rglob("*.json")):
        const = (
            _document(path).get("properties", {}).get(discriminator, {}).get("const")
        )
        if isinstance(const, str):
            index[const] = path
    return index


def _resolve(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve core references while retaining sibling constraints."""
    if "$ref" not in schema:
        return dict(schema)
    ref = schema["$ref"]
    match = re.fullmatch(r"(?:\.\./)*core\.json#/\$defs/(\w+)", ref)
    if match is None:
        raise ValueError(f"Unsupported schema reference: {ref}")
    definition = _document(_SCHEMA_ROOT / "core.json")["$defs"][match[1]]
    return {**definition, **{k: v for k, v in schema.items() if k != "$ref"}}


def _types(schema: Mapping[str, Any]) -> list[str]:
    """Return the explicit JSON types of an item."""
    value = schema.get("type", [])
    return [value] if isinstance(value, str) else value


def _compile(item: Mapping[str, Any]) -> _Checks:
    """
    Compile an item schema, failing closed on unknown validation keywords.

    Parameters
    ----------
    item : mapping
        A column's ``items`` schema, possibly a core ``$ref``.

    Returns
    -------
    _Checks
        Arrow type, nullability and the constraints Arrow cannot express.

    Raises
    ------
    ValueError
        If the item type, a constraint or a pattern cannot be mapped.
    """
    schema = _resolve(item)
    if unknown := schema.keys() - _ITEM_KEYWORDS:
        raise ValueError(f"Unsupported item constraints: {sorted(unknown)}")
    types = [t for t in _types(schema) if t != "null"]
    number_union = len(types) == 2 and set(types) == {"integer", "number"}
    if len(types) != 1 and not number_union:
        raise ValueError(f"Cannot map JSON item schema: {item}")
    kind = "number" if number_union else types[0]
    common = {
        "nullable": "null" in _types(schema),
        "lengths": tuple(
            (key, op, schema[key]) for key, op in _LENGTHS if key in schema
        ),
    }
    if kind == "array":
        prefix = schema.get("prefixItems")
        if prefix is not None:
            children = tuple(_compile(child) for child in prefix)
            if (
                len(children) == 2
                and schema.get("minItems") == schema.get("maxItems") == 2
                and schema.get("items") is False
                and all(child.type == pa.int64() for child in children)
            ):
                return _Checks(
                    pa.list_(pa.int64(), 2),
                    prefix=children,
                    **common,  # type: ignore[arg-type]
                )
        elif isinstance(schema.get("items"), dict):
            child = _compile(schema["items"])
            return _Checks(pa.list_(child.type), items=child, **common)  # type: ignore[arg-type]
        raise ValueError(f"Cannot map JSON item schema: {item}")
    if kind not in _SCALARS:
        raise ValueError(f"Cannot map JSON item schema: {item}")
    target = _SCALARS[kind]
    enum = schema.get("enum")
    return _Checks(
        target,
        bounds=tuple((key, op, schema[key]) for key, op in _BOUNDS if key in schema),
        enum=None if enum is None else pa.array(enum, type=target),
        enum_nullable=enum is not None and None in enum,
        pattern=None if "pattern" not in schema else _bopp_pattern(schema["pattern"]),
        **common,  # type: ignore[arg-type]
    )


def _columns(
    payload_kind: str, extent_kind: str | None
) -> Iterator[tuple[str, dict[str, Any], bool]]:
    """Yield name, item schema and requiredness in extent then payload order."""
    kinds = [("payload", payload_kind)]
    if extent_kind is not None:
        kinds.insert(0, ("extent", extent_kind))
    for category, kind in kinds:
        path = _tag_index(category).get(kind)
        if path is None:
            raise ValueError(f"Unknown {category} kind: {kind!r}")
        schema = _document(path)
        properties = schema.get("properties", {})
        if not properties:
            raise ValueError(f"Non-columnar {category}: {kind}")
        for name, column in properties.items():
            if name == f"{category}_type":
                continue
            if not column.get("$ref", "").endswith("core.json#/$defs/Array"):
                raise ValueError(f"Non-columnar {category}: {kind}.{name}")
            resolved = _resolve(column)
            if resolved.get("type") != "array" or "items" not in resolved:
                raise ValueError(f"Unsupported column: {kind}.{name}")
            if resolved.keys() - {"type", "description", "items", "title"}:
                raise ValueError(f"Unsupported column constraints: {kind}.{name}")
            yield name, column["items"], name in schema.get("required", [])


@cache
def _plan(
    payload_kind: str, extent_kind: str | None, index_columns: tuple[str, ...]
) -> tuple[pa.Schema, tuple[_Column, ...]]:
    """
    Derive and compile a kind's Arrow schema once.

    Parameters
    ----------
    payload_kind : str
        Registry payload tag.
    extent_kind : str or None
        Registry extent tag; None omits extent columns.
    index_columns : tuple of str
        Required string columns prepended in the given order.

    Returns
    -------
    schema : pyarrow.Schema
        Ordered fields with original item schemas in ``bopp:items`` metadata.
    columns : tuple of _Column
        The same columns with their compiled constraints, in field order.

    Raises
    ------
    ValueError
        If a kind, item type, constraint or column name cannot be mapped.
    """
    schema_columns = list(_columns(payload_kind, extent_kind))
    if not schema_columns:
        raise ValueError(f"Non-columnar payload: {payload_kind}")
    items = [(name, {"type": "string"}, True) for name in index_columns]
    items.extend(schema_columns)
    columns: list[_Column] = []
    fields = []
    for name, item, required in items:
        if any(column.name == name for column in columns):
            raise ValueError(f"Duplicate column: {name}")
        checks = _compile(item)
        columns.append(_Column(name, item, required, checks))
        fields.append(
            pa.field(
                name,
                checks.type,
                nullable=not required or checks.nullable,
                metadata={b"bopp:items": json.dumps(item).encode()},
            )
        )
    return pa.schema(fields), tuple(columns)


def arrow_schema(
    payload_kind: str,
    extent_kind: str | None = None,
    *,
    index_columns: Sequence[str] = ("corpus", "piece"),
) -> pa.Schema:
    """
    Derive ordered Arrow fields from installed BOPP JSON schemas.

    Parameters
    ----------
    payload_kind : str
        Registry payload tag, including dotted control event kinds.
    extent_kind : str or None, optional
        Registry extent tag; None omits extent columns.
    index_columns : sequence of str, optional
        Required string columns prepended in the given order.

    Returns
    -------
    pyarrow.Schema
        Typed fields with original item schemas in ``bopp:items`` metadata.
        Cached per argument triple; the result must not be mutated.

    Raises
    ------
    ValueError
        If a kind, item type, constraint or column name cannot be mapped.
    """
    return _plan(payload_kind, extent_kind, tuple(index_columns))[0]


def _translate(pattern: str, dot: str) -> str:
    """Translate an ECMA pattern to RE2, spelling an unescaped ``.`` as ``dot``."""
    result = []
    i = 0
    in_class = False
    while i < len(pattern):
        char = pattern[i]
        if not in_class and pattern.startswith(r"(?![\s\S])", i):
            result.append(r"\z")
            i += len(r"(?![\s\S])")
            continue
        if char == "\\":
            if i + 1 == len(pattern):
                raise ValueError("Incomplete pattern escape")
            escaped = pattern[i + 1]
            if escaped == "u":
                digits = pattern[i + 2 : i + 6]
                if not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                    raise ValueError("Unsupported Unicode escape")
                if 0xD800 <= int(digits, 16) <= 0xDFFF:
                    raise ValueError("Unsupported surrogate escape")
                result.append(r"\x{" + digits + "}")
                i += 6
                continue
            if escaped not in r"\.^$|?*+()[]{}-/nrtfv":
                raise ValueError(f"Unsupported pattern escape: \\{escaped}")
            result.append("\\" + escaped)
            i += 2
            continue
        if char == "[" and not in_class:
            in_class = True
        elif char == "]" and in_class:
            in_class = False
        elif not in_class:
            if pattern.startswith("(?", i) and not pattern.startswith("(?:", i):
                raise ValueError("Unsupported pattern group or lookaround")
            if char in "*+?}" and pattern[i + 1 : i + 2] == "+":
                raise ValueError("Unsupported possessive repetition")
            if char == ".":
                result.append(dot)
                i += 1
                continue
        result.append(char)
        i += 1
    translated = "".join(result)
    try:
        pc.match_substring_regex(pa.array([], type=pa.string()), translated)
    except pa.ArrowInvalid as err:
        raise ValueError(f"Unsupported pattern: {err}") from err
    return translated


def ecma_to_re2(pattern: str) -> str:
    r"""
    Translate the repository's ECMA patterns without adding anchors.

    Parameters
    ----------
    pattern : str
        JSON Schema pattern, with JSON escapes already decoded.

    Returns
    -------
    str
        RE2 pattern; strict end assertions become ``\z``, ``.`` becomes the
        ECMA-262 class excluding all line terminators and Unicode escapes
        become RE2 hexadecimal escapes. Search semantics are retained.

    Raises
    ------
    ValueError
        If an unsupported escape, group or repetition construct occurs.
    """
    return _translate(pattern, _ECMA_DOT)


def _fail(name: str, rule: str, count: int, first: int | None) -> None:
    """Raise one uniform, row-oriented diagnostic."""
    raise ArrowValidationError(
        f"Column '{name}': {rule}; {count} offending rows; first offending row index {first}"
    )


@cache
def _bopp_pattern(pattern: str) -> str:
    """Reproduce the Python ``re`` semantics applied by msgspec decoding."""
    translated = _translate(pattern, _PYTHON_DOT)
    backslashes = len(pattern[:-1]) - len(pattern[:-1].rstrip("\\"))
    if pattern.endswith("$") and backslashes % 2 == 0:
        # All dollar assertions in the installed payload schemas are terminal.
        # Python accepts a final LF here; ECMA and RE2 require the actual end.
        translated = translated[:-1] + r"(?:\n?\z)"
    return translated


def _check(
    name: str,
    rule: str,
    mask: pa.Array | pa.ChunkedArray,
    parents: pa.Array | None = None,
) -> None:
    """Report distinct table rows, even when multiple children fail."""
    mask = pc.fill_null(mask, False)
    if not pc.any(mask).as_py():
        return
    if parents is None:
        count = pc.sum(pc.cast(mask, pa.int64())).as_py()
        first = pc.index(mask, True).as_py()
    else:
        rows = pc.unique(pc.filter(parents, mask))
        count = len(rows)
        first = pc.min(rows).as_py()
    _fail(name, rule, count, first)


def _validate_values(
    values: pa.Array | pa.ChunkedArray,
    checks: _Checks,
    name: str,
    parents: pa.Array | None = None,
) -> None:
    """Check scalars and nested items using kernels and parent row indices."""
    if not len(values):
        return
    target = checks.type
    if pa.types.is_dictionary(values.type):
        values = pc.dictionary_decode(values)
    if not checks.nullable:
        _check(name, "null", pc.is_null(values), parents)
    if pa.types.is_null(values.type):
        if checks.enum is not None and not checks.enum_nullable:
            _check(name, "enum", pc.is_null(values), parents)
        return
    is_list = any(
        check(values.type)
        for check in (
            pa.types.is_list,
            pa.types.is_large_list,
            pa.types.is_fixed_size_list,
        )
    )
    if pa.types.is_list(target) or pa.types.is_fixed_size_list(target):
        if not is_list:
            _check(name, "type (array)", pc.is_valid(values), parents)
            return
        lengths = pc.list_value_length(values)
        for key, op, bound in checks.lengths:
            _check(name, key, op(lengths, bound), parents)
        if checks.prefix is not None:
            for position, child_checks in enumerate(checks.prefix):
                child = pc.list_element(values, position)
                # Null parents have no children subject to validation.
                present = pc.is_valid(values)
                row_ids = pc.indices_nonzero(present)
                if parents is not None:
                    row_ids = pc.take(parents, row_ids)
                _validate_values(pc.filter(child, present), child_checks, name, row_ids)
        else:
            row_ids = pc.list_parent_indices(values)
            if parents is not None:
                row_ids = pc.take(parents, row_ids)
            _validate_values(pc.list_flatten(values), checks.items, name, row_ids)
        return
    # BOPP decoding rejects a float for an integer column, so Arrow must too.
    compatible = (
        pa.types.is_integer(target)
        and pa.types.is_integer(values.type)
        or pa.types.is_floating(target)
        and (pa.types.is_integer(values.type) or pa.types.is_floating(values.type))
        or pa.types.is_string(target)
        and (pa.types.is_string(values.type) or pa.types.is_large_string(values.type))
        or pa.types.is_boolean(target)
        and pa.types.is_boolean(values.type)
    )
    if not compatible:
        _check(name, f"type ({target})", pc.is_valid(values), parents)
        return
    if pa.types.is_integer(target) and pa.types.is_unsigned_integer(values.type):
        _check(
            name,
            "int64 range",
            pc.greater(values, pa.scalar(2**63 - 1, pa.uint64())),
            parents,
        )
    elif pa.types.is_floating(target) and pa.types.is_integer(values.type):
        # Arrow's safe cast rejects integers outside the exact float64 range.
        _check(name, "float64 exact range", pc.greater(values, 2**53), parents)
        if pa.types.is_signed_integer(values.type):
            _check(name, "float64 exact range", pc.less(values, -(2**53)), parents)
    for key, op, bound in checks.bounds:
        _check(name, key, pc.invert(op(values, bound)), parents)
    if checks.enum is not None:
        bad = pc.invert(pc.is_in(pc.cast(values, target), value_set=checks.enum))
        _check(name, "enum", bad, parents)
    if checks.pattern is not None:
        valid = pc.match_substring_regex(values, checks.pattern)
        _check(name, "pattern", pc.invert(valid), parents)


def validate_table(
    table: pa.Table,
    payload_kind: str,
    extent_kind: str | None = None,
    *,
    index_columns: Sequence[str] = ("corpus", "piece"),
) -> pa.Table:
    """
    Validate column constraints and cast a table to its exact BOPP schema.

    Parameters
    ----------
    table : pyarrow.Table
        Input table; optional columns may be absent and types may be wider
        physical representations such as large strings or narrower integers.
        Floats are not accepted for integer columns, as in BOPP decoding.
        Coercions that happened while the table was built from Python values
        (e.g. a boolean inferred as 1.0) are invisible here; build columns with
        column_to_arrow, or call reject_booleans first.
    payload_kind : str
        Registry payload tag.
    extent_kind : str or None, optional
        Registry extent tag.
    index_columns : sequence of str, optional
        Required string index columns.

    Returns
    -------
    pyarrow.Table
        Ordered, safely cast table, with absent optional columns filled by nulls.

    Raises
    ------
    ArrowValidationError
        On the first constraint failure, with column, rule and row statistics.
    ValueError
        If the requested schema cannot be mapped.
    """
    schema, columns = _plan(payload_kind, extent_kind, tuple(index_columns))
    _check_names(table, schema)
    present = set(table.column_names)
    arrays = []
    for field, column in zip(schema, columns, strict=True):
        if column.name not in present:
            if column.required:
                _fail(
                    column.name,
                    "missing required column",
                    table.num_rows,
                    0 if table.num_rows else None,
                )
            arrays.append(pa.nulls(table.num_rows, field.type))
            continue
        arrays.append(_validated_column(table[column.name], field, column))
    return pa.Table.from_arrays(arrays, schema=schema)


def _check_names(table: pa.Table, schema: pa.Schema) -> None:
    """Reject unknown and duplicate table columns."""
    names = set(schema.names)
    for name in table.column_names:
        if name not in names:
            _fail(name, "unknown column", table.num_rows, 0 if table.num_rows else None)
        if table.column_names.count(name) != 1:
            _fail(
                name, "duplicate column", table.num_rows, 0 if table.num_rows else None
            )


def _validated_column(
    values: pa.ChunkedArray, field: pa.Field, column: _Column
) -> pa.ChunkedArray | pa.Array:
    """Check one present column and cast it safely to its schema type."""
    if not column.required and values.null_count == len(values):
        return pa.nulls(len(values), field.type)
    _validate_values(values, column.checks, column.name)
    return pc.cast(values, field.type, safe=True)


def validate_columns(
    table: pa.Table,
    payload_kind: str,
    extent_kind: str | None = None,
    *,
    index_columns: Sequence[str] = ("corpus", "piece"),
) -> pa.Table:
    """
    Validate a column subset, such as a projection, with validate_table's rules.

    Parameters
    ----------
    table : pyarrow.Table
        Any subset of the schema columns, in any order.
    payload_kind : str
        Registry payload tag.
    extent_kind : str or None, optional
        Registry extent tag.
    index_columns : sequence of str, optional
        Required string index columns.

    Returns
    -------
    pyarrow.Table
        The given columns in table order, checked and safely cast exactly as
        validate_table checks and casts them.

    Raises
    ------
    ArrowValidationError
        On an unknown or duplicate column, or the first constraint failure of a
        present column. Absent columns, required or not, are not reported.
    ValueError
        If the requested schema cannot be mapped.
    """
    schema, columns = _plan(payload_kind, extent_kind, tuple(index_columns))
    _check_names(table, schema)
    specs = {column.name: column for column in columns}
    fields = [schema.field(name) for name in table.column_names]
    arrays = [
        _validated_column(table[field.name], field, specs[field.name])
        for field in fields
    ]
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def restore_number_union(
    values: Sequence[int | float | None],
) -> list[int | float | None]:
    """
    Restore integral floats from an Arrow integer/number union column as ints.

    Parameters
    ----------
    values : sequence of int, float or None
        Values from a column whose item schema admits integer and number.

    Returns
    -------
    list of int, float or None
        Integral floats converted to ints; all other values, including nulls
        and non-finite floats, retained. Source integral floats and integers
        cannot be distinguished after storage as float64.
    """
    return [
        int(value) if isinstance(value, float) and value.is_integer() else value
        for value in values
    ]


def _boolean_types() -> tuple[type, ...]:
    """Return the boolean types a column may hold; NumPy only once imported."""
    numpy = sys.modules.get("numpy")
    return (bool,) if numpy is None else (bool, numpy.bool_)


#: Row containers walked by the guards; others are left to Arrow's type checks.
_ROW_TYPES = frozenset({list, tuple, type(None)})


def _numpy_row_may_hold(value: Any) -> bool:
    """
    Whether a NumPy array row can hold a boolean or an out-of-range integer.

    Arrow infers a boolean array inside a list column as integers (``[[2],
    np.array([True])]`` becomes ``[[2], [1]]``), so such rows need the walk.
    The test is one dtype check per row: boolean and object arrays may hold
    booleans, unsigned arrays integers above the int64 range; signed integer
    and float arrays hold neither, whatever their length.

    Parameters
    ----------
    value : object
        One row of a list column.

    Returns
    -------
    bool
        True for a NumPy array of boolean, unsigned or object dtype.
    """
    numpy = sys.modules.get("numpy")
    return (
        numpy is not None
        and isinstance(value, numpy.ndarray)
        and value.dtype.kind in "buO"
    )


def _is_leaf(checks: _Checks) -> bool:
    """Whether an item schema is a scalar rather than an array."""
    return checks.prefix is None and checks.items is None


def _may_hold(values: Sequence[Any], checks: _Checks, kinds: frozenset[type]) -> bool:
    """
    Screen a column for scalars of the given types in non-boolean positions.

    One type-set pass per nesting level, without per-item Python calls. A False
    result is exact; a True result only selects the exact walk.

    Parameters
    ----------
    values : sequence
        Python values of one column, or of one nesting level of it.
    checks : _Checks
        Compiled item schema of ``values``.
    kinds : frozenset of type
        Scalar types to look for.

    Returns
    -------
    bool
        False if no such scalar can occur at a non-boolean scalar position.
    """
    types = set(map(type, values))
    if _is_leaf(checks):
        return not pa.types.is_boolean(checks.type) and not types.isdisjoint(kinds)
    if not types <= _ROW_TYPES:
        return True
    # Only None and empty rows are falsy here, and neither holds items.
    flat = chain.from_iterable(filter(None, values))
    children = (checks.items,) if checks.items is not None else checks.prefix
    assert children is not None
    if checks.items is not None and not _is_leaf(checks.items):
        return _may_hold(list(flat), checks.items, kinds)
    if not all(map(_is_leaf, children)) or any(
        pa.types.is_boolean(child.type) for child in children
    ):
        return True
    return not set(map(type, flat)).isdisjoint(kinds)


def _offending_rows(
    values: Sequence[Any],
    checks: _Checks,
    offends: Callable[[Any, _Checks], bool],
) -> set[int]:
    """
    Find rows holding a scalar that offends its item schema.

    Parameters
    ----------
    values : sequence
        Python values of one column, or of one nesting level of it.
    checks : _Checks
        Compiled item schema of ``values``.
    offends : callable
        Test of one scalar against its compiled scalar item schema.

    Returns
    -------
    set of int
        Indices into ``values``. NumPy array rows of boolean, unsigned or
        object dtype are walked as nested lists; other non-list values of list
        items are left to the Arrow type checks.
    """
    if _is_leaf(checks):
        return {i for i, value in enumerate(values) if offends(value, checks)}
    rows = [
        (i, value if isinstance(value, list | tuple) else value.tolist())
        for i, value in enumerate(values)
        if isinstance(value, list | tuple) or _numpy_row_may_hold(value)
    ]
    if checks.prefix is not None:
        bad: set[int] = set()
        for position, child in enumerate(checks.prefix):
            picked = [(i, row[position]) for i, row in rows if len(row) > position]
            hits = _offending_rows([value for _, value in picked], child, offends)
            bad.update(picked[hit][0] for hit in hits)
        return bad
    assert checks.items is not None
    parents = [i for i, row in rows for _ in row]
    hits = _offending_rows([v for _, row in rows for v in row], checks.items, offends)
    return {parents[hit] for hit in hits}


def _out_of_range(value: Any, checks: _Checks) -> bool:
    """Whether a Python int exceeds the exact range of its Arrow target."""
    if type(value) is not int:
        return False
    if pa.types.is_floating(checks.type):
        return abs(value) > 2**53
    if pa.types.is_integer(checks.type):
        return not -(2**63) <= value <= 2**63 - 1
    return False


def _range_rule(checks: _Checks) -> str:
    """Name the range rule of a column's scalar items, as validate_table does."""
    while not _is_leaf(checks):
        checks = checks.prefix[0] if checks.prefix is not None else checks.items  # type: ignore[assignment]
    return "float64 exact range" if pa.types.is_floating(checks.type) else "int64 range"


def _column_checks(
    payload_kind: str,
    extent_kind: str | None,
    index_columns: Sequence[str],
) -> dict[str, _Checks]:
    """Compiled item schemas of a kind's columns, by name (cached; read only)."""
    return _checks_by_name(payload_kind, extent_kind, tuple(index_columns))


@cache
def _checks_by_name(
    payload_kind: str, extent_kind: str | None, index_columns: tuple[str, ...]
) -> dict[str, _Checks]:
    """Index a compiled plan by column name once."""
    plan = _plan(payload_kind, extent_kind, index_columns)[1]
    return {column.name: column.checks for column in plan}


def _checks_of(name: str, checks: Mapping[str, _Checks], size: int) -> _Checks:
    """Look a column up, reporting unknown names like validate_table."""
    if name not in checks:
        _fail(name, "unknown column", size, 0 if size else None)
    return checks[name]


def reject_booleans(
    columns: Mapping[str, Sequence[Any]],
    payload_kind: str,
    extent_kind: str | None = None,
    *,
    index_columns: Sequence[str] = ("corpus", "piece"),
) -> None:
    """
    Reject Python and NumPy booleans that Arrow inference would coerce.

    ``pa.array`` turns a boolean that follows a float into 1.0 or 0.0, and a
    NumPy boolean that follows an int into 1 or 0, so validate_table can no
    longer see them. Call this on the Python values of every column before
    building Arrow arrays from them; boolean columns are left alone. Columns
    are screened with one type-set pass per nesting level, and only a column
    that holds a boolean is walked item by item.

    Parameters
    ----------
    columns : mapping of str to sequence
        Python values per column, in wire form (fraction pairs and integer
        lists as lists or tuples, None for nulls); any subset of the columns.
    payload_kind : str
        Registry payload tag.
    extent_kind : str or None, optional
        Registry extent tag.
    index_columns : sequence of str, optional
        Required string index columns.

    Raises
    ------
    ArrowValidationError
        If a boolean occupies a non-boolean scalar position, nested items
        included, or a column name is unknown.
    ValueError
        If the requested schema cannot be mapped.
    """
    checks = _column_checks(payload_kind, extent_kind, index_columns)
    kinds = frozenset(_boolean_types())
    for name, values in columns.items():
        item = _checks_of(name, checks, len(values))
        if not _may_hold(values, item, kinds):
            continue
        if bad := _offending_rows(
            values,
            item,
            lambda value, leaf: (
                type(value) in kinds and not pa.types.is_boolean(leaf.type)
            ),
        ):
            _fail(name, f"type ({item.type}): boolean", len(bad), min(bad))


def column_to_arrow(
    name: str,
    values: Sequence[Any],
    payload_kind: str,
    extent_kind: str | None = None,
    *,
    index_columns: Sequence[str] = ("corpus", "piece"),
) -> pa.Array:
    """
    Build one column's Arrow array by inference, guarded against coercions.

    The array is inferred, not built with its schema type: an explicit type
    would still coerce booleans and would truncate 2.5 to 2 in an integer
    column, whereas inference plus validate_table rejects such values. The
    result still needs validate_table (or validate_columns).

    Parameters
    ----------
    name : str
        Column name.
    values : sequence
        Python values in wire form, as for reject_booleans.
    payload_kind : str
        Registry payload tag.
    extent_kind : str or None, optional
        Registry extent tag.
    index_columns : sequence of str, optional
        Required string index columns.

    Returns
    -------
    pyarrow.Array
        Inferred array; validate_table checks its type and constraints.

    Raises
    ------
    ArrowValidationError
        For booleans (see reject_booleans), for Python ints outside the exact
        range of the column's type (``float64 exact range``: magnitude above
        2**53; ``int64 range``), and for any other value Arrow cannot convert.
    ValueError
        If the requested schema cannot be mapped.
    """
    reject_booleans(
        {name: values}, payload_kind, extent_kind, index_columns=index_columns
    )
    try:
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError) as err:
        item = _column_checks(payload_kind, extent_kind, index_columns)[name]
        if bad := _offending_rows(values, item, _out_of_range):
            _fail(name, _range_rule(item), len(bad), min(bad))
        raise ArrowValidationError(
            f"Column '{name}': type ({item.type}): not convertible ({err})"
        ) from err


def annotation_to_arrow(
    ann: Annotation,
    *,
    index: Mapping[str, str] | None = None,
) -> pa.Table:
    """
    Convert a columnar BOPP annotation to a validated Arrow table.

    Parameters
    ----------
    ann : bopp.models.v1.annotation.Annotation
        Annotation containing a supported payload and extent.
    index : mapping of str to str or None, optional
        Constant index columns, in insertion order; None adds no index.

    Returns
    -------
    pyarrow.Table
        Table with schema-derived column types, constraints and metadata.

    Raises
    ------
    ArrowValidationError
        If annotation columns violate a schema constraint.
    ValueError
        If payload or extent cannot be mapped, or lengths differ.
    TypeError
        If an index constant is not a string.
    """
    index = {} if index is None else index
    payload_kind: str = type(ann.payload).__struct_config__.tag
    extent_kind: str = type(ann.extent).__struct_config__.tag
    schema = _plan(payload_kind, extent_kind, tuple(index))[0]
    columns: dict[str, Any] = {}
    for struct in (ann.extent, ann.payload):
        for name in struct.__struct_fields__:
            values = getattr(struct, name)
            if values is msgspec.UNSET:
                continue
            if pa.types.is_fixed_size_list(schema.field(name).type):
                values = [
                    list(value) if value is not None else None for value in values
                ]
            columns[name] = values
    size = len(next(iter(columns.values())))
    if len({len(values) for values in columns.values()}) != 1:
        raise ValueError("Annotation columns differ in length")
    for name, value in index.items():
        if not isinstance(value, str):
            raise TypeError(f"Index column '{name}' requires a constant string")
        columns[name] = [value] * size
    # Infer lists first so invalid fraction lengths receive row diagnostics;
    # the guard keeps inference from coercing booleans BOPP rejects.
    table = pa.table(
        {
            name: column_to_arrow(
                name, values, payload_kind, extent_kind, index_columns=tuple(index)
            )
            for name, values in columns.items()
        }
    )
    return validate_table(table, payload_kind, extent_kind, index_columns=tuple(index))
