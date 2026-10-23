"""Define the canonical DLC facet tables shared by every storage benchmark.

Tables use object columns, Python scalar values, Fraction positions and rational
payload values, tuples for integer lists, and None for absent values. Column
order follows the generated extent and payload structs, including every optional
payload column. Piece order is lexical (corpus, piece); row order within each
piece is the reader's order. No reader failure is suppressed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from functools import cache
from math import copysign, isnan
from pathlib import Path
from typing import Any, TypedDict

import msgspec
import numpy as np
import pandas as pd
from msgspec import inspect as mi

import bopp
from bopp.exceptions import BoppArgumentError
from bopp.models.v1.annotation import Annotation
from bopp.registries.v1 import EXTENT_TYPE_REGISTRY, PAYLOAD_TYPE_REGISTRY
from experiments.dlc_chords import chords_to_control_events
from experiments.dlc_labels import harmonies_to_dcml_label
from experiments.dlc_measures import measures_to_score_measure
from experiments.dlc_notes import notes_to_score_note


class _Facet(TypedDict):
    facet_dir: str
    payload_kind: str
    reader: Callable[..., Any]


#: Source directory, payload kind and piece reader for each benchmark facet.
FACETS: dict[str, _Facet] = {
    "notes": {
        "facet_dir": "notes",
        "payload_kind": "score_note",
        "reader": notes_to_score_note,
    },
    "measures": {
        "facet_dir": "measures",
        "payload_kind": "score_measure",
        "reader": measures_to_score_measure,
    },
    "labels": {
        "facet_dir": "harmonies",
        "payload_kind": "dcml_label",
        "reader": harmonies_to_dcml_label,
    },
    **{
        f"control_events.{kind}": {
            "facet_dir": "chords",
            "payload_kind": f"score_control_event.{kind}",
            "reader": chords_to_control_events,
        }
        for kind in (
            "chord",
            "dynamic",
            "spanner",
            "figured_bass",
            "staff_text",
            "system_text",
            "tempo",
        )
    },
}
_EXTENT_KIND = "quarters_interval.fraction"
_MAX_DIFFERENCES = 20
_SCALAR_TYPES = {int, float, str, bool, Fraction, type(None)}


def _unwrap(info: mi.Type) -> mi.Type:
    """Remove metadata and nullable wrappers from a schema type."""
    if isinstance(info, mi.Metadata):
        return _unwrap(info.type)
    if isinstance(info, mi.UnionType):
        non_null = [t for t in info.types if not isinstance(t, mi.NoneType)]
        if len(non_null) == 1:
            return _unwrap(non_null[0])
    return info


def _converter(info: mi.Type) -> Callable[[Any], Any]:
    """Choose canonical cell conversion from the generated column item type."""
    column = _unwrap(info)
    if not isinstance(column, mi.ListType):
        raise BoppArgumentError("Canonical facets require parallel array columns")
    item = _unwrap(column.item_type)
    if isinstance(item, mi.TupleType) and len(item.item_types) == 2:
        return lambda value: Fraction(*value)
    if isinstance(item, mi.ListType) and isinstance(
        _unwrap(item.item_type), mi.IntType
    ):
        return lambda value: tuple(int(v) for v in value)
    for schema_type, scalar_type in (
        (mi.IntType, int),
        (mi.FloatType, float),
        (mi.StrType, str),
        (mi.BoolType, bool),
    ):
        if isinstance(item, schema_type):
            return scalar_type
    raise BoppArgumentError(f"Unsupported canonical column type: {item!r}")


@cache
def _columns(struct: type) -> tuple[tuple[str, bool, Callable[[Any], Any]], ...]:
    """Read field order, requiredness and cell conversion from a payload/extent."""
    info = mi.type_info(struct)
    if not isinstance(info, mi.StructType):
        raise BoppArgumentError("Expected a generated extent or payload struct")
    return tuple((f.name, f.required, _converter(f.type)) for f in info.fields)


def _frame_columns(payload_kind: str, extent_kind: str) -> list[str]:
    return [
        "corpus",
        "piece",
        *(name for name, _, _ in _columns(EXTENT_TYPE_REGISTRY[extent_kind])),
        *(name for name, _, _ in _columns(PAYLOAD_TYPE_REGISTRY[payload_kind])),
    ]


def annotation_to_frame(ann: Annotation, *, corpus: str, piece: str) -> pd.DataFrame:
    """
    Convert an annotation to the canonical table for one DLC piece.

    Parameters
    ----------
    ann : Annotation
        Reader-produced annotation with a columnar extent and payload.
    corpus, piece : str
        Source identity to prepend to every row.

    Returns
    -------
    pandas.DataFrame
        Object columns in identity, extent, payload order. Every payload schema
        field is present; unset optional columns and null cells become None.
        Fraction pairs become Fraction and integer lists become tuples.

    Raises
    ------
    BoppArgumentError
        If an extent is missing, a schema type is unsupported, or a value is NaN.
    """
    if ann.extent is msgspec.UNSET:
        raise BoppArgumentError("A canonical facet requires an extent")
    extent_columns = _columns(type(ann.extent))
    count = len(getattr(ann.extent, extent_columns[0][0]))
    columns: dict[str, Any] = {"corpus": [corpus] * count, "piece": [piece] * count}
    for struct in (ann.extent, ann.payload):
        for name, _, convert in _columns(type(struct)):
            values = getattr(struct, name)
            if values is msgspec.UNSET or values is None:
                columns[name] = [None] * count
            else:
                canonical = [None if v is None else convert(v) for v in values]
                if any(type(v) is float and isnan(v) for v in canonical):
                    raise BoppArgumentError(f"Column '{name}' contains NaN")
                columns[name] = canonical
    return pd.DataFrame(columns, dtype=object)


def _wire(value: Any) -> Any:
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, tuple):
        return list(value)
    return value


def frame_to_annotation(
    df: pd.DataFrame, *, payload_kind: str, extent_kind: str, media_id: str
) -> Annotation:
    """
    Reconstruct a reader annotation from one piece's canonical table.

    Parameters
    ----------
    df : pandas.DataFrame
        Canonical rows covering exactly one corpus/piece pair.
    payload_kind, extent_kind : str
        Registered BOPP payload and extent kinds.
    media_id : str
        Annotation media identifier; use the reader's identifier to retain its id.

    Returns
    -------
    Annotation
        Annotation created by bopp.create, omitting wholly None optional columns.
        Fractions and integer tuples are restored to wire-format lists.

    Raises
    ------
    BoppArgumentError
        If identity is ambiguous, required columns are missing, or unexpected
        columns would otherwise be discarded.
    """
    if (
        not {"corpus", "piece"}.issubset(df.columns)
        or len(df[["corpus", "piece"]].drop_duplicates()) != 1
    ):
        raise BoppArgumentError("Expected a single corpus/piece pair")
    expected = _frame_columns(payload_kind, extent_kind)
    if not df.columns.is_unique or any(name not in expected for name in df.columns):
        raise BoppArgumentError("Unexpected or duplicate canonical columns")
    columns = {}
    for struct in (
        EXTENT_TYPE_REGISTRY[extent_kind],
        PAYLOAD_TYPE_REGISTRY[payload_kind],
    ):
        for name, required, _ in _columns(struct):
            if name not in df:
                if required:
                    raise BoppArgumentError(f"Column '{name}' is missing")
                continue
            values = df[name].tolist()
            if not required and all(v is None for v in values):
                continue
            columns[name] = [_wire(v) for v in values]
    return bopp.create(
        media_id=media_id, payload_kind=payload_kind, extent_kind=extent_kind, **columns
    )


def _piece_paths(
    root: str | Path,
    facet_dir: str,
    corpora: Sequence[str] | None,
    pieces: Sequence[str] | None,
) -> list[tuple[str, str, Path]]:
    """List the selected (corpus, piece, path) triples of a facet directory."""
    selected_corpora = None if corpora is None else set(corpora)
    selected_pieces = None if pieces is None else set(pieces)
    suffix = f".{facet_dir}.tsv"
    paths = []
    for path in Path(root).glob(f"*/{facet_dir}/*{suffix}"):
        corpus = path.parent.parent.name
        piece = path.name.removesuffix(suffix)
        if selected_corpora is not None and corpus not in selected_corpora:
            continue
        if selected_pieces is not None and piece not in selected_pieces:
            continue
        paths.append((corpus, piece, path))
    return sorted(paths)


def _read_piece(path: Path, corpus: str, piece: str) -> pd.DataFrame:
    """Read one facet TSV as strings, prefixed with its source identity."""
    source = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    source.insert(0, "piece", piece)
    source.insert(0, "corpus", corpus)
    return source


def _piece_end(corpus_dir: Path, corpus: str, piece: str) -> Fraction:
    """
    Compute the exact quarterbeat where a piece ends, from its measure table.

    The end of a measure unit is its ``quarters_interval.fraction`` extent start
    plus its duration, as built by `measures_to_score_measure`. Pieces with
    voltas repeat quarterbeats across endings, so the maximum over all units is
    taken; without voltas that is the last unit in score order.
    """
    path = corpus_dir / "measures" / f"{piece}.measures.tsv"
    if not path.is_file():
        raise BoppArgumentError(f"Measure table is missing: {str(path)!r}")
    extent = measures_to_score_measure(_read_piece(path, corpus, piece)).extent
    return max(
        Fraction(*quarter) + Fraction(*duration)
        for quarter, duration in zip(extent.quarter, extent.duration, strict=True)
    )


def _piece_annotations(
    facet_dir: str,
    facets: Sequence[str],
    source: pd.DataFrame,
    path: Path,
    corpus: str,
    piece: str,
) -> dict[str, Annotation | None]:
    """Call a facet directory's reader once and select each requested facet."""
    if facet_dir == "chords":
        events = chords_to_control_events(source)
        return {facet: events.get(facet.split(".", 1)[1]) for facet in facets}
    reader_kwargs: dict[str, Any] = {}
    if facet_dir == "harmonies" and (
        "duration_qb" in source and len(source) and source["duration_qb"].iloc[-1] == ""
    ):
        # Only a final empty duration_qb needs the measure table, so the extra
        # TSV is read for the handful of pieces that end inside a volta.
        reader_kwargs["piece_end"] = _piece_end(path.parent.parent, corpus, piece)
    annotation = FACETS[facets[0]]["reader"](source, **reader_kwargs)
    return dict.fromkeys(facets, annotation)


def load_dlc_facets(
    root: str | Path,
    facets: Sequence[str],
    *,
    corpora: Sequence[str] | None = None,
    pieces: Sequence[str] | None = None,
    errors: list[tuple[str, str, str, Exception]] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load several DLC facets, parsing every source TSV exactly once.

    Parameters
    ----------
    root : str or pathlib.Path
        DLC checkout containing corpus directories.
    facets : sequence of str
        Keys in FACETS; duplicates are collapsed. Facets sharing a source
        directory share one read and one reader call per piece, which matters
        for the seven control-event kinds of the chords tables.
    corpora, pieces : sequence of str or None, optional
        Corpus directory names and piece basenames to select. None selects all;
        an empty sequence selects none. Piece names apply across selected corpora.
    errors : list or None, optional
        None (default) lets reader errors propagate. A list instead collects
        ``(facet_dir, corpus, piece, exception)`` for every piece whose reader
        raised BoppArgumentError, and that piece contributes no rows to the
        facets of that directory.

    Returns
    -------
    dict of str to pandas.DataFrame
        One canonical table per requested facet, keyed as requested, each as
        `load_dlc_facet` returns it.

    Raises
    ------
    BoppArgumentError
        If a facet is unknown. Reader errors propagate unless collected.
    """
    requested = list(dict.fromkeys(facets))
    for facet in requested:
        if facet not in FACETS:
            raise BoppArgumentError(f"Unknown DLC facet: {facet!r}")
    groups: dict[str, list[str]] = {}
    for facet in requested:
        groups.setdefault(FACETS[facet]["facet_dir"], []).append(facet)
    frames: dict[str, list[pd.DataFrame]] = {facet: [] for facet in requested}
    for facet_dir, group in groups.items():
        for corpus, piece, path in _piece_paths(root, facet_dir, corpora, pieces):
            source = _read_piece(path, corpus, piece)
            try:
                annotations = _piece_annotations(
                    facet_dir, group, source, path, corpus, piece
                )
            except BoppArgumentError as err:
                if errors is None:
                    raise
                errors.append((facet_dir, corpus, piece, err))
                continue
            for facet, annotation in annotations.items():
                if annotation is not None:
                    frames[facet].append(
                        annotation_to_frame(annotation, corpus=corpus, piece=piece)
                    )
    return {
        facet: pd.concat(frames[facet], ignore_index=True)
        if frames[facet]
        else pd.DataFrame(
            columns=_frame_columns(FACETS[facet]["payload_kind"], _EXTENT_KIND),
            dtype=object,
        )
        for facet in requested
    }


def load_dlc_facet(
    root: str | Path,
    facet: str,
    *,
    corpora: Sequence[str] | None = None,
    pieces: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load the selected DLC facet into one canonical pandas table.

    Parameters
    ----------
    root : str or pathlib.Path
        DLC checkout containing corpus directories.
    facet : str
        Key in FACETS.
    corpora, pieces : sequence of str or None, optional
        Corpus directory names and piece basenames to select. None selects all;
        an empty sequence selects none. Piece names apply across selected corpora.

    Returns
    -------
    pandas.DataFrame
        Canonical rows sorted by corpus and piece, preserving reader row order.
        Missing TSVs or absent event kinds contribute no rows. An empty selection
        still has all schema columns with object dtype and a reset index.

    Raises
    ------
    BoppArgumentError
        If the facet is unknown. Reader errors propagate unchanged.

    See Also
    --------
    load_dlc_facets : Load several facets, sharing one reader call per piece.
    """
    return load_dlc_facets(root, [facet], corpora=corpora, pieces=pieces)[facet]


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, (tuple, list)):
        # Lists are not canonical, but a loader returning them must still be
        # compared element-wise, and a list never equals a tuple.
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is float:
        return left == right and copysign(1.0, left) == copysign(1.0, right)
    return bool(left == right)


def frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """
    Compare canonical tables exactly, including cell types and column order.

    Parameters
    ----------
    left, right : pandas.DataFrame
        Tables to compare by row position; index labels are not part of a facet.

    Returns
    -------
    list of str
        At most 20 human-readable differences, or an empty list when equal.
        Every column whose dtype is not object is reported, because a canonical
        facet stores Python scalars, never numpy or pyarrow-backed values.
        Tuples and lists are compared recursively with the same type strictness,
        and 0.0 differs from -0.0. Scalar columns use array comparisons after
        checking each raw cell's Python type.
    """
    differences = []
    if list(left.columns) != list(right.columns):
        differences.append(
            f"Columns differ (names/order): {list(left.columns)!r} != {list(right.columns)!r}"
        )
    if len(left) != len(right):
        differences.append(f"Row count differs: {len(left)} != {len(right)}")
    if list(left.columns) != list(right.columns):
        return differences
    for side, frame in (("left", left), ("right", right)):
        for name, dtype in frame.dtypes.items():
            if dtype != object:
                differences.append(
                    f"Column {name!r}: dtype {dtype} != object ({side} frame)"
                )
                if len(differences) >= _MAX_DIFFERENCES:
                    return differences
    count = min(len(left), len(right))
    cell_types = np.frompyfunc(type, 1, 1)
    for column_index, name in enumerate(left.columns):
        # Raw values: converting to object dtype first would turn numpy and
        # pyarrow-backed cells into Python scalars and hide their types.
        a = left.iloc[:count, column_index].to_numpy()
        b = right.iloc[:count, column_index].to_numpy()
        a_types, b_types = cell_types(a), cell_types(b)
        if set(a_types) <= _SCALAR_TYPES and set(b_types) <= _SCALAR_TYPES:
            # Object arrays preserve None and Python scalar types, unlike pandas'
            # nullable equality and numeric coercion. Type masks distinguish 1/True.
            equal = np.asarray((a_types == b_types) & (a == b), dtype=bool)
            signed = np.flatnonzero(equal & (a_types == float))
            if len(signed):
                equal[signed] &= np.signbit(a[signed].astype(float)) == np.signbit(
                    b[signed].astype(float)
                )
            mismatches = np.flatnonzero(~equal)
        else:
            mismatches = (
                i
                for i, (x, y) in enumerate(zip(a, b, strict=True))
                if not _strict_equal(x, y)
            )
        for index in mismatches:
            differences.append(
                f"Row {index}, column {name!r}: "
                f"{a[index]!r} ({type(a[index]).__name__}) != "
                f"{b[index]!r} ({type(b[index]).__name__})"
            )
            if len(differences) >= _MAX_DIFFERENCES:
                return differences
    return differences


def assert_frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """
    Assert exact equality of two canonical facet tables.

    Parameters
    ----------
    left, right : pandas.DataFrame
        Canonical tables to compare.

    Raises
    ------
    AssertionError
        If frames_equal reports differences, including its diagnostic messages.
    """
    differences = frames_equal(left, right)
    if differences:
        raise AssertionError("\n".join(differences))
