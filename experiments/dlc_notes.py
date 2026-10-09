"""Read DLC-style symbolic note TSVs into ``score_note`` annotations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

import bopp
from bopp.base import BoppBase
from bopp.exceptions import BoppArgumentError

#: TSV columns that map to a scalar score_note column, with their cast functions.
_SCALAR_COLUMNS: dict[str, Callable[[str], Any]] = {
    "midi": int,
    "tpc": int,
    "octave": int,
    "staff": int,
    "voice": int,
    "mc": int,
    "mn": int,
    "chord_id": int,
    "name": str,
    "gracenote": str,
    "tremolo": str,
    "tied": int,
    "volta": int,
    "tuning": float,
}

#: TSV columns that map to a [numerator, denominator] score_note column.
_FRACTION_COLUMNS = ("nominal_duration", "scalar")

#: Columns whose score_note items allow null; missing cells become None.
_NULLABLE_COLUMNS = frozenset({"tied", "gracenote", "tremolo", "volta", "tuning"})


def read_notes_tsv(path: str | Path) -> pd.DataFrame:
    """
    Read a DLC-style note TSV as an all-string DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the tab-separated note file.

    Returns
    -------
    pandas.DataFrame
        One row per note, all columns as strings; empty string means a
        missing value.
    """
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _fraction_pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _column_values(
    df: pd.DataFrame, name: str, convert: Callable[[str], Any]
) -> list[Any] | None:
    """Build a score_note column from a TSV column, or None if it is absent."""
    cells = df[name]
    non_empty = cells != ""
    if not non_empty.any():
        return None
    if name in _NULLABLE_COLUMNS:
        return [convert(v) if v != "" else None for v in cells]
    if not non_empty.all():
        raise BoppArgumentError(f"Column '{name}' has missing values for some rows")
    return [convert(v) for v in cells]


def notes_to_score_note(df: pd.DataFrame, *, media_id: str | None = None) -> BoppBase:
    """
    Convert a DLC-style note DataFrame into a ``score_note`` annotation.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame as returned by `read_notes_tsv`, covering a single
        ``corpus``/``piece`` pair.
    media_id : str or None, optional
        Media identifier for the annotation. If None, defaults to
        ``f"dlc:{corpus}/{piece}"``.

    Returns
    -------
    BoppBase
        An Annotation instance with a ``score_note`` payload and a
        ``quarters_interval.fraction`` extent.

    Raises
    ------
    BoppArgumentError
        If `df` does not cover exactly one ``corpus``/``piece`` pair, or if a
        non-nullable score_note column has missing cells for some rows.
    """
    corpora = df["corpus"].unique()
    pieces = df["piece"].unique()
    if len(corpora) != 1 or len(pieces) != 1:
        raise BoppArgumentError(
            "notes_to_score_note requires a single corpus/piece pair in the frame"
        )
    corpus, piece = corpora[0], pieces[0]
    if media_id is None:
        media_id = f"dlc:{corpus}/{piece}"

    quarter = [
        _fraction_pair(Fraction(qb_all if qb_all else qb))
        for qb, qb_all in zip(df["quarterbeats"], df["quarterbeats_all_endings"], strict=True)
    ]
    duration = [_fraction_pair(Fraction(d) * 4) for d in df["duration"]]

    columns: dict[str, Any] = {}
    for name, convert in _SCALAR_COLUMNS.items():
        values = _column_values(df, name, convert)
        if values is not None:
            columns[name] = values
    for name in _FRACTION_COLUMNS:
        values = _column_values(df, name, lambda v: _fraction_pair(Fraction(v)))
        if values is not None:
            columns[name] = values

    return bopp.create(
        media_id=media_id,
        payload_kind="score_note",
        extent_kind="quarters_interval.fraction",
        quarter=quarter,
        duration=duration,
        **columns,
    )
