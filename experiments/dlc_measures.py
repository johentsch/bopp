"""Read DLC measure TSVs into ``score_measure`` annotations.

Measure units retain encoded score order, including all volta endings. Every
TSV column except quarterbeats, quarterbeats_all_endings and duration_qb is
reconstructed exactly from the payload. The extent start is
quarterbeats_all_endings (falling back to quarterbeats), and its duration is
4 * act_dur. The unfolded quarterbeats column (which differs from
quarterbeats_all_endings after a volta) and duration_qb (equal to 4 * act_dur
in all DLC rows) are not stored, by design.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

import bopp
from bopp.base import BoppBase
from bopp.exceptions import BoppArgumentError

_REQUIRED = ("mc", "mn", "keysig", "timesig", "act_dur", "mc_offset", "next")
_NULLABLE = (
    "numbering_offset",
    "volta",
    "dont_count",
    "barline",
    "breaks",
    "repeats",
    "markers",
    "jump_bwd",
    "jump_fwd",
    "play_until",
)
_INTS = frozenset({"mc", "mn", "keysig", "numbering_offset", "volta"})
_FRACTIONS = frozenset({"act_dur", "mc_offset"})
_IGNORED = frozenset(
    {"corpus", "piece", "quarterbeats", "quarterbeats_all_endings", "duration_qb"}
)


def read_measures_tsv(path: str | Path) -> pd.DataFrame:
    """
    Read a DLC measure TSV as an all-string DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the tab-separated measure file.

    Returns
    -------
    pandas.DataFrame
        One row per measure unit, with empty cells preserved as empty strings.
        Native DLC files need corpus and piece columns inserted by the caller.
    """
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _cell(name: str, value: str) -> Any:
    if name in _INTS:
        return int(value)
    if name in _FRACTIONS:
        return _pair(Fraction(value))
    if name == "next":
        return [int(mc) for mc in value.split(",")]
    if name == "dont_count":
        if value != "1":
            raise BoppArgumentError(f"Column 'dont_count' has invalid value: {value!r}")
        return True
    return value


def measures_to_score_measure(
    df: pd.DataFrame, *, media_id: str | None = None
) -> BoppBase:
    """
    Convert one DLC measure table into a ``score_measure`` annotation.

    Parameters
    ----------
    df : pandas.DataFrame
        All-string measure table covering exactly one corpus/piece pair.
        Nullable columns may be absent; entirely empty ones are omitted.
    media_id : str or None, optional
        Media identifier, defaulting to ``f"dlc:{corpus}/{piece}"``.

    Returns
    -------
    BoppBase
        Annotation with parallel measure columns and a
        ``quarters_interval.fraction`` extent. Starts use
        quarterbeats_all_endings, falling back to quarterbeats per cell;
        durations are four times act_dur. Fraction pairs are reduced [n, d] lists.

    Raises
    ------
    BoppArgumentError
        If corpus/piece does not identify a single pair, neither quarterbeats
        nor quarterbeats_all_endings is present (required for the extent),
        a required column has missing cells, a cell cannot be converted, or a non-empty source
        column cannot be represented without data loss.
    """
    if (
        not {"corpus", "piece"}.issubset(df.columns)
        or len(df[["corpus", "piece"]].drop_duplicates()) != 1
    ):
        raise BoppArgumentError("Expected a single corpus/piece pair")
    if media_id is None:
        media_id = f"dlc:{df.iloc[0]['corpus']}/{df.iloc[0]['piece']}"

    if not {"quarterbeats", "quarterbeats_all_endings"}.intersection(df.columns):
        raise BoppArgumentError(
            "A quarterbeats or quarterbeats_all_endings column is required for the extent"
        )

    allowed = {*_REQUIRED, *_NULLABLE} | _IGNORED
    for name in df.columns:
        if name not in allowed and (df[name] != "").any():
            raise BoppArgumentError(f"Column '{name}' is unsupported")

    columns: dict[str, Any] = {}
    for name in (*_REQUIRED, *_NULLABLE):
        cells = df[name].tolist() if name in df else [""] * len(df)
        if name in _REQUIRED:
            if any(value == "" for value in cells):
                raise BoppArgumentError(f"Column '{name}' has missing values")
        elif all(value == "" for value in cells):
            continue
        try:
            columns[name] = [
                _cell(name, value) if value != "" else None for value in cells
            ]
        except (ValueError, TypeError, ZeroDivisionError) as err:
            raise BoppArgumentError(f"Column '{name}' has invalid values") from err

    quarters = []
    for row in df.to_dict("records"):
        name = (
            "quarterbeats_all_endings"
            if row.get("quarterbeats_all_endings", "")
            else "quarterbeats"
        )
        try:
            quarters.append(_pair(Fraction(row[name])))
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as err:
            raise BoppArgumentError(f"Column '{name}' is missing or invalid") from err
    durations = [_pair(Fraction(*value) * 4) for value in columns["act_dur"]]

    return bopp.create(
        media_id=media_id,
        payload_kind="score_measure",
        extent_kind="quarters_interval.fraction",
        quarter=quarters,
        duration=durations,
        **columns,
    )
