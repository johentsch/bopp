"""Read DLC chords TSVs into columnar control events or one mixed ext annotation.

This reader uses only bopp's generated payload definitions, independently of the
score extension plugin. Empty strings represent missing cells. Non-empty source
columns must be represented by the event's payload, except documented extent and
source bookkeeping columns, so unsupported DLC features fail without data loss.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
from msgspec import inspect as mi

import bopp
from bopp.exceptions import BoppArgumentError
from bopp.models.v1.annotation import Annotation
from bopp.registries.v1 import PAYLOAD_TYPE_REGISTRY

MIXED_EXT_SCHEMA = "io.github.johentsch.score_control_event:v1"
_EVENT_TAGS = {
    "chord": "Chord",
    "dynamic": "Dynamic",
    "spanner": "Spanner",
    "figured_bass": "FiguredBass",
    "staff_text": "StaffText",
    "system_text": "SystemText",
    "tempo": "Tempo",
}
_IGNORED = frozenset(
    {
        "corpus",
        "piece",
        "i",
        "event",
        "quarterbeats",
        "quarterbeats_all_endings",
        "duration_qb",
        "duration",
        "mc_onset",
        "mn_onset",
        "timesig",
    }
)
_INTS = frozenset({"staff", "voice", "mc", "mn", "volta", "chord_id"})
_FRACTIONS = frozenset({"nominal_duration", "scalar", "thoroughbass_duration"})
_FLOATS = frozenset({"qpm", "metronome_number"})


def _column_info(kind: str) -> tuple[tuple[str, bool, bool], ...]:
    info = mi.type_info(PAYLOAD_TYPE_REGISTRY[f"score_control_event.{kind}"])
    assert isinstance(info, mi.StructType)
    result = []
    for field in info.fields:
        column = field.type
        while isinstance(column, mi.Metadata):
            column = column.type
        assert isinstance(column, mi.ListType)
        item = column.item_type
        while isinstance(item, mi.Metadata):
            item = item.type
        nullable = isinstance(item, mi.UnionType) and any(
            isinstance(t, mi.NoneType) for t in item.types
        )
        result.append((field.name, field.required, nullable))
    return tuple(result)


_COLUMNS = {kind: _column_info(kind) for kind in _EVENT_TAGS}


def read_chords_tsv(path: str | Path) -> pd.DataFrame:
    """Read an all-string frame, preserving empty cells as empty strings."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _cell(name: str, value: str) -> Any:
    if name in _INTS:
        return int(value)
    if name in _FRACTIONS:
        return _pair(Fraction(value))
    if name in _FLOATS:
        return float(value)
    if name == "tempo_visible":
        if value in {"1.0", "1", "True"}:
            return True
        if value in {"0.0", "0", "False"}:
            return False
        raise BoppArgumentError(f"Invalid tempo_visible value: {value!r}")
    return value


def _extent(df: pd.DataFrame) -> dict[str, Any]:
    quarters: list[list[int]] = []
    durations: list[list[int]] = []
    for row in df.to_dict("records"):
        quarter_column = (
            "quarterbeats_all_endings"
            if row.get("quarterbeats_all_endings", "")
            else "quarterbeats"
        )
        for name, multiplier, values in (
            (quarter_column, 1, quarters),
            ("duration", 4, durations),
        ):
            try:
                values.append(_pair(Fraction(row[name]) * multiplier))
            except (KeyError, ValueError, TypeError, ZeroDivisionError) as err:
                raise BoppArgumentError(
                    f"Column '{name}' is missing or invalid for event '{row['event']}'"
                ) from err
    return {
        "extent_kind": "quarters_interval.fraction",
        "quarter": quarters,
        "duration": durations,
    }


def _prepare(
    df: pd.DataFrame,
    media_id: str | None,
) -> tuple[str, dict[str, tuple[pd.DataFrame, dict[str, Any]]]]:
    if (
        not {"corpus", "piece"}.issubset(df.columns)
        or len(df[["corpus", "piece"]].drop_duplicates()) != 1
    ):
        raise BoppArgumentError("Expected a single corpus/piece pair")
    if media_id is None:
        media_id = f"dlc:{df.iloc[0]['corpus']}/{df.iloc[0]['piece']}"
    if "event" not in df:
        raise BoppArgumentError("Missing event column")
    for event in df["event"].unique():
        if event not in _EVENT_TAGS.values():
            raise BoppArgumentError(f"Unknown event value: {event!r}")

    groups = {}
    for kind, event in _EVENT_TAGS.items():
        subset = df[df["event"] == event]
        if subset.empty:
            continue
        allowed = {name for name, _, _ in _COLUMNS[kind]}
        for name in subset.columns:
            if name not in allowed | _IGNORED and (subset[name] != "").any():
                raise BoppArgumentError(
                    f"Column '{name}' is unsupported for event '{event}'"
                )
        columns: dict[str, Any] = {}
        for name, required, nullable in _COLUMNS[kind]:
            cells = subset[name].tolist() if name in subset else [""] * len(subset)
            if not required and all(value == "" for value in cells):
                continue
            if (required or not nullable) and any(value == "" for value in cells):
                raise BoppArgumentError(
                    f"Column '{name}' has missing values for event '{event}'"
                )
            try:
                columns[name] = [
                    _cell(name, value) if value != "" else None for value in cells
                ]
            except (ValueError, TypeError, ZeroDivisionError) as err:
                raise BoppArgumentError(
                    f"Column '{name}' has invalid values for event '{event}'"
                ) from err
        groups[kind] = (subset, columns)
    return media_id, groups


def chords_to_control_events(
    df: pd.DataFrame,
    *,
    media_id: str | None = None,
) -> dict[str, Annotation]:
    """Build one annotation per present event kind, retaining TSV order within it."""
    media_id, groups = _prepare(df, media_id)
    return {
        kind: bopp.create(
            media_id=media_id,
            payload_kind=f"score_control_event.{kind}",
            **_extent(subset),
            **columns,
        )
        for kind, (subset, columns) in groups.items()
    }


def chords_to_mixed_ext(df: pd.DataFrame, *, media_id: str | None = None) -> Annotation:
    """Build tagged, null-free rows in original TSV order without plugin resolution."""
    media_id, groups = _prepare(df, media_id)
    per_event = {}
    for kind, (_, columns) in groups.items():
        names = tuple(columns)
        per_event[_EVENT_TAGS[kind]] = iter(
            [
                {
                    name: value
                    for name, value in zip(names, values, strict=True)
                    if value is not None
                }
                for values in zip(*columns.values(), strict=True)
            ]
        )
    rows = [{"event": event, **next(per_event[event])} for event in df["event"]]
    return bopp.create(
        media_id=media_id,
        payload_kind="ext",
        ext_schema=MIXED_EXT_SCHEMA,
        value=rows,
        resolve_ext=False,
        **_extent(df),
    )
