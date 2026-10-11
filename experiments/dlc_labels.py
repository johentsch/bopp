"""Read DLC harmony TSVs into ``dcml_label`` annotations.

The payload stores the label verbatim. All other harmony columns are positional
(mc, mn, onsets, timesig, staff, voice, volta, quarterbeats) or derived by ms3's
label parser (globalkey, localkey, numeral, form, figbass, changes, relativeroot,
cadence, phraseend, chord_type, chord_tones, and related fields), and are
intentionally not stored in the payload. Extents use all-ending quarterbeats
with a per-cell fallback to quarterbeats. Float duration_qb values are recovered
as bounded-denominator fractions only when the residual is below 1e-6.
Empty duration_qb cells in alternate endings use the difference to the next
label's exact start in TSV order. A final empty cell has no next start and needs
the caller's ``piece_end``, which the measure table provides; without it, and for
a backwards interval, an error is raised: the harmony table alone cannot
establish that interval.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pandas as pd

import bopp
from bopp.exceptions import BoppArgumentError
from bopp.models.v1.annotation import Annotation


def read_harmonies_tsv(path: str | Path) -> pd.DataFrame:
    """
    Read a DLC harmony TSV as an all-string DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the tab-separated harmony file.

    Returns
    -------
    pandas.DataFrame
        One row per label, preserving empty cells as empty strings. Native DLC
        files need corpus and piece columns inserted by the caller.
    """
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def harmonies_to_dcml_label(
    df: pd.DataFrame,
    *,
    media_id: str | None = None,
    max_denominator: int = 1000,
    piece_end: Fraction | None = None,
) -> Annotation:
    """
    Convert one DLC harmony table into a ``dcml_label`` annotation.

    Parameters
    ----------
    df : pandas.DataFrame
        All-string harmony table covering exactly one corpus/piece pair.
    media_id : str or None, optional
        Media identifier, defaulting to ``f"dlc:{corpus}/{piece}"``.
    max_denominator : int, optional
        Maximum denominator when recovering rational durations, default 1000.
        Start positions are parsed exactly, without this approximation.
    piece_end : fractions.Fraction or None, optional
        Exact quarterbeat where the piece ends, as derived from its measure
        table. Used only to close a final row whose duration_qb is empty, which
        happens in the last alternate ending of a piece.

    Returns
    -------
    Annotation
        Verbatim labels with a ``quarters_interval.fraction`` extent.

    Raises
    ------
    BoppArgumentError
        If the frame does not identify one corpus/piece pair, a required column
        is absent or invalid, max_denominator is not a positive integer,
        piece_end is neither a Fraction nor None, or a recovered duration
        differs from duration_qb by at least 1e-6. Empty durations require a
        following label, or piece_end for a final row, at the same or a later
        exact start.
    """
    if (
        not {"corpus", "piece"}.issubset(df.columns)
        or len(df[["corpus", "piece"]].drop_duplicates()) != 1
    ):
        raise BoppArgumentError("Expected a single corpus/piece pair")
    if type(max_denominator) is not int or max_denominator < 1:
        raise BoppArgumentError("max_denominator must be a positive integer")
    if piece_end is not None and not isinstance(piece_end, Fraction):
        raise BoppArgumentError("piece_end must be an exact Fraction or None")
    if media_id is None:
        media_id = f"dlc:{df.iloc[0]['corpus']}/{df.iloc[0]['piece']}"
    for name in ("label", "duration_qb"):
        if name not in df:
            raise BoppArgumentError(f"Column '{name}' is missing")
    if not all(isinstance(value, str) for value in df["label"]):
        raise BoppArgumentError("Column 'label' must contain strings")

    quarters = []
    for row in df.to_dict("records"):
        name = (
            "quarterbeats_all_endings"
            if row.get("quarterbeats_all_endings", "")
            else "quarterbeats"
        )
        try:
            quarter = Fraction(row[name])
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as err:
            raise BoppArgumentError(f"Column '{name}' is missing or invalid") from err
        quarters.append(quarter)

    durations = []
    for index, source in enumerate(df["duration_qb"]):
        if source == "":
            if index + 1 == len(quarters):
                if piece_end is None:
                    raise BoppArgumentError(
                        f"Column 'duration_qb' at row {index} is empty with no next "
                        "start and no piece_end"
                    )
                end = piece_end
            else:
                end = quarters[index + 1]
            duration = end - quarters[index]
            if duration < 0:
                raise BoppArgumentError(
                    f"Column 'duration_qb' at row {index} is empty with a backwards "
                    "next start"
                )
        else:
            try:
                duration = Fraction(source).limit_denominator(max_denominator)
                residual = abs(float(duration) - float(source))
            except (ValueError, TypeError, ZeroDivisionError, OverflowError) as err:
                raise BoppArgumentError("Column 'duration_qb' has invalid values") from err
            if residual >= 1e-6:
                raise BoppArgumentError(
                    f"Column 'duration_qb' at row {index} has residual {residual!r} >= 1e-6"
                )
        durations.append([duration.numerator, duration.denominator])

    return bopp.create(
        media_id=media_id,
        payload_kind="dcml_label",
        extent_kind="quarters_interval.fraction",
        quarter=[[q.numerator, q.denominator] for q in quarters],
        duration=durations,
        value=df["label"].tolist(),
    )
