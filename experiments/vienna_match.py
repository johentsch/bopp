"""Read Vienna 4x22 match files into exact score and performance tables.

Positions remain fractions throughout. The Vienna MusicXML scores write
repeats out (no repeat signs; bars and anchors are already numbered as
unfolded performance order), and the match files follow the same numbering,
so score positions here are unfolded: there is no notion of folding a
repeated anchor onto an earlier pass. No bopp payloads are constructed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pandas as pd

_INTEGER = r"-?\d+"
_FRACTION = rf"{_INTEGER}(?:/\d+)?"
_DECIMAL = rf"{_INTEGER}(?:\.\d+)?"
_ANCHOR = re.compile(r"(n\d+)(?:-(\d+))?")
_INFO = re.compile(r"info\(([^,()]+),(.*)\)\.")
_SCOREPROP = re.compile(
    rf"scoreprop\(([^,()]+),([^,()]+),({_INTEGER}):({_INTEGER}),"
    rf"({_FRACTION}),({_DECIMAL})\)\."
)
_NOTE = re.compile(r"note\(([^,()]+)," + ",".join([rf"({_INTEGER})"] * 6) + r"\)")
_SNOTE = re.compile(
    rf"snote\((n\d+(?:-\d+)?),\[([A-G]),([n#b])\],({_INTEGER}),"
    rf"({_INTEGER}):({_INTEGER}),({_FRACTION}),({_FRACTION}),"
    rf"({_DECIMAL}),({_DECIMAL}),\[([^\[\]()]*)\]\)-(deletion|note\(.*\))\."
)
_INSERTION = re.compile(r"insertion-(note\(.*\))\.")
_PEDAL = re.compile(rf"(sustain|soft)\(({_INTEGER}),({_INTEGER})\)\.")
_TIME_SIGNATURE = re.compile(r"([1-9]\d*)/([1-9]\d*)")
_PART = re.compile(r'<part\s+id="[^"]*">(.*?)</part>', re.DOTALL)
_MEASURE = re.compile(r"<measure\b")

_SCOREPROP_COLUMNS = ("name", "value", "bar", "beat", "offset", "onset_in_beats")
_SNOTE_COLUMNS = (
    "anchor",
    "step",
    "alter",
    "octave",
    "bar",
    "beat",
    "offset",
    "duration",
    "onset_in_beats",
    "offset_in_beats",
    "attributes",
)
_NOTE_COLUMNS = ("note_id", "midi", "onset", "offset", "velocity", "channel", "track")
_ALIGNMENT_COLUMNS = ("kind", "anchor", "note_id")


@dataclass(frozen=True)
class MatchFile:
    """
    Parsed tables and metadata from one Vienna match file.

    Parameters
    ----------
    path : pathlib.Path
        Source file path.
    info : dict of str to str
        Metadata values as written, without numeric conversion.
    scoreprops : pandas.DataFrame
        Score properties with exact offsets and onsets.
    snotes : pandas.DataFrame
        Score note occurrences in file order, including deletions.
    notes : pandas.DataFrame
        Performed notes in file order, including insertions.
    alignment : pandas.DataFrame
        Match, deletion and insertion links in file order.
    sustain, soft : pandas.DataFrame
        Pedal ticks and values in file order.

    Notes
    -----
    Field assignment is frozen; the contained dictionary and tables are mutable.
    """

    path: Path
    info: dict[str, str]
    scoreprops: pd.DataFrame
    snotes: pd.DataFrame
    notes: pd.DataFrame
    alignment: pd.DataFrame
    sustain: pd.DataFrame
    soft: pd.DataFrame


def _note(text: str) -> tuple:
    """Parse the performed side of a match or insertion."""
    match = _NOTE.fullmatch(text)
    if match is None:
        raise ValueError("Invalid performed note")
    note_id, *numbers = match.groups()
    return (note_id, *(int(value) for value in numbers))


def _anchor(anchor: str) -> tuple[str, int]:
    """Split a score anchor into its base and numeric repetition index."""
    match = _ANCHOR.fullmatch(anchor)
    if match is None:
        raise ValueError(f"Invalid score anchor: {anchor!r}")
    base, repetition = match.groups()
    return base, int(repetition) if repetition is not None else 0


def _table(rows: list, columns: tuple[str, ...], integers: tuple[str, ...] = ()):
    """Build a table with stable integer types, including for empty tables."""
    # Object columns preserve literal None links under pandas' string inference.
    return pd.DataFrame(rows, columns=columns, dtype=object).astype(
        dict.fromkeys(integers, "int64")
    )


def parse_match(path: str | Path) -> MatchFile:
    """
    Parse every nonempty line of a Vienna v1.0.0 match file.

    Parameters
    ----------
    path : str or pathlib.Path
        UTF-8 match file to read.

    Returns
    -------
    MatchFile
        Metadata and tables in source order. Rational and decimal score
        positions are fractions; attributes are tuples of strings.

    Raises
    ------
    ValueError
        If a line is unknown or malformed, with its path and line number.
    """
    path = Path(path)
    info: dict[str, str] = {}
    scoreprops: list[tuple] = []
    snotes: list[tuple] = []
    notes: list[tuple] = []
    alignment: list[tuple] = []
    pedals: dict[str, list[tuple[int, int]]] = {"sustain": [], "soft": []}
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                if match := _INFO.fullmatch(line):
                    name, value = match.groups()
                    if name in info:
                        raise ValueError(f"Duplicate info key: {name!r}")
                    info[name] = value
                elif match := _SCOREPROP.fullmatch(line):
                    name, value, bar, beat, offset, onset = match.groups()
                    scoreprops.append(
                        (
                            name,
                            value,
                            int(bar),
                            int(beat),
                            Fraction(offset),
                            Fraction(onset),
                        )
                    )
                elif match := _SNOTE.fullmatch(line):
                    (
                        anchor,
                        step,
                        alter,
                        octave,
                        bar,
                        beat,
                        offset,
                        duration,
                        onset,
                        end,
                        attributes,
                        performed,
                    ) = match.groups()
                    attrs = tuple(attributes.split(",")) if attributes else ()
                    if any(not attr or attr.strip() != attr for attr in attrs):
                        raise ValueError("Invalid score attributes")
                    snotes.append(
                        (
                            anchor,
                            step,
                            alter,
                            int(octave),
                            int(bar),
                            int(beat),
                            Fraction(offset),
                            Fraction(duration),
                            Fraction(onset),
                            Fraction(end),
                            attrs,
                        )
                    )
                    if performed == "deletion":
                        alignment.append(("deletion", anchor, None))
                    else:
                        note = _note(performed)
                        notes.append(note)
                        alignment.append(("match", anchor, note[0]))
                elif match := _INSERTION.fullmatch(line):
                    note = _note(match[1])
                    notes.append(note)
                    alignment.append(("insertion", None, note[0]))
                elif match := _PEDAL.fullmatch(line):
                    kind, tick, value = match.groups()
                    pedals[kind].append((int(tick), int(value)))
                else:
                    raise ValueError("Unknown or malformed fact")
            except (ValueError, ZeroDivisionError) as err:
                raise ValueError(f"{path}:{line_number}: {err}: {line}") from err

    return MatchFile(
        path=path,
        info=info,
        scoreprops=_table(scoreprops, _SCOREPROP_COLUMNS, ("bar", "beat")),
        snotes=_table(snotes, _SNOTE_COLUMNS, ("octave", "bar", "beat")),
        notes=_table(notes, _NOTE_COLUMNS, _NOTE_COLUMNS[1:]),
        alignment=_table(alignment, _ALIGNMENT_COLUMNS),
        sustain=_table(pedals["sustain"], ("tick", "value"), ("tick", "value")),
        soft=_table(pedals["soft"], ("tick", "value"), ("tick", "value")),
    )


def score_quarters(mf: MatchFile) -> pd.DataFrame:
    """
    Add quarter positions and quarter durations to the score table.

    Parameters
    ----------
    mf : MatchFile
        Parsed file with exactly one time signature. Beats count quarters,
        while onset_in_beats counts units of the time signature denominator.
        Bars and anchors are already unfolded (repeats written out), so no
        folding is performed.

    Returns
    -------
    pandas.DataFrame
        A copy of the score table with Fraction columns quarter (position
        from the start of the piece, bar 0 being an anacrusis) and
        quarter_duration.

    Raises
    ------
    ValueError
        If the time signature is missing, repeated or invalid, or any
        computed position disagrees with onset_in_beats.
    """
    signatures = mf.scoreprops.loc[mf.scoreprops["name"] == "timeSignature"]
    if len(signatures) != 1:
        raise ValueError(f"{mf.path}: Expected exactly one timeSignature")
    signature = signatures.iloc[0]
    match = _TIME_SIGNATURE.fullmatch(signature["value"])
    if match is None:
        raise ValueError(f"{mf.path}: Invalid timeSignature {signature['value']!r}")
    num, den = map(int, match.groups())
    pickup = signature["onset_in_beats"] / den
    quarters: list[Fraction] = []
    for row in mf.snotes.itertuples(index=False):
        start = pickup if row.bar == 0 else Fraction((row.bar - 1) * num, den)
        quarter = 4 * (start + Fraction(row.beat - 1, 4) + row.offset)
        if quarter != 4 * row.onset_in_beats / den:
            raise ValueError(
                f"{mf.path}: {row.anchor}: quarter disagrees with onset_in_beats"
            )
        quarters.append(quarter)
    result = mf.snotes.copy()
    result["quarter"] = pd.Series(quarters, index=result.index, dtype=object)
    result["quarter_duration"] = result["duration"].map(lambda duration: 4 * duration)
    return result


def _comparable_score(quarters: pd.DataFrame) -> pd.DataFrame:
    """Score quarters with voice_overlap removed, for cross-performance diffing."""
    result = quarters.copy()
    result["attributes"] = result["attributes"].map(
        lambda attrs: tuple(attr for attr in attrs if attr != "voice_overlap")
    )
    return result.sort_values("anchor", kind="stable").reset_index(drop=True)


def score_notes(match_dir: str | Path, piece: str) -> pd.DataFrame:
    """
    Verify all performances and return the full score side of p01.

    Parameters
    ----------
    match_dir : str or pathlib.Path
        Directory holding the piece's match files.
    piece : str
        Piece filename prefix, without the performer or extension.

    Returns
    -------
    pandas.DataFrame
        Score quarters from p01, one row per snote (no deduplication of
        repeated passes), with p01's attributes kept as-is, sorted by
        (quarter, anchor number, repetition index).

    Raises
    ------
    ValueError
        If any performance's score side differs from p01 once voice_overlap
        is disregarded, naming the first differing file in filename order,
        or quarters are invalid.
    FileNotFoundError
        If the reference p01 file is absent.

    Notes
    -----
    Across the 22 Mozart performances, voice_overlap is always carried by
    `n238-1`. Its second-pass partner carries voice_overlap as `n238-2` in
    20 performances, and as `n239-2` in p05 and p07; the other of the pair
    is matched to a performed note in that file instead of deleted. p01
    follows the 20-file pattern (`n238-2`), which is why its attributes are
    returned unchanged rather than stripped.
    """
    match_dir = Path(match_dir)
    reference_path = match_dir / f"{piece}_p01.match"
    reference = score_quarters(parse_match(reference_path))
    reference_comparable = _comparable_score(reference)
    for path in sorted(match_dir.glob(f"{piece}_p*.match")):
        if path == reference_path:
            continue
        candidate = _comparable_score(score_quarters(parse_match(path)))
        if not reference_comparable.equals(candidate):
            raise ValueError(f"{path}: Score side differs from {reference_path.name}")

    bases = [_anchor(anchor) for anchor in reference["anchor"]]
    result = reference.assign(
        _anchor_number=[int(base[1:]) for base, _ in bases],
        _repetition=[repetition for _, repetition in bases],
    )
    return (
        result.sort_values(
            ["quarter", "_anchor_number", "_repetition"], kind="stable"
        )
        .drop(columns=["_anchor_number", "_repetition"])
        .reset_index(drop=True)
    )


def performance_notes(mf: MatchFile) -> pd.DataFrame:
    """
    Add the matched score anchor to every performed note.

    Parameters
    ----------
    mf : MatchFile
        Parsed match file.

    Returns
    -------
    pandas.DataFrame
        A copy of the performance table in file order, with match containing
        the full score anchor (e.g. 'n1-2') or 'NOMATCH' for insertions.

    Raises
    ------
    ValueError
        If note identifiers are duplicated or alignment links are incomplete.
    """
    links = mf.alignment.loc[mf.alignment["kind"] != "deletion"]
    if links["note_id"].duplicated().any() or mf.notes["note_id"].duplicated().any():
        raise ValueError(f"{mf.path}: Duplicate performed note identifiers")
    matches = {
        row.note_id: row.anchor if row.kind == "match" else "NOMATCH"
        for row in links.itertuples(index=False)
    }
    if set(matches) != set(mf.notes["note_id"]):
        raise ValueError(f"{mf.path}: Incomplete performance alignment")
    result = mf.notes.copy()
    result["match"] = result["note_id"].map(matches)
    return result


def score_measure_count(path: str | Path) -> int:
    """
    Count the measures in the first part of a MusicXML score.

    Parameters
    ----------
    path : str or pathlib.Path
        MusicXML file.

    Returns
    -------
    int
        Number of `<measure` elements inside the first `<part id="...">`
        element.

    Raises
    ------
    ValueError
        If no part element is found.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    match = _PART.search(text)
    if match is None:
        raise ValueError(f"{path}: No part element found")
    return len(_MEASURE.findall(match[1]))
