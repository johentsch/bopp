"""Read BPS-FH sonatas into score_note, music21_chord and segment_open annotations.

BPS-FH (Chen & Su 2018) stores 32 Beethoven sonata first movements as
``<n>/notes.csv`` (onset, MIDI, morphetic pitch, duration, staff from 0,
measure with -1 for an incomplete measure; no header), ``chords.xlsx`` (onset,
offset, key, degree, quality, inversion, Roman numeral) and ``phrases.xlsx``
(onset, offset, section, subsection, letter). Positions are floats in quarter
notes, so every extent is ``quarters_interval.float``; the anacrusis starts at
-1.0.

Conversions:

- Morphetic pitch to TPC: step = morphetic - 60 (C4), letter =
  ``CDEFGAB[step mod 7]``, accidental = MIDI - natural MIDI of that letter and
  octave, tpc = fifths(letter) + 7 * accidental. ``name`` and ``octave`` follow.
- Staff numbers become 1-based.
- Keys spell sharps with ``+`` in BPS-FH and are translated to ``#``.
- A phrase label is ``section/subsection/letter`` (``exp/1st/a``).

Source errors are kept, not dropped, and reported through optional lists:
nine phrases and two chords end before they start (``read_phrases`` and
``read_chords`` swap onset and offset, so that ``bopp.create``, which does not
check value constraints, cannot produce an annotation that typed decoding
rejects),
and two notes have a morphetic pitch more than a double accidental away from
their MIDI pitch (14/row 1129: MIDI 64, morphetic 42; 17/row 3190: MIDI 66,
morphetic 53), which ``read_notes`` spells from MIDI alone (sharps). Four
Roman numerals do not parse (``RN_CORRECTIONS``); ``read_chords`` replaces them
with the numeral their degree, quality and inversion columns describe.

The chords' degree, quality and inversion columns are not stored: the payload
keeps the Roman numeral and key, which determine them except for 39 of the
1,185 distinct key/numeral pairs (a known loss of ``music21_chord``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import bopp
from bopp.models.v1.annotation import Annotation

MEDIA_PREFIX = "bpsfh"
CORPUS = "sonatas"
_LETTERS = "CDEFGAB"
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_FIFTHS = {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}
_ACCIDENTALS = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
#: Roman numerals that do not parse, corrected from the same row's degree,
#: quality and inversion columns (16/rows 65, 168: 5/3 D7 root position;
#: 21/row 522: 7/5 d7 first inversion; 26/row 110: 5/4 D7 root position).
RN_CORRECTIONS = {"Vi/III": "V7/III", "vii-65V": "vii-65/V", "V7IV": "V7/IV"}
_SHARP_SPELLING = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def media_id(piece: str) -> str:
    """Media identifier of a sonata, e.g. ``bpsfh:sonatas/01``."""
    return f"{MEDIA_PREFIX}:{CORPUS}/{piece}"


def pieces(root: str | Path) -> list[str]:
    """Sorted two-digit piece directories of a BPS_FH_Dataset checkout."""
    return sorted(p.name for p in Path(root).iterdir() if (p / "notes.csv").is_file())


def _spelling(midi: int, morphetic: int | None) -> tuple[int, str, int] | None:
    """TPC, name and octave; None if the morphetic pitch cannot spell `midi`."""
    if morphetic is None:
        name = _SHARP_SPELLING[midi % 12]
        letter, accidental, octave = name[0], len(name) - 1, midi // 12 - 1
    else:
        step = morphetic - 60
        letter = _LETTERS[step % 7]
        octave = 4 + step // 7
        accidental = midi - (12 * (octave + 1) + _NATURAL[letter])
        if accidental not in _ACCIDENTALS:
            return None
    tpc = _FIFTHS[letter] + 7 * accidental
    return tpc, f"{letter}{_ACCIDENTALS[accidental]}{octave}", octave


def read_notes(
    root: str | Path, piece: str, *, respelled: list[tuple[str, int]] | None = None
) -> Annotation:
    """
    Read one sonata's notes as a score_note annotation.

    Parameters
    ----------
    root : str or pathlib.Path
        BPS_FH_Dataset directory.
    piece : str
        Two-digit piece directory.
    respelled : list or None, optional
        Receives ``(piece, row)`` for every note whose morphetic pitch cannot
        spell its MIDI pitch; such notes are spelled from MIDI alone.

    Returns
    -------
    Annotation
        Score notes with ``quarters_interval.float`` extents.
    """
    table = pd.read_csv(Path(root) / piece / "notes.csv", header=None)
    table.columns = ["onset", "midi", "morphetic", "duration", "staff", "measure"]
    spelled = []
    for row, (m, p) in enumerate(zip(table["midi"], table["morphetic"], strict=True)):
        spelling = _spelling(int(m), int(p))
        if spelling is None:
            if respelled is not None:
                respelled.append((piece, row))
            spelling = _spelling(int(m), None)
        spelled.append(spelling)
    return bopp.create(
        media_id=media_id(piece),
        payload_kind="score_note",
        extent_kind="quarters_interval.float",
        quarter=[float(v) for v in table["onset"]],
        duration=[float(v) for v in table["duration"]],
        midi=[int(v) for v in table["midi"]],
        tpc=[s[0] for s in spelled],
        name=[s[1] for s in spelled],
        octave=[s[2] for s in spelled],
        staff=[int(v) + 1 for v in table["staff"]],
        measure=[int(v) for v in table["measure"]],
    )


def _key(value: str) -> str:
    return value.replace("+", "#")


def _intervals(
    table: pd.DataFrame, piece: str, inverted: list[tuple[str, int]] | None
) -> tuple[list[float], list[float]]:
    """Quarters and durations; intervals that end before they start are swapped."""
    quarters, durations = [], []
    for row, (onset, offset) in enumerate(zip(table["onset"], table["offset"])):
        onset, offset = float(onset), float(offset)
        if offset < onset:
            if inverted is not None:
                inverted.append((piece, row))
            onset, offset = offset, onset
        quarters.append(onset)
        durations.append(offset - onset)
    return quarters, durations


def read_chords(
    root: str | Path, piece: str, *, inverted: list[tuple[str, int]] | None = None
) -> Annotation:
    """
    Read one sonata's chords as a music21_chord annotation.

    Parameters
    ----------
    root : str or pathlib.Path
        BPS_FH_Dataset directory.
    piece : str
        Two-digit piece directory.
    inverted : list or None, optional
        Receives ``(piece, row)`` for every chord whose offset precedes its
        onset; such chords are stored with onset and offset swapped.

    Returns
    -------
    Annotation
        Roman numerals and keys with ``quarters_interval.float`` extents.
    """
    table = pd.read_excel(Path(root) / piece / "chords.xlsx", header=None)
    table.columns = ["onset", "offset", "key", "degree", "quality", "inversion", "rn"]
    quarters, durations = _intervals(table, piece, inverted)
    return bopp.create(
        media_id=media_id(piece),
        payload_kind="music21_chord",
        extent_kind="quarters_interval.float",
        quarter=quarters,
        duration=durations,
        value=[RN_CORRECTIONS.get(str(v), str(v)) for v in table["rn"]],
        key=[_key(str(v)) for v in table["key"]],
    )


def read_phrases(
    root: str | Path, piece: str, *, inverted: list[tuple[str, int]] | None = None
) -> Annotation:
    """
    Read one sonata's phrases as a segment_open annotation.

    Parameters
    ----------
    root : str or pathlib.Path
        BPS_FH_Dataset directory.
    piece : str
        Two-digit piece directory.
    inverted : list or None, optional
        Receives ``(piece, row)`` for every phrase whose offset precedes its
        onset; such phrases are stored with onset and offset swapped.

    Returns
    -------
    Annotation
        Phrases with labels ``section/subsection/letter``.
    """
    table = pd.read_excel(Path(root) / piece / "phrases.xlsx", header=None)
    table.columns = ["onset", "offset", "section", "subsection", "letter"]
    quarters, durations = _intervals(table, piece, inverted)
    return bopp.create(
        media_id=media_id(piece),
        payload_kind="segment_open",
        extent_kind="quarters_interval.float",
        quarter=quarters,
        duration=durations,
        value=[
            f"{a}/{b}/{c}"
            for a, b, c in zip(table["section"], table["subsection"], table["letter"])
        ],
    )
