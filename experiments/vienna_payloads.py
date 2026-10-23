"""Turn parsed Vienna 4x22 match files into score_note, performance_note and pedal.

Identity: corpus ``vienna4x22``; the score side of a piece is the item
``<piece>`` (e.g. ``Chopin_op10_no3``), each performance the item
``<piece>_<performer>`` (``Chopin_op10_no3_p01``). Media ids are
``vienna:vienna4x22/<item>``.

- Score notes come from ``vienna_match.score_notes`` (p01, unfolded, verified
  against all 22 performances), with ``quarters_interval.fraction`` extents.
  ``anchor`` is the full match anchor (``n1-2``), the value performance notes
  link to in ``match``. The ``v<n>`` and ``staff<n>`` attributes become
  ``voice`` and ``staff``, ``grace`` becomes ``gracenote = "grace"``; every
  other attribute (``accent``, ``staccato``, ``voice_overlap``) has no column
  and is counted by ``dropped_attributes``.
- Performance notes keep the file order, with ``midi_interval`` extents in
  ticks (offset - onset as duration), ``anchor`` = the note id and ``match``
  = the score anchor or ``NOMATCH``.
- Pedal events (sustain and soft controller changes) are instants, so they
  use ``midi_ticks``, ordered by tick with sustain before soft at equal ticks.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

import bopp
from bopp.models.v1.annotation import Annotation
from experiments.vienna_match import parse_match, performance_notes, score_notes

MEDIA_PREFIX = "vienna"
CORPUS = "vienna4x22"
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_FIFTHS = {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}
_ALTER = {"n": 0, "#": 1, "b": -1}
_SIGN = {0: "", 1: "#", -1: "b"}
_LAYER = re.compile(r"(v|staff)(\d+)")


def media_id(item: str) -> str:
    """Media identifier of a score or performance item."""
    return f"{MEDIA_PREFIX}:{CORPUS}/{item}"


def pieces(match_dir: str | Path) -> list[str]:
    """Sorted piece names that have a p01 match file."""
    return sorted(
        p.name.removesuffix("_p01.match") for p in Path(match_dir).glob("*_p01.match")
    )


def _pair(value) -> list[int]:
    return [value.numerator, value.denominator]


def _layers(attributes: tuple[str, ...], counter: Counter | None) -> tuple:
    voice = staff = grace = None
    for attribute in attributes:
        match = _LAYER.fullmatch(attribute)
        if attribute == "grace":
            grace = "grace"
        elif match is None:
            if counter is not None:
                counter[attribute] += 1
        elif match[1] == "v":
            voice = int(match[2])
        else:
            staff = int(match[2])
    return voice, staff, grace


def score_note_annotation(
    match_dir: str | Path, piece: str, *, dropped_attributes: Counter | None = None
) -> Annotation:
    """
    Build the score_note annotation of one piece.

    Parameters
    ----------
    match_dir : str or pathlib.Path
        Directory with the piece's match files.
    piece : str
        Piece name without performer and extension.
    dropped_attributes : collections.Counter or None, optional
        Counts every attribute other than voice and staff, which no score_note
        column holds.

    Returns
    -------
    Annotation
        One row per score note occurrence, unfolded.
    """
    table = score_notes(match_dir, piece)
    layers = [_layers(attrs, dropped_attributes) for attrs in table["attributes"]]
    alters = [_ALTER[a] for a in table["alter"]]
    midi = [
        12 * (int(o) + 1) + _NATURAL[s] + a
        for s, a, o in zip(table["step"], alters, table["octave"], strict=True)
    ]
    columns = {
        "staff": [layer[1] for layer in layers],
        "voice": [layer[0] for layer in layers],
        "gracenote": [layer[2] for layer in layers],
    }
    return bopp.create(
        media_id=media_id(piece),
        payload_kind="score_note",
        extent_kind="quarters_interval.fraction",
        quarter=[_pair(q) for q in table["quarter"]],
        duration=[_pair(d) for d in table["quarter_duration"]],
        midi=midi,
        tpc=[_FIFTHS[s] + 7 * a for s, a in zip(table["step"], alters, strict=True)],
        name=[
            f"{s}{_SIGN[a]}{int(o)}"
            for s, a, o in zip(table["step"], alters, table["octave"], strict=True)
        ],
        octave=[int(o) for o in table["octave"]],
        measure=[int(b) for b in table["bar"]],
        anchor=[str(a) for a in table["anchor"]],
        **{k: v for k, v in columns.items() if any(x is not None for x in v)},
    )


def performance_annotations(path: str | Path) -> tuple[Annotation, Annotation]:
    """
    Build the performance_note and pedal annotations of one match file.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``<piece>_<performer>.match`` file.

    Returns
    -------
    tuple of Annotation
        Performance notes (``midi_interval``) and pedal events (``midi_ticks``).
    """
    mf = parse_match(path)
    item = Path(path).stem
    notes = performance_notes(mf)
    performance = bopp.create(
        media_id=media_id(item),
        payload_kind="performance_note",
        extent_kind="midi_interval",
        tick=[int(v) for v in notes["onset"]],
        duration=[
            int(off) - int(on) for on, off in zip(notes["onset"], notes["offset"])
        ],
        midi=[int(v) for v in notes["midi"]],
        velocity=[int(v) for v in notes["velocity"]],
        channel=[int(v) for v in notes["channel"]],
        track=[int(v) for v in notes["track"]],
        anchor=[str(v) for v in notes["note_id"]],
        match=[str(v) for v in notes["match"]],
    )
    events = pd.concat(
        [
            mf.sustain.assign(controller="sustain", order=0),
            mf.soft.assign(controller="soft", order=1),
        ],
        ignore_index=True,
    ).sort_values(["tick", "order"], kind="stable")
    pedal = bopp.create(
        media_id=media_id(item),
        payload_kind="pedal",
        extent_kind="midi_ticks",
        tick=[int(v) for v in events["tick"]],
        controller=list(events["controller"]),
        value=[int(v) for v in events["value"]],
    )
    return performance, pedal
