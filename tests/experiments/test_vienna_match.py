"""Validate Vienna match parsing, exact positions and score consistency."""

import re
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from experiments.vienna_match import (
    parse_match,
    performance_notes,
    score_measure_count,
    score_notes,
    score_quarters,
)

FIXTURE = Path(__file__).parent / "data/vienna_Mozart_K331_1st-mov_p05_excerpt.match"
PIECE = "Mozart_K331_1st-mov"


@pytest.fixture
def mf():
    """Parse the bundled, verbatim Mozart p05 excerpt."""
    return parse_match(FIXTURE)


def test_parse_fields(mf):
    """Preserve metadata, score fields, performed notes and pedal events."""
    assert mf.path == FIXTURE
    assert mf.info == {
        "matchFileVersion": "1.0.0",
        "piece": PIECE,
        "scoreFileName": f"{PIECE}.musicxml",
        "midiFileName": f"{PIECE}_p05.mid",
        "composer": "W. A. Mozart",
        "performer": "Pianist 05",
        "midiClockUnits": "480",
        "midiClockRate": "500000",
    }
    assert mf.scoreprops.to_dict("records") == [
        {
            "name": "keySignature",
            "value": "A",
            "bar": 1,
            "beat": 1,
            "offset": Fraction(0),
            "onset_in_beats": Fraction(0),
        },
        {
            "name": "timeSignature",
            "value": "6/8",
            "bar": 1,
            "beat": 1,
            "offset": Fraction(0),
            "onset_in_beats": Fraction(0),
        },
    ]
    assert mf.snotes.iloc[0].to_dict() == {
        "anchor": "n1-1",
        "step": "C",
        "alter": "#",
        "octave": 5,
        "bar": 1,
        "beat": 1,
        "offset": Fraction(0),
        "duration": Fraction(3, 16),
        "onset_in_beats": Fraction(0),
        "offset_in_beats": Fraction(3, 2),
        "attributes": ("v1", "staff1"),
    }
    assert mf.snotes.iloc[3]["offset"] == Fraction(3, 16)
    assert mf.snotes.iloc[-1]["attributes"] == ("v2", "staff1", "voice_overlap")
    assert mf.notes.iloc[0].to_dict() == {
        "note_id": "n0",
        "midi": 73,
        "onset": 1407,
        "offset": 2109,
        "velocity": 91,
        "channel": 0,
        "track": 0,
    }
    assert mf.notes.iloc[3].to_dict() == {
        "note_id": "n3",
        "midi": 64,
        "onset": 1543,
        "offset": 2073,
        "velocity": 1,
        "channel": 0,
        "track": 0,
    }
    assert mf.sustain.to_dict("list") == {"tick": [0] * 10, "value": [12, 11] * 5}
    assert mf.soft.to_dict("list") == {
        "tick": [0, 448, 2887, 5325, 7764, 8128, 8148, 8167, 8224, 8877],
        "value": [0, 0, 0, 0, 0, 2, 4, 5, 5, 4],
    }
    assert len(mf.snotes) == 44
    assert len(mf.notes) == 42
    assert len(mf.alignment) == 45
    with pytest.raises(FrozenInstanceError):
        mf.path = Path("another.match")


def test_fraction_and_integer_types(mf):
    """All positions are fractions and MIDI and pedal columns are integers."""
    for table, names in (
        (mf.snotes, ("offset", "duration", "onset_in_beats", "offset_in_beats")),
        (mf.scoreprops, ("offset", "onset_in_beats")),
    ):
        for name in names:
            assert all(isinstance(value, Fraction) for value in table[name])
    for table, names in (
        (mf.snotes, ("octave", "bar", "beat")),
        (mf.scoreprops, ("bar", "beat")),
        (mf.notes, ("midi", "onset", "offset", "velocity", "channel", "track")),
        (mf.sustain, ("tick", "value")),
        (mf.soft, ("tick", "value")),
    ):
        for name in names:
            assert pd.api.types.is_integer_dtype(table[name])
    assert all(isinstance(attrs, tuple) for attrs in mf.snotes["attributes"])


def test_alignment_and_performance_order(mf):
    """Keep all three link kinds and map performed notes by their IDs."""
    assert mf.alignment.iloc[:4].to_dict("records") == [
        {"kind": "match", "anchor": "n1-1", "note_id": "n0"},
        {"kind": "match", "anchor": "n6-1", "note_id": "n2"},
        {"kind": "match", "anchor": "n7-1", "note_id": "n1"},
        {"kind": "insertion", "anchor": None, "note_id": "n3"},
    ]
    deleted = mf.alignment.loc[mf.alignment["kind"] == "deletion"]
    assert deleted["anchor"].tolist() == ["n99-1", "n238-1", "n239-2"]
    assert all(note_id is None for note_id in deleted["note_id"])
    result = performance_notes(mf)
    pd.testing.assert_frame_equal(result.drop(columns="match"), mf.notes)
    assert result["match"].iloc[:5].tolist() == [
        "n1-1",
        "n6-1",
        "n7-1",
        "NOMATCH",
        "n2-1",
    ]
    assert result["match"].iloc[-1] == "n1-2"
    assert "match" not in mf.notes
    reversed_links = replace(mf, alignment=mf.alignment.iloc[::-1])
    pd.testing.assert_frame_equal(performance_notes(reversed_links), result)


@pytest.mark.parametrize(
    "line",
    [
        "unknown(1).",
        "sustain(0,1). trailing",
        "sustain(0,1.5).",
        "soft(0,0)",
        "insertion-note(n3,64,1,2,3,0).",
        "snote(n1,[C,n],4,1:1,0,1/0,0.0000,1.0000,[v1])-deletion.",
        "snote(n1,[C,n],4,1:1,0,1/4,0.0000,1.0000,[v1])-unknown.",
        "snote(n1,[C,n],4,1:1,0,1/4,0.0000,1.0000,[v1])-note(n1,60,0,1,x,0,0).",
    ],
)
def test_bad_line_names_source(tmp_path, line):
    """Never skip unknown syntax or lose location information on cast errors."""
    path = tmp_path / "bad.match"
    path.write_text(f"info(piece,test).\n\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        parse_match(path)
    assert f"{path}:3:" in str(error.value)


def test_quarters(mf):
    """Use quarter beats within 6/8 bars; positions stay unfolded."""
    result = score_quarters(mf)
    pd.testing.assert_frame_equal(result[mf.snotes.columns], mf.snotes)
    for row in result.itertuples(index=False):
        expected = (row.bar - 1) * 3 + row.beat - 1 + 4 * row.offset
        assert row.quarter == expected == row.onset_in_beats / 2
        assert row.quarter_duration == 4 * row.duration
        assert all(
            isinstance(value, Fraction) for value in (row.quarter, row.quarter_duration)
        )
    indexed = result.set_index("anchor")
    assert indexed.loc["n1-1", "quarter"] == 0
    assert indexed.loc["n1-2", "quarter"] == 24
    assert indexed.loc["n239-2", "quarter"] == Fraction(213, 2)
    assert indexed.loc["n2-1", "quarter"] == Fraction(3, 4)
    reversed_score = score_quarters(replace(mf, snotes=mf.snotes.iloc[::-1]))
    pd.testing.assert_frame_equal(reversed_score.iloc[::-1], result)
    assert "quarter" not in mf.snotes


@pytest.mark.parametrize(
    ("meter", "pickup", "expected"),
    [
        ("2/4", "-0.5000", Fraction(-1, 2)),
        ("6/8", "-4.0000", Fraction(-2)),
        ("3/4", "-1.0000", Fraction(-1)),
        ("2/4", "-0.1000", Fraction(-1, 10)),
    ],
)
def test_anacrusis_exact_decimals(tmp_path, meter, pickup, expected):
    """Pickup onsets use signature denominator units without float rounding."""
    path = tmp_path / "pickup.match"
    path.write_text(
        f"scoreprop(timeSignature,{meter},0:1,0,{pickup}).\n"
        f"snote(n1,[B,b],3,0:1,0,0,{pickup},{pickup},[])-deletion.\n",
        encoding="utf-8",
    )
    parsed = parse_match(path)
    row = score_quarters(parsed).iloc[0]
    assert row["quarter"] == expected
    assert row["onset_in_beats"] == Fraction(pickup)
    assert row["quarter_duration"] == Fraction(0)
    assert row["attributes"] == ()
    assert row["alter"] == "b"
    assert performance_notes(parsed).empty
    assert pd.api.types.is_integer_dtype(parsed.notes["onset"])


def test_invalid_onset(mf):
    """Reject mismatched coordinates with the file and score anchor."""
    mf.snotes.loc[0, "onset_in_beats"] = Fraction(1, 10)
    with pytest.raises(ValueError, match=r"p05_excerpt.match: n1-1: quarter"):
        score_quarters(mf)


@pytest.mark.parametrize(
    ("meter", "bar", "beat", "offset", "onset_in_beats", "expected"),
    [
        ("2/4", 3, 2, Fraction(0), "5.0000", Fraction(5)),
        ("3/4", 2, 3, Fraction(1, 8), "5.5000", Fraction(11, 2)),
    ],
)
def test_bar_formula_simple_meters(tmp_path, meter, bar, beat, offset, onset_in_beats, expected):
    """Apply the unfolded quarter formula to ordinary bars outside 6/8."""
    path = tmp_path / "meter.match"
    path.write_text(
        f"scoreprop(timeSignature,{meter},1:1,0,0.0000).\n"
        f"snote(n1,[C,n],4,{bar}:{beat},{offset},1/4,{onset_in_beats},"
        f"{onset_in_beats},[])-deletion.\n",
        encoding="utf-8",
    )
    row = score_quarters(parse_match(path)).iloc[0]
    assert row["quarter"] == expected


def test_no_anacrusis_bar(tmp_path):
    """A piece without any bar 0 row starts counting at bar 1 directly."""
    path = tmp_path / "no_pickup.match"
    path.write_text(
        "scoreprop(timeSignature,4/4,1:1,0,0.0000).\n"
        "snote(n1,[C,n],4,1:1,0,1/4,0.0000,1.0000,[])-deletion.\n"
        "snote(n2,[C,n],4,2:1,0,1/4,4.0000,5.0000,[])-deletion.\n",
        encoding="utf-8",
    )
    result = score_quarters(parse_match(path)).set_index("anchor")
    assert result.loc["n1", "quarter"] == 0
    assert result.loc["n2", "quarter"] == 4


def test_duplicate_info_key_raises(tmp_path):
    """Reject a repeated info key instead of silently overwriting it."""
    path = tmp_path / "dup.match"
    path.write_text("info(piece,first).\ninfo(piece,second).\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate info key"):
        parse_match(path)


def test_score_measure_count_synthetic(tmp_path):
    """Count only the first part's measures in a tiny synthetic score."""
    path = tmp_path / "tiny.musicxml"
    path.write_text(
        '<?xml version="1.0"?>\n'
        "<score-partwise>\n"
        '  <part-list><score-part id="P1"/></part-list>\n'
        '  <part id="P1">\n'
        '    <measure number="1"></measure>\n'
        '    <measure number="2"></measure>\n'
        "  </part>\n"
        '  <part id="P2">\n'
        '    <measure number="1"></measure>\n'
        "  </part>\n"
        "</score-partwise>\n",
        encoding="utf-8",
    )
    assert score_measure_count(path) == 2


@pytest.mark.parametrize("mode", ["missing", "duplicate", "invalid"])
def test_invalid_signature(mf, mode):
    """Enforce the single valid time signature required by the formula."""
    if mode == "missing":
        props = mf.scoreprops.iloc[:1]
    elif mode == "duplicate":
        props = pd.concat([mf.scoreprops, mf.scoreprops.iloc[1:]], ignore_index=True)
    else:
        props = mf.scoreprops.copy()
        props.loc[1, "value"] = "6/0"
    with pytest.raises(ValueError, match="timeSignature"):
        score_quarters(replace(mf, scoreprops=props))


def _anchor_sort_key(anchor):
    """Mirror the module's (number, repetition) ordering key for a full anchor."""
    number, repetition = re.fullmatch(r"n(\d+)(?:-(\d+))?", anchor).groups()
    return int(number), int(repetition) if repetition is not None else 0


def test_score_notes_keeps_full_anchors(tmp_path, mf):
    """Verify every performance and return every snote, unstripped and unfolded."""
    original = FIXTURE.read_text(encoding="utf-8")
    first = tmp_path / f"{PIECE}_p01.match"
    second = tmp_path / f"{PIECE}_p02.match"
    first.write_text(original, encoding="utf-8")
    second.write_text(original, encoding="utf-8")
    result = score_notes(tmp_path, PIECE)
    assert len(result) == len(mf.snotes) == 44
    assert result["anchor"].is_unique
    assert {"n1-1", "n1-2"} <= set(result["anchor"])
    ordering = [
        (row.quarter, *_anchor_sort_key(row.anchor)) for row in result.itertuples()
    ]
    assert ordering == sorted(ordering)
    # p01's own attributes are kept as-is: voice_overlap is not stripped.
    indexed = result.set_index("anchor")
    assert "voice_overlap" in indexed.loc["n238-1", "attributes"]
    assert "voice_overlap" in indexed.loc["n239-2", "attributes"]

    quarters = score_quarters(mf)
    keys = [
        (row.quarter, *_anchor_sort_key(row.anchor))
        for row in quarters.itertuples(index=False)
    ]
    order = sorted(range(len(quarters)), key=keys.__getitem__)
    expected = quarters.iloc[order].reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)

    second.write_text(original.replace(",voice_overlap", ""), encoding="utf-8")
    pd.testing.assert_frame_equal(score_notes(tmp_path, PIECE), result)
    third = tmp_path / f"{PIECE}_p03.match"
    third.write_text(
        original.replace("snote(n1-2,[C,#],5", "snote(n1-2,[C,#],4"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=f"{PIECE}_p03.match"):
        score_notes(tmp_path, PIECE)


def test_blank_lines_and_raw_strings(tmp_path):
    """Allow empty lines while preserving UTF-8 metadata without coercion."""
    path = tmp_path / "metadata.match"
    path.write_text(
        "\ninfo(composer,Frèdéryk Chopin).\ninfo(raw, 001 ).\n \n", encoding="utf-8"
    )
    mf = parse_match(path)
    assert mf.info == {"composer": "Frèdéryk Chopin", "raw": " 001 "}
    assert mf.snotes.empty
    assert mf.snotes.columns.tolist() == [
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
    ]
    assert mf.alignment.columns.tolist() == ["kind", "anchor", "note_id"]
    assert mf.soft.columns.tolist() == ["tick", "value"]
