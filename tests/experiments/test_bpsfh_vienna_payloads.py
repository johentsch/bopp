"""Check the BPS-FH reader, the Vienna payloads and their benchmark tables."""

import json
import shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from experiments import bench, bpsfh, canonical, vienna_payloads
from experiments import layouts_parquet as parquet

DATA = Path(__file__).parent / "data"
EXCERPT = DATA / "vienna_Mozart_K331_1st-mov_p05_excerpt.match"


@pytest.fixture(scope="module")
def bps_root(tmp_path_factory):
    """One synthetic sonata with every source error the reader repairs."""
    root = tmp_path_factory.mktemp("bps")
    piece = root / "01"
    piece.mkdir()
    # onset, midi, morphetic, duration, staff, measure
    (piece / "notes.csv").write_text(
        "-1.0,60,60,1.0,0,0\n0.0,61,60,0.5,1,1\n0.5,66,53,0.5,1,1\n"
    )
    pd.DataFrame(
        [
            [-1, 4, "C", 1, "M", 0, "I"],
            [4, 2, "f+", "5/3", "D7", 0, "Vi/III"],
        ]
    ).to_excel(piece / "chords.xlsx", header=False, index=False)
    pd.DataFrame([[-1, 2.0, "exp", "1st", "a"], [8, 4.5, "exp", "tr", "b"]]).to_excel(
        piece / "phrases.xlsx", header=False, index=False
    )
    return root


def test_bps_notes(bps_root):
    respelled = []
    ann = bpsfh.read_notes(bps_root, "01", respelled=respelled)
    assert ann.media_id == "bpsfh:sonatas/01"
    assert ann.extent.quarter == [-1.0, 0.0, 0.5]
    assert ann.payload.name == ["C4", "C#4", "F#4"]
    assert ann.payload.tpc == [0, 7, 6]
    assert ann.payload.staff == [1, 2, 2]
    assert ann.payload.measure == [0, 1, 1]
    assert respelled == [("01", 2)]


def test_bps_chords_and_phrases(bps_root):
    inverted = []
    chords = bpsfh.read_chords(bps_root, "01", inverted=inverted)
    assert chords.payload.value == ["I", "V7/III"]
    assert chords.payload.key == ["C", "f#"]
    assert chords.extent.duration == [5.0, 2.0]
    assert inverted == [("01", 1)]
    phrases = bpsfh.read_phrases(bps_root, "01", inverted=inverted)
    assert phrases.payload.value == ["exp/1st/a", "exp/tr/b"]
    assert phrases.extent.quarter == [-1.0, 4.5]
    assert phrases.extent.duration == [3.0, 3.5]
    assert inverted == [("01", 1), ("01", 1)]


@pytest.fixture(scope="module")
def match_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("match")
    shutil.copy(EXCERPT, directory / "Mozart_K331_1st-mov_p01.match")
    return directory


def test_vienna_score_notes(match_dir):
    dropped = Counter()
    ann = vienna_payloads.score_note_annotation(
        match_dir, "Mozart_K331_1st-mov", dropped_attributes=dropped
    )
    assert ann.media_id == "vienna:vienna4x22/Mozart_K331_1st-mov"
    assert type(ann.extent).__struct_config__.tag == "quarters_interval.fraction"
    first = ann.payload.anchor.index("n1-1")
    assert ann.payload.name[first] == "C#5"
    assert ann.payload.midi[first] == 73
    assert ann.payload.tpc[first] == 7
    assert (ann.payload.voice[first], ann.payload.staff[first]) == (1, 1)
    assert Fraction(*ann.extent.duration[first]) == Fraction(3, 4)
    assert dropped == {"voice_overlap": 2}


def test_vienna_performance_and_pedal(match_dir):
    performance, pedal = vienna_payloads.performance_annotations(
        match_dir / "Mozart_K331_1st-mov_p01.match"
    )
    assert performance.extent.tick[:2] == [1407, 1438]
    assert all(type(t) is int for t in performance.extent.tick)
    assert performance.payload.match[:2] == ["n1-1", "n6-1"]
    assert "NOMATCH" in performance.payload.match
    ticks = pedal.extent.tick
    assert ticks == sorted(ticks)
    assert set(pedal.payload.controller) <= {"sustain", "soft"}


def test_number_union_round_trips_through_parquet(match_dir, tmp_path):
    performance, _ = vienna_payloads.performance_annotations(
        match_dir / "Mozart_K331_1st-mov_p01.match"
    )
    frame = canonical.annotation_to_frame(
        performance, corpus="vienna4x22", piece="Mozart_K331_1st-mov_p01"
    )
    parquet.write(
        frame,
        "vienna.performance_notes",
        tmp_path,
        layout="single",
        extent_kind="midi_interval",
    )
    loaded = parquet.load(
        tmp_path,
        "vienna.performance_notes",
        layout="single",
        extent_kind="midi_interval",
    )
    assert canonical.frames_equal(frame, loaded) == []
    assert type(loaded["tick"].iloc[0]) is int


def test_bench_vienna(match_dir, tmp_path):
    bench.run(
        [
            "--dataset",
            "vienna",
            "--source",
            str(match_dir),
            "--out",
            str(tmp_path),
            "--facets",
            "vienna.performance_notes",
            "--repeat",
            "1",
        ]
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 18
    by_cell = {bench.cell_name(row): row for row in rows}
    # CSV stores int|float ticks as floats: the id check fails, the plain load differs
    assert "ID mismatch" in by_cell["bopp-per_item-csv-none"]["error"]
    # upstream's CSV header carries the id now, so csv_upstream fails the same way
    assert "ID mismatch" in by_cell["bopp-per_item-csv_upstream-none"]["error"]
    others = [r for name, r in by_cell.items() if "csv" not in name]
    assert {r["roundtrip"] for r in others} == {"ok"}
    assert json.loads((tmp_path / "vienna-source.json").read_text())[
        "positions_from_onset_in_beats"
    ] == {"Mozart_K331_1st-mov": 0}
    assert "ID mismatch" in bench.report(tmp_path / "results.jsonl")
