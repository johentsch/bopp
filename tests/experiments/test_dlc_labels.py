"""Verify DLC harmony labels, rational extents and serialization."""

from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

import bopp
from bopp.exceptions import BoppArgumentError
from bopp.io import load_bopp_json, save_bopp_json
from experiments.dlc_labels import harmonies_to_dcml_label, read_harmonies_tsv

FIXTURE = Path(__file__).parent / "data/dlc_harmonies_chopin_mazurkas_BI163op67-4.tsv"


@pytest.fixture
def frame():
    """Use the bundled harmony table without an external DLC checkout."""
    return read_harmonies_tsv(FIXTURE)


def test_reader_fixture(frame):
    """Retain all 117 labels verbatim and construct exact all-ending extents."""
    assert len(frame) == 117
    assert all(type(value) is str for value in frame.to_numpy().flat)
    annotation = harmonies_to_dcml_label(frame)
    assert bopp.validate(annotation)
    assert annotation.media_id == "dlc:chopin_mazurkas/BI163op67-4"
    assert annotation.payload.value == frame.label.tolist()
    assert annotation.payload.value[:4] == ["a.V{", "i", "V7(6)", "V7"]
    for index, quarter, duration in [
        (0, 0, 1),
        (1, 1, 4),
        (42, 94, 1),
        (43, 95, 2),
        (46, 100, 4),
        (68, 145, 1),
        (69, 146, 2),
        (116, 245, 2),
    ]:
        assert annotation.extent.quarter[index] == [quarter, 1]
        assert annotation.extent.duration[index] == [duration, 1]
    for source, actual in zip(
        frame.duration_qb, annotation.extent.duration, strict=True
    ):
        if source:
            assert abs(float(Fraction(*actual)) - float(source)) < 1e-6


def test_empty_durations_use_exact_next_start(frame):
    """Fill consecutive empty durations exactly without mutating source cells."""
    frame = frame.iloc[:4].copy().set_axis([9, 3, 7, 1])
    frame["duration_qb"] = ["", "", "", "2.0"]
    frame["quarterbeats_all_endings"] = ["1/1001", "", "2/1001", "5/1001"]
    frame["quarterbeats"] = ["99", "2/1001", "99", "99"]
    original = frame.copy(deep=True)
    annotation = harmonies_to_dcml_label(frame, max_denominator=2)
    assert annotation.extent.duration == [[1, 1001], [0, 1], [3, 1001], [2, 1]]
    assert bopp.validate(annotation)
    pd.testing.assert_frame_equal(frame, original)


def test_final_empty_duration_uses_piece_end(frame):
    """Close a final empty duration with the exact end of the piece."""
    frame = frame.iloc[:2].copy()
    frame["duration_qb"] = ["", ""]
    frame["quarterbeats_all_endings"] = ["1/3", "2/3"]
    annotation = harmonies_to_dcml_label(frame, piece_end=Fraction(7, 6))
    assert annotation.extent.duration == [[1, 3], [1, 2]]
    assert bopp.validate(annotation)
    # piece_end only closes a final empty cell; stated durations keep precedence.
    frame["duration_qb"] = ["", "2.0"]
    assert harmonies_to_dcml_label(frame, piece_end=Fraction(99)).extent.duration == [
        [1, 3],
        [2, 1],
    ]


@pytest.mark.parametrize("piece_end", [Fraction(1, 3), Fraction(0)])
def test_piece_end_before_the_final_start(frame, piece_end):
    """A piece end preceding the final label is backwards and must fail."""
    frame = frame.iloc[:1].copy()
    frame["duration_qb"] = [""]
    frame["quarterbeats_all_endings"] = ["2/3"]
    with pytest.raises(BoppArgumentError, match="duration_qb.*backwards"):
        harmonies_to_dcml_label(frame, piece_end=piece_end)


@pytest.mark.parametrize("piece_end", [2.5, "5/2", 2, True])
def test_invalid_piece_end(frame, piece_end):
    """Inexact piece ends are rejected rather than silently converted."""
    with pytest.raises(BoppArgumentError, match="piece_end"):
        harmonies_to_dcml_label(frame.iloc[:1], piece_end=piece_end)


@pytest.mark.parametrize("mode", ["final", "backwards"])
def test_empty_duration_requires_an_endpoint(frame, mode):
    """An unknown endpoint or reversed interval must fail without inventing data."""
    frame = frame.iloc[:2].copy()
    if mode == "final":
        frame.loc[1, "duration_qb"] = ""
        message = "no next start"
    else:
        frame.loc[0, "duration_qb"] = ""
        frame["quarterbeats_all_endings"] = ["2", "1"]
        message = "backwards next start"
    with pytest.raises(BoppArgumentError, match=f"duration_qb.*{message}"):
        harmonies_to_dcml_label(frame)


def test_duration_recovery_and_exact_starts(frame):
    """Recover tuplets without rounding exact starts or stripping label strings."""
    frame = frame.iloc[:3].copy()
    frame["duration_qb"] = ["0.3333333333333144", "0.1428571428571428", "0.0"]
    frame["quarterbeats_all_endings"] = ["1/1001", "", "1/3"]
    frame["quarterbeats"] = ["9", "2/7", "9"]
    frame["label"] = ["", "I", "i|PAC"]
    annotation = harmonies_to_dcml_label(frame, media_id="test:override")
    assert annotation.media_id == "test:override"
    assert annotation.extent.quarter == [[1, 1001], [2, 7], [1, 3]]
    assert annotation.extent.duration == [[1, 3], [1, 7], [0, 1]]
    assert annotation.payload.value == ["", "I", "i|PAC"]
    assert bopp.validate(annotation)
    frame["label"] = [" NA ", " I ", ""]
    assert harmonies_to_dcml_label(frame).payload.value == [" NA ", " I ", ""]


def test_start_column_fallback(frame):
    """Absent and empty all-ending starts use quarterbeats, parsed as fractions."""
    frame = frame.iloc[:2].copy()
    without_all = harmonies_to_dcml_label(
        frame.drop(columns="quarterbeats_all_endings")
    )
    without_unfolded = harmonies_to_dcml_label(frame.drop(columns="quarterbeats"))
    assert without_all.id == without_unfolded.id


@pytest.mark.parametrize("duration", ["0.3333333333333144", "0.000001", "0.123456789"])
def test_duration_residual_rejected(frame, duration):
    """Reject recovery residuals at or above the specified threshold."""
    frame = frame.iloc[:1].copy()
    frame.loc[0, "duration_qb"] = duration
    with pytest.raises(BoppArgumentError, match="duration_qb.*residual"):
        harmonies_to_dcml_label(frame, max_denominator=2)


def test_duration_residual_below_threshold(frame):
    """A residual strictly below 1e-6 remains within the recovery contract."""
    frame = frame.iloc[:1].copy()
    frame.loc[0, "duration_qb"] = "0.000000999"
    assert harmonies_to_dcml_label(frame).extent.duration == [[0, 1]]


@pytest.mark.parametrize("maximum", [0, -1, 1.5, True])
def test_invalid_max_denominator(frame, maximum):
    """Denominator limits must be positive integers."""
    with pytest.raises(BoppArgumentError, match="max_denominator"):
        harmonies_to_dcml_label(frame.iloc[:1], max_denominator=maximum)


@pytest.mark.parametrize("name", ["corpus", "piece"])
@pytest.mark.parametrize("mode", ["absent", "multiple", "empty_frame"])
def test_single_pair_required(frame, name, mode):
    """Require both identity columns and exactly one corpus/piece pair."""
    if mode == "absent":
        frame = frame.drop(columns=name)
    elif mode == "multiple":
        frame.loc[0, name] = "another"
    else:
        frame = frame.iloc[:0]
    with pytest.raises(BoppArgumentError, match="corpus/piece"):
        harmonies_to_dcml_label(frame)


@pytest.mark.parametrize("name", ["label", "duration_qb", "quarterbeats"])
def test_missing_required_column(frame, name):
    """Missing source columns raise an error naming the column."""
    frame = frame.iloc[:1].drop(columns="quarterbeats_all_endings")
    with pytest.raises(BoppArgumentError, match=name):
        harmonies_to_dcml_label(frame.drop(columns=name))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("duration_qb", "bad"),
        ("duration_qb", "nan"),
        ("duration_qb", "inf"),
        ("quarterbeats_all_endings", "1/0"),
        ("label", None),
    ],
)
def test_invalid_cell(frame, name, value):
    """Malformed values are reported against their source column."""
    frame = frame.iloc[:1].copy()
    frame.loc[0, name] = value
    with pytest.raises(BoppArgumentError, match=name):
        harmonies_to_dcml_label(frame)


def test_strings_preserved(tmp_path):
    """The TSV reader preserves empty labels and strings pandas treats as NA."""
    path = tmp_path / "labels.tsv"
    pd.DataFrame({"label": ["NA", "", "null", " I "]}).to_csv(
        path, sep="\t", index=False
    )
    assert read_harmonies_tsv(path).label.tolist() == ["NA", "", "null", " I "]


def test_json_roundtrip_id(tmp_path, frame):
    """JSON round trips retain the annotation id, labels and exact extents."""
    annotation = harmonies_to_dcml_label(frame)
    path = tmp_path / "labels.json"
    save_bopp_json(annotation, path)
    restored = load_bopp_json(path)
    assert bopp.validate(restored)
    assert restored.id == annotation.id
    assert restored.payload.value == annotation.payload.value
    assert [Fraction(*v) for v in restored.extent.quarter] == [
        Fraction(*v) for v in annotation.extent.quarter
    ]
    assert [Fraction(*v) for v in restored.extent.duration] == [
        Fraction(*v) for v in annotation.extent.duration
    ]
