"""Validate symbolic score note columns and annotation serialization."""

import msgspec
import pytest

import bopp
from bopp.exceptions import BoppArgumentError, BoppArrayError, BoppValidationError
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.models.v1.payload.score_note import ScoreNotePayload
from bopp.registries.v1 import COMPLEX_FIELDS_REGISTRY, PAYLOAD_TYPE_REGISTRY

MINIMAL = {"midi": [60, 61, 127], "tpc": [0, 7, -2]}
FULL = {
    **MINIMAL,
    "tpc": [0, 7, 1],
    "name": ["C4", "C#4", "G9"],
    "octave": [4, 4, 9],
    "staff": [1, 2, 1],
    "voice": [1, 2, 1],
    "mc": [1, 1, 2],
    "mn": [0, 0, 1],
    "measure": [0, 0, 1],
    "tied": [None, 1, -1],
    "gracenote": [None, "acciaccatura", "grace16"],
    "nominal_duration": [[1, 4], [1, 8], [1, 16]],
    "scalar": [[1, 1], [3, 2], [2, 3]],
    "chord_id": [0, 0, 1],
    "tremolo": [None, "1/2_r16_0", "1/2_r16_0"],
    "volta": [None, 1, 2],
    "tuning": [None, 0.0, -12.5],
    "anchor": ["n1", "n2", "n3"],
}
NULL_REPLACEMENTS = {
    "tied": 0,
    "gracenote": "appoggiatura",
    "tremolo": "1/2_r16_0",
    "volta": 1,
    "tuning": 0.0,
}
NONULL = {
    name: [NULL_REPLACEMENTS[name] if value is None else value for value in values]
    for name, values in FULL.items()
}
QUARTERS = {
    "quarter": [[0, 1], [1, 1], [3, 2]],
    "duration": [[1, 1], [1, 2], [1, 2]],
}
ONE_ROW = {name: values[:1] for name, values in NONULL.items()}
INVALID_COLUMNS = [
    ("midi", {**ONE_ROW, "midi": [128]}, "midi_above_range"),
    ("midi", {**ONE_ROW, "midi": [-1]}, "midi_below_range"),
    ("midi", {**ONE_ROW, "midi": [60.5]}, "midi_float"),
    ("midi", {**ONE_ROW, "midi": ["60"]}, "midi_string"),
    ("tpc", {**ONE_ROW, "tpc": [0.5]}, "tpc_float"),
    ("tied", {**ONE_ROW, "tied": [2]}, "tied_above_range"),
    ("volta", {**ONE_ROW, "volta": [0]}, "volta_zero"),
    ("staff", {**ONE_ROW, "staff": [0]}, "staff_zero"),
    ("voice", {**ONE_ROW, "voice": [0]}, "voice_zero"),
    ("mc", {**ONE_ROW, "mc": [0]}, "mc_zero"),
    ("chord_id", {**ONE_ROW, "chord_id": [-1]}, "chord_id_negative"),
    ("nominal_duration", {**ONE_ROW, "nominal_duration": [[1, 0]]}, "duration_zero_denominator"),
    ("nominal_duration", {**ONE_ROW, "nominal_duration": [[-1, 2]]}, "duration_negative"),
    ("nominal_duration", {**ONE_ROW, "nominal_duration": [0.25]}, "duration_float"),
    ("name", {**ONE_ROW, "name": [None]}, "name_null"),
    ("scalar", {**ONE_ROW, "scalar": [[-1, 2]]}, "scalar_negative"),
    ("scalar", {**ONE_ROW, "scalar": [[1, 0]]}, "scalar_zero_denominator"),
    ("tied", {**ONE_ROW, "tied": [-2]}, "tied_below_range"),
    ("volta", {**ONE_ROW, "volta": [-1]}, "volta_negative"),
    ("scalar", {**ONE_ROW, "scalar": [[3, 0]]}, "scalar_three_zero_denominator"),
    ("anchor", {**ONE_ROW, "anchor": [None]}, "anchor_null"),
]


def assert_columns(payload, columns):
    """Compare all columns while normalizing decoded fraction tuples."""
    for name in FULL:
        actual = getattr(payload, name)
        if name not in columns:
            assert actual is msgspec.UNSET
        else:
            if name in ("nominal_duration", "scalar"):
                actual = [list(value) for value in actual]
            assert actual == columns[name]


def create_annotation(columns):
    """Build a three-note annotation with fractional quarter extents."""
    return bopp.create(
        media_id="test:score_note",
        payload_kind="score_note",
        extent_kind="quarters_interval.fraction",
        **columns,
        **QUARTERS,
    )


def assert_annotation(annotation, columns):
    """Check validation, media identity, payload columns and quarter extents."""
    assert bopp.validate(annotation)
    assert annotation.media_id == "test:score_note"
    assert isinstance(annotation.payload, ScoreNotePayload)
    assert_columns(annotation.payload, columns)
    for name, expected in QUARTERS.items():
        assert [list(value) for value in getattr(annotation.extent, name)] == expected


def test_payload_registry():
    """The score note tag resolves to the generated payload class."""
    assert PAYLOAD_TYPE_REGISTRY["score_note"] is ScoreNotePayload


@pytest.mark.parametrize("columns", [MINIMAL, FULL], ids=["minimal", "full"])
def test_decode_columns(columns):
    """Decode required and optional columns, preserving nulls and absent columns."""
    decoded = msgspec.json.decode(
        msgspec.json.encode({"payload_type": "score_note", **columns}),
        type=ScoreNotePayload,
    )
    assert_columns(decoded, columns)
    for name in ("nominal_duration", "scalar"):
        actual = getattr(decoded, name)
        if actual is not msgspec.UNSET:
            assert all(isinstance(value, tuple) for value in actual)
            assert [list(value) for value in actual] == columns[name]


@pytest.mark.parametrize(
    ("field", "columns", "reason"),
    INVALID_COLUMNS,
    ids=[reason for _, _, reason in INVALID_COLUMNS],
)
def test_decode_invalid_columns(field, columns, reason):
    """Reject invalid item types, numeric bounds and fraction representations."""
    with pytest.raises(msgspec.ValidationError, match=rf"\$\.{field}\["):
        msgspec.json.decode(
            msgspec.json.encode({"payload_type": "score_note", **columns}),
            type=ScoreNotePayload,
        )


@pytest.mark.parametrize("missing", ["midi", "tpc"], ids=["missing_midi", "missing_tpc"])
def test_missing_required_column(missing):
    """Both pitch columns must be present during decoding."""
    columns = {name: values for name, values in MINIMAL.items() if name != missing}
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(
            msgspec.json.encode({"payload_type": "score_note", **columns}),
            type=ScoreNotePayload,
        )


def test_create_unknown_column():
    """Creating an annotation rejects unknown payload columns."""
    with pytest.raises(BoppArgumentError):
        create_annotation({**MINIMAL, "pitch": [60, 61, 127]})


def test_create_unequal_lengths():
    """Creating an annotation rejects pitch columns of different lengths."""
    with pytest.raises(BoppArrayError):
        create_annotation({**MINIMAL, "tpc": [0, 7]})


def test_validate_invalid_midi():
    """Validation checks item bounds that creation does not enforce."""
    annotation = create_annotation({**MINIMAL, "midi": [60, 64, 200]})
    with pytest.raises(BoppValidationError):
        bopp.validate(annotation)


@pytest.mark.parametrize("columns", [MINIMAL, FULL], ids=["minimal", "full"])
@pytest.mark.parametrize(
    ("save", "load", "suffix"),
    [
        (save_bopp_json, load_bopp_json, "json"),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
    ],
    ids=["json", "msgpack"],
)
def test_annotation_roundtrip(tmp_path, columns, save, load, suffix):
    """JSON and MessagePack preserve columns, absent values and annotation ids."""
    annotation = create_annotation(columns)
    assert_annotation(annotation, columns)
    path = tmp_path / f"score_note.{suffix}"
    save(annotation, path)
    loaded = load(path)
    assert_annotation(loaded, columns)
    assert loaded.id == annotation.id


@pytest.mark.parametrize("columns", [MINIMAL, NONULL], ids=["minimal", "full_nonull"])
def test_csv_roundtrip(tmp_path, columns):
    """CSV preserves nonnull columns and leaves absent optional columns unset."""
    annotation = create_annotation(columns)
    assert_annotation(annotation, columns)
    path = tmp_path / "score_note.csv"
    save_bopp_csv(annotation, path)
    assert_annotation(load_bopp_csv(path), columns)


def test_json_to_csv_roundtrip(tmp_path):
    """CSV handles fraction tuples produced by loading a JSON annotation."""
    json_path = tmp_path / "score_note.json"
    save_bopp_json(create_annotation(NONULL), json_path)
    annotation = load_bopp_json(json_path)
    assert isinstance(annotation.payload.nominal_duration[0], tuple)
    csv_path = tmp_path / "score_note.csv"
    save_bopp_csv(annotation, csv_path)
    assert_annotation(load_bopp_csv(csv_path), NONULL)


def test_complex_fields_registry():
    """Fraction columns require complex serialization while pitch columns do not."""
    fields = COMPLEX_FIELDS_REGISTRY["payload_type"]["score_note"]
    assert "nominal_duration" in fields
    assert "scalar" in fields
    assert "midi" not in fields
    assert "tpc" not in fields
