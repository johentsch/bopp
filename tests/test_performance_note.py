"""Validate performed MIDI note columns and serialization behavior."""

import msgspec
import pytest

import bopp
from bopp.exceptions import BoppValidationError
from bopp.io import (
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.models.v1.payload.performance_note import PerformanceNotePayload
from bopp.models.v1.payloads import AnyPayload
from bopp.registries.v1 import COMPLEX_FIELDS_REGISTRY, PAYLOAD_TYPE_REGISTRY

MINIMAL = {"midi": [0, 60, 127], "velocity": [0, 64, 127]}
FULL = {
    **MINIMAL,
    "channel": [None, 0, 15],
    "track": [None, 0, 128],
    "anchor": [None, "n12", "source note 3"],
    "match": [None, "n238", "NOMATCH"],
}
INVALID = [
    *[(name, value) for name in MINIMAL for value in [-1, 128, None, "60", 1.5, True]],
    *[(name, value) for name in ["channel", "track"] for value in [-1, "0", 1.5, True]],
    ("channel", 16),
    ("anchor", 12),
    ("match", 238),
]


def create_annotation(columns, tick_type=float):
    """Build three performed notes with MIDI tick intervals."""
    return bopp.create(
        media_id="test:performance_note",
        payload_kind="performance_note",
        extent_kind="midi_interval",
        tick=[tick_type(value) for value in [0, 480, 960]],
        duration=[tick_type(value) for value in [120, 480, 240]],
        **columns,
    )


def test_registry():
    """Register the payload and its optional columns."""
    assert PAYLOAD_TYPE_REGISTRY["performance_note"] is PerformanceNotePayload
    assert (
        set(COMPLEX_FIELDS_REGISTRY["payload_type"]["performance_note"])
        == FULL.keys() - MINIMAL.keys()
    )


@pytest.mark.parametrize("columns", [MINIMAL, FULL], ids=["minimal", "full"])
@pytest.mark.parametrize(
    "tick_type", [float, int], ids=["float_ticks", "integer_ticks"]
)
def test_valid_annotation(columns, tick_type):
    """Bounds, omitted columns and nullable optional items validate."""
    annotation = create_annotation(columns, tick_type)
    assert bopp.validate(annotation)
    decoded = msgspec.json.decode(
        msgspec.json.encode(annotation.payload), type=AnyPayload
    )
    assert isinstance(decoded, PerformanceNotePayload)
    assert msgspec.to_builtins(decoded) == {
        "payload_type": "performance_note",
        **columns,
    }
    for name in FULL.keys() - columns.keys():
        assert getattr(decoded, name) is msgspec.UNSET


@pytest.mark.parametrize("name", ["anchor", "match"])
def test_free_source_strings(name):
    """Source identifiers are free strings, including empty and spaced values."""
    assert bopp.validate(
        create_annotation({**FULL, name: ["", "arbitrary anchor / 2", None]})
    )


@pytest.mark.parametrize(("name", "value"), INVALID)
def test_invalid_columns(name, value):
    """Validation enforces integer bounds, nonnull required items and text types.

    MIDI channels are capped at 15, so 16 is rejected while 15 stays in FULL.
    """
    annotation = create_annotation({**FULL, name: [value, *FULL[name][1:]]})
    with pytest.raises(BoppValidationError):
        bopp.validate(annotation)


@pytest.mark.parametrize("name", ["payload_type", *MINIMAL])
def test_missing_required_column(name):
    """Typed decoding requires MIDI, velocity and the discriminator."""
    document = {"payload_type": "performance_note", **MINIMAL}
    del document[name]
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(msgspec.json.encode(document), type=AnyPayload)


@pytest.mark.parametrize("columns", [MINIMAL, FULL], ids=["minimal", "full"])
@pytest.mark.parametrize(
    "tick_type", [float, int], ids=["float_ticks", "integer_ticks"]
)
@pytest.mark.parametrize(
    ("save", "load", "suffix"),
    [
        (save_bopp_json, load_bopp_json, "json"),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
    ],
)
def test_roundtrip(tmp_path, columns, tick_type, save, load, suffix):
    """JSON and MessagePack preserve ids, nullable note columns and intervals."""
    annotation = create_annotation(columns, tick_type)
    assert bopp.validate(annotation)
    path = tmp_path / f"performance_note.{suffix}"
    save(annotation, path)
    loaded = load(path)
    assert bopp.validate(loaded)
    assert loaded.id == annotation.id
    assert msgspec.to_builtins(loaded) == msgspec.to_builtins(annotation)
    # Integer ticks must decode back as integers, or the id would not match.
    assert {type(value) for value in loaded.extent.tick} == {tick_type}
    assert {type(value) for value in loaded.extent.duration} == {tick_type}
