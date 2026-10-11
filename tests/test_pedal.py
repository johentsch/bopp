"""Validate MIDI pedal controller columns and serialization behavior."""

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
from bopp.models.v1.payload.pedal import PedalPayload
from bopp.models.v1.payloads import AnyPayload
from bopp.registries.v1 import PAYLOAD_TYPE_REGISTRY

MINIMAL = {"controller": ["sustain"], "value": [127]}
FULL = {"controller": ["sustain", "soft", "sustain"], "value": [0, 64, 127]}


def create_annotation(columns, tick_type=float):
    """Build pedal events spaced by 480 MIDI ticks."""
    return bopp.create(
        media_id="test:pedal",
        payload_kind="pedal",
        extent_kind="midi_ticks",
        tick=[tick_type(480 * i) for i in range(len(columns["value"]))],
        **columns,
    )


def test_registry():
    """Register the pedal payload in the public payload registry."""
    assert PAYLOAD_TYPE_REGISTRY["pedal"] is PedalPayload


@pytest.mark.parametrize("columns", [MINIMAL, FULL], ids=["minimal", "full"])
@pytest.mark.parametrize(
    "tick_type", [float, int], ids=["float_ticks", "integer_ticks"]
)
def test_valid_annotation(columns, tick_type):
    """Both pedal controllers and boundary values validate."""
    annotation = create_annotation(columns, tick_type)
    assert bopp.validate(annotation)
    decoded = msgspec.json.decode(
        msgspec.json.encode(annotation.payload), type=AnyPayload
    )
    assert isinstance(decoded, PedalPayload)
    assert msgspec.to_builtins(decoded) == {"payload_type": "pedal", **columns}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("controller", "damper"),
        ("controller", "Sustain"),
        ("controller", ""),
        ("controller", None),
        ("controller", 64),
        *[("value", value) for value in [-1, 128, None, "64", 1.5, True]],
    ],
)
def test_invalid_columns(name, value):
    """Validation rejects unknown controllers, nulls and invalid MIDI values."""
    annotation = create_annotation({**MINIMAL, name: [value]})
    with pytest.raises(BoppValidationError):
        bopp.validate(annotation)


@pytest.mark.parametrize("name", ["payload_type", *MINIMAL])
def test_missing_required_column(name):
    """Typed decoding requires controller, value and the discriminator."""
    document = {"payload_type": "pedal", **MINIMAL}
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
    """JSON and MessagePack preserve ids, pedal values and MIDI tick positions."""
    annotation = create_annotation(columns, tick_type)
    assert bopp.validate(annotation)
    path = tmp_path / f"pedal.{suffix}"
    save(annotation, path)
    loaded = load(path)
    assert bopp.validate(loaded)
    assert loaded.id == annotation.id
    assert msgspec.to_builtins(loaded) == msgspec.to_builtins(annotation)
    # Integer ticks must decode back as integers, or the id would not match.
    assert {type(value) for value in loaded.extent.tick} == {tick_type}
