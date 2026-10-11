"""Validate BPS-FH chord grammar, local keys and serialization behavior."""

import json
import re
from pathlib import Path

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
from bopp.models.v1.payload.music21_chord import Music21ChordPayload
from bopp.models.v1.payloads import AnyPayload
from bopp.registries.v1 import PAYLOAD_TYPE_REGISTRY

SCHEMA = Path(__file__).resolve().parents[1] / "schemas/v1/payload/music21_chord.json"
# Observed BPS-FH labels, including quality marks, inversions and secondary chains.
# The dataset's three malformed labels appear here in the form the BPS-FH reader
# normalizes them to ('Vi/III' -> 'VI/III', 'V7IV' -> 'V7/IV', 'vii-65V' -> 'vii-65/V');
# the raw spellings are rejected, see BAD_RN.
OBSERVED = [
    "I",
    "I+",
    "I+6",
    "I+64",
    "I42",
    "I6",
    "I64",
    "I7",
    "III+6",
    "IV6",
    "IV65",
    "V7",
    "V65/V",
    "V42/V",
    "V6/bII",
    "V7/bVII",
    "V43/iv",
    "V/vi",
    "V7/IV",
    "VI/III",
    "bII42",
    "bIII",
    "bVI6",
    "bVII6",
    "i",
    "i64",
    "ii-",
    "ii-6",
    "ii-64",
    "ii=42",
    "ii=43",
    "ii=64",
    "ii=65",
    "ii=7",
    "iv",
    "v/ii",
    "vii-7/V",
    "vii-65/V",
    "vii=65/III",
    "vii=7/bVII",
    "It+6",
    "Fr+6",
    "Gr+6",
    "Gr+6/IV",
    "N6",
]
COMPOSED = ["viio7/V", "#iv-65/ii/V", "V7/V/ii", "N6/V", "Fr+6/ii"]
# music21 RomanNumeral features the BPS-FH dialect never uses.
MUSIC21 = [
    "V9",
    "V11",
    "V13",
    "I53",
    "V63",
    "viio",
    "vii\u00f87",
    "ii\u00f865/V",
    "Cad64",
    "Cad64/V",
    "N",
    "N64",
    "N53",
    "It",
    "It6",
    "Fr43",
    "Ger",
    "Ger65",
    "Sw7",
]
KEYS = ["C", "c", "B-", "f#", "E--", "g##"]
BAD_RN = [
    "",
    " I",
    "I ",
    "X7",
    "V//V",
    "Q",
    "I6/",
    "iv\n",
    "V7 /V",
    "IIII",
    "/V",
    "V\r",
    "V\u2028",
    "V\u2029",
    # Raw BPS-FH typos: mixed-case degrees and omitted secondary slashes.
    "Vi/III",
    "V7IV",
    "vii-65V",
    "vII",
    # music21 features outside this grammar's scope.
    "V[no5]",
    "-VI",
    "Gr6",
    "It+",
    "Cad",
    "N7",
    "Sw9",
]
BAD_KEY = ["", "H", "C+", "Cb", "C# ", "f+", "C#-", "C-#", "C\n"]
MINIMAL = {"value": ["I"], "key": ["C"]}


def create_annotation(columns, extent_kind="quarters_interval.float"):
    """Build a chord annotation with one quarter-note interval per row."""
    count = len(columns["value"])
    fraction = extent_kind.endswith("fraction")
    return bopp.create(
        media_id="test:music21_chord",
        payload_kind="music21_chord",
        extent_kind=extent_kind,
        quarter=[[i, 1] for i in range(count)]
        if fraction
        else list(map(float, range(count))),
        duration=[[1, 1] for _ in range(count)] if fraction else [1.0] * count,
        **columns,
    )


def test_registry():
    """Register the chord payload in the public payload union."""
    assert PAYLOAD_TYPE_REGISTRY["music21_chord"] is Music21ChordPayload
    annotation = create_annotation(MINIMAL)
    decoded = msgspec.json.decode(
        msgspec.json.encode(annotation.payload), type=AnyPayload
    )
    assert isinstance(decoded, Music21ChordPayload)


@pytest.mark.parametrize(
    "extent_kind", ["quarters_interval.float", "quarters_interval.fraction"]
)
@pytest.mark.parametrize(
    "values",
    [["I"], OBSERVED, COMPOSED, MUSIC21],
    ids=["minimal", "observed", "composed", "music21"],
)
def test_valid_annotation(extent_kind, values):
    """Both quarter extents accept observed labels and wider music21 figures."""
    columns = {
        "value": values,
        "key": [KEYS[i % len(KEYS)] for i in range(len(values))],
    }
    annotation = create_annotation(columns, extent_kind)
    assert bopp.validate(annotation)
    assert msgspec.to_builtins(annotation.payload) == {
        "payload_type": "music21_chord",
        **columns,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        *[("value", value) for value in [*BAD_RN, None, 7]],
        *[("key", value) for value in [*BAD_KEY, None, 7]],
    ],
)
def test_invalid_columns(name, value):
    """Reject malformed labels and keys, trailing newlines and nonstring items."""
    annotation = create_annotation({**MINIMAL, name: [value]})
    with pytest.raises(BoppValidationError):
        bopp.validate(annotation)


@pytest.mark.parametrize("name", ["payload_type", *MINIMAL])
def test_missing_required_column(name):
    """Typed decoding requires both columns and the discriminator."""
    document = {"payload_type": "music21_chord", **MINIMAL}
    del document[name]
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(msgspec.json.encode(document), type=AnyPayload)


@pytest.mark.parametrize(
    ("save", "load", "suffix"),
    [
        (save_bopp_json, load_bopp_json, "json"),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
    ],
)
@pytest.mark.parametrize(
    "extent_kind", ["quarters_interval.float", "quarters_interval.fraction"]
)
def test_roundtrip(tmp_path, save, load, suffix, extent_kind):
    """JSON and MessagePack preserve ids, chord labels, keys and extents."""
    annotation = create_annotation(
        {"value": OBSERVED, "key": ["f#"] * len(OBSERVED)}, extent_kind
    )
    path = tmp_path / f"music21_chord.{suffix}"
    save(annotation, path)
    loaded = load(path)
    assert bopp.validate(loaded)
    assert loaded.id == annotation.id
    # Fraction arrays decode to tuples; compare their common wire representation.
    assert msgspec.json.decode(msgspec.json.encode(loaded)) == msgspec.json.decode(
        msgspec.json.encode(annotation)
    )


def test_jsonschema_pattern_agreement():
    """JSON Schema, Python regexes and bopp agree on RN and key constraints."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    for name, valid, invalid in [
        ("value", [*OBSERVED, *COMPOSED, *MUSIC21], BAD_RN),
        ("key", KEYS, BAD_KEY),
    ]:
        items = schema["properties"][name]["items"]
        validator = jsonschema.Draft202012Validator(items)
        for value in valid:
            assert re.search(items["pattern"], value) is not None
            validator.validate(value)
            assert bopp.validate(create_annotation({**MINIMAL, name: [value]}))
        for value in [*invalid, None, 7]:
            if isinstance(value, str):
                assert re.search(items["pattern"], value) is None
            with pytest.raises(jsonschema.ValidationError):
                validator.validate(value)
            with pytest.raises(BoppValidationError):
                bopp.validate(create_annotation({**MINIMAL, name: [value]}))
