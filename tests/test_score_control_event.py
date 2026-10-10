"""Validate the seven DLC score control event payloads and their serialization."""

import json
from pathlib import Path

import msgspec
import pytest

import bopp
from bopp.exceptions import BoppValidationError
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.models.v1.payload.score_control_event.chord import ScoreChordPayload
from bopp.models.v1.payload.score_control_event.dynamic import ScoreDynamicPayload
from bopp.models.v1.payload.score_control_event.figured_bass import (
    ScoreFiguredBassPayload,
)
from bopp.models.v1.payload.score_control_event.spanner import ScoreSpannerPayload
from bopp.models.v1.payload.score_control_event.staff_text import ScoreStaffTextPayload
from bopp.models.v1.payload.score_control_event.system_text import (
    ScoreSystemTextPayload,
)
from bopp.models.v1.payload.score_control_event.tempo import ScoreTempoPayload
from bopp.models.v1.payloads import AnyPayload
from bopp.registries.v1 import COMPLEX_FIELDS_REGISTRY, PAYLOAD_TYPE_REGISTRY

CLASSES = {
    "chord": ScoreChordPayload,
    "dynamic": ScoreDynamicPayload,
    "spanner": ScoreSpannerPayload,
    "figured_bass": ScoreFiguredBassPayload,
    "staff_text": ScoreStaffTextPayload,
    "system_text": ScoreSystemTextPayload,
    "tempo": ScoreTempoPayload,
}
SPANNERS = (
    "slur",
    "crescendo_hairpin",
    "decrescendo_hairpin",
    "crescendo_line",
    "diminuendo_line",
    "pedal",
)
COMMON = {
    "staff": [1, 2, 1],
    "voice": [1, 2, 1],
    "mc": [1, 1, 2],
    "mn": [0, 0, 1],
    "volta": [1, 2, 1],
    **{name: ["0", "92, 93", "1"] for name in SPANNERS},
}
SPECIFIC = {
    "chord": {
        "chord_id": [0, 1, 2],
        "nominal_duration": [[1, 4], [1, 8], [0, 1]],
        "scalar": [[1, 1], [3, 2], [2, 3]],
        "gracenote": ["acciaccatura", "appoggiatura", "grace16"],
        "articulation": ["staccato", "accent", "tenuto"],
        "tremolo": ["1/2_r16_0"] * 3,
        "lyrics_1": ["one", "two", "three"],
        "lyrics_2": ["four", "five", "six"],
        "lyrics_3": ["seven", "eight", "nine"],
    },
    "dynamic": {"dynamics": ["mf", "p", "ff"]},
    "spanner": {},
    "figured_bass": {
        "thoroughbass_duration": [[1, 4], [1, 2], [0, 1]],
        "thoroughbass_level_1": ["6", "5", "#4"],
        "thoroughbass_level_2": ["4", "3", "b2"],
        "thoroughbass_level_3": ["2", "3", "#4"],
        "thoroughbass_level_4": ["3", "4", "b5"],
    },
    "staff_text": {"staff_text": ["dolce", "espressivo", "cantabile"]},
    "system_text": {"system_text": ["Fine", "Trio", "Coda"]},
    "tempo": {
        "qpm": [120.0, 90.0, 60.5],
        "tempo_visible": [True, False, True],
        "tempo": ["Allegro", "Andante", "Adagio"],
        "metronome_base": ["quarter", "half", "eighth"],
        "metronome_number": [120.0, 45.0, 121.0],
    },
}
REQUIRED = {
    kind: ("staff", "voice", "mc", "mn", *fields)
    for kind, fields in {
        "chord": ("chord_id",),
        "dynamic": ("dynamics",),
        "spanner": (),
        "figured_bass": ("thoroughbass_duration",),
        "staff_text": (),
        "system_text": ("system_text",),
        "tempo": ("qpm", "tempo_visible"),
    }.items()
}
NULLABLE = {
    kind: ("volta", *SPANNERS, *fields)
    for kind, fields in {
        "chord": (
            "gracenote",
            "articulation",
            "tremolo",
            "lyrics_1",
            "lyrics_2",
            "lyrics_3",
        ),
        "dynamic": (),
        "spanner": (),
        "figured_bass": tuple(f"thoroughbass_level_{level}" for level in range(1, 5)),
        "staff_text": ("staff_text",),
        "system_text": (),
        "tempo": ("tempo", "metronome_base", "metronome_number"),
    }.items()
}
NONULL = {kind: {**COMMON, **columns} for kind, columns in SPECIFIC.items()}
FULL = {
    kind: {
        name: [None, *values[1:]] if name in NULLABLE[kind] else values
        for name, values in columns.items()
    }
    for kind, columns in NONULL.items()
}
QUARTERS = {
    "quarter": [[0, 1], [1, 1], [3, 2]],
    "duration": [[1, 1], [1, 2], [0, 1]],
}


def create_annotation(kind, columns):
    """Build a three-event annotation with fractional quarter extents."""
    return bopp.create(
        media_id="test:score_control_event",
        payload_kind=f"score_control_event.{kind}",
        extent_kind="quarters_interval.fraction",
        **columns,
        **QUARTERS,
    )


def assert_payload(kind, payload, columns):
    """Compare columns after normalizing fraction tuples to JSON arrays."""
    assert type(payload) is CLASSES[kind]
    assert msgspec.json.decode(msgspec.json.encode(payload)) == {
        "payload_type": f"score_control_event.{kind}",
        **columns,
    }


@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize("minimal", [False, True], ids=["full", "minimal"])
def test_create_and_validate(kind, minimal):
    """Every kind accepts all columns, null items and omitted optional columns."""
    columns = FULL[kind]
    if minimal:
        columns = {name: columns[name] for name in REQUIRED[kind]}
    annotation = create_annotation(kind, columns)
    assert bopp.validate(annotation)
    assert_payload(kind, annotation.payload, columns)
    decoded = msgspec.json.decode(
        msgspec.json.encode(annotation.payload), type=AnyPayload
    )
    assert_payload(kind, decoded, columns)
    for name in NONULL[kind].keys() - columns.keys():
        assert getattr(decoded, name) is msgspec.UNSET


@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize(
    ("save", "load", "suffix"),
    [
        (save_bopp_json, load_bopp_json, "json"),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
        (save_bopp_csv, load_bopp_csv, "csv"),
    ],
    ids=["json", "msgpack", "csv"],
)
def test_roundtrip(tmp_path, kind, save, load, suffix):
    """All formats preserve payload classes; CSV uses nonnull item values."""
    columns = NONULL[kind] if suffix == "csv" else FULL[kind]
    annotation = create_annotation(kind, columns)
    assert bopp.validate(annotation)
    path = tmp_path / f"{kind}.{suffix}"
    save(annotation, path)
    loaded = load(path)
    assert bopp.validate(loaded)
    assert_payload(kind, loaded.payload, columns)
    assert msgspec.json.decode(
        msgspec.json.encode(loaded.payload)
    ) == msgspec.json.decode(msgspec.json.encode(annotation.payload))
    # Constructors retain fraction lists; typed decoding produces core tuples.
    expected_payload = msgspec.convert(
        msgspec.to_builtins(annotation.payload), type=CLASSES[kind]
    )
    assert loaded.payload == expected_payload
    assert loaded.media_id == annotation.media_id
    for name, expected in QUARTERS.items():
        assert [list(value) for value in getattr(loaded.extent, name)] == expected


@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize(
    ("save", "load", "suffix"),
    [
        (save_bopp_json, load_bopp_json, "json"),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
        (save_bopp_csv, load_bopp_csv, "csv"),
    ],
    ids=["json", "msgpack", "csv"],
)
def test_roundtrip_id(tmp_path, kind, save, load, suffix):
    """Check format-specific ids separately from passing payload round trips."""
    annotation = create_annotation(kind, NONULL[kind])
    path = tmp_path / f"{kind}.{suffix}"
    save(annotation, path)
    loaded = load(path)
    if suffix == "csv":
        # The CSV frontmatter does not carry the id.
        assert loaded.id is msgspec.UNSET
    else:
        assert loaded.id == annotation.id


@pytest.mark.xfail(strict=True,
    raises=msgspec.ValidationError,
    reason="Upstream bopp/io.py: pandas NaN for empty CSV cells breaks nullable ints",
)
def test_csv_roundtrip_nullable_int(tmp_path):
    """CSV should preserve null items in nullable integer columns."""
    columns = {**NONULL["spanner"], "volta": [None, 1, 2]}
    annotation = create_annotation("spanner", columns)
    assert bopp.validate(annotation)
    path = tmp_path / "nullable_int.csv"
    save_bopp_csv(annotation, path)
    loaded = load_bopp_csv(path)
    assert bopp.validate(loaded)
    assert_payload("spanner", loaded.payload, columns)


@pytest.mark.xfail(strict=True,
    raises=msgspec.ValidationError,
    reason="Upstream bopp/io.py: pandas NaN for empty CSV cells breaks nullable strings",
)
def test_csv_roundtrip_nullable_string(tmp_path):
    """CSV should preserve null items in nullable string columns."""
    columns = {**NONULL["spanner"], "slur": [None, "92, 93", "1"]}
    annotation = create_annotation("spanner", columns)
    assert bopp.validate(annotation)
    path = tmp_path / "nullable_string.csv"
    save_bopp_csv(annotation, path)
    loaded = load_bopp_csv(path)
    assert bopp.validate(loaded)
    assert_payload("spanner", loaded.payload, columns)


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        pytest.param(
            "figured_bass", "thoroughbass_level_1", "(6)",
            marks=pytest.mark.xfail(strict=True,
                raises=msgspec.ValidationError,
                reason=(
                    "Upstream bopp/io.py: literal_eval on COMPLEX_FIELDS_REGISTRY "
                    "columns converts figured-bass '(6)' to int"
                ),
            ),
        ),
        pytest.param(
            "staff_text", "staff_text", "(sic)",
            marks=pytest.mark.xfail(strict=True,
                raises=ValueError,
                reason=(
                    "Upstream bopp/io.py: literal_eval on COMPLEX_FIELDS_REGISTRY "
                    "columns rejects staff text '(sic)' as a malformed node"
                ),
            ),
        ),
    ],
    ids=["figured-bass-symbol", "staff-text"],
)
def test_csv_roundtrip_parenthesized_string(tmp_path, kind, field, value):
    """Parentheses in musical text should survive CSV without evaluation."""
    columns = {**NONULL[kind], field: [value, *NONULL[kind][field][1:]]}
    annotation = create_annotation(kind, columns)
    assert bopp.validate(annotation)
    path = tmp_path / f"{kind}.csv"
    save_bopp_csv(annotation, path)
    loaded = load_bopp_csv(path)
    assert bopp.validate(loaded)
    assert_payload(kind, loaded.payload, columns)


@pytest.mark.parametrize(
    ("kind", "missing"),
    [(kind, name) for kind in CLASSES for name in ("payload_type", *REQUIRED[kind])],
)
def test_missing_required_column(kind, missing):
    """The annotation payload union rejects missing tags and required columns."""
    document = {"payload_type": f"score_control_event.{kind}", **FULL[kind]}
    del document[missing]
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(msgspec.json.encode(document), type=AnyPayload)


@pytest.mark.parametrize(
    ("kind", "field"),
    [(kind, name) for kind, columns in NONULL.items() for name in columns],
)
def test_item_nullability(kind, field):
    """Only explicitly nullable columns accept a null item."""
    columns = {**NONULL[kind], field: [None, *NONULL[kind][field][1:]]}
    annotation = create_annotation(kind, columns)
    if field in NULLABLE[kind]:
        assert bopp.validate(annotation)
        decoded = msgspec.json.decode(
            msgspec.json.encode(annotation.payload), type=AnyPayload
        )
        assert_payload(kind, decoded, columns)
    else:
        with pytest.raises(BoppValidationError):
            bopp.validate(annotation)


@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize("field", ["staff", "voice", "mc", "volta"])
def test_one_based_columns(kind, field):
    """Staff, voice, measure counts and volta numbers start at one."""
    annotation = create_annotation(kind, {**NONULL[kind], field: [0, 1, 2]})
    with pytest.raises(BoppValidationError):
        bopp.validate(annotation)


@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize("field", SPANNERS)
def test_spanner_ids_are_strings(kind, field):
    """Spanner membership stores ms3 string lists, including single ids."""
    annotation = create_annotation(kind, {**NONULL[kind], field: [0, "92, 93", "1"]})
    with pytest.raises(BoppValidationError):
        bopp.validate(annotation)


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        ("chord", "chord_id", -1),
        ("chord", "scalar", [1, 0]),
        ("chord", "scalar", [-1, 2]),
        ("chord", "nominal_duration", [-1, 4]),
        ("chord", "nominal_duration", [1, 0]),
        ("figured_bass", "thoroughbass_duration", [1, 0]),
        ("figured_bass", "thoroughbass_duration", [-1, 4]),
        ("tempo", "qpm", 0),
        ("tempo", "metronome_number", 0),
        ("tempo", "tempo_visible", 1),
        ("dynamic", "dynamics", None),
    ],
)
def test_invalid_specific_columns(kind, field, value):
    """Validate type-specific bounds, fraction denominators and item types."""
    columns = {**NONULL[kind], field: [value, *NONULL[kind][field][1:]]}
    with pytest.raises(BoppValidationError):
        bopp.validate(create_annotation(kind, columns))


@pytest.mark.parametrize("kind", CLASSES)
def test_payload_registry(kind):
    """Each dot-separated tag resolves to its generated payload class."""
    assert PAYLOAD_TYPE_REGISTRY[f"score_control_event.{kind}"] is CLASSES[kind]


@pytest.mark.parametrize("kind", CLASSES)
def test_payload_fields_and_tag(kind):
    """Columns are Struct fields; the discriminator is separate msgspec metadata."""
    cls = CLASSES[kind]
    assert set(cls.__struct_fields__) == set(NONULL[kind])
    assert cls.__struct_config__.tag_field == "payload_type"
    assert cls.__struct_config__.tag == f"score_control_event.{kind}"
    payload = create_annotation(kind, FULL[kind]).payload
    wire = msgspec.json.decode(msgspec.json.encode(payload))
    assert set(wire) == {*NONULL[kind], "payload_type"}
    assert wire["payload_type"] == f"score_control_event.{kind}"


@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        ("chord", ["nominal_duration", "scalar"]),
        ("figured_bass", ["thoroughbass_duration"]),
    ],
)
def test_fraction_registry(kind, fields):
    """CSV recognizes fractional payload columns as complex values."""
    registered = COMPLEX_FIELDS_REGISTRY["payload_type"][f"score_control_event.{kind}"]
    assert set(fields) <= set(registered)


def test_union_schema():
    """The documented grouping lists exactly the seven payload schemas."""
    path = (
        Path(__file__).resolve().parents[1]
        / "schemas/v1/payload/score_control_event.json"
    )
    schema = json.loads(path.read_text())
    assert schema["$id"] == "payload/score_control_event.json"
    assert schema["title"] == "Score Control Event Payload"
    assert schema["oneOf"] == [
        {"$ref": f"score_control_event/{kind}.json"} for kind in CLASSES
    ]
    assert schema["discriminator"] == {"propertyName": "payload_type"}
