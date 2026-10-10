"""Control-event reader, extension wire contract, and mixed-union experiment."""

import json
import warnings
from pathlib import Path

import msgspec
import pandas as pd
import pytest
from msgspec import inspect as mi

import bopp
from bopp.exceptions import BoppArgumentError
from bopp.extensions import get_extensions, reset_extensions
from bopp.io import (
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.models.v1.extent.quarters_interval_fraction import QuartersIntervalFraction
from bopp.registries.v1 import PAYLOAD_TYPE_REGISTRY
from experiments import dlc_chords
from experiments.dlc_chords import (
    MIXED_EXT_SCHEMA,
    chords_to_control_events,
    chords_to_mixed_ext,
    read_chords_tsv,
)

try:
    import bopp_score_ext as plugin
    from bopp_score_ext.convert import _BY_SCHEMA

    _PLUGIN_AVAILABLE = all(
        schema in get_extensions() for schema in plugin.EVENT_EXT_SCHEMAS.values()
    )
except ImportError:
    _PLUGIN_AVAILABLE = False

FIXTURE = Path(__file__).parent / "data/dlc_chords_chopin_mazurkas_BI163op67-4.tsv"
SCHEMAS = Path(__file__).resolve().parents[2] / "schemas/v1/payload"
KINDS = (
    "chord",
    "dynamic",
    "spanner",
    "figured_bass",
    "staff_text",
    "system_text",
    "tempo",
)
TAGS = dict(
    zip(
        KINDS,
        (
            "Chord",
            "Dynamic",
            "Spanner",
            "FiguredBass",
            "StaffText",
            "SystemText",
            "Tempo",
        ),
    )
)
COUNTS = {"chord": 563, "spanner": 132, "staff_text": 14, "dynamic": 5, "tempo": 2}
FIRST_ROWS = {
    "tempo": {
        "staff": 1,
        "voice": 1,
        "mc": 1,
        "mn": 0,
        "qpm": 114.0,
        "tempo_visible": True,
        "tempo": "Moderatoanimato",
    },
    "chord": {
        "staff": 1,
        "voice": 1,
        "mc": 1,
        "mn": 0,
        "chord_id": 0,
        "nominal_duration": [1, 4],
        "scalar": [1, 1],
        "articulation": "articAccentAbove",
    },
    "dynamic": {"staff": 1, "voice": 1, "mc": 2, "mn": 1, "dynamics": "mf"},
    "spanner": {"staff": 2, "voice": 1, "mc": 2, "mn": 1, "pedal": "0"},
}


@pytest.fixture
def frame():
    return read_chords_tsv(FIXTURE)


@pytest.fixture
def annotations(frame):
    return chords_to_control_events(frame)


def _schema(kind):
    return json.loads((SCHEMAS / f"score_control_event/{kind}.json").read_text())


def _plain(value):
    return msgspec.json.decode(msgspec.json.encode(value))


def _synthetic(kind):
    columns = {
        "figured_bass": {
            "thoroughbass_duration": [[3, 2], [1, 4]],
            "thoroughbass_level_1": ["6", None],
        },
        "system_text": {"system_text": ["Allegro", "Fine"]},
    }[kind]
    return bopp.create(
        media_id="test:synthetic",
        payload_kind=f"score_control_event.{kind}",
        staff=[1, 2],
        voice=[1, 1],
        mc=[1, 2],
        mn=[0, 1],
        extent_kind="quarters_interval.fraction",
        quarter=[[0, 1], [3, 2]],
        duration=[[0, 1], [1, 1]],
        **columns,
    )


def _annotation(kind, annotations):
    return annotations[kind] if kind in annotations else _synthetic(kind)


def test_reader_fixture(annotations, frame):
    assert len(frame) == 716
    assert all(isinstance(cell, str) for cell in frame.to_numpy().flat)
    assert {k: len(a.payload.staff) for k, a in annotations.items()} == COUNTS
    for ann in annotations.values():
        assert ann.media_id == "dlc:chopin_mazurkas/BI163op67-4"
        assert bopp.validate(ann)
    tempo = annotations["tempo"]
    assert _plain(tempo.extent.quarter) == [[0, 1], [0, 1]]
    assert tempo.payload.qpm == [114.0, 138.0]
    assert all(type(v) is float for v in tempo.payload.qpm)
    assert tempo.payload.tempo_visible == [True, True]
    assert all(type(v) is bool for v in tempo.payload.tempo_visible)
    assert tempo.payload.tempo == ["Moderatoanimato", "\U0001d15f=138"]
    assert tempo.payload.metronome_base == [None, "\U0001d15f"]
    assert tempo.payload.metronome_number == [None, 138.0]
    for name in ("slur", "volta", "pedal"):
        assert getattr(tempo.payload, name) is msgspec.UNSET
    dynamic = annotations["dynamic"]
    assert _plain(dynamic.extent.quarter) == [
        [1, 1],
        [127, 2],
        [281, 2],
        [142, 1],
        [151, 1],
    ]
    assert _plain(dynamic.extent.duration) == [[0, 1]] * 5
    assert dynamic.payload.dynamics == ["mf", "p", "f", "p", "mf"]
    assert dynamic.payload.slur == [None, "13", "26", None, None]
    assert dynamic.payload.mc == [2, 22, 48, 49, 52]
    assert dynamic.payload.mn == [1, 21, 46, 47, 49]
    chord = annotations["chord"]
    assert _plain(chord.extent.quarter[:3]) == [[0, 1], [1, 1], [5, 2]]
    assert _plain(chord.extent.duration[:3]) == [[1, 1], [3, 2], [1, 2]]
    assert _plain(chord.payload.nominal_duration[:3]) == [[1, 4], [1, 4], [1, 8]]
    assert _plain(chord.payload.scalar[:3]) == [[1, 1], [3, 2], [1, 1]]
    assert chord.payload.chord_id[:4] == [0, 1, 2, 3]
    assert chord.payload.slur[:4] == [None, "0", "0, 1", "0, 1"]
    assert chord.payload.articulation[:4] == [
        "articAccentAbove",
        None,
        None,
        "ornamentShortTrill",
    ]
    for name, count in (("volta", 24), ("slur", 317), ("pedal", 114)):
        assert sum(v is not None for v in getattr(chord.payload, name)) == count
    assert [v for v in chord.payload.gracenote if v is not None] == ["acciaccatura"] * 2
    for name in ("tremolo", "lyrics_1", "lyrics_2", "lyrics_3"):
        assert getattr(chord.payload, name) is msgspec.UNSET
    spanner = annotations["spanner"].payload
    assert sum(v is not None for v in spanner.pedal) == 52
    assert sum(v is not None for v in spanner.volta) == 16
    assert spanner.pedal[:2] == ["0", None]
    assert annotations["staff_text"].payload.staff_text[:3] == [
        "riten.",
        "marcato",
        "a tempo",
    ]


@pytest.mark.parametrize("kind", KINDS)
def test_reader_metadata_matches_schema(kind):
    schema = _schema(kind)
    columns = dlc_chords._COLUMNS[kind]
    assert tuple(n for n, _, _ in columns) == tuple(
        n
        for n in PAYLOAD_TYPE_REGISTRY[f"score_control_event.{kind}"].__struct_fields__
        if n != "payload_type"
    )
    assert {n for n, required, _ in columns if required} == set(schema["required"]) - {
        "payload_type"
    }
    for name, _, nullable in columns:
        item = schema["properties"][name]["items"]
        assert nullable == ("null" in item.get("type", []))
        assert (name in dlc_chords._FRACTIONS) == ("$ref" in item)
        assert (name in dlc_chords._FLOATS) == ("number" in item.get("type", []))
        assert (name in dlc_chords._INTS) == ("integer" in item.get("type", []))


@pytest.mark.parametrize("reader", [chords_to_control_events, chords_to_mixed_ext])
def test_reader_extent_tag_preserves_wire_and_id(reader, frame, tmp_path):
    result = reader(frame)
    annotations = result.values() if isinstance(result, dict) else [result]
    for ann in annotations:
        extent = ann.extent
        assert type(extent) is QuartersIntervalFraction
        assert type(extent).__struct_config__.tag == "quarters_interval.fraction"
        generated_extent = QuartersIntervalFraction(
            quarter=extent.quarter, duration=extent.duration
        )
        generated = msgspec.structs.replace(ann, extent=generated_extent)
        assert bopp.validate(ann)
        assert bopp.core.validate_annotation_id(ann)
        assert bopp.core.compute_annotation_id(generated) == ann.id
        for encode, save, load, suffix in (
            (msgspec.json.encode, save_bopp_json, load_bopp_json, "json"),
            (msgspec.msgpack.encode, save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
        ):
            assert encode(ann) == encode(generated)
            path = tmp_path / f"extent.{suffix}"
            save(ann, path)
            loaded = load(path, resolve_ext=False)
            assert type(loaded.extent) is QuartersIntervalFraction
            assert loaded.id == ann.id
            assert encode(loaded) == encode(ann)


def test_mixed_reader(frame, annotations, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ann = chords_to_mixed_ext(frame)
    assert ann.payload.ext_schema == MIXED_EXT_SCHEMA
    assert len(ann.payload.value) == 716
    assert [row["event"] for row in ann.payload.value] == frame.event.tolist()
    assert [row["event"] for row in ann.payload.value[:6]] == [
        "Tempo",
        "Tempo",
        "Chord",
        "Dynamic",
        "Chord",
        "Chord",
    ]
    assert ann.payload.value[0] == {"event": "Tempo", **FIRST_ROWS["tempo"]}
    assert _plain(ann.extent.quarter[:4]) == [[0, 1], [0, 1], [0, 1], [1, 1]]
    for row in ann.payload.value:
        assert next(iter(row)) == "event"
        assert all(value is not None for value in row.values())
    # Independently rebuild every row without importing the extension plugin.
    rows_by_tag = {}
    for kind, source in annotations.items():
        columns = _plain(source.payload)
        columns.pop("payload_type")
        rows_by_tag[TAGS[kind]] = iter(
            [
                {
                    name: column[i]
                    for name, column in columns.items()
                    if column[i] is not None
                }
                for i in range(COUNTS[kind])
            ]
        )
    rows = [{"event": event, **next(rows_by_tag[event])} for event in frame.event]
    assert msgspec.json.encode(rows) == msgspec.json.encode(ann.payload.value)
    rebuilt = bopp.create(
        media_id=ann.media_id,
        payload_kind="ext",
        ext_schema=MIXED_EXT_SCHEMA,
        value=rows,
        resolve_ext=False,
        extent_kind="quarters_interval.fraction",
        quarter=ann.extent.quarter,
        duration=ann.extent.duration,
    )
    assert rebuilt.id == ann.id
    for save, load, suffix in (
        (save_bopp_json, load_bopp_json, "json"),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
    ):
        path = tmp_path / f"mixed.{suffix}"
        save(ann, path)
        with pytest.warns(UserWarning, match="No extension registered"):
            loaded = load(path)
        assert loaded.id == ann.id
        assert all(isinstance(row, dict) for row in loaded.payload.value)


@pytest.mark.parametrize("reader", [chords_to_control_events, chords_to_mixed_ext])
@pytest.mark.parametrize(
    ("event", "column", "value", "match"),
    [
        ("Dynamic", "articulation", "accent", "articulation.*Dynamic"),
        ("Spanner", "Ottava:8va", "0", "Ottava:8va.*Spanner"),
        ("Tempo", "tempo_visible", "yes", "tempo_visible"),
        ("Tempo", "event", "Unknown", "Unknown"),
        ("Tempo", "piece", "another", "single corpus/piece"),
        ("Chord", "staff", "", "staff.*Chord"),
        ("Chord", "nominal_duration", "", "nominal_duration.*Chord"),
    ],
)
def test_reader_errors(frame, reader, event, column, value, match):
    if column not in frame:
        frame[column] = ""
    frame.loc[frame.index[frame.event == event][0], column] = value
    with pytest.raises(BoppArgumentError, match=match):
        reader(frame)


@pytest.mark.parametrize("reader", [chords_to_control_events, chords_to_mixed_ext])
@pytest.mark.parametrize(
    ("column", "value", "cause"),
    [
        ("duration", None, KeyError),
        ("duration", "", ValueError),
        ("staff", "x", ValueError),
        ("quarterbeats", "x", ValueError),
        ("scalar", "1/0", ZeroDivisionError),
        ("qpm", "x", ValueError),
    ],
)
def test_reader_malformed_cells(frame, reader, column, value, cause):
    event = "Tempo" if column == "qpm" else "Chord"
    frame = frame[frame.event == event].iloc[:1].copy()
    if value is None:
        frame = frame.drop(columns=column)
    else:
        frame[column] = value
    if column == "quarterbeats":
        frame["quarterbeats_all_endings"] = ""
    with pytest.raises(BoppArgumentError, match=f"{column}.*{event}") as excinfo:
        reader(frame)
    assert isinstance(excinfo.value.__cause__, cause)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.0", True),
        ("0.0", False),
        ("1", True),
        ("0", False),
        ("True", True),
        ("False", False),
    ],
)
def test_tempo_visible_spellings(frame, value, expected):
    frame.loc[frame.event == "Tempo", "tempo_visible"] = value
    assert (
        chords_to_control_events(frame)["tempo"].payload.tempo_visible == [expected] * 2
    )


def test_reader_optional_empty_and_required_missing(frame):
    frame["ignored_empty_column"] = ""
    frame.loc[frame.event == "Chord", "scalar"] = ""
    frame.loc[frame.event == "Dynamic", "slur"] = ""
    anns = chords_to_control_events(frame, media_id="test:custom")
    assert anns["chord"].payload.scalar is msgspec.UNSET
    assert anns["dynamic"].payload.slur is msgspec.UNSET
    assert all(a.media_id == "test:custom" for a in anns.values())
    for df in (frame.drop(columns="staff"), frame.assign(staff="")):
        with pytest.raises(BoppArgumentError, match="staff"):
            chords_to_control_events(df)


def test_reader_synthetic_events():
    df = pd.DataFrame(
        [
            {
                "corpus": "c",
                "piece": "p",
                "event": "FiguredBass",
                "staff": "1",
                "voice": "1",
                "mc": "1",
                "mn": "0",
                "quarterbeats": "3/2",
                "quarterbeats_all_endings": "",
                "duration": "1/4",
                "duration_qb": "999",
                "thoroughbass_duration": "3/2",
                "thoroughbass_level_1": "6",
                "system_text": "",
            },
            {
                "corpus": "c",
                "piece": "p",
                "event": "SystemText",
                "staff": "1",
                "voice": "1",
                "mc": "1",
                "mn": "0",
                "quarterbeats": "2",
                "quarterbeats_all_endings": "5/2",
                "duration": "0",
                "duration_qb": "999",
                "thoroughbass_duration": "",
                "thoroughbass_level_1": "",
                "system_text": "Fine",
            },
        ],
        index=[7, 7],
    )
    anns = chords_to_control_events(df)
    assert anns["figured_bass"].payload.thoroughbass_duration == [[3, 2]]
    assert anns["system_text"].payload.system_text == ["Fine"]
    mixed = chords_to_mixed_ext(df)
    assert mixed.extent.quarter == [[3, 2], [5, 2]]
    assert mixed.extent.duration == [[1, 1], [0, 1]]
    assert [r["event"] for r in mixed.payload.value] == ["FiguredBass", "SystemText"]


@pytest.mark.skipif(
    not _PLUGIN_AVAILABLE, reason="bopp-score-ext plugin is not installed or registered"
)
class TestPlugin:
    @pytest.mark.parametrize(
        ("kind", "column", "overlong"),
        [
            ("score_note", "scalar", [1, 1, 7]),
            ("chord", "scalar", [1, 2, 3]),
            ("figured_bass", "thoroughbass_duration", [1, 4, 9, 9]),
        ],
    )
    @pytest.mark.parametrize("shape", ["overlong", "short", "valid"])
    def test_fraction_pair_length(self, kind, column, overlong, shape):
        pair = {"overlong": overlong, "short": [1], "valid": [1, 2]}[shape]
        if kind == "score_note":
            row = {"midi": 60, "tpc": 0, column: pair}
            schema = plugin.EXT_SCHEMA
        else:
            row = {"staff": 1, "voice": 1, "mc": 1, "mn": 0, column: pair}
            if kind == "chord":
                row["chord_id"] = 0
                row["nominal_duration"] = [1, 4]
            schema = plugin.EVENT_EXT_SCHEMAS[kind]
        kwargs = {
            "media_id": "test:fraction-length",
            "payload_kind": "ext",
            "ext_schema": schema,
            "value": [row],
        }
        if shape == "valid":
            resolved = bopp.create(**kwargs, resolve_ext=True)
            raw = bopp.create(**kwargs, resolve_ext=False)
            assert msgspec.to_builtins(resolved.payload.value) == [row]
            assert resolved.id == raw.id
        else:
            with pytest.raises(msgspec.ValidationError, match=column):
                bopp.create(**kwargs, resolve_ext=True)

        if kind != "score_note":
            mixed_row = {"event": TAGS[kind], **row}
            mixed = bopp.create(
                **{**kwargs, "ext_schema": MIXED_EXT_SCHEMA, "value": [mixed_row]},
                resolve_ext=False,
            )
            if shape == "valid":
                assert msgspec.to_builtins(plugin.decode_mixed(mixed)) == [mixed_row]
            else:
                with pytest.raises(msgspec.ValidationError, match=column):
                    plugin.decode_mixed(mixed)

    def test_registry_and_exports(self):
        assert plugin.EVENT_KINDS == KINDS
        assert plugin.EVENT_TAGS == TAGS
        assert plugin.MIXED_EXT_SCHEMA == MIXED_EXT_SCHEMA
        assert MIXED_EXT_SCHEMA not in get_extensions()
        for kind, tag in TAGS.items():
            schema = f"io.github.johentsch.score_control_event.{kind}:v1"
            assert plugin.EVENT_EXT_SCHEMAS[kind] == schema
            row_type = getattr(plugin, f"Score{tag}")
            assert get_extensions()[schema] is plugin.EVENT_ROW_TYPES[kind] is row_type
            tagged = getattr(plugin, f"{tag}Event")
            assert issubclass(tagged, row_type)
            assert tagged.__struct_config__.tag == tag
            assert tagged.__struct_config__.tag_field == "event"
        assert set(plugin.__all__) == {
            "EXT_SCHEMA",
            "ScoreNote",
            "to_ext",
            "from_ext",
            "EVENT_KINDS",
            "EVENT_EXT_SCHEMAS",
            "EVENT_ROW_TYPES",
            "EVENT_TAGS",
            "MIXED_EXT_SCHEMA",
            "ScoreControlEvent",
            "decode_mixed",
            *(f"Score{tag}" for tag in TAGS.values()),
            *(f"{tag}Event" for tag in TAGS.values()),
        }

    @pytest.mark.parametrize("kind", (*KINDS, "score_note"))
    def test_schema_contract(self, kind):
        if kind == "score_note":
            schema = json.loads((SCHEMAS / "score_note.json").read_text())
            row_type, ext_schema, payload_kind = (
                plugin.ScoreNote,
                plugin.EXT_SCHEMA,
                kind,
            )
        else:
            schema = _schema(kind)
            row_type, ext_schema = (
                plugin.EVENT_ROW_TYPES[kind],
                plugin.EVENT_EXT_SCHEMAS[kind],
            )
            payload_kind = f"score_control_event.{kind}"
        payload_type = PAYLOAD_TYPE_REGISTRY[payload_kind]
        assert row_type.__struct_fields__ == tuple(
            n for n in payload_type.__struct_fields__ if n != "payload_type"
        )
        fields = msgspec.structs.fields(row_type)
        assert {f.name for f in fields if f.required} == set(schema["required"]) - {
            "payload_type"
        }
        assert all(f.default is msgspec.UNSET for f in fields if not f.required)
        assert row_type.__struct_config__.forbid_unknown_fields
        assert row_type.__struct_config__.omit_defaults
        encoding = _BY_SCHEMA[ext_schema]
        info = mi.type_info(row_type)
        for col, field in zip(encoding.columns, info.fields, strict=True):
            item = schema["properties"][col.name]["items"]
            assert col.name == field.name
            assert col.required == (col.name in schema["required"])
            assert col.nullable == ("null" in item.get("type", []))
            assert col.fraction == ("$ref" in item)
            assert col.floating == ("number" in item.get("type", []))
            scalar = field.type
            if isinstance(scalar, mi.Metadata):
                scalar = scalar.type
            if col.fraction:
                assert scalar.cls.__name__ == "FractionPair"
                assert scalar.array_like
                assert scalar.fields[0].type.ge == 0
                assert scalar.fields[1].type.ge == 1
            else:
                name = item["type"]
                name = name[0] if isinstance(name, list) else name
                assert isinstance(
                    scalar,
                    {
                        "integer": mi.IntType,
                        "number": mi.FloatType,
                        "string": mi.StrType,
                        "boolean": mi.BoolType,
                    }[name],
                )
                if name in {"integer", "number"}:
                    for bound, attr in (
                        ("minimum", "ge"),
                        ("maximum", "le"),
                        ("exclusiveMinimum", "gt"),
                    ):
                        assert getattr(scalar, attr) == item.get(bound)

    @pytest.mark.parametrize("kind", KINDS)
    def test_conversion_and_io(self, kind, annotations, tmp_path):
        source = _annotation(kind, annotations)
        snapshot = msgspec.json.encode(source)
        resolved = plugin.to_ext(source)
        raw = plugin.to_ext(source, resolve_ext=False)
        assert all(
            type(row) is plugin.EVENT_ROW_TYPES[kind] for row in resolved.payload.value
        )
        assert all(v is not None for row in raw.payload.value for v in row.values())
        assert msgspec.json.encode(resolved.payload.value) == msgspec.json.encode(
            raw.payload.value
        )
        assert (
            msgspec.msgpack.decode(msgspec.msgpack.encode(resolved.payload.value))
            == raw.payload.value
        )
        assert resolved.id == raw.id
        if kind in FIRST_ROWS:
            assert raw.payload.value[0] == FIRST_ROWS[kind]
        for ext in (resolved, raw):
            restored = plugin.from_ext(ext)
            assert restored.id == source.id
            assert _plain(restored.payload) == _plain(source.payload)
        assert msgspec.json.encode(source) == snapshot
        for save, load, suffix in (
            (save_bopp_json, load_bopp_json, "json"),
            (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
        ):
            path = tmp_path / f"{kind}.{suffix}"
            save(resolved, path)
            for resolve in (True, False):
                loaded = load(path, resolve_ext=resolve)
                assert loaded.id == raw.id
                assert bopp.core.validate_annotation_id(loaded)
                assert isinstance(
                    loaded.payload.value[0],
                    plugin.EVENT_ROW_TYPES[kind] if resolve else dict,
                )

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("failure", ["unknown", "null", "staff"])
    def test_common_validation(self, kind, failure, annotations):
        row = plugin.to_ext(
            _annotation(kind, annotations), resolve_ext=False
        ).payload.value[0]
        row.update(
            {
                "unknown": {"unexpected": 1},
                "null": {"pedal": None},
                "staff": {"staff": 0},
            }[failure]
        )
        with pytest.raises(msgspec.ValidationError):
            bopp.create(
                media_id="test:invalid",
                payload_kind="ext",
                ext_schema=plugin.EVENT_EXT_SCHEMAS[kind],
                value=[row],
            )

    @pytest.mark.parametrize(
        ("kind", "column", "value"),
        [
            ("chord", "chord_id", -1),
            ("chord", "scalar", [1, 0]),
            ("tempo", "qpm", 0.0),
            ("tempo", "metronome_number", 0.0),
            ("tempo", "tempo_visible", 1),
            ("dynamic", "dynamics", msgspec.UNSET),
            ("figured_bass", "thoroughbass_duration", [1, 0]),
            ("system_text", "system_text", msgspec.UNSET),
        ],
    )
    def test_specific_validation(self, kind, column, value, annotations):
        row = plugin.to_ext(
            _annotation(kind, annotations), resolve_ext=False
        ).payload.value[0]
        if value is msgspec.UNSET:
            del row[column]
        else:
            row[column] = value
        with pytest.raises(msgspec.ValidationError):
            bopp.create(
                media_id="test:invalid",
                payload_kind="ext",
                ext_schema=plugin.EVENT_EXT_SCHEMAS[kind],
                value=[row],
            )

    @pytest.mark.parametrize("kind", KINDS)
    def test_registry_cleared(self, kind, annotations, tmp_path):
        ann = plugin.to_ext(_annotation(kind, annotations), resolve_ext=False)
        path = tmp_path / "missing.json"
        save_bopp_json(ann, path)
        try:
            get_extensions().clear()
            with pytest.warns(UserWarning, match="No extension registered"):
                loaded = load_bopp_json(path)
            assert loaded.id == ann.id
            assert all(type(row) is dict for row in loaded.payload.value)
        finally:
            reset_extensions()
        assert (
            get_extensions()[plugin.EVENT_EXT_SCHEMAS[kind]]
            is plugin.EVENT_ROW_TYPES[kind]
        )

    def test_mixed_decode(self, frame, annotations):
        ann = chords_to_mixed_ext(frame)
        snapshot = msgspec.json.encode(ann)
        decoded = plugin.decode_mixed(ann)
        assert len(decoded) == 716
        assert msgspec.json.encode(decoded) == msgspec.json.encode(ann.payload.value)
        assert (
            msgspec.msgpack.decode(msgspec.msgpack.encode(decoded)) == ann.payload.value
        )
        assert all(
            type(row) is getattr(plugin, f"{event}Event")
            for row, event in zip(decoded, frame.event, strict=True)
        )
        assert msgspec.json.encode(ann) == snapshot
        for kind, source in annotations.items():
            rows = [
                {k: v for k, v in row.items() if k != "event"}
                for row in ann.payload.value
                if row["event"] == TAGS[kind]
            ]
            assert msgspec.json.encode(rows) == msgspec.json.encode(
                plugin.to_ext(source).payload.value
            )
        # Also exercise the two union members absent from the fixture.
        for kind in ("figured_bass", "system_text"):
            ext = plugin.to_ext(_synthetic(kind), resolve_ext=False)
            mixed = bopp.create(
                media_id="test:synthetic",
                payload_kind="ext",
                ext_schema=MIXED_EXT_SCHEMA,
                resolve_ext=False,
                value=[{"event": TAGS[kind], **r} for r in ext.payload.value],
            )
            assert all(
                type(r) is getattr(plugin, f"{TAGS[kind]}Event")
                for r in plugin.decode_mixed(mixed)
            )
        ann.payload.value[0]["event"] = "Unknown"
        with pytest.raises(msgspec.ValidationError):
            plugin.decode_mixed(ann)

    def test_reject_unsupported_payloads(self, annotations):
        other = bopp.create(media_id="test:x", payload_kind="tag_open", value=["x"])
        wrong_schema = bopp.create(
            media_id="test:x",
            payload_kind="ext",
            ext_schema="wrong",
            value=[],
            resolve_ext=False,
        )
        for function in (plugin.to_ext, plugin.from_ext, plugin.decode_mixed):
            with pytest.raises(BoppArgumentError):
                function(other)
        for ann in (wrong_schema, annotations["chord"]):
            with pytest.raises(BoppArgumentError):
                plugin.decode_mixed(ann)
        with pytest.raises(BoppArgumentError):
            plugin.from_ext(wrong_schema)

    def test_partial_and_all_missing_columns(self, annotations):
        ext = plugin.to_ext(annotations["chord"], resolve_ext=False)
        del ext.payload.value[0]["scalar"]
        with pytest.raises(BoppArgumentError, match="scalar"):
            plugin.from_ext(ext)
        for row in ext.payload.value:
            row.pop("scalar", None)
        assert plugin.from_ext(ext).payload.scalar is msgspec.UNSET
        source = annotations["dynamic"]
        source.payload.pedal = [None] * len(source.payload.staff)
        assert plugin.from_ext(plugin.to_ext(source)).payload.pedal is msgspec.UNSET
