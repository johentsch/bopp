"""Check schema-derived Arrow types, constraints and fixture Parquet round trips."""

import json
import math
import re
from pathlib import Path

import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import bopp
from bopp.exceptions import BoppValidationError
from bopp.registries.v1 import EXTENT_TYPE_REGISTRY, PAYLOAD_TYPE_REGISTRY
from experiments import arrow_schema as module
from experiments.arrow_schema import (
    ArrowValidationError,
    annotation_to_arrow,
    arrow_schema,
    ecma_to_re2,
    restore_number_union,
    validate_table,
)
from experiments.dlc_chords import chords_to_control_events, read_chords_tsv
from experiments.dlc_measures import measures_to_score_measure, read_measures_tsv
from experiments.dlc_notes import notes_to_score_note, read_notes_tsv

DATA = Path(__file__).parent / "data"
SCHEMAS = Path(bopp.__file__).resolve().parents[1] / "schemas/v1"
EXTENT = "quarters_interval.fraction"
TIME_FRACTION = "quarters_time.fraction"
MINIMAL = {
    "mc": [1, 2, 3],
    "mn": [0, 1, 2],
    "keysig": [-7, 0, 7],
    "timesig": ["3/4", "12/16", "33/32"],
    "act_dur": [[1, 4], [3, 4], [0, 1]],
    "mc_offset": [[-1, 2], [0, 1], [3, 4]],
    "next": [[2], [1, 3], [-1]],
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Keep the module's schema caches from leaking between tests."""
    caches = (module._document, module._tag_index, module._plan, module._bopp_pattern)
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:
        cached.cache_clear()


def tagged_document(category, tag):
    """Read the installed schema whose ``<category>_type`` const is ``tag``."""
    paths = {}
    for path in sorted((SCHEMAS / category).rglob("*.json")):
        source = json.loads(path.read_text())
        const = source.get("properties", {}).get(f"{category}_type", {}).get("const")
        if isinstance(const, str):
            paths[const] = source
    return paths[tag]


def is_columnar(schema):
    """Report typed core Array payloads, excluding unmappable object/Any items."""
    columns = [
        v for k, v in schema.get("properties", {}).items() if k != "payload_type"
    ]
    return bool(columns) and all(
        c.get("$ref", "").endswith("core.json#/$defs/Array")
        and c.get("items")
        and c["items"].get("type") != "object"
        for c in columns
    )


@pytest.mark.parametrize("tag", sorted(PAYLOAD_TYPE_REGISTRY))
def test_every_registered_payload(tag):
    """Resolve every registered tag, not file stems, including dotted kinds."""
    source = tagged_document("payload", tag)
    if not is_columnar(source):
        with pytest.raises(ValueError):
            arrow_schema(tag)
        return
    schema = arrow_schema(tag)
    assert schema.names == [
        "corpus",
        "piece",
        *[n for n in source["properties"] if n != "payload_type"],
    ]
    for field in list(schema)[2:]:
        assert (
            json.loads(field.metadata[b"bopp:items"])
            == source["properties"][field.name]["items"]
        )


@pytest.mark.parametrize(
    "kind",
    [
        "object",
        "ext",
        "onset",
        "pitch_contour",
        "relation",
        "score_control_event",
        "pitch_contour_hz",
        "segment_multi",
        "times",
        "unknown",
    ],
)
def test_unmappable_payload(kind):
    """Reject object payloads, base schemas and file stems that are not tags."""
    with pytest.raises(ValueError):
        arrow_schema(kind)


@pytest.mark.parametrize("tag", sorted(EXTENT_TYPE_REGISTRY))
def test_every_registered_extent(tag):
    """Cover every registered extent tag, including ``time`` and MIDI variants."""
    source = tagged_document("extent", tag)
    schema = arrow_schema("score_note", tag, index_columns=("piece",))
    extent_names = [name for name in source["properties"] if name != "extent_type"]
    assert schema.names[: 1 + len(extent_names)] == ["piece", *extent_names]


@pytest.mark.parametrize("extent", ["midi_ticks", "midi_interval"])
@pytest.mark.parametrize("dtype", [pa.int64(), pa.float64()])
def test_midi_number_union(extent, dtype):
    """Accept integer and floating MIDI columns and retain union metadata."""
    columns = {"tick": pa.array([0, 480], type=dtype), "value": [60, 62]}
    if extent == "midi_interval":
        columns["duration"] = pa.array([120, 240], type=dtype)
    result = validate_table(pa.table(columns), "note_midi", extent, index_columns=())
    for name in columns.keys() - {"value"}:
        assert result[name].type == pa.float64()
        item = json.loads(result.schema.field(name).metadata[b"bopp:items"])
        assert item == tagged_document("extent", extent)["properties"][name]["items"]


@pytest.mark.parametrize("column", ["tick", "duration"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1, 1.5])
def test_midi_union_constraint_parity(column, value):
    """Follow BOPP's unbounded ticks and nonnegative durations, including NaN."""
    extent = "midi_ticks" if column == "tick" else "midi_interval"
    columns = (
        {"tick": [value]} if column == "tick" else {"tick": [0], "duration": [value]}
    )
    ann = bopp.create(
        media_id="test:midi",
        payload_kind="note_midi",
        extent_kind=extent,
        value=[60],
        **columns,
    )
    table = pa.table({**columns, "value": [60]})
    if column == "duration" and (math.isnan(value) or value < 0):
        with pytest.raises(BoppValidationError):
            bopp.validate(ann)
        with pytest.raises(ArrowValidationError, match="duration.*minimum"):
            validate_table(table, "note_midi", extent, index_columns=())
    else:
        assert bopp.validate(ann)
        result = validate_table(table, "note_midi", extent, index_columns=())
        assert result[column].type == pa.float64()


@pytest.mark.parametrize("nullable", [False, True])
def test_nullable_number_union(monkeypatch, nullable):
    """Map only the numeric union, preserving optional nulls and both bounds."""
    original = module._document

    def document(path):
        schema = json.loads(json.dumps(original(path)))
        if path.name == "midi_ticks.json":
            schema["properties"]["tick"]["items"] = {
                "type": ["integer", "number", *(["null"] if nullable else [])],
                "minimum": 0,
                "maximum": 480,
            }
        return schema

    monkeypatch.setattr(module, "_document", document)
    field = arrow_schema("note_midi", "midi_ticks", index_columns=()).field("tick")
    assert field.type == pa.float64()
    assert field.nullable is nullable
    samples = [(-1, "minimum"), (float("nan"), "minimum"), (481, "maximum")]
    if not nullable:
        samples.append((None, "null"))
    for value, rule in samples:
        with pytest.raises(ArrowValidationError, match=f"tick.*{rule}"):
            validate_table(
                pa.table({"tick": [value], "value": [60]}),
                "note_midi",
                "midi_ticks",
                index_columns=(),
            )
    good = [0, None, 480] if nullable else [0, 1.5, 480]
    result = validate_table(
        pa.table({"tick": good, "value": [60] * 3}),
        "note_midi",
        "midi_ticks",
        index_columns=(),
    )
    assert result["tick"].to_pylist() == good


@pytest.mark.parametrize(
    "types", [["integer", "string"], ["integer", "number", "boolean"]]
)
def test_other_unions_remain_unsupported(types):
    """Do not broaden the supported mappings beyond the numeric union."""
    with pytest.raises(ValueError, match="Cannot map"):
        module._compile({"type": types})


def test_restore_number_union():
    """Restore exact Python integer types without changing fractional floats."""
    restored = restore_number_union([0.0, 480.0, 1.5])
    assert restored == [0, 480, 1.5]
    assert [type(value) for value in restored] == [int, int, float]
    extra = restore_number_union([None, 7, float("nan"), float("inf"), -float("inf")])
    assert extra[:2] == [None, 7]
    assert math.isnan(extra[2])
    assert extra[3:] == [float("inf"), -float("inf")]


@pytest.mark.parametrize("extent", ["midi_ticks", "midi_interval"])
def test_midi_integer_parquet_roundtrip(tmp_path, extent):
    """Restore integer MIDI extents and the annotation id after Parquet storage."""
    columns = {"tick": [0, 480, 960]}
    if extent == "midi_interval":
        columns["duration"] = [120, 240, 480]
    ann = bopp.create(
        media_id="test:midi",
        payload_kind="note_midi",
        extent_kind=extent,
        value=[60.0, 62.0, 64.0],
        **columns,
    )
    assert bopp.validate(ann)
    path = tmp_path / "midi.parquet"
    pq.write_table(annotation_to_arrow(ann), path)
    table = validate_table(pq.read_table(path), "note_midi", extent, index_columns=())
    restored = {name: restore_number_union(table[name].to_pylist()) for name in columns}
    assert all(type(value) is int for values in restored.values() for value in values)
    rebuilt = bopp.create(
        media_id=ann.media_id,
        payload_kind="note_midi",
        extent_kind=extent,
        value=table["value"].to_pylist(),
        **restored,
    )
    assert bopp.validate(rebuilt)
    assert rebuilt.id == ann.id


@pytest.mark.parametrize("column", ["tick", "duration"])
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("flag_first", [False, True], ids=["float-first", "bool-first"])
def test_midi_union_rejects_booleans(column, flag, flag_first):
    """Reject booleans in MIDI unions in either order, before any table exists."""
    extent = "midi_ticks" if column == "tick" else "midi_interval"
    values = [flag, 0.5] if flag_first else [0.5, flag]
    columns = (
        {"tick": values} if column == "tick" else {"tick": [0, 1], "duration": values}
    )
    ann = bopp.create(
        media_id="test:midi",
        payload_kind="note_midi",
        extent_kind=extent,
        value=[60.0, 62.0],
        **columns,
    )
    with pytest.raises(BoppValidationError, match="got `bool`"):
        bopp.validate(ann)
    row = 0 if flag_first else 1
    with pytest.raises(
        ArrowValidationError,
        match=rf"'{column}': type \(double\): boolean; 1 offending rows; "
        rf"first offending row index {row}",
    ):
        annotation_to_arrow(ann)


def _measure_columns(**overrides):
    """Return minimal valid score measure columns with a fraction extent."""
    return {
        **MINIMAL,
        "quarter": [[0, 1], [1, 1], [4, 1]],
        "duration": [[1, 1], [3, 1], [0, 1]],
        **overrides,
    }


@pytest.mark.parametrize(
    ("payload", "extent", "columns", "column", "row"),
    [
        # number: [0.5, True] became [0.5, 1.0] before the fix
        (
            "note_midi",
            "midi_ticks",
            {"tick": [0, 1], "value": [60.5, True]},
            "value",
            1,
        ),
        (
            "note_midi",
            "midi_ticks",
            {"tick": [0, 1], "value": [True, 60.5]},
            "value",
            0,
        ),
        (
            "note_midi",
            "midi_ticks",
            {"tick": [0, 1], "value": [60.5, np.True_]},
            "value",
            1,
        ),
        # integer/number union with a NumPy boolean after an int
        (
            "note_midi",
            "midi_ticks",
            {"tick": [0, np.True_], "value": [60.0, 1.0]},
            "tick",
            1,
        ),
        # integer: [1, np.True_] became [1, 1] before the fix
        ("beat", "time", {"time": [0.0, 1.0], "value": [1, True]}, "value", 1),
        ("beat", "time", {"time": [0.0, 1.0], "value": [1, np.True_]}, "value", 1),
        # fraction pair items: [1, np.True_] became [1, 1] before the fix
        (
            "beat",
            TIME_FRACTION,
            {"quarter": [[0, 1], [1, True]], "value": [1, 2]},
            "quarter",
            1,
        ),
        (
            "beat",
            TIME_FRACTION,
            {"quarter": [[0, 1], [1, np.True_]], "value": [1, 2]},
            "quarter",
            1,
        ),
        # string
        ("chord", "time", {"time": [0.0, 1.0], "value": ["C", True]}, "value", 1),
        ("chord", "time", {"time": [0.0, 1.0], "value": ["C", np.False_]}, "value", 1),
        # integer list items: [[2], [np.True_]] became [[2], [1]] before the fix
        (
            "score_measure",
            EXTENT,
            _measure_columns(next=[[2], [True], [-1]]),
            "next",
            1,
        ),
        (
            "score_measure",
            EXTENT,
            _measure_columns(next=[[2], [1, np.True_], [-1]]),
            "next",
            1,
        ),
    ],
)
def test_booleans_rejected_in_every_column_kind(payload, extent, columns, column, row):
    """Reject booleans wherever BOPP does, independently of their position."""
    numpy_bool = any(
        isinstance(value, np.bool_)
        for values in columns.values()
        for item in values
        for value in (item if isinstance(item, list) else [item])
    )
    ann = bopp.create(
        media_id="test:bool",
        payload_kind=payload,
        extent_kind=extent,
        generate_id=not numpy_bool,
        **columns,
    )
    # msgspec cannot encode NumPy booleans at all; Python booleans fail decoding.
    with pytest.raises(TypeError if numpy_bool else BoppValidationError):
        bopp.validate(ann)
    with pytest.raises(
        ArrowValidationError,
        match=rf"'{column}': type \(.*\): boolean; 1 offending rows; "
        rf"first offending row index {row}",
    ):
        annotation_to_arrow(ann)


def test_boolean_columns_keep_booleans():
    """Leave genuine boolean columns and boolean-free numeric columns alone."""
    columns = _measure_columns(dont_count=[True, None, False])
    ann = bopp.create(
        media_id="test:bool",
        payload_kind="score_measure",
        extent_kind=EXTENT,
        **columns,
    )
    assert bopp.validate(ann)
    table = annotation_to_arrow(ann)
    assert table["dont_count"].to_pylist() == [True, None, False]
    assert table["next"].to_pylist() == [[2], [1, 3], [-1]]


def test_multi_segment_tag():
    """The ``multi_segment`` tag resolves although its file stem differs."""
    schema = arrow_schema("multi_segment", "time")
    assert schema.names == ["corpus", "piece", "time", "label", "level"]
    assert schema.field("level").type == pa.int64()
    ann = bopp.create(
        media_id="test:multi_segment",
        payload_kind="multi_segment",
        extent_kind="time",
        time=[0.0, 1.5],
        label=["A", "B"],
        level=[0, 1],
    )
    table = annotation_to_arrow(ann, index={"corpus": "c", "piece": "p"})
    assert table.schema.equals(schema, check_metadata=True)
    assert table["label"].to_pylist() == ["A", "B"]
    assert table["level"].to_pylist() == [0, 1]
    with pytest.raises(ArrowValidationError, match="level.*minimum"):
        validate_table(
            pa.table({"time": [0.0], "label": ["A"], "level": [-1]}),
            "multi_segment",
            "time",
            index_columns=(),
        )


def test_types_nullability_and_order():
    """Required columns, nullable items and optional columns remain distinct."""
    schema = arrow_schema("score_note", EXTENT)
    assert schema.names[:6] == ["corpus", "piece", "quarter", "duration", "midi", "tpc"]
    for name in ("corpus", "piece"):
        assert schema.field(name).type == pa.string()
        assert not schema.field(name).nullable
    for name in ("midi", "tpc"):
        assert schema.field(name).type == pa.int64()
        assert not schema.field(name).nullable
    for name in ("tied", "staff", "octave"):
        assert schema.field(name).type == pa.int64()
        assert schema.field(name).nullable
    assert schema.field("tuning").type == pa.float64()
    assert schema.field("gracenote").type == pa.string()
    assert schema.field("quarter").type == pa.list_(pa.int64(), 2)
    measures = arrow_schema("score_measure")
    assert measures.field("next").type == pa.list_(pa.int64())
    assert measures.field("act_dur").type == pa.list_(pa.int64(), 2)
    assert measures.field("dont_count").type == pa.bool_()
    assert measures.field("dont_count").nullable
    assert not measures.field("mc").nullable


@pytest.fixture(params=["notes", "measures", "chords"])
def annotations(request):
    """Build all annotations from the three small bundled DLC fixtures."""
    path = DATA / f"dlc_{request.param}_chopin_mazurkas_BI163op67-4.tsv"
    if request.param == "notes":
        return [notes_to_score_note(read_notes_tsv(path))]
    if request.param == "measures":
        return [measures_to_score_measure(read_measures_tsv(path))]
    return list(chords_to_control_events(read_chords_tsv(path)).values())


@pytest.mark.parametrize("decode", [False, True], ids=["constructed", "decoded"])
def test_fixture_roundtrips(annotations, tmp_path, decode):
    """Every fixture event survives conversion, validation and Parquet storage."""
    for ann in annotations:
        if decode:
            ann = msgspec.json.decode(msgspec.json.encode(ann), type=type(ann))
        kind = type(ann.payload).__struct_config__.tag
        extent_kind = type(ann.extent).__struct_config__.tag
        assert "payload_type" not in ann.payload.__struct_fields__
        assert "extent_type" not in ann.extent.__struct_fields__
        index = {"corpus": "chopin_mazurkas", "piece": "BI163op67-4"}
        table = annotation_to_arrow(ann, index=index)
        assert table.schema.equals(arrow_schema(kind, extent_kind), check_metadata=True)
        assert validate_table(table, kind, extent_kind).equals(table)
        for struct in (ann.extent, ann.payload):
            for name in struct.__struct_fields__:
                values = getattr(struct, name)
                if values is msgspec.UNSET:
                    assert table[name].null_count == len(table)
                else:
                    assert table[name].to_pylist() == msgspec.json.decode(
                        msgspec.json.encode(values)
                    )
        pq.write_table(table, tmp_path / f"{kind}.parquet")
        restored = validate_table(
            pq.read_table(tmp_path / f"{kind}.parquet"), kind, extent_kind
        )
        assert restored.equals(table, check_metadata=True)
        no_index = annotation_to_arrow(ann)
        assert "corpus" not in no_index.column_names
        assert no_index.equals(table.drop(["corpus", "piece"]))


@pytest.mark.parametrize(
    ("column", "value", "rule"),
    [
        ("keysig", 8, "maximum"),
        ("keysig", -8, "minimum"),
        ("mc", 0, "minimum"),
        ("timesig", "3/0", "pattern"),
        ("mc", None, "null"),
        ("act_dur", [1, 0], "minimum"),
        ("act_dur", [-1, 4], "minimum"),
        ("act_dur", [1, -1], "minimum"),
        ("act_dur", [1], "minItems"),
        ("act_dur", [1, 2, 3], "maxItems"),
        ("act_dur", [None, 4], "null"),
        ("next", [], "minItems"),
        ("next", [-2], "minimum"),
        ("next", [None], "null"),
        ("next", [0, -2, -3], "minimum"),
    ],
)
def test_measure_rejections(column, value, rule):
    """Identify the rule and original row, including failures inside lists."""
    values = {**MINIMAL, column: [MINIMAL[column][0], value, MINIMAL[column][2]]}
    table = pa.table(values)
    with pytest.raises(ArrowValidationError) as caught:
        validate_table(table, "score_measure", index_columns=())
    message = str(caught.value)
    assert f"'{column}'" in message
    assert rule in message
    assert "1 offending rows" in message
    assert "first offending row index 1" in message


@pytest.mark.parametrize("label", ["XYZ", "V7 ", "V7\n", "IIIIII[II"])
def test_dcml_rejections(label):
    """Retain the strict DCML syntax and end-of-input assertion."""
    with pytest.raises(ArrowValidationError, match="value.*pattern.*1 offending rows"):
        validate_table(pa.table({"value": [label]}), "dcml_label", index_columns=())


@pytest.mark.parametrize(
    "mode", ["extra", "missing", "duplicate", "missing_index", "null_index"]
)
def test_column_structure(mode):
    """Reject extra, duplicate, missing and null identity columns."""
    table = pa.table(MINIMAL)
    indices = ()
    if mode == "extra":
        table = table.append_column("unknown", pa.array([1, 2, 3]))
        match = "unknown.*unknown column"
    elif mode == "missing":
        table = table.drop(["mc"])
        match = "mc.*missing required column"
    elif mode == "duplicate":
        table = table.append_column("mc", table["mc"])
        match = "mc.*duplicate column"
    elif mode == "missing_index":
        indices = ("corpus",)
        match = "corpus.*missing required column"
    else:
        indices = ("corpus",)
        table = table.append_column("corpus", pa.array(["a", None, "b"]))
        match = "corpus.*null"
    with pytest.raises(ArrowValidationError, match=match):
        validate_table(table, "score_measure", index_columns=indices)


def test_optional_and_casts():
    """Fill absent optional columns, reorder fields and safely widen types."""
    columns = {
        **MINIMAL,
        "mc": pa.array([1, 2, 3], type=pa.int32()),
        "timesig": pa.array(MINIMAL["timesig"], type=pa.large_string()),
    }
    table = pa.table(dict(reversed(list(columns.items()))))
    result = validate_table(table, "score_measure", index_columns=())
    assert result.schema.equals(
        arrow_schema("score_measure", index_columns=()), check_metadata=True
    )
    assert result["mc"].type == pa.int64()
    assert result["timesig"].type == pa.string()
    assert result["volta"].to_pylist() == [None] * 3
    assert validate_table(result, "score_measure", index_columns=()).equals(result)


def test_optional_nonnullable_items():
    """An omitted optional column differs from a present column with null items."""
    table = pa.table({"midi": [60, 61], "tpc": [0, 1]})
    result = validate_table(table, "score_note", index_columns=())
    assert result["staff"].null_count == 2
    with pytest.raises(ArrowValidationError, match="staff.*null"):
        validate_table(
            table.append_column("staff", pa.array([1, None])),
            "score_note",
            index_columns=(),
        )


def test_chunked_nested_failure_counts():
    """Count failing rows once across chunks, offsets and repeated bad children."""
    columns = {**MINIMAL, "next": [[-2, -3], [2], [-4, -5]]}
    table = pa.concat_tables([pa.table(MINIMAL), pa.table(columns)]).slice(1)
    with pytest.raises(
        ArrowValidationError,
        match=("next.*minimum.*2 offending rows.*first offending row index 2"),
    ):
        validate_table(table, "score_measure", index_columns=())


def test_enum(monkeypatch):
    """Exercise enum validation without changing an installed JSON schema."""
    original = module._document

    def document(path):
        schema = json.loads(json.dumps(original(path)))
        if path.name == "score_measure.json":
            schema["properties"]["barline"]["items"]["enum"] = [None, "double"]
        return schema

    monkeypatch.setattr(module, "_document", document)
    good = pa.table({**MINIMAL, "barline": [None, "double", "double"]})
    validate_table(good, "score_measure", index_columns=())
    bad = good.set_column(
        good.column_names.index("barline"),
        "barline",
        pa.array([None, "single", "double"]),
    )
    with pytest.raises(ArrowValidationError, match="barline.*enum.*1 offending rows"):
        validate_table(bad, "score_measure", index_columns=())


def test_exclusive_minimum():
    """Tempo control events enforce the positive-only bounds from their schema."""
    table = pa.table(
        {
            "staff": [1],
            "voice": [1],
            "mc": [1],
            "mn": [0],
            "qpm": [0.0],
            "tempo_visible": [True],
        }
    )
    with pytest.raises(ArrowValidationError, match="qpm.*exclusiveMinimum"):
        validate_table(table, "score_control_event.tempo", index_columns=())


LABELS = [
    "",
    "I",
    "V7",
    "ii%65",
    "Ger65",
    "It6",
    "Fr43",
    "@none",
    "bVII",
    "C.V7",
    "g.i",
    ".C.V7",
    "V/V",
    "V7(+9)",
    "I(6)",
    "I|PAC",
    "I|PAC.comment",
    "I|PAC.é",
    "I|PAC.🙂",
    "V7-I",
    "-",
    "|PAC",
    "{",
    "I}{",
    "I\\\\",
    "XYZ",
    "V7 ",
    " V7",
    "V7\n",
    "I|PAC.x\n",
    "I|PAC.x\r",
    "I|PAC.x\u2028",
    "I|PAC.x\u2029",
    "IIIIII[II",
    "C:maj",
    "V7\x00",
    "I|PAC.\x00",
    "I|PAC. ",
    "I|PAC.\t",
]


def test_dcml_pattern_agreement():
    """Match Python's strict pattern on valid, invalid and Unicode examples."""
    pattern = json.loads((SCHEMAS / "payload/dcml_label.json").read_text())[
        "properties"
    ]["value"]["items"]["pattern"]
    translated = ecma_to_re2(pattern)
    assert translated.endswith(r"\z")
    expected = [re.fullmatch(pattern, label) is not None for label in LABELS]
    assert (
        pc.match_substring_regex(pa.array(LABELS), translated).to_pylist() == expected
    )
    good = [label for label, valid in zip(LABELS, expected, strict=True) if valid]
    assert (
        validate_table(pa.table({"value": good}), "dcml_label", index_columns=())[
            "value"
        ].to_pylist()
        == good
    )


@pytest.mark.parametrize(
    "pattern", [r"a(?=b)", r"(?<!a)b", r"(a)\1", r"\w+", r"(?i)a", r"a++", r"\uD800"]
)
def test_unsupported_patterns(pattern):
    """Reject constructs whose semantics are not translated."""
    with pytest.raises(ValueError):
        ecma_to_re2(pattern)


def test_patterns_stay_unanchored():
    """JSON Schema pattern is a search; existing alternation precedence survives."""
    assert pc.match_substring_regex(
        pa.array(["abc", "ba", "z"]), ecma_to_re2("a")
    ).to_pylist() == [True, True, False]
    # The key_mode schema has an ungrouped alternation, intentionally retained.
    table = pa.table({"value": ["Nonsense", "prefixC:major"]})
    assert validate_table(table, "key_mode", index_columns=()).equals(
        table.cast(arrow_schema("key_mode", index_columns=()))
    )


def test_empty_table_and_collisions():
    """Empty typed columns validate, and colliding index names fail early."""
    table = pa.table(MINIMAL).slice(0, 0)
    assert len(validate_table(table, "score_measure", index_columns=())) == 0
    with pytest.raises(ValueError, match="Duplicate column"):
        arrow_schema("score_measure", index_columns=("mc",))


@pytest.mark.parametrize(
    ("kind", "column", "samples"),
    [
        ("score_measure", "timesig", ["3/4", "3/4\n", "3/4\n\n", "3/0", "3/4\r"]),
        ("chord", "value", ["C:maj", "C:maj\n", "C:maj\n\n", "X", "C:maj "]),
        ("key_mode", "value", ["C:major\n", "prefixC:major", "Nonsense", "Z"]),
        ("dcml_label", "value", ["V7", "V7\n", "I|PAC.x\u2028", "I|PAC.é"]),
    ],
)
def test_python_pattern_validation_parity(kind, column, samples):
    """Follow BOPP decoding for terminal dollars, without weakening DCML's end."""
    pattern = json.loads((SCHEMAS / f"payload/{kind}.json").read_text())["properties"][
        column
    ]["items"]["pattern"]
    for sample in samples:
        columns = (
            {name: values[:1] for name, values in MINIMAL.items()}
            if (kind == "score_measure")
            else {}
        )
        columns[column] = [sample]
        table = pa.table(columns)
        if re.search(pattern, sample):
            validate_table(table, kind, index_columns=())
        else:
            with pytest.raises(ArrowValidationError, match=f"{column}.*pattern"):
                validate_table(table, kind, index_columns=())
    # The public translator itself keeps ECMA's strict dollar semantics.
    assert not pc.match_substring_regex(
        pa.array(["3/4\n"]), ecma_to_re2(r"^3/4$")
    ).to_pylist()[0]


@pytest.mark.parametrize(
    ("kind", "column", "values", "rule"),
    [
        ("tempo", "value", [1.0, float("nan")], "minimum"),
        ("score_note", "midi", [60.0, 60.5], "type"),
        ("score_note", "midi", [60.0, float("inf")], "type"),
        ("score_note", "tpc", pa.array([0, 2**63], type=pa.uint64()), "int64 range"),
        ("note_midi", "value", [60, 2**53 + 1], "float64 exact range"),
        ("score_note", "midi", ["60", "61"], "type"),
    ],
)
def test_numeric_cast_and_nan_errors(kind, column, values, rule):
    """Invalid casts and bounded NaN values receive row-oriented diagnostics."""
    columns = {"midi": [60, 61], "tpc": [0, 1]} if kind == "score_note" else {}
    columns[column] = values
    with pytest.raises(ArrowValidationError, match=f"{column}.*{rule}.*offending rows"):
        validate_table(pa.table(columns), kind, index_columns=())


def test_integer_columns_reject_floats():
    """BOPP rejects a float for an integer column, even when it is integral."""
    table = pa.table({"midi": pa.array([60.0, 61.0], type=pa.float64()), "tpc": [0, 1]})
    with pytest.raises(ArrowValidationError) as caught:
        validate_table(table, "score_note", index_columns=())
    message = str(caught.value)
    assert "'midi'" in message
    assert "type (int64)" in message
    assert "2 offending rows" in message
    assert "first offending row index 0" in message
    narrower = pa.table(
        {
            "midi": pa.array([60, 61], type=pa.int16()),
            "tpc": pa.array([0, 1], type=pa.uint8()),
        }
    )
    result = validate_table(narrower, "score_note", index_columns=())
    assert result["midi"].type == pa.int64()
    assert result["tpc"].to_pylist() == [0, 1]


def test_dot_follows_python_semantics():
    """Validation reproduces Python's ``.``, which excludes the line feed only."""
    samples = ["a\rb", "a\nb", "axb"]
    expected = [re.search("a.b", sample) is not None for sample in samples]
    assert expected == [True, False, True]
    assert (
        pc.match_substring_regex(
            pa.array(samples), module._bopp_pattern("a.b")
        ).to_pylist()
        == expected
    )
    # The public ECMA translator excludes every line terminator instead.
    assert pc.match_substring_regex(
        pa.array(samples), ecma_to_re2("a.b")
    ).to_pylist() == [False, False, True]


def test_fixed_list_chunk_offsets():
    """Fixed-list fraction checks retain row indices after chunking and slicing."""
    valid = validate_table(pa.table(MINIMAL), "score_measure", index_columns=())
    position = valid.column_names.index("act_dur")
    bad = valid.set_column(
        position,
        valid.schema.field("act_dur"),
        pa.array([[1, 4], [1, 0], [2, 0]], type=pa.list_(pa.int64(), 2)),
    )
    table = pa.concat_tables([valid, bad]).slice(2)
    with pytest.raises(
        ArrowValidationError,
        match=("act_dur.*minimum.*2 offending rows.*first offending row index 2"),
    ):
        validate_table(table, "score_measure", index_columns=())


def test_dictionary_nulls():
    """Dictionary values, as well as indices, must obey required nullability."""
    values = pa.DictionaryArray.from_arrays(pa.array([0, 1]), pa.array(["V7", None]))
    with pytest.raises(
        ArrowValidationError,
        match=("value.*null.*1 offending rows.*first offending row index 1"),
    ):
        validate_table(pa.table({"value": values}), "dcml_label", index_columns=())
