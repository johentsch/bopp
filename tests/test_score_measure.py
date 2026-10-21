"""Validate measure columns, DLC conversion and serialization behavior."""

import json
from fractions import Fraction
from pathlib import Path

import msgspec
import pytest

import bopp
from bopp.exceptions import BoppArgumentError, BoppValidationError
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.models.v1.payload.score_measure import ScoreMeasurePayload
from bopp.models.v1.payloads import AnyPayload
from bopp.registries.v1 import COMPLEX_FIELDS_REGISTRY, PAYLOAD_TYPE_REGISTRY
from experiments.dlc_measures import measures_to_score_measure, read_measures_tsv

FIXTURE = (
    Path(__file__).parent
    / "experiments/data/dlc_measures_chopin_mazurkas_BI163op67-4.tsv"
)
SCHEMA = Path(__file__).resolve().parents[1] / "schemas/v1/payload/score_measure.json"
MINIMAL = {
    "mc": [1, 2, 3],
    "mn": [0, 1, 1],
    "keysig": [-7, 0, 7],
    "timesig": ["3/4", "12/16", "33/32"],
    "act_dur": [[1, 4], [3, 4], [0, 1]],
    "mc_offset": [[1, 2], [0, 1], [3, 4]],
    "next": [[2], [1, 3], [3, -1]],
}
NONULL = {
    **MINIMAL,
    "numbering_offset": [-1, 0, 1],
    "volta": [1, 2, 3],
    "dont_count": [True, True, True],
    "barline": ["double", "end-start-repeat", "end"],
    "breaks": ["line", "page & section", "section & page"],
    "repeats": ["firstMeasure", "startend", "lastMeasure"],
    "markers": ["fine", "codab", "4e"],
    "jump_bwd": ["start", "segno", "start"],
    "jump_fwd": ["coda", "2e", "4e"],
    "play_until": ["fine", "segno", "/"],
}
FULL = {
    name: values if name in MINIMAL else [None, *values[1:]]
    for name, values in NONULL.items()
}
QUARTERS = {
    "quarter": [[0, 1], [1, 1], [4, 1]],
    "duration": [[1, 1], [3, 1], [0, 1]],
}
INVALID = [
    ("mc", 0),
    ("mc", "1"),
    ("mn", 1.5),
    ("keysig", -8),
    ("keysig", 8),
    ("timesig", "3/0"),
    ("timesig", "3/4 "),
    ("timesig", "0/4"),
    ("act_dur", [-1, 4]),
    ("act_dur", [1, 0]),
    ("mc_offset", [1, 0]),
    ("next", []),
    ("next", [-2]),
    ("next", ["2"]),
    ("volta", 0),
    ("numbering_offset", 1.5),
    ("dont_count", 1),
    *[(name, None) for name in MINIMAL],
    *[
        (name, 1)
        for name in FULL
        if name not in {*MINIMAL, "numbering_offset", "volta", "dont_count"}
    ],
]


def wire(value):
    """Normalize fraction tuples and lists to the same JSON representation."""
    return msgspec.json.decode(msgspec.json.encode(value))


def create_annotation(columns):
    """Build a three-measure annotation for validation and I/O tests."""
    return bopp.create(
        media_id="test:score_measure",
        payload_kind="score_measure",
        extent_kind="quarters_interval.fraction",
        **QUARTERS,
        **columns,
    )


def test_registry():
    """Register the payload and nested columns used by CSV serialization."""
    assert PAYLOAD_TYPE_REGISTRY["score_measure"] is ScoreMeasurePayload
    fields = COMPLEX_FIELDS_REGISTRY["payload_type"]["score_measure"]
    assert {"act_dur", "mc_offset", "next"} <= set(fields)


@pytest.mark.parametrize("columns", [MINIMAL, FULL], ids=["minimal", "full"])
def test_valid_annotation(columns):
    """Required and optional columns validate, including null and unset items."""
    annotation = create_annotation(columns)
    assert bopp.validate(annotation)
    assert wire(annotation.payload) == {"payload_type": "score_measure", **columns}
    decoded = msgspec.json.decode(
        msgspec.json.encode(annotation.payload), type=AnyPayload
    )
    assert isinstance(decoded, ScoreMeasurePayload)
    assert wire(decoded) == wire(annotation.payload)
    assert all(isinstance(value, list) for value in decoded.next)
    assert all(type(value) is int for values in decoded.next for value in values)
    for name in FULL.keys() - columns.keys():
        assert getattr(decoded, name) is msgspec.UNSET


@pytest.mark.parametrize(("name", "value"), INVALID)
def test_invalid_columns(name, value):
    """Validation enforces item bounds and types, including nested next lists."""
    columns = {**FULL, name: [value, *FULL[name][1:]]}
    annotation = create_annotation(columns)
    with pytest.raises(BoppValidationError):
        bopp.validate(annotation)


@pytest.mark.parametrize("name", ["payload_type", *MINIMAL])
def test_missing_required_column(name):
    """Typed decoding requires the discriminator and every required column."""
    document = {"payload_type": "score_measure", **MINIMAL}
    del document[name]
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(msgspec.json.encode(document), type=AnyPayload)


@pytest.fixture
def frame():
    """Use the bundled measure table without needing the DLC checkout."""
    return read_measures_tsv(FIXTURE)


def test_reader_fixture(frame):
    """Preserve the full table and use all-ending starts for volta measures."""
    assert len(frame) == 83
    assert all(isinstance(value, str) for value in frame.to_numpy().flat)
    annotation = measures_to_score_measure(frame)
    assert bopp.validate(annotation)
    assert annotation.media_id == "dlc:chopin_mazurkas/BI163op67-4"
    assert annotation.payload.mc == list(range(1, 84))
    for values in (
        annotation.payload.act_dur,
        annotation.payload.mc_offset,
        annotation.extent.quarter,
        annotation.extent.duration,
    ):
        assert all(isinstance(value, list) for value in values)
    assert annotation.payload.next[0] == [2]
    assert annotation.payload.next[-1] == [-1]
    for mc, start, following in [
        (33, 94, 18),
        (34, 97, 35),
        (50, 145, 35),
        (51, 148, 52),
    ]:
        i = mc - 1
        assert annotation.extent.quarter[i] == [start, 1]
        assert annotation.extent.duration[i] == [3, 1]
        assert annotation.payload.next[i] == [following]
    assert frame.iloc[32]["quarterbeats"] == ""
    assert annotation.payload.dont_count[:2] == [True, None]
    assert annotation.payload.volta[31:34] == [None, 1, 2]
    for name in FULL.keys() - MINIMAL.keys():
        values = getattr(annotation.payload, name)
        if name not in frame or (frame[name] == "").all():
            assert values is msgspec.UNSET
        else:
            assert [v is None for v in values] == (frame[name] == "").tolist()
    # Reconstruct every source payload column independently of the reader casts.
    for name in frame.columns:
        if name not in FULL:
            continue
        values = getattr(annotation.payload, name)
        if values is msgspec.UNSET:
            assert (frame[name] == "").all()
            continue
        restored = []
        for value in values:
            if value is None:
                restored.append("")
            elif name in {"act_dur", "mc_offset"}:
                restored.append(str(Fraction(*value)))
            elif name == "next":
                restored.append(", ".join(map(str, value)))
            elif name == "dont_count":
                restored.append("1" if value else "0")
            else:
                restored.append(str(value))
        assert restored == frame[name].tolist()


def test_reader_conversions_and_fallback(frame):
    """Reduce fractions, parse branches and preserve optional text verbatim."""
    frame = frame.iloc[:2].copy()
    frame.loc[0, "act_dur"] = "2/8"
    frame.loc[0, "mc_offset"] = "2/4"
    frame.loc[0, "next"] = "9, -1"
    frame.loc[0, "quarterbeats_all_endings"] = ""
    frame.loc[0, "quarterbeats"] = "1/3"
    frame["markers"] = [" NA ", ""]
    annotation = measures_to_score_measure(frame, media_id="test:override")
    assert bopp.validate(annotation)
    assert annotation.media_id == "test:override"
    assert annotation.payload.next[0] == [9, -1]
    assert annotation.payload.act_dur[0] == [1, 4]
    assert annotation.payload.mc_offset[0] == [1, 2]
    assert annotation.payload.markers == [" NA ", None]
    assert annotation.extent.quarter[0] == [1, 3]
    assert annotation.extent.duration[0] == [1, 1]
    fallback = measures_to_score_measure(frame.drop(columns="quarterbeats_all_endings"))
    assert fallback.extent.quarter == annotation.extent.quarter
    all_endings = measures_to_score_measure(frame.iloc[1:].drop(columns="quarterbeats"))
    assert all_endings.extent.quarter == annotation.extent.quarter[1:]


@pytest.mark.parametrize("name", MINIMAL)
@pytest.mark.parametrize("mode", ["absent", "one_empty", "all_empty"])
def test_reader_required_column_errors(frame, name, mode):
    """Absent and empty required values report their source column."""
    if mode == "absent":
        frame = frame.drop(columns=name)
    elif mode == "one_empty":
        frame.loc[0, name] = ""
    else:
        frame[name] = ""
    with pytest.raises(BoppArgumentError, match=name):
        measures_to_score_measure(frame)


@pytest.mark.parametrize("name", ["corpus", "piece"])
@pytest.mark.parametrize("mode", ["absent", "multiple", "empty_frame"])
def test_reader_identity_errors(frame, name, mode):
    """Require both identity columns and exactly one corpus/piece pair."""
    if mode == "absent":
        frame = frame.drop(columns=name)
    elif mode == "multiple":
        frame.loc[0, name] = "another"
    else:
        frame = frame.iloc[:0]
    with pytest.raises(BoppArgumentError, match="corpus/piece"):
        measures_to_score_measure(frame)


def test_reader_unknown_column(frame):
    """Unknown empty columns are harmless, but non-empty data cannot be lost."""
    frame["unsupported"] = ""
    assert bopp.validate(measures_to_score_measure(frame))
    frame.loc[0, "unsupported"] = "value"
    with pytest.raises(BoppArgumentError, match="unsupported"):
        measures_to_score_measure(frame)


@pytest.mark.parametrize(
    ("name", "value"),
    [("mc", "bad"), ("act_dur", "1/0"), ("next", "2, bad"), ("dont_count", "0")],
)
def test_reader_invalid_conversion(frame, name, value):
    """Conversion failures identify the offending source column."""
    frame.loc[0, name] = value
    with pytest.raises(BoppArgumentError, match=name):
        measures_to_score_measure(frame)


@pytest.mark.parametrize("offset", [[-9, 4], [-7, 8]])
def test_negative_mc_offset(offset):
    """Offsets are signed while actual durations remain nonnegative."""
    assert bopp.validate(create_annotation({**MINIMAL, "mc_offset": [offset] * 3}))
    with pytest.raises(BoppValidationError):
        bopp.validate(create_annotation({**MINIMAL, "act_dur": [offset] * 3}))


def test_reader_missing_start_columns(frame):
    """At least one quarterbeats column is required to construct the extent."""
    frame = frame.drop(columns=["quarterbeats", "quarterbeats_all_endings"])
    with pytest.raises(BoppArgumentError, match="quarterbeats.*required for the extent"):
        measures_to_score_measure(frame)


def test_reader_missing_start(frame):
    """A measure with neither start coordinate reports the fallback column."""
    frame.loc[0, ["quarterbeats", "quarterbeats_all_endings"]] = ""
    with pytest.raises(BoppArgumentError, match="quarterbeats"):
        measures_to_score_measure(frame)


@pytest.mark.parametrize("source", ["minimal", "full", "fixture"])
@pytest.mark.parametrize(
    ("save", "load", "suffix"),
    [
        (save_bopp_json, load_bopp_json, "json"),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
    ],
)
def test_roundtrip(tmp_path, frame, source, save, load, suffix):
    """JSON and MessagePack retain ids, all payload values and extents."""
    annotation = (
        measures_to_score_measure(frame)
        if source == "fixture"
        else create_annotation(MINIMAL if source == "minimal" else FULL)
    )
    assert bopp.validate(annotation)
    path = tmp_path / f"score_measure.{suffix}"
    save(annotation, path)
    loaded = load(path)
    assert bopp.validate(loaded)
    assert loaded.id == annotation.id
    assert wire(loaded) == wire(annotation)


@pytest.mark.parametrize("columns", [MINIMAL, FULL], ids=["minimal", "full"])
def test_csv_roundtrip(tmp_path, columns):
    """CSV preserves nested next lists and nullable columns, but omits the id."""
    annotation = create_annotation(columns)
    path = tmp_path / "score_measure.csv"
    save_bopp_csv(annotation, path)
    loaded = load_bopp_csv(path)
    assert bopp.validate(loaded)
    assert wire(loaded.payload) == wire(annotation.payload)
    assert wire(loaded.extent) == wire(annotation.extent)
    assert loaded.media_id == annotation.media_id
    assert annotation.id is not msgspec.UNSET
    assert loaded.id is msgspec.UNSET


@pytest.mark.parametrize("name", ["numbering_offset", "volta", "barline", "dont_count"])
def test_csv_roundtrip_nulls(tmp_path, name):
    """CSV preserves null integers, strings and booleans."""
    annotation = create_annotation({**MINIMAL, name: FULL[name]})
    assert bopp.validate(annotation)
    path = tmp_path / f"{name}.csv"
    save_bopp_csv(annotation, path)
    loaded = load_bopp_csv(path)
    assert bopp.validate(loaded)
    assert wire(loaded.payload) == wire(annotation.payload)


@pytest.mark.parametrize(
    ("name", "valid", "invalid"),
    [
        ("timesig", ["3/4", "6/8", "12/16", "33/32"], ["3/0", "3/4 ", "0/4", 3, None]),
        ("next", [[2], [1, 9], [9, -1]], [[], [-2], [None], ["2"], 2, None]),
    ],
)
def test_jsonschema_item_agreement(name, valid, invalid):
    """JSON Schema and generated msgspec constraints agree on item values."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema["properties"][name]["items"])
    for value in valid:
        validator.validate(value)
        assert bopp.validate(create_annotation({**MINIMAL, name: [value] * 3}))
    for value in invalid:
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(value)
        with pytest.raises(BoppValidationError):
            bopp.validate(create_annotation({**MINIMAL, name: [value] * 3}))
