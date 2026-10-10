"""Validate DCML harmony labels, parsing, and annotation serialization."""

import json
import re
import signal
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
from bopp.models.v1.payload.dcml_label import DcmlLabelPayload
from bopp.registries.v1 import PAYLOAD_TYPE_REGISTRY
from experiments.dcml import (
    DCML_PATTERN,
    DCML_REGEX,
    MS3_DCML_DOUBLE_REGEX,
    portable_pattern,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas/v1/payload/dcml_label.json"
VALID_LABELS = [
    "F.I",
    "V65/vi",
    "#viio7(6)/vi",
    r"V\\",
    "I|PAC}",
    "@none",
    "V-I",
    "ii6(11#7b6)",
    "V[I6",
    "V2]",
    "I|HC.SPC",
    "Ger6",
    "",
]
INVALID_LABELS = ["H7", "V7(9", "I|XYZ", "Ger65 ", "C.I.V.x", "VIII"]
INVALID_VALUES = [*INVALID_LABELS, None, 7, 2.5, True, [], {}]


def create_annotation(labels):
    """Build an annotation with one fractional quarter interval per label."""
    return bopp.create(
        media_id="test:dcml_label",
        payload_kind="dcml_label",
        extent_kind="quarters_interval.fraction",
        value=labels,
        quarter=[[i, 1] for i in range(len(labels))],
        duration=[[1, 1] for _ in labels],
    )


def assert_annotation(annotation, labels):
    """Check labels, media identity, and extents after validation."""
    assert bopp.validate(annotation)
    assert annotation.media_id == "test:dcml_label"
    assert isinstance(annotation.payload, DcmlLabelPayload)
    assert annotation.payload.value == labels
    assert [list(q) for q in annotation.extent.quarter] == [
        [i, 1] for i in range(len(labels))
    ]
    assert [list(d) for d in annotation.extent.duration] == [[1, 1] for _ in labels]


def test_schema_pattern():
    """The schema uses exactly the portable expression from the parsing helper."""
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["properties"]["value"]["items"]["pattern"] == DCML_PATTERN
    assert DCML_PATTERN == portable_pattern(DCML_REGEX)
    assert len(DCML_PATTERN) == 1222
    assert "(?P<" not in DCML_PATTERN
    assert r"\#" not in DCML_PATTERN
    assert not re.search(r"\s", DCML_PATTERN)
    assert DCML_PATTERN.startswith("^")
    assert r"\d" not in DCML_PATTERN
    assert DCML_PATTERN.endswith(r"(?![\s\S])")


def test_schema_description_credits_ms3():
    """The schema description credits ms3 and its GPL v3 licence."""
    schema = json.loads(SCHEMA_PATH.read_text())
    description = schema["description"]
    assert "ms3" in description
    assert "GPL" in description


def test_named_groups():
    """The verbose reference remains useful for parsing harmonic components."""
    assert DCML_REGEX.flags & re.VERBOSE
    match = DCML_REGEX.fullmatch("V65/vi|PAC")
    assert match is not None
    assert match.group("numeral", "figbass", "relativeroot", "cadence") == (
        "V",
        "65",
        "vi",
        "PAC",
    )
    alternative = DCML_REGEX.fullmatch("V-I")
    assert alternative is not None
    assert alternative.group("numeral", "numeral2") == ("V", "I")


def test_ms3_regex_agreement():
    """DCML_REGEX and ms3's reference agree on samples, including groupdict()."""
    for label in [*VALID_LABELS, *INVALID_LABELS]:
        ours = DCML_REGEX.fullmatch(label)
        ref = MS3_DCML_DOUBLE_REGEX.fullmatch(label)
        assert (ours is not None) == (ref is not None), label
        if ours is not None:
            assert ours.groupdict() == ref.groupdict()


def test_deambiguated_numeral_chain_rejects_run_on():
    """Slash-free numeral run-ons that ms3 accepts are rejected by design."""
    label = "IIIIII[II"
    assert MS3_DCML_DOUBLE_REGEX.fullmatch(label) is not None
    assert DCML_REGEX.fullmatch(label) is None


class _Timeout(Exception):
    """Raised by the SIGALRM handler used to guard against catastrophic backtracking."""


@pytest.mark.parametrize("label", ["III/" * 12, "V/" * 30 + "x"])
def test_no_catastrophic_backtracking(label):
    """Labels that are exponential for ms3's pattern validate (accepted or not) in well under a second."""

    def _on_alarm(signum, frame):
        raise _Timeout

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(3)
    try:
        annotation = create_annotation([label])
        try:
            bopp.validate(annotation)
        except BoppValidationError:
            pass
    except _Timeout:
        pytest.fail(f"validating {label!r} did not finish in 3 s: catastrophic backtracking")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("label", ["V65\n", "V(٣)", "I|PAC.\r"])
def test_ecma_divergent_constructs_rejected(label):
    """\\d, ., and $ mean different things in Python re and ECMA-262; pin the Python side down."""
    assert re.fullmatch(DCML_PATTERN, label) is None


@pytest.mark.parametrize("label", ["-", "V-", "|PAC", "{", "}{"])
def test_degenerate_labels_accepted(label):
    """The pattern is a syntax filter, not a well-formedness check; ms3 accepts these too."""
    assert MS3_DCML_DOUBLE_REGEX.fullmatch(label) is not None
    assert DCML_REGEX.fullmatch(label) is not None
    assert re.fullmatch(DCML_PATTERN, label) is not None
    assert_annotation(create_annotation([label]), [label])


def test_portable_pattern_rejects_unflattenable_verbose_source():
    """Whitespace inside a character class or an unescaped '#' would be corrupted by flattening."""
    with pytest.raises(ValueError):
        portable_pattern(re.compile(r"[a b]", re.VERBOSE))
    with pytest.raises(ValueError):
        portable_pattern(re.compile(r"abc # comment", re.VERBOSE))
    # The actual DCML pattern passes: ms3 escapes '#' as '\#' and uses no
    # character-class whitespace.
    portable_pattern(DCML_REGEX)


def test_payload_registry():
    """The DCML tag resolves to the generated payload class."""
    assert PAYLOAD_TYPE_REGISTRY["dcml_label"] is DcmlLabelPayload


@pytest.mark.parametrize("label", VALID_LABELS)
def test_valid_labels(label):
    """Accept the reference syntax through both annotation and msgspec checks."""
    assert DCML_REGEX.fullmatch(label) is not None
    assert re.fullmatch(DCML_PATTERN, label) is not None
    assert_annotation(create_annotation([label]), [label])
    decoded = msgspec.json.decode(
        msgspec.json.encode({"payload_type": "dcml_label", "value": [label]}),
        type=DcmlLabelPayload,
    )
    assert decoded.value == [label]


def test_empty_label():
    """All reference components are optional, so an empty label is valid."""
    assert DCML_REGEX.fullmatch("") is not None
    assert re.fullmatch(DCML_PATTERN, "") is not None
    assert_annotation(create_annotation([""]), [""])


@pytest.mark.parametrize("value", INVALID_VALUES)
def test_invalid_values(value):
    """Validation, unlike construction, rejects malformed labels and types."""
    annotation = create_annotation([value])
    with pytest.raises(BoppValidationError):
        bopp.validate(annotation)
    with pytest.raises(msgspec.ValidationError, match=r"\$\.value\[0\]"):
        msgspec.json.decode(
            msgspec.json.encode({"payload_type": "dcml_label", "value": [value]}),
            type=DcmlLabelPayload,
        )


def test_missing_value_column():
    """Decoding requires the single value column."""
    with pytest.raises(msgspec.ValidationError, match="value"):
        msgspec.json.decode(b'{"payload_type":"dcml_label"}', type=DcmlLabelPayload)


@pytest.mark.parametrize("label", [*VALID_LABELS, *INVALID_LABELS])
def test_json_schema_pattern_agreement(label):
    """JSON Schema and the named reference agree on valid and invalid samples."""
    jsonschema = pytest.importorskip("jsonschema")
    item_schema = json.loads(SCHEMA_PATH.read_text())["properties"]["value"]["items"]
    jsonschema.Draft202012Validator.check_schema(item_schema)
    valid = jsonschema.Draft202012Validator(item_schema).is_valid(label)
    assert valid == (label in VALID_LABELS)
    assert valid == (DCML_REGEX.search(label) is not None)


@pytest.mark.parametrize(
    ("save", "load", "suffix"),
    [
        (save_bopp_json, load_bopp_json, "json"),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
    ],
    ids=["json", "msgpack"],
)
def test_annotation_roundtrip(tmp_path, save, load, suffix):
    """JSON and MessagePack preserve empty labels, backslashes, and ids."""
    annotation = create_annotation(VALID_LABELS)
    path = tmp_path / f"dcml_label.{suffix}"
    save(annotation, path)
    loaded = load(path)
    assert_annotation(loaded, VALID_LABELS)
    assert loaded.id == annotation.id


def test_csv_roundtrip(tmp_path):
    """CSV preserves nonempty labels and both backslashes, but drops the id."""
    labels = [label for label in VALID_LABELS if label]
    annotation = create_annotation(labels)
    path = tmp_path / "dcml_label.csv"
    save_bopp_csv(annotation, path)
    loaded = load_bopp_csv(path)
    assert_annotation(loaded, labels)
    assert loaded.payload.value[3] == "V" + "\\" * 2
    assert annotation.id is not msgspec.UNSET
    assert loaded.id is msgspec.UNSET


def test_csv_empty_label(tmp_path):
    """The current CSV reader turns empty cells into floats, which are rejected."""
    path = tmp_path / "dcml_label_empty.csv"
    save_bopp_csv(create_annotation([""]), path)
    with pytest.raises(msgspec.ValidationError, match=r"\$\.payload\.value\[0\]"):
        load_bopp_csv(path)
