"""Validate the bopp-score-ext plugin: registration, conversions, and I/O round trips."""

from pathlib import Path

import msgspec
import pytest

import bopp
import bopp.extensions
from bopp.exceptions import BoppArgumentError, BoppValidationError
from bopp.io import (
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_json,
    save_bopp_msgpack,
)

try:
    import bopp_score_ext
    from bopp_score_ext.convert import from_ext, to_ext
    from bopp_score_ext.models import ScoreNote

    _PLUGIN_AVAILABLE = bopp_score_ext.EXT_SCHEMA in bopp.extensions.get_extensions()
except ImportError:
    _PLUGIN_AVAILABLE = False

# A plain module-level skip (pytest.importorskip / pytest.skip(allow_module_level=True))
# would stop collection before any test item exists, which pytest reports as exit code 5
# ("no tests ran") rather than a normal, zero-exit skipped run. Marking every test with
# skipif instead keeps each test collected (so pytest exits 0) while still skipping all of
# them cleanly when the plugin is absent.
pytestmark = pytest.mark.skipif(
    not _PLUGIN_AVAILABLE, reason="bopp-score-ext plugin is not installed or not registered"
)

from experiments.dlc_notes import notes_to_score_note, read_notes_tsv


def _to_plain(value):
    """Recursively convert tuples (e.g. decoded fraction pairs) into lists for comparison."""
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    return value


FIXTURE = (
    Path(__file__).resolve().parent
    / "data"
    / "dlc_notes_chopin_mazurkas_BI163op67-4.tsv"
)

EXT_ROWS = [
    {"midi": 60, "tpc": 0, "gracenote": "acciaccatura", "nominal_duration": [1, 4], "scalar": [1, 1]},
    {"midi": 62, "tpc": 2, "tied": 1, "nominal_duration": [1, 8], "scalar": [1, 1]},
    {"midi": 64, "tpc": 4, "nominal_duration": [1, 2], "scalar": [2, 3]},
]
EXT_EXTENT = {
    "extent_kind": "quarters_interval.fraction",
    "quarter": [[0, 1], [1, 4], [1, 2]],
    "duration": [[1, 4], [1, 4], [1, 2]],
}

SCORE_NOTE_COLUMNS = {
    "midi": [60, 62, 64],
    "tpc": [0, 2, 4],
    "tied": [None, 1, None],
    "gracenote": ["acciaccatura", None, None],
    "nominal_duration": [[1, 4], [1, 8], [1, 2]],
    "scalar": [[1, 1], [1, 1], [2, 3]],
}


def make_ext_annotation(*, resolve_ext=True):
    """Build the hand-written three-row ext annotation used throughout this module."""
    return bopp.create(
        media_id="track:1",
        payload_kind="ext",
        ext_schema=bopp_score_ext.EXT_SCHEMA,
        value=[dict(row) for row in EXT_ROWS],
        resolve_ext=resolve_ext,
        **EXT_EXTENT,
    )


def make_score_note_annotation():
    """Build the equivalent columnar score_note annotation."""
    return bopp.create(
        media_id="track:1",
        payload_kind="score_note",
        **SCORE_NOTE_COLUMNS,
        **EXT_EXTENT,
    )


# a. Registry resolution and struct shape
# -----------------------------------------------------------------------------


def test_registry_resolves_score_note():
    """The entry point resolves EXT_SCHEMA to the ScoreNote struct."""
    assert bopp.extensions.get_extensions()[bopp_score_ext.EXT_SCHEMA] is ScoreNote


def test_score_note_fields_match_schema():
    """ScoreNote has exactly the score_note schema's columns, midi/tpc required."""
    fields = {f.name: f for f in msgspec.structs.fields(ScoreNote)}
    expected = {
        "midi", "tpc", "name", "octave", "staff", "voice", "mc", "mn", "measure",
        "tied", "gracenote", "nominal_duration", "scalar", "chord_id", "tremolo",
        "volta", "tuning", "anchor",
    }
    assert set(fields) == expected
    assert fields["midi"].required
    assert fields["tpc"].required
    for name in expected - {"midi", "tpc"}:
        assert not fields[name].required
        assert fields[name].default is msgspec.UNSET


def test_score_note_struct_config():
    """ScoreNote forbids unknown fields and omits defaults on serialization."""
    config = ScoreNote.__struct_config__
    assert config.forbid_unknown_fields is True
    assert config.omit_defaults is True


# b. Hand-written ext annotation
# -----------------------------------------------------------------------------


def test_create_ext_resolved_and_unresolved():
    """create(resolve_ext=True/False) agree on id and content, differing only in row type."""
    resolved = make_ext_annotation(resolve_ext=True)
    unresolved = make_ext_annotation(resolve_ext=False)

    assert all(isinstance(row, ScoreNote) for row in resolved.payload.value)
    assert all(isinstance(row, dict) for row in unresolved.payload.value)
    assert resolved.id == unresolved.id
    assert _to_plain(msgspec.to_builtins(resolved.payload.value)) == EXT_ROWS


# c. to_ext / from_ext against the columnar annotation
# -----------------------------------------------------------------------------


def test_to_ext_matches_hand_written_rows():
    """to_ext reproduces the hand-written rows exactly, without mutating the input."""
    score_note_ann = make_score_note_annotation()
    snapshot = msgspec.to_builtins(score_note_ann)

    ext_ann = to_ext(score_note_ann)

    assert msgspec.to_builtins(score_note_ann) == snapshot
    assert [_to_plain(msgspec.to_builtins(row)) for row in ext_ann.payload.value] == EXT_ROWS
    for row in ext_ann.payload.value:
        for value in msgspec.to_builtins(row).values():
            assert value is not None
    assert ext_ann.extent == score_note_ann.extent
    assert ext_ann.media_id == score_note_ann.media_id


def test_to_ext_id_matches_direct_ext_annotation():
    """to_ext of the columnar annotation has the same id as the directly built ext one."""
    score_note_ann = make_score_note_annotation()
    ext_ann = to_ext(score_note_ann)
    direct_ext_ann = make_ext_annotation()
    assert ext_ann.id == direct_ext_ann.id


def test_from_ext_round_trip():
    """from_ext(to_ext(ann)) reproduces ann's id and payload."""
    score_note_ann = make_score_note_annotation()
    ext_ann = to_ext(score_note_ann)
    round_tripped = from_ext(ext_ann)

    assert round_tripped.id == score_note_ann.id
    assert msgspec.to_builtins(round_tripped.payload) == msgspec.to_builtins(score_note_ann.payload)


def test_from_ext_accepts_unresolved_rows():
    """from_ext also accepts an ext annotation whose rows are still plain dicts."""
    unresolved = make_ext_annotation(resolve_ext=False)
    round_tripped = from_ext(unresolved)
    assert round_tripped.id == make_score_note_annotation().id


# d. JSON and msgpack round trips
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("save", "load", "suffix"),
    [
        (save_bopp_json, load_bopp_json, "json"),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack"),
    ],
    ids=["json", "msgpack"],
)
def test_ext_annotation_io_roundtrip(tmp_path, save, load, suffix):
    """JSON and MsgPack round trips preserve ids and resolve rows as configured."""
    ann = make_ext_annotation(resolve_ext=False)
    path = tmp_path / f"ext.{suffix}"
    save(ann, path)

    loaded_resolved = load(path, resolve_ext=True)
    assert all(isinstance(row, ScoreNote) for row in loaded_resolved.payload.value)

    loaded_unresolved = load(path, resolve_ext=False)
    assert all(isinstance(row, dict) for row in loaded_unresolved.payload.value)

    assert loaded_resolved.id == ann.id == loaded_unresolved.id
    assert bopp.core.validate_annotation_id(loaded_resolved)
    assert bopp.core.validate_annotation_id(loaded_unresolved)


# e. Missing plugin simulated in-process
# -----------------------------------------------------------------------------


def test_missing_extension_warns_and_keeps_dicts(tmp_path):
    """Clearing the registry simulates a missing plugin: warn, keep dicts, keep the id."""
    ann = make_ext_annotation(resolve_ext=False)
    path = tmp_path / "ext.json"
    save_bopp_json(ann, path)

    try:
        bopp.extensions.REGISTRY.clear()
        with pytest.warns(UserWarning, match="No extension registered"):
            loaded = load_bopp_json(path)
        assert all(isinstance(row, dict) for row in loaded.payload.value)
        assert loaded.id == ann.id
    finally:
        bopp.extensions.reset_extensions()

    assert bopp_score_ext.EXT_SCHEMA in bopp.extensions.get_extensions()


# f. Validation
# -----------------------------------------------------------------------------

INVALID_ROWS = [
    ({"midi": 128, "tpc": 0}, "midi_above_range"),
    ({"midi": 60, "tpc": 0, "unknown": 1}, "unknown_field"),
    ({"midi": "60", "tpc": 0}, "midi_string"),
    ({"midi": 60, "tpc": 0, "tied": 2}, "tied_above_range"),
    ({"midi": 60}, "missing_tpc"),
    ({"midi": 60, "tpc": 0, "scalar": [1, 0]}, "scalar_zero_denominator"),
]


@pytest.mark.parametrize(("row", "reason"), INVALID_ROWS, ids=[r for _, r in INVALID_ROWS])
def test_invalid_row_rejected_by_create(row, reason):
    """create(..., resolve_ext=True) raises on every invalid row shape."""
    with pytest.raises((msgspec.ValidationError, BoppValidationError)):
        bopp.create(
            media_id="track:invalid",
            payload_kind="ext",
            ext_schema=bopp_score_ext.EXT_SCHEMA,
            value=[row],
            resolve_ext=True,
            extent_kind="quarters_interval.fraction",
            quarter=[[0, 1]],
            duration=[[1, 1]],
        )


@pytest.mark.parametrize(("row", "reason"), INVALID_ROWS, ids=[r for _, r in INVALID_ROWS])
def test_invalid_row_rejected_by_from_ext(row, reason):
    """from_ext on an unresolved annotation raises on every invalid row shape."""
    ann = bopp.create(
        media_id="track:invalid",
        payload_kind="ext",
        ext_schema=bopp_score_ext.EXT_SCHEMA,
        value=[row],
        resolve_ext=False,
        extent_kind="quarters_interval.fraction",
        quarter=[[0, 1]],
        duration=[[1, 1]],
    )
    with pytest.raises((msgspec.ValidationError, BoppValidationError)):
        from_ext(ann)


def test_to_ext_rejects_non_score_note_payload():
    """to_ext raises BoppArgumentError on a payload that is not score_note."""
    ann = bopp.create(media_id="track:tags", payload_kind="tag_open", value=["a", "b"])
    with pytest.raises(BoppArgumentError):
        to_ext(ann)


def test_from_ext_rejects_other_ext_schema():
    """from_ext raises BoppArgumentError on an ext annotation with a different ext_schema."""
    ann = bopp.create(
        media_id="track:other",
        payload_kind="ext",
        ext_schema="org.example.other:v1",
        value=[{"anything": 1}],
        resolve_ext=False,
    )
    with pytest.raises(BoppArgumentError):
        from_ext(ann)


# g. Fixture-based reader round trip
# -----------------------------------------------------------------------------


def test_fixture_notes_to_score_note():
    """read_notes_tsv + notes_to_score_note match hand-derived expectations."""
    df = read_notes_tsv(FIXTURE)
    ann = notes_to_score_note(df)

    assert ann.media_id == "dlc:chopin_mazurkas/BI163op67-4"
    assert bopp.validate(ann)

    payload = ann.payload
    assert payload.midi[:3] == [76, 45, 76]
    assert payload.tpc[:3] == [4, 3, 4]
    assert payload.name[:3] == ["E5", "A2", "E5"]
    assert payload.octave[:3] == [5, 2, 5]
    assert payload.staff[:3] == [1, 2, 1]
    assert payload.voice[:3] == [1, 1, 1]
    assert payload.mc[:3] == [1, 2, 2]
    assert payload.mn[:3] == [0, 1, 1]
    assert payload.chord_id[:3] == [0, 5, 1]
    assert [list(v) for v in payload.nominal_duration[:3]] == [[1, 4], [1, 4], [1, 4]]
    assert [list(v) for v in payload.scalar[:3]] == [[1, 1], [1, 1], [3, 2]]
    assert payload.tied[:3] == [1, None, -1]

    extent = ann.extent
    assert [list(v) for v in extent.quarter[:3]] == [[0, 1], [1, 1], [1, 1]]
    assert [list(v) for v in extent.duration[:3]] == [[1, 1], [1, 1], [3, 2]]

    assert sum(1 for v in payload.tied if v is not None) == 16
    assert sum(1 for v in payload.volta if v is not None) == 32
    assert {v for v in payload.volta if v is not None} == {1, 2}
    assert sum(1 for v in payload.gracenote if v is not None) == 2
    assert all(v == "acciaccatura" for v in payload.gracenote if v is not None)

    assert payload.tremolo is msgspec.UNSET
    assert payload.tuning is msgspec.UNSET
    assert payload.measure is msgspec.UNSET
    assert payload.anchor is msgspec.UNSET


def test_fixture_to_ext_and_round_trip(tmp_path):
    """to_ext of the fixture annotation matches the first two expected rows and round-trips."""
    ann = notes_to_score_note(read_notes_tsv(FIXTURE))
    ext_ann = to_ext(ann)

    assert len(ext_ann.payload.value) == 822
    assert _to_plain(msgspec.to_builtins(ext_ann.payload.value[0])) == {
        "midi": 76, "tpc": 4, "name": "E5", "octave": 5, "staff": 1, "voice": 1,
        "mc": 1, "mn": 0, "tied": 1, "nominal_duration": [1, 4], "scalar": [1, 1],
        "chord_id": 0,
    }
    assert "tied" not in msgspec.to_builtins(ext_ann.payload.value[1])

    assert from_ext(ext_ann).id == ann.id

    path = tmp_path / "fixture_ext.json"
    save_bopp_json(ext_ann, path)
    loaded = load_bopp_json(path)
    assert loaded.id == ext_ann.id


# h. from_ext rejects inconsistent non-nullable columns
# -----------------------------------------------------------------------------


def test_from_ext_rejects_partial_nonnullable_column():
    """from_ext names the offending column when a non-nullable field is present on some rows only."""
    ann = bopp.create(
        media_id="track:partial",
        payload_kind="ext",
        ext_schema=bopp_score_ext.EXT_SCHEMA,
        value=[{"midi": 60, "tpc": 0, "name": "C4"}, {"midi": 61, "tpc": 1}],
        resolve_ext=False,
        extent_kind="quarters_interval.fraction",
        quarter=[[0, 1], [1, 1]],
        duration=[[1, 1], [1, 1]],
    )
    with pytest.raises(BoppArgumentError, match="name"):
        from_ext(ann)


# i. Explicit nulls are rejected rather than resolving to UNSET
# -----------------------------------------------------------------------------

EXPLICIT_NULL_ROWS = [
    ({"midi": 60, "tpc": 0, "tied": None}, "tied_null"),
    ({"midi": 60, "tpc": 0, "gracenote": None}, "gracenote_null"),
    ({"midi": 60, "tpc": 0, "nominal_duration": None}, "nominal_duration_null"),
]


@pytest.mark.parametrize(("row", "reason"), EXPLICIT_NULL_ROWS, ids=[r for _, r in EXPLICIT_NULL_ROWS])
def test_explicit_null_rejected_by_create(row, reason):
    """An explicit null for an optional field raises rather than resolving to UNSET."""
    with pytest.raises((msgspec.ValidationError, BoppValidationError)):
        bopp.create(
            media_id="track:explicit-null",
            payload_kind="ext",
            ext_schema=bopp_score_ext.EXT_SCHEMA,
            value=[row],
            resolve_ext=True,
            extent_kind="quarters_interval.fraction",
            quarter=[[0, 1]],
            duration=[[1, 1]],
        )


@pytest.mark.parametrize(("row", "reason"), EXPLICIT_NULL_ROWS, ids=[r for _, r in EXPLICIT_NULL_ROWS])
def test_explicit_null_rejected_by_from_ext(row, reason):
    """from_ext on an unresolved annotation also rejects an explicit null."""
    ann = bopp.create(
        media_id="track:explicit-null",
        payload_kind="ext",
        ext_schema=bopp_score_ext.EXT_SCHEMA,
        value=[row],
        resolve_ext=False,
        extent_kind="quarters_interval.fraction",
        quarter=[[0, 1]],
        duration=[[1, 1]],
    )
    with pytest.raises((msgspec.ValidationError, BoppValidationError)):
        from_ext(ann)


def test_explicit_null_load_fails_resolution_not_id_mismatch(tmp_path):
    """
    Reproduces the reviewer's finding: an explicit null must fail at resolution,
    never surface as an 'ID mismatch', and loading unresolved must still work.
    """
    ann = bopp.create(
        media_id="track:explicit-null-io",
        payload_kind="ext",
        ext_schema=bopp_score_ext.EXT_SCHEMA,
        value=[{"midi": 60, "tpc": 0, "tied": None}],
        resolve_ext=False,
        extent_kind="quarters_interval.fraction",
        quarter=[[0, 1]],
        duration=[[1, 1]],
    )
    path = tmp_path / "explicit_null.json"
    save_bopp_json(ann, path)

    with pytest.raises((msgspec.ValidationError, BoppValidationError)) as excinfo:
        load_bopp_json(path)
    assert "ID mismatch" not in str(excinfo.value)

    loaded_unresolved = load_bopp_json(path, resolve_ext=False)
    assert loaded_unresolved.id == ann.id


def test_resolved_row_missing_field_is_unset():
    """A resolved row's absent optional field is UNSET, and is omitted from to_builtins."""
    ann = bopp.create(
        media_id="track:unset",
        payload_kind="ext",
        ext_schema=bopp_score_ext.EXT_SCHEMA,
        value=[{"midi": 60, "tpc": 0}],
        resolve_ext=True,
        extent_kind="quarters_interval.fraction",
        quarter=[[0, 1]],
        duration=[[1, 1]],
    )
    row = ann.payload.value[0]
    assert row.tied is msgspec.UNSET
    assert "tied" not in msgspec.to_builtins(row)
