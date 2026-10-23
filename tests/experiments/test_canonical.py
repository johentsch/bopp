"""Verify canonical DLC facets and exact, type-strict storage comparisons."""

from fractions import Fraction
from pathlib import Path

import msgspec
import pandas as pd
import pytest
from msgspec import inspect as mi

import bopp
from bopp.exceptions import BoppArgumentError
from bopp.registries.v1 import PAYLOAD_TYPE_REGISTRY
from experiments import canonical
from experiments.canonical import (
    DLC_FACETS,
    FACETS,
    annotation_to_frame,
    assert_frames_equal,
    frame_to_annotation,
    frames_equal,
    load_dlc_facet,
)
from experiments.dlc_notes import notes_to_score_note, read_notes_tsv

DATA = Path(__file__).parent / "data"
PIECE = "BI163op67-4"
CORPUS = "chopin_mazurkas"
PRESENT = [
    "notes",
    "measures",
    "labels",
    "control_events.chord",
    "control_events.dynamic",
    "control_events.spanner",
    "control_events.staff_text",
    "control_events.tempo",
]


def source_frame(facet):
    """Read an identity-prefixed fixture without any external checkout."""
    directory = FACETS[facet]["facet_dir"]
    return pd.read_csv(
        DATA / f"dlc_{directory}_{CORPUS}_{PIECE}.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


def source_annotation(facet, source=None):
    """Select the piece annotation for one facet from its reader."""
    result = FACETS[facet]["reader"](source_frame(facet) if source is None else source)
    if facet.startswith("control_events."):
        return result[facet.split(".", 1)[1]]
    return result


def test_facet_registry():
    """Expose exactly the benchmark facets and their source readers."""
    assert set(DLC_FACETS) == {
        "notes",
        "measures",
        "labels",
        *{
            f"control_events.{kind}"
            for kind in (
                "chord",
                "dynamic",
                "spanner",
                "figured_bass",
                "staff_text",
                "system_text",
                "tempo",
            )
        },
    }
    assert FACETS["labels"]["facet_dir"] == "harmonies"


@pytest.mark.parametrize("facet", PRESENT)
def test_annotation_roundtrip(facet):
    """Every fixture facet retains its id, values, types and schema column order."""
    annotation = source_annotation(facet)
    frame = annotation_to_frame(annotation, corpus=CORPUS, piece=PIECE)
    assert len(frame) == len(annotation.extent.quarter)
    assert all(dtype == object for dtype in frame.dtypes)
    fields = mi.type_info(PAYLOAD_TYPE_REGISTRY[FACETS[facet]["payload_kind"]]).fields
    assert list(frame.columns) == [
        "corpus",
        "piece",
        "quarter",
        "duration",
        *(f.name for f in fields),
    ]
    assert all(type(v) is str for v in frame.corpus)
    assert all(type(v) is str for v in frame.piece)
    for name in ("quarter", "duration"):
        assert all(type(v) is Fraction for v in frame[name])
    for name in annotation.payload.__struct_fields__:
        values = getattr(annotation.payload, name)
        if values is msgspec.UNSET:
            assert all(v is None for v in frame[name])
            continue
        for original, cell in zip(values, frame[name], strict=True):
            if original is None:
                assert cell is None
            elif isinstance(original, (tuple, list)):
                if name == "next":
                    assert type(cell) is tuple
                    assert all(type(v) is int for v in cell)
                    assert cell == tuple(original)
                else:
                    assert type(cell) is Fraction
                    assert cell == Fraction(*original)
            else:
                assert type(cell) is type(original)
                assert cell == original
    restored = frame_to_annotation(
        frame,
        payload_kind=FACETS[facet]["payload_kind"],
        extent_kind="quarters_interval.fraction",
        media_id=annotation.media_id,
    )
    assert bopp.validate(restored)
    assert restored.id == annotation.id
    assert_frames_equal(
        frame, annotation_to_frame(restored, corpus=CORPUS, piece=PIECE)
    )
    assert frames_equal(frame, frame.copy()) == []


def test_signed_fractions_and_optional_types():
    """Negative offsets, optional booleans and floats retain their exact types."""
    measures = source_frame("measures")
    measures.loc[0, "mc_offset"] = "-7/8"
    ann = source_annotation("measures", measures)
    frame = annotation_to_frame(ann, corpus=CORPUS, piece=PIECE)
    assert frame.loc[0, "mc_offset"] == Fraction(-7, 8)
    assert frame.loc[0, "dont_count"] is True
    assert frame.loc[1, "dont_count"] is None
    notes = source_frame("notes")
    notes.loc[0, "tuning"] = "0.0"
    notes_ann = notes_to_score_note(notes)
    frame = annotation_to_frame(notes_ann, corpus=CORPUS, piece=PIECE)
    assert type(frame.loc[0, "tuning"]) is float
    assert frame.loc[0, "tuning"] == 0.0
    assert frame.loc[1, "tuning"] is None
    restored = frame_to_annotation(
        frame,
        payload_kind="score_note",
        extent_kind="quarters_interval.fraction",
        media_id=notes_ann.media_id,
    )
    assert restored.id == notes_ann.id


@pytest.mark.parametrize("name", ["tremolo", "gracenote", "volta", "tuning", "tied"])
def test_absent_nullable_note_column(name):
    """An absent nullable source column equals an entirely empty one, including id."""
    source = read_notes_tsv(DATA / f"dlc_notes_{CORPUS}_{PIECE}.tsv")
    source[name] = ""
    expected = notes_to_score_note(source)
    actual = notes_to_score_note(source.drop(columns=name))
    assert actual.id == expected.id
    assert msgspec.json.encode(actual) == msgspec.json.encode(expected)


@pytest.mark.parametrize("name", ["midi", "tpc", "nominal_duration", "mc"])
def test_missing_required_note_source_column(name):
    """Non-nullable source columns still report their names when absent."""
    with pytest.raises(BoppArgumentError, match=name):
        notes_to_score_note(source_frame("notes").drop(columns=name))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (60, 60.0),
        (True, 1),
        (None, 0),
        (Fraction(1, 3), Fraction(1, 2)),
        (Fraction(1), 1),
        ((1, 2), (1, 2.0)),
        ((True,), (1,)),
        ((1, (2,)), (1, (2.0,))),
        ((1,), [1]),
        ((1,), (1, 2)),
        (0.0, -0.0),
        (-0.0, 0.0),
        ([1], [1.0]),
        ([1, [2]], [1, [2.0]]),
        ([1], [1, 2]),
        ((0.0,), (-0.0,)),
        ("60", 60),
        (None, float("nan")),
    ],
)
def test_strict_cell_comparison(left, right):
    """Numerically equal values with different types are unequal, including tuples."""
    a = pd.DataFrame({"value": [left]}, dtype=object)
    b = pd.DataFrame({"value": [right]}, dtype=object)
    differences = frames_equal(a, b)
    assert len(differences) == 1
    assert "Row 0" in differences[0] and "value" in differences[0]
    with pytest.raises(AssertionError, match="value"):
        assert_frames_equal(a, b)


@pytest.mark.parametrize("backend", ["int64", "pyarrow"])
def test_non_object_dtypes_are_reported(backend):
    """Numpy- and pyarrow-backed twins of a canonical frame are not equal to it."""
    canonical = annotation_to_frame(
        source_annotation("notes"), corpus=CORPUS, piece=PIECE
    )
    if backend == "int64":
        twin = canonical.astype({"midi": "int64", "mc": "int64"})
        expected = ["'midi'", "'mc'"]
    else:
        twin = canonical.convert_dtypes(dtype_backend="pyarrow")
        expected = ["'midi'", "'name'"]
    differences = frames_equal(canonical, twin)
    assert differences
    assert all("dtype" in difference for difference in differences[: len(expected)])
    for name in expected:
        assert any(name in difference for difference in differences)
    assert frames_equal(twin, canonical)
    with pytest.raises(AssertionError, match="dtype"):
        assert_frames_equal(canonical, twin)
    assert frames_equal(canonical, canonical.copy()) == []


def test_frame_shape_order_and_diagnostic_limit():
    """Check column names/order, row count, index independence and bounded reports."""
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, dtype=object)
    assert "Columns" in frames_equal(frame, frame[["b", "a"]])[0]
    assert "Columns" in frames_equal(frame, frame.rename(columns={"a": "c"}))[0]
    assert "Row count" in frames_equal(frame, frame.iloc[:1])[0]
    assert frames_equal(frame, frame.set_axis([12, 13])) == []
    assert frames_equal(frame.iloc[:0], frame.iloc[:0].copy()) == []
    a = pd.DataFrame({"a": [1] * 100}, dtype=object)
    b = pd.DataFrame({"a": [1.0] * 100}, dtype=object)
    assert len(frames_equal(a, b)) == 20


@pytest.mark.parametrize("facet", PRESENT)
def test_load_dlc_facet(tmp_path, facet):
    """Load strict-suffix paths, restore identities and sort pieces and corpora."""
    spec = FACETS[facet]
    source = source_frame(facet).drop(columns=["corpus", "piece"])
    pairs = [("z_corpus", "a.piece"), ("a_corpus", "z_piece"), ("a_corpus", "a_piece")]
    expected = []
    for corpus, piece in pairs:
        directory = tmp_path / corpus / spec["facet_dir"]
        directory.mkdir(parents=True, exist_ok=True)
        source.to_csv(
            directory / f"{piece}.{spec['facet_dir']}.tsv", sep="\t", index=False
        )
        # The Winterreise duplicate naming convention must not be loaded.
        source.to_csv(directory / "n01.tsv", sep="\t", index=False)
    for corpus, piece in sorted(pairs):
        piece_source = source.assign(corpus=corpus, piece=piece)
        annotation = source_annotation(facet, piece_source)
        expected.append(annotation_to_frame(annotation, corpus=corpus, piece=piece))
    expected_frame = pd.concat(expected, ignore_index=True)
    loaded = load_dlc_facet(tmp_path, facet)
    assert_frames_equal(loaded, expected_frame)
    assert loaded.index.equals(pd.RangeIndex(len(loaded)))
    selected = load_dlc_facet(tmp_path, facet, corpora=["a_corpus"], pieces=["z_piece"])
    assert_frames_equal(selected, expected[1])
    for kwargs in ({"corpora": []}, {"pieces": []}, {"pieces": ["absent"]}):
        empty = load_dlc_facet(tmp_path, facet, **kwargs)
        assert_frames_equal(empty, expected_frame.iloc[:0])
        assert all(dtype == object for dtype in empty.dtypes)


def dlc_tree(tmp_path, facets, corpus=CORPUS, piece=PIECE):
    """Lay out a DLC-like checkout of fixture facets without identity columns."""
    for facet in facets:
        directory = tmp_path / corpus / FACETS[facet]["facet_dir"]
        directory.mkdir(parents=True, exist_ok=True)
        source_frame(facet).drop(columns=["corpus", "piece"]).to_csv(
            directory / f"{piece}.{FACETS[facet]['facet_dir']}.tsv",
            sep="\t",
            index=False,
        )
    return tmp_path


def test_load_dlc_facets_shares_one_reader_call(tmp_path, monkeypatch):
    """Control-event facets parse and convert each chords table exactly once."""
    root = dlc_tree(tmp_path, ["notes", "control_events.chord"])
    calls = []
    original = canonical.chords_to_control_events
    monkeypatch.setattr(
        canonical,
        "chords_to_control_events",
        lambda df: (calls.append(len(df)), original(df))[1],
    )
    kinds = ["control_events.chord", "control_events.dynamic", "control_events.tempo"]
    loaded = canonical.load_dlc_facets(root, [*kinds, "notes", "notes"])
    assert len(calls) == 1
    assert list(loaded) == [*kinds, "notes"]
    for facet in loaded:
        assert_frames_equal(loaded[facet], load_dlc_facet(root, facet))
    assert canonical.load_dlc_facets(root, []) == {}
    with pytest.raises(BoppArgumentError, match="Unknown DLC facet"):
        canonical.load_dlc_facets(root, ["notes", "bad_facet"])


def test_load_labels_closes_a_final_empty_duration(tmp_path):
    """A final empty duration_qb is closed with the piece end of the measures TSV."""
    root = dlc_tree(tmp_path, ["labels", "measures"])
    harmonies = root / CORPUS / "harmonies" / f"{PIECE}.harmonies.tsv"
    source = pd.read_csv(harmonies, sep="\t", dtype=str, keep_default_na=False)
    source.loc[source.index[-1], "duration_qb"] = ""
    source.to_csv(harmonies, sep="\t", index=False)
    loaded = load_dlc_facet(root, "labels")
    # Measure 83 starts at 244 and lasts 3 quarters, so the piece ends at 247.
    assert loaded.quarter.iloc[-1] == Fraction(245)
    assert loaded.duration.iloc[-1] == Fraction(2)
    (root / CORPUS / "measures").rename(root / CORPUS / "hidden")
    with pytest.raises(BoppArgumentError, match="Measure table is missing"):
        load_dlc_facet(root, "labels")


@pytest.mark.parametrize("facet", list(DLC_FACETS))
def test_empty_facet(tmp_path, facet):
    """Empty roots and absent kinds still expose every schema column."""
    source = source_frame("control_events.chord").drop(columns=["corpus", "piece"])
    directory = tmp_path / CORPUS / "chords"
    directory.mkdir(parents=True)
    source.to_csv(directory / f"{PIECE}.chords.tsv", sep="\t", index=False)
    empty = load_dlc_facet(tmp_path / "missing", facet)
    assert empty.empty
    expected_fields = PAYLOAD_TYPE_REGISTRY[
        FACETS[facet]["payload_kind"]
    ].__struct_fields__
    assert list(empty.columns) == [
        "corpus",
        "piece",
        "quarter",
        "duration",
        *expected_fields,
    ]
    assert all(dtype == object for dtype in empty.dtypes)
    if facet in {"control_events.figured_bass", "control_events.system_text"}:
        assert_frames_equal(load_dlc_facet(tmp_path, facet), empty)


def test_loader_errors_propagate(tmp_path):
    """Unsupported control-event data must fail even when requesting another kind."""
    source = source_frame("control_events.chord").drop(columns=["corpus", "piece"])
    source["unsupported"] = "non-empty"
    directory = tmp_path / CORPUS / "chords"
    directory.mkdir(parents=True)
    source.to_csv(directory / f"{PIECE}.chords.tsv", sep="\t", index=False)
    with pytest.raises(BoppArgumentError, match="unsupported"):
        load_dlc_facet(tmp_path, "control_events.system_text")
    with pytest.raises(BoppArgumentError, match="Unknown DLC facet"):
        load_dlc_facet(tmp_path, "bad_facet")


@pytest.mark.parametrize("mode", ["multiple", "missing", "extra", "required"])
def test_inverse_rejects_data_loss(mode):
    """The inverse refuses ambiguous identities and unrepresentable columns."""
    ann = source_annotation("notes")
    frame = annotation_to_frame(ann, corpus=CORPUS, piece=PIECE)
    if mode == "multiple":
        frame.loc[0, "piece"] = "another"
    elif mode == "missing":
        frame = frame.drop(columns="corpus")
    elif mode == "extra":
        frame["unknown"] = 1
    else:
        frame = frame.drop(columns="midi")
    with pytest.raises(BoppArgumentError):
        frame_to_annotation(
            frame,
            payload_kind="score_note",
            extent_kind="quarters_interval.fraction",
            media_id=ann.media_id,
        )
