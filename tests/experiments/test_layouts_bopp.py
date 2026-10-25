"""Verify BOPP benchmark layouts against canonical DLC facet tables."""

import csv
from io import StringIO
from pathlib import Path

import msgspec
import pandas as pd
import pyarrow as pa
import pytest

import bopp
from bopp.core import ensure_annotation_id
from bopp.exceptions import BoppArgumentError, BoppValidationError
from bopp.io import load_bopp_csv, load_bopp_json, load_bopp_msgpack, save_bopp_csv
from bopp.models.v1.annotation import Annotation
from experiments import canonical, layouts_bopp

DATA = Path(__file__).parent / "data"
SUPPORTED = [
    (layout, fmt, compression)
    for layout, formats in layouts_bopp.FORMATS.items()
    for fmt in formats
    for compression in ("none", "zstd")
    if not (fmt.startswith("csv") and compression == "zstd")
]
BINARY_CELLS = [cell for cell in SUPPORTED if not cell[1].startswith("csv")]
CSV_FORMATS = ("csv", "csv_upstream")
# Chopin's chords TSV has none of these events; test_events_absent_from_fixture
# covers them with synthetic data, test_empty covers empty facets.
EMPTY_IN_FIXTURE = ("control_events.figured_bass", "control_events.system_text")
NONEMPTY = [facet for facet in canonical.DLC_FACETS if facet not in EMPTY_IN_FIXTURE]
# zstd is a byte layer in front of the same decode, so its round trips run on
# notes only; zstd on other payloads: test_untyped_without_annotation_conversion
# (measures), test_events_absent_from_fixture (figured bass, system text).
ROUNDTRIP_CELLS = [
    (facet, *cell)
    for facet in NONEMPTY
    for cell in SUPPORTED
    if cell[2] == "none" or facet == "notes"
]
PUBLIC_LOADERS = {
    "json": load_bopp_json,
    "msgpack": load_bopp_msgpack,
    "csv": load_bopp_csv,
    "csv_upstream": load_bopp_csv,
}


def _normalised(value):
    """Builtins with tuples turned into lists, as JSON would store them."""
    return msgspec.json.decode(msgspec.json.encode(value))


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """Read two corpora with two fake pieces each from bundled DLC fixtures."""
    root = tmp_path_factory.mktemp("dlc")
    for directory in ("notes", "measures", "chords", "harmonies"):
        source = pd.read_csv(
            DATA / f"dlc_{directory}_chopin_mazurkas_BI163op67-4.tsv",
            sep="\t",
            dtype=str,
            keep_default_na=False,
        ).drop(columns=["corpus", "piece"])
        for corpus in ("z_corpus", "a_corpus"):
            target = root / corpus / directory
            target.mkdir(parents=True)
            for piece, table in (("z.piece", source), ("a_piece", source.iloc[:20])):
                table.to_csv(target / f"{piece}.{directory}.tsv", sep="\t", index=False)
    return {facet: canonical.load_dlc_facet(root, facet) for facet in canonical.DLC_FACETS}


def test_fixture_facets(frames):
    """The pruned round-trip matrix skips exactly the facets the fixture lacks."""
    assert [facet for facet in canonical.DLC_FACETS if frames[facet].empty] == list(
        EMPTY_IN_FIXTURE
    )


def test_format_constants():
    """Every format is documented, and only upstream-exact cells have no note."""
    assert layouts_bopp.FORMATS == {
        "per_item": ("json", "msgpack", "csv", "csv_upstream"),
        "collection": ("json", "msgpack", "jsonl"),
    }
    formats = {fmt for formats in layouts_bopp.FORMATS.values() for fmt in formats}
    assert set(layouts_bopp.FORMAT_NOTES) == formats
    assert {fmt for fmt, note in layouts_bopp.FORMAT_NOTES.items() if note} == {
        "jsonl",
        "csv",
        "csv_upstream",
    }
    for layout, formats in layouts_bopp.FORMATS.items():
        for fmt in formats:
            assert layouts_bopp.supported(layout, fmt, "notes") is True


@pytest.mark.parametrize(("facet", "layout", "fmt", "compression"), ROUNDTRIP_CELLS)
@pytest.mark.parametrize("validate", [True, False])
def test_roundtrip(tmp_path, frames, facet, layout, fmt, compression, validate):
    """Every supported cell retains exact values, types, columns and row order."""
    frame = frames[facet]
    paths = layouts_bopp.write(
        frame, facet, tmp_path, layout=layout, fmt=fmt, compression=compression
    )
    assert set(paths) == {path for path in tmp_path.rglob("*") if path.is_file()}
    assert sum(path.stat().st_size for path in paths) == sum(
        path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()
    )
    extension = "csv" if fmt in CSV_FORMATS else fmt
    suffix = f".{extension}" + (".zst" if compression == "zstd" else "")
    if layout == "collection":
        assert paths == [tmp_path / f"{facet}{suffix}"]
    else:
        assert paths == [
            tmp_path / corpus / f"{piece}.{facet}{suffix}"
            for corpus, piece in frame[["corpus", "piece"]]
            .drop_duplicates()
            .itertuples(index=False)
        ]
    loaded = layouts_bopp.load(
        tmp_path,
        facet,
        layout=layout,
        fmt=fmt,
        compression=compression,
        validate=validate,
    )
    assert canonical.frames_equal(loaded, frame) == []
    # decode_only itself: test_decode_only_does_not_convert_frames.
    if layout == "collection" and validate:
        from_file = layouts_bopp.load(
            paths[0],
            facet,
            layout=layout,
            fmt=fmt,
            compression=compression,
            validate=validate,
        )
        assert canonical.frames_equal(from_file, frame) == []


@pytest.mark.parametrize("fmt", CSV_FORMATS)
def test_csv_zstd_refused(tmp_path, frames, fmt):
    """Both CSV arms refuse whole-file compression with a reason, before writing."""
    status = layouts_bopp.supported("per_item", fmt, "notes", compression="zstd")
    assert isinstance(status, str) and "CSV" in status
    with pytest.raises(BoppArgumentError, match="CSV"):
        layouts_bopp.write(
            frames["notes"],
            "notes",
            tmp_path,
            layout="per_item",
            fmt=fmt,
            compression="zstd",
        )
    for validate in (True, False):
        with pytest.raises(BoppArgumentError, match="CSV"):
            layouts_bopp.load(
                tmp_path,
                "notes",
                layout="per_item",
                fmt=fmt,
                compression="zstd",
                validate=validate,
            )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(("layout", "fmt", "compression"), SUPPORTED)
@pytest.mark.parametrize("validate", [True, False])
def test_empty(tmp_path, frames, layout, fmt, compression, validate):
    """Empty facets retain schema columns even when no piece is represented."""
    frame = frames["notes"].iloc[:0]
    layouts_bopp.write(
        frame, "notes", tmp_path, layout=layout, fmt=fmt, compression=compression
    )
    assert (
        canonical.frames_equal(
            layouts_bopp.load(
                tmp_path,
                "notes",
                layout=layout,
                fmt=fmt,
                compression=compression,
                validate=validate,
            ),
            frame,
        )
        == []
    )


@pytest.mark.parametrize(
    "media_id,identity",
    [
        ("dlc:corpus/piece", ("corpus", "piece")),
        ("dlc:corpus_name/opus.1-ä", ("corpus_name", "opus.1-ä")),
        ("bpsfh:sonatas/01", ("sonatas", "01")),
    ],
)
def test_media_identity(media_id, identity):
    """Dataset identity preserves literal punctuation and Unicode."""
    assert layouts_bopp.media_id_to_identity(media_id) == identity


@pytest.mark.parametrize(
    "media_id",
    [
        "bps_fh:corpus/piece",
        ":corpus/piece",
        "corpus/piece",
        "dlc:piece",
        "dlc:/piece",
        "dlc:corpus/",
        "dlc:corpus/a/b",
        "dlc:../piece",
        "dlc:corpus/..",
        "dlc:corpus/a\\b",
    ],
)
def test_bad_media_identity(media_id):
    """Ambiguous identifiers cannot silently become a different identity."""
    with pytest.raises(BoppArgumentError):
        layouts_bopp.media_id_to_identity(media_id)


@pytest.mark.parametrize(("layout", "fmt", "compression"), BINARY_CELLS)
@pytest.mark.parametrize("corruption", ["id", "type", "missing_id"])
def test_validation(tmp_path, frames, layout, fmt, compression, corruption):
    """Typed decode validates ids and schema independently; untyped skips both."""
    frame = frames["notes"].iloc[:3]
    path = layouts_bopp.write(
        frame, "notes", tmp_path, layout=layout, fmt=fmt, compression=compression
    )[0]
    codec = msgspec.msgpack if fmt == "msgpack" else msgspec.json
    document = codec.decode(layouts_bopp._read_bytes(path, compression))
    ann = document[0] if layout == "collection" and fmt != "jsonl" else document
    if corruption == "id":
        ann["id"] = "00000000-0000-5000-8000-000000000000"
    else:
        del ann["id"]
        if corruption == "type":
            ann["payload"]["midi"][0] = "x"
    data = codec.encode(document) + (b"\n" if fmt == "jsonl" else b"")
    path.write_bytes(layouts_bopp._compress(data) if compression == "zstd" else data)
    raw = layouts_bopp.decode_only(
        tmp_path, layout=layout, fmt=fmt, compression=compression, validate=False
    )
    assert isinstance(raw[0], dict)
    if corruption == "type":
        assert raw[0]["payload"]["midi"][0] == "x"
    if corruption == "missing_id":
        loaded = layouts_bopp.load(
            tmp_path, "notes", layout=layout, fmt=fmt, compression=compression
        )
        assert canonical.frames_equal(loaded, frame) == []
    else:
        error = BoppValidationError if corruption == "id" else msgspec.ValidationError
        with pytest.raises(error):
            layouts_bopp.decode_only(
                tmp_path, layout=layout, fmt=fmt, compression=compression, validate=True
            )
        with pytest.raises(error):
            layouts_bopp.load(
                tmp_path, "notes", layout=layout, fmt=fmt, compression=compression
            )
    if corruption == "id":
        loaded = layouts_bopp.load(
            tmp_path,
            "notes",
            layout=layout,
            fmt=fmt,
            compression=compression,
            validate=False,
        )
        assert canonical.frames_equal(loaded, frame) == []


@pytest.mark.parametrize(("layout", "fmt", "compression"), SUPPORTED)
def test_untyped_without_annotation_conversion(
    tmp_path, frames, monkeypatch, layout, fmt, compression
):
    """The untyped path builds its frame directly from builtin dictionaries."""
    frame = frames["measures"]
    layouts_bopp.write(
        frame, "measures", tmp_path, layout=layout, fmt=fmt, compression=compression
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Untyped loading constructed or validated an annotation")

    monkeypatch.setattr(canonical, "annotation_to_frame", forbidden)
    monkeypatch.setattr(canonical, "frame_to_annotation", forbidden)
    monkeypatch.setattr(layouts_bopp, "Annotation", forbidden)
    monkeypatch.setattr(layouts_bopp, "validate_annotation_id", forbidden)
    monkeypatch.setattr(msgspec, "convert", forbidden)
    loaded = layouts_bopp.load(
        tmp_path,
        "measures",
        layout=layout,
        fmt=fmt,
        compression=compression,
        validate=False,
    )
    assert canonical.frames_equal(loaded, frame) == []


@pytest.mark.parametrize(("layout", "fmt", "compression"), SUPPORTED)
@pytest.mark.parametrize("validate", [True, False])
def test_overrides(tmp_path, frames, layout, fmt, compression, validate):
    """Alternate extents and arbitrary media ids preserve canonical identity."""
    frame = frames["notes"].iloc[:2].copy().rename(columns={"quarter": "time"})
    frame["time"] = pd.Series([0.0, 1.5], dtype=object)
    frame["duration"] = pd.Series([1.0, 0.25], dtype=object)
    paths = layouts_bopp.write(
        frame,
        "notes",
        tmp_path,
        layout=layout,
        fmt=fmt,
        compression=compression,
        extent_kind="time_interval",
        media_id=lambda corpus, piece: f"test:{piece}",
    )
    loaded = layouts_bopp.load(
        tmp_path,
        "notes",
        layout=layout,
        fmt=fmt,
        compression=compression,
        validate=validate,
    )
    assert canonical.frames_equal(loaded, frame) == []
    decoded = layouts_bopp.decode_only(
        paths[0], layout=layout, fmt=fmt, compression=compression, validate=False
    )
    assert decoded[0]["media_id"] == f"test:{frame.iloc[0]['piece']}"


@pytest.mark.parametrize("facet", NONEMPTY)
def test_csv_upstream_bytes(tmp_path, frames, facet):
    """csv_upstream is upstream's exact bytes; csv is identical now that upstream writes the id."""
    frame = frames[facet][frames[facet].corpus == "a_corpus"]
    upstream = layouts_bopp.write(
        frame, facet, tmp_path / "upstream", layout="per_item", fmt="csv_upstream"
    )
    injected = layouts_bopp.write(
        frame, facet, tmp_path / "injected", layout="per_item", fmt="csv"
    )
    groups = frame.groupby(["corpus", "piece"], sort=True)
    for (corpus, piece), group in groups:
        ann = canonical.frame_to_annotation(
            group,
            payload_kind=canonical.FACETS[facet]["payload_kind"],
            extent_kind="quarters_interval.fraction",
            media_id=f"dlc:{corpus}/{piece}",
        )
        reference = tmp_path / "reference.csv"
        save_bopp_csv(ann, reference)
        expected = reference.read_bytes()
        name = f"{piece}.{facet}.csv"
        plain = tmp_path / "upstream" / corpus / name
        with_id = tmp_path / "injected" / corpus / name
        assert plain in upstream and with_id in injected
        assert plain.read_bytes() == expected
        assert f'# id = "{ann.id}"\n'.encode() in expected
        assert with_id.read_bytes() == expected
        assert load_bopp_csv(plain).id == ann.id
        assert load_bopp_csv(with_id).id == ann.id


def test_csv_crlf(tmp_path, frames, monkeypatch):
    """The id goes into CRLF frontmatter (Windows) with the file's line ending."""
    save = layouts_bopp.save_bopp_csv

    def save_crlf(ann, path):
        save(ann, path)
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    monkeypatch.setattr(layouts_bopp, "save_bopp_csv", save_crlf)
    frame = frames["notes"]
    paths = layouts_bopp.write(frame, "notes", tmp_path, layout="per_item", fmt="csv")
    for path in paths:
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        assert sum(line.startswith(b'# id = "') for line in lines) == 1
        assert all(line.endswith(b"\r\n") for line in lines)
        assert data.count(b"\n") == data.count(b"\r\n")
    for validate in (True, False):
        loaded = layouts_bopp.load(
            tmp_path, "notes", layout="per_item", fmt="csv", validate=validate
        )
        assert canonical.frames_equal(loaded, frame) == []
    ids = {load_bopp_csv(path).id for path in paths}
    assert msgspec.UNSET not in ids and len(ids) == len(paths)


@pytest.mark.parametrize(
    "data,message",
    [
        (b'"a","b"\n1,2\n', "delimiter '# ---' not found"),
        (b"", "delimiter '# ---' not found"),
        (b'# ---\n# bopp_version = "1.0"\n"a"\n1\n', "no closing"),
    ],
)
def test_csv_frontmatter_missing(tmp_path, data, message):
    """Injection fails clearly, naming the file, without a closed frontmatter."""
    path = tmp_path / "broken.csv"
    with pytest.raises(BoppArgumentError, match=message) as info:
        layouts_bopp._inject_csv_id(data, "id", path)
    assert str(path) in str(info.value)


def _edit_csv_body(path, column, value):
    """Rewrite one cell of a CSV file whose frontmatter uses LF line endings."""
    _, header, body = path.read_text().split("# ---\n", 2)
    rows = list(csv.reader(StringIO(body)))
    rows[1][rows[0].index(column)] = value
    stream = StringIO()
    csv.writer(stream).writerows(rows)
    path.write_text("# ---\n" + header + "# ---\n" + stream.getvalue())


def test_csv_tampered_id(tmp_path, frames):
    """A wrong id in the frontmatter fails typed loading; untyped skips it."""
    frame = frames["notes"].iloc[:3]
    path = layouts_bopp.write(frame, "notes", tmp_path, layout="per_item", fmt="csv")[0]
    lines = path.read_text().splitlines(keepends=True)
    path.write_text(
        "".join(
            '# id = "00000000-0000-5000-8000-000000000000"\n'
            if line.startswith("# id =")
            else line
            for line in lines
        )
    )
    with pytest.raises(BoppValidationError):
        layouts_bopp.load(path, "notes", layout="per_item", fmt="csv")
    loaded = layouts_bopp.load(
        path, "notes", layout="per_item", fmt="csv", validate=False
    )
    assert canonical.frames_equal(loaded, frame) == []


@pytest.mark.parametrize("fmt", CSV_FORMATS)
def test_csv_parser_failure(tmp_path, frames, fmt):
    """Parser failure, not validation: polars rejects midi 'x' in both arms."""
    from polars.exceptions import ComputeError

    path = layouts_bopp.write(
        frames["notes"].iloc[:3], "notes", tmp_path, layout="per_item", fmt=fmt
    )[0]
    _edit_csv_body(path, "payload:score_note:midi", "x")
    for validate in (True, False):
        with pytest.raises(ComputeError):
            layouts_bopp.load(
                path, "notes", layout="per_item", fmt=fmt, validate=validate
            )


@pytest.mark.parametrize("fmt", CSV_FORMATS)
def test_csv_schema_violation(tmp_path, frames, fmt):
    """
    A parseable midi of 200 breaks the schema maximum of 127.

    validate=True rejects it in the typed decode (before any id check: the id
    in the csv arm matches the tampered content). validate=False has no schema
    check and returns the value unchanged.
    """
    path = layouts_bopp.write(
        frames["notes"].iloc[:3], "notes", tmp_path, layout="per_item", fmt=fmt
    )[0]
    ann = load_bopp_csv(path)
    ann.payload.midi = [200, *ann.payload.midi[1:]]
    ann.id = msgspec.UNSET
    save_bopp_csv(ann, path)
    if fmt == "csv":
        ensure_annotation_id(ann)
        path.write_bytes(layouts_bopp._inject_csv_id(path.read_bytes(), ann.id, path))
    with pytest.raises(msgspec.ValidationError, match="<= 127"):
        layouts_bopp.load(path, "notes", layout="per_item", fmt=fmt)
    loaded = layouts_bopp.load(
        path, "notes", layout="per_item", fmt=fmt, validate=False
    )
    assert loaded["midi"].iloc[0] == 200


@pytest.mark.parametrize("fmt", layouts_bopp.FORMATS["per_item"])
def test_per_item_matches_public_loaders(tmp_path, frames, fmt):
    """
    Pin the private upstream helpers to the public bopp.io loaders.

    Typed decoding (_VersionHeader dispatch) must equal load_bopp_json/msgpack/
    csv; untyped decoding (_build_polars_csv_schema, _parse_complex_cell) must
    equal the public annotation as builtins; and the untyped frame
    (canonical._columns) must equal canonical.annotation_to_frame.
    """
    loader = PUBLIC_LOADERS[fmt]
    for facet in NONEMPTY:
        frame = frames[facet][frames[facet].corpus == "a_corpus"]
        paths = layouts_bopp.write(
            frame, facet, tmp_path / facet, layout="per_item", fmt=fmt
        )
        for path in paths:
            public = loader(path)
            (typed,) = layouts_bopp.decode_only(
                path, layout="per_item", fmt=fmt, validate=True
            )
            assert typed == public
            (raw,) = layouts_bopp.decode_only(
                path, layout="per_item", fmt=fmt, validate=False
            )
            assert _normalised(raw) == _normalised(public)
            corpus, piece = layouts_bopp.media_id_to_identity(public.media_id)
            assert (
                canonical.frames_equal(
                    layouts_bopp._builtin_frame(raw, corpus, piece),
                    canonical.annotation_to_frame(public, corpus=corpus, piece=piece),
                )
                == []
            )


@pytest.mark.parametrize("fmt", layouts_bopp.FORMATS["collection"])
def test_collection_matches_public_loaders(tmp_path, frames, fmt):
    """Collection items equal what load_bopp_json reads from per-item files."""
    for facet in NONEMPTY:
        frame = frames[facet][frames[facet].corpus == "a_corpus"]
        items, cell = tmp_path / facet / "items", tmp_path / facet / "collection"
        per_item = layouts_bopp.write(
            frame, facet, items, layout="per_item", fmt="json"
        )
        public = [load_bopp_json(path) for path in per_item]
        layouts_bopp.write(frame, facet, cell, layout="collection", fmt=fmt)
        typed = layouts_bopp.decode_only(
            cell, layout="collection", fmt=fmt, validate=True
        )
        assert typed == public
        raw = layouts_bopp.decode_only(
            cell, layout="collection", fmt=fmt, validate=False
        )
        assert _normalised(raw) == _normalised(public)


@pytest.mark.parametrize(
    ("layout", "fmt", "facet", "compression"),
    [
        ("missing", "json", "notes", "none"),
        ("collection", "csv", "notes", "none"),
        ("collection", "csv_upstream", "notes", "none"),
        ("per_item", "jsonl", "notes", "none"),
        ("per_item", "json", "missing", "none"),
        ("per_item", "json", "notes", "gzip"),
    ],
)
def test_unsupported(tmp_path, frames, layout, fmt, facet, compression):
    """Unsupported cells return a reason and fail explicitly before writing."""
    assert isinstance(
        layouts_bopp.supported(layout, fmt, facet, compression=compression), str
    )
    with pytest.raises(BoppArgumentError):
        layouts_bopp.write(
            frames["notes"],
            facet,
            tmp_path,
            layout=layout,
            fmt=fmt,
            compression=compression,
        )
    with pytest.raises(BoppArgumentError):
        layouts_bopp.load(
            tmp_path, facet, layout=layout, fmt=fmt, compression=compression
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(("layout", "fmt", "compression"), SUPPORTED)
def test_decode_only_does_not_convert_frames(
    tmp_path, frames, monkeypatch, layout, fmt, compression
):
    """The timed decode operation does not perform pandas frame conversion."""
    layouts_bopp.write(
        frames["notes"],
        "notes",
        tmp_path,
        layout=layout,
        fmt=fmt,
        compression=compression,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Decode-only built a frame")

    monkeypatch.setattr(pd, "DataFrame", forbidden)
    monkeypatch.setattr(canonical, "annotation_to_frame", forbidden)
    monkeypatch.setattr(layouts_bopp, "_builtin_frame", forbidden)
    for validate in (True, False):
        decoded = layouts_bopp.decode_only(
            tmp_path,
            layout=layout,
            fmt=fmt,
            compression=compression,
            validate=validate,
        )
        assert isinstance(decoded, list) and len(decoded) == 4
        assert all(isinstance(ann, Annotation if validate else dict) for ann in decoded)


@pytest.mark.parametrize(
    ("layout", "fmt"), [(a, b) for a, b, c in BINARY_CELLS if c == "zstd"]
)
def test_compressed_container(tmp_path, frames, layout, fmt):
    """The size header describes identical bytes and notes compress smaller."""
    paths = layouts_bopp.write(
        frames["notes"], "notes", tmp_path / "plain", layout=layout, fmt=fmt
    )
    compressed = layouts_bopp.write(
        frames["notes"],
        "notes",
        tmp_path / "zstd",
        layout=layout,
        fmt=fmt,
        compression="zstd",
    )
    for plain, packed in zip(paths, compressed, strict=True):
        data = packed.read_bytes()
        expected = plain.read_bytes()
        assert data[:8] == len(expected).to_bytes(8, "little")
        assert data[8:12] == b"\x28\xb5\x2f\xfd"
        assert (
            pa.Codec("zstd").decompress(
                data[8:], decompressed_size=len(expected), asbytes=True
            )
            == expected
        )
        assert packed.stat().st_size < plain.stat().st_size


@pytest.mark.parametrize(
    "corruption,message",
    [
        ("short", "truncated zstd size header"),
        ("plain", "not a zstd container"),
        ("huge_size", "corrupt zstd size header"),
        ("truncated_frame", "zstd decompression failed"),
        ("trailing_bytes", "zstd decompression failed"),
        ("wrong_size", "zstd decompression failed|incorrect zstd decoded size"),
    ],
)
def test_corrupt_zstd(tmp_path, frames, corruption, message):
    """Corrupt containers raise a ValueError naming the file, not MemoryError."""
    frame = frames["notes"].iloc[:50]
    plain = layouts_bopp.write(
        frame, "notes", tmp_path / "plain", layout="collection", fmt="json"
    )[0].read_bytes()
    path = layouts_bopp.write(
        frame,
        "notes",
        tmp_path / "zstd",
        layout="collection",
        fmt="json",
        compression="zstd",
    )[0]
    data = path.read_bytes()
    data = {
        "short": data[:10],
        "plain": plain,
        "huge_size": (2**40).to_bytes(8, "little") + data[8:],
        "truncated_frame": data[:-5],
        "trailing_bytes": data + b"junk",
        "wrong_size": (len(plain) - 1).to_bytes(8, "little") + data[8:],
    }[corruption]
    path.write_bytes(data)
    for validate in (True, False):
        with pytest.raises(BoppArgumentError, match=message) as info:
            layouts_bopp.decode_only(
                path,
                layout="collection",
                fmt="json",
                compression="zstd",
                validate=validate,
            )
        assert isinstance(info.value, ValueError)
        assert str(path) in str(info.value)


@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_jsonl_blank_lines(tmp_path, frames, compression):
    """Blank and whitespace-only JSONL lines are skipped in both arms."""
    frame = frames["labels"]
    path = layouts_bopp.write(
        frame,
        "labels",
        tmp_path,
        layout="collection",
        fmt="jsonl",
        compression=compression,
    )[0]
    lines = layouts_bopp._read_bytes(path, compression).splitlines(keepends=True)
    data = b"\n" + lines[0] + b"  \r\n" + b"".join(lines[1:]) + b"\n\t\n"
    path.write_bytes(layouts_bopp._compress(data) if compression == "zstd" else data)
    for validate in (True, False):
        loaded = layouts_bopp.load(
            tmp_path,
            "labels",
            layout="collection",
            fmt="jsonl",
            compression=compression,
            validate=validate,
        )
        assert canonical.frames_equal(loaded, frame) == []


@pytest.mark.parametrize("kind", ["figured_bass", "system_text"])
@pytest.mark.parametrize(("layout", "fmt", "compression"), SUPPORTED)
@pytest.mark.parametrize("validate", [True, False])
def test_events_absent_from_fixture(tmp_path, kind, layout, fmt, compression, validate):
    """Nonempty synthetic events cover the two kinds absent from Chopin's TSV."""
    columns = (
        {"thoroughbass_duration": [[1, 4], [1, 2]], "thoroughbass_level_1": ["6", None]}
        if kind == "figured_bass"
        else {"system_text": ["Fine", "1"]}
    )
    ann = bopp.create(
        media_id="dlc:synthetic/piece",
        payload_kind=f"score_control_event.{kind}",
        extent_kind="quarters_interval.fraction",
        quarter=[[0, 1], [1, 2]],
        duration=[[1, 1], [2, 1]],
        staff=[1, 1],
        voice=[1, 2],
        mc=[1, 2],
        mn=[0, 1],
        **columns,
    )
    frame = canonical.annotation_to_frame(ann, corpus="synthetic", piece="piece")
    facet = f"control_events.{kind}"
    layouts_bopp.write(
        frame, facet, tmp_path, layout=layout, fmt=fmt, compression=compression
    )
    loaded = layouts_bopp.load(
        tmp_path,
        facet,
        layout=layout,
        fmt=fmt,
        compression=compression,
        validate=validate,
    )
    assert canonical.frames_equal(loaded, frame) == []
