"""Verify exact Parquet facet round trips, pruning and validation fairness."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

import bopp
from bopp.exceptions import BoppValidationError
from experiments import layouts_parquet as parquet
from experiments.arrow_schema import ArrowValidationError, arrow_schema
from experiments.canonical import (
    FACETS,
    annotation_to_frame,
    frames_equal,
    load_dlc_facet,
)

DATA = Path(__file__).parent / "data"
LAYOUTS = ["per_item", "single", "partitioned"]
PAIRS = [(c, p) for c in ("a_corpus", "z_corpus") for p in ("a.piece", "z_piece")]


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """Read two corpora with two pieces each from the bundled DLC fixtures."""
    root = tmp_path_factory.mktemp("dlc")
    for directory in ("notes", "measures", "chords", "harmonies"):
        source = pd.read_csv(
            DATA / f"dlc_{directory}_chopin_mazurkas_BI163op67-4.tsv",
            sep="\t",
            dtype=str,
            keep_default_na=False,
        ).drop(columns=["corpus", "piece"])
        for corpus, piece in reversed(PAIRS):
            target = root / corpus / directory
            target.mkdir(parents=True, exist_ok=True)
            subset = source if piece == "a.piece" else source.iloc[:20]
            subset.to_csv(target / f"{piece}.{directory}.tsv", sep="\t", index=False)
    return {facet: load_dlc_facet(root, facet) for facet in FACETS}


@pytest.mark.parametrize("facet", FACETS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("compression", ["zstd", "none"])
@pytest.mark.parametrize("validate", [True, False])
def test_roundtrip(tmp_path, frames, facet, layout, compression, validate):
    """Every facet/layout/codec/validation arm reproduces canonical cells exactly."""
    original = frames[facet]
    paths = parquet.write(
        original, facet, tmp_path, layout=layout, compression=compression
    )
    assert all(isinstance(path, Path) and path.is_file() for path in paths)
    loaded = parquet.load(tmp_path, facet, layout=layout, validate=validate)
    assert frames_equal(loaded, original) == []
    table = parquet.read_table(tmp_path, facet, layout=layout)
    assert table.schema.field("corpus").type == pa.string()
    assert table.schema.field("piece").type == pa.string()
    assert table.column_names == list(original.columns)


@pytest.mark.parametrize("facet", FACETS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("validate", [True, False])
def test_projection_and_selection(tmp_path, frames, facet, layout, validate):
    """Projections and exact pair filters preserve order, types and empty schemas."""
    original = frames[facet]
    parquet.write(original, facet, tmp_path, layout=layout)
    columns = [original.columns[-1], "duration", original.columns[4]]
    projected = list(dict.fromkeys(["corpus", "piece", *columns]))
    for pieces in (None, [PAIRS[3], PAIRS[0], PAIRS[3]], [], [("missing", "piece")]):
        expected = original
        if pieces is not None:
            mask = [(c, p) in pieces for c, p in zip(original.corpus, original.piece)]
            expected = original.loc[mask]
        for requested in (None, columns, []):
            selected = (
                list(original.columns)
                if requested is None
                else projected
                if requested
                else ["corpus", "piece"]
            )
            loaded = parquet.load(
                tmp_path,
                facet,
                layout=layout,
                validate=validate,
                pieces=pieces,
                columns=requested,
            )
            assert frames_equal(loaded, expected[selected]) == []


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("compression", ["zstd", "none"])
def test_row_groups_and_paths(tmp_path, frames, layout, compression):
    """Each piece occupies one row group with statistics usable for pruning."""
    frame = frames["notes"]
    paths = parquet.write(
        frame, "notes", tmp_path, layout=layout, compression=compression
    )
    if layout == "per_item":
        assert paths == [tmp_path / c / f"{p}.notes.parquet" for c, p in PAIRS]
    elif layout == "single":
        assert paths == [tmp_path / "notes.parquet"]
    else:
        assert paths == [
            tmp_path / f"corpus={c}" / "notes.parquet" for c in ("a_corpus", "z_corpus")
        ]
    found = []
    for path in paths:
        file = pq.ParquetFile(path)
        assert (
            file.metadata.num_row_groups
            == {"per_item": 1, "single": 4, "partitioned": 2}[layout]
        )
        for i in range(file.metadata.num_row_groups):
            group = file.metadata.row_group(i)
            ids = [group.column(j).statistics for j in (0, 1)]
            assert all(stat.min == stat.max for stat in ids)
            pair = tuple(stat.min for stat in ids)
            found.append(pair)
            assert group.num_rows == len(
                frame[(frame.corpus == pair[0]) & (frame.piece == pair[1])]
            )
            assert group.column(0).compression == (
                "ZSTD" if compression == "zstd" else "UNCOMPRESSED"
            )
            assert "RLE_DICTIONARY" in group.column(0).encodings
        dataset = ds.dataset(path, format="parquet")
        expression = (ds.field("corpus") == PAIRS[0][0]) & (
            ds.field("piece") == PAIRS[0][1]
        )
        groups = [
            group
            for fragment in dataset.get_fragments()
            for group in fragment.split_by_row_group(filter=expression)
        ]
        assert len(groups) == (
            1 if PAIRS[0] in found[-file.metadata.num_row_groups :] else 0
        )
    assert found == PAIRS


@pytest.fixture
def io_log(monkeypatch):
    """Return a callable that starts recording the reader's Parquet I/O.

    Call it after writing: it records every file opened, every read and every
    ``pq.read_table`` call (passed through), and forbids validation and pandas
    conversion.
    """
    log = {"opened": [], "groups": [], "read": [], "read_table": []}

    class Recording(pq.ParquetFile):
        def __init__(self, source, *args, **kwargs):
            log["opened"].append(Path(source))
            super().__init__(source, *args, **kwargs)

        def read_row_groups(self, row_groups, *args, **kwargs):
            log["groups"].append((list(row_groups), kwargs.get("columns")))
            return super().read_row_groups(row_groups, *args, **kwargs)

        def read(self, *args, **kwargs):
            log["read"].append(kwargs.get("columns"))
            return super().read(*args, **kwargs)

    read_schema = pq.read_schema

    def recording_schema(path, *args, **kwargs):
        log["opened"].append(Path(path))
        return read_schema(path, *args, **kwargs)

    read_table = pq.read_table

    def recording_read_table(source, *args, **kwargs):
        log["read_table"].append(([Path(path) for path in source], kwargs))
        return read_table(source, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("read_table must not validate or convert")

    def start():
        monkeypatch.setattr(pq, "ParquetFile", Recording)
        monkeypatch.setattr(pq, "read_schema", recording_schema)
        monkeypatch.setattr(pq, "read_table", recording_read_table)
        monkeypatch.setattr(parquet.validation, "validate_table", forbidden)
        monkeypatch.setattr(parquet.validation, "validate_columns", forbidden)
        monkeypatch.setattr(parquet, "_validate_projection", forbidden)
        monkeypatch.setattr(parquet, "table_to_frame", forbidden)
        return log

    return start


@pytest.mark.parametrize("layout", LAYOUTS)
def test_reader_pushdown(tmp_path, frames, layout, io_log):
    """The I/O step opens one file once and reads only the selected row group."""
    paths = parquet.write(frames["notes"], "notes", tmp_path, layout=layout)
    io_log = io_log()
    table = parquet.read_table(
        tmp_path, "notes", layout=layout, columns=["midi"], pieces=[PAIRS[0]]
    )
    assert table.column_names == ["corpus", "piece", "midi"]
    assert set(zip(table["corpus"].to_pylist(), table["piece"].to_pylist())) == {
        PAIRS[0]
    }
    assert io_log["opened"] == [paths[0]]
    assert io_log["read_table"] == []
    if layout == "per_item":
        # One pruned file: ParquetFile.read measured faster than read_table.
        assert io_log["read"] == [["corpus", "piece", "midi"]]
        assert io_log["groups"] == []
    else:
        # PAIRS[0] is the first row group of the single file and of its corpus file.
        assert io_log["groups"] == [([0], ["corpus", "piece", "midi"])]
        assert io_log["read"] == []


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "pieces", [None, [PAIRS[3], PAIRS[1]], []], ids=["all", "two", "none"]
)
def test_each_file_opened_once(tmp_path, frames, layout, pieces, io_log):
    """Full reads, multi-piece selections and empty selections open no file twice."""
    paths = parquet.write(frames["notes"], "notes", tmp_path, layout=layout)
    io_log = io_log()
    table = parquet.read_table(tmp_path, "notes", layout=layout, pieces=pieces)
    opened = io_log["opened"]
    assert len(opened) == len(set(opened))
    frame = frames["notes"]
    if pieces == []:
        assert opened == paths[:1]
        assert io_log["read_table"] == []
        assert table.num_rows == 0
        assert table.column_names == list(frame.columns)
        return
    expected = (
        len(frame)
        if pieces is None
        else sum((frame.corpus == c) & (frame.piece == p) for c, p in pieces).sum()
    )
    assert table.num_rows == expected
    if layout == "per_item":
        # Several per-item files: one dataset read over the pruned path list,
        # with no filter, no imposed schema and no Hive partitioning.
        assert opened == []
        wanted = (
            paths
            if pieces is None
            else [tmp_path / c / f"{p}.notes.parquet" for c, p in PAIRS[1::2]]
        )
        assert io_log["read_table"] == [
            (wanted, {"columns": None, "partitioning": None})
        ]
        return
    assert io_log["read_table"] == []
    assert opened == paths
    if pieces is None:
        # Full reads use ParquetFile.read, measured faster than read_row_groups.
        assert io_log["read"] == [None] * len(paths)
        assert io_log["groups"] == []
    elif layout == "single":
        assert io_log["groups"] == [([1, 3], None)]
    else:
        assert io_log["groups"] == [([1], None), ([1], None)]


@pytest.fixture(scope="module")
def trees(tmp_path_factory, frames):
    """Write every facet in every layout once, one directory per layout."""
    root = tmp_path_factory.mktemp("trees")
    for layout in LAYOUTS:
        for facet, frame in frames.items():
            parquet.write(frame, facet, root / layout, layout=layout)
    return root


def _pair_filter(pieces):
    """Express a piece selection as the dataset filter of a plain read_table."""
    expression = ds.scalar(False)
    for corpus, piece in pieces:
        expression = expression | (
            (ds.field("corpus") == corpus) & (ds.field("piece") == piece)
        )
    return expression


@pytest.mark.parametrize("layout", LAYOUTS)
def test_reader_matches_plain_read_table(trees, frames, layout):
    """Every facet, projection and selection equals one plain filtered read_table.

    The plain reference (no pruning, no row-group selection, a filter
    expression) is the reader of round 2, which round 3 matched in the same
    case matrix, so the reader choice per layout changes no result: rows,
    order, types, nullability and schema metadata.
    """
    selections = [
        None,
        [],
        [("missing", "piece")],
        [PAIRS[0]],
        [PAIRS[3], PAIRS[0], PAIRS[3]],
        [("a_corpus", "z_piece"), ("z_corpus", "a.piece")],
    ]
    out = trees / layout
    for facet, frame in frames.items():
        paths = [str(path) for path in parquet._paths(out, facet, layout)]
        last = frame.columns[-1]
        for columns in (None, [last], [last, "duration", "quarter"], ["piece"], []):
            projected = parquet._projection(columns)
            for pieces in selections:
                table = parquet.read_table(
                    out, facet, layout=layout, columns=columns, pieces=pieces
                )
                reference = (
                    pq.read_table(
                        paths,
                        columns=projected,
                        filters=None if pieces is None else _pair_filter(pieces),
                        partitioning=None,
                    )
                    if paths
                    # Facets absent from the fixture write no per-item or
                    # partitioned file; the result is the derived empty schema.
                    else parquet._schema(facet)
                    .empty_table()
                    .select(projected or parquet._schema(facet).names)
                )
                assert table.equals(reference, check_metadata=True), (
                    facet,
                    columns,
                    pieces,
                )


def test_per_item_unknown_projection_and_relaxed_nulls(tmp_path, frames):
    """The multi-file read reports unknown columns and relaxes later nulls."""
    frame = frames["notes"]
    paths = parquet.write(frame, "notes", tmp_path, layout="per_item")
    with pytest.raises(ValueError, match="bogus"):
        parquet.read_table(tmp_path, "notes", layout="per_item", columns=["bogus"])
    # A later file with a null in a field the first file stores as required.
    table = pq.read_table(paths[-1])
    assert not pq.read_schema(paths[0]).field("midi").nullable
    values = table["midi"].to_pylist()
    values[0] = None
    pq.write_table(
        table.set_column(
            table.schema.get_field_index("midi"), "midi", pa.array(values)
        ),
        paths[-1],
    )
    loaded = parquet.read_table(tmp_path, "notes", layout="per_item")
    assert loaded.schema.field("midi").nullable
    assert loaded["midi"].null_count == 1
    assert not loaded.schema.field("piece").nullable
    with pytest.raises(ArrowValidationError, match="midi"):
        parquet.load(tmp_path, "notes", layout="per_item")


@pytest.mark.parametrize("layout", ["single", "partitioned"])
@pytest.mark.parametrize("statistics", [True, False])
def test_null_identities_bypass_statistics(tmp_path, layout, statistics):
    """A row group with null corpus rows needs the row filter despite min == max."""
    table = pa.table(
        {
            "corpus": ["a", None, "a", "a"],
            "piece": ["p", "p", "p", None],
            "x": [1, 2, 3, 4],
        }
    )
    path = (
        tmp_path / "notes.parquet"
        if layout == "single"
        else tmp_path / "corpus=a" / "notes.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, write_statistics=statistics)
    group = pq.ParquetFile(path).metadata.row_group(0)
    if statistics:
        for j, column in enumerate(("corpus", "piece")):
            stats = group.column(j).statistics
            assert stats.min == stats.max == {"corpus": "a", "piece": "p"}[column]
            assert stats.null_count == 1
    for pieces in ([("a", "p")], [("a", "p"), ("b", "q")]):
        loaded = parquet.read_table(tmp_path, "notes", layout=layout, pieces=pieces)
        reference = pq.read_table(path, filters=_pair_filter(pieces))
        assert loaded["x"].to_pylist() == reference["x"].to_pylist() == [1, 3]


def test_rows_without_pair_statistics_are_filtered(tmp_path, frames):
    """A foreign multi-piece row group is read once and filtered to exact pairs."""
    frame = frames["notes"]
    table = parquet.frame_to_table(frame, "notes")
    pq.write_table(table, tmp_path / "notes.parquet", row_group_size=len(frame))
    assert pq.ParquetFile(tmp_path / "notes.parquet").metadata.num_row_groups == 1
    for pieces in ([PAIRS[2]], [PAIRS[0], PAIRS[3]]):
        loaded = parquet.load(tmp_path, "notes", layout="single", pieces=pieces)
        mask = [(c, p) in pieces for c, p in zip(frame.corpus, frame.piece)]
        assert frames_equal(loaded, frame.loc[mask]) == []
        projected = parquet.load(
            tmp_path, "notes", layout="single", pieces=pieces, columns=["midi"]
        )
        assert (
            frames_equal(projected, frame.loc[mask, ["corpus", "piece", "midi"]]) == []
        )


def test_unknown_projection(tmp_path, frames):
    """Projected names outside the stored schema are a ValueError."""
    parquet.write(frames["notes"], "notes", tmp_path, layout="single")
    with pytest.raises(ValueError, match="bogus"):
        parquet.read_table(tmp_path, "notes", layout="single", columns=["bogus"])


def _corrupt(frame, column, value, row):
    """Copy a frame with one cell replaced, keeping object dtype."""
    bad = frame.copy()
    bad[column] = bad[column].astype(object)
    bad.at[bad.index[row], column] = value
    return bad


def _bopp_rejects(frame, facet):
    """BOPP refuses the same piece (create or validate) that Parquet refuses."""
    piece = frame[(frame.corpus == PAIRS[0][0]) & (frame.piece == PAIRS[0][1])]
    columns = {}
    for name in piece.columns[2:]:
        values = piece[name].tolist()
        if all(value is None for value in values):
            continue
        columns[name] = [parquet._wire(value) for value in values]
    with pytest.raises((BoppValidationError, TypeError)):
        ann = bopp.create(
            media_id="test:bool",
            payload_kind=FACETS[facet]["payload_kind"],
            extent_kind="quarters_interval.fraction",
            generate_id=False,
            **columns,
        )
        bopp.validate(ann)


@pytest.mark.parametrize("flag", [True, np.True_, False, np.False_])
@pytest.mark.parametrize("first", [False, True], ids=["flag-after", "flag-first"])
@pytest.mark.parametrize(
    ("facet", "column", "good", "bad"),
    [
        # float scalar (number | null): [0.5, True] became [0.5, 1.0]
        ("notes", "tuning", 0.5, lambda flag: flag),
        # int scalar: np.True_ after an int became 1
        ("notes", "midi", 60, lambda flag: flag),
        # nullable int scalar
        ("notes", "tied", 1, lambda flag: flag),
        # integer list cell
        ("measures", "next", (2,), lambda flag: (2, flag)),
        # fraction column given a list cell instead of a Fraction
        ("measures", "act_dur", Fraction(3, 4), lambda flag: [1, flag]),
        # string column
        ("measures", "timesig", "3/4", lambda flag: flag),
    ],
)
def test_frame_to_table_rejects_booleans(frames, facet, column, good, bad, flag, first):
    """Booleans never reach Arrow inference, as BOPP rejects them in every kind."""
    frame = frames[facet].copy()
    frame[column] = pd.Series([good] * len(frame), index=frame.index, dtype=object)
    row = 0 if first else 1
    corrupt = _corrupt(frame, column, bad(flag), row)
    with pytest.raises(
        ArrowValidationError,
        match=rf"'{column}': type \(.*\): boolean; 1 offending rows; "
        rf"first offending row index {row}",
    ):
        parquet.frame_to_table(corrupt, facet)
    _bopp_rejects(corrupt, facet)
    # The same frame without the boolean converts.
    parquet.frame_to_table(frame, facet)


@pytest.mark.parametrize(
    ("facet", "column", "value", "rule"),
    [
        ("notes", "midi", "60", r"type \(int64\)"),
        ("notes", "tuning", 2**53 + 1, "float64 exact range"),
        ("notes", "tpc", 2**63, "int64 range"),
        ("measures", "act_dur", [1, 2**63], "int64 range"),
        ("measures", "next", 7, r"type \(list<item: int64>\)"),
    ],
)
def test_frame_to_table_wraps_arrow_errors(frames, facet, column, value, rule):
    """Values Arrow cannot convert raise ArrowValidationError naming the column."""
    frame = frames[facet]
    if column == "tuning":
        frame = frame.assign(
            tuning=pd.Series([0.5] * len(frame), index=frame.index, dtype=object)
        )
    with pytest.raises(ArrowValidationError, match=rf"'{column}': {rule}"):
        parquet.frame_to_table(_corrupt(frame, column, value, 1), facet)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_extent_kind_override(tmp_path, frames, layout):
    """write, read_table and load honour a non-DLC extent kind end to end."""
    extent = "quarters_interval.float"
    frame = frames["measures"].copy()
    for name in ("quarter", "duration"):
        frame[name] = pd.Series(
            [float(value) for value in frame[name]], index=frame.index, dtype=object
        )
    with pytest.raises(ArrowValidationError):
        parquet.write(frame, "measures", tmp_path / "default", layout=layout)
    parquet.write(frame, "measures", tmp_path, layout=layout, extent_kind=extent)
    loaded = parquet.load(tmp_path, "measures", layout=layout, extent_kind=extent)
    assert frames_equal(loaded, frame) == []
    projected = parquet.load(
        tmp_path, "measures", layout=layout, columns=["quarter"], extent_kind=extent
    )
    assert frames_equal(projected, frame[["corpus", "piece", "quarter"]]) == []
    # The default fraction extent rejects the stored float quarters.
    with pytest.raises(ArrowValidationError, match="quarter"):
        parquet.load(tmp_path, "measures", layout=layout)
    (tmp_path / "empty").mkdir()
    empty = parquet.read_table(
        tmp_path / "empty", "measures", layout="per_item", extent_kind=extent
    )
    assert empty.num_rows == 0
    assert empty.schema.field("quarter").type == pa.float64()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    ("column", "bad"),
    [
        ("keysig", 9),
        ("timesig", "3/0"),
        ("next", [-2]),
        ("mc_offset", [1, 0]),
        ("mc", None),
    ],
)
def test_invalid_values(tmp_path, frames, layout, column, bad):
    """Full and projected validation reject bounds, patterns, nested and null errors."""
    frame = frames["measures"]
    paths = parquet.write(frame, "measures", tmp_path, layout=layout)
    table = pq.ParquetFile(paths[0]).read()
    values = table[column].to_pylist()
    values[0] = bad
    # Bypass the writer and relax physical nullability to allow corrupt input.
    table = table.set_column(
        table.schema.get_field_index(column),
        column,
        pa.array(values, type=table[column].type),
    )
    pq.write_table(table, paths[0])
    if column not in {"mc_offset"}:
        loaded = parquet.load(tmp_path, "measures", layout=layout, validate=False)
        assert loaded[column].iloc[0] == (tuple(bad) if column == "next" else bad)
    for columns in (None, [column]):
        with pytest.raises(ArrowValidationError, match=column):
            parquet.load(tmp_path, "measures", layout=layout, columns=columns)
    # A projection must not read or validate corrupt unselected data.
    loaded = parquet.load(tmp_path, "measures", layout=layout, columns=["mn"])
    assert frames_equal(loaded, frame[["corpus", "piece", "mn"]]) == []


@pytest.mark.parametrize("facet", FACETS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_polars(tmp_path, frames, facet, layout):
    """Polars collects the same rows and scalar values, including empty facets."""
    frame = frames[facet]
    parquet.write(frame, facet, tmp_path, layout=layout)
    loaded = parquet.polars_load(tmp_path, facet, layout=layout)
    assert loaded.height == len(frame)
    for name in ("corpus", "piece", frame.columns[4]):
        assert loaded[name].to_list() == frame[name].tolist()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("kind", ["figured_bass", "system_text"])
def test_populated_absent_fixture_kinds(tmp_path, layout, kind):
    """Kinds absent from the fixture also support actual payload rows."""
    facet = f"control_events.{kind}"
    payload = (
        {"thoroughbass_duration": [[1, 4]], "thoroughbass_level_1": ["6"]}
        if kind == "figured_bass"
        else {"system_text": ["Allegro"]}
    )
    annotation = bopp.create(
        media_id="test:piece",
        payload_kind=FACETS[facet]["payload_kind"],
        extent_kind="quarters_interval.fraction",
        quarter=[[0, 1]],
        duration=[[1, 1]],
        staff=[1],
        voice=[1],
        mc=[1],
        mn=[0],
        **payload,
    )
    frame = annotation_to_frame(annotation, corpus="test", piece="piece")
    parquet.write(frame, facet, tmp_path, layout=layout)
    assert frames_equal(parquet.load(tmp_path, facet, layout=layout), frame) == []


def test_conversion_types_and_optional_values(frames):
    """Null columns, signed zero, fractions and integer tuples retain exact types."""
    frame = frames["notes"].iloc[:3].copy()
    frame["tuning"] = pd.Series([-0.0, None, 0.125], dtype=object)
    frame.loc[0, "quarter"] = Fraction(-1, 3)
    table = parquet.frame_to_table(
        frame, "notes", extent_kind="quarters_interval.fraction"
    )
    assert table.schema == arrow_schema("score_note", "quarters_interval.fraction")
    assert frames_equal(parquet.table_to_frame(table), frame) == []
    measures = frames["measures"]
    assert (
        frames_equal(
            parquet.table_to_frame(parquet.frame_to_table(measures, "measures")),
            measures,
        )
        == []
    )
    assert any(value is True for value in measures.dont_count)
    assert any(value is None for value in measures.dont_count)
    assert all(type(value) is tuple for value in measures.next)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_empty_populated_facet(tmp_path, frames, layout):
    """A deliberately empty notes frame round-trips in every layout."""
    frame = frames["notes"].iloc[:0]
    parquet.write(frame, "notes", tmp_path, layout=layout)
    assert frames_equal(parquet.load(tmp_path, "notes", layout=layout), frame) == []


def test_writer_rejects_data_loss(tmp_path, frames):
    """Reject missing/extra columns, invalid values, options and stale output."""
    frame = frames["notes"]
    for bad in (frame.drop(columns="tuning"), frame.assign(unknown=1)):
        with pytest.raises(ValueError, match="schema column"):
            parquet.frame_to_table(bad, "notes")
    bad = frame.copy()
    bad.loc[0, "midi"] = 60.5
    with pytest.raises(ArrowValidationError, match="midi"):
        parquet.frame_to_table(bad, "notes")
    for kwargs in (
        {"layout": "unknown"},
        {"layout": "single", "compression": "snappy"},
    ):
        with pytest.raises(ValueError):
            parquet.write(frame, "notes", tmp_path, **kwargs)
    parquet.write(frame, "notes", tmp_path, layout="single")
    with pytest.raises(FileExistsError):
        parquet.write(frame, "notes", tmp_path, layout="single")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_prefix_names_and_shared_output(tmp_path, frames, layout):
    """Sort identity basenames, independent of path suffixes and sibling facets.

    Piece names include a dot and another facet's name, so a loose basename
    pattern would select a sibling facet's files in the shared directory.
    """
    originals = {}
    written = {}
    names = ("a", "a.b", "a.measures", "a.notes")
    for facet in ("notes", "measures"):
        rows = []
        for corpus in ("a", "a.b"):
            for piece in names:
                row = frames[facet].iloc[:1].copy()
                # Plain assignment of a str would create a pandas 3 str-dtype
                # column, which is not canonical; keep Python str in object.
                row["corpus"] = pd.Series([corpus], index=row.index, dtype=object)
                row["piece"] = pd.Series([piece], index=row.index, dtype=object)
                rows.append(row)
        frame = pd.concat(rows, ignore_index=True)
        assert frames_equal(frame, frame) == []
        originals[facet] = frame
        written[facet] = parquet.write(frame, facet, tmp_path, layout=layout)
    assert not set(written["notes"]) & set(written["measures"])
    for facet, frame in originals.items():
        loaded = parquet.load(tmp_path, facet, layout=layout)
        assert list(loaded.columns) == list(frames[facet].columns)
        assert frames_equal(loaded, frame) == []
        table = parquet.read_table(tmp_path, facet, layout=layout)
        assert table.column_names == list(frames[facet].columns)
        polars = parquet.polars_load(tmp_path, facet, layout=layout)
        assert polars.columns == list(frames[facet].columns)
        assert polars["piece"].to_list() == frame["piece"].tolist()
        pair = ("a.b", "a.notes")
        selected = parquet.load(tmp_path, facet, layout=layout, pieces=[pair])
        expected = frame[(frame.corpus == pair[0]) & (frame.piece == pair[1])]
        assert frames_equal(selected, expected) == []
