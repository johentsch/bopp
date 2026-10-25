import shutil
from pathlib import Path

import msgspec
import pytest

import bopp
from bopp.exceptions import BoppArgumentError, BoppValidationError
from bopp.models.v1.grouping import Grouping
from experiments import boppard as bd
from experiments import vienna_payloads as vp

EXCERPT = (
    Path(__file__).parent / "data" / "vienna_Mozart_K331_1st-mov_p05_excerpt.match"
)
PIECE = "Mozart_K331_1st-mov"
MATCH = bd.Link(
    name="match",
    source_payload="performance_note",
    source_field="match",
    target_payload="score_note",
    target_field="anchor",
    missing=["NOMATCH"],
)


@pytest.fixture(scope="module")
def vienna(tmp_path_factory):
    """Score notes, two performances with notes and pedal, and their timelines."""
    directory = tmp_path_factory.mktemp("match")
    for performer in ("p01", "p02"):
        shutil.copy(EXCERPT, directory / f"{PIECE}_{performer}.match")
    score = vp.score_note_annotation(directory, PIECE)
    members = [(PIECE, score)]
    timelines = [bd.timeline_for(score, PIECE)]
    for path in sorted(directory.glob("*.match")):
        performance, pedal = vp.performance_annotations(path)
        timelines.append(bd.timeline_for(performance, path.stem, aligned_to=[PIECE]))
        members += [(path.stem, performance), (path.stem, pedal)]
    return members, timelines


def _wire(ann):
    return msgspec.json.encode(ann, order="deterministic")


def tags(media_id, values, *, annotator="a"):
    return bopp.create(
        media_id=media_id,
        payload_kind="tag_open",
        extent_kind="time_interval",
        time=[float(i) for i in range(len(values))],
        duration=[1.0] * len(values),
        value=values,
        sandbox={"annotator": annotator},
    )


def test_roundtrip_ids(vienna):
    members, timelines = vienna
    packed = bd.pack(members, timelines, collection_id="vienna:excerpt", links=[MATCH])
    assert [bd._payload_type(s) for s in packed.shards] == [
        "score_note",
        "performance_note",
        "pedal",
    ]
    for fmt in ("msgpack", "json"):
        loaded = bd.decode(bd.encode(packed, fmt), fmt)
        rebuilt = bd.split(loaded)
        assert sorted(a.id for a in rebuilt) == sorted(a.id for _, a in members)
        wire = {a.id: _wire(a) for _, a in members}
        assert {a.id: _wire(a) for a in rebuilt} == wire


def test_shard_is_upstream_annotation(vienna):
    members, timelines = vienna
    shard = bd.pack(members, timelines, collection_id="vienna:excerpt").shards[1]
    ann = shard.annotation
    assert ann.media_id == "vienna:excerpt"
    assert bopp.validate(ann)
    (grouping,) = ann.groupings
    assert grouping.name == bd.TIMELINE_GROUPING
    assert grouping.labels == {f"{PIECE}_p01": 1, f"{PIECE}_p02": 2}
    rows = shard.members[0].rows
    assert grouping.ids == [1] * rows + [2] * rows


def test_split_one_timeline(vienna):
    members, timelines = vienna
    packed = bd.pack(members, timelines, collection_id="vienna:excerpt")
    rebuilt = bd.split(packed, timelines=[f"{PIECE}_p02"])
    assert [a.id for a in rebuilt] == [a.id for t, a in members if t.endswith("p02")]


def test_links_resolve_across_aligned_timelines(vienna):
    members, timelines = vienna
    packed = bd.pack(members, timelines, collection_id="vienna:excerpt", links=[MATCH])
    table = bd.resolve(packed, MATCH)
    performance = packed.shards[1].annotation.payload
    score = packed.shards[0].annotation.payload
    assert table.num_rows == sum(m != "NOMATCH" for m in performance.match)
    for source, target in zip(
        table["source_row"].to_pylist(), table["target_row"].to_pylist(), strict=True
    ):
        assert performance.match[source] == score.anchor[target]


def test_unresolved_link_fails(vienna):
    members, timelines = vienna
    broken = msgspec.structs.replace(MATCH, missing=[])
    with pytest.raises(BoppValidationError, match="matches 0 target rows"):
        bd.pack(members, timelines, collection_id="vienna:excerpt", links=[broken])


def test_member_grouping_only_when_needed():
    first, second = tags("x:one", ["a", "b"]), tags("x:one", ["c"], annotator="b")
    other = tags("x:two", ["d"])
    timelines = [
        bd.Timeline(id="one", media_id="x:one"),
        bd.Timeline(id="two", media_id="x:two"),
    ]
    packed = bd.pack(
        [("one", first), ("one", second), ("two", other)],
        timelines,
        collection_id="x:all",
    )
    (shard,) = packed.shards
    names = [g.name for g in shard.annotation.groupings]
    assert names == [bd.TIMELINE_GROUPING, bd.MEMBER_GROUPING]
    assert shard.annotation.groupings[1].ids == [0, 0, 1, 2]
    assert [a.id for a in bd.split(packed)] == [first.id, second.id, other.id]


def test_split_survives_reordered_rows():
    one, two = tags("x:one", ["a", "b"]), tags("x:two", ["c", "d"])
    timelines = [
        bd.Timeline(id="one", media_id="x:one"),
        bd.Timeline(id="two", media_id="x:two"),
    ]
    packed = bd.pack([("one", one), ("two", two)], timelines, collection_id="x:all")
    shard = packed.shards[0].annotation
    order = [0, 2, 1, 3]  # interleave the members, keeping each member's own order
    payload = msgspec.structs.replace(
        shard.payload, value=[shard.payload.value[i] for i in order]
    )
    extent = msgspec.structs.replace(
        shard.extent,
        time=[shard.extent.time[i] for i in order],
        duration=[shard.extent.duration[i] for i in order],
    )
    (grouping,) = shard.groupings
    grouping = msgspec.structs.replace(grouping, ids=[grouping.ids[i] for i in order])
    shuffled = msgspec.structs.replace(
        shard, payload=payload, extent=extent, groupings=[grouping]
    )
    packed.shards[0] = msgspec.structs.replace(packed.shards[0], annotation=shuffled)
    assert [a.id for a in bd.split(packed)] == [one.id, two.id]


def test_different_scalars_make_separate_shards():
    one = tags("x:one", ["a"])
    two = bopp.create(
        media_id="x:one",
        payload_kind="tag_open",
        extent_kind="time_interval",
        time=[0.0],
        duration=[1.0],
        value=["b"],
        sandbox={"annotator": "b"},
    )
    timelines = [bd.Timeline(id="one", media_id="x:one")]
    packed = bd.pack([("one", one), ("one", two)], timelines, collection_id="x:all")
    assert len(packed.shards) == 1  # sandbox is member header, not a shard key
    assert [m.sandbox for m in packed.shards[0].members] == [
        {"annotator": "a"},
        {"annotator": "b"},
    ]


def test_merge(vienna):
    members, timelines = vienna
    score, rest = members[:1], members[1:]
    left = bd.pack(score, timelines[:1], collection_id="vienna:excerpt")
    right = bd.pack(rest, timelines, collection_id="vienna:excerpt")
    merged = bd.merge(left, right)
    assert sorted(a.id for a in bd.split(merged)) == sorted(a.id for _, a in members)
    assert [t.id for t in merged.timelines] == [t.id for t in timelines]


def test_pack_rejects_wrong_timeline_and_reserved_names():
    one = tags("x:one", ["a"])
    timelines = [bd.Timeline(id="one", media_id="x:other")]
    with pytest.raises(BoppArgumentError, match="does not live on"):
        bd.pack([("one", one)], timelines, collection_id="x:all")
    with pytest.raises(BoppArgumentError, match="Unknown timeline"):
        bd.pack([("two", one)], timelines, collection_id="x:all")
    grouped = msgspec.structs.replace(
        one, groupings=[Grouping(name=bd.TIMELINE_GROUPING, ids=[0])], id=msgspec.UNSET
    )
    bopp.core.ensure_annotation_id(grouped)
    with pytest.raises(BoppArgumentError, match="reserved"):
        bd.pack(
            [("one", grouped)],
            [bd.Timeline(id="one", media_id="x:one")],
            collection_id="x:all",
        )


def test_tampered_shard_fails_validation(vienna):
    members, timelines = vienna
    packed = bd.pack(members, timelines, collection_id="vienna:excerpt")
    raw = msgspec.to_builtins(packed)
    raw["shards"][2]["annotation"]["payload"]["value"][0] += 1
    with pytest.raises(BoppValidationError, match="ID mismatch"):
        bd.decode(msgspec.msgpack.encode(raw))


def test_bench_vienna_excerpt(tmp_path):
    from experiments import bench, boppard_bench

    shutil.copy(EXCERPT, tmp_path / f"{PIECE}_p01.match")
    tables, _ = bench._vienna_tables(tmp_path)
    row = boppard_bench.measure("vienna", tables)
    assert row["roundtrip_ids"] == "ok"
    assert row["members"] == 3 and row["timelines"] == 2
    assert row["links"]["match"]["rows"] > 0
    assert row["bytes"]["offsets_only_msgpack"] < row["bytes"]["boppard_msgpack"]


def test_upstream_filter_by_on_timeline_grouping(vienna):
    """Upstream filter_by selects one timeline; split is what restores the member."""
    members, timelines = vienna
    shard = bd.pack(members, timelines, collection_id="vienna:excerpt").shards[1]
    index = shard.annotation.groupings[0].labels[f"{PIECE}_p02"]
    filtered = bopp.filter_by(
        shard.annotation,
        lambda i: i == index,
        facet="groupings",
        target=bd.TIMELINE_GROUPING,
    )
    original = members[3][1]
    assert filtered.payload.match == original.payload.match
    assert filtered.parents == [shard.annotation.id]
    assert filtered.media_id == "vienna:excerpt" and filtered.id != original.id


def _variant(ann, media_id, **payload):
    out = msgspec.structs.replace(
        ann, media_id=media_id, payload=msgspec.structs.replace(ann.payload, **payload)
    )
    msgspec.structs.force_setattr(out, "id", bopp.core.compute_annotation_id(out))
    return out


@pytest.mark.parametrize(
    ("column", "shards"), [("gracenote", 1), ("staff", 2)], ids=["nullable", "not_null"]
)
def test_absent_optional_column(vienna, column, shards):
    """Without an optional column a member joins the shard if the column takes None."""
    members, timelines = vienna
    score = members[0][1]
    rows = len(score.payload.midi)
    full = _variant(score, score.media_id, gracenote=[None] * rows, staff=[1] * rows)
    reduced = _variant(full, "vienna:other", **{column: msgspec.UNSET})
    timelines = [timelines[0], bd.Timeline(id="other", media_id="vienna:other")]
    packed = bd.pack(
        [(PIECE, full), ("other", reduced)], timelines, collection_id="vienna:excerpt"
    )
    assert len(packed.shards) == shards
    if shards == 1:
        assert packed.shards[0].members[1].absent == [f"payload.{column}"]
    assert [a.id for a in bd.split(packed)] == [full.id, reduced.id]
