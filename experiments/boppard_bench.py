"""Pack each benchmark dataset into one .boppard and measure it against per-item BOPP.

Usage: ``python -m experiments.boppard_bench --work <dir> --out <results.json>``
where ``<dir>/<dataset>/canonical-<dataset>.pkl`` are the tables cached by
``bench.prepare``. For every dataset the members are rebuilt from the canonical
tables, packed, encoded, decoded with id validation and split back; the split
must reproduce every member id.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgspec
import pyarrow as pa

from experiments import bench, boppard, canonical

_CODEC = pa.Codec("zstd")
_LIST = msgspec.msgpack.Decoder(list[boppard.Annotation])


def _median(fn: Callable[[], Any], repeat: int = 3) -> tuple[float, Any]:
    times, result = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def _timeline_id(dataset: str, facet: str, corpus: str, piece: str) -> str:
    return f"{corpus}/{piece}" if dataset == "dlc" else piece


def members(dataset: str, tables: dict) -> tuple[list, list, list]:
    """(timeline id, annotation) pairs, the manifest and the links of a dataset."""
    out: list[tuple[str, Any]] = []
    timelines: dict[str, boppard.Timeline] = {}
    for facet, frame in tables.items():
        make_media = bench.media_id_function(facet) or (lambda c, p: f"dlc:{c}/{p}")
        for (corpus, piece), group in frame.groupby(["corpus", "piece"], sort=True):
            ann = canonical.frame_to_annotation(
                group,
                payload_kind=canonical.FACETS[facet]["payload_kind"],
                extent_kind=canonical.facet_extent_kind(facet),
                media_id=make_media(corpus, piece),
            )
            tid = _timeline_id(dataset, facet, corpus, piece)
            aligned = []
            if facet in ("vienna.performance_notes", "vienna.pedal"):
                aligned = [piece.rsplit("_", 1)[0]]
            if tid not in timelines:
                timelines[tid] = boppard.timeline_for(ann, tid, aligned_to=aligned)
            out.append((tid, ann))
    links = []
    if dataset == "vienna":
        links.append(
            boppard.Link(
                name="match",
                source_payload="performance_note",
                source_field="match",
                target_payload="score_note",
                target_field="anchor",
                missing=["NOMATCH"],
            )
        )
    return out, sorted(timelines.values(), key=lambda t: t.id), links


def measure(dataset: str, tables: dict) -> dict[str, Any]:
    pairs, timelines, links = members(dataset, tables)
    annotations = [ann for _, ann in pairs]
    collection_id = {"dlc": "dlc:distant_listening_corpus@v3.2"}.get(
        dataset, f"{dataset}:{dataset}"
    )
    t_pack, packed = _median(
        lambda: boppard.pack(pairs, timelines, collection_id=collection_id, links=links)
    )
    data = boppard.encode(packed)
    per_item = [msgspec.msgpack.encode(a) for a in annotations]
    collection = msgspec.msgpack.encode(annotations)
    t_decode, loaded = _median(lambda: boppard.decode(data))
    t_decode_list, _ = _median(
        lambda: [boppard.validate_annotation_id(a) for a in _LIST.decode(collection)]
    )
    t_split, rebuilt = _median(lambda: boppard.split(loaded))
    assert sorted(a.id for a in rebuilt) == sorted(a.id for a in annotations)
    one = timelines[len(timelines) // 2].id
    t_one, _ = _median(lambda: boppard.split(loaded, timelines=[one]))
    grouping_bytes = sum(
        len(msgspec.msgpack.encode(s.annotation.groupings)) for s in packed.shards
    )
    # offsets-only variant: rows located by Member.rows alone, no grouping column
    bare = msgspec.structs.replace(
        packed,
        shards=[
            msgspec.structs.replace(
                sh,
                annotation=msgspec.structs.replace(
                    sh.annotation, groupings=msgspec.UNSET
                ),
            )
            for sh in packed.shards
        ],
    )
    bare_data = boppard.encode(bare)
    header_bytes = len(
        msgspec.msgpack.encode([packed.timelines, [s.members for s in packed.shards]])
    )
    row = {
        "dataset": dataset,
        "members": len(annotations),
        "timelines": len(timelines),
        "shards": [
            {
                "payload": boppard._payload_type(s),
                "members": len(s.members),
                "rows": len(s.annotation.groupings[0].ids),
                "groupings": [g.name for g in s.annotation.groupings],
            }
            for s in packed.shards
        ],
        "bytes": {
            "per_item_msgpack": sum(map(len, per_item)),
            "per_item_msgpack_zstd": sum(
                len(_CODEC.compress(b, asbytes=True)) for b in per_item
            ),
            "collection_msgpack": len(collection),
            "collection_msgpack_zstd": len(_CODEC.compress(collection, asbytes=True)),
            "boppard_msgpack": len(data),
            "boppard_msgpack_zstd": len(_CODEC.compress(data, asbytes=True)),
            "offsets_only_msgpack": len(bare_data),
            "offsets_only_msgpack_zstd": len(_CODEC.compress(bare_data, asbytes=True)),
            "boppard_json": len(boppard.encode(packed, "json")),
            "timeline_groupings": grouping_bytes,
            "manifest_and_members": header_bytes,
        },
        "seconds": {
            "pack": t_pack,
            "boppard_decode_validated": t_decode,
            "collection_decode_validated": t_decode_list,
            "split_all_validated": t_split,
            "split_one_timeline": t_one,
        },
        "roundtrip_ids": "ok",
    }
    for link in links:
        t_link, table = _median(lambda link=link: boppard.resolve(loaded, link))
        row.setdefault("links", {})[link.name] = {
            "rows": table.num_rows,
            "seconds": t_link,
        }
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=["bpsfh", "vienna", "dlc"])
    args = parser.parse_args(argv)
    rows = []
    for dataset in args.datasets:
        cache = args.work / dataset / f"canonical-{dataset}.pkl"
        tables = pickle.loads(cache.read_bytes())
        rows.append(measure(dataset, tables))
        print(json.dumps(rows[-1]), flush=True)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
