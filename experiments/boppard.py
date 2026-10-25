"""BOPPARD: BOPP for aggregated research data, a collection format over BOPP annotations.

A ``.boppard`` file holds many BOPP annotations ("members") in one object:

- ``timelines`` is the manifest: one entry per timeline (a score, a
  performance, a recording, a scan), with the ``media_id`` that the members
  living on it carried, the TimeToAlign coordinate description (domain, unit,
  number type), the timelines it is aligned to, and an opaque ``tts`` slot for
  the TimeToShare manifest entry.
- ``shards`` concatenate the members that share payload type, extent type and
  every scalar field, so that each shard is one ordinary BOPP ``Annotation``
  (loadable and filterable by upstream bopp). The reserved grouping
  ``boppard.timeline`` links every row to its timeline: ids index
  ``timelines`` and the labels map timeline ids to those indices. Only when a
  shard holds several members on the same timeline (two annotators, two
  versions) is a second reserved grouping ``boppard.member`` added, indexing
  the shard's ``members``.
- ``members`` keep the header of each original annotation (id, parents,
  metadata, sandbox), so ``split`` rebuilds every member byte for byte and its
  content id is checked against the stored one.
- ``links`` declare cross-references by value: a source payload field whose
  values name rows of a target payload field, either on the same timeline or
  on a timeline the source timeline is aligned to (e.g. Vienna
  ``performance_note.match`` -> ``score_note.anchor``). ``resolve`` turns a
  link into a row-level join table.

The shard's own ``media_id`` is the collection id: inside a BOPPARD, media
identity is a property of the timeline, not of the annotation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import msgspec
import pyarrow as pa

from bopp.core import compute_annotation_id, validate_annotation_id
from bopp.exceptions import BoppArgumentError, BoppValidationError
from bopp.models.v1.annotation import Annotation

BOPPARD_VERSION = "0.1"
TIMELINE_GROUPING = "boppard.timeline"
MEMBER_GROUPING = "boppard.member"
_RESERVED = (TIMELINE_GROUPING, MEMBER_GROUPING)
_FACETS = ("payload", "extent", "confidence")

#: TimeToAlign coordinate description implied by each extent type.
EXTENT_TIMELINES: dict[str, tuple[str, str, str]] = {
    "quarters_time.fraction": ("logical", "quarters", "fraction"),
    "quarters_interval.fraction": ("logical", "quarters", "fraction"),
    "quarters_time.float": ("logical", "quarters", "float"),
    "quarters_interval.float": ("logical", "quarters", "float"),
    "midi_ticks": ("physical", "midi_ticks", "int"),
    "midi_interval": ("physical", "midi_ticks", "int"),
    "time": ("physical", "seconds", "float"),
    "time_interval": ("physical", "seconds", "float"),
}


class Timeline(msgspec.Struct, omit_defaults=True):
    """One timeline of the manifest."""

    id: str
    media_id: str
    domain: str | None = None
    unit: str | None = None
    number_type: str | None = None
    aligned_to: list[str] = []
    tts: dict[str, Any] = {}


class Member(msgspec.Struct, omit_defaults=True):
    """Header of one original annotation inside a shard."""

    timeline: str
    id: str
    rows: int
    parents: list[str] | msgspec.UnsetType = msgspec.UNSET
    metadata: Any = msgspec.UNSET
    sandbox: dict[str, Any] | msgspec.UnsetType = msgspec.UNSET


class Shard(msgspec.Struct):
    """Concatenated members that share every non-column field."""

    annotation: Annotation
    members: list[Member]


class Link(msgspec.Struct, omit_defaults=True):
    """A cross-reference by value from one payload field to another."""

    name: str
    source_payload: str
    source_field: str
    target_payload: str
    target_field: str
    scope: str = "aligned"
    """``aligned``: targets live on a timeline the source timeline is aligned
    to; ``timeline``: on the source's own timeline."""
    missing: list[str] = []
    """Source values that deliberately reference nothing (e.g. ``NOMATCH``)."""


class Boppard(msgspec.Struct, omit_defaults=True):
    """A collection of BOPP annotations on many timelines."""

    collection_id: str
    timelines: list[Timeline]
    shards: list[Shard]
    links: list[Link] = []
    boppard_version: str = BOPPARD_VERSION


def _split_facet(data: dict[str, Any] | None) -> tuple[dict, dict]:
    """Columns (list-valued fields, as upstream treats them) and scalar fields."""
    if data is None:
        return {}, {}
    columns = {k: v for k, v in data.items() if isinstance(v, list)}
    scalars = {k: v for k, v in data.items() if not isinstance(v, list)}
    return columns, scalars


def _member_key(raw: dict[str, Any]) -> bytes:
    """Everything that must be equal for two annotations to share a shard."""
    key: dict[str, Any] = {"bopp_version": raw["bopp_version"]}
    for facet in _FACETS:
        columns, scalars = _split_facet(raw.get(facet))
        if facet in raw:
            key[facet] = {"scalars": scalars, "columns": sorted(columns)}
    key["groupings"] = [
        {"name": g["name"], "labels": g.get("labels")} for g in raw.get("groupings", [])
    ]
    return msgspec.json.encode(key, order="deterministic")


def timeline_for(
    ann: Annotation, timeline_id: str | None = None, **kwargs: Any
) -> Timeline:
    """Default manifest entry of an annotation's timeline, from its extent type."""
    extent = msgspec.to_builtins(ann.extent) if ann.extent is not msgspec.UNSET else {}
    domain, unit, number_type = EXTENT_TIMELINES.get(
        extent.get("extent_type"), (None, None, None)
    )
    return Timeline(
        id=timeline_id or ann.media_id,
        media_id=ann.media_id,
        domain=domain,
        unit=unit,
        number_type=number_type,
        **kwargs,
    )


def pack(
    members: Iterable[tuple[str, Annotation]],
    timelines: Sequence[Timeline],
    *,
    collection_id: str,
    links: Sequence[Link] = (),
) -> Boppard:
    """
    Concatenate annotations into a BOPPARD.

    Parameters
    ----------
    members : iterable of (timeline id, Annotation)
        Annotations in the order they are stored; each must carry an id and
        live on a timeline listed in ``timelines``.
    timelines : sequence of Timeline
        The manifest. Every member's ``media_id`` must equal its timeline's.
    collection_id : str
        Collection identifier, used as every shard's ``media_id``
        (``<prefix>:<name>``).
    links : sequence of Link, optional
        Cross-references, checked with ``resolve``.

    Returns
    -------
    Boppard
    """
    index = {t.id: i for i, t in enumerate(timelines)}
    if len(index) != len(timelines):
        raise BoppArgumentError("Timeline ids must be unique")
    groups: dict[bytes, list[tuple[str, dict[str, Any]]]] = {}
    for timeline_id, ann in members:
        if timeline_id not in index:
            raise BoppArgumentError(f"Unknown timeline {timeline_id!r}")
        if ann.media_id != timelines[index[timeline_id]].media_id:
            raise BoppArgumentError(
                f"{ann.media_id!r} does not live on timeline {timeline_id!r}"
            )
        if ann.id is msgspec.UNSET:
            raise BoppArgumentError("Members need an id; create them with bopp.create")
        raw = msgspec.to_builtins(ann)
        if any(g["name"] in _RESERVED for g in raw.get("groupings", [])):
            raise BoppArgumentError(f"Grouping names {_RESERVED} are reserved")
        groups.setdefault(_member_key(raw), []).append((timeline_id, raw))
    shards = [_shard(group, index, collection_id) for group in groups.values()]
    boppard = Boppard(
        collection_id=collection_id,
        timelines=list(timelines),
        shards=shards,
        links=list(links),
    )
    for link in links:
        resolve(boppard, link)
    return boppard


def _shard(
    group: list[tuple[str, dict[str, Any]]], index: dict[str, int], collection_id: str
) -> Shard:
    first = group[0][1]
    shard: dict[str, Any] = {
        "media_id": collection_id,
        "bopp_version": first["bopp_version"],
    }
    members = []
    timeline_ids: list[int] = []
    member_ids: list[int] = []
    for number, (timeline_id, raw) in enumerate(group):
        rows = max(
            (len(v) for f in _FACETS for v in _split_facet(raw.get(f))[0].values()),
            default=0,
        )
        for facet in _FACETS:
            if facet not in raw:
                continue
            columns, scalars = _split_facet(raw[facet])
            target = shard.setdefault(facet, {**scalars, **{k: [] for k in columns}})
            for name, values in columns.items():
                target[name].extend(values)
        for position, grouping in enumerate(raw.get("groupings", [])):
            target = shard.setdefault("_own", [])
            if len(target) <= position:
                target.append({**grouping, "ids": []})
            target[position]["ids"].extend(grouping["ids"])
        timeline_ids += [index[timeline_id]] * rows
        member_ids += [number] * rows
        members.append(
            Member(
                timeline=timeline_id,
                id=raw["id"],
                rows=rows,
                parents=raw.get("parents", msgspec.UNSET),
                metadata=raw.get("metadata", msgspec.UNSET),
                sandbox=raw.get("sandbox", msgspec.UNSET),
            )
        )
    used = sorted({m.timeline for m in members}, key=index.__getitem__)
    groupings = [
        {
            "name": TIMELINE_GROUPING,
            "ids": timeline_ids,
            "labels": {t: index[t] for t in used},
        }
    ]
    if len(used) < len(members):
        groupings.append({"name": MEMBER_GROUPING, "ids": member_ids})
    shard["groupings"] = groupings + shard.pop("_own", [])
    annotation = msgspec.convert(shard, Annotation)
    msgspec.structs.force_setattr(annotation, "id", compute_annotation_id(annotation))
    return Shard(annotation=annotation, members=members)


def _member_rows(shard: Shard) -> list[list[int]]:
    """Row indices of each member, in stored order (robust to reordered rows)."""
    groupings = {g.name: g for g in shard.annotation.groupings}
    if MEMBER_GROUPING in groupings:
        ids = groupings[MEMBER_GROUPING].ids
        keys = range(len(shard.members))
    else:
        ids = groupings[TIMELINE_GROUPING].ids
        labels = groupings[TIMELINE_GROUPING].labels
        keys = [labels[m.timeline] for m in shard.members]
    rows: dict[int, list[int]] = defaultdict(list)
    for row, key in enumerate(ids):
        rows[key].append(row)
    return [rows[key] for key in keys]


def split(
    boppard: Boppard, *, timelines: Iterable[str] | None = None
) -> list[Annotation]:
    """
    Rebuild the original annotations, optionally only those on some timelines.

    Every rebuilt annotation's content id is recomputed and must equal the id
    stored for it; otherwise BoppValidationError is raised.
    """
    wanted = None if timelines is None else set(timelines)
    media = {t.id: t.media_id for t in boppard.timelines}
    out = []
    for shard in boppard.shards:
        raw = msgspec.to_builtins(shard.annotation)
        own = [g for g in raw.get("groupings", []) if g["name"] not in _RESERVED]
        for member, rows in zip(shard.members, _member_rows(shard), strict=True):
            if wanted is not None and member.timeline not in wanted:
                continue
            if len(rows) != member.rows:
                raise BoppValidationError(
                    f"Member {member.id} has {len(rows)} rows, expected {member.rows}"
                )
            rebuilt: dict[str, Any] = {
                "media_id": media[member.timeline],
                "bopp_version": raw["bopp_version"],
                "id": member.id,
            }
            for name in ("parents", "metadata", "sandbox"):
                value = getattr(member, name)
                if value is not msgspec.UNSET:
                    rebuilt[name] = value
            for facet in _FACETS:
                if facet in raw:
                    rebuilt[facet] = {
                        k: [v[i] for i in rows] if isinstance(v, list) else v
                        for k, v in raw[facet].items()
                    }
            if own:
                rebuilt["groupings"] = [
                    {**g, "ids": [g["ids"][i] for i in rows]} for g in own
                ]
            ann = msgspec.convert(rebuilt, Annotation)
            validate_annotation_id(ann)
            out.append(ann)
    return out


def merge(*boppards: Boppard, collection_id: str | None = None) -> Boppard:
    """Concatenate BOPPARDs; shared timeline ids must have identical entries."""
    timelines: dict[str, Timeline] = {}
    links: dict[str, Link] = {}
    members: list[tuple[str, Annotation]] = []
    for boppard in boppards:
        for timeline in boppard.timelines:
            if timelines.setdefault(timeline.id, timeline) != timeline:
                raise BoppArgumentError(
                    f"Conflicting entries for timeline {timeline.id!r}"
                )
        for link in boppard.links:
            if links.setdefault(link.name, link) != link:
                raise BoppArgumentError(
                    f"Conflicting definitions of link {link.name!r}"
                )
        members += [(ann.media_id, ann) for ann in split(boppard)]
    by_media = {t.media_id: t.id for t in timelines.values()}
    return pack(
        [(by_media[media_id], ann) for media_id, ann in members],
        list(timelines.values()),
        collection_id=collection_id or boppards[0].collection_id,
        links=list(links.values()),
    )


def _payload_type(shard: Shard) -> str:
    return type(shard.annotation.payload).__struct_config__.tag


def resolve(boppard: Boppard, link: Link) -> pa.Table:
    """
    Join table of a link: one row per resolved source row.

    Columns are ``source_shard``, ``source_row``, ``target_shard`` and
    ``target_row`` (row indices within the shards). Values listed in
    ``link.missing`` are skipped. Raises BoppValidationError for a source
    value that names no target row, or names more than one.
    """
    timelines = {t.id: t for t in boppard.timelines}
    targets: dict[tuple[str, Any], list[tuple[int, int]]] = defaultdict(list)
    sources = []
    for number, shard in enumerate(boppard.shards):
        kind = _payload_type(shard)
        if kind not in (link.source_payload, link.target_payload):
            continue
        labels = {v: k for k, v in shard.annotation.groupings[0].labels.items()}
        ids = shard.annotation.groupings[0].ids
        if kind == link.target_payload:
            values = getattr(shard.annotation.payload, link.target_field)
            for row, value in enumerate(values):
                targets[(labels[ids[row]], value)].append((number, row))
        if kind == link.source_payload:
            sources.append((number, shard, labels, ids))
    columns: dict[str, list[int]] = defaultdict(list)
    missing = set(link.missing)
    for number, shard, labels, ids in sources:
        values = getattr(shard.annotation.payload, link.source_field)
        for row, value in enumerate(values):
            if value in missing:
                continue
            timeline = labels[ids[row]]
            scope = (
                timelines[timeline].aligned_to
                if link.scope == "aligned"
                else [timeline]
            )
            found = [hit for t in scope for hit in targets.get((t, value), [])]
            if len(found) != 1:
                raise BoppValidationError(
                    f"Link {link.name!r}: {value!r} on {timeline!r} matches "
                    f"{len(found)} target rows"
                )
            columns["source_shard"].append(number)
            columns["source_row"].append(row)
            columns["target_shard"].append(found[0][0])
            columns["target_row"].append(found[0][1])
    names = ("source_shard", "source_row", "target_shard", "target_row")
    return pa.table({n: pa.array(columns[n], pa.int32()) for n in names})


_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder(Boppard)
_JSON_DECODER = msgspec.json.Decoder(Boppard)


def encode(boppard: Boppard, fmt: str = "msgpack") -> bytes:
    """Serialize to ``msgpack`` (``.boppard``) or ``json`` (``.boppard.json``)."""
    return (
        _ENCODER.encode(boppard) if fmt == "msgpack" else msgspec.json.encode(boppard)
    )


def decode(data: bytes, fmt: str = "msgpack", *, validate: bool = True) -> Boppard:
    """Typed decode; with ``validate``, also check every shard's content id."""
    boppard = _DECODER.decode(data) if fmt == "msgpack" else _JSON_DECODER.decode(data)
    if validate:
        for shard in boppard.shards:
            validate_annotation_id(shard.annotation)
    return boppard


def save(boppard: Boppard, path: str | Path) -> None:
    """Write ``.boppard`` (msgpack) or ``.boppard.json``, by suffix."""
    path = Path(path)
    path.write_bytes(encode(boppard, "json" if path.suffix == ".json" else "msgpack"))


def load(path: str | Path, *, validate: bool = True) -> Boppard:
    """Read a file written by ``save``."""
    path = Path(path)
    fmt = "json" if path.suffix == ".json" else "msgpack"
    return decode(path.read_bytes(), fmt, validate=validate)
