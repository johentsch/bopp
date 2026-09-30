"""Export Distant Listening Corpus facets as BOPP objects.

Homogeneous facets (notes, expanded harmonies) become one ``Annotation`` per
(corpus, piece) pair.  The heterogeneous ``chords`` facet, whose rows are
events of seven different kinds, becomes one ``ScoreObject`` per piece: every
event type keeps its own strictly typed payload block, and all blocks travel in
the same file.  Every object is written in MessagePack, JSON, and CSV so the
three representations remain directly comparable.

All extents are ``time_interval`` extents measured in ``quarters``, i.e. exact
fractions, following the timetoalign coordinate model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import msgspec
import numpy as np
import pandas as pd
from dimcat import Dataset

from bopp.io import decode_arrow, encode_arrow, load_bopp_csv, load_bopp_json, load_bopp_msgpack
from bopp.io import save_bopp_json, save_bopp_msgpack, to_csv
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.core import FractionBuffer
from bopp.models.v1.score_object import ScoreObject
from bopp.util import _field_types, _payload_struct, fractions_to_buffer, to_dataframe


INDEX_COLUMNS = {"corpus", "piece", "i"}
EXTENT_COLUMNS = {"quarterbeats_all_endings", "duration_qb"}


@dataclass(frozen=True)
class FacetSpec:
    """Rules for translating one DLC resource into BOPP payloads."""

    name: str
    resource_name: str
    payload_type: str | None = None
    payload_start: str | None = None
    # For heterogeneous facets: value of the discriminating column -> payload type.
    event_column: str | None = None
    event_payloads: dict[str, str] | None = None

    @property
    def is_score_object(self) -> bool:
        return self.event_column is not None


FACETS = {
    "notes": FacetSpec("notes", "distant_listening_corpus.notes", "score_note"),
    # DLC calls this resource "expanded".  Accept "extended" as a convenient
    # public alias because it is commonly used to describe this facet.
    "extended": FacetSpec("extended", "distant_listening_corpus.expanded", "dcml_harmony", "label"),
    "chords": FacetSpec(
        "chords",
        "distant_listening_corpus.chords",
        event_column="event",
        event_payloads={
            "Chord": "score_chord",
            "Dynamic": "score_dynamic",
            "Spanner": "score_spanner",
            "FiguredBass": "score_figured_bass",
            "StaffText": "score_staff_text",
            "SystemText": "score_system_text",
            "Tempo": "score_tempo",
        },
    ),
}

# DLC columns that dimcat leaves as strings but whose BOPP payload declares a number/bool.
FLOAT_COLUMNS = {"qpm", "metronome_number", "scalar"}
BOOL_COLUMNS = {"tempo_visible"}


def load_dlc(root: str | Path) -> tuple[dict[str, Any], Any]:
    """Load the DLC descriptor and dimcat package rooted at *root*."""
    root = Path(root)
    descriptor = msgspec.json.decode((root / "distant_listening_corpus.datapackage.json").read_bytes())
    dataset = Dataset()
    dataset.load(str(root / "distant_listening_corpus.datapackage.json"))
    return descriptor, dataset.inputs.get_package("distant_listening_corpus")


def get_facet(descriptor: dict[str, Any], package: Any, facet: str) -> tuple[FacetSpec, list[dict[str, Any]], pd.DataFrame]:
    """Return the translation spec, descriptor fields, and DataFrame for a facet."""
    try:
        spec = FACETS[facet]
    except KeyError as exc:
        raise ValueError(f"Unknown facet {facet!r}; choose from {sorted(FACETS)}") from exc
    resource = next(item for item in descriptor["resources"] if item["name"] == spec.resource_name)
    fields = resource["schema"]["fields"]
    return spec, fields, package.get_resource_by_name(spec.resource_name).df


def payload_columns(spec: FacetSpec, fields: Iterable[dict[str, Any]]) -> list[str]:
    """Derive payload columns from the descriptor, without hard-coded lists."""
    if spec.is_score_object:
        return sorted({name for ptype in spec.event_payloads.values() for name in struct_columns(ptype)})
    fields = list(fields)
    if spec.payload_start:
        names = [field["name"] for field in fields]
        return names[names.index(spec.payload_start) :]
    return [field["name"] for field in fields if field["name"] not in INDEX_COLUMNS | EXTENT_COLUMNS]


def struct_columns(payload_type: str) -> list[str]:
    """The payload columns a BOPP payload schema declares (everything but the tag)."""
    return [name for name in _field_types(_payload_struct(payload_type)) if name != "payload_type"]


def scalar(value: Any) -> Any:
    """Convert pandas, NumPy, and Fraction scalars to serializable Python values."""
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    # DLC uses empty TSV fields for missing values, including in columns that
    # are otherwise numeric.  Preserve the missingness as an Arrow null rather
    # than passing an invalid empty string to a numeric BOPP buffer.
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (tuple, list, np.ndarray)):
        return [scalar(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Fraction):
        return float(value)
    return value


def to_fraction(value: Any) -> Fraction | None:
    """Exact rational from whatever dimcat hands us (Fraction, '1/4', 0.5, int)."""
    value = scalar(value) if not isinstance(value, Fraction) else value
    if value is None:
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        # Floats such as duration_qb are decimal renderings of exact durations.
        return Fraction(value).limit_denominator(4096)
    return Fraction(str(value))


def fraction_column(values: Iterable[Any]) -> dict[str, list[int | None]]:
    return fractions_to_buffer(to_fraction(value) for value in values)


def quarter_extent(frame: pd.DataFrame) -> dict[str, Any]:
    """time_interval extent in quarters: exact start positions and durations."""
    starts = [to_fraction(value) for value in frame["quarterbeats_all_endings"].tolist()]
    if "duration" in frame.columns:
        # `duration` is the exact fraction of a whole note; quarters = 4 * whole notes.
        durations = [None if (d := to_fraction(value)) is None else d * 4 for value in frame["duration"].tolist()]
    else:
        durations = [to_fraction(value) for value in frame["duration_qb"].tolist()]
    return {
        "extent_type": "time_interval",
        "start": {"unit": "quarters", "values": fractions_to_buffer(starts)},
        "duration": {"unit": "quarters", "values": fractions_to_buffer(durations)},
    }


def payload_from_frame(payload_type: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Build the raw payload dict, converting each column to the type its schema declares."""
    payload: dict[str, Any] = {"payload_type": payload_type}
    for column, field_type in _field_types(_payload_struct(payload_type)).items():
        if column == "payload_type":
            continue
        values = frame[column].tolist() if column in frame.columns else [None] * len(frame)
        if field_type is FractionBuffer:
            payload[column] = fraction_column(values)
        elif column in FLOAT_COLUMNS:
            payload[column] = [None if (v := scalar(value)) is None else float(v) for value in values]
        elif column in BOOL_COLUMNS:
            payload[column] = [None if (v := scalar(value)) is None else bool(float(v)) for value in values]
        else:
            payload[column] = [scalar(value) for value in values]
    return payload


def annotation_from_piece(spec: FacetSpec, fields: list[dict[str, Any]], corpus: str, piece: str, frame: pd.DataFrame) -> Annotation:
    """Build a validated BOPP annotation for one DLC piece of a homogeneous facet."""
    raw = {
        "object_type": "annotation",
        "media_id": f"dlc:{corpus}/{piece}",
        "bopp_version": "1.0.0",
        "extent": quarter_extent(frame),
        "payload": payload_from_frame(spec.payload_type, frame),
    }
    return msgspec.convert(raw, type=Annotation, dec_hook=decode_arrow)


def score_object_from_piece(spec: FacetSpec, corpus: str, piece: str, frame: pd.DataFrame) -> ScoreObject:
    """Build a validated ScoreObject: one strictly typed event block per event type present."""
    events = []
    for event, block_frame in frame.groupby(spec.event_column, sort=True):
        payload_type = spec.event_payloads[event]
        events.append({"extent": quarter_extent(block_frame), "payload": payload_from_frame(payload_type, block_frame)})
    raw = {
        "object_type": "score_object",
        "media_id": f"dlc:{corpus}/{piece}",
        "bopp_version": "1.0.0",
        "events": events,
    }
    return msgspec.convert(raw, type=ScoreObject, dec_hook=decode_arrow)


def object_from_piece(spec: FacetSpec, fields: list[dict[str, Any]], corpus: str, piece: str, frame: pd.DataFrame) -> Annotation | ScoreObject:
    if spec.is_score_object:
        return score_object_from_piece(spec, corpus, piece, frame)
    return annotation_from_piece(spec, fields, corpus, piece, frame)


def _safe_stem(corpus: str, piece: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{corpus}__{piece}")


def _csv_value(value: Any) -> Any:
    """Canonicalize the intentional CSV type losses before comparing values."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        return [_csv_value(item) for item in value]
    if isinstance(value, (bool, Fraction)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _csv_values_equal(left: Any, right: Any) -> bool:
    """Compare CSV-restored values, allowing its decimal float rounding."""
    left, right = _csv_value(left), _csv_value(right)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_csv_values_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7)
    return left == right


def verify_round_trips(obj: Annotation | ScoreObject, paths: dict[str, Path]) -> None:
    """Check lossless binary/JSON round trips and CSV's value-level round trip."""
    encoded = msgspec.msgpack.encode(obj, enc_hook=encode_arrow)
    assert msgspec.msgpack.encode(load_bopp_msgpack(str(paths["msgpack"])), enc_hook=encode_arrow) == encoded
    assert msgspec.msgpack.encode(load_bopp_json(str(paths["json"])), enc_hook=encode_arrow) == encoded
    original, restored = to_dataframe(obj), to_dataframe(load_bopp_csv(str(paths["csv"])))
    assert original.shape == restored.shape and list(original.columns) == list(restored.columns), (original.shape, restored.shape)
    for column in original.columns:
        assert all(_csv_values_equal(left, right) for left, right in zip(original[column], restored[column])), column


def export_facet(root: str | Path, facet: str, output_dir: str | Path | None = None, max_pieces: int | None = None, dlc: tuple[dict[str, Any], Any] | None = None) -> pd.DataFrame:
    """Export pieces of one facet to all formats and return their file sizes."""
    root = Path(root)
    descriptor, package = dlc or load_dlc(root)
    spec, fields, frame = get_facet(descriptor, package, facet)
    destination = Path(output_dir) if output_dir else root / "dlc_bopp_exports" / spec.name
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for piece_number, ((corpus, piece), piece_frame) in enumerate(
        frame.groupby(level=["corpus", "piece"], sort=True)
    ):
        if max_pieces is not None and piece_number >= max_pieces:
            break
        base = destination / _safe_stem(corpus, piece)
        paths = {"msgpack": base.with_suffix(".bopp.msgpack"), "json": base.with_suffix(".bopp.json"), "csv": base.with_suffix(".bopp.csv")}
        if not all(path.exists() for path in paths.values()):
            obj = object_from_piece(spec, fields, corpus, piece, piece_frame)
            save_bopp_msgpack(obj, str(paths["msgpack"]))
            save_bopp_json(obj, str(paths["json"]))
            to_csv(obj, str(paths["csv"]))
            # Full CSV reconstruction is intentionally done once per facet: it
            # is representative of every shared schema but avoids re-reading
            # the complete corpus three times for every individual piece.
            if piece_number == 0:
                verify_round_trips(obj, paths)
        rows.extend({"facet": spec.name, "corpus": corpus, "piece": piece, "observations": len(piece_frame), "format": name, "bytes": path.stat().st_size} for name, path in paths.items())
    sizes = pd.DataFrame(rows)
    sizes["kib"] = sizes["bytes"] / 1024
    sizes["bytes_per_observation"] = sizes["bytes"] / sizes["observations"]
    return sizes


def export_all(root: str | Path, output_dir: str | Path | None = None, max_pieces: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export all facets, plus per-facet/format size summaries."""
    root = Path(root)
    output_dir = Path(output_dir) if output_dir else root / "dlc_bopp_exports"
    dlc = load_dlc(root)
    sizes = pd.concat([export_facet(root, facet, output_dir / facet, max_pieces, dlc) for facet in FACETS], ignore_index=True)
    summary = sizes.groupby(["facet", "format"], as_index=False).agg(
        files=("bytes", "size"), observations=("observations", "sum"), total_bytes=("bytes", "sum"), mean_bytes_per_observation=("bytes_per_observation", "mean")
    )
    summary["total_kib"] = summary["total_bytes"] / 1024
    summary.to_csv(output_dir / "format_size_comparison.csv", index=False)
    sizes.to_csv(output_dir / "format_sizes_by_piece.csv", index=False)
    return sizes, summary
