"""Benchmark BOPP storage layouts against Parquet on canonical DLC facets.

One *cell* is (facet, system, layout, format, compression). For every cell the
facet's canonical table is written once to a dedicated directory and then
measured:

``bytes``, ``files``
    Every stored byte and file of the cell.
``write_s``
    One write, canonical table to files (not repeated).
``decode_s``
    Files to an in-memory representation without validation: BOPP
    ``layouts_bopp.decode_only(validate=False)`` (builtins), Parquet
    ``layouts_parquet.read_table`` (an Arrow table).
``validated_s``
    The same with validation: BOPP typed ``Annotation`` decoding with id checks
    (``validate=True``), Parquet ``read_table`` plus
    ``arrow_schema.validate_table``. This is the main comparison.
``piece_s``
    Random access to one piece (the one with the median row count), validated.
    BOPP collections have no index, so their value is the whole validated
    decode (``piece_note = "whole file"``).
``polars_s``
    Parquet only: ``layouts_parquet.polars_load`` (native Polars, no checks).
``load_s``, ``roundtrip``
    One full load back to the canonical pandas table (validated), and whether it
    equals the source table exactly (``canonical.frames_equal``). The load is
    dominated by Python object construction on both sides; it guarantees that
    every cell stores the same information, its time is secondary.
``peak_mb``
    With ``--memory``: peak RSS growth in MiB of a fresh process during one
    validated decode (``VmHWM`` after minus before; the process has imported
    everything and read nothing yet). Linux only.

Timed operations run ``--repeat`` times (at least once, and stop early after
``--budget`` seconds); the result is the median, ``*_min`` the minimum. The
garbage collector runs before every repetition and stays enabled during it.

Usage::

    python -m experiments.bench --source <DLC tree or release zip> --out <workdir>
    python -m experiments.bench --dataset bpsfh --source <BPS_FH_Dataset> --out <workdir>
    python -m experiments.bench --dataset vienna --source <vienna4x22/match> --out <workdir>
    python -m experiments.bench report <workdir>/results.jsonl

A DLC release zip is split with ``dlc_release.split_release`` into
``<workdir>/dlc``; DLC pieces whose reader fails are skipped and listed in
``<workdir>/skipped.tsv``. BPS-FH tables come from ``experiments.bpsfh`` and
Vienna tables from ``experiments.vienna_payloads``; what the readers report
about the source goes to ``<workdir>/<dataset>-source.json``. Results append to
``<workdir>/results.jsonl``, one JSON object per cell, and ``report`` renders
them as markdown.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import pickle
import platform
import shutil
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import msgspec
import pandas as pd
import polars as pl
import pyarrow as pa

from experiments import (
    arrow_schema,
    bpsfh,
    canonical,
    dlc_release,
    vienna_match,
    vienna_payloads,
)
from experiments import layouts_bopp as bopp_layouts
from experiments import layouts_parquet as parquet_layouts

SYSTEMS = ("bopp", "parquet")


def cells(facet: str) -> Iterator[dict[str, str]]:
    """Yield every supported cell of a facet, BOPP first."""
    for layout, formats in bopp_layouts.FORMATS.items():
        for fmt in formats:
            for compression in ("none", "zstd"):
                if (
                    bopp_layouts.supported(layout, fmt, facet, compression=compression)
                    is True
                ):
                    yield {
                        "facet": facet,
                        "system": "bopp",
                        "layout": layout,
                        "format": fmt,
                        "compression": compression,
                    }
    for layout in ("per_item", "single", "partitioned"):
        for compression in ("none", "zstd"):
            yield {
                "facet": facet,
                "system": "parquet",
                "layout": layout,
                "format": "parquet",
                "compression": compression,
            }


def cell_name(cell: dict[str, str]) -> str:
    """Directory name of a cell, unique within a facet."""
    return "-".join(
        (cell["system"], cell["layout"], cell["format"], cell["compression"])
    )


def timed(
    func: Callable[[], Any], repeat: int, budget: float
) -> tuple[float, float, Any]:
    """Return median and minimum seconds of up to `repeat` runs, and a result."""
    times = []
    result = None
    start = time.perf_counter()
    for _ in range(max(1, repeat)):
        result = None
        gc.collect()
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)
        if time.perf_counter() - start > budget:
            break
    return statistics.median(times), min(times), result


class Cell:
    """Write, decode and load callables of one benchmark cell."""

    def __init__(self, spec: dict[str, str], directory: Path):
        self.spec = spec
        self.dir = directory
        self.facet = spec["facet"]
        self.payload_kind = canonical.FACETS[self.facet]["payload_kind"]
        self.extent_kind = canonical.facet_extent_kind(self.facet)

    @property
    def _bopp_kwargs(self) -> dict[str, str]:
        return {
            "layout": self.spec["layout"],
            "fmt": self.spec["format"],
            "compression": self.spec["compression"],
        }

    def write(self, frame: pd.DataFrame) -> list[Path]:
        if self.spec["system"] == "bopp":
            return bopp_layouts.write(
                frame,
                self.facet,
                self.dir,
                extent_kind=self.extent_kind,
                media_id=media_id_function(self.facet),
                **self._bopp_kwargs,
            )
        return parquet_layouts.write(
            frame,
            self.facet,
            self.dir,
            layout=self.spec["layout"],
            compression=self.spec["compression"],
            extent_kind=self.extent_kind,
        )

    def decode(self, validate: bool) -> Any:
        if self.spec["system"] == "bopp":
            return bopp_layouts.decode_only(
                self.dir, validate=validate, **self._bopp_kwargs
            )
        table = parquet_layouts.read_table(
            self.dir,
            self.facet,
            layout=self.spec["layout"],
            extent_kind=self.extent_kind,
        )
        if validate:
            table = arrow_schema.validate_table(
                table, self.payload_kind, self.extent_kind
            )
        return table

    def piece(self, path: Path | None, pair: tuple[str, str]) -> Any:
        if self.spec["system"] == "parquet":
            table = parquet_layouts.read_table(
                self.dir,
                self.facet,
                layout=self.spec["layout"],
                pieces=[pair],
                extent_kind=self.extent_kind,
            )
            return arrow_schema.validate_table(
                table, self.payload_kind, self.extent_kind
            )
        if self.spec["layout"] == "per_item":
            return bopp_layouts.decode_only(path, validate=True, **self._bopp_kwargs)
        return self.decode(True)

    def polars(self) -> pl.DataFrame:
        return parquet_layouts.polars_load(
            self.dir,
            self.facet,
            layout=self.spec["layout"],
            extent_kind=self.extent_kind,
        )

    def load(self) -> pd.DataFrame:
        if self.spec["system"] == "bopp":
            return bopp_layouts.load(
                self.dir, self.facet, validate=True, **self._bopp_kwargs
            )
        return parquet_layouts.load(
            self.dir,
            self.facet,
            layout=self.spec["layout"],
            validate=True,
            extent_kind=self.extent_kind,
        )


def _median_piece(frame: pd.DataFrame) -> tuple[str, str]:
    sizes = (
        frame.groupby(["corpus", "piece"], sort=True).size().sort_values(kind="stable")
    )
    return sizes.index[len(sizes) // 2]


def _piece_path(files: list[Path], pair: tuple[str, str], facet: str) -> Path | None:
    corpus, piece = pair
    for path in files:
        if path.parent.name == corpus and path.name.startswith(f"{piece}.{facet}."):
            return path
    return None


def _hwm_kib() -> int:
    """Peak RSS of this process image in KiB (VmHWM).

    ``ru_maxrss`` is not used: Linux carries it across ``execve``, so a spawned
    child would report the parent's peak.
    """
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise RuntimeError("VmHWM is not available")


def _memory_child(spec: dict[str, str], directory: str, queue: Any) -> None:
    cell = Cell(spec, Path(directory))
    gc.collect()
    before = _hwm_kib()
    result = cell.decode(True)
    after = _hwm_kib()
    del result
    queue.put((after - before) / 1024)


def peak_mb(spec: dict[str, str], directory: Path) -> float:
    """Peak RSS growth in MiB of one validated decode in a fresh process."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_memory_child, args=(spec, str(directory), queue))
    process.start()
    value = queue.get()
    process.join()
    return value


def measure(
    spec: dict[str, str],
    frame: pd.DataFrame,
    work: Path,
    *,
    repeat: int,
    budget: float,
    memory: bool,
    roundtrip: bool,
) -> dict[str, Any]:
    """Write one cell, measure it, and return its result row."""
    directory = work / "cells" / spec["facet"] / cell_name(spec)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    cell = Cell(spec, directory)
    row: dict[str, Any] = dict(
        spec,
        rows=len(frame),
        pieces=frame[["corpus", "piece"]].drop_duplicates().shape[0],
    )
    t0 = time.perf_counter()
    files = cell.write(frame)
    row["write_s"] = time.perf_counter() - t0
    row["files"] = len(files)
    row["bytes"] = sum(path.stat().st_size for path in files)
    try:
        for key, validate in (("decode_s", False), ("validated_s", True)):
            row[key], row[f"{key[:-2]}_min_s"], _ = timed(
                lambda validate=validate: cell.decode(validate), repeat, budget
            )
        pair = _median_piece(frame)
        row["piece"] = "/".join(pair)
        if spec["system"] == "bopp" and spec["layout"] == "collection":
            row["piece_s"], row["piece_note"] = row["validated_s"], "whole file"
        else:
            path = _piece_path(files, pair, spec["facet"])
            row["piece_s"], _, _ = timed(
                lambda: cell.piece(path, pair), max(repeat, 5), budget
            )
        if spec["system"] == "parquet":
            row["polars_s"], _, _ = timed(cell.polars, repeat, budget)
        if roundtrip:
            t0 = time.perf_counter()
            loaded = cell.load()
            row["load_s"] = time.perf_counter() - t0
            differences = canonical.frames_equal(frame, loaded)
            row["roundtrip"] = "ok" if not differences else "; ".join(differences[:3])
            del loaded
        if memory:
            row["peak_mb"] = peak_mb(spec, directory)
    except Exception as err:  # noqa: BLE001 (a failing cell is a result)
        row["error"] = f"{type(err).__name__}: {err}"[:300]
    return row


def environment() -> dict[str, Any]:
    """Versions and machine facts recorded with every run."""
    import bopp

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "bopp": getattr(bopp, "__version__", "?"),
        "msgspec": msgspec.__version__,
        "pyarrow": pa.__version__,
        "polars": pl.__version__,
        "pandas": pd.__version__,
    }


def dataset_of(facet: str) -> str:
    """Dataset a facet belongs to: dlc, bpsfh or vienna."""
    return facet.split(".", 1)[0] if facet in canonical.OTHER_FACETS else "dlc"


def media_id_function(facet: str) -> Callable[[str, str], str] | None:
    """Media id builder of a facet's dataset; None keeps the DLC default."""
    dataset = dataset_of(facet)
    if dataset == "dlc":
        return None
    return lambda corpus, piece: f"{dataset}:{corpus}/{piece}"


DATASETS = {
    "dlc": canonical.DLC_FACETS,
    **{
        dataset: tuple(f for f in canonical.OTHER_FACETS if dataset_of(f) == dataset)
        for dataset in ("bpsfh", "vienna")
    },
}
"""Facets of each benchmark dataset."""


def _frames(annotations: list[tuple[str, str, Any]]) -> pd.DataFrame:
    return pd.concat(
        [
            canonical.annotation_to_frame(ann, corpus=corpus, piece=piece)
            for corpus, piece, ann in sorted(annotations, key=lambda t: t[:2])
        ],
        ignore_index=True,
    )


def _bpsfh_tables(source: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    inverted: list[tuple[str, int]] = []
    inverted_chords: list[tuple[str, int]] = []
    respelled: list[tuple[str, int]] = []
    rows: dict[str, list] = {f: [] for f in DATASETS["bpsfh"]}
    for piece in bpsfh.pieces(source):
        for facet, ann in (
            ("bpsfh.notes", bpsfh.read_notes(source, piece, respelled=respelled)),
            (
                "bpsfh.chords",
                bpsfh.read_chords(source, piece, inverted=inverted_chords),
            ),
            ("bpsfh.phrases", bpsfh.read_phrases(source, piece, inverted=inverted)),
        ):
            rows[facet].append((bpsfh.CORPUS, piece, ann))
    notes = {
        "inverted_phrases": inverted,
        "inverted_chords": inverted_chords,
        "respelled_notes": respelled,
    }
    return {facet: _frames(anns) for facet, anns in rows.items()}, notes


def _vienna_tables(source: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    dropped: Counter = Counter()
    rows: dict[str, list] = {f: [] for f in DATASETS["vienna"]}
    fallbacks = {}
    for piece in vienna_payloads.pieces(source):
        ann = vienna_payloads.score_note_annotation(
            source, piece, dropped_attributes=dropped
        )
        rows["vienna.score_notes"].append((vienna_payloads.CORPUS, piece, ann))
        fallbacks[piece] = int(
            vienna_match.score_notes(source, piece)["from_onset_in_beats"].sum()
        )
    for path in sorted(Path(source).glob("*.match")):
        performance, pedal = vienna_payloads.performance_annotations(path)
        rows["vienna.performance_notes"].append(
            (vienna_payloads.CORPUS, path.stem, performance)
        )
        rows["vienna.pedal"].append((vienna_payloads.CORPUS, path.stem, pedal))
    notes = {
        "dropped_attributes": dict(dropped),
        "positions_from_onset_in_beats": fallbacks,
    }
    return {facet: _frames(anns) for facet, anns in rows.items()}, notes


def _dlc_tables(
    source: Path, work: Path, corpora: list[str] | None
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if source.suffix == ".zip":
        tree = work / "dlc"
        if not tree.exists():
            print(f"splitting {source} into {tree}", flush=True)
            dlc_release.split_release(source, tree)
        source = tree
    errors: list[tuple[str, str, str, Exception]] = []
    tables = canonical.load_dlc_facets(
        source, list(canonical.DLC_FACETS), corpora=corpora, errors=errors
    )
    pd.DataFrame(
        [(d, c, p, f"{type(e).__name__}: {e}") for d, c, p, e in errors],
        columns=["facet_dir", "corpus", "piece", "error"],
    ).to_csv(work / "skipped.tsv", sep="\t", index=False)
    return tables, {"skipped_pieces": len(errors)}


def prepare(
    dataset: str,
    source: Path,
    work: Path,
    facets: list[str],
    corpora: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build (or reuse) the canonical tables of a dataset.

    DLC pieces whose reader fails are skipped and listed in ``skipped.tsv``.
    What the readers report about the source (skipped pieces, BPS-FH phrases
    swapped and notes respelled, Vienna attributes without a column) is written
    to ``<dataset>-source.json``.
    """
    name = dataset if corpora is None else f"{dataset}-{'+'.join(corpora)}"
    cache = work / f"canonical-{name}.pkl"
    if cache.exists():
        tables = pickle.loads(cache.read_bytes())
        if all(facet in tables for facet in facets):
            return {facet: tables[facet] for facet in facets}
    t0 = time.perf_counter()
    if dataset == "dlc":
        tables, notes = _dlc_tables(source, work, corpora)
    elif dataset == "bpsfh":
        tables, notes = _bpsfh_tables(source)
    else:
        tables, notes = _vienna_tables(source)
    print(f"canonical tables in {time.perf_counter() - t0:.0f}s: {notes}", flush=True)
    (work / f"{dataset}-source.json").write_text(json.dumps(notes, indent=1))
    cache.write_bytes(pickle.dumps(tables))
    return {facet: tables[facet] for facet in facets}


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="dlc")
    parser.add_argument(
        "--source",
        "--dlc",
        type=Path,
        required=True,
        help="DLC tree or release zip, BPS_FH_Dataset directory, or Vienna match directory",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--facets", nargs="*", default=None)
    parser.add_argument("--corpora", nargs="*", default=None)
    parser.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--no-roundtrip", dest="roundtrip", action="store_false")
    parser.add_argument("--keep", action="store_true", help="keep cell directories")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.facets is None:
        args.facets = list(DATASETS[args.dataset])
    tables = prepare(args.dataset, args.source, args.out, args.facets, args.corpora)
    env = environment()
    results = args.out / "results.jsonl"
    with results.open("a") as stream:
        for facet in args.facets:
            for spec in cells(facet):
                if spec["system"] not in args.systems:
                    continue
                row = measure(
                    spec,
                    tables[facet],
                    args.out,
                    repeat=args.repeat,
                    budget=args.budget,
                    memory=args.memory,
                    roundtrip=args.roundtrip,
                )
                row["env"] = env
                row["corpora"] = args.corpora
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(_line(row), flush=True)
                if not args.keep:
                    shutil.rmtree(args.out / "cells" / facet / cell_name(spec))


def _fmt(value: Any, spec: str, scale: float = 1.0) -> str:
    return "" if value is None or pd.isna(value) else format(value * scale, spec)


def _line(row: dict[str, Any]) -> str:
    if "error" in row:
        return (
            f"{row['facet']:<28} {cell_name(row):<36} "
            f"{row['bytes'] / 2**20:8.1f} MiB {row['error']}"
        )
    return (
        f"{row['facet']:<28} {cell_name(row):<36} {row['bytes'] / 2**20:8.1f} MiB "
        f"decode {row['decode_s']:7.3f}s validated {row['validated_s']:7.3f}s "
        f"piece {row['piece_s'] * 1e3:8.1f}ms {row.get('roundtrip', '')}"
    )


def report(path: Path) -> str:
    """Render results as markdown, one table per facet, last run per cell."""
    rows: dict[tuple, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[(row["facet"], cell_name(row))] = row
    frame = pd.DataFrame(rows.values())
    out = []
    env = frame["env"].iloc[-1]
    out.append("Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()))
    for facet, group in frame.groupby("facet", sort=False):
        first = group.iloc[0]
        out.append(
            f"\n### {facet} ({first['rows']:,} rows, {first['pieces']} pieces)\n"
        )
        out.append(
            "| cell | MiB | files | write s | decode s | validated s | piece ms | polars s | load s | round trip |"
        )
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in group.iterrows():
            piece = _fmt(r.get("piece_s"), ".1f", 1e3) + (
                " (whole)" if r.get("piece_note") == "whole file" else ""
            )
            error = r.get("error")
            status = r.get("roundtrip", "") if not isinstance(error, str) else error
            out.append(
                f"| {cell_name(r)} | {r['bytes'] / 2**20:.1f} | {r['files']} "
                f"| {r['write_s']:.1f} | {_fmt(r.get('decode_s'), '.3f')} "
                f"| {_fmt(r.get('validated_s'), '.3f')} | {piece} "
                f"| {_fmt(r.get('polars_s'), '.3f')} | {_fmt(r.get('load_s'), '.1f')} "
                f"| {status} |"
            )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "report":
        print(report(Path(argv[1])))
    else:
        run(argv)


if __name__ == "__main__":
    main()
