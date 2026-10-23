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
    With ``--memory``: peak RSS growth of a fresh process during one validated
    decode (``ru_maxrss`` after minus before; the process has imported
    everything and read nothing yet).

Timed operations run ``--repeat`` times (at least once, and stop early after
``--budget`` seconds); the result is the median, ``*_min`` the minimum. The
garbage collector runs before every repetition and stays enabled during it.

Usage::

    python -m experiments.bench --dlc <tree or release zip> --out <workdir>
    python -m experiments.bench report <workdir>/results.jsonl

``--dlc`` accepts a DLC checkout-like tree or a release zip, which is split
with ``dlc_release.split_release`` into ``<workdir>/dlc``. Pieces whose reader
fails are skipped and listed in ``<workdir>/skipped.tsv``. Results append to
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
import resource
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import msgspec
import pandas as pd
import polars as pl
import pyarrow as pa

from experiments import arrow_schema, canonical, dlc_release
from experiments import layouts_bopp as bopp_layouts
from experiments import layouts_parquet as parquet_layouts

_EXTENT_KIND = "quarters_interval.fraction"
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

    @property
    def _bopp_kwargs(self) -> dict[str, str]:
        return {
            "layout": self.spec["layout"],
            "fmt": self.spec["format"],
            "compression": self.spec["compression"],
        }

    def write(self, frame: pd.DataFrame) -> list[Path]:
        if self.spec["system"] == "bopp":
            return bopp_layouts.write(frame, self.facet, self.dir, **self._bopp_kwargs)
        return parquet_layouts.write(
            frame,
            self.facet,
            self.dir,
            layout=self.spec["layout"],
            compression=self.spec["compression"],
        )

    def decode(self, validate: bool) -> Any:
        if self.spec["system"] == "bopp":
            return bopp_layouts.decode_only(
                self.dir, validate=validate, **self._bopp_kwargs
            )
        table = parquet_layouts.read_table(
            self.dir, self.facet, layout=self.spec["layout"]
        )
        if validate:
            table = arrow_schema.validate_table(table, self.payload_kind, _EXTENT_KIND)
        return table

    def piece(self, path: Path | None, pair: tuple[str, str]) -> Any:
        if self.spec["system"] == "parquet":
            table = parquet_layouts.read_table(
                self.dir, self.facet, layout=self.spec["layout"], pieces=[pair]
            )
            return arrow_schema.validate_table(table, self.payload_kind, _EXTENT_KIND)
        if self.spec["layout"] == "per_item":
            return bopp_layouts.decode_only(path, validate=True, **self._bopp_kwargs)
        return self.decode(True)

    def polars(self) -> pl.DataFrame:
        return parquet_layouts.polars_load(
            self.dir, self.facet, layout=self.spec["layout"]
        )

    def load(self) -> pd.DataFrame:
        if self.spec["system"] == "bopp":
            return bopp_layouts.load(
                self.dir, self.facet, validate=True, **self._bopp_kwargs
            )
        return parquet_layouts.load(
            self.dir, self.facet, layout=self.spec["layout"], validate=True
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


def _memory_child(spec: dict[str, str], directory: str, queue: Any) -> None:
    cell = Cell(spec, Path(directory))
    gc.collect()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = cell.decode(True)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    del result
    queue.put((after - before) / 1024)  # ru_maxrss is in KiB on Linux


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


def prepare(
    dlc: Path, work: Path, facets: list[str], corpora: list[str] | None
) -> dict[str, pd.DataFrame]:
    """Build (or reuse) the canonical tables, skipping pieces whose reader fails."""
    cache = work / (
        "canonical.pkl" if corpora is None else f"canonical-{'+'.join(corpora)}.pkl"
    )
    if cache.exists():
        tables = pickle.loads(cache.read_bytes())
        if all(facet in tables for facet in facets):
            return {facet: tables[facet] for facet in facets}
    if dlc.suffix == ".zip":
        tree = work / "dlc"
        if not tree.exists():
            print(f"splitting {dlc} into {tree}", flush=True)
            dlc_release.split_release(dlc, tree)
        dlc = tree
    errors: list[tuple[str, str, str, Exception]] = []
    t0 = time.perf_counter()
    tables = canonical.load_dlc_facets(
        dlc, list(canonical.FACETS), corpora=corpora, errors=errors
    )
    print(
        f"canonical tables in {time.perf_counter() - t0:.0f}s, {len(errors)} pieces skipped",
        flush=True,
    )
    pd.DataFrame(
        [(d, c, p, f"{type(e).__name__}: {e}") for d, c, p, e in errors],
        columns=["facet_dir", "corpus", "piece", "error"],
    ).to_csv(work / "skipped.tsv", sep="\t", index=False)
    cache.write_bytes(pickle.dumps(tables))
    return {facet: tables[facet] for facet in facets}


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dlc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--facets", nargs="*", default=list(canonical.FACETS))
    parser.add_argument("--corpora", nargs="*", default=None)
    parser.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--no-roundtrip", dest="roundtrip", action="store_false")
    parser.add_argument("--keep", action="store_true", help="keep cell directories")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tables = prepare(args.dlc, args.out, args.facets, args.corpora)
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


def _line(row: dict[str, Any]) -> str:
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
            piece = f"{r['piece_s'] * 1e3:.1f}" + (
                " (whole)" if r.get("piece_note") == "whole file" else ""
            )
            polars = "" if pd.isna(r.get("polars_s")) else f"{r['polars_s']:.3f}"
            load = "" if pd.isna(r.get("load_s")) else f"{r['load_s']:.1f}"
            out.append(
                f"| {cell_name(r)} | {r['bytes'] / 2**20:.1f} | {r['files']} | {r['write_s']:.1f} "
                f"| {r['decode_s']:.3f} | {r['validated_s']:.3f} | {piece} | {polars} | {load} "
                f"| {r.get('roundtrip', '')} |"
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
