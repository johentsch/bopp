"""Smoke-test the benchmark harness and the DLC release splitter on the fixtures."""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from experiments import bench, canonical
from experiments.dlc_release import RELEASE_TABLES, split_release

DATA = Path(__file__).parent / "data"
SOURCES = {
    "notes": "notes",
    "measures": "measures",
    "expanded": "harmonies",
    "chords": "chords",
}


def _fixture(directory: str) -> pd.DataFrame:
    return pd.read_csv(
        DATA / f"dlc_{directory}_chopin_mazurkas_BI163op67-4.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """A release-shaped zip: an extra empty union column and upcast spanner ids."""
    path = tmp_path_factory.mktemp("release") / "distant_listening_corpus.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, directory in RELEASE_TABLES.items():
            table = _fixture(directory)
            table["Ottava:8va"] = ""
            if name == "chords":
                for column in ("crescendo_hairpin", "decrescendo_hairpin"):
                    table[column] = table[column].where(
                        table[column] == "", table[column] + ".0"
                    )
            if name == "measures":
                table.insert(2, "i", [str(i) for i in range(len(table))])
            archive.writestr(
                f"distant_listening_corpus.{name}.tsv",
                table.to_csv(sep="\t", index=False),
            )
    return path


def test_split_release_reproduces_checkout_tables(release, tmp_path):
    counts = split_release(release, tmp_path)
    assert counts == dict.fromkeys(SOURCES.values(), 1)
    for directory in SOURCES.values():
        written = pd.read_csv(
            tmp_path / "chopin_mazurkas" / directory / f"BI163op67-4.{directory}.tsv",
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        source = _fixture(directory).drop(columns=["corpus", "piece"])
        kept = [c for c in source if (source[c] != "").any() or c in written]
        assert list(written.columns) == kept
        pd.testing.assert_frame_equal(written, source[kept])


def test_load_dlc_facets_collects_reader_errors(release, tmp_path):
    split_release(release, tmp_path)
    path = tmp_path / "chopin_mazurkas" / "chords" / "BI163op67-4.chords.tsv"
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    table["Ottava:8va"] = ""
    table.loc[0, "Ottava:8va"] = "1"
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(Exception, match="Ottava"):
        canonical.load_dlc_facets(tmp_path, ["control_events.chord"])
    errors = []
    frames = canonical.load_dlc_facets(
        tmp_path, ["notes", "control_events.chord"], errors=errors
    )
    assert [e[:3] for e in errors] == [("chords", "chopin_mazurkas", "BI163op67-4")]
    assert frames["control_events.chord"].empty
    assert len(frames["notes"]) == 822


def test_cells_cover_both_systems():
    names = [bench.cell_name(c) for c in bench.cells("notes")]
    assert len(names) == len(set(names)) == 18
    assert "bopp-per_item-csv-none" in names
    assert "bopp-per_item-csv-zstd" not in names
    assert "parquet-partitioned-parquet-zstd" in names


def test_run_and_report(release, tmp_path):
    bench.run(
        [
            "--dlc",
            str(release),
            "--out",
            str(tmp_path),
            "--facets",
            "measures",
            "--repeat",
            "1",
        ]
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 18
    assert {row["roundtrip"] for row in rows} == {"ok"}
    assert all(row["bytes"] > 0 and row["validated_s"] > 0 for row in rows)
    assert {row["piece"] for row in rows} == {"chopin_mazurkas/BI163op67-4"}
    assert not (tmp_path / "cells" / "measures").exists() or not any(
        (tmp_path / "cells" / "measures").iterdir()
    )
    markdown = bench.report(tmp_path / "results.jsonl")
    assert "### measures (83 rows, 1 pieces)" in markdown
    assert markdown.count("| bopp-") + markdown.count("| parquet-") == 18
