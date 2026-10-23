"""Unpack a DLC release into the per-piece tree that canonical.load_dlc_facets reads.

A DLC release (e.g. v3.2, ``distant_listening_corpus.zip``) concatenates every
piece into one TSV per facet, with leading ``corpus`` and ``piece`` columns and
the union of all pieces' columns. ``split_release`` writes
``<out>/<corpus>/<facet_dir>/<piece>.<facet_dir>.tsv`` so that each piece looks
like its file in a DLC checkout:

- ``corpus`` and ``piece`` are dropped (canonical re-inserts them);
- columns empty throughout the piece are dropped, except the quarterbeat and
  duration columns, which the readers always look up;
- the measures ``i`` column is dropped (checkout measures TSVs have none);
- in chords, columns whose non-empty values are all integral floats (``"0.0"``)
  lose the ``.0``: the concatenation upcast integer spanner ids. Genuine float
  columns (``KEEP_FLOAT``) are left alone.

On the bundled Chopin fixture this reproduces the checkout TSVs value for value.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

#: Release table name and the facet directory it is written to.
RELEASE_TABLES = {
    "notes": "notes",
    "measures": "measures",
    "expanded": "harmonies",
    "chords": "chords",
}
#: Columns kept even when a piece leaves them empty.
ALWAYS = frozenset({"quarterbeats", "quarterbeats_all_endings", "duration_qb"})
#: Chords columns whose integral values stay floats.
KEEP_FLOAT = frozenset(
    {
        "quarterbeats",
        "quarterbeats_all_endings",
        "duration_qb",
        "qpm",
        "metronome_number",
        "tempo_visible",
    }
)


def _read(source: Path, name: str) -> pd.DataFrame:
    inner = f"distant_listening_corpus.{name}.tsv"
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive, archive.open(inner) as stream:
            return pd.read_csv(stream, sep="\t", dtype=str, keep_default_na=False)
    return pd.read_csv(source / inner, sep="\t", dtype=str, keep_default_na=False)


def _piece_table(group: pd.DataFrame, name: str) -> pd.DataFrame:
    group = group.drop(columns=["corpus", "piece"])
    group = group.loc[:, (group != "").any() | group.columns.isin(ALWAYS)]
    if name == "measures":
        group = group.drop(columns=["i"], errors="ignore")
    if name == "chords":
        for column in group.columns:
            if column in KEEP_FLOAT:
                continue
            values = group[column][group[column] != ""]
            if len(values) and values.str.fullmatch(r"-?\d+\.0").all():
                group[column] = group[column].str.removesuffix(".0")
    return group


def split_release(source: str | Path, out: str | Path) -> dict[str, int]:
    """
    Write a release's concatenated TSVs as one TSV per piece and facet.

    Parameters
    ----------
    source : str or pathlib.Path
        The release zip, or a directory holding its extracted TSVs.
    out : str or pathlib.Path
        Target root; existing piece files are overwritten.

    Returns
    -------
    dict of str to int
        Number of pieces written per facet directory.
    """
    source, out = Path(source), Path(out)
    counts = {}
    for name, facet_dir in RELEASE_TABLES.items():
        table = _read(source, name)
        count = 0
        for (corpus, piece), group in table.groupby(["corpus", "piece"], sort=False):
            directory = out / corpus / facet_dir
            directory.mkdir(parents=True, exist_ok=True)
            _piece_table(group, name).to_csv(
                directory / f"{piece}.{facet_dir}.tsv", sep="\t", index=False
            )
            count += 1
        counts[facet_dir] = count
    return counts
