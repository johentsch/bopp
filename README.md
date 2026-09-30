BOPP: The Bounded Observation Payload Protocol
==============================================

Overview
--------

**BOPP** (Bounded Observation Payload Protocol) is a high-performance, strongly typed annotation framework for audio and music signal processing data built on top of [`msgspec`](https://jcristharif.com/msgspec/). It provides strict schema validation, fast JSON and MessagePack serialization, and seamless conversion to tabular formats (Pandas DataFrames, Apache Arrow, and CSV).

Key Differences from JAMS
-------------------------

While JAMS (JSON Annotated Music Specification) organizes annotations as collections of individual observation dictionaries, BOPP approaches audio annotations with a **columnar/tabular mindset**:

1. **Columnar Data Layout**:
   - *JAMS*: Stores annotations as a list of observation objects, where each observation contains scalar fields (`time`, `duration`, `value`, `confidence`).
   - *BOPP*: Stores annotation facets (`extents`, `payloads`, `confidences`) in parallel, synchronized arrays (e.g., `time` array, `duration` array, `value` array). This layout aligns natively with tabular data structures (DataFrames / Arrow arrays) and enables zero-copy operations and fast vectorized processing.

2. **Strict Typing & Tagged Unions**:
   - *JAMS*: Uses JSON Schema validation over generic dictionaries.
   - *BOPP*: Enforces runtime type-safety through C-accelerated `msgspec.Struct` classes using explicit tagged unions (`payload_type`, `extent_type`, `metadata_type`, `confidence_type`).

3. **Performance & Efficiency**:
   - *JAMS*: Relies on pure-Python dictionary parsing and `jsonschema` validation, which can become a bottleneck during large-scale ML data loading.
   - *BOPP*: Achieves sub-millisecond serialization and validation speeds via C-optimized parsing and efficient MessagePack binary representations.

Example Use Cases
-----------------

- **Machine Learning Data Pipelines**: Rapidly load and batch audio metadata, timestamps, and target payloads (e.g., chords, beats, notes, pitch contours) into training loops without deserialization bottlenecks.
- **Data Analysis & Querying**: Instantly convert annotations to Pandas DataFrames or Apache Arrow tables via `bopp.util.to_dataframe` or `bopp.io.read_bopp_csv` for data manipulation, filtering, and visualization.
- **Interoperability & Interchange**: Export annotations losslessly to CSV or MessagePack for compact storage, sharing, or downstream consumption.

Coordinates (proposal)
----------------------

Extents are expressed as **coordinates** following the [timetoalign](https://timetoalign.github.io) model: a
coordinate is a columnar array of values measured in a *unit*, and every unit is coupled to exactly one number type
(`schemas/v1/coordinate.json`, a `oneOf` discriminated by `unit`):

| unit class          | units                                                                     | number type | buffer                                     |
|---------------------|---------------------------------------------------------------------------|-------------|--------------------------------------------|
| discrete            | `ticks`, `samples`, `frames`, `pixels`                                    | `int`       | `Int64Buffer`                              |
| continuous symbolic | `quarters`, `whole_note`                                                  | `fraction`  | `FractionBuffer` (`numerator[]`, `denominator[]`) |
| continuous          | `seconds`, `milliseconds`, `minutes`, `floating_measures`, `number`, `meters`, `centimeters`, `millimeters`, `inches`, `points` | `float` | `Float64Buffer` |

`timestamps` carries one coordinate (`time`); `time_interval` and `TimeFrequencyBox` carry two (`start`, `duration`),
which must share a unit. Because the unit tag fully determines the buffer type, readers never guess precision from
the data, and non-second timelines no longer need a dedicated extent type (`quarter_interval` is now
`time_interval` + `unit: quarters`). See `coordinates_demo.ipynb`.

Objects (proposal)
------------------

The root of a BOPP file is an **`Object`** (`schemas/v1/object.json`, discriminated by `object_type`):

- **`Annotation`** – one timeline, one extent + one payload (+ confidence) as parallel arrays.
- **`ScoreObject`** – one media, one timeline, many **`EventBlock`s**. Each block is an extent/payload pair with its
  own strictly typed payload schema (e.g. `score_chord`, `score_dynamic`, `score_tempo`), yet all blocks live in the
  same file. `to_dataframe` / `to_csv` union the blocks into a sparse table with a `payload_type` column, which
  `load_bopp_csv` reads back into the typed blocks. See `distant_listening_chords_bopp.ipynb`.

When updating the schema, run 

```
hatch run codegen:build
```

## Silly benchmarks

The following is just an initial benchmark comparing `jams.load` and `bopp` deserialization from JSON or msgpack on a representative example file of chord annotations.
Don't take it too seriously, this is just to demonstrate the performance difference between the two libraries.

```
In [14]: %timeit load_bopp_file("drive.bopp")
103 μs ± 2.01 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [13]: %timeit load_from_msgpack("drive.bopp.msgpack")
97.4 μs ± 2.24 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [15]: %timeit jams.load("/home/bmcfee/drive.jams")
3.4 ms ± 106 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [16]: %timeit jams.load("/home/bmcfee/drive.jams", validate=False)
301 μs ± 3.4 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```
