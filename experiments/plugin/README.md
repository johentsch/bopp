# bopp-score-ext

A proof-of-concept extension for [bmcfee/bopp#8](https://github.com/bmcfee/bopp/pull/8)
that represents score notes and control events as row observations. Install with
`pip install experiments/plugin` from a checkout with bopp available.

## Registered schemas

These eight entry points register individual row types:

```toml
[project.entry-points."bopp_extension"]
"io.github.johentsch.score_note:v1" = "bopp_score_ext.models:ScoreNote"
"io.github.johentsch.score_control_event.chord:v1" = "bopp_score_ext.events:ScoreChord"
"io.github.johentsch.score_control_event.dynamic:v1" = "bopp_score_ext.events:ScoreDynamic"
"io.github.johentsch.score_control_event.spanner:v1" = "bopp_score_ext.events:ScoreSpanner"
"io.github.johentsch.score_control_event.figured_bass:v1" = "bopp_score_ext.events:ScoreFiguredBass"
"io.github.johentsch.score_control_event.staff_text:v1" = "bopp_score_ext.events:ScoreStaffText"
"io.github.johentsch.score_control_event.system_text:v1" = "bopp_score_ext.events:ScoreSystemText"
"io.github.johentsch.score_control_event.tempo:v1" = "bopp_score_ext.events:ScoreTempo"
```

`to_ext(annotation, resolve_ext=True)` converts any of these columnar payloads
into an `ext` annotation. `from_ext(annotation)` restores its columnar form.
Both preserve the other facets and compute a new id for the new representation.
A round trip restores the original columnar id, except for entirely null columns.

```python
import bopp
from bopp.io import load_bopp_json, save_bopp_json
from bopp_score_ext import EVENT_EXT_SCHEMAS, ScoreDynamic, from_ext, to_ext

ann = bopp.create(
    media_id="track:1", payload_kind="ext",
    ext_schema=EVENT_EXT_SCHEMAS["dynamic"],
    value=[{"staff": 1, "voice": 1, "mc": 1, "mn": 0, "dynamics": "mf"}],
)
assert isinstance(ann.payload.value[0], ScoreDynamic)
assert to_ext(from_ext(ann)).id == ann.id
save_bopp_json(ann, "dynamics.json")
assert load_bopp_json("dynamics.json", resolve_ext=False).id == ann.id
```

## Mixed experiment

`io.github.johentsch.score_control_event:v1` holds all seven event kinds in one
annotation, in original piece order. Each row begins with `event`, using the DLC
tags `Chord`, `Dynamic`, `Spanner`, `FiguredBass`, `StaffText`, `SystemText`, or
`Tempo`. `ScoreControlEvent` is the union of the corresponding tagged subclasses
(`ChordEvent`, etc.).

**Finding for Brian:** on this branch, an entry point or
`REGISTRY.register_entry(name, "module:Union")` fails on lookup with
`TypeError: Resolved extension for '...' must be a type, got UnionType`.
Assigning the union directly with `REGISTRY[name] = ScoreControlEvent` bypasses
that check, and resolution then works because msgspec converts tagged unions
fine. The mixed schema is therefore not registered as an entry point;
`decode_mixed` decodes it explicitly:

```python
from experiments.dlc_chords import read_chords_tsv, chords_to_mixed_ext
from bopp_score_ext import decode_mixed

mixed = chords_to_mixed_ext(read_chords_tsv("chords.tsv"))
events = decode_mixed(mixed)
# The annotation retains dict rows; events is a separate typed list.
```

The reader works without this plugin and creates mixed annotations with
`resolve_ext=False`. Loading mixed JSON/MsgPack with default resolution warns
"No extension registered", retaining dicts and the same id. Use
`load_bopp_json(path, resolve_ext=False)` followed by `decode_mixed` to avoid it.

## Wire contract and limitations

- Optional fields use `UNSET`; missing keys encode absence. Explicit nulls and
  unknown fields fail strict row validation. Nullable column cells become absent
  row keys. Entirely null columns cannot survive conversion back to columns.
- Row fields follow generated payload order. Mixed rows put `event` first.
  Producer floats (`tuning`, `qpm`, `metronome_number`) must be floats on the
  wire: msgspec promotes integers even with strict conversion, changing bytes.
- Fraction columns reuse the array-like `FractionPair`, preserving two-element
  integer lists. Canonically produced dict rows and resolved structs serialize
  identically in JSON, preserving ids across plugin installation and JSON/MsgPack I/O.
  MsgPack map headers may differ between structs and dicts; decoded values agree.
- CSV round trips of `ext` payloads are not id-stable upstream yet; this
  experiment tests JSON and MsgPack.
