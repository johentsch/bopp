# bopp-score-ext

A proof-of-concept `bopp` extension for [bmcfee/bopp#8](https://github.com/bmcfee/bopp/pull/8),
answering Brian's request for an example plugin that defines score element
types. It registers a `ScoreNote` observation struct that mirrors the columns
of the built-in `score_note` payload, so a `score_note` annotation can be
re-encoded as an `ext` annotation (and back) without losing any information
or changing its deterministic id.

## Install

From the repository root, with `bopp` itself available from this checkout:

```bash
pip install -e .                  # bopp
pip install experiments/plugin    # this plugin
```

The entry point it registers:

```toml
[project.entry-points."bopp_extension"]
"io.github.johentsch.score_note:v1" = "bopp_score_ext.models:ScoreNote"
```

## Usage

```python
import bopp
from bopp.io import load_bopp_json, save_bopp_json
import bopp_score_ext

ann = bopp.create(
    media_id="track:1",
    payload_kind="ext",
    ext_schema=bopp_score_ext.EXT_SCHEMA,
    value=[{"midi": 60, "tpc": 0}, {"midi": 64, "tpc": 4}],
    extent_kind="quarters_interval.fraction",
    quarter=[[0, 1], [1, 1]],
    duration=[[1, 1], [1, 1]],
)
save_bopp_json(ann, "notes.json")

# With the plugin installed, rows resolve to ScoreNote instances.
with_plugin = load_bopp_json("notes.json")
assert isinstance(with_plugin.payload.value[0], bopp_score_ext.ScoreNote)

# Without it (e.g. resolve_ext=False), rows stay as plain dicts; the id is
# unchanged either way.
without_plugin = load_bopp_json("notes.json", resolve_ext=False)
assert with_plugin.id == without_plugin.id
```

`bopp_score_ext.convert.to_ext`/`from_ext` convert between this encoding and
the columnar `score_note` payload directly.

## Invariants

- A missing value is represented only by an absent key. Every optional
  `ScoreNote` field defaults to `msgspec.UNSET`, not `None`; an explicit
  `null` on the wire is invalid for this schema and is rejected at
  resolution (`msgspec.ValidationError`), it never silently decodes to
  `msgspec.UNSET`. This keeps a resolved row's re-serialized bytes, and
  therefore the annotation's id, independent of whether this plugin is
  installed: a row that resolves at all always re-serializes byte for byte.
- `ScoreNote` uses `omit_defaults=True` so a resolved row and its dict form
  serialize identically (`msgspec.UNSET` fields are omitted on encoding
  regardless of this flag; it is kept to document the intent).
- `tuning` is always written as a float on the wire, since `msgspec` would
  otherwise promote an int to a float during strict conversion and change
  the serialized bytes.
- `ScoreNote` uses `forbid_unknown_fields=True`, so unexpected keys fail
  loudly instead of being silently dropped.

## Known limitations

- CSV round trips of `ext` payloads are not id-stable upstream yet (verified
  2026-10-09 on PR #8's head), so this plugin's tests cover JSON and MsgPack
  only.
- A column that is entirely `None` across all rows does not survive
  `to_ext`: it is indistinguishable from an absent column, so it cannot be
  reconstructed by `from_ext`.
