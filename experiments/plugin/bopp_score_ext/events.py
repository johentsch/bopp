"""Score control-event rows and an explicitly decoded mixed tagged union.

On this branch, an entry point or REGISTRY.register_entry(name, "module:Union")
fails on lookup with TypeError: Resolved extension for '...' must be a type, got
UnionType. Assigning REGISTRY[name] = ScoreControlEvent bypasses that check, and
resolution then works because msgspec converts tagged unions fine. The mixed
schema is not registered as an entry point; decode_mixed decodes it explicitly.
Optional values are absent keys, never explicit nulls;
field order and canonical floats/fraction lists preserve the wire bytes.
"""

from __future__ import annotations

from typing import Annotated

import msgspec
from msgspec import UNSET, Meta, UnsetType

from bopp.exceptions import BoppArgumentError
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.payload.ext import ExtensionPayload

from .models import FractionPair

EVENT_KINDS = (
    "chord",
    "dynamic",
    "spanner",
    "figured_bass",
    "staff_text",
    "system_text",
    "tempo",
)
EVENT_EXT_SCHEMAS = {
    kind: f"io.github.johentsch.score_control_event.{kind}:v1" for kind in EVENT_KINDS
}
EVENT_TAGS = {
    "chord": "Chord",
    "dynamic": "Dynamic",
    "spanner": "Spanner",
    "figured_bass": "FiguredBass",
    "staff_text": "StaffText",
    "system_text": "SystemText",
    "tempo": "Tempo",
}
MIXED_EXT_SCHEMA = "io.github.johentsch.score_control_event:v1"


class ScoreChord(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One chord observation, in generated payload field order."""

    staff: Annotated[int, Meta(ge=1)]
    voice: Annotated[int, Meta(ge=1)]
    mc: Annotated[int, Meta(ge=1)]
    mn: int
    chord_id: Annotated[int, Meta(ge=0)]
    volta: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    slur: str | UnsetType = UNSET
    crescendo_hairpin: str | UnsetType = UNSET
    decrescendo_hairpin: str | UnsetType = UNSET
    crescendo_line: str | UnsetType = UNSET
    diminuendo_line: str | UnsetType = UNSET
    pedal: str | UnsetType = UNSET
    nominal_duration: FractionPair | UnsetType = UNSET
    scalar: FractionPair | UnsetType = UNSET
    gracenote: str | UnsetType = UNSET
    articulation: str | UnsetType = UNSET
    tremolo: str | UnsetType = UNSET
    lyrics_1: str | UnsetType = UNSET
    lyrics_2: str | UnsetType = UNSET
    lyrics_3: str | UnsetType = UNSET


class ScoreDynamic(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One dynamic observation, in generated payload field order."""

    staff: Annotated[int, Meta(ge=1)]
    voice: Annotated[int, Meta(ge=1)]
    mc: Annotated[int, Meta(ge=1)]
    mn: int
    dynamics: str
    volta: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    slur: str | UnsetType = UNSET
    crescendo_hairpin: str | UnsetType = UNSET
    decrescendo_hairpin: str | UnsetType = UNSET
    crescendo_line: str | UnsetType = UNSET
    diminuendo_line: str | UnsetType = UNSET
    pedal: str | UnsetType = UNSET


class ScoreSpanner(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One spanner observation, in generated payload field order."""

    staff: Annotated[int, Meta(ge=1)]
    voice: Annotated[int, Meta(ge=1)]
    mc: Annotated[int, Meta(ge=1)]
    mn: int
    volta: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    slur: str | UnsetType = UNSET
    crescendo_hairpin: str | UnsetType = UNSET
    decrescendo_hairpin: str | UnsetType = UNSET
    crescendo_line: str | UnsetType = UNSET
    diminuendo_line: str | UnsetType = UNSET
    pedal: str | UnsetType = UNSET


class ScoreFiguredBass(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One figured_bass observation, in generated payload field order."""

    staff: Annotated[int, Meta(ge=1)]
    voice: Annotated[int, Meta(ge=1)]
    mc: Annotated[int, Meta(ge=1)]
    mn: int
    thoroughbass_duration: FractionPair
    volta: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    slur: str | UnsetType = UNSET
    crescendo_hairpin: str | UnsetType = UNSET
    decrescendo_hairpin: str | UnsetType = UNSET
    crescendo_line: str | UnsetType = UNSET
    diminuendo_line: str | UnsetType = UNSET
    pedal: str | UnsetType = UNSET
    thoroughbass_level_1: str | UnsetType = UNSET
    thoroughbass_level_2: str | UnsetType = UNSET
    thoroughbass_level_3: str | UnsetType = UNSET
    thoroughbass_level_4: str | UnsetType = UNSET


class ScoreStaffText(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One staff_text observation, in generated payload field order."""

    staff: Annotated[int, Meta(ge=1)]
    voice: Annotated[int, Meta(ge=1)]
    mc: Annotated[int, Meta(ge=1)]
    mn: int
    volta: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    slur: str | UnsetType = UNSET
    crescendo_hairpin: str | UnsetType = UNSET
    decrescendo_hairpin: str | UnsetType = UNSET
    crescendo_line: str | UnsetType = UNSET
    diminuendo_line: str | UnsetType = UNSET
    pedal: str | UnsetType = UNSET
    staff_text: str | UnsetType = UNSET


class ScoreSystemText(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One system_text observation, in generated payload field order."""

    staff: Annotated[int, Meta(ge=1)]
    voice: Annotated[int, Meta(ge=1)]
    mc: Annotated[int, Meta(ge=1)]
    mn: int
    system_text: str
    volta: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    slur: str | UnsetType = UNSET
    crescendo_hairpin: str | UnsetType = UNSET
    decrescendo_hairpin: str | UnsetType = UNSET
    crescendo_line: str | UnsetType = UNSET
    diminuendo_line: str | UnsetType = UNSET
    pedal: str | UnsetType = UNSET


class ScoreTempo(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """One tempo observation, in generated payload field order."""

    staff: Annotated[int, Meta(ge=1)]
    voice: Annotated[int, Meta(ge=1)]
    mc: Annotated[int, Meta(ge=1)]
    mn: int
    qpm: Annotated[float, Meta(gt=0)]
    tempo_visible: bool
    volta: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    slur: str | UnsetType = UNSET
    crescendo_hairpin: str | UnsetType = UNSET
    decrescendo_hairpin: str | UnsetType = UNSET
    crescendo_line: str | UnsetType = UNSET
    diminuendo_line: str | UnsetType = UNSET
    pedal: str | UnsetType = UNSET
    tempo: str | UnsetType = UNSET
    metronome_base: str | UnsetType = UNSET
    metronome_number: Annotated[float, Meta(gt=0)] | UnsetType = UNSET


EVENT_ROW_TYPES: dict[str, type[msgspec.Struct]] = {
    "chord": ScoreChord,
    "dynamic": ScoreDynamic,
    "spanner": ScoreSpanner,
    "figured_bass": ScoreFiguredBass,
    "staff_text": ScoreStaffText,
    "system_text": ScoreSystemText,
    "tempo": ScoreTempo,
}


class ChordEvent(ScoreChord, tag_field="event", tag="Chord"):
    """Tagged Chord observation for the mixed arm."""


class DynamicEvent(ScoreDynamic, tag_field="event", tag="Dynamic"):
    """Tagged Dynamic observation for the mixed arm."""


class SpannerEvent(ScoreSpanner, tag_field="event", tag="Spanner"):
    """Tagged Spanner observation for the mixed arm."""


class FiguredBassEvent(ScoreFiguredBass, tag_field="event", tag="FiguredBass"):
    """Tagged FiguredBass observation for the mixed arm."""


class StaffTextEvent(ScoreStaffText, tag_field="event", tag="StaffText"):
    """Tagged StaffText observation for the mixed arm."""


class SystemTextEvent(ScoreSystemText, tag_field="event", tag="SystemText"):
    """Tagged SystemText observation for the mixed arm."""


class TempoEvent(ScoreTempo, tag_field="event", tag="Tempo"):
    """Tagged Tempo observation for the mixed arm."""


ScoreControlEvent = (
    ChordEvent
    | DynamicEvent
    | SpannerEvent
    | FiguredBassEvent
    | StaffTextEvent
    | SystemTextEvent
    | TempoEvent
)


def decode_mixed(annotation: Annotation) -> list[ScoreControlEvent]:
    """Strictly decode mixed rows without registering or mutating the annotation."""
    payload = annotation.payload
    if (
        not isinstance(payload, ExtensionPayload)
        or payload.ext_schema != MIXED_EXT_SCHEMA
    ):
        raise BoppArgumentError(
            f"decode_mixed requires an 'ext' payload with ext_schema '{MIXED_EXT_SCHEMA}'"
        )
    return msgspec.convert(payload.value, list[ScoreControlEvent], strict=True)
