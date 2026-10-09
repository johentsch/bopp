"""Observation struct for the ``io.github.johentsch.score_note:v1`` extension schema."""

from __future__ import annotations

from typing import Annotated

import msgspec
from msgspec import UNSET, Meta, UnsetType

EXT_SCHEMA = "io.github.johentsch.score_note:v1"


class FractionPair(msgspec.Struct, array_like=True, frozen=True):
    """
    A ``[numerator, denominator]`` pair, the resolved form of a fraction column.

    This mirrors ``bopp.models.v1.core.FractionNonnegative`` (a
    ``tuple[int, int]`` with the same ``ge=0``/``ge=1`` bounds), but is a
    dedicated ``array_like`` struct rather than a bare tuple: ``msgspec``
    decodes a plain ``tuple[int, int]`` field into a Python ``tuple``, and
    ``msgspec.to_builtins`` then renders that tuple as-is, which would make a
    resolved row's wire form diverge from the raw dict row's two-element
    *list*. An ``array_like`` struct still encodes as a two-element array on
    the wire and under ``msgspec.to_builtins``, but round-trips as a ``list``,
    matching the unresolved dict row exactly.
    """

    numerator: Annotated[int, Meta(ge=0)]
    denominator: Annotated[int, Meta(ge=1)]


class ScoreNote(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """
    One row of a ``score_note`` payload, resolved as an extension observation.

    A missing optional value is represented only by an absent key: every
    optional field defaults to `msgspec.UNSET`, never to ``None``, and an
    explicit ``null`` on the wire is invalid for this schema and fails
    resolution (a `msgspec.ValidationError`) rather than decoding to
    `msgspec.UNSET`. The columnar ``score_note`` payload mirrors this by only
    ever emitting a key when the corresponding column is present *and* the
    value at that row is not ``None`` (see :mod:`bopp_score_ext.convert`), so
    a wire row never carries an explicit ``null`` in the first place. Combined
    with ``omit_defaults=True`` (which omits `msgspec.UNSET` fields on
    encoding regardless, but documents the intent), this keeps the wire form
    of a resolved row identical, byte for byte, to the raw dict form it was
    converted from, which is required for the annotation's UUIDv5 id to stay
    the same whether or not this plugin is installed: a row that resolves at
    all always re-serializes byte for byte. ``tuning`` must be written as a
    float on the wire: ``msgspec`` promotes an int to a float losslessly even
    under ``strict=True`` conversion, but an int and a float serialize to
    different bytes, which would otherwise break that id invariance. The
    fraction columns use `FractionPair` rather than a bare tuple for the same
    reason (see its docstring).
    """

    midi: Annotated[int, Meta(ge=0, le=127)]
    tpc: int
    name: str | UnsetType = UNSET
    octave: int | UnsetType = UNSET
    staff: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    voice: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    mc: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    mn: int | UnsetType = UNSET
    measure: int | UnsetType = UNSET
    tied: Annotated[int, Meta(ge=-1, le=1)] | UnsetType = UNSET
    gracenote: str | UnsetType = UNSET
    nominal_duration: FractionPair | UnsetType = UNSET
    scalar: FractionPair | UnsetType = UNSET
    chord_id: Annotated[int, Meta(ge=0)] | UnsetType = UNSET
    tremolo: str | UnsetType = UNSET
    volta: Annotated[int, Meta(ge=1)] | UnsetType = UNSET
    tuning: float | UnsetType = UNSET
    anchor: str | UnsetType = UNSET
