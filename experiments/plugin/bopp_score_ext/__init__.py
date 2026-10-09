"""Proof-of-concept bopp extension registering score note observations (bmcfee/bopp#8)."""

from .convert import from_ext, to_ext
from .models import EXT_SCHEMA, ScoreNote

__all__ = ["EXT_SCHEMA", "ScoreNote", "from_ext", "to_ext"]
