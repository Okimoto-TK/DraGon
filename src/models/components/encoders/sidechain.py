"""Backward-compatible import shim for the adaptive STEP sidechain encoder."""
from __future__ import annotations

from src.models.components.encoders.adaptive_step import SidechainEncoder

__all__ = ["SidechainEncoder"]
