"""Processor constants and configurations."""
from __future__ import annotations

# === Processing Window Constants ===

MACRO_LOOKBACK = 64
MEZZO_LOOKBACK = 64
MICRO_LOOKBACK = 48
LABEL_WINDOW = 10
LABEL_WEIGHTS = [1, 1, 1, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.1]

# === Chunking Configuration ===

CHUNK_DAYS = 128  # Number of days to process in one chunk for high-volume data
