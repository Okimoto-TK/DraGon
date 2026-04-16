"""Processor constants and configurations."""
from __future__ import annotations

# === Processing Window Constants ===

FEAT_WINDOW = 5
LABEL_WINDOW = 4

# === Chunking Configuration ===

CHUNK_DAYS = 128  # Number of days to process in one chunk for high-volume data
# Carry 5 full trading days of 5-minute bars per code so intraday F4 can use the
# previous 5 sessions' same-slot moving average across chunk boundaries.
CHUNK_TAIL_BARS = 5 * 48
# Daily adj factors / limit prices must cover the same 5 carried-over sessions.
CHUNK_TAIL_DAYS = 5
