"""Processor constants and configurations."""
from __future__ import annotations

# === Processing Window Constants ===

FEAT_WINDOW = 5
LABEL_WINDOW = 10
LABEL_WEIGHTS = [
    0.070,
    0.075,
    0.080,
    0.085,
    0.090,
    0.100,
    0.105,
    0.120,
    0.130,
    0.145,
]

# === Chunking Configuration ===

CHUNK_DAYS = 48  # Number of days to process in one chunk for high-volume data
CHUNK_TAIL_BARS = 1  # 5-min bars to carry over (covers Mezzo lookback: 64 * 6 = 384)
CHUNK_TAIL_DAYS = 1  # Daily adj factors to carry over
