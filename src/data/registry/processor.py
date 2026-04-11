"""Processor constants and configurations."""
from __future__ import annotations

# === Processing Window Constants ===

FEAT_WINDOW = 5
LABEL_WINDOW = 10
LABEL_WEIGHTS = [1, 1, 1, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.1]

# === Chunking Configuration ===

CHUNK_DAYS = 48  # Number of days to process in one chunk for high-volume data
CHUNK_TAIL_BARS = 1  # 5-min bars to carry over (covers Mezzo lookback: 64 * 6 = 384)
CHUNK_TAIL_DAYS = 1  # Daily adj factors to carry over
