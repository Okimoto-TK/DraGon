# --- Lookback windows (in native bar count) ---
MACRO_LOOKBACK = 64        # daily bars (unchanged)
MEZZO_LOOKBACK = 96        # 30-min bars  (12 days × 8 bars/day)
MICRO_LOOKBACK = 144       # 5-min bars   (3 days  × 48 bars/day)

# --- Warmup: extra bars retained per scale before the usable window ---
WARMUP_BARS = 48
