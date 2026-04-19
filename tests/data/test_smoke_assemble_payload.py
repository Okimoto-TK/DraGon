from __future__ import annotations

import numpy as np
import polars as pl
import warnings

from src.data.assembler.assemble import (
    LABEL_COLS,
    MACRO_FLOAT_FEATURES,
    MACRO_INT8_FEATURES,
    MACRO_TOTAL_BARS,
    MEZZO_BARS_PER_DAY,
    MEZZO_FLOAT_FEATURES,
    MEZZO_INT8_FEATURES,
    MICRO_BARS_PER_DAY,
    MICRO_FLOAT_FEATURES,
    MICRO_INT8_FEATURES,
    SIDECHAIN_FEATURES,
    _build_packed_payload,
    _list_non_null_expr,
    _safe_list_int8_expr,
)


def test_build_packed_payload_smoke_sidechain_width() -> None:
    n_rows = 120
    float_width = (
        2
        + len(LABEL_COLS)
        + len(MACRO_FLOAT_FEATURES)
        + len(SIDECHAIN_FEATURES)
        + len(MEZZO_FLOAT_FEATURES) * MEZZO_BARS_PER_DAY
        + len(MICRO_FLOAT_FEATURES) * MICRO_BARS_PER_DAY
    )
    int8_width = (
        len(MACRO_INT8_FEATURES)
        + len(MEZZO_INT8_FEATURES) * MEZZO_BARS_PER_DAY
        + len(MICRO_INT8_FEATURES) * MICRO_BARS_PER_DAY
    )

    float_data = np.zeros((n_rows, float_width), dtype=np.float32)
    float_data[:, 0] = np.arange(n_rows, dtype=np.float32) + 20260101.0
    float_data[:, 1] = 1.0
    int8_data = np.zeros((n_rows, int8_width), dtype=np.int8)

    payload = _build_packed_payload(float_data=float_data, int8_data=int8_data)

    assert payload["date"].shape[0] > 0
    assert payload["sidechain"].shape[1] == len(SIDECHAIN_FEATURES)
    assert payload["sidechain"].shape[2] == MACRO_TOTAL_BARS


def test_int8_nulls_mark_invalid_and_are_zero_filled_for_safe_cast() -> None:
    mezzo_pos = list(range(1, MEZZO_BARS_PER_DAY + 1))
    micro_pos = list(range(1, MICRO_BARS_PER_DAY + 1))

    df = pl.DataFrame(
        {
            "mcr_f6": [None, 1],
            "mcr_f7": [None, 2],
            "mzo_f6": [[None] * MEZZO_BARS_PER_DAY, [0] * MEZZO_BARS_PER_DAY],
            "mzo_f7": [mezzo_pos, mezzo_pos],
            "mic_f6": [[None] * MICRO_BARS_PER_DAY, [0] * MICRO_BARS_PER_DAY],
            "mic_f7": [micro_pos, micro_pos],
        },
        strict=False,
    )

    validity = df.select(
        [
            pl.all_horizontal([pl.col(f).is_not_null() for f in MACRO_INT8_FEATURES]).alias("macro_ok"),
            _list_non_null_expr(MEZZO_INT8_FEATURES, MEZZO_BARS_PER_DAY).alias("mezzo_ok"),
            _list_non_null_expr(MICRO_INT8_FEATURES, MICRO_BARS_PER_DAY).alias("micro_ok"),
        ]
    )
    assert validity.row(0, named=True) == {
        "macro_ok": False,
        "mezzo_ok": False,
        "micro_ok": False,
    }
    assert validity.row(1, named=True) == {
        "macro_ok": True,
        "mezzo_ok": True,
        "micro_ok": True,
    }

    int8_df = df.select(
        [
            pl.col("mcr_f6").fill_null(0).cast(pl.Int8),
            pl.col("mcr_f7").fill_null(0).cast(pl.Int8),
            _safe_list_int8_expr("mzo_f6", MEZZO_BARS_PER_DAY)
            .list.to_struct(fields=[f"mzo_f6_{j}" for j in range(MEZZO_BARS_PER_DAY)])
            .struct.field("*"),
            _safe_list_int8_expr("mzo_f7", MEZZO_BARS_PER_DAY)
            .list.to_struct(fields=[f"mzo_f7_{j}" for j in range(MEZZO_BARS_PER_DAY)])
            .struct.field("*"),
            _safe_list_int8_expr("mic_f6", MICRO_BARS_PER_DAY)
            .list.to_struct(fields=[f"mic_f6_{j}" for j in range(MICRO_BARS_PER_DAY)])
            .struct.field("*"),
            _safe_list_int8_expr("mic_f7", MICRO_BARS_PER_DAY)
            .list.to_struct(fields=[f"mic_f7_{j}" for j in range(MICRO_BARS_PER_DAY)])
            .struct.field("*"),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        int8_cols = []
        for c in int8_df.columns:
            int8_cols.append(int8_df[c].to_numpy().astype(np.int8))
    int8_numpy = np.column_stack(int8_cols)

    assert int8_numpy.shape[0] == 2
    assert int8_numpy[0, 0] == 0
    assert int8_numpy[0, 1] == 0
    assert np.all(int8_numpy[0, 2 : 2 + MEZZO_BARS_PER_DAY] == 0)
    assert np.all(int8_numpy[0, -MICRO_BARS_PER_DAY * 2 : -MICRO_BARS_PER_DAY] == 0)
