from __future__ import annotations

from datetime import date

import polars as pl

from src.data.processor.basic import process_index, process_mask


def test_process_index_smoke() -> None:
    suspend_df = pl.DataFrame(
        {
            "code": [
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "000002.SZ",
            ],
            "trade_date": [
                date(2026, 1, 5),
                date(2026, 1, 6),
                date(2026, 1, 7),
                date(2026, 1, 5),
                date(2026, 1, 6),
            ],
            "is_suspend": [False, True, False, False, False],
        }
    )

    out = process_index(suspend_df)
    code1 = out.filter(pl.col("code") == "000001.SZ").sort("trade_date")

    assert out.height == 4
    assert code1["logic_index"].to_list() == [1, 2]


def test_process_mask_smoke() -> None:
    days = [date(2026, 1, d) for d in range(5, 11)]
    suspend_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * len(days),
            "trade_date": days,
            "is_suspend": [False] * len(days),
        }
    )
    namechange_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * len(days),
            "trade_date": days,
            "name": ["AAA", "AAA", "AAA", "AAA", "AAA", "STAAA"],
        }
    )
    index_df = process_index(suspend_df)

    out = process_mask(
        suspend_df=suspend_df,
        namechange_df=namechange_df,
        index_df=index_df,
    ).sort("trade_date")

    assert out["filter_mask"].to_list() == [True, False, False, False, False, False]
