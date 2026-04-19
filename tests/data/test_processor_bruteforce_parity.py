from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
import math
from typing import Any

import polars as pl
from scipy.stats import norm

from src.data.processor.ohlcv import process_macro, process_mezzo, process_micro
from src.data.processor.sidechain import process_sidechain
from src.data.registry.dataset import WARMUP_BARS

EPS = 1e-6
_OHLCV_EPS = 1e-8
_LIMIT_TOL = 1e-4


def _fracdiff_weights_naive(d: float, window: int) -> list[float]:
    weights = [0.0] * window
    weights[0] = 1.0
    for k in range(1, window):
        weights[k] = -weights[k - 1] * (d - k + 1.0) / k
    return weights


def _fracdiff_naive(values: list[float | None], d: float, window: int) -> list[float]:
    weights = _fracdiff_weights_naive(d, window)
    out: list[float] = []
    for idx in range(len(values)):
        acc = 0.0
        for k, weight in enumerate(weights):
            src_idx = idx - k
            if src_idx < 0:
                break
            value = values[src_idx]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = 0.0
            acc += weight * float(value)
        out.append(acc)
    return out


def _normal_rank_naive(values: list[float | None]) -> list[float]:
    valid = [
        (idx, float(value))
        for idx, value in enumerate(values)
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    if not valid:
        return [math.nan] * len(values)

    sorted_valid = sorted(valid, key=lambda item: item[1])
    avg_ranks: dict[int, float] = {}
    pos = 1
    i = 0
    while i < len(sorted_valid):
        j = i + 1
        while j < len(sorted_valid) and sorted_valid[j][1] == sorted_valid[i][1]:
            j += 1
        avg_rank = (pos + (pos + (j - i) - 1)) / 2.0
        for k in range(i, j):
            avg_ranks[sorted_valid[k][0]] = avg_rank
        pos += j - i
        i = j

    n_valid = len(valid)
    out = [math.nan] * len(values)
    for idx in avg_ranks:
        out[idx] = float(norm.ppf((avg_ranks[idx] - 0.5) / n_valid))
    return out


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _trading_days() -> list[date]:
    return [
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
        date(2026, 1, 12),
        date(2026, 1, 13),
    ]


def _slot_time(slot: int) -> time:
    start = datetime(2026, 1, 5, 9, 35, 0)
    return (start + timedelta(minutes=5 * (slot - 1))).time()


def _build_fixture_data() -> dict[str, pl.DataFrame]:
    days = _trading_days()
    codes = ["000001.SZ", "000002.SZ"]

    index_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    adj_rows: list[dict[str, object]] = []
    limit_rows: list[dict[str, object]] = []
    moneyflow_rows: list[dict[str, object]] = []
    min5_rows: list[dict[str, object]] = []

    for code_idx, code in enumerate(codes):
        for day_idx, trade_day in enumerate(days, start=1):
            index_rows.append(
                {"code": code, "trade_date": trade_day, "logic_index": day_idx}
            )

            adj_factor = 1.0 + code_idx * 0.03 + day_idx * 0.01
            base = 10.0 + code_idx * 6.0 + day_idx * 0.8
            daily_open = base
            daily_high = base + 0.55
            daily_low = base - 0.45
            daily_close = base + (0.2 if day_idx % 2 == 0 else -0.1)
            daily_amount = 900.0 + code_idx * 150.0 + day_idx * 70.0
            up_limit = base + 0.6
            down_limit = base - 0.6
            if code_idx == 0 and day_idx == 5:
                daily_high = up_limit
                daily_close = up_limit
            if code_idx == 1 and day_idx == 6:
                daily_low = down_limit
                daily_close = down_limit

            daily_rows.append(
                {
                    "code": code,
                    "trade_date": trade_day,
                    "open": daily_open,
                    "high": daily_high,
                    "low": daily_low,
                    "close": daily_close,
                    "amount": daily_amount,
                }
            )
            adj_rows.append(
                {
                    "code": code,
                    "trade_date": trade_day,
                    "adj_factor": adj_factor,
                }
            )
            limit_rows.append(
                {
                    "code": code,
                    "trade_date": trade_day,
                    "up_limit": up_limit,
                    "down_limit": down_limit,
                }
            )

            buy_lg = 140.0 + 18.0 * day_idx + 25.0 * code_idx
            buy_elg = 70.0 + 6.0 * day_idx + 9.0 * code_idx
            sell_lg = 120.0 + 15.0 * day_idx + 17.0 * code_idx
            sell_elg = 55.0 + 5.0 * day_idx + 7.0 * code_idx
            moneyflow_rows.append(
                {
                    "code": code,
                    "trade_date": trade_day,
                    "buy_lg_amount": buy_lg,
                    "buy_elg_amount": buy_elg,
                    "sell_lg_amount": sell_lg,
                    "sell_elg_amount": sell_elg,
                }
            )

            for slot in range(1, 13):
                open_px = base + slot * 0.025
                high_px = open_px + 0.06 + (slot % 3) * 0.01
                low_px = open_px - 0.05 - (slot % 2) * 0.005
                close_px = open_px + 0.015 - (slot % 2) * 0.02
                if code_idx == 0 and day_idx == 4 and slot == 3:
                    high_px = up_limit
                    close_px = up_limit
                if code_idx == 1 and day_idx == 7 and slot == 10:
                    low_px = down_limit
                    close_px = down_limit
                min5_rows.append(
                    {
                        "code": code,
                        "trade_date": trade_day,
                        "time": _slot_time(slot),
                        "open": open_px,
                        "high": high_px,
                        "low": low_px,
                        "close": close_px,
                        "amount": 200.0 + code_idx * 30.0 + day_idx * 20.0 + slot * 7.0,
                    }
                )

    return {
        "index_df": pl.DataFrame(index_rows),
        "daily_df": pl.DataFrame(daily_rows),
        "adj_factor_df": pl.DataFrame(adj_rows),
        "limit_df": pl.DataFrame(limit_rows),
        "moneyflow_df": pl.DataFrame(moneyflow_rows),
        "min5_df": pl.DataFrame(min5_rows),
    }


def _lookup_by_key(
    rows: list[dict[str, Any]],
    key_cols: tuple[str, ...],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[col] for col in key_cols): row for row in rows}


def _adjust_ohlcv_rows(
    rows: list[dict[str, Any]],
    adj_lookup: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["code"], row["trade_date"])
        adj = float(adj_lookup[key]["adj_factor"])
        out.append(
            {
                **row,
                "open": float(row["open"]) * adj,
                "high": float(row["high"]) * adj,
                "low": float(row["low"]) * adj,
                "close": float(row["close"]) * adj,
            }
        )
    return out


def _adjust_limit_rows(
    rows: list[dict[str, Any]],
    adj_lookup: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["code"], row["trade_date"])
        adj = float(adj_lookup[key]["adj_factor"])
        out.append(
            {
                **row,
                "up_limit": float(row["up_limit"]) * adj,
                "down_limit": float(row["down_limit"]) * adj,
            }
        )
    return out


def _with_time_index(min5_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, date], list[dict[str, Any]]] = defaultdict(list)
    for row in min5_rows:
        grouped[(row["code"], row["trade_date"])].append(dict(row))

    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        sorted_rows = sorted(grouped[key], key=lambda row: row["time"])
        for idx, row in enumerate(sorted_rows, start=1):
            copied = dict(row)
            copied["time_index"] = idx
            copied.pop("time", None)
            out.append(copied)
    return out


def _aggregate_30min_naive(min5_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, date], list[dict[str, Any]]] = defaultdict(list)
    for row in min5_rows:
        grouped[(row["code"], row["trade_date"])].append(dict(row))

    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        sorted_rows = sorted(grouped[key], key=lambda row: row["time"])
        chunks = [sorted_rows[idx: idx + 6] for idx in range(0, len(sorted_rows), 6)]
        for time_index, chunk in enumerate(chunks, start=1):
            out.append(
                {
                    "code": key[0],
                    "trade_date": key[1],
                    "time_index": time_index,
                    "open": float(chunk[0]["open"]),
                    "high": max(float(row["high"]) for row in chunk),
                    "low": min(float(row["low"]) for row in chunk),
                    "close": float(chunk[-1]["close"]),
                    "amount": sum(float(row["amount"]) for row in chunk),
                }
            )
    return out


def _process_ohlcv_naive(
    index_df: pl.DataFrame,
    ohlcv_rows: list[dict[str, Any]],
    limit_rows: list[dict[str, Any]],
    freq: str,
    diff_d: float,
    prefix: str,
) -> pl.DataFrame:
    index_lookup = _lookup_by_key(index_df.to_dicts(), ("code", "trade_date"))
    limit_lookup = _lookup_by_key(limit_rows, ("code", "trade_date"))

    joined_rows: list[dict[str, Any]] = []
    for row in ohlcv_rows:
        key = (row["code"], row["trade_date"])
        if key not in index_lookup:
            continue
        joined_rows.append({**row, **limit_lookup[key], **index_lookup[key]})

    has_time_index = "time_index" in joined_rows[0]
    sort_key = (
        lambda row: (row["code"], row["trade_date"], row["time_index"])
        if has_time_index
        else (row["code"], row["trade_date"])
    )
    joined_rows.sort(key=sort_key)

    amount_hist: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    prev_close_by_code: dict[str, float] = {}
    raw_rows: list[dict[str, Any]] = []
    frac_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)

    for row in joined_rows:
        code = str(row["code"])
        trade_day = row["trade_date"]
        lag_key = (code, int(row["time_index"])) if has_time_index else (code,)
        amount_series = amount_hist[lag_key]

        prev_close = prev_close_by_code.get(code)
        if freq == "macro":
            f0_raw = (
                math.log(float(row["close"])) - math.log(prev_close)
                if prev_close is not None
                else math.nan
            )
        else:
            f0_raw = math.log(float(row["close"])) - math.log(float(row["open"]))

        amount_ma5 = (
            sum(amount_series[-5:]) / 5.0 if len(amount_series) >= 5 else math.nan
        )
        amount_prev = amount_series[-1] if amount_series else None
        up_limit = float(row["up_limit"])
        down_limit = float(row["down_limit"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        amount = float(row["amount"])
        up_limit_tol = abs(up_limit) * _LIMIT_TOL + 1e-6
        down_limit_tol = abs(down_limit) * _LIMIT_TOL + 1e-6

        if high == low:
            f2_raw = 0.5
        else:
            f2_raw = _clip((close - low) / (high - low), 0.0, 1.0)

        f3_raw = _clip(
            (close - down_limit) / (up_limit - down_limit + _OHLCV_EPS),
            0.0,
            1.0,
        )
        f4_raw = (
            _clip(amount / (amount_ma5 + _OHLCV_EPS), 0.0, 10.0)
            if not math.isnan(amount_ma5)
            else math.nan
        )
        amihud_raw = f0_raw / (amount + _OHLCV_EPS) if not math.isnan(f0_raw) else math.nan
        vol_ratio_log_raw = (
            math.log(amount + _OHLCV_EPS) - math.log(amount_prev + _OHLCV_EPS)
            if amount_prev is not None
            else math.nan
        )

        if freq == "micro":
            step_idx = int(row["time_index"]) % 48 or 48
        elif freq == "mezzo":
            step_idx = int(row["time_index"]) % 8 or 8
        else:
            step_idx = min(trade_day.isoweekday(), 5)

        hit_up = 1 if high >= up_limit - up_limit_tol else 0
        hit_down = 1 if low <= down_limit + down_limit_tol else 0
        close_up = 1 if abs(close - up_limit) <= up_limit_tol else 0
        close_down = 1 if abs(close - down_limit) <= down_limit_tol else 0

        raw_row = {
            "code": code,
            "trade_date": trade_day,
            "time_index": row.get("time_index"),
            "f0_raw": f0_raw,
            "f1_raw": math.log(high / (low + _OHLCV_EPS)),
            "f2_raw": f2_raw,
            "f3_raw": f3_raw,
            "f4_raw": f4_raw,
            "f5_raw": amihud_raw,
            "_vol_ratio_log_raw": vol_ratio_log_raw,
            "f6_raw": int(hit_up * 8 + hit_down * 4 + close_up * 2 + close_down),
            "f7_raw": step_idx,
        }
        frac_groups[lag_key].append(len(raw_rows))
        raw_rows.append(raw_row)

        amount_series.append(amount)
        prev_close_by_code[code] = close

    for group_indices in frac_groups.values():
        ret_values = [raw_rows[idx]["f0_raw"] for idx in group_indices]
        vol_values = [raw_rows[idx]["_vol_ratio_log_raw"] for idx in group_indices]
        ret_diff = _fracdiff_naive(ret_values, d=diff_d, window=WARMUP_BARS)
        vol_diff = _fracdiff_naive(vol_values, d=diff_d, window=WARMUP_BARS)
        for local_idx, row_idx in enumerate(group_indices):
            raw_rows[row_idx]["f8_raw"] = ret_diff[local_idx]
            raw_rows[row_idx]["f9_raw"] = vol_diff[local_idx]
            raw_rows[row_idx]["f10_raw"] = ret_diff[local_idx] - vol_diff[local_idx]

    out_rows: list[dict[str, Any]] = []
    feature_names = [f"{prefix}_f{i}" for i in range(11)]
    for row in raw_rows:
        out = {
            "code": row["code"],
            "trade_date": row["trade_date"],
        }
        if has_time_index:
            out["time_index"] = int(row["time_index"])
        for i, feature_name in enumerate(feature_names):
            out[feature_name] = row[f"f{i}_raw"]
        out_rows.append(out)

    sort_cols = ["code", "trade_date"] + (["time_index"] if has_time_index else [])
    return pl.DataFrame(out_rows).sort(sort_cols)


def _process_sidechain_naive(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    moneyflow_df: pl.DataFrame,
    diff_d: float,
) -> pl.DataFrame:
    index_lookup = _lookup_by_key(index_df.to_dicts(), ("code", "trade_date"))
    adj_lookup = _lookup_by_key(adj_factor_df.to_dicts(), ("code", "trade_date"))
    money_lookup = _lookup_by_key(moneyflow_df.to_dicts(), ("code", "trade_date"))

    rows: list[dict[str, Any]] = []
    for daily_row in daily_df.to_dicts():
        key = (daily_row["code"], daily_row["trade_date"])
        if key not in index_lookup or key not in money_lookup:
            continue
        adj = float(adj_lookup[key]["adj_factor"])
        rows.append(
            {
                "code": daily_row["code"],
                "trade_date": daily_row["trade_date"],
                "open": float(daily_row["open"]) * adj,
                "close": float(daily_row["close"]) * adj,
                "amount": float(daily_row["amount"]),
                **money_lookup[key],
            }
        )

    rows.sort(key=lambda row: (row["code"], row["trade_date"]))

    prev_close_by_code: dict[str, float] = {}
    by_trade_date: dict[date, list[int]] = defaultdict(list)
    by_code: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        code = str(row["code"])
        prev_close = prev_close_by_code.get(code)
        velocity_raw = (
            math.log(float(row["close"])) - math.log(prev_close)
            if prev_close is not None
            else math.nan
        )
        gap = (
            math.log(float(row["open"])) - math.log(prev_close)
            if prev_close is not None
            else math.nan
        )
        amihud_raw = (
            velocity_raw / (float(row["amount"]) + _OHLCV_EPS)
            if not math.isnan(velocity_raw)
            else math.nan
        )
        buy_main = float(row["buy_lg_amount"]) + float(row["buy_elg_amount"])
        sell_main = float(row["sell_lg_amount"]) + float(row["sell_elg_amount"])
        mf_net_ratio = (buy_main - sell_main) / (float(row["amount"]) + _OHLCV_EPS)
        mf_concentration = (buy_main + sell_main) / (float(row["amount"]) + _OHLCV_EPS)
        mf_main_amount_log = math.log(buy_main + sell_main + _OHLCV_EPS)

        row["gap"] = gap
        row["velocity_raw"] = velocity_raw
        row["amihud_raw"] = amihud_raw
        row["mf_net_ratio"] = mf_net_ratio
        row["mf_concentration"] = mf_concentration
        row["mf_main_amount_log"] = mf_main_amount_log

        by_trade_date[row["trade_date"]].append(idx)
        by_code[code].append(idx)
        prev_close_by_code[code] = float(row["close"])

    for indices in by_trade_date.values():
        gap_rank = _normal_rank_naive([rows[idx]["gap"] for idx in indices])
        velocity_rank = _normal_rank_naive([rows[idx]["velocity_raw"] for idx in indices])
        amount_rank = _normal_rank_naive([rows[idx]["amount"] for idx in indices])
        amihud_rank = _normal_rank_naive([rows[idx]["amihud_raw"] for idx in indices])
        mf_net_rank = _normal_rank_naive([rows[idx]["mf_net_ratio"] for idx in indices])
        mf_concentration_rank = _normal_rank_naive(
            [rows[idx]["mf_concentration"] for idx in indices]
        )
        mf_main_amount_log_rank = _normal_rank_naive(
            [rows[idx]["mf_main_amount_log"] for idx in indices]
        )
        for local_idx, row_idx in enumerate(indices):
            rows[row_idx]["gap_rank"] = gap_rank[local_idx]
            rows[row_idx]["velocity_rank"] = velocity_rank[local_idx]
            rows[row_idx]["amount_rank"] = amount_rank[local_idx]
            rows[row_idx]["amihud_rank"] = amihud_rank[local_idx]
            rows[row_idx]["mf_net_rank"] = mf_net_rank[local_idx]
            rows[row_idx]["mf_concentration_rank"] = mf_concentration_rank[local_idx]
            rows[row_idx]["mf_main_amount_log_rank"] = mf_main_amount_log_rank[local_idx]

    for indices in by_code.values():
        concentration_diff = _fracdiff_naive(
            [rows[idx]["mf_concentration"] for idx in indices],
            d=diff_d,
            window=WARMUP_BARS,
        )
        main_amount_diff = _fracdiff_naive(
            [rows[idx]["mf_main_amount_log"] for idx in indices],
            d=diff_d,
            window=WARMUP_BARS,
        )
        for local_idx, row_idx in enumerate(indices):
            rows[row_idx]["mf_concentration_diff"] = concentration_diff[local_idx]
            rows[row_idx]["mf_main_amount_log_diff"] = main_amount_diff[local_idx]

    result_rows = [
        {
            "code": row["code"],
            "trade_date": row["trade_date"],
            "gap": row["gap"],
            "gap_rank": row["gap_rank"],
            "mf_net_ratio": row["mf_net_ratio"],
            "mf_net_rank": row["mf_net_rank"],
            "mf_concentration": row["mf_concentration"],
            "mf_concentration_diff": row["mf_concentration_diff"],
            "mf_concentration_rank": row["mf_concentration_rank"],
            "mf_main_amount_log": row["mf_main_amount_log"],
            "mf_main_amount_log_diff": row["mf_main_amount_log_diff"],
            "mf_main_amount_log_rank": row["mf_main_amount_log_rank"],
            "amount_rank": row["amount_rank"],
            "velocity_rank": row["velocity_rank"],
            "amihud_rank": row["amihud_rank"],
        }
        for row in rows
    ]
    return pl.DataFrame(result_rows).sort(["code", "trade_date"])


def _assert_frames_match_with_eps(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    *,
    key_cols: list[str],
    numeric_cols: list[str],
    eps: float = EPS,
) -> None:
    actual_sorted = actual.sort(key_cols)
    expected_sorted = expected.sort(key_cols)

    assert actual_sorted.columns == expected_sorted.columns
    assert actual_sorted.height == expected_sorted.height

    actual_rows = actual_sorted.to_dicts()
    expected_rows = expected_sorted.to_dicts()

    for row_idx, (actual_row, expected_row) in enumerate(zip(actual_rows, expected_rows, strict=True)):
        for col in key_cols:
            assert actual_row[col] == expected_row[col], (row_idx, col, actual_row[col], expected_row[col])
        for col in numeric_cols:
            actual_value = actual_row[col]
            expected_value = expected_row[col]
            actual_missing = actual_value is None or (
                isinstance(actual_value, float) and math.isnan(actual_value)
            )
            expected_missing = expected_value is None or (
                isinstance(expected_value, float) and math.isnan(expected_value)
            )
            if actual_missing and expected_missing:
                continue
            assert not actual_missing and not expected_missing, (
                row_idx,
                col,
                actual_value,
                expected_value,
            )
            assert abs(float(actual_value) - float(expected_value)) <= eps, (
                row_idx,
                col,
                actual_value,
                expected_value,
            )


def test_process_macro_matches_naive_bruteforce_with_eps() -> None:
    data = _build_fixture_data()
    actual = process_macro(
        index_df=data["index_df"],
        daily_df=data["daily_df"],
        adj_factor_df=data["adj_factor_df"],
        limit_df=data["limit_df"],
        diff_d=0.5,
    )

    adj_lookup = _lookup_by_key(data["adj_factor_df"].to_dicts(), ("code", "trade_date"))
    daily_adj_rows = _adjust_ohlcv_rows(data["daily_df"].to_dicts(), adj_lookup)
    limit_adj_rows = _adjust_limit_rows(data["limit_df"].to_dicts(), adj_lookup)
    expected = _process_ohlcv_naive(
        index_df=data["index_df"],
        ohlcv_rows=daily_adj_rows,
        limit_rows=limit_adj_rows,
        freq="macro",
        diff_d=0.5,
        prefix="mcr",
    )

    _assert_frames_match_with_eps(
        actual,
        expected,
        key_cols=["code", "trade_date"],
        numeric_cols=[f"mcr_f{i}" for i in range(11)],
    )


def test_process_micro_matches_naive_bruteforce_with_eps() -> None:
    data = _build_fixture_data()
    actual = process_micro(
        index_df=data["index_df"],
        min5_df=data["min5_df"],
        adj_factor_df=data["adj_factor_df"],
        limit_df=data["limit_df"],
        diff_d=0.5,
    )

    adj_lookup = _lookup_by_key(data["adj_factor_df"].to_dicts(), ("code", "trade_date"))
    min5_indexed_rows = _with_time_index(data["min5_df"].to_dicts())
    min5_adj_rows = _adjust_ohlcv_rows(min5_indexed_rows, adj_lookup)
    limit_adj_rows = _adjust_limit_rows(data["limit_df"].to_dicts(), adj_lookup)
    expected = _process_ohlcv_naive(
        index_df=data["index_df"],
        ohlcv_rows=min5_adj_rows,
        limit_rows=limit_adj_rows,
        freq="micro",
        diff_d=0.5,
        prefix="mic",
    )

    _assert_frames_match_with_eps(
        actual,
        expected,
        key_cols=["code", "trade_date", "time_index"],
        numeric_cols=[f"mic_f{i}" for i in range(11)],
    )


def test_process_mezzo_matches_naive_bruteforce_with_eps() -> None:
    data = _build_fixture_data()
    actual = process_mezzo(
        index_df=data["index_df"],
        min5_df=data["min5_df"],
        adj_factor_df=data["adj_factor_df"],
        limit_df=data["limit_df"],
        diff_d=0.5,
    )

    adj_lookup = _lookup_by_key(data["adj_factor_df"].to_dicts(), ("code", "trade_date"))
    min30_rows = _aggregate_30min_naive(data["min5_df"].to_dicts())
    min30_adj_rows = _adjust_ohlcv_rows(min30_rows, adj_lookup)
    limit_adj_rows = _adjust_limit_rows(data["limit_df"].to_dicts(), adj_lookup)
    expected = _process_ohlcv_naive(
        index_df=data["index_df"],
        ohlcv_rows=min30_adj_rows,
        limit_rows=limit_adj_rows,
        freq="mezzo",
        diff_d=0.5,
        prefix="mzo",
    )

    _assert_frames_match_with_eps(
        actual,
        expected,
        key_cols=["code", "trade_date", "time_index"],
        numeric_cols=[f"mzo_f{i}" for i in range(11)],
    )


def test_process_sidechain_matches_naive_bruteforce_with_eps() -> None:
    data = _build_fixture_data()
    actual = process_sidechain(
        index_df=data["index_df"],
        daily_df=data["daily_df"],
        adj_factor_df=data["adj_factor_df"],
        moneyflow_df=data["moneyflow_df"],
        diff_d=0.5,
    )
    expected = _process_sidechain_naive(
        index_df=data["index_df"],
        daily_df=data["daily_df"],
        adj_factor_df=data["adj_factor_df"],
        moneyflow_df=data["moneyflow_df"],
        diff_d=0.5,
    )

    _assert_frames_match_with_eps(
        actual,
        expected,
        key_cols=["code", "trade_date"],
        numeric_cols=[
            "gap",
            "gap_rank",
            "mf_net_ratio",
            "mf_net_rank",
            "mf_concentration",
            "mf_concentration_diff",
            "mf_concentration_rank",
            "mf_main_amount_log",
            "mf_main_amount_log_diff",
            "mf_main_amount_log_rank",
            "amount_rank",
            "velocity_rank",
            "amihud_rank",
        ],
    )
