"""Processors for transforming raw data into processed features."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm
from tqdm import tqdm

import config.config as config

from src.data.registry.processor import LABEL_WINDOW, LABEL_WEIGHTS, MACRO_LOOKBACK
from src.data.schemas.processed import (
    PROCESSED_INDEX_SCHEMA,
    PROCESSED_LABEL_SCHEMA,
    PROCESSED_MACRO_SCHEMA,
    PROCESSED_MASK_SCHEMA,
    PROCESSED_MEZZO_SCHEMA,
    PROCESSED_MICRO_SCHEMA,
    PROCESSED_SIDECHAIN_SCHEMA,
)
from src.data.validators import validate_table


_EPS = 1e-8


def _normal_rank(s: pl.Series) -> pl.Series:
    """Apply normal rank transformation: Φ^{-1}((rank - 0.5) / N).

    NaN values are preserved as NaN and excluded from ranking.
    Only valid (non-null) values are ranked and transformed.
    """
    arr = s.to_numpy()
    valid_mask = ~np.isnan(arr)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return s

    result = np.full(len(arr), np.nan)
    valid_values = arr[valid_mask]

    # Rank only valid values
    ranked = pl.Series(valid_values).rank(method="average").to_numpy()
    result[valid_mask] = norm.ppf((ranked - 0.5) / n_valid)

    return pl.Series(s.name, result)


def process_index(suspend_df: pl.DataFrame, **_kwargs) -> pl.DataFrame:
    """Process suspend data into logical index table.

    Args:
        suspend_df: DataFrame with code, trade_date, is_suspend columns.
        **_kwargs: Unused additional arguments.

    Returns:
        DataFrame with code, trade_date, logic_index columns.
    """
    result = (
        suspend_df.filter(pl.col("is_suspend") == False)
        .sort(["code", "trade_date"])
        .with_columns(
            logic_index=pl.int_range(1, pl.len() + 1).over("code").cast(pl.Int32)
        )
        .select(["code", "trade_date", "logic_index"])
    )
    validate_table(result, PROCESSED_INDEX_SCHEMA)
    return result


def process_mask(
    suspend_df: pl.DataFrame,
    namechange_df: pl.DataFrame,
    index_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process suspend and namechange data into filter mask table.

    Args:
        suspend_df: DataFrame with code, trade_date, is_suspend columns.
        namechange_df: DataFrame with code, trade_date, name columns.
        index_df: DataFrame with code, trade_date, logic_index columns.

    Returns:
        DataFrame with code, trade_date, filter_mask columns.
    """
    # Preprocess is_st: True when name starts with "ST" or "*ST"
    nc = namechange_df.with_columns(
        is_st=pl.col("name").str.starts_with("ST")
        | pl.col("name").str.starts_with("*ST")
    )

    # Merge suspend and namechange on full outer join
    df = suspend_df.join(
        nc.select(["code", "trade_date", "is_st"]),
        on=["code", "trade_date"],
        how="full",
        coalesce=True,
    ).with_columns(
        pl.col("is_suspend").fill_null(False),
        pl.col("is_st").fill_null(False),
    )

    # Join with index to get logic_index
    df = df.join(
        index_df.select(["code", "trade_date", "logic_index"]),
        on=["code", "trade_date"],
        how="left",
    )

    # Sort by code and trade_date
    df = df.sort(["code", "trade_date"])

    # Compute filter_mask per code group with progress bar
    codes = df["code"].unique().to_list()
    results = []
    for code in tqdm(codes, desc="Processing mask", disable=config.debug):
        group = df.filter(pl.col("code") == code)
        n = len(group)
        suspend = group["is_suspend"].to_numpy()
        st = group["is_st"].to_numpy()
        logic_idx = group["logic_index"].to_numpy()

        mask = []
        for i in range(n):
            current_idx = logic_idx[i]

            # Calendar window [T, T+LABEL_WINDOW]: no suspend and no ST
            future_end = min(i + LABEL_WINDOW + 1, n)
            future_suspend = suspend[i:future_end].any()
            future_st = st[i:future_end].any()

            # Index window [T-MACRO_LOOKBACK+1, T]: no ST (by logic_index)
            lookback_start = current_idx - MACRO_LOOKBACK + 1
            lookback_mask = (logic_idx >= lookback_start) & (logic_idx <= current_idx)
            window_st = st[lookback_mask].any()

            # filter_mask is True only when all conditions are satisfied
            mask.append(not future_suspend and not future_st and not window_st)

        results.append(group.with_columns(filter_mask=pl.Series(mask, dtype=pl.Boolean)))

    result = pl.concat(results)

    # Filter to only keep rows that exist in index_df (exclude suspend days)
    result = result.filter(pl.col("logic_index").is_not_null())

    result = result.sort(["code", "trade_date"]).select(
        ["code", "trade_date", "filter_mask"]
    )
    validate_table(result, PROCESSED_MASK_SCHEMA)
    return result


# === Processed Data Processors ===

def process_macro(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    lookback: int = MACRO_LOOKBACK,
    **_kwargs,
) -> pl.DataFrame:
    """Process daily OHLCV data into macro backbone features.

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        daily_df: DataFrame with code, trade_date, open, high, low, close, amount columns.
        adj_factor_df: DataFrame with code, trade_date, adj_factor columns.
        limit_df: DataFrame with code, trade_date, up_limit, down_limit columns.
        lookback: Lookback window for feature calculation.

    Returns:
        DataFrame with code, trade_date, mcr_f0..mcr_f6 columns.
    """
    # Adjust prices by adj_factor with progress
    tqdm.pandas(disable=config.debug, desc="Adjusting prices")
    daily_adj = daily_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")

    # Process OHLCV features with progress
    tqdm.write(f"Processing {lookback}-step features...")
    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_lf=pl.LazyFrame(daily_adj),
        limit_df=limit_df,
        lookback=lookback,
        freq="macro",
    )

    # Rename f* to mcr_f*
    for i in range(9):
        result = result.rename({f"f{i}": f"mcr_f{i}"})

    validate_table(result, PROCESSED_MACRO_SCHEMA)
    return result


def process_micro(
    index_df: pl.DataFrame,
    min5_lf: pl.LazyFrame,
    adj_factor_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    lookback: int = MACRO_LOOKBACK,
    **_kwargs,
) -> pl.LazyFrame:
    """Process 5-minute OHLCV data into micro backbone features using Lazy evaluation.

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        min5_lf: LazyFrame with code, trade_date, time, open, high, low, close, amount columns.
        adj_factor_df: DataFrame with code, trade_date, adj_factor columns.
        limit_df: DataFrame with code, trade_date, up_limit, down_limit columns.
        lookback: Lookback window for feature calculation.

    Returns:
        LazyFrame with code, trade_date, time_index, mic_f0..mic_f6 columns.
    """
    # Generate time_index lazily
    min5_indexed = min5_lf.sort(["code", "trade_date", "time"]).with_columns(
        time_index=pl.int_range(1, pl.len() + 1).over(["code", "trade_date"]).cast(pl.Int32)
    ).drop("time")

    # Adjust prices by adj_factor lazily
    min5_adj = min5_indexed.join(
        pl.LazyFrame(adj_factor_df).select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")

    # Process OHLCV features for micro (time_index present) lazily
    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_lf=min5_adj,
        limit_df=limit_df,
        lookback=lookback,
        freq="micro",
    )

    # Rename f* to mic_f*
    for i in range(9):
        result = result.rename({f"f{i}": f"mic_f{i}"})

    validate_table(result, PROCESSED_MICRO_SCHEMA)
    return result


def process_mezzo(
    index_df: pl.DataFrame,
    min5_lf: pl.LazyFrame,
    adj_factor_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    lookback: int = MACRO_LOOKBACK,
    **_kwargs,
) -> pl.LazyFrame:
    """Process 5-minute OHLCV data into mezzo (30-min) backbone features using Lazy evaluation.

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        min5_lf: LazyFrame with code, trade_date, time, open, high, low, close, amount columns.
        adj_factor_df: DataFrame with code, trade_date, adj_factor columns.
        limit_df: DataFrame with code, trade_date, up_limit, down_limit columns.
        lookback: Lookback window for feature calculation.

    Returns:
        LazyFrame with code, trade_date, time_index, mzo_f0..mzo_f6 columns.
    """
    # Aggregate 5-min bars to 30-min bars lazily
    min30 = _aggregate_30min_lazy(min5_lf)

    # Adjust prices by adj_factor lazily
    min30_adj = min30.join(
        pl.LazyFrame(adj_factor_df).select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")

    # Process OHLCV features for mezzo (time_index present) lazily
    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_lf=min30_adj,
        limit_df=limit_df,
        lookback=lookback,
        freq="mezzo",
    )

    # Rename f* to mzo_f*
    for i in range(9):
        result = result.rename({f"f{i}": f"mzo_f{i}"})

    validate_table(result, PROCESSED_MEZZO_SCHEMA)
    return result


def process_sidechain(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    moneyflow_df: pl.DataFrame,
    lookback: int = MACRO_LOOKBACK,
    **_kwargs,
) -> pl.DataFrame:
    """Process moneyflow and daily data into sidechain features.

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        daily_df: DataFrame with code, trade_date, open, high, low, close, amount columns.
        adj_factor_df: DataFrame with code, trade_date, adj_factor columns.
        moneyflow_df: DataFrame with code, trade_date, buy/sell large and extra-large amount columns.
        lookback: Lookback window for feature calculation.

    Returns:
        DataFrame with code, trade_date, gap, gap_rank, mf_net_ratio, mf_concentration,
        amt_surge_rank, velocity_rank columns.
    """
    # Adjust prices by adj_factor
    daily_adj = daily_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")

    # Merge daily with moneyflow
    df = daily_adj.join(
        moneyflow_df,
        on=["code", "trade_date"],
        how="inner",
    )

    # Join with index to filter suspend days
    df = df.join(
        index_df.select(["code", "trade_date"]),
        on=["code", "trade_date"],
        how="inner",
    )

    df = df.sort(["code", "trade_date"])

    # === 1. Gap: ln(Open_t / Close_{t-1}) ===
    df = df.with_columns(
        (pl.col("open").log() - pl.col("close").shift(1).over("code").log()).alias("gap")
    )

    # === 2. Gap Rank: NormalRank(gap) ===
    group_key = ["trade_date"]
    df = df.with_columns(
        pl.col("gap").map_batches(_normal_rank).over(group_key).alias("gap_rank")
    )

    # === 3 & 4. Money flow features ===
    buy_main = pl.col("buy_lg_amount") + pl.col("buy_elg_amount")
    sell_main = pl.col("sell_lg_amount") + pl.col("sell_elg_amount")

    df = df.with_columns([
        # mf_net_ratio: (buy_main - sell_main) / Amount
        ((buy_main - sell_main) / (pl.col("amount") + _EPS)).alias("mf_net_ratio"),
        # mf_concentration: (buy_main + sell_main) / Amount
        ((buy_main + sell_main) / (pl.col("amount") + _EPS)).alias("mf_concentration"),
    ])

    # === 5. Volume Surge Rank: NormalRank(Amount / MA(Amount, 5)) ===
    amt_ma5 = pl.col("amount").rolling_mean(5).over("code")
    df = df.with_columns(
        (pl.col("amount") / (amt_ma5 + _EPS)).map_batches(_normal_rank).over(group_key).alias("amt_surge_rank")
    )

    # === 6. Velocity Rank: NormalRank(ln(Close_t / Close_{t-1})) ===
    velocity = pl.col("close").log() - pl.col("close").shift(1).over("code").log()
    df = df.with_columns(
        velocity.map_batches(_normal_rank).over(group_key).alias("velocity_rank")
    )

    # Select output columns
    result = df.select([
        "code", "trade_date", "gap", "gap_rank",
        "mf_net_ratio", "mf_concentration",
        "amt_surge_rank", "velocity_rank",
    ]).sort(["code", "trade_date"])

    validate_table(result, PROCESSED_SIDECHAIN_SCHEMA)
    return result


def process_label(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process daily adjusted prices into prediction labels.

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        daily_df: DataFrame with code, trade_date, open, close columns.
        adj_factor_df: DataFrame with code, trade_date, adj_factor columns.

    Returns:
        DataFrame with code, trade_date, label_final columns.
    """
    # Adjust prices by adj_factor with progress
    tqdm.write("Adjusting prices...")
    daily_adj = daily_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        adj_open=pl.col("open") * pl.col("adj_factor"),
        adj_close=pl.col("close") * pl.col("adj_factor"),
    ).sort(["code", "trade_date"])

    # Join with index to filter suspend days
    df = daily_adj.join(
        index_df.select(["code", "trade_date"]),
        on=["code", "trade_date"],
        how="inner",
    )

    # Compute future log returns and weights with progress
    tqdm.write("Computing label features...")
    H = LABEL_WINDOW
    w = LABEL_WEIGHTS

    # Compute weighted sum S_t and max peak M_t
    # S_t = sum_{k=1}^{H} w_k * ln(C_{t+k} / O_{t+1})
    # M_t = max_{k=1}^{H} ln(C_{t+k} / O_{t+1})
    next_open_log = pl.col("adj_open").log().shift(-1).over("code")

    weighted_sum = pl.lit(0.0)
    max_ret = pl.lit(-float("inf"))

    for k in range(1, H + 1):
        future_log_ret = pl.col("adj_close").log().shift(-k).over("code") - next_open_log
        weighted_sum = weighted_sum + w[k - 1] * future_log_ret
        max_ret = pl.max_horizontal(max_ret, future_log_ret)

    df = df.with_columns([
        weighted_sum.alias("S"),
        max_ret.alias("M"),
    ])

    # Cross-sectional NormalRank within each trade_date
    group_key = ["trade_date"]
    df = df.with_columns([
        pl.col("S").map_batches(_normal_rank).over(group_key).alias("R_S"),
        pl.col("M").map_batches(_normal_rank).over(group_key).alias("R_M"),
    ])

    # Weight fusion: F_t = NormalRank(0.5 * R_S + 0.5 * R_M) within each trade_date
    df = df.with_columns(
        (0.5 * pl.col("R_S") + 0.5 * pl.col("R_M")).map_batches(_normal_rank).over(group_key).alias("F")
    )

    # Label_final = F (直接输出，不再做 ^3 增强)
    df = df.with_columns(
        pl.col("F").alias("label_final")
    )

    # Select output columns
    result = df.select(["code", "trade_date", "label_final"]).sort(["code", "trade_date"])

    validate_table(result, PROCESSED_LABEL_SCHEMA)
    return result


# === Internal OHLCV Feature Processor ===


def _process_ohlcv(
    index_df: pl.DataFrame,
    ohlcv_lf: pl.LazyFrame,
    limit_df: pl.DataFrame,
    lookback: int = MACRO_LOOKBACK,
    freq: str = "macro",
) -> pl.LazyFrame:
    """Process OHLCV data into 9 backbone features (f0-f8) using Lazy evaluation.

    Features:
        f0: Velocity (瞬时速度) - Raw
        f1: Volume Delta (动能加速度) - Raw
        f2: Amplitude (极限振幅) - Raw
        f3: Bar Form (线内多空胜率) - Raw [0,1]
        f4: Gravity (边界引力场) - Raw [0,1]
        f5: Phase X (时钟相位 X) - Raw [-1,1]
        f6: Phase Y (时钟相位 Y) - Raw [-1,1]
        f7: Velocity NormalRank - NormalRank
        f8: Volume Delta NormalRank - NormalRank

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        ohlcv_lf: LazyFrame with code, trade_date, open, high, low, close, amount columns.
        limit_df: DataFrame with code, trade_date, up_limit, down_limit columns.
        lookback: Lookback window for NormalRank (unused for raw features).
        freq: Frequency type - "macro", "mezzo", or "micro".

    Returns:
        LazyFrame with code, trade_date, [time_index], f0-f8 columns.
    """
    # Join with index (inner to exclude suspend days)
    df = ohlcv_lf.join(
        pl.LazyFrame(index_df).select(["code", "trade_date", "logic_index"]),
        on=["code", "trade_date"],
        how="inner",
    )

    # Join with limit data for Gravity (F4)
    df = df.join(
        pl.LazyFrame(limit_df).select(["code", "trade_date", "up_limit", "down_limit"]),
        on=["code", "trade_date"],
        how="left",
    )

    # Determine structure
    schema_names = df.collect_schema().names()
    has_time_index = "time_index" in schema_names
    sort_cols = ["code", "trade_date"] + (["time_index"] if has_time_index else [])
    group_key = ["trade_date"] + (["time_index"] if has_time_index else [])

    df = df.sort(sort_cols)

    # === F0: Velocity (瞬时速度) & F1: Volume Delta (动能加速度) ===
    if freq == "macro":
        # Macro 级别：保留隔夜跳空，使用 C/C
        f0_expr = pl.col("close").log() - pl.col("close").shift(1).over("code").log()
        # Macro 级别：成交量对比昨日
        f1_expr = (pl.col("amount") + 1).log() - (pl.col("amount").shift(1).over("code") + 1).log()
    else:
        # Micro/Mezzo 级别：彻底剥离隔夜跳空，使用纯粹的线内动能 C/O
        f0_expr = pl.col("close").log() - pl.col("open").log()
        # Micro/Mezzo 级别：成交量只在日内进行差分对比，每天第一根K线(无同日历史)不与昨日尾盘对比
        f1_expr = (pl.col("amount") + 1).log() - (pl.col("amount").shift(1).over(["code", "trade_date"]) + 1).log()

    df = df.with_columns([
        f0_expr.alias("f0_raw"),
        f1_expr.alias("f1_raw")
    ])

    # === F2: Amplitude (极限振幅) ===
    # at = ln(High_t / (Low_t + ε)), 永远 >= 0
    df = df.with_columns(
        (pl.col("high") / (pl.col("low") + _EPS)).log().alias("f2_raw")
    )

    # === F3: Bar Form (线内多空胜率) ===
    # pt = (Close_t - Low_t) / (High_t - Low_t), [0, 1]
    # 一字板 (High = Low) 时多空交火空间为 0，设为中性 0.5
    df = df.with_columns(
        pl.when(pl.col("high") == pl.col("low"))
        .then(pl.lit(0.5))
        .otherwise((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
        .clip(0.0, 1.0)
        .alias("f3_raw")
    )

    # === F4: Gravity (边界引力场) ===
    # gt = (Close_t - LimitDown_t) / (LimitUp_t - LimitDown_t + ε), [0, 1]
    # 使用真实的涨跌停价格
    df = df.with_columns(
        ((pl.col("close") - pl.col("down_limit")) /
         (pl.col("up_limit") - pl.col("down_limit") + _EPS)).clip(0.0, 1.0).alias("f4_raw")
    )

    # === F5 & F6: Phase X & Y (时钟相位) ===
    # 根据频率不同使用不同的周期
    if freq == "micro":
        # Micro: 每天 48 根 K 线，周期为 48
        period = 48
        df = df.with_columns(
            step_idx=pl.when(pl.col("time_index") % period == 0)
            .then(pl.lit(period))
            .otherwise(pl.col("time_index") % period)
        )
    elif freq == "mezzo":
        # Mezzo: 每天 8 根 K 线，周期为 8
        period = 8
        df = df.with_columns(
            step_idx=pl.when(pl.col("time_index") % period == 0)
            .then(pl.lit(period))
            .otherwise(pl.col("time_index") % period)
        )
    else:  # macro
        # Macro: 使用星期几，周一=1, 周五=5
        period = 5  # 5个交易日
        df = df.with_columns(
            weekday=pl.col("trade_date").dt.weekday()  # 1=Monday, 5=Friday
        )
        df = df.with_columns(
            step_idx=pl.when(pl.col("weekday") <= 5)
            .then(pl.col("weekday"))
            .otherwise(pl.lit(5))  # 周末统一为周五
        )

    # Phase X = sin(2π * step_idx / period), Phase Y = cos(2π * step_idx / period)
    df = df.with_columns([
        (pl.col("step_idx") * 2 * np.pi / period).sin().alias("f5_raw"),
        (pl.col("step_idx") * 2 * np.pi / period).cos().alias("f6_raw"),
    ])

    # === Apply NormalRank to f0, f1 -> f7, f8 ===
    # F7 = NormalRank(F0_raw), F8 = NormalRank(F1_raw)
    df = df.with_columns([
        pl.col("f0_raw").map_batches(_normal_rank).over(group_key).alias("f7_raw"),
        pl.col("f1_raw").map_batches(_normal_rank).over(group_key).alias("f8_raw"),
    ])

    # Drop helper columns
    cols_to_drop = ["logic_index", "up_limit", "down_limit", "step_idx", "weekday"]
    actual_drop = [c for c in cols_to_drop if c in df.collect_schema().names()]
    df = df.drop(actual_drop)

    # Rename f*_raw to f*
    for i in range(9):
        df = df.rename({f"f{i}_raw": f"f{i}"})

    # Select output columns
    select_cols = (
        ["code", "trade_date"]
        + (["time_index"] if has_time_index else [])
        + [f"f{i}" for i in range(9)]
    )
    result = df.select(select_cols)

    # Drop time column if present
    if "time" in result.collect_schema().names():
        result = result.drop("time")

    return result


# === Internal Aggregation Functions ===

def _aggregate_30min(min5_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 5-minute OHLCV bars into 30-minute bars (eager version).

    Groups every 6 consecutive 5-min bars within each (code, trade_date).
    - open: first bar's open
    - high: max high across 6 bars
    - low: min low across 6 bars
    - close: last bar's close
    - amount: sum across 6 bars

    Args:
        min5_df: DataFrame with code, trade_date, time, open, high, low, close, amount columns.

    Returns:
        DataFrame with code, trade_date, time_index, open, high, low, close, amount columns.
    """
    # Sort and assign bar index within each (code, trade_date)
    sorted_df = min5_df.sort(["code", "trade_date", "time"])

    # Assign bar index and group index (every 6 bars = 1 group)
    with_bar_idx = sorted_df.with_columns(
        bar_idx=pl.int_range(0, pl.len()).over(["code", "trade_date"]),
    )
    with_bar_idx = with_bar_idx.with_columns(
        group_idx=(pl.col("bar_idx") // 6).cast(pl.Int32),
    )

    # Aggregate OHLCV within each 30-min group
    aggregated = with_bar_idx.group_by(["code", "trade_date", "group_idx"]).agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("amount").sum().alias("amount"),
    )

    # Assign time_index within each (code, trade_date)
    result = aggregated.sort(["code", "trade_date", "group_idx"]).with_columns(
        time_index=pl.int_range(1, pl.len() + 1).over(["code", "trade_date"]).cast(pl.Int32)
    ).drop("group_idx")

    return result


def _aggregate_30min_lazy(min5_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate 5-minute OHLCV bars into 30-minute bars (lazy version).

    Groups every 6 consecutive 5-min bars within each (code, trade_date).
    - open: first bar's open
    - high: max high across 6 bars
    - low: min low across 6 bars
    - close: last bar's close
    - amount: sum across 6 bars

    Args:
        min5_lf: LazyFrame with code, trade_date, time, open, high, low, close, amount columns.

    Returns:
        LazyFrame with code, trade_date, time_index, open, high, low, close, amount columns.
    """
    # Sort and assign bar index within each (code, trade_date)
    sorted_lf = min5_lf.sort(["code", "trade_date", "time"])

    # Assign bar index and group index (every 6 bars = 1 group)
    with_bar_idx = sorted_lf.with_columns(
        bar_idx=pl.int_range(0, pl.len()).over(["code", "trade_date"]),
    )
    with_bar_idx = with_bar_idx.with_columns(
        group_idx=(pl.col("bar_idx") // 6).cast(pl.Int32),
    )

    # Aggregate OHLCV within each 30-min group
    aggregated = with_bar_idx.group_by(["code", "trade_date", "group_idx"]).agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("amount").sum().alias("amount"),
    )

    # Assign time_index within each (code, trade_date)
    result = aggregated.sort(["code", "trade_date", "group_idx"]).with_columns(
        time_index=pl.int_range(1, pl.len() + 1).over(["code", "trade_date"]).cast(pl.Int32)
    ).drop("group_idx")

    return result
