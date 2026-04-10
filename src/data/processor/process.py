"""Processors for transforming raw data into processed features."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
from src.utils.log import vlog

_SRC = "OHLCV"


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
    lookback: int = MACRO_LOOKBACK,
    **_kwargs,
) -> pl.DataFrame:
    """Process daily OHLCV data into macro backbone features.

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        daily_df: DataFrame with code, trade_date, open, high, low, close, amount columns.
        adj_factor_df: DataFrame with code, trade_date, adj_factor columns.
        lookback: Lookback window for feature calculation.

    Returns:
        DataFrame with code, trade_date, mcr_f1..mcr_f10 columns.
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
        ohlcv_df=daily_adj,
        lookback=lookback,
    )

    # Rename f* to mcr_f*
    for i in range(1, 11):
        result = result.rename({f"f{i}": f"mcr_f{i}"})

    validate_table(result, PROCESSED_MACRO_SCHEMA)
    return result


def process_micro(
    index_df: pl.DataFrame,
    min5_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    lookback: int = MACRO_LOOKBACK,
    **_kwargs,
) -> pl.DataFrame:
    """Process 5-minute OHLCV data into micro backbone features.

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        min5_df: DataFrame with code, trade_date, time, open, high, low, close, amount columns.
        adj_factor_df: DataFrame with code, trade_date, adj_factor columns.
        lookback: Lookback window for feature calculation.

    Returns:
        DataFrame with code, trade_date, time_index, mic_f1..mic_f10 columns.
    """
    # Generate time_index with progress
    min5_indexed = min5_df.sort(["code", "trade_date", "time"]).with_columns(
        time_index=pl.int_range(1, pl.len() + 1).over(["code", "trade_date"]).cast(pl.Int32)
    ).drop("time")

    # Adjust prices by adj_factor with progress
    min5_adj = min5_indexed.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")

    # Process OHLCV features for micro (time_index present) with progress
    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=min5_adj,
        lookback=lookback,
    )

    # Rename f* to mic_f*
    for i in range(1, 11):
        result = result.rename({f"f{i}": f"mic_f{i}"})

    validate_table(result, PROCESSED_MICRO_SCHEMA)
    return result


def process_mezzo(
    index_df: pl.DataFrame,
    min5_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    lookback: int = MACRO_LOOKBACK,
    **_kwargs,
) -> pl.DataFrame:
    """Process 5-minute OHLCV data into mezzo (30-min) backbone features.

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        min5_df: DataFrame with code, trade_date, time, open, high, low, close, amount columns.
        adj_factor_df: DataFrame with code, trade_date, adj_factor columns.
        lookback: Lookback window for feature calculation.

    Returns:
        DataFrame with code, trade_date, time_index, mzo_f1..mzo_f10 columns.
    """
    # Aggregate 5-min bars to 30-min bars with progress
    tqdm.write("Aggregating to 30-min bars...")
    min30 = _aggregate_30min(min5_df)

    # Adjust prices by adj_factor with progress
    tqdm.write("Adjusting prices...")
    min30_adj = min30.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")

    # Process OHLCV features for mezzo (time_index present) with progress
    tqdm.write(f"Processing {lookback}-step mezzo features...")
    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=min30_adj,
        lookback=lookback,
    )

    # Rename f* to mzo_f*
    for i in tqdm(range(1, 11), desc="Renaming mezzo features", disable=config.debug):
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
    """Process moneyflow and daily data into sidechain energy modulation features.

    Args:
        index_df: DataFrame with code, trade_date, logic_index columns.
        daily_df: DataFrame with code, trade_date, open, high, low, close, amount columns.
        adj_factor_df: DataFrame with code, trade_date, adj_factor columns.
        moneyflow_df: DataFrame with code, trade_date, buy/sell large and extra-large amount columns.
        lookback: Lookback window for feature calculation.

    Returns:
        DataFrame with code, trade_date, mf_abs_rank, mf_impact, mf_conviction, energy_factor, mkt_vola_rank columns.
    """
    # Adjust prices by adj_factor with progress
    tqdm.write("Adjusting prices...")
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

    # Merge daily with moneyflow with progress
    tqdm.write("Merging moneyflow data...")
    df = daily_adj.join(
        moneyflow_df,
        on=["code", "trade_date"],
        how="inner",
    )

    # Compute main force metrics with progress
    tqdm.write("Computing sidechain features...")
    buy_main = pl.col("buy_lg_amount") + pl.col("buy_elg_amount")
    sell_main = pl.col("sell_lg_amount") + pl.col("sell_elg_amount")
    net_main = buy_main - sell_main

    df = df.with_columns([
        net_main.alias("net_main_amt"),
        buy_main.alias("buy_main_amt"),
        sell_main.alias("sell_main_amt"),
    ])

    # F3 and F5 computation for energy_factor
    short_lookback = max(lookback // 4, 1)
    df = df.sort(["code", "trade_date"]).with_columns([
        pl.col("close").shift(1).over("code").alias("close_prev"),
        pl.col("amount").rolling_mean(short_lookback).over("code").alias("ma_amt_short"),
    ])

    # F3: sgn(rt) * ln(Amt / MA(Amt, short_lookback))
    f3 = (pl.col("close").log() - pl.col("close_prev").log()).sign() * (
        pl.col("amount") / (pl.col("ma_amt_short") + _EPS)
    ).log()

    # F5: ln(Ht / Lt)
    f5 = (pl.col("high") / (pl.col("low") + _EPS)).log()

    # Compute sidechain features
    group_key = ["trade_date"]
    df = df.with_columns([
        # mf_abs_rank: cross-sectional normal rank of Net_Main_Amt
        pl.col("net_main_amt").map_batches(_normal_rank).over(group_key).alias("mf_abs_rank"),
        # mf_impact: Net_Main_Amt / MA(Amount, short_lookback)
        (pl.col("net_main_amt") / (pl.col("ma_amt_short") + _EPS)).alias("mf_impact"),
        # mf_conviction: Buy_Main / (Sell_Main + ε)
        (pl.col("buy_main_amt") / (pl.col("sell_main_amt") + _EPS)).alias("mf_conviction"),
        # energy_factor: Rank(F3 * F5)
        (f3 * f5).map_batches(_normal_rank).over(group_key).alias("energy_factor"),
        # mkt_vola_rank (relative_vol): Amount_t / MA(Amount, lookback//4)
        (pl.col("amount") / (pl.col("ma_amt_short") + _EPS)).alias("mkt_vola_rank"),
    ])

    # Select output columns
    result = df.select([
        "code", "trade_date", "mf_abs_rank", "mf_impact",
        "mf_conviction", "energy_factor", "mkt_vola_rank",
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

    df = daily_adj.with_columns([
        weighted_sum.alias("S"),
        max_ret.alias("M"),
    ])

    # Cross-sectional NormalRank within each trade_date
    group_key = ["trade_date"]
    df = df.with_columns([
        pl.col("S").map_batches(_normal_rank).over(group_key).alias("R_S"),
        pl.col("M").map_batches(_normal_rank).over(group_key).alias("R_M"),
    ])

    # Weight fusion: F_t = NormalRank(0.5 * R_S + 0.5 * R_M)
    df = df.with_columns(
        (0.5 * pl.col("R_S") + 0.5 * pl.col("R_M")).map_batches(_normal_rank).alias("F")
    )

    # Gradient enhancement: Label_final = F^3
    df = df.with_columns(
        pl.col("F").pow(3).alias("label_final")
    )

    # Select output columns
    result = df.select(["code", "trade_date", "label_final"]).sort(["code", "trade_date"])

    validate_table(result, PROCESSED_LABEL_SCHEMA)
    return result


# === Internal OHLCV Feature Processor ===

_EPS = 1e-8


def _normal_rank(s: pl.Series) -> pl.Series:
    """Apply normal rank transformation: Φ^{-1}((rank - 0.5) / N)."""
    n = len(s)
    if n == 0:
        return s
    ranked = s.rank(method="average")
    return pl.Series(norm.ppf((ranked.to_numpy() - 0.5) / n))


def vectorized_rolling_slope(s: pl.Series, n: int) -> pl.Series:
    """
    全量向量化：1次函数调用，算完一整只股票的所有斜率
    """
    y = s.to_numpy()
    L = len(y)

    if L < n:
        return pl.Series(s.name, np.full(L, np.nan))

    # 1. 把你代码里的核心常数提出来（完全一样）
    x = np.arange(n, dtype=np.float64)
    x_mean = (n - 1) / 2.0
    denom = (n * n * n - n) / 12.0

    if denom == 0:
        return pl.Series(s.name, np.full(L, 0.0))

    # 2. 核心魔法：直接切出所有滚动窗口，形成 (L-n+1, n) 的矩阵
    # 这一步是内存零拷贝的，瞬间完成
    y_windows = sliding_window_view(y, window_shape=n)

    # 3. 对应你代码里的：y - y_mean
    # 这里我们按行求均值，并保持维度 (L-n+1, 1) 以便广播相减
    y_means = y_windows.mean(axis=1, keepdims=True)

    # 4. 对应你代码里的：np.sum((x - x_mean) * (y - y_mean))
    # 直接利用矩阵乘法（np.dot），让 C 语言底层跑满 CPU
    # y_windows - y_means 是 (L-n+1, n)
    # x - x_mean 是 (n,)
    numer = np.dot(y_windows - y_means, x - x_mean)

    # 5. 除以固定的分母
    slopes = numer / denom

    # 6. 前面补齐 n-1 个 NaN，对齐原始长度
    result = np.full(L, np.nan)
    result[n - 1:] = slopes

    return pl.Series(s.name, result)


def _process_ohlcv(
    index_df: pl.DataFrame,
    ohlcv_df: pl.DataFrame,
    lookback: int = MACRO_LOOKBACK,
) -> pl.DataFrame:
    """Process OHLCV data into 10 backbone features."""
    vlog(_SRC, "Joining index...")
    # Join with index
    df = ohlcv_df.join(
        index_df.select(["code", "trade_date", "logic_index"]),
        on=["code", "trade_date"],
        how="left",
    )

    # Determine structure
    has_time_index = "time_index" in df.columns
    sort_cols = ["code", "trade_date"] + (["time_index"] if has_time_index else [])
    group_key = ["trade_date"] + (["time_index"] if has_time_index else [])

    df = df.sort(sort_cols)

    # === Helper columns ===
    vlog(_SRC, "Calculating helper columns...")
    short_lookback = max(lookback // 4, 1)
    df = df.with_columns([
        pl.col("close").shift(1).over("code").alias("close_prev"),
        pl.col("amount").rolling_mean(short_lookback).over("code").alias("ma_amt"),
        pl.col("close").rolling_mean(lookback).over("code").alias("ma_c"),
        pl.col("high").rolling_max(lookback).over("code").alias("roll_high"),
        pl.col("low").rolling_min(lookback).over("code").alias("roll_low"),
    ])

    # F1: log return (NormalRank)
    df = df.with_columns(
        (pl.col("close").log() - pl.col("close_prev").log()).alias("f1_raw")
    )

    # F2: price range position (Raw)
    df = df.with_columns(
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + _EPS)).alias("f2_raw")
    )

    # F3: volume-price divergence (NormalRank): sgn(rt) * ln(Amt / MA(Amt, lookback//4))
    df = df.with_columns(
        (pl.col("f1_raw").sign() * (pl.col("amount") / (pl.col("ma_amt") + _EPS)).log()).alias("f3_raw")
    )

    # F4: center of gravity bias (NormalRank): (Ct - (Ht+Lt)/2) / Ct-1
    df = df.with_columns(
        ((pl.col("close") - (pl.col("high") + pl.col("low")) / 2) / (pl.col("close_prev") + _EPS)).alias("f4_raw")
    )

    # F5: log volatility (NormalRank): ln(Ht / Lt)
    df = df.with_columns(
        (pl.col("high") / (pl.col("low") + _EPS)).log().alias("f5_raw")
    )

    # T6: MA bias (NormalRank): Ct / MA(C, n) - 1
    df = df.with_columns(
        (pl.col("close") / (pl.col("ma_c") + _EPS) - 1).alias("f6_raw")
    )

    # T7: channel position (Raw): (Ct - min(L,n)) / (max(H,n) - min(L,n))
    df = df.with_columns(
        ((pl.col("close") - pl.col("roll_low")) / (pl.col("roll_high") - pl.col("roll_low") + _EPS)).alias("f7_raw")
    )

    # T8: normalized slope (NormalRank): LinReg(C,n).slope / ATR(n)
    vlog(_SRC, "Computing T8 (ATR/Slope)...")
    # True Range
    tr = pl.max_horizontal([
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close_prev")).abs(),
        (pl.col("low") - pl.col("close_prev")).abs(),
    ])
    df = df.with_columns(tr.alias("tr"))
    df = df.with_columns(
        pl.col("tr").rolling_mean(lookback).over("code").alias("atr")
    )
    vlog(_SRC, "Computing Slope...")
    # Rolling LinReg slope
    df = df.with_columns(
        pl.col("close")
        .map_batches(lambda s: vectorized_rolling_slope(s, lookback))
        .over("code")
        .alias("slope")
    )
    df = df.with_columns(
        (pl.col("slope") / (pl.col("atr") + _EPS)).alias("f8_raw")
    )
    vlog(_SRC, "Computing T9...")
    # T9: OBV ratio (Raw): sum(sgn(ri)*Amount_i) / sum(Amount_i)
    df = df.with_columns(
        ((pl.col("f1_raw").sign() * pl.col("amount")).rolling_sum(lookback).over("code") /
         (pl.col("amount").rolling_sum(lookback).over("code") + _EPS)).alias("f9_raw")
    )

    # T10: cross-period momentum (NormalRank): ln(Ct / Ct-n)
    df = df.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(lookback).over("code")).alias("f10_raw")
    )

    # === Apply NormalRank ===
    vlog(_SRC, "Applying NormalRank...")
    norm_rank_cols = ["f1_raw", "f3_raw", "f4_raw", "f5_raw", "f6_raw", "f8_raw", "f10_raw"]
    for col in norm_rank_cols:
        df = df.with_columns(
            pl.col(col).map_batches(_normal_rank).over(group_key).alias(col)
        )

    # Drop helper columns
    cols_to_drop = [
        "close_prev", "ma_amt", "ma_c", "roll_high", "roll_low",
        "tr", "atr", "slope", "logic_index",
    ]
    df = df.drop([c for c in cols_to_drop if c in df.columns])

    # Rename f*_raw to f*
    for i in range(1, 11):
        df = df.rename({f"f{i}_raw": f"f{i}"})

    # Select output columns
    vlog(_SRC, "Selecting output columns...")
    select_cols = (
        ["code", "trade_date"]
        + (["time_index"] if has_time_index else [])
        + [f"f{i}" for i in range(1, 11)]
    )
    result = df.select([c for c in select_cols if c in df.columns])

    # Drop time column if present
    if "time" in result.columns:
        result = result.drop("time")

    return result


# === Internal Aggregation Functions ===

def _aggregate_30min(min5_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 5-minute OHLCV bars into 30-minute bars.

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
