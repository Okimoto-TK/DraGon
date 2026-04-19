import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import zlib

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from config.data import assembled_dir, debug, label_schema_version as PACKED_LABEL_SCHEMA_VERSION, packed_min_files_per_code, processed_path, train_seed
from src.data.registry.dataset import MACRO_LOOKBACK, MEZZO_LOOKBACK, MICRO_LOOKBACK, WARMUP_BARS


# --- Feature column definitions ---
MACRO_FEATURES = [f"mcr_f{i}" for i in range(11)]
SIDECHAIN_FEATURES = [
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
]
LABEL_COLS = ["label_ret", "label_rv"]
MEZZO_FEATURES = [f"mzo_f{i}" for i in range(11)]
MICRO_FEATURES = [f"mic_f{i}" for i in range(11)]
MICRO_USED_FEATURES = MICRO_FEATURES

# Int8 feature indices within each scale's feature list (f6, f7)
MACRO_INT8_FEATURES = ["mcr_f6", "mcr_f7"]
MEZZO_INT8_FEATURES = ["mzo_f6", "mzo_f7"]
MICRO_INT8_FEATURES = ["mic_f6", "mic_f7"]

# Float features (f0-f5, f8-f10) for each scale = 9 total
MACRO_FLOAT_FEATURES = [f"mcr_f{i}" for i in list(range(6)) + list(range(8, 11))]
MEZZO_FLOAT_FEATURES = [f"mzo_f{i}" for i in list(range(6)) + list(range(8, 11))]
MICRO_FLOAT_FEATURES = [f"mic_f{i}" for i in list(range(6)) + list(range(8, 11))]

# Bars per day for each high-frequency scale
MEZZO_BARS_PER_DAY = 8
MICRO_BARS_PER_DAY = 48

# Daily window sizes: lookback + warmup, converted to days
MACRO_WINDOW_DAYS = MACRO_LOOKBACK + WARMUP_BARS                          # 64 + 32 = 96
MEZZO_WINDOW_DAYS = MEZZO_LOOKBACK // MEZZO_BARS_PER_DAY + (WARMUP_BARS + MEZZO_BARS_PER_DAY - 1) // MEZZO_BARS_PER_DAY  # 12 + 4 = 16
MICRO_WINDOW_DAYS = MICRO_LOOKBACK // MICRO_BARS_PER_DAY + (WARMUP_BARS + MICRO_BARS_PER_DAY - 1) // MICRO_BARS_PER_DAY  # 3 + 1 = 4

# Total bar counts per scale (after windowing & flattening)
MACRO_TOTAL_BARS = MACRO_WINDOW_DAYS                                      # 96
MEZZO_TOTAL_BARS = MEZZO_WINDOW_DAYS * MEZZO_BARS_PER_DAY                 # 16 * 8 = 128
MICRO_TOTAL_BARS = MICRO_WINDOW_DAYS * MICRO_BARS_PER_DAY                 # 4 * 48 = 192

_LABEL_NAMES_ARRAY = np.asarray(LABEL_COLS, dtype="<U32")


def _compute_sample_valid(step_valid: np.ndarray) -> np.ndarray:
    macro_valid = sliding_window_view(step_valid, MACRO_WINDOW_DAYS).all(axis=-1)
    mezzo_valid = sliding_window_view(step_valid, MEZZO_WINDOW_DAYS).all(axis=-1)
    micro_valid = sliding_window_view(step_valid, MICRO_WINDOW_DAYS).all(axis=-1)
    start_idx = max(MACRO_WINDOW_DAYS, MEZZO_WINDOW_DAYS, MICRO_WINDOW_DAYS) - 1
    return (
        macro_valid[start_idx - MACRO_WINDOW_DAYS + 1 :]
        & mezzo_valid[start_idx - MEZZO_WINDOW_DAYS + 1 :]
        & micro_valid[start_idx - MICRO_WINDOW_DAYS + 1 :]
    )


def _window_samples(values: np.ndarray, lookback: int) -> np.ndarray:
    windows = sliding_window_view(values, lookback, axis=0)
    return np.transpose(windows, (0, 2, 1)).astype(np.float32, copy=False)


def _window_samples_int8(values: np.ndarray, lookback: int) -> np.ndarray:
    """Window samples preserving int8 dtype for limit-hit and time features."""
    windows = sliding_window_view(values, lookback, axis=0)
    return np.transpose(windows, (0, 2, 1)).astype(np.int8, copy=False)


def _build_packed_payload(
    float_data: np.ndarray,
    int8_data: np.ndarray,
) -> dict[str, np.ndarray]:
    start_idx = max(MACRO_WINDOW_DAYS, MEZZO_WINDOW_DAYS, MICRO_WINDOW_DAYS) - 1
    n_rows = float_data.shape[0]
    if n_rows <= start_idx:
        return {
            "date": np.empty((0,), dtype=np.float32),
            "label": np.empty((0, len(LABEL_COLS)), dtype=np.float32),
            "macro": np.empty((0, len(MACRO_FLOAT_FEATURES), MACRO_TOTAL_BARS), dtype=np.float32),
            "sidechain": np.empty((0, len(SIDECHAIN_FEATURES), MACRO_TOTAL_BARS), dtype=np.float32),
            "mezzo": np.empty((0, len(MEZZO_FLOAT_FEATURES), MEZZO_TOTAL_BARS), dtype=np.float32),
            "micro": np.empty((0, len(MICRO_FLOAT_FEATURES), MICRO_TOTAL_BARS), dtype=np.float32),
            "macro_i8": np.empty((0, len(MACRO_INT8_FEATURES), MACRO_TOTAL_BARS), dtype=np.int8),
            "mezzo_i8": np.empty((0, len(MEZZO_INT8_FEATURES), MEZZO_TOTAL_BARS), dtype=np.int8),
            "micro_i8": np.empty((0, len(MICRO_INT8_FEATURES), MICRO_TOTAL_BARS), dtype=np.int8),
        }

    sample_valid = _compute_sample_valid(float_data[:, 1] > 0.5)

    # --- Float data layout ---
    # float_data columns: date_val(1), is_valid_step(1), labels(2),
    #   macro_float(9), sidechain(13), mezzo_float_flat(9*8), micro_float_flat(9*48)
    label_end = 2 + len(LABEL_COLS)
    macro_start = label_end
    macro_end = macro_start + len(MACRO_FLOAT_FEATURES)
    sidechain_start = macro_end
    sidechain_end = sidechain_start + len(SIDECHAIN_FEATURES)
    mezzo_start = sidechain_end
    mezzo_end = mezzo_start + len(MEZZO_FLOAT_FEATURES) * MEZZO_BARS_PER_DAY
    micro_start = mezzo_end
    micro_end = micro_start + len(MICRO_FLOAT_FEATURES) * MICRO_BARS_PER_DAY

    macro = _window_samples(float_data[:, macro_start:macro_end], MACRO_WINDOW_DAYS).transpose(0, 2, 1)
    sidechain = _window_samples(float_data[:, sidechain_start:sidechain_end], MACRO_WINDOW_DAYS).transpose(0, 2, 1)

    mezzo_flat = float_data[:, mezzo_start:mezzo_end]
    mezzo_windows = _window_samples(mezzo_flat, MEZZO_WINDOW_DAYS)[start_idx - MEZZO_WINDOW_DAYS + 1 :]
    mezzo = (
        mezzo_windows
        .reshape(-1, MEZZO_WINDOW_DAYS, len(MEZZO_FLOAT_FEATURES), MEZZO_BARS_PER_DAY)
        .transpose(0, 2, 1, 3)
        .reshape(-1, len(MEZZO_FLOAT_FEATURES), MEZZO_TOTAL_BARS)
        .astype(np.float32, copy=False)
    )

    micro_flat = float_data[:, micro_start:micro_end]
    micro_windows = _window_samples(micro_flat, MICRO_WINDOW_DAYS)[start_idx - MICRO_WINDOW_DAYS + 1 :]
    micro = (
        micro_windows
        .reshape(-1, MICRO_WINDOW_DAYS, len(MICRO_FLOAT_FEATURES), MICRO_BARS_PER_DAY)
        .transpose(0, 2, 1, 3)
        .reshape(-1, len(MICRO_FLOAT_FEATURES), MICRO_TOTAL_BARS)
        .astype(np.float32, copy=False)
    )

    date = np.asarray(float_data[start_idx:, 0], dtype=np.float32)
    label = np.asarray(float_data[start_idx:, 2:label_end], dtype=np.float32)

    # --- Int8 data layout ---
    # int8_data columns: macro_i8(2), mezzo_i8_flat(2*8), micro_i8_flat(2*48)
    i8_macro_end = len(MACRO_INT8_FEATURES)
    i8_mezzo_start = i8_macro_end
    i8_mezzo_end = i8_mezzo_start + len(MEZZO_INT8_FEATURES) * MEZZO_BARS_PER_DAY
    i8_micro_start = i8_mezzo_end
    i8_micro_end = i8_micro_start + len(MICRO_INT8_FEATURES) * MICRO_BARS_PER_DAY

    macro_i8 = _window_samples_int8(int8_data[:, 0:i8_macro_end], MACRO_WINDOW_DAYS).transpose(0, 2, 1)

    mezzo_i8_flat = int8_data[:, i8_mezzo_start:i8_mezzo_end]
    mezzo_i8_windows = _window_samples_int8(mezzo_i8_flat, MEZZO_WINDOW_DAYS)[start_idx - MEZZO_WINDOW_DAYS + 1 :]
    mezzo_i8 = (
        mezzo_i8_windows
        .reshape(-1, MEZZO_WINDOW_DAYS, len(MEZZO_INT8_FEATURES), MEZZO_BARS_PER_DAY)
        .transpose(0, 2, 1, 3)
        .reshape(-1, len(MEZZO_INT8_FEATURES), MEZZO_TOTAL_BARS)
        .astype(np.int8, copy=False)
    )

    micro_i8_flat = int8_data[:, i8_micro_start:i8_micro_end]
    micro_i8_windows = _window_samples_int8(micro_i8_flat, MICRO_WINDOW_DAYS)[start_idx - MICRO_WINDOW_DAYS + 1 :]
    micro_i8 = (
        micro_i8_windows
        .reshape(-1, MICRO_WINDOW_DAYS, len(MICRO_INT8_FEATURES), MICRO_BARS_PER_DAY)
        .transpose(0, 2, 1, 3)
        .reshape(-1, len(MICRO_INT8_FEATURES), MICRO_TOTAL_BARS)
        .astype(np.int8, copy=False)
    )

    keep = np.asarray(sample_valid, dtype=bool)

    return {
        "date": date[keep],
        "label": label[keep],
        "macro": macro[keep],
        "sidechain": sidechain[keep],
        "mezzo": mezzo[keep],
        "micro": micro[keep],
        "macro_i8": macro_i8[keep],
        "mezzo_i8": mezzo_i8[keep],
        "micro_i8": micro_i8[keep],
    }


def _shard_suffix(index: int) -> str:
    return f"__{index:03d}"


def _shuffled_shard_indices(count: int, *, code: str) -> list[np.ndarray]:
    if count <= 0:
        return []

    num_shards = min(max(1, int(packed_min_files_per_code)), count)
    seed = np.uint64(train_seed) ^ np.uint64(zlib.crc32(code.encode("utf-8")))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(count)
    return [np.asarray(part, dtype=np.int32) for part in np.array_split(shuffled, num_shards) if part.size > 0]


def _clear_existing_packed_files(code: str) -> None:
    legacy_path = assembled_dir / f"{code}.npz"
    if legacy_path.exists():
        legacy_path.unlink()
    for path in assembled_dir.glob(f"{code}__*.npz"):
        path.unlink()


def _write_packed_samples(code: str, float_data: np.ndarray, int8_data: np.ndarray) -> None:
    payload = _build_packed_payload(float_data, int8_data)
    _clear_existing_packed_files(code)

    shards = _shuffled_shard_indices(int(payload["date"].shape[0]), code=code)
    if not shards:
        np.savez(
            assembled_dir / f"{code}.npz",
            label_schema_version=np.asarray(PACKED_LABEL_SCHEMA_VERSION, dtype=np.int32),
            label_names=_LABEL_NAMES_ARRAY,
            date=payload["date"],
            label=payload["label"],
            macro=payload["macro"],
            sidechain=payload["sidechain"],
            mezzo=payload["mezzo"],
            micro=payload["micro"],
            macro_i8=payload["macro_i8"],
            mezzo_i8=payload["mezzo_i8"],
            micro_i8=payload["micro_i8"],
        )
        return

    single_shard = len(shards) == 1
    for shard_idx, order in enumerate(shards):
        shard_path = (
            assembled_dir / f"{code}.npz"
            if single_shard
            else assembled_dir / f"{code}{_shard_suffix(shard_idx)}.npz"
        )
        np.savez(
            shard_path,
            label_schema_version=np.asarray(PACKED_LABEL_SCHEMA_VERSION, dtype=np.int32),
            label_names=_LABEL_NAMES_ARRAY,
            date=payload["date"][order],
            label=payload["label"][order],
            macro=payload["macro"][order],
            sidechain=payload["sidechain"][order],
            mezzo=payload["mezzo"][order],
            micro=payload["micro"][order],
            macro_i8=payload["macro_i8"][order],
            mezzo_i8=payload["mezzo_i8"][order],
            micro_i8=payload["micro_i8"][order],
        )


def _scan_parquet(path: str | os.PathLike[str], columns: list[str]) -> pl.LazyFrame:
    return pl.scan_parquet(path).select(columns)


def _scalar_finite_expr(cols: list[str]) -> pl.Expr:
    exprs = [
        (pl.col(col).is_not_null() & pl.col(col).is_finite()).fill_null(False)
        for col in cols
    ]
    return pl.all_horizontal(exprs)


def _list_finite_expr(cols: list[str], expected_len: int) -> pl.Expr:
    exprs = []
    for col in cols:
        exprs.append(
            pl.when(pl.col(col).is_not_null())
            .then(
                (pl.col(col).list.len() == expected_len)
                & pl.col(col).list.eval(pl.element().is_null()).list.any().not_()
                & pl.col(col).list.eval(pl.element().is_finite()).list.all()
            )
            .otherwise(False)
            .fill_null(False)
        )
    return pl.all_horizontal(exprs)


def _list_non_null_expr(cols: list[str], expected_len: int) -> pl.Expr:
    exprs = []
    for col in cols:
        exprs.append(_single_list_non_null_expr(col, expected_len))
    return pl.all_horizontal(exprs)


def _single_list_non_null_expr(col: str, expected_len: int) -> pl.Expr:
    return (
        pl.when(pl.col(col).is_not_null())
        .then(
            (pl.col(col).list.len() == expected_len)
            & pl.col(col).list.eval(pl.element().is_null()).list.any().not_()
        )
        .otherwise(False)
        .fill_null(False)
    )


def _safe_list_int8_expr(col: str, expected_len: int) -> pl.Expr:
    zero_list = pl.lit([0] * expected_len, dtype=pl.List(pl.Int8))
    return (
        pl.when(_single_list_non_null_expr(col, expected_len))
        .then(pl.col(col))
        .otherwise(zero_list)
        .cast(pl.List(pl.Int8))
    )


def process_single_stock(code: str):
    mask_file = processed_path.mask_dir / f"{code}.parquet"
    if not mask_file.exists():
        return

    try:
        # 1. Scan only the columns needed for assembly.
        mask_lf = _scan_parquet(mask_file, ["code", "trade_date", "filter_mask"])
        label_lf = _scan_parquet(
            processed_path.label_dir / f"{code}.parquet",
            ["code", "trade_date", *LABEL_COLS],
        )
        macro_lf = _scan_parquet(
            processed_path.macro_dir / f"{code}.parquet",
            ["code", "trade_date", *MACRO_FEATURES],
        )
        sidechain_lf = _scan_parquet(
            processed_path.sidechain_dir / f"{code}.parquet",
            ["code", "trade_date", *SIDECHAIN_FEATURES],
        )
        mezzo_lf = _scan_parquet(
            processed_path.mezzo_dir / f"{code}.parquet",
            ["code", "trade_date", "time_index", *MEZZO_FEATURES],
        )
        micro_lf = _scan_parquet(
            processed_path.micro_dir / f"{code}.parquet",
            ["code", "trade_date", "time_index", *MICRO_FEATURES],
        )

        # 2. Aggregate high-frequency into per-day list payloads.
        mzo_agg = (
            mezzo_lf
            .sort(["trade_date", "time_index"])
            .group_by(["code", "trade_date"], maintain_order=True)
            .agg([pl.col(f) for f in MEZZO_FEATURES])
        )
        mic_agg = (
            micro_lf
            .sort(["trade_date", "time_index"])
            .group_by(["code", "trade_date"], maintain_order=True)
            .agg([pl.col(f) for f in MICRO_FEATURES])
        )

        # 3. Join (full date range as backbone)
        df = (
            mask_lf
            .join(label_lf, on=["code", "trade_date"], how="left")
            .join(macro_lf, on=["code", "trade_date"], how="left")
            .join(sidechain_lf, on=["code", "trade_date"], how="left")
            .join(mzo_agg, on=["code", "trade_date"], how="left")
            .join(mic_agg, on=["code", "trade_date"], how="left")
        )

        # Validity: float features must be finite; int8 features must be non-null.
        scalar_valid_expr = _scalar_finite_expr(LABEL_COLS + MACRO_FLOAT_FEATURES + SIDECHAIN_FEATURES)
        macro_int8_valid_expr = (
            pl.all_horizontal([pl.col(f).is_not_null() for f in MACRO_INT8_FEATURES])
        )
        # For list columns, split into float and int8
        mezzo_list_valid_expr = _list_finite_expr(MEZZO_FLOAT_FEATURES, expected_len=MEZZO_BARS_PER_DAY)
        mezzo_int8_list_valid_expr = _list_non_null_expr(
            MEZZO_INT8_FEATURES,
            expected_len=MEZZO_BARS_PER_DAY,
        )
        micro_list_valid_expr = _list_finite_expr(MICRO_FLOAT_FEATURES, expected_len=MICRO_BARS_PER_DAY)
        micro_int8_list_valid_expr = _list_non_null_expr(
            MICRO_INT8_FEATURES,
            expected_len=MICRO_BARS_PER_DAY,
        )

        # 4. Build date and validity flag
        df = df.with_columns([
            pl.col("trade_date").cast(pl.String).str.replace_all("-", "").cast(pl.Float32).alias("date_val"),
            (
                pl.col("filter_mask").fill_null(False)
                & scalar_valid_expr
                & macro_int8_valid_expr
                & mezzo_list_valid_expr
                & mezzo_int8_list_valid_expr
                & micro_list_valid_expr
                & micro_int8_list_valid_expr
            ).cast(pl.Float32).alias("is_valid_step")
        ])

        # 5. Sort
        df = df.sort("trade_date")

        # 6a. Float features: date_val, is_valid_step, labels, macro float, sidechain,
        #     mezzo float list, micro float list
        float_select = [pl.col("date_val"), pl.col("is_valid_step")]
        float_select.extend([pl.col(f).cast(pl.Float32) for f in LABEL_COLS])
        float_select.extend([pl.col(f).cast(pl.Float32) for f in MACRO_FLOAT_FEATURES])
        float_select.extend([pl.col(f).cast(pl.Float32) for f in SIDECHAIN_FEATURES])

        for f in MEZZO_FLOAT_FEATURES:
            float_select.append(
                pl.col(f).list.to_struct(fields=[f"{f}_{j}" for j in range(MEZZO_BARS_PER_DAY)]).struct.field("*")
            )
        for f in MICRO_FLOAT_FEATURES:
            float_select.append(
                pl.col(f).list.to_struct(fields=[f"{f}_{j}" for j in range(MICRO_BARS_PER_DAY)]).struct.field("*")
            )

        float_df = (
            df.sort("trade_date")
            .select(float_select)
            .select(pl.col(pl.NUMERIC_DTYPES))
            .collect()
        )
        float_numpy = float_df.to_numpy().astype(np.float32, copy=False)

        # 6b. Int8 features: macro int8, mezzo int8 list, micro int8 list
        int8_select = [pl.col(f).fill_null(0).cast(pl.Int8) for f in MACRO_INT8_FEATURES]
        for f in MEZZO_INT8_FEATURES:
            int8_select.append(
                _safe_list_int8_expr(
                    f,
                    expected_len=MEZZO_BARS_PER_DAY,
                ).list.to_struct(
                    fields=[f"{f}_{j}" for j in range(MEZZO_BARS_PER_DAY)]
                ).struct.field("*")
            )
        for f in MICRO_INT8_FEATURES:
            int8_select.append(
                _safe_list_int8_expr(
                    f,
                    expected_len=MICRO_BARS_PER_DAY,
                ).list.to_struct(
                    fields=[f"{f}_{j}" for j in range(MICRO_BARS_PER_DAY)]
                ).struct.field("*")
            )

        int8_df = (
            df.sort("trade_date")
            .select(int8_select)
            .collect()
        )
        # Build int8 numpy manually to preserve dtype (polars to_numpy may upcast)
        int8_cols = []
        for c in int8_df.columns:
            int8_cols.append(int8_df[c].to_numpy().astype(np.int8))
        int8_numpy = np.column_stack(int8_cols) if int8_cols else np.empty((float_numpy.shape[0], 0), dtype=np.int8)

        # 7. Write training-ready packed tensor bundles
        _write_packed_samples(code, float_numpy, int8_numpy)

    except Exception as e:
        print(f" {code} assembly failed: {e}")
        raise e


def assemble_all() -> None:
    """Assemble processed per-code parquet files directly into packed tensor ``.npz`` files."""
    all_codes = [f.stem for f in processed_path.mask_dir.glob("*.parquet")]
    max_workers = min(6, max(1, (os.cpu_count() or 1) // 2))
    if debug or max_workers <= 1:
        for code in all_codes:
            process_single_stock(code)
        return

    from tqdm import tqdm

    ctx = mp.get_context("spawn")
    chunksize = max(1, min(64, len(all_codes) // (max_workers * 4) or 1))
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for _ in tqdm(
            executor.map(process_single_stock, all_codes, chunksize=chunksize),
            total=len(all_codes),
            desc="Assembling",
            disable=debug,
        ):
            pass
