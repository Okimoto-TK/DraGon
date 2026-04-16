import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import zlib

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from config.config import assembled_dir, debug, label_schema_version as PACKED_LABEL_SCHEMA_VERSION, packed_min_files_per_code, processed_path, train_seed


# --- 特征列定义 ---
MACRO_FEATURES = [f"mcr_f{i}" for i in range(9)]
SIDECHAIN_FEATURES = ["gap", "gap_rank", "mf_net_ratio", "mf_net_rank", "mf_concentration", "amount_rank", "velocity_rank", "amihud_impact"]
LABEL_COLS = ["label_ret", "label_rv"]
MEZZO_FEATURES = [f"mzo_f{i}" for i in range(9)]
MICRO_FEATURES = [f"mic_f{i}" for i in range(9)]
MICRO_USED_FEATURES = MICRO_FEATURES

MACRO_LOOKBACK = 64
MEZZO_DAYS = 8
MICRO_DAYS = 1
_LABEL_NAMES_ARRAY = np.asarray(LABEL_COLS, dtype="<U32")


def _compute_sample_valid(step_valid: np.ndarray) -> np.ndarray:
    macro_valid = sliding_window_view(step_valid, MACRO_LOOKBACK).all(axis=-1)
    mezzo_valid = sliding_window_view(step_valid, MEZZO_DAYS).all(axis=-1)
    micro_valid = sliding_window_view(step_valid, MICRO_DAYS).all(axis=-1)
    start_idx = max(MACRO_LOOKBACK, MEZZO_DAYS, MICRO_DAYS) - 1
    return (
        macro_valid[start_idx - MACRO_LOOKBACK + 1 :]
        & mezzo_valid[start_idx - MEZZO_DAYS + 1 :]
        & micro_valid[start_idx - MICRO_DAYS + 1 :]
    )


def _window_samples(values: np.ndarray, lookback: int) -> np.ndarray:
    windows = sliding_window_view(values, lookback, axis=0)
    return np.transpose(windows, (0, 2, 1)).astype(np.float32, copy=False)


def _build_packed_payload(data: np.ndarray) -> dict[str, np.ndarray]:
    start_idx = max(MACRO_LOOKBACK, MEZZO_DAYS, MICRO_DAYS) - 1
    n_rows = data.shape[0]
    if n_rows <= start_idx:
        return {
            "date": np.empty((0,), dtype=np.float32),
            "label": np.empty((0, len(LABEL_COLS)), dtype=np.float32),
            "macro": np.empty((0, len(MACRO_FEATURES), MACRO_LOOKBACK), dtype=np.float32),
            "sidechain": np.empty((0, len(SIDECHAIN_FEATURES), MACRO_LOOKBACK), dtype=np.float32),
            "mezzo": np.empty((0, len(MEZZO_FEATURES), 64), dtype=np.float32),
            "micro": np.empty((0, len(MICRO_USED_FEATURES), 48), dtype=np.float32),
        }

    sample_valid = _compute_sample_valid(data[:, 1] > 0.5)
    label_end = 2 + len(LABEL_COLS)
    macro_start = label_end
    macro_end = macro_start + len(MACRO_FEATURES)
    sidechain_start = macro_end
    sidechain_end = sidechain_start + len(SIDECHAIN_FEATURES)
    mezzo_start = sidechain_end
    mezzo_end = mezzo_start + len(MEZZO_FEATURES) * 8
    micro_start = mezzo_end
    micro_end = micro_start + len(MICRO_USED_FEATURES) * 48

    macro = _window_samples(data[:, macro_start:macro_end], MACRO_LOOKBACK).transpose(0, 2, 1)
    sidechain = _window_samples(data[:, sidechain_start:sidechain_end], MACRO_LOOKBACK).transpose(0, 2, 1)

    mezzo_flat = data[:, mezzo_start:mezzo_end]
    mezzo_windows = _window_samples(mezzo_flat, MEZZO_DAYS)[start_idx - MEZZO_DAYS + 1 :]
    mezzo = (
        mezzo_windows
        .reshape(-1, MEZZO_DAYS, len(MEZZO_FEATURES), 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, len(MEZZO_FEATURES), MEZZO_DAYS * 8)
        .astype(np.float32, copy=False)
    )

    micro_flat = data[:, micro_start:micro_end]
    micro_windows = _window_samples(micro_flat, MICRO_DAYS)[start_idx - MICRO_DAYS + 1 :]
    micro = (
        micro_windows
        .reshape(-1, MICRO_DAYS, len(MICRO_USED_FEATURES), 48)
        .transpose(0, 2, 1, 3)
        .reshape(-1, len(MICRO_USED_FEATURES), MICRO_DAYS * 48)
        .astype(np.float32, copy=False)
    )

    date = np.asarray(data[start_idx:, 0], dtype=np.float32)
    label = np.asarray(data[start_idx:, 2:label_end], dtype=np.float32)
    keep = np.asarray(sample_valid, dtype=bool)

    return {
        "date": date[keep],
        "label": label[keep],
        "macro": macro[keep],
        "sidechain": sidechain[keep],
        "mezzo": mezzo[keep],
        "micro": micro[keep],
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


def _write_packed_samples(code: str, data: np.ndarray) -> None:
    payload = _build_packed_payload(data)
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

        # 2. 高频聚合为按日 list 载荷。
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

        # 3. Join (以全量日期为基石)
        df = (
            mask_lf
            .join(label_lf, on=["code", "trade_date"], how="left")
            .join(macro_lf, on=["code", "trade_date"], how="left")
            .join(sidechain_lf, on=["code", "trade_date"], how="left")
            .join(mzo_agg, on=["code", "trade_date"], how="left")
            .join(mic_agg, on=["code", "trade_date"], how="left")
        )

        scalar_valid_expr = _scalar_finite_expr(LABEL_COLS + MACRO_FEATURES + SIDECHAIN_FEATURES)
        mezzo_valid_expr = _list_finite_expr(MEZZO_FEATURES, expected_len=8)
        micro_valid_expr = _list_finite_expr(MICRO_FEATURES, expected_len=48)

        # 4. 构造日期和有效位
        # is_valid_step: 当天切片只有在 filter_mask 为 True 且所有会进入模型的
        # scalar/list 特征都完整且 finite 时才为 1.0
        df = df.with_columns([
            pl.col("trade_date").cast(pl.String).str.replace_all("-", "").cast(pl.Float32).alias("date_val"),
            (
                pl.col("filter_mask").fill_null(False)
                & scalar_valid_expr
                & mezzo_valid_expr
                & micro_valid_expr
            ).cast(pl.Float32).alias("is_valid_step")
        ])

        # 5. 排序，保留缺失值以便 invalid rows 在 assembled 中显式暴露
        df = df.sort("trade_date")

        # 6. 构造最终特征矩阵
        select_exprs = [pl.col("date_val"), pl.col("is_valid_step")]
        select_exprs.extend([pl.col(f).cast(pl.Float32) for f in LABEL_COLS])
        select_exprs.extend([pl.col(f).cast(pl.Float32) for f in MACRO_FEATURES])
        select_exprs.extend([pl.col(f).cast(pl.Float32) for f in SIDECHAIN_FEATURES])

        for f in MEZZO_FEATURES:
            select_exprs.append(pl.col(f).list.to_struct(fields=[f"{f}_{j}" for j in range(8)]).struct.field("*"))
        for f in MICRO_FEATURES:
            select_exprs.append(pl.col(f).list.to_struct(fields=[f"{f}_{j}" for j in range(48)]).struct.field("*"))

        final_df = (
            df.sort("trade_date")
            .select(select_exprs)
            .select(pl.col(pl.NUMERIC_DTYPES))
            .collect()
        )
        numpy_data = final_df.to_numpy().astype(np.float32, copy=False)

        # 7. 直接写训练就绪的 packed tensor 包，不再落原始 assembled npy
        _write_packed_samples(code, numpy_data)

    except Exception as e:
        print(f" {code} 组装失败: {e}")
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
