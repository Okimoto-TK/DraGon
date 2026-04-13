import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from config.config import processed_path, assembled_dir, debug
from src.data.storage.npy_io import write_npy


# --- 特征列定义 (保持不变) ---
MACRO_FEATURES = [f"mcr_f{i}" for i in range(9)]
SIDECHAIN_FEATURES = ["gap", "gap_rank", "mf_net_ratio", "mf_net_rank", "mf_concentration", "amt_surge_rank", "velocity_rank", "amihud_impact"]
LABEL_COLS = ["label_S", "label_M", "label_MDD", "label_RV"]
MEZZO_FEATURES = [f"mzo_f{i}" for i in range(9)]
MICRO_FEATURES = [f"mic_f{i}" for i in range(9)]

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

        # 7. 使用 open_memmap 写入标准 NPY (带 Header，方便以后 np.load)
        target_path = assembled_dir / f"{code}.npy"
        write_npy(target_path, numpy_data)

    except Exception as e:
        print(f" {code} 组装失败: {e}")
        raise e


def assemble_all() -> None:
    """Assemble all processed per-code parquet files into ``.npy`` arrays."""
    assembled_dir.mkdir(parents=True, exist_ok=True)
    all_codes = [f.stem for f in processed_path.mask_dir.glob("*.parquet")]
    max_workers = min(4, max(1, (os.cpu_count() or 1) // 2))
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
