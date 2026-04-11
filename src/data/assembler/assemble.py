import numpy as np
import polars as pl

from config.config import processed_path, assembled_dir
from src.data.storage.npy_io import write_npy
from src.data.storage.parquet_io import read_parquet
from src.data.schemas.processed import (
    PROCESSED_MASK_SCHEMA,
    PROCESSED_MACRO_SCHEMA,
    PROCESSED_MEZZO_SCHEMA,
    PROCESSED_MICRO_SCHEMA,
    PROCESSED_SIDECHAIN_SCHEMA,
    PROCESSED_LABEL_SCHEMA
)


# --- 特征列定义 (保持不变) ---
MACRO_FEATURES = [f"mcr_f{i}" for i in range(9)]
SIDECHAIN_FEATURES = ["gap", "gap_rank", "mf_net_ratio", "mf_net_rank", "mf_concentration", "amt_surge_rank", "velocity_rank", "amihud_impact"]
LABEL_COLS = ["label_S", "label_M", "label_MDD", "label_RV"]
MEZZO_FEATURES = [f"mzo_f{i}" for i in range(9)]
MICRO_FEATURES = [f"mic_f{i}" for i in range(9)]

check_cols = LABEL_COLS + MACRO_FEATURES + SIDECHAIN_FEATURES + MEZZO_FEATURES + MICRO_FEATURES


def process_single_stock(code: str):
    mask_file = processed_path.mask_dir / f"{code}.parquet"
    if not mask_file.exists():
        return

    try:
        # 1. 加载全量日期基准 (不再执行 .filter)
        # 这样能保证 20120104 这种预热期日期也能进入矩阵
        mask_df = read_parquet(mask_file, PROCESSED_MASK_SCHEMA, "mask").select(["code", "trade_date", "filter_mask"])

        # 2. 读取其他特征
        label_df = read_parquet(processed_path.label_dir / f"{code}.parquet", PROCESSED_LABEL_SCHEMA, "label")
        macro_df = read_parquet(processed_path.macro_dir / f"{code}.parquet", PROCESSED_MACRO_SCHEMA, "macro")
        sidechain_df = read_parquet(processed_path.sidechain_dir / f"{code}.parquet", PROCESSED_SIDECHAIN_SCHEMA, "sidechain")
        mezzo_df = read_parquet(processed_path.mezzo_dir / f"{code}.parquet", PROCESSED_MEZZO_SCHEMA, "mezzo")
        micro_df = read_parquet(processed_path.micro_dir / f"{code}.parquet", PROCESSED_MICRO_SCHEMA, "micro")

        # 3. Join (以全量日期为基石)
        df = mask_df.join(label_df, on=["code", "trade_date"], how="left")
        df = df.join(macro_df, on=["code", "trade_date"], how="left")
        df = df.join(sidechain_df, on=["code", "trade_date"], how="left")

        # 高频聚合
        mzo_agg = mezzo_df.sort("time_index").group_by(["code", "trade_date"]).agg([pl.col(f).implode() for f in MEZZO_FEATURES])
        mic_agg = micro_df.sort("time_index").group_by(["code", "trade_date"]).agg([pl.col(f).implode() for f in MICRO_FEATURES])
        df = df.join(mzo_agg, on=["code", "trade_date"], how="left").join(mic_agg, on=["code", "trade_date"], how="left")

        # 4. 构造日期和有效位
        # is_valid_step: 只有当 filter_mask 为 True 且关键特征不为空时才为 1.0
        df = df.with_columns([
            pl.col("trade_date").cast(pl.String).str.replace_all("-", "").cast(pl.Float32).alias("date_val"),
            (
                pl.col("filter_mask") &
                pl.all_horizontal(pl.col(check_cols).is_not_null())
            ).cast(pl.Float32).alias("is_valid_step")
        ])

        # 5. 排序并填充 (1月份没数据的地方会自动填 0)
        df = df.sort("trade_date").fill_null(0.0)

        # 6. 构造最终特征矩阵
        select_exprs = [pl.col("date_val"), pl.col("is_valid_step")]
        select_exprs.extend([pl.col(f).cast(pl.Float32) for f in LABEL_COLS])
        select_exprs.extend([pl.col(f).cast(pl.Float32) for f in MACRO_FEATURES])
        select_exprs.extend([pl.col(f).cast(pl.Float32) for f in SIDECHAIN_FEATURES])

        for f in MEZZO_FEATURES:
            select_exprs.append(pl.col(f).list.to_struct(fields=[f"{f}_{j}" for j in range(8)]).struct.field("*"))
        for f in MICRO_FEATURES:
            select_exprs.append(pl.col(f).list.to_struct(fields=[f"{f}_{j}" for j in range(48)]).struct.field("*"))

        final_df = df.select(select_exprs).select(pl.col(pl.NUMERIC_DTYPES))
        numpy_data = final_df.to_numpy().astype(np.float32)

        # 7. 使用 open_memmap 写入标准 NPY (带 Header，方便以后 np.load)
        target_path = assembled_dir / f"{code}.npy"
        write_npy(target_path, numpy_data)

    except Exception as e:
        print(f" {code} 组装失败: {e}")
        raise e
