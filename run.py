"""Parquet file column alignment utility.

Aligns column order across all parquet files in a directory
to match a predefined schema.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from tqdm import tqdm

# ================= Configuration =================
# Target column order (must include all column names)
TARGET_COLUMNS = ["code", "trade_date", "up_limit", "down_limit"]

# Directory containing parquet files to process
SOURCE_DIR = "data/raw/limit"
# ================================================


def align_parquet_columns(folder_path: str, schema_list: list[str]) -> None:
    """Align parquet file columns to match the target schema order.

    Reads each parquet file in the directory, reorders columns to match
    the schema list, and overwrites the file in place.

    Args:
        folder_path: Path to directory containing parquet files.
        schema_list: Ordered list of column names.
    """
    base_dir = Path(folder_path)
    files = list(base_dir.glob("*.parquet"))

    if not files:
        print(f"未在 {folder_path} 下找到文件。")
        return

    print(f"开始对齐列顺序，目标列数: {len(schema_list)}")

    for file_path in tqdm(files, desc="Aligning"):
        try:
            df = pl.read_parquet(file_path)

            # Verify all target columns exist in the file
            current_cols = set(df.columns)
            target_cols_set = set(schema_list)

            if not target_cols_set.issubset(current_cols):
                missing = target_cols_set - current_cols
                print(f"\n跳过文件 {file_path.name}: 缺少列 {missing}")
                continue

            # Reorder columns to match schema
            df_aligned = df.select(schema_list)
            df_aligned.write_parquet(file_path)

        except Exception as e:
            print(f"\n处理 {file_path.name} 时出错: {e}")


if __name__ == "__main__":
    align_parquet_columns(SOURCE_DIR, TARGET_COLUMNS)
