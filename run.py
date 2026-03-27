import polars as pl
from pathlib import Path
from tqdm import tqdm

# ================= 配置区 =================
# 1. 在这里填入你希望的最终列顺序（必须包含所有列名）
TARGET_COLUMNS = ["code", "trade_date", "up_limit", "down_limit"]

# 2. 待处理的文件夹路径
SOURCE_DIR = "data/raw/limit"


# =========================================

def align_parquet_columns(folder_path: str, schema_list: list):
    base_dir = Path(folder_path)
    # 获取目录下所有 parquet 文件
    files = list(base_dir.glob("*.parquet"))

    if not files:
        print(f"未在 {folder_path} 下找到文件。")
        return

    print(f"开始对齐列顺序，目标列数: {len(schema_list)}")

    for file_path in tqdm(files, desc="Aligning"):
        try:
            # 1. 读取文件（使用 read_parquet 即可，因为只是调整顺序）
            df = pl.read_parquet(file_path)

            # 2. 检查列是否匹配（可选，防止因列名缺失导致报错）
            current_cols = set(df.columns)
            target_cols_set = set(schema_list)

            if not target_cols_set.issubset(current_cols):
                missing = target_cols_set - current_cols
                print(f"\n跳过文件 {file_path.name}: 缺少列 {missing}")
                continue

            # 3. 核心操作：重新选择列顺序
            # 这只改变排列，不改变数据类型或内容
            df_aligned = df.select(schema_list)

            # 4. 覆写文件
            df_aligned.write_parquet(file_path)

        except Exception as e:
            print(f"\n处理 {file_path.name} 时出错: {e}")


if __name__ == "__main__":
    align_parquet_columns(SOURCE_DIR, TARGET_COLUMNS)