import polars as pl
from pathlib import Path
from tqdm import tqdm


def convert_parquet_types(root_path: str):
    # 1. 初始化路径
    base_dir = Path(root_path)

    # 2. 递归查找所有子文件夹下的 parquet 文件 (rglob)
    files = list(base_dir.rglob("*.parquet"))

    if not files:
        print(f"在 {root_path} 下未找到任何 .parquet 文件")
        return

    print(f"找到 {len(files)} 个文件，开始转换类型...")

    for file_path in tqdm(files, desc="Converting"):
        try:
            # 3. 扫描文件结构 (Lazy 模式)
            lf = pl.scan_parquet(file_path)
            cols = lf.collect_schema().names()

            # 4. 构造转换逻辑
            # 4. 构造转换逻辑
            expressions = []

            # 转换 trade_date: 针对 "20150623" 这种格式
            if "trade_date" in cols:
                # 首先确保它是字符串，然后按格式解析
                expressions.append(
                    pl.col("trade_date").cast(pl.String).str.to_date("%Y%m%d")
                )

            # 转换 time: 假设你的格式是 "09:30:00"
            if "time" in cols:
                # 如果 time 格式是 "093000"，请改为 .str.to_time("%H%M%S")
                # 如果是 "09:30:00"，通常 cast(pl.Time) 即可，但 str.to_time 更稳健
                expressions.append(
                    pl.col("time").cast(pl.String).str.to_time("%H:%M:%S")
                )

            # 如果没有任何匹配的列，跳过
            if not expressions:
                continue

            # 5. 执行转换并覆写文件
            df = lf.with_columns(expressions).collect()
            df.write_parquet(file_path)

        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {e}")


if __name__ == "__main__":
    # 请根据实际情况修改这里的路径
    DATA_ROOT = "data"
    convert_parquet_types(DATA_ROOT)