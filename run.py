import pandas as pd
from pathlib import Path


def process_moneyflow_data(directory):
    # 转换为 Path 对象方便路径操作
    root_dir = Path(directory)
    
    # 获取目录下所有 parquet 文件
    files = list(root_dir.glob("*.parquet"))
    
    if not files:
        print(f"在 {directory} 下未找到 parquet 文件。")
        return

    for file_path in files:
        try:
            # 读取文件
            df = pd.read_parquet(file_path)
            
            # 记录是否有修改，避免不必要的保存
            modified = False
            
            for col in df.columns:
                # 处理包含 'amount' 的列 (不区分大小写用 .lower())
                if 'amount' in col.lower():
                    df[col] = df[col] * 10000
                    modified = True
                
                # 处理包含 'vol' 的列
                elif 'vol' in col.lower():
                    df[col] = df[col] * 100
                    modified = True
            
            if modified:
                # 覆盖原文件保存
                df.to_parquet(file_path, index=False)
                print(f"成功处理并更新: {file_path.name}")
            else:
                print(f"跳过（未匹配到列）: {file_path.name}")
                
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")


if __name__ == "__main__":
    target_path = "data/raw/moneyflow"
    process_moneyflow_data(target_path
                           )