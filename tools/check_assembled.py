import numpy as np

from config.config import assembled_dir

# 替换成你实际的文件路径
file_path = assembled_dir / (input("Input file path: ") + ".npy")

# 加载数据（只读模式）
data = np.load(file_path, mmap_mode='r')

# 提取第二列 (is_valid_step)
valid_data = data[:, 1]

# 计算统计信息
total_count = len(valid_data)
valid_count = np.sum(valid_data)
ratio = valid_count / total_count

print(f"--- 数据统计: {file_path} ---")
print(f"总天数 (Rows): {total_count}")
print(f"有效天数 (Valid): {int(valid_count)}")
print(f"无效天数 (Invalid): {int(total_count - valid_count)}")
print(f"可用比例 (Ratio): {ratio:.2%}")