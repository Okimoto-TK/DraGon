import os
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 配置区 =================
TRACKING_URI = "http://117.50.83.183:11111"
EXPERIMENT_NAME = "dragon"
RUN_NAME = "run00_edge"
SAVE_DIR = "./run00_edge_metrics"
MAX_WORKERS = 16  # 并行进程数，建议根据 CPU 核心数或网络带宽调整


# =========================================

def download_single_metric(uri, run_id, key, save_dir):
    """
    单个指标下载的任务函数，将被多进程调用
    注意：在多进程中，每个进程需要重新初始化自己的 MlflowClient
    """
    try:
        client = MlflowClient(tracking_uri=uri)
        history = client.get_metric_history(run_id, key)

        if not history:
            return f"{key}: ⚠️ 无数据"

        # 解析数据
        data = [{
            "step": m.step,
            "value": m.value,
            "timestamp": m.timestamp
        } for m in history]

        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 文件名清洗
        safe_filename = key.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(save_dir, f"{safe_filename}.csv")

        df.to_csv(save_path, index=False)
        return f"{key}: ✅ 成功"
    except Exception as e:
        return f"{key}: ❌ 失败 ({e})"


def download_mlflow_metrics():
    os.makedirs(SAVE_DIR, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    print(f"正在连接到 MLflow 服务器: {TRACKING_URI}")

    # 1. 查找实验
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"❌ 找不到名为 '{EXPERIMENT_NAME}' 的实验。")
        return

    # 2. 查找特定的 Run
    query = f"tags.`mlflow.runName` = '{RUN_NAME}'"
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=query)

    if not runs:
        print(f"❌ 找不到 Run: {RUN_NAME}")
        return

    target_run = runs[0]
    run_id = target_run.info.run_id
    metric_keys = list(target_run.data.metrics.keys())
    if not metric_keys:
        print("⚠️ 该 Run 没有记录任何指标。")
        return

    print(f"\n共找到 {len(metric_keys)} 个指标，启动 {MAX_WORKERS} 个进程并行下载...")

    # 3. 使用进程池进行并行下载
    # 使用 list 转换 keys 以便迭代
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {
            executor.submit(download_single_metric, TRACKING_URI, run_id, key, SAVE_DIR): key
            for key in metric_keys
        }

        # 实时打印完成情况
        for future in as_completed(futures):
            res = future.result()
            print(res)

    print(f"\n🎉 任务结束。保存目录: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    # Windows 下使用多进程必须放在 if __name__ == "__main__": 之下
    download_mlflow_metrics()