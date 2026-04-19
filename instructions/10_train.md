# Train 模块实现规范（最终版，给 coder 的确定性说明）

本模块的总规则：

- 参数分为两种，一种是**隐参数**，放在 `src/models/config/hparams.py` 或 `hparams.yaml`
- 另一种是**开放参数**，放在 `config/models.py`
- 隐参数统一使用 `_` 开头
- 开放参数不使用 `_` 开头
- 本文档只定义 **train / dataloader / runtime / console logging**
- 本文档不重新定义 network 结构
- 本文档当前总目标只有一个：

> **先把整条训练链路稳定跑起来。**

也就是说，当前版本优先级如下：

1. 数据读取正确
2. batch 对接 network-facing contract 正确
3. DataLoader 高效，不出现明显 IO 瓶颈
4. `torch.compile(mode="max-autotune")` 正确接入
5. cudagraph 约束明确，避免图碎和编译时间爆炸
6. Rich 控制台日志可用
7. train/val 正常循环
8. checkpoint 正常保存/恢复

当前版本**不做**：

- 数据可视化
- TensorBoard/MLflow
- 复杂 profiler
- 在线样本分析
- 自动调参
- 分布式训练
- 复杂 curriculum
- 回测联动

---

## 0. 给 coder 的确定性 prompt

你要实现的是**训练模块**，用于把 assembled 数据读入、适配成 network-facing batch、送入 `MultiScaleForecastNetwork`，并完成 train/val 两个阶段的单机单卡训练。

你必须严格按本规范实现，不要自行加入额外框架，不要在当前版本中引入可视化系统，不要把日志采集做得过重，不要把 compile / cudagraph 逻辑写得过碎。

### 必须实现的东西

1. 一个 **assembled 数据集**，从 `.npz` 文件读取样本
2. 一个 **batch adapter / collate**，把数据转成 network-facing contract
3. 一个 **高效 DataLoader 构建器**
4. 一个 **Trainer** 类
5. 支持：
   - train epoch
   - val epoch
   - checkpoint save/load
   - compile(max-autotune)
   - AMP
   - grad clip
   - Rich 控制台进度条与日志
6. 每个 epoch 内只有 **一个 step 进度条**，train 和 val 阶段在同一个条上覆盖更新
7. ETA 的估计方式固定为：

\[
\text{ETA}
=
\frac{t_{\text{now}} - t_{\text{last\_log}}}{\Delta \text{logged\_steps}}
\times
(\text{steps\_remaining})
\]

也就是：

- 以两次 log 之间的时间差
- 除以两次 log 之间跨过的 step 数
- 估算平均 step 时间
- 再乘剩余 step 数

### 明确禁止的东西

- 不要上来就做 DDP / FSDP
- 不要在 dataloader 里做复杂在线特征计算
- 不要在每个 step 都把大量 tensor 拉回 CPU
- 不要在每个 step 都刷新富文本大表格
- 不要在 `forward()` 里迁移模型 device
- 不要在训练循环里频繁 `torch.cuda.synchronize()`
- 不要让 DataLoader 产出不定长 batch
- 不要在当前版本做动态图形可视化
- 不要在当前版本引入复杂 callback 系统

---

## 1. 当前 network-facing batch 契约（必须对齐，不允许偏离）

训练模块必须严格对齐当前 `MultiScaleForecastNetwork` 的输入契约。  
也就是说，DataLoader 最终给到 network 的 batch 必须包含：

```python
{
    "macro_float_long": torch.Tensor,   # [B, 11, 112], float32
    "macro_i8_long": torch.Tensor,      # [B, 2, 112], int8/int64

    "mezzo_float_long": torch.Tensor,   # [B, 11, 144], float32
    "mezzo_i8_long": torch.Tensor,      # [B, 2, 144], int8/int64

    "micro_float_long": torch.Tensor,   # [B, 11, 192], float32
    "micro_i8_long": torch.Tensor,      # [B, 2, 192], int8/int64

    "sidechain_cond": torch.Tensor,     # [B, 8, 64], float32

    "target_ret": torch.Tensor,         # [B, 1], float32
    "target_rv": torch.Tensor,          # [B, 1], float32
    "target_q": torch.Tensor,           # [B, 1], float32
}
```

当前 train 模块**必须以这个契约作为唯一真相**。  
不要直接把旧版 `DATA_SPEC` 里残留的冲突 shape 往 network 里塞。当前 network 装配文档已经明确裁决：train/data 侧必须做 adapter，把 assembled 样本转成上述 network-facing batch。

---

## 2. 目录与文件结构（必须按此落位）

```text
src/
    train/
        __init__.py
        dataset.py
        collate.py
        dataloaders.py
        runtime.py
        trainer.py
        checkpoint.py
        console.py
        train_entry.py
```

### 2.1 `src/train/dataset.py`

必须包含以下类：

- `AssembledNPZDataset`

### 2.2 `src/train/collate.py`

必须包含以下函数：

- `collate_network_batch`

### 2.3 `src/train/dataloaders.py`

必须包含以下函数：

- `build_train_dataloader`
- `build_val_dataloader`

### 2.4 `src/train/runtime.py`

必须包含以下函数：

- `build_model`
- `maybe_compile_model`
- `build_optimizer`
- `build_scheduler`
- `move_batch_to_device`

### 2.5 `src/train/checkpoint.py`

必须包含以下函数：

- `save_checkpoint`
- `load_checkpoint`

### 2.6 `src/train/console.py`

必须包含以下类：

- `EpochConsoleLogger`

### 2.7 `src/train/trainer.py`

必须包含以下类：

- `Trainer`

### 2.8 `src/train/train_entry.py`

必须包含以下函数：

- `run_training`

### 2.9 `src/train/__init__.py`

必须导出：

```python
from .trainer import Trainer
from .train_entry import run_training
```

---

## 3. 数据读取策略（必须定死）

## 3.1 数据源

当前训练只从：

- `data/assembled/`

读取 `.npz` 文件。  
允许：

- 单文件：`<code>.npz`
- 分片：`<code>__000.npz`, `__001.npz`, ...`

## 3.2 当前版本只做 NPZ 读取，不直接读 processed parquet

原因：

- train 目标是先跑起来
- assembled 已经是面向样本级消费的数据层
- 当前版本不把 processed parquet 在线拼装放进 dataloader

## 3.3 Dataset 的最小职责

`AssembledNPZDataset` 只负责：

1. 扫描训练文件列表
2. 建立 shard/sample 索引
3. 按索引读取单样本
4. 返回“原始 assembled sample dict”

**不要**在 `Dataset.__getitem__` 里做复杂的 batch 拼接。  
**不要**在 `__getitem__` 里做大量 numpy->torch->numpy 往返。

---

## 4. 关于 assembled 到 network-facing batch 的 adapter（必须明确）

当前上传文件里 `DATA_SPEC` 仍有历史残留冲突，当前 train 必须新增一个**明确 adapter 层**。

### 4.1 adapter 的职责

把 assembled sample 中的原始键，统一映射成 network 所需键。

### 4.2 当前 adapter 的目标输出（定死）

最终样本必须整理为：

```python
{
    "macro_float_long": np.ndarray,   # [11, 112], float32
    "macro_i8_long": np.ndarray,      # [2, 112], int64/int8

    "mezzo_float_long": np.ndarray,   # [11, 144], float32
    "mezzo_i8_long": np.ndarray,      # [2, 144], int64/int8

    "micro_float_long": np.ndarray,   # [11, 192], float32
    "micro_i8_long": np.ndarray,      # [2, 192], int64/int8

    "sidechain_cond": np.ndarray,     # [8, 64], float32

    "target_ret": np.ndarray,         # [1], float32
    "target_rv": np.ndarray,          # [1], float32
    "target_q": np.ndarray,           # [1], float32
}
```

### 4.3 adapter 必须拆成独立私有函数

在 `dataset.py` 内至少拆出：

- `_adapt_macro(...)`
- `_adapt_mezzo(...)`
- `_adapt_micro(...)`
- `_adapt_sidechain(...)`
- `_adapt_targets(...)`

### 4.4 当前不在 spec 里写死 assembled 原始键名

原因：

- 当前上传的 `DATA_SPEC` 仍混有旧版说明
- train 规范只锁定 network-facing contract
- coder 必须根据当前 assembled 实际 `.npz` 键，写一个集中 adapter
- 一旦 assembled 键名后续统一，train 侧只需要改 adapter，不动 network

---

## 5. Dataset 设计（必须定死）

## 5.1 `AssembledNPZDataset` 接口

```python
class AssembledNPZDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_paths: list[str],
        mmap_mode: str | None = None,
        validate_shapes: bool = True,
    ) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        ...
```

## 5.2 建索引方式

初始化时必须：

1. 扫描所有 `.npz`
2. 读取每个文件样本数
3. 建立一个全局 sample index：
   - `global_index -> (file_id, local_index)`

不要每次 `__getitem__` 都重新统计文件长度。

## 5.3 文件打开策略

当前版本建议：

- 每次 `__getitem__` 时按需 `np.load(file, mmap_mode=mmap_mode)` 读取
- worker 进程各自维护自己的文件句柄缓存字典

也就是：
- 不在主进程预先打开所有 NPZ 再 fork
- 避免多进程共享句柄问题
- 每个 worker 内部可做简单 LRU 或 dict cache

### 5.3.1 必须允许 `mmap_mode="r"`
默认建议：

```python
mmap_mode = "r"
```

用于减少大文件重复读开销。

## 5.4 Dataset 输出类型

`__getitem__` 返回 **numpy dict**，不要在这里转 torch。  
torch tensor 转换放到 `collate_network_batch` 中统一做。

---

## 6. Collate 设计（必须定死）

## 6.1 函数接口

```python
def collate_network_batch(samples: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    ...
```

## 6.2 职责

只负责：

1. 将同名 numpy 数组 stack 成 batch
2. 转成 torch tensor
3. 设定正确 dtype
4. 返回 network-facing batch

## 6.3 dtype 规则（必须定死）

### float 类
以下键必须转成 `torch.float32`：

- `macro_float_long`
- `mezzo_float_long`
- `micro_float_long`
- `sidechain_cond`
- `target_ret`
- `target_rv`
- `target_q`

### int 类
以下键必须转成 `torch.int64`：

- `macro_i8_long`
- `mezzo_i8_long`
- `micro_i8_long`

理由：
- network 内会直接 `.long()`
- collate 阶段统一成 `int64` 更简单稳妥

## 6.4 collate 中的形状检查

当前必须检查：

- `macro_float_long.shape == (11, 112)`
- `macro_i8_long.shape == (2, 112)`
- `mezzo_float_long.shape == (11, 144)`
- `mezzo_i8_long.shape == (2, 144)`
- `micro_float_long.shape == (11, 192)`
- `micro_i8_long.shape == (2, 192)`
- `sidechain_cond.shape == (8, 64)`
- `target_ret.shape == (1,)`
- `target_rv.shape == (1,)`
- `target_q.shape == (1,)`

若任一样本不满足，必须报 `ValueError`。

---

## 7. DataLoader 设计（必须定死）

## 7.1 总体目标

DataLoader 必须优先保证：

- 多进程预取
- pin memory
- persistent workers
- 固定 batch shape
- 减少 IO 抖动

## 7.2 train dataloader 配置（默认推荐）

```python
batch_size = ...
shuffle = True
drop_last = True
num_workers = ...
pin_memory = True
persistent_workers = (num_workers > 0)
prefetch_factor = 4
collate_fn = collate_network_batch
```

### 关键要求：`drop_last=True`
当前训练必须固定：

- `train_loader.drop_last = True`

原因：
1. `torch.compile(mode="max-autotune")` 更喜欢稳定 shape
2. 若未来开启 cudagraph，固定 batch shape 是硬要求
3. 避免最后一个小 batch 导致图碎 / 重新编译 / graph break

## 7.3 val dataloader 配置（默认推荐）

```python
shuffle = False
drop_last = False
num_workers = ...
pin_memory = True
persistent_workers = (num_workers > 0)
prefetch_factor = 2
collate_fn = collate_network_batch
```

### 为什么 val 不强制 `drop_last=True`
因为 val 不做 backward，也不需要最严格的 shape 稳定。  
当前先让验证全量覆盖样本。

---

## 8. 运行时与 compile 设计（必须定死）

## 8.1 `build_model()`

必须返回：

- `MultiScaleForecastNetwork`

不要在这里自动 compile。

## 8.2 `maybe_compile_model()`

接口建议：

```python
def maybe_compile_model(
    model: torch.nn.Module,
    enabled: bool,
    mode: str = "max-autotune",
) -> torch.nn.Module:
    ...
```

### 当前 compile 规则（定死）

若 `enabled=True`，必须：

```python
model = torch.compile(model, mode="max-autotune")
```

不要改成别的 mode。

## 8.3 关于 cudagraph 的明确约束（必须写死）

当前要求里已经明确提到：

- 要用标准 `max-autotune`
- 要注意 cudagraph 的要求
- 图不要太碎
- 注意 `drop_last`

因此当前训练模块必须遵守以下规则：

### 8.3.1 不主动在 trainer 里手写 cudagraph capture
当前版本：
- **不自己手写 `torch.cuda.CUDAGraph`**
- 只允许依赖 `torch.compile(mode="max-autotune")` 的默认 runtime 优化

原因：
- 当前总目标是先跑起来
- 手写 cudagraph 会显著增加复杂度
- 容易和网络中仍未完全静态化的路径打架

### 8.3.2 但必须为 compile/cudagraph 做准备
为避免 graph break / 反复编译，必须做到：

1. train batch 固定 shape
2. `drop_last=True`
3. 不在 `forward()` 中返回巨量 aux（默认 `return_aux=False`）
4. 不在 train step 中插入频繁 CPU 读取
5. 不在 step 中做动态分支过多的 debug 逻辑
6. 不在 dataloader 输出不规则 batch

### 8.3.3 不要让图太碎
当前版本必须：
- 每个 step 只执行一个完整的 `forward_loss`
- 不要把 forward 拆成大量独立小 compile 单元
- 不要对每个子模块单独 compile

也就是：
- **compile 整个 network 主体**
- 不 compile 单个局部模块

原因：
- 太碎会导致 `max-autotune` 编译时间过长
- 也会让 cudagraph 价值下降

---

## 9. 优化器 / AMP / 梯度处理（必须定死）

## 9.1 优化器

当前默认：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
)
```

不要在当前版本引入复杂 optimizer。

## 9.2 AMP

当前默认：

- CUDA 下开启 `torch.autocast(device_type="cuda", dtype=torch.float16 or bfloat16)`
- 使用 `torch.cuda.amp.GradScaler`（如果 dtype 是 float16）
- 若使用 bfloat16，可不使用 scaler

### 推荐策略
如果硬件支持良好：
- 优先 `bfloat16`
- 否则 `float16 + GradScaler`

## 9.3 梯度裁剪

当前必须支持：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

默认建议：
```python
max_grad_norm = 1.0
```

---

## 10. Trainer 设计（必须定死）

## 10.1 `Trainer` 类接口

```python
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader,
        val_loader,
        device: torch.device,
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        max_grad_norm: float = 1.0,
        log_every: int = 50,
        console_logger = None,
    ) -> None:
        ...

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        ...

    def validate_one_epoch(self, epoch: int) -> dict[str, float]:
        ...

    def fit(self, num_epochs: int) -> None:
        ...
```

## 10.2 step 内部必须调用 `forward_loss()`

训练 step 当前必须统一调用：

```python
out = model.forward_loss(batch, return_aux=False)
loss = out["loss_total"]
```

不要自己在 trainer 内重复拼 loss。

## 10.3 `move_batch_to_device()`

必须在 `runtime.py` 中集中实现：

```python
def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    ...
```

要求：

- `non_blocking=True`
- 只搬运 tensor
- 不修改键名
- 不在这里做 dtype 改写（除非是极少数必须项；当前默认不做）

---

## 11. console logging（必须定死）

## 11.1 当前只允许 Rich 控制台日志

不接：

- TensorBoard
- MLflow
- WandB
- matplotlib

当前只用控制台 Rich。

## 11.2 一个 epoch 内只允许一个 step 进度条

也就是：

- train 阶段开始时创建一个条
- val 阶段复用同一个条
- 不要 train 和 val 各开一个条长期并存

### 推荐实现
使用 `rich.progress.Progress`，在同一个 task 上更新：

- description：
  - `train`
  - `val`
- total：
  - 当前阶段 steps 总数
- completed：
  - 当前阶段 step

进入 val 时：
- reset 同一个 task
- 改 description 为 `val`

## 11.3 日志刷新的频率（必须定死）

只有在：

```python
global_step % log_every == 0
```

时才做一次日志采集。

### 关键要求
每 `log_every` 次，才允许：
- 从 GPU 抽取 loss 标量
- 更新 ETA
- 打印控制台信息

其他 step：
- 不做频繁 `.item()`
- 不做频繁 CPU-GPU 同步
- 不做大量 metric 聚合输出

## 11.4 ETA 计算方式（必须定死）

设：

- 上一次 log 发生在 step `s_prev`，时间 `t_prev`
- 当前 log 发生在 step `s_now`，时间 `t_now`

定义：

\[
\Delta s = s_{now} - s_{prev}
\]
\[
\Delta t = t_{now} - t_{prev}
\]

平均 step 时间：

\[
\bar{t}_{step} = \frac{\Delta t}{\Delta s}
\]

剩余 step 数：

\[
s_{remain} = s_{total} - s_{now}
\]

则：

\[
\text{ETA} = \bar{t}_{step} \cdot s_{remain}
\]

当前必须按这个方式计算，不要使用更复杂的 EMA。

## 11.5 Rich 日志内容（当前最小集合）

每次 log 只输出最必要信息：

- phase: `train` / `val`
- epoch
- step / total_steps
- `loss_total`
- `loss_ret`
- `loss_rv`
- `loss_q`
- learning rate
- ETA

不要输出大表格，不要输出中间层统计，不要输出大量辅助键。

---

## 12. checkpoint 设计（必须定死）

## 12.1 保存内容

必须保存：

- `model.state_dict()`
- `optimizer.state_dict()`
- `scheduler.state_dict()`（若存在）
- `scaler.state_dict()`（若存在）
- `epoch`
- `global_step`

## 12.2 保存时机

当前至少支持：

- 每 epoch 保存一次 latest
- val loss 最优时保存一次 best

### 文件名建议
- `latest.pt`
- `best.pt`

## 12.3 恢复要求

`load_checkpoint()` 必须支持：

- 仅加载模型
- 完整恢复训练状态

---

## 13. train loop（必须定死）

## 13.1 train step

每个 train step 固定顺序：

```text
1. 取 batch
2. move_batch_to_device(non_blocking=True)
3. autocast
4. model.forward_loss(batch, return_aux=False)
5. backward
6. grad clip
7. optimizer.step
8. scaler.update (if needed)
9. optimizer.zero_grad(set_to_none=True)
10. scheduler.step (if per-step)
```

### 细节要求

#### zero grad
必须用：

```python
optimizer.zero_grad(set_to_none=True)
```

#### backward
- float16 下通过 `GradScaler`
- bfloat16 下可直接 backward

## 13.2 val step

每个 val step 固定顺序：

```text
1. 取 batch
2. move_batch_to_device(non_blocking=True)
3. autocast
4. model.forward_loss(batch, return_aux=False)
5. 只读 loss，不 backward
```

---

## 14. 开放参数与隐参数（train 级）

### 14.1 开放参数（放到 `config/models.py`）

- `batch_size`
- `val_batch_size`
- `num_workers`
- `lr`
- `weight_decay`
- `max_grad_norm`
- `log_every`
- `num_epochs`
- `compile_model`
- `amp_dtype`

### 14.2 隐参数（放到 `src/models/config/hparams.py`）

- `_pin_memory = True`
- `_prefetch_factor_train = 4`
- `_prefetch_factor_val = 2`
- `_persistent_workers = True`
- `_drop_last_train = True`
- `_drop_last_val = False`
- `_compile_mode = "max-autotune"`

---

## 15. 默认值（当前推荐）

```python
batch_size = 256
val_batch_size = 256
num_workers = 8
lr = 3e-4
weight_decay = 1e-4
max_grad_norm = 1.0
log_every = 50
num_epochs = 20
compile_model = True
amp_dtype = "bfloat16"
```

隐参数默认：

```python
_pin_memory = True
_prefetch_factor_train = 4
_prefetch_factor_val = 2
_persistent_workers = True
_drop_last_train = True
_drop_last_val = False
_compile_mode = "max-autotune"
```

---

## 16. smoke test 要求

测试文件建议放在：

```text
tests/train/test_trainer_smoke.py
```

至少包含以下测试：

### 测试 1：dataset 可读取
给一个最小 `.npz` 文件集合，验证：

- `len(dataset) > 0`
- `dataset[0]` 返回 dict
- 包含 adapter 后所需键

### 测试 2：collate 后 shape 正确
验证 batch 输出 shape 精确等于 network-facing contract。

### 测试 3：train loader drop_last=True
验证 train loader 的 `drop_last` 正确。

### 测试 4：整链 train step 可运行
用一个极小网络和极小 batch，验证：
- forward_loss 正常
- backward 正常
- optimizer.step 正常

### 测试 5：val step 可运行
验证验证阶段不 backward，loss 可正常返回。

### 测试 6：compile 路径可运行
在可用环境下验证：
- `torch.compile(mode="max-autotune")` 路径能跑通

### 测试 7：checkpoint save/load 正常
验证：
- latest 保存成功
- load 后 epoch/global_step 恢复正确

---

## 17. 验收标准

实现完成后，以下条件必须同时满足：

1. train 侧严格对齐当前 network-facing batch contract
2. assembled -> adapter -> collate -> network 全链打通
3. DataLoader 支持多进程预取、pin memory、persistent workers
4. train loader 使用 `drop_last=True`
5. `torch.compile(mode="max-autotune")` 可选启用
6. 不手写 cudagraph，但训练路径满足其静态 shape 约束
7. Rich 控制台日志按 `log_every` 更新
8. 只在 log step 获取 GPU 标量
9. train / val / checkpoint 全部可运行
10. 当前版本不包含任何数据可视化系统

---

## 18. 最终一句话要求

你要实现的不是一个“能凑合跑两步”的脚本，而是一个：

> **围绕当前 network-facing batch contract，具备高效多进程数据加载、稳定 compile/max-autotune 运行时、低开销 Rich 控制台日志，以及可正常 train/val/save/load 的最小可用训练模块。**

除此之外，什么都不要做。
