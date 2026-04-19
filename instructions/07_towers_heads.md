# Multi-Task Towers and Heads 实现规范（最终版，给 coder 的确定性说明）

本模块的总规则：

- 参数分为两种，一种是**隐参数**，放在 `models/hparams.yaml` 或 `hparams.py`
- 另一种是**开放参数**，放在 `config/models.py`
- 隐参数统一使用 `_` 开头
- 开放参数不使用 `_` 开头
- 本文档只定义 **multi-task towers + heads**
- 本文档不定义 trunk、encoder、fusion、loss
- 本文档当前固定三任务：
  1. `ret`：预计时间平均收益率
  2. `rv`：波动率
  3. `q`：`p_xx` 分位数收益（风险标签）

当前已确定的结构原则如下：

1. 使用 **shared trunk**，其输出为：
   - `fused_latents: [B, hidden_dim, num_latents]`
   - `fused_global: [B, hidden_dim]`
2. 三个任务使用 **三个独立的 task query towers**
3. 每个 tower：
   - 具有一个 task-specific learned token
   - 使用 `fused_global` 进行 task-specific 条件注入
   - 通过一次 cross-attention 读取 `fused_latents`
   - 再通过一个小 FFN 做 task-specific refinement
4. 每个任务都有 **两个独立 head**：
   - 一个 value head
   - 一个 uncertainty head
5. 输出必须严格对齐 `multi_task_loss_spec.md` 的输入接口

---

## 0. 给 coder 的确定性 prompt

你要实现的是三任务预测的 **task-specific towers and heads**。

你必须严格按本规范实现，不要自行改成共享 head，不要把 tower 简化成只有一层线性，不要更改输出字段，不要改变任务语义。

### 必须实现的东西

1. 一个 `TaskQueryTower` 类
2. 一个 `MultiTaskHeads` 类
3. 三个 tower：
   - `ret_tower`
   - `rv_tower`
   - `q_tower`
4. 六个 head：
   - `ret_value_head`
   - `ret_uncertainty_head`
   - `rv_value_head`
   - `rv_uncertainty_head`
   - `q_value_head`
   - `q_uncertainty_head`
5. 输入：
   - `fused_latents [B, hidden_dim, num_latents]`
   - `fused_global [B, hidden_dim]`
6. 输出一个字典，包含：
   - `pred_mu_ret`
   - `pred_scale_ret_raw`
   - `pred_mean_rv_raw`
   - `pred_shape_rv_raw`
   - `pred_mu_q`
   - `pred_scale_q_raw`

### 明确禁止的东西

- 不要把三个任务合并成一个共享 tower
- 不要只用 `fused_global` 做预测，忽略 `fused_latents`
- 不要只用 `fused_latents.mean()` 替代 task query 机制
- 不要把三个任务的 head 共享参数
- 不要让 `ret / rv / q` 共用同一个 task token
- 不要在这里加入 loss
- 不要在这里加入 `softplus` 正值化
- 不要在这里加入 `nu_ret`
- 不要在这里加入 side / conditioning 分支
- 不要在这里加入 cross-scale fusion
- 不要接 prediction label
- 不要把 head 写成完整大 backbone

这个模块唯一职责：

> 读取 trunk 融合后的 `fused_latents` 和 `fused_global`，通过三个 task-specific query towers 形成三个任务表示，再分别输出六个 raw prediction heads，对接后续分布式 loss。

---

## 1. 本规范依赖的上游接口（按当前已定规格写死）

### 1.1 上游输入

本模块当前只消费 `CrossScaleFusion` 的输出：

- `fused_latents: [B, hidden_dim, num_latents]`
- `fused_global: [B, hidden_dim]`

当前默认值为：

- `hidden_dim = 128`
- `num_latents = 8`

### 1.2 下游 loss 对接要求

本模块输出必须严格对齐 `MultiTaskDistributionLoss` 所需输入：

- `pred_mu_ret: [B, 1]`
- `pred_scale_ret_raw: [B, 1]`
- `pred_mean_rv_raw: [B, 1]`
- `pred_shape_rv_raw: [B, 1]`
- `pred_mu_q: [B, 1]`
- `pred_scale_q_raw: [B, 1]`

注意：

- 所有 uncertainty 相关输出此处都是 **raw**
- 正值化在 loss 模块内部做
- `nu_ret` 由 loss 模块内部持有的 task-level 全局参数决定，不在 head 中输出

---

## 2. 模块职责边界

### 2.1 本模块负责的事情

本模块只负责：

1. 把 `fused_latents` 和 `fused_global` 转成三个任务专属表示
2. 为每个任务输出：
   - 一个中心预测 raw 值
   - 一个 uncertainty raw 值
3. 返回统一字典供 loss 使用

### 2.2 本模块不负责的事情

以下事情不属于本模块：

- trunk 编码
- within-scale fusion
- side bridge fusion
- cross-scale fusion
- loss 计算
- target 变换
- 正值参数化
- task-level `nu_ret` 参数

---

## 3. 设计决策（全部定死）

### 3.1 三个任务使用三个独立的 task query towers

三个任务语义不同：

- `ret`：收益
- `rv`：波动
- `q`：风险 / 分位数收益

因此：

- tower 不共享
- task token 不共享
- head 不共享

### 3.2 每个 tower 的输入相同，但读法不同

每个 tower 都读取：

- `fused_latents`
- `fused_global`

但由于：

- task token 不同
- global projection 不同
- cross-attention 权重不同
- FFN 权重不同

所以每个任务会形成不同的任务表征。

### 3.3 每个 tower 固定只做一层 task query block

当前默认与推荐：

- 1 个 task token
- 1 次 cross-attention
- 1 个 FFN
- 不做多层堆叠

不要加深。

### 3.4 每个任务两个 head

每个任务必须有：

- value head
- uncertainty head

都从该任务自己的 tower 输出上计算。

---

## 4. 文件结构（必须按此落位）

```text
src/models/
    heads/
        __init__.py
        task_query_tower.py
        multi_task_heads.py
```

### 4.1 `src/models/heads/task_query_tower.py`

必须包含以下类：

- `TaskQueryTower`

### 4.2 `src/models/heads/multi_task_heads.py`

必须包含以下类：

- `MultiTaskHeads`

### 4.3 `src/models/heads/__init__.py`

必须导出：

```python
from .task_query_tower import TaskQueryTower
from .multi_task_heads import MultiTaskHeads
```

---

## 5. 参数分类（必须明确）

### 5.1 开放参数（放到 `config/models.py`）

这些参数是未来可能调的，不加 `_` 前缀：

- `hidden_dim`
- `num_latents`
- `tower_num_heads`
- `tower_ffn_ratio`
- `tower_dropout`

### 5.2 隐参数（放到 `models/hparams.yaml` 或 `hparams.py`）

这些参数是不希望频繁调的，统一加 `_` 前缀：

- `_tower_norm_eps`

### 5.3 当前默认值

#### 开放参数默认值
```python
hidden_dim = 128
num_latents = 8
tower_num_heads = 4
tower_ffn_ratio = 2.0
tower_dropout = 0.0
```

#### 隐参数默认值
```python
_tower_norm_eps = 1e-6
```

---

## 6. 类接口定义（必须完全一致）

## 6.1 `TaskQueryTower`

文件：`src/models/heads/task_query_tower.py`

```python
class TaskQueryTower(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        fused_latents: torch.Tensor,
        fused_global: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入

- `fused_latents: [B, hidden_dim, num_latents]`
- `fused_global: [B, hidden_dim]`

### 输出

- `task_repr: [B, hidden_dim]`

---

## 6.2 `MultiTaskHeads`

文件：`src/models/heads/multi_task_heads.py`

```python
class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_latents: int = 8,
        tower_num_heads: int = 4,
        tower_ffn_ratio: float = 2.0,
        tower_dropout: float = 0.0,
        _tower_norm_eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        fused_latents: torch.Tensor,
        fused_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...
```

### 输入

- `fused_latents: [B, hidden_dim, num_latents]`
- `fused_global: [B, hidden_dim]`

### 输出字典（必须定死）

```python
{
    "pred_mu_ret": ...,
    "pred_scale_ret_raw": ...,
    "pred_mean_rv_raw": ...,
    "pred_shape_rv_raw": ...,
    "pred_mu_q": ...,
    "pred_scale_q_raw": ...,
}
```

所有值的 shape 都必须为：

```text
[B, 1]
```

---

## 7. `TaskQueryTower` 的结构（必须定死）

### 7.1 总体流程

每个 task tower 必须严格按下面顺序实现：

```text
fused_latents [B, D, L]
-> transpose to [B, L, D]

fused_global [B, D]
-> task-specific global projection
-> add to task token

task token
-> cross-attend fused_latents
-> residual add

task token
-> FFN
-> residual add

squeeze token dimension
-> task_repr [B, D]
```

### 7.2 具体实现要求

#### 第 1 步：输入变换

将：

```python
fused_latents: [B, D, L]
```

转为：

```python
latents = fused_latents.transpose(1, 2)   # [B, L, D]
```

其中：

- `L = num_latents`

#### 第 2 步：定义 task token

每个 `TaskQueryTower` 必须持有一个 learnable task token：

```python
self.task_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
```

不要让多个任务共享同一个 task token。

#### 第 3 步：global conditioning 注入

定义：

```python
self.global_proj = nn.Linear(hidden_dim, hidden_dim)
```

然后：

```python
q0 = self.task_token.expand(B, -1, -1) + self.global_proj(fused_global).unsqueeze(1)
```

得到：

```text
q0: [B, 1, D]
```

#### 第 4 步：task token 读取 fused_latents

使用：

```python
self.q_norm = nn.LayerNorm(hidden_dim, eps=_norm_eps)
self.kv_norm = nn.LayerNorm(hidden_dim, eps=_norm_eps)
self.cross_attn = nn.MultiheadAttention(
    embed_dim=hidden_dim,
    num_heads=num_heads,
    dropout=dropout,
    batch_first=True,
)
```

然后：

```python
q = self.q_norm(q0)
kv = self.kv_norm(latents)
delta_attn, _ = self.cross_attn(q, kv, kv, need_weights=False)
q1 = q0 + delta_attn
```

得到：

```text
q1: [B, 1, D]
```

#### 第 5 步：FFN refinement

定义：

```python
_ffn_dim = int(hidden_dim * ffn_ratio)

self.ffn_norm = nn.LayerNorm(hidden_dim, eps=_norm_eps)
self.ffn = nn.Sequential(
    nn.Linear(hidden_dim, _ffn_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(_ffn_dim, hidden_dim),
    nn.Dropout(dropout),
)
```

然后：

```python
q2 = q1 + self.ffn(self.ffn_norm(q1))
```

#### 第 6 步：输出 task representation

```python
task_repr = q2.squeeze(1)   # [B, D]
```

返回：

```python
return task_repr
```

### 7.3 明确禁止的替代实现

- 不要把 tower 改成只用 `fused_global`
- 不要把 tower 改成只用 `fused_latents.mean()`
- 不要把 tower 改成 MLP(concat(mean_latents, global))
- 不要把 tower 改成多 token query
- 不要把 tower 改成 self-attention over latents
- 不要在 tower 内加 side / conditioning 输入

---

## 8. `MultiTaskHeads` 的结构（必须定死）

### 8.1 必须包含三个独立 tower

必须定义：

- `self.ret_tower = TaskQueryTower(...)`
- `self.rv_tower = TaskQueryTower(...)`
- `self.q_tower = TaskQueryTower(...)`

不要共享。

### 8.2 必须包含六个独立 head

每个 head 都必须是：

```python
nn.Sequential(
    nn.LayerNorm(hidden_dim, eps=_tower_norm_eps),
    nn.Linear(hidden_dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(tower_dropout),
    nn.Linear(hidden_dim, 1),
)
```

具体必须定义为：

- `self.ret_value_head`
- `self.ret_uncertainty_head`
- `self.rv_value_head`
- `self.rv_uncertainty_head`
- `self.q_value_head`
- `self.q_uncertainty_head`

### 8.3 forward 过程（逐步，不可歧义）

#### 第 1 步：三个 tower 分别生成任务表示

```python
h_ret = self.ret_tower(fused_latents, fused_global)   # [B, D]
h_rv  = self.rv_tower(fused_latents, fused_global)    # [B, D]
h_q   = self.q_tower(fused_latents, fused_global)     # [B, D]
```

#### 第 2 步：各自进入独立 head

```python
pred_mu_ret         = self.ret_value_head(h_ret)
pred_scale_ret_raw  = self.ret_uncertainty_head(h_ret)

pred_mean_rv_raw    = self.rv_value_head(h_rv)
pred_shape_rv_raw   = self.rv_uncertainty_head(h_rv)

pred_mu_q           = self.q_value_head(h_q)
pred_scale_q_raw    = self.q_uncertainty_head(h_q)
```

#### 第 3 步：返回字典

必须返回：

```python
{
    "pred_mu_ret": pred_mu_ret,
    "pred_scale_ret_raw": pred_scale_ret_raw,
    "pred_mean_rv_raw": pred_mean_rv_raw,
    "pred_shape_rv_raw": pred_shape_rv_raw,
    "pred_mu_q": pred_mu_q,
    "pred_scale_q_raw": pred_scale_q_raw,
}
```

---

## 9. 输入输出与 loss 的严格对齐（必须写死）

`MultiTaskHeads.forward()` 的输出必须可以**直接**送入 `MultiTaskDistributionLoss.forward()`。

### 对应关系

- `pred_mu_ret` → `mu_ret`
- `pred_scale_ret_raw` → `scale_ret_raw`

- `pred_mean_rv_raw` → `mean_rv_raw`
- `pred_shape_rv_raw` → `shape_rv_raw`

- `pred_mu_q` → `mu_q`
- `pred_scale_q_raw` → `scale_q_raw`

### 重要说明

这里全部输出 **raw** 参数，不在 head 中做：

- `softplus`
- `exp`
- `clamp`
- `nu_ret`

这些都在 loss 模块内部处理。

---

## 10. 初始化与 dtype/device 要求

### 10.1 初始化

- `Linear`、`MultiheadAttention` 使用 PyTorch 默认初始化
- 每个 `task_token` 初始化为全零：

```python
nn.Parameter(torch.zeros(1, 1, hidden_dim))
```

### 10.2 dtype/device

必须满足：

- 所有输出与输入 `fused_latents` 保持同 dtype/device
- 不要在 `forward()` 里调用 `self.to(...)`

---

## 11. 错误处理要求

以下情况必须报 `ValueError`：

1. `hidden_dim <= 0`
2. `num_latents <= 0`
3. `tower_num_heads <= 0`
4. `hidden_dim % tower_num_heads != 0`
5. `tower_ffn_ratio <= 0`
6. `tower_dropout < 0` 或 `tower_dropout >= 1`
7. `_tower_norm_eps <= 0`
8. `fused_latents.ndim != 3`
9. `fused_global.ndim != 2`
10. `fused_latents.shape[1] != hidden_dim`
11. `fused_latents.shape[2] != num_latents`
12. `fused_global.shape[1] != hidden_dim`
13. batch 不一致

错误信息必须指出：

- 哪个值非法
- 当前值是什么
- 合法范围是什么

---

## 12. smoke test 要求

测试文件建议放在：

```text
tests/models/heads/test_multi_task_heads.py
```

至少包含以下测试：

### 测试 1：forward 可运行
```python
fused_latents = torch.randn(2, 128, 8)
fused_global = torch.randn(2, 128)

model = MultiTaskHeads()
out = model(fused_latents, fused_global)

assert out["pred_mu_ret"].shape == (2, 1)
assert out["pred_scale_ret_raw"].shape == (2, 1)
assert out["pred_mean_rv_raw"].shape == (2, 1)
assert out["pred_shape_rv_raw"].shape == (2, 1)
assert out["pred_mu_q"].shape == (2, 1)
assert out["pred_scale_q_raw"].shape == (2, 1)
```

### 测试 2：三个 tower 不共享 task token
验证：
- `ret_tower.task_token is not rv_tower.task_token`
- `ret_tower.task_token is not q_tower.task_token`

### 测试 3：三个 value heads 不共享参数
验证：
- `ret_value_head` 与 `rv_value_head` 参数对象不同
- `ret_value_head` 与 `q_value_head` 参数对象不同

### 测试 4：dtype/device 一致
输入是什么 dtype/device，输出就必须是什么 dtype/device。

### 测试 5：非法 hidden_dim 报错
`fused_latents.shape[1] != hidden_dim` 时必须报 `ValueError`

### 测试 6：非法 num_latents 报错
`fused_latents.shape[2] != num_latents` 时必须报 `ValueError`

### 测试 7：batch 不一致报错
batch 不一致时必须报 `ValueError`

### 测试 8：梯度可回传
验证：
- `fused_latents.requires_grad_(True)` 后
- 所有六个 head 至少有梯度
- 三个 tower 的 task token 至少有梯度

---

## 13. 当前默认实例化方式（按已定规格写死）

```python
multi_task_heads = MultiTaskHeads(
    hidden_dim=128,
    num_latents=8,
    tower_num_heads=4,
    tower_ffn_ratio=2.0,
    tower_dropout=0.0,
)
```

典型调用方式：

```python
out = multi_task_heads(
    fused_latents=fused_latents,   # [B, 128, 8]
    fused_global=fused_global,     # [B, 128]
)
```

返回：

```python
{
    "pred_mu_ret": [B, 1],
    "pred_scale_ret_raw": [B, 1],
    "pred_mean_rv_raw": [B, 1],
    "pred_shape_rv_raw": [B, 1],
    "pred_mu_q": [B, 1],
    "pred_scale_q_raw": [B, 1],
}
```

---

## 14. 验收标准

实现完成后，以下条件必须同时满足：

1. 三个任务使用三个独立 `TaskQueryTower`
2. 每个 tower 都通过 task token + global conditioning + cross-attention 读取 `fused_latents`
3. 六个 head 都独立，不共享参数
4. 输出字段严格对齐 `MultiTaskDistributionLoss`
5. 不在 head 中做任何正值化
6. 不在 head 中持有 `nu_ret`
7. 所有 smoke test 通过

---

## 15. 最终一句话要求

你要实现的不是一个“共享一个向量后接六个线性层”的简化 head，而是一个：

> **对 `fused_latents` 和 `fused_global` 使用三个任务专属 query towers 读取不同融合信息，再分别输出 `ret / rv / q` 的 value raw 和 uncertainty raw 的多任务 towers + heads 模块。**

除此之外，什么都不要做。
