# WithinScaleSTARFusion 实现规范（最终版，给 coder 的确定性说明）

本模块的总规则：

- 参数分为两种，一种是**隐参数**，放在 `models/hparams.yaml` 或 `hparams.py`
- 另一种是**开放参数**，放在 `config/models.py`
- 隐参数统一使用 `_` 开头
- 开放参数不使用 `_` 开头
- 本模块只负责 **within-scale feature fusion**
- 本模块 **不消费** `ConditioningEncoder` 的输出
- 本模块当前只针对 `ModernTCNFiLMEncoder` 的输出接口实现
- 本模块采用 **STAR-style aggregate-redistribute** 机制
- **STAR 只沿 feature/channel 维做融合，不对时间/patch 维做任何压缩**

---

## 0. 给 coder 的确定性 prompt

你要实现一个 **WithinScaleSTARFusion**，用于对同一尺度内、已经被 `ModernTCNFiLMEncoder` **逐特征独立编码**后的隐藏表示做 **STAR 风格的晚融合**。

这里的 STAR 指 **STar Aggregate-Redistribute** 风格的中心化通道交互：

1. 在每个时间/patch 位置上，聚合同一尺度下的所有 feature 表示
2. 得到一个 centralized core
3. 将该 core 复制并回灌到每个 feature 表示中
4. 形成 feature 维的晚融合结果

你必须严格按本规范实现，不要自行补充结构，不要把它改成 attention，不要把 conditioning 分支并进来，不要把它写成完整模型。

### 必须实现的东西

1. 一个 `WithinScaleSTARFusion` 类
2. 输入 `z_scale`，形状 `[B, F, D, N]`
3. 输出两路：
   - `z_fused: [B, F, D, N]`
   - `scale_seq: [B, D, N]`
4. 内部使用 `STARAggregateRedistributeBlock`
5. 使用 `StochasticPooling1D` 做 centralized core aggregation
6. 只做同一尺度内的 feature fusion，不做任何 cross-scale fusion
7. 提供最小必要的 smoke test

### 明确禁止的东西

- 不要接 prediction head
- 不要接 cross-scale fusion
- 不要接 conditioning encoder 的输出
- 不要把 `cond_seq / cond_global` 拼进来
- 不要加 self-attention
- 不要加 cross-attention
- 不要加 patching
- 不要改写主干 encoder
- 不要做 feature 级早融合
- 不要把它做成完整 backbone
- **不要压缩时间/patch 维**
- **不要把 STAR 实现成时间聚合模块**

这个模块唯一职责：

> 输入 `[B, F, D, N]` 的单尺度已编码特征表示，沿 feature 维执行 STAR 风格集中式聚合-回灌融合，输出融合后的特征表示和尺度级序列表示。

---

## 1. 本规范依赖的上游接口（按已上传规格写死）

### 1.1 上游主干 encoder 输出

`ModernTCNFiLMEncoder` 的输出固定为：

```text
z: [B, 11, hidden_dim, num_patches]
```

也就是说，本模块当前只接收：

- `F = 11`
- `D = hidden_dim`
- `N = num_patches`

### 1.2 当前三个尺度的默认 patch 配置

按当前已定 `ModernTCNFiLMEncoder` 规格：

- `patch_len = 8`
- `patch_stride = 4`
- `hidden_dim = 128`

因此，在默认配置下：

#### Macro
- 输入来自 `x_float [B, 11, 64]`
- 编码输出 `z_macro [B, 11, 128, 16]`

#### Mezzo
- 输入来自 `x_float [B, 11, 96]`
- 编码输出 `z_mezzo [B, 11, 128, 24]`

#### Micro
- 输入来自 `x_float [B, 11, 144]`
- 编码输出 `z_micro [B, 11, 128, 36]`

### 1.3 上游 conditioning encoder 输出（本模块当前不消费）

`ConditioningEncoder` 的输出固定为：

- `cond_seq: [B, d_cond, 64]`
- `cond_global: [B, d_cond]`

本模块 **当前不得消费这两个输出**。  
side / conditioning 融合属于后续模块，不属于本模块职责。

---

## 2. 模块职责边界

### 2.1 本模块负责的事情

本模块只负责：

1. 接收某一尺度的已编码 feature 表示 `z_scale [B, F, D, N]`
2. 在每个 patch 位置上，对 feature 维做 STAR 风格 centralized interaction
3. 输出：
   - 融合后的 per-feature 表示 `z_fused [B, F, D, N]`
   - 尺度级序列表示 `scale_seq [B, D, N]`

### 2.2 本模块不负责的事情

以下事情不属于本模块：

- conditioning 融合
- side / regime 融合
- cross-scale fusion
- 最终 prediction head
- 时间维编码
- denoise
- label 预测
- loss 计算

---

## 3. 设计决策（全部定死）

### 3.1 当前采用的是 STAR 风格的 within-scale fusion

这里使用的是 **STAR-style aggregate-redistribute fusion**，而不是把 SOFTS 整体直接搬过来。

本规范保留 STAR 的核心机制：

1. centralized core aggregation
2. stochastic pooling
3. core repeat + concat
4. MLP fuse back to each feature
5. residual connection

但本规范只适配你的输入接口 `[B, F, D, N]`，不试图复现 SOFTS 的完整 forecasting 框架。

### 3.2 融合轴固定为 feature 维

所有 STAR 操作都只沿 `F` 维进行。

也就是说：

- 同一 patch 内 feature 与 feature 交互
- 不做时间维 attention
- 不做尺度间交互

### 3.3 时间/patch 维必须完整保留

本模块 **绝不对时间/patch 维做任何压缩**。

即：

- 输入是 `[B, F, D, N]`
- 输出 `z_fused` 仍然必须是 `[B, F, D, N]`
- `scale_seq` 必须是 `[B, D, N]`

这里的 `scale_seq` 只表示：

- feature 维被聚合掉了
- 时间/patch 维 `N` 仍然完整保留

### 3.4 尺度级表示的生成方式定死为 feature 维均值池化

在得到 `z_fused [B, F, D, N]` 后，尺度级表示固定为：

```python
scale_seq = z_fused.mean(dim=1)
```

输出：

```text
[B, D, N]
```

当前不要加 learnable pooling，不要加 attention pooling。

---

## 4. 文件结构（必须按此落位）

```text
src/models/
    layers/
        __init__.py
        stochastic_pooling1d.py

    fusions/
        __init__.py
        within_scale_star_fusion.py
```

### 4.1 `src/models/layers/stochastic_pooling1d.py`

必须包含以下类：

- `StochasticPooling1D`

### 4.2 `src/models/fusions/within_scale_star_fusion.py`

必须包含以下类：

- `STARAggregateRedistributeBlock`
- `WithinScaleSTARFusion`

### 4.3 `src/models/fusions/__init__.py`

只导出：

```python
from .within_scale_star_fusion import WithinScaleSTARFusion
```

### 4.4 `src/models/layers/__init__.py`

必须导出：

```python
from .stochastic_pooling1d import StochasticPooling1D
```

---

## 5. 参数分类（必须明确）

### 5.1 开放参数（放到 `config/models.py`）

这些参数是未来可能调的，不加 `_` 前缀：

- `hidden_dim`
- `num_features`
- `core_dim`
- `num_layers`
- `dropout`

### 5.2 隐参数（放到 `models/hparams.yaml` 或 `hparams.py`）

这些参数是不希望频繁调的，统一加 `_` 前缀：

- `_norm_eps`
- `_pool_temperature`

### 5.3 当前默认值

#### 开放参数默认值
```python
hidden_dim = 128
num_features = 11
core_dim = 64
num_layers = 1
dropout = 0.0
```

#### 隐参数默认值
```python
_norm_eps = 1e-6
_pool_temperature = 1.0
```

---

## 6. 类接口定义（必须完全一致）

## 6.1 `StochasticPooling1D`

文件：`src/models/layers/stochastic_pooling1d.py`

```python
class StochasticPooling1D(nn.Module):
    def __init__(
        self,
        _pool_temperature: float = 1.0,
    ) -> None:
        ...

    def forward(
        self,
        values: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入

- `values`: `[Bq, F, C]`
- `scores`: `[Bq, F, 1]`

其中：

- `Bq` 表示合并后的 batch 维
- `F` 是 feature 数
- `C` 是待池化的通道数（这里等于 `core_dim`）

### 输出

- `pooled`: `[Bq, 1, C]`

### 实现要求

1. 先把 `scores` 除以 `_pool_temperature`
2. 然后沿 `F` 维做 softmax 得到 feature 维概率：

```python
probs = torch.softmax(scores / _pool_temperature, dim=1)
```

3. 在训练态：
   - 使用 `torch.distributions.Categorical` 按 `probs` 采样 feature 索引
   - 从 `values` 中 gather 被采样的 feature 向量
   - 输出 shape 必须为 `[Bq, 1, C]`

4. 在 eval 态：
   - 使用加权期望替代采样：
   ```python
   pooled = (probs * values).sum(dim=1, keepdim=True)
   ```
   - 输出 shape 必须为 `[Bq, 1, C]`

5. 不允许改成 max pooling
6. 不允许改成 average pooling
7. 不允许改成 Gumbel-Softmax

---

## 6.2 `STARAggregateRedistributeBlock`

文件：`src/models/fusions/within_scale_star_fusion.py`

```python
class STARAggregateRedistributeBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        core_dim: int,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
        _pool_temperature: float = 1.0,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

### 输入

- `x`: `[Bq, F, hidden_dim]`

### 输出

- `out`: `[Bq, F, hidden_dim]`

### block 结构（必须定死）

每个 block 必须严格按下面顺序实现：

```text
x
-> LayerNorm(hidden_dim)
-> core projection
-> score projection
-> stochastic pooling over feature axis
-> repeat core to all features
-> concat(original_normed_x, repeated_core)
-> fuse MLP
-> residual add with original x
```

### 具体实现要求

#### 第 1 步：归一化
```python
u = self.norm(x)
```

其中：
- `x`: `[Bq, F, D]`
- `u`: `[Bq, F, D]`

使用：

```python
nn.LayerNorm(hidden_dim, eps=_norm_eps)
```

#### 第 2 步：生成 core candidates
```python
core_values = self.core_proj(u)
```

要求：
- `self.core_proj = nn.Linear(hidden_dim, core_dim)`
- 输出 shape：`[Bq, F, core_dim]`

#### 第 3 步：生成 pooling scores
```python
core_scores = self.score_proj(u)
```

要求：
- `self.score_proj = nn.Linear(hidden_dim, 1)`
- 输出 shape：`[Bq, F, 1]`

#### 第 4 步：stochastic pooling
```python
core = self.pool(core_values, core_scores)
```

输出：
- `core`: `[Bq, 1, core_dim]`

#### 第 5 步：repeat + concat
```python
core_rep = core.expand(-1, F, -1)
fuse_in = torch.cat([u, core_rep], dim=-1)
```

输出：
- `core_rep`: `[Bq, F, core_dim]`
- `fuse_in`: `[Bq, F, hidden_dim + core_dim]`

#### 第 6 步：MLP fuse
使用两层 MLP：

```python
self.fuse_mlp = nn.Sequential(
    nn.Linear(hidden_dim + core_dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Dropout(dropout),
)
```

得到：

```python
delta = self.fuse_mlp(fuse_in)
```

输出：
- `delta`: `[Bq, F, hidden_dim]`

#### 第 7 步：残差
```python
out = x + delta
```

不要额外加 gate，不要再套第二个 FFN。

---

## 6.3 `WithinScaleSTARFusion`

文件：`src/models/fusions/within_scale_star_fusion.py`

```python
class WithinScaleSTARFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_features: int = 11,
        core_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
        _pool_temperature: float = 1.0,
    ) -> None:
        ...

    def forward(self, z_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

### 输入

- `z_scale`: `[B, num_features, hidden_dim, N]`

### 输出

- `z_fused`: `[B, num_features, hidden_dim, N]`
- `scale_seq`: `[B, hidden_dim, N]`

---

## 7. forward 过程（逐步，不可歧义）

### 第 1 步：输入形状检查

检查：

- `z_scale.ndim == 4`
- `z_scale.shape == [B, F, D, N]`
- `F == self.num_features`
- `D == self.hidden_dim`

若不满足，必须抛出 `ValueError`

### 第 2 步：变换维度，按 patch 位置做 feature fusion

将：

```text
[B, F, D, N]
```

变为：

```text
[B, N, F, D]
```

具体写法：

```python
z = z_scale.permute(0, 3, 1, 2)
```

然后 reshape 为：

```text
[B*N, F, D]
```

具体写法：

```python
z = z.reshape(B * N, F, D)
```

### 第 3 步：通过若干层 STAR block

```python
for block in self.blocks:
    z = block(z)
```

输出保持：

```text
[B*N, F, D]
```

### 第 4 步：恢复回原始尺度格式

先 reshape：

```python
z = z.view(B, N, F, D)
```

再 permute：

```python
z_fused = z.permute(0, 2, 3, 1)
```

得到：

```text
[B, F, D, N]
```

### 第 5 步：生成尺度级序列表示

固定使用 feature 维平均：

```python
scale_seq = z_fused.mean(dim=1)
```

输出：

```text
[B, D, N]
```

注意：

- 这里只消掉 feature 维 `F`
- **必须保留时间/patch 维 `N`**
- 不允许把 `scale_seq` 继续池化成 `[B, D]`

### 第 6 步：返回

```python
return z_fused, scale_seq
```

---

## 8. 当前三个尺度的默认实例化方式（按已上传主干规格写死）

### 8.1 Macro

当前主干默认输出：

- `z_macro: [B, 11, 128, 16]`

对应实例化：

```python
fusion_macro = WithinScaleSTARFusion(
    hidden_dim=128,
    num_features=11,
    core_dim=64,
    num_layers=1,
    dropout=0.0,
)
```

### 8.2 Mezzo

当前主干默认输出：

- `z_mezzo: [B, 11, 128, 24]`

对应实例化：

```python
fusion_mezzo = WithinScaleSTARFusion(
    hidden_dim=128,
    num_features=11,
    core_dim=64,
    num_layers=1,
    dropout=0.0,
)
```

### 8.3 Micro

当前主干默认输出：

- `z_micro: [B, 11, 128, 36]`

对应实例化：

```python
fusion_micro = WithinScaleSTARFusion(
    hidden_dim=128,
    num_features=11,
    core_dim=64,
    num_layers=1,
    dropout=0.0,
)
```

---

## 9. 初始化与 dtype/device 要求

### 9.1 初始化

- `Linear` 使用 PyTorch 默认初始化
- 不要额外自定义复杂初始化

### 9.2 dtype/device

必须满足：

- `z_fused` 与输入 `z_scale` 保持同 dtype/device
- `scale_seq` 与输入 `z_scale` 保持同 dtype/device

---

## 10. 错误处理要求

以下情况必须报 `ValueError`：

1. `hidden_dim <= 0`
2. `num_features <= 0`
3. `core_dim <= 0`
4. `num_layers <= 0`
5. `dropout < 0` 或 `dropout >= 1`
6. `_norm_eps <= 0`
7. `_pool_temperature <= 0`
8. `z_scale.ndim != 4`
9. `z_scale.shape[1] != num_features`
10. `z_scale.shape[2] != hidden_dim`

错误信息必须指出：

- 哪个值非法
- 当前值是什么
- 合法范围是什么

---

## 11. smoke test 要求

测试文件建议放在：

```text
tests/models/fusions/test_within_scale_star_fusion.py
```

至少包含以下测试：

### 测试 1：macro shape
```python
x = torch.randn(2, 11, 128, 16)
fusion = WithinScaleSTARFusion()
z_fused, scale_seq = fusion(x)

assert z_fused.shape == (2, 11, 128, 16)
assert scale_seq.shape == (2, 128, 16)
```

### 测试 2：mezzo shape
```python
x = torch.randn(2, 11, 128, 24)
fusion = WithinScaleSTARFusion()
z_fused, scale_seq = fusion(x)

assert z_fused.shape == (2, 11, 128, 24)
assert scale_seq.shape == (2, 128, 24)
```

### 测试 3：micro shape
```python
x = torch.randn(2, 11, 128, 36)
fusion = WithinScaleSTARFusion()
z_fused, scale_seq = fusion(x)

assert z_fused.shape == (2, 11, 128, 36)
assert scale_seq.shape == (2, 128, 36)
```

### 测试 4：dtype/device 一致
输入是什么 dtype/device，输出就必须是什么 dtype/device。

### 测试 5：非法 feature 数报错
输入 `shape[1] != 11` 时必须报 `ValueError`

### 测试 6：非法 hidden_dim 报错
输入 `shape[2] != hidden_dim` 时必须报 `ValueError`

### 测试 7：train / eval 都可 forward
在 `model.train()` 和 `model.eval()` 下都必须正常运行。

### 测试 8：时间/patch 维保持不变
例如输入 `[2, 11, 128, 24]`，输出 `z_fused` 的最后一维必须仍然是 `24`，`scale_seq` 的最后一维也必须仍然是 `24`。

---

## 12. import 约定

### `within_scale_star_fusion.py` 内必须这样导入

```python
from src.models.layers import StochasticPooling1D
```

### `src/models/fusions/__init__.py`

```python
from .within_scale_star_fusion import WithinScaleSTARFusion
```

---

## 13. 验收标准

实现完成后，以下条件必须同时满足：

1. 本模块只承担 within-scale feature fusion 职责
2. 输入接口严格对齐 `ModernTCNFiLMEncoder` 输出 `[B, 11, D, N]`
3. 当前版本不消费 `ConditioningEncoder` 输出
4. STAR block 保留 centralized aggregate-redistribute 结构
5. 使用 stochastic pooling，不改成 attention / average / max
6. 输出同时提供 `z_fused` 与 `scale_seq`
7. **不对时间/patch 维做任何压缩**
8. 所有 smoke test 通过

---

## 14. 最终一句话要求

你要实现的不是一个“新主干”，而是一个：

> **对单尺度已编码特征 `[B, F, D, N]` 沿 feature 维执行 STAR 风格中心化聚合-回灌融合，并在完整保留时间/patch 维的前提下，输出融合后特征表示与尺度级序列表示的 within-scale fusion 模块。**

除此之外，什么都不要做。
