# ConditioningEncoder 实现规范（给 coder 的确定性说明，最终版）

本模块的总规则：

- 参数分为两种，一种是**隐参数**，放在 `models/hparams.yaml` 或 `hparams.py`
- 另一种是**开放参数**，放在 `config/models.py`
- 隐参数统一使用 `_` 开头
- 开放参数不使用 `_` 开头
- 本模块只实现 **轻量版 ConditioningEncoder**

---

## 0. 给 coder 的确定性 prompt

你要实现一个 **ConditioningEncoder**，用于把 conditioning 序列编码成条件表示，供主干模型做 FiLM / gate / context injection 使用。

这个模块是 **conditioning encoder**，不是 backbone，不承担主预测时序建模职责。

你必须严格按本规范实现，不要自行补充设计，不要替换结构，不要加入“你认为更好”的大模块。

### 必须实现的东西

1. 一个 `ConditioningEncoder` 类，输入 `[B, 8, 64]`
2. 输出两路条件：
   - `cond_seq: [B, d_cond, 64]`
   - `cond_global: [B, d_cond]`
3. 默认实现采用 **轻量 TSMixer 风格**
4. 提供最小必要的 smoke test，验证 shape、dtype/device、一致性和非法输入报错

### 明确禁止的东西

- 不要把它做成第四条 backbone
- 不要使用 ModernTCN 作为 conditioning encoder
- 不要加 attention
- 不要加 patching
- 不要加 cross-scale fusion
- 不要加 prediction head
- 不要加多分支输出头
- 不要让它输出主预测 representation
- 不要加入不必要的复杂 gating / mixture-of-experts

这个模块唯一职责：

> 输入 `[B, 8, 64]` 的日级 conditioning 序列，做轻量 temporal mixing 与 feature mixing，输出序列条件 `cond_seq` 和全局条件 `cond_global`。

---

## 1. 模块职责边界

### 1.1 模块职责

本模块只负责：

1. 接收 conditioning 输入 `[B, 8, 64]`
2. 将输入投影到隐藏维
3. 做轻量 TSMixer 风格编码
4. 输出：
   - `cond_seq [B, d_cond, 64]`
   - `cond_global [B, d_cond]`

### 1.2 本模块不负责的事情

以下功能不属于本模块：

- 主干序列表征学习
- 多尺度融合
- 预测 head
- 时间序列 patch 化
- 主干 cross-feature fusion
- attention 融合
- denoise
- label 预测
- loss 计算

这些职责必须留给主干 encoder、fusion 模块或 head。

---

## 2. 输入数据语义（定死）

conditioning 输入固定为：

```text
x_cond: [B, 8, 64]
```

8 个特征按固定顺序为：

1. `gap`
2. `gap_rank`
3. `mf_net_ratio`
4. `mf_net_rank`
5. `mf_concentration`
6. `amount_rank`
7. `velocity_rank`
8. `amihud_impact`

时间长度固定为：

- `64`：crop 后目标窗口长度

注意：

- 这里不是 `96`
- 这里不是带 warmup 的长窗
- ConditioningEncoder 默认只吃 crop 后的目标窗口 `[B, 8, 64]`

---

## 3. 输出定义（定死）

### 3.1 序列条件

```text
cond_seq: [B, d_cond, 64]
```

含义：

- 保留日级时间轴
- 后续可用于 macro 分支的时序调制

### 3.2 全局条件

```text
cond_global: [B, d_cond]
```

含义：

- conditioning 的全局状态向量
- 后续可用于 macro / mezzo / micro 的全局 FiLM / gate

---

## 4. 当前实现结论

当前必须实现且只实现的版本是：

> **Light Conditioning TSMixer Encoder**

默认推荐：

- `d_cond = 32`
- `num_blocks = 1`
- `dropout = 0.0`

---

## 5. 文件结构（必须按此落位）

```text
src/models/
    layers/
        __init__.py
        temporal_mixing1d.py
        feature_mixing1d.py

    encoders/
        __init__.py
        conditioning_encoder.py
```

### 5.1 `src/models/layers/temporal_mixing1d.py`

必须包含以下类：

- `TemporalMixing1D`

### 5.2 `src/models/layers/feature_mixing1d.py`

必须包含以下类：

- `FeatureMixing1D`

### 5.3 `src/models/encoders/conditioning_encoder.py`

必须包含以下类：

- `ConditioningMixerBlock`
- `ConditioningEncoder`

### 5.4 `src/models/encoders/__init__.py`

只导出：

```python
from .conditioning_encoder import ConditioningEncoder
```

---

## 6. 参数分类（必须明确）

### 6.1 开放参数（放到 `config/models.py`）

这些是未来可能调参的参数，不加 `_` 前缀：

- `d_cond`
- `num_blocks`
- `dropout`

### 6.2 隐参数（放到 `models/hparams.yaml` 或 `hparams.py`）

这些是不希望频繁调的参数，统一用 `_` 开头：

- `_temporal_mlp_mult`
- `_feature_mlp_mult`
- `_norm_eps`
- `_pool_type`

### 6.3 当前默认值

#### 开放参数默认值
```python
d_cond = 32
num_blocks = 1
dropout = 0.0
```

#### 隐参数默认值
```python
_temporal_mlp_mult = 2
_feature_mlp_mult = 2
_norm_eps = 1e-6
_pool_type = "mean"
```

---

## 7. 类接口定义（必须完全一致）

```python
class ConditioningEncoder(nn.Module):
    def __init__(
        self,
        d_cond: int = 32,
        num_blocks: int = 1,
        dropout: float = 0.0,
        _temporal_mlp_mult: int = 2,
        _feature_mlp_mult: int = 2,
        _norm_eps: float = 1e-6,
        _pool_type: str = "mean",
    ) -> None:
        ...

    def forward(self, x_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

---

## 8. 中间过程（逐步、不可歧义）

### 第 1 步：校验输入

输入必须满足：

- `x_cond.ndim == 3`
- `x_cond.shape == [B, 8, 64]`

若不满足，必须抛出 `ValueError`

### 第 2 步：转成时间在前的格式

将：

```text
[B, 8, 64]
```

变为：

```text
[B, 64, 8]
```

具体写法：

```python
x = x_cond.transpose(1, 2)
```

### 第 3 步：输入投影

对每个时间步共享一个线性层：

```python
x = self.input_proj(x)
```

其中：

- 输入 `[B, 64, 8]`
- 输出 `[B, 64, d_cond]`

### 第 4 步：经过若干个 `ConditioningMixerBlock`

block 数量固定为：

- `num_blocks`

每个 block 输入输出 shape 都必须保持：

```text
[B, 64, d_cond]
```

### 第 5 步：构造 `cond_seq`

将 block 最终输出 `x` 转回：

```python
cond_seq = x.transpose(1, 2)
```

shape 固定为：

```text
[B, d_cond, 64]
```

### 第 6 步：构造 `cond_global`

如果 `_pool_type == "mean"`，则：

```python
cond_global = x.mean(dim=1)
```

输出 shape 固定为：

```text
[B, d_cond]
```

当前默认只允许 `"mean"`。

### 第 7 步：返回

```python
return cond_seq, cond_global
```

---

## 9. `ConditioningMixerBlock` 的结构（必须定死）

### 9.1 block 输入输出

输入：

```text
x: [B, 64, d_cond]
```

输出：

```text
x: [B, 64, d_cond]
```

### 9.2 block 内部流程

每个 block 必须严格按下面顺序实现：

```text
u = Norm(x)
x = x + TemporalMixing1D(u)

v = Norm(x)
x = x + FeatureMixing1D(v)
```

不要改顺序，不要多加额外路径。

### 9.3 归一化

使用：

- `LayerNorm(d_cond)`

不要改成 BatchNorm，不要改成 RMSNorm，当前版本保持统一。

---

## 10. `TemporalMixing1D` 的实现要求

### 10.1 输入输出

输入：

```text
[B, 64, d_cond]
```

输出：

```text
[B, 64, d_cond]
```

### 10.2 具体做法

`TemporalMixing1D` 只沿时间维做 mixing。

实现时必须：

1. 先转置到 `[B, d_cond, 64]`
2. 对每个 hidden channel 独立地沿时间轴做两层 MLP
3. 再转回 `[B, 64, d_cond]`

推荐实现方式：

```python
self.fc1 = nn.Linear(64, _temporal_hidden)
self.fc2 = nn.Linear(_temporal_hidden, 64)
```

其中：

```python
_temporal_hidden = _temporal_mlp_mult * 64
```

具体过程：

```python
x = x.transpose(1, 2)        # [B, d_cond, 64]
x = self.fc2(dropout(gelu(self.fc1(x))))
x = x.transpose(1, 2)        # [B, 64, d_cond]
```

注意：

- 这是沿时间维做 mixing
- 不是沿特征维
- 不是卷积
- 不是 attention

---

## 11. `FeatureMixing1D` 的实现要求

### 11.1 输入输出

输入：

```text
[B, 64, d_cond]
```

输出：

```text
[B, 64, d_cond]
```

### 11.2 具体做法

`FeatureMixing1D` 只沿最后一维隐藏特征维做 mixing。

实现方式：

```python
_feature_hidden = _feature_mlp_mult * d_cond

self.fc1 = nn.Linear(d_cond, _feature_hidden)
self.fc2 = nn.Linear(_feature_hidden, d_cond)
```

具体过程：

```python
x = self.fc2(dropout(gelu(self.fc1(x))))
```

注意：

- 这是沿隐藏维做 mixing
- 不是时间 mixing
- 不是跨尺度 mixing

---

## 12. 初始化与 dtype/device 要求

### 12.1 初始化

- `Linear` 使用 PyTorch 默认初始化
- 不要自定义复杂初始化

### 12.2 dtype/device

必须满足：

- 输出 `cond_seq` 与输入 `x_cond` 保持同 dtype
- 输出 `cond_seq` 与输入 `x_cond` 保持同 device
- 输出 `cond_global` 与输入 `x_cond` 保持同 dtype/device

---

## 13. 错误处理要求

以下情况必须报 `ValueError`：

1. `d_cond <= 0`
2. `num_blocks <= 0`
3. `dropout < 0` 或 `dropout >= 1`
4. `_temporal_mlp_mult <= 0`
5. `_feature_mlp_mult <= 0`
6. `x_cond.ndim != 3`
7. `x_cond.shape[1] != 8`
8. `x_cond.shape[2] != 64`
9. `_pool_type != "mean"`

错误信息必须指出：

- 哪个值非法
- 当前值是什么
- 合法范围是什么

---

## 14. smoke test 要求

至少包含以下测试：

### 测试 1：shape 正确
```python
x = torch.randn(2, 8, 64)
encoder = ConditioningEncoder()
cond_seq, cond_global = encoder(x)

assert cond_seq.shape == (2, 32, 64)
assert cond_global.shape == (2, 32)
```

### 测试 2：dtype/device 一致
输入是什么 dtype/device，输出就必须是什么 dtype/device。

### 测试 3：非法 shape 报错
例如：

- `[2, 8, 63]`
- `[2, 7, 64]`
- `[2, 8]`

都必须报 `ValueError`

### 测试 4：不同 `d_cond` 可运行
例如 `d_cond=64` 时：

```python
x = torch.randn(2, 8, 64)
encoder = ConditioningEncoder(d_cond=64)
cond_seq, cond_global = encoder(x)

assert cond_seq.shape == (2, 64, 64)
assert cond_global.shape == (2, 64)
```

### 测试 5：不同 `num_blocks` 可运行
例如 `num_blocks=2` 时仍可正确 forward。

---

## 15. 最终一句话要求

你要实现的不是一个“side backbone”，而是一个 **轻量的 TSMixer 风格 conditioning encoder**。

它只做：

> `[B, 8, 64]` → 输入投影 → 1~2 个轻量 mixer blocks → `cond_seq [B, d_cond, 64]` + `cond_global [B, d_cond]`

除此之外，什么都不要做。
