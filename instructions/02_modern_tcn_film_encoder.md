# ModernTCN-FiLM Encoder 实现规范（最终版，给 coder 的确定性说明）

## 0. 给 coder 的确定性 prompt

你要实现一个 **ModernTCN-FiLM Encoder**，用于对单一尺度（macro / mezzo / micro）的 **11 个 float backbone 特征**进行 **先独立编码、后续再融合** 的时序表示学习。

这是一个 **单尺度 encoder**，不是完整模型，不负责：

- denoise
- cross-scale fusion
- sidechain encoder
- prediction head
- late fusion
- loss

它只负责：

1. 接收一个尺度的 float 特征张量 `x_float`，形状 `[B, F, T]`
2. 接收两个 int 条件张量 `x_state`、`x_pos`，形状均为 `[B, T]`
3. 对每个 float feature **独立编码**，但 **共享同一套 encoder 参数**
4. 在每个 block 内使用由 `x_state/x_pos` 生成的条件进行 **FiLM 调制**
5. 输出 patch 级隐藏表示 `z`，形状 `[B, F, D, N]`

你必须严格按本规范实现，不要自行补充结构，不要修改职责边界，不要加入额外模块，不要改变输入输出语义。

### 必须实现的东西

1. `Patch1D`：将 `[B*F, 1, T]` patchify 为 `[B*F, D, N]`
2. `ConditionEmbedding1D`：将两个 int 序列嵌入并对齐到 patch 级条件 `[B*F, D_cond, N]`
3. `FiLM1D`：根据条件生成 `gamma/beta` 并调制主干激活
4. `ChannelFFN1D`：对 hidden channel 维做轻量前馈变换
5. `ModernTCNFiLMBlock`：一个 block，包含
   - 预归一化
   - FiLM
   - temporal depthwise conv
   - 再次预归一化
   - FiLM
   - ChannelFFN
   - 残差连接
6. `ModernTCNFiLMEncoder`：完整单尺度 encoder

### 明确禁止的东西

- 不要做 feature 之间的早期融合
- 不要做 cross-feature attention / mixer
- 不要做 cross-scale fusion
- 不要接 prediction head
- 不要加入 sidechain
- 不要加入 self-attention
- 不要加入 BatchNorm
- 不要加入多路 backbone
- 不要让 `x_state/x_pos` 直接与 float 通道拼接卷积
- 不要让不同 feature 各自拥有独立 encoder 参数
- 不要改成多变量 ModernTCN 原始版本的 `M` 维 mixing

这个模块唯一职责：

> 在单一尺度上，对 11 个 float backbone 特征逐特征独立编码（共享参数），并用两个离散条件序列在 block 内做 FiLM 调制，输出 patch 级隐藏表示。

---

## 1. 模块职责边界

### 1.1 本模块负责的事情

本模块只负责：

1. 单尺度 float 特征的 patch 化与编码
2. 将两个 int 条件嵌入为 patch 级条件表示
3. 在每个 block 内用 FiLM 对主干激活做调制
4. 输出编码后的 patch 级表示 `[B, F, D, N]`

### 1.2 本模块不负责的事情

以下功能一律不属于本模块：

- denoise
- sidechain 处理
- feature late fusion
- scale fusion
- label 预测
- uncertainty 估计
- reconstruction
- pooling 为最终分类/回归输出

这些职责必须留给其他模块，不允许塞进本模块。

---

## 2. 命名规则（必须遵守）

### 2.1 公开/未来可能调参的参数

这些参数名 **不加下划线前缀**：

- `patch_len`
- `patch_stride`
- `hidden_dim`
- `cond_dim`
- `kernel_size`
- `ffn_ratio`
- `num_layers`
- `state_vocab_size`
- `pos_vocab_size`
- `dropout`
- `num_features`
- `seq_len`

规则：

- 这些参数属于未来可能会调的超参数
- 必须使用清晰、无下划线前缀的名字

### 2.2 隐参数 / 内部实现参数

这些参数名必须 **加 `_` 前缀**：

- `_num_patches`
- `_expected_seq_len`
- `_groups`
- `_in_channels`
- `_out_channels`
- `_padding`
- `_eps`

规则：

- 这些参数属于内部推导值、实现细节、通常不直接调
- 必须统一使用 `_` 前缀

---

## 3. 文件结构（必须按此落位）

```text
src/models/
    layers/
        __init__.py
        patch1d.py
        film1d.py
        channel_ffn1d.py

    embeddings/
        __init__.py
        condition_embedding1d.py

    encoders/
        __init__.py
        modern_tcn_film_encoder.py
```

### 3.1 `src/models/layers/patch1d.py`

必须包含以下类：

- `Patch1D`

### 3.2 `src/models/layers/film1d.py`

必须包含以下类：

- `FiLM1D`

### 3.3 `src/models/layers/channel_ffn1d.py`

必须包含以下类：

- `ChannelFFN1D`

### 3.4 `src/models/embeddings/condition_embedding1d.py`

必须包含以下类：

- `ConditionEmbedding1D`

### 3.5 `src/models/encoders/modern_tcn_film_encoder.py`

必须包含以下类：

- `ModernTCNFiLMBlock`
- `ModernTCNFiLMEncoder`

### 3.6 `src/models/encoders/__init__.py`

只导出：

```python
from .modern_tcn_film_encoder import ModernTCNFiLMEncoder
```

### 3.7 其他 `__init__.py`

#### `src/models/layers/__init__.py`

必须导出：

```python
from .patch1d import Patch1D
from .film1d import FiLM1D
from .channel_ffn1d import ChannelFFN1D
```

#### `src/models/embeddings/__init__.py`

必须导出：

```python
from .condition_embedding1d import ConditionEmbedding1D
```

---

## 4. 数据输入语义（按当前数据规范写死）

本模块只消费某一个尺度的数据。

### 4.1 Float 输入

输入 `x_float` 只来自该尺度的 11 个 float backbone 特征：

- `f0..f5`
- `f8..f12`

不接收 `f6/f7`。

### 4.2 Int 条件输入

- `x_state`：来自该尺度 `f6`，即涨跌停 bitmask
- `x_pos`：来自该尺度 `f7`，即步内位置索引

### 4.3 单尺度输入 shape

#### Macro

- `x_float`: `[B, 11, 64]`
- `x_state`: `[B, 64]`
- `x_pos`: `[B, 64]`

#### Mezzo

- `x_float`: `[B, 11, 96]`
- `x_state`: `[B, 96]`
- `x_pos`: `[B, 96]`

#### Micro

- `x_float`: `[B, 11, 144]`
- `x_state`: `[B, 144]`
- `x_pos`: `[B, 144]`

### 4.4 本模块输出 shape

输出：

- `z`: `[B, 11, hidden_dim, num_patches]`

其中 `num_patches` 由 `patch_len` 和 `patch_stride` 决定。

---

## 5. 设计决策（全部定死）

### 5.1 编码策略

对每个 float feature **独立编码**，但 **共享同一套 encoder 参数**。

具体做法：

```text
[B, F, T] -> reshape -> [B*F, 1, T] -> shared encoder -> [B*F, D, N] -> reshape back -> [B, F, D, N]
```

不允许每个 feature 独立维护一套不同参数。

### 5.2 条件策略

两个 int 条件 `x_state/x_pos`：

- 不与 float 直接拼接
- 不作为卷积输入通道
- 只能先做 embedding，再变成 patch 级条件，再进入 FiLM

### 5.3 归一化策略

使用 `LayerNorm` 风格的 **channel-last 归一化**。

实现要求：

- 在 `[B*F, D, N]` 形状上，先转为 `[B*F, N, D]`
- 对最后一维 `D` 做 `nn.LayerNorm(D)`
- 再转回 `[B*F, D, N]`

不要使用 `BatchNorm1d`。

### 5.4 激活函数

统一使用 `GELU`。

### 5.5 dropout

允许在 FFN 内使用 dropout，但默认 `dropout=0.0`。

---

## 6. 类接口定义（必须完全一致）

## 6.1 `Patch1D`

文件：`src/models/layers/patch1d.py`

```python
class Patch1D(nn.Module):
    def __init__(
        self,
        patch_len: int,
        patch_stride: int,
        hidden_dim: int,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

### 输入

- `x`: `[Bf, 1, T]`

### 输出

- `z`: `[Bf, hidden_dim, N]`

### 实现要求

1. 使用 `nn.Conv1d(1, hidden_dim, kernel_size=patch_len, stride=patch_stride)`
2. 为保证最后一个 patch 可覆盖尾部，必须在右侧做 replicate pad：

```python
pad_right = patch_len - patch_stride
```

3. pad 必须写成：

```python
F.pad(x, (0, pad_right), mode="replicate")
```

4. 不允许加 bias-free 特殊版本，不允许加 norm，不允许加 activation

---

## 6.2 `ConditionEmbedding1D`

文件：`src/models/embeddings/condition_embedding1d.py`

```python
class ConditionEmbedding1D(nn.Module):
    def __init__(
        self,
        state_vocab_size: int,
        pos_vocab_size: int,
        cond_dim: int,
        patch_len: int,
        patch_stride: int,
        num_features: int,
    ) -> None:
        ...

    def forward(
        self,
        x_state: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入

- `x_state`: `[B, T]`
- `x_pos`: `[B, T]`

### 输出

- `c`: `[B*F, cond_dim, N]`

其中 `F = num_features`。

### 实现要求

1. `x_state` 先经 `nn.Embedding(state_vocab_size, cond_dim)`
2. `x_pos` 先经 `nn.Embedding(pos_vocab_size, cond_dim)`
3. 两者直接相加：

```python
e = e_state + e_pos
```

不要 concat。

4. 然后转为 patch 级：

```python
[B, T, cond_dim] -> [B, cond_dim, T] -> AvgPool1d(kernel=patch_len, stride=patch_stride, ceil_mode=False)
```

5. 为对齐 `Patch1D` 的尾部 pad，必须在 pooling 前也做同样的右侧 replicate pad：

```python
pad_right = patch_len - patch_stride
```

6. 得到 `[B, cond_dim, N]` 后，沿 feature 维复制 `num_features` 次：

```python
[B, cond_dim, N] -> [B, F, cond_dim, N] -> [B*F, cond_dim, N]
```

7. 不允许让不同 feature 拿到不同的条件副本；所有 feature 共享同一时刻的状态/位置条件。

---

## 6.3 `FiLM1D`

文件：`src/models/layers/film1d.py`

```python
class FiLM1D(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
    ) -> None:
        ...

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入

- `x`: `[Bf, hidden_dim, N]`
- `cond`: `[Bf, cond_dim, N]`

### 输出

- `[Bf, hidden_dim, N]`

### 实现要求

1. 使用 `Conv1d(cond_dim, 2 * hidden_dim, kernel_size=1)` 生成 `gamma_beta`
2. 拆分为：

```python
gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
```

3. 调制公式必须是：

```python
out = x * (1.0 + gamma) + beta
```

4. 不允许加入额外 gate，不允许使用 sigmoid/tanh 包裹 gamma/beta

---

## 6.4 `ChannelFFN1D`

文件：`src/models/layers/channel_ffn1d.py`

```python
class ChannelFFN1D(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

### 输入输出

- 输入：`[Bf, hidden_dim, N]`
- 输出：`[Bf, hidden_dim, N]`

### 实现要求

1. 中间维度：

```python
_ffn_dim = int(hidden_dim * ffn_ratio)
```

2. 使用：

```python
Conv1d(hidden_dim, _ffn_dim, kernel_size=1)
GELU
Dropout(dropout)
Conv1d(_ffn_dim, hidden_dim, kernel_size=1)
Dropout(dropout)
```

3. 不允许做 depthwise，不允许 groups > 1

---

## 6.5 `ModernTCNFiLMBlock`

文件：`src/models/encoders/modern_tcn_film_encoder.py`

```python
class ModernTCNFiLMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        kernel_size: int,
        ffn_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入输出

- `x`: `[Bf, hidden_dim, N]`
- `cond`: `[Bf, cond_dim, N]`
- 输出：`[Bf, hidden_dim, N]`

### 实现要求

block 结构必须严格为：

```text
x
-> Norm1
-> FiLM1(cond)
-> DepthwiseConv1d(kernel_size, padding=same)
-> Residual Add
-> Norm2
-> FiLM2(cond)
-> ChannelFFN1D
-> Residual Add
```

#### 归一化细节

由于 `nn.LayerNorm` 作用于最后一维，必须写一个内部私有函数：

```python
def _apply_ln(self, ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    # x: [Bf, D, N]
    x = x.transpose(1, 2)      # [Bf, N, D]
    x = ln(x)
    x = x.transpose(1, 2)      # [Bf, D, N]
    return x
```

#### temporal depthwise conv

要求：

- `in_channels = out_channels = hidden_dim`
- `groups = hidden_dim`
- `padding = kernel_size // 2`
- stride = 1

不允许 dilation，不允许 causal conv，不允许再加 pointwise conv。

#### 残差写法

必须：

```python
x = x + temporal_branch
x = x + ffn_branch
```

---

## 6.6 `ModernTCNFiLMEncoder`

文件：`src/models/encoders/modern_tcn_film_encoder.py`

```python
class ModernTCNFiLMEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_features: int = 11,
        patch_len: int = 8,
        patch_stride: int = 4,
        hidden_dim: int = 128,
        cond_dim: int = 64,
        kernel_size: int = 7,
        ffn_ratio: float = 2.0,
        num_layers: int = 2,
        state_vocab_size: int = 16,
        pos_vocab_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(
        self,
        x_float: torch.Tensor,
        x_state: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入

- `x_float`: `[B, num_features, seq_len]`
- `x_state`: `[B, seq_len]`
- `x_pos`: `[B, seq_len]`

### 输出

- `z`: `[B, num_features, hidden_dim, num_patches]`

### 公开属性

必须保留以下公开属性：

- `seq_len`
- `num_features`
- `patch_len`
- `patch_stride`
- `hidden_dim`
- `cond_dim`
- `kernel_size`
- `ffn_ratio`
- `num_layers`
- `state_vocab_size`
- `pos_vocab_size`
- `dropout`

### 隐属性

内部推导值必须使用 `_` 前缀，例如：

- `_expected_seq_len = seq_len`
- `_num_patches`

### `_num_patches` 计算公式

设：

```python
pad_right = patch_len - patch_stride
```

则：

```python
_num_patches = ((seq_len + pad_right - patch_len) // patch_stride) + 1
```

---

## 7. forward 过程（逐步，不可歧义）

以下过程必须逐步实现，不允许自行改语义。

### 第 1 步：输入形状检查

检查：

- `x_float.ndim == 3`
- `x_state.ndim == 2`
- `x_pos.ndim == 2`
- `x_float.shape == [B, F, T]`
- `x_state.shape == [B, T]`
- `x_pos.shape == [B, T]`
- `F == self.num_features`
- `T == self.seq_len`

若不满足，必须抛出 `ValueError`。

### 第 2 步：把 float 特征拉平成独立样本

```python
x = x_float.reshape(B * F, 1, T)
```

注意：

- 这里的含义是“每个 feature 单独编码”
- 不是 feature 维卷积
- 不是 feature mixing

### 第 3 步：Patch1D

```python
z = self.patch(x)
```

输出：

- `z`: `[B*F, hidden_dim, N]`

### 第 4 步：ConditionEmbedding1D

```python
cond = self.condition_embedding(x_state, x_pos)
```

输出：

- `cond`: `[B*F, cond_dim, N]`

### 第 5 步：重复通过 blocks

对 `self.blocks` 中每个 block：

```python
for block in self.blocks:
    z = block(z, cond)
```

### 第 6 步：reshape 回 `[B, F, D, N]`

```python
z = z.view(B, F, self.hidden_dim, self._num_patches)
```

返回 `z`。

---

## 8. 数学定义（用于避免 coder 误解）

给定输入：

- `x_float ∈ R^{B×F×T}`
- `x_state ∈ Z^{B×T}`
- `x_pos ∈ Z^{B×T}`

### 8.1 Patch 映射

对每个 feature 独立处理：

- `x_float -> x_indep ∈ R^{(B·F)×1×T}`
- 经 patch conv 得到 `z0 ∈ R^{(B·F)×D×N}`

### 8.2 条件嵌入

- `E_state(x_state) ∈ R^{B×T×Dc}`
- `E_pos(x_pos) ∈ R^{B×T×Dc}`
- `E = E_state + E_pos`
- 经过 patch 对齐池化得 `C ∈ R^{B×Dc×N}`
- 在 feature 维复制得到 `C_rep ∈ R^{(B·F)×Dc×N}`

### 8.3 FiLM

对于 block 中任一激活 `h ∈ R^{(B·F)×D×N}`：

- `gamma, beta = Conv1x1(C_rep)`
- `FiLM(h, C_rep) = h * (1 + gamma) + beta`

### 8.4 Block

一个 block 定义为：

- `u = DWConv(FiLM(LN(h), C))`
- `h1 = h + u`
- `v = FFN(FiLM(LN(h1), C))`
- `h2 = h1 + v`

最终输出 `h2`。

---

## 9. 初始化要求

### 9.1 默认初始化

使用 PyTorch 默认初始化即可，不做额外自定义初始化。

### 9.2 Condition Embedding 初始化

- `nn.Embedding` 使用默认初始化
- 不要手工置零

### 9.3 FiLM 初始化

允许使用默认初始化，不要求 identity init。

---

## 10. 错误处理要求

以下情况必须报 `ValueError`：

1. `seq_len <= 0`
2. `num_features <= 0`
3. `patch_len <= 0`
4. `patch_stride <= 0`
5. `hidden_dim <= 0`
6. `cond_dim <= 0`
7. `kernel_size <= 0`
8. `num_layers <= 0`
9. `state_vocab_size <= 0`
10. `pos_vocab_size <= 0`
11. `kernel_size % 2 == 0`
12. `patch_len < patch_stride`

报错信息必须说明：

- 参数名
- 当前值
- 期望条件

---

## 11. 不同尺度的实例化方式（必须明确给出）

### 11.1 Macro

```python
encoder_macro = ModernTCNFiLMEncoder(
    seq_len=64,
    num_features=11,
    patch_len=8,
    patch_stride=4,
    hidden_dim=128,
    cond_dim=64,
    kernel_size=7,
    ffn_ratio=2.0,
    num_layers=2,
    state_vocab_size=16,
    pos_vocab_size=8,
    dropout=0.0,
)
```

### 11.2 Mezzo

```python
encoder_mezzo = ModernTCNFiLMEncoder(
    seq_len=96,
    num_features=11,
    patch_len=8,
    patch_stride=4,
    hidden_dim=128,
    cond_dim=64,
    kernel_size=7,
    ffn_ratio=2.0,
    num_layers=2,
    state_vocab_size=16,
    pos_vocab_size=16,
    dropout=0.0,
)
```

### 11.3 Micro

```python
encoder_micro = ModernTCNFiLMEncoder(
    seq_len=144,
    num_features=11,
    patch_len=8,
    patch_stride=4,
    hidden_dim=128,
    cond_dim=64,
    kernel_size=7,
    ffn_ratio=2.0,
    num_layers=2,
    state_vocab_size=16,
    pos_vocab_size=64,
    dropout=0.0,
)
```

说明：

- `state_vocab_size=16` 因为 `f6` 是 bitmask，按 0..15 处理
- `pos_vocab_size` 只要求覆盖实际取值范围即可，允许稍大于真实最大值

---

## 12. import 约定

### `modern_tcn_film_encoder.py` 内必须这样导入

```python
from src.models.layers import Patch1D, FiLM1D, ChannelFFN1D
from src.models.embeddings import ConditionEmbedding1D
```

---

## 13. 单元测试要求

测试文件建议放在：

```text
tests/models/encoders/test_modern_tcn_film_encoder.py
```

### 必须包含的测试

#### 测试 1：macro forward shape

输入：

- `x_float = torch.randn(2, 11, 64)`
- `x_state = torch.randint(0, 16, (2, 64))`
- `x_pos = torch.randint(0, 8, (2, 64))`

验证：

- 输出 shape 为 `[2, 11, 128, N_macro]`

#### 测试 2：mezzo forward shape

输入长度 96，验证 shape 正确。

#### 测试 3：micro forward shape

输入长度 144，验证 shape 正确。

#### 测试 4：dtype/device 一致

确保输出 dtype/device 与输入 float 张量一致。

#### 测试 5：非法输入长度报错

构造错误 `T`，必须 `raise ValueError`。

#### 测试 6：feature 维不匹配报错

构造 `x_float.shape[1] != num_features`，必须 `raise ValueError`。

#### 测试 7：条件对齐后的 patch 数一致

验证 `Patch1D` 与 `ConditionEmbedding1D` 的输出 `N` 完全一致。

---

## 14. 验收标准

实现完成后，以下条件必须同时满足：

1. `ModernTCNFiLMEncoder` 只承担单尺度 per-feature encoder 职责。
2. 11 个 float feature 独立编码，但共享参数。
3. `x_state/x_pos` 不直接拼到 float 输入里，而是只通过 `ConditionEmbedding1D -> FiLM1D` 调制主干。
4. block 内没有任何 cross-feature mixing。
5. 使用 `LayerNorm`，不使用 `BatchNorm1d`。
6. 文件路径与类归属严格符合本规范。
7. 公开参数名与隐参数命名规则严格符合本规范。
8. 所有单元测试通过。

---

## 15. 最终一句话要求

你要实现的不是一个“多变量直接混合的 ModernTCN”，而是一个：

> **对 11 个 float backbone 特征逐特征独立编码（共享参数），并用两个离散条件序列在 block 内做 FiLM 调制的单尺度 ModernTCN 风格 encoder。**

除此之外，什么都不要做。
