# CrossScaleFusion 实现规范（最终版，给 coder 的确定性说明）

本模块的总规则：

- 参数分为两种，一种是**隐参数**，放在 `models/hparams.yaml` 或 `hparams.py`
- 另一种是**开放参数**，放在 `config/models.py`
- 隐参数统一使用 `_` 开头
- 开放参数不使用 `_` 开头
- 本模块只负责 **cross-scale fusion**
- 本模块当前只消费三个已经完成 side bridge 的尺度表示
- 本模块采用 **bottleneck-latent cross-scale fusion**
- 本模块 **不压缩**输入的时间/patch 维
- 本模块 **不负责** side / conditioning 融合
- 本模块 **不负责** prediction head

---

## 0. 给 coder 的确定性 prompt

你要实现一个 **CrossScaleFusion**，用于把三路已经完成单尺度编码、within-scale fusion、以及 exogenous bridge fusion 的尺度序列表示做最终的跨尺度融合。

这里的 cross-scale fusion 采用：

1. 保留三路输入各自完整的时间/patch 维
2. 为每一路加一个 learnable scale embedding
3. 将三路 token 直接拼接成一个统一的 multi-scale token set
4. 使用少量 learnable bottleneck latents 作为融合载体
5. 让 bottleneck latents 去 cross-attend 所有尺度 tokens
6. 再在 bottleneck latent 空间做 self-attention 和 FFN
7. 输出融合后的 latent 序列与全局表示

你必须严格按本规范实现，不要自行补充结构，不要加入 side 分支，不要把它改成完整模型。

### 必须实现的东西

1. 一个 `CrossScaleFusion` 类
2. 输入三路：
   - `macro_seq: [B, hidden_dim, 16]`
   - `mezzo_seq: [B, hidden_dim, 24]`
   - `micro_seq: [B, hidden_dim, 36]`
3. 输出两路：
   - `fused_latents: [B, hidden_dim, num_latents]`
   - `fused_global: [B, hidden_dim]`
4. 内部使用 `CrossScaleFusionBlock`
5. 使用 `nn.MultiheadAttention(batch_first=True)` 做：
   - latent -> multi-scale tokens 的 cross-attention
   - latent 内部 self-attention
6. 使用三组 learnable scale embeddings 区分 macro / mezzo / micro
7. 提供最小必要的 smoke test

### 明确禁止的东西

- 不要再接 `cond_seq / cond_global`
- 不要再接 `s1 / s2 / s3`
- 不要再接 `bridge_global`
- 不要做 cross-scale 之前的 token 压缩
- 不要做 patching
- 不要做 Deformable Attention
- 不要做 Mamba
- 不要做图结构学习
- 不要做 prediction head
- 不要把三路时间长度强行对齐成相同长度
- 不要改成“只池化成三个向量再融合”

这个模块唯一职责：

> 输入三路已经吸收过 side 信息的尺度级序列表示，保留其完整 token 序列，通过 bottleneck latents 做跨尺度晚融合，输出最终融合 latent 序列和全局表示。

---

## 1. 本规范依赖的上游接口（按当前已上传规格写死）

### 1.1 上游 within-scale 输出

`WithinScaleSTARFusion` 的输出之一是：

```text
scale_seq: [B, hidden_dim, N]
```

当前默认三个尺度分别为：

- macro: `scale_seq_macro [B, 128, 16]`
- mezzo: `scale_seq_mezzo [B, 128, 24]`
- micro: `scale_seq_micro [B, 128, 36]`

### 1.2 上游 exogenous bridge 输出

`ExogenousBridgeFusion` 的输出为：

- `endogenous_fused: [B, hidden_dim, N_endo]`
- `bridge_global: [B, hidden_dim]`

本模块当前只消费这一步的 `endogenous_fused`，也就是：

- `macro_fused [B, 128, 16]`
- `mezzo_fused [B, 128, 24]`
- `micro_fused [B, 128, 36]`

本模块 **当前不得消费** `bridge_global`。

### 1.3 当前典型调用方式

```python
fused_latents, fused_global = cross_scale_fusion(
    macro_seq=macro_fused,   # [B, 128, 16]
    mezzo_seq=mezzo_fused,   # [B, 128, 24]
    micro_seq=micro_fused,   # [B, 128, 36]
)
```

---

## 2. 模块职责边界

### 2.1 本模块负责的事情

本模块只负责：

1. 接收三路尺度级序列表示
2. 给三路 tokens 加上不同的 scale embeddings
3. 拼接成统一的 multi-scale token set
4. 用少量 bottleneck latents 吸收三路尺度信息
5. 输出：
   - `fused_latents [B, hidden_dim, num_latents]`
   - `fused_global [B, hidden_dim]`

### 2.2 本模块不负责的事情

以下事情不属于本模块：

- within-scale feature fusion
- side hierarchy 生成
- exogenous bridge fusion
- prediction head
- label 预测
- loss 计算

---

## 3. 设计决策（全部定死）

### 3.1 当前采用 bottleneck-latent cross-scale fusion

当前跨尺度融合采用：

- 不对输入 token 做压缩
- 不要求不同尺度 token 数相同
- 使用少量 learned bottleneck latents 做融合载体

### 3.2 三路输入 token 全部保留

本模块绝不压缩输入时间/patch 维：

- macro 保留 `16`
- mezzo 保留 `24`
- micro 保留 `36`

拼接后总 token 数：

```python
N_total = 16 + 24 + 36 = 76
```

### 3.3 bottleneck latents 数量固定为小数目

当前默认：

```python
num_latents = 8
```

这些 latents 是 learnable 参数，不是从输入池化得到。

### 3.4 scale embedding 是必须的

必须为三路输入分别定义三个 learnable scale embeddings：

- `macro_scale_embedding`
- `mezzo_scale_embedding`
- `micro_scale_embedding`

每个 shape 固定为：

```python
[1, 1, hidden_dim]
```

### 3.5 当前只在 latent 空间做 self-attention

- 输入 tokens 不做 self-attention
- 只有 bottleneck latents 做 self-attention
- multi-scale tokens 只作为 cross-attention 的 K/V memory

---

## 4. 文件结构（必须按此落位）

```text
src/models/
    fusions/
        __init__.py
        cross_scale_fusion.py
```

### 4.1 `src/models/fusions/cross_scale_fusion.py`

必须包含以下类：

- `CrossScaleFusionBlock`
- `CrossScaleFusion`

### 4.2 `src/models/fusions/__init__.py`

只导出：

```python
from .cross_scale_fusion import CrossScaleFusion
```

---

## 5. 参数分类（必须明确）

### 5.1 开放参数（放到 `config/models.py`）

这些参数是未来可能调的，不加 `_` 前缀：

- `hidden_dim`
- `num_latents`
- `num_heads`
- `ffn_ratio`
- `num_layers`
- `dropout`

### 5.2 隐参数（放到 `models/hparams.yaml` 或 `hparams.py`）

这些参数是不希望频繁调的，统一加 `_` 前缀：

- `_norm_eps`

### 5.3 当前默认值

#### 开放参数默认值
```python
hidden_dim = 128
num_latents = 8
num_heads = 4
ffn_ratio = 2.0
num_layers = 1
dropout = 0.0
```

#### 隐参数默认值
```python
_norm_eps = 1e-6
```

---

## 6. 类接口定义（必须完全一致）

## 6.1 `CrossScaleFusionBlock`

文件：`src/models/fusions/cross_scale_fusion.py`

```python
class CrossScaleFusionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        latents: torch.Tensor,
        scale_tokens: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入

- `latents: [B, num_latents, hidden_dim]`
- `scale_tokens: [B, N_total, hidden_dim]`

### 输出

- `latents_out: [B, num_latents, hidden_dim]`

---

## 6.2 `CrossScaleFusion`

文件：`src/models/fusions/cross_scale_fusion.py`

```python
class CrossScaleFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_latents: int = 8,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        num_layers: int = 1,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        macro_seq: torch.Tensor,
        mezzo_seq: torch.Tensor,
        micro_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

### 输入

- `macro_seq: [B, hidden_dim, 16]`
- `mezzo_seq: [B, hidden_dim, 24]`
- `micro_seq: [B, hidden_dim, 36]`

### 输出

- `fused_latents: [B, hidden_dim, num_latents]`
- `fused_global: [B, hidden_dim]`

---

## 7. `CrossScaleFusionBlock` 的结构（必须定死）

### 7.1 总体流程

每个 block 必须严格按下面顺序实现：

```text
latents
-> LayerNorm
-> cross-attend to scale_tokens
-> residual add

latents
-> LayerNorm
-> self-attention among latents
-> residual add

latents
-> LayerNorm
-> FFN
-> residual add
```

### 7.2 具体实现要求

#### 第 1 步：latent -> scale_tokens cross-attention

使用：

```python
self.cross_norm = nn.LayerNorm(hidden_dim, eps=_norm_eps)
self.cross_attn = nn.MultiheadAttention(
    embed_dim=hidden_dim,
    num_heads=num_heads,
    dropout=dropout,
    batch_first=True,
)
```

然后：

```python
q = self.cross_norm(latents)
cross_delta, _ = self.cross_attn(q, scale_tokens, scale_tokens, need_weights=False)
latents = latents + cross_delta
```

#### 第 2 步：latent self-attention

使用：

```python
self.self_norm = nn.LayerNorm(hidden_dim, eps=_norm_eps)
self.self_attn = nn.MultiheadAttention(
    embed_dim=hidden_dim,
    num_heads=num_heads,
    dropout=dropout,
    batch_first=True,
)
```

然后：

```python
q = self.self_norm(latents)
self_delta, _ = self.self_attn(q, q, q, need_weights=False)
latents = latents + self_delta
```

#### 第 3 步：latent FFN

使用：

```python
_ffn_dim = int(hidden_dim * ffn_ratio)

self.ffn = nn.Sequential(
    nn.LayerNorm(hidden_dim, eps=_norm_eps),
    nn.Linear(hidden_dim, _ffn_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(_ffn_dim, hidden_dim),
    nn.Dropout(dropout),
)
```

然后：

```python
latents = latents + self.ffn(latents)
```

返回：

```python
return latents
```

---

## 8. `CrossScaleFusion` 的 forward 过程（逐步，不可歧义）

### 第 1 步：输入形状检查

检查：

- `macro_seq.ndim == 3`
- `mezzo_seq.ndim == 3`
- `micro_seq.ndim == 3`

并检查：

- `macro_seq.shape == [B, hidden_dim, 16]`
- `mezzo_seq.shape == [B, hidden_dim, 24]`
- `micro_seq.shape == [B, hidden_dim, 36]`

三路 batch 必须一致。

若不满足，必须抛出 `ValueError`

### 第 2 步：转成 time-major

```python
macro = macro_seq.transpose(1, 2)   # [B, 16, hidden_dim]
mezzo = mezzo_seq.transpose(1, 2)   # [B, 24, hidden_dim]
micro = micro_seq.transpose(1, 2)   # [B, 36, hidden_dim]
```

### 第 3 步：加 scale embeddings

定义三个 learnable 参数：

```python
self.macro_scale_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
self.mezzo_scale_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
self.micro_scale_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
```

然后：

```python
macro = macro + self.macro_scale_embedding
mezzo = mezzo + self.mezzo_scale_embedding
micro = micro + self.micro_scale_embedding
```

### 第 4 步：拼接 multi-scale tokens

```python
scale_tokens = torch.cat([macro, mezzo, micro], dim=1)
```

输出：

```text
[B, 76, hidden_dim]
```

### 第 5 步：构造初始 bottleneck latents

定义 learnable 参数：

```python
self.latents = nn.Parameter(torch.zeros(1, num_latents, hidden_dim))
```

然后：

```python
latents = self.latents.expand(scale_tokens.shape[0], -1, -1)
```

输出：

```text
[B, num_latents, hidden_dim]
```

### 第 6 步：通过若干个 `CrossScaleFusionBlock`

```python
for block in self.blocks:
    latents = block(latents, scale_tokens)
```

### 第 7 步：构造输出

先转回 channel-first：

```python
fused_latents = latents.transpose(1, 2)   # [B, hidden_dim, num_latents]
```

再做全局平均：

```python
fused_global = latents.mean(dim=1)        # [B, hidden_dim]
```

### 第 8 步：返回

```python
return fused_latents, fused_global
```

---

## 9. 当前默认实例化方式（按已上传规格写死）

```python
cross_scale_fusion = CrossScaleFusion(
    hidden_dim=128,
    num_latents=8,
    num_heads=4,
    ffn_ratio=2.0,
    num_layers=1,
    dropout=0.0,
)
```

当前调用接口固定为：

```python
fused_latents, fused_global = cross_scale_fusion(
    macro_seq=macro_fused,    # [B, 128, 16]
    mezzo_seq=mezzo_fused,    # [B, 128, 24]
    micro_seq=micro_fused,    # [B, 128, 36]
)
```

---

## 10. 初始化与 dtype/device 要求

### 10.1 初始化

- `Linear`、`MultiheadAttention` 使用 PyTorch 默认初始化
- `macro_scale_embedding / mezzo_scale_embedding / micro_scale_embedding` 初始化为全零
- `latents` 初始化为全零

### 10.2 dtype/device

必须满足：

- `fused_latents` 与输入 `macro_seq` 保持同 dtype/device
- `fused_global` 与输入 `macro_seq` 保持同 dtype/device

不要在 `forward()` 里调用 `self.to(...)`。

---

## 11. 错误处理要求

以下情况必须报 `ValueError`：

1. `hidden_dim <= 0`
2. `num_latents <= 0`
3. `num_heads <= 0`
4. `hidden_dim % num_heads != 0`
5. `ffn_ratio <= 0`
6. `num_layers <= 0`
7. `dropout < 0` 或 `dropout >= 1`
8. `_norm_eps <= 0`
9. `macro_seq.ndim != 3`
10. `mezzo_seq.ndim != 3`
11. `micro_seq.ndim != 3`
12. `macro_seq.shape[1] != hidden_dim`
13. `mezzo_seq.shape[1] != hidden_dim`
14. `micro_seq.shape[1] != hidden_dim`
15. `macro_seq.shape[2] != 16`
16. `mezzo_seq.shape[2] != 24`
17. `micro_seq.shape[2] != 36`
18. 三路 batch 不一致

错误信息必须指出：

- 哪个值非法
- 当前值是什么
- 合法范围是什么

---

## 12. smoke test 要求

测试文件建议放在：

```text
tests/models/fusions/test_cross_scale_fusion.py
```

至少包含以下测试：

### 测试 1：shape 正确
```python
macro = torch.randn(2, 128, 16)
mezzo = torch.randn(2, 128, 24)
micro = torch.randn(2, 128, 36)

fusion = CrossScaleFusion()
fused_latents, fused_global = fusion(macro, mezzo, micro)

assert fused_latents.shape == (2, 128, 8)
assert fused_global.shape == (2, 128)
```

### 测试 2：dtype/device 一致
输入是什么 dtype/device，输出就必须是什么 dtype/device。

### 测试 3：非法 macro 长度报错
`macro_seq.shape[2] != 16` 时必须报 `ValueError`

### 测试 4：非法 mezzo 长度报错
`mezzo_seq.shape[2] != 24` 时必须报 `ValueError`

### 测试 5：非法 micro 长度报错
`micro_seq.shape[2] != 36` 时必须报 `ValueError`

### 测试 6：batch 不一致报错
三路 batch 不一致时必须报 `ValueError`

### 测试 7：train / eval 都可 forward
在 `model.train()` 和 `model.eval()` 下都必须正常运行。

---

## 13. 验收标准

实现完成后，以下条件必须同时满足：

1. 本模块只承担 cross-scale fusion 职责
2. 当前版本只消费三路 side-aware scale sequences
3. 输入三路 token 长度完整保留，不做输入压缩
4. 使用 learned scale embeddings 区分三种尺度
5. 使用 learned bottleneck latents 作为融合载体
6. latents 先 cross-attend 三路 tokens，再在 latent 空间 self-attend
7. 输出同时提供 `fused_latents` 与 `fused_global`
8. 所有 smoke test 通过

---

## 14. 最终一句话要求

你要实现的不是一个“再编码三尺度主干”的模块，而是一个：

> **对已经完成 side bridge 的三路尺度序列表示，在完整保留其 token 序列的前提下，使用 learned bottleneck latents 做跨尺度晚融合，并输出融合 latent 序列与全局表示的 cross-scale fusion 模块。**

除此之外，什么都不要做。
