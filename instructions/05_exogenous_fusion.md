# ExogenousBridgeFusion 实现规范（TimeXer-style，最终版，给 coder 的确定性说明）

本模块的总规则：

- 参数分为两种，一种是**隐参数**，放在 `models/hparams.yaml` 或 `hparams.py`
- 另一种是**开放参数**，放在 `config/models.py`
- 隐参数统一使用 `_` 开头
- 开放参数不使用 `_` 开头
- 本模块只负责 **单尺度 endogenous 表示 与 单条 exogenous memory 的桥接融合**
- 本模块 **不负责** cross-scale fusion
- 本模块 **不负责** side memory 的层级构造（`s1 / s2 / s3` 的生成不在本模块内）
- 本模块采用 **TimeXer-style endogenous global token bridge** 思路
- 本模块 **不压缩 endogenous 时间/patch 维**
- 本模块 **不要求 endogenous 与 exogenous 的时间长度相同**

---

## 0. 给 coder 的确定性 prompt

你要实现一个 **ExogenousBridgeFusion**，用于把某一尺度的 endogenous 序列表示与一条 exogenous memory 序列做桥接融合。

这里的 “TimeXer-style bridge” 指：

1. 先从 endogenous token 序列中构造一个 **global endogenous bridge token**
2. 再让这个 bridge token 去 cross-attend exogenous memory
3. 得到吸收了 exogenous 信息的 bridge token
4. 再把这个 bridge token 回灌到原始 endogenous tokens 中
5. 输出融合后的 endogenous 序列表示

你必须严格按本规范实现，不要自行补充结构，不要把它改成完整模型，不要把 cross-scale 逻辑塞进来。

### 必须实现的东西

1. 一个 `ExogenousBridgeFusion` 类
2. 输入三路：
   - `endogenous_seq: [B, hidden_dim, N_endo]`
   - `exogenous_seq: [B, exogenous_dim, N_exo]`
   - `exogenous_global: [B, exogenous_dim]`
3. 输出两路：
   - `endogenous_fused: [B, hidden_dim, N_endo]`
   - `bridge_global: [B, hidden_dim]`
4. 内部使用一个 `ExogenousBridgeBlock`
5. 使用标准 `nn.MultiheadAttention(batch_first=True)` 做 bridge token 到 exogenous memory 的 cross-attention
6. 使用 `exogenous_global` 生成全局 gate，对 bridge 回灌分支做门控
7. 保留 endogenous 的完整时间/patch 维 `N_endo`

### 明确禁止的东西

- 不要做 cross-scale fusion
- 不要把 `macro / mezzo / micro` 三路一起输入本模块
- 不要在本模块内部生成 `s1 / s2 / s3`
- 不要加 self-attention 到 endogenous tokens 上
- 不要加 self-attention 到 exogenous tokens 上
- 不要压缩 endogenous 时间/patch 维
- 不要要求 endogenous / exogenous 时间长度一致
- 不要加 prediction head
- 不要加 patching
- 不要把它写成完整 TimeXer 模型
- 不要加入额外复杂路由、MoE、图结构

这个模块唯一职责：

> 输入一条 endogenous 序列表示和一条 exogenous memory，使用 TimeXer-style global bridge token 做单次桥接融合，并输出融合后的 endogenous 序列表示。

---

## 1. 本规范依赖的上游接口（按当前已上传规格写死）

### 1.1 endogenous 输入来源

本模块当前对接的是 `WithinScaleSTARFusion` 输出的 `scale_seq`：

```text
scale_seq: [B, hidden_dim, N]
```

当前默认三个尺度分别为：

- macro: `[B, 128, 16]`
- mezzo: `[B, 128, 24]`
- micro: `[B, 128, 36]`

这里的 `N` 必须完整保留。

### 1.2 exogenous 输入来源

本模块当前对接的是 side / conditioning 分支的层级 memory：

- `s1`: `[B, d_cond, N_s1]`
- `s2`: `[B, d_cond, N_s2]`
- `s3`: `[B, d_cond, N_s3]`

以及相应的全局向量：

- `g1`: `[B, d_cond]`
- `g2`: `[B, d_cond]`
- `g3`: `[B, d_cond]`

注意：

- `s1 / s2 / s3` 的生成不属于本模块
- 本模块只接收一条当前尺度对应的 `exogenous_seq` 和 `exogenous_global`

### 1.3 当前典型调用方式

#### Macro
```python
macro_fused, macro_bridge = fusion_macro(
    endogenous_seq=scale_seq_macro,   # [B, 128, 16]
    exogenous_seq=s1,                 # [B, d_cond, N_s1]
    exogenous_global=g1,              # [B, d_cond]
)
```

#### Mezzo
```python
mezzo_fused, mezzo_bridge = fusion_mezzo(
    endogenous_seq=scale_seq_mezzo,   # [B, 128, 24]
    exogenous_seq=s2,                 # [B, d_cond, N_s2]
    exogenous_global=g2,              # [B, d_cond]
)
```

#### Micro
```python
micro_fused, micro_bridge = fusion_micro(
    endogenous_seq=scale_seq_micro,   # [B, 128, 36]
    exogenous_seq=s3,                 # [B, d_cond, N_s3]
    exogenous_global=g3,              # [B, d_cond]
)
```

---

## 2. 模块职责边界

### 2.1 本模块负责的事情

本模块只负责：

1. 从 endogenous token 序列生成一个 bridge token
2. 让 bridge token 去 cross-attend exogenous memory
3. 用 `exogenous_global` 生成全局 gate
4. 用 bridge token 回灌 endogenous tokens
5. 输出：
   - `endogenous_fused`
   - `bridge_global`

### 2.2 本模块不负责的事情

以下事情不属于本模块：

- side hierarchy 构造
- s1/s2/s3 递归生成
- cross-scale fusion
- prediction head
- denoise
- label 预测
- loss 计算

---

## 3. 设计决策（全部定死）

### 3.1 TimeXer-style bridge 的当前解释

本规范中的 TimeXer-style bridge 具体解释为：

1. endogenous tokens 先通过平均汇聚构造一个 **endogenous bridge token**
2. 再加上一个 **learnable bridge token parameter**
3. 这个 bridge token 作为 query，去 cross-attend exogenous tokens
4. 得到更新后的 bridge token
5. 再把更新后的 bridge token 复制回每个 endogenous token，并通过 MLP 融合回灌

### 3.2 endogenous 时间/patch 维必须完整保留

本模块绝不压缩 `N_endo`：

- 输入 `endogenous_seq [B, D, N_endo]`
- 输出 `endogenous_fused [B, D, N_endo]`

### 3.3 exogenous 时间长度允许变化

本模块必须支持任意 `N_exo >= 1`：

- 不要求等于 `N_endo`
- 不要求固定长度
- 只要求 batch 对齐

### 3.4 bridge token 数固定为 1

当前版本只允许：

- 单个 bridge token

不要改成多个 learned bridge tokens。

### 3.5 当前只做单层 bridge

当前默认只做：

- `num_layers = 1`

虽然允许参数上设为 `>1`，但默认与推荐值均为 1。

---

## 4. 文件结构（必须按此落位）

```text
src/models/
    fusions/
        __init__.py
        exogenous_bridge_fusion.py
```

### 4.1 `src/models/fusions/exogenous_bridge_fusion.py`

必须包含以下类：

- `ExogenousBridgeBlock`
- `ExogenousBridgeFusion`

### 4.2 `src/models/fusions/__init__.py`

只导出：

```python
from .exogenous_bridge_fusion import ExogenousBridgeFusion
```

---

## 5. 参数分类（必须明确）

### 5.1 开放参数（放到 `config/models.py`）

这些参数是未来可能调的，不加 `_` 前缀：

- `hidden_dim`
- `exogenous_dim`
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
exogenous_dim = 32
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

## 6.1 `ExogenousBridgeBlock`

文件：`src/models/fusions/exogenous_bridge_fusion.py`

```python
class ExogenousBridgeBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        exogenous_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        endogenous_tokens: torch.Tensor,
        exogenous_tokens: torch.Tensor,
        exogenous_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

### 输入

- `endogenous_tokens`: `[B, N_endo, hidden_dim]`
- `exogenous_tokens`: `[B, N_exo, exogenous_dim]`
- `exogenous_global`: `[B, exogenous_dim]`

### 输出

- `endogenous_out`: `[B, N_endo, hidden_dim]`
- `bridge_global`: `[B, hidden_dim]`

---

## 6.2 `ExogenousBridgeFusion`

文件：`src/models/fusions/exogenous_bridge_fusion.py`

```python
class ExogenousBridgeFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        exogenous_dim: int = 32,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        num_layers: int = 1,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        endogenous_seq: torch.Tensor,
        exogenous_seq: torch.Tensor,
        exogenous_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

### 输入

- `endogenous_seq`: `[B, hidden_dim, N_endo]`
- `exogenous_seq`: `[B, exogenous_dim, N_exo]`
- `exogenous_global`: `[B, exogenous_dim]`

### 输出

- `endogenous_fused`: `[B, hidden_dim, N_endo]`
- `bridge_global`: `[B, hidden_dim]`

---

## 7. `ExogenousBridgeBlock` 的结构（必须定死）

### 7.1 总体流程

每个 block 必须严格按下面顺序实现：

```text
endogenous tokens
-> build bridge token from endogenous mean + learned token
-> cross-attend bridge token to exogenous tokens
-> residual update on bridge token
-> gate generation from exogenous_global
-> repeat bridge token to endogenous length
-> concat(endogenous tokens, repeated bridge token)
-> fuse MLP
-> gate the fuse delta
-> residual add back to endogenous tokens
```

### 7.2 具体实现要求

#### 第 1 步：归一化 exogenous tokens

使用：

```python
self.exo_proj = nn.Linear(exogenous_dim, hidden_dim)
self.exo_norm = nn.LayerNorm(hidden_dim, eps=_norm_eps)
```

先做：

```python
exo = self.exo_proj(exogenous_tokens)     # [B, N_exo, hidden_dim]
exo = self.exo_norm(exo)
```

#### 第 2 步：构造 bridge token

定义一个可学习参数：

```python
self.bridge_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
```

然后：

```python
endo_mean = endogenous_tokens.mean(dim=1, keepdim=True)     # [B, 1, hidden_dim]
bridge0 = endo_mean + self.bridge_token                     # [B, 1, hidden_dim]
```

不要改成 max pooling，不要改成 attention pooling。

#### 第 3 步：bridge token cross-attend exogenous tokens

使用：

```python
self.bridge_norm = nn.LayerNorm(hidden_dim, eps=_norm_eps)
self.cross_attn = nn.MultiheadAttention(
    embed_dim=hidden_dim,
    num_heads=num_heads,
    dropout=dropout,
    batch_first=True,
)
```

然后：

```python
q = self.bridge_norm(bridge0)   # [B, 1, hidden_dim]
bridge_delta, _ = self.cross_attn(q, exo, exo, need_weights=False)
bridge1 = bridge0 + bridge_delta
```

得到：

```text
bridge1: [B, 1, hidden_dim]
```

#### 第 4 步：从 exogenous_global 生成全局 gate

使用：

```python
self.global_gate = nn.Linear(exogenous_dim, hidden_dim)
```

然后：

```python
gate = torch.sigmoid(self.global_gate(exogenous_global)).unsqueeze(1)   # [B, 1, hidden_dim]
```

不要改成两层 MLP，不要改成 softmax gate。

#### 第 5 步：bridge token 回灌 endogenous tokens

先复制：

```python
bridge_rep = bridge1.expand(-1, endogenous_tokens.shape[1], -1)   # [B, N_endo, hidden_dim]
```

再拼接：

```python
fuse_in = torch.cat([endogenous_tokens, bridge_rep], dim=-1)      # [B, N_endo, 2*hidden_dim]
```

#### 第 6 步：fuse MLP

使用：

```python
_ffn_dim = int(hidden_dim * ffn_ratio)

self.fuse_mlp = nn.Sequential(
    nn.LayerNorm(2 * hidden_dim, eps=_norm_eps),
    nn.Linear(2 * hidden_dim, _ffn_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(_ffn_dim, hidden_dim),
    nn.Dropout(dropout),
)
```

然后：

```python
delta = self.fuse_mlp(fuse_in)     # [B, N_endo, hidden_dim]
delta = delta * gate               # [B, N_endo, hidden_dim]
endogenous_out = endogenous_tokens + delta
```

### 7.3 输出 bridge_global

定义：

```python
bridge_global = bridge1.squeeze(1)   # [B, hidden_dim]
```

返回：

```python
return endogenous_out, bridge_global
```

---

## 8. `ExogenousBridgeFusion` 的 forward 过程（逐步，不可歧义）

### 第 1 步：输入形状检查

检查：

- `endogenous_seq.ndim == 3`
- `exogenous_seq.ndim == 3`
- `exogenous_global.ndim == 2`

记：

- `endogenous_seq.shape == [B, D, N_endo]`
- `exogenous_seq.shape == [B, E, N_exo]`
- `exogenous_global.shape == [B, E]`

并检查：

- `D == self.hidden_dim`
- `E == self.exogenous_dim`
- 三路 batch 大小一致

若不满足，必须抛出 `ValueError`

### 第 2 步：转成 time-major

```python
endo = endogenous_seq.transpose(1, 2)   # [B, N_endo, hidden_dim]
exo = exogenous_seq.transpose(1, 2)     # [B, N_exo, exogenous_dim]
```

### 第 3 步：通过若干个 bridge block

```python
for block in self.blocks:
    endo, bridge_global = block(
        endogenous_tokens=endo,
        exogenous_tokens=exo,
        exogenous_global=exogenous_global,
    )
```

### 第 4 步：转回输出格式

```python
endogenous_fused = endo.transpose(1, 2)   # [B, hidden_dim, N_endo]
```

### 第 5 步：返回

```python
return endogenous_fused, bridge_global
```

---

## 9. 当前典型实例化方式（按已上传规格写死）

### 9.1 Macro bridge

```python
bridge_macro = ExogenousBridgeFusion(
    hidden_dim=128,
    exogenous_dim=32,
    num_heads=4,
    ffn_ratio=2.0,
    num_layers=1,
    dropout=0.0,
)
```

用于：

- `endogenous_seq = scale_seq_macro [B, 128, 16]`
- `exogenous_seq = s1 [B, 32, N_s1]`
- `exogenous_global = g1 [B, 32]`

### 9.2 Mezzo bridge

```python
bridge_mezzo = ExogenousBridgeFusion(
    hidden_dim=128,
    exogenous_dim=32,
    num_heads=4,
    ffn_ratio=2.0,
    num_layers=1,
    dropout=0.0,
)
```

用于：

- `endogenous_seq = scale_seq_mezzo [B, 128, 24]`
- `exogenous_seq = s2 [B, 32, N_s2]`
- `exogenous_global = g2 [B, 32]`

### 9.3 Micro bridge

```python
bridge_micro = ExogenousBridgeFusion(
    hidden_dim=128,
    exogenous_dim=32,
    num_heads=4,
    ffn_ratio=2.0,
    num_layers=1,
    dropout=0.0,
)
```

用于：

- `endogenous_seq = scale_seq_micro [B, 128, 36]`
- `exogenous_seq = s3 [B, 32, N_s3]`
- `exogenous_global = g3 [B, 32]`

---

## 10. 初始化与 dtype/device 要求

### 10.1 初始化

- `Linear`、`MultiheadAttention` 使用 PyTorch 默认初始化
- `bridge_token` 初始化为全零：
  ```python
  nn.Parameter(torch.zeros(1, 1, hidden_dim))
  ```

### 10.2 dtype/device

必须满足：

- `endogenous_fused` 与 `endogenous_seq` 保持同 dtype/device
- `bridge_global` 与 `endogenous_seq` 保持同 dtype/device

不要在 `forward()` 里调用 `self.to(...)`。

---

## 11. 错误处理要求

以下情况必须报 `ValueError`：

1. `hidden_dim <= 0`
2. `exogenous_dim <= 0`
3. `num_heads <= 0`
4. `hidden_dim % num_heads != 0`
5. `ffn_ratio <= 0`
6. `num_layers <= 0`
7. `dropout < 0` 或 `dropout >= 1`
8. `_norm_eps <= 0`
9. `endogenous_seq.ndim != 3`
10. `exogenous_seq.ndim != 3`
11. `exogenous_global.ndim != 2`
12. `endogenous_seq.shape[1] != hidden_dim`
13. `exogenous_seq.shape[1] != exogenous_dim`
14. `exogenous_global.shape[1] != exogenous_dim`
15. 三路 batch 不一致
16. `endogenous_seq.shape[2] <= 0`
17. `exogenous_seq.shape[2] <= 0`

错误信息必须指出：

- 哪个值非法
- 当前值是什么
- 合法范围是什么

---

## 12. smoke test 要求

测试文件建议放在：

```text
tests/models/fusions/test_exogenous_bridge_fusion.py
```

至少包含以下测试：

### 测试 1：macro 形状正确
```python
endo = torch.randn(2, 128, 16)
exo = torch.randn(2, 32, 64)
exo_g = torch.randn(2, 32)

fusion = ExogenousBridgeFusion()
endo_fused, bridge_global = fusion(endo, exo, exo_g)

assert endo_fused.shape == (2, 128, 16)
assert bridge_global.shape == (2, 128)
```

### 测试 2：mezzo 形状正确（变长 exogenous）
```python
endo = torch.randn(2, 128, 24)
exo = torch.randn(2, 32, 12)
exo_g = torch.randn(2, 32)

fusion = ExogenousBridgeFusion()
endo_fused, bridge_global = fusion(endo, exo, exo_g)

assert endo_fused.shape == (2, 128, 24)
assert bridge_global.shape == (2, 128)
```

### 测试 3：micro 形状正确（更短 exogenous）
```python
endo = torch.randn(2, 128, 36)
exo = torch.randn(2, 32, 3)
exo_g = torch.randn(2, 32)

fusion = ExogenousBridgeFusion()
endo_fused, bridge_global = fusion(endo, exo, exo_g)

assert endo_fused.shape == (2, 128, 36)
assert bridge_global.shape == (2, 128)
```

### 测试 4：dtype/device 一致
输入是什么 dtype/device，输出就必须是什么 dtype/device。

### 测试 5：非法 hidden_dim 报错
`endogenous_seq.shape[1] != hidden_dim` 时必须报 `ValueError`

### 测试 6：非法 exogenous_dim 报错
`exogenous_seq.shape[1] != exogenous_dim` 或 `exogenous_global.shape[1] != exogenous_dim` 时必须报 `ValueError`

### 测试 7：batch 不一致报错
三路 batch 不一致时必须报 `ValueError`

### 测试 8：train / eval 都可 forward
在 `model.train()` 和 `model.eval()` 下都必须正常运行。

---

## 13. 验收标准

实现完成后，以下条件必须同时满足：

1. 本模块只承担单尺度 endogenous-exogenous bridge fusion 职责
2. endogenous 时间/patch 维 `N_endo` 完整保留
3. exogenous 时间长度 `N_exo` 允许变化
4. 使用单个 bridge token
5. 使用 `MultiheadAttention` 做 bridge token 到 exogenous memory 的 cross-attention
6. 使用 `exogenous_global` 生成全局 gate
7. 不消费 cross-scale 输入
8. 所有 smoke test 通过

---

## 14. 最终一句话要求

你要实现的不是一个“完整 TimeXer”，而是一个：

> **对单尺度 endogenous 序列表示 `[B, D, N_endo]` 和一条 exogenous memory `[B, E, N_exo]`，使用单个 endogenous bridge token 做 TimeXer-style bridge fusion，并完整保留 endogenous 时间/patch 维的融合模块。**

除此之外，什么都不要做。
