# Network 总装配实现规范（最终版，给 coder 的确定性说明）

本模块的总规则：

- 参数分为两种，一种是**隐参数**，放在 `models/hparams.yaml` 或 `hparams.py`
- 另一种是**开放参数**，放在 `config/models.py`
- 隐参数统一使用 `_` 开头
- 开放参数不使用 `_` 开头
- 本文档只定义 **network 总装配 / forward 链路 / side hierarchy / 输入输出契约**
- 本文档不定义 train loop
- 本文档不定义 data prepare / assembler
- 本文档默认接你已经定下来的子模块实现：
  - `WaveletDenoise1D`
  - `ModernTCNFiLMEncoder`
  - `ConditioningEncoder`
  - `WithinScaleSTARFusion`
  - `ExogenousBridgeFusion`
  - `CrossScaleFusion`
  - `MultiTaskHeads`
  - `MultiTaskDistributionLoss`（仅在 `forward_loss` 中调用，可选）

---

## 0. 给 coder 的确定性 prompt

你要实现的是**完整网络装配模块**，把已经定好的各子模块按固定顺序接起来，形成单次前向预测链路。

你必须严格按本规范实现，不要自行改动模块顺序，不要加入额外 backbone，不要提前做 cross-scale，不要把 loss 强绑进 `forward()`。

### 必须实现的东西

1. 一个 **SideMemoryHierarchy** 模块
2. 一个 **完整网络主类**
3. 网络必须按以下固定顺序执行：

```text
long-window inputs
-> wavelet denoise (3 scales, float only)
-> int crop (3 scales)
-> per-scale ModernTCN-FiLM encoder
-> per-scale STAR within-scale fusion
-> ConditioningEncoder
-> SideMemoryHierarchy 生成 s1/s2/s3 + g1/g2/g3
-> 3 个 ExogenousBridgeFusion
-> CrossScaleFusion
-> MultiTaskHeads
-> raw prediction dict
```

4. 主类必须同时提供：
   - `forward()`：只返回预测 raw 和必要中间结果
   - `forward_loss()`：调用 loss 模块并返回总 loss 字典（可选但推荐）
5. 提供最小必要 smoke tests，覆盖整链 shape 与 key 对齐

### 明确禁止的东西

- 不要把所有逻辑塞进一个超长 `forward()` 而不拆辅助私有函数
- 不要在 network 内重新实现子模块算法
- 不要把 `bridge_global` 送进 `CrossScaleFusion`
- 不要在 network 内做 train loop
- 不要在 `forward()` 里调用 `self.to(...)`
- 不要在 `forward()` 里做 optimizer / scaler / backward
- 不要在这里加数据可视化
- 不要把旧版 DATA_SPEC 里冲突的 shape 继续往下传

这个模块唯一职责：

> 统一接收 network-facing batch，按既定模块顺序完成从输入到三任务 raw 输出的整条前向链路，并提供可选的 loss 对接入口。

---

## 1. 当前总装配所依赖的子模块契约（按已定文档写死）

### 1.1 WaveletDenoise1D

三个尺度分别接收带 warmup 的长窗，输出 crop 后目标窗口：

- macro: 输入 `[B, C, 112]`，输出 `[B, C, 64]`
- mezzo: 输入 `[B, C, 144]`，输出 `[B, C, 96]`
- micro: 输入 `[B, C, 192]`，输出 `[B, C, 144]`

### 1.2 ModernTCNFiLMEncoder

单尺度 encoder 接收：

- `x_float: [B, 11, T]`
- `x_state: [B, T]`
- `x_pos: [B, T]`

输出：

- `z: [B, 11, hidden_dim, num_patches]`

当前默认：
- `hidden_dim = 128`
- `patch_len = 8`
- `patch_stride = 4`

### 1.3 WithinScaleSTARFusion

每尺度输入：

- `z_scale: [B, 11, 128, N]`

输出：

- `z_fused: [B, 11, 128, N]`
- `scale_seq: [B, 128, N]`

当前默认：
- macro `N=16`
- mezzo `N=24`
- micro `N=36`

### 1.4 ConditioningEncoder

输入：

- `x_cond: [B, 8, 64]`

输出：

- `cond_seq: [B, d_cond, 64]`
- `cond_global: [B, d_cond]`

当前默认：
- `d_cond = 32`

### 1.5 ExogenousBridgeFusion

每尺度输入：

- `endogenous_seq: [B, 128, N_endo]`
- `exogenous_seq: [B, 32, N_exo]`
- `exogenous_global: [B, 32]`

输出：

- `endogenous_fused: [B, 128, N_endo]`
- `bridge_global: [B, 128]`

### 1.6 CrossScaleFusion

输入：

- `macro_seq: [B, 128, 16]`
- `mezzo_seq: [B, 128, 24]`
- `micro_seq: [B, 128, 36]`

输出：

- `fused_latents: [B, 128, 8]`
- `fused_global: [B, 128]`

### 1.7 MultiTaskHeads

输入：

- `fused_latents: [B, 128, 8]`
- `fused_global: [B, 128]`

输出字典：

- `pred_mu_ret: [B, 1]`
- `pred_scale_ret_raw: [B, 1]`
- `pred_mean_rv_raw: [B, 1]`
- `pred_shape_rv_raw: [B, 1]`
- `pred_mu_q: [B, 1]`
- `pred_scale_q_raw: [B, 1]`

### 1.8 MultiTaskDistributionLoss

当前 loss 契约：

- `ret`：Student-t NLL，`nu_ret` 为 task-level 全局可学习参数
- `rv`：Gamma NLL
- `q`：ALD NLL

---

## 2. 关于 DATA_SPEC 冲突的明确裁决（必须写死）

当前你上传的 `DATA_SPEC` 存在历史版本残留，至少有以下冲突：

### 2.1 旧版 assembled shape 之一

存在版本写的是：

- `label: [N, 2]`
- `macro: [N, 11, 96]`
- `sidechain: [N, 8, 96]`
- `mezzo: [N, 11, 128]`
- `micro: [N, 11, 192]`
- `macro_i8: [N, 2, 96]`
- `mezzo_i8: [N, 2, 128]`
- `micro_i8: [N, 2, 192]`

### 2.2 另一版 old assembled shape

还存在版本写的是：

- `label: [N, 2]`
- `macro: [N, 9, 112]`
- `sidechain: [N, 13, 112]`
- `mezzo: [N, 9, 144]`
- `micro: [N, 9, 192]`
- `macro_i8: [N, 2, 112]`
- `mezzo_i8: [N, 2, 144]`
- `micro_i8: [N, 2, 192]`

### 2.3 当前 network 的裁决

**network-facing contract 不再跟随这些旧版冲突说明。**  
当前 network 以你已经定下来的模块规格为准，因此统一裁定：

#### 网络前向所需 batch shape（定死）
- `macro_float_long: [B, 11, 112]`
- `macro_i8_long: [B, 2, 112]`
- `mezzo_float_long: [B, 11, 144]`
- `mezzo_i8_long: [B, 2, 144]`
- `micro_float_long: [B, 11, 192]`
- `micro_i8_long: [B, 2, 192]`
- `sidechain_cond: [B, 8, 64]`

#### labels（按当前任务数定死）
- `target_ret: [B, 1]`
- `target_rv: [B, 1]`
- `target_q: [B, 1]`

> 也就是说：train/data 侧稍后必须实现一个 adapter，把当前 assembled 文件读出来后转成上述 **network-facing batch**。  
> network 模块本身不负责兼容旧版 assembled 多种 shape。

---

## 3. Network-facing batch 契约（必须定死）

主类 `forward()` 当前只接受一个字典 `batch`，键必须如下：

```python
{
    "macro_float_long": torch.Tensor,   # [B, 11, 112], float
    "macro_i8_long": torch.Tensor,      # [B, 2, 112], int8/int64

    "mezzo_float_long": torch.Tensor,   # [B, 11, 144], float
    "mezzo_i8_long": torch.Tensor,      # [B, 2, 144], int8/int64

    "micro_float_long": torch.Tensor,   # [B, 11, 192], float
    "micro_i8_long": torch.Tensor,      # [B, 2, 192], int8/int64

    "sidechain_cond": torch.Tensor,     # [B, 8, 64], float

    # 以下 target 仅在 forward_loss 时要求存在
    "target_ret": torch.Tensor,         # [B, 1]
    "target_rv": torch.Tensor,          # [B, 1]
    "target_q": torch.Tensor,           # [B, 1]
}
```

### 3.1 int 序列通道定义（定死）

每个 `*_i8_long` 的两个通道固定为：

- channel 0：`x_state`（即 `f6`，涨跌停 bitmask）
- channel 1：`x_pos`（即 `f7`，步内位置）

### 3.2 dtype 规则（定死）

- `*_float_long`, `sidechain_cond`, `target_*`：`float32`
- `*_i8_long`：`int8` 或 `int64` 均可进入 network，但在 network 内必须转换为 `long`

---

## 4. SideMemoryHierarchy（这是当前缺失但必须补完的模块）

当前在 01~08 文档里，side hierarchy 尚未单独落文档，但这是 network 总装配必须包含的内容。  
本节现在将其**定死**。

### 4.1 目标

输入：

- `cond_seq: [B, 32, 64]`
- `cond_global: [B, 32]`

输出：

- `s1: [B, 32, 64]`
- `g1: [B, 32]`
- `s2: [B, 32, 12]`
- `g2: [B, 32]`
- `s3: [B, 32, 3]`
- `g3: [B, 32]`

### 4.2 语义

- `s1`：完整 64 日驱动记忆
- `s2`：面向 mezzo 的中程驱动记忆
- `s3`：面向 micro 的短程驱动记忆

### 4.3 结构（必须定死）

#### Step 1: cond_seq 自身 self-attention 生成 s1

先转为 time-major：

```python
cond_tokens = cond_seq.transpose(1, 2)   # [B, 64, 32]
```

使用 1 层标准自注意力 block：

```text
cond_tokens
-> LayerNorm
-> SelfAttention
-> residual
-> LayerNorm
-> FFN
-> residual
```

输出：

```python
s1_tokens: [B, 64, 32]
s1 = s1_tokens.transpose(1, 2)   # [B, 32, 64]
```

#### Step 2: g1 的定义（定死）
直接均值池化：

```python
g1 = s1.mean(dim=-1)   # [B, 32]
```

不做 learnable pooling。

#### Step 3: 从 s1 生成 s2（定死）

由于 mezzo 对应 96 个 30min bars = 12 天，因此：

- Query 取 `s1` 的最后 12 天
- Key/Value 取 `s1` 的前 52 天

具体：

```python
q2 = s1_tokens[:, -12:, :]      # [B, 12, 32]
kv2 = s1_tokens[:, :-12, :]     # [B, 52, 32]
```

做一次 cross-attention block：

```text
q2
-> LayerNorm
-> CrossAttention(q=q2, k=kv2, v=kv2)
-> residual
-> LayerNorm
-> FFN
-> residual
```

输出：

```python
s2_tokens: [B, 12, 32]
s2 = s2_tokens.transpose(1, 2)   # [B, 32, 12]
g2 = s2.mean(dim=-1)             # [B, 32]
```

#### Step 4: 从 s2 生成 s3（定死）

由于 micro 对应 144 个 5min bars = 3 天，因此：

- Query 取 `s2` 的最后 3 天
- Key/Value 取 `s2` 的前 9 天

具体：

```python
q3 = s2_tokens[:, -3:, :]      # [B, 3, 32]
kv3 = s2_tokens[:, :-3, :]     # [B, 9, 32]
```

做一次 cross-attention block：

```text
q3
-> LayerNorm
-> CrossAttention(q=q3, k=kv3, v=kv3)
-> residual
-> LayerNorm
-> FFN
-> residual
```

输出：

```python
s3_tokens: [B, 3, 32]
s3 = s3_tokens.transpose(1, 2)   # [B, 32, 3]
g3 = s3.mean(dim=-1)             # [B, 32]
```

### 4.4 必须禁止的东西

- 不要让 `s2` 的 K/V 使用完整 `s1`
- 不要让 `s3` 的 K/V 使用完整 `s2`
- 不要把 `cond_global` 直接当 `g1/g2/g3`
- 不要给 `s1/s2/s3` 加 patching
- 不要把 side hierarchy 写进 `ConditioningEncoder`
- 不要做多层深堆叠，全部固定 1 层

---

## 5. 文件结构（必须按此落位）

```text
src/models/
    networks/
        __init__.py
        side_memory_hierarchy.py
        multi_scale_forecast_network.py
```

### 5.1 `src/models/networks/side_memory_hierarchy.py`

必须包含以下类：

- `SelfAttentionBlock1D`
- `CrossAttentionBlock1D`
- `SideMemoryHierarchy`

### 5.2 `src/models/networks/multi_scale_forecast_network.py`

必须包含以下类：

- `MultiScaleForecastNetwork`

### 5.3 `src/models/networks/__init__.py`

只导出：

```python
from .side_memory_hierarchy import SideMemoryHierarchy
from .multi_scale_forecast_network import MultiScaleForecastNetwork
```

---

## 6. 开放参数与隐参数（network 级）

### 6.1 开放参数（放到 `config/models.py`）

这些参数未来可能调参：

- `hidden_dim`
- `cond_dim`
- `num_latents`

当前默认：
```python
hidden_dim = 128
cond_dim = 32
num_latents = 8
```

### 6.2 隐参数（放到 `src/models/config/hparams.py`）

这些是 network 装配内部固定值：

- `_macro_target_len = 64`
- `_macro_warmup_len = 48`
- `_mezzo_target_len = 96`
- `_mezzo_warmup_len = 48`
- `_micro_target_len = 144`
- `_micro_warmup_len = 48`
- `_side_len = 64`
- `_mezzo_days = 12`
- `_micro_days = 3`
- `_norm_eps = 1e-6`

---

## 7. MultiScaleForecastNetwork 的 `__init__`（必须定死）

主类必须显式实例化以下子模块：

### 7.1 三个 denoise front-end
```python
self.denoise_macro = WaveletDenoise1D(
    n_channels=11,
    target_len=64,
    warmup_len=48,
)
self.denoise_mezzo = WaveletDenoise1D(
    n_channels=11,
    target_len=96,
    warmup_len=48,
)
self.denoise_micro = WaveletDenoise1D(
    n_channels=11,
    target_len=144,
    warmup_len=48,
)
```

### 7.2 三个主干 encoder
```python
self.encoder_macro = ModernTCNFiLMEncoder(...)
self.encoder_mezzo = ModernTCNFiLMEncoder(...)
self.encoder_micro = ModernTCNFiLMEncoder(...)
```

必须使其最终输出符合：
- macro `z_macro [B,11,128,16]`
- mezzo `z_mezzo [B,11,128,24]`
- micro `z_micro [B,11,128,36]`

### 7.3 三个 STAR fusion
```python
self.star_macro = WithinScaleSTARFusion(...)
self.star_mezzo = WithinScaleSTARFusion(...)
self.star_micro = WithinScaleSTARFusion(...)
```

### 7.4 一个 ConditioningEncoder
```python
self.conditioning_encoder = ConditioningEncoder(d_cond=32, ...)
```

### 7.5 一个 SideMemoryHierarchy
```python
self.side_memory_hierarchy = SideMemoryHierarchy(d_cond=32, ...)
```

### 7.6 三个 ExogenousBridgeFusion
```python
self.bridge_macro = ExogenousBridgeFusion(hidden_dim=128, exogenous_dim=32, ...)
self.bridge_mezzo = ExogenousBridgeFusion(hidden_dim=128, exogenous_dim=32, ...)
self.bridge_micro = ExogenousBridgeFusion(hidden_dim=128, exogenous_dim=32, ...)
```

### 7.7 一个 CrossScaleFusion
```python
self.cross_scale_fusion = CrossScaleFusion(hidden_dim=128, num_latents=8, ...)
```

### 7.8 一个 MultiTaskHeads
```python
self.multi_task_heads = MultiTaskHeads(hidden_dim=128, num_latents=8, ...)
```

### 7.9 （可选但推荐）一个 loss 模块
```python
self.loss_fn = MultiTaskDistributionLoss(...)
```

---

## 8. `forward()` 过程（逐步，不可歧义）

### Step 1: 读取 batch

从 `batch` 中读取：

- `macro_float_long`
- `macro_i8_long`
- `mezzo_float_long`
- `mezzo_i8_long`
- `micro_float_long`
- `micro_i8_long`
- `sidechain_cond`

### Step 2: 基础 shape 检查

必须检查：

- `macro_float_long.shape == [B, 11, 112]`
- `macro_i8_long.shape == [B, 2, 112]`
- `mezzo_float_long.shape == [B, 11, 144]`
- `mezzo_i8_long.shape == [B, 2, 144]`
- `micro_float_long.shape == [B, 11, 192]`
- `micro_i8_long.shape == [B, 2, 192]`
- `sidechain_cond.shape == [B, 8, 64]`

不满足必须报 `ValueError`。

### Step 3: 三尺度 float denoise

```python
macro_float = self.denoise_macro(macro_float_long)   # [B,11,64]
mezzo_float = self.denoise_mezzo(mezzo_float_long)   # [B,11,96]
micro_float = self.denoise_micro(micro_float_long)   # [B,11,144]
```

### Step 4: 三尺度 int crop

#### Macro
```python
macro_state = macro_i8_long[:, 0, -64:].long()   # [B,64]
macro_pos   = macro_i8_long[:, 1, -64:].long()   # [B,64]
```

#### Mezzo
```python
mezzo_state = mezzo_i8_long[:, 0, -96:].long()   # [B,96]
mezzo_pos   = mezzo_i8_long[:, 1, -96:].long()   # [B,96]
```

#### Micro
```python
micro_state = micro_i8_long[:, 0, -144:].long()  # [B,144]
micro_pos   = micro_i8_long[:, 1, -144:].long()  # [B,144]
```

### Step 5: 三尺度主干 encoder

```python
z_macro = self.encoder_macro(macro_float, macro_state, macro_pos)   # [B,11,128,16]
z_mezzo = self.encoder_mezzo(mezzo_float, mezzo_state, mezzo_pos)   # [B,11,128,24]
z_micro = self.encoder_micro(micro_float, micro_state, micro_pos)   # [B,11,128,36]
```

### Step 6: 三尺度 STAR

```python
z_macro_fused, scale_seq_macro = self.star_macro(z_macro)   # [B,11,128,16], [B,128,16]
z_mezzo_fused, scale_seq_mezzo = self.star_mezzo(z_mezzo)   # [B,11,128,24], [B,128,24]
z_micro_fused, scale_seq_micro = self.star_micro(z_micro)   # [B,11,128,36], [B,128,36]
```

注意：
- `z_*_fused` 当前仅保留用于调试/可选返回
- 后续主链只消费 `scale_seq_*`

### Step 7: conditioning 编码

```python
cond_seq, cond_global = self.conditioning_encoder(sidechain_cond)   # [B,32,64], [B,32]
```

注意：
- 当前 `cond_global` 不直接送进 bridge
- side hierarchy 会自行生成 `g1/g2/g3`

### Step 8: side hierarchy

```python
s1, g1, s2, g2, s3, g3 = self.side_memory_hierarchy(cond_seq, cond_global)
```

输出必须为：

- `s1 [B,32,64]`
- `g1 [B,32]`
- `s2 [B,32,12]`
- `g2 [B,32]`
- `s3 [B,32,3]`
- `g3 [B,32]`

### Step 9: 三尺度 exogenous bridge

```python
macro_fused, macro_bridge = self.bridge_macro(scale_seq_macro, s1, g1)   # [B,128,16], [B,128]
mezzo_fused, mezzo_bridge = self.bridge_mezzo(scale_seq_mezzo, s2, g2)   # [B,128,24], [B,128]
micro_fused, micro_bridge = self.bridge_micro(scale_seq_micro, s3, g3)   # [B,128,36], [B,128]
```

注意：
- `macro_bridge / mezzo_bridge / micro_bridge` 当前不再继续向下游传给 `CrossScaleFusion`

### Step 10: cross-scale fusion

```python
fused_latents, fused_global = self.cross_scale_fusion(
    macro_seq=macro_fused,
    mezzo_seq=mezzo_fused,
    micro_seq=micro_fused,
)
```

输出：

- `fused_latents [B,128,8]`
- `fused_global [B,128]`

### Step 11: multi-task heads

```python
pred_dict = self.multi_task_heads(fused_latents, fused_global)
```

输出字典必须包含：

- `pred_mu_ret`
- `pred_scale_ret_raw`
- `pred_mean_rv_raw`
- `pred_shape_rv_raw`
- `pred_mu_q`
- `pred_scale_q_raw`

### Step 12: 返回值（定死）

`forward()` 必须返回一个字典，至少包含：

```python
{
    "pred_mu_ret": ...,
    "pred_scale_ret_raw": ...,
    "pred_mean_rv_raw": ...,
    "pred_shape_rv_raw": ...,
    "pred_mu_q": ...,
    "pred_scale_q_raw": ...,
    "fused_latents": ...,
    "fused_global": ...,
}
```

并允许可选返回更多中间结果（通过 `return_aux=True` 控制）：

```python
{
    "scale_seq_macro": ...,
    "scale_seq_mezzo": ...,
    "scale_seq_micro": ...,
    "s1": ..., "g1": ...,
    "s2": ..., "g2": ...,
    "s3": ..., "g3": ...,
    "macro_fused": ...,
    "mezzo_fused": ...,
    "micro_fused": ...,
}
```

默认 `return_aux=False` 时，不要返回这些辅助结果，以减少内存占用。

---

## 9. `forward_loss()`（推荐实现）

主类建议实现：

```python
def forward_loss(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> dict[str, torch.Tensor]:
    ...
```

流程：

1. 调用 `forward(batch, return_aux=return_aux)`
2. 取出：
   - `target_ret`
   - `target_rv`
   - `target_q`
3. 与预测 raw 输出一起送入 `self.loss_fn`
4. 返回 loss 字典，并附加预测字典中的必要值

输出至少包含：

```python
{
    "loss_total": ...,
    "loss_ret": ...,
    "loss_rv": ...,
    "loss_q": ...,
    "nu_ret": ...,
    "sigma_ret_pred": ...,
    "sigma_rv_pred": ...,
    "sigma_q_pred": ...,
    "pred_mu_ret": ...,
    "pred_mean_rv_raw": ...,
    "pred_mu_q": ...,
}
```

---

## 10. 类接口定义（必须完全一致）

### 10.1 `SideMemoryHierarchy`

文件：`src/models/networks/side_memory_hierarchy.py`

```python
class SideMemoryHierarchy(nn.Module):
    def __init__(
        self,
        d_cond: int = 32,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        cond_seq: torch.Tensor,      # [B, d_cond, 64]
        cond_global: torch.Tensor,   # [B, d_cond]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...
```

### 10.2 `MultiScaleForecastNetwork`

文件：`src/models/networks/multi_scale_forecast_network.py`

```python
class MultiScaleForecastNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        cond_dim: int = 32,
        num_latents: int = 8,
        return_aux_default: bool = False,
    ) -> None:
        ...

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        ...

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        ...
```

---

## 11. 初始化与 dtype/device 要求

### 11.1 初始化

所有子模块都使用各自文档规定的默认初始化。

### 11.2 dtype/device

必须满足：

- 输出 dtype/device 与输入 float batch 保持一致
- 不要在 `forward()` 里调用 `self.to(...)`
- int 张量只允许在局部转 `.long()`，不能改模型参数 dtype/device

---

## 12. 错误处理要求

以下情况必须报 `ValueError`：

1. 任何必需 batch key 缺失
2. 任一输入 shape 不符
3. 任一 batch size 不一致
4. 任一 float tensor 不是 3 维
5. 任一 int tensor 不是 3 维
6. `sidechain_cond` 不是 `[B,8,64]`
7. `target_*` 在 `forward_loss()` 中缺失

错误信息必须指出：

- 缺了哪个 key，或
- 哪个张量 shape 错了
- 当前 shape 是什么
- 期望 shape 是什么

---

## 13. smoke test 要求

测试文件建议放在：

```text
tests/models/networks/test_multi_scale_forecast_network.py
```

至少包含以下测试：

### 测试 1：整链 forward 可运行
构造：

- `macro_float_long [2,11,112]`
- `macro_i8_long [2,2,112]`
- `mezzo_float_long [2,11,144]`
- `mezzo_i8_long [2,2,144]`
- `micro_float_long [2,11,192]`
- `micro_i8_long [2,2,192]`
- `sidechain_cond [2,8,64]`

验证：

- forward 输出包含六个预测 raw 键
- `fused_latents.shape == (2, 128, 8)`
- `fused_global.shape == (2, 128)`

### 测试 2：forward_loss 可运行
构造额外：

- `target_ret [2,1]`
- `target_rv [2,1]`
- `target_q [2,1]`

验证：

- 返回 `loss_total`
- 返回 `loss_ret/loss_rv/loss_q`
- 返回 `nu_ret`

### 测试 3：aux 返回可运行
设置 `return_aux=True`，验证：
- 包含 `s1/s2/s3`
- 包含 `macro_fused/mezzo_fused/micro_fused`

### 测试 4：非法 key 报错
缺一个 batch key 必须报 `ValueError`

### 测试 5：非法 shape 报错
例如 `macro_float_long.shape != [B,11,112]` 时必须报 `ValueError`

### 测试 6：train / eval 都可 forward
在 `model.train()` 和 `model.eval()` 下都必须正常运行

---

## 14. 验收标准

实现完成后，以下条件必须同时满足：

1. network 装配顺序严格符合本文档
2. 三尺度 denoise / encoder / STAR / bridge 全部独立实例化
3. `SideMemoryHierarchy` 按固定 64 -> 12 -> 3 路径生成 `s1/s2/s3`
4. `CrossScaleFusion` 只消费三路 bridge 后序列
5. `MultiTaskHeads` 只消费 `fused_latents/fused_global`
6. `MultiTaskDistributionLoss` 只在 `forward_loss()` 中调用
7. 不在 `forward()` 中做任何 train loop 逻辑
8. 所有 smoke tests 通过

---

## 15. 最终一句话要求

你要实现的不是一个“把若干子模块随便串起来”的临时脚本，而是一个：

> **严格对齐当前模块规格的、多尺度去噪 + 单尺度编码 + STAR 晚融合 + side hierarchy + TimeXer-style bridge + cross-scale bottleneck fusion + multi-task heads 的完整 network 装配模块。**

除此之外，什么都不要做。
