本模块的总规则：
参数分为两种，一种是隐参数，放在src/models/config/hparams.py
另一种是开放到config/models.py里的参数，这种参数应该是有必要调参的参数。

# WaveletDenoise1D 实现规范（给 coder 的确定性说明，最终版）

## 0. 给 coder 的确定性 prompt

你要实现一个**纯 denoise front-end**，用于金融时序模型输入前的去噪处理。

这是一个**职责单一**的模块，不允许承担任何 encoding、embedding、feature extraction、cross-scale fusion、attention、conv backbone、tokenization、head 输出等职责。

你必须严格按本规范实现，不要自行补充设计，不要替换算法，不要更改输入输出语义，不要加入“你认为更好”的结构。

### 必须实现的东西

1. 一个 `WaveletDenoise1D` 类，输入 `[B, C, T_total]`，输出 `[B, C, T_target]`。
2. 使用**固定小波基**做 1D 多层小波分解与重建。
3. 仅对 **detail subbands** 做**可学习三参数 shrinkage**。
4. **approximation subband 不做非线性处理，不做可学习修正**，保持原样。
5. 使用**真实历史 warmup**，不做 artificial left pad。
6. 在 denoise 完成后，只保留最后 `target_len` 段输出。
7. 三个尺度（macro / mezzo / micro）分别实例化，不共享参数。
8. 提供最小必要的单元测试，验证输入输出长度、shape、一致性和 warmup crop 逻辑。

### 明确禁止的东西

- 不要加 encoder
- 不要加 conv block
- 不要加 attention
- 不要加 hidden projection
- 不要加 token mixer
- 不要加 cross-attention
- 不要加多分支融合
- 不要加 residual head
- 不要输出 embedding
- 不要把它做成 backbone
- 不要加入可学习小波滤波器
- 不要把 wavelet transform 改成 WNO / MAWNO / PIWNO / WPT / scattering / STFT
- 不要自行增加新的 learnable 参数类型（例如 band mixing MLP、coefficient attention、low-pass gain、noise estimator）

这个模块唯一职责：

> 输入一段“真实历史 warmup + 目标窗口”的时间序列，对其进行固定小波分解，对 detail 子带做可学习三参数收缩，再做逆变换，最后裁出目标窗口长度，返回 denoised 结果。

---

## 1. 模块职责边界

### 1.1 模块职责

本模块只负责：

1. 接收一个长度为 `target_len + warmup_len` 的时间序列。
2. 在时间维上做固定小波分解。
3. 对每一层 detail 子带做可学习三参数 shrinkage。
4. 用逆小波重建回时域序列。
5. 裁出最后 `target_len` 的有效目标窗口。
6. 返回 `x_denoised`。

### 1.2 本模块不负责的事情

以下功能一律不属于本模块：

- 表征学习
- 编码语义
- 生成 embedding
- 跨尺度信息交互
- sidechain 调制
- feature fusion
- 主干时序建模
- 预测输出
- 不确定性估计
- K 线几何修正

这些职责必须留给后续主模型或数据处理层，不允许塞进本模块。

---

## 2. 固定设计决策（不得修改）

以下配置全部定死：

### 2.1 算法形态

使用：

- 固定小波基的 **1D 多层 DWT/IDWT**
- 可学习 **detail-band 三参数 shrinkage**
- approximation band 原样保留

### 2.2 小波库

实现必须基于 **ptwt**（PyTorch Wavelet Toolbox）。

原因：

- 直接工作在 PyTorch 张量上
- forward 路径可放在模型里统一执行
- 不需要离线预处理整段序列
- 与训练/验证/推理路径一致

不要改用 PyWavelets 作为主实现路径。

### 2.3 输入格式

输入张量固定为：

- `x_long: torch.Tensor`
- shape: `[B, C, T_total]`

其中：

- `B` = batch size
- `C` = feature / channel 数
- `T_total = target_len + warmup_len`

### 2.4 输出格式

输出张量固定为：

- `y: torch.Tensor`
- shape: `[B, C, target_len]`

### 2.5 时间维定义

小波分解必须沿最后一维时间维执行，即：

- 输入 `[B, C, T]`
- 对每个 batch、每个通道独立做 1D wavelet transform

不允许在通道维上做小波，不允许把通道维和时间维混合。

### 2.6 warmup 处理

本项目**已经提供真实历史 warmup**，因此：

- 不做 artificial left pad
- 不做 reflection / symmetric / periodization pad 设计
- 不在模块内部补造历史

模块只吃真实的 `target + warmup` 长窗，然后在最后裁切。

### 2.7 v2 固定超参数

模块默认固定配置如下：

- `wavelet = "db4"`
- `level = 2`
- `shrink_mode = "soft"`
- `learn_lowpass = False`
- `learn_gain = True`
- `learn_bypass = True`
- `eps = 1e-6`

不允许 coder 自行改为别的小波、别的 level、别的 shrinkage 形式，除非上层配置显式传入，但默认值必须是上述内容。

---

## 3. 三个尺度的确定输入输出

### 3.1 Macro

输入：

- 目标长度 `target_len = 64`
- warmup 长度 `warmup_len = 48`
- 总长度 `T_total = 112`

shape：

- 输入 `[B, C_macro, 112]`
- 输出 `[B, C_macro, 64]`

### 3.2 Mezzo

输入：

- 目标长度 `target_len = 96`
- warmup 长度 `warmup_len = 48`
- 总长度 `T_total = 144`

shape：

- 输入 `[B, C_mezzo, 144]`
- 输出 `[B, C_mezzo, 96]`

### 3.3 Micro

输入：

- 目标长度 `target_len = 144`
- warmup 长度 `warmup_len = 48`
- 总长度 `T_total = 192`

shape：

- 输入 `[B, C_micro, 192]`
- 输出 `[B, C_micro, 144]`

### 3.4 参数共享规则

三者：

- 代码结构相同
- 类相同
- 参数**不共享**

也就是说，必须实例化三个对象：

- `denoise_macro`
- `denoise_mezzo`
- `denoise_micro`

不能共享 `theta/phi/psi` 参数。

---

## 4. 文件结构（必须按此落位）

```text
src/models/utils/
    __init__.py
    wavelet_denoise.py
```

### 4.1 `wavelet_denoise.py`

必须包含：

- `class WaveletDenoise1D(nn.Module)`
- 一个内部静态/私有函数用于 `rms` 计算
- 一个内部静态/私有函数用于 `soft threshold shrinkage`

### 4.2 `__init__.py`

只导出：

```python
from .wavelet_denoise import WaveletDenoise1D
```

### 4.3 smoke test

必须包含最少 5 个测试：

1. shape 测试
2. crop 长度测试
3. detail shrinkage 后输出仍与输入 dtype/device 一致
4. 三个尺度配置实例可正常 forward
5. 非法输入长度时正确报错

---

## 5. 类接口定义（必须完全一致）

```python
class WaveletDenoise1D(nn.Module):
    def __init__(
        self,
        n_channels: int,
        target_len: int,
        warmup_len: int,
        wavelet: str = "db4",
        level: int = 2,
        eps: float = 1e-6,
    ) -> None:
        ...

    def forward(self, x_long: torch.Tensor) -> torch.Tensor:
        ...
```

### 5.1 构造参数语义

- `n_channels`: 输入通道数 `C`
- `target_len`: 最终输出保留的目标长度
- `warmup_len`: 输入中历史 warmup 长度
- `wavelet`: 默认 `"db4"`
- `level`: 默认 `2`
- `eps`: 数值稳定常数

### 5.2 forward 输入要求

`x_long` 必须满足：

- `x_long.ndim == 3`
- `x_long.shape == [B, C, target_len + warmup_len]`
- `x_long.shape[1] == n_channels`

若不满足，必须 `raise ValueError`，报错信息必须清楚指出期望 shape 与实际 shape。

### 5.3 forward 输出要求

返回值 `y` 必须满足：

- `y.ndim == 3`
- `y.shape == [B, C, target_len]`
- `y.device == x_long.device`
- `y.dtype == x_long.dtype`

---

## 6. 可学习参数定义（必须完全一致）

本模块唯一允许的可学习参数共有三类，全部只作用于 detail 子带：

```python
theta_detail: nn.Parameter  # shape [level, n_channels]
phi_detail: nn.Parameter    # shape [level, n_channels]
psi_detail: nn.Parameter    # shape [level, n_channels]
```

除这三类参数外，不要定义别的 learnable 参数。

### 6.1 参数语义

`theta_detail[j, c]` 对应：

- 第 `j` 个 detail level
- 第 `c` 个通道
- 一个阈值强度参数

`phi_detail[j, c]` 对应：

- 第 `j` 个 detail level
- 第 `c` 个通道
- shrink 后 detail 的保留增益参数

`psi_detail[j, c]` 对应：

- 第 `j` 个 detail level
- 第 `c` 个通道
- 原始 detail 旁路比例参数

注意：

- 这里的 `j` 是实现内部对 detail 子带列表的索引
- 只要求前后保持一致
- 不要在文档里再自行更改 level 顺序约定

### 6.2 参数到实际量的映射

实际阈值由：

```python
tau = softplus(theta)
```

得到，确保非负。

shrink 后增益由：

```python
alpha = sigmoid(phi)
```

得到，范围在 `(0, 1)`。

原始 detail 旁路比例由：

```python
rho = sigmoid(psi)
```

得到，范围在 `(0, 1)`。

然后阈值乘以当前子带的通道内尺度：

```python
thr = tau * sigma
```

其中 `sigma` 是该 detail 子带的 RMS 估计。

---

## 7. 中间过程（逐步、不可歧义）

以下 forward 过程必须逐步实现，不允许删改语义。

### 第 1 步：校验输入形状

读取：

- `B, C, T_total = x_long.shape`

检查：

- `C == self.n_channels`
- `T_total == self.target_len + self.warmup_len`

若不满足，抛出 `ValueError`。

### 第 2 步：执行 1D 多层小波分解

对 `x_long` 在时间维做多层 DWT。

输出系数列表约定为：

```text
coeffs = [a_L, d_L, d_{L-1}, ..., d_1]
```

其中：

- `a_L` = 最顶层 approximation coefficients
- `d_*` = 各层 detail coefficients

不要改变该语义。

### 第 3 步：保留 approximation 子带

```python
a = coeffs[0]
```

并且：

- 不做阈值
- 不做 gain
- 不做非线性
- 不做 learnable 修正

也就是直接原样使用。

### 第 4 步：遍历所有 detail 子带

对 `coeffs[1:]` 中的每个 detail 子带，逐个执行：

#### 4.1 计算 RMS 尺度

若当前子带为 `d`，shape 可记作 `[B, C, T_j]`，则：

```python
sigma = sqrt(mean(d^2, dim=-1, keepdim=True)).clamp_min(eps)
```

要求：

- 只沿时间维 `dim=-1` 统计
- 保留维度 `keepdim=True`
- 结果 shape 必须为 `[B, C, 1]`

#### 4.2 读取本层参数

```python
tau = F.softplus(self.theta_detail[j]).view(1, C, 1)
alpha = torch.sigmoid(self.phi_detail[j]).view(1, C, 1)
rho = torch.sigmoid(self.psi_detail[j]).view(1, C, 1)
```

其中：

- `tau` shape 必须为 `[1, C, 1]`
- `alpha` shape 必须为 `[1, C, 1]`
- `rho` shape 必须为 `[1, C, 1]`
- 要与 `sigma` 广播兼容

#### 4.3 构造最终阈值

```python
thr = tau * sigma
```

shape 必须为：

- `[B, C, 1]`

#### 4.4 执行 soft-threshold shrinkage

先对 detail 子带逐元素执行 soft-threshold：

```python
d_shrunk = sign(d) * relu(abs(d) - thr)
```

然后执行三参数组合输出：

```python
d_new = rho * d + (1.0 - rho) * (alpha * d_shrunk)
```

必须严格使用这两个步骤，不允许替换为别的 shrinkage 公式。

不要：

- 改成 hard threshold
- 改成 garrote
- 改成 band mixing
- 改成 MLP threshold predictor
- 改成 coefficient attention
- 改成额外 conv/encoder

#### 4.5 收集处理后的 detail 子带

将每层 `d_new` 按原顺序存回列表。

顺序必须与 `waverec` 所需顺序完全一致。

### 第 5 步：执行逆小波重建

将：

```python
[a] + new_details
```

送入逆变换，得到：

```python
y_long
```

要求：

- `y_long` 是完整长度重建结果
- 时间长度至少应覆盖原始 `T_total`
- 若因库实现边界行为导致长度略有差异，必须在后处理阶段严格裁齐

### 第 6 步：对重建结果做长度裁齐

先将 `y_long` 对齐到输入长度 `T_total`，然后只保留最后 `target_len` 段：

```python
y = y_long[..., -self.target_len:]
```

注意：

- 这里的裁切语义固定为“保留末尾目标段”
- 绝对不能裁前面
- 绝对不能返回整段

### 第 7 步：返回结果

返回：

```python
y
```

shape 为：

- `[B, C, target_len]`

---

## 8. 数学定义（用于避免 coder 误解）

给定输入：

- `x ∈ R^{B×C×T_total}`

经多层小波分解得到：

- `a_L`
- `d_L, d_{L-1}, ..., d_1`

定义每层每通道参数：

- `θ_{j,c}` 为阈值强度参数
- `φ_{j,c}` 为 shrink 后增益参数
- `ψ_{j,c}` 为原始 detail 旁路比例参数
- `τ_{j,c} = softplus(θ_{j,c})`
- `α_{j,c} = sigmoid(φ_{j,c})`
- `ρ_{j,c} = sigmoid(ψ_{j,c})`

定义当前样本、当前通道、当前 detail 子带尺度：

- `σ_{b,c,j} = sqrt(mean_t(d_{b,c,j,t}^2)) + eps`

定义实际阈值：

- `thr_{b,c,j} = τ_{j,c} * σ_{b,c,j}`

定义 soft-threshold 后的 detail 系数：

- `d_shrunk = sign(d) * max(|d| - thr, 0)`

定义最终输出的 detail 系数：

- `d_out = ρ * d + (1 - ρ) * (α * d_shrunk)`

最终：

- approximation 子带 `a_L` 原样保留
- detail 子带全部替换为 `d_out`
- 使用逆小波变换重建得到 `y_long`
- 返回 `y = y_long[..., -target_len:]`

---

## 9. 初始化要求

### 9.1 参数初始化

以下三个参数初始化必须全部为全零：

```python
self.theta_detail = nn.Parameter(torch.zeros(level, n_channels))
self.phi_detail = nn.Parameter(torch.zeros(level, n_channels))
self.psi_detail = nn.Parameter(torch.zeros(level, n_channels))
```

含义：

- 初始 `tau = softplus(0) ≈ 0.693`
- 初始 `alpha = sigmoid(0) = 0.5`
- 初始 `rho = sigmoid(0) = 0.5`
- 初始状态为“适中阈值 + 适中 shrink 保留 + 一半原始旁路”
- 不要自行改成随机初始化

### 9.2 数值稳定

`eps` 默认：

```python
eps = 1e-6
```

必须在 `sigma` 计算后做 `clamp_min(eps)`。

---

## 10. 类型与设备要求

实现必须满足：

- 支持 CPU / CUDA
- 保持输入 dtype 不变
- 保持输入 device 不变

不要在模块内强行 `.float()`、`.cpu()` 或 `.cuda()`。

若底层 wavelet 库要求特殊处理，必须显式保证输出最终恢复为输入 dtype/device。

---

## 11. 错误处理要求

以下情况必须报错：

1. `x_long.ndim != 3`
2. 输入通道数不等于 `n_channels`
3. 输入长度不等于 `target_len + warmup_len`
4. `target_len <= 0`
5. `warmup_len < 0`
6. `level <= 0`
7. `n_channels <= 0`

报错类型统一为：

- `ValueError`

报错信息必须明确指出：

- 哪个参数非法
- 当前值是什么
- 期望范围是什么

---

## 12. 三个尺度的实例化方式（必须明确给出）

### Macro

```python
denoise_macro = WaveletDenoise1D(
    n_channels=C_macro,
    target_len=64,
    warmup_len=48,
    wavelet="db4",
    level=2,
)
```

### Mezzo

```python
denoise_mezzo = WaveletDenoise1D(
    n_channels=C_mezzo,
    target_len=96,
    warmup_len=48,
    wavelet="db4",
    level=2,
)
```

### Micro

```python
denoise_micro = WaveletDenoise1D(
    n_channels=C_micro,
    target_len=144,
    warmup_len=48,
    wavelet="db4",
    level=2,
)
```

---

## 13. forward 示例伪代码（必须按此语义实现） 不要照抄，可能有错

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt


class WaveletDenoise1D(nn.Module):
    def __init__(self, n_channels, target_len, warmup_len, wavelet="db4", level=2, eps=1e-6):
        super().__init__()
        if n_channels <= 0:
            raise ValueError(f"n_channels must be > 0, got {n_channels}")
        if target_len <= 0:
            raise ValueError(f"target_len must be > 0, got {target_len}")
        if warmup_len < 0:
            raise ValueError(f"warmup_len must be >= 0, got {warmup_len}")
        if level <= 0:
            raise ValueError(f"level must be > 0, got {level}")

        self.n_channels = int(n_channels)
        self.target_len = int(target_len)
        self.warmup_len = int(warmup_len)
        self.wavelet = str(wavelet)
        self.level = int(level)
        self.eps = float(eps)

        self.theta_detail = nn.Parameter(torch.zeros(self.level, self.n_channels))
        self.phi_detail = nn.Parameter(torch.zeros(self.level, self.n_channels))
        self.psi_detail = nn.Parameter(torch.zeros(self.level, self.n_channels))

    @staticmethod
    def _rms(x, eps):
        return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)

    @staticmethod
    def _soft_threshold(x, thr):
        return torch.sign(x) * F.relu(x.abs() - thr)

    def forward(self, x_long):
        if x_long.ndim != 3:
            raise ValueError(f"x_long must have shape [B, C, T], got ndim={x_long.ndim}, shape={tuple(x_long.shape)}")

        b, c, t = x_long.shape
        expected_t = self.target_len + self.warmup_len
        if c != self.n_channels:
            raise ValueError(f"channel mismatch: expected {self.n_channels}, got {c}")
        if t != expected_t:
            raise ValueError(f"time length mismatch: expected {expected_t}, got {t}")

        coeffs = ptwt.wavedec(x_long, self.wavelet, mode="zero", level=self.level, axis=-1)
        a = coeffs[0]
        details = coeffs[1:]

        new_details = []
        for j, d in enumerate(details):
            sigma = self._rms(d, self.eps)
            tau = F.softplus(self.theta_detail[j]).view(1, self.n_channels, 1)
            alpha = torch.sigmoid(self.phi_detail[j]).view(1, self.n_channels, 1)
            rho = torch.sigmoid(self.psi_detail[j]).view(1, self.n_channels, 1)
            thr = tau * sigma
            d_shrunk = self._soft_threshold(d, thr)
            d_new = rho * d + (1.0 - rho) * (alpha * d_shrunk)
            new_details.append(d_new)

        y_long = ptwt.waverec([a] + new_details, self.wavelet, axis=-1)
        y_long = y_long[..., -expected_t:]
        y = y_long[..., -self.target_len:]
        return y
```

### 关于 `mode`

这里固定写成 `mode="zero"`，目的不是引入 artificial left pad，而是满足底层小波库对边界计算的接口要求。

注意这里的真实语义仍然是：

- 输入已经包含真实历史 warmup
- 模块依赖 warmup 降低边界污染
- 最终只保留最后 `target_len` 段

coder 不要把这件事误解成“模块内部在构造历史”。

---

## 14. 单元测试要求

### 测试 1：Macro shape

构造：

- `x = torch.randn(2, 5, 112)`
- `module = WaveletDenoise1D(n_channels=5, target_len=64, warmup_len=48)`

验证：

- `y.shape == (2, 5, 64)`

### 测试 2：Mezzo shape

构造：

- `x = torch.randn(2, 7, 144)`
- `module = WaveletDenoise1D(n_channels=7, target_len=96, warmup_len=48)`

验证：

- `y.shape == (2, 7, 96)`

### 测试 3：Micro shape

构造：

- `x = torch.randn(2, 9, 192)`
- `module = WaveletDenoise1D(n_channels=9, target_len=144, warmup_len=48)`

验证：

- `y.shape == (2, 9, 144)`

### 测试 4：dtype / device 一致

输入是什么 dtype/device，输出就必须是什么 dtype/device。

### 测试 5：非法长度报错

构造错误长度输入，必须报 `ValueError`。

---

## 15. 验收标准

实现完成后，以下条件必须同时满足：

1. `WaveletDenoise1D` 只承担 denoise 职责。
2. 只有 `theta_detail`、`phi_detail`、`psi_detail` 三类参数是 learnable 参数。
3. approximation 子带原样保留。
4. detail 子带按 RMS-scaled soft-threshold + gain + bypass 处理。
5. 输入输出 shape 完全符合本规范。
6. 三个尺度都可独立实例化并 forward 成功。
7. 没有引入任何 encode / backbone / attention / fusion 逻辑。
8. 单元测试全部通过。

---

## 16. 最终一句话要求

你要实现的不是一个“可学习表示模块”，而是一个**职责极窄、过程固定、参数极少的纯小波去噪前端**。

它只做：

> 长窗输入 → 固定 DWT → detail 三参数 shrinkage → IDWT → 裁最后 target 段 → 输出 denoised 序列

除此之外，什么都不要做。
