# Multi-Task Loss 实现规范（最终版，给 coder 的确定性说明）

本模块的总规则：

- 参数分为两种，一种是**隐参数**，放在 `models/hparams.yaml` 或 `hparams.py`
- 另一种是**开放参数**，放在 `config/models.py`
- 隐参数统一使用 `_` 开头
- 开放参数不使用 `_` 开头
- 本文档只定义 **loss / likelihood / task output parameterization**
- 本文档不定义 trunk、tower、fusion 结构
- 本文档当前固定三任务：
  1. `ret`：预计时间平均收益率
  2. `rv`：波动率
  3. `q`：`p_xx` 分位数收益（风险标签）

当前已确定的决策如下：

1. `ret` 使用 **Student-t NLL**
2. `ret` 的自由度 `ν` 使用 **task-level 全局可学习参数**
3. `rv` 使用 **Gamma NLL**
4. `q` 使用 **Asymmetric Laplace NLL (ALD NLL)**
5. 三个任务都需要输出不可信度
6. `ret` / `q` 的 target **不做 log 变换**
7. `rv` 的 target **不做 log 变换**
8. 所有正值参数统一用 `softplus + eps` 做正值化

---

## 0. 给 coder 的确定性 prompt

你要实现的是三任务预测的 **分布式输出与对应 NLL loss**。

你必须严格按本规范实现，不要自行替换分布，不要更改参数化方式，不要把 target 改成别的空间，不要引入额外损失。

### 三个任务及对应分布

- `ret`：
  - 分布：Student-t
  - 参数：`mu_ret`, `scale_ret`, `nu_ret`
  - 其中 `nu_ret` 为 **task-level 全局可学习参数**

- `rv`：
  - 分布：Gamma（mean-shape parameterization）
  - 参数：`mean_rv`, `shape_rv`
  - 不使用 log-rv loss
  - 不使用 QLIKE 作为独立实现，因为 QLIKE 只是 Gamma(shape=1) 的特例

- `q`：
  - 分布：Asymmetric Laplace Distribution (ALD)
  - 参数：`mu_q`, `scale_q`
  - `tau_q` 固定为配置给定的 quantile level，不是网络输出

### 明确禁止的东西

- 不要把 `ret` 改成 Gaussian
- 不要把 `ret` 的 `nu` 改成逐样本输出
- 不要把 `rv` 改成 QLIKE-only 单参数版本
- 不要把 `rv` target 改成 `log(rv)`
- 不要把 `q` 只写成 pinball 而丢掉 scale 参数
- 不要把 `q` 的 target 做 log
- 不要把任一尺度参数直接 `exp(raw)` 而不加数值保护
- 不要把 `softplus` 后的正值参数允许为 0
- 不要把三个任务的 loss 混成一个没有语义的统一公式

---

## 1. 文件结构（必须按此落位）

```text
src/models/
    losses/
        __init__.py
        student_t_nll.py
        gamma_nll.py
        asymmetric_laplace_nll.py
        multi_task_loss.py
```

### 1.1 `src/models/losses/student_t_nll.py`

必须包含以下类：

- `StudentTNLLLoss`

### 1.2 `src/models/losses/gamma_nll.py`

必须包含以下类：

- `GammaNLLLoss`

### 1.3 `src/models/losses/asymmetric_laplace_nll.py`

必须包含以下类：

- `AsymmetricLaplaceNLLLoss`

### 1.4 `src/models/losses/multi_task_loss.py`

必须包含以下类：

- `MultiTaskDistributionLoss`

### 1.5 `src/models/losses/__init__.py`

必须导出：

```python
from .student_t_nll import StudentTNLLLoss
from .gamma_nll import GammaNLLLoss
from .asymmetric_laplace_nll import AsymmetricLaplaceNLLLoss
from .multi_task_loss import MultiTaskDistributionLoss
```

---

## 2. 参数分类（必须明确）

### 2.1 开放参数（放到 `config/models.py`）

这些参数是未来可能调的，不加 `_` 前缀：

- `ret_loss_weight`
- `rv_loss_weight`
- `q_loss_weight`
- `q_tau`

### 2.2 隐参数（放到 `models/hparams.yaml` 或 `hparams.py`）

这些参数是不希望频繁调的，统一加 `_` 前缀：

- `_eps`
- `_nu_ret_init`
- `_nu_ret_min`
- `_gamma_shape_min`
- `_ald_scale_min`

### 2.3 当前默认值

#### 开放参数默认值
```python
ret_loss_weight = 1.0
rv_loss_weight = 1.0
q_loss_weight = 1.0
q_tau = 0.05
```

#### 隐参数默认值
```python
_eps = 1e-6
_nu_ret_init = 8.0
_nu_ret_min = 2.01
_gamma_shape_min = 1e-4
_ald_scale_min = 1e-6
```

---

## 3. 三个任务的模型输出定义（必须定死）

假设每个 task tower 输出一个 task representation：

```text
h_ret:  [B, D]
h_rv:   [B, D]
h_q:    [B, D]
```

那么三个任务的 head 输出必须严格定义为：

### 3.1 ret 任务输出

```text
mu_ret_raw:    [B, 1]
scale_ret_raw: [B, 1]
```

实际参数为：

\[
\mu_{\text{ret}} = \text{mu\_ret\_raw}
\]

\[
s_{\text{ret}} = \operatorname{softplus}(\text{scale\_ret\_raw}) + \_eps
\]

### 3.2 rv 任务输出

```text
mean_rv_raw:  [B, 1]
shape_rv_raw: [B, 1]
```

实际参数为：

\[
m_{\text{rv}} = \operatorname{softplus}(\text{mean\_rv\_raw}) + \_eps
\]

\[
\alpha_{\text{rv}} = \operatorname{softplus}(\text{shape\_rv\_raw}) + \_gamma\_shape\_min
\]

### 3.3 q 任务输出

```text
mu_q_raw:    [B, 1]
scale_q_raw: [B, 1]
```

实际参数为：

\[
\mu_q = \text{mu\_q\_raw}
\]

\[
b_q = \operatorname{softplus}(\text{scale\_q\_raw}) + \_ald\_scale\_min
\]

---

## 4. ret 任务：Student-t NLL（必须定死）

## 4.1 分布假设

设收益标签记为：

\[
y_{\text{ret}} \in \mathbb{R}
\]

假设：

\[
y_{\text{ret}} \mid x \sim \operatorname{StudentT}(\mu_{\text{ret}}, s_{\text{ret}}, \nu_{\text{ret}})
\]

其中：

- \(\mu_{\text{ret}} \in \mathbb{R}\)
- \(s_{\text{ret}} > 0\)
- \(\nu_{\text{ret}} > 2\)

### 4.2 概率密度函数

Student-t 的密度为：

\[
p(y \mid \mu, s, \nu)
=
\frac{
\Gamma\left(\frac{\nu+1}{2}\right)
}{
\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\, s
}
\left(
1 + \frac{(y-\mu)^2}{\nu s^2}
\right)^{-\frac{\nu+1}{2}}
\]

### 4.3 NLL 公式

负对数似然为：

\[
\mathcal{L}_{\text{ret}}
=
\log s
+
\frac{1}{2}\log(\nu \pi)
+
\log \Gamma\left(\frac{\nu}{2}\right)
-
\log \Gamma\left(\frac{\nu+1}{2}\right)
+
\frac{\nu+1}{2}
\log\left(
1 + \frac{(y-\mu)^2}{\nu s^2}
\right)
\]

实现时必须按这个公式计算。

### 4.4 `ν` 的参数化（必须定死）

`ret` 的自由度使用 **task-level 全局可学习参数**，不是逐样本输出。

定义一个可学习标量参数：

```python
self._nu_ret_raw = nn.Parameter(torch.tensor(init_value))
```

其中 `init_value` 必须按 `_nu_ret_init` 反解或直接取一个合理常数初始化。

实际自由度定义为：

\[
\nu_{\text{ret}} = \_nu\_ret\_min + \operatorname{softplus}(\_nu\_ret\_raw)
\]

必须保证：

\[
\nu_{\text{ret}} > 2
\]

不要改成：
- 逐样本输出 `ν`
- 固定常数 `ν`（当前已确定是可学习）
- `exp(raw)` 直接参数化

### 4.5 不可信度输出

Student-t 的 scale 参数 `s_ret` 本身就是 aleatoric uncertainty 的核心参数。

若需要换算为标准差，则在 \(\nu>2\) 下：

\[
\operatorname{Var}(Y \mid x) = \frac{\nu}{\nu-2} s^2
\]

所以预测标准差为：

\[
\sigma_{\text{ret,pred}}
=
s_{\text{ret}}
\sqrt{\frac{\nu_{\text{ret}}}{\nu_{\text{ret}}-2}}
\]

### 4.6 ret 的 log 处理规则（定死）

- `y_ret` **不做 log**
- `mu_ret` 在原始收益空间预测
- 只对 `scale_ret` 做正值参数化

---

## 5. rv 任务：Gamma NLL（必须定死）

## 5.1 分布假设

设波动率标签记为：

\[
y_{\text{rv}} > 0
\]

假设：

\[
y_{\text{rv}} \mid x \sim \operatorname{Gamma}(\alpha_{\text{rv}}, \theta_{\text{rv}})
\]

但本规范不用 `(shape, scale)` 直接输出，而是使用 **mean-shape parameterization**：

- 均值 \(m_{\text{rv}} > 0\)
- 形状参数 \(\alpha_{\text{rv}} > 0\)

并定义：

\[
m_{\text{rv}} = \alpha_{\text{rv}} \theta_{\text{rv}}
\]

所以：

\[
\theta_{\text{rv}} = \frac{m_{\text{rv}}}{\alpha_{\text{rv}}}
\]

### 5.2 Gamma 概率密度函数

Gamma 密度：

\[
p(y \mid \alpha, \theta)
=
\frac{1}{\Gamma(\alpha)\theta^\alpha}
y^{\alpha-1} e^{-y/\theta}
\]

代入：

\[
\theta = \frac{m}{\alpha}
\]

得到 mean-shape 参数化密度：

\[
p(y \mid m, \alpha)
=
\frac{1}{\Gamma(\alpha)}
\left(\frac{\alpha}{m}\right)^\alpha
y^{\alpha-1}
\exp\left(-\frac{\alpha y}{m}\right)
\]

### 5.3 NLL 公式

负对数似然为：

\[
\mathcal{L}_{\text{rv}}
=
\alpha \frac{y}{m}
+
\alpha \log m
-
\alpha \log \alpha
+
\log \Gamma(\alpha)
-
(\alpha - 1)\log y
\]

这里：

- \(y = y_{\text{rv}}\)
- \(m = m_{\text{rv}}\)
- \(\alpha = \alpha_{\text{rv}}\)

实现时必须按此公式计算。

### 5.4 与 QLIKE 的关系（必须说明清楚）

QLIKE 定义为：

\[
\operatorname{QLIKE}(y,m)
=
\frac{y}{m}
-
\log \frac{y}{m}
-
1
\]

展开为：

\[
\operatorname{QLIKE}(y,m)
=
\frac{y}{m} + \log m - \log y - 1
\]

当 Gamma 形状参数固定为：

\[
\alpha = 1
\]

时，Gamma NLL 变成：

\[
\mathcal{L}_{\text{rv}}
=
\frac{y}{m} + \log m
\]

与 QLIKE 只差与模型参数无关的常数项：

\[
-\log y - 1
\]

因此：

- **QLIKE 是 Gamma NLL 在 \(\alpha=1\) 时的特例**
- 当前我们不采用这个特例
- 当前采用更一般的 **Gamma NLL**
- 因为你已经明确要求 **rv 也需要独立不可信度输出**

### 5.5 不可信度输出

Gamma 分布下：

\[
\mathbb{E}[Y \mid x] = m
\]

\[
\operatorname{Var}(Y \mid x) = \frac{m^2}{\alpha}
\]

所以预测标准差为：

\[
\sigma_{\text{rv,pred}}
=
\frac{m_{\text{rv}}}{\sqrt{\alpha_{\text{rv}}}}
\]

这里：

- `mean_rv` 负责中心预测
- `shape_rv` 负责不可信度强弱
- `shape` 越大，方差越小
- `shape` 越小，方差越大

### 5.6 rv 的 log 处理规则（定死）

- `y_rv` **不做 log**
- 不训练 `log(rv)`
- 不使用 Gaussian NLL on log-rv
- 必须在原始正值 rv 空间做 Gamma NLL

### 5.7 数值稳定要求

Gamma NLL 中包含：

\[
\log y
\]

因此实现时必须对 target 做：

```python
y_safe = y.clamp_min(_eps)
```

然后用 `y_safe` 参与：

- `log(y_safe)`
- 其他正值运算

---

## 6. q 任务：Asymmetric Laplace NLL（必须定死）

## 6.1 分布假设

设分位数收益标签记为：

\[
y_q \in \mathbb{R}
\]

对应目标分位数水平：

\[
\tau = q\_tau \in (0,1)
\]

假设：

\[
y_q \mid x \sim \operatorname{ALD}(\mu_q, b_q, \tau)
\]

其中：

- \(\mu_q\)：目标 \(\tau\)-分位数位置参数
- \(b_q > 0\)：ALD 的 scale
- \(\tau\)：固定给定，不是网络输出

### 6.2 pinball loss 定义

令残差：

\[
u = y_q - \mu_q
\]

pinball loss 定义为：

\[
\rho_\tau(u)
=
u\bigl(\tau - \mathbf{1}(u < 0)\bigr)
\]

等价分段形式：

\[
\rho_\tau(u)
=
\begin{cases}
\tau u, & u \ge 0 \\
(\tau - 1)u, & u < 0
\end{cases}
\]

### 6.3 ALD 概率密度函数

Asymmetric Laplace Distribution 的密度为：

\[
p(y \mid \mu, b, \tau)
=
\frac{\tau(1-\tau)}{b}
\exp\left(
-\rho_\tau\left(\frac{y-\mu}{b}\right)
\right)
\]

### 6.4 NLL 公式

负对数似然为：

\[
\mathcal{L}_{q}
=
\rho_\tau\left(\frac{y-\mu}{b}\right)
+
\log b
-
\log\bigl(\tau(1-\tau)\bigr)
\]

这里：

- \(y = y_q\)
- \(\mu = \mu_q\)
- \(b = b_q\)
- \(\tau = q\_tau\)

实现时必须按此公式计算。

### 6.5 为什么不能只写 pinball

如果只写 pinball loss：

\[
\rho_\tau(y-\mu)
\]

那相当于把 ALD 的 scale 固定成常数，不再有独立不可信度输出。

当前你已经明确要求：

- `q` 任务也需要不可信度输出

所以必须使用 **完整 ALD NLL**，不能只实现 pinball。

### 6.6 不可信度输出

ALD 的 scale \(b_q\) 就是不可信度核心参数。

ALD 的方差为：

\[
\operatorname{Var}(Y \mid x)
=
b_q^2
\cdot
\frac{1 - 2\tau + 2\tau^2}{\tau^2(1-\tau)^2}
\]

因此预测标准差为：

\[
\sigma_{q,\text{pred}}
=
b_q
\sqrt{
\frac{1 - 2\tau + 2\tau^2}{\tau^2(1-\tau)^2}
}
\]

### 6.7 q 的 log 处理规则（定死）

- `y_q` **不做 log**
- `mu_q` 在原始收益空间预测
- 只对 `scale_q` 做正值参数化

---

## 7. 三个单任务 loss 类的接口定义（必须完全一致）

## 7.1 `StudentTNLLLoss`

文件：`src/models/losses/student_t_nll.py`

```python
class StudentTNLLLoss(nn.Module):
    def __init__(
        self,
        _nu_min: float = 2.01,
        _eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入

- `target: [B, 1]`
- `mu: [B, 1]`
- `scale: [B, 1]`
- `nu: []` 或 `[1]` 或可广播标量

### 输出

- 标量 loss

### 实现要求

1. `scale` 必须假设已经是正值
2. `nu` 必须假设已经大于 `_nu_min`
3. 返回 batch mean NLL

---

## 7.2 `GammaNLLLoss`

文件：`src/models/losses/gamma_nll.py`

```python
class GammaNLLLoss(nn.Module):
    def __init__(
        self,
        _eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        target: torch.Tensor,
        mean: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入

- `target: [B, 1]`
- `mean: [B, 1]`
- `shape: [B, 1]`

### 输出

- 标量 loss

### 实现要求

1. `target` 内部先做：
   ```python
   target_safe = target.clamp_min(_eps)
   ```
2. `mean` 和 `shape` 必须假设已经是正值
3. 返回 batch mean NLL

---

## 7.3 `AsymmetricLaplaceNLLLoss`

文件：`src/models/losses/asymmetric_laplace_nll.py`

```python
class AsymmetricLaplaceNLLLoss(nn.Module):
    def __init__(
        self,
        tau: float,
        _eps: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 输入

- `target: [B, 1]`
- `mu: [B, 1]`
- `scale: [B, 1]`

### 输出

- 标量 loss

### 实现要求

1. `tau` 必须满足 `0 < tau < 1`
2. `scale` 必须假设已经是正值
3. 返回 batch mean NLL

---

## 8. 三任务总 loss 类接口（必须完全一致）

文件：`src/models/losses/multi_task_loss.py`

```python
class MultiTaskDistributionLoss(nn.Module):
    def __init__(
        self,
        q_tau: float,
        ret_loss_weight: float = 1.0,
        rv_loss_weight: float = 1.0,
        q_loss_weight: float = 1.0,
        _eps: float = 1e-6,
        _nu_ret_init: float = 8.0,
        _nu_ret_min: float = 2.01,
        _gamma_shape_min: float = 1e-4,
        _ald_scale_min: float = 1e-6,
    ) -> None:
        ...

    def forward(
        self,
        target_ret: torch.Tensor,
        pred_mu_ret: torch.Tensor,
        pred_scale_ret_raw: torch.Tensor,
        target_rv: torch.Tensor,
        pred_mean_rv_raw: torch.Tensor,
        pred_shape_rv_raw: torch.Tensor,
        target_q: torch.Tensor,
        pred_mu_q: torch.Tensor,
        pred_scale_q_raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...
```

### 8.1 内部必须持有的可学习参数

`ret` 的 task-level 全局自由度参数必须作为模块参数定义：

```python
self._nu_ret_raw = nn.Parameter(torch.tensor(init_raw_value))
```

然后在 `forward()` 中计算：

\[
\nu_{\text{ret}} = \_nu\_ret\_min + \operatorname{softplus}(\_nu\_ret\_raw)
\]

### 8.2 forward 内部必须做的正值化

#### ret
\[
s_{\text{ret}} = \operatorname{softplus}(\text{pred\_scale\_ret\_raw}) + \_eps
\]

#### rv
\[
m_{\text{rv}} = \operatorname{softplus}(\text{pred\_mean\_rv\_raw}) + \_eps
\]

\[
\alpha_{\text{rv}} = \operatorname{softplus}(\text{pred\_shape\_rv\_raw}) + \_gamma\_shape\_min
\]

#### q
\[
b_q = \operatorname{softplus}(\text{pred\_scale\_q\_raw}) + \_ald\_scale\_min
\]

### 8.3 输出字典（必须定死）

返回字典至少包含：

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
}
```

其中：

#### ret
\[
\sigma_{\text{ret,pred}}
=
s_{\text{ret}}
\sqrt{\frac{\nu_{\text{ret}}}{\nu_{\text{ret}}-2}}
\]

#### rv
\[
\sigma_{\text{rv,pred}}
=
\frac{m_{\text{rv}}}{\sqrt{\alpha_{\text{rv}}}}
\]

#### q
\[
\sigma_{q,\text{pred}}
=
b_q
\sqrt{
\frac{1 - 2\tau + 2\tau^2}{\tau^2(1-\tau)^2}
}
\]

### 8.4 总 loss

总 loss 必须定义为：

\[
\mathcal{L}_{\text{total}}
=
w_{\text{ret}}\mathcal{L}_{\text{ret}}
+
w_{\text{rv}}\mathcal{L}_{\text{rv}}
+
w_q\mathcal{L}_q
\]

其中：

- \(w_{\text{ret}} = \text{ret\_loss\_weight}\)
- \(w_{\text{rv}} = \text{rv\_loss\_weight}\)
- \(w_q = \text{q\_loss\_weight}\)

---

## 9. 数值稳定要求（必须严格遵守）

### 9.1 所有正值参数都必须经过正值化

绝对不允许直接使用 raw 输出当正值参数。

### 9.2 所有涉及 `log(y)` 的地方必须做 target clamp

当前只在 Gamma NLL 中有：

\[
\log y
\]

所以必须使用：

```python
target_safe = target.clamp_min(_eps)
```

### 9.3 所有 batch loss 必须做 mean reduction

不要返回 sum，不要返回 none reduction。

---

## 10. 错误处理要求

以下情况必须报 `ValueError`：

1. `ret_loss_weight < 0`
2. `rv_loss_weight < 0`
3. `q_loss_weight < 0`
4. `q_tau <= 0` 或 `q_tau >= 1`
5. `_eps <= 0`
6. `_nu_ret_min <= 2`
7. `_gamma_shape_min <= 0`
8. `_ald_scale_min <= 0`
9. 任一输入 shape 不为 `[B, 1]`
10. batch 大小不一致

错误信息必须指出：

- 哪个值非法
- 当前值是什么
- 合法范围是什么

---

## 11. smoke test 要求

测试文件建议放在：

```text
tests/models/losses/test_multi_task_loss.py
```

至少包含以下测试：

### 测试 1：三任务 forward 可运行
构造：

- `target_ret, target_rv, target_q`: `[B,1]`
- 各 raw 输出也为 `[B,1]`

验证：
- 返回 dict
- 包含所有必须键
- `loss_total` 是标量张量

### 测试 2：`nu_ret` 可反传
验证：
- `self._nu_ret_raw.grad is not None`

### 测试 3：rv 的 `shape` 可反传
验证：
- `pred_shape_rv_raw.grad is not None`

### 测试 4：q 的 `scale` 可反传
验证：
- `pred_scale_q_raw.grad is not None`

### 测试 5：非法 `q_tau` 报错
`q_tau=0` 或 `1` 必须报 `ValueError`

### 测试 6：Gamma NLL 对极小正 target 不崩
例如：
```python
target_rv = torch.full((B,1), 1e-12)
```
仍然必须数值稳定。

---

## 12. 最终一句话要求

你要实现的不是一个“随便凑的三任务 loss”，而是一个：

> **`ret` 用带 task-level 全局可学习自由度的 Student-t NLL，`rv` 用 mean-shape 参数化的 Gamma NLL，`q` 用带可学习 scale 的 Asymmetric Laplace NLL，并统一输出三任务各自 aleatoric uncertainty 的严格分布式多任务 loss 模块。**

除此之外，什么都不要做。
