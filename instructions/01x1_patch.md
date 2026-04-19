# 01_1 patch 说明

## 对应对象

- 原始文档：`01_wavelet_denoise.md`
- patch 序号：`01_1`
- patch 类型：**说明性补丁**

## 本 patch 的目的

本 patch 不修改 `WaveletDenoise1D` 的核心算法、输入输出语义、张量 shape、三参数 shrinkage 逻辑，也不修改三尺度实例化方式。

本 patch 只补充一件事：

> 按项目统一规则，明确区分 **接口参数**、**隐参数**、**开放参数**，避免 coder 在实现时自行猜测哪些参数应该暴露到 `config/models.py`，哪些参数应该收进 `src/models/config/hparams.py`。

## 背景规则

根据当前总规则：

- 参数分为两种：
  1. **隐参数**：放在 `src/models/config/hparams.py`
  2. **开放参数**：放在 `config/models.py`，这些参数应当是未来确实有调参必要的参数

但在 `01_wavelet_denoise.md` 当前版本中，`WaveletDenoise1D.__init__` 里仍直接列出了：

- `wavelet`
- `level`
- `eps`

这会让 coder 产生歧义：

- 它们是不是开放参数？
- 是不是应该写进模型配置？
- 未来实例化时是否允许外部频繁覆盖？

因此需要补一份 patch 说明，把这件事定死。

## 本 patch 的核心结论

对于 `WaveletDenoise1D`，参数应分成三类：

### 1. 接口参数（必须显式传入，但不属于调参项）

这些参数由输入数据结构决定，属于模块实例化所必需的结构参数，不属于“开放调参参数”：

- `n_channels`
- `target_len`
- `warmup_len`

说明：

- 这三个值来自具体尺度的数据 shape
- 它们不是超参数搜索对象
- 它们不应被放入 `config/models.py` 作为可调模型超参数
- 它们应由上层装配代码在实例化时按数据规范显式传入

### 2. 隐参数（收进 hparams，不在 config/models.py 暴露）

以下参数属于当前版本中**不建议频繁调参**、且已经在文档中被定死默认值的参数：

- `_wavelet = "db4"`
- `_level = 2`
- `_eps = 1e-6`

说明：

- 这三个参数可以保留在类构造中，但项目层面应视为**隐参数**
- 如果需要统一管理，应放在：
  - `src/models/config/hparams.py`
- 它们不应该出现在 `config/models.py` 中给业务侧/实验侧频繁改动

### 3. 开放参数（当前版本没有）

当前 `WaveletDenoise1D` 的 v2 规范中，**没有需要开放到 `config/models.py` 的参数**。

原因：

1. 文档已经明确写死了默认设计：
   - `wavelet = "db4"`
   - `level = 2`
   - `eps = 1e-6`
2. 该模块职责极窄，目标是稳定、简单、不可歧义
3. 现阶段不希望把小波前端重新变成一个需要频繁搜索超参数的模块

因此，本 patch 明确规定：

> `WaveletDenoise1D` 当前版本 **没有开放参数**。

## 对原文档应补充的说明

建议在 `01_wavelet_denoise.md` 中新增一个小节，例如：

---

## 参数分类说明（01_1 patch 新增）

本模块参数分为三类：

### 接口参数

以下参数由调用方在实例化时显式提供，但不属于开放调参项：

- `n_channels`
- `target_len`
- `warmup_len`

它们由输入数据 shape 决定，不进入 `config/models.py`。

### 隐参数

以下参数属于隐参数，应统一放在 `src/models/config/hparams.py` 中管理：

- `_wavelet`
- `_level`
- `_eps`

它们在当前版本中默认分别取：

- `_wavelet = "db4"`
- `_level = 2`
- `_eps = 1e-6`

### 开放参数

当前版本无开放参数。

`WaveletDenoise1D` 不应向 `config/models.py` 暴露任何可调项。

---

## 对 coder 的直接要求

coder 在实现 `WaveletDenoise1D` 时，必须按以下规则执行：

1. `n_channels / target_len / warmup_len` 作为实例化必需参数保留。
2. `wavelet / level / eps` 在项目层面视为隐参数，不作为公开调参项。
3. 当前版本不要向 `config/models.py` 增加任何 `WaveletDenoise1D` 的开放参数。
4. 不要自行把 `wavelet`、`level`、`eps` 扩展成可搜索的实验参数。
5. 不要新增新的开放参数。

## 本 patch 不做的事

本 patch 不涉及以下任何修改：

- 不修改 `WaveletDenoise1D` 的文件路径
- 不修改三参数 shrinkage 公式
- 不修改 `theta_detail / phi_detail / psi_detail` 定义
- 不修改输入输出 shape
- 不修改三尺度实例化方式
- 不修改测试项
- 不修改 `ptwt` 依赖

## 一句话结论

`01_1` patch 的作用只有一个：

> 把 `WaveletDenoise1D` 的参数边界写死：`n_channels / target_len / warmup_len` 是接口参数，`_wavelet / _level / _eps` 是隐参数，当前版本没有开放参数。

