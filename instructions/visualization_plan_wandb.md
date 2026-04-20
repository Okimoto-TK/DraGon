# 多尺度多任务预测模型可视化方案（W&B）

## 0. 文档目的

本方案面向 AI coder，目标不是“把指标记上去”，而是让可视化直接回答两类核心问题：

1. **学习效果**：模型是否真的在学、学到了哪一层、哪个尺度、哪个任务。
2. **健康状态**：模型是否在塌缩、饱和、失衡、过度依赖某一路信息、或出现数值异常。

本方案基于当前模型链路：

`原始多尺度序列 -> 小波去噪 -> 三个单尺度编码器 -> 尺度内融合 -> 条件编码/层级外生记忆 -> 单尺度桥接融合 -> 跨尺度融合 -> 多任务输出头 -> 分布式损失`

可视化结构必须严格跟随模型结构走，而不是按“loss / grad / attention”随意堆指标。

---

## 1. W&B 部署与运行约束

### 1.1 运行环境

- Linux
- 使用自托管 W&B
- 服务暴露到：`0.0.0.0:11111`
- 训练侧统一使用：
  - `WANDB_BASE_URL=http://<server_ip>:11111`
  - `WANDB_PROJECT=<project_name>`
  - `WANDB_RUN_GROUP=<experiment_group>`

### 1.2 Run 分组规则

每个 run 至少包含以下 config 字段，供 W&B 侧筛选与分组：

- `model.hidden_dim`
- `model.cond_dim`
- `model.num_latents`
- `model.q_tau`
- `data.macro_len`
- `data.mezzo_len`
- `data.micro_len`
- `train.batch_size`
- `train.lr`
- `train.weight_decay`
- `train.grad_clip`
- `train.compile`
- `train.seed`
- `loss.ret_weight`
- `loss.rv_weight`
- `loss.q_weight`

### 1.3 W&B 面板分类

W&B workspace 按以下 section 固定组织：

1. `00_overview`
2. `01_input`
3. `02_wavelet`
4. `03_encoder_macro`
5. `04_encoder_mezzo`
6. `05_encoder_micro`
7. `06_within_scale_fusion`
8. `07_conditioning`
9. `08_side_memory`
10. `09_bridge_fusion`
11. `10_cross_scale`
12. `11_heads`
13. `12_loss`
14. `13_optimizer_and_system`
15. `14_case_study`

不要把所有 scalar 扔在一个总面板里。

---

## 2. 日志层级与记录频率

### 2.1 三层日志粒度

必须分三层记录：

#### A. 高频标量层（step 级）
用途：观察训练过程是否稳定。

记录频率：`every log_every step`

包括：
- loss
- lr
- grad norm
- param norm
- activation mean/std
- gate mean/std
- 关键分布参数均值
- 吞吐、step_time、data_time

#### B. 中频分布层（step/epoch 混合）
用途：观察是否塌缩、是否极端偏态、是否有长尾异常。

记录频率：`every hist_every step` 或 `every eval epoch`

包括：
- activation histogram
- weight histogram
- grad histogram
- raw head output histogram
- legal parameter histogram（如 sigma / shape / nu）

#### C. 低频结构层（固定验证批）
用途：观察语义结构是否合理。

记录频率：`every viz_every step` 或 `every val epoch`

包括：
- heatmap
- attention map
- feature-patch map
- scale-latent map
- case study table

### 2.2 固定验证样本

所有 heatmap / attention / case study 都必须基于 **固定验证子集**，否则不同 epoch 图像不可比较。

固定验证子集要求：

- 固定随机种子抽样
- 至少覆盖 3 类样本：
  - 正常波动样本
  - 强趋势样本
  - 高噪声/高冲击样本
- 固定保存样本 ID，并在所有 epoch 复用

### 2.3 train / val 分开记

所有指标必须显式区分：

- `train/...`
- `val/...`

严禁把 train 和 val 画在同一命名空间里再靠 legend 猜。

---

## 3. 命名规范

### 3.1 命名模板

统一命名：

`{split}/{module}/{submodule_or_scale}/{metric_name}`

例如：

- `train/loss/total`
- `val/loss/ret_nll`
- `train/encoder_macro/block_02/act_std`
- `val/bridge_micro/gate/mean`
- `val/cross_scale/latent_token_usage/entropy`

### 3.2 图表对象命名

图表标题中必须显式包含语义，不要只写 `heatmap_1`。

正确示例：

- `macro encoder feature-patch activation heatmap`
- `mezzo bridge gate over token position`
- `cross-scale latent to scale attention`
- `ret head predicted sigma distribution`

错误示例：

- `viz1`
- `panel_a`
- `debug_map`

---

## 4. 总览页（00_overview）

这是默认首页，必须先回答“能不能继续训”。

### 4.1 必须包含的图

#### 1) 总损失与分任务损失
- 图表：**折线图**
- 指标：
  - `train/loss/total`
  - `val/loss/total`
  - `train/loss/ret`
  - `train/loss/rv`
  - `train/loss/q`
  - `val/loss/ret`
  - `val/loss/rv`
  - `val/loss/q`
- 目的：判断总体收敛，以及是否某一任务成为瓶颈。

#### 2) 学习效果主指标
- 图表：**折线图**
- 指标：按你的评估模块接入，例如：
  - `val/metrics/ret_corr`
  - `val/metrics/ret_rank_ic`
  - `val/metrics/rv_mae`
  - `val/metrics/q_pinball_or_nll_proxy`
- 目的：loss 降并不等于任务有效，这一页必须能看任务效果。

#### 3) 健康状态四联图
- 图表：**四个折线图**
- 指标：
  - `train/health/global_grad_norm`
  - `train/health/global_param_norm`
  - `train/health/global_act_mean`
  - `train/health/global_act_std`
- 目的：第一时间判断 exploding / vanishing / collapse。

#### 4) 吞吐与系统状态
- 图表：**折线图**
- 指标：
  - `train/system/step_time_ms`
  - `train/system/data_time_ms`
  - `train/system/forward_time_ms`
  - `train/system/backward_time_ms`
  - `train/system/tokens_or_samples_per_sec`
  - `train/system/gpu_mem_alloc_mb`
- 目的：确认性能问题不是模型问题。

#### 5) 多任务 loss 占比
- 图表：**堆叠面积图** 或 **100% stacked area**
- 指标：
  - `train/loss_share/ret`
  - `train/loss_share/rv`
  - `train/loss_share/q`
- 目的：看训练是否长期被某一任务主导。

---

## 5. 输入组成（01_input）

模型输入包含 `macro / mezzo / micro` 三个连续特征序列、两个离散条件通道 `state / pos`，以及 `sidechain_cond` 外生序列。

### 5.1 可视化目标

这个模块回答：

1. 输入本身是否分布漂移。
2. 三个尺度的数据是否在数值上健康。
3. 离散条件是否失衡。
4. `sidechain_cond` 是否经常为零、常数、或异常尖峰。

### 5.2 必须记录的指标

#### 连续输入统计
- 图表：**折线图**
- 每个 split、每个 scale、每个连续 feature 记录：
  - mean
  - std
  - min
  - max
  - p01 / p50 / p99
- 命名：
  - `train/input/macro/feature_<name>/mean`
  - `train/input/macro/feature_<name>/std`

#### 连续输入分布
- 图表：**histogram**
- 记录：
  - `macro_float_long`
  - `mezzo_float_long`
  - `micro_float_long`
  - `sidechain_cond`
- 注意：不要混在一起；每个 scale 独立 histogram。

#### 离散条件占比
- 图表：**柱状图**
- 指标：
  - `state` 类别频次
  - `pos` 类别频次
- 呈现方式：每个 epoch 记录一个 category count bar chart。
- 目的：看 `state / pos` 是否偏到几乎单一类别。

#### 固定样本输入热力图
- 图表：**heatmap**
- 对象：固定验证样本
- 画法：
  - x 轴：时间位置
  - y 轴：feature 维
  - 值：标准化后的 feature value
- 分别画：
  - `macro`
  - `mezzo`
  - `micro`
  - `sidechain_cond`
- 目的：让人直接看输入纹理，不做均值压扁。

---

## 6. 小波去噪前端（02_wavelet）

`WaveletDenoise1D` 负责从带 warmup 的长序列中做固定小波分解与去噪，然后输出目标长度序列。

### 6.1 可视化目标

这个模块回答：

1. 去噪到底去掉了什么。
2. 去噪是否过强，导致有用波动被抹平。
3. 三个尺度的去噪强度是否一致且合理。

### 6.2 必须记录的指标

#### 去噪前后能量比
- 图表：**折线图**
- 指标：
  - `train/wavelet/<scale>/energy_raw`
  - `train/wavelet/<scale>/energy_denoised`
  - `train/wavelet/<scale>/energy_ratio_denoised_over_raw`
- 定义建议：
  - energy = mean(x^2)
- 目的：看去噪强度是否异常飙升或塌到过低。

#### 细节带收缩比例
- 图表：**折线图**
- 指标：
  - `train/wavelet/<scale>/detail_band_<k>/shrink_ratio`
- 如果实现中有多层 detail band，就逐层记录。
- 目的：看哪一层频带被过分压制。

#### 去噪残差分布
- 图表：**histogram**
- 指标：
  - `train/wavelet/<scale>/residual_hist`
- 其中 residual = raw_tail - denoised
- 目的：看被去掉的是小噪声还是大量主结构。

#### 固定样本对比图
- 图表：**两行折线图 image**
- 每个 scale 画 3 张固定样本图：
  - 原始输入曲线
  - 去噪输出曲线
  - 两者差值曲线
- 对象：建议只选 1~2 个代表性 feature，不要 9 个全堆。
- 目的：这个图最能直接判断“去噪有没有抹掉趋势”。

#### 特征-时间去噪强度图
- 图表：**heatmap**
- 画法：
  - x 轴：输出时间位置
  - y 轴：feature
  - 值：`abs(raw_tail - denoised)` 的 batch 平均
- 不允许再沿 feature 或时间做一次平均。
- 目的：看去噪强度集中在哪些 feature 和哪个时段。

---

## 7. 单尺度编码器（03/04/05_encoder_*）

三个尺度编码器 `ModernTCNFiLMEncoder` 独立工作，输入经 patch 化后输出 `[B, F, D, N]`。内部包含 `Patch1D`、`ConditionEmbedding1D`、`FiLM1D`、depthwise temporal conv、`ChannelFFN1D` 等子结构。

三个尺度必须拆成三个 section：

- `03_encoder_macro`
- `04_encoder_mezzo`
- `05_encoder_micro`

不要三者混在一起。

### 7.1 可视化目标

这个模块回答：

1. patch 化后是否有信息损失或异常静态化。
2. block 内部 activations 是否健康。
3. FiLM 是否真的在起作用，还是退化为恒等映射。
4. depthwise temporal conv 是否在不同 patch 上形成结构。
5. 某个尺度是否单独塌缩。

### 7.2 Patch1D

#### patch token 范数
- 图表：**折线图**
- 指标：
  - `train/encoder_<scale>/patch/token_l2_mean`
  - `train/encoder_<scale>/patch/token_l2_std`
- 目的：看 patch 输出是否迅速接近常数。

#### feature-patch 激活热力图
- 图表：**heatmap**
- 画法：
  - x 轴：patch index
  - y 轴：feature index / feature name
  - 值：`mean(abs(token))`，再对 D 维求均值
- 目的：看不同 feature 是否在 patch 级别仍有结构。
- 这是语义图，不能再把 patch 平均掉。

### 7.3 每个 ModernTCNFiLMBlock

对每个 block 记录以下指标，命名为：

- `train/encoder_<scale>/block_<i>/...`
- `val/encoder_<scale>/block_<i>/...`

#### activation 健康指标
- 图表：**折线图**
- 指标：
  - `act_mean`
  - `act_std`
  - `act_abs_mean`
  - `residual_delta_l2`
- 目的：看 block 是否不工作或过强放大。

#### activation 分布
- 图表：**histogram**
- 指标：
  - block output histogram
- 目的：看是否过度集中在 0 附近或出现爆炸长尾。

#### LayerNorm 输出统计
- 图表：**折线图**
- 指标：
  - `ln1_mean`
  - `ln1_std`
  - `ln2_mean`
  - `ln2_std`
- 目的：看 pre-norm 是否已经失效。

### 7.4 FiLM1D

#### gamma / beta 统计
- 图表：**折线图** + **histogram**
- 指标：
  - `film_gamma_mean`
  - `film_gamma_std`
  - `film_beta_mean`
  - `film_beta_std`
  - gamma histogram
  - beta histogram
- 目的：
  - gamma 长期约等于 1 且 std 极小：说明调制很弱
  - beta 长期约等于 0：说明偏移近乎无效

#### FiLM 作用强度图
- 图表：**heatmap**
- 画法：
  - x 轴：patch index
  - y 轴：feature
  - 值：`mean(abs(film_out - film_in))`，对 D 维均值
- 目的：看 FiLM 影响集中在哪些 feature / patch。
- 注意：不能只记全局平均，否则语义丢失。

#### state / pos 条件分组效果
- 图表：**分组柱状图**
- 指标：
  - 不同 `state` 类别下的 gamma_mean
  - 不同 `pos` 类别下的 gamma_mean
- 目的：看条件嵌入是否真的对调制有分辨作用。

### 7.5 depthwise temporal conv

#### 时序卷积响应强度
- 图表：**heatmap**
- 画法：
  - x 轴：patch index
  - y 轴：channel block / feature
  - 值：conv 输出绝对值均值
- 目的：看卷积是否只在少数 patch 位置响应。

#### 卷积残差贡献
- 图表：**折线图**
- 指标：
  - `temporal_branch_delta_l2`
- 定义：conv residual 前后差值的 L2 均值
- 目的：看 temporal branch 是否完全空转。

### 7.6 ChannelFFN1D

#### FFN 门控/增益强度
- 图表：**折线图**
- 指标：
  - `ffn_delta_l2`
  - `ffn_out_std`
- 目的：看 FFN 是否退化为近线性恒等。

#### FFN 输出分布
- 图表：**histogram**
- 指标：
  - `ffn_out_hist`

### 7.7 编码器总览比较

#### 三尺度编码器健康对比
- 图表：**分组三柱状图**（每个 epoch 一次）
- 指标：
  - `final_block_act_std`
  - `final_block_residual_delta_l2`
  - `final_block_film_gamma_std`
- 目的：比较 macro / mezzo / micro 哪一路最弱或最不稳定。

---

## 8. 尺度内融合（06_within_scale_fusion）

`WithinScaleSTARFusion` 在单尺度内沿 feature 轴做 aggregate-redistribute，输出 `z_fused [B, F, D, N]` 和 `scale_seq [B, D, N]`。

### 8.1 可视化目标

这个模块回答：

1. feature 融合是否真的发生。
2. 融合后是否出现 feature collapse。
3. core 表示是否被少数 feature 垄断。

### 8.2 必须记录的指标

#### 融合前后 feature 差异度
- 图表：**折线图**
- 指标：
  - `train/within_scale/<scale>/feature_var_pre`
  - `train/within_scale/<scale>/feature_var_post`
  - `train/within_scale/<scale>/feature_diversity_ratio`
- 可用定义：feature 维 pairwise cosine distance 的均值。
- 目的：看融合后是否把 feature 全部抹成一样。

#### feature-patch 融合热力图
- 图表：**heatmap**
- 画法：
  - x 轴：patch index
  - y 轴：feature
  - 值：`mean(abs(z_fused - z_in))`，对 D 维均值
- 目的：看 STAR 融合主要改动哪些 feature token。

#### core usage 图
- 图表：**折线图**
- 指标：
  - `core_norm_mean`
  - `core_norm_std`
  - `redistribute_gain_mean`
- 目的：看聚合 core 是否存在、是否起作用。

#### scale_seq 纹理图
- 图表：**heatmap**
- 画法：
  - x 轴：patch index
  - y 轴：channel 维（可抽样 32 个 channel 或 PCA 后 8 维）
  - 值：activation
- 目的：看输出序列是否保留时序结构。

---

## 9. 外生条件编码器（07_conditioning）

`ConditioningEncoder` 对 `sidechain_cond [B, 13, 64]` 做轻量编码，输出 `cond_seq [B, d_cond, 64]` 和 `cond_global [B, d_cond]`。

### 9.1 可视化目标

这个模块回答：

1. 外生条件编码是否有动态结构。
2. `cond_global` 是否过快塌为常数。
3. 13 个 side 特征是否都有贡献。

### 9.2 必须记录的指标

#### cond_seq 激活统计
- 图表：**折线图**
- 指标：
  - `cond_seq_mean`
  - `cond_seq_std`
  - `cond_seq_abs_mean`

#### cond_global 范数
- 图表：**折线图**
- 指标：
  - `cond_global_l2_mean`
  - `cond_global_l2_std`
- 目的：看全局条件是否塌缩。

#### side feature -> encoded token 热力图
- 图表：**heatmap**
- 画法：
  - x 轴：time index (64)
  - y 轴：encoded channel（或 side feature）
  - 值：activation magnitude
- 目的：看编码后的时间结构。

#### TemporalMixing / FeatureMixing 贡献
- 图表：**折线图**
- 指标：
  - `temporal_mixing_delta_l2`
  - `feature_mixing_delta_l2`
- 目的：看两个 mixer 是否都在工作。

#### cond_global 分布
- 图表：**histogram**
- 目的：看全局条件向量是否高度尖峰或接近常数。

---

## 10. 层级外生记忆（08_side_memory）

`SideMemoryHierarchy` 将 `cond_seq` 组织成三级记忆 `s1/g1, s2/g2, s3/g3`，分别对应完整 64 天、最后 12 天、最后 3 天上下文。

### 10.1 可视化目标

这个模块回答：

1. 三层记忆是否真正呈现不同时间尺度。
2. `s2/s3` 是否只是在复制 `s1`。
3. 层级截断后信息是否过度丢失。

### 10.2 必须记录的指标

#### 三层记忆长度语义检查
- 图表：**heatmap**
- 对象：注意力权重图
- 画法：
  - `s1`: self-attn map, x/y 均为 64 天 index
  - `s2`: query 为最后 12 天，key 为较长历史
  - `s3`: query 为最后 3 天，key 为 `s2` 的更长上下文
- 注意：
  - 每个 head 单独画，不要先对 head 求平均后只保留一张。
  - 可以选择 head 维做 facet，每个 head 一张图。
- 目的：因为这里的语义就在“query 看向哪段历史”，平均掉就失真。

#### 三层全局向量差异度
- 图表：**折线图**
- 指标：
  - `cos(g1, g2)`
  - `cos(g2, g3)`
  - `cos(g1, g3)`
- 目的：看三级记忆是否有层级区分。

#### 记忆压缩强度
- 图表：**折线图**
- 指标：
  - `s1_l2_mean`
  - `s2_l2_mean`
  - `s3_l2_mean`
  - `g1_l2_mean`
  - `g2_l2_mean`
  - `g3_l2_mean`
- 目的：看后面层级是否过度压扁。

#### 三层记忆概览对比
- 图表：**分组柱状图**
- 指标：
  - norm
  - std
  - entropy（若有 attention 权重）
- 目的：直接比较三层记忆“活性”。

---

## 11. 单尺度外生桥接融合（09_bridge_fusion）

`ExogenousBridgeFusion` 将每个尺度的内生序列与对应层级外生记忆做桥接融合，流程是：bridge token -> 对外生记忆 cross-attention -> gate 回灌到内生 token。

三个尺度必须拆开看：

- `bridge_macro`
- `bridge_mezzo`
- `bridge_micro`

### 11.1 可视化目标

这个模块回答：

1. 外生信息是否真的注入了内生序列。
2. bridge token 是否有信息量。
3. gate 是否塌到 0 / 1，或某尺度长期失活。
4. 不同尺度是否真的消费不同层级记忆。

### 11.2 必须记录的指标

#### bridge token 范数
- 图表：**折线图**
- 指标：
  - `bridge_token_l2_mean`
  - `bridge_global_l2_mean`
- 目的：看 bridge 本身是否工作。

#### gate 健康性
- 图表：**折线图**
- 指标：
  - `gate_mean`
  - `gate_std`
  - `gate_min`
  - `gate_max`
  - `gate_entropy`
- 目的：
  - gate_mean 长期接近 0：外生分支未被使用
  - gate_mean 长期接近 1：主干被外生强行覆盖
  - gate_std 很小：门控塌缩

#### gate over token position
- 图表：**heatmap**
- 画法：
  - x 轴：token / patch index
  - y 轴：channel（可抽样）或样本
  - 值：gate value
- 目的：看桥接回灌集中在哪些时序位置。
- 这里不能做简单加权平均，因为你要看到“靠近尾部 patch 是否更强”。

#### bridge attention 图
- 图表：**attention heatmap**
- 画法：
  - x 轴：side memory time index
  - y 轴：bridge query/head
  - 值：attention 权重
- 每个尺度单独画。
- 如果 multi-head：每个 head 单独一张图，不要先均值。
- 目的：看不同尺度分别在消费哪段外生历史。

#### 融合增量强度
- 图表：**折线图**
- 指标：
  - `delta_l2 = ||endogenous_fused - endogenous_in||`
- 目的：看外生注入强度是否过弱或过强。

#### 三尺度桥接对比
- 图表：**分组柱状图**
- 指标：
  - `gate_mean`
  - `gate_std`
  - `delta_l2`
  - `attention_entropy`
- 目的：判断 macro / mezzo / micro 哪个尺度最依赖 side memory。

---

## 12. 跨尺度融合（10_cross_scale）

`CrossScaleFusion` 将三个尺度 token 加 scale embedding 后拼接成统一 token 集，再由 `num_latents=8` 的 bottleneck latents 做 cross-attention、自注意力与 FFN 融合，输出 `fused_latents` 和 `fused_global`。

### 12.1 可视化目标

这个模块回答：

1. latent bottleneck 是否真的吸收了三尺度信息。
2. 是否只偏向某一个尺度。
3. 8 个 latent 是否都被使用，还是只有少数活跃。

### 12.2 必须记录的指标

#### 各尺度 token 使用比例
- 图表：**堆叠柱状图**
- 指标：
  - `latent_attn_to_macro`
  - `latent_attn_to_mezzo`
  - `latent_attn_to_micro`
- 计算：对 cross-attention 权重按 source scale 汇总。
- 目的：看 bottleneck 是否单押某一尺度。

#### latent-to-scale 注意力图
- 图表：**heatmap**
- 画法：
  - x 轴：source token index（按 macro / mezzo / micro 拼接，并在图上分段标记）
  - y 轴：latent index (0~7)
  - 值：attention weight
- 目的：这是本模块最关键的结构图。
- 注意：不能先把 latent 平均，否则看不出 latent 分工。

#### latent usage entropy
- 图表：**折线图**
- 指标：
  - `latent_usage_entropy`
  - `latent_usage_gini`
- 目的：看 8 个 latent 是否被平均使用。

#### 每个 latent 的激活范数
- 图表：**分组柱状图** 或 **多折线图**
- 指标：
  - `latent_00_norm`
  - ...
  - `latent_07_norm`
- 目的：看 latent collapse。

#### fused_global 健康性
- 图表：**折线图** + **histogram**
- 指标：
  - `fused_global_l2_mean`
  - `fused_global_std`
  - `fused_global_hist`
- 目的：看共享 trunk 顶层表示是否塌缩。

#### scale embedding 分离度
- 图表：**PCA/UMAP scatter image**
- 对象：固定验证 batch 的 `scale_tokens`
- 上色：macro / mezzo / micro
- 目的：看 scale embedding 是否建立了可分结构。
- 频率低，不要每 step 记。

---

## 13. 多任务输出头（11_heads）

`MultiTaskHeads` 基于共享 trunk 的 `fused_latents / fused_global` 为 `ret / rv / q` 三个任务生成任务特定表示，并输出各自的 raw 分布参数。

### 13.1 可视化目标

这个模块回答：

1. 三个任务是否真的学出了不同表示。
2. 某个任务塔是否失活。
3. raw 输出参数是否数值健康。
4. 不确定性头是否只会输出常数。

### 13.2 TaskQueryTower

#### 任务查询注意力图
- 图表：**heatmap**
- 画法：
  - x 轴：latent index
  - y 轴：task (`ret`, `rv`, `q`) × head index
  - 值：attention weight
- 目的：看不同任务是否在读取不同 latent。
- 不要先对 head 求平均。

#### 任务表示差异度
- 图表：**折线图**
- 指标：
  - `cos(task_repr_ret, task_repr_rv)`
  - `cos(task_repr_ret, task_repr_q)`
  - `cos(task_repr_rv, task_repr_q)`
- 目的：看三任务塔是否学成同一个向量。

#### 任务表示范数
- 图表：**折线图**
- 指标：
  - `task_repr_ret_l2`
  - `task_repr_rv_l2`
  - `task_repr_q_l2`
- 目的：看某一塔是否长期偏弱。

### 13.3 Value head / Uncertainty head

#### raw 输出统计
- 图表：**折线图** + **histogram**
- 指标：
  - `pred_mu_ret`
  - `pred_scale_ret_raw`
  - `pred_mean_rv_raw`
  - `pred_shape_rv_raw`
  - `pred_mu_q`
  - `pred_scale_q_raw`
- 每个指标至少记录：
  - mean
  - std
  - p01 / p99
  - histogram
- 目的：看 raw 参数是否数值失真。

#### 三任务输出稳定性对比
- 图表：**分组柱状图**
- 指标：
  - 各头输出 std
  - 各头输出 gradient norm
- 目的：看哪个头在主导或失活。

#### 预测-目标散点图
- 图表：**scatter**
- 对象：固定验证 batch
- 分别画：
  - `pred_mu_ret` vs `target_ret`
  - `pred_mean_rv_or_expectation` vs `target_rv`
  - `pred_mu_q` vs `target_q`
- 目的：给最直观的学习效果图。
- 散点图比单纯 corr 更有语义。

---

## 14. 分布式损失层（12_loss）

`MultiTaskDistributionLoss` 将 raw 参数转成合法分布参数，分别计算：

- `ret -> Student-t NLL`
- `rv -> Gamma NLL`
- `q -> Asymmetric Laplace NLL`

并返回 `loss_total, loss_ret, loss_rv, loss_q, nu_ret, sigma_ret_pred, sigma_rv_pred, sigma_q_pred`。

### 14.1 可视化目标

这个模块回答：

1. loss 是否真在下降。
2. 合法分布参数是否数值合理。
3. uncertainty 是否有分辨率，还是输出常数。
4. 某任务的 NLL 是否因为参数退化而“假性变好/变坏”。

### 14.2 必须记录的指标

#### 总 loss 与分任务 NLL
- 图表：**折线图**
- 指标：
  - `train/loss/total`
  - `train/loss/ret_nll`
  - `train/loss/rv_nll`
  - `train/loss/q_nll`
  - `val/loss/...`

#### loss 分解柱状图
- 图表：**分组柱状图**
- 每个 eval epoch 画 train/val 的 ret, rv, q 对比。
- 目的：一眼看出瓶颈任务。

### 14.3 ret: Student-t

#### `nu_ret`
- 图表：**折线图**
- 指标：
  - `train/loss_params/ret/nu`
  - `val/loss_params/ret/nu`
- 目的：看自由度是否走向不合理极端。

#### `sigma_ret_pred`
- 图表：**折线图** + **histogram**
- 指标：
  - mean/std/p01/p99/hist
- 目的：看不确定性是否塌缩。

#### ret calibration 图
- 图表：**分箱折线图 / reliability plot**
- 做法：按 `sigma_ret_pred` 分桶，统计每桶的实际绝对误差或 NLL。
- 目的：看预测不确定性是否有校准意义。

### 14.4 rv: Gamma

#### `sigma_rv_pred` / shape 健康性
- 图表：**折线图** + **histogram**
- 指标：
  - legal mean
  - legal std
  - p01/p99
  - hist
- 目的：看正值化后是否常贴下界。

#### rv calibration 图
- 图表：**分箱折线图**
- 做法：按预测 scale 或 shape 分桶，统计实际 rv 误差。

### 14.5 q: Asymmetric Laplace

#### `sigma_q_pred`
- 图表：**折线图** + **histogram**
- 指标同上。

#### 分位偏差图
- 图表：**分箱柱状图**
- 指标：
  - 预测分位覆盖率偏差
- 目的：看 `q_tau` 对应分位是否有系统偏差。

### 14.6 全任务 uncertainty 对比

#### uncertainty 三任务对比
- 图表：**分组三柱状图**
- 指标：
  - `sigma_ret_pred_mean`
  - `sigma_rv_pred_mean`
  - `sigma_q_pred_mean`
  - 对应 std
- 目的：看不确定性头是否只有某一路在工作。

#### uncertainty vs error 散点图
- 图表：**scatter**
- 每个任务都画：
  - x：predicted uncertainty
  - y：actual abs error / NLL contribution
- 目的：判断不确定性是否有排序意义。

---

## 15. 优化器、梯度与系统状态（13_optimizer_and_system）

这一部分不属于模型主结构，但对“健康状态”是强制项。

### 15.1 必须记录的指标

#### 全局优化状态
- 图表：**折线图**
- 指标：
  - `train/optim/lr`
  - `train/optim/grad_norm_global`
  - `train/optim/param_norm_global`
  - `train/optim/update_to_weight_ratio`

#### 分模块梯度范数
- 图表：**分组柱状图**
- 模块：
  - wavelet
  - encoder_macro
  - encoder_mezzo
  - encoder_micro
  - within_scale_fusion
  - conditioning
  - side_memory
  - bridge_macro
  - bridge_mezzo
  - bridge_micro
  - cross_scale
  - heads_ret
  - heads_rv
  - heads_q
- 目的：看梯度是否只流向后端 head。

#### 分模块权重范数
- 图表：**分组柱状图**
- 目的：辅助判断某模块是否长期近零。

#### 梯度/权重直方图
- 图表：**histogram**
- 只对关键模块记录，不要全模型每步都记。
- 推荐频率：每个 val epoch。

#### 性能指标
- 图表：**折线图**
- 指标：
  - step_time
  - data_time
  - forward_time
  - backward_time
  - optimizer_time
  - gpu_mem_alloc
  - gpu_mem_reserved
- 目的：避免把 IO/compile/perf 问题误判成模型问题。

---

## 16. 个案分析页（14_case_study）

这是最有语义的一页，用来回答：**模型到底在“看什么、依赖什么、输出什么”。**

### 16.1 固定 case table

- 图表：**W&B Table**
- 每个样本一行，字段至少包含：
  - `sample_id`
  - `date`
  - `code`
  - `target_ret`
  - `pred_mu_ret`
  - `sigma_ret`
  - `target_rv`
  - `pred_rv`
  - `sigma_rv`
  - `target_q`
  - `pred_mu_q`
  - `sigma_q`
  - `ret_abs_error`
  - `rv_abs_error`
  - `q_abs_error`
  - `case_type`（normal / trend / noisy）

### 16.2 每个 case 的结构图

每个固定样本至少保存以下图：

1. `macro/mezzo/micro` 输入热力图
2. wavelet 前后对比图
3. encoder feature-patch heatmap（每尺度一张）
4. side memory attention 图
5. bridge gate 图（每尺度一张）
6. cross-scale latent-to-scale attention 图
7. task query attention 图
8. 预测分布参数摘要

这组图的目的不是大量抽样，而是形成“剖面图”，让你能追一个样本从输入一路看到输出。

---

## 17. 图表类型选择原则（强制）

以下原则必须执行：

### 17.1 什么时候用折线图

用于看：
- 随 step/epoch 的趋势
- 收敛过程
- 稳定性变化

适用：
- loss
- norm
- mean/std
- entropy
- throughput

### 17.2 什么时候用 histogram

用于看：
- 分布是否塌缩
- 是否长尾爆炸
- 是否贴边界

适用：
- activation
- raw head outputs
- legal sigma / shape
- gradients

### 17.3 什么时候用 heatmap

用于看：
- 有两个语义轴，且平均会丢失语义

适用：
- feature × patch
- token × time
- latent × scale token
- head × memory index
- gate × token position

只要图的语义依赖二维结构，就必须优先用 heatmap，而不是简单平均后画线。

### 17.4 什么时候用柱状图

用于看：
- 模块间比较
- 任务间比较
- 类别分布

适用：
- 离散 state/pos 分布
- 三任务 loss 对比
- 三尺度 gate 对比
- 分模块 grad norm 对比

### 17.5 什么时候用 scatter

用于看：
- 预测值与真值关系
- uncertainty 与 error 的相关性
- 表示空间分离度

适用：
- pred vs target
- uncertainty vs abs error
- scale token PCA/UMAP

---

## 18. 最低实现清单（MVP）

如果第一阶段只做最小可用版，必须先实现以下内容：

### 18.1 必做 scalar

- `train/val loss_total`
- `train/val loss_ret, loss_rv, loss_q`
- `lr`
- `global_grad_norm`
- `global_param_norm`
- `gate_mean/std` for each bridge scale
- `nu_ret`
- `sigma_ret_pred_mean`
- `sigma_rv_pred_mean`
- `sigma_q_pred_mean`
- `fused_global_l2_mean`
- `task_repr_*_l2`

### 18.2 必做 heatmap

- encoder feature-patch heatmap for each scale
- bridge gate heatmap for each scale
- cross-scale latent-to-scale attention heatmap
- task query attention heatmap

### 18.3 必做 histogram

- final encoder activation hist for each scale
- gate hist for each bridge scale
- raw head output hist for each task
- legal sigma hist for each task

### 18.4 必做 case study

- 固定验证样本表
- 每个样本至少保存：
  - wavelet 前后图
  - bridge gate 图
  - cross-scale attention 图
  - pred/target 摘要

---

## 19. 实现注意事项

### 19.1 不要过度记录

以下内容不要每 step 记：

- attention heatmap
- full histogram of every layer
- PCA/UMAP
- case study tables

建议：
- scalar 高频
- histogram 中频
- heatmap / case study 低频

### 19.2 所有 heatmap 必须保留原始轴语义

不要把：
- head 维直接平均掉
- patch 维直接平均掉
- feature 维直接平均掉

除非这个维度本身不是分析重点。

### 19.3 所有比较图必须可横向对比

例如：
- macro / mezzo / micro 三个 bridge gate 图要同量纲
- ret / rv / q 的 uncertainty 图要固定各自坐标范围策略
- train / val 同类图要布局一致

### 19.4 所有固定样本图必须复用同一批样本

否则 epoch 间不可比较，热力图会失去意义。

---

## 20. 一句话落地要求

这套可视化不是为了“展示有很多图”，而是为了让你在 W&B 里能直接回答下面这些问题：

1. 模型有没有真正学到任务，而不是只把 loss 压下去。
2. 哪个尺度在工作，哪个尺度在掉线。
3. side memory 是否真的被不同尺度消费。
4. bridge gate 是否健康，是否塌缩。
5. latent bottleneck 是否在整合三尺度，而不是偏向单一路。
6. 三个任务头是否真的学出不同表示。
7. uncertainty 是否有校准意义，而不是常数输出。
8. 如果训练坏了，坏在输入、编码器、融合层、头，还是 loss 参数化层。

如果某个图无法帮助回答这些问题，就不要记录。
