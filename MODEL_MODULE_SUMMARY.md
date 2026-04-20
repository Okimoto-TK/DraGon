# 模型模块组成总结

主装配入口在 [src/models/arch/networks/multi_scale_forecast_network.py](/root/DraGon/src/models/arch/networks/multi_scale_forecast_network.py:35)。

整体链路可以概括为：

`原始多尺度序列 -> 小波去噪 -> 三个单尺度编码器 -> 尺度内融合 -> 条件编码/层级外生记忆 -> 单尺度桥接融合 -> 跨尺度融合 -> 多任务输出头 -> 分布式损失`

## 1. 整体定位

这是一个面向时序预测的多尺度、多任务网络，核心特点有三点：

- 同时建模 `macro / mezzo / micro` 三个时间尺度
- 引入 `sidechain_cond` 作为外生条件分支
- 对三个任务 `ret / rv / q` 采用分布式参数化输出，而不是单点回归

## 2. 输入组成

模型前向依赖的主要 batch 字段如下：

- `macro_float_long [B, 9, 112]`
- `macro_i8_long [B, 2, 112]`
- `mezzo_float_long [B, 9, 144]`
- `mezzo_i8_long [B, 2, 144]`
- `micro_float_long [B, 9, 192]`
- `micro_i8_long [B, 2, 192]`
- `sidechain_cond [B, 13, 64]`

其中：

- `*_float_long` 是三个尺度的连续特征序列
- `*_i8_long` 的两个通道分别承载离散 `state` 和 `pos`
- `sidechain_cond` 是外生条件序列

## 3. 模块主链路

主网络 `MultiScaleForecastNetwork` 的执行顺序如下：

1. `WaveletDenoise1D`
2. `ModernTCNFiLMEncoder` x 3
3. `WithinScaleSTARFusion` x 3
4. `ConditioningEncoder`
5. `SideMemoryHierarchy`
6. `ExogenousBridgeFusion` x 3
7. `CrossScaleFusion`
8. `MultiTaskHeads`
9. `MultiTaskDistributionLoss`

## 4. 各模块职责

### 4.1 小波去噪前端

模块：[src/models/arch/layers/wavelet_denoise.py](/root/DraGon/src/models/arch/layers/wavelet_denoise.py:21)

类：`WaveletDenoise1D`

职责：

- 对带 warmup 的长序列做固定小波分解与细节带收缩
- 输出送入编码器的目标长度序列

三个尺度的目标长度分别是：

- `macro: 64`
- `mezzo: 96`
- `micro: 144`

对应地，原始输入长度包含额外 warmup：

- `macro: 64 + 48 = 112`
- `mezzo: 96 + 48 = 144`
- `micro: 144 + 48 = 192`

这个模块只负责去噪，不负责编码、融合或预测。

### 4.2 单尺度编码器

模块：[src/models/arch/encoders/modern_tcn_film_encoder.py](/root/DraGon/src/models/arch/encoders/modern_tcn_film_encoder.py:122)

类：`ModernTCNFiLMEncoder`

职责：

- 对每个尺度单独编码
- 以“每个 feature 一条序列”的方式做 patch 化和时序建模
- 用离散条件 `state / pos` 对编码过程做 FiLM 调制

内部关键子模块：

- `Patch1D`：[src/models/arch/layers/patch1d.py](/root/DraGon/src/models/arch/layers/patch1d.py:10)
- `ConditionEmbedding1D`：[src/models/arch/embeddings/condition_embedding1d.py](/root/DraGon/src/models/arch/embeddings/condition_embedding1d.py:10)
- `FiLM1D`：[src/models/arch/layers/film1d.py](/root/DraGon/src/models/arch/layers/film1d.py:9)
- `ChannelFFN1D`：[src/models/arch/layers/channel_ffn1d.py](/root/DraGon/src/models/arch/layers/channel_ffn1d.py:9)

每个 `ModernTCNFiLMBlock` 的结构大致是：

- `LayerNorm -> FiLM -> depthwise temporal conv -> residual`
- `LayerNorm -> FiLM -> channel FFN -> residual`

编码器输出张量形状为：

- `[B, F, D, N]`

其中：

- `F = 9`
- `D = hidden_dim = 128`
- `N` 为 patch 数

三个尺度的 patch 数分别是：

- `macro: 16`
- `mezzo: 24`
- `micro: 36`

### 4.3 尺度内融合

模块：[src/models/arch/fusions/within_scale_star_fusion.py](/root/DraGon/src/models/arch/fusions/within_scale_star_fusion.py:84)

类：`WithinScaleSTARFusion`

职责：

- 在单个尺度内部沿 feature 维度做融合
- 不做跨尺度操作

做法：

- 将 `[B, F, D, N]` 重排后，对每个 patch 位置在 feature 轴上做 STAR 风格的 aggregate-redistribute
- 先聚合出一个 core 表示，再回灌到各 feature token

输出有两个：

- `z_fused [B, F, D, N]`
- `scale_seq [B, D, N]`

在主网络中，后续主干继续使用的是 `scale_seq`。

### 4.4 外生条件编码器

模块：[src/models/arch/encoders/conditioning_encoder.py](/root/DraGon/src/models/arch/encoders/conditioning_encoder.py:88)

类：`ConditioningEncoder`

职责：

- 对 `sidechain_cond [B, 13, 64]` 做轻量编码
- 提供外生序列记忆和全局条件向量

内部是轻量 TSMixer 风格：

- `TemporalMixing1D`：[src/models/arch/layers/temporal_mixing1d.py](/root/DraGon/src/models/arch/layers/temporal_mixing1d.py:10)
- `FeatureMixing1D`：[src/models/arch/layers/feature_mixing1d.py](/root/DraGon/src/models/arch/layers/feature_mixing1d.py:10)

输出：

- `cond_seq [B, d_cond, 64]`
- `cond_global [B, d_cond]`

当前默认：

- `d_cond = 32`

### 4.5 层级外生记忆

模块：[src/models/arch/networks/side_memory_hierarchy.py](/root/DraGon/src/models/arch/networks/side_memory_hierarchy.py:139)

类：`SideMemoryHierarchy`

职责：

- 将 64 天外生条件记忆组织成三级 side memory
- 让不同尺度使用不同时间范围的外生上下文

输出为：

- `s1, g1`：完整 64 天记忆
- `s2, g2`：最后 12 天层级记忆
- `s3, g3`：最后 3 天层级记忆

内部逻辑：

- 先对完整条件序列做一次 self-attention，形成 `s1`
- 再让最后 12 天 query 前面更长窗口的 key/value，形成 `s2`
- 再让最后 3 天 query `s2` 里更长的上下文，形成 `s3`

这一步的作用是把一条 64 天条件序列转换成适合不同尺度消费的外生记忆。

### 4.6 单尺度外生桥接融合

模块：[src/models/arch/fusions/exogenous_bridge_fusion.py](/root/DraGon/src/models/arch/fusions/exogenous_bridge_fusion.py:112)

类：`ExogenousBridgeFusion`

职责：

- 将单尺度内生序列与对应级别的外生记忆做桥接融合
- 不做跨尺度融合

三个尺度的对应关系是：

- `macro <- s1 / g1`
- `mezzo <- s2 / g2`
- `micro <- s3 / g3`

机制上是 TimeXer 风格：

- 先从内生序列的均值生成一个 bridge token
- 用 bridge token 对外生序列做 cross-attention
- 再把融合后的 bridge 信息按 gate 回灌到每个内生 token

输出：

- `endogenous_fused [B, D, N]`
- `bridge_global [B, D]`

### 4.7 跨尺度融合

模块：[src/models/arch/fusions/cross_scale_fusion.py](/root/DraGon/src/models/arch/fusions/cross_scale_fusion.py:98)

类：`CrossScaleFusion`

职责：

- 将三个尺度的序列表示统一到一个 latent bottleneck 中
- 完成真正的跨尺度信息汇聚

做法：

- 给 `macro / mezzo / micro` 各加一个 scale embedding
- 把三组 token 拼接成统一的 `scale_tokens`
- 使用一组可学习的 latent tokens 对所有尺度 token 做 cross-attention
- 再让 latent 自注意力和 FFN 继续融合

输出：

- `fused_latents [B, D, num_latents]`
- `fused_global [B, D]`

当前默认：

- `hidden_dim = 128`
- `num_latents = 8`

### 4.8 多任务输出头

模块：[src/models/arch/heads/multi_task_heads.py](/root/DraGon/src/models/arch/heads/multi_task_heads.py:9)

类：`MultiTaskHeads`

依赖的任务塔模块：

- [TaskQueryTower](/root/DraGon/src/models/arch/heads/task_query_tower.py:7)

职责：

- 基于共享 trunk 的 `fused_latents / fused_global`
- 为 `ret / rv / q` 三个任务分别生成任务特定表示
- 再分别输出分布参数

结构上：

- 三个任务各自有独立 `TaskQueryTower`
- 三个任务各自有独立 value head
- 三个任务各自有独立 uncertainty head

各任务输出字段：

- `ret`
  - `pred_mu_ret`
  - `pred_scale_ret_raw`
- `rv`
  - `pred_mean_rv_raw`
  - `pred_shape_rv_raw`
- `q`
  - `pred_mu_q`
  - `pred_scale_q_raw`

`TaskQueryTower` 的机制是：

- 用一个 task token 加上 `fused_global`
- 作为 query 去 cross-attend `fused_latents`
- 得到该任务自己的表示向量

### 4.9 分布式损失层

模块：[src/models/arch/losses/multi_task_loss.py](/root/DraGon/src/models/arch/losses/multi_task_loss.py:14)

类：`MultiTaskDistributionLoss`

职责：

- 将三个任务的 raw 输出参数转换成合法分布参数
- 计算各自的 NLL / 分布损失
- 按权重汇总成总 loss

三个任务的建模方式分别是：

- `ret -> Student-t NLL`
- `rv -> Gamma NLL`
- `q -> Asymmetric Laplace NLL`

这里有两个关键点：

- heads 输出的是 raw 参数，不直接保证正值
- 正值化和下界约束都在 loss 层内部完成

最终返回的核心项包括：

- `loss_total`
- `loss_ret`
- `loss_rv`
- `loss_q`
- `nu_ret`
- `sigma_ret_pred`
- `sigma_rv_pred`
- `sigma_q_pred`

## 5. 当前默认超参数骨架

配置文件：[config/models.py](/root/DraGon/config/models.py:1)

主要开放超参数包括：

- `hidden_dim = 128`
- `cond_dim = 32`
- `num_latents = 8`
- `WithinScaleSTARFusion.num_features = 9`
- `ExogenousBridgeFusion.num_heads = 4`
- `CrossScaleFusion.num_heads = 4`
- `MultiTaskHeads.tower_num_heads = 4`
- `MultiTaskLoss.q_tau = 0.05`

隐含长度和尺度相关参数在：

- [src/models/config/hparams.py](/root/DraGon/src/models/config/hparams.py:95)

## 6. 一句话总结

你的模型本质上是一个：

> 先对 `macro / mezzo / micro` 三个尺度分别做去噪和单尺度编码，再将 `sidechain_cond` 编码成层级外生记忆注入各尺度，之后通过 bottleneck latents 完成跨尺度汇聚，最后对 `ret / rv / q` 三个任务分别输出分布参数并计算对应分布损失的多尺度多任务时序预测网络。
