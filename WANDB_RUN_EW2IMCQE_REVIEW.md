# W&B Run `ew2imcqe` 全窗口审查与代码对照

审查时间：2026-04-20

审查对象：
- Run: `https://wandb.ai/okimotolyx-huazhong-university-of-science-and-technology/dragon/runs/ew2imcqe`
- 代码入口：
  - `src/train/trainer.py`
  - `src/train/wandb_logger.py`
  - `src/models/arch/networks/multi_scale_forecast_network.py`
  - `src/models/arch/fusions/cross_scale_fusion.py`
  - `src/models/arch/fusions/exogenous_bridge_fusion.py`
  - `src/models/arch/fusions/within_scale_star_fusion.py`
  - `src/models/arch/heads/multi_task_heads.py`
  - `src/models/arch/heads/task_query_tower.py`
  - `src/models/arch/losses/multi_task_loss.py`

数据来源：
- 用 `wandb.Api().run(...).scan_history()` 拉取全量 history。
- 当前导出的数值指标汇总文件：`wandb_run_ew2imcqe_metric_summary.csv`

## 0. Run 状态与时间窗口

- 这条 run 在审查时仍是 `running`，不是已结束 run。
- `created_at`: `2026-04-20T09:56:15Z`
- `heartbeat_at`: `2026-04-20T11:50:08Z`
- `global_step` 已到 `32776`
- `val` 出现在 `8218`、`16436`、`24654`
- 因为 run 仍在运行，`24654 -> 32776` 这段不是“训练结束后没验证”，而是“第 4 个 epoch 正在跑，尚未到 epoch 末尾”

这点会影响结论：
- 不能把 `24654` 之后没有 `val` 解读成训练流程漏验证
- 但可以明确说：截至当前抓取窗口，结构性指标已经出现了很强的偏移

## 1. 代码里实际记录了什么

`src/train/wandb_logger.py` 里实际使用的 W&B 类型只有三类：
- 标量：`self._wandb.log(...)`
- Histogram：`self._wandb.Histogram(...)`
- Image：`self._wandb.Image(...)`

没有看到这些类型：
- `wandb.Table`
- `wandb.Audio`
- `wandb.Video`
- `wandb.Html`
- `wandb.Plotly`

对应代码位置：
- 标量定义与 step 绑定：`src/train/wandb_logger.py:167-201`
- Histogram：`src/train/wandb_logger.py:568-585`
- Image：`src/train/wandb_logger.py:633-774`

当前 run 中实际拿到的内容：
- 数值标量：145 个
- Histogram summary key：9 个
- `viz/*` 图片：0 个

这和代码一致的部分：
- 标量训练日志每 `50` step 打一次
- Histogram 每 `500` step 打一次

这和代码不一致的预期：
- 代码有 `Image`，但这条 run 没有任何 `viz/*`

## 2. 为什么你看不到除了 histogram 和 scalar 之外的其他类型

原因不是 W&B 前端隐藏，而是这条 run 根本没有把 `Image` 打出来。

证据：
- `run.summary.keys()` 里没有任何 `viz/*`
- `run.files()` 只有 `requirements.txt` 和 `wandb-metadata.json`
- 本地 `wandb/run-20260420_175614-ew2imcqe/` 下也没有 media 文件

根因在代码逻辑：
- `Trainer` 只有在验证阶段才可能打 `log_fixed_val_snapshot(...)`
- 触发条件是 `self.wandb_logger.should_log_visuals(self.global_step)`
- `should_log_visuals` 要求 `global_step % viz_every == 0`
- 默认 `viz_every=1000`

对应代码：
- `src/train/wandb_logger.py:191-195`
- `src/train/trainer.py:149-153`
- `src/train/trainer.py:224-237`
- `src/train/trainer.py:277-283`

而这条 run 的验证 step 是：
- `8218`
- `16436`
- `24654`

这三个 step 都不是 `1000` 的整数倍，所以 `log_fixed_val_snapshot(...)` 一次都没进。

结论：
- 代码里除了 scalar 和 histogram 之外，确实还有 `Image`
- 但当前 run 因为调度条件没有命中，图片完全没有被记录

## 3. 145 个数值指标的全窗口审查

### 3.1 指标覆盖

145 个数值指标全部被覆盖，按代码模块可归类为：
- `train_loss`: 4
- `val_loss`: 4
- `train_loss_share`: 3
- `val_loss_share`: 3
- `train_health`: 4
- `train_system`: 7
- `train_input_*`: 16
- `train_wavelet_*`: 9
- `train_encoder_*`: 9
- `train_within_scale_*`: 9
- `train_conditioning`: 5
- `train_side_memory`: 9
- `train_bridge_*`: 21
- `train_cross_scale`: 13
- `train_heads`: 6
- `train_heads_pred_*`: 12
- `train_loss_params*`: 7
- `trainer/*`: 2
- `val_optimizer`: 1

### 3.2 Loss 与验证

训练 loss 全窗口整体改善：
- `train_loss/total`: `-7.364 -> -7.983 -> -8.076`
- `train_loss/ret_nll`: `-2.102 -> -2.312 -> -2.347`
- `train_loss/rv_nll`: `-3.497 -> -3.638 -> -3.663`
- `train_loss/q_nll`: `-1.765 -> -2.033 -> -2.067`

验证 loss 3 次都在改善：
- `val_loss/total`: `-7.753 -> -7.965 -> -8.038`
- `val_loss/ret_nll`: `-2.233 -> -2.311 -> -2.330`
- `val_loss/rv_nll`: `-3.573 -> -3.639 -> -3.658`
- `val_loss/q_nll`: `-1.947 -> -2.014 -> -2.050`

判断：
- 优化本身是有效的
- 没有出现 train/val 明显背离
- 但验证频率只在 epoch 末，结构性退化是否同步反映到泛化上，目前证据不够

### 3.3 Loss share 与分任务重心

`q` 权重占比持续升高，`rv` 持续下降：
- `train_loss_share/q`: `0.239 -> 0.255 -> 0.256`
- `train_loss_share/rv`: `0.475 -> 0.456 -> 0.454`
- `train_loss_share/ret`: `0.286 -> 0.290 -> 0.291`

验证端也是同方向：
- `val_loss_share/q`: `0.251 -> 0.253 -> 0.255`
- `val_loss_share/rv`: `0.461 -> 0.457 -> 0.455`

判断：
- 训练后期越来越偏向 `q`
- 这不一定是问题，但需要和任务权重设计一起看

### 3.4 Health 与 System

整体稳定，没有数值层面的训练崩坏：
- `global_grad_norm` 没有随时间失控
- `global_param_norm`: `147.06 -> 151.82 -> 156.81`
- `global_act_std`: `1.33 -> 1.60 -> 2.03`

系统性能基本稳定：
- `samples_per_sec`: `4210 -> 4224 -> 4169`
- `step_time_ms`: `245.8 -> 243.3 -> 247.3`
- `gpu_mem_alloc_mb`: `564.2 -> 559.1 -> 559.1`

判断：
- 没有显著的吞吐或显存异常
- 但激活尺度在扩大，后面会和 heads / encoder 的尺度漂移对上

### 3.5 输入、wavelet、conditioning

输入统计基本不动：
- `train_input_macro/mean,std,min,max`
- `train_input_mezzo/mean,std,min,max`
- `train_input_micro/mean,std,min,max`
- `train_input_sidechain/mean,std,min,max`

这说明数据分布本身没有漂。

wavelet 组整体健康：
- `macro` 去噪比例轻微下降：`0.814 -> 0.805 -> 0.801`
- `mezzo` 去噪比例轻微上升：`0.830 -> 0.839 -> 0.845`
- `micro` 去噪比例轻微上升：`0.792 -> 0.796 -> 0.804`

conditioning 组变化温和：
- `cond_global_l2_mean`: `10.70 -> 11.43 -> 11.79`
- `cond_global_l2_std`: `1.12 -> 0.93 -> 0.88`
- `cond_seq_std`: `1.81 -> 1.90 -> 1.97`

判断：
- 输入、预处理、conditioning 没有明显病灶
- 结构性偏移更可能发生在 fusion / cross-scale / heads

### 3.6 Encoder 与 within-scale fusion

Encoder 三个尺度的激活绝对值和标准差都在上升：
- `macro act_abs_mean`: `1.09 -> 1.32 -> 1.49`
- `macro act_std`: `1.66 -> 1.97 -> 2.25`
- `mezzo act_abs_mean`: `1.04 -> 1.53 -> 1.81`
- `mezzo act_std`: `1.53 -> 2.36 -> 2.87`
- `micro act_abs_mean`: `0.89 -> 1.14 -> 1.30`
- `micro act_std`: `1.30 -> 1.80 -> 2.05`

within-scale 指标的真正含义需要注意：
- 代码里 `feature_var_pre/post` 不是 variance
- 它实际是 `_offdiag_mean_cosine_distance(...)`
- 对应代码：
  - `src/train/wandb_logger.py:117-126`
  - `src/train/wandb_logger.py:430-434`

所以这里更准确的解释是“特征间平均余弦距离”，不是“方差”。

分尺度看：
- `macro` 最稳，`feature_diversity_ratio` 近似 `1.0`
- `micro` 明显收缩，`1.002 -> 0.912 -> 0.879`
- `mezzo` 最差，`0.838 -> 0.580 -> 0.541`

其中 `mezzo` 的特征距离表现最异常：
- `feature_var_pre` 上升：`0.356 -> 0.417 -> 0.431`
- `feature_var_post` 下降：`0.298 -> 0.242 -> 0.234`
- `feature_diversity_ratio` 大幅下降

判断：
- `macro` 内部融合基本稳定
- `micro` 有一定特征收缩
- `mezzo` 的 within-scale 融合在持续压缩差异，且与后面的 cross-scale/bridge 指标一致，说明 `mezzo` 通路在系统性失活

### 3.7 Side memory

side memory 的向量范数和相似度都在下降：
- `g1_l2_mean`: `7.74 -> 5.06 -> 4.33`
- `g2_l2_mean`: `6.10 -> 3.82 -> 3.78`
- `g3_l2_mean`: `6.13 -> 4.87 -> 5.11`
- `cos_g1_g2`: `0.772 -> 0.598 -> 0.592`
- `cos_g1_g3`: `0.497 -> 0.363 -> 0.372`
- `cos_g2_g3`: `0.823 -> 0.815 -> 0.799`

判断：
- side memory 在整体收缩
- `g1` 和 `g2` 的耦合下降比较明显
- 单看这些指标还不能判定“坏”，但它会放大 downstream bridge 的路径偏置

### 3.8 Bridge 三组

#### macro bridge

- `gate_mean`: `0.592 -> 0.799 -> 0.802`
- `gate_entropy`: `0.580 -> 0.425 -> 0.414`
- `delta_l2`: `3.59 -> 6.44 -> 8.05`

解释：
- `macro` gate 越来越常开
- gate 分布越来越确定
- 融合注入强度持续提高

#### mezzo bridge

- `gate_mean`: `0.475 -> 0.379 -> 0.327`
- `gate_entropy`: `0.645 -> 0.628 -> 0.591`
- `bridge_token_l2_mean`: `18.91 -> 38.90 -> 55.09`
- `delta_l2`: `2.80 -> 1.84 -> 2.66`

解释：
- `mezzo` gate 在持续关小
- 但 bridge token 本身范数在疯狂变大
- 这不是“完全不用 exogenous”，而是“中间表示变大，但真正注入 endogenous 的 gate 越来越保守”

这是非常值得警惕的一组。

#### micro bridge

- `gate_mean`: `0.468 -> 0.439 -> 0.424`
- `gate_entropy`: `0.635 -> 0.627 -> 0.601`
- `delta_l2`: `7.40 -> 11.08 -> 13.40`

解释：
- `micro` gate 轻微收缩
- 但实际融合增量越来越大
- 这说明 `micro` 仍在工作，而且注入强度上升

综合 bridge 三组：
- `macro` 越来越主导
- `micro` 次之
- `mezzo` 逐步被压弱

### 3.9 Cross-scale

这是当前最关键的一组。

首先看路由：
- `latent_attn_to_macro`: `0.659 -> 0.769 -> 0.774`
- `latent_attn_to_mezzo`: `0.028 -> 0.003 -> 0.003`
- `latent_attn_to_micro`: `0.313 -> 0.228 -> 0.223`

结论很直接：
- cross-scale 融合几乎不再看 `mezzo`
- 注意力质量越来越压到 `macro`
- `micro` 也在退，但没有 `mezzo` 那么极端

然后看 `latent_usage_entropy` 和 `latent_usage_gini`：
- entropy 全程恒等于 `2.079442 = ln(8)`
- gini 全程恒等于 `0`

这不是模型真的“完美均匀使用 8 个 latent”，而是日志公式有问题。

原因：
- `cross_attn` 的每个 latent query 对 source token 的注意力本来就会在 token 维求和为 1
- 当前代码却对 token 维先做 mean：
  - `usage = cross_attn.detach().float().mean(dim=(0, 1, 3))`
  - `src/train/wandb_logger.py:503-507`
- 对 token 维取平均后，每个 latent 天然都会得到同一个常数
- 后续再归一化，必然是严格均匀分布
- 所以这两个指标按现在的实现，数学上几乎注定是常数，没有诊断价值

然后看 `latent_00_norm` 到 `latent_07_norm`：
- 8 条曲线逐点完全一致

这里不是 logger 公式问题，而是架构存在对称性风险。

原因在 `CrossScaleFusion`：
- `self.latents` 初始化为全 0：`src/models/arch/fusions/cross_scale_fusion.py:171`
- 没有任何 latent id embedding / positional identity
- 在对称初始化下，8 个 latent query 的前向路径完全同构

结果是：
- 每个 latent 的范数曲线完全相同
- 实际上 8 个 latent 很可能始终没有分化出职责

这是比 `latent_usage_entropy` 更严重的问题，因为它是模型结构问题，不是单纯日志定义问题。

### 3.10 Heads 与 loss params

任务表征范数持续抬升，非常明显：
- `task_repr_ret_l2`: `30.38 -> 73.29 -> 107.50`
- `task_repr_rv_l2`: `31.72 -> 56.86 -> 82.70`
- `task_repr_q_l2`: `34.87 -> 89.49 -> 136.86`

但任务间余弦相似度没有塌缩到同向：
- `ret-rv`: `-0.068 -> -0.063 -> -0.061`
- `ret-q`: `-0.073 -> -0.067 -> -0.076`
- `rv-q`: `0.072 -> 0.089 -> 0.084`

解释：
- 任务表征方向关系还算稳定
- 但表征尺度在持续上飘

head 原始输出分布：
- `pred_scale_ret_raw/mean`: 更负，代表 `scale_ret` 在下降
- `pred_scale_q_raw/mean`: 更负，代表 `scale_q` 在下降
- `pred_shape_rv_raw/mean`: 明显上升，代表 Gamma shape 增大

和 `loss_params` 一起看：
- `ret_nu`: `7.28 -> 4.57 -> 3.62`
- `ret_sigma mean`: `0.0317 -> 0.0277 -> 0.0288`
- `rv_sigma mean`: `0.00808 -> 0.00695 -> 0.00683`
- `q_sigma mean`: `0.0639 -> 0.0500 -> 0.0487`

判断：
- 三个头的预测不确定性整体在收缩
- 这和 loss 变好是同方向的
- 但和 task repr / encoder act 的持续放大同时出现，提示“表征尺度增大 + 输出分布变尖”的组合
- 当前还没失控，但值得跟踪

## 4. 与代码直接相关的明确问题

### 问题 1：`viz/*` 图片日志调度几乎不会命中

证据：
- `should_log_visuals(global_step)` 要求 `global_step % viz_every == 0`
- 只在验证阶段检查
- 当前 epoch 长度和 `viz_every` 不对齐，导致 0 次图片日志

影响：
- 代码里有 `Image`
- 但 run 里完全看不到任何 `viz/*`

### 问题 2：所谓 “fixed val snapshot” 实际上不是 fixed

证据：
- `capture_fixed_val_batch()` 和 `get_fixed_val_batch()` 是空实现：`src/train/wandb_logger.py:185-189`
- `Trainer` 在验证时直接把当前 batch 传给 `log_fixed_val_snapshot(...)`：`src/train/trainer.py:277-283`

影响：
- 即使未来打出了图，也未必是同一批样本
- 跨 epoch 图像比较会混入样本差异

### 问题 3：`latent_usage_entropy` / `latent_usage_gini` 当前实现没有信息量

证据：
- `src/train/wandb_logger.py:503-507`

影响：
- 这两个指标会天然恒定
- 不能用于判断 latent 是否被均匀使用

### 问题 4：8 个 latent 没有显式打破对称性

证据：
- `self.latents` 全 0 初始化：`src/models/arch/fusions/cross_scale_fusion.py:171`
- 没有额外 latent identity
- 8 个 `latent_*_norm` 曲线逐点完全一致

影响：
- latent 专业化很可能起不来
- cross-scale 模块更像 8 份重复副本，而不是 8 个有职责分化的 bottleneck

### 问题 5：`feature_var_pre/post` 命名与真实含义不一致

证据：
- 实际计算的是 off-diagonal mean cosine distance，不是 variance

影响：
- 指标解释容易误导
- 审阅时会误以为是二阶统计量

### 问题 6：`include_quantiles=True` 被传入，但 quantile 并没有真的记录

证据：
- `_quantile(...)` 定义存在：`src/train/wandb_logger.py:68-72`
- `_add_stats(... include_quantiles=True ...)` 多处传入
- 但 `_add_stats` 只写了 `mean/std/min/max`：`src/train/wandb_logger.py:288-301`

影响：
- 代码表意和实际行为不一致
- 如果本来想用 quantile 看训练分布尾部，现在完全没有这些数据

### 问题 7：绝大部分结构性 debug 指标只有 train，没有 val

现状：
- `val` 侧只记录了 `loss` / `loss_share` / `lr`
- 所有结构类信息都只在 `train` 侧记录

影响：
- 你能看到训练过程中结构如何漂移
- 但看不到这些结构现象是否也出现在验证样本上

## 5. 明确建议加入修改清单的部分

这些部分修改方向明确，可以直接进 TODO。

### 修改项 A：修正图片日志触发逻辑

建议：
- 不要用 `global_step % viz_every == 0` 去卡验证图
- 改为：
  - 每个 val epoch 固定打 1 次
  - 或按 `epoch % k == 0` 打
  - 或维护单独的 `last_viz_step`，只要距离上次超过阈值就在 val 时打

目标：
- 保证 `viz/*` 图片实际落盘

### 修改项 B：真正缓存固定验证 batch

建议：
- 在训练开始时从 val loader 固定抓一批样本缓存
- 后续所有 `log_fixed_val_snapshot(...)` 都使用这批缓存

目标：
- 让跨 epoch 图片可比

### 修改项 C：把 `feature_var_*` 改成与真实语义一致的名字

建议：
- 重命名为类似：
  - `feature_cosdist_pre`
  - `feature_cosdist_post`
  - `feature_cosdist_ratio`

目标：
- 避免把余弦距离误解成方差

### 修改项 D：把 quantile 真正打出来，或者删掉伪接口

建议：
- 如果保留 `include_quantiles=True`，就补齐 p05/p25/p50/p75/p95
- 如果不打算记录 quantile，就删掉 `_quantile` 和 `include_quantiles`

目标：
- 消除代码意图和实际行为的偏差

### 修改项 E：补一组低频 `val_debug` 标量

建议：
- 对固定 val batch 每个 epoch 或每 `k` 个 epoch 记录一次：
  - bridge gate
  - cross-scale attention
  - task repr l2
  - loss params

目标：
- 把当前只在 train 端可见的结构性漂移同步到 val 端

## 6. 暂时只写问题，不先定修改方案的部分

### 问题 P1：`cross_scale` 的 latent 专业化是否需要结构性打破对称

当前证据：
- 8 个 latent norm 完全一致
- `mezzo` attention 近乎归零

但这里先不直接定方案，因为可能有几种路线：
- 只改初始化
- 加 latent id embedding
- 改 cross-attn / self-attn 结构
- 改正则或 loss

先记录问题，不在本轮文档里强行指定唯一修改方案。

### 问题 P2：`latent_usage_entropy/gini` 应该改成什么定义

当前实现是错位的，但目标定义要先确认：
- 想看的是 latent 之间的职责分配
- 还是 source token 被关注的分配
- 还是各 scale 被关注的分配

这需要先定观测目标，再改指标。

### 问题 P3：`mezzo` 通路的持续失活是 bug 还是模型选择结果

证据链已经很强：
- `within_scale_mezzo` 收缩
- `bridge_mezzo/gate_mean` 持续下降
- `latent_attn_to_mezzo` 几乎归零

但是否需要强行“救活 mezzo”，要先结合任务设计确认。

## 7. 总结

这条 run 不是简单的“loss 正常下降”。

更准确的结论是：
- 优化稳定，train/val loss 都在改善
- 输入、wavelet、system 没有明显坏信号
- 结构性漂移非常明显，尤其是：
  - `mezzo` 通路持续失活
  - `macro` 通路越来越主导
  - task repr / encoder act 尺度持续上升
  - cross-scale 的部分指标实现本身有问题
- 非标量日志现在看不到，不是 W&B UI 的锅，而是当前调度逻辑导致图片从未被记录

如果下一步要动代码，优先级我建议是：
- 先修 `viz` 调度和固定 val batch
- 再修明显错误的 `latent_usage_*` 与命名问题
- 然后再决定是否对 `CrossScaleFusion` 的 latent 对称性下手
