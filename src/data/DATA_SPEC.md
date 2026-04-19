# Data 规范（Processed / Assembled）

本文档描述 `src/data` 当前实现下的“实际规范”，以代码行为为准，重点覆盖 `processed` 与 `assembled` 两层。

## 1. 目录与文件组织

- 根目录：`data/`
- Processed 根目录：`data/processed/`
- Assembled 根目录：`data/assembled/`

### 1.1 Processed 落盘规则

- `index`：单文件 `data/processed/index.parquet`
- 其余 processed 表：按 `code` 分区写入，每个股票 1 个文件
  - `data/processed/mask/<code>.parquet`
  - `data/processed/macro/<code>.parquet`
  - `data/processed/mezzo/<code>.parquet`
  - `data/processed/micro/<code>.parquet`
  - `data/processed/sidechain/<code>.parquet`
  - `data/processed/label/<code>.parquet`

### 1.2 Assembled 落盘规则

- 每个 `code` 输出一个或多个 `.npz`：
  - 单文件：`data/assembled/<code>.npz`
  - 分片：`data/assembled/<code>__000.npz`, `__001.npz`, ...
- 分片数量受 `packed_min_files_per_code` 控制（当前配置为 `32`），并结合样本量自适应。
- 每个 `.npz` 都包含同一套键（见第 4 节）。

## 2. 通用字段约定（Processed）

- `code`
  - 类型：`String`
  - 格式：`^\d{6}\.[A-Z]{2}$`（例如 `000001.SZ`）
- `trade_date`
  - 类型：`Date`
  - 格式说明：`%Y%m%d`（schema 描述格式）
- 主键
  - 每张表必须满足 schema 中定义的主键唯一性。
- 额外字段
  - 当前默认允许额外列存在，但会在非 debug 模式给出 warning；debug 模式下会直接报错。

## 3. Processed 表规范

## 3.1 processed_index（`index.parquet`）

- 主键：`(code, trade_date)`
- 字段：
  - `code: String`
  - `trade_date: Date`
  - `logic_index: Int32`

说明：`logic_index` 是按交易有效日连续编号，用于“跨停牌拼接”的逻辑时间轴。

## 3.2 processed_mask（`mask/<code>.parquet`）

- 主键：`(code, trade_date)`
- 字段：
  - `code: String`
  - `trade_date: Date`
  - `filter_mask: Boolean`

说明：`filter_mask` 同时考虑未来标签窗口停牌/ST风险与特征回看窗口 ST 风险。

## 3.3 processed_macro（`macro/<code>.parquet`）

- 主键：`(code, trade_date)`
- 字段：
  - `code: String`
  - `trade_date: Date`
  - `mcr_f0..mcr_f10`
- dtype 规则：
  - `mcr_f6`, `mcr_f7`: `Int8`
  - 其余 `mcr_f*`: `Float64`

## 3.4 processed_mezzo（`mezzo/<code>.parquet`）

- 主键：`(code, trade_date, time_index)`
- 字段：
  - `code: String`
  - `trade_date: Date`
  - `time_index: Int32`（30 分钟 bar 序号）
  - `mzo_f0..mzo_f10`
- dtype 规则：
  - `mzo_f6`, `mzo_f7`: `Int8`
  - 其余 `mzo_f*`: `Float64`

## 3.5 processed_micro（`micro/<code>.parquet`）

- 主键：`(code, trade_date, time_index)`
- 字段：
  - `code: String`
  - `trade_date: Date`
  - `time_index: Int32`（5 分钟 bar 序号）
  - `mic_f0..mic_f10`
- dtype 规则：
  - `mic_f6`, `mic_f7`: `Int8`
  - 其余 `mic_f*`: `Float64`

## 3.6 processed_sidechain（`sidechain/<code>.parquet`）

- 主键：`(code, trade_date)`
- 字段（均 `Float64`）：
  - `gap`
  - `gap_rank`
  - `mf_net_ratio`
  - `mf_net_rank`
  - `mf_concentration`
  - `mf_concentration_diff`
  - `mf_concentration_rank`
  - `mf_main_amount_log`
  - `mf_main_amount_log_diff`
  - `mf_main_amount_log_rank`
  - `amount_rank`
  - `velocity_rank`
  - `amihud_rank`

说明：

- `mf_concentration` 是比例项：`(buy_lg+buy_elg+sell_lg+sell_elg) / amount`
- `mf_main_amount_log` 是大资金总额 raw log：  
  `log(buy_lg+buy_elg+sell_lg+sell_elg)`
- `mf_main_amount_log_diff` 为 `mf_main_amount_log` 的分数阶差分（窗口=`WARMUP_BARS`）
- `mf_main_amount_log_rank` 为 `mf_main_amount_log` 的截面 `NormalRank`

## 3.7 processed_label（`label/<code>.parquet`）

- 主键：`(code, trade_date)`
- 字段：
  - `label_ret: Float64`
  - `label_rv: Float64`

说明：

- `label_ret`: `mean(open_{t+2}..open_{t+H}) / open_{t+1} - 1`
  - 其中 `H = LABEL_WINDOW`
- `label_rv`: 未来 `H` 日 Parkinson 波动率
  - `sqrt(mean(log(high/low)^2) / (4 * ln(2)))`
  - 其中 `H = LABEL_WINDOW`

## 3.8 Backbone 特征语义（f0-f10）

`macro/mezzo/micro` 共用同一语义框架，只是时间尺度不同：

- `f0`: 收益项
  - macro: `log(close_t) - log(close_{t-1})`
  - intraday: `log(close_t) - log(open_t)`
- `f1`: `log(high/low)`
- `f2`: 收盘在高低区间中的相对位置（裁剪到 `[0,1]`）
- `f3`: 收盘在涨跌停区间中的相对位置（裁剪到 `[0,1]`）
- `f4`: 成交额相对近 5 期均值（裁剪到 `[0,10]`）
- `f5`: Amihud 风格项（收益 / 金额）
- `f6` (`Int8`): 涨跌停 bitmask  
  `hit_up*8 + hit_down*4 + close_up*2 + close_down`
- `f7` (`Int8`): 步内位置  
  - macro: 周内交易日（1..5）
  - mezzo: 日内 30 分钟序号（1..8）
  - micro: 日内 5 分钟序号（1..48）
- `f8`: 收益率分数差分（分数阶，窗口截断）
- `f9`: 量比对数差分（分数阶，窗口截断）
- `f10`: 收益率与量比分数差分之差  
  `f8 - f9`

其中：

- `diff_d`：分数差分阶数，取值范围 `(0,1)`，当前默认 `0.5`
- `f8` 输入序列（log 收益）：
  - macro：`log(close_t) - log(close_{t-1})`（`c-c`）
  - intraday：`log(close_t) - log(open_t)`（`c-o`）
- `f9` 输入序列（log 量比）：
  - macro：`log(amount_t) - log(amount_{t-1})`
  - intraday：`log(amount_t) - log(amount_{prev_day_same_slot})`
- 分数阶差分公式（按每条序列独立计算）：
  - `omega_0 = 1`
  - `omega_k = -omega_{k-1} * (diff_d - k + 1) / k`
  - `X_tilde_t = sum_{k=0}^{K-1} omega_k * X_{t-k}`
  - `K = WARMUP_BARS`（滑动窗口长度，截断到最近 `K` 项）

## 4. Assembled（`.npz`）规范

## 4.1 输入来源

按 `code + trade_date` 对齐并拼接：

- `mask`（含 `filter_mask`）
- `label`
- `macro`
- `sidechain`
- `mezzo`（按 `time_index` 聚合为日内 list，长度应为 `8`）
- `micro`（按 `time_index` 聚合为日内 list，长度应为 `48`）

## 4.2 有效性规则

单日 `is_valid_step` 由以下条件与运算得到：

- `filter_mask == True`
- 标量 float 字段（label/macro_float/sidechain）非空且 finite
- macro 的 int8 字段非空
- mezzo/micro 的 float list 长度分别为 `8/48`，且元素非空且 finite
- mezzo/micro 的 int8 list 非空

样本级有效性 `keep`：在窗口内（macro/mezzo/micro 各自窗口）所有 `is_valid_step` 都为真。

## 4.3 窗口参数（当前配置）

- `MACRO_LOOKBACK=64`, `MEZZO_LOOKBACK=96`, `MICRO_LOOKBACK=144`
- `WARMUP_BARS=48`
- `MEZZO_BARS_PER_DAY=8`, `MICRO_BARS_PER_DAY=48`
- 推导：
  - `MACRO_WINDOW_DAYS=96`
  - `MEZZO_WINDOW_DAYS=16`，总 bar=`128`
  - `MICRO_WINDOW_DAYS=4`，总 bar=`192`

## 4.4 `.npz` 键、dtype、shape

设有效样本数为 `N`：

- `label_schema_version`: `int32`（标量数组）
- `label_names`: `str[2]`，当前为 `["label_ret", "label_rv"]`
- `date`: `float32`, shape=`[N]`，格式为 `YYYYMMDD` 数值
- `label`: `float32`, shape=`[N, 2]`
- `macro`: `float32`, shape=`[N, 9, 112]`
- `sidechain`: `float32`, shape=`[N, 13, 112]`
- `mezzo`: `float32`, shape=`[N, 9, 144]`
- `micro`: `float32`, shape=`[N, 9, 192]`
- `macro_i8`: `int8`, shape=`[N, 2, 112]`
- `mezzo_i8`: `int8`, shape=`[N, 2, 144]`
- `micro_i8`: `int8`, shape=`[N, 2, 192]`

其中：

- float 特征维（9）对应 `f0..f5, f8..f10`
- int8 特征维（2）对应 `f6, f7`

## 5. 处理顺序建议

推荐按以下顺序准备数据（避免依赖缺失）：

1. `processed.index`
2. `processed.mask`
3. `processed.macro`
4. `processed.mezzo`
5. `processed.micro`
6. `processed.sidechain`
7. `processed.label`
8. `assembled`

CLI 示例：

```bash
python -m src.cli prepare processed index,mask,macro,mezzo,micro,sidechain,label
python -m src.cli prepare assembled
```

## 6. 兼容性约束

- 调整 `LABEL_COLS` 时，必须同步更新：
  - 组装键 `label_names`
  - 消费侧读取逻辑
  - `label_schema_version`（建议递增）
- 调整 `f0..f10` 的含义或顺序时，必须同步更新：
  - processed schema 说明
  - assembler 的 float/int8 列拆分逻辑
  - 下游模型输入映射
- 调整 lookback/warmup 时，必须同步检查所有输出 shape。
