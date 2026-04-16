# Model Frontend Architecture

Current frontend design for the rebuilt `src/models` stack.

## Feature Layout

OHLCV input uses `f0..f8`:

```text
f0 = return
f1 = range
f2 = close_position
f3 = limit_band_position
f4 = volume_ratio          # clamp to [0, 10]
f5 = signed_amihud
f6 = limit_state_id        # 0..15
f7 = sin(time)
f8 = cos(time)
```

Sidechain currently keeps:

```text
gap
gap_rank
mf_net_ratio
mf_net_rank
mf_concentration
amount_rank
velocity_rank
amihud_impact
```

## Branch Split

```text
Path core    = {f0, f1, f2, f3}
Liquidity    = {f4, f5}
Status       = {f6}
Time         = {f7, f8}
Sidechain    = {gap, gap_rank, mf_net_ratio, mf_net_rank, mf_concentration, amount_rank, velocity_rank, amihud_impact}
```

## Design Rules

1. `f0-f3` use per-feature WNO encoders.
2. `f7-f8` do not enter the WNO operator directly.
3. `f7-f8` are injected after WNO in the path branch, and directly projected in the liquidity/state branches.
4. Path branch uses explicit pairwise crosses only for Tier 1 and Tier 2 pairs.
5. Liquidity branch keeps `R = clamp(f4, 0, 10)` as explicit magnitude.
6. `f4` also produces an RBF encoding `e4` for FiLM conditioning.
7. `f5` uses `SymLog + Fourier(num_freqs=2)`.
8. Liquidity should remain as tokenized sequence output, not a single early-pooled vector.
9. Price and liquidity should perform symmetric dual fusion and then build a `Kp x Kv` joint interaction grid.
10. `f6` is a discrete hard state anchor. It should read from the `price-liquidity joint tokens`, not directly from raw liquidity memory.
11. State and liquidity outputs should remain separate unless a later stage explicitly needs interaction.
12. Sidechain should preserve directional semantics: `moneyflow <-> liquidity -> gap`.
13. Sidechain cross-group interaction should not use symmetric self-attention as the main structure.
14. Sidechain should write into `Z_joint` through gated cross-attention and block conditioning, not through an early head override.
15. Cross-scale fusion should use a mezzo-centric route, not full three-scale all-to-all attention.
16. The cross-scale route is: `macro -> mezzo -> micro -> mezzo -> FFN(mezzo)`.
17. `micro -> mezzo` should use a stronger selection module than dense cross-attention.

## Frozen Hyperparameters

The first implementation fixes all structural sizes to keep tensor shapes static
and CUDA Graph friendly:

```text
D   = 64
heads = 4

Kp  = 4   # path tokens
Kv  = 3   # liquid tokens
Kj  = 4   # joint tokens
Ks  = 4   # side tokens

topk = 4

macro_wno_blocks = 2
mezzo_wno_blocks = 2
micro_wno_blocks = 2

macro_wno_levels = 2
mezzo_wno_levels = 2
micro_wno_levels = 2
```

## Pairwise Cross Set

Tier 1:

```text
(f0, f1)
(f0, f2)
(f1, f2)
```

Tier 2:

```text
(f0, f3)
(f1, f3)
```

Each pair keeps:

```text
prod
diff
absdiff
```

So the path token set is:

```text
4 single-feature tokens
+ 5 pairs * 3 cross-types
= 19 tokens
```

## Frontend Network Overview

```mermaid
flowchart TD
  IN["Input OHLCV f0..f8"]
  SIDEIN["Input side_daily"]

  IN --> P0["f0 return"]
  IN --> P1["f1 range"]
  IN --> P2["f2 close_position"]
  IN --> P3["f3 limit_band_position"]
  IN --> L4["f4 volume_ratio"]
  IN --> L5["f5 signed_amihud"]
  IN --> S6["f6 limit_state_id"]
  IN --> T7["f7 sin_time"]
  IN --> T8["f8 cos_time"]

  subgraph PATH["Path Core Branch"]
    P0 --> SPE0["SingleFeaturePathEncoder"]
    P1 --> SPE1["SingleFeaturePathEncoder"]
    P2 --> SPE2["SingleFeaturePathEncoder"]
    P3 --> SPE3["SingleFeaturePathEncoder"]

    T7 --> XY0["XYConditionedMixer"]
    T8 --> XY0
    T7 --> XY1["XYConditionedMixer"]
    T8 --> XY1
    T7 --> XY2["XYConditionedMixer"]
    T8 --> XY2
    T7 --> XY3["XYConditionedMixer"]
    T8 --> XY3

    SPE0 --> XY0 --> H0["h0"]
    SPE1 --> XY1 --> H1["h1"]
    SPE2 --> XY2 --> H2["h2"]
    SPE3 --> XY3 --> H3["h3"]

    H0 --> C01["PairwiseCross 01"]
    H1 --> C01
    H0 --> C02["PairwiseCross 02"]
    H2 --> C02
    H1 --> C12["PairwiseCross 12"]
    H2 --> C12
    H0 --> C03["PairwiseCross 03"]
    H3 --> C03
    H1 --> C13["PairwiseCross 13"]
    H3 --> C13

    H0 --> STACK19["Stack 19 tokens"]
    H1 --> STACK19
    H2 --> STACK19
    H3 --> STACK19
    C01 --> STACK19
    C02 --> STACK19
    C12 --> STACK19
    C03 --> STACK19
    C13 --> STACK19

    STACK19 --> ATTN19["FeatureSetAttentionBlock"]
    ATTN19 --> GPATH["GatedAttentionPooling"]
    GPATH --> REFP["SmallSwiGLURefine"]
    REFP --> ZP["Z_price"]
  end

  subgraph LIQ["Liquidity Branch"]
    L4 --> CLAMP4["r4 clamp 0 to 10"]
    CLAMP4 --> LOG4["log r4"]
    LOG4 --> RBF4["RBF Encoder"]
    CLAMP4 --> RVAL["R = r4"]

    L5 --> SYM5["SymLog"]
    SYM5 --> FOUR5["Fourier num_freqs=2"]

    RBF4 --> FILM4["FiLM Head"]
    FOUR5 --> MOD5["Modulate Fourier"]
    FILM4 --> MOD5
    MOD5 --> LN5["LayerNorm direction"]

    RVAL --> LIQCAT["Concat R direction X Y"]
    LN5 --> LIQCAT
    T7 --> LIQCAT
    T8 --> LIQCAT

    LIQCAT --> LPROJ["Linear SiLU Linear"]
    LPROJ --> TOKL["LiquidityTokenPooling"]
    TOKL --> ZL["Z_liquid"]
  end

  subgraph PV["Price Liquidity Dual Fusion"]
    ZP --> PSELF["Price Token SelfAttention"]
    ZL --> VSELF["Liquidity Token SelfAttention"]

    PSELF --> PCROSS["CrossAttention P queries V"]
    VSELF --> VCROSS["CrossAttention V queries P"]

    VSELF --> PCROSS
    PSELF --> VCROSS

    PCROSS --> PDUAL["Z_price_dual"]
    VCROSS --> VDUAL["Z_liquid_dual"]

    PDUAL --> GRID["PairInteractionGrid Kp x Kv"]
    VDUAL --> GRID
    GRID --> JREAD["JointQueryReadout"]
    JREAD --> ZJ["Z_joint"]
  end

  subgraph SIDE["Shared Side Daily Context"]
    SIDEIN --> SIDEENC["SidechainContextEncoder"]
    SIDEENC --> ED["E_d"]
  end

  subgraph STATE["State Query Branch"]
    S6 --> EMB6["Embedding 16 to D"]
    T7 --> TPROJ["Time Projection"]
    T8 --> TPROJ
    EMB6 --> ADD6["Add and LayerNorm"]
    TPROJ --> ADD6
    ADD6 --> CONV6["DWConv PWConv SiLU Residual"]
    CONV6 --> S6CTX["s6_ctx"]
  end

  subgraph SIDEWRITE["Side Writes Into Joint"]
    ZJ --> JWRITE["Gated Side Write"]
    ED --> JWRITE
    S6CTX --> JWRITE
    JWRITE --> ZJCTX["Z_joint_ctx"]
  end

  subgraph READ["State Reads Joint Tokens"]
    S6CTX --> READJOINT["CrossAttention Q=s6_ctx"]
    ZJCTX --> READJOINT
    READJOINT --> ZSTATE["Z_state"]
  end

  ZP --> OUT["Front Outputs"]
  ZL --> OUT
  ZJCTX --> OUT
  ZSTATE --> OUT
  ED --> OUT
```

## Sidechain Causal Semantics

```text
moneyflow <-> liquidity -> gap
```

Interpretation:

- `moneyflow` and `liquidity` are mutually verifying causes.
- `gap` is a downstream result / manifestation.
- `E_d` should preserve this direction, not flatten the three groups into one symmetric context token.

## Sidechain Group Split

```text
GapState         = {gap, gap_rank}
MoneyFlow        = {mf_net_ratio, mf_net_rank, mf_concentration}
LiquidityRegime  = {amount_rank, velocity_rank, amihud_impact}
```

## Sidechain Output Rule

Do not collapse sidechain into a single token too early.

Instead:

```text
E_d = stack([z_mf1, z_liqreg1, z_cause, z_gap_ctx], dim=token_axis)
```

with:

- `z_mf1`: moneyflow token after reading liquidity
- `z_liqreg1`: liquidity-regime token after reading moneyflow
- `z_cause`: fused latent from `moneyflow <-> liquidity`
- `z_gap_ctx`: gap token after reading `z_cause`

So the sidechain output is:

```text
E_d: [B, Td, 4, D]
```

Recommended use:

```text
E = E_d
delta_J = CrossAttn(Q=Z_joint, K,V=E)
alpha_evt = sigmoid(MLP([Pool(s6_ctx), Pool(E), Pool(Z_joint)]))
Z_joint = Z_joint + alpha_evt * delta_J
```

Optional block conditioning:

```text
[gamma, beta, g_attn, g_ffn] = SideStateMLP([Pool(E), Pool(s6_ctx)])
Z_joint = Z_joint + g_attn * SelfAttn(AdaLN(Z_joint, gamma, beta))
Z_joint = Z_joint + g_ffn  * FFN(AdaLN(Z_joint, gamma, beta))
```

Do not use sidechain as a separate early override head in the first version.

## Module Graphs

### SingleFeaturePathEncoder

```mermaid
flowchart TD
  A["f_i B,T,1"] --> B["Lift 1 to D"]
  B --> C["LayerNorm"]
  C --> D1["FeatureWNOBlock"]
  D1 --> D2["FeatureWNOBlock"]
  D2 --> E["LocalRefineConv k=3 or 5"]
  E --> F["h_i B,T,D"]
```

### FeatureWNOBlock

```mermaid
flowchart TD
  A["h B,T,D"] --> B["PreNorm"]
  B --> C["WaveletDecompose"]
  C --> A1["Approx"]
  C --> D1["Detail 1"]
  C --> D2["Detail 2"]

  A1 --> A2["DWConv k3"]
  A2 --> A3["PWConv"]
  A3 --> A4["SiLU"]

  D1 --> B2["DWConv k3"]
  B2 --> B3["PWConv"]
  B3 --> B4["SiLU"]

  D2 --> C2["DWConv k3"]
  C2 --> C3["PWConv"]
  C3 --> C4["SiLU"]

  A4 --> R["InverseWaveletReconstruct"]
  B4 --> R
  C4 --> R
  R --> RES["ResidualAdd"]
  RES --> N["OutputNorm"]
  N --> O["h_out B,T,D"]
```

### XYConditionedMixer

```mermaid
flowchart TD
  A["h_i B,T,D"] --> C["Concat"]
  X["f7 sin_time"] --> C
  Y["f8 cos_time"] --> C
  C --> D["Conv1d k=3 or 5"]
  D --> E["SiLU"]
  E --> F["Conv1d 1x1"]
  F --> G["ResidualAdd"]
  G --> H["LayerNorm"]
  H --> O["h_i_tilde B,T,D"]
```

### PairwiseCross

```mermaid
flowchart TD
  A["h_i B,T,D"] --> P["prod"]
  B["h_j B,T,D"] --> P

  A --> D["diff"]
  B --> D

  A --> AD["absdiff"]
  B --> AD

  P --> O1["prod_ij B,T,D"]
  D --> O2["diff_ij B,T,D"]
  AD --> O3["absdiff_ij B,T,D"]
```

### FeatureSetAttentionBlock

```mermaid
flowchart TD
  A["G B,T,19,D"] --> B["reshape to B*T,19,D"]
  B --> C["SelfAttention"]
  C --> D["ResidualAdd"]
  D --> E["LayerNorm"]
  E --> F["FFN Linear SwiGLU Linear"]
  F --> G["ResidualAdd"]
  G --> H["LayerNorm"]
  H --> I["reshape back"]
  I --> O["G_attn B,T,19,D"]
```

### GatedAttentionPooling

```mermaid
flowchart TD
  A["G_attn B,T,19,D"] --> B["Score Network"]
  B --> C["Token Scores"]
  A --> D["Weighted Pooling"]
  C --> D
  D --> O["G_pool B,T,Kp,D"]
```

### SmallSwiGLURefine

```mermaid
flowchart TD
  A["G_pool B,T,Kp,D"] --> B["Linear"]
  B --> C["SwiGLU"]
  C --> D["Linear"]
  D --> E["ResidualAdd"]
  E --> F["LayerNorm"]
  F --> O["Z_price B,T,Kp,D"]
```

### Liquidity Branch

```mermaid
flowchart TD
  A["f4 volume_ratio"] --> B["r4 clamp 0 to 10"]
  B --> C["log r4"]
  C --> D["RBF Encoder"]
  B --> R["R = r4"]

  E["f5 signed_amihud"] --> F["SymLog"]
  F --> G["Fourier num_freqs=2"]

  D --> H["FiLM Head"]
  G --> I["Modulate Fourier"]
  H --> I
  I --> J["LayerNorm direction"]

  R --> K["Concat R direction X Y"]
  J --> K
  X["f7 sin_time"] --> K
  Y["f8 cos_time"] --> K

  K --> L["Linear"]
  L --> M["SiLU"]
  M --> N["Linear"]
  N --> P["LiquidityTokenPooling"]
  P --> O["Z_liquid B,T,Kv,D"]
```

### LiquidityTokenPooling

`Z_liquid_base_seq` below is the pre-pooling sequence produced by the preceding
`Linear -> SiLU -> Linear` block in the liquidity branch.

```mermaid
flowchart TD
  A["Z_liquid_base_seq B,T,D"] --> B["Token Query Pooling"]
  B --> O["Z_liquid B,T,Kv,D"]
```

### StateQueryEncoder

```mermaid
flowchart TD
  A["f6 limit_state_id"] --> B["Embedding 16 to D"]
  X["f7 sin_time"] --> C["Time Projection"]
  Y["f8 cos_time"] --> C
  B --> D["Add and LayerNorm"]
  C --> D
  D --> E["DWConv"]
  E --> F["PWConv"]
  F --> G["SiLU"]
  G --> H["ResidualAdd"]
  H --> I["LayerNorm"]
  I --> O["s6_ctx B,T,1,D"]
```

### PriceLiquidityDualFusion

```mermaid
flowchart TD
  A["Z_price B,T,Kp,D"] --> B["Price Token SelfAttention"]
  C["Z_liquid B,T,Kv,D"] --> D["Liquidity Token SelfAttention"]

  B --> E["CrossAttention P queries V"]
  D --> E

  D --> F["CrossAttention V queries P"]
  B --> F

  E --> G["Z_price_dual B,T,Kp,D"]
  F --> H["Z_liquid_dual B,T,Kv,D"]

  G --> I["PairInteractionGrid"]
  H --> I
  I --> J["JointQueryReadout"]
  J --> O["Z_joint B,T,Kj,D"]
```

### StateQueryJointReader

```mermaid
flowchart TD
  A["s6_ctx B,T,1,D"] --> Q["Q"]
  B["Z_joint B,T,Kj,D"] --> KV["K V"]
  Q --> C["ScaleAware CrossAttention"]
  KV --> C
  C --> O["Z_state B,T,1,D"]
```

### ScaleAware Window Policy

```mermaid
flowchart TD
  A["StateQueryJointReader"] --> M1["micro full T x T"]
  A --> M2["mezzo window plus minus 8"]
  A --> M3["macro window plus minus 4"]
```

### SidechainContextEncoder

```mermaid
flowchart TD
  G0["gap gap_rank"] --> GE["GapEncoder"]
  GE --> ZG0["z_gap0"]

  M0["mf_net_ratio mf_net_rank mf_concentration"] --> ME["MoneyFlowEncoder"]
  ME --> ZM0["z_mf"]

  L0["amount_rank velocity_rank amihud_impact"] --> LE["LiquidityRegimeEncoder"]
  LE --> ZL0["z_liqreg"]

  ZM0 --> CA1["CrossAttention Q=z_mf"]
  ZL0 --> CA1
  CA1 --> DMF["delta_mf"]

  ZL0 --> CA2["CrossAttention Q=z_liqreg"]
  ZM0 --> CA2
  CA2 --> DLIQ["delta_liq"]

  ZM0 --> GMF["GateHead"]
  DMF --> GMF
  GMF --> ZM1["z_mf1"]

  ZL0 --> GLIQ["GateHead"]
  DLIQ --> GLIQ
  GLIQ --> ZL1["z_liqreg1"]

  ZM1 --> CFUSE["CausalFuse prod absdiff SwiGLU"]
  ZL1 --> CFUSE
  CFUSE --> ZCAUSE["z_cause"]

  ZG0 --> CA3["CrossAttention Q=z_gap0"]
  ZCAUSE --> CA3
  CA3 --> DGAP["delta_gap"]

  ZG0 --> GGAP["GateHead"]
  DGAP --> GGAP
  GGAP --> ZG1["z_gap_ctx"]

  ZM1 --> ZSIDE["Stack token axis"]
  ZL1 --> ZSIDE
  ZCAUSE --> ZSIDE
  ZG1 --> ZSIDE
  ZSIDE --> ZS["E_d B,T,4,D"]
```

### GapEncoder

```mermaid
flowchart TD
  A["gap gap_rank"] --> B["Linear"]
  B --> C["DWConv k=3 or 5"]
  C --> D["SiLU"]
  D --> E["LayerNorm"]
  E --> O["z_gap0 B,T,D"]
```

### MoneyFlowEncoder

```mermaid
flowchart TD
  A["mf_net_ratio mf_net_rank mf_concentration"] --> B["Linear"]
  B --> C["DWConv k=3 or 5"]
  C --> D["SiLU"]
  D --> E["LayerNorm"]
  E --> O["z_mf B,T,D"]
```

### LiquidityRegimeEncoder

```mermaid
flowchart TD
  A["amount_rank velocity_rank amihud_impact"] --> B["Linear"]
  B --> C["DWConv k=3 or 5"]
  C --> D["SiLU"]
  D --> E["LayerNorm"]
  E --> O["z_liqreg B,T,D"]
```

### Sidechain CausalFuse

```mermaid
flowchart TD
  A["z_mf1 B,T,D"] --> P["Projected prod"]
  B["z_liqreg1 B,T,D"] --> P

  A --> D["Projected absdiff"]
  B --> D

  P --> C["Concat"]
  D --> C
  C --> E["Linear"]
  E --> F["SwiGLU"]
  F --> G["Linear"]
  G --> O["z_cause B,T,D"]
```

## Cross-Scale Fusion

Each scale first runs the full single-scale frontend independently, producing
its own `J_*` and `Z_state_*`. Only after that does the model enter the
cross-scale fusion stage below.

The cross-scale route is intentionally asymmetric:

```text
macro -> mezzo
mezzo -> micro
micro -> mezzo
FFN on mezzo
```

Interpretation:

- `macro -> mezzo`: macro provides regime / stage anchor to mezzo.
- `mezzo -> micro`: mezzo tells micro what local detail is worth reading.
- `micro -> mezzo`: micro writes decisive fine-grained evidence back to mezzo.
- final fusion and readout stay on mezzo.

This avoids:

```text
macro <-> micro direct attention
three-scale all-to-all attention
full dense micro -> mezzo writeback
```

### Cross-Scale Inputs

```text
J_macro : [B, Ta, Ka, D]
J_mezzo : [B, Tm, Km, D]
J_micro : [B, Ti, Ki, D]

Z_state_macro : [B, Ta, 1, D]
Z_state_mezzo : [B, Tm, 1, D]
Z_state_micro : [B, Ti, 1, D]

E_d : [B, Td, 4, D]
```

### Cross-Scale Flow

```text
Jm0 = J_mezzo
Ji0 = J_micro
Ja0 = J_macro
```

```text
cond_A2M = MLP([Pool(E_d), Pool(Z_state_macro), Pool(Z_state_mezzo)])
Jm1 = Jm0 + g_A2M ⊙ CA(q=AdaLN(Jm0, cond_A2M), kv=Ja0)
```

```text
cond_M2I = MLP([Pool(E_d), Pool(Z_state_mezzo), Pool(Z_state_micro)])
Ji1 = Ji0 + g_M2I ⊙ CA(q=AdaLN(Ji0, cond_M2I), kv=Jm1)
```

```text
J_micro_sig = LocalTopKCrossReadout(
  q      = Jm1,
  kv     = Ji1,
  cond   = [E_d, Z_state_mezzo, Z_state_micro],
  window = local aligned micro window,
  k      = top-k
)
```

```text
cond_I2M = MLP([Pool(E_d), Pool(Z_state_micro), Pool(Z_state_mezzo)])
Jm2 = Jm1 + g_I2M ⊙ CA(q=AdaLN(Jm1, cond_I2M), kv=J_micro_sig)
```

```text
cond_FFN = MLP([Pool(E_d), Pool(Z_state_mezzo)])
J_fused = Jm2 + g_FFN ⊙ FFN(AdaLN(Jm2, cond_FFN))
```

### MacroToMezzoAdapter

```mermaid
flowchart TD
  A["J_macro B,Ta,Ka,D"] --> B["kv"]
  C["J_mezzo B,Tm,Km,D"] --> D["AdaLN q"]
  E["Pool E_d + S_macro + S_mezzo"] --> F["Condition MLP"]
  F --> D
  D --> G["CrossAttention mezzo queries macro"]
  B --> G
  G --> H["gated residual"]
  H --> O["Jm1 B,Tm,Km,D"]
```

### MezzoToMicroAdapter

```mermaid
flowchart TD
  A["Jm1 B,Tm,Km,D"] --> B["kv"]
  C["J_micro B,Ti,Ki,D"] --> D["AdaLN q"]
  E["Pool E_d + S_mezzo + S_micro"] --> F["Condition MLP"]
  F --> D
  D --> G["CrossAttention micro queries mezzo"]
  B --> G
  G --> H["gated residual"]
  H --> O["Ji1 B,Ti,Ki,D"]
```

### LocalTopKCrossReadout

```mermaid
flowchart TD
  A["Jm1 B,Tm,Km,D"] --> Q["mezzo queries"]
  B["Ji1 B,Ti,Ki,D"] --> W["local aligned window"]
  C["Pool E_d + S_mezzo + S_micro"] --> S["score MLP"]
  Q --> S
  W --> S
  S --> T["Top-k select"]
  Q --> CA["CrossAttention on selected micro tokens"]
  T --> CA
  CA --> O["J_micro_sig B,Tm,Km,D"]
```

### MicroToMezzoAdapter

```mermaid
flowchart TD
  A["J_micro_sig B,Tm,Km,D"] --> B["kv"]
  C["Jm1 B,Tm,Km,D"] --> D["AdaLN q"]
  E["Pool E_d + S_micro + S_mezzo"] --> F["Condition MLP"]
  F --> D
  D --> G["CrossAttention mezzo queries decisive micro"]
  B --> G
  G --> H["gated residual"]
  H --> O["Jm2 B,Tm,Km,D"]
```

### MezzoFusionFFN

```mermaid
flowchart TD
  A["Jm2 B,Tm,Km,D"] --> B["AdaLN"]
  C["Pool E_d + S_mezzo"] --> D["Condition MLP"]
  D --> B
  B --> E["FFN"]
  E --> F["gated residual"]
  F --> O["J_fused B,Tm,Km,D"]
```

## Prediction Heads

The first version should expose three prediction families:

```text
ret
one configured p_q per run
rv
```

Training is **single-target, single-task**:

- one run trains `ret`
- or one run trains `rv`
- or one run trains one chosen quantile `p_q`

Head input:

```text
H = Pool(J_fused)
```

Shared head stem:

```text
H
-> Linear
-> SwiGLU
-> Linear
-> H_head
```

### ret Head

```text
[H_head]
-> Linear
-> ret_mu

[H_head]
-> Linear
-> ret_log_sigma2

ret_sigma2 = exp(ret_log_sigma2)
```

`ret_mu` is the mean in **log-return-ratio space**:

```text
z_ret = log(label_ret)
```

### rv Head

```text
[H_head]
-> Linear
-> rv_log_var

[H_head]
-> Linear
-> rv_log_sigma2

rv_var_hat = exp(rv_log_var)
rv_hat = exp(0.5 * rv_log_var)
rv_sigma2 = exp(rv_log_sigma2)
```

`rv_log_var` is the forecasted **log realized variance**, not the auxiliary uncertainty.

### Quantile Head

For one configured quantile `q` in each run:

```text
[H_head]
-> Linear
-> p_mu_raw[q]

[H_head]
-> Linear
-> p_log_b[q]

p_mu[q] = softplus(p_mu_raw[q]) + eps
p_b[q]  = exp(p_log_b[q])
```

Recommended first enabled set:

```text
Q = {0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99}
```

The dense `p01..p99` grid can be enabled later without changing the head family, but each run still activates only one `p_q`.

### Head Graph

```mermaid
flowchart TD
  A["J_fused B,Tm,Km,D"] --> B["Pool"]
  B --> C["Shared Head Stem Linear SwiGLU Linear"]

  C --> D1["ret_mu"]
  C --> D2["ret_log_sigma2"]

  C --> E1["rv_log_var"]
  C --> E2["rv_log_sigma2"]

  C --> F1["p_mu_raw[q]"]
  C --> F2["p_log_b[q]"]
```

## Losses

All tasks follow the same pattern:

```text
L_total = L_main + lambda * L_NLL
```

The auxiliary `NLL` term must use a **detached** mean / location target so the model cannot reduce the total loss by only inflating scale:

```text
mu_for_nll = stop_gradient(mu)
```

### ret Loss

For `ret`, predict:

```text
mu       = ret_mu
s        = ret_log_sigma2
sigma2   = exp(s)
target y = label_ret
z        = log(y)
```

Use correlation in log space as the main objective:

$$
L^{ret}_{main} = 1 - \mathrm{Corr}(\mu, z)
$$

Use log-Gaussian NLL as the auxiliary term:

$$
\mathrm{NLL}^{ret}
=
\frac{1}{2}s
+
\frac{1}{2}e^{-s}\bigl(z-\mathrm{stopgrad}(\mu)\bigr)^2
$$

Total loss:

$$
L^{ret}_{total}
=
L^{ret}_{main}
+
\lambda_{ret}\,\mathrm{NLL}^{ret}
$$

### rv Loss

For `rv`, predict:

```text
log_var_hat = rv_log_var
var_hat     = exp(log_var_hat)
s           = rv_log_sigma2
sigma2      = exp(s)
target y = label_rv
v        = y^2
z_v      = log(v)
```

Use QLIKE on the forecasted realized variance as the main objective:

$$
L^{rv}_{main}
=
\log(\hat v) + \frac{v}{\hat v}
=
\text{log\_var\_hat} + v\,e^{-\text{log\_var\_hat}}
$$

Important:

- `QLIKE` uses the forecasted variance level `var_hat`
- it does **not** use the auxiliary uncertainty `sigma2`

Use Gaussian NLL on log realized variance as the auxiliary term:

$$
\mathrm{NLL}^{rv}
=
\frac{1}{2}s
+
\frac{1}{2}e^{-s}\bigl(z_v-\mathrm{stopgrad}(\text{log\_var\_hat})\bigr)^2
$$

Total loss:

$$
L^{rv}_{total}
=
L^{rv}_{main}
+
\lambda_{rv}\,\mathrm{NLL}^{rv}
$$

### Quantile Loss

For one chosen quantile `q`, predict:

```text
mu_q
log_b_q
b_q
target y = label_ret
z      = log(y)
```

Define the log residual:

$$
u_q = z - \log(\hat{\mu}_q)
$$

Main loss:

$$
\rho_q(u_q)=\max(q\,u_q,\ (q-1)\,u_q)
$$

$$
L^{q}_{main}=\rho_q(u_q)
$$

Auxiliary Log-ALD term:

$$
\mathrm{NLL}^{q}
=
\log(b_q)
+
\frac{\rho_q\!\left(z-\log(\mathrm{stopgrad}(\hat{\mu}_q))\right)}{b_q}
$$

Total loss:

$$
L^{q}_{total}
=
L^{q}_{main}
+
\lambda_{q}\,\mathrm{NLL}^{q}
$$

There is no multi-task weighted sum in the first version. One run trains exactly one target.

## Config Cleanup Notes

After the training stack is migrated to single-target `ret / rv / p_q`, the old task-specific config knobs should be removed:

```text
persist_theta
persist_tau
student_t_nu
uncertainty_floor
uncertainty_loss_weight
persist_probability_loss_weight
edge_huber_beta
persist_logit_huber_beta
downrisk_log_huber_beta
freeze_scale_s0_S
freeze_scale_s0_M
freeze_scale_s0_MDD
freeze_scale_s0_RV
```

Recommended replacement group:

```text
ret_nll_weight
rv_nll_weight
quantile_nll_weight
quantile_set
variance_eps
variance_log_clamp_min
variance_log_clamp_max
active_target
active_quantile
```

## Open Points

Still intentionally unresolved:

1. `Kp` in path pooling.
2. `Kv` in liquidity pooling.
3. `Kj` in joint token readout.
4. `Ks` in shared side latent.
5. `k` in `LocalTopKCrossReadout`.
6. Exact local window definition for `micro -> mezzo` readout.
7. Whether `f5` Fourier frequencies should remain fixed or become learnable.
8. Whether `f5` needs any later tail compression beyond `SymLog`.
9. Exact quantile set to enable in the first training run.
