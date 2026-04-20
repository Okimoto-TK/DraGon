# Diagnostic: Persistent Mezzo Branch Degeneration

## Scope

This note is a pure analysis of the current `dragon` training run behavior.
It does not prescribe code changes yet.

Observed top-line fact:

- Training loss is decreasing normally.
- The `mezzo` branch is still degenerating structurally.
- The degeneration is consistent across `within-scale`, `bridge`, and `cross-scale`.

This is not well explained by transient numerical noise.

## Executive Summary

The most likely interpretation is:

- The model is converging to a stable two-scale solution dominated by `macro` and, secondarily, `micro`.
- `mezzo` is becoming a redundant middle branch with weak routing priority.
- Once `cross-scale` stops using `mezzo`, gradient pressure on `bridge_mezzo` weakens.
- Once `bridge_mezzo` weakens, `within-scale mezzo` no longer needs to preserve feature diversity.
- The three observed symptoms therefore look like one causal chain, not three independent failures.

Most likely root-cause ranking:

1. `CrossScaleFusion` has strong symmetry and weak identity structure, so it can collapse onto the easiest token sources.
2. `ExogenousBridgeFusion` turns side-memory drift into a shrinking injection gate for `mezzo`.
3. `WithinScaleSTARFusion` diversity collapse is real, but more likely a downstream consequence and amplifier than the first trigger.

## Symptom Chain

### 1. Within-scale mezzo diversity keeps shrinking

Source:

- `src/models/arch/fusions/within_scale_star_fusion.py`

Observed:

- `train_within_scale_mezzo/feature_diversity_ratio`
- `0.838 -> 0.580 -> 0.541`

Important interpretation:

- `feature_var_pre/post` is not variance.
- The logger computes off-diagonal mean cosine distance between features.
- So the metric means fused feature tokens are becoming more similar to each other.

Implication:

- `mezzo` is losing internal feature separation.
- This is materially worse than both `macro` and `micro`.

### 2. Bridge injection into mezzo keeps weakening

Source:

- `src/models/arch/fusions/exogenous_bridge_fusion.py`

Observed:

- `train_bridge_mezzo/gate_mean`
- `0.475 -> 0.379 -> 0.327`

At the same time:

- `bridge_token_l2_mean` rises instead of collapsing.

Implication:

- The bridge path is not dead.
- Exogenous information is still forming a nontrivial bridge representation.
- But that representation is increasingly not being injected back into endogenous `mezzo` tokens.

This points to a routing or gating problem, not simple absence of signal.

### 3. Cross-scale fusion nearly stops attending to mezzo

Source:

- `src/models/arch/fusions/cross_scale_fusion.py`

Observed:

- `train_cross_scale/latent_attn_to_mezzo`
- `0.028 -> 0.003 -> 0.003`

Implication:

- Final fusion almost completely stops using `mezzo`.
- `macro` becomes dominant.
- `micro` weakens somewhat but remains meaningfully used.

This is the strongest signal in the chain.

## Why This Is Not Just Noise

Three different modules point in the same direction:

- `within-scale`: `mezzo` tokens become more homogeneous
- `bridge`: `mezzo` accepts less external injection
- `cross-scale`: `mezzo` is nearly ignored downstream

If this were mostly logging noise or short-lived instability, those three layers would not usually align this cleanly.

## Ranked Root-Cause Hypotheses

## Hypothesis 1: Cross-scale symmetry is the primary failure point

This is the highest-probability root cause.

### Structural evidence

In `CrossScaleFusion`:

- `self.latents` is initialized to all zeros.
- `macro_scale_embedding`, `mezzo_scale_embedding`, and `micro_scale_embedding` are also initialized to zeros.
- There is no latent identity embedding.

Source:

- `src/models/arch/fusions/cross_scale_fusion.py`

Consequence:

- At initialization, all latent queries are functionally symmetric.
- The module has weak built-in pressure to diversify latent roles.
- The optimizer can learn a cheap routing strategy where most useful mass goes to one or two source groups.

### Why this explains the observed pattern

If cross-scale routing discovers that:

- `macro` already carries strong coarse information
- `micro` already carries sharp local information
- `mezzo` is partially redundant with both

then the easiest stable solution is:

- mostly use `macro`
- keep some `micro`
- largely drop `mezzo`

Once that happens, `mezzo` gets less downstream gradient and starts to decay in utility.

### Why the tiny mezzo attention is meaningful

This is not explainable by source-token length alone.

Token counts are:

- `macro = 16`
- `mezzo = 24`
- `micro = 36`
- total `= 76`

If attention were merely length-proportional, `mezzo` would receive about:

- `24 / 76 ~= 0.316`

Observed is:

- `0.003`

That is not a mild imbalance.
It is an explicit routing rejection.

## Hypothesis 2: Bridge gate design amplifies `g2` drift into under-injection

This is the second most likely cause, and may interact tightly with Hypothesis 1.

### Structural evidence

In `ExogenousBridgeFusion`:

- `gate = sigmoid(global_gate(exogenous_global))`
- gate is applied multiplicatively to the fusion delta
- there is no normalization on `exogenous_global`
- there is no floor or residual path protecting minimum injection strength

Source:

- `src/models/arch/fusions/exogenous_bridge_fusion.py`

### Why this matters for mezzo specifically

`mezzo` uses `s2 / g2` from `SideMemoryHierarchy`.

In the reviewed run, side-memory norms and couplings drift downward, especially around `g1/g2`.
That means the distribution seen by `bridge_mezzo.global_gate` is moving during training.

So the bridge does not just learn content.
It also learns against changing amplitude and geometry in `g2`.

The observed pattern:

- `bridge_token_l2_mean` rises
- `gate_mean` falls
- actual fusion strength does not rise with token norm

is exactly what a shrinking multiplicative gate would look like.

Interpretation:

- useful bridge content exists
- but the gate increasingly suppresses its write-back into endogenous `mezzo`

## Hypothesis 3: Within-scale mezzo collapse is mostly downstream

This is real, but likely not the first trigger.

### Structural evidence

`WithinScaleSTARFusion` outputs:

- fused feature map `z_fused`
- `scale_seq = z_fused.mean(dim=1)`

Source:

- `src/models/arch/fusions/within_scale_star_fusion.py`

That means downstream modules only consume the feature-averaged sequence.

### Why this likely comes later

If downstream training no longer values `mezzo`, then `within-scale mezzo` no longer receives strong incentive to preserve feature-specific variation.

In that regime the optimizer can cheaply reduce feature diversity while still maintaining enough average sequence signal for whatever little remains useful.

So:

- the diversity collapse is likely genuine
- but it is more consistent with gradient starvation than with being the first independent bug

## Supporting Module-Level Reasoning

## SideMemoryHierarchy makes mezzo the most replaceable branch

Source:

- `src/models/arch/networks/side_memory_hierarchy.py`

Current construction:

- `macro <- s1 / g1`
- `mezzo <- s2 / g2`
- `micro <- s3 / g3`

`s2/g2` is built from a middle-stage query-memory interaction:

- recent 12-day queries against earlier 52-day memory

Compared with the other two:

- `macro` gets the broadest full-memory view
- `micro` gets the shortest, sharpest local view
- `mezzo` sits between them and has the weakest unique identity

This makes `mezzo` the branch most vulnerable to being treated as redundant if downstream routing is not explicitly balanced.

## The branch is not empty; it is being bypassed

This is a critical distinction.

Evidence against "mezzo has no representation":

- encoder activations do not indicate obvious branch death
- bridge token norm grows rather than collapses
- the degradation appears specifically at gate and routing stages

So the better description is:

- `mezzo` still forms representations
- but those representations are progressively not selected, injected, or consumed

## What Is Probably Not The Main Cause

## Not a logger artifact for the three main symptoms

The following observations remain meaningful:

- `within_scale_mezzo/feature_diversity_ratio`
- `bridge_mezzo/gate_mean`
- `cross_scale/latent_attn_to_mezzo`

Some other debug metrics are flawed or low-value, especially:

- `latent_usage_entropy`
- `latent_usage_gini`

because of how they are aggregated in the logger.

But that does not invalidate the `mezzo` degeneration diagnosis.

## Not a pure token-count bias

As noted earlier:

- length-proportional routing cannot explain `mezzo ~= 0.003`

The collapse is too strong.

## Not best explained by encoder failure

Nothing in the current evidence strongly suggests:

- the `mezzo` encoder alone is broken
- or the branch fails before fusion begins

The strongest evidence points later, at:

- bridge write-back
- cross-scale routing

## Most Plausible Causal Story

The current most plausible chain is:

1. `CrossScaleFusion` starts from a highly symmetric setup.
2. Training discovers a cheap solution dominated by `macro`, with some contribution from `micro`.
3. `mezzo` receives weak downstream credit assignment.
4. `bridge_mezzo` responds by shrinking effective write-back through its gate.
5. `within-scale mezzo` then has less incentive to preserve feature diversity.
6. The branch remains numerically active but functionally sidelined.

This matches all three observed symptoms without needing separate ad hoc explanations.

## Confidence Assessment

High confidence:

- `mezzo` degeneration is real
- the problem is structural rather than random noise
- the main failure sits in selection/routing, not total absence of representation

Medium-to-high confidence:

- `cross-scale` symmetry is the leading root cause
- `bridge` gating is the second major contributor

Medium confidence:

- `within-scale` diversity collapse is mostly downstream rather than primary

## Open Questions For Further Analysis

These should be verified before deciding on a code change.

1. Does `mezzo` receive systematically lower gradient norm than `macro/micro` after cross-scale begins to specialize?
2. Does `bridge_mezzo` gate shrink because `g2` norm drifts, because `g2` direction drifts, or because the learned gate weights explicitly prefer suppression?
3. If `cross-scale` attention is inspected per latent rather than averaged, do all latents ignore `mezzo`, or only a subset?
4. Is `mezzo` genuinely redundant for the task, or is it being dropped only because the architecture does not protect a unique role for it?

## Bottom Line

The best current diagnosis is:

- `mezzo` is not failing because it cannot represent information.
- `mezzo` is failing because the current architecture gives the optimizer a stable path to stop using it.

The strongest candidate first cause is the symmetric, weakly differentiated design of `CrossScaleFusion`.
The strongest secondary amplifier is the multiplicative `bridge` gate on `mezzo`.
The `within-scale` diversity collapse is likely the visible upstream consequence of that downstream abandonment.
