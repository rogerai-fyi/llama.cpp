# llama-moe-route-stats

Measure which experts a mixture-of-experts model actually routes to, and which it
*can* route to, from the GGUF you already serve.

## Why

Expert pruning tools score experts by how much routing traffic they carry on a
calibration set. That is **profiling** — it tells you what happened, not what can
happen. The question that licenses removing an expert is **reachability**: can
this expert be selected at all, for any input in the workload?

The two answers can differ, but no cross-architecture magnitude is currently
claimed. Results produced before the 2026-07-30 stride and score-election fixes
are withdrawn and must not be cited.

## What it does

One streaming pass over your prompts, observing tensors llama.cpp already names
(`ffn_moe_topk` and architecture-dependent score nodes). **No core patch.** It records:

- per-layer selection histograms (the usual frequency view)
- router-logit mean and covariance
- the **exact support function of the convex hull** of observed logits, as a
  running max of pairwise differences
- per-neuron gate maxima inside each expert

Token-hash routed layers (for example, DeepSeek-V4's early
`ffn_gate_tid2eid` layers) are labeled separately. Their actual selections
contribute to traffic histograms, but they are excluded from score-based
reachability because no learned selection score exists for those layers.

The collector reads non-contiguous tensors through their recorded strides. It
elects the score tensor that reproduces the model's actual top-k, verifies the
choice on subsequent tokens, serializes the agreement result, and exits nonzero
if no comparison occurred or any token-selection differs.

## Use

```
llama-moe-route-stats -m model.gguf -f prompts.txt -ngl 0 -c 1024
MOE_STATS_OUT=stats.json llama-moe-route-stats -m model.gguf -f prompts.txt
MOE_STATS_COLLECT_GATE=1 MOE_STATS_OUT=stats.json llama-moe-route-stats -m model.gguf -f prompts.txt
```

One prompt per line. It prints a concentration table and, with `MOE_STATS_OUT`
set, writes the full statistics for downstream analysis. Internal expert-gate
activation capture is intentionally opt-in because it can dominate runtime and
GPU-to-host traffic on large MoE models; route and router-score collection do
not require it.

The companion analysis (hull reachability test, ellipsoid test at chosen radii,
and per-expert routing margins) is not part of llama.cpp; see the project this
was developed in.

## Interpreting the output

`experts needed to cover 50/80/90% of routes` is reported against the uniform
baseline. Do not infer that a general workload must route uniformly. That was a
withdrawn conclusion produced by the former strided-read defect. Compare frozen,
sample-matched corpora and retain the raw self-check evidence.
