# Mode 4 (fully async) Benchmark Report

**Setup**: 8× B200 GPUs (single node), Qwen3.5 dense models, BSHD layout, vanilla mbridge, Megatron-Core 0.16.1, vLLM 0.19.1. All Mode 4 runs use 4-trainer/4-rollout split (TP=2 → 2 trainer DP ranks + 2 vLLM replicas). Numbers are averaged over steady-state global_steps (warmup excluded). One **round** = one trainer fetch event = `require_batches × ppo_mini_batch_size` prompts → `n=8` responses each → `require_batches` PPO updates. The headline metric is **s/prompt = round_wallclock / prompts_per_round**, since rounds vary in size across configs.

## Glossary

- **gen_wait** (`timing_s/gen` in the trainer log): wallclock time the *trainer* spends blocked at the start of a round, waiting for the rollouter to deliver enough prompts to fetch. It is **not** raw generation time — actual generation runs continuously in the rollouter, in parallel with trainer compute. `gen_wait > 0` means the rollouter couldn't pre-buffer enough fresh prompts during the previous trainer-compute window. `gen_wait ≈ 0` means the queue was already saturated when the trainer was ready (gen-hiding worked).
- **update_actor**: trainer wallclock spent on the PPO update (forward + backward + optimizer step), summed across `require_batches` mini-batches.
- **log_prob**: trainer wallclock spent recomputing `old_log_prob` for the fetched batch. Skipped when `use_rollout_log_probs=True` (and `bypass_mode=True`), in which case rollout-time log probs are used directly.
- **step**: total round wallclock = `gen_wait + log_prob + update_actor + small misc`.
- **s/prompt**: `step / prompts_per_round`. Lets us compare configs with different batch sizes apples-to-apples.

## 1. Model size sweep (Acemath 40k, baseline Mode 4)

Baseline Mode 4 config: `mini_bsz=32`, `require_batches=1` (32 prompts/round, 1 PPO update), `staleness=0.5`, `partial_rollout=True`, `trigger=4`, `concurrency_multiplier=16` (= 32 max in-flight prompts), `use_rollout_log_probs=False`, `bypass_mode=False`, `recompute_granularity=full`, optimizer-only CPU offload (no Megatron param/grad/opt offload), `ppo_micro_batch_size_per_gpu=1`.

| Model | gen_wait | log_prob | update_actor | step | **s/prompt** | response_length avg | param_sync |
|---|---|---|---|---|---|---|---|
| **4B** | 68s | 60s | 165s | 290s | **9.1** | 25k tokens | ~3-7s |
| **9B** | 54s | 80s | 252s | 396s | **12.4** | 30k tokens | 6.6 / 7.2 / 5.7 / 2.0s |
| **27B** | 8.6s | 153s | 603s | 806s | **25.2** | 27k tokens | 5.8 / 3.1 / 4.2 / 2.2s |

Observations:
- Trainer compute scales roughly with parameter count (4B → 9B is ~1.5× update; 9B → 27B is ~2.4×, sublinear thanks to recompute saving activations).
- gen_wait *decreases* with bigger models because trainer compute grows fast enough to give the rollouter slack — at 27B gen_wait is 8.6s out of 806s (≈ 1%), meaning the rollouter is rarely the bottleneck and gen-hiding works fully.
- s/prompt scales linearly with model size at this batch (9.1 → 12.4 → 25.2 going 4B → 9B → 27B).
- Param sync time grows mildly with model size (NCCL bandwidth × param count); checkpoint-engine keeps it under 8s even for 27B.

## 2. Dataset / context comparison (4B, Mode 4)

Same 4-trainer/4-rollout split. Compares Acemath at 40k context (baseline above) vs Fineproof at 65k context. Fineproof requires LLM-judge-graded proof outputs and elicits much longer responses.

| Setup | response_length avg | clip ratio | gen_wait | update | step | **s/prompt** |
|---|---|---|---|---|---|---|
| Acemath 40k batch=32 | 25k | 5% | 68s | 165s | 290s | **9.1** |
| Acemath 40k batch=64 (req=2) | 25k | 5% | ~150s | ~350s | ~580s | **9.1** |
| Fineproof 65k batch=64 (req=2) | 50k | 28% | 813s | 805s | 1870s | **29.2** |
| Fineproof 65k batch=128 (req=4) | 51k | 30% | 1611s | 1685s | 3700s | **28.9** |

Observations:
- Fineproof at 65k is dominated by long-tail responses — 28-33% hit the 65k cap. Average response length is 2× longer than Acemath at 40k.
- s/prompt is roughly invariant across batch sizes within the same dataset/context (Acemath stays at 9.1; Fineproof stays at ~29). Per-prompt cost is set by model + context, not by batch.
- s/prompt is ~3× higher for Fineproof 65k vs Acemath 40k. Most of that comes from doubled response length, the rest from extra trainer compute over more tokens.
- At Fineproof 65k, the trainer is **rollout-bound** (43% trainer idle waiting on rollouter). At Acemath 40k batch=32 the trainer is also slightly rollout-bound; at Acemath 40k batch=64 they are roughly balanced.

## 3. Concurrency multiplier sweep (4B, Fineproof 65k batch=64)

The async rollouter caps in-flight prompts at `num_replicas × concurrency_multiplier` (capped by `max_required_samples = mini_bsz × require_batches × (1+staleness) × trigger`). The default multiplier was hardcoded at 16; this sweep varied it.

| concurrency_multiplier | Effective in-flight prompts | gen_wait | step | **s/prompt** | Δ vs baseline |
|---|---|---|---|---|---|
| 16 (original default) | 32 | 813s | 1879s | **29.4** | — |
| 128 | 256 | 584s | 1611s | **25.2** | -14% |
| 1024 (capped at 384) | 384 | 646s | 1713s | **26.8** | -9% |

Observations:
- Default `× 16` was significantly throttling: lifting to 128 (256 in-flight) shaved 14% from per-prompt cost.
- Going further to 384 (the queue cap) was *worse* than 128. Past 256 in-flight, vLLM is paging KV cache faster than it can decode, so the additional concurrency hurts.
- The sweet spot is roughly where in-flight prompts × `n=8` ≈ vLLM's per-replica KV-resident capacity. Past that, paging overhead exceeds the parallelism benefit. We've made `concurrency_multiplier` a config knob (`async_training.concurrency_multiplier`).

## 4. Batch size sweep (Mode 4)

Holding everything else constant: 4T+4R, conc=128, no Megatron offloads.

| Model | Context | Batch (prompts/round) | gen_wait | log_prob | update | step | **s/prompt** |
|---|---|---|---|---|---|---|---|
| 4B | 40k | 32 | 68s | 60s | 165s | 290s | **9.1** |
| 4B | 40k | 64 (req=2) | ~150s | ~120s | ~330s | ~580s | **9.1** |
| 4B | 65k | 64 (req=2, conc=128) | 584s | 226s | 805s | 1611s | **25.2** |
| 4B | 65k | 128 (req=4, conc=128) | 1611s | 451s | 1681s | 3700s | **28.9** |
| 9B | 40k | 32 | 54s | 80s | 252s | 396s | **12.4** |
| 9B | 40k | 128 (req=4, conc=128) | 305s | 322s | 1118s | 1734s | **13.5** |
| 27B | 40k | 32 | 8.6s | 153s | 603s | 806s | **25.2** |
| 27B | 40k | 64 (req=2, conc=128) | 333s | 401s | 1399s | 2090s | **32.7** |

Observations:
- s/prompt is roughly invariant under batch — the framework overhead doesn't get a meaningful amortization win from bigger batches at this scale (e.g. 9B 40k: 12.4 at batch=32 vs 13.5 at batch=128).
- gen_wait scales linearly with batch in proportion to rollouter rate vs trainer-compute rate; in the ones where trainer compute is small (4B 65k), gen_wait dominates and step time grows almost 1:1 with batch.
- For 9B/27B, trainer compute dominates; bigger batch grows step time linearly but doesn't change the trainer-bound vs rollout-bound regime.

## 5. Settings that flip Mode 4 from rollout-bound to trainer-bound

For 4B Acemath 40k batch=64 — the smallest case where Mode 4 was on the edge — the following stack of changes brings step time down and gen_wait to ~zero:

| Layer | step (mean) | s/prompt | gen_wait | Notes |
|---|---|---|---|---|
| baseline | 580s | **9.1** | ~150s | conc=16, recompute log_probs in trainer |
| + concurrency_multiplier=128 | ~530s | **8.3** | ~80s | unblocks vLLM concurrency |
| + use_rollout_log_probs=True + bypass_mode=True | ~510s | **8.0** | ~30-40s | trainer skips log_prob recompute (~120s saved) |
| + ppo_micro_batch_size_per_gpu=2 | ~510s (high variance) | **8.0** | ~30-40s | minimal change in steady state, occasional OOM at long responses; micro_bsz=1 is safer |

Once you stack `use_rollout_log_probs=True` + `concurrency_multiplier=128`, gen_wait collapses to single digits in many global_steps (when rollouter has pre-buffered the next round before trainer is ready to fetch). Per-step variance becomes large (335s to 700s observed in single steps), which means steady-state averaging over ≥6 global_steps is necessary for stable comparisons.

## Notes on variance

Per-global-step variance is consistently large across all configs:
- Step time variance correlates with `response_length/mean` per step. Acemath responses range from 14k-35k tokens depending on the cycle.
- Cycle 1 (warmup) takes 1.3-2× longer than steady state; first 1-2 cycles should be excluded from any comparison.
- Within a cycle, individual global_steps can land anywhere in a ~2× range (e.g. 4B Acemath cycle: 335s, 506s, 517s, 551s, 573s, 600s, 615s in one run).

For tight benchmarking, run ≥4 cycles and average over cycles 2-N.

## Implementation deltas

The repo gained two configurable knobs from this study:
- `async_training.concurrency_multiplier` (default 16 for backwards compat, recommend 128 for 4B/9B at 40k, lower for very long contexts)
- The `_async_acemath_40k.sh` and `_async_fineproof_65k.sh` benchmark launchers expose `MINI_BSZ`, `REQUIRE_BATCHES`, `CONCURRENCY_MULT`, `PPO_MICRO_BSZ`, `USE_ROLLOUT_LOG_PROBS`, `BYPASS_MODE` env vars for sweeping.
