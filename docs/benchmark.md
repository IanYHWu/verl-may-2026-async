# Benchmark Report

This document combines two benchmark studies on `fully_async_policy` Mode 4
and (where applicable) the colocate / hybrid-engine path:

- **§1 — H100 (single node, 8× SM90):** Mode 4 vs colocate at 4B, with
  variants of the GPU split (4R/4T, 2R/6T).
- **§2–§6 — B200 (single node, 8× SM100):** the original Mode 4 study
  covering model size (4B/9B/27B), context (40k/65k), concurrency
  multiplier, batch sizes, and the settings stack that flips Mode 4 from
  rollout-bound to trainer-bound.

A direct H100 ↔ B200 comparison appears at the end of §1.

## Glossary

- **gen_wait** (`timing_s/gen` in the trainer log): wallclock time the
  *trainer* spends blocked at the start of a round, waiting for the
  rollouter to deliver enough prompts to fetch. It is **not** raw
  generation time — actual generation runs continuously in the rollouter,
  in parallel with trainer compute. `gen_wait > 0` means the rollouter
  couldn't pre-buffer enough fresh prompts during the previous
  trainer-compute window. `gen_wait ≈ 0` means the queue was already
  saturated when the trainer was ready (gen-hiding worked).
- **update_actor**: trainer wallclock spent on the PPO update (forward +
  backward + optimizer step), summed across `require_batches` mini-batches.
- **log_prob**: trainer wallclock spent recomputing `old_log_prob` for the
  fetched batch. Skipped when `use_rollout_log_probs=True` (and
  `bypass_mode=True`), in which case rollout-time log probs are used
  directly.
- **step**: total round wallclock = `gen_wait + log_prob + update_actor +
  small misc`.
- **s/prompt**: `step / prompts_per_round`. Lets us compare configs with
  different batch sizes apples-to-apples.
- **round**: one trainer fetch event = `require_batches × ppo_mini_batch_size`
  prompts, each producing `n=8` responses, leading to `require_batches` PPO
  updates inside one `_update_actor` call.

---

## §1. H100 single-node — Mode 4 vs Colocate (Qwen3.5-4B, Acemath 40k)

**Setup.** 8× H100 80GB on a single node, Qwen3.5-4B, Acemath dataset, 40k
max response length, 2k prompt context. All three configs match on
training math (32 prompts/round equivalent, 2 PPO mini-batch updates per
round), DAPO loss (clip 0.20/0.28), constant lr=1e-6, full Megatron CPU
optimizer offload, ref param_offload=True.

H100-specific deviations from the B200 baseline (necessary to fit at 40k
on 80 GB):
- **`calculate_entropy=false`** in both colocate and Mode 4. The entropy
  forward materializes a `(seq × vocab)` tensor; with Qwen3.5-4B
  (vocab=248K) and seq=43K, that's ~17 GiB at bf16 per TP shard, which
  doesn't fit alongside the rest of the trainer state. Loss math is
  unaffected because `entropy_coeff=0`.
- **`bypass_mode=True`** in Mode 4 only. The trainer-side recompute of
  `old_log_prob` materializes the same shape and OOMs at 18 GiB on 4-GPU
  trainer setups; bypass uses the rollouter's stored log probs as the old
  policy reference.

(There is now an upstream `transformer_impl.py` patch that skips the
logits clone for monitor-only entropy, which would let `calculate_entropy=true`
fit on H100 at this context length. Numbers below were collected before
that patch and so use `calculate_entropy=false`.)

### Configs

| Config | Trainer (TP, DP) | Rollouter (vLLM) | mini_bsz | require_batches | Prompts/round | Trajectories/round |
|---|---|---|---|---|---|---|
| Colocate (bsz=32) | 8 GPUs (TP=2, DP=4) | shared 8 GPUs, hybrid_engine | 16 | n/a (`ppo_epochs=1`, 2 mini-batches) | 32 | 256 |
| Mode 4 4R/4T | 4 GPUs (TP=2, DP=2) | 4 GPUs (vLLM TP=2 × 2 replicas) | 16 | 2 | 32 | 256 |
| Mode 4 2R/6T | 6 GPUs (TP=2, DP=3) | 2 GPUs (vLLM TP=2 × 1 replica) | 15 | 2 | 30 | 240 |

(2R/6T uses bsz=30 / mbsz=15 instead of 32/16 because DP=3 requires
trajectories evenly divisible by 3.)

### Per-round timings

**Colocate (32 prompts / round, 2 PPO updates / round):**

| Step | total | gen | old_log_prob | update_actor | update_weights |
|---|---|---|---|---|---|
| 1 (warmup) | 754 | 471 | 95 | 182 | 4 |
| 2 | 772 | 480 | 107 | 180 | 4 |
| 3 | 748 | 454 | 122 | 167 | 4 |
| 4 | 697 | 438 | 95 | 159 | 4 |
| **avg 2-4 (steady)** | **739** | **457** | **108** | **169** | **4** |

**Mode 4 4R/4T (32 prompts / round, 2 PPO updates / round):**

| step:N | total | gen_wait | update_actor |
|---|---|---|---|
| 2 (cold start) | 1264 | 932 | 331 |
| 3 | 861 | 546 | 315 |
| 4 (best in cycle 1) | 588 | 287 | 301 |
| 5 (post-sync) | 808 | 470 | 285 |
| 6 | 853 | 541 | 311 |
| **avg 3-6 (steady)** | **778** | **461** | **303** |

**Mode 4 2R/6T (30 prompts / round, 2 PPO updates / round):**

| step:N | total | gen_wait | update_actor |
|---|---|---|---|
| 2 (cold start) | 1839 | 1624 | 214 |
| 3 | 1624 | 1427 | 196 |
| 4 (still trending down) | 1346 | 1156 | 189 |
| **avg 2-4** | **1603** | **1402** | **200** |

### Per-prompt comparison

| Config | gen / gen_wait | log_prob | update_actor | **s/prompt** | Notes |
|---|---|---|---|---|---|
| Colocate | 14.3 | 3.4 | 5.3 | **23.1** | hybrid engine, no async wait |
| Mode 4 4R/4T (avg) | 14.4 | 0 (bypassed) | 9.5 | **24.3** | ~5% slower than colocate on avg |
| Mode 4 4R/4T (best step) | 9.0 | 0 | 9.4 | **18.4** | 26% faster than colocate when pipeline is fully primed |
| Mode 4 2R/6T (avg) | 46.7 | 0 | 6.7 | **53.4** | rollout-starved by 2-GPU vLLM |

End-to-end throughput (prompts / sec):

| Config | prompts/s | vs colocate |
|---|---|---|
| Colocate | 0.0433 | 1.00× |
| Mode 4 4R/4T avg | 0.0411 | 0.95× (5% slower) |
| Mode 4 4R/4T best step | 0.0544 | 1.26× (faster) |
| Mode 4 2R/6T avg | 0.0187 | 0.43× (2.3× slower) |

### Per fwd-bwd consistency check

Update time scales with the number of sequential fwd-bwd passes per DP
rank, which equals `prompts_per_round / DP × ppo_epochs (=1)` (with
`micro_batch=1` and `mini_bsz=16` or `15`). Per-fwd-bwd wallclock should
be roughly constant across configs:

| Config | update_actor (s) | DP | Prompts per DP per round | Per-fwd-bwd (s) |
|---|---|---|---|---|
| Colocate | 169 | 4 | 8 | **21.1** |
| Mode 4 4R/4T | 303 | 2 | 16 | **18.9** |
| Mode 4 2R/6T | 192 | 3 | 10 | **19.2** |

All three land at ~19–21 s/fwd-bwd, confirming the update-time
differences are explained entirely by fwd-bwd count per DP rank, not by
per-call efficiency.

### Why Mode 4 doesn't beat colocate by much at H100 / 4B / 32-prompt round

1. **Update side**: Mode 4 4R/4T gets only 4 trainer GPUs vs colocate's 8,
   so DP=2 vs DP=4. Update time is roughly 2× per round (303 s vs 169 s),
   which adds 4.2 s/prompt to Mode 4's bill. On B200, update_actor is
   smaller and gen_wait dominates more, so the trade is more favourable.
2. **Rollout side is roughly tied per-prompt**. Mode 4's 4 dedicated
   rollout GPUs (gen_wait 14.4 s/prompt) about-match colocate's 8 shared
   GPUs (gen 14.3 s/prompt). On the colocate side, vLLM benefits from 8
   GPUs but pays for memory pressure (gpu_memory_utilization=0.7 of a
   GPU also holding Megatron) and sleep/wake overhead between phases.
3. **Cycle averaging penalises Mode 4**. With `trigger_parameter_sync_step=4`,
   every 4 rounds the rollouter is paused for weight sync. Step 5 (first
   post-sync round) reliably lands ~30–40% slower than the cycle-end
   step 4 because the queue was drained. On a 2-cycle run (`N_CYCLES=2`,
   8 rounds total), this penalty has fewer rounds to amortize over than
   the B200 study's 4–6 cycles.
4. **2R/6T is wrong for this workload.** Two vLLM GPUs serve a single
   TP=2 replica, so concurrent in-flight rollouts are capped low; trainer
   wait absolutely dominates (46.7 s/prompt). The 6-trainer-GPU benefit
   (DP=3 → fewer fwd-bwds per rank → 6.7 s/prompt update vs 9.5 in 4R/4T)
   is small comfort when the trainer sits idle most of the round.

### H100 vs B200 — same experiment

For the corresponding experiment in the B200 study (4B, Acemath, 40k,
~32 prompts/round, Mode 4 4T/4R), the headline numbers are:

| Hardware | Config | Round | gen_wait | log_prob | update_actor | step | **s/prompt** |
|---|---|---|---|---|---|---|---|
| **B200** (8× SM100) | Mode 4 4T/4R, mbsz=32, req=1 | 32 | 68 | 60 | 165 | 290 | **9.1** |
| **B200** | Mode 4 4T/4R, mbsz=32, req=2 (bsz=64) | 64 | ~150 | ~120 | ~330 | ~580 | **9.1** |
| **H100** (8× SM90) | Mode 4 4R/4T, mbsz=16, req=2 (bsz=32) | 32 | 461 | 0 (bypass) | 303 | 778 | **24.3** |
| **H100** | Colocate, bsz=32, mbsz=16 | 32 | 457 | 108 | 169 | 739 | **23.1** |

H100 is **~2.7× slower per prompt** than B200 on the same Mode 4 4R/4T
4B/40k workload. The split of the gap:
- update_actor: 303 s vs 165 s → 1.8× slower on H100 (sm90 vs sm100,
  smaller HBM bandwidth, fewer FP8 / MXFP4 paths exercised at bf16).
- gen_wait: 461 s vs 68 s → 6.8× slower on H100. This is the dominant
  factor and it's primarily a vLLM-throughput gap on the rollouter side
  combined with the cycle-sync penalty (the H100 run was 2 cycles
  averaging over post-sync recovery; the B200 run averaged over 4+).
- The B200 study has no colocate baseline to compare against; on H100,
  colocate and Mode 4 are within 5%.

This is consistent with the broader pattern: at smaller models / shorter
contexts on slower hardware, the rollouter rate limits Mode 4 hard, so
gen-hiding only kicks in transiently (best step on H100 4R/4T was 18.4
s/prompt — within 2× of B200). At larger models / longer contexts, the
ratio compresses because trainer compute grows fast enough to give the
rollouter slack.

---

## §2. Model size sweep (B200, Acemath 40k, baseline Mode 4)

**Setup**: 8× B200 GPUs (single node), Qwen3.5 dense models, BSHD layout,
vanilla mbridge, Megatron-Core 0.16.1, vLLM 0.19.1. All Mode 4 runs in
this section use 4-trainer/4-rollout split (TP=2 → 2 trainer DP ranks +
2 vLLM replicas). Numbers averaged over steady-state global_steps
(warmup excluded). One **round** = one trainer fetch event =
`require_batches × ppo_mini_batch_size` prompts → `n=8` responses each
→ `require_batches` PPO updates.

Baseline Mode 4 config: `mini_bsz=32`, `require_batches=1` (32
prompts/round, 1 PPO update), `staleness=0.5`, `partial_rollout=True`,
`trigger=4`, `concurrency_multiplier=16` (= 32 max in-flight prompts),
`use_rollout_log_probs=False`, `bypass_mode=False`,
`recompute_granularity=full`, optimizer-only CPU offload (no Megatron
param/grad/opt offload), `ppo_micro_batch_size_per_gpu=1`.

| Model | gen_wait | log_prob | update_actor | step | **s/prompt** | response_length avg | param_sync |
|---|---|---|---|---|---|---|---|
| **4B** | 68s | 60s | 165s | 290s | **9.1** | 25k tokens | ~3-7s |
| **9B** | 54s | 80s | 252s | 396s | **12.4** | 30k tokens | 6.6 / 7.2 / 5.7 / 2.0s |
| **27B** | 8.6s | 153s | 603s | 806s | **25.2** | 27k tokens | 5.8 / 3.1 / 4.2 / 2.2s |

Observations:
- Trainer compute scales roughly with parameter count (4B → 9B is ~1.5×
  update; 9B → 27B is ~2.4×, sublinear thanks to recompute saving
  activations).
- gen_wait *decreases* with bigger models because trainer compute grows
  fast enough to give the rollouter slack — at 27B gen_wait is 8.6s out
  of 806s (≈ 1%), meaning the rollouter is rarely the bottleneck and
  gen-hiding works fully.
- s/prompt scales linearly with model size at this batch (9.1 → 12.4 →
  25.2 going 4B → 9B → 27B).
- Param sync time grows mildly with model size (NCCL bandwidth × param
  count); checkpoint-engine keeps it under 8s even for 27B.

## §3. Dataset / context comparison (B200, 4B, Mode 4)

Same 4-trainer/4-rollout split. Compares Acemath at 40k context (baseline
above) vs Fineproof at 65k context. Fineproof requires LLM-judge-graded
proof outputs and elicits much longer responses.

| Setup | response_length avg | clip ratio | gen_wait | update | step | **s/prompt** |
|---|---|---|---|---|---|---|
| Acemath 40k batch=32 | 25k | 5% | 68s | 165s | 290s | **9.1** |
| Acemath 40k batch=64 (req=2) | 25k | 5% | ~150s | ~350s | ~580s | **9.1** |
| Fineproof 65k batch=64 (req=2) | 50k | 28% | 813s | 805s | 1870s | **29.2** |
| Fineproof 65k batch=128 (req=4) | 51k | 30% | 1611s | 1685s | 3700s | **28.9** |

Observations:
- Fineproof at 65k is dominated by long-tail responses — 28-33% hit the
  65k cap. Average response length is 2× longer than Acemath at 40k.
- s/prompt is roughly invariant across batch sizes within the same
  dataset/context (Acemath stays at 9.1; Fineproof stays at ~29).
  Per-prompt cost is set by model + context, not by batch.
- s/prompt is ~3× higher for Fineproof 65k vs Acemath 40k. Most of that
  comes from doubled response length, the rest from extra trainer
  compute over more tokens.
- At Fineproof 65k, the trainer is **rollout-bound** (43% trainer idle
  waiting on rollouter). At Acemath 40k batch=32 the trainer is also
  slightly rollout-bound; at Acemath 40k batch=64 they are roughly
  balanced.

## §4. Concurrency multiplier sweep (B200, 4B, Fineproof 65k batch=64)

The async rollouter caps in-flight prompts at
`num_replicas × concurrency_multiplier` (capped by `max_required_samples
= mini_bsz × require_batches × (1+staleness) × trigger`). The default
multiplier was hardcoded at 16; this sweep varied it.

| concurrency_multiplier | Effective in-flight prompts | gen_wait | step | **s/prompt** | Δ vs baseline |
|---|---|---|---|---|---|
| 16 (original default) | 32 | 813s | 1879s | **29.4** | — |
| 128 | 256 | 584s | 1611s | **25.2** | -14% |
| 1024 (capped at 384) | 384 | 646s | 1713s | **26.8** | -9% |

Observations:
- Default `× 16` was significantly throttling: lifting to 128 (256
  in-flight) shaved 14% from per-prompt cost.
- Going further to 384 (the queue cap) was *worse* than 128. Past 256
  in-flight, vLLM is paging KV cache faster than it can decode, so the
  additional concurrency hurts.
- The sweet spot is roughly where in-flight prompts × `n=8` ≈ vLLM's
  per-replica KV-resident capacity. Past that, paging overhead exceeds
  the parallelism benefit. We've made `concurrency_multiplier` a config
  knob (`async_training.concurrency_multiplier`).

## §5. Batch size sweep (B200, Mode 4)

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
- s/prompt is roughly invariant under batch — the framework overhead
  doesn't get a meaningful amortization win from bigger batches at this
  scale (e.g. 9B 40k: 12.4 at batch=32 vs 13.5 at batch=128).
- gen_wait scales linearly with batch in proportion to rollouter rate vs
  trainer-compute rate; in the ones where trainer compute is small (4B
  65k), gen_wait dominates and step time grows almost 1:1 with batch.
- For 9B/27B, trainer compute dominates; bigger batch grows step time
  linearly but doesn't change the trainer-bound vs rollout-bound regime.

## §6. Settings that flip Mode 4 from rollout-bound to trainer-bound (B200)

For 4B Acemath 40k batch=64 — the smallest case where Mode 4 was on the
edge — the following stack of changes brings step time down and gen_wait
to ~zero:

| Layer | step (mean) | s/prompt | gen_wait | Notes |
|---|---|---|---|---|
| baseline | 580s | **9.1** | ~150s | conc=16, recompute log_probs in trainer |
| + concurrency_multiplier=128 | ~530s | **8.3** | ~80s | unblocks vLLM concurrency |
| + use_rollout_log_probs=True + bypass_mode=True | ~510s | **8.0** | ~30-40s | trainer skips log_prob recompute (~120s saved) |
| + ppo_micro_batch_size_per_gpu=2 | ~510s (high variance) | **8.0** | ~30-40s | minimal change in steady state, occasional OOM at long responses; micro_bsz=1 is safer |

Once you stack `use_rollout_log_probs=True` + `concurrency_multiplier=128`,
gen_wait collapses to single digits in many global_steps (when rollouter
has pre-buffered the next round before trainer is ready to fetch).
Per-step variance becomes large (335s to 700s observed in single steps),
which means steady-state averaging over ≥6 global_steps is necessary for
stable comparisons.

## Notes on variance

Per-global-step variance is consistently large across all configs:
- Step time variance correlates with `response_length/mean` per step.
  Acemath responses range from 14k-35k tokens depending on the cycle.
- Cycle 1 (warmup) takes 1.3-2× longer than steady state; first 1-2
  cycles should be excluded from any comparison.
- Within a cycle, individual global_steps can land anywhere in a ~2×
  range (e.g. 4B Acemath cycle: 335s, 506s, 517s, 551s, 573s, 600s, 615s
  in one run).

For tight benchmarking, run ≥4 cycles and average over cycles 2-N.

## Implementation deltas

The repo gained two configurable knobs from this study:
- `async_training.concurrency_multiplier` (default 16 for backwards
  compat, recommend 128 for 4B/9B at 40k, lower for very long contexts)
- The `_async_acemath_40k.sh` and `_async_fineproof_65k.sh` benchmark
  launchers expose `MINI_BSZ`, `REQUIRE_BATCHES`, `CONCURRENCY_MULT`,
  `PPO_MICRO_BSZ`, `USE_ROLLOUT_LOG_PROBS`, `BYPASS_MODE` env vars for
  sweeping.
