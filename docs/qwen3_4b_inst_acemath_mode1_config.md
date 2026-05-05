# Mode 1 reference run — config snapshot

The canonical config for the
`qwen3_4b_inst_acemath_mode1_16k_bsz64_mbsz32_clip028_lr1e6_wd001`
experiment in wandb project `async_verl_debug`. Companion to
[`ASYNC_EXPERIMENT.md`](../ASYNC_EXPERIMENT.md). Launcher:
[`scripts/qwen3_4b_inst_acemath_async_mode1.sh`](../scripts/qwen3_4b_inst_acemath_async_mode1.sh).

## Model & data

| | |
|---|---|
| Model | Qwen3-4B-Instruct-2507 |
| Dataset | `HerrHruby/acemath_rl_4b_inst_hard` (DAPO-formatted parquet at `/tmp/ianwu/data/acemath_dapo_format.parquet`) |
| Reward | `math_dapo` rule-based, **1.0 correct / 0.0 wrong**, overlong penalty disabled |

## Sampling

| | |
|---|---|
| `temperature` | 1.0 |
| `top_p` | 1.0 |
| `top_k` | -1 |
| `n` (responses per prompt) | 8 |
| `max_prompt_length` | 2048 |
| `max_response_length` | 16384 |

## Batch / update math

| | |
|---|---|
| `require_batches` | 2 |
| `ppo_mini_batch_size` | 32 prompts |
| Prompts collected per training step | 64 (= 2 × 32) |
| Trajectories per training step | 512 (= 64 × 8) |
| Mini-batches per training step | 2 of 32 prompts each (1 on-policy + 1 off-policy) |
| `ppo_epochs` | 1 |
| `ppo_micro_batch_size_per_gpu` | 1 |
| `use_dynamic_bsz` | False |

## Optimization

| | |
|---|---|
| Algorithm | GRPO (`algorithm.adv_estimator=grpo`) |
| `lr` | 1e-6 (constant, no warmup) |
| `weight_decay` | 0.01 |
| `clip_ratio_low` | 0.20 |
| `clip_ratio_high` | 0.28 (DAPO clip-higher) |
| `clip_ratio_c` | 10.0 |
| `entropy_coeff` | 0 (logged but not part of loss) |
| `kl_loss_coef` | 0 (`use_kl_loss=False`) |
| `algorithm.use_kl_in_reward` | False |
| `loss_agg_mode` | `token-mean` |

## Async topology

| | |
|---|---|
| Mode | 1 (on-policy pipeline) |
| `trigger_parameter_sync_step` | 1 |
| `staleness_threshold` | 0.0 |
| `partial_rollout` | False |
| Trainer | 4× H100, TP=2 → DP=2 |
| Rollouter | 4× H100, vLLM TP=2 → 2 servers, `gpu_memory_utilization=0.7` |
| `max_concurrent_samples` | 32 (= 2 servers × 16, upstream default) |
| `algorithm.rollout_correction.bypass_mode` | False (so trainer recomputes `old_log_probs`) |
| `actor.use_rollout_log_probs` | False (no-op for megatron actor) |
| Weight sync | NCCL via upstream `CheckpointEngineManager` (`backend=nccl`) |

## Notes

- **Comparison target**: `scripts/qwen3_4b_inst_acemath_colocate.sh`
  uses the same hyperparams (clip / lr / wd / batch math / sampling /
  reward) on `hybrid_engine=True` (8 GPUs shared between trainer and
  rollout, DP=4 × TP=2). Mode 1 should track its reward curve within
  ~5 pp.
- **Health metric**: `rollout_corr/log_ppl_diff` should stay around
  0.0006–0.005. If it grows beyond that, the rollouter isn't getting
  fresh weights.
- **`max_concurrent_samples`** is hardcoded to `num_servers * 16` in
  upstream (vs the fork's `* 8` patch we used to mitigate H100 KV
  pressure with `partial_rollout=True`). For Mode 1 with
  `partial_rollout=False` this should be fine.
