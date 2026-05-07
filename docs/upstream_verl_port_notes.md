# Porting from this fork to upstream verl: regression notes

Companion to the [Install + Limitations](../README.md#install) sections in
the top-level README (which absorbed the former `b200_qwen35_megatron_bringup.md`).

This fork (`verl-may-2026`, root commit `8f16fc3`) is roughly aligned to a verl
snapshot from before upstream's "engine workers" refactor. Upstream HEAD at the
time of writing (`a435148`) has diverged in ways that affect the use cases this
fork has been validated on. This doc enumerates the regressions, what stays the
same, and what to re-validate when rebasing.

## Use cases this fork has been validated on

1. **Colocate sync GRPO** with `verl.trainer.main_ppo`, `hybrid_engine=True`,
   Megatron-LM + mbridge for training, vLLM for rollout, on 8× B200 (and 8× H100
   on the FLAME cluster).
2. **B200 / Qwen3.5** dense + GatedDeltaNet hybrid attention, BSHD layout
   (because GDN rejects packed sequences).
3. **DAPO reward path** with `math_dapo` rule-based scoring, optionally a 0.0 /
   1.0 patch (instead of upstream's 1.0 / −1.0).
4. **Async paths (Mode 1–4)**: **NOT working in this fork** for the new
   `ServerAdapter` rollout — see "Async path (already broken in this fork)"
   below for the reason. Upstream main has fixed this.

## Status of fork patches when rebased on upstream main

### (a) MTP routing for Qwen3.5 — STILL NEEDED, will conflict on rebase

- Upstream `verl/models/mcore/registry.py:33-34` adds `QWEN3_5_MOE_VL` and
  `QWEN3_5_VL` to `SupportedVLM`. The fork deliberately removes `QWEN3_5_VL`
  from that enum (registry.py:33-40 in fork).
- After rebase, the same one-line removal must be re-applied. Otherwise
  text-only Qwen3.5 routes through `model_forward_gen(True)` (THD), which
  GatedDeltaNet rejects with `NotImplementedError: GDN does not support packed
  sequence for now.`
- `verl/models/mcore/mtp_patch.py` is byte-identical between fork and
  upstream — no action needed.

### (b) MRoPE + BSHD fix in `model_forward.py` — NEEDED, non-trivial re-port

- Upstream heavily refactored `model_forward.py`. Function renamed
  `gptmodel_forward_no_padding` → `gptmodel_forward_model_engine`, and the
  BSHD branch (`else: # bshd`) now does
  `position_ids=None if vision_model else new_position_ids` — i.e. it only
  forces `None` when `vision_model=True`.
- The fork's `mrope_section`-based detection (model_forward.py:141-151) is
  gone upstream. Under upstream's logic Qwen3.5 routed as non-vision will pass
  1D `position_ids` to a model expecting 3D MRoPE → `IndexError: too many
  indices for tensor of dimension 2`.
- Re-port required: detect `tf_config.mrope_section is not None` and force
  `position_ids=None` even on the non-VLM branch. Localized but tricky —
  the surrounding MTP/CP scaffolding is new.
- Also: fork's loosened `assert not (vision_model and has_vision_inputs)`
  was tightened back upstream. Re-loosen if you need the BSHD path to accept
  text-only batches under a VL routing.

### (c) `AutoModelForVision2Seq` alias — UPSTREAM NOW (drop the fork patch)

- Upstream added `verl/utils/transformers_compat.py` with
  `get_auto_model_for_vision2seq()` (lru-cached). `verl/utils/model.py`
  uses this helper.
- Fork's manual aliasing in `utils/model.py:33,42-50` is now redundant and
  conflicts with upstream's pattern. **Drop the fork patch** and use the
  upstream API as-is.

### (d) `apply_chat_template_kwargs.return_dict=False` — no change needed

- Hydra-passable in both versions.

### (e) `mtp_num_layers=null` override — STILL NEEDED for dense Qwen3.5

- Upstream `config_converter.py:281-304` adds
  `hf_to_mcore_config_qwen3_5_moe` that auto-sets
  `transformer_config.mtp_num_layers = mtp_num_hidden_layers` for the
  **MoE** variant. Dense Qwen3.5 (e.g. 2B / 4B) does **not** use this
  converter, so the user's `+actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=null`
  Hydra override is still required for dense Qwen3.5 with
  `mtp_num_hidden_layers > 0`. `override_transformer_config.*` is still
  honored upstream.

### (f) mbridge from git for `qwen3_5` — UNCHANGED

- Both upstream and fork declare bare `mbridge` in `setup.py:60` with no
  version pin. PyPI `mbridge==0.15.1` still lacks `qwen3_5`. Same git pin
  needed:
  ```bash
  pip install --no-build-isolation --no-deps \
      "git+https://github.com/ISEEKYAN/mbridge.git@4cfd6f5eab84ed5424a8202e1a282e6ac584fce5"
  ```
  See `docs/b200_qwen35_megatron_bringup.md` for the full pin set.

### (g) `math_dapo.py` 0/1 reward patch — STILL NEEDED, re-applies cleanly

- Upstream `verl/utils/reward_score/math_dapo.py:217,239,257,265` still
  returns `1 / -1` and `1.0 / -1.0`.
- Fork's 0.0 patch is the only divergence and is a clean re-apply.
- DAPO reward manager (`verl/workers/reward_manager/dapo.py`) is
  byte-identical.

## Major architectural risk: the engine_workers cutover

Upstream removed the `use_legacy_worker_impl` switch in `verl/trainer/main_ppo.py`.
Only the new unified `engine_workers.ActorRolloutRefWorker` path remains
(main_ppo.py:128-145).

The fork's colocate Megatron flow uses
`verl.workers.megatron_workers.AsyncActorRolloutRefWorker` (the legacy class).
That class is gone upstream. The user's
`scripts/qwen3_4b_inst_acemath_colocate.sh` flow has never been validated
against `engine_workers.ActorRolloutRefWorker`.

`verl/trainer/ppo/ray_trainer.py` has a ~600-line diff: new
`CheckpointEngineManager`, `LLMServerManager`, `DistillationConfig`,
`EngineConfig` (replaces `FSDPEngineConfig`), `GDPO` advantage estimator.
Anything calling old internals will break.

`hybrid_engine: true` is still the default in `ppo_trainer.yaml:53`, but
the worker class it instantiates is different. Behavior on the
Megatron + vLLM colocate path needs end-to-end re-validation.

## Async path: already broken in this fork

The fork's `verl/experimental/fully_async_policy/` infrastructure
(`param_sync.py`, `megatron_worker.py`, `fsdp_workers.py`,
`base_detach_sync.py`, `checkpoint_engine.py`) is a custom sync stack
designed for the **legacy direct vLLM rollout** (with
`rollout.inference_engine` exposed).

The fork upgraded the rollout to `ServerAdapter` (vLLM HTTP server,
engine in a separate process) for `rollout.mode=async`, but the sync
stack was never updated. Both per-step sync paths
(`param_synchronizer.sync_weights` and `_fit_update_weights →
checkpoint_manager.update_weights`) have their actual NCCL/IPC transfer
calls **commented out**. With them uncommented they raise
`AttributeError: 'ServerAdapter' object has no attribute
'inference_engine'` in `BaseDetachNcclSync.get_inference_model`.

Symptom: log_ppl_diff grows unbounded (rollouter's vLLM stays at the
initial weights), reward bounces in a narrow band, no learning.

Upstream replaced the entire fork-custom sync stack with
`verl/checkpoint_engine/` (CheckpointEngineManager + per-replica
RolloutReplica + ServerAdapter IPC `update_weights`). Upstream's
`FullyAsyncTrainer.__init__` initializes
`self.checkpoint_manager = None`, `set_rollouter(rollouter)` calls
`_setup_checkpoint_manager`, and `_fit_update_weights` calls
`await self.checkpoint_manager.update_weights(...)`. Upstream main is
the right target if the user wants async to work.

## B200 / Qwen3.5 hardware story

- **Toolchain not pinned upstream**. No upstream record of torch 2.10 +
  TE 2.13 + flash_attn 2.8.3 + vLLM 0.19.1 + megatron-core 0.16.1 +
  mbridge `4cfd6f5`.
- **`docs/b200_qwen35_megatron_bringup.md` is fork-only** and stays the
  authoritative install recipe. The LD_LIBRARY_PATH workaround for
  cudnn `dlopen` from the `nvidia-cudnn-cu12` wheel is not solved
  upstream.
- The custom build-time CPATH / LIBRARY_PATH dance for compiling
  `transformer_engine_torch` against torch 2.10 + nvcc 12.8 is also
  documented only in the fork.

## H100-specific gotchas (from the recent run)

- **flash-linear-attention** required on Hopper for Qwen3.5's GDN
  layers (`pip install flash-linear-attention`). Upstream doesn't
  declare this dependency either.
- **tilelang** required on Hopper to work around a Triton ≥ 3.4 bug in
  `gated chunk_bwd_dqkwg` (FLA error message points at this directly).
  Same situation upstream.
- **vLLM `gpu_memory_utilization`** budget is tighter on H100 than B200
  (80 GB vs 180 GB). The B200 doc's `0.7` setting works on H100 only at
  smaller `max_response_length` (e.g. 16k–32k) and reduced concurrency.
  See `scripts/qwen3_4b_inst_acemath_colocate.sh` for a known-working H100
  configuration: `max_response_length=16k`, `gpu_memory_utilization=0.8`,
  `train_batch_size=64`, `ppo_mini_batch_size=32`, `rollout.n=8`.

## Net porting assessment

| Item | Severity | Effort |
|---|---|---|
| (a) Drop `QWEN3_5_VL` from `SupportedVLM` | Low | 1 line |
| (b) MRoPE BSHD re-port onto renamed forward | High | ~1–2 hours |
| (c) Drop legacy `AutoModelForVision2Seq` patch | Low | delete code |
| (d) `apply_chat_template_kwargs.return_dict` | None | no change |
| (e) `mtp_num_layers=null` Hydra override | None | no change |
| (f) mbridge git pin | None | no change |
| (g) `math_dapo` 0/1 reward patch | Low | 4 lines |
| **engine_workers cutover** | **High** | **half day to a day to re-validate colocate Megatron path** |
| Async path | Done upstream | use upstream's `CheckpointEngineManager` sync; lose fork-specific MIS proximal-anchor swap unless re-ported |

**Plan:** fork from upstream HEAD, accept the validation cost on the new
unified `engine_workers.ActorRolloutRefWorker`, and pick up the post-cutover
features (working async sync, GDPO advantage estimator, distillation,
LLMServerManager).

Steps:
- Re-apply (a), (b), (g) as small, surgical patches on top of HEAD.
- Drop (c) entirely.
- Keep `docs/b200_qwen35_megatron_bringup.md` and the H100 launcher tweaks
  in this doc as the install / configuration recipe.
- **Validate** `scripts/qwen3_4b_inst_acemath_colocate.sh` end-to-end against
  the new worker class before any production run. Watch in particular for:
  - vLLM hybrid-engine init under the new `EngineConfig` shape (replaces the
    old `FSDPEngineConfig`)
  - Megatron weight handoff to vLLM (the colocate path used to share the
    underlying tensors; the unified worker may go through `CheckpointEngineManager`
    even with `hybrid_engine=True`)
  - TP=2 sharding behavior on the actor
  - Reward parity vs the fork on the first 30 steps of the same workload
- For async, drop the fork's `verl/experimental/fully_async_policy/{param_sync,megatron_worker,fsdp_workers,base_detach_sync,checkpoint_engine}.py` and use upstream's `verl/checkpoint_engine/` infrastructure as-is.
