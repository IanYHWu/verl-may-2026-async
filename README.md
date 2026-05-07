# Verl (May 2026 Fork)

A fork of [verl](https://github.com/volcengine/verl) tracked against May 2026
upstream HEAD, with the patches needed to run **Qwen3.5** RL on Blackwell
(B200, SM100) with the **fully-async** trainer/rollouter pipeline. Headline
additions on top of upstream:

- **Qwen3.5 hybrid attention (dense + GatedDeltaNet) trains end-to-end** with
  Megatron-Core + mbridge in BSHD layout. The GDN linear-attention path
  rejects packed (THD) sequences, so we route Qwen3.5 through a non-VL
  forward and pad statically; details in
  [`docs/b200_qwen35_megatron_bringup.md`](docs/b200_qwen35_megatron_bringup.md).
- **Async (decoupled trainer/rollouter) is the default for long-context RL.**
  We've validated Qwen3.5-4B on `fully_async_policy` Mode 1 (on-policy
  pipeline) and Mode 4 (async stream + partial rollout) end-to-end at 40k
  and 65k response lengths.
- **Per-step wandb logging** in the fully-async trainer. The upstream code
  only flushed metrics at param-sync time (every `trigger_parameter_sync_step`
  iters), which collapses within-cycle reward dynamics into a single average
  in Mode 4. Now every grad update emits a per-step row.
- **LLM-as-judge reward manager.** A new `llm_judge` reward manager calls a
  hosted OpenAI-compatible chat-completions endpoint (e.g. a Cloudflare
  Worker proxying gpt-oss / qwen / Anthropic) for rubric-based grading of
  proof-style problems. Strips policy `</think>` regions before grading.
  Details in [`verl/utils/judge/README.md`](verl/utils/judge/README.md).

> **Tested with Megatron only.** The FSDP path likely still works (we
> haven't broken it), but none of our changes have been validated against it.
> Reach for this fork if you specifically need Qwen3.5 + Megatron + async on
> Blackwell; otherwise stock upstream verl will probably do.

## Install

The Blackwell + Megatron + Qwen3.5 stack is non-trivial to assemble. The
sequence below is what produced our verified env. Runtime versions:
torch 2.10 · TE 2.13 · flash-attn 2.8.3 · megatron-core 0.16.1 · vLLM 0.19.1
· mbridge 0.15.1 · flash-linear-attention 0.4.2.

```bash
# 1. Fresh conda env (Python 3.10).
conda create -n verl_megatron python=3.10
conda activate verl_megatron

# 2. cudnn first — TransformerEngine dlopens libcudnn_graph.so.9 at import.
pip install nvidia-cudnn-cu12

# 3. Run the upstream stack installer, BUT stop before the TE step (step 4
#    in the script) — we'll handle TE ourselves below to control build flags.
#    Comment out the TE + Megatron-LM lines in the script first, or run by
#    hand up to that point.
USE_MEGATRON=1 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh

# 4. Install TransformerEngine from source against the active torch.
#    --no-build-isolation reuses the already-installed torch / cudnn /
#    nccl wheels rather than pulling a fresh isolated copy.
NVTE_FRAMEWORK=pytorch pip install --no-build-isolation --no-deps \
    git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
# If the v2.6 ABI doesn't line up with torch 2.10 on your stack, repeat
# against @v2.13 — that's what landed in the verified env.

# 5. Install Megatron-LM (the line we skipped from the install script).
pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1

# 6. Build apex from source (CPP + CUDA extensions). Slow.
git clone https://github.com/NVIDIA/apex.git && cd apex && \
    MAX_JOB=16 pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./ && \
    cd ..

# 7. mbridge pinned to the commit the bring-up validated.
pip install git+https://github.com/ISEEKYAN/mbridge.git@4cfd6f5eab84ed5424a8202e1a282e6ac584fce5

# 8. flash-linear-attention — Megatron-Core's GatedDeltaNet imports
#    `fla.modules.l2norm.l2norm` and `fla.ops.gated_delta_rule.chunk_gated_delta_rule`.
pip install flash-linear-attention==0.4.2

# 9. Install verl itself (this fork) in editable mode.
pip install --no-deps -e .
```

### Runtime LD_LIBRARY_PATH

TransformerEngine `dlopen`s `libcudnn_graph.so.9` at import time. The
cudnn + nccl wheel `lib/` directories must be on `LD_LIBRARY_PATH` for any
launcher you run outside the conda activation that initially sourced them:

```bash
CUDNN_LOC=$(pip show nvidia-cudnn-cu12 | grep Location | cut -d' ' -f2)
NCCL_LOC=$(pip show nvidia-nccl-cu12  | grep Location | cut -d' ' -f2)
export LD_LIBRARY_PATH=$CUDNN_LOC/nvidia/cudnn/lib:$NCCL_LOC/nvidia/nccl/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

The launchers in `scripts/` set this themselves. Without it you'll see
`OSError: libcudnn_graph.so.9: cannot open shared object file` at import.

### Sanity check

```bash
python -c "
import torch, transformer_engine, transformer_engine_torch, flash_attn
import megatron.core, vllm, mbridge
import verl
print('torch', torch.__version__, 'TE', transformer_engine.__version__,
      'TE_torch', transformer_engine_torch.__version__,
      'FA', flash_attn.__version__,
      'megatron-core', megatron.core.__version__,
      'vLLM', vllm.__version__,
      'mbridge', mbridge.__version__)
"
```

If any import fails, see the env-bringup notes in
[`docs/b200_qwen35_megatron_bringup.md`](docs/b200_qwen35_megatron_bringup.md)
— it documents which wheel ABIs needed source rebuilds against torch 2.10
on Blackwell, and the `CPATH` / `LIBRARY_PATH` settings the source builds
require.

## Repo layout

```
verl/                       # forked verl framework
docs/
  b200_qwen35_megatron_bringup.md     # ABI-pinned env + Qwen3.5 patches
  qwen3_4b_inst_acemath_mode1_config.md  # Mode 1 reference run config
  advance/fully_async.md              # upstream's async-training doc
ASYNC_EXPERIMENT.md         # async-vs-colocate verification protocol
verl/utils/judge/           # JudgeClient, parser, templates, README
verl/utils/judge/README.md  # full LLM-judge usage + customization guide
verl/utils/reward_score/math_proof.py   # async compute_score for rubrics
verl/experimental/reward_loop/reward_manager/
  llm_judge.py              # the LLMJudgeRewardManager class
scripts/
  qwen35_4b_32k_colocate.sh            # vanilla GRPO, hybrid engine, 32k
  qwen3_4b_inst_acemath_async_mode1.sh # fully-async Mode 1 reference
```

## Running training

All launchers below are thin wrappers around `python -m verl.trainer.main_ppo`
(colocate path) or `python -m verl.experimental.fully_async_policy.fully_async_main`
(async path), with the Qwen3.5-specific Hydra overrides pre-set. See the
script you're running for the full command.

### Colocate (hybrid engine)

Trainer and rollout share the same GPUs via vLLM's hybrid engine. Best for
short-response workloads or as the known-good baseline.

```bash
# Edit HF_MODEL_PATH and TRAIN_FILE / TEST_FILE at the top of the script
# (or pass them as env vars).
bash scripts/qwen35_4b_32k_colocate.sh
```

The reference colocate config: 8 GPUs shared, `hybrid_engine=True`, TP=2,
gpu_memory_utilization=0.7, recompute_granularity=full, optimizer state
offloaded to CPU.

### Fully-async (separated trainer + rollouter)

Trainer and rollouter run on disjoint GPU pools. Three knobs define the
operating mode (see [`docs/advance/fully_async.md`](docs/advance/fully_async.md)):

- `async_training.trigger_parameter_sync_step` — local grad updates per param
  sync. `=1` is on-policy, larger is more off-policy.
- `async_training.staleness_threshold` — fraction of stale (older-version)
  trajectories the rollouter is allowed to ship. `0.0` is strict on-policy,
  `0.5` is the validated Mode 4 default.
- `async_training.partial_rollout` — interrupt and resume in-flight rollouts
  during param sync (only matters when `staleness_threshold > 0`).

**Mode 1 — on-policy pipeline** (`trigger=1`, `staleness=0`,
`partial_rollout=False`). Cleanest reward signal; matches colocate within
~5pp at matched step counts. Use as the async sanity baseline.

```bash
bash scripts/qwen3_4b_inst_acemath_async_mode1.sh
```

The Mode 1 reference config (Qwen3-4B-Instruct on AceMath DAPO, 4 trainer +
4 rollout, 16k responses, GRPO with DAPO clip-higher 0.20/0.28, lr=1e-6,
wd=0.01, mbsz=32, n=8, 100 cycles → 200 grad updates) is documented in
[`docs/qwen3_4b_inst_acemath_mode1_config.md`](docs/qwen3_4b_inst_acemath_mode1_config.md).

**Mode 4 — async stream pipeline + partial rollout** (`trigger=4`,
`staleness=0.5`, `partial_rollout=True`, `require_batches=1`). Best
throughput on long-response workloads; tolerates a small amount of staleness
in exchange for keeping both pools saturated. At 32k responses, Mode 4 ran
~1.6–1.8× faster than colocate on Qwen3.5-4B/9B/27B.

To build a Mode 4 launcher, take the Mode 1 launcher and override
`TRIGGER_SYNC_STEP=4 STALENESS_THRESHOLD=0.5 PARTIAL_ROLLOUT=True
REQUIRE_BATCHES=1`.

### Health checks during training

The async path emits the standard verl signals plus a few we lean on:

- `rollout_corr/log_ppl_diff` should sit ≲ 5e-4 throughout. If it grows
  toward 1e-2+, the rollouter isn't getting fresh weights — sync is broken.
- `timing_s/param_sync` non-zero per cycle confirms the
  `CheckpointEngineManager` NCCL+IPC path is doing real work.
- `critic/score/mean` trends up over 20–30 grad updates — actual learning.

## LLM-as-judge reward

Set `reward.reward_manager.name=llm_judge` and populate
`reward.reward_kwargs.judge.*` to score rollouts via a hosted chat-completions
endpoint instead of a rule-based function. Drop-in with the same launcher
machinery — same per-sample interface, same `{reward_score, reward_extra_info}`
return shape — so the trainer is unchanged.

Minimum config:

```yaml
reward:
  reward_manager:
    name: llm_judge
  reward_kwargs:
    judge:
      endpoint_url: https://<your-worker>.workers.dev/v1/chat/completions
      model: gpt-oss-20b
      api_key_env: LLM_JUDGE_API_KEY        # never inline keys
      max_score: 7                           # rubric maximum
      max_output_tokens: 4096                # gpt-oss needs headroom for reasoning
      temperature: 0.6
      top_p: 1.0
      reasoning_effort: medium               # gpt-oss-style
      strip_thinking: true                   # default; see judge README
      max_concurrency: 16
      timeout_s: 180
      on_error_score: 0.0
```

By default the judge looks for the *last* `</think>` tag in the policy's
response and only sends what comes after to the grader. If `</think>` is
missing the reward is forced to `on_error_score` (default 0) — we fail
closed rather than grading the raw chain of thought.

Customizable extension points:
- **Prompt templates** in `verl/utils/judge/templates/` (sentinel-based
  `<<problem>>` / `<<response>>` / `<<rubric>>` substitution — no Jinja
  dependency, safe with LaTeX-heavy text).
- **Score parser** for non-`<score>N</score>` formats.
- **Extra dataset fields** — declare a dotted path in
  `reward.reward_kwargs.judge.extra_fields` and reference it as
  `<<your_field>>` in your template (e.g., a reference solution alongside
  the rubric). No code needed.

Full guide, including dataset schema requirements and a checker
(`python -m verl.utils.judge.check_dataset`), is in
[`verl/utils/judge/README.md`](verl/utils/judge/README.md).

## Limitations

**Tested only with Megatron + vLLM.** The FSDP backend should still work but
nothing in this fork has been validated against it.

**Qwen3.5 hybrid architecture is the messy case.** Megatron-Core's
`GatedDeltaNet` (the linear-attention path in the hybrid layers) does not
accept packed sequences, which cascades into several constraints:

- **No THD / no `use_remove_padding`** — must run with
  `actor.megatron.use_remove_padding=false` *and* `model.use_remove_padding=false`.
  The forward dispatches in BSHD throughout.
- **Static micro batches required.** `actor.use_dynamic_bsz=False`,
  `rollout.log_prob_use_dynamic_bsz=False`,
  `actor.ppo_micro_batch_size_per_gpu=1`, and the equivalent for
  log-prob recompute. Padding is to the longest sequence in the batch.
- **No context parallel.** CP in Megatron requires THD; our TF config runs
  with `context_parallel_size=1`. Sequence parallel (TP-side) is fine.
- **Sequence parallel keeps working**, just not context parallel.
- **Vision tower frozen.** mbridge builds the Qwen3.5 VL stack even for
  text-only RL; we set `actor.freeze_vision_tower=True` to skip its
  optimizer / grad bookkeeping.
- **Optimizer state offloaded to CPU.** Required to fit the 4B model with
  long contexts on a single B200 trainer pool. Set via
  `actor.optim.override_optimizer_config.optimizer_cpu_offload=True`.

The full rationale, plus the upstream patches that make this work, lives in
[`docs/b200_qwen35_megatron_bringup.md`](docs/b200_qwen35_megatron_bringup.md).

**Async-mode caveats.**

- The fully-async trainer logs metrics at the end of every grad update *and*
  at every param-sync cycle. Cycle-level keys (e.g. `processing_time/p99`,
  `rollouter/idle_ratio`) only land at sync events; per-step keys (loss,
  reward, log_ppl_diff) land every step.
- `staleness_threshold > 0` produces stale trajectories — they show up
  in the `fully_async/count/stale_trajectory_processed` counter. With
  `partial_rollout=True` + the canonical 0.5 threshold this is well-behaved
  on 4B–27B; we haven't pushed it harder.

## Pointers

- [`docs/b200_qwen35_megatron_bringup.md`](docs/b200_qwen35_megatron_bringup.md) —
  ABI-pinned env (torch 2.10 / TE 2.13 / FA 2.8.3 / mbridge / megatron-core
  0.16.1 / vLLM 0.19.1) and the verl patches required to run Qwen3.5.
- [`ASYNC_EXPERIMENT.md`](ASYNC_EXPERIMENT.md) — async verification protocol
  (what to watch, what to compare, how to tell if sync is broken).
- [`docs/qwen3_4b_inst_acemath_mode1_config.md`](docs/qwen3_4b_inst_acemath_mode1_config.md) —
  Mode 1 reference run hyperparameters.
- [`verl/utils/judge/README.md`](verl/utils/judge/README.md) — LLM-as-judge
  full guide: architecture, dataset format, customization, Cloudflare hosting.
- [`docs/advance/fully_async.md`](docs/advance/fully_async.md) — upstream's
  fully-async design doc; canonical reference for the four operating modes.
