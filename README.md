# Verl (May 2026 Fork)

A fork of [verl](https://github.com/verl-project/verl) tracked against May 2026
upstream HEAD, with the patches needed to run **Qwen3.5** RL with the
**fully-async** trainer/rollouter pipeline. Validated end-to-end on **B200
(SM100 / Blackwell)** and **H100 (SM90 / Hopper)** GPUs. Headline additions
on top of upstream:

- **Qwen3.5 hybrid attention (dense + GatedDeltaNet) trains end-to-end** with
  Megatron-Core + mbridge in BSHD layout. The GDN linear-attention path
  in the pinned Megatron-Core 0.16.1 stack rejects packed (THD) sequences,
  so we route dense text-only Qwen3.5 through a non-VL forward and use padded
  BSHD. The complete FSDP/Megatron settings, FP32-accumulating fused-CE paths,
  and architecture rationale are in
  [`docs/training_qwen35.md`](docs/training_qwen35.md).
- **Async (decoupled trainer/rollouter) is the default for long-context RL.**
  We've validated Qwen3.5-4B on `fully_async_policy` Mode 1 (on-policy
  pipeline) at 40k responses and Mode 4 (async stream + partial rollout) at
  40k and 65k. Throughput data — including a Mode 4 vs
  colocate comparison on H100 and a model-size / context / concurrency
  sweep on B200 — is in [`docs/benchmark.md`](docs/benchmark.md).
- **LLM-as-judge reward manager.** A new `llm_judge` reward manager calls a
  hosted OpenAI-compatible chat-completions endpoint (e.g. a Cloudflare
  Worker proxying gpt-oss / qwen / Anthropic) for rubric-based grading of
  proof-style problems. Strips policy `</think>` regions before grading.
  Details in [`verl/utils/judge/README.md`](verl/utils/judge/README.md).

> **Tested with Megatron only.** The FSDP path likely still works (we
> haven't broken it), and its implemented Qwen3.5 settings are documented, but
> none of our changes have been validated against it end to end.
> This fork's compatibility claims apply to its pinned May 2026 stack. For a
> new deployment, compare against current upstream as its Qwen3.5 and
> Megatron-Core support has continued to evolve.

## Install

The B200 / H100 + Megatron + Qwen3.5 stack is non-trivial to assemble. We
ship an end-to-end installer at
[`scripts/install_verl_megatron.sh`](scripts/install_verl_megatron.sh) that
performs every step below in order and stops when an invoked step reports a
failure, so you can correct the environment and re-run it.

```bash
conda create -n verl_megatron python=3.10
conda activate verl_megatron
export CUDA_HOME=/usr/local/cuda      # or wherever your toolkit lives
bash scripts/install_verl_megatron.sh
```

Total wall time on a fresh env is ~30–60 minutes (apex + flash-attn from
source dominate).

### Verified version pins

| Component | Version |
|---|---|
| python | 3.10 |
| CUDA toolkit | 12.8 |
| torch | 2.10.0 |
| triton | 3.6.0 |
| transformers | 5.5.4 |
| transformer_engine | 2.13.0 (cu12, source build) |
| flash_attn | 2.8.3 (sm100 source build for B200) |
| megatron-core | 0.16.1 |
| vllm | 0.19.1 |
| apex | 0.1 (source build) |
| mbridge | 0.15.1 (git: `4cfd6f5e`) |
| flash-linear-attention | 0.4.2 |
| nvidia-cudnn-cu12 | 9.10.2.21 |
| nvidia-nccl-cu12 | 2.27.5 |

### What the installer does

The installer is structured as 10 ordered, best-effort rerunnable steps. If
you'd rather run them by hand, the same steps are reproduced here.

1. **`pip install nvidia-cudnn-cu12`.** TransformerEngine `dlopen`s
   `libcudnn_graph.so.9` at import time, so cudnn must be on disk before
   TE is built.
2. **`USE_MEGATRON=1 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh`.**
   The upstream verl stack installer. It pins older versions of vLLM, TE,
   and Megatron-LM that don't ABI-match torch 2.10 + sm100; we upgrade
   each in subsequent steps.
3. **Discover cudnn + nccl include / lib dirs.** Export `CPATH`,
   `LIBRARY_PATH`, and `LD_LIBRARY_PATH` from
   `pip show nvidia-cudnn-cu12 | grep Location` and the nccl equivalent so
   the source builds in steps 4 and 7 link correctly.
4. **TransformerEngine 2.13 from source.** The prebuilt 2.6 wheel fails
   against torch 2.10 with
   `undefined symbol: _ZNK3c106SymInt6sym_neERKS0_`. We build 2.13 from
   source against the active torch:
   ```
   NVTE_FRAMEWORK=pytorch pip install --no-build-isolation --no-deps --upgrade \
       git+https://github.com/NVIDIA/TransformerEngine.git@v2.13
   ```
5. **Megatron-Core 0.16.1.** Newer than what step 2 pinned.
   ```
   pip install --no-deps --upgrade \
       git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.1
   ```
6. **vLLM 0.19.1.** Upgrade past the older pin.
   ```
   pip install --upgrade vllm==0.19.1
   ```
7. **flash-attn 2.8.3 source build.** The prebuilt wheel was linked
   against an older torch and fails with `undefined symbol:
   c10::cuda::c10_cuda_check_implementation(...)`. The source build runs
   on B200 (sm100) and H100 (sm90).
   ```
   pip install --upgrade --no-build-isolation flash-attn==2.8.3
   ```
8. **apex from source.** CPP + CUDA extensions. Slow.
9. **mbridge pinned commit.** mbridge isn't on PyPI as a wheel that
   matches Megatron-Core 0.16.1 — install from git at `4cfd6f5e`.
10. **`flash-linear-attention==0.4.2`** plus `pip install --no-deps -e .`
    to install verl itself.

### Runtime LD_LIBRARY_PATH

The same `LD_LIBRARY_PATH` that the installer set must be on the env at
*runtime* too — otherwise import fails with
`OSError: libcudnn_graph.so.9: cannot open shared object file`:

```bash
CUDNN_LOC=$(pip show nvidia-cudnn-cu12 | grep Location | cut -d' ' -f2)
NCCL_LOC=$(pip show nvidia-nccl-cu12  | grep Location | cut -d' ' -f2)
export LD_LIBRARY_PATH=$CUDNN_LOC/nvidia/cudnn/lib:$NCCL_LOC/nvidia/nccl/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

The hardware-specific async launchers set this themselves. Other launchers,
including the basic colocate example, expect it in the calling environment.
The installer prints this snippet at the end as a reminder.

### Sanity check

The installer ends with a Python import check. Re-run it manually any time:

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

## Repo layout

```
verl/                       # forked verl framework
verl/utils/judge/           # JudgeClient, parser, templates, README
verl/utils/judge/README.md  # full LLM-judge usage + customization guide
verl/utils/reward_score/math_proof.py   # async compute_score for rubrics
verl/experimental/reward_loop/reward_manager/
  llm_judge.py              # the LLMJudgeRewardManager class
docs/
  training_qwen35.md       # Qwen3.5 FSDP/Megatron settings + constraints
  advance/fully_async.md    # upstream's async-training doc
  benchmark.md              # H100 Mode 4 vs colocate, B200 Mode 4 sweep
  quantized_rl_learnings.md # FP8/NVFP4 bring-up notes and compatibility
scripts/data/
  convert_fineproof_to_dapo.py  # parquet → DAPO chat format converter
scripts/sample_scripts/     # portable launcher templates
  qwen35_4b_32k_colocate.sh                  # GRPO, hybrid engine, 32k
  qwen35_4b_dapo_async_mode1_40k.sh          # Qwen3.5 + DAPO Math, Mode 1
  qwen35_4b_dapo_async_mode4_40k.sh          # Qwen3.5 + DAPO Math, Mode 4
  qwen35_4b_fineproof_async_mode4_judge_65k.sh  # Mode 4 + LLM judge, 65k
  qwen3_4b_inst_acemath_async_mode1.sh       # Qwen3-Inst + AceMath, Mode 1
  qwen3_4b_inst_acemath_async_mode4.sh       # Qwen3-Inst + AceMath, Mode 4
  qwen3_4b_inst_acemath_colocate.sh          # Qwen3-Inst + AceMath, colocate
```

## Running training

All launchers below are thin wrappers around `python -m verl.trainer.main_ppo`
(colocate path) or `python -m verl.experimental.fully_async_policy.fully_async_main`
(async path), with the Qwen3.5-specific Hydra overrides pre-set. See the
script you're running for the full command. Before adapting one, read
[`docs/training_qwen35.md`](docs/training_qwen35.md); in particular, export
`VERL_MEGATRON_MEM_EFFICIENT_CE=1` to enable the BSHD-compatible
FP32-accumulating CE path.

### Colocate (hybrid engine)

Trainer and rollout share the same GPUs via vLLM's hybrid engine.

```bash
# Set HF_MODEL_PATH and TRAIN_FILE (DAPO-format parquet) as env vars,
# or edit the defaults at the top of the script.
VERL_MEGATRON_MEM_EFFICIENT_CE=1 \
TRAIN_FILE=/path/to/train.parquet \
    bash scripts/sample_scripts/qwen35_4b_32k_colocate.sh
```

The reference colocate config: 8 GPUs shared, `hybrid_engine=True`, TP=2,
gpu_memory_utilization=0.7, recompute_granularity=full, optimizer state
offloaded to CPU.

### Fully-async (separated trainer + rollouter)

Trainer and rollouter run on disjoint GPU pools. Three knobs define the
operating mode (see [`docs/advance/fully_async.md`](docs/advance/fully_async.md)):

- `async_training.trigger_parameter_sync_step` — local grad updates per param
  sync. `=1` is on-policy, larger is more off-policy.
- `async_training.staleness_threshold` — maximum proportion of stale samples
  that training may consume; it also expands the rollouter's between-sync
  sample budget. `0.0` refuses stale samples; `0.5` is the validated Mode 4
  setting.
- `async_training.partial_rollout` — interrupt and resume in-flight rollouts
  during param sync (only matters when `staleness_threshold > 0`).

**Mode 1 — on-policy pipeline** (`trigger=1`, `staleness=0`,
`partial_rollout=False`). This is the strict freshness configuration and the
recommended async sanity baseline.

```bash
VERL_MEGATRON_MEM_EFFICIENT_CE=1 \
TRAIN_FILE=/path/to/train.parquet \
    bash scripts/sample_scripts/qwen35_4b_dapo_async_mode1_40k.sh
```

The Qwen3.5 Mode 1 reference config uses 4 trainer + 4 rollout B200s, 40k
responses, GRPO with DAPO clip-higher 0.20/0.28, lr=1e-6, wd=0.01, mbsz=32,
n=8, and 50 cycles → 100 grad updates. The launcher exposes its main sizing
and schedule values as environment variables.

**Mode 4 — async stream pipeline + partial rollout** (`trigger=4`,
`staleness=0.5`, `partial_rollout=True`, `require_batches=1`). Designed to
hide long rollout latency; it tolerates some staleness in exchange for trying
to keep both pools saturated. Performance is workload and
topology dependent: the measured H100 4B run tied colocate within 1%, while
H100 9B favored colocate; the B200 study covers Mode 4 scaling at 4B, 9B, and
27B. See [`docs/benchmark.md`](docs/benchmark.md) instead of assuming a fixed
speedup.

The Qwen3.5 Mode 4 launcher is
`scripts/sample_scripts/qwen35_4b_dapo_async_mode4_40k.sh`. Its defaults can
be changed with `TRIGGER_SYNC_STEP`, `STALENESS_THRESHOLD`,
`PARTIAL_ROLLOUT`, and `REQUIRE_BATCHES` environment variables.

### Health checks during training

The async path emits the standard verl signals plus a few we lean on:

- `rollout_corr/log_ppl_diff` should sit ≲ 5e-4 throughout. If it grows
  toward 1e-2+, the rollouter isn't getting fresh weights — sync is broken.
- `timing_s/param_sync` non-zero per cycle confirms the
  `CheckpointEngineManager` NCCL+IPC path is doing real work.
- `fully_async/processing_time/tp99` and
  `fully_async/rollouter/idle_ratio` expose long-tail rollout latency and
  rollouter under-utilization.
- `critic/score/mean` should be finite and consistent with the configured
  reward scale. Its trend is task-dependent; it is not by itself proof that
  weight synchronization is healthy.

## Reducing memory pressure

Long-context RL on Qwen3.5 burns memory on three fronts: optimizer and master
parameter state (roughly 12 bytes/parameter for FP32 master weights plus FP32
Adam moments before sharding), activations during the trainer's backward pass,
and the rollouter's vLLM KV cache. The knobs below are listed roughly in order
of "free" → "expensive in throughput". Stack them as needed.

### Activation checkpointing (recompute)

The first thing to enable. Trades compute for activation memory by
re-running selected forward layers during backward. The bundled launchers
already set:

```yaml
actor_rollout_ref.actor.megatron.override_transformer_config:
  recompute_granularity: full      # checkpoint all activations between layers
  recompute_method: uniform        # uniform layer split
  recompute_num_layers: 1          # number of layers in each recompute group
```

`full` + `uniform` recompute_method + `recompute_num_layers=1` is the most
fine-grained full-recompute setting: each layer is its own checkpointed unit.
With `uniform`, `recompute_num_layers=2` checkpoints two-layer units; it saves
fewer boundary activations but still recomputes all layers during backward. It
does not halve recompute FLOPs. Use `block` or selective recomputation if the
goal is to recompute only part of the model.

### Optimizer state offload (Adam moments → CPU)

FP32 Adam moments plus master-parameter state are large even after distributed
optimizer sharding: at 4B parameters, 12 bytes/parameter is roughly 24 GB per
rank when sharded two ways. Offloading optimizer work and state to CPU is the
single biggest GPU-memory win for single-trainer-pool runs:

```yaml
actor_rollout_ref.actor.optim:
  override_optimizer_config:
    optimizer_cpu_offload: true              # move Adam state to CPU
    optimizer_offload_fraction: 1            # 0.0–1.0; 1.0 = all of it
    use_precision_aware_optimizer: true      # required by MCore CPU-offload path
    overlap_cpu_optimizer_d2h_h2d: true      # hide PCIe transfer behind compute
```

`overlap_cpu_optimizer_d2h_h2d=true` is critical — without it the H↔D
transfer runs serial with the optimizer step and dominates wall time.

### Param / grad offload (Megatron Distributed Optimizer)

For colocate runs where vLLM and the trainer share GPUs, offloading
parameters and gradients to CPU between rollout and training keeps the
rollouter's KV cache from getting starved. Set on the trainer side:

```yaml
actor_rollout_ref.actor.megatron:
  param_offload: true             # parameters → CPU when not in use
  grad_offload: true              # grad buffers → CPU
  optimizer_offload: true         # whole DistOpt state, not just Adam moments
```

In **separated trainer/rollouter** (Mode 1/4) these are typically `false`
because the trainer pool isn't competing with vLLM for memory. In
**colocate**, set all three to `true`.

### Sequence parallel (TP-side)

Megatron's sequence parallel splits eligible activations along the sequence
dimension within each tensor-parallel group, reducing their per-rank
duplication. It is enabled by default when `TP > 1` and automatically disabled
when TP is 1. The bundled Qwen3.5 launchers use
`tensor_model_parallel_size=2`, so Megatron SP is on by default.

> Sequence parallel is not context parallel. CP is an independent parallel
> dimension that partitions context computation and is the closer Megatron
> analogue to Ulysses. CP > 1 has not been validated for Qwen3.5 in the pinned
> MCore stack, so keep it at 1. See
> [`docs/training_qwen35.md`](docs/training_qwen35.md#recommended-settings).

### vLLM KV cache budget

On the rollouter side, control the fraction of GPU memory vLLM reserves
for its KV cache:

```yaml
actor_rollout_ref.rollout:
  gpu_memory_utilization: 0.7    # leave headroom outside vLLM's KV cache
  enable_chunked_prefill: true    # smaller per-step prefill chunks
  enforce_eager: false            # let vLLM compile cudagraphs
```

For Mode 4 with 65k responses on dedicated B200 rollout GPUs we bump this to
0.85. In colocate mode the trainer shares those GPUs, so keep it ≤ 0.7.

### Reducing micro-batch size

The bundled long-context Qwen3.5 launchers use
`ppo_micro_batch_size_per_gpu=1`, but BSHD does not semantically require a
static batch size of one. If a single sample still does not fit, reduce the
response-length cap, increase Megatron tensor parallelism to create more
parameter shards, enable stronger recomputation/offload, or use a larger GPU.
Reducing tensor parallelism creates fewer shards and generally increases
per-rank model memory.

### Quick recipe by GPU

| Situation | Stack |
|---|---|
| 8× B200 (180 GiB), Qwen3.5-4B, 32k–65k responses | Recompute=full, optimizer_cpu_offload=true, gpu_mem_util=0.7–0.85, async (Mode 4 for 65k) |
| 8× H100 (80 GiB), Qwen3.5-4B, 16k responses | Recompute=full, optimizer_cpu_offload=true, gpu_mem_util=0.7, async Mode 1 (validated reference) |
| Colocate (any GPU), Qwen3.5-4B | All of the above + `param_offload=true`, `grad_offload=true`, `optimizer_offload=true` on the trainer side |

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
      response_skip_special_tokens: false    # preserve structural policy tags
      max_concurrency: 16
      timeout_s: 180
      on_error_score: 0.0
```

By default the judge looks for the *last* `</think>` tag in the policy's
response and only sends what comes after to the grader. If `</think>` is
missing the reward is forced to `on_error_score` (default 0) — we fail
closed rather than grading the raw chain of thought. Set
`response_skip_special_tokens=false` when structural tags are registered as
tokenizer special tokens; otherwise decoding can erase them before the judge
or custom scoring code sees the response.

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

## Custom rollouts

Use this pattern when you want non-vanilla rollout behavior — e.g. a
self-reflection loop where the policy generates an answer, gets a critique,
and revises; or a multi-agent debate; or any other rollout shape that
goes beyond a single `generate_sequences` call. The pattern keeps the
trainer's gradient/sync machinery intact and only swaps the rollout step.

### Recommended layout

Put each recipe in **its own top-level package** at the same level as
`verl/`. Each recipe is self-contained: main entry point, custom trainer,
custom rollout class, configs, scripts.

```
verl/                              # the framework — don't fork it
my_recipe/                         # your recipe
  __init__.py
  main.py                          # entry point — Hydra dispatch
  trainer.py                       # MyRayPPOTrainer(RayPPOTrainer)
  rollout.py                       # MyRollout — the actual custom rollout
  config/
    my_recipe_trainer.yaml         # extends ppo_trainer.yaml
  scripts/
    qwen35_4b_my_recipe.sh         # launcher
```

Why a sibling package, not a subdir of `verl/`: keeps your code separable
from the framework, so you can rebase against upstream verl without
conflicts.

### Five pieces

**1. Custom rollout class** (`my_recipe/rollout.py`).

The verl trainer talks to rollouts through one method:
`async_rollout_manager.generate_sequences(batch) -> batch`. Wrap the
existing `AgentLoopManager` and add your custom logic around its calls.

```python
# my_recipe/rollout.py
from verl import DataProto

class MyRollout:
    """Self-reflection rollout: generate -> critique -> revise."""

    def __init__(self, base_manager, max_turns: int = 3):
        self._base = base_manager   # the original AgentLoopManager
        self.max_turns = max_turns

    def generate_sequences(self, batch: DataProto) -> DataProto:
        current = self._base.generate_sequences(batch)
        for turn in range(self.max_turns - 1):
            current = self._build_reflection_batch(batch, current)
            current = self._base.generate_sequences(current)
        return current

    def _build_reflection_batch(self, original: DataProto, prev: DataProto) -> DataProto:
        # Construct a follow-up prompt that includes the previous response
        # plus a critique header. Implementation-specific.
        ...

    # Forward any other AgentLoopManager methods you need (validate path,
    # checkpoint hooks, etc.) to self._base.
    def __getattr__(self, name):
        return getattr(self._base, name)
```

#### What `generate_sequences` must return

Return a `DataProto` (verl's `{batch, non_tensor_batch, meta_info}`
container) with these core fields. The trainer reads them directly during
reward, log-prob recompute, advantage, and update phases — get the shapes
wrong and downstream collation breaks. `response_mask` is optional only in
the simple single-turn case described below.

`batch.batch` (a `TensorDict`):

| Key | Shape | Meaning |
|---|---|---|
| `prompts` | `(bsz, prompt_length)` | Prompt token ids, **left-padded** (e.g., `[0,0,1,2,3,4]`). |
| `responses` | `(bsz, response_length)` | Response token ids, **right-padded** (e.g., `[5,6,7,8,0,0]`). |
| `input_ids` | `(bsz, prompt_length + response_length)` | Concatenation of `prompts` + `responses`. |
| `attention_mask` | `(bsz, prompt_length + response_length)` | 1s for real tokens, 0s for padding. The prompt portion is left-padded, the response portion is right-padded. |
| `position_ids` | `(bsz, prompt_length + response_length)` *or* `(bsz, 3, prompt_length + response_length)` for MRoPE | Incremental positions over the full sequence; padding positions are 0. |
| `response_mask` | `(bsz, response_length)` | 1 for tokens the policy generated, 0 for tool-response tokens / padding. Optional — the trainer falls back to deriving it from `responses + attention_mask` via `compute_response_mask(batch)` if absent, but emit it yourself if you have tool turns. |

**Optional fields** the trainer uses if present:

| Key | Shape | When |
|---|---|---|
| `rollout_log_probs` | `(bsz, response_length)` | When `actor.use_rollout_log_probs=true` (fully-async default), used as `old_log_prob` for importance sampling. If absent, the trainer recomputes via the actor engine. |
| `routed_experts` | per-expert | MoE routing replay. |
| `teacher_logprobs`, `teacher_ids` | response_length | Distillation paths. |

`batch.non_tensor_batch` should preserve at least the dataset-supplied
fields (`data_source`, `reward_model`, `extra_info`, etc.) so the reward
manager can read them downstream. The agent loop additionally adds:

- `__num_turns__` — number of agent-loop iterations per sample.
- `multi_modal_inputs` — only for VL data; pass through if your rollout
  doesn't change vision tokens.
- `raw_prompt` — chat-format messages, when `data.return_raw_chat=true`.

`batch.meta_info`:

- `timing` — dict of stage timings; trainer pops it into its own timing
  log. Optional but useful.
- `temperature` — the trainer fills this from the rollout configuration before
  log-prob recompute; a custom trainer must preserve the actual sampling
  temperature if it differs.

**Reference implementation:**
`AgentLoopWorker._postprocess` in
`verl/experimental/agent_loop/agent_loop.py` is the canonical place that
builds this shape from raw rollout outputs. If your custom rollout
produces token streams in any other shape, mirror its padding /
concatenation logic.

**2. Custom trainer** (`my_recipe/trainer.py`).

Inherit from `RayPPOTrainer` and override `fit` to point at the custom
rollout. You can either swap the rollout manager at `init_workers` time
(every `generate_sequences` call now goes through your wrapper) **or**
override `fit` directly and call `MyRollout` at the rollout points
explicitly. Pick whichever matches the shape of your change:

```python
# my_recipe/trainer.py
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from .rollout import MyRollout

class MyRayPPOTrainer(RayPPOTrainer):
    # Option A: swap the manager once, get every rollout call wrapped.
    def init_workers(self):
        super().init_workers()
        max_turns = self.config.my_recipe.reflection_max_turns
        self.async_rollout_manager = MyRollout(
            self.async_rollout_manager, max_turns=max_turns
        )

    # Option B: override fit() if you also need to change the *order* of
    # PPO steps (e.g. an extra phase between rollout and reward), or if
    # you want different rollout behavior per call site (train vs val).
    # Copy verl/trainer/ppo/ray_trainer.py:fit() and edit the rollout
    # call(s) — the rest stays identical.
```

Option A is far less code to keep in sync with upstream — prefer it
unless you specifically need to reorder the PPO phases.

**3. Custom main module** (`my_recipe/main.py`).

Mirror `verl/trainer/main_ppo.py`. The framework already accepts a
`task_runner_class` argument to `run_ppo`; subclass `TaskRunner` and
override `run()` to instantiate `MyRayPPOTrainer` instead of the vanilla
`RayPPOTrainer` at the corresponding line.

```python
# my_recipe/main.py
import hydra
import ray
from verl.trainer.main_ppo import run_ppo, TaskRunner
from .trainer import MyRayPPOTrainer


@ray.remote(num_cpus=1)
class MyTaskRunner(TaskRunner):
    """Same as TaskRunner but builds MyRayPPOTrainer."""

    def run(self, config):
        # Copy verl/trainer/main_ppo.py:TaskRunner.run() and replace the
        # `trainer = RayPPOTrainer(...)` line with `trainer = MyRayPPOTrainer(...)`.
        # Everything else (resource pool setup, dataset, sampler, init_workers,
        # fit) stays identical.
        ...


@hydra.main(config_path="config", config_name="my_recipe_trainer", version_base=None)
def main(config):
    run_ppo(config, task_runner_class=MyTaskRunner)


if __name__ == "__main__":
    main()
```

**4. Config** (`my_recipe/config/my_recipe_trainer.yaml`).

Inherit the base Hydra config and add recipe-specific knobs:

```yaml
defaults:
  - /ppo_trainer
  - override /model_engine: megatron
  - _self_

my_recipe:
  reflection_max_turns: 3
```

**5. Launcher** (`my_recipe/scripts/qwen35_4b_my_recipe.sh`).

Mirror the launchers in [`scripts/sample_scripts/`](scripts/sample_scripts/),
but invoke your `main` instead of `verl.trainer.main_ppo`:

```bash
#!/usr/bin/env bash
set -x
# ...env setup, LD_LIBRARY_PATH, etc — copy from sample_scripts/...
python -m my_recipe.main \
    --config-path=$(pwd)/my_recipe/config \
    --config-name='my_recipe_trainer.yaml' \
    +my_recipe.reflection_max_turns=3 \
    actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    data.train_files="$TRAIN_FILE" \
    ... # everything else as in sample_scripts
```

### Async runs

If you're running async (Mode 1/4), the entry point is
`verl.experimental.fully_async_policy.fully_async_main`. The live training
rollout manager belongs to the separate `FullyAsyncRollouter`, not the
trainer, so wrapping the trainer's `async_rollout_manager` does not customize
training rollouts. Provide a compatible manager class and configure its fully
qualified name instead:

```yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_manager_class: my_recipe.rollout.MyFullyAsyncManager
```

The rollouter loads this class and calls its async `create(...)` factory with
the config, fully-async LLM client, and optional reward-loop handles. Subclass
or mirror `FullyAsyncAgentLoopManager` so `generate_sequences_single` and
statistics remain compatible. Trainer-side validation currently creates the
standard `AgentLoopManager`, so validate a custom fully-async rollout through
the rollouter unless that validation path is customized too.

## Limitations

**Tested only with Megatron + vLLM.** The FSDP backend should still work but
nothing in this fork has been validated against it.

### Qwen3.5 layout and precision constraints

The verified Megatron-Core 0.16.1 stack does not support Qwen3.5 GDN with
packed THD sequences. Later upstream MCore work does not change the behavior
of this pinned environment. The safe invariant for both FSDP and Megatron is
`actor_rollout_ref.model.use_remove_padding=false`; Megatron launchers should
also set `actor_rollout_ref.actor.megatron.use_remove_padding=false`.

FSDP obtains memory-efficient FP32-accumulating CE with
`use_fused_kernels=true` and the `triton` backend. Stock Megatron fused mode
requires THD, so this fork instead uses
`VERL_MEGATRON_MEM_EFFICIENT_CE=1` with `use_fused_kernels=false`. These paths
keep BF16 model weights and hidden states; they retain the output-projection
accumulator and CE statistics in FP32 rather than creating an FP32-parameter
head.

Dynamic batching is independent of packing, and BSHD does not intrinsically
require a microbatch size of one. The bundled long-context launchers disable
dynamic batching and use microbatch one as conservative, validated defaults.
FSDP Ulysses must stay at one because the current integration couples it to
remove-padding. Megatron context parallelism above one is not validated for
Qwen3.5 in this stack; Megatron's TP-side sequence parallelism remains usable.

Dense `Qwen3_5ForConditionalGeneration` is deliberately routed through the
non-VL forward for text-only BSHD training. Actual image/video training and
Qwen3.5 MoE variants are outside the validated scope. Transformers 5.x
natively supports the official architecture, and current tokenizer helpers
normalize both flat token lists and `BatchEncoding`; the launchers' explicit
`return_dict=false` remains compatible but is no longer a hard requirement.

See [`docs/training_qwen35.md`](docs/training_qwen35.md) for the complete
recommended-settings table, copyable FSDP and Megatron overrides, CE
collective explanation, checkpoint notes, and troubleshooting.

### Async-mode caveats

- The fully-async trainer logs metrics at the end of every grad update *and*
  at every param-sync cycle. Cycle-level keys such as
  `fully_async/processing_time/tp99` and
  `fully_async/rollouter/idle_ratio` only land at sync events; per-step keys
  such as loss, reward, and `rollout_corr/log_ppl_diff` land every step.
- `staleness_threshold > 0` produces stale trajectories — they show up
  in the `fully_async/count/stale_trajectory_processed` counter. With
  `partial_rollout=True` + the canonical 0.5 threshold this is well-behaved
  on 4B–27B; we haven't pushed it harder.

## Pointers

- [`docs/training_qwen35.md`](docs/training_qwen35.md) — the authoritative
  Qwen3.5 text-training guide: BSHD requirements, FSDP/Megatron settings,
  FP32-accumulating fused CE, parallelism limits, and troubleshooting.
- [`scripts/sample_scripts/`](scripts/sample_scripts/) — portable launcher
  templates for the validated configurations (colocate / Mode 1 / Mode 4 on
  Qwen3-Instruct + AceMath, Qwen3.5 + DAPO Math, Qwen3.5 + Fineproofs with
  LLM judge). Set `HF_MODEL_PATH` and `TRAIN_FILE` and run.
- [`scripts/data/`](scripts/data/) — dataset preprocessing utilities (e.g.,
  the Fineproofs → DAPO chat format converter that uploaded
  [`HerrHruby/fineproofs`](https://huggingface.co/datasets/HerrHruby/fineproofs)).
- [`verl/utils/judge/README.md`](verl/utils/judge/README.md) — LLM-as-judge
  full guide: architecture, dataset format, customization, Cloudflare hosting.
- [`docs/advance/fully_async.md`](docs/advance/fully_async.md) — upstream's
  fully-async design doc; canonical reference for the four operating modes.
