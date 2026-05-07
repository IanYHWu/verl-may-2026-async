# Verl (May 2026 Fork)

A fork of [verl](https://github.com/volcengine/verl) tracked against May 2026
upstream HEAD, with the patches needed to run **Qwen3.5** RL with the
**fully-async** trainer/rollouter pipeline. Validated end-to-end on **B200
(SM100 / Blackwell)** and **H100 (SM90 / Hopper)** GPUs. Headline additions
on top of upstream:

- **Qwen3.5 hybrid attention (dense + GatedDeltaNet) trains end-to-end** with
  Megatron-Core + mbridge in BSHD layout. The GDN linear-attention path
  rejects packed (THD) sequences, so we route Qwen3.5 through a non-VL
  forward and pad statically. Architecture and patch rationale in
  [Qwen3.5 limitations](#qwen35-limitations--patch-rationale) below.
- **Async (decoupled trainer/rollouter) is the default for long-context RL.**
  We've validated Qwen3.5-4B on `fully_async_policy` Mode 1 (on-policy
  pipeline) and Mode 4 (async stream + partial rollout) end-to-end at 40k
  and 65k response lengths.
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

The B200 / H100 + Megatron + Qwen3.5 stack is non-trivial to assemble. We
ship an end-to-end installer at
[`scripts/install_verl_megatron.sh`](scripts/install_verl_megatron.sh) that
performs every step below in order and aborts on the first failure so you
can re-run from where it stopped.

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
| torch | 2.10.0 |
| triton | 3.6.0 |
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

The installer is structured as 10 ordered, idempotent-on-rerun steps. If
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

The launchers in `scripts/sample_scripts/` set this themselves at the top
of each file. The installer prints this snippet at the end as a reminder.

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
  advance/fully_async.md    # upstream's async-training doc
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
script you're running for the full command.

### Colocate (hybrid engine)

Trainer and rollout share the same GPUs via vLLM's hybrid engine.

```bash
# Set HF_MODEL_PATH and TRAIN_FILE (DAPO-format parquet) as env vars,
# or edit the defaults at the top of the script.
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
- `async_training.staleness_threshold` — fraction of stale (older-version)
  trajectories the rollouter is allowed to ship. `0.0` is strict on-policy,
  `0.5` is the validated Mode 4 default.
- `async_training.partial_rollout` — interrupt and resume in-flight rollouts
  during param sync (only matters when `staleness_threshold > 0`).

**Mode 1 — on-policy pipeline** (`trigger=1`, `staleness=0`,
`partial_rollout=False`). Cleanest reward signal; matches colocate within
~5pp at matched step counts. Use as the async sanity baseline.

```bash
TRAIN_FILE=/path/to/train.parquet \
    bash scripts/sample_scripts/qwen3_4b_inst_acemath_async_mode1.sh
```

The Mode 1 reference config (Qwen3-4B-Instruct on AceMath DAPO, 4 trainer +
4 rollout, 16k responses, GRPO with DAPO clip-higher 0.20/0.28, lr=1e-6,
wd=0.01, mbsz=32, n=8, 100 cycles → 200 grad updates) is set inside the
launcher itself.

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

## Reducing memory pressure

Long-context RL on Qwen3.5 burns memory on three fronts: optimizer state
(Adam moments, ~12 bytes/param), activations during the trainer's backward
pass, and the rollouter's vLLM KV cache. The knobs below are listed roughly
in order of "free" → "expensive in throughput". Stack them as needed.

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
aggressive setting — every layer is re-run during backward. For shorter
contexts where you have memory headroom, `recompute_num_layers=2` (half
the recompute cost, slightly more activation memory) is a reasonable
intermediate.

### Optimizer state offload (Adam moments → CPU)

Adam's first/second moment buffers double the parameter footprint. With
distributed-Adam they're sharded, but at 4B params still ~24 GiB sharded
2-way. Offloading them to CPU is the single biggest GPU-memory win for
single-trainer-pool runs:

```yaml
actor_rollout_ref.actor.optim:
  override_optimizer_config:
    optimizer_cpu_offload: true              # move Adam state to CPU
    optimizer_offload_fraction: 1            # 0.0–1.0; 1.0 = all of it
    use_precision_aware_optimizer: true      # fp32 master, bf16 grads
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

Megatron's sequence parallel splits the activations along the sequence
dimension within each tensor-parallel group, cutting per-rank activation
memory by `TP×`. It's enabled automatically when `TP > 1` and the model
opts in; you'll see `sequence_parallel=True` in the printed
`Qwen3_5VLTransformerConfig`. The bundled launchers all use
`tensor_model_parallel_size=2`, so SP is on by default.

> Sequence parallel ≠ context parallel. CP would split the sequence
> *across* TP groups (not within them), but it requires THD packing,
> which Qwen3.5's GDN rejects. CP must stay at 1 — see
> [Qwen3.5 limitations](#qwen35-limitations--patch-rationale).

### vLLM KV cache budget

On the rollouter side, control the fraction of GPU memory vLLM reserves
for its KV cache:

```yaml
actor_rollout_ref.rollout:
  gpu_memory_utilization: 0.7    # 0.7 leaves ~30% for the trainer / Megatron
  enable_chunked_prefill: true    # smaller per-step prefill chunks
  enforce_eager: false            # let vLLM compile cudagraphs
```

For Mode 4 with 65k responses on B200 we bump this to 0.85; for colocate
where the trainer also lives on the rollout GPUs, keep it ≤ 0.7.

### Reducing micro-batch size

Qwen3.5 already requires `ppo_micro_batch_size_per_gpu=1` because BSHD
forces static batches; you can't go lower. If a single prompt's longest
response still doesn't fit, the only knobs are dropping the response
length cap, dropping `tensor_model_parallel_size` (more shards), or moving
to a bigger GPU.

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

### Qwen3.5 limitations & patch rationale

Megatron-Core's `GatedDeltaNet` (the linear-attention path in Qwen3.5's
hybrid attention layers) does not accept packed sequences:
`NotImplementedError: GDN does not support packed sequence for now`.
This cascades into the constraints below.

#### THD vs BSHD

verl defaults to **THD** activations
(`(total_tokens, num_heads, head_dim)` — packed variable-length sequences
with `cu_seqlens` boundaries) when `actor.megatron.use_remove_padding=true`.
THD is efficient but not all attention variants consume it. Qwen3.5 forces
**BSHD** (`(batch, seq_len, num_heads, head_dim)` — padded fixed-shape) so
the GDN layer can run.

#### Required Hydra overrides for Qwen3.5

These apply on top of the stock `verl/trainer/config/ppo_megatron_trainer.yaml`
config. The bundled launchers in `scripts/sample_scripts/` already set them.

| Override | Reason |
|---|---|
| `actor.megatron.use_remove_padding=false` | BSHD forward (GDN refuses THD). |
| `model.use_remove_padding=false` | The fully-async path reads from `model.use_remove_padding` (separate from `actor.megatron.use_remove_padding`); both must be set. |
| `actor.megatron.use_mbridge=true` | mbridge owns the Megatron ↔ HF Qwen3.5 weight bridge. |
| `actor.megatron.vanilla_mbridge=true` | Routes through the non-VL converter. |
| `actor.use_dynamic_bsz=false` | BSHD requires static micro batches. |
| `actor.ppo_micro_batch_size_per_gpu=1` | Same; pad to longest in the batch. |
| `rollout.log_prob_use_dynamic_bsz=false` | Same on the log-prob recompute path. |
| `rollout.log_prob_micro_batch_size_per_gpu=1` | Same. |
| `actor.freeze_vision_tower=true` | mbridge builds the VL stack even for text-only RL; freezing skips its optimizer / grad bookkeeping. |
| `actor.optim.override_optimizer_config.optimizer_cpu_offload=true` | Required to fit 4B with long contexts on a single trainer pool. |
| `tensor_model_parallel_size=2`, `pipeline_model_parallel_size=1` | Validated. |
| `context_parallel_size=1` | **CP is not supported** — see below. |
| `model.trust_remote_code=true` | Qwen3.5 config has custom Python. |

#### No context parallel

Context parallel (CP) in Megatron requires THD packing. Since Qwen3.5
forces BSHD, `context_parallel_size` must stay at 1. Sequence parallel (the
TP-side variant) is unaffected and remains enabled by default.

#### In-tree patches under `verl/`

These are already in this fork; you don't need to apply them by hand.

- **`verl/utils/model.py`** — `transformers` 5.x dropped
  `AutoModelForVision2Seq`; aliased to `AutoModelForImageTextToText` so verl
  imports still work.
- **`verl/models/mcore/registry.py`** — Qwen3.5 dense
  (`Qwen3_5ForConditionalGeneration`) is deliberately **not** in
  `SupportedVLM`. That routes it through `model_forward_gen(False)` (the
  non-VL forward) so the training-side dispatch can take BSHD. The VL
  forward would force THD, which the GDN rejects.
- **`verl/models/mcore/model_forward.py`** (BSHD branch) —
  - Loosened `assert not vision_model` so a VL-routed forward can take BSHD
    when the batch is text-only.
  - **MRoPE fix**: when `tf_config.mrope_section` is set
    (Qwen3.5 / Qwen3-VL), pass `position_ids=None` so `Qwen3_5VLModel`
    auto-computes 3-D position ids via `get_rope_index`. verl's 1-D
    `position_ids` would otherwise trip
    `mbridge/.../rope_utils.py` with
    `IndexError: too many indices for tensor of dimension 2`.
- **`verl/experimental/agent_loop/agent_loop.py`** `_compute_position_ids` —
  - Text-only fast path: when no `image_grid_thw` / `video_grid_thw` is
    present, skip multimodal RoPE and fall back to
    `compute_position_id_with_mask(attention_mask)`.
  - **`mm_token_type_ids` synthesis**: transformers 5.x made it a required
    positional arg of `Qwen3VLModel.get_rope_index`. The patch synthesizes
    it from `input_ids` (0 for text, vision-start-token-id for vision
    tokens) so the call succeeds.

#### Tokenizer compatibility (transformers 5.x)

Any caller of `tokenizer.apply_chat_template(..., tokenize=True)` should
pass `return_dict=False`. transformers 5.x changed the default return to a
`BatchEncoding` instead of a flat list of ids, breaking
`torch.tensor(ids, dtype=torch.long)` with
`TypeError: 'str' object cannot be interpreted as an integer`. The
launchers append `+data.apply_chat_template_kwargs.return_dict=false` for
this reason.

### Async-mode caveats

- The fully-async trainer logs metrics at the end of every grad update *and*
  at every param-sync cycle. Cycle-level keys (e.g. `processing_time/p99`,
  `rollouter/idle_ratio`) only land at sync events; per-step keys (loss,
  reward, log_ppl_diff) land every step.
- `staleness_threshold > 0` produces stale trajectories — they show up
  in the `fully_async/count/stale_trajectory_processed` counter. With
  `partial_rollout=True` + the canonical 0.5 threshold this is well-behaved
  on 4B–27B; we haven't pushed it harder.

## Pointers

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
