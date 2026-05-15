# Quantized training in verl — getting NVFP4 / FP8 RL working

Notes from getting NVFP4 QAT, FP8 rollout-only, and FP8 E2E running in
`verl-may-2026-async` on B200 + CUDA 12.8. Goal of this doc: capture the
non-obvious bits so the next person doesn't re-derive them.

This is on branch **`feat/quantization-training`**.

---

## TL;DR

| Variant | Status | Run script | Notes |
|---|---|---|---|
| **FP8 rollout-only** | ✅ Working, healthy curves | [`scripts/sample_scripts/qwen3_4b_fp8_rollout_only_acemath_colocate_16k.sh`](../scripts/sample_scripts/qwen3_4b_fp8_rollout_only_acemath_colocate_16k.sh) | BF16 training + FP8 blockwise rollout. ~3 min/step on 8×B200. |
| **FP8 E2E** | ✅ Working with `delayed` recipe | [`scripts/sample_scripts/qwen3_4b_fp8_e2e_acemath_colocate_16k.sh`](../scripts/sample_scripts/qwen3_4b_fp8_e2e_acemath_colocate_16k.sh) | `blockwise` requires CUDA 12.9+ — we're on 12.8, so default `FP8_RECIPE=delayed`. ~3 min/step, ~6% slower than rollout-only. |
| **NVFP4 W4A16 QAT** | ⚠️ Pipeline runs end-to-end, model collapses | [`scripts/sample_scripts/qwen3_4b_nvfp4_qat_colocate_16k.sh`](../scripts/sample_scripts/qwen3_4b_nvfp4_qat_colocate_16k.sh) | All wiring works; on-the-fly W4 quant of Qwen3-4B-Instruct produces degenerate output (16k responses, entropy 0.005, 0% solve rate). Not a verl bug — see [§NVFP4 collapse](#nvfp4-collapse-discussion). |

All three variants share the same scaffolding:
- Qwen3-4B-Instruct-2507 (Qwen3.5 not supported by the QAT-required megatron-bridge)
- acemath_rl_4b_inst_hard (DAPO-formatted), 16k max response
- DAPO judge (1.0 / 0.0 rewards + overlong soft penalty in last 4k)
- Colocate (sync) mode, GRPO, n=8, bsz=64, mbsz=32
- TP=2, ckpt + optimizer + grad offload, recompute = full
- **Token-level TIS** (`rollout_correction.rollout_is=token`, threshold=2.0)

---

## Installs (env: `verl_megatron` on B200)

Done from scratch on top of an existing BF16-training env:

```bash
# 1. ModelOpt — QAT fake-quantization, NVFP4 packing helpers.
#    Peripheral; no torch/cuda conflict.
pip install nvidia-modelopt
# (installs nvidia-modelopt-0.44.0 + pulp + a few small deps)

# 2. Megatron-Bridge — REQUIRED for QAT path. Verl's
#    verl/utils/modelopt/megatron_qat_patch.py imports from
#    megatron.bridge.models.conversion.* and megatron.bridge.models.gpt_provider.
#    Latest pypi release (0.3.1) is INCOMPATIBLE with verl QAT — it dropped
#    the quantization_layer_spec symbol that verl's patch imports. You MUST
#    install the same pinned commit the verl-recipe upstream pins:
pip install --no-deps "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@e940d997d7bdb7810f621f5b32bf70255b5aa2d9"

# 3. Patch the bridge for transformers 5.x compatibility (one-shot).
#    See "transformers 5.x rope_theta" below.
python recipe/qat/patch_megatron_bridge_rope.py
```

Versions in our env (do not change without asking):
- `torch 2.10.0+cu128`, CUDA toolkit 12.8 (system nvcc).
- `transformers 5.5.4`, `vllm 0.19.1`, `megatron-core 0.16.1`,
  `transformer_engine 2.13.0`, `compressed-tensors 0.15.0.1`.
- New, peripheral: `nvidia-modelopt 0.44.0`, `megatron-bridge 0.3.0rc0` (pinned), `pulp`.

`mbridge` (the legacy community bridge from ISEEKYAN, used when
`vanilla_mbridge=True`) and `megatron.bridge` (the NVIDIA one, used when
`vanilla_mbridge=False`) are **different packages** with different top-level
namespaces. They coexist fine — installing one doesn't break the other.

---

## Verl code patches

All on branch `feat/quantization-training`. None of these are version bumps —
they're compatibility patches against installed lib versions.

### 1. `verl/utils/modelopt/quantize.py` — modelopt 0.39+ config format

Old verl code builds the `mtq.quantize` config as a **dict**:
```python
quant_cfg = { **_NVFP4_W4A16_QUANTIZER_CFG, **_default_disabled_quantizer_cfg, **ignore_cfg }
```
But modelopt 0.39+ changed `_default_disabled_quantizer_cfg` to a **list of
QuantizerCfgEntry dicts**, and `mtq.quantize` now expects `quant_cfg` to be a list
in priority order. Symptom: `TypeError: 'list' object is not a mapping`.

Fix: build a list, with the NVFP4 weight-quantizer rule first, then modelopt's
disabled defaults, then ignore patterns. Idempotent if you have an older
modelopt — accepts both dict and list shapes for the default-disabled cfg.

### 2. `verl/utils/modelopt/vllm_modelopt_patch.py` — vLLM 0.19 marlin API drift

Two changes to `_modelopt_dense_process_weights` for vLLM 0.19+:

a. `nvfp4_marlin_process_scales` now returns `(tensor, scale_factor)`, not just a tensor. Symptom: `AttributeError: 'tuple' object has no attribute 'detach'`. Fix: unpack tuple, keep tensor.

b. The forward path (`apply_nvfp4_linear` in `vllm/.../nvfp4_utils.py:176`) reads
`layer.weight_global_scale`, `layer.input_global_scale_inv`, and `layer.alpha`
**regardless of backend**. The verl patch only set `weight_scale_2` and deleted
the old attrs. Symptom: `AttributeError: 'QKVParallelLinear' object has no
attribute 'weight_global_scale'`. Fix: after marlin packing, also set
`weight_global_scale = weight_scale_2_max` (fp32 scalar — must be float32, not
the marlin-processed bf16), `input_global_scale_inv = 1.0` (W4A16: no
activation quant), `alpha = weight_global_scale`.

### 3. `recipe/qat/config/nvfp4_w4a16_megatron.json` (new)

The verl QAT path requires this JSON. It's not in this repo by default — it
lives in the separate `verl-project/verl-recipe` repo. Contents copied verbatim
from there. Same for the FSDP variant `nvfp4_w4a16.json`.

### 4. `recipe/qat/patch_megatron_bridge_rope.py` (new) — transformers 5.x

The pinned megatron-bridge reads `hf_config.rope_theta` directly. transformers
5.x moved it to `hf_config.rope_parameters["rope_theta"]`. Symptom on every
Qwen2/3, Llama, Gemma, GLM, etc. bridge: `AttributeError: 'Qwen3Config' object
has no attribute 'rope_theta'`.

This script idempotently rewrites every `*_bridge.py` in the installed
megatron-bridge package to fall back to `rope_parameters['rope_theta']` when
the direct attribute is missing. Run it once after installing the bridge.
Re-running is a no-op (uses a sentinel check).

### 5. Script changes — required knobs

Verbatim, but worth calling out — every quantized variant **needs** these on top of the
normal verl flags:

| Knob | Why |
|---|---|
| `algorithm.rollout_correction.rollout_is=token` | Token-level TIS. **Without it**, FP8/NVFP4 rollout vs BF16 training drift compounds. We saw length-bloat → 32% clip → reward decay by step 30. With TIS, ppl_ratio stays flat at ~1.006 for 75+ steps. |
| `algorithm.rollout_correction.rollout_is_threshold=2.0` | Standard upper-bound clamp for TIS IS weights. |
| `+reward.reward_kwargs.overlong_buffer_cfg.enable=True` | DAPO overlong soft penalty (linear penalty over the last 4k tokens). Without it the model learns to ramble until it hits the cap. |
| `+actor_rollout_ref.actor.megatron.override_transformer_config.use_arbitrary_attention_mask=False` | **NVFP4 QAT only.** The `quantization_layer_spec` in megatron-bridge sets `use_arbitrary_attention_mask=True` by default when context_parallel_size=1. That arbitrary mask makes TE FlashAttention bail with "No dot product attention backend is available" because FusedAttention is also disabled by env. Setting False keeps FlashAttention live. |
| `actor_rollout_ref.actor.megatron.vanilla_mbridge=False` | **QAT only.** QAT's weight exporter imports from `megatron.bridge.*` (the NVIDIA one), not `mbridge` (the legacy one). For FP8 rollout-only and FP8 E2E, keep `vanilla_mbridge=True` because Qwen3.5 isn't supported by megatron-bridge (only mbridge has Qwen3_5). |

---

## Run-time configuration gotchas

### vLLM NVFP4 backend selection on B200

B200 reports compute capability 10.0, so vLLM auto-picks `FLASHINFER_CUTLASS`
for NVFP4 GEMM. But the verl modelopt patch packs weights in **Marlin** layout.
Mismatch produces:

```
torch._dynamo.exc.Unsupported: Data-dependent assertion failed
  assert weight.dtype == torch.uint8
```

(asserts in `vllm/.../nvfp4_utils.py:226` — these are the non-marlin branch
asserts, hit because cutlass branch expects different weight format).

**Fix**: explicitly pin marlin in the NVFP4 script:
```bash
export VLLM_NVFP4_GEMM_BACKEND=marlin
```

### FP8 E2E: CUDA 12.9 vs 12.8

The docs prescribe `fp8_recipe: "blockwise"`. That requires CUDA 12.9+ /
cuBLAS 12.9+. With CUDA 12.8:
```
AssertionError: FP8 block scaled GEMM requires compute capability 9.0 or
higher and CUDA >= 12.9.
```

**Fix**: the FP8 E2E script defaults to `FP8_RECIPE=delayed` (legacy TE recipe,
works on CUDA 12.x). Slightly lower fidelity than blockwise but no crashes.
Override via env if you're on 12.9+:
```bash
FP8_RECIPE=blockwise bash qwen3_4b_fp8_e2e_acemath_colocate_16k.sh
```

### Megatron-Bridge model support

The pinned bridge (e940d99) supports `Qwen3ForCausalLM` but **not**
`Qwen3_5ForConditionalGeneration`. So `Qwen3.5-4B` cannot be used with QAT —
QAT requires `vanilla_mbridge=False`, which requires `megatron.bridge`, which
requires Qwen3. We use `Qwen3-4B-Instruct-2507` for the NVFP4 variant. FP8
variants stay on `vanilla_mbridge=True` and could use Qwen3.5, but we picked
Qwen3-4B-Instruct uniformly for comparable runs.

### Prompt length filtering

`filter_overlong_prompts=False` (the verl-recipe default) lets prompts >
`max_prompt_length` through. They're left-truncated for the *training* tensor,
but the **agent loop** re-tokenizes via the chat template and the result
sometimes exceeds the budget — then `torch.cat([input.prompt_ids for ...])` at
`agent_loop.py:794` blows up. Symptom: `RuntimeError: Sizes of tensors must
match except in dimension 0. Expected size 2048 but got size 2219 for tensor
number 8`. Hit consistently around step 4 on acemath.

**Fix**: set `filter_overlong_prompts=True`. Costs a few minutes of initial
filter time. All three scripts have this.

### Old Ray processes hold GPU memory

After killing a trainer, residual Ray dashboard/runtime-agent processes
(`ray::DashboardAgent`, `ray::RuntimeEnvAgent`) plus `VLLM::EngineCore` /
`VLLM::Worker_TP*` processes can sit around holding GPU memory (saw ~128 GB
held after a kill). vLLM startup of a subsequent run then fails with:

```
ValueError: Free memory on device cuda:0 (48.48/178.35 GiB) on startup is less
than desired GPU memory utilization (0.7, 124.85 GiB)
```

**Fix before relaunching**:
```bash
pkill -9 -f "VLLM::"
pkill -9 -f "ray::"
pkill -9 -f "verl.trainer.main_ppo"
```
Then `nvidia-smi --query-gpu=memory.used` should be ~0 MiB.

### Hydra `outputs/` symlink

On this cluster, the repo has `outputs` symlinked to `/tmp/ianwu/hydra_outputs`
but the target dir doesn't exist on a fresh node — Hydra crashes immediately
with `FileExistsError: outputs`. **Fix once per node**: `mkdir -p /tmp/ianwu/hydra_outputs`.

---

## NVFP4 collapse — discussion

The pipeline runs end-to-end with no errors. But Qwen3-4B-Instruct under
verl's NVFP4 W4A16 QAT collapses to degenerate output from step 1:
- `actor/entropy: 0.005` (vs healthy ~0.32)
- `response_length/mean: 16384.0` (100% of responses hit the 16k cap)
- `critic/rewards/mean: 0.0` (no answer ever parseable by DAPO judge)

**Root cause**: W4 weight-only quantization of an un-calibrated, instruct-tuned
4B model is too lossy. verl's QAT path:
1. Loads BF16 weights.
2. Wraps Linear layers with `QATLinear` (fake quant during forward).
3. At vLLM init, dynamically computes per-block scales from the BF16 max and
   packs to real NVFP4 for the rollout.

There is no calibration step — scales are dynamic. For a 4B instruct model
whose output structure (chat template, stop-token timing) depends on tight
weight magnitudes, the round-trip BF16 → packed NVFP4 → dequant-back-to-bf16 at
inference is lossy enough to break the model's output format → max-length
rambling → 0 reward → no learning signal.

The verl QAT docs only document tests on `Qwen3-8B-Base` and `Qwen3-30B-A3B-Base`
— both **base** (foundation) models, both **larger** than ours. Both regimes
help: base models don't have a fragile instruction-following format to lose,
and bigger models have more parameters to absorb per-weight quant error.

### Things we tried that didn't fix it

**Dequant init from `OPENZEKA/Qwen3-4B-Instruct-2507-NVFP4`.** The idea: load
that pre-quantized NVFP4 checkpoint, dequantize it back to BF16 (so the BF16
weights are pre-rounded to the W4 grid), use as init. Theory: verl's
on-the-fly W4 packing should then be near-lossless. See
[`recipe/qat/dequantize_nvfp4.py`](../recipe/qat/dequantize_nvfp4.py). The script works
(8 GB BF16 model output) and Megatron loads it fine, but **the resulting run
collapsed identically** (entropy 0.005, all 16k, 0% reward).

Why it didn't help: re-reading the OPENZEKA config, that checkpoint is
**W4A4** (the `input_activations` field is `num_bits: 4`), not W4A16. Its
weights were calibrated *assuming* FP4 activations would also be applied. Our
verl QAT path is W4A16 — W4 weights, BF16 activations. The weight×activation
interaction is wrong; the per-block weight scales OPENZEKA chose are simply
not optimal for a regime where activations stay BF16.

### What would probably work (untried)

1. **PTQ calibration** with `mtq.calibrate` on a small calibration set before
   opening RL — fits dynamic block scales to actual activation stats. Requires
   a small change to verl's QAT init path.
2. **A genuinely W4A16 pre-quantized init** — much harder to find publicly;
   most NVFP4 models on HF are W4A4.
3. **Switch to `Qwen3-4B-Base` or any 8B+ model** — should reduce the
   accuracy hit enough to make W4A16 viable.

For our purposes, NVFP4 is recorded as **"pipeline works, model accuracy
collapses for this base model"** rather than fixed.

---

## RL dynamics observed (FP8 variants)

The first FP8 rollout-only attempt **without** TIS and **without** the overlong
penalty (just the recipe-as-written defaults) showed classic length-bloat:

| step | reward | resp_len | clip_ratio | ppl_ratio |
|---|---|---|---|---|
| 5 | 0.43 | 6320 | 6% | 1.008 |
| 20 | 0.36 | 7550 | 11% | 1.007 |
| 30 | 0.32 | 9749 | **31%** | **1.011** |
| 34 | 0.34 | 9690 | **32%** | **1.017** |

Both lengths and rollout-vs-train mismatch growing. After enabling TIS +
overlong penalty, the same first 30 steps look like:

| step | reward | resp_len | clip_ratio | ppl_ratio |
|---|---|---|---|---|
| 5 | 0.34 | 6063 | 5% | 1.007 |
| 15 | 0.39 | 5344 | 0.2% | 1.006 |
| 25 | 0.35 | 5183 | 0% | 1.006 |
| 32 | **0.45** | 5861 | 0% | 1.006 |
| 75 | **0.39** avg, max 0.48 | ~6500 | <1% | 1.006 |

Stable, slight upward trend in reward, ppl_ratio rock-solid. **TIS + overlong
should be considered mandatory** for FP8 rollout-only on this stack; the
verl-recipe FP8 doc says as much ("with TIS, FP8 rollout aligns with BF16;
obvious accuracy drop when TIS is not enabled") but verl's bare
`ppo_megatron_trainer.yaml` leaves `rollout_is=None`.

FP8 E2E (`delayed` recipe, with TIS + overlong) behaves similarly so far
through 8 steps — slightly higher per-step `rollout_probs_diff` than rollout-only
(~0.020 vs ~0.006, expected because both training *and* inference now have FP8
error), but ppl_ratio is flat at ~1.012.

---

## Performance numbers (8×B200, 16k responses, bsz 64 × n 8)

Steady-state step time (steps 5–10 average):

- **BF16 baseline**: not directly measured this run, but similar workloads
  are ~165 s/step.
- **FP8 rollout-only**: 171 s/step
- **FP8 E2E (delayed)**: 181 s/step (~6% slower than rollout-only)

FP8 E2E being *slightly slower* than rollout-only is expected at this scale:
4B is small enough that the BF16→FP8 conversion overhead doesn't pay back via
faster matmuls, and the `delayed` recipe pays bookkeeping cost (amax history
all-reduce, scale updates) on every linear. FP8 E2E's wins are bigger at 30B+
and substantially bigger with `blockwise` (CUDA 12.9+).

Memory at this scale is dominated by activations + offload bandwidth, not
weight footprint, because we run with `param_offload`, `optimizer_offload`,
`grad_offload`, and `optimizer_cpu_offload` all True. Static weight memory
savings of FP8 E2E vs BF16 are real (~4 GB resident weight, ~36 GB optimizer
states) but mostly hidden by offload. Visible delta on this hardware is
~4–8 GB / GPU.

---

## Reproduction

```bash
# 1. Source the cluster env.
source /home/schmidt/ssci-ianwu/scripts/startup.sh
unset WANDB_API_KEY   # let the script use its bundled wandb_v1 key

# 2. Make sure Hydra outputs target exists (one-shot per node).
mkdir -p /tmp/ianwu/hydra_outputs

# 3. Optional: apply the bridge rope_theta patch (idempotent).
python recipe/qat/patch_megatron_bridge_rope.py

# 4. Run one of:
export TRAIN_FILE=/scratch/schmidt/ssci-aviralku/ianwu/data/acemath_rl_4b_inst_hard_dapofmt_train.parquet

# FP8 rollout only (recommended starter — fastest, healthiest curve):
bash scripts/sample_scripts/qwen3_4b_fp8_rollout_only_acemath_colocate_16k.sh

# FP8 E2E (delayed recipe; default; works on CUDA 12.8):
bash scripts/sample_scripts/qwen3_4b_fp8_e2e_acemath_colocate_16k.sh
# Override to blockwise if on CUDA 12.9+:
FP8_RECIPE=blockwise bash scripts/sample_scripts/qwen3_4b_fp8_e2e_acemath_colocate_16k.sh

# NVFP4 QAT (pipeline runs; model collapses on Qwen3-4B-Instruct — see above):
bash scripts/sample_scripts/qwen3_4b_nvfp4_qat_colocate_16k.sh
```

All three log to `ianwu-cmu/async_verl_debug` on wandb.

---

## File index

| File | Purpose |
|---|---|
| `docs/quantized_rl_learnings.md` | This doc |
| `recipe/qat/config/nvfp4_w4a16_megatron.json` | ModelOpt-format quant config (used by NVFP4 script) |
| `recipe/qat/config/nvfp4_w4a16.json` | compressed-tensors-format quant config (FSDP variant; not used here) |
| `recipe/qat/patch_megatron_bridge_rope.py` | Idempotent patcher for transformers 5.x `rope_theta` compat |
| `recipe/qat/dequantize_nvfp4.py` | Stand-alone tool: dequantize a ModelOpt-NVFP4 HF checkpoint to BF16 |
| `verl/utils/modelopt/quantize.py` | Verl patch: list-format `quant_cfg` for modelopt 0.39+ |
| `verl/utils/modelopt/vllm_modelopt_patch.py` | Verl patch: marlin tuple unpack + missing `weight_global_scale` attrs for vLLM 0.19+ |
| `scripts/sample_scripts/qwen3_4b_nvfp4_qat_colocate_16k.sh` | NVFP4 run script |
| `scripts/sample_scripts/qwen3_4b_fp8_rollout_only_acemath_colocate_16k.sh` | FP8 rollout-only run script |
| `scripts/sample_scripts/qwen3_4b_fp8_e2e_acemath_colocate_16k.sh` | FP8 E2E run script |
