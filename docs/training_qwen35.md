# Training Qwen3.5

This guide covers text-only RL training of dense Qwen3.5 checkpoints such as
`Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, and `Qwen/Qwen3.5-27B` in this fork. The
Megatron + vLLM path is validated end to end. The FSDP settings below follow the
implemented Qwen3.5 and fused-cross-entropy paths, but have not received the
same end-to-end hardware validation in this fork.

The verified Megatron environment is documented in the top-level
[`README.md`](../README.md#verified-version-pins). In particular, this fork pins
Megatron-Core 0.16.1 and cuDNN 9.10.2. Newer Megatron-Core releases have since
added GatedDeltaNet (GDN) packed-sequence support, but that does not make THD
safe in the pinned environment. Keep the settings in this guide unless the
whole stack is deliberately upgraded and revalidated.

## The important architectural constraint

Qwen3.5 interleaves full-attention layers with recurrent GDN linear-attention
layers. Every sample needs an independent GDN state. Flattening multiple
samples into one token stream without passing their sequence boundaries lets
state from one sample leak into the next.

For that reason, both training engines should use padded BSHD inputs in this
fork:

- Megatron maps `use_remove_padding=true` to formal THD packing and
  `use_remove_padding=false` to padded BSHD.
- FSDP uses ordinary BSHD when `use_remove_padding=false`. Its remove-padding
  path flattens examples but does not pass the boundary metadata required by
  Qwen3.5 GDN.

A remove-padding microbatch containing exactly one sample happens to have no
cross-sample boundary, but that is not a robust training configuration.

## Recommended settings

These settings assume that the training-side output projection should be
accumulated and consumed by cross entropy in FP32 without materializing the
full `[tokens, vocabulary]` logits tensor. The hidden states and output-head
weights remain BF16; this is FP32 accumulation, not an FP32-parameter head.

| Setting | What it does | FSDP | Megatron |
| --- | --- | --- | --- |
| Remove padding / packing | Controls whether separate samples are flattened or packed. Disable it so GDN state cannot cross sample boundaries. | `actor_rollout_ref.model.use_remove_padding=false` | `actor_rollout_ref.model.use_remove_padding=false` and `actor_rollout_ref.actor.megatron.use_remove_padding=false` |
| Sequence parallel | Distributes sequence-related activations across ranks. FSDP uses Ulysses; Megatron SP works inside the tensor-parallel group and is not equivalent to Ulysses. | `actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1` | `actor_rollout_ref.actor.megatron.sequence_parallel=true` when TP > 1; automatically disabled at TP = 1 |
| Context parallel | Partitions long-context computation across ranks. This is Megatron's closest analogue to Ulysses, but it is not validated for Qwen3.5 GDN in the pinned stack. | Not applicable | `actor_rollout_ref.actor.megatron.context_parallel_size=1` |
| Fused CE | Avoids materializing the full logits tensor. The Triton implementation retains the projection accumulator and CE statistics in FP32. | `actor_rollout_ref.model.use_fused_kernels=true` | `actor_rollout_ref.model.use_fused_kernels=false`, because stock Megatron fused mode requires THD |
| Fused backend | Selects the implementation behind FSDP's fused-CE switch. `triton` provides the desired FP32-retained tiled computation; the default `torch` backend first writes each logits chunk in the model dtype. | `actor_rollout_ref.model.fused_kernel_options.impl_backend=triton` | Not applicable; the custom path directly invokes the Triton kernel |
| Dynamic batching | Forms variable-sized microbatches according to token counts. It is separate from sequence packing, but is disabled in the conservative, known-good configuration. | `actor_rollout_ref.actor.use_dynamic_bsz=false` | `actor_rollout_ref.actor.use_dynamic_bsz=false` |
| Custom Megatron CE | Enables this fork's BSHD-compatible memory-efficient CE path, avoiding stock Megatron's THD requirement. The variable must reach every trainer worker. | Unset: `unset VERL_MEGATRON_MEM_EFFICIENT_CE` | `export VERL_MEGATRON_MEM_EFFICIENT_CE=1` |

The model-level `use_remove_padding` and `use_fused_kernels` values are the
authoritative settings: the unified worker copies them into the selected
engine. Megatron launchers should nevertheless set the engine-level
`use_remove_padding=false` too, both for clarity and for compatibility with
alternate entry points.

The reference and rollout log-prob configurations normally inherit the
actor's `use_dynamic_bsz` value. If a launcher overrides
`actor_rollout_ref.ref.log_prob_use_dynamic_bsz` or
`actor_rollout_ref.rollout.log_prob_use_dynamic_bsz` explicitly, keep those
false for the conservative baseline as well.

## FSDP configuration

Apply the following overrides to an ordinary `ppo_trainer.yaml` launcher:

```bash
unset VERL_MEGATRON_MEM_EFFICIENT_CE

python -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    ...
```

`ppo_micro_batch_size_per_gpu=1` is a conservative long-context starting
point, not a semantic BSHD requirement. Increase it if padded BSHD memory
permits. Likewise, dynamic batching can be tested independently after the
baseline is stable; enabling it does not by itself concatenate GDN states.

Do not set Ulysses above one. The current FSDP configuration requires
remove-padding when Ulysses is enabled, which conflicts with the safe GDN
layout. Supporting Qwen3.5 with Ulysses requires both correct distributed GDN
recurrence and explicit sample-boundary handling.

## Megatron configuration

Export the custom CE switch before starting Ray or the training process, then
apply these overrides:

```bash
export VERL_MEGATRON_MEM_EFFICIENT_CE=1

python -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    model_engine=megatron \
    actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.actor.megatron.use_remove_padding=false \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=true \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    ...
```

`ppo_megatron_trainer.yaml` remains as a backward-compatible alias, which is
why older sample scripts still use it. New launchers should select
`ppo_trainer.yaml` with `model_engine=megatron`.

Megatron sequence parallel is tied to tensor parallelism. Keep it enabled for
TP > 1; verl automatically disables it at TP = 1. Context parallelism is a
different parallel dimension and should remain at one for the pinned Qwen3.5
stack. This is a validation limit, not a claim that all newer GDN
implementations fundamentally require THD.

Stock verl's Megatron fused path would instead use:

```bash
actor_rollout_ref.model.use_fused_kernels=true
actor_rollout_ref.model.use_remove_padding=true
```

That selects the THD fused forward and is therefore the wrong configuration
for Qwen3.5 with Megatron-Core 0.16.1. When stock fused mode is active,
`VERL_MEGATRON_MEM_EFFICIENT_CE` is deliberately ignored to prevent
double-patching.

## Starting from a bundled launcher

The sample launchers under [`scripts/sample_scripts/`](../scripts/sample_scripts/)
contain the tested Megatron, vLLM, long-context, offload, and fully-async
settings. For example:

```bash
conda activate verl_megatron
export CUDA_HOME=/usr/local/cuda
export VERL_MEGATRON_MEM_EFFICIENT_CE=1

TRAIN_FILE=/path/to/train.parquet \
HF_MODEL_PATH=Qwen/Qwen3.5-4B \
    bash scripts/sample_scripts/qwen35_4b_dapo_async_mode1_40k.sh
```

Useful starting points are:

- `qwen35_4b_32k_colocate.sh`: synchronous colocate GRPO.
- `qwen35_4b_dapo_async_mode1_40k.sh`: fully-async Mode 1, strict on-policy.
- `qwen35_4b_dapo_async_mode4_40k.sh`: Mode 4 with staleness and partial
  rollouts.
- `qwen35_4b_fineproof_async_mode4_judge_65k.sh`: Mode 4 with the LLM judge.
- `qwen35_4b_bf16_acemath_colocate_40k_debug.sh`: BF16 debugging baseline.
- `qwen35_4b_fp8_rollout_only_acemath_colocate_40k.sh` and
  `qwen35_4b_fp8_e2e_acemath_colocate_40k.sh`: quantized variants; read
  [`quantized_rl_learnings.md`](quantized_rl_learnings.md) before using them.

The async entry point is
`python -m verl.experimental.fully_async_policy.fully_async_main`; its
Mode 1/Mode 4 controls and checkpoint behavior are documented in
[`advance/fully_async.md`](advance/fully_async.md).

## Data and tokenizer setup

The bundled launchers expect a DAPO-style Parquet dataset. At minimum, each
row should provide a chat-format `prompt`; the selected reward manager may
require `data_source`, `reward_model.ground_truth`, or additional fields. See
[`preparation/prepare_data.rst`](preparation/prepare_data.rst) and, when using
the judge, [`verl/utils/judge/README.md`](../verl/utils/judge/README.md).

Qwen3.5 is natively available in the pinned Transformers 5.x stack. The sample
launchers retain `actor_rollout_ref.model.trust_remote_code=true` for
compatibility with derived checkpoints, but official Qwen3.5 architecture
loading does not depend on remote model code.

The launchers also explicitly set:

```bash
+data.apply_chat_template_kwargs.return_dict=false
```

That is compatible with the verified environment. Current code normalizes
both a flat token-id list and a `BatchEncoding`, so this override is no longer
a hard Qwen3.5 requirement.

## What FP32 accumulation means

The Triton kernel multiplies BF16 hidden states by BF16 output-head weights,
starts each tiled logits accumulator in FP32, and keeps the max, log-sum-exp,
target log probability, and entropy statistics in FP32. It never writes the
complete logits matrix to memory.

This avoids the important BF16 rounding that occurs when a conventional
output projection writes a BF16 logits tensor before cross entropy. It does
not create FP32 output-head parameters and does not change the model's BF16
parameter or gradient storage.

On FSDP, each rank logically evaluates the full vocabulary head after FSDP
parameter gathering, so the CE operation needs no vocabulary-parallel
collectives. On Megatron, the vocabulary head is tensor-parallel sharded. The
kernel obtains the exact full-vocabulary result with reductions of per-token
statistics. Conceptually it needs the global maximum, softmax denominator,
target-token logit, and entropy numerator. The current implementation obtains
them with three collective calls:

1. an `all_reduce(MAX)` for the global maximum logit;
2. an `all_reduce(SUM)` for the target-token contribution; and
3. one `all_reduce(SUM)` over a combined buffer containing the denominator
   and entropy statistics.

The full `[tokens, vocabulary]` tensor is never all-gathered.

The memory-efficient CE path assumes a scalar temperature and cannot service
features that require the complete logits tensor, such as distillation top-k
selection or `calculate_sum_pi_squared`. The Megatron implementation warns or
falls back when an incompatible option is enabled.

## Text-only and multimodal scope

Dense `Qwen3_5ForConditionalGeneration` is deliberately routed through the
language-model forward for text-only RL so Megatron can use BSHD. Derived
checkpoints that preserve Qwen3.5's GDN/full-attention layer pattern have the
same packing and parallelism constraints, even if they configure the GDN state
dtype as FP32.

Actual image/video training is not covered by this route. The full
vision-language forward and Qwen3.5 MoE variants have not been validated with
the BSHD workaround in this fork; do not infer multimodal support from a
checkpoint merely having `Qwen3_5ForConditionalGeneration` in its config.

## Checkpoint resume

Normal Megatron checkpoints save the actor, optimizer, scheduler, and random
state. Fully-async checkpoints additionally snapshot queued and in-flight
rollout work. Resume with the same parallel topology unless the checkpoint
format explicitly supports resharding.

This branch also carries the Megatron-Core 0.16.1 compatibility fix required
to restore precision-aware optimizer state correctly. Do not remove the
version guard or assume that the backport applies unchanged to another
Megatron-Core release.

## Troubleshooting

- **GDN packed-sequence error:** confirm both model-level and Megatron
  engine-level `use_remove_padding` are false.
- **Megatron still reports stock fused mode:** set
  `actor_rollout_ref.model.use_fused_kernels=false` when using
  `VERL_MEGATRON_MEM_EFFICIENT_CE=1`.
- **FSDP uses the chunked torch backend:** explicitly set
  `fused_kernel_options.impl_backend=triton`.
- **GDN state contamination or unexplained quality loss:** verify that
  remove-padding is false and Ulysses is one.
- **`libcudnn_graph.so.9` import failure:** restore the cuDNN/NCCL wheel paths
  in `LD_LIBRARY_PATH` as shown in the top-level README.
- **Single-sample OOM:** reduce the prompt/response cap, enable full activation
  recomputation, increase TP for Megatron, or use a larger GPU. Reducing TP
  creates fewer parameter shards and generally increases per-rank model
  memory.

Upstream Megatron-Core tracks later GDN THD support in
[NVIDIA/Megatron-LM#5044](https://github.com/NVIDIA/Megatron-LM/issues/5044).
Treat that as an upgrade path requiring a new MCore/cuDNN compatibility and
correctness qualification, not as evidence that packing is safe with this
fork's pinned stack.
