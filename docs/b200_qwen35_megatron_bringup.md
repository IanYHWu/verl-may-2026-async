# B200 + Qwen3.5-4B + Megatron: vanilla-RL bring-up

Running vanilla GRPO on Qwen3.5-4B end-to-end with Megatron-Core + mbridge
for training and vLLM for rollout, on 8× B200. This doc summarizes the
env work and the verl patches that were needed to make it work.

## Quick glossary: THD vs BSHD

Attention kernels and Megatron's forward path accept the activation batch
in two layouts:

- **BSHD** — `(batch, seq_len, num_heads, head_dim)`. Padded, fixed-shape
  batch. Simple but wastes compute when sequences in the batch have very
  different lengths.
- **THD** — `(total_tokens, num_heads, head_dim)`. *Packed* variable-length
  sequences concatenated into one flat tensor with `cu_seqlens` boundaries;
  no padding. Efficient for RL rollouts with heterogeneous lengths, but
  not all attention variants can consume it.

verl defaults to THD when `actor.megatron.use_remove_padding=true`. For
Qwen3.5 we had to flip to BSHD because Megatron-Core's `GatedDeltaNet`
(the linear-attention path in the hybrid layers) raises
`NotImplementedError: GDN does not support packed sequence for now.`

## Environment (outside the repo)

### Verified version pins (as of 2026-05-03)

ABI-coupled stack confirmed working end-to-end on 8× B200:

```
torch                    2.10.0
triton                   3.6.0
transformer_engine       2.13.0  (cu12)
flash_attn               2.8.3
megatron-core            0.16.1
vllm                     0.19.1
apex                     0.1
mbridge                  0.15.1  (git: 4cfd6f5eab84ed5424a8202e1a282e6ac584fce5)

nvidia-cuda-runtime-cu12 12.8.90
nvidia-cudnn-cu12        9.10.2.21
nvidia-nccl-cu12         2.27.5
nvidia-cublas-cu12       12.8.4.1
```

mbridge isn't on PyPI as a wheel that matches Megatron-Core 0.16.1 — install
from git pinned to the verified commit:

```bash
pip install git+https://github.com/ISEEKYAN/mbridge.git@4cfd6f5eab84ed5424a8202e1a282e6ac584fce5
```

### Build / install notes

- **`transformer_engine` 2.6 → 2.13.** The prebuilt 2.6 extension was
  compiled against an older torch and failed against torch 2.10 with
  `undefined symbol: _ZNK3c106SymInt6sym_neERKS0_`. Installed prebuilt
  `transformer_engine-2.13.0` + `transformer_engine_cu12-2.13.0` and built
  `transformer_engine_torch-2.13.0` from source against torch 2.10. The
  source build needs `CPATH` / `LIBRARY_PATH` pointing at cudnn + nccl
  headers from the `nvidia-cudnn-cu12` / `nvidia-nccl-cu12` wheels.
- **`flash_attn` 2.8.3 rebuilt from source** against torch 2.10 + sm100.
  The prebuilt wheel was linked against older torch and failed with
  `undefined symbol: c10::cuda::c10_cuda_check_implementation(...)`.
- **`flash-linear-attention` 0.4.2 installed.** Megatron-Core's
  `GatedDeltaNet` imports `fla.modules.l2norm.l2norm` and
  `fla.ops.gated_delta_rule.chunk_gated_delta_rule`.
- **`LD_LIBRARY_PATH` must include the cudnn + nccl wheel lib dirs** at
  runtime. `transformer_engine` dlopens `libcudnn_graph.so.9` during
  `import`; without these paths it dies with
  `OSError: libcudnn_graph.so.9: cannot open shared object file`. The
  `scripts/startup.sh` snippet that sets `CUDNN_PATH` is incomplete —
  it exports the var but doesn't add `$CUDNN_PATH/lib` to
  `LD_LIBRARY_PATH`, and the assignment order is reversed (`CUDNN_PATH`
  is set before `CUDNN_LOC` is defined). Any launcher run outside the
  tmux session that sourced startup also needs this explicitly:
  ```bash
  CUDNN_LOC=$(pip show nvidia-cudnn-cu12 | grep Location | cut -d' ' -f2)
  NCCL_LOC=$(pip show nvidia-nccl-cu12  | grep Location | cut -d' ' -f2)
  export LD_LIBRARY_PATH=$CUDNN_LOC/nvidia/cudnn/lib:$NCCL_LOC/nvidia/nccl/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

## verl patches under `verl/verl/`

- `utils/model.py` — `transformers` 5.x dropped `AutoModelForVision2Seq`;
  alias it to `AutoModelForImageTextToText` so verl's imports still work.
- `models/mcore/registry.py` — deliberately do **not** add Qwen3.5 to the
  `SupportedVLM` enum. This routes Qwen3.5 through `model_forward_gen(False)`
  (the non-VL forward) so the training-side forward can take BSHD. The VL
  forward would force THD packing, which the GDN rejects.
- `models/mcore/model_forward.py` (BSHD branch):
  - Loosened the `assert not vision_model` so a VL-routed forward can
    take BSHD when the batch is text-only.
  - **MRoPE fix**: when `tf_config.mrope_section` is set (Qwen3.5 /
    Qwen3-VL), pass `position_ids=None` to the model — `Qwen3_5VLModel`
    auto-computes the 3-D position ids via `get_rope_index`. verl's 1-D
    `position_ids` would otherwise trip
    `mbridge/models/qwen3_vl/rope_utils.py` with
    `IndexError: too many indices for tensor of dimension 2`.

## Tokenizer compatibility note (transformers 5.x)

Any caller of `tokenizer.apply_chat_template(..., tokenize=True)` should pass
`return_dict=False`. In transformers 5.x the default return became
`BatchEncoding` instead of a flat list of ids, which breaks
`torch.tensor(ids, dtype=torch.long)` with
`TypeError: 'str' object cannot be interpreted as an integer`.

## Agent-loop patch in `verl/verl/experimental/agent_loop/agent_loop.py`

`_compute_position_ids`:

- **Text-only fast path**: when no `image_grid_thw` / `video_grid_thw` is
  present, skip the multimodal RoPE entirely and fall back to
  `compute_position_id_with_mask(attention_mask)`.
- **`mm_token_type_ids` synthesis**: transformers 5.x made
  `mm_token_type_ids` a required positional arg of
  `Qwen3VLModel.get_rope_index`. When the bound method's signature
  includes that parameter, synthesize it from `input_ids` (0 for text,
  vision-start-token-id marks vision tokens) so the call succeeds.

## Required Hydra overrides for the Qwen3.5 hybrid architecture

These are the settings the bring-up validated. They apply on top of the
stock `verl/trainer/config/ppo_megatron_trainer.yaml` config:

- `actor_rollout_ref.actor.strategy=megatron`,
  `actor_rollout_ref.actor.megatron.use_mbridge=true`,
  `actor_rollout_ref.actor.megatron.vanilla_mbridge=true`.
- **`actor_rollout_ref.actor.megatron.use_remove_padding=false`** → forward
  dispatches in BSHD, not THD. The critical flip that lets the GDN run.
- **`actor_rollout_ref.actor.use_dynamic_bsz=false`**,
  **`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`**,
  **`actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false`**,
  **`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1`** —
  BSHD requires static micro batches on both the training step and the
  log-prob recompute.
- **`actor_rollout_ref.actor.freeze_vision_tower=true`** — the vision tower
  is built by mbridge but unused for text-only RL; freezing it keeps the
  optimizer step small and avoids wasted grad computation.
- TP=2, PP=1, CP=1; vLLM rollout `gpu_memory_utilization=0.7`,
  `cudagraph_mode=PIECEWISE`.

See [`scripts/qwen35_4b_32k_colocate.sh`](../scripts/qwen35_4b_32k_colocate.sh)
for a full launcher applying these overrides to vanilla GRPO.

## Net effect

`bash scripts/qwen35_4b_32k_colocate.sh` on 8× B200 runs vanilla GRPO on
Qwen3.5-4B end-to-end: rollout via vLLM (`Qwen3_5ForConditionalGeneration`),
training via Megatron-Core + mbridge in BSHD, Adam distributed optimizer,
weight sync back to vLLM (~2.7 s/step). Steady state ≈ 89 s/step, all 8 GPUs
active, ~73 GiB peak during training and ~158 GiB peak during rollout
(gpu_mem_util=0.7, well within the 180 GiB budget per B200).
