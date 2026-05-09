#!/usr/bin/env bash
# Parameterized async benchmark base. Override env vars from a wrapper:
#   MODEL_HUB_OR_PATH  — Qwen3.5-{4B,9B,27B} hub id or path
#   MODE               — "mode1" or "mode4"
#   TP                 — tensor model parallel size (2 or 4)
#   N_GPUS_ROLLOUT     — rollout GPUs (default 4); rest are trainer
#   PARAM_OFFLOAD      — true/false (default false)
#   GRAD_OFFLOAD       — true/false (default false)
#   OPTIMIZER_OFFLOAD  — true/false (default false; whole DistOpt)
#   ROLLOUT_GMU        — vLLM gpu_memory_utilization (default 0.7)
#   N_CYCLES           — default 4
#   PROJECT_NAME, EXPERIMENT_NAME — wandb tags

set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export NCCL_IB_DISABLE=1

CUDNN_LOC=$(pip show nvidia-cudnn-cu12 2>/dev/null | grep Location | cut -d' ' -f2)
NCCL_LOC=$(pip show nvidia-nccl-cu12  2>/dev/null | grep Location | cut -d' ' -f2)
export LD_LIBRARY_PATH="${CUDNN_LOC}/nvidia/cudnn/lib:${NCCL_LOC}/nvidia/nccl/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

MODEL_HUB_OR_PATH=${MODEL_HUB_OR_PATH:?MODEL_HUB_OR_PATH must be set}
MODE=${MODE:-mode4}
TP=${TP:-2}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-false}
GRAD_OFFLOAD=${GRAD_OFFLOAD:-false}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}
ROLLOUT_GMU=${ROLLOUT_GMU:-0.7}
N_CYCLES=${N_CYCLES:-4}

NGPUS_PER_NODE=8
NNODES=1
n_gpus_training=$((NGPUS_PER_NODE - N_GPUS_ROLLOUT))

n_resp_per_prompt=8
train_prompt_mini_bsz=${MINI_BSZ:-32}
train_prompt_bsz=0
gen_prompt_bsz=1

if [[ "$MODE" == "mode1" ]]; then
    require_batches=2
    trigger_parameter_sync_step=1
    staleness_threshold=0.0
    partial_rollout=False
elif [[ "$MODE" == "mode4" ]]; then
    require_batches=${REQUIRE_BATCHES:-1}
    trigger_parameter_sync_step=4
    staleness_threshold=0.5
    partial_rollout=True
else
    echo "ERROR: MODE must be mode1 or mode4 (got $MODE)" >&2
    exit 1
fi
total_rollout_steps=$((N_CYCLES * trigger_parameter_sync_step * require_batches * train_prompt_mini_bsz))

max_prompt_length=2048
max_response_length=40960
max_model_len=$((max_prompt_length + max_response_length + 256))

PROJECT_NAME=${PROJECT_NAME:-"verl_benchmark"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?EXPERIMENT_NAME must be set}
TRAIN_FILE=${TRAIN_FILE:-"/scratch/schmidt/ssci-aviralku/ianwu/data/acemath_rl_4b_inst_hard_dapofmt_train.parquet"}

CKPT_DIR=${CKPT_DIR:-"/tmp/ianwu/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TRAIN_FILE" \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.return_dict=false \
    actor_rollout_ref.model.path="$MODEL_HUB_OR_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.lr_decay_steps=1000000 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BSZ:-1} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.use_rollout_log_probs=${USE_ROLLOUT_LOG_PROBS:-False} \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=true \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GMU} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_len} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=${BYPASS_MODE:-False} \
    reward.reward_model.enable=False \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${n_gpus_training} \
    rollout.nnodes=${NNODES} \
    rollout.n_gpus_per_node=${N_GPUS_ROLLOUT} \
    rollout.n=${n_resp_per_prompt} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout} \
    async_training.concurrency_multiplier=${CONCURRENCY_MULT:-16}
