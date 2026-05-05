#!/usr/bin/env bash
# Sample launcher: colocate (hybrid) sync GRPO. 8x B200, Qwen3.5-4B, vLLM + Megatron.
# 32k max response, FULL_AND_PIECEWISE cudagraphs. Compare with
# scripts/qwen35_4b_32k_mode4.sh for the fully-async Mode 4 alternative.
set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1

HF_MODEL_PATH=${HF_MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-4B"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/dapo-math-17k/train.parquet"}
TEST_FILE=${TEST_FILE:-"$HOME/data/dapo-math-17k/train.parquet"}  # unused; val disabled

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Small batch to keep wall-clock tractable with 32k responses.
# Per-step sample budget: train_batch_size * rollout.n = 16 * 8 = 128 trajectories.
# 6 training steps * 128 = 768 trajectories total.
train_prompt_bsz=16
n_resp_per_prompt=8
train_prompt_mini_bsz=16
total_training_steps=6

# 32k response + 1k prompt + vLLM slack.
max_prompt_length=1024
max_response_length=32768
max_model_len=$((max_prompt_length + max_response_length + 256))
max_num_batched_tokens=${max_model_len}

python -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='bench_mode4_vs_colocate_32k' \
    trainer.experiment_name='qwen35_4b_32k_colocate' \
    trainer.total_training_steps=${total_training_steps} \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
