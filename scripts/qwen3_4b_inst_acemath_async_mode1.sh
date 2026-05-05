#!/usr/bin/env bash
# Async Mode 1 (on-policy pipeline) counterpart to the colocate run, designed
# to be functionally identical settings-wise so the comparison isolates
# colocate vs separated trainer/rollouter.
#
# Mode 1 = trigger_parameter_sync_step=1, staleness_threshold=0,
#          partial_rollout=False. Rollouter generates require_batches
#          * ppo_mini_batch_size prompts per cycle, trainer takes one
#          training step on them, then weight sync.
#
# Same hyperparams as the colocate run:
#   - clip_ratio_high = 0.28, clip_ratio_low = 0.20
#   - top_p = 1.0, temperature = 1.0
#   - max_response_length = 16384
#   - require_batches=2, ppo_mini_batch_size=32, ppo_epochs=1
#     -> 64 prompts collected, 2 mini-batches per training step
#        (1 on-policy + 1 off-policy mini-batch)
#   - rollout.n = 8 -> 64 prompts * 8 = 512 trajectories per step
#   - lr = 1e-6 (constant), weight_decay = 0.01
#   - reward: 1.0 correct / 0.0 wrong, overlong penalty disabled
#
# Differs from colocate only in resource allocation: 4 trainer + 4 rollout H100s
# (vs 8 shared in colocate). Trainer DP=2 x TP=2 (vs DP=4 x TP=2 colocate).

set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export NCCL_IB_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

HF_MODEL_PATH=${HF_MODEL_PATH:-"/project/flame/ianwu/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"}
TRAIN_FILE=${TRAIN_FILE:-"/tmp/ianwu/data/acemath_dapo_format.parquet"}
TEST_FILE=${TEST_FILE:-"$TRAIN_FILE"}  # validation disabled

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
n_gpus_rollout=${N_GPUS_ROLLOUT:-4}
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

# Match colocate batch math: 64 prompts collected, 2 mini-batches of 32
# (1 on-policy + 1 off-policy), ppo_epochs=1.
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}
train_prompt_bsz=0
gen_prompt_bsz=1
require_batches=2
trigger_parameter_sync_step=1
staleness_threshold=0.0
partial_rollout=False
N_CYCLES=${N_CYCLES:-100}
total_rollout_steps=$((N_CYCLES * trigger_parameter_sync_step * require_batches * train_prompt_mini_bsz))

# Length budget.
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
max_model_len=$((max_prompt_length + max_response_length + 256))
max_num_batched_tokens=${max_model_len}

save_freq=${SAVE_FREQ:-50}

PROJECT_NAME=${PROJECT_NAME:-"async_verl_debug"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_4b_inst_acemath_mode1_16k_bsz64_mbsz32_clip028_lr1e6_wd001"}

CKPT_DIR=${CKPT_DIR:-"/tmp/ianwu/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.return_dict=false \
    actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.lr_decay_steps=1000000 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.use_rollout_log_probs=False \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=true \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.cudagraph_mode=FULL_AND_PIECEWISE \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    reward.reward_model.enable=False \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${n_gpus_training} \
    rollout.nnodes=${NNODES} \
    rollout.n_gpus_per_node=${n_gpus_rollout} \
    rollout.n=${n_resp_per_prompt} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    rollout.test_freq=-1 \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout}
