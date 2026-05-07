#!/usr/bin/env bash
# B200 (SCHMIDT) Async Mode 4 + Qwen3.5-4B + Fineproofs + LLM judge.
#   - Data: HerrHruby/fineproofs (local parquet copy)
#   - Reward: LLM judge via Cloudflare-tunneled gpt-oss-20b endpoint
#   - 65k max response length
#   - Mode 4: trigger=4, staleness=0.5, partial_rollout=True, require_batches=1
#   - clip_high=0.27 (DAPO), clip_low=0.20
#   - strip_thinking: True (policy's </think> stripped before grading;
#     reward forced to 0 if the policy didn't emit </think>)

set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export NCCL_IB_DISABLE=1

# B200: TE dlopens libcudnn_graph.so.9; ensure cudnn + nccl wheel lib dirs on LD_LIBRARY_PATH.
CUDNN_LOC=$(pip show nvidia-cudnn-cu12 2>/dev/null | grep Location | cut -d' ' -f2)
NCCL_LOC=$(pip show nvidia-nccl-cu12  2>/dev/null | grep Location | cut -d' ' -f2)
export LD_LIBRARY_PATH="${CUDNN_LOC}/nvidia/cudnn/lib:${NCCL_LOC}/nvidia/nccl/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen3.5-4B"}
TRAIN_FILE=${TRAIN_FILE:?TRAIN_FILE must be set (DAPO-format parquet)}
TEST_FILE=${TEST_FILE:-"$TRAIN_FILE"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
n_gpus_rollout=${N_GPUS_ROLLOUT:-4}
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

# Mode 4 cycle math: per cycle = 4 * 1 * 32 = 128 prompts, 4 grad updates.
# N_CYCLES=50 -> 200 grad updates, 6400 total prompts.
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}
train_prompt_bsz=0
gen_prompt_bsz=1
require_batches=${REQUIRE_BATCHES:-1}
trigger_parameter_sync_step=${TRIGGER_SYNC_STEP:-4}
staleness_threshold=${STALENESS_THRESHOLD:-0.5}
partial_rollout=${PARTIAL_ROLLOUT:-True}
N_CYCLES=${N_CYCLES:-50}
total_rollout_steps=$((N_CYCLES * trigger_parameter_sync_step * require_batches * train_prompt_mini_bsz))

# Length budget: 65k response + 2k prompt + slack.
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-65536}
max_model_len=$((max_prompt_length + max_response_length + 256))
max_num_batched_tokens=${max_model_len}

save_freq=${SAVE_FREQ:-25}

PROJECT_NAME=${PROJECT_NAME:-"async_verl_debug"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen35_4b_fineproof_mode4_b200_65k_bsz32_trig4_stale05_partial_clip027_lr1e6_wd001_judge_gptoss20b"}

CKPT_DIR=${CKPT_DIR:-"./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

# Judge config (cloudflare-tunneled gpt-oss-20b).
JUDGE_URL=${JUDGE_URL:-"https://sustained-pop-provinces-government.trycloudflare.com/v1/chat/completions"}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-oss-20b"}

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
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.return_dict=false \
    actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=True \
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
    actor_rollout_ref.actor.clip_ratio_high=0.27 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=true \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
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
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    reward.reward_model.enable=False \
    reward.reward_manager.name=llm_judge \
    +reward.reward_kwargs.judge.endpoint_url="$JUDGE_URL" \
    +reward.reward_kwargs.judge.model="$JUDGE_MODEL" \
    +reward.reward_kwargs.judge.api_key_env=LLM_JUDGE_API_KEY \
    +reward.reward_kwargs.judge.temperature=0.6 \
    +reward.reward_kwargs.judge.top_p=1.0 \
    +reward.reward_kwargs.judge.reasoning_effort=medium \
    +reward.reward_kwargs.judge.max_output_tokens=4096 \
    +reward.reward_kwargs.judge.max_input_tokens=24576 \
    +reward.reward_kwargs.judge.max_score=7 \
    +reward.reward_kwargs.judge.prompt_template=proof_rubric \
    +reward.reward_kwargs.judge.strip_thinking=true \
    +reward.reward_kwargs.judge.on_error_score=0.0 \
    +reward.reward_kwargs.judge.timeout_s=180 \
    +reward.reward_kwargs.judge.max_retries=3 \
    +reward.reward_kwargs.judge.retry_backoff_s=2.0 \
    +reward.reward_kwargs.judge.max_concurrency=16 \
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
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout}
