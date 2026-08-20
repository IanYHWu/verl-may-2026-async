# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared critic configuration helpers for colocated and separated PPO trainers."""


def prepare_critic_worker_config_kwargs(critic_config) -> dict:
    """Translate a ``CriticConfig`` into ``TrainingWorkerConfig`` keyword arguments.

    The unified model engine owns backend selection. Keeping that translation here
    ensures colocated PPO and resource-separated/fully-async PPO support the same
    value-model backends and configuration schema.
    """
    if not critic_config.use_dynamic_bsz:
        legacy_micro_batch_size = getattr(critic_config, "ppo_micro_batch_size", None)
        if legacy_micro_batch_size is not None:
            raise ValueError(
                "The unified critic TrainingWorker cannot infer a per-GPU micro-batch size from the deprecated "
                f"critic.ppo_micro_batch_size={legacy_micro_batch_size}. Set "
                "critic.ppo_micro_batch_size_per_gpu explicitly."
            )
        if critic_config.ppo_micro_batch_size_per_gpu is None:
            raise ValueError("critic.ppo_micro_batch_size_per_gpu must be set when critic.use_dynamic_bsz is false")
        legacy_forward_batch_size = getattr(critic_config, "forward_micro_batch_size", None)
        forward_batch_size_per_gpu = getattr(critic_config, "forward_micro_batch_size_per_gpu", None)
        if (
            critic_config.ppo_infer_micro_batch_size_per_gpu is None
            and forward_batch_size_per_gpu is None
            and legacy_forward_batch_size is not None
        ):
            raise ValueError(
                "The unified critic TrainingWorker cannot infer a per-GPU inference batch size from the deprecated "
                f"critic.forward_micro_batch_size={legacy_forward_batch_size}. Set "
                "critic.ppo_infer_micro_batch_size_per_gpu or critic.forward_micro_batch_size_per_gpu explicitly."
            )

    engine_config = critic_config.engine
    engine_config.use_dynamic_bsz = critic_config.use_dynamic_bsz
    engine_config.infer_max_token_len_per_gpu = critic_config.ppo_infer_max_token_len_per_gpu
    infer_micro_batch_size_per_gpu = critic_config.ppo_infer_micro_batch_size_per_gpu
    if infer_micro_batch_size_per_gpu is None:
        infer_micro_batch_size_per_gpu = getattr(critic_config, "forward_micro_batch_size_per_gpu", None)
    if infer_micro_batch_size_per_gpu is None:
        infer_micro_batch_size_per_gpu = critic_config.ppo_micro_batch_size_per_gpu
    engine_config.infer_micro_batch_size_per_gpu = infer_micro_batch_size_per_gpu
    engine_config.max_token_len_per_gpu = critic_config.ppo_max_token_len_per_gpu
    engine_config.micro_batch_size_per_gpu = critic_config.ppo_micro_batch_size_per_gpu

    return {
        "model_type": "value_model",
        "model_config": critic_config.model,
        "engine_config": engine_config,
        "optimizer_config": critic_config.optim,
        "checkpoint_config": critic_config.checkpoint,
    }
