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
    engine_config = critic_config.engine
    engine_config.infer_max_token_len_per_gpu = critic_config.ppo_infer_max_token_len_per_gpu
    engine_config.max_token_len_per_gpu = critic_config.ppo_max_token_len_per_gpu

    return {
        "model_type": "value_model",
        "model_config": critic_config.model,
        "engine_config": engine_config,
        "optimizer_config": critic_config.optim,
        "checkpoint_config": critic_config.checkpoint,
    }
