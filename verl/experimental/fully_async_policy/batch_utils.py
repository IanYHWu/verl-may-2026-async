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

"""Dependency-light batch sizing helpers for fully asynchronous PPO."""

import math


def get_train_batch_divisor(
    *,
    actor_dp_size: int,
    actor_mini_batch_rows: int | None,
    critic_dp_size: int | None = None,
    critic_mini_batch_rows: int | None = None,
) -> int:
    """Return the row-count multiple required by every enabled train worker.

    ``actor_mini_batch_rows=None`` represents actor full-batch mode: only the
    actor data-parallel divisibility requirement applies. Critic training always
    retains its configured PPO mini-batch size.
    """
    divisors = [actor_dp_size]
    if actor_dp_size <= 0:
        raise ValueError(f"actor data-parallel size must be positive, got {actor_dp_size}")
    if actor_mini_batch_rows is not None:
        if actor_mini_batch_rows % actor_dp_size != 0:
            raise ValueError(
                f"actor mini-batch rows ({actor_mini_batch_rows}) must be divisible by "
                f"actor data-parallel size ({actor_dp_size})"
            )
        divisors.append(actor_mini_batch_rows)

    if (critic_dp_size is None) != (critic_mini_batch_rows is None):
        raise ValueError("critic data-parallel size and mini-batch rows must be provided together")
    if critic_dp_size is not None:
        assert critic_mini_batch_rows is not None
        if critic_dp_size <= 0:
            raise ValueError(f"critic data-parallel size must be positive, got {critic_dp_size}")
        if critic_mini_batch_rows % critic_dp_size != 0:
            raise ValueError(
                f"critic mini-batch rows ({critic_mini_batch_rows}) must be divisible by "
                f"critic data-parallel size ({critic_dp_size})"
            )
        divisors.extend((critic_dp_size, critic_mini_batch_rows))

    if any(divisor <= 0 for divisor in divisors):
        raise ValueError(f"batch divisors must be positive, got {divisors}")
    return math.lcm(*divisors)


def get_fully_async_train_steps(
    *,
    total_rollout_steps: int,
    required_samples: int,
    trigger_parameter_sync_step: int,
) -> int:
    """Compute parameter versions before trainer model initialization."""
    divisors = (required_samples, trigger_parameter_sync_step)
    if total_rollout_steps < 0 or any(divisor <= 0 for divisor in divisors):
        raise ValueError(
            "total rollout steps must be non-negative and async training divisors must be positive, "
            f"got total_rollout_steps={total_rollout_steps}, required_samples={required_samples}, "
            f"trigger_parameter_sync_step={trigger_parameter_sync_step}"
        )
    return total_rollout_steps // (required_samples * trigger_parameter_sync_step)


def get_fully_async_optimizer_steps(*, total_rollout_steps: int, required_samples: int) -> int:
    """Compute local actor/critic optimizer updates across all parameter versions."""
    if total_rollout_steps < 0 or required_samples <= 0:
        raise ValueError(
            "total rollout steps must be non-negative and required samples must be positive, "
            f"got total_rollout_steps={total_rollout_steps}, required_samples={required_samples}"
        )
    return total_rollout_steps // required_samples
