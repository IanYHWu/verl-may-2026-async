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


def get_role_batch_divisor(*, role: str, dp_size: int, mini_batch_rows: int | None) -> int:
    """Return one train role's required row-count multiple.

    Actor and critic batches must be padded independently. Taking an LCM across
    roles can add an entire all-masked optimizer mini-batch to the role with the
    smaller mini-batch, which still advances AdamW and its LR scheduler.

    ``mini_batch_rows=None`` represents full-batch mode, where data-parallel
    divisibility is the only constraint.
    """
    if dp_size <= 0:
        raise ValueError(f"{role} data-parallel size must be positive, got {dp_size}")
    if mini_batch_rows is None:
        return dp_size
    if mini_batch_rows <= 0:
        raise ValueError(f"{role} mini-batch rows must be positive, got {mini_batch_rows}")
    if mini_batch_rows % dp_size != 0:
        raise ValueError(
            f"{role} mini-batch rows ({mini_batch_rows}) must be divisible by {role} data-parallel size ({dp_size})"
        )
    return mini_batch_rows


def get_fully_async_train_steps(
    *,
    total_rollout_steps: int,
    required_samples: int,
    trigger_parameter_sync_step: int,
) -> int:
    """Compute parameter versions before trainer model initialization."""
    if trigger_parameter_sync_step <= 0:
        raise ValueError(
            f"trigger_parameter_sync_step must be positive, trigger_parameter_sync_step={trigger_parameter_sync_step}"
        )
    optimizer_steps = get_fully_async_optimizer_steps(
        total_rollout_steps=total_rollout_steps,
        required_samples=required_samples,
    )
    if optimizer_steps % trigger_parameter_sync_step != 0:
        raise ValueError(
            "fully async training requires a complete final parameter-sync cycle: "
            f"optimizer_steps ({optimizer_steps}) must be divisible by trigger_parameter_sync_step "
            f"({trigger_parameter_sync_step})"
        )
    return optimizer_steps // trigger_parameter_sync_step


def get_fully_async_optimizer_steps(*, total_rollout_steps: int, required_samples: int) -> int:
    """Compute local actor/critic optimizer updates across all parameter versions."""
    if total_rollout_steps < 0 or required_samples <= 0:
        raise ValueError(
            "total rollout steps must be non-negative and required samples must be positive, "
            f"got total_rollout_steps={total_rollout_steps}, required_samples={required_samples}"
        )
    if total_rollout_steps % required_samples != 0:
        raise ValueError(
            "fully async training cannot consume a partial final optimizer batch: "
            f"total_rollout_steps ({total_rollout_steps}) must be divisible by required_samples "
            f"({required_samples})"
        )
    return total_rollout_steps // required_samples
