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
    actor_world_size: int,
    actor_mini_batch_rows: int | None,
    critic_world_size: int | None = None,
    critic_mini_batch_rows: int | None = None,
) -> int:
    """Return the row-count multiple required by every enabled train worker.

    ``actor_mini_batch_rows=None`` represents actor full-batch mode: only the
    actor worker-group divisibility requirement applies. Critic training always
    retains its configured PPO mini-batch size.
    """
    divisors = [actor_world_size]
    if actor_mini_batch_rows is not None:
        divisors.append(actor_mini_batch_rows)

    if (critic_world_size is None) != (critic_mini_batch_rows is None):
        raise ValueError("critic world size and mini-batch rows must be provided together")
    if critic_world_size is not None:
        assert critic_mini_batch_rows is not None
        divisors.extend((critic_world_size, critic_mini_batch_rows))

    if any(divisor <= 0 for divisor in divisors):
        raise ValueError(f"batch divisors must be positive, got {divisors}")
    return math.lcm(*divisors)
