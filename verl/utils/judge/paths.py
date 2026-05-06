# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Dotted-path resolution into mixed dict/list structures.

Shared between the reward manager (resolves ``extra_fields`` against
``data_item.non_tensor_batch`` at training time) and the dataset checker
(resolves the same paths against parquet rows ahead of time).
"""

from __future__ import annotations

from typing import Any


def resolve_dotted(root: Any, path: str) -> Any:
    """Walk a dotted path through dict-like / list-like structures.

    Examples:
        >>> resolve_dotted({"reward_model": {"ground_truth": "x"}}, "reward_model.ground_truth")
        'x'
        >>> resolve_dotted({"items": ["a", "b"]}, "items.1")
        'b'

    Raises:
        KeyError / TypeError / IndexError when a segment cannot be walked.
    """
    cur = root
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur[key]
        elif key.isdigit():
            cur = cur[int(key)]
        else:
            cur = cur[key]
    return cur
