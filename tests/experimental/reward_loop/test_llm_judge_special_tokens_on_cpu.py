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

import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl.experimental.reward_loop.reward_manager.llm_judge import LLMJudgeRewardManager


class _ResponseTokenizer:
    def __init__(self):
        self.skip_special_tokens_calls = []

    def decode(self, token_ids, *, skip_special_tokens):
        del token_ids
        self.skip_special_tokens_calls.append(skip_special_tokens)
        if skip_special_tokens:
            return "answer"
        return "<summary>answer</summary>"


class _SingleItemData:
    def __init__(self):
        self.item = SimpleNamespace(
            batch={
                "responses": torch.tensor([10, 11, 12]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
            non_tensor_batch={
                "data_source": "test",
                "reward_model": {"ground_truth": "rubric"},
                "raw_prompt": [{"role": "user", "content": "question"}],
            },
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.item


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("judge_cfg", "expected_skip", "expected_response"),
    [
        ({}, True, "answer"),
        ({"response_skip_special_tokens": False}, False, "<summary>answer</summary>"),
    ],
)
async def test_policy_response_special_token_decoding(judge_cfg, expected_skip, expected_response):
    seen = {}

    async def compute_score(**kwargs):
        seen.update(kwargs)
        return {"score": 1.0}

    tokenizer = _ResponseTokenizer()
    manager = object.__new__(LLMJudgeRewardManager)
    manager._judge_cfg = judge_cfg
    manager._extra_fields_paths = {}
    manager.tokenizer = tokenizer
    manager.compute_score = compute_score
    manager.judge_client = object()
    manager.loop = asyncio.get_running_loop()

    result = await manager.run_single(_SingleItemData())

    assert seen["solution_str"] == expected_response
    assert tokenizer.skip_special_tokens_calls == [expected_skip]
    assert result["reward_score"] == 1.0
