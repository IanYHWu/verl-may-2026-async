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

"""Reward manager that delegates per-sample scoring to a hosted LLM judge.

Drop-in replacement for the rule-based managers — same registry interface
(``@register("llm_judge")``), same ``run_single`` signature, same return
dict shape ``{"reward_score", "reward_extra_info"}`` so the trainer / reward
loop are unchanged.

Reads judge config from ``config.reward.reward_kwargs.judge.*``. To switch
on, set ``reward.reward_manager.name=llm_judge`` and populate the kwargs.
"""

from __future__ import annotations

import inspect
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.judge.client import JudgeClient
from verl.utils.reward_score import math_proof


@register("llm_judge")
class LLMJudgeRewardManager(RewardManagerBase):
    """Score each sample by calling a hosted LLM judge endpoint.

    Owns one ``JudgeClient`` per worker. Decodes the response (and the
    question, preferring ``raw_prompt`` when available) and hands them to
    ``math_proof.compute_score`` together with the rubric (ground truth).
    """

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        compute_score: Any,
        reward_router_address: str | None = None,
        reward_model_tokenizer: Any = None,
    ):
        super().__init__(config, tokenizer, compute_score)
        # If user supplies a custom_reward_function, prefer that; otherwise
        # default to math_proof.compute_score (LLM-judge scoring).
        self.compute_score = compute_score or math_proof.compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        if not self.is_async_reward_score:
            raise ValueError(
                "LLMJudgeRewardManager requires an async compute_score. "
                "Either omit custom_reward_function (uses math_proof.compute_score) "
                "or supply an `async def` reward function."
            )

        judge_cfg = config.reward.get("reward_kwargs", {}).get("judge", None)
        if judge_cfg is None:
            raise ValueError(
                "LLMJudgeRewardManager requires reward.reward_kwargs.judge.* config "
                "(endpoint_url, model, ...)"
            )
        self._judge_cfg = judge_cfg

        self.judge_client = JudgeClient(
            endpoint_url=judge_cfg["endpoint_url"],
            model=judge_cfg["model"],
            api_key_env=judge_cfg.get("api_key_env", "LLM_JUDGE_API_KEY"),
            headers=dict(judge_cfg.get("headers", {}) or {}),
            timeout_s=float(judge_cfg.get("timeout_s", 60.0)),
            max_retries=int(judge_cfg.get("max_retries", 3)),
            retry_backoff_s=float(judge_cfg.get("retry_backoff_s", 2.0)),
            max_concurrency=int(judge_cfg.get("max_concurrency", 32)),
        )

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        question = await self.loop.run_in_executor(
            None,
            lambda: self._extract_question(data_item),
        )

        data_source = data_item.non_tensor_batch.get("data_source")
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}

        cfg = self._judge_cfg
        result = await self.compute_score(
            solution_str=response_str,
            ground_truth=ground_truth,
            question=question,
            judge_client=self.judge_client,
            template_name=cfg.get("prompt_template", math_proof.DEFAULT_TEMPLATE),
            max_score=float(cfg.get("max_score", math_proof.DEFAULT_MAX_SCORE)),
            max_input_tokens=cfg.get("max_input_tokens"),
            max_output_tokens=int(cfg.get("max_output_tokens", math_proof.DEFAULT_MAX_OUTPUT_TOKENS)),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            reasoning_effort=cfg.get("reasoning_effort"),
            thinking_mode=cfg.get("thinking_mode"),
            on_error_score=float(cfg.get("on_error_score", 0.0)),
            tokenizer=self.tokenizer,
            data_source=data_source,
            extra_info=extra_info,
        )

        score = float(result["score"])
        reward_extra_info = {k: v for k, v in result.items() if k != "score"}
        # Keep parity with rule-based managers that always emit ``acc``.
        reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}

    def _extract_question(self, data_item: Any) -> str:
        """Recover the user-visible question from the data item.

        Prefers the chat-format ``raw_prompt`` (available when the dataset
        was loaded with ``data.return_raw_chat=True``). Falls back to
        decoding the prompt token tensor.
        """
        raw_prompt = data_item.non_tensor_batch.get("raw_prompt")
        if raw_prompt is not None:
            try:
                # Concatenate all user-turn contents. Most datasets put the
                # problem in a single user turn; multi-turn chats just join.
                parts = []
                for msg in raw_prompt:
                    role = msg.get("role") if isinstance(msg, dict) else None
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if role == "user" and isinstance(content, str):
                        parts.append(content)
                if parts:
                    return "\n\n".join(parts)
            except Exception:  # noqa: BLE001
                pass

        # Fallback: decode the prompt token tensor.
        prompts = data_item.batch["prompts"]
        attention_mask = data_item.batch["attention_mask"]
        prompt_length = prompts.shape[-1]
        prompt_attn = attention_mask[:prompt_length]
        valid_prompt_ids = prompts[prompt_attn.bool()]
        return self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
