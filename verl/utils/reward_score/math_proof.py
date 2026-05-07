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

"""LLM-judge scoring for proof-based math problems with structured rubrics.

Pairs with ``LLMJudgeRewardManager`` (the reward manager owns the ``JudgeClient``
lifecycle and config plumbing; this module is a pure scoring function).

Returns a dict shaped like the rest of verl's score returns
(``{"score": ..., **extra_info}``) so reward managers can flow it through.
The score is normalized to ``[0, 1]``; the unnormalized rubric score is also
returned in ``raw_score`` for inspection.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from verl.utils.judge.client import JudgeClient
from verl.utils.judge.parser import parse_score
from verl.utils.judge.prompt import render_template

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = "proof_rubric"
DEFAULT_MAX_SCORE = 7.0
DEFAULT_MAX_OUTPUT_TOKENS = 512
THINK_CLOSE_TAG = "</think>"


async def compute_score(
    *,
    solution_str: str,
    ground_truth: str,
    question: str,
    judge_client: JudgeClient,
    template_name: str = DEFAULT_TEMPLATE,
    max_score: float = DEFAULT_MAX_SCORE,
    max_input_tokens: Optional[int] = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    reasoning_effort: Optional[str] = None,
    thinking_mode: Optional[bool] = None,
    on_error_score: float = 0.0,
    strip_thinking: bool = True,
    tokenizer: Any = None,
    data_source: Optional[str] = None,
    extra_info: Optional[dict[str, Any]] = None,
    template_fields: Optional[dict[str, Any]] = None,
    **_: Any,
) -> dict[str, Any]:
    """Score a proof solution by asking an LLM judge to walk a rubric.

    Args:
        solution_str: the student's response (decoded from token ids).
        ground_truth: the rubric text (from ``reward_model.ground_truth``).
        question: the original problem statement.
        judge_client: pre-built async HTTP client.
        template_name: short name (looked up in ``utils/judge/templates/``)
            or absolute path to a template file.
        max_score: rubric maximum (e.g., 7 for olympiad rubrics). The returned
            ``score`` is normalized to ``[0, 1]`` against this max.
        max_input_tokens: optional cap on the prompt's token count. When
            ``tokenizer`` is provided and the rendered prompt exceeds this
            budget, the *response* portion is truncated (problem + rubric
            preserved). Soft check; if no tokenizer is given we just log.
        max_output_tokens: forwarded as ``max_tokens`` to the chat API.
        temperature, top_p: judge sampling params. Default to greedy.
        reasoning_effort: gpt-oss style ``reasoning_effort`` (low/medium/high).
        thinking_mode: qwen-style ``enable_thinking``; passed through
            ``extra_body.chat_template_kwargs.enable_thinking``.
        on_error_score: returned (normalized) when the call fails or the
            judge output cannot be parsed.
        strip_thinking: when True (default), look for a final ``</think>`` tag
            in the response and pass only the post-tag text to the judge. If
            no ``</think>`` is found, the reward is **forced to**
            ``on_error_score`` and the judge is **not** called — we fail
            closed rather than grading the raw chain of thought.
        template_fields: optional extra ``name -> value`` pairs forwarded to
            ``render_template``. Use to inject custom sentinels referenced by
            a custom template (e.g., ``<<reference_solution>>``).

    Returns:
        Dict with at least ``score`` (normalized [0, 1]). Extras: ``raw_score``,
        ``judge_text``, ``judge_latency_s``, ``judge_attempts``, ``judge_error``,
        ``had_think_tag``.
    """
    # Strip the thinking section before sending to the judge. We use rfind so
    # that nested or repeated </think> tags resolve to "everything after the
    # *last* one" — the model's final emitted answer.
    had_think_tag: Optional[bool] = None
    judge_input = solution_str
    if strip_thinking:
        idx = solution_str.rfind(THINK_CLOSE_TAG)
        had_think_tag = idx != -1
        if not had_think_tag:
            # Fail closed. No judge call.
            return {
                "score": (float(on_error_score) / float(max_score)) if max_score else 0.0,
                "raw_score": None,
                "judge_text": "",
                "judge_latency_s": 0.0,
                "judge_attempts": 0,
                "judge_error": "missing_think_tag",
                "had_think_tag": False,
            }
        judge_input = solution_str[idx + len(THINK_CLOSE_TAG):].lstrip()

    fields: dict[str, Any] = {
        "problem": question,
        "response": judge_input,
        "rubric": ground_truth,
        "max_score": int(max_score) if float(max_score).is_integer() else max_score,
    }
    if template_fields:
        # template_fields may shadow defaults if user wants — explicit > implicit.
        fields.update(template_fields)

    judge_prompt = render_template(template_name, **fields)

    # Optional input-token guard. We truncate the *response* preferentially —
    # the problem and rubric are required for grading; the response is what
    # may be unbounded long.
    if max_input_tokens is not None and tokenizer is not None:
        ids = tokenizer.encode(judge_prompt, add_special_tokens=False)
        if len(ids) > max_input_tokens:
            judge_prompt = _truncate_response_in_prompt(
                template_name=template_name,
                fields=fields,
                tokenizer=tokenizer,
                budget=max_input_tokens,
            )

    try:
        result = await judge_client.chat_with_semaphore(
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            thinking_mode=thinking_mode,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM judge call failed (data_source=%s): %s", data_source, exc)
        return {
            "score": float(on_error_score) / float(max_score) if max_score else 0.0,
            "raw_score": None,
            "judge_text": "",
            "judge_latency_s": 0.0,
            "judge_attempts": 0,
            "judge_error": str(exc)[:300],
            "had_think_tag": had_think_tag,
        }

    judge_text = result["content"] or ""
    raw = parse_score(judge_text, max_score=max_score)
    if raw is None:
        finish_reason = result.get("finish_reason")
        logger.warning(
            "LLM judge produced unparseable output "
            "(data_source=%s, finish_reason=%s, head=%r)",
            data_source,
            finish_reason,
            judge_text[:200].replace("\n", " "),
        )
        # If the judge ran out of tokens before emitting content, it's a
        # budget problem (raise max_output_tokens), not a model problem.
        err = "truncated_no_content" if finish_reason == "length" and not judge_text else "parse_failure"
        return {
            "score": float(on_error_score) / float(max_score) if max_score else 0.0,
            "raw_score": None,
            "judge_text": judge_text,
            "judge_latency_s": result["latency_s"],
            "judge_attempts": result["attempts"],
            "judge_error": err,
            "had_think_tag": had_think_tag,
        }

    clamped = max(0.0, min(float(raw), float(max_score)))
    normalized = clamped / float(max_score) if max_score else 0.0

    return {
        "score": normalized,
        "raw_score": float(raw),
        "judge_text": judge_text,
        "judge_latency_s": result["latency_s"],
        "judge_attempts": result["attempts"],
        "judge_error": None,
        "had_think_tag": had_think_tag,
    }


def _truncate_response_in_prompt(
    *,
    template_name: str,
    fields: dict[str, Any],
    tokenizer: Any,
    budget: int,
) -> str:
    """Re-render the prompt with the response truncated to fit the token budget.

    Truncates the ``response`` field; problem, rubric, and any user-supplied
    extra fields are preserved.
    """
    overhead_fields = dict(fields)
    overhead_fields["response"] = ""
    overhead = render_template(template_name, **overhead_fields)
    overhead_tokens = len(tokenizer.encode(overhead, add_special_tokens=False))
    response_budget = max(0, budget - overhead_tokens - 32)  # keep some slack
    response_ids = tokenizer.encode(fields["response"], add_special_tokens=False)
    if len(response_ids) > response_budget:
        response_ids = response_ids[:response_budget]
        truncated = tokenizer.decode(response_ids, skip_special_tokens=True)
        truncated = truncated + "\n\n[…response truncated for grader budget…]"
        fields = dict(fields)
        fields["response"] = truncated
    return render_template(template_name, **fields)
