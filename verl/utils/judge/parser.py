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

"""Parse a numeric score from an LLM judge's free-form output.

Strategy, in order:
  1. ``<score>N</score>`` — the preferred tagged form (templates instruct this).
  2. ``Final score: N`` / ``Score: N`` — common natural-language form.
  3. Trailing ``N/M`` fraction (rescaled to ``max_score`` if given).
  4. Last bare number in the text.

Returns ``None`` when nothing parseable is found.
"""

from __future__ import annotations

import re
from typing import Optional

# Tagged: <score>5</score> or <score>5.5</score>. Case-insensitive, multiline.
_RE_SCORE_TAG = re.compile(
    r"<\s*score\s*>\s*([0-9]+(?:\.[0-9]+)?)\s*<\s*/\s*score\s*>",
    re.IGNORECASE,
)

# "Final score: 5", "Score = 5", "score: 5/7"
_RE_LABELED_SCORE = re.compile(
    r"(?:final\s+score|score)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*([0-9]+(?:\.[0-9]+)?))?",
    re.IGNORECASE,
)

# Bare fraction near the end: "5/7"
_RE_FRACTION = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)")

# Last bare number in the text (after stripping).
_RE_TRAILING_NUMBER = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*[.\)]?\s*$")


def parse_score(text: str, max_score: Optional[float] = None) -> Optional[float]:
    """Extract a raw score from ``text``.

    Args:
        text: judge model's free-form output.
        max_score: rubric maximum used to rescale fraction matches like ``5/7``.
            If ``None``, the numerator is returned as-is.

    Returns:
        Raw float score, or ``None`` if no parse.
    """
    if not text:
        return None

    m = _RE_SCORE_TAG.search(text)
    if m:
        return float(m.group(1))

    m = _RE_LABELED_SCORE.search(text)
    if m:
        num = float(m.group(1))
        den = m.group(2)
        if den is not None and float(den) > 0:
            num = (num / float(den)) * (max_score if max_score is not None else float(den))
        return num

    # Search the last 200 chars for a fraction; fractions earlier in the text
    # are likely part of the rubric and not the verdict.
    tail = text[-300:]
    m = _RE_FRACTION.search(tail)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den > 0:
            return (num / den) * (max_score if max_score is not None else den)

    m = _RE_TRAILING_NUMBER.search(text.strip())
    if m:
        return float(m.group(1))

    return None
