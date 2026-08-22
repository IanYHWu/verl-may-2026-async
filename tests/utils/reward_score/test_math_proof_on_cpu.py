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

from verl.utils.reward_score.math_proof import _truncate_response_in_prompt


class _StructuralTokenTokenizer:
    special_tokens = {"<summary>", "</summary>"}

    def __init__(self):
        self.decode_skip_special_tokens = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return text.split()

    def decode(self, token_ids, *, skip_special_tokens):
        self.decode_skip_special_tokens.append(skip_special_tokens)
        if skip_special_tokens:
            token_ids = [token for token in token_ids if token not in self.special_tokens]
        return " ".join(token_ids)


def test_truncation_preserves_already_decoded_structural_tokens(tmp_path):
    template = tmp_path / "judge_template.txt"
    template.write_text("Problem Rubric <<response>>")
    tokenizer = _StructuralTokenTokenizer()
    response = "<summary> kept </summary> " + " ".join(f"tail-{i}" for i in range(20))

    prompt = _truncate_response_in_prompt(
        template_name=str(template),
        fields={"response": response},
        tokenizer=tokenizer,
        budget=39,
    )

    assert "<summary> kept </summary>" in prompt
    assert "tail-19" not in prompt
    assert "[…response truncated for grader budget…]" in prompt
    assert tokenizer.decode_skip_special_tokens == [False]
