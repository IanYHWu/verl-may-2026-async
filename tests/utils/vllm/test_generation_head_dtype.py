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
"""CPU tests for the vLLM generation head_dtype backport."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.utils.vllm.generation_head_dtype import (
    apply_vllm_generation_head_dtype_patch,
)


@pytest.fixture(scope="module", autouse=True)
def _install_backport():
    apply_vllm_generation_head_dtype_patch()


class _FakeLmHead:
    def __init__(self, weight: torch.Tensor, *, quantized: bool = False):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        self.weight = weight
        self.quant_method = object() if quantized else UnquantizedEmbeddingMethod()
        self.shard_indices = SimpleNamespace(
            num_org_vocab_padding=0,
            org_vocab_start_index=0,
        )


def _processor(vocab_size: int, head_dtype: torch.dtype):
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    # Avoid LogitsProcessor.__init__: production constructs it inside a vLLM
    # config context, while these focused projection tests intentionally do not
    # create an engine or require a GPU.
    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.head_dtype = head_dtype
    processor.org_vocab_size = vocab_size
    processor.scale = 1.0
    processor.soft_cap = None
    processor._gather_logits = lambda logits: logits
    return processor


def test_fp32_projection_preserves_bf16_parameters_and_returns_fp32():
    vocab_size, hidden_size = 64, 16
    processor = _processor(vocab_size, torch.float32)
    hidden_states = torch.randn(4, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
    lm_head = _FakeLmHead(weight)

    logits = processor._get_logits(hidden_states, lm_head, None)

    assert hidden_states.dtype == torch.bfloat16
    assert lm_head.weight.dtype == torch.bfloat16
    assert logits.dtype == torch.float32
    expected = torch.nn.functional.linear(hidden_states.float(), weight.float())
    torch.testing.assert_close(logits, expected)


def test_same_head_and_model_dtype_keeps_vllm_quant_method_path():
    processor = _processor(32, torch.bfloat16)
    hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)
    expected = torch.randn(2, 32, dtype=torch.bfloat16)

    class _RecordingMethod:
        called = False

        def apply(self, layer, inputs, bias=None):
            self.called = True
            assert inputs is hidden_states
            return expected

    method = _RecordingMethod()
    lm_head = SimpleNamespace(weight=None, quant_method=method)

    actual = processor._get_logits(hidden_states, lm_head, None)

    assert method.called
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_different_head_dtype_rejects_quantized_lm_head():
    processor = _processor(64, torch.float32)
    lm_head = _FakeLmHead(
        torch.randn(64, 16, dtype=torch.bfloat16),
        quantized=True,
    )

    with pytest.raises(ValueError, match="unquantized"):
        processor._get_logits(
            torch.randn(4, 16, dtype=torch.bfloat16),
            lm_head,
            None,
        )


def test_get_top_tokens_honors_fp32_projection(monkeypatch):
    import vllm.distributed as vllm_distributed

    vocab_size, hidden_size = 64, 16
    processor = _processor(vocab_size, torch.float32)
    hidden_states = torch.randn(4, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
    lm_head = _FakeLmHead(weight)
    monkeypatch.setattr(
        vllm_distributed,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    top_tokens = processor.get_top_tokens(lm_head, hidden_states)

    expected = torch.nn.functional.linear(
        hidden_states.float(),
        weight.float(),
    ).argmax(dim=-1)
    assert torch.equal(top_tokens, expected)


def test_fp32_head_rejected_with_lora():
    from vllm.lora.layers.logits_processor import LogitsProcessorWithLoRA

    processor = _processor(64, torch.float32)
    with pytest.raises(ValueError, match="not yet supported with LoRA"):
        LogitsProcessorWithLoRA(
            processor,
            hidden_size=16,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            sharded_to_full_mapping=None,
        )


def test_patch_is_idempotent():
    assert apply_vllm_generation_head_dtype_patch() is False


def test_generation_model_head_dtype_override_is_honored():
    from vllm.config.model import ModelConfig

    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(head_dtype="float32"),
        dtype=torch.bfloat16,
        runner_type="generate",
    )

    assert ModelConfig.head_dtype.fget(model_config) == torch.float32


def test_cuda_path_uses_fp32_out_dtype_mm_under_device_guard():
    source = inspect.getsource(apply_vllm_generation_head_dtype_patch)
    guard_start = source.index("self.head_dtype == torch.float32")
    mm_start = source.index("torch.mm(", guard_start)
    fallback_start = source.index("return F.linear(", mm_start)
    guarded_path = source[guard_start:fallback_start]

    assert "and current_platform.is_cuda()" in guarded_path
    assert "and hidden_states.is_cuda" in guarded_path
    assert "out_dtype=self.head_dtype" in guarded_path
    assert guard_start < mm_start < fallback_start


def test_controller_and_worker_install_backport_before_model_construction():
    repo_root = Path(__file__).resolve().parents[3]
    server_source = (
        repo_root
        / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    ).read_text()
    worker_source = (
        repo_root / "verl/workers/rollout/vllm_rollout/utils.py"
    ).read_text()

    assert "async def launch_server" in server_source
    assert "apply_vllm_generation_head_dtype_patch()" in server_source
    worker_new = worker_source.split("def __new__(cls, **kwargs):", maxsplit=1)[1]
    assert worker_new.index("apply_vllm_generation_head_dtype_patch()") < worker_new.index(
        "instance = super().__new__(cls)"
    )
