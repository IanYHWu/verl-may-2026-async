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
"""Backport generation ``head_dtype`` support to vLLM releases before 0.26.

This is a runtime backport of vLLM commit 107a03ba (PR #48390).  In
particular, ``head_dtype=float32`` keeps the hidden states and unquantized
LM-head weights in their model dtype and asks CUDA ``torch.mm`` to accumulate
and return the projection in fp32.  It does not create or checkpoint an fp32
copy of the LM-head weights.

The patch is deliberately inert unless a model requests a head dtype different
from its model dtype, normally through ``hf_overrides.head_dtype``.  Apply it in
both the vLLM controller and worker processes: engine-core workers may be
spawned rather than forked, so patching only the controller is insufficient.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_UPSTREAM_COMMIT = "107a03ba63e005ff03424fed9c4e6cf551b98bb2"
_PATCH_MARKER = "_verl_generation_head_dtype_backport"


def apply_vllm_generation_head_dtype_patch() -> bool:
    """Install vLLM's generation ``head_dtype`` implementation if necessary.

    Returns ``True`` when this call installs the backport and ``False`` when it
    was already installed or the running vLLM already has upstream support.
    """

    import torch
    import torch.nn.functional as F
    import vllm.distributed as vllm_distributed

    from vllm.config import get_current_vllm_config
    from vllm.config.model import ModelConfig, _get_head_dtype
    from vllm.lora.layers.logits_processor import LogitsProcessorWithLoRA
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.platforms import current_platform

    if getattr(LogitsProcessor, _PATCH_MARKER, False):
        return False

    # PR #48390 introduced _apply_head.  Feature detection is safer than a
    # version check for source builds and downstream backports.
    if hasattr(LogitsProcessor, "_apply_head"):
        setattr(LogitsProcessor, _PATCH_MARKER, f"upstream:{_UPSTREAM_COMMIT}")
        return False

    original_head_dtype = ModelConfig.head_dtype
    original_logits_init = LogitsProcessor.__init__
    original_lora_init = LogitsProcessorWithLoRA.__init__

    def model_head_dtype(self: Any) -> torch.dtype:
        """Allow generation heads to honor the configured head dtype."""

        head_dtype = _get_head_dtype(
            config=self.hf_config,
            dtype=self.dtype,
            runner_type=self.runner_type,
        )
        if head_dtype not in current_platform.supported_dtypes:
            logger.warning(
                "The current platform does not support [%s] head dtype; "
                "falling back to model dtype [%s].",
                head_dtype,
                self.dtype,
            )
            return self.dtype
        return head_dtype

    ModelConfig.head_dtype = property(  # type: ignore[method-assign]
        model_head_dtype,
        doc=original_head_dtype.fget.__doc__,
    )

    def logits_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_logits_init(self, *args, **kwargs)
        model_config = get_current_vllm_config().model_config
        self.head_dtype = model_config.head_dtype if model_config is not None else None

    def apply_head(
        self: Any,
        lm_head: Any,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project hidden states through the LM head in ``head_dtype``."""

        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
            return lm_head.quant_method.apply(
                lm_head,
                hidden_states,
                bias=embedding_bias,
            )

        if not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError(
                "A head_dtype different from the model dtype is only "
                "supported for an unquantized lm_head."
            )

        if (
            self.head_dtype == torch.float32
            and current_platform.is_cuda()
            and hidden_states.is_cuda
        ):
            # CUDA supports an fp32 output/accumulator for fp16/bf16 inputs.
            # Unlike casting both operands, this does not materialize an fp32
            # copy of the large LM-head weight on every decode step.
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.mm(
                flat,
                lm_head.weight.t(),
                out_dtype=self.head_dtype,
            )
            if embedding_bias is not None:
                logits = logits + embedding_bias.to(self.head_dtype)
            return logits.reshape(*hidden_states.shape[:-1], -1)

        # CPU and non-CUDA backends do not support torch.mm(out_dtype=...).
        # This path also makes the behavior unit-testable without a GPU.
        return F.linear(
            hidden_states.to(self.head_dtype),
            lm_head.weight.to(self.head_dtype),
            embedding_bias.to(self.head_dtype) if embedding_bias is not None else None,
        )

    def get_logits(
        self: Any,
        hidden_states: torch.Tensor,
        lm_head: Any,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(
        self: Any,
        lm_head: Any,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax while honoring ``head_dtype``."""

        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = vllm_distributed.get_tensor_model_parallel_world_size()

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start
        if tp_size == 1:
            return global_indices

        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()],
            dim=-1,
        )
        gathered = vllm_distributed.tensor_model_parallel_all_gather(
            local_pair,
            dim=-1,
        )
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    def lora_init(
        self: Any,
        base_layer: Any,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        sharded_to_full_mapping: list[int] | None,
    ) -> None:
        # The LoRA wrapper bypasses LogitsProcessor._get_logits, so it would
        # silently lose the fp32 projection.  Match upstream and reject it.
        head_dtype = getattr(base_layer, "head_dtype", None)
        if head_dtype is not None and head_dtype != dtype:
            raise ValueError(
                "A head_dtype different from the model dtype (e.g. an fp32 "
                "lm_head) is not yet supported with LoRA."
            )
        original_lora_init(
            self,
            base_layer,
            hidden_size,
            dtype,
            device,
            sharded_to_full_mapping,
        )

    LogitsProcessor.__init__ = logits_init
    LogitsProcessor._apply_head = apply_head
    LogitsProcessor._get_logits = get_logits
    LogitsProcessor.get_top_tokens = get_top_tokens
    LogitsProcessorWithLoRA.__init__ = lora_init
    setattr(LogitsProcessor, _PATCH_MARKER, _UPSTREAM_COMMIT)
    logger.info("Applied vLLM generation head_dtype backport from %s", _UPSTREAM_COMMIT)
    return True


__all__ = ["apply_vllm_generation_head_dtype_patch"]
