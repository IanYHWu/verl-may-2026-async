# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Backport non-tensor optimizer-state filtering to Megatron-Core 0.16.1.

Megatron-Core 0.16.1 assumes every per-parameter optimizer-state value is a
tensor. Optimizers can also store booleans and integers, such as ``found_inf``
or ``store_param_remainders``. Those values break distributed checkpoint save
and both precision-aware and conventional restore paths when Megatron applies
tensor-only operations to them.

Upstream fixed all affected paths in NVIDIA/Megatron-LM commit 23dd639cf3de:

* filter non-tensors while collecting state for checkpoint save;
* filter values sent to the precision-aware restore API; and
* build tensor-only source and destination mappings for conventional restore.

This module installs the same behavior on the repository's pinned
Megatron-Core 0.16.1. The patch is process-local, so the checkpoint manager
applies it in each worker that imports the manager.
"""

import inspect
import logging
from functools import update_wrapper

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

_TARGET_VERSION = Version("0.16.1")
_PATCH_MARKER = "__verl_non_tensor_optimizer_state_guard__"


def is_megatron_core_0161(raw_version: object) -> bool:
    """Return whether a version is the 0.16.1 release, including local builds.

    PEP 440 local metadata (for example ``0.16.1+cu12``) describes a rebuild
    of the same public release. Pre, post, and development releases have a
    different public version and are deliberately excluded.
    """
    try:
        return Version(str(raw_version)).public == _TARGET_VERSION.public
    except InvalidVersion:
        return False


def _compact_source(function) -> str:
    try:
        return "".join(inspect.getsource(function).split())
    except (OSError, TypeError):
        return ""


def _has_complete_upstream_guard(getter, setter) -> bool:
    """Recognize the complete upstream save/load fix, not a partial guard."""
    getter_source = _compact_source(getter)
    setter_source = _compact_source(setter)
    return all(
        marker in source
        for source, marker in (
            (
                getter_source,
                "ifnotisinstance(self.optimizer.state[sharded_model_param][k],torch.Tensor):",
            ),
            (getter_source, "ifisinstance(v,torch.Tensor):"),
            (setter_source, "ifnotisinstance(v,torch.Tensor):"),
            (setter_source, "ifisinstance(v,torch.Tensor):"),
            (setter_source, "ifnotisinstance(tensors[key],torch.Tensor):"),
        )
    )


def apply_non_tensor_optimizer_state_guard() -> bool:
    """Ensure Megatron has the complete non-tensor optimizer-state guard.

    Returns ``True`` when the process has an effective complete guard, whether
    supplied by upstream or installed here. Returns ``False`` when Megatron is
    unavailable or the installed version is outside the validated target and
    lacks the upstream guard.

    Raises:
        RuntimeError: Megatron-Core 0.16.1 is present but its expected private
            methods cannot be patched safely.
    """
    try:
        import megatron.core
        import torch
        from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    except ImportError as exc:
        logger.debug("Megatron optimizer compatibility imports unavailable: %s", exc)
        return False

    raw_version = getattr(megatron.core, "__version__", "")
    try:
        installed_version = Version(str(raw_version))
    except InvalidVersion:
        logger.warning("Cannot interpret megatron.core version %r; compatibility guard not applied", raw_version)
        return False

    getter = getattr(DistributedOptimizer, "_get_main_param_and_optimizer_states", None)
    setter = getattr(DistributedOptimizer, "_set_main_param_and_optimizer_states", None)
    if not callable(getter) or not callable(setter):
        message = (
            "Megatron-Core DistributedOptimizer does not expose the expected "
            "checkpoint state methods; cannot install the non-tensor state guard"
        )
        if is_megatron_core_0161(installed_version):
            raise RuntimeError(message)
        logger.warning("%s (version %s)", message, installed_version)
        return False

    if getattr(getter, _PATCH_MARKER, False) and getattr(setter, _PATCH_MARKER, False):
        return True
    if _has_complete_upstream_guard(getter, setter):
        return True

    if not is_megatron_core_0161(installed_version):
        logger.warning(
            "Megatron-Core %s lacks the recognized complete non-tensor optimizer-state guard; "
            "the verl backport is validated only for %s and was not applied",
            installed_version,
            _TARGET_VERSION,
        )
        return False

    def _guarded_get_main_param_and_optimizer_states(self, model_param):
        group_index, group_order = self.model_param_group_index_map[model_param]
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            sharded_model_param = self.optimizer.param_groups[group_index]["params"][group_order]
            tensors = {}
            for k, v in self.optimizer.state[sharded_model_param].items():
                if not isinstance(v, torch.Tensor):
                    continue
                if isinstance(self.optimizer, HybridDeviceOptimizer):
                    tensors[k] = v
                    continue
                tensors[k] = self.optimizer.get_unscaled_state(sharded_model_param, k)
            tensors["param"] = tensors.pop("master_param")
        else:
            main_param = self.optimizer.param_groups[group_index]["params"][group_order]
            optim_state = self.optimizer.state[main_param]
            tensors = {"param": main_param}
            for k, v in optim_state.items():
                if isinstance(v, torch.Tensor):
                    tensors[k] = v
        return tensors

    def _guarded_set_main_param_and_optimizer_states(self, model_param, tensors):
        group_index, group_order = self.model_param_group_index_map[model_param]
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            sharded_model_param = self.optimizer.param_groups[group_index]["params"][group_order]
            for k, v in tensors.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if isinstance(self.optimizer, HybridDeviceOptimizer):
                    if k == "param":
                        k = "master_param"
                    self.optimizer.state[sharded_model_param][k] = v
                    continue
                if k == "param":
                    self.optimizer.set_scaled_state(sharded_model_param, "master_param", v)
                else:
                    self.optimizer.set_scaled_state(sharded_model_param, k, v)
        else:
            main_param = self.optimizer.param_groups[group_index]["params"][group_order]
            optim_state = self.optimizer.state[main_param]
            dst_tensors = {"param": main_param}
            for k, v in optim_state.items():
                if isinstance(v, torch.Tensor):
                    dst_tensors[k] = v
            for key in dst_tensors:
                if not isinstance(tensors[key], torch.Tensor):
                    continue
                dst_tensors[key].copy_(tensors[key])

    update_wrapper(_guarded_get_main_param_and_optimizer_states, getter)
    update_wrapper(_guarded_set_main_param_and_optimizer_states, setter)
    setattr(_guarded_get_main_param_and_optimizer_states, _PATCH_MARKER, True)
    setattr(_guarded_set_main_param_and_optimizer_states, _PATCH_MARKER, True)

    try:
        DistributedOptimizer._get_main_param_and_optimizer_states = _guarded_get_main_param_and_optimizer_states
        DistributedOptimizer._set_main_param_and_optimizer_states = _guarded_set_main_param_and_optimizer_states
    except Exception as exc:
        DistributedOptimizer._get_main_param_and_optimizer_states = getter
        DistributedOptimizer._set_main_param_and_optimizer_states = setter
        raise RuntimeError("Failed to install Megatron non-tensor optimizer-state guard") from exc

    logger.info(
        "Installed complete non-tensor optimizer-state save/load guard for Megatron-Core %s",
        installed_version,
    )
    return True
