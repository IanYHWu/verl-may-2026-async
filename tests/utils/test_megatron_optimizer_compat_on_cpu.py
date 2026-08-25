"""Regression tests for Megatron's non-tensor optimizer-state backport."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from verl.utils import megatron_optimizer_compat as compat


class _FakeHybridDeviceOptimizer:
    pass


class _PrecisionAwareOptimizer:
    def __init__(self, sharded_param):
        self.param_groups = [{"params": [sharded_param]}]
        self.state = {
            sharded_param: {
                "master_param": torch.zeros(2),
                "exp_avg": torch.zeros(2),
                "exp_avg_sq": torch.zeros(2),
                "found_inf": False,
            }
        }
        self.get_calls = []
        self.set_calls = []

    def get_unscaled_state(self, sharded_param, key):
        value = self.state[sharded_param][key]
        _ = value.dtype
        self.get_calls.append(key)
        return value

    def set_scaled_state(self, sharded_param, key, value):
        _ = value.dtype
        self.set_calls.append((key, value.clone()))


def _vulnerable_distributed_optimizer_class():
    class VulnerableDistributedOptimizer:
        def _get_main_param_and_optimizer_states(self, model_param):
            group_index, group_order = self.model_param_group_index_map[model_param]
            if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                sharded_model_param = self.optimizer.param_groups[group_index]["params"][group_order]
                tensors = {}
                for k in self.optimizer.state[sharded_model_param]:
                    if isinstance(self.optimizer, _FakeHybridDeviceOptimizer):
                        tensors[k] = self.optimizer.state[sharded_model_param][k]
                        continue
                    tensors[k] = self.optimizer.get_unscaled_state(sharded_model_param, k)
                tensors["param"] = tensors.pop("master_param")
            else:
                main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                optim_state = self.optimizer.state[main_param]
                tensors = {"param": main_param, **optim_state}
            return tensors

        def _set_main_param_and_optimizer_states(self, model_param, tensors):
            group_index, group_order = self.model_param_group_index_map[model_param]
            if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                sharded_model_param = self.optimizer.param_groups[group_index]["params"][group_order]
                for k, v in tensors.items():
                    if isinstance(self.optimizer, _FakeHybridDeviceOptimizer):
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
                dst_tensors = {"param": main_param, **optim_state}
                for key in dst_tensors:
                    dst_tensors[key].copy_(tensors[key])

    return VulnerableDistributedOptimizer


def _upstream_fixed_distributed_optimizer_class():
    class UpstreamFixedDistributedOptimizer:
        def _get_main_param_and_optimizer_states(self, model_param):
            sharded_model_param = model_param
            tensors = {}
            for k in self.optimizer.state[sharded_model_param]:
                if not isinstance(self.optimizer.state[sharded_model_param][k], torch.Tensor):
                    continue
                tensors[k] = self.optimizer.state[sharded_model_param][k]
            optim_state = self.optimizer.state[model_param]
            for k, v in optim_state.items():
                if isinstance(v, torch.Tensor):
                    tensors[k] = v
            return tensors

        def _set_main_param_and_optimizer_states(self, model_param, tensors):
            for k, v in tensors.items():
                if not isinstance(v, torch.Tensor):
                    continue
            optim_state = self.optimizer.state[model_param]
            for k, v in optim_state.items():
                if isinstance(v, torch.Tensor):
                    pass
            for key in optim_state:
                if not isinstance(tensors[key], torch.Tensor):
                    continue

    return UpstreamFixedDistributedOptimizer


def _install_fake_megatron(monkeypatch, distributed_optimizer_class, version="0.16.1"):
    megatron_module = ModuleType("megatron")
    core_module = ModuleType("megatron.core")
    optimizer_module = ModuleType("megatron.core.optimizer")
    distrib_module = ModuleType("megatron.core.optimizer.distrib_optimizer")
    cpu_offloading_module = ModuleType("megatron.core.optimizer.cpu_offloading")

    megatron_module.__path__ = []
    core_module.__path__ = []
    optimizer_module.__path__ = []
    core_module.__version__ = version
    distrib_module.DistributedOptimizer = distributed_optimizer_class
    cpu_offloading_module.HybridDeviceOptimizer = _FakeHybridDeviceOptimizer

    megatron_module.core = core_module
    core_module.optimizer = optimizer_module
    optimizer_module.distrib_optimizer = distrib_module
    optimizer_module.cpu_offloading = cpu_offloading_module

    modules = {
        "megatron": megatron_module,
        "megatron.core": core_module,
        "megatron.core.optimizer": optimizer_module,
        "megatron.core.optimizer.distrib_optimizer": distrib_module,
        "megatron.core.optimizer.cpu_offloading": cpu_offloading_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _distributed_optimizer_instance(distributed_optimizer_class, optimizer, model_param, precision_aware):
    instance = distributed_optimizer_class()
    instance.model_param_group_index_map = {model_param: (0, 0)}
    instance.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware,
    )
    instance.optimizer = optimizer
    return instance


def test_precision_aware_save_and_load_filter_non_tensors(monkeypatch):
    distributed_optimizer_class = _vulnerable_distributed_optimizer_class()
    _install_fake_megatron(monkeypatch, distributed_optimizer_class)
    original_getter = distributed_optimizer_class._get_main_param_and_optimizer_states

    assert compat.apply_non_tensor_optimizer_state_guard() is True
    assert distributed_optimizer_class._get_main_param_and_optimizer_states is not original_getter

    model_param = object()
    sharded_param = object()
    optimizer = _PrecisionAwareOptimizer(sharded_param)
    instance = _distributed_optimizer_instance(distributed_optimizer_class, optimizer, model_param, True)

    saved = instance._get_main_param_and_optimizer_states(model_param)
    assert set(saved) == {"param", "exp_avg", "exp_avg_sq"}
    assert optimizer.get_calls == ["master_param", "exp_avg", "exp_avg_sq"]

    instance._set_main_param_and_optimizer_states(
        model_param,
        {
            "param": torch.ones(2),
            "exp_avg": torch.ones(2),
            "exp_avg_sq": torch.ones(2),
            "found_inf": False,
            "step_count": 200,
        },
    )
    assert [key for key, _ in optimizer.set_calls] == ["master_param", "exp_avg", "exp_avg_sq"]


def test_conventional_save_and_load_filter_non_tensors(monkeypatch):
    distributed_optimizer_class = _vulnerable_distributed_optimizer_class()
    _install_fake_megatron(monkeypatch, distributed_optimizer_class)
    assert compat.apply_non_tensor_optimizer_state_guard() is True

    model_param = object()
    main_param = torch.zeros(2)
    exp_avg = torch.zeros(2)
    optimizer = SimpleNamespace(
        param_groups=[{"params": [main_param]}],
        state={main_param: {"exp_avg": exp_avg, "found_inf": False}},
    )
    instance = _distributed_optimizer_instance(distributed_optimizer_class, optimizer, model_param, False)

    saved = instance._get_main_param_and_optimizer_states(model_param)
    assert set(saved) == {"param", "exp_avg"}

    instance._set_main_param_and_optimizer_states(
        model_param,
        {"param": torch.ones(2), "exp_avg": torch.full((2,), 2.0), "found_inf": True},
    )
    torch.testing.assert_close(main_param, torch.ones(2))
    torch.testing.assert_close(exp_avg, torch.full((2,), 2.0))
    assert optimizer.state[main_param]["found_inf"] is False


def test_repeated_application_is_idempotent(monkeypatch):
    distributed_optimizer_class = _vulnerable_distributed_optimizer_class()
    _install_fake_megatron(monkeypatch, distributed_optimizer_class)

    assert compat.apply_non_tensor_optimizer_state_guard() is True
    first_getter = distributed_optimizer_class._get_main_param_and_optimizer_states
    first_setter = distributed_optimizer_class._set_main_param_and_optimizer_states
    assert compat.apply_non_tensor_optimizer_state_guard() is True
    assert distributed_optimizer_class._get_main_param_and_optimizer_states is first_getter
    assert distributed_optimizer_class._set_main_param_and_optimizer_states is first_setter


def test_local_build_of_vulnerable_release_is_patched(monkeypatch):
    distributed_optimizer_class = _vulnerable_distributed_optimizer_class()
    _install_fake_megatron(monkeypatch, distributed_optimizer_class, version="0.16.1+cu12")
    original_getter = distributed_optimizer_class._get_main_param_and_optimizer_states

    assert compat.apply_non_tensor_optimizer_state_guard() is True
    assert distributed_optimizer_class._get_main_param_and_optimizer_states is not original_getter


@pytest.mark.parametrize(
    ("raw_version", "expected"),
    [
        ("0.16.1", True),
        ("0.16.1+cu12", True),
        ("0.16.1+nv.1", True),
        ("0.16.1rc1", False),
        ("0.16.1.post1", False),
        ("0.16.1.dev1", False),
        ("not-a-version", False),
    ],
)
def test_vulnerable_version_detection(raw_version, expected):
    assert compat.is_megatron_core_0161(raw_version) is expected


def test_complete_upstream_guard_is_left_unchanged(monkeypatch):
    distributed_optimizer_class = _upstream_fixed_distributed_optimizer_class()
    _install_fake_megatron(monkeypatch, distributed_optimizer_class, version="0.17.0")
    original_getter = distributed_optimizer_class._get_main_param_and_optimizer_states
    original_setter = distributed_optimizer_class._set_main_param_and_optimizer_states

    assert compat.apply_non_tensor_optimizer_state_guard() is True
    assert compat.apply_non_tensor_optimizer_state_guard() is True
    assert distributed_optimizer_class._get_main_param_and_optimizer_states is original_getter
    assert distributed_optimizer_class._set_main_param_and_optimizer_states is original_setter


def test_unguarded_non_target_version_is_not_patched(monkeypatch, caplog):
    distributed_optimizer_class = _vulnerable_distributed_optimizer_class()
    _install_fake_megatron(monkeypatch, distributed_optimizer_class, version="0.15.0")
    original_getter = distributed_optimizer_class._get_main_param_and_optimizer_states

    with caplog.at_level("WARNING"):
        assert compat.apply_non_tensor_optimizer_state_guard() is False
    assert distributed_optimizer_class._get_main_param_and_optimizer_states is original_getter
    assert "validated only for 0.16.1" in caplog.text


def test_missing_megatron_returns_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "megatron", None)
    for name in tuple(sys.modules):
        if name.startswith("megatron."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    assert compat.apply_non_tensor_optimizer_state_guard() is False


def test_vulnerable_version_with_missing_method_raises(monkeypatch):
    class MissingSetter:
        def _get_main_param_and_optimizer_states(self, model_param):
            return {"param": model_param}

    _install_fake_megatron(monkeypatch, MissingSetter)

    with pytest.raises(RuntimeError, match="expected checkpoint state methods"):
        compat.apply_non_tensor_optimizer_state_guard()
