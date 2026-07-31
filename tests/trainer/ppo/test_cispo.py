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

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.config import RolloutCorrectionConfig
from verl.trainer.ppo.core_algos import (
    compute_policy_loss_bypass_mode,
    compute_policy_loss_cispo,
    get_policy_loss_fn,
)
from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
from verl.utils.config import _validate_policy_loss_rollout_correction
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def _actor_config(loss_mode: str) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=10.0,
        clip_ratio_high=0.2,
        policy_loss=PolicyLossConfig(loss_mode=loss_mode),
    )


def test_cispo_matches_detached_clipped_is_objective():
    config = _actor_config("cispo")
    old_log_prob = torch.tensor([[-2.0, -2.0, -2.0]])
    log_prob = torch.tensor([[-1.5, -3.0, -1.0]], requires_grad=True)
    advantages = torch.tensor([[2.0, -3.0, 100.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])

    loss, metrics = compute_policy_loss_cispo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )
    loss.backward()

    expected_weights = torch.tensor([[1.2, torch.exp(torch.tensor(-1.0)), 1.2]])
    expected_loss = (-expected_weights * advantages * log_prob.detach() * response_mask).sum() / response_mask.sum()
    expected_grad = -expected_weights * advantages * response_mask / response_mask.sum()

    torch.testing.assert_close(loss.detach(), expected_loss)
    torch.testing.assert_close(log_prob.grad, expected_grad)
    assert log_prob.grad[0, 0] != 0  # CISPO retains the upper-clipped token's gradient.
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.5)


def test_cispo_applies_decoupled_rollout_correction_weights():
    config = _actor_config("cispo")
    old_log_prob = torch.tensor([[-2.0, -2.0]])
    log_prob = torch.tensor([[-1.5, -3.0]], requires_grad=True)
    advantages = torch.tensor([[2.0, -3.0]])
    response_mask = torch.ones_like(advantages)
    rollout_is_weights = torch.tensor([[0.5, 2.0]])

    loss, _ = compute_policy_loss_cispo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    loss.backward()

    clipped_ratio = torch.tensor([[1.2, torch.exp(torch.tensor(-1.0))]])
    expected_grad = -clipped_ratio * advantages * rollout_is_weights / response_mask.sum()
    torch.testing.assert_close(log_prob.grad, expected_grad)


@pytest.mark.parametrize("loss_mode", ["vanilla", "cispo"])
def test_bypass_preserves_selected_policy_loss(loss_mode):
    rollout_log_probs = torch.tensor([[-1.0, -2.0]])
    batch = DataProto(
        batch=TensorDict(
            {"rollout_log_probs": rollout_log_probs},
            batch_size=[1],
        )
    )
    policy_loss_config = OmegaConf.structured(PolicyLossConfig(loss_mode=loss_mode))

    apply_bypass_mode(
        batch=batch,
        rollout_corr_config=RolloutCorrectionConfig(bypass_mode=True),
        policy_loss_config=policy_loss_config,
    )

    torch.testing.assert_close(batch.batch["old_log_probs"], rollout_log_probs)
    assert policy_loss_config.loss_mode == loss_mode
    assert get_policy_loss_fn(policy_loss_config.loss_mode) is get_policy_loss_fn(loss_mode)


@pytest.mark.parametrize(
    "rollout_corr_config",
    [
        RolloutCorrectionConfig.bypass_pg_is(),
        RolloutCorrectionConfig.bypass_ppo_clip_geo_rs(),
    ],
)
def test_legacy_bypass_presets_are_resolved_before_worker_serialization(rollout_corr_config):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "loss_mode": "vanilla",
                        "rollout_correction": {},
                    }
                }
            },
            "algorithm": {
                "rollout_correction": OmegaConf.to_container(
                    OmegaConf.structured(rollout_corr_config),
                    resolve=True,
                )
            },
        }
    )

    _validate_policy_loss_rollout_correction(config)
    worker_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    assert worker_config.actor_rollout_ref.actor.policy_loss.loss_mode == "bypass_mode"
    assert worker_config.actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode is True
    assert (
        worker_config.actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type
        == rollout_corr_config.loss_type
    )


@pytest.mark.parametrize(
    ("loss_mode", "algorithm_bypass_mode"),
    [
        ("vanilla", False),
        ("vanilla", True),
        ("cispo", False),
        ("cispo", True),
        ("bypass_mode", True),
    ],
)
def test_policy_loss_and_algorithm_bypass_valid_combinations(loss_mode, algorithm_bypass_mode):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "loss_mode": loss_mode,
                        "rollout_correction": {},
                    }
                }
            },
            "algorithm": {
                "rollout_correction": {
                    "bypass_mode": algorithm_bypass_mode,
                    "loss_type": "ppo_clip",
                }
            },
        }
    )

    _validate_policy_loss_rollout_correction(config)

    if loss_mode == "bypass_mode":
        assert config.actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode is True


def test_policy_loss_bypass_mode_requires_algorithm_bypass():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "loss_mode": "bypass_mode",
                        "rollout_correction": {},
                    }
                }
            },
            "algorithm": {
                "rollout_correction": {
                    "bypass_mode": False,
                    "loss_type": "ppo_clip",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="requires algorithm.rollout_correction.bypass_mode=true"):
        _validate_policy_loss_rollout_correction(config)


def test_bypass_loss_runtime_guard_rejects_decoupled_config():
    config = _actor_config("bypass_mode")
    tensor = torch.zeros((1, 1))

    with pytest.raises(ValueError, match="requires algorithm.rollout_correction.bypass_mode=true"):
        compute_policy_loss_bypass_mode(
            old_log_prob=tensor,
            log_prob=tensor,
            advantages=tensor,
            response_mask=torch.ones_like(tensor),
            config=config,
        )


def test_run_ppo_validates_before_dispatching_custom_task_runner(monkeypatch):
    main_ppo = pytest.importorskip("verl.trainer.main_ppo", reason="Ray training dependencies are not installed")
    observed = {}
    config = OmegaConf.create({"ray_kwargs": {}})

    def fake_validate_config(*, config, use_reference_policy, use_critic):
        observed["validated_config"] = config
        observed["use_reference_policy"] = use_reference_policy
        observed["use_critic"] = use_critic

    class FakeRemoteRun:
        @staticmethod
        def remote(dispatched_config):
            observed["dispatched_config"] = dispatched_config
            return "run-result"

    class FakeRunner:
        run = FakeRemoteRun()

    class FakeTaskRunner:
        @staticmethod
        def remote():
            return FakeRunner()

    monkeypatch.setattr(main_ppo, "validate_config", fake_validate_config)
    monkeypatch.setattr(main_ppo, "need_reference_policy", lambda _: False)
    monkeypatch.setattr(main_ppo, "need_critic", lambda _: False)
    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(main_ppo.ray, "get", lambda result: result)

    main_ppo.run_ppo(config, task_runner_class=FakeTaskRunner)

    assert observed["validated_config"] is config
    assert observed["dispatched_config"] is config
    assert observed["use_reference_policy"] is False
    assert observed["use_critic"] is False
