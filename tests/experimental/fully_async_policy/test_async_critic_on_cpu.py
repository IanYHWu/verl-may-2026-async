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

"""CPU-only regression tests for critic PPO in fully asynchronous training."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dependency_light_module(name: str, relative_path: str):
    """Load a pure helper without importing ``verl.protocol`` (which requires Ray)."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


critic_utils = _load_dependency_light_module("critic_utils", "verl/trainer/ppo/critic_utils.py")
batch_utils = _load_dependency_light_module(
    "fully_async_batch_utils", "verl/experimental/fully_async_policy/batch_utils.py"
)


class TestCriticWorkerConfig(unittest.TestCase):
    def test_current_critic_schema_supports_all_value_engine_backends(self):
        for strategy in ("fsdp", "fsdp2", "megatron"):
            with self.subTest(strategy=strategy):
                engine = SimpleNamespace(strategy=strategy)
                model = object()
                optim = object()
                checkpoint = object()
                critic_config = SimpleNamespace(
                    strategy=strategy,
                    engine=engine,
                    model=model,
                    optim=optim,
                    checkpoint=checkpoint,
                    ppo_infer_max_token_len_per_gpu=1234,
                    ppo_max_token_len_per_gpu=5678,
                )

                kwargs = critic_utils.prepare_critic_worker_config_kwargs(critic_config)

                self.assertEqual(kwargs["model_type"], "value_model")
                self.assertIs(kwargs["model_config"], model)
                self.assertIs(kwargs["engine_config"], engine)
                self.assertIs(kwargs["optimizer_config"], optim)
                self.assertIs(kwargs["checkpoint_config"], checkpoint)
                self.assertEqual(engine.infer_max_token_len_per_gpu, 1234)
                self.assertEqual(engine.max_token_len_per_gpu, 5678)


class TestFullyAsyncBatchDivisor(unittest.TestCase):
    def test_lcm_covers_actor_and_critic_requirements(self):
        divisor = batch_utils.get_train_batch_divisor(
            actor_world_size=6,
            actor_mini_batch_rows=8,
            critic_world_size=4,
            critic_mini_batch_rows=10,
        )
        self.assertEqual(divisor, 120)

    def test_actor_full_batch_still_honors_critic_minibatch(self):
        divisor = batch_utils.get_train_batch_divisor(
            actor_world_size=8,
            actor_mini_batch_rows=None,
            critic_world_size=8,
            critic_mini_batch_rows=12,
        )
        self.assertEqual(divisor, 24)

    def test_critic_requirements_must_be_provided_together(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            batch_utils.get_train_batch_divisor(
                actor_world_size=8,
                actor_mini_batch_rows=16,
                critic_world_size=8,
            )


try:
    import ray  # noqa: F401
except ModuleNotFoundError:
    _RAY_AVAILABLE = False
else:
    _RAY_AVAILABLE = True


@unittest.skipUnless(_RAY_AVAILABLE, "Ray training dependencies are not installed")
class TestFullyAsyncCriticIntegration(unittest.TestCase):
    def test_separated_builder_uses_unified_worker_config_for_all_backends(self):
        from verl.experimental.separation import ray_trainer as separation_trainer
        from verl.trainer.ppo.utils import Role

        for strategy in ("fsdp", "fsdp2", "megatron"):
            with self.subTest(strategy=strategy):
                engine = SimpleNamespace(strategy=strategy)
                critic_config = SimpleNamespace(
                    strategy=strategy,
                    engine=engine,
                    model=object(),
                    optim=object(),
                    checkpoint=object(),
                    ppo_infer_max_token_len_per_gpu=1234,
                    ppo_max_token_len_per_gpu=5678,
                )
                trainer = object.__new__(separation_trainer.SeparateRayPPOTrainer)
                trainer.use_critic = True
                trainer.config = SimpleNamespace(critic=object())
                trainer.resource_pool_manager = SimpleNamespace(get_resource_pool=lambda _: "trainer_pool")
                trainer.resource_pool_to_cls = {"trainer_pool": {}}
                trainer.role_worker_mapping = {Role.Critic: object()}

                with patch.object(separation_trainer, "omega_conf_to_dataclass", return_value=critic_config):
                    trainer._create_critic_class()

                worker = trainer.resource_pool_to_cls["trainer_pool"][str(Role.Critic)]
                worker_config = worker.kwargs["config"]
                self.assertEqual(worker_config.model_type, "value_model")
                self.assertIs(worker_config.engine_config, engine)
                self.assertIs(worker_config.model_config, critic_config.model)

    def test_fully_async_initializes_training_worker_critic_contract(self):
        from verl.experimental.fully_async_policy import fully_async_trainer
        from verl.trainer.ppo.utils import Role

        class FakeCriticWorkerGroup:
            def __init__(self):
                self.reset_called = False
                self.loss_fn = None

            def reset(self):
                self.reset_called = True

            def set_loss_fn(self, loss_fn):
                self.loss_fn = loss_fn

        class FakeActorWorkerGroup:
            def __init__(self):
                self.init_called = False

            def init_model(self):
                self.init_called = True

        raw_trainer_cls = fully_async_trainer.FullyAsyncTrainer.__ray_actor_class__
        trainer = object.__new__(raw_trainer_cls)
        trainer.use_critic = True
        trainer.use_reference_policy = False
        trainer.ref_in_actor = False
        trainer.train_role = Role.Actor
        trainer.orig_critic_cfg = object()
        critic_wg = FakeCriticWorkerGroup()
        actor_wg = FakeActorWorkerGroup()
        trainer.all_wg = {str(Role.Critic): critic_wg, str(Role.Actor): actor_wg}

        trainer._init_models()

        self.assertTrue(critic_wg.reset_called)
        self.assertIsNotNone(critic_wg.loss_fn)
        self.assertIs(critic_wg.loss_fn.keywords["config"], trainer.orig_critic_cfg)
        self.assertTrue(actor_wg.init_called)


if __name__ == "__main__":
    unittest.main()
