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

import asyncio
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
                    use_dynamic_bsz=False,
                    ppo_micro_batch_size_per_gpu=2,
                    ppo_infer_micro_batch_size_per_gpu=3,
                    ppo_infer_max_token_len_per_gpu=1234,
                    ppo_max_token_len_per_gpu=5678,
                )

                kwargs = critic_utils.prepare_critic_worker_config_kwargs(critic_config)

                self.assertEqual(kwargs["model_type"], "value_model")
                self.assertIs(kwargs["model_config"], model)
                self.assertIs(kwargs["engine_config"], engine)
                self.assertIs(kwargs["optimizer_config"], optim)
                self.assertIs(kwargs["checkpoint_config"], checkpoint)
                self.assertFalse(engine.use_dynamic_bsz)
                self.assertEqual(engine.micro_batch_size_per_gpu, 2)
                self.assertEqual(engine.infer_micro_batch_size_per_gpu, 3)
                self.assertEqual(engine.infer_max_token_len_per_gpu, 1234)
                self.assertEqual(engine.max_token_len_per_gpu, 5678)

    def test_inference_micro_batch_falls_back_to_legacy_forward_setting(self):
        engine = SimpleNamespace(strategy="fsdp2")
        critic_config = SimpleNamespace(
            engine=engine,
            model=object(),
            optim=object(),
            checkpoint=object(),
            use_dynamic_bsz=False,
            ppo_micro_batch_size_per_gpu=2,
            ppo_infer_micro_batch_size_per_gpu=None,
            forward_micro_batch_size_per_gpu=5,
            ppo_infer_max_token_len_per_gpu=1234,
            ppo_max_token_len_per_gpu=5678,
        )

        critic_utils.prepare_critic_worker_config_kwargs(critic_config)

        self.assertEqual(engine.infer_micro_batch_size_per_gpu, 5)


class TestFullyAsyncBatchDivisor(unittest.TestCase):
    def test_lcm_covers_actor_and_critic_requirements(self):
        divisor = batch_utils.get_train_batch_divisor(
            actor_dp_size=6,
            actor_mini_batch_rows=12,
            critic_dp_size=4,
            critic_mini_batch_rows=20,
        )
        self.assertEqual(divisor, 60)

    def test_actor_full_batch_still_honors_critic_minibatch(self):
        divisor = batch_utils.get_train_batch_divisor(
            actor_dp_size=8,
            actor_mini_batch_rows=None,
            critic_dp_size=4,
            critic_mini_batch_rows=12,
        )
        self.assertEqual(divisor, 24)

    def test_critic_requirements_must_be_provided_together(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            batch_utils.get_train_batch_divisor(
                actor_dp_size=8,
                actor_mini_batch_rows=16,
                critic_dp_size=8,
            )

    def test_actor_minibatch_must_be_divisible_by_actor_dp(self):
        with self.assertRaisesRegex(ValueError, "actor mini-batch rows"):
            batch_utils.get_train_batch_divisor(actor_dp_size=4, actor_mini_batch_rows=6)

    def test_critic_minibatch_must_be_divisible_by_critic_dp(self):
        with self.assertRaisesRegex(ValueError, "critic mini-batch rows"):
            batch_utils.get_train_batch_divisor(
                actor_dp_size=4,
                actor_mini_batch_rows=8,
                critic_dp_size=4,
                critic_mini_batch_rows=6,
            )

    def test_fully_async_train_steps_are_available_before_worker_init(self):
        total_train_steps = batch_utils.get_fully_async_train_steps(
            total_rollout_steps=128,
            required_samples=16,
            trigger_parameter_sync_step=4,
        )
        total_optimizer_steps = batch_utils.get_fully_async_optimizer_steps(
            total_rollout_steps=128,
            required_samples=16,
        )
        self.assertEqual(total_train_steps, 2)
        self.assertEqual(total_optimizer_steps, 8)


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
                    use_dynamic_bsz=False,
                    ppo_micro_batch_size_per_gpu=2,
                    ppo_infer_micro_batch_size_per_gpu=3,
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

    def test_fully_async_resolves_actor_and_critic_dp_meshes_before_training(self):
        from verl.experimental.fully_async_policy import fully_async_trainer

        raw_trainer_cls = fully_async_trainer.FullyAsyncTrainer.__ray_actor_class__
        trainer = object.__new__(raw_trainer_cls)
        actor_wg = object()
        critic_wg = object()
        trainer.actor_wg = actor_wg
        trainer.critic_wg = critic_wg
        trainer.use_critic = True
        trainer.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(n=2),
                actor=SimpleNamespace(ppo_mini_batch_size=8, get=lambda _key, default=None: default),
            ),
            critic=SimpleNamespace(ppo_mini_batch_size=6),
        )
        queries = []

        def get_dp_size(worker_group, mesh_name):
            queries.append((worker_group, mesh_name))
            return 4 if mesh_name == "actor" else 3

        trainer._get_dp_size = get_dp_size
        trainer._init_train_batch_divisor()

        self.assertEqual(queries, [(actor_wg, "actor"), (critic_wg, "train")])
        self.assertEqual(trainer.train_batch_divisor, 48)

    def test_worker_initialization_injects_total_steps_before_models_start(self):
        from verl.experimental.fully_async_policy import fully_async_main

        events = []

        class RemoteCall:
            def __init__(self, name, result=None):
                self.name = name
                self.result = result

            def remote(self, *args):
                events.append((self.name, args))
                return self.result

        trainer = SimpleNamespace(
            set_total_train_steps=RemoteCall("set_total_train_steps"),
            init_workers=RemoteCall("trainer_init_workers"),
        )
        rollouter = SimpleNamespace(
            get_total_train_steps=RemoteCall("get_total_train_steps", result=7),
            get_total_optimizer_steps=RemoteCall("get_total_optimizer_steps", result=28),
            init_workers=RemoteCall("rollouter_init_workers"),
            set_max_required_samples=RemoteCall("set_max_required_samples"),
        )
        raw_runner_cls = fully_async_main.FullyAsyncTaskRunner.__ray_actor_class__
        runner = object.__new__(raw_runner_cls)
        runner.components = {"trainer": trainer, "rollouter": rollouter}

        with patch.object(fully_async_main.ray, "get", side_effect=lambda value: value):
            runner._initialize_worker_components()

        self.assertEqual(
            events,
            [
                ("get_total_train_steps", ()),
                ("get_total_optimizer_steps", ()),
                ("set_total_train_steps", (7, 28)),
                ("trainer_init_workers", ()),
                ("rollouter_init_workers", ()),
                ("set_max_required_samples", ()),
            ],
        )

    def test_fully_async_fit_step_runs_critic_dataflow_before_actor(self):
        from verl.experimental.fully_async_policy import fully_async_trainer

        events = []
        raw_trainer_cls = fully_async_trainer.FullyAsyncTrainer.__ray_actor_class__
        trainer = object.__new__(raw_trainer_cls)
        trainer.global_steps = 3
        trainer.epoch = 0

        async def generate(_batch):
            events.append("generate")
            return "batch"

        async def update_weights():
            events.append("update_weights")

        async def validate():
            events.append("validate")

        def batch_stage(name):
            def run(batch):
                events.append(name)
                return batch

            return run

        trainer._fit_start_profile = lambda: events.append("start_profile")
        trainer._fit_generate = generate
        trainer._fit_compute_reward = batch_stage("reward")
        trainer._fit_compute_log_prob = batch_stage("log_prob")
        trainer._fit_compute_ref_log_prob = batch_stage("ref_log_prob")
        trainer._fit_compute_critic = batch_stage("values")
        trainer._fit_compute_advantage = batch_stage("advantage")
        trainer._fit_update_critic = batch_stage("update_critic")
        trainer._fit_update_actor = batch_stage("update_actor")
        trainer._fit_update_local_step = lambda: events.append("local_step")
        trainer._fit_update_weights = update_weights
        trainer._fit_dump_data = lambda _batch: events.append("dump")
        trainer._fit_validate = validate
        trainer._fit_save_checkpoint = lambda: events.append("save")
        trainer._fit_stop_profile = lambda: events.append("stop_profile")
        trainer._fit_collect_metrics = lambda _batch: events.append("collect_metrics")
        trainer._fit_postprocess_step = lambda: events.append("postprocess")

        asyncio.run(trainer.fit_step())

        self.assertLess(events.index("values"), events.index("advantage"))
        self.assertLess(events.index("advantage"), events.index("update_critic"))
        self.assertLess(events.index("update_critic"), events.index("update_actor"))


if __name__ == "__main__":
    unittest.main()
