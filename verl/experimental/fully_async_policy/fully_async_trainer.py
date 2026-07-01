# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import logging
import os
import time
from datetime import datetime
from pprint import pprint
from typing import Any

import ray
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    ValidateMetrics,
    assemble_batch_from_rollout_samples,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerManager

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # ==================== fully async config ====================

        self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # use trainer to do validation
        if self.config.async_training.use_trainer_do_validate:
            from verl.trainer.main_ppo import create_rl_dataset
            from verl.utils.dataset.rl_dataset import collate_fn

            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            print(f"[FullyAsyncTrainer] split before val_dataset total len: {len(val_dataset)}")
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[rollout_gpus:]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            print(f"[FullyAsyncTrainer] split after val_dataset total len: {len(val_dataset)}")
            self.val_dataset = val_dataset
            # update val_dataloader
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(val_dataset)
            from torchdata.stateful_dataloader import StatefulDataLoader

            print(f"[FullyAsyncTrainer] create val_dataloader with batch_size: {val_batch_size}")
            self.val_dataloader = StatefulDataLoader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                num_workers=self.config.data["dataloader_num_workers"],
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # when use_trainer_do_validate == Ture, use colocate_checkpoint_manager to sync params
        self.colocate_checkpoint_manager = None

    def _setup_checkpoint_manager(self, rollouter):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = ray.get(rollouter.get_replicas.remote())
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def set_rollouter(self, rollouter):
        """Set rollouter reference for parameter synchronization"""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        self._setup_checkpoint_manager(rollouter)

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = await self.message_queue_client.get_sample()

            if sample is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            queue_samples.append(sample)

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds. "
            f"mq_len: {queue_len}"
        )

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch. Balancing is DEFERRED to _fit_generate (AFTER the pad that makes the
        # row count DP-divisible): _balance_batch uses equal_size=True (asserts len % dp_size == 0),
        # but a variable/odd per-step row count (e.g. the v3 flat MR/E/FA rollout) crashes
        # karmarkar_karp if balanced here, PRE-pad. Assemble without balancing.
        batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_reward_model_class(self):
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_reward_loop()
        await self._init_async_rollout_manager()

    def _init_reward_loop(self):
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] Init reward loop")
            super()._init_reward_loop()

    async def _init_async_rollout_manager(self):
        # use async rollout do validate
        print(f"[FullyAsyncTrainer] use_trainer_do_validate: {self.config.async_training.use_trainer_do_validate}")
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] Init async rollout manager")

            # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
            # agent_reward_loop: streaming reward computation with actor rollout
            # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
            enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

            # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
            # to stream reward computation with actor rollout
            reward_loop_worker_handles = (
                self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
            )

            # create async rollout manager and request scheduler
            assert self.config.actor_rollout_ref.rollout.mode == "async"

            self.async_rollout_mode = True
            from verl.experimental.agent_loop import AgentLoopManager

            self.llm_server_manager = await LLMServerManager.create(
                config=self.config, worker_group=self.actor_rollout_wg
            )
            self.async_rollout_manager = await AgentLoopManager.create(
                config=self.config,
                llm_client=self.llm_server_manager.get_client(),
                reward_loop_worker_handles=reward_loop_worker_handles,
            )
            print("[FullyAsyncTrainer] async_rollout_manager initialized")

            # Modify checkpoint_engine config to use naive backend
            checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
            original_backend = checkpoint_engine_cfg.backend
            with open_dict(checkpoint_engine_cfg):
                checkpoint_engine_cfg.backend = "naive"
            checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

            print(f"[FullyAsyncTrainer] checkpoint_engine_config: {checkpoint_engine_config}")

            self.colocate_checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_engine_config,
                trainer=self.actor_rollout_wg,
                replicas=self.llm_server_manager.get_replicas(),
            )

            # sleep all replicas to load checkpoint
            await self.colocate_checkpoint_manager.sleep_replicas()

            # Restore original backend value
            with open_dict(checkpoint_engine_cfg):
                checkpoint_engine_cfg.backend = original_backend

            print("[FullyAsyncTrainer] colocate_checkpoint_manager initialized")

        else:
            print("[FullyAsyncTrainer] Skip async rollout manager (use_trainer_do_validate=False)")

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_postprocess_step()

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
            # The collected batch may not be a multiple of the actor's sequence-level
            # mini-batch (ppo_mini_batch_size * n) when the rollout produces a variable
            # number of rows per sample (e.g. per-step / multi-row agent recipes). Pad
            # to a clean multiple and mask the pad rows out of the gradient so the
            # mini-batch split is valid. No-op when already divisible.
            import torch as _torch

            from verl.protocol import pad_dataproto_to_divisor

            n = self.config.actor_rollout_ref.rollout.n
            ppo_mini = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            # Default optimizer mini-batch is ppo_mini * n rows; an explicit
            # train_minibatch_rows overrides it. Must match the split size that
            # _update_actor (ray_trainer) passes to make_iterator.
            train_minibatch_rows = self.config.actor_rollout_ref.actor.get("train_minibatch_rows", None)
            if train_minibatch_rows == 0:
                # Full-batch mode: exactly ONE optimizer step over the whole batch, so pad ONLY
                # to DP-divisibility (world_size ⊇ dp_size) instead of up to a fixed row count.
                # _update_actor sets mini_batch_size = len(batch), so make_iterator's
                # `batch % mini_batch_size == 0` holds trivially. Minimizes masked-pad waste while
                # keeping opt-steps-per-train-step = 1 regardless of the variable per-step row count.
                divisor = self.actor_wg.world_size
            else:
                mini_rows = int(train_minibatch_rows) if train_minibatch_rows else ppo_mini * n
                divisor = max(self.actor_wg.world_size, mini_rows)
            if len(batch) % divisor != 0:
                orig_size = len(batch)
                # pad_dataproto_to_divisor concats slices, and DataProto.concat asserts
                # meta_info[k] == v per key — which raises on array-valued meta_info
                # (e.g. trajectory_param_versions). meta_info is batch-level and already
                # consumed by _collect_metrics_from_samples above, so clear it across the
                # pad, then restore it and recompute the per-row global_token_num.
                saved_meta = batch.meta_info
                batch.meta_info = {}
                batch, pad_size = pad_dataproto_to_divisor(batch, divisor)
                batch.meta_info = saved_meta
                if "response_mask" in batch.batch:
                    batch.batch["response_mask"][orig_size:] = 0
                for _k in ("rm_scores", "token_level_rewards", "token_level_scores", "advantages"):
                    if _k in batch.batch:
                        batch.batch[_k][orig_size:] = 0
                if "attention_mask" in batch.batch:
                    batch.meta_info["global_token_num"] = _torch.sum(
                        batch.batch["attention_mask"], dim=-1).tolist()
                metrics["fully_async/batch_pad_rows"] = float(pad_size)
            # Balance sequence lengths across DP ranks AFTER the pad above (which made len a
            # multiple of max(world_size, mini_rows) ⊇ dp_size). Doing it here — not in
            # _get_samples_from_queue, pre-pad — lets _balance_batch's equal_size=True seqlen
            # partition see a DP-divisible row count, so it no longer asserts `len % dp_size != 0`
            # on a variable/odd v3 per-step row count. Pad rows are already masked, so the
            # reorder is loss-neutral.
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        # Reset staleness in rollouter
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        # Anchor cycle-level logs to the global_step that _fit_postprocess_step is
        # about to set, so they share the wandb step with the per-step log (which
        # then takes precedence for any overlapping keys). Avoids wandb step regression
        # when current_param_version trails global_steps (e.g. trigger_sync_step>1).
        cycle_log_step = self.global_steps + 1
        self.logger.log(
            data=timing_raw,
            step=cycle_log_step,
        )

        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=cycle_log_step,
        )
        self.metrics_aggregator.reset()

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Capture validation generations for deferred logging in _fit_validate.

        When use_trainer_do_validate=True, the trainer also runs _validate(True) which
        calls this method. We capture instead of logging immediately so that we can
        merge with rollouter-side generations and log once with the correct step.
        """
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            self._captured_val_generations = []
            return

        import numpy as np

        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        self._captured_val_generations = samples[:generations_to_log]

    async def _validate_process(self):
        """Run trainer-side validation using async rollout manager"""
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] _validate_process")
            from verl.utils.profiler import marked_timer

            # Wake up rollouter replicas and sync weights
            print("[FullyAsyncTrainer] wake up replicas before validation")
            await self.colocate_checkpoint_manager.update_weights(global_steps=self.current_param_version)

            with marked_timer("trainer/validate_time", self.timing_raw):
                train_val_metrics = self._validate(True)

            # Sleep rollouter replicas to free GPU memory for validation
            print("[FullyAsyncTrainer] sleep replicas after validation")
            await self.colocate_checkpoint_manager.sleep_replicas()

            print(f"[FullyAsyncTrainer] validate timing: {self.timing_raw['trainer/validate_time']}")
            return train_val_metrics
        else:
            print("[FullyAsyncTrainer] _validate_process without async_rollout_manager")
            return None

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return

        # Trigger rollouter validation and get future
        val_future = self.rollouter.do_validate.remote()

        # Run trainer-side validation
        self._captured_val_generations = []
        train_val_metrics = await self._validate_process()

        # Wait for rollouter validation result and log
        val_metrics: ValidateMetrics = await asyncio.wrap_future(val_future.future())
        # Log val on the SAME wandb step axis as the train metrics. The train series
        # advances `step` by self.global_steps (≈ trigger * current_param_version), so
        # logging val at step=current_param_version is a BACKWARD step (e.g. 15 vs the
        # current ~30) -> wandb silently drops non-monotonic logs and val never appears.
        # global_steps+1 is the step _fit_postprocess_step/_fit_update_weights use for
        # this same fit_step, so val rides on the matching train point.
        val_step = self.global_steps + 1
        if train_val_metrics:
            # Merge trainer and rollouter validation results
            with marked_timer("timing_s/merge_val", self.timing_raw):
                new_metrics = self._merge_validation_results(train_val_metrics, val_metrics.metrics)
            if new_metrics:
                self.logger.log(data=new_metrics, step=val_step)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {self.current_param_version} "
                    f"Validation metrics: {new_metrics}, timing: {self.timing_raw['timing_s/merge_val']}"
                )
        else:
            if val_metrics.metrics:
                self.logger.log(data=val_metrics.metrics, step=val_step)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {self.current_param_version} "
                    f"Validation metrics: {val_metrics.metrics}"
                )
        self.logger.log(data=val_metrics.timing_raw, step=val_step)

        # Merge and log validation generations from rollouter (and trainer if applicable)
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log > 0:
            import numpy as np

            all_generations = list(self._captured_val_generations)
            if val_metrics.val_generations:
                all_generations.extend(val_metrics.val_generations)
            if all_generations:
                all_generations.sort(key=lambda x: x[0])
                rng = np.random.RandomState(42)
                rng.shuffle(all_generations)
                all_generations = all_generations[:generations_to_log]
                self.validation_generations_logger.log(
                    self.config.trainer.logger, all_generations, self.current_param_version
                )

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            force or self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1

        # Per-step wandb log: flush self.metrics every step (not just at param sync).
        # Without this, with trigger_parameter_sync_step>1, only cycle-aggregated values
        # (avg over the cycle) appear in wandb — masking per-step trajectories.
        self.logger.log(data=self.metrics, step=self.global_steps)

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        if self.colocate_checkpoint_manager:
            await self.colocate_checkpoint_manager.update_weights(self.current_param_version)
            await self.colocate_checkpoint_manager.sleep_replicas()

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            # Keys a custom rollout manager contributed (tagged by the rollouter in
            # get_statistics). assemble_batch_from_rollout_samples prefixes ALL
            # rollout_status with "fully_async/"; for the manager's own metrics we ALSO
            # log them at their native top-level name so a recipe's async metrics match
            # the namespace it uses in colocate. Driven by the manager's declared keys —
            # verl never hardcodes recipe-specific metric names.
            mgr_keys = set(batch.meta_info.get("fully_async/manager_metric_keys") or [])
            for key, value in batch.meta_info.items():
                if key == "fully_async/manager_metric_keys":
                    continue  # bookkeeping list, not a metric
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value
                    if key.startswith("fully_async/") and key[len("fully_async/"):] in mgr_keys:
                        metrics[key[len("fully_async/"):]] = value
