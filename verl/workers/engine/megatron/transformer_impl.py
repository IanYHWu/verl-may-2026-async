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

import inspect
import logging
import os
from functools import partial
from typing import Any, Callable, ContextManager, Iterator, Optional

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.models.mcore import get_mcore_weight_converter
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    merge_router_topk_indices,
    pp_gather,
    reorder_and_merge_vpp_layers,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import (
    vocab_parallel_entropy,
    vocab_parallel_entropy_chunked,
    vocab_parallel_log_probs_from_logits,
    vocab_parallel_sum_pi_squared,
)
from verl.utils.megatron_peft_utils import add_base_layer_suffix, build_peft_config_for_vllm
from verl.utils.megatron_utils import (
    check_mtp_config,
    get_megatron_module_device,
    get_megatron_mtp_loss,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    patch_engine_mtp,
    register_megatron_training_hooks,
    unwrap_model,
)
from verl.utils.model import extract_multi_modal_inputs, load_mcore_dist_weights
from verl.utils.seqlen_balancing import restore_dynamic_batch
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches
from .utils import set_random_seed

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronEngine(BaseEngine):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.use_mbridge, "use_mbridge must be True"
        self._init_device_mesh()

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_grad = self.engine_config.grad_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.mode = None

        # Set by _maybe_enable_mem_efficient_ce() during initialize(); safe default here so the
        # forward path can always read it even before initialize() runs.
        self._mem_efficient_ce = False

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None

        # QAT configuration
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            if self.engine_config.vanilla_mbridge:
                raise ValueError(
                    "QAT requires non-vanilla Megatron bridge. "
                    "Please set 'use_mbridge=True' and 'vanilla_mbridge=False'."
                )
            logger.info(f"QAT enabled in MegatronEngine: mode={self._qat_config.mode}")

        # Router replay configuration for MoE models
        self.enable_routing_replay = self.engine_config.router_replay.mode != "disabled"
        logger.info(f"enable_routing_replay in MegatronEngine: {self.enable_routing_replay}")
        if self.enable_routing_replay:
            apply_router_replay_patch()
            self.mini_layer_topk_idx_list = []
        # Apply checkpoint patch for MoE models
        from verl.utils.device import is_cuda_available

        if is_cuda_available:
            from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

            apply_patch_megatron_recomputation_backward()

    def _init_device_mesh(self):
        # TODO: set different parallelism for actor, critic, ref
        if mpu.is_initialized():
            return

        extra_args = dict()

        if self.engine_config.dynamic_context_parallel:
            assert "dynamic_context_parallel" in inspect.signature(mpu.initialize_model_parallel).parameters, (
                "dynamic_context_parallel is not supported in your megatron version, "
                + "please update your megatron version to the latest version"
            )
            assert self.engine_config.max_seqlen_per_dp_cp_rank is not None, (
                "max_seqlen_per_dp_cp_rank is required when dynamic_context_parallel is enabled"
            )
            extra_args["dynamic_context_parallel"] = self.engine_config.dynamic_context_parallel

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.engine_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
            **extra_args,
        )

    def _build_tf_config(self):
        from verl.utils.megatron_utils import mapping_string_to_attn_backend
        from verl.utils.torch_dtypes import PrecisionType

        check_mtp_config(self.model_config, self.engine_config)

        self.param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        override_transformer_config = mapping_string_to_attn_backend({**self.engine_config.override_transformer_config})

        self.provider = None
        self.vanilla_bridge = self.engine_config.vanilla_mbridge

        if self.vanilla_bridge:
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
            if self.engine_config.dynamic_context_parallel:
                override_transformer_config["max_seqlen_per_dp_cp_rank"] = self.engine_config.max_seqlen_per_dp_cp_rank
                # note(baiyan): we must set the transformer_config.dynamic_context_parallel to False
                # because of the bad coupling design in Megatron-LM
                # https://github.com/xiaoyao0115/Megatron-LM/blob/88733ab6614e3e91b9d095172f41e7d8b5d8e9d4/megatron/core/pipeline_parallel/dynamic_cp_schedule.py#L552-L553
                # but it does not affect the functionality of dynamic CP, so we can use it to avoid the coupling.
                override_transformer_config["dynamic_context_parallel"] = False
                override_transformer_config["context_parallel_size"] = mpu.get_data_parallel_world_size()
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            tf_config.fp16 = self.param_dtype == torch.float16
            tf_config.bf16 = self.param_dtype == torch.bfloat16
        else:
            from verl.models.mcore.bridge import AutoBridge

            # Use Megatron-Bridge to convert HF config to Megatron config
            bridge = AutoBridge.from_hf_pretrained(
                self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
            )
            # Get Megatron provider and configure it
            provider = bridge.to_megatron_provider(load_weights=False)

            # In case of invalid overrides, we need to make sure some critical params are set correctly
            provider.params_dtype = self.param_dtype

            # Ensure dtype settings propagate to Megatron-Bridge/TE
            provider.fp16 = self.param_dtype == torch.float16
            provider.bf16 = self.param_dtype == torch.bfloat16

            # Pass distributed info
            provider.tensor_model_parallel_size = self.engine_config.tensor_model_parallel_size
            provider.pipeline_model_parallel_size = self.engine_config.pipeline_model_parallel_size
            provider.expert_model_parallel_size = self.engine_config.expert_model_parallel_size
            provider.expert_tensor_parallel_size = self.engine_config.expert_tensor_parallel_size
            provider.virtual_pipeline_model_parallel_size = self.engine_config.virtual_pipeline_model_parallel_size
            provider.context_parallel_size = self.engine_config.context_parallel_size
            provider.sequence_parallel = self.engine_config.sequence_parallel

            # Match verl implementation (need variable_seq_lengths)
            from megatron.core.transformer.enums import AttnBackend

            provider.attention_backend = AttnBackend.flash
            provider.variable_seq_lengths = True
            provider.moe_token_dispatcher_type = "alltoall"
            provider.moe_router_load_balancing_type = "none"

            # Apply QAT: set quantization layer spec and patch Megatron-Bridge
            if self._qat_enabled:
                from verl.utils.modelopt import patch_provider_for_qat

                patch_provider_for_qat(provider)

            # Apply transformer config overrides
            for key, value in override_transformer_config.items():
                setattr(provider, key, value)

            if self.enable_routing_replay:
                provider.enable_routing_replay = True

            provider.finalize()
            self.provider = provider
            tf_config = None  # Will be set after model creation
        self.bridge = bridge

        if not self.bridge:
            self.weight_converter = get_mcore_weight_converter(self.model_config.hf_config, self.dtype)

        # Set enable_routing_replay directly on tf_config instead of passing through
        # override_transformer_config, because dataclass subclasses like MLATransformerConfig
        # generate their own __init__ and don't inherit the patched TransformerConfig.__init__
        # that accepts this kwarg.
        if self.enable_routing_replay and tf_config is not None:
            tf_config.enable_routing_replay = True

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.tf_config = tf_config

        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.model_config, bridge=self.bridge, provider=self.provider, dtype=self.param_dtype
        )

    def _build_megatron_module(self):
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        self.is_value_model = self.model_config.model_type == "value_model"
        if self.engine_config.forward_only:
            wrap_with_ddp = False
        else:
            wrap_with_ddp = True

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=self.is_value_model,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
        )
        if self.is_value_model:
            self.model_config.hf_config.tie_word_embeddings = False

        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.model_config.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=self.engine_config.override_mcore_model_config,
            override_ddp_config=self.engine_config.override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.model_config.get("lora", None),
        )
        self.tf_config = updated_tf_config
        print(f"module: {len(module)}")

        if self.engine_config.use_dist_checkpointing:
            load_mcore_dist_weights(
                module, self.engine_config.dist_checkpointing_path, is_value_model=self.is_value_model
            )
        else:
            if self.vanilla_bridge:
                self.bridge.load_weights(module, self.model_config.local_path)
            else:
                allowed_mismatched_params = []
                if self.is_value_model:
                    allowed_mismatched_params = ["output_layer.weight"]
                self.bridge.load_hf_weights(
                    module, self.model_config.local_path, allowed_mismatched_params=allowed_mismatched_params
                )

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])

        if self.enable_routing_replay:
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")

        return module

    def _maybe_enable_fused_kernels(self):
        if not self.engine_config.use_fused_kernels:
            return

        if self.is_value_model or self.model_config.mtp.enable:
            logger.warning_once(
                "Fused kernels are not supported for value models or when MTP is enabled in Megatron engine; disabling."
            )
            self.engine_config.use_fused_kernels = False
            return

        from verl.models.mcore.model_forward_fused import patch_fused_forward

        for model in self.module:
            patch_fused_forward(model)

    def _maybe_enable_mem_efficient_ce(self):
        """Opt-in (env ``VERL_MEGATRON_MEM_EFFICIENT_CE=1``): route the *non-fused* bshd path
        through the fused ``linear_cross_entropy`` kernel to avoid the full ``[tokens x vocab_shard]``
        logits + gradient materialization that OOMs on long (~100k-token) rows.

        Unlike ``use_fused_kernels`` (which requires ``use_remove_padding=True`` / THD packing that
        the GatedDeltaNet layers reject), this patches the model to return HIDDEN states and runs the
        LM-head + CE fusion inside the engine's ``logits_processor``. It is numerically equivalent to
        the legacy ``vocab_parallel_log_probs_from_logits`` / ``vocab_parallel_entropy`` path.
        """
        self._mem_efficient_ce = os.getenv("VERL_MEGATRON_MEM_EFFICIENT_CE", "0") == "1"
        if not self._mem_efficient_ce:
            return

        if self.is_value_model or self.model_config.mtp.enable:
            logger.warning_once(
                "Memory-efficient CE is not supported for value models or when MTP is enabled; disabling."
            )
            self._mem_efficient_ce = False
            return

        if self.engine_config.use_fused_kernels:
            # use_fused_kernels already patches the model to the (thd) fused forward; don't double-patch.
            logger.warning_once(
                "VERL_MEGATRON_MEM_EFFICIENT_CE ignored because use_fused_kernels=True is already active."
            )
            self._mem_efficient_ce = False
            return

        from verl.models.mcore.model_forward_fused import patch_hidden_forward

        for model in self.module:
            patch_hidden_forward(model)
        logger.warning_once(
            "Memory-efficient CE ENABLED (VERL_MEGATRON_MEM_EFFICIENT_CE=1): non-fused Megatron path "
            "will fuse LM head + cross-entropy via linear_cross_entropy (returns hidden, no full logits)."
        )

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config

        optim_config_megatron = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
        )
        optimizer = get_megatron_optimizer(model=self.module, config=optim_config_megatron)
        register_megatron_training_hooks(self.module, optimizer)
        return optimizer

    def _build_lr_scheduler(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer_param_scheduler

        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer, config=self.optimizer_config
        )
        return optimizer_scheduler

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )

    def initialize(self):
        self._build_tf_config()

        self.module = self._build_megatron_module()

        if self._qat_enabled and not self.engine_config.forward_only:
            from verl.utils.modelopt import apply_qat_to_modules

            self.module = apply_qat_to_modules(self.module, self._qat_config)

        self._maybe_enable_fused_kernels()
        self._maybe_enable_mem_efficient_ce()

        if self.model_config.mtp.enable:
            patch_engine_mtp(self.module, self.model_config)

        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
            self.to(device="cpu", model=self._is_offload_param, optimizer=False, grad=False)
            log_gpu_memory_usage("After offload model during init (forward_only)", logger=logger)
            return

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        full_reshardable = self.engine_config.dist_ckpt_optim_fully_reshardable
        mem_eff = self.engine_config.distrib_optim_fully_reshardable_mem_efficient

        tmp_config = OmegaConf.create(
            {
                "model": {"path": self.model_config.local_path},
                "megatron": {
                    "dist_ckpt_optim_fully_reshardable": full_reshardable,
                    "distrib_optim_fully_reshardable_mem_efficient": mem_eff,
                },
            }
        )

        role = "actor" if not self.is_value_model else "critic"

        self.checkpoint_mananager = MegatronCheckpointManager(
            config=tmp_config,
            checkpoint_config=self.checkpoint_config,
            model_config=self.model_config.hf_config,
            transformer_config=self.tf_config,
            role=role,
            model=self.module,
            arch=self.model_config.architectures[0],
            hf_config=self.model_config.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.model_config.share_embeddings_and_output_weights,
            processing_class=self.model_config.get_processor(),
            optimizer=self.optimizer,
            optimizer_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.optimizer_config.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            provider=self.provider,
            peft_cls=self.peft_cls,
            use_dist_checkpointing=self.engine_config.use_dist_checkpointing,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def train_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        return EngineEvalModeCtx(self, **kwargs)

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        self.optimizer.zero_grad()
        # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        for chunk in self.module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError("Megatron optimizer step failed. This should not happen")

        return grad_norm

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        self.lr_scheduler.step(1)
        return get_megatron_last_lr(self.optimizer)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_megatron_model_to_gpu(self.module, load_grad=grad)
            if optimizer and self.optimizer is not None:
                load_megatron_optimizer(self.optimizer)
        elif device == "cpu":
            if model:
                offload_megatron_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_megatron_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def get_data_parallel_rank(self):
        if self.engine_config.dynamic_context_parallel:
            # in order to let every dp-cp group has full data to split, we set dp=1
            return 0
        return mpu.get_data_parallel_rank()

    def get_data_parallel_size(self):
        if self.engine_config.dynamic_context_parallel:
            # in order to let every dp-cp group has full data to split, we set dp=1
            return 1
        return mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self):
        return mpu.get_data_parallel_group()

    def get_model_parallel_group(self):
        return mpu.get_model_parallel_group()

    def get_context_parallel_group(self):
        return mpu.get_context_parallel_group()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        origin_module_device = get_megatron_module_device(self.module)
        if self._is_offload_param or origin_module_device == "cpu":
            load_megatron_model_to_gpu(self.module, load_grad=True)
        self.checkpoint_mananager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            num_batches_divided_by = self.tf_config.microbatch_group_size_per_vp_stage
        else:
            num_batches_divided_by = None

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=True,
            min_num_micro_batch=None,
        )

        if num_batches_divided_by is not None:
            assert len(micro_batches) % num_batches_divided_by == 0, (
                f"micro_batches {micro_batches} must be divisible by num_batches_divided_by "
                f"{num_batches_divided_by} for megatron backend"
            )

        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)

        forward_backward_func = get_forward_backward_func()

        postprocess_micro_batch_func = partial(
            self.postprocess_micro_batch_func,
            forward_only=forward_only,
            loss_function=loss_function,
        )

        tu.assign_non_tensor(data, num_micro_batch=n_micro_batch)

        forward_step = partial(
            self.forward_step,
            logits_processor_func=loss_function,
            postprocess_micro_batch_func=postprocess_micro_batch_func,
        )

        enable_routing_replay = tu.get_non_tensor_data(data, key="enable_routing_replay", default=False)

        if enable_routing_replay:
            # Set to REPLAY mode: for R3 mode or actor update phase in R2 mode
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            if forward_only and self.engine_config.router_replay.mode == "R2":
                # In R2 mode, forward_only calls (e.g., compute_log_probs) need to record routing information
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.module,
            num_microbatches=n_micro_batch,
            seq_length=1,  # the communication shape is obtained via p2p comm
            micro_batch_size=1,  # the communication shape is obtained via p2p comm
            forward_only=forward_only,
        )

        if self.model_config.mtp.enable and mpu.is_pipeline_last_stage(ignore_virtual=True):
            # All CP ranks must participate in the all_reduce inside get_megatron_mtp_loss,
            # because save_loss_to_tracker uses avg_group=DP+CP group.
            # Only collect metrics on the src rank afterward.
            metrics = get_megatron_mtp_loss(n_micro_batch)
            if self.is_mp_src_rank_with_outputs():
                if "metrics" not in losses_reduced[0]:
                    losses_reduced[0]["metrics"] = {}
                losses_reduced[0]["metrics"].update(metrics)

        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                # config = self.actor_module[0].module.module.config
                vp_size = len(self.module)
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                bs = n_micro_batch
                topk_idx_td = reorder_and_merge_vpp_layers(
                    self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage
                )
            else:
                tensors = [tensor for nt in self.mini_layer_topk_idx_list for tensor in nt.unbind()]
                topk_idx_td = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            self.mini_layer_topk_idx_list = []

            layers_topk_idx = pp_gather(topk_idx_td.to(torch.uint8), self.tf_config)
            use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
            if use_dynamic_bsz and indices is not None:
                layers_topk_idx = restore_dynamic_batch(layers_topk_idx, indices)

        output = {}
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            output = postprocess_batch_func(output_lst=losses_reduced, indices=indices, data=data)
            if RouterReplayHelper.is_r2_record_action(self.tf_config):
                output["model_output"]["routed_experts"] = layers_topk_idx
        if enable_routing_replay:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()
        return output

    def get_per_tensor_param(self, base_sync_done=False, **kwargs):
        peft_config = None
        non_merge_lora_sync = self.peft_cls is not None and not self.model_config.lora.get("merge", False)
        adapter_only = base_sync_done and non_merge_lora_sync
        if non_merge_lora_sync:
            peft_config = build_peft_config_for_vllm(self.model_config.lora)
        # when lora adapter only, we only load adapter weights when base sync is done, otherwise load all weights
        load_megatron_model_to_gpu(self.module, load_grad=False, load_frozen_params=not adapter_only)
        if self.vanilla_bridge:
            per_tensor_param = self.bridge.export_weights(self.module)
        elif adapter_only:
            per_tensor_param = self.bridge.export_adapter_weights(self.module)
        else:
            per_tensor_param = (
                self.bridge.export_hf_weights(self.module, merge_adapter_weights=False)
                if non_merge_lora_sync
                else self.bridge.export_hf_weights(self.module)
            )
            if non_merge_lora_sync:
                per_tensor_param = add_base_layer_suffix(
                    per_tensor_param, model_type=self.model_config.hf_config.model_type
                )

        # QAT: process weights through QATWeightExporter for quantized weight sync to vLLM
        if self._qat_enabled:
            from verl.utils.modelopt import export_qat_weights

            per_tensor_param = export_qat_weights(per_tensor_param, self.module, self._qat_config.mode, self.bridge)

        return per_tensor_param, peft_config

    def disable_adapter(self) -> ContextManager:
        return self.peft_cls.disable_adapter(self.module)

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        raise NotImplementedError("postprocess_micro_batch_func must be implemented in subclass")


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    def prepare_model_inputs(self, batch: TensorDict):
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].to(bool)
        multi_modal_inputs = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []))

        routed_experts = batch.get("routed_experts", None)

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
            "routed_experts": routed_experts,
        }

    def prepare_model_outputs(self, output: dict, data: TensorDict):
        return output

    def forward_step(
        self, batch_iter: Iterator[TensorDict], model, logits_processor_func, postprocess_micro_batch_func
    ):
        batch: TensorDict = next(batch_iter)

        if self.engine_config.dynamic_context_parallel:
            # split the batch and give the sub-batches to each dp-cp group
            from verl.utils.megatron_utils import dynamic_cp_split_batch

            batch = dynamic_cp_split_batch(
                batch=batch,
                engine_config=self.engine_config,
                dp_size=mpu.get_data_parallel_world_size(),
                dp_rank=mpu.get_data_parallel_rank(),
            )

        batch = batch.to(get_device_id())
        use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        # When True, compute entropy without keeping logits in the autograd graph — the engine
        # skips the ~vocab_size × seq_len × dtype defensive clone normally needed to avoid
        # autograd version-counter conflicts between entropy.backward and log_prob.backward.
        # Only safe when entropy is purely for logging (entropy_coeff=0) or when the forward
        # is in eval/no_grad mode (compute_log_prob path).
        entropy_no_grad = tu.get_non_tensor_data(batch, key="entropy_no_grad", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(batch, key="calculate_sum_pi_squared", default=False)
        distillation_use_topk = tu.get_non_tensor_data(batch, key="distillation_use_topk", default=False)

        if calculate_sum_pi_squared and use_fused_kernels:
            raise NotImplementedError(
                "calculate_sum_pi_squared=True is not supported with use_fused_kernels=True: "
                "fused kernels do not materialize the full logits tensor needed for Σπ²."
            )
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = batch["temperature"]
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        local_cp_size = tu.get_non_tensor_data(data=batch, key="local_cp_size", default=None)
        loss_mask = model_inputs["loss_mask"]

        unwrapped_model = unwrap_model(model)
        if hasattr(unwrapped_model, "vp_stage"):
            vp_rank = unwrapped_model.vp_stage
        else:
            vp_rank = 0

        if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            layers_topk_idx = model_inputs["routed_experts"]
            set_router_replay_data(layers_topk_idx, None, self.tf_config, vp_rank)

        if pad_mode == DatasetPadMode.NO_PADDING:
            label = input_ids.clone()
        else:
            raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

        if use_fused_kernels:
            if not self.engine_config.use_remove_padding:
                logger.warning_once(
                    "Fused kernels require `use_remove_padding=True` for Megatron engine. Falling back to non-fused."
                )
                use_fused_kernels = False
            elif isinstance(temperature, torch.Tensor):
                if temperature.numel() != 1:
                    logger.warning_once(
                        "Fused kernels do not support per-sample temperature. Falling back to non-fused."
                    )
                    use_fused_kernels = False
                else:
                    temperature_value = float(temperature.item())
            else:
                temperature_value = float(temperature)

        if use_fused_kernels:
            from verl.models.mcore import get_mcore_forward_fused_model_engine_fn

            fused_forward_fn = get_mcore_forward_fused_model_engine_fn(self.model_config.hf_config)
            output = fused_forward_fn(
                model=model,
                input_ids=input_ids,
                labels=label,
                multi_modal_inputs=multi_modal_inputs,
                temperature=temperature_value,
                calculate_entropy=calculate_entropy,
                pad_token_id=self.model_config.tokenizer.pad_token_id,
            )
        else:
            # Memory-efficient CE (opt-in): the model is patched to return HIDDEN states and the
            # LM head + cross-entropy are fused via linear_cross_entropy, so the full
            # [tokens x vocab_shard] logits (and its grad) are never materialized. The kernel only
            # supports a single scalar temperature, so capture it here before per-token expansion.
            mem_efficient_ce = getattr(self, "_mem_efficient_ce", False)
            if mem_efficient_ce:
                if isinstance(temperature, torch.Tensor):
                    if temperature.numel() == 1:
                        mem_ce_temperature = float(temperature.item())
                    elif bool((temperature == temperature.reshape(-1)[0]).all().item()):
                        mem_ce_temperature = float(temperature.reshape(-1)[0].item())
                    else:
                        logger.warning_once(
                            "Memory-efficient CE does not support per-sample temperature; "
                            "falling back to the full-logits path for this batch."
                        )
                        mem_efficient_ce = False
                else:
                    mem_ce_temperature = float(temperature)
                if mem_efficient_ce and (calculate_sum_pi_squared or distillation_use_topk):
                    # These need the full logits tensor; the fused path never materializes it.
                    logger.warning_once(
                        "Memory-efficient CE cannot serve calculate_sum_pi_squared / distillation_use_topk; "
                        "falling back to the full-logits path for this batch."
                    )
                    mem_efficient_ce = False

            if not isinstance(temperature, torch.Tensor):
                temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

            temperature = temperature.to(torch.float32)
            assert temperature.shape[0] == input_ids.shape[0]
            temperature = verl_F.expand_as_nested(temperature, input_ids)  # (bsz, j1)
            from verl.models.mcore import get_mcore_engine_forward_fn

            forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)
            data_format = "thd" if self.engine_config.use_remove_padding else "bshd"

            def mem_efficient_logits_processor(output_orig, label, temperature):
                """Fused LM-head + CE on hidden states (see _hidden_only_GPTModel_forward).

                ``output_orig`` is a CausalLMOutputForPPO carrying ``hidden_states`` [b, s, H] and
                ``output_weight`` [vocab_shard, H]. Returns log_probs / entropy of shape [b, s],
                numerically equivalent to the vocab_parallel_log_probs_from_logits / entropy path.
                """
                from megatron.core import parallel_state as _mpu

                from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

                hidden = output_orig.hidden_states  # Megatron is sequence-first: [s, b, H]
                output_weight = output_orig.output_weight  # [vocab_shard, H]
                b, s = label.shape[:2]
                # Megatron hidden states are [seq, batch, H]. Transpose to [batch, seq, H] so the
                # row-major flatten below aligns token-for-token with labels [batch, seq]. The offline
                # validation used synthetic [b, s, H] tensors, so it never exercised this; the real
                # forward returns [s, b, H], which failed the old shape assert (and would misalign for
                # batch>1). Transpose is autograd-transparent (grad flows back to [s, b, H] hidden).
                if hidden.dim() == 3 and hidden.shape[0] == s and hidden.shape[1] == b:
                    hidden = hidden.transpose(0, 1).contiguous()  # [s, b, H] -> [b, s, H]
                assert hidden.shape[:2] == label.shape[:2], (hidden.shape, label.shape)

                hidden_flat = hidden.reshape(-1, hidden.shape[-1]).contiguous()
                labels_flat = label.reshape(-1).contiguous()

                tp_size = _mpu.get_tensor_model_parallel_world_size()
                # TP=1 uses the single-rank fast path (no all-reduce); TP>1 passes the TP group so the
                # kernel does the vocab-parallel max / sum / logprob all-reduces.
                pg = None if tp_size == 1 else _mpu.get_tensor_model_parallel_group()

                log_probs, entropy = linear_cross_entropy(
                    hidden_flat,
                    output_weight,
                    labels_flat,
                    mem_ce_temperature,
                    "none",
                    pg,
                )
                ret = {"log_probs": log_probs.reshape(b, s)}
                if calculate_entropy:
                    ret["entropy"] = entropy.reshape(b, s)
                return ret

            # When the model is patched for memory-efficient CE it ALWAYS returns hidden states
            # (a CausalLMOutputForPPO), never logits. So the full-logits branch below is only
            # reachable when the model was NOT patched. If mem-efficient CE is enabled but this
            # particular batch can't use the fused kernel (per-sample temperature / sum_pi_squared /
            # distillation topk), we must reconstruct logits from hidden here so the legacy branch
            # still works. This is the rare fallback; the common case stays fused.
            mem_efficient_ce_patched = getattr(self, "_mem_efficient_ce", False)

            def logits_processor(logits, label, temperature):
                if mem_efficient_ce:
                    return mem_efficient_logits_processor(logits, label, temperature)
                if mem_efficient_ce_patched:
                    # Model returns hidden; reconstruct sharded logits for the legacy path.
                    output_orig = logits
                    hidden = output_orig.hidden_states  # Megatron sequence-first: [s, b, H]
                    output_weight = output_orig.output_weight
                    b_, s_ = label.shape[:2]
                    # Match the fused path: transpose [s, b, H] -> [b, s, H] so reconstructed
                    # logits are [b, s, vocab] and pass the shape assert below.
                    if hidden.dim() == 3 and hidden.shape[0] == s_ and hidden.shape[1] == b_:
                        hidden = hidden.transpose(0, 1).contiguous()  # [s, b, H] -> [b, s, H]
                    logits = torch.matmul(hidden, output_weight.t())
                assert logits.shape[:2] == label.shape[:2]
                # avoid non-positive temperature such as padding
                temperature[temperature <= 0] = 1e-8
                assert torch.all(temperature > 0).item(), f"temperature tensor must be positive. Got {temperature}"
                logits.div_(temperature.unsqueeze(dim=-1).to(logits.dtype))
                ret = {}
                # sum_pi_squared is non-destructive — must run before vocab_parallel_entropy.
                if calculate_sum_pi_squared:
                    ret["sum_pi_squared"] = vocab_parallel_sum_pi_squared(logits)
                if calculate_entropy:
                    if entropy_no_grad:
                        # Monitor-only path: no autograd through entropy → no version-counter
                        # conflict with log_prob backward → no defensive clone of logits.
                        # vocab_parallel_entropy.forward is non-destructive on its input, and
                        # without autograd save_for_backward there is no in-place mutation
                        # later. logits stays usable for log_prob below.
                        with torch.no_grad():
                            # Chunked over tokens: monitoring-only entropy must not materialize the
                            # full [tokens × vocab_shard] copy (OOMs on a long FA row × 248k vocab).
                            entropy = vocab_parallel_entropy_chunked(logits.detach())
                        logits_bak = logits
                    else:
                        # Loss-contributing entropy: clone logits so log_prob.backward and
                        # entropy.backward don't fight over the same tensor's version
                        # counter (entropy.backward does in-place sub_/add_ on the saved
                        # logits; even though it restores the value, the bumped version
                        # would invalidate log_prob's saved view).
                        # # disable the hint until the fused_kernel is optimized for triton>=3.3
                        # if torch.distributed.get_rank() == 0:
                        #     logger.warning_once(
                        #         "For memory-efficient computation, enable fused kernels via "
                        #         "`actor_rollout_ref.model.use_fused_kernels=True`. "
                        #         "The current `clone()` operation ensures correctness but increases memory usage."
                        #     )
                        logits_bak = logits.clone()
                        entropy = vocab_parallel_entropy(logits)
                    ret["entropy"] = entropy
                else:
                    logits_bak = logits

                # logits_processor_func return tensors with shape (1, total_nnz/cp_size)
                if distillation_use_topk:
                    ret.update(logits_processor_func(student_logits=logits_bak, data=batch, data_format=data_format))
                log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                ret["log_probs"] = log_probs
                return ret

            logits_processor_args = {"label": label, "temperature": temperature, "loss_mask": loss_mask}

            output = forward_fn(
                model,
                input_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                vision_model=hasattr(self.model_config.hf_config, "vision_config"),
                pad_token_id=self.model_config.tokenizer.pad_token_id,
                data_format=data_format,
                mtp_enable_train=self.model_config.mtp.enable and self.model_config.mtp.enable_train,
                local_cp_size=local_cp_size,
            )

        # Router replay: record routing decisions for R2 mode
        if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
            merge_router_topk_indices(None, input_ids, self.mini_layer_topk_idx_list, self.tf_config, vp_rank)

        # Router replay: switch to backward replay mode for next backward pass
        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

        return output, partial(postprocess_micro_batch_func, data=batch, local_cp_size=local_cp_size)

    def postprocess_micro_batch_func(
        self, output, data: TensorDict, forward_only: bool, loss_function, local_cp_size=None
    ):
        # For memory efficiency
        # We move calculation of entropy to compute_log_probs, forward_only == True
        device = data["input_ids"].device
        model_output = self.prepare_model_outputs(output, data)

        if loss_function is not None:
            # TODO(baiyan): How to support hybrid context parallel with dp_group,
            # now the dp_group is not used, so just leave it as is, but what if we need to use it?
            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
            # scale loss by num_micro_batch because megatron will scale loss
            # by n_micro_batch inside pp schedule
            scaled_loss = loss * data["num_micro_batch"]
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device)
            scaled_loss = loss
            metrics = {}
        if local_cp_size is not None:
            # aggregate model_output by DP-CP groups
            from verl.utils.megatron_utils import dynamic_cp_merge_output

            model_output = dynamic_cp_merge_output(
                model_output,
                dp_size=mpu.get_data_parallel_world_size(),
                dp_rank=mpu.get_data_parallel_rank(),
                local_cp_size=local_cp_size,
            )

        output = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }

        # return loss and stats
        return scaled_loss, output


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngineWithLMHead):
    # for value head
    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        from verl.models.mcore import get_mcore_engine_forward_fn

        forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)

        output = forward_fn(
            model,
            input_ids,
            multi_modal_inputs,
            value_model=True,
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            data_format="thd" if self.engine_config.use_remove_padding else "bshd",
        )

        return output, partial(postprocess_micro_batch_func, data=batch)

    def prepare_model_outputs(self, output: dict | torch.Tensor, data: TensorDict):
        return {"values": output}
