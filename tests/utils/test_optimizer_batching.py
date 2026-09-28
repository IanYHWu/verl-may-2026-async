"""CPU tests for variable-row optimizer batches and actual worker iteration."""

import ast
import time
from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from codetiming import Timer
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric.utils import Metric
from verl.utils.optimizer_batching import batching_divisor, count_minibatch_order, minibatch_rows, summarize_minibatches
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions


def config(**kwargs):
    return OmegaConf.create(dict(train_minibatch_mode="count", train_num_minibatches=kwargs.pop("k", 1),
                                 train_minibatch_rows=None, ppo_mini_batch_size=32, **kwargs))


@pytest.mark.parametrize("n,k,padded,rows", [(1652, 2, 1656, 828), (1245, 1, 1248, 1248), (1652, 3, 1656, 552)])
def test_count_padding_and_resolved_size(n, k, padded, rows):
    cfg = config(k=k)
    divisor = batching_divisor(cfg, 4, 8)
    assert (n + divisor - 1) // divisor * divisor == padded
    assert minibatch_rows(cfg, padded, 4, 8) == rows


def test_legacy_rows_and_full_batch():
    cfg = config()
    cfg.train_minibatch_mode = "rows"
    assert batching_divisor(cfg, 4, 8) == 256
    cfg.train_minibatch_rows = 640
    assert minibatch_rows(cfg, 1920, 4, 8) == 640
    cfg.train_minibatch_rows = 0
    assert batching_divisor(cfg, 4, 8) == 4
    assert minibatch_rows(cfg, 1652, 4, 8) == 1652


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_count(k):
    with pytest.raises(ValueError, match="positive integer"):
        batching_divisor(config(k=k), 4, 8)


def test_conflicting_modes_and_empty_minibatches():
    cfg = config()
    cfg.train_minibatch_rows = 640
    with pytest.raises(ValueError, match="requires train_minibatch_rows=null"):
        batching_divisor(cfg, 4, 8)
    with pytest.raises(ValueError, match="Not enough"):
        count_minibatch_order([10] * 12, [1] + [0] * 11, 2, 3, 42, None)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("balanced", [False, True])
def test_membership_survives_dp_dispatch(k, balanced):
    n, dp = 480, 4
    work = [24576 * (i + 1) + (i + 1) ** 2 for i in range(n)]
    loss = [i + 1 for i in range(n - 13)] + [0] * 13
    balance = get_seqlen_balanced_partitions if balanced else None
    order = count_minibatch_order(work, loss, dp, k, 73, balance)
    assert sorted(order) == list(range(n))
    assert order == count_minibatch_order(work, loss, dp, k, 73, balance)
    assert order != count_minibatch_order(work, loss, dp, k, 74, balance)
    width = n // dp // k
    membership = []
    for j in range(k):
        rows = [i for rank in range(dp) for i in order[rank * n // dp + j * width:rank * n // dp + (j + 1) * width]]
        assert len(rows) == n // k
        assert sum(loss[i] > 0 for i in rows) >= (n - 13) // k
        membership.append(set(rows))
    # Balancing changes rank assignment/order, never optimizer membership.
    unbalanced = count_minibatch_order(work, loss, dp, k, 73, None)
    for j in range(k):
        expected = {
            i for rank in range(dp)
            for i in unbalanced[rank * n // dp + j * width:rank * n // dp + (j + 1) * width]
        }
        assert membership[j] == expected


def test_metrics_use_actual_tokens_and_exclude_padding():
    # Two ranks per minibatch. Padding costs compute but contributes zero loss.
    minis = [[[[10, 2, 2, 1, "continue_mr"]], [[10, 2, 0, 0, "continue_mr"]]],
             [[[10, 6, 6, 1, "continue_mr"]], [[10, 4, 4, 1, "fa"]]]]
    result = summarize_minibatches(minis, [2, 10], 2)
    assert result["batching/coefficient_max_min_ratio"] == 5
    assert result["batching/mini_0_relative_token_coefficient"] == 3
    assert result["batching/mr_short_long_coefficient_ratio"] == 5
    assert result["batching/real_rows"] == 3
    assert result["batching/pad_rows"] == 1
    assert result["batching/mini_0_fa_row_count"] == 0
    assert result["batching/mini_0_fa_loss_token_fraction"] == 0
    assert "batching/mini_0_fa_response_tokens_mean" not in result
    assert result["batching/continue_mr_coefficient_row_mean"] == 1.8
    assert all(key.count("/") == 1 for key in result)
    assert summarize_minibatches(minis, [6, 6], 2)["batching/coefficient_max_min_ratio"] == 1


def worker_method():
    # Execute the production method without importing GPU/Ray worker construction.
    path = Path(__file__).resolve().parents[2] / "verl/workers/engine_workers.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingWorker")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train_mini_batch")
    method.decorator_list = []
    namespace = dict(
        torch=torch, tu=tu, time=time, chain=chain, Timer=Timer, Metric=Metric,
        TensorDict=TensorDict, NonTensorData=NonTensorData, append_to_dict=append_to_dict,
        maybe_fix_3d_position_ids=lambda data: None,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["train_mini_batch"]


@pytest.mark.parametrize("shuffle", [False, True])
def test_worker_logs_actual_iterator_and_steps(shuffle):
    seen = []
    engine = SimpleNamespace(get_data_parallel_size=lambda: 1, get_data_parallel_rank=lambda: 0,
                             get_data_parallel_group=lambda: None, train_mode=lambda **kw: nullcontext(),
                             is_mp_src_rank_with_outputs=lambda: True)
    def train_batch(data):
        seen.append(data["row_id"].tolist())
        assert "batching_row_stats" not in data
        tu.assign_non_tensor(data, batch_num_tokens=float(data["loss_mask"].sum()))
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"grad_norm": 0.5}})
    worker = SimpleNamespace(engine=engine, train_batch=train_batch, optimizer_config=SimpleNamespace(clip_grad=1.0))
    lengths = torch.arange(1, 13)
    data = TensorDict({"row_id": torch.arange(12), "loss_mask": lengths[:, None],
                       "batching_row_stats": torch.stack([lengths, lengths, lengths, torch.ones(12, dtype=torch.long),
                                                           torch.full((12,), 4)], dim=-1)}, batch_size=[12])
    tu.assign_non_tensor(data, mini_batch_size=4, epochs=2, seed=42, dataloader_kwargs={"shuffle": shuffle},
                         monitor_batching=True, batching_requested_count=3)
    result = tu.get(worker_method()(worker, data), "metrics")
    assert len(seen) == 6
    assert sorted(sum(seen[:3], [])) == list(range(12))
    assert sorted(sum(seen[3:], [])) == list(range(12))
    assert result["batching/num_minibatches"] == 3
    for j, indices in enumerate(seen[:3]):
        assert result[f"batching/mini_{j}_loss_denominator"] == sum(i + 1 for i in indices)
    assert result["batching/epoch_1_loss_tokens"] == 78
    assert result["batching/mini_0_grad_clipped"] == 0
    assert result["batching/epoch_1_mini_0_grad_clipped"] == 0
    assert "batching/mini_3_grad_clipped" not in result
    assert all(key.count("/") == 1 for key in result if key.startswith("batching/"))


@pytest.mark.parametrize("balanced", [False, True])
def test_controller_dispatch_preserves_rows_metadata_and_caller(balanced):
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    cfg = OmegaConf.create({
        "actor_rollout_ref": {
            "actor": {"train_minibatch_mode": "count", "train_num_minibatches": 3,
                      "train_minibatch_rows": None, "batching_metrics": True, "ppo_epochs": 1,
                      "data_loader_seed": None, "shuffle": True, "calculate_entropy": False,
                      "entropy_coeff": 0, "loss_agg_mode": "token-mean"},
            "rollout": {"n": 8, "temperature": 1, "multi_turn": {"enable": False}},
        }, "trainer": {"balance_batch": balanced},
    })
    n, real, dp = 24, 20, 2
    tokens = torch.arange(n)[:, None].expand(n, 4).clone() + 1
    response_mask = torch.ones(n, 2, dtype=torch.long)
    response_mask[real:] = 0
    batch = DataProto.from_dict(tensors={
        "input_ids": tokens, "prompts": tokens[:, :2], "responses": tokens[:, 2:],
        "position_ids": torch.arange(4).expand(n, 4), "attention_mask": torch.ones_like(tokens),
        "response_mask": response_mask, "row_id": torch.arange(n),
        "batching_valid_row": torch.arange(n) < real,
    })
    original = batch.batch["row_id"].clone()
    captured = []
    def update_actor(td):
        captured.append(td)
        assert tu.get(td, "mini_batch_size") == 8
        assert tu.get(td, "global_batch_size") == 8
        assert tu.get(td, "dataloader_kwargs") == {"shuffle": False}
        assert tu.get(td, "seed") == 49
        assert td["input_ids"].is_nested
        assert td["batching_row_stats"][:, 3].sum() == real
        assert torch.equal(td["batching_row_stats"][:, 3].bool(), td["row_id"] < real)
        for j in range(3):
            idx = [r * 12 + i for r in range(dp) for i in range(j * 4, (j + 1) * 4)]
            assert td["response_mask"][idx].sum() > 0
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"mfu": 0, "batching/loss_tokens": 40}})
    trainer = SimpleNamespace(config=cfg, global_steps=7, actor_rollout_wg=SimpleNamespace(update_actor=update_actor),
                              _get_dp_size=lambda *args: dp)
    result = RayPPOTrainer._update_actor(trainer, batch)
    assert torch.equal(batch.batch["row_id"], original)
    assert "batching_row_stats" not in batch.batch
    assert result.meta_info["metrics"]["batching/optimizer_steps"] == 3
    assert result.meta_info["metrics"]["batching/loss_tokens"] == 40
    assert "actor/batching/loss_tokens" not in result.meta_info["metrics"]


def test_async_padding_divisor_count_mode():
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    cls = FullyAsyncTrainer.__ray_actor_class__
    trainer = object.__new__(cls)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"actor": {
        "train_minibatch_mode": "count", "train_num_minibatches": 3, "train_minibatch_rows": None,
    }, "rollout": {"n": 8}}})
    trainer.actor_wg = object()
    trainer.use_critic = False
    trainer._get_dp_size = lambda *args: 4
    cls._init_train_batch_divisor(trainer)
    assert trainer.actor_batch_divisor == 12
