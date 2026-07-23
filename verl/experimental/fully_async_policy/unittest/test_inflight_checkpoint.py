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
"""Unit tests for in-flight sample checkpoint snapshot/restore (resume re-dispatch)."""
import os

import torch

from verl.experimental.fully_async_policy.inflight_checkpoint import (
    INFLIGHT_FILENAME,
    load_inflight_snapshot,
    save_inflight_snapshot,
)


def _records():
    # full_batch stands in for a DataProto — torch.save round-trips tensors like data.pt does.
    return [
        {"sample_id": "sample_0_1", "epoch": 0, "full_batch": torch.arange(4)},
        {"sample_id": "sample_0_2", "epoch": 0, "full_batch": torch.ones(2, 3)},
    ]


def test_save_load_roundtrip(tmp_path):
    recs = _records()
    path = str(tmp_path / INFLIGHT_FILENAME)
    save_inflight_snapshot(recs, path)

    assert not os.path.exists(f"{path}.tmp")  # atomic write leaves no temp behind
    loaded = load_inflight_snapshot(path)
    assert loaded is not None
    assert [r["sample_id"] for r in loaded] == ["sample_0_1", "sample_0_2"]
    assert [r["epoch"] for r in loaded] == [0, 0]
    assert torch.equal(loaded[0]["full_batch"], torch.arange(4))
    assert torch.equal(loaded[1]["full_batch"], torch.ones(2, 3))


def test_load_missing_returns_none(tmp_path):
    assert load_inflight_snapshot(str(tmp_path / "does_not_exist.pt")) is None


def test_load_corrupt_degrades_to_none(tmp_path):
    # A truncated / garbage file must not raise (would brick a resume) — degrade to no re-dispatch.
    path = str(tmp_path / INFLIGHT_FILENAME)
    with open(path, "wb") as f:
        f.write(b"not a valid torch save stream")
    assert load_inflight_snapshot(path) is None


def test_roundtrip_real_dataproto(tmp_path):
    # The real input is a DataProto with tensors AND a numpy object array in non_tensor_batch
    # (e.g. agent_name from prepare_single_generation_data); the bare-tensor tests above don't
    # exercise that pickle path. Round-trip a real one to cover it.
    import numpy as np
    from verl import DataProto

    dp = DataProto.from_single_dict(
        {
            "input_ids": torch.arange(12).reshape(2, 6),
            "agent_name": np.array(["single_turn_agent", "single_turn_agent"], dtype=object),
        }
    )
    recs = [{"sample_id": "resumed_sample_0_1", "epoch": 0, "full_batch": dp}]
    path = str(tmp_path / INFLIGHT_FILENAME)
    save_inflight_snapshot(recs, path)

    loaded = load_inflight_snapshot(path)
    assert loaded is not None
    assert loaded[0]["sample_id"] == "resumed_sample_0_1"
    lb = loaded[0]["full_batch"]
    assert torch.equal(lb.batch["input_ids"], torch.arange(12).reshape(2, 6))
    assert list(lb.non_tensor_batch["agent_name"]) == ["single_turn_agent", "single_turn_agent"]
