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
"""Unit tests for message-queue checkpoint snapshot/restore (rollout-queue saving).

Covers the disk round-trip (pure functions, no Ray) and the live `MessageQueue` actor
snapshot/restore round-trip (FIFO order + None-sentinel exclusion + counters).
"""
import os
import pickle

import pytest
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.message_queue import (
    QUEUE_FILENAME,
    MessageQueue,
    load_message_queue_snapshot,
    save_message_queue_snapshot,
)


def test_save_load_snapshot_roundtrip(tmp_path):
    blobs = [pickle.dumps({"i": i}) for i in range(5)]
    snap = {"samples": blobs, "total_produced": 7, "total_consumed": 2, "dropped_samples": 1}
    path = str(tmp_path / QUEUE_FILENAME)

    save_message_queue_snapshot(snap, path, required_samples=4, max_queue_size=10)

    # atomic write leaves no *.tmp behind, and the meta sidecar is present
    assert not os.path.exists(f"{path}.tmp")
    assert os.path.exists(f"{path}.meta.json")

    loaded, meta = load_message_queue_snapshot(path)
    assert loaded["samples"] == blobs
    assert loaded["total_produced"] == 7
    assert loaded["total_consumed"] == 2
    assert loaded["dropped_samples"] == 1
    assert meta["n_samples"] == 5
    assert meta["required_samples"] == 4
    assert meta["max_queue_size"] == 10


def test_load_missing_snapshot_returns_none(tmp_path):
    loaded, meta = load_message_queue_snapshot(str(tmp_path / "does_not_exist.pkl"))
    assert loaded is None
    assert meta is None


def test_load_corrupt_snapshot_degrades_to_none(tmp_path):
    # A truncated / garbage .pkl must not raise (would brick a resume) — degrade to empty.
    path = str(tmp_path / QUEUE_FILENAME)
    with open(path, "wb") as f:
        f.write(b"not a valid pickle stream")
    loaded, meta = load_message_queue_snapshot(path)
    assert loaded is None
    assert meta is None


def test_load_snapshot_missing_meta_sidecar(tmp_path):
    # Samples present but the .meta.json is gone: still load the samples (meta empty).
    blobs = [pickle.dumps({"i": i}) for i in range(3)]
    path = str(tmp_path / QUEUE_FILENAME)
    save_message_queue_snapshot({"samples": blobs}, path, required_samples=2, max_queue_size=5)
    os.remove(f"{path}.meta.json")
    loaded, meta = load_message_queue_snapshot(path)
    assert loaded["samples"] == blobs
    assert meta == {}


def test_queue_actor_snapshot_restore_roundtrip(tmp_path):
    ray = pytest.importorskip("ray")
    os.environ.pop("RAY_ADDRESS", None)  # never attach to a shared cluster from a unit test
    ray.init(num_cpus=8, include_dashboard=False, ignore_reinit_error=True)
    try:
        cfg = OmegaConf.create({})
        q = MessageQueue.remote(cfg, 10)
        blobs = [pickle.dumps({"i": i}) for i in range(5)]
        for b in blobs:
            assert ray.get(q.put_sample.remote(b)) is True
        # a None termination sentinel must NOT be captured (it would poison a restore)
        ray.get(q.put_sample.remote(None))

        snap = ray.get(q.snapshot.remote())
        assert snap["samples"] == blobs  # None excluded, FIFO order preserved
        assert snap["total_produced"] == 6

        path = str(tmp_path / QUEUE_FILENAME)
        save_message_queue_snapshot(snap, path, required_samples=4, max_queue_size=10)

        # fresh, empty queue resumes from the snapshot
        q2 = MessageQueue.remote(cfg, 10)
        loaded, meta = load_message_queue_snapshot(path)
        assert meta["required_samples"] == 4
        n = ray.get(q2.restore.remote(loaded))
        assert n == 5
        assert ray.get(q2.get_queue_size.remote()) == 5

        # FIFO preserved end-to-end: first restored sample is the first produced
        out, remaining = ray.get(q2.get_sample.remote())
        assert out == blobs[0]
        assert remaining == 4

        # Overflow: restoring more than the cap keeps the NEWEST (drops oldest), FIFO.
        q3 = MessageQueue.remote(cfg, 3)
        big = [pickle.dumps({"j": j}) for j in range(6)]
        assert ray.get(q3.restore.remote({"samples": big})) == 3  # actual landed count
        assert ray.get(q3.get_queue_size.remote()) == 3
        first, _ = ray.get(q3.get_sample.remote())
        assert first == big[3]  # the 3 oldest were dropped, big[3:] kept in order
    finally:
        ray.shutdown()
