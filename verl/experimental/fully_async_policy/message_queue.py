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
import json
import logging
import os
import pickle
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Filename of the message-queue snapshot inside a `global_step_{N}/` checkpoint dir.
# Written next to verl's `actor/` and the rollouter's `data.pt` so a resume restores the
# completed-but-untrained rollouts that were waiting in the queue at checkpoint time.
QUEUE_FILENAME = "message_queue.pkl"


def save_message_queue_snapshot(snapshot: dict, path: str, *, required_samples: int, max_queue_size: int) -> None:
    """Persist a `MessageQueue.snapshot()` dict to `path`, atomically (`*.tmp` then
    `os.replace`), plus a small `.meta.json` sidecar for the resume integrity check.

    The samples are already cloudpickled `RolloutSample` bytes, so we just pickle the
    list of blobs; the cheap counters/config go in the JSON sidecar so resume can
    validate without unpickling the (potentially large) sample payload."""
    samples = snapshot["samples"]
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

    meta = {
        "n_samples": len(samples),
        "required_samples": int(required_samples),
        "max_queue_size": int(max_queue_size),
        "total_produced": snapshot.get("total_produced"),
        "total_consumed": snapshot.get("total_consumed"),
        "dropped_samples": snapshot.get("dropped_samples"),
    }
    meta_tmp = f"{path}.meta.json.tmp"
    with open(meta_tmp, "w") as f:
        json.dump(meta, f)
    os.replace(meta_tmp, f"{path}.meta.json")


def load_message_queue_snapshot(path: str) -> tuple[dict | None, dict | None]:
    """Load a snapshot written by `save_message_queue_snapshot`. Returns
    `(snapshot_dict, meta)`, or `(None, None)` when no snapshot exists at `path`
    (old checkpoint, or snapshotting was disabled) — the caller then resumes with an
    empty queue, exactly as before this feature."""
    if not os.path.exists(path):
        return None, None
    meta_path = f"{path}.meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    with open(path, "rb") as f:
        samples = pickle.load(f)
    snapshot = {
        "samples": samples,
        "total_produced": meta.get("total_produced"),
        "total_consumed": meta.get("total_consumed"),
        "dropped_samples": meta.get("dropped_samples"),
    }
    return snapshot, meta


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)

        self.val_queue = deque()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        print(f"[MessageQueue] initialized with max_queue_size={max_queue_size}")

    async def put_sample(self, sample: Any) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        async with self._lock:
            # If queue is full, remove the oldest sample (rarely happens)
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self.queue.append(sample)
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            if is_drop:
                return False
            return True

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample
            data = self.queue.popleft()
            self.total_consumed += 1
            return data, len(self.queue)

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "max_queue_size": self.max_queue_size,
            }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def snapshot(self) -> dict[str, Any]:
        """Point-in-time copy of the queue for checkpointing. Returns the raw per-sample
        blobs (already cloudpickled `RolloutSample` bytes) plus counters. Held under the
        lock, so no `put`/`get` interleaves and the snapshot is internally consistent.
        The `None` termination sentinel is excluded so a restore never poisons the
        trainer with a premature stop signal."""
        async with self._lock:
            return {
                "samples": [s for s in self.queue if s is not None],
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
            }

    async def restore(self, snapshot: dict[str, Any]) -> int:
        """Repopulate the queue from a checkpoint `snapshot`, preserving FIFO order.
        Called once during resume init before producers/consumers start, so there is no
        contention — the lock is held only for consistency. `deque(maxlen=...)` drops the
        oldest if the snapshot exceeds the current cap. Returns the number restored."""
        async with self._lock:
            samples = snapshot.get("samples", [])
            self.queue.extend(samples)
            for key in ("total_produced", "total_consumed", "dropped_samples"):
                if snapshot.get(key) is not None:
                    setattr(self, key, snapshot[key])
            self._consumer_condition.notify_all()
            print(f"[MessageQueue] restored {len(samples)} samples from checkpoint (queue_size={len(self.queue)})")
            return len(samples)

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any) -> bool:
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(sample)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def snapshot(self) -> dict[str, Any]:
        """Snapshot the queue for checkpointing (async)"""
        future = self.queue_actor.snapshot.remote()
        return await asyncio.wrap_future(future.future())

    def restore_sync(self, snapshot: dict[str, Any]) -> int:
        """Restore a queue snapshot (sync — called from the rollouter's sync
        load_checkpoint, before the event loop is serving producers/consumers)"""
        return ray.get(self.queue_actor.restore.remote(snapshot))

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())
