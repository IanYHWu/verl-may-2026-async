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
"""Checkpoint persistence for the rollouter's in-flight sample inputs.

Fully-async training advances the dataloader at *dispatch*, so at checkpoint time the
problems still mid-generation (or waiting in the pending queue) sit behind the saved
cursor and would be silently skipped on resume. Snapshotting their INPUTS here lets the
rollouter re-dispatch (regenerate) them on resume — the mid-decode state itself cannot be
serialized, but no problem is dropped.

A record is ``{"sample_id": str, "epoch": int, "full_batch": DataProto}`` where
``full_batch`` is the un-generated input. ``torch.save`` handles the DataProto tensors,
matching how the sibling ``data.pt`` is written. (Standard pickle, unlike the message
queue's ``ray.cloudpickle`` — fine because a dataloader input is plain CPU tensors +
numpy arrays; a hypothetical cloudpickle-only object would just degrade to
skip-re-dispatch via the load guard below, not crash.)
"""
import os

import torch

# Filename of the in-flight snapshot inside a ``global_step_{N}/`` checkpoint dir.
INFLIGHT_FILENAME = "inflight_samples.pt"


def save_inflight_snapshot(records: list, path: str) -> None:
    """Persist the in-flight input records to ``path``, atomically (``*.tmp`` + os.replace)."""
    tmp = f"{path}.tmp"
    torch.save(records, tmp)
    os.replace(tmp, path)


def load_inflight_snapshot(path: str) -> list | None:
    """Load records written by ``save_inflight_snapshot``. Returns ``None`` when the file is
    absent (old checkpoint / disabled) or unreadable (corrupt) — the caller then re-dispatches
    nothing and resumes exactly as before this feature."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:  # corrupt / truncated — degrade to no re-dispatch
        print(f"[inflight-ckpt] WARNING: could not read in-flight snapshot at {path} ({e}); skipping re-dispatch")
        return None
