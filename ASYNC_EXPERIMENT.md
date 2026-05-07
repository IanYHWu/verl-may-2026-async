# Async-training verification experiments

## Why these experiments exist

The previous fork (`verl-may-2026`) had a half-finished migration in its
async stack: the new `ServerAdapter`-based vLLM rollout was wired up,
but the per-step weight-sync code was never updated to match, so for
every async-mode launcher the trainer's gradient updates were silently
**not transferred** to the rollouter. Symptom: reward bouncing in the
cold-policy band, `rollout_corr/log_ppl_diff` exploding, no learning.

This new repo (`verl-may-2026-async`) is upstream verl HEAD with the
fork's small surgical patches re-applied (see
[`docs/upstream_verl_port_notes.md`](docs/upstream_verl_port_notes.md)).
Upstream has a working `CheckpointEngineManager` that performs the
NCCL + IPC transfer correctly to `ServerAdapter`. We need to verify
that on this hardware (H100s in particular, since most upstream
validation has been on B200) the async path actually trains.

## Goals

1. **Colocate works as a known-good baseline.** Run
   `scripts/qwen3_4b_inst_acemath_colocate.sh` end-to-end and confirm
   reward goes up monotonically (we previously saw 0.34 → 0.45 over
   ~30 steps on the prior fork). This is the reference curve.

2. **Mode 1 async actually trains** (was broken on the prior fork).
   Run `scripts/qwen3_4b_inst_acemath_async_mode1.sh` (Mode 1 = on-policy
   pipeline: `trigger_parameter_sync_step=1`, `staleness_threshold=0`,
   `partial_rollout=False`). Confirm:
   - `rollout_corr/log_ppl_diff` stays ~0.004 throughout (does NOT
     grow into the 0.01–0.1 range we saw in the broken fork — that
     is the smoking-gun for broken sync).
   - Reward trajectory tracks colocate within ~5 pp at matched step
     counts.
   - No `EngineDeadError` / `TimeoutError: RPC call to sample_tokens
     timed out` (those signaled vLLM workers stalled because nothing
     was syncing weights and prefill backlogs grew).

3. **Qwen3.5 can be trained.** The custom Qwen3.5 hybrid attention
   (dense + GatedDeltaNet) plus MTP plus MRoPE is the messiest
   architecture in this stack and is what motivated the fork's
   custom patches in the first place. Running
   `scripts/qwen35_4b_32k_colocate.sh` end-to-end (B200 reference)
   confirms the patches in
   [`docs/upstream_verl_port_notes.md`](docs/upstream_verl_port_notes.md)
   (a, b, e, f) hold against upstream HEAD.

## Setup

### Hardware

- **H100 (FLAME cluster)**: use the launchers in `scripts/` as-is.
  Tested values: `max_response_length=16384`, `gpu_memory_utilization=0.8`,
  `train_batch_size=64`, `ppo_mini_batch_size=32`, `rollout.n=8`.
- **B200**: read the [Install + Limitations](README.md#install) sections in the top-level README
  before doing anything. It has the verified version pins, the
  build-time CPATH/LIBRARY_PATH trick for `transformer_engine_torch`,
  the runtime `LD_LIBRARY_PATH` workaround for cudnn `dlopen`, and
  the Qwen3.5 hybrid attention / BSHD requirements. Mode 4 with
  longer responses (`max_response_length=32768+`,
  `gpu_memory_utilization=0.7`) is appropriate on B200.

### Conda env

Use the existing `verl_megatron` env (now repointed at this repo via
`pip install --no-deps -e .`). Don't reinstall the ABI-pinned stack
(torch 2.10 / TE 2.13 / flash-attn 2.8.3 / mbridge git / vLLM 0.19.1 /
megatron-core 0.16.1 / FLA / tilelang) — see the B200 doc for what
each is for.

### Wandb

All async-debug runs log to **wandb project `async_verl_debug`**.
The launchers' `PROJECT_NAME` defaults already point there. Wandb
auth is already configured in `~/.netrc` for the active account.

### Hydra outputs

This repo has `outputs -> /tmp/ianwu/hydra_outputs` symlinked at
the repo root so Hydra run dirs don't blow out the home quota.

## Running

```bash
# from anywhere — the scripts cd to ${REPO_ROOT}
bash /home/ianwu/code/verl-may-2026-async/scripts/qwen3_4b_inst_acemath_colocate.sh
# or
bash /home/ianwu/code/verl-may-2026-async/scripts/qwen3_4b_inst_acemath_async_mode1.sh
```

Tail the log; the key things to watch are:

- `[FullyAsyncTrainer] Checkpoint manager initialized` — the upstream
  sync path is wired up.
- `timing_s/update_weights:` non-zero in step metrics — actual NCCL
  transfer is happening per step.
- `rollout_corr/log_ppl_diff:` stays ≲ 0.005 over many steps — the
  rollouter is receiving fresh weights.
- `critic/score/mean:` trends up over 20–30 steps — reward learning
  is real.

## Comparison protocol

For each new experiment, log to `async_verl_debug` and tag the run
with the workload (`qwen3_4b_inst_acemath`,
`qwen35_4b_dapo_math_17k`) and topology (`colocate` or `mode1`).
Compare `critic/score/mean` chunks of 5 (or 10) steps:

| Steps | Colocate | Mode 1 | Δ |
|---|---|---|---|
| 1–10 | ... | ... | ... |
| 11–20 | ... | ... | ... |
| 21–30 | ... | ... | ... |

Mode 1 should land within ~5 pp of colocate. If it lags much more
than that, check `log_ppl_diff` first — that's the canonical
"is sync working" health metric.

## Known good / known broken

| Path | Hardware | Status |
|---|---|---|
| Colocate (`hybrid_engine=True`) | H100 8×, B200 8× | Validated on prior fork |
| Mode 1 (`hybrid_engine=False`, `trigger=1`, `staleness=0`) | H100 8× (4T+4R) | **What we're verifying here** — was broken on prior fork |
| Mode 4 (`trigger=4`, `staleness=0.5`, `partial_rollout=True`) | B200 8× (4T+4R) | Validated upstream-style; not yet on this repo |
| Qwen3.5 + GatedDeltaNet | B200 8× | Validated on prior fork (see B200 doc) |
| Qwen3.5 dense (e.g. 2B) with MTP | H100 8× | Needs `+actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=null` |
