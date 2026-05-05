# verl-b200-qwen35

A fork of [verl](https://github.com/volcengine/verl) with the patches needed to
run vanilla GRPO on **Qwen3.5** with **Megatron-Core + mbridge** on **B200**
(SM100 / Blackwell) GPUs, plus support for **external (off-cluster) judge
endpoints** as a reward model.

This repo carries no project-specific (e.g. meta-reasoning) logic — it is just
the upstream verl framework with the hybrid-attention / B200 / external-judge
patches applied. If a stock verl release works for your hardware, you probably
do not need this fork.

## What's in this repo

```
verl/                       # forked verl framework — the actual training code
docs/
  b200_qwen35_megatron_bringup.md
                            # detailed write-up of the env + verl patches
                            # required to make Qwen3.5-4B run on 8× B200
scripts/
  qwen35_4b_32k_colocate.sh # vanilla GRPO, hybrid (colocated) engine, 32k resp
  qwen35_4b_4k_mode1.sh     # fully-async Mode 1 (separate trainer/rollout pools)
  qwen35_4b_32k_mode4.sh    # fully-async Mode 4 (partial rollout + staleness)
verl/verl/experimental/fully_async_policy/async_experiments/FINAL_REPORT.md
                            # async-vs-colocate benchmark report (Qwen3.5
                            # 4B/9B/27B at 4k/32k/48k/64k responses on 8× B200)
```

## Install

The full B200 + transformers-5 + Megatron + mbridge environment is non-trivial.
See [`docs/b200_qwen35_megatron_bringup.md`](docs/b200_qwen35_megatron_bringup.md)
for the rebuild steps that were actually needed (`transformer_engine` 2.13 from
source, `flash_attn` 2.8.3 from source, `flash-linear-attention`, `LD_LIBRARY_PATH`
gotchas, the verl patches themselves).

For a stock GPU / non-Qwen3.5 setup, follow the upstream verl install in
[`verl/README.md`](verl/README.md) — the patches in this fork are additive and
should not affect non-Qwen3.5 / non-B200 runs.

After the env is in place:

```bash
cd verl
pip install -e .
```

## Run vanilla GRPO

```bash
# Edit HF_MODEL_PATH and TRAIN_FILE / TEST_FILE at the top of the script.
bash scripts/qwen35_4b_32k_colocate.sh
```

The launcher invokes `python -m verl.trainer.main_ppo` against the stock
`ppo_megatron_trainer.yaml` config with the Qwen3.5-specific Hydra overrides
documented in the bring-up doc.

## External judge (off-cluster reward model)

Set `reward.reward_model.backend=external` and point at any OpenAI-compatible
chat completions endpoint (vLLM, a Cloudflare Worker proxy, OpenAI/Anthropic).
See `verl/verl/trainer/config/reward/reward.yaml` for the
`reward_model.external.{base_url,api_key_env,timeout_seconds,max_concurrency}`
fields. The endpoint runs entirely off-cluster — no GPU pool is reserved and
no `Role.RewardModel` is registered.

## Async vs colocate guidance

For long-response RL workloads (32k+ response tokens), the fully-async pipeline
materially beats the hybrid colocate engine on Qwen3.5-family models. The
[`FINAL_REPORT.md`](verl/verl/experimental/fully_async_policy/async_experiments/FINAL_REPORT.md)
walks through the configurations swept and the recipe that won at each scale.
TL;DR: at 32k responses, Mode 4 with `trigger_parameter_sync_step=4` and
`staleness_threshold=0.5` ran ~1.6–1.8× faster than colocate at 4B/9B/27B.
