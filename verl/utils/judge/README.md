# LLM-as-judge reward computation

A reward manager (`llm_judge`) that scores rollouts by calling a hosted
OpenAI-compatible chat-completions endpoint (e.g., a Cloudflare Worker that
proxies to a model). Drop-in replacement for the rule-based managers
(`dapo`, `naive`, etc.) — selected via `reward.reward_manager.name=llm_judge`.

## Architecture

```
verl/utils/judge/
  client.py     JudgeClient   — aiohttp + retries + concurrency cap
  parser.py     parse_score() — <score>N</score> / "N/M" / trailing number
  prompt.py     render_template() — tag-based "<<var>>" substitution
  templates/    *.txt         — prompt templates (proof_rubric, proof_with_reference)

verl/utils/reward_score/
  math_proof.py compute_score() — orchestrates: render -> call -> parse -> normalize

verl/experimental/reward_loop/reward_manager/
  llm_judge.py  LLMJudgeRewardManager — owns one JudgeClient per worker;
                                         decodes prompt+response from data_item;
                                         resolves extra fields; returns reward dict.
```

The trainer flow is unchanged — the manager returns the standard
`{"reward_score": float, "reward_extra_info": {...}}` shape, and reward dispatches
through the same `RewardLoopWorker.compute_score_batch`.

## Quick start

Minimum config to switch on (Hydra overrides):

```yaml
reward:
  reward_manager:
    name: llm_judge
  reward_kwargs:
    judge:
      endpoint_url: https://<your-worker>.workers.dev/v1/chat/completions
      model: gpt-oss-120b
      api_key_env: LLM_JUDGE_API_KEY        # never inline keys
      max_score: 7                           # rubric maximum
      max_output_tokens: 512
      temperature: 0.0
      top_p: 1.0
      reasoning_effort: low                  # gpt-oss only
      thinking_mode: false                   # qwen only
      strip_thinking: true                   # default; see below
      max_concurrency: 32
      timeout_s: 60
      max_retries: 3
      on_error_score: 0.0
```

Set `LLM_JUDGE_API_KEY` in the launcher's environment. The rest of the launcher
(model, dataset, sampling) doesn't change.

## `strip_thinking` (default: `true`)

When the policy emits a `<think>...</think>` segment before its final answer,
we don't want the judge to grade the chain-of-thought. With
`strip_thinking: true`:

- The reward manager looks for the **last** `</think>` tag in the response and
  passes only the post-tag text to the judge.
- If no `</think>` is found, **the reward is forced to `on_error_score`**
  (default `0.0`) and the judge is *not* called. The fallback is **not** to
  grade the raw CoT.

If you don't want this (e.g., model has no thinking section), set
`strip_thinking: false`.

The flag `had_think_tag` (true/false/null) is added to `reward_extra_info`
for diagnostics.

## Customizing

Three independent extension points. You can mix and match.

### 1. Custom prompt template

Templates live in `verl/utils/judge/templates/`. The default `proof_rubric.txt`
substitutes four sentinels:

- `<<problem>>` — the question (extracted from `raw_prompt`)
- `<<response>>` — the policy response (after `</think>` stripping if enabled)
- `<<rubric>>` — `data_item.non_tensor_batch["reward_model"]["ground_truth"]`
- `<<max_score>>` — from config

**To plug in your own template**: write a plain-text file with these
sentinels (any subset is fine; missing ones simply aren't substituted) and
point `prompt_template` at either a built-in name (matching `templates/<name>.txt`)
or an absolute path:

```yaml
reward.reward_kwargs.judge.prompt_template: /path/to/my_template.txt
# or
reward.reward_kwargs.judge.prompt_template: proof_with_reference  # built-in
```

We use sentinel-based substitution rather than Jinja or Python format strings
because problem statements and rubrics are LaTeX-heavy and frequently contain
bare `{` / `}` / `$`.

### 2. Custom score format

The default parser (`verl/utils/judge/parser.py`) tries, in order:

1. `<score>N</score>` — preferred; templates instruct the judge to emit this.
2. `Final score: N` / `Score: N/M` — common natural-language forms.
3. Bare `N/M` fraction near the end of the output.
4. Last bare number in the text.

If your judge emits something else (e.g. JSON `{"score": 0.8}`, or a
single letter `A`/`B`/`C`/`D`/`E` mapped to scores), wire in your own parser
by writing a custom `compute_score`:

```python
# my_proj/my_judge.py
from verl.utils.reward_score.math_proof import compute_score as base_compute_score
import json, re

def parse_my_format(text):
    m = re.search(r'\{[^}]*"score"\s*:\s*([0-9.]+)[^}]*\}', text)
    return float(m.group(1)) if m else None

async def compute_score(*args, **kwargs):
    # Quickest path: monkey-patch the parser and delegate.
    import verl.utils.reward_score.math_proof as mp
    original = mp.parse_score
    mp.parse_score = lambda text, max_score=None: parse_my_format(text) or original(text, max_score)
    try:
        return await base_compute_score(*args, **kwargs)
    finally:
        mp.parse_score = original
```

Then point Hydra at it:

```yaml
reward:
  custom_reward_function:
    path: my_proj/my_judge.py
    name: compute_score
```

A cleaner long-term approach is to factor `parse_score` into a configurable
strategy; the current code is intentionally minimal.

### 3. Custom data fields (e.g., a Reference Solution)

Suppose your dataset has an extra column `reference_solution` (or nested
under `reward_model.reference_solution`) and you want a template like
`templates/proof_with_reference.txt` to receive it as `<<reference_solution>>`.

**Step 1** — point the manager at the source path with `extra_fields`:

```yaml
reward.reward_kwargs.judge:
  prompt_template: proof_with_reference        # provided in templates/
  extra_fields:
    reference_solution: reward_model.reference_solution
    # or, for a top-level non_tensor_batch column:
    # reference_solution: reference_solution
```

The keys on the left are the sentinel names that will appear as
`<<reference_solution>>` in the template; the values on the right are
**dotted paths into `data_item.non_tensor_batch`**. Missing paths are logged
and substituted with the empty string (training won't crash on a malformed row).

**Step 2** — write a template that uses the new sentinel. The bundled
`proof_with_reference.txt` is an example.

That's it — no manager subclass, no custom `compute_score`, just a config
addition and a template. Multiple extra fields are supported.

If you need richer extraction logic (e.g., concatenating two columns,
decoding a tokenized field, calling an external service), subclass
`LLMJudgeRewardManager` and override `_resolve_extra_fields`.

## Hosting the judge on Cloudflare

A typical setup:

1. **Cloudflare Worker** that exposes `POST /v1/chat/completions`. The
   Worker can either:
   - Forward to **Cloudflare Workers AI** (built-in models like
     `@cf/openai/gpt-oss-120b`, `@cf/qwen/qwen3-...`).
   - Forward to a remote OpenAI-compatible provider, with the API key stored
     as a Worker secret.
   - Hit a model running on a separate machine through a tunnel.

2. **Cloudflare AI Gateway** in front of the Worker for auth, rate-limiting,
   logging, and observability. Add gateway tokens to the `headers` config
   (e.g., `cf-aig-authorization: Bearer ...`).

3. **API key** kept in the env on the training node:
   ```bash
   export LLM_JUDGE_API_KEY=...
   ```
   Never commit it. The launcher reads it via `api_key_env`.

The client expects an OpenAI-compatible response shape:
`response["choices"][0]["message"]["content"]`. Most Worker-AI variants and
gateway integrations return this shape natively.

## Concurrency and rate limiting

- One `JudgeClient` per reward worker actor (one per training process). Each
  has its own `aiohttp.ClientSession` and a `Semaphore(max_concurrency)`.
- The reward loop already fans out per-sample via `asyncio.gather`, so each
  reward worker independently issues up to `max_concurrency` calls in flight.
- For a global QPS cap across workers, compose with the existing
  `RateLimitedRewardManager` wrapper (not enabled here by default).

## Failure handling

| Condition                           | Behavior                                         |
|-------------------------------------|--------------------------------------------------|
| Missing `</think>` (when stripping) | reward = `on_error_score`; judge **not** called  |
| HTTP 5xx / 429                      | retry with exponential backoff                   |
| HTTP 4xx (other)                    | raise — non-retryable client error               |
| All retries exhausted               | reward = `on_error_score`; error in extra_info   |
| Judge output unparseable            | reward = `on_error_score`; error in extra_info   |

`reward_extra_info` always includes `judge_error` (`None`,
`"missing_think_tag"`, `"parse_failure"`, or the HTTP error string), plus
`judge_text`, `judge_latency_s`, `judge_attempts`, `had_think_tag` for
debugging via wandb.

## Switching back to rule-based grading

```yaml
reward.reward_manager.name: dapo
```

That's the whole switch. The launcher's other reward kwargs (`overlong_buffer_cfg`,
etc.) are ignored by `llm_judge` and used by `dapo` — both can coexist in
the config.
