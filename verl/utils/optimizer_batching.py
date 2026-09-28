"""Optimizer minibatch planning and diagnostics (independent of async collection)."""

import random
import statistics

ROLE_NAMES = ("unknown", "mr", "e", "fa", "continue_mr", "terminal_mr", "malformed_mr", "truncated_mr")


def batching_divisor(config, dp_size, rollout_n):
    """Rows required for DP dispatch and the selected optimizer splitting mode."""
    if dp_size < 1:
        raise ValueError("Actor DP size must be positive")
    mode = config.get("train_minibatch_mode", "rows")
    if mode == "count":
        count = config.get("train_num_minibatches", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("train_num_minibatches must be a positive integer")
        if config.get("train_minibatch_rows") is not None:
            raise ValueError("Count mode requires train_minibatch_rows=null; select rows mode for a row override")
        return dp_size * count
    if mode != "rows":
        raise ValueError("train_minibatch_mode must be 'count' or 'rows'")
    rows = config.get("train_minibatch_rows")
    if rows is None:
        rows = config.ppo_mini_batch_size * rollout_n
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError("train_minibatch_rows must be a nonnegative integer or null")
    if rows == 0:
        return dp_size
    if rows % dp_size:
        raise ValueError("Optimizer minibatch rows must be divisible by actor DP size")
    return rows


def minibatch_rows(config, total_rows, dp_size, rollout_n):
    divisor = batching_divisor(config, dp_size, rollout_n)
    if total_rows <= 0 or total_rows % divisor:
        raise ValueError(f"Padded actor batch ({total_rows}) must be a positive multiple of {divisor}")
    if config.get("train_minibatch_mode", "rows") == "count":
        return total_rows // config.get("train_num_minibatches", 1)
    return total_rows if config.get("train_minibatch_rows") == 0 else divisor


def count_minibatch_order(workloads, loss_tokens, dp_size, count, seed, balance):
    """Random membership first; optional DP balance independently inside each mini.

    Returns rank-major indices for contiguous DP dispatch. Padding/zero-loss rows
    are spread round-robin so that every optimizer update has training tokens.
    ``balance`` accepts the usual equal-size partition function, or None.
    """
    size = len(workloads)
    if size == 0 or size % (dp_size * count):
        raise ValueError("Count-mode rows must be padded to DP * minibatch count")
    active = [i for i, tokens in enumerate(loss_tokens) if tokens > 0]
    inactive = [i for i, tokens in enumerate(loss_tokens) if tokens <= 0]
    if len(active) < count:
        raise ValueError("Not enough nonzero-loss rows for the requested optimizer minibatches")
    rng = random.Random(seed)
    rng.shuffle(active)
    rng.shuffle(inactive)
    minis = [(active + inactive)[j::count] for j in range(count)]
    ranks = [[] for _ in range(dp_size)]
    for mini in minis:
        rng.shuffle(mini)
        if balance is not None:
            parts = balance([workloads[i] for i in mini], k_partitions=dp_size, equal_size=True)
        else:
            width = len(mini) // dp_size
            parts = [list(range(d * width, (d + 1) * width)) for d in range(dp_size)]
        for d, part in enumerate(parts):
            # Any scheduling sort is confined to this optimizer minibatch.
            rows = [mini[i] for i in part]
            if balance is not None:
                rows.sort(key=lambda i: (workloads[i], i))
                rows = rows[::2] + rows[1::2][::-1]
            ranks[d].extend(rows)
    return [i for rank in ranks for i in rank]


def summarize_minibatches(minis, denominators, dp_size, requested_count=0):
    """Summarize actual DP-gathered rows: (prompt, response, loss, valid, role).

    Each mini is a list of DP-rank row lists. Padding participates in workload,
    but never in real-row/role statistics. Coefficients describe token-mean loss.
    """
    rows = [[row for rank in mini for row in rank] for mini in minis]
    counts = [sum(row[2] for row in mini) for mini in rows]
    if not counts or min(counts) <= 0 or min(denominators) <= 0:
        raise ValueError("All-masked optimizer minibatch encountered")
    k, total = len(rows), sum(counts)
    alpha = [total / k / d for d in denominators]
    metrics = {
        "num_minibatches": k, "dp_size": dp_size, "minibatch_rows": len(rows[0]),
        "requested_num_minibatches": requested_count, "loss_tokens": total,
        "real_rows": sum(bool(row[3]) for mini in rows for row in mini),
        "pad_rows": sum(not row[3] for mini in rows for row in mini),
        "token_count_max_min_ratio": max(counts) / min(counts),
        "token_count_cv": statistics.pstdev(counts) / statistics.mean(counts),
        "coefficient_max_min_ratio": max(alpha) / min(alpha),
    }
    by_role = {}
    continuing = []
    roles = sorted({r[4] for mini in rows for r in mini if r[3]})
    for j, (mini, token_count, a) in enumerate(zip(rows, counts, alpha, strict=True)):
        prefix = f"mini_{j}_"
        work = [sum(24576 * (r[0] + r[1]) + (r[0] + r[1]) ** 2 for r in rank) for rank in minis[j]]
        metrics.update({prefix + "loss_tokens": token_count, prefix + "loss_denominator": denominators[j],
                        prefix + "relative_token_coefficient": a,
                        prefix + "real_rows": sum(bool(r[3]) for r in mini),
                        prefix + "pad_rows": sum(not r[3] for r in mini),
                        prefix + "dp_workload_max_mean": max(work) / max(statistics.mean(work), 1)})
        for role in roles:
            subset = [r for r in mini if r[3] and r[4] == role]
            rp = prefix + role + "_"
            # Absence is a zero count/fraction, not a missing data point. Means
            # remain undefined for empty subsets.
            if not subset:
                metrics[rp + "row_count"] = 0
                metrics[rp + "loss_token_fraction"] = 0.0
                continue
            metrics.update({rp + "row_count": len(subset),
                            rp + "loss_token_fraction": sum(r[2] for r in subset) / token_count,
                            rp + "prompt_tokens_mean": statistics.mean(r[0] for r in subset),
                            rp + "response_tokens_mean": statistics.mean(r[1] for r in subset)})
            by_role.setdefault(role, []).extend([a] * len(subset))
            if role == "continue_mr":
                continuing.extend((r[1], a) for r in subset)
    for role, values in by_role.items():
        metrics[role + "_coefficient_row_mean"] = statistics.mean(values)
    if continuing:
        cutoff = statistics.median(r[0] for r in continuing)
        short = [a for length, a in continuing if length <= cutoff]
        long = [a for length, a in continuing if length > cutoff]
        metrics.update(mr_length_median=cutoff, mr_short_rows=len(short), mr_long_rows=len(long))
        if short and long:
            metrics["mr_short_long_coefficient_ratio"] = statistics.mean(short) / statistics.mean(long)
    return {"batching/" + key: value for key, value in metrics.items()}
