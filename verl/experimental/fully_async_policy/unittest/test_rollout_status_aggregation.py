from types import SimpleNamespace

import pytest

from verl.experimental.fully_async_policy.detach_utils import aggregate_rollout_statuses


def sample(status):
    return SimpleNamespace(rollout_status=status)


def test_uses_exact_consumed_sample_metrics_not_last_rolling_snapshot():
    samples = [
        sample(
            {
                "count/total_generated_samples": 100,
                "reward/reward_mean": 0.25,
                "reward/judge_reward_mean": 0.5,
                "manager_metric_keys": ["reward/reward_mean", "reward/judge_reward_mean"],
                "manager_metrics_are_sample_local": True,
            }
        ),
        sample(
            {
                "count/total_generated_samples": 101,
                "reward/reward_mean": 0.75,
                "reward/judge_reward_mean": 1.0,
                "manager_metric_keys": ["reward/reward_mean", "reward/judge_reward_mean"],
                "manager_metrics_are_sample_local": True,
            }
        ),
    ]

    status = aggregate_rollout_statuses(samples)

    # Counters remain freshest-snapshot values; manager metrics describe this batch.
    assert status["count/total_generated_samples"] == 101
    assert status["reward/reward_mean"] == pytest.approx(0.5)
    assert status["reward/judge_reward_mean"] == pytest.approx(0.75)
    assert status["manager_metric_sample_coverage"] == 1.0
    assert "manager_metrics_are_sample_local" not in status


def test_suppresses_stale_manager_metrics_for_legacy_resume_batch():
    samples = [
        sample(
            {
                # Legacy queue entry: these are rolling-window values, not local.
                "reward/reward_mean": 0.95,
                "manager_metric_keys": ["reward/reward_mean"],
            }
        ),
        sample(
            {
                "count/total_generated_samples": 200,
                "reward/reward_mean": 0.10,
                "manager_metric_keys": ["reward/reward_mean"],
                "manager_metrics_are_sample_local": True,
            }
        ),
    ]

    status = aggregate_rollout_statuses(samples)

    assert "reward/reward_mean" not in status
    assert "manager_metric_keys" not in status
    assert status["count/total_generated_samples"] == 200
    assert status["manager_metric_sample_coverage"] == pytest.approx(0.5)


def test_rejects_empty_batch():
    with pytest.raises(ValueError, match="Empty rollout_samples"):
        aggregate_rollout_statuses([])
