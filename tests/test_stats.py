import numpy as np
import pytest

from xjevboost import FakeProvider, XJevBoostClassifier
from xjevboost.stats import summarize_traces


def test_routing_cache_and_token_reductions_have_separate_denominators():
    traces = [dict(calls=1, provider_calls=1, stop_reason="leaf", accounted_tokens=20,
                   actual_tokens=20, new_actual_tokens=20, full_view_estimated_tokens=100),
              dict(calls=2, provider_calls=0, stop_reason="token_limit", accounted_tokens=40,
                   actual_tokens=40, new_actual_tokens=0, full_view_estimated_tokens=100)]
    stats = summarize_traces(traces, max_depth=3, n_views=10)
    assert stats["calls"] == 3
    assert stats["provider_calls"] == 1
    assert stats["cache_hits"] == 2
    assert stats["routing_reduction_vs_depth_pct"] == 50
    assert stats["routing_reduction_vs_all_views_pct"] == 85
    assert stats["provider_call_reduction_vs_all_views_pct"] == 95
    assert stats["token_reduction_vs_full_estimate_pct"] == 70
    assert stats["cache_hit_pct"] == pytest.approx(200 / 3)


def test_unknown_usage_and_no_full_estimate_stay_unknown():
    traces = [dict(calls=1, provider_calls=1, stop_reason="leaf", accounted_tokens=30,
                   actual_tokens=None, new_actual_tokens=None)]
    stats = summarize_traces(traces, max_depth=5, n_views=1)
    assert stats["depth_call_upper_bound"] == 1
    assert stats["actual_tokens"] is None
    assert stats["token_reduction_vs_full_estimate_pct"] is None
    assert summarize_traces([], max_depth=2, n_views=3)["cache_hit_pct"] is None


def test_last_run_stats_and_training_reduction_are_consistent():
    X = np.arange(80).reshape(40, 2)
    y = np.arange(40) % 2
    clf = XJevBoostClassifier(provider=FakeProvider(), n_views=3, max_depth=1,
                             min_samples_leaf=2, random_state=42, calibration_fraction=.3).fit(X, y)
    training = clf.training_stats_
    dense = training["query_view_upper_bound"]
    assert training["provider_call_reduction_pct"] == pytest.approx(100 * (1 - training["provider_calls"] / dense))
    assert training["query_view_pairs_skipped"] + training["unique_query_view_pairs"] == dense
    assert training["calls_saved_by_cache"] + training["provider_calls"] == training["unique_query_view_pairs"]
    traces = clf.predict_with_trace(X + 100)
    first = clf.inference_stats_.copy()
    assert first["calls"] == len(X)
    calls_before = clf.provider.calls
    summary = clf.summarize_traces(traces)
    assert summary["provider_calls"] == first["provider_calls"]
    assert clf.provider.calls == calls_before
    clf.predict(X + 100)
    assert clf.inference_stats_["provider_calls"] == 0
    assert clf.inference_stats_["routing_reduction_vs_all_views_pct"] == first["routing_reduction_vs_all_views_pct"]
    assert clf.inference_stats_["cache_hit_pct"] == 100
