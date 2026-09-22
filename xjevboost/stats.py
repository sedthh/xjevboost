"""Cost summaries with explicit reference baselines; no provider calls."""
from collections import Counter


def summarize_traces(traces, *, max_depth, n_views, _group=True):
    """Aggregate recorded traces. Reductions are percentages, not fractions.

    Full-view token comparisons use a hypothetical estimate, not a benchmark
    request. The all-views/depth references need not be feasible under budgets.
    Cached usage is retained in logical policy cost and excluded from new cost.
    """
    n = len(traces)
    calls = sum(t["calls"] for t in traces)
    requests = sum(t["provider_calls"] for t in traces)
    depth_reference = n * min(max_depth, n_views)
    all_views = n * n_views

    def total_if_known(key):
        values = [t.get(key) for t in traces]
        return sum(values) if all(v is not None for v in values) else None

    def reduction(actual, reference):
        return 100 * (1 - actual / reference) if reference else None

    full = total_if_known("full_view_estimated_tokens")
    accounted = sum(t["accounted_tokens"] for t in traces)
    actual = total_if_known("actual_tokens")
    result = {
        "n_queries": n, "calls": calls, "provider_calls": requests,
        "cache_hits": calls - requests,
        "average_calls": calls / n if n else 0.0,
        "average_actual_tokens_per_query": actual / n if actual is not None and n else None,
        "average_actual_tokens_per_view": actual / calls if actual is not None and calls else None,
        "average_accounted_tokens_per_query": accounted / n if n else None,
        "average_accounted_tokens_per_view": accounted / calls if calls else None,
        "call_count_distribution": dict(sorted(Counter(t["calls"] for t in traces).items())),
        "stop_reasons": dict(Counter(t["stop_reason"] for t in traces)),
        "depth_call_upper_bound": depth_reference,
        "all_views_call_count": all_views,
        "routing_reduction_vs_depth_pct": reduction(calls, depth_reference),
        "routing_reduction_vs_all_views_pct": reduction(calls, all_views),
        "cache_hit_pct": 100 * (calls - requests) / calls if calls else None,
        "provider_call_reduction_vs_all_views_pct": reduction(requests, all_views),
        "actual_tokens": total_if_known("actual_tokens"),
        "new_actual_tokens": total_if_known("new_actual_tokens"),
        "accounted_tokens": accounted,
        "full_view_estimated_tokens": full,
        "token_reduction_vs_full_estimate_pct": reduction(accounted, full),
    }
    if _group:
        groups = {}
        for trace in traces:
            key = (trace.get("stop_node", "unknown"), trace["stop_reason"])
            groups.setdefault(key, []).append(trace)
        result["leaf_stats"] = [dict(stop_node=node, stop_reason=reason,
            **summarize_traces(rows, max_depth=max_depth, n_views=n_views, _group=False))
            for (node, reason), rows in sorted(groups.items())]
    return result
