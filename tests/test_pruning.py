import numpy as np
import pytest

from xjevboost import FakeProvider, XJevBoostClassifier
from xjevboost.core import AdaptivePolicy
from test_xjevboost import branch_data, scripted_policy, ScriptedProvider, walk


def policy_with_pruning(prune_labels, limit=None):
    rows, labels = branch_data()
    template = scripted_policy(rows, labels)
    pruning = [(.4, .99, .5)] * 10 + [(.6, .5, .01)] * 10
    policy = AdaptivePolicy(ScriptedProvider(), template.task, template.builder, template.recipes,
                            max_depth=2, min_samples_leaf=4, screening_samples=100,
                            gamma=.0001, random_state=7, leaf_l2=.001,
                            max_training_calls=limit)
    return policy.fit(tuple(rows + pruning), np.array(labels + prune_labels), pruning_rows=20)


def test_held_out_bad_continuations_are_removed_without_refitting():
    good = policy_with_pruning([1] * 10 + [0] * 10)
    bad = policy_with_pruning([0] * 10 + [1] * 10)
    assert good.stats['growth_brier_before_pruning'] == bad.stats['growth_brier_before_pruning']
    assert good.single_view == bad.single_view == 0
    assert bad.tree.child.kind == 'stop'
    assert np.allclose(bad.tree.child.weights, [1.])
    assert bad.stats['nodes_after_pruning'] == 2
    assert bad.stats['pruning_brier'] == bad.stats['pruning_single_view_brier']
    assert good.stats['pruning_brier'] < good.stats['pruning_single_view_brier']
    assert good.stats['nodes_after_pruning'] > 2
    assert bad.stats['growth_rows'] == 100
    assert bad.stats['pruning_rows'] == 20
    assert bad.stats['growth_class_counts'] == [50, 50]
    assert any(r['reason'] == 'insufficient_improvement' for r in bad.pruning_history)


def test_pruning_reuses_existing_requests():
    policy = policy_with_pruning([0] * 10 + [1] * 10)
    # The repeated projected queries were acquired during growth.
    assert policy.stats['pruning_provider_calls'] == 0
    assert policy.stats['cache_hits'] > 0
    assert len(list(walk(policy.tree))) == policy.stats['nodes_after_pruning']


def test_wrapper_rejects_overlapping_pruning_rows_before_api_calls():
    provider = FakeProvider()
    clf = XJevBoostClassifier(provider=provider, min_samples_leaf=1)
    with pytest.raises(ValueError, match='overlap'):
        clf.fit([[0], [1]], [0, 1], calibration_set=([[2], [3]], [0, 1]),
                pruning_set=([[2], [4]], [1, 0]))
    assert provider.calls == 0


def test_wrapper_exposes_pruning_diagnostics():
    X = np.arange(120).reshape(60, 2)
    y = np.arange(60) % 2
    clf = XJevBoostClassifier(provider=FakeProvider(), n_views=2, max_depth=2,
                             min_samples_leaf=2, random_state=3).fit(
        X[:20], y[:20], calibration_set=(X[20:40], y[20:40]), pruning_set=(X[40:], y[40:]))
    assert clf.training_stats_['pruning_rows'] == 20
    assert clf.training_stats_['pruning_brier'] <= clf.training_stats_['pruning_single_view_brier'] + 1e-12
    assert isinstance(clf.pruning_history_, list)
    assert clf.export_tree()['root']['sample_count'] == 20


def test_budget_exhaustion_does_not_keep_unvalidated_branches():
    rows, labels = branch_data()
    template = scripted_policy(rows, labels)
    limit = template.stats['provider_calls']
    policy = AdaptivePolicy(ScriptedProvider(), template.task, template.builder, template.recipes,
                            max_depth=2, min_samples_leaf=4, screening_samples=100,
                            gamma=.0001, random_state=7, leaf_l2=.001, max_training_calls=limit)
    novel = [(.41, .98, .51)] * 10 + [(.61, .51, .02)] * 10
    policy.fit(tuple(rows + novel), np.array(labels + [0] * 10 + [1] * 10), pruning_rows=20)
    assert policy.stats['provider_calls'] == limit
    assert policy.tree.child.kind == 'stop'
    assert policy.stats['pruning_brier'] is None
    assert policy.pruning_history[-1]['reason'] == 'validation_budget_unavailable'


def test_tiny_pruning_branch_is_removed_instead_of_assumed_valid():
    rows, labels = branch_data()
    template = scripted_policy(rows, labels)
    policy = AdaptivePolicy(ScriptedProvider(), template.task, template.builder, template.recipes,
                            max_depth=2, min_samples_leaf=4, screening_samples=100,
                            gamma=.0001, random_state=7, leaf_l2=.001)
    pruning = [(.1, .5, .5)] * 8 + [(.4, .99, .5), (.6, .5, .01)]
    policy.fit(tuple(rows + pruning), np.array(labels + [0] * 8 + [1, 0]), pruning_rows=10)
    assert policy.tree.child.kind == 'stop'
    assert any(r['reason'] == 'insufficient_pruning_rows' for r in policy.pruning_history)


@pytest.mark.parametrize("auto_cal,auto_prune", [(True, True), (True, False), (False, True)])
def test_fraction_splits_use_original_rows_and_are_disjoint(auto_cal, auto_prune):
    from sklearn.base import clone
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100) % 2
    settings = dict(provider=FakeProvider(), n_views=1, max_depth=1,
                    min_samples_leaf=2, random_state=42,
                    calibration_fraction=.2 if auto_cal else None,
                    pruning_fraction=.3 if auto_prune else None)
    sets = {}
    if not auto_cal:
        sets["calibration_set"] = [X[:20] + 1000, y[:20]]
    if not auto_prune:
        sets["pruning_set"] = [X[:30] + 2000, y[:30]]
    a = XJevBoostClassifier(**settings).fit(X, y, **sets)
    b = clone(a).fit(X, y, **sets)
    indices = [a.example_pool_indices_]
    for name, expected in [("calibration_indices_", 20), ("pruning_indices_", 30)]:
        value = getattr(a, name)
        if value is not None:
            assert len(value) == expected
            assert np.array_equal(value, getattr(b, name))
            indices.append(value)
    assert len(np.concatenate(indices)) == 100
    assert len(set(np.concatenate(indices))) == 100
    assert a.training_stats_["growth_rows"] == 20
    assert a.training_stats_["pruning_rows"] == 30
    assert a.views_ == b.views_
    for recipe in a.views_:
        assert set(recipe.example_ids) <= set(a.example_pool_indices_)


@pytest.mark.parametrize("fraction", [0, 1, -.1, True, float("nan"), float("inf")])
def test_invalid_pruning_fraction_has_no_calls(fraction):
    provider = FakeProvider()
    with pytest.raises(ValueError, match="pruning_fraction"):
        XJevBoostClassifier(provider=provider, calibration_fraction=.2,
                            pruning_fraction=fraction).fit(np.arange(100).reshape(50, 2), np.arange(50) % 2)
    assert provider.calls == 0


def test_fraction_conflicts_and_total_have_no_calls():
    provider = FakeProvider()
    X, y = np.arange(100).reshape(50, 2), np.arange(50) % 2
    for fraction in [.8, .9]:
        with pytest.raises(ValueError, match="sum to less"):
            XJevBoostClassifier(provider=provider, calibration_fraction=.2,
                                pruning_fraction=fraction).fit(X, y)
    with pytest.raises(ValueError, match="not both"):
        XJevBoostClassifier(provider=provider, calibration_fraction=.2, pruning_fraction=.2).fit(
            X, y, pruning_set=(X + 1000, y))
    with pytest.raises(ValueError, match="overlap"):
        XJevBoostClassifier(provider=provider).fit(X, y, calibration_set=(X, y))
    assert provider.calls == 0


def test_grouped_ratio_splits_preserve_balance_and_all_rows():
    from xjevboost.splitting import stratified_split
    rows = tuple((i,) for i in range(40) for _ in range(3))
    labels = np.array([i % 2 for i in range(40) for _ in range(3)])
    a, b = stratified_split(rows, labels, np.arange(120), 30, 42)
    assert len(b) == 30
    assert abs(labels[b].mean() - .5) <= .1
    assert {rows[i] for i in a}.isdisjoint({rows[i] for i in b})
    assert set(a) | set(b) == set(range(120))
    assert np.array_equal(b, stratified_split(rows, labels, np.arange(120), 30, 42)[1])
    clf = XJevBoostClassifier(provider=FakeProvider(), calibration_fraction=.25,
        pruning_fraction=.25, random_state=42, min_samples_leaf=2, n_views=1,
        max_depth=1).fit(rows, labels)
    partitions = [clf.example_pool_indices_, clf.calibration_indices_, clf.pruning_indices_]
    for i, ids in enumerate(partitions):
        for other in partitions[i + 1:]:
            assert {rows[j] for j in ids}.isdisjoint({rows[j] for j in other})
