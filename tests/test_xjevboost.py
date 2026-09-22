import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError

from xjevboost import FakeProvider, Prediction, PredictionCache, Task, TokenEstimate, XJevBoostClassifier
from xjevboost.core import AdaptivePolicy, BudgetError, Node, fit_weights
from xjevboost.jev import JevProvider
from xjevboost.views import View, ViewBuilder, ViewRecipe, canonical, table


class ScriptedProvider:
    namespace = "scripted:v1"
    context_limit = 1000

    def __init__(self):
        self.calls = []

    def estimate(self, view, task):
        return TokenEstimate(20 * len(view.columns), 1, True)

    def predict(self, view, task):
        self.calls.append(view)
        p = view.query[0]
        estimate = self.estimate(view, task)
        return Prediction((1 - p, p), estimate.input_tokens, estimate.output_tokens)


def scripted_policy(rows, labels, **settings):
    builder = ViewBuilder(((0., 0., 0.), (1., 1., 1.)), np.array([0, 1]), ("root", "left", "right"))
    recipes = [ViewRecipe(f"v{i}", (name,), (i,), (0, 1), i)
               for i, name in enumerate(builder.columns)]
    config = dict(min_samples_leaf=4, screening_samples=len(rows), max_depth=2,
                  gamma=0.0001, leaf_l2=0.001, random_state=7, max_nodes=31)
    config.update(settings)
    policy = AdaptivePolicy(ScriptedProvider(), Task((0, 1)), builder, recipes, **config)
    return policy.fit(tuple(rows), np.array(labels))


def branch_data():
    rows = ([(.1, .5, .5)] * 40 + [(.9, .5, .5)] * 40
            + [(.4, .99, .5)] * 10 + [(.6, .5, .01)] * 10)
    return rows, [0] * 40 + [1] * 40 + [1] * 10 + [0] * 10


def walk(node):
    yield node
    if node.kind == "acquire":
        yield from walk(node.child)
    elif node.kind == "route":
        yield from walk(node.inside)
        yield from walk(node.outside)


def test_learns_branch_specific_views_and_variable_calls():
    rows, labels = branch_data()
    policy = scripted_policy(rows, labels)
    traces = [policy.predict_one(row) for row in [rows[0], rows[40], rows[80], rows[90]]]
    assert policy.tree.view == 0
    assert [t["calls"] for t in traces] == [1, 1, 2, 2]
    assert traces[2]["steps"][-1]["view_id"] == "v1"
    assert traces[3]["steps"][-1]["view_id"] == "v2"
    assert [np.argmax(t["probabilities"]) for t in traces] == [0, 1, 1, 0]
    assert len(list(walk(policy.tree))) <= policy.max_nodes
    assert policy.stats["policy_fit_brier"] < policy.stats["single_view_brier"]


def test_learns_nonmonotonic_middle_stopping_region():
    rows = [(.75, .5, .5)] * 80 + [(.4, .99, .5)] * 10 + [(.9, .5, .01)] * 10
    policy = scripted_policy(rows, [1] * 90 + [0] * 10)
    traces = [policy.predict_one(row) for row in [rows[80], rows[0], rows[90]]]
    assert [t["calls"] for t in traces] == [2, 1, 2]
    assert all(t["branches"] for t in traces)


def test_projected_query_dedup_keeps_separate_labels():
    provider = FakeProvider()
    builder = ViewBuilder(((0, "a"), (1, "b")), np.array([0, 1]), ("signal", "ignored"))
    recipe = ViewRecipe("v0", ("signal",), (0,), (0, 1), 1)
    policy = AdaptivePolicy(provider, Task((0, 1)), builder, [recipe], max_depth=1,
                            min_samples_leaf=1, screening_samples=8)
    rows = ((0, "secret-a"), (0, "secret-b"), (1, "secret-c"), (1, "secret-d"))
    policy.fit(rows, np.array([0, 1, 0, 1]))
    assert provider.calls == 2
    assert len(policy.target) == 4
    assert len(policy._matrix) == 4
    assert policy.stats["cache_hits"] == 2
    assert "secret" not in canonical(builder.build(recipe, rows[0]).payload())
    before = provider.calls
    policy.predict_one((0, "previously-unseen-ignored-value"))
    assert provider.calls == before


def test_view_dedup_and_label_exclusion():
    rows = (("example", 1), ("example", 1))
    builder = ViewBuilder(rows, np.array([0, 0]), ("text", "number"))
    recipes = builder.recipes(20, 1, 1, 42)
    assert len(recipes) == 1
    payload = builder.build(recipes[0], ("QUERY", 99)).payload()
    assert payload["query"] == {"values": ("QUERY", 99)}
    assert all(e["values"][0] == "example" for e in payload["examples"])
    assert len(payload["examples"]) == 2  # duplicates in the prompt retain their original meaning


def test_balanced_recipe_keeps_every_class_with_small_fraction():
    builder = ViewBuilder(tuple((i,) for i in range(6)), np.array([0, 0, 0, 0, 1, 2]), ("x",))
    recipe = builder.recipes(1, .01, 1, 42)[0]
    assert {label for _, label in builder.build(recipe, (99,)).examples} == {0, 1, 2}


@pytest.mark.parametrize("subsample", [.01, .25, 1.0])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_every_recipe_covers_imbalanced_classes(subsample, n_classes):
    labels = np.array([0] * 28 + list(range(1, n_classes)))
    rows = tuple((i, i + 100) for i in range(len(labels)))
    source_ids = tuple(range(1000, 1000 + len(rows)))
    builder = ViewBuilder(rows, labels, ("a", "b"), row_ids=source_ids)
    recipes = builder.recipes(20, subsample, .5, 42)
    assert recipes == builder.recipes(20, subsample, .5, 42)
    if subsample < 1:
        assert {r.kind for r in recipes} == {"balanced", "broader", "random"}
    for recipe in recipes:
        view = builder.build(recipe, (999, 999))
        assert {label for _, label in view.examples} == set(range(n_classes))
        assert len(set(recipe.example_ids)) == len(recipe.example_ids)
        assert set(recipe.example_ids) <= set(source_ids)
        target = int(np.ceil(len(rows) * subsample))
        if recipe.kind == "broader":
            target = min(len(rows), int(np.ceil(target * 1.5)))
        assert len(recipe.example_ids) == max(n_classes, target)


def test_persistent_cache_reused_across_fits_and_is_task_sensitive(tmp_path):
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 0, 1])
    provider = FakeProvider()
    path = tmp_path / "cache.sqlite"
    kwargs = dict(provider=provider, n_views=1, subsample=1, max_depth=1, min_samples_leaf=1, cache=path)
    first = XJevBoostClassifier(**kwargs).fit(X, y, calibration_set=(X + 10, y))
    calls = provider.calls
    second = XJevBoostClassifier(**kwargs).fit(X, y, calibration_set=(X + 10, y))
    assert provider.calls == calls
    assert second.training_stats_["provider_calls"] == 0
    assert np.allclose(first.predict_proba(X + 10), second.predict_proba(X + 10))
    XJevBoostClassifier(**kwargs, task_instructions="different task").fit(X, y, calibration_set=(X + 10, y))
    assert provider.calls > calls


def test_two_partial_views_cannot_exceed_hypothetical_full_budget():
    class EightyPercent(ScriptedProvider):
        def estimate(self, view, task):
            return TokenEstimate(100 if len(view.columns) == 3 else 80, 0, True)

    rows, labels = branch_data()
    policy = scripted_policy(rows, labels)
    policy.provider = EightyPercent()
    # Cached synthetic results came from a different namespace/version.
    policy.cache = PredictionCache()
    trace = policy.predict_one(rows[80])
    assert trace["calls"] == 1
    assert trace["stop_reason"] == "token_limit"
    assert trace["accounted_tokens"] == 80
    assert trace["token_ceiling"] == 100
    assert len(policy.provider.calls) == 1  # hypothetical full view never sent


def test_absolute_context_and_depth_limits():
    rows, labels = branch_data()
    policy = scripted_policy(rows, labels)
    policy.max_total_tokens = 21
    trace = policy.predict_one(rows[80])
    assert trace["calls"] == 1 and trace["stop_reason"] == "token_limit"
    assert np.allclose(trace["probabilities"], [.6, .4])
    policy.max_total_tokens = 20
    with pytest.raises(BudgetError, match="First view"):
        policy.predict_one(rows[80])
    policy.max_total_tokens = None
    assert policy.predict_one(rows[80], max_depth=1)["calls"] == 1
    policy.provider.context_limit = 20
    with pytest.raises(BudgetError, match="First view"):
        policy.predict_one(rows[0])


def test_unknown_usage_reserved_and_approximate_bounds_flagged():
    class UnknownUsage(ScriptedProvider):
        def estimate(self, view, task):
            return TokenEstimate(20, 1, False)

        def predict(self, view, task):
            return Prediction((.5, .5))

    rows, labels = branch_data()
    policy = scripted_policy(rows, labels)
    policy.provider, policy.cache = UnknownUsage(), PredictionCache()
    policy.strict_tokens = False  # approximate accounting is an explicit opt-in
    policy.full_view_budget = False
    policy.max_total_tokens = 40
    trace = policy.predict_one(rows[80])
    assert trace["budget_kind"] == "estimated"
    assert trace["actual_tokens"] is None
    assert trace["accounted_tokens"] >= 25
    assert trace["calls"] == 1
    policy.strict_tokens = True
    with pytest.raises(BudgetError, match="upper bound"):
        policy.predict_one(rows[0])


def test_staged_search_calls_less_than_dense_bound():
    class Ranked(ScriptedProvider):
        def predict(self, view, task):
            self.calls.append(view)
            correct = int(view.query[0]) % 2
            confidence = .99 if view.columns == ("x0",) else .55
            return Prediction((1 - confidence, confidence) if correct else (confidence, 1 - confidence), 20, 1)

    n_views, n = 16, 100
    builder = ViewBuilder((tuple([0] * n_views), tuple([1] * n_views)), np.array([0, 1]), tuple(f"x{i}" for i in range(n_views)))
    recipes = [ViewRecipe(f"v{i}", (f"x{i}",), (i,), (0, 1), i) for i in range(n_views)]
    rows = tuple(tuple([i] * n_views) for i in range(n))
    provider = Ranked()
    policy = AdaptivePolicy(provider, Task((0, 1)), builder, recipes, max_depth=1,
                            screening_samples=8, min_samples_leaf=2, random_state=0)
    policy.fit(rows, np.arange(n) % 2)
    assert policy.stats["provider_calls"] < n * n_views // 2
    assert policy.single_view == 0
    # Each full request is called at most once, despite repeated stages.
    assert len({canonical(v.payload()) for v in provider.calls}) == len(provider.calls)


def test_training_limit_and_progress(capsys):
    X = np.arange(60).reshape(30, 2)
    provider = FakeProvider()
    clf = XJevBoostClassifier(provider=provider, n_views=1, max_depth=2, min_samples_leaf=1,
                             max_training_calls=5)
    with pytest.raises(BudgetError):
        clf.fit(X[:10], np.arange(10) % 2, calibration_set=(X[10:], np.arange(20) % 2), verbose=True)
    assert provider.calls == 5
    assert "calls=" in capsys.readouterr().err


@pytest.mark.parametrize("limit", [None, 100])
def test_progress_distinguishes_budget_from_batch_completion(capsys, limit):
    X = np.arange(80).reshape(40, 2)
    cache = PredictionCache()
    for cached in (False, True):
        clf = XJevBoostClassifier(
            provider=FakeProvider(), cache=cache, n_views=1, max_depth=1,
            min_samples_leaf=2, random_state=42, max_training_calls=limit)
        clf.fit(X[:10], np.arange(10) % 2,
                calibration_set=(X[10:30], np.arange(20) % 2),
                pruning_set=(X[30:], np.arange(10) % 2), verbose=True)
        log = capsys.readouterr().err
        assert "Growth |" in log and "Pruning |" in log and "Done |" in log
        assert "rows=10/10 (100.0% of batch)" in log
        assert "% of budget" not in log
        assert "calls=0/" not in log
        if cached:
            assert clf.training_stats_["provider_calls"] == 0
            assert "calls=0 " in log


def test_multiclass_mixed_values_sklearn_and_traces():
    X = pd.DataFrame({"text": ["a", "b", None, "a", "b", "c"],
                      "category": pd.Categorical(["red", "blue", "green"] * 2),
                      "number": [1., np.nan, 3., 4., 5., 6.]})
    y = np.array(["cat", "dog", "bird"] * 2)
    clf = XJevBoostClassifier(provider=FakeProvider(), n_views=4, min_samples_leaf=1, max_depth=2, random_state=4)
    assert is_classifier(clf)
    assert clone(clf).get_params()["n_views"] == 4
    with pytest.raises(NotFittedError):
        clf.predict(X)
    clf.fit(X, y, calibration_set=(X.assign(number=X.number.fillna(0) + 10), y))
    assert clf.classes_.tolist() == ["bird", "cat", "dog"]
    assert clf.feature_names_in_.tolist() == list(X.columns)
    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1)
    traces = clf.predict_with_trace(X)
    assert all(t["calls"] <= 2 for t in traces)
    for trace in traces:
        assert trace["prediction"] in clf.classes_
        assert trace["steps"][0]["columns"]
        assert trace["steps"][0]["example_ids"]
        assert trace["steps"][0]["actual_input_tokens"] > 0
        assert trace["accounted_tokens"] <= trace["token_ceiling"]
        json.dumps(trace, allow_nan=False)
    with pytest.raises(ValueError, match="Columns"):
        clf.predict(X[list(reversed(X.columns))])
    assert clf.predict_proba(X.iloc[:0]).shape == (0, 3)
    for policy in ("adaptive", "single_view", "fixed_sequence"):
        metrics = clf.evaluate(X, y, policy=policy, view_ids=[clf.views_[clf.policy_.single_view].id])
        assert 0 <= metrics["brier"] <= 1
        assert metrics["average_calls"] >= 1


def test_repeatable_sampling_and_learning():
    X = np.arange(120).reshape(30, 4)
    y = np.arange(30) % 3
    kwargs = dict(n_views=7, min_samples_leaf=2, max_depth=2, random_state=19)
    a = XJevBoostClassifier(provider=FakeProvider(), **kwargs).fit(X, y, calibration_set=(X + 400, y))
    b = XJevBoostClassifier(provider=FakeProvider(), **kwargs).fit(X, y, calibration_set=(X + 400, y))
    assert a.views_ == b.views_
    assert a.training_stats_["provider_calls"] == b.training_stats_["provider_calls"]
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_jev_adapter_payload_probability_order_and_no_sdk():
    seen = []

    def transport(payload):
        seen.append(payload)
        return {"model": "jev-1.13.0", "answers": {"classification": {
            "probabilities": {"class_1": .8, "class_0": .2}}},
            "usage": {"input_tokens": 12, "output_tokens": 4}}

    provider = JevProvider(model="jev-1.13.0", context_limit=1000, transport=transport)
    view = View(("a",), (((1,), 0),), (2,))
    task = Task(("no", "yes"), "Decide whether query.a exceeds one.", ("At most one", "Above one"))
    assert provider.estimate(view, task).is_upper_bound is False
    assert not seen
    result = provider.predict(view, task)
    assert result.probabilities == (.2, .8)
    assert seen[0]["state"]["query"] == {"features": {"a": 2}}
    assert seen[0]["state"]["examples"][0]["label"] == "class_0"
    with pytest.raises(ValueError, match="task_instructions"):
        provider.estimate(view, Task((0, 1)))
    with pytest.raises(ValueError, match="pinned"):
        JevProvider(model="jev-latest", context_limit=1000)
    code = "import sys; from xjevboost import FakeProvider, XJevBoostClassifier; assert 'xjevboost.jev' not in sys.modules; assert 'typesafe_sdk' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_weights_are_regularized_convex_combination():
    target = np.eye(2)[[0, 1, 0, 1]]
    outputs = np.stack([target, 1 - target], axis=1)
    weights = fit_weights(outputs, target, l2=.01)
    assert np.isclose(weights.sum(), 1) and (weights >= 0).all()
    assert weights[0] > .95


def test_calibration_fraction_is_reproducible_and_excludes_reserved_rows():
    X = pd.DataFrame({"id": np.arange(100), "text": ["example"] * 100})
    y = np.arange(100) % 2
    settings = dict(provider=FakeProvider(), calibration_fraction=.2, n_views=3,
                    random_state=42, min_samples_leaf=2, max_depth=1)
    first = XJevBoostClassifier(**settings).fit(X, y)
    second = clone(first).fit(X, y)
    pool, calibration = set(first.example_pool_indices_), set(first.calibration_indices_)
    assert len(pool) == 80 and len(calibration) == 20
    assert pool.isdisjoint(calibration)
    assert pool | calibration == set(range(100))
    assert first.calibration_indices_.tolist() == second.calibration_indices_.tolist()
    assert first.views_ == second.views_
    assert first.feature_names_in_.tolist() == ["id", "text"]
    for recipe in first.views_:
        assert set(recipe.example_ids) <= pool
        assert set(recipe.example_ids).isdisjoint(calibration)
        if "id" in recipe.columns:
            view = first.policy_.builder.build(recipe, (999, "new"))
            id_column = recipe.columns.index("id")
            assert [r[id_column] for r, _ in view.examples] == list(recipe.example_ids)
    assert all(first.policy_.queries[i][0] in calibration for i in range(20))


def test_calibration_mode_is_explicit_and_unambiguous():
    X, y = np.arange(80).reshape(40, 2), np.arange(40) % 2
    with pytest.raises(ValueError, match="either"):
        XJevBoostClassifier(provider=FakeProvider()).fit(X, y)
    with pytest.raises(ValueError, match="not both"):
        XJevBoostClassifier(provider=FakeProvider(), calibration_fraction=.2).fit(X, y, calibration_set=(X, y))
    for fraction in (0, 1, -1, True):
        with pytest.raises(ValueError, match="calibration_fraction"):
            XJevBoostClassifier(provider=FakeProvider(), calibration_fraction=fraction).fit(X, y)
    # Explicit data remain unsplit and the source IDs cover the full pool.
    clf = XJevBoostClassifier(provider=FakeProvider(), n_views=1).fit(X, y, calibration_set=(X + 100, y))
    assert clf.calibration_indices_ is None
    assert clf.example_pool_indices_.tolist() == list(range(40))


@pytest.mark.parametrize("max_nodes", [2, 3, 4, 5, 6, 7, 9, 11])
def test_tree_size_limit(max_nodes):
    rows, labels = branch_data()
    policy = scripted_policy(rows, labels, max_nodes=max_nodes)
    assert len(list(walk(policy.tree))) <= max_nodes


def test_core_import_is_independent_of_sklearn_and_jev():
    code = "import sys; from xjevboost.core import AdaptivePolicy; assert 'sklearn' not in sys.modules; assert 'xjevboost.jev' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_strict_budget_rejects_approximate_jev_before_any_request():
    called = []
    provider = JevProvider(model="jev-1.13.0", context_limit=32000,
                           transport=lambda payload: called.append(payload))
    clf = XJevBoostClassifier(provider=provider, n_views=1, min_samples_leaf=1, strict_tokens=True,
                             task_instructions="Classify the numeric observation.",
                             class_descriptions={0: "Group zero", 1: "Group one"})
    assert clf.strict_tokens is True
    with pytest.raises(BudgetError, match="upper bound"):
        clf.fit([[0], [1]], [0, 1], calibration_set=([[2], [3]], [0, 1]))
    assert not called


def test_approximate_default_warns_once_and_allows_oversized_full_reference():
    class Approximate(ScriptedProvider):
        context_limit = 40

        def estimate(self, view, task):
            return TokenEstimate(1000 if len(view.columns) == 3 else 20, 1, False)

    rows, labels = branch_data()
    policy = scripted_policy(rows, labels)
    provider = Approximate()
    policy.provider, policy.cache = provider, PredictionCache()
    assert policy.strict_tokens is False
    with pytest.warns(RuntimeWarning, match="approximate") as caught:
        trace = policy.predict_one(rows[80])
        policy.predict_one(rows[90])
    assert len(caught) == 1
    assert trace["calls"] == 2
    assert trace["token_ceiling"] > provider.context_limit
    assert all(len(v.columns) == 1 for v in provider.calls)


def test_rejects_bad_probabilities_and_violated_token_bound():
    class Bad(ScriptedProvider):
        def predict(self, view, task):
            return Prediction((float('nan'), 1))

    rows, labels = branch_data()
    policy = scripted_policy(rows, labels)
    policy.provider, policy.cache = Bad(), PredictionCache()
    with pytest.raises(ValueError, match="probabilities"):
        policy.predict_one(rows[0])

    class BrokenBound(ScriptedProvider):
        def predict(self, view, task):
            return Prediction((.5, .5), 100, 1)

    policy.provider = BrokenBound()
    with pytest.raises(BudgetError, match="violated"):
        policy.predict_one(rows[0])


@pytest.mark.parametrize("settings", [{"max_depth": 0}, {"n_views": 0}, {"max_nodes": 1},
                                     {"subsample": 0}, {"colsample_bytree": 2}, {"gamma": -1},
                                     {"token_margin": .9}, {"max_training_calls": 0}])
def test_invalid_settings_fail_before_provider_calls(settings):
    provider = FakeProvider()
    with pytest.raises(ValueError):
        XJevBoostClassifier(provider=provider, **settings).fit([[0], [1]], [0, 1], calibration_set=([[2], [3]], [0, 1]))
    assert provider.calls == 0

