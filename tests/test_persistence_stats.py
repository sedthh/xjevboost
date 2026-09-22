import json
import numpy as np
import pandas as pd
import pytest
from xjevboost import XJevBoostClassifier, FakeProvider, Prediction, PredictionCache
from xjevboost.stats import summarize_traces


def fitted(k=2, provider=None):
    X = pd.DataFrame({"income": np.arange(60), "category": ["a", "b", "c"] * 20})
    y = np.arange(60) % k
    clf = XJevBoostClassifier(provider=provider or FakeProvider(), n_views=3, max_depth=2,
        min_samples_leaf=2, calibration_fraction=.4, pruning_fraction=.2, random_state=42)
    clf.fit(X, y)
    return clf, X.iloc[:8]


@pytest.mark.parametrize("k", [2, 3])
def test_json_roundtrip_credentials_and_inspection(tmp_path, k):
    provider = FakeProvider()
    provider.api_key = "DO_NOT_SERIALIZE_THIS_SECRET"
    provider.extra_state = {"password": "ALSO_PRIVATE"}
    clf, X = fitted(k, provider)
    path = tmp_path / "model.json"
    before = provider.calls
    exported = clf.export_tree()
    clf.save_model(path)
    assert provider.calls == before
    content = path.read_text(encoding="utf8")
    assert "DO_NOT_SERIALIZE" not in content and "ALSO_PRIVATE" not in content
    data = json.loads(content)
    assert "provider" not in data and "cache" not in data
    new_provider = FakeProvider()
    restored = XJevBoostClassifier.load_model(path, provider=new_provider, cache=tmp_path / "new.sqlite")
    assert new_provider.calls == 0
    assert restored.export_tree() == exported
    assert restored.feature_names_in_.tolist() == clf.feature_names_in_.tolist()
    assert restored.classes_.tolist() == clf.classes_.tolist()
    np.testing.assert_allclose(restored.predict_proba(X), clf.predict_proba(X))
    assert restored.predict(X).tolist() == clf.predict(X).tolist()
    for policy in ("adaptive", "single_view", "fixed_sequence"):
        kwargs = {"view_ids": [clf.views_[0].id]} if policy == "fixed_sequence" else {}
        a = clf.evaluate(X, np.arange(len(X)) % k, policy=policy, **kwargs)
        b = restored.evaluate(X, np.arange(len(X)) % k, policy=policy, **kwargs)
        for name in ("brier", "accuracy", "average_calls", "actual_tokens", "leaf_stats"):
            if name != "leaf_stats":
                assert a[name] == b[name]
    restored.save_model(tmp_path / "again.json")
    assert json.loads((tmp_path / "again.json").read_text()) == data
    with pytest.raises(ValueError, match="Columns"):
        restored.predict(X[X.columns[::-1]])


@pytest.mark.parametrize("mutation", ["version", "weights", "kind", "view", "prior"])
def test_rejects_invalid_models(tmp_path, mutation):
    clf, _ = fitted()
    p = tmp_path / "bad.json"
    clf.save_model(p)
    d = json.loads(p.read_text())
    if mutation == "version": d["version"] = 99
    if mutation == "weights": d["tree"]["weights"] = [-1]
    if mutation == "kind": d["tree"]["kind"] = "execute"
    if mutation == "view": d["tree"]["view"] = 1000
    if mutation == "prior": d["prior"] = [2, -1]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError): XJevBoostClassifier.load_model(p, provider=FakeProvider())


def test_requires_matching_provider(tmp_path):
    clf, _ = fitted()
    p = tmp_path / "model.json"
    clf.save_model(p)
    provider = FakeProvider()
    provider.namespace = "different-model"
    with pytest.raises(ValueError, match="namespace"):
        XJevBoostClassifier.load_model(p, provider=provider)
    with pytest.raises(TypeError): XJevBoostClassifier.load_model(p)


def test_training_and_leaf_costs_require_no_calls():
    clf, X = fitted()
    calls = clf.provider.calls
    exported = clf.export_tree()
    assert clf.provider.calls == calls
    def visit(n):
        if n["kind"] == "stop": yield n
        for key in ("child", "inside", "outside"):
            if key in n: yield from visit(n[key])
    leaves = list(visit(exported["root"]))
    assert sum(n["sample_count"] for n in leaves) == clf.training_stats_["growth_rows"]
    for n in leaves:
        assert n["training_path_stats"]["average_calls"] == n["acquisitions"]
        assert n["training_path_stats"]["average_actual_tokens_per_query"] >= 0
    stats = clf.training_stats_
    rows = stats["growth_rows"] + stats["pruning_rows"]
    assert stats["average_provider_calls_per_training_query"] == stats["provider_calls"] / rows
    first = clf.predict_with_trace(X)
    second = clf.predict_with_trace(X)
    a, b = clf.summarize_traces(first), clf.summarize_traces(second)
    assert b["provider_calls"] == 0 and b["new_actual_tokens"] == 0
    assert a["actual_tokens"] == b["actual_tokens"]
    assert b["average_actual_tokens_per_query"] == b["actual_tokens"] / len(X)
    assert b["average_actual_tokens_per_view"] == b["actual_tokens"] / b["calls"]
    assert sum(g["n_queries"] for g in b["leaf_stats"]) == len(X)
    assert sum(g["actual_tokens"] for g in b["leaf_stats"]) == b["actual_tokens"]
    assert all(t["stop_node"].startswith("root.child") for t in second)


def test_unknown_usage_and_empty_stats():
    class Unknown(FakeProvider):
        namespace = "unknown"
        def predict(self, view, task):
            p = super().predict(view, task)
            return Prediction(p.probabilities)
    clf, X = fitted(provider=Unknown())
    assert clf.training_stats_["average_new_actual_tokens_per_training_query"] is None
    stats = clf.summarize_traces(clf.predict_with_trace(X))
    assert stats["average_actual_tokens_per_query"] is None
    assert stats["average_accounted_tokens_per_query"] > 0
    empty = summarize_traces([], max_depth=2, n_views=3)
    assert empty["leaf_stats"] == [] and empty["average_actual_tokens_per_query"] is None


def test_budget_stops_have_separate_groups():
    common = dict(calls=1, provider_calls=0, accounted_tokens=10, actual_tokens=10,
                  new_actual_tokens=0, full_view_estimated_tokens=20, stop_node="root.child")
    traces = [dict(common, stop_reason="leaf"), dict(common, stop_reason="token_limit")]
    stats = summarize_traces(traces, max_depth=2, n_views=3)
    assert len(stats["leaf_stats"]) == 2
    assert {g["stop_reason"] for g in stats["leaf_stats"]} == {"leaf", "token_limit"}


def test_branched_tree_roundtrip_and_budget_stop(tmp_path):
    from test_xjevboost import scripted_policy, branch_data, ScriptedProvider
    rows, labels = branch_data()
    policy = scripted_policy(rows, labels)
    clf = XJevBoostClassifier(provider=policy.provider, max_depth=2)
    clf.policy_, clf.tree_, clf.views_ = policy, policy.tree, policy.recipes
    clf.classes_ = np.array([0, 1])
    clf.n_features_in_ = 3
    clf.n_views_ = len(policy.recipes)
    clf.training_stats_ = dict(policy.stats)
    path = tmp_path / "branched.json"
    clf.save_model(path)
    loaded = XJevBoostClassifier.load_model(path, provider=ScriptedProvider())
    # Three regions include both continuation branches and the stopping region.
    X = np.array([rows[0], rows[-1], rows[-11]])
    expected = clf.predict_with_trace(X)
    actual = loaded.predict_with_trace(X)
    assert len({t["stop_node"] for t in actual}) > 1
    for a, b in zip(expected, actual):
        for key in ("probabilities", "branches", "stop_node", "stop_reason", "calls", "actual_tokens"):
            assert a[key] == b[key]
    loaded.policy_.max_depth = 1
    limited = loaded.predict_with_trace(X)
    assert any(t["stop_reason"] == "max_depth" for t in limited)
    assert all(t["calls"] == 1 for t in limited)
    assert sum(g["n_queries"] for g in loaded.inference_stats_["leaf_stats"]) == len(X)


def test_loaded_plotting_and_refit(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clf, X = fitted()
    path = tmp_path / "model.json"
    clf.save_model(path)
    loaded = XJevBoostClassifier.load_model(path, provider=FakeProvider())
    before = loaded.provider.calls
    ax = loaded.plot_tree()
    plt.close(ax.figure)
    assert loaded.provider.calls == before
    loaded.set_params(calibration_fraction=.4, pruning_fraction=.2, random_state=42)
    frame = pd.DataFrame({"new_feature": np.arange(60)})
    loaded.fit(frame, np.arange(60) % 2)
    assert not hasattr(loaded, "_saved_inspection")
    assert loaded.feature_names_in_.tolist() == ["new_feature"]
    assert loaded.export_tree()["root"]["training_path_stats"]["n_queries"] == 24


def test_array_input_and_string_classes_roundtrip(tmp_path):
    X = np.arange(120).reshape(60, 2)
    clf = XJevBoostClassifier(provider=FakeProvider(), n_views=2, min_samples_leaf=2,
                             calibration_fraction=.4, random_state=2)
    clf.fit(X, np.array(["lost", "won"] * 30))
    path = tmp_path / "model.json"
    clf.save_model(path)
    loaded = XJevBoostClassifier.load_model(path, provider=FakeProvider())
    assert not hasattr(loaded, "feature_names_in_")
    assert loaded.predict(X[:5]).tolist() == clf.predict(X[:5]).tolist()


@pytest.mark.parametrize("field", ["column_indices", "columns", "example_ids"])
def test_invalid_recipe_rejected(tmp_path, field):
    clf, _ = fitted()
    path = tmp_path / "model.json"
    clf.save_model(path)
    d = json.loads(path.read_text())
    d["recipes"][0][field] = [999999] if field != "columns" else ["wrong"]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="recipe"):
        XJevBoostClassifier.load_model(path, provider=FakeProvider())
