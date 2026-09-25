"""Wire-contract tests; no downloaded weights or hosted credentials needed."""
import io
import json

import numpy as np
import pytest

from xjevboost import LayaProvider, OpenJevProvider, SystemOneProvider, XJevBoostClassifier
from xjevboost.cache import PredictionCache
from xjevboost.providers import Task, TokenEstimate
from xjevboost.views import View


TASK = Task(("no", "yes"), "Predict whether the robot finishes.", ("Does not finish", "Finishes"))
VIEW = View(("battery",), (((20,), 0), ((90,), 1)), (70, True), query_columns=("battery", "traction"))


def provider(cls=SystemOneProvider, **kwargs):
    settings = dict(model="openjev-0.1", endpoint="http://localhost:8080/v1/systemone",
                    revision="weights-abc-server-def", context_limit=32000)
    if cls is LayaProvider:
        settings["model"] = "english"
    settings.update(kwargs)
    return cls(**settings)


def response(model="openjev-0.1", **kwargs):
    return dict(model=model, answers={"classification": {
        "probabilities": {"class_1": 3, "class_0": 1}}}, **kwargs)


@pytest.mark.parametrize("cls", [SystemOneProvider, OpenJevProvider, LayaProvider])
def test_wire_request_and_probability_order(monkeypatch, cls):
    requests = []
    def send(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(response(routing={"model": "english"},
                            usage={"input_tokens": 100, "output_tokens": 0})).encode())
    monkeypatch.setattr("xjevboost.jev.urlopen", send)
    monkeypatch.setenv("TYPESAFE_API_KEY", "must-not-leak-to-local-server")
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.delenv("OPENJEV_API_KEY", raising=False)
    p = provider(cls)
    result = p.predict(VIEW, TASK)
    assert result.probabilities == (.25, .75)
    assert result.actual_tokens == 100
    req = requests[0]
    assert req.get_header("Authorization") is None
    payload = json.loads(req.data)
    assert payload["state"]["query"]["features"] == {"battery": 70, "traction": True}
    assert payload["state"]["examples"][0] == {"features": {"battery": 20}, "label": "class_0"}
    assert payload["questions"]["classification"]["criteria"]["class_0"] == "no: Does not finish"
    assert "label" not in payload["state"]["query"]


def test_authentication_is_explicit(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "local-key")
    assert provider(LayaProvider).request_headers()["Authorization"] == "Bearer local-key"
    assert provider(LayaProvider, api_key="override").request_headers()["Authorization"] == "Bearer override"
    assert "Authorization" not in provider(LayaProvider, api_key="").request_headers()
    assert "Authorization" not in provider().request_headers()


def test_estimate_is_local_and_uses_serialized_payload():
    seen = []
    p = provider(estimator=lambda payload: seen.append(payload) or TokenEstimate(300, 0, True),
                 transport=lambda payload: pytest.fail("Estimator made a call"))
    assert p.estimate(VIEW, TASK).total == 300
    assert seen == [p.payload(VIEW, TASK)]
    assert not provider().estimate(VIEW, TASK).is_upper_bound


def test_laya_does_not_reserve_generated_output():
    estimate = provider(LayaProvider, context_limit=512).estimate(VIEW, TASK)
    assert estimate.output_tokens == 0
    assert estimate.total < 512
    assert not estimate.is_upper_bound
    custom = TokenEstimate(400, 1, True)
    assert provider(LayaProvider, estimator=lambda _: custom).estimate(VIEW, TASK) == custom


@pytest.mark.parametrize("change", [dict(revision="new"), dict(model="other"),
    dict(endpoint="http://localhost:9999/v1/systemone"), dict(response_model="canonical")])
def test_cache_separates_deployments(change):
    assert PredictionCache.key(provider(), TASK, VIEW) != PredictionCache.key(provider(**change), TASK, VIEW)
    assert provider(api_key="secret").namespace == provider(api_key="rotated").namespace


@pytest.mark.parametrize("payload", [response(model="wrong"), {"answers": {}}])
def test_unexpected_model_rejected(payload):
    with pytest.raises(ValueError, match="unexpected model"):
        provider(transport=lambda _: payload).predict(VIEW, TASK)


def test_model_alias_and_missing_usage():
    p = provider(response_model="canonical-id", transport=lambda _: response(model="canonical-id"))
    assert p.predict(VIEW, TASK).actual_tokens is None


@pytest.mark.parametrize("routing", [{}, {"model": "multilingual"}])
def test_laya_cannot_silently_change_checkpoint(routing):
    with pytest.raises(ValueError, match="checkpoint"):
        provider(LayaProvider, transport=lambda _: response(routing=routing)).predict(VIEW, TASK)


@pytest.mark.parametrize("probs", [{"class_0": 1}, {"class_0": -1, "class_1": 2},
    {"class_0": float("nan"), "class_1": 1}, {"class_0": 0, "class_1": 0}])
def test_invalid_probabilities(probs):
    data = response()
    data["answers"]["classification"]["probabilities"] = probs
    with pytest.raises(ValueError):
        provider(transport=lambda _: data).predict(VIEW, TASK)


def test_multiclass_and_class_limit():
    task = Task(("a", "b", "c"), "Pick the class", ("First", "Second", "Third"))
    data = response()
    data["answers"]["classification"]["probabilities"] = {"class_2": 2, "class_0": 1, "class_1": 1}
    assert provider(transport=lambda _: data).predict(VIEW, task).probabilities == (.25, .25, .5)
    with pytest.raises(ValueError, match="at most 2"):
        provider(max_classes=2).payload(VIEW, task)
    with pytest.raises(ValueError, match="100"):
        provider(LayaProvider).validate_task(Task(tuple(range(101)), "Choose", ("Description",) * 101))


@pytest.mark.parametrize("settings", [dict(revision=""), dict(model="model-latest"),
    dict(endpoint="http://user:secret@localhost/api"), dict(endpoint="http://localhost/api?key=secret"),
    dict(max_classes=1), dict(response_model="")])
def test_configuration_validation(settings):
    with pytest.raises(ValueError):
        provider(**settings)


@pytest.mark.parametrize("cls", [SystemOneProvider, OpenJevProvider, LayaProvider])
def test_fit_cache_trace_and_save_load(tmp_path, cls):
    calls = []
    def predict(payload):
        calls.append(payload)
        return response(routing={"model": "english"}, usage={"input_tokens": 50, "output_tokens": 0})
    p = provider(cls, transport=predict, api_key="never-serialize-this")
    clf = XJevBoostClassifier(provider=p, n_views=2, max_depth=1, min_samples_leaf=2,
        calibration_fraction=.4, task_instructions=TASK.instructions,
        class_descriptions={0: "Does not finish", 1: "Finishes"}, random_state=42)
    X = np.arange(60).reshape(30, 2)
    clf.fit(X, np.arange(30) % 2)
    first = clf.predict_proba(X[:3])
    count = len(calls)
    np.testing.assert_allclose(clf.predict_proba(X[:3]), first)
    assert len(calls) == count
    assert all(t["calls"] == 1 for t in clf.predict_with_trace(X[:3]))
    path = tmp_path / "policy.json"
    clf.save_model(path)
    assert "never-serialize-this" not in path.read_text()
    loaded = XJevBoostClassifier.load_model(path, provider=p)
    np.testing.assert_allclose(loaded.predict_proba(X[:3]), first)
    with pytest.raises(ValueError, match="namespace"):
        XJevBoostClassifier.load_model(path, provider=provider(cls, revision="changed"))
