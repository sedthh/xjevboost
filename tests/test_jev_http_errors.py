import io
import json
from urllib.error import HTTPError
import pytest
from xjevboost.jev import JevProvider, JevContextLimitError
from xjevboost.providers import Task
from xjevboost.views import View

@pytest.mark.parametrize("status,error_type,expected", [
    (400, "max_tokens_exceeded", JevContextLimitError),
    (413, "max_tokens_exceeded", JevContextLimitError),
    (401, "unauthorized", HTTPError), (429, "rate_limit", HTTPError),
    (400, "invalid_question", HTTPError)])
def test_context_rejection_only(monkeypatch, status, error_type, expected):
    calls = []
    def reject(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.invalid", status, "rejected", {},
                        io.BytesIO(json.dumps({"detail": {"error_type": error_type}}).encode()))
    monkeypatch.setattr("xjevboost.jev.urlopen", reject)
    provider = JevProvider(model="jev-1.13.0", context_limit=32000, api_key="secret")
    with pytest.raises(expected) as exc:
        provider.predict(View(("feature",), (), (1,)), Task((0,1), "Predict outcome", ("No", "Yes")))
    assert "secret" not in str(exc.value)
    assert len(calls) == 1
