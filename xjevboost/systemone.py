"""HTTP adapters for Jev-compatible decision servers; no model SDK required."""
import os
from urllib.parse import urlsplit

from .jev import JevProvider
from .providers import TokenEstimate
from .views import digest


class SystemOneProvider(JevProvider):
    """Connect to a Choice-compatible /v1/systemone endpoint.

    revision identifies the deployed weights, server version and configuration.
    Change it whenever any of those change; it is a cache identity, not a server
    argument or a mechanism for pinning remote weights. Authentication is opt-in.
    response_model can name a server's canonical model ID instead of its alias.
    """

    def __init__(self, *, endpoint, model, revision, context_limit, api_key=None,
                 api_key_env=None, response_model=None, max_classes=255,
                 timeout=60, output_token_allowance=256, estimator=None,
                 transport=None):
        url = urlsplit(endpoint)
        if (url.scheme not in ("http", "https") or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("endpoint must be an HTTP(S) URL without credentials, query or fragment")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must identify the deployed weights and server configuration")
        if isinstance(max_classes, bool) or not isinstance(max_classes, int) or not 2 <= max_classes <= 255:
            raise ValueError("max_classes must be between 2 and 255")
        if response_model is not None and (not isinstance(response_model, str) or not response_model.strip()):
            raise ValueError("response_model must be a nonempty model ID")
        super().__init__(model=model, context_limit=context_limit, api_key=api_key,
                         endpoint=endpoint, timeout=timeout,
                         output_token_allowance=output_token_allowance,
                         estimator=estimator, transport=transport)
        self.revision = revision
        self.api_key_env = api_key_env
        self.response_model = response_model or model
        self.max_classes = max_classes
        self.namespace = "systemone:choice-text-v1:" + digest({
            "adapter": type(self).__name__, "endpoint": endpoint, "model": model,
            "revision": revision, "response_model": self.response_model,
            "max_classes": max_classes})

    def validate_task(self, task):
        super().validate_task(task)
        if len(task.classes) > self.max_classes:
            raise ValueError(f"This server supports at most {self.max_classes} classes")

    def payload(self, view, task):
        payload = super().payload(view, task)
        # String criteria are the common denominator across compatible servers.
        payload["questions"]["classification"]["criteria"] = {
            f"class_{i}": f"{label}: {description}"
            for i, (label, description) in enumerate(zip(task.classes, task.descriptions))}
        return payload

    def request_headers(self):
        key = self.api_key
        if key is None and self.api_key_env:
            key = os.environ.get(self.api_key_env)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def validate_response_model(self, response):
        if response.get("model") != self.response_model:
            raise ValueError("Server returned an unexpected model; refusing to cache its response")


class OpenJevProvider(SystemOneProvider):
    """Choice adapter for Jev-compatible OpenJev HTTP deployments.

    endpoint is explicit because several unrelated projects use this name.
    """

    def __init__(self, *, api_key_env="OPENJEV_API_KEY", **kwargs):
        super().__init__(api_key_env=api_key_env, **kwargs)


class LayaProvider(SystemOneProvider):
    """Connect to NandhaKishorM/laya's laya-serve HTTP server.

    Pin an explicit router checkpoint rather than allowing automatic routing.
    Laya reports the selected checkpoint in routing.model, separately from the
    top-level model identifier. Context allowance must match your deployment.
    """

    def __init__(self, *, model="english", endpoint="http://127.0.0.1:8000/v1/systemone",
                 api_key_env="LAYA_API_KEY", **kwargs):
        if model not in ("english", "multilingual", "typed-decisions"):
            raise ValueError("Laya model must be english, multilingual or typed-decisions")
        super().__init__(model=model, endpoint=endpoint, api_key_env=api_key_env,
                         max_classes=100, **kwargs)

    def validate_response_model(self, response):
        routing = response.get("routing") or {}
        if routing.get("model") != self.model:
            raise ValueError("Laya returned an unexpected router checkpoint; refusing to cache its response")

    def estimate(self, view, task):
        estimate = super().estimate(view, task)
        if self.estimator is not None:
            return estimate
        # laya-serve's encoder returns output_tokens=0. Reserving Jev's generic
        # output allowance would reject small views in a 512-token checkpoint.
        return TokenEstimate(estimate.input_tokens, 0, False)
