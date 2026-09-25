"""Optional Jev HTTP adapter. Importing xjevboost never imports this module."""
import json
import math
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .providers import Prediction, TokenEstimate, normalized
from .views import canonical


class JevContextLimitError(ValueError):
    """The server rejected the request because its context was too large."""


class JevProvider:
    """Use a pinned model version for stable persistent caching.

    context_limit is explicit: model capabilities can change. The built-in
    byte-based estimator is approximate, not a strict guarantee. Supply an
    estimator(payload) -> TokenEstimate with a reliable bound for strict mode.
    transport(payload) -> dict can be injected for tests or custom HTTP clients.
    """
    def __init__(self, *, model, context_limit, api_key=None,
                 endpoint="https://api.typesafe.ai/v1/systemone", timeout=60,
                 output_token_allowance=256, estimator=None, transport=None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("An explicit Jev model version is required")
        if model.endswith("latest"):
            raise ValueError("Use a pinned Jev model version, not a mutable 'latest' alias")
        if not isinstance(context_limit, int) or context_limit <= 0:
            raise ValueError("context_limit must be a positive integer")
        if not isinstance(output_token_allowance, int) or output_token_allowance < 1:
            raise ValueError("output_token_allowance must be positive")
        self.model, self.context_limit, self.api_key = model, context_limit, api_key
        self.endpoint, self.timeout = endpoint, timeout
        self.output_token_allowance, self.estimator, self.transport = output_token_allowance, estimator, transport
        self.namespace = f"jev-http:choice-records-v1:{endpoint}:{model}"

    @staticmethod
    def validate_task(task):
        if not isinstance(task.instructions, str) or not task.instructions.strip():
            raise ValueError("Jev requires meaningful task_instructions")
        if (len(task.descriptions) != len(task.classes)
                or any(not isinstance(d, str) or not d.strip() for d in task.descriptions)):
            raise ValueError("Jev requires a nonempty description for every class")
        if len(task.classes) > 255:
            raise ValueError("Jev Choice supports at most 255 classes")

    def payload(self, view, task):
        self.validate_task(task)
        names = [f"class_{i}" for i in range(len(task.classes))]
        return {"model": self.model,
                "state": {"columns": list(view.columns),
                          "examples": [{"features": dict(zip(view.columns, row)), "label": names[label]}
                                       for row, label in view.examples],
                          "query": {"features": dict(zip(view.effective_query_columns, view.query))}},
                "questions": {"classification": {
                    "type": "choice",
                    "instructions": task.instructions + "\nClassify only state.query. state.examples are labeled reference records.",
                    "criteria": {name: {"label": str(label), "description": description}
                                 for name, label, description in zip(names, task.classes, task.descriptions)}}}}

    def estimate(self, view, task):
        payload = self.payload(view, task)
        if self.estimator is not None:
            return self.estimator(payload)
        # Includes repeated instructions, criteria, headers, examples and query.
        return TokenEstimate(math.ceil(len(canonical(payload).encode("utf-8")) / 2) + 64,
                             max(self.output_token_allowance, 32 * len(task.classes)), False)

    def predict(self, view, task):
        payload = self.payload(view, task)
        if self.transport is not None:
            response = self.transport(payload)
        else:
            headers = self.request_headers()
            request = Request(self.endpoint, canonical(payload).encode("utf-8"),
                              headers, method="POST")
            try:
                with urlopen(request, timeout=self.timeout) as reply:
                    response = json.load(reply)
            except HTTPError as exc:
                # Only translate the specific size rejection; do not hide other
                # failures or echo server content that might contain request data.
                body = exc.read()
                try:
                    detail = json.loads(body).get("detail", {})
                except (ValueError, AttributeError):
                    detail = {}
                if exc.code in (400, 413) and isinstance(detail, dict) and detail.get("error_type") == "max_tokens_exceeded":
                    raise JevContextLimitError(
                        "Jev rejected this request: max_tokens_exceeded. Reduce view size; "
                        "the local token estimate is approximate.") from None
                raise
        self.validate_response_model(response)
        answer = response["answers"]["classification"]
        probabilities = answer["probabilities"]
        expected = {f"class_{i}" for i in range(len(task.classes))}
        if set(probabilities) != expected:
            raise ValueError("Jev response class keys do not match the task")
        usage = response.get("usage") or {}
        return normalized(Prediction(tuple(probabilities[f"class_{i}"] for i in range(len(task.classes))),
                                     usage.get("input_tokens"), usage.get("output_tokens")), len(task.classes))

    def request_headers(self):
        key = self.api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY or pass api_key to JevProvider")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def validate_response_model(self, response):
        if response.get("model") != self.model:
            raise ValueError("Jev returned a different model version; refusing to cache under the requested version")
