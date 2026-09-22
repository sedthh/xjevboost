"""Adapters implement this small protocol; the policy never imports Jev."""
from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from .views import View, canonical


@dataclass(frozen=True)
class Task:
    classes: tuple
    instructions: str = ""
    descriptions: tuple[str, ...] = ()

    def payload(self):
        return {"classes": self.classes, "instructions": self.instructions,
                "descriptions": self.descriptions}


@dataclass(frozen=True)
class TokenEstimate:
    input_tokens: int
    output_tokens: int
    is_upper_bound: bool = False

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0
               for v in (self.input_tokens, self.output_tokens)):
            raise ValueError("Token estimates must be nonnegative integers")

    @property
    def total(self):
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Prediction:
    probabilities: tuple[float, ...]
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def actual_tokens(self):
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def normalized(result, n_classes):
    p = np.asarray(result.probabilities, dtype=float)
    if p.shape != (n_classes,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Provider must return finite nonnegative probabilities for every class")
    for usage in (result.input_tokens, result.output_tokens):
        if usage is not None and (not isinstance(usage, (int, np.integer)) or usage < 0):
            raise ValueError("Provider usage must be a nonnegative integer or None")
    return Prediction(tuple((p / p.sum()).tolist()), result.input_tokens, result.output_tokens)


class Provider(Protocol):
    """namespace must identify model version, configuration and adapter format.

    estimate() is local and must not make a prediction call. context_limit is
    the combined input/output allowance, or None if no limit applies.
    """
    namespace: str
    context_limit: int | None

    def estimate(self, view: View, task: Task) -> TokenEstimate: ...
    def predict(self, view: View, task: Task) -> Prediction: ...


class FakeProvider:
    """Deterministic mixed-value nearest-example classifier, entirely offline.

    Subclass predict() for task-specific synthetic behavior and give that
    subclass a different namespace when sharing a persistent cache.
    """
    namespace = "fake-nearest-example:v1"

    def __init__(self, context_limit=100_000):
        self.context_limit = context_limit
        self.calls = 0

    def estimate(self, view, task):
        # This fake provider defines its own exact synthetic token accounting.
        size = len(canonical({"view": view.payload(), "task": task.payload()}).encode("utf-8"))
        return TokenEstimate(math.ceil(size / 4), 8 * len(task.classes), True)

    def predict(self, view, task):
        self.calls += 1
        scores = np.full(len(task.classes), 0.05)
        query_by_column = dict(zip(view.effective_query_columns, view.query))
        projected_query = tuple(query_by_column[c] for c in view.columns)
        for row, label in view.examples:
            distance = 0.0
            for a, b in zip(row, projected_query):
                if a is None or b is None:
                    distance += float(a != b)
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    distance += min(abs(a - b) / (1 + abs(a) + abs(b)), 1.0)
                else:
                    distance += float(a != b)
            scores[label] += np.exp(-6 * distance)
        estimate = self.estimate(view, task)
        return Prediction(tuple(scores / scores.sum()), estimate.input_tokens, estimate.output_tokens)

