"""Adaptive acquisition of fixed classifier views. No hosted SDK is imported."""

from .providers import FakeProvider, Prediction, Provider, Task, TokenEstimate
from .cache import PredictionCache

__all__ = ["XJevBoostClassifier", "FakeProvider", "Prediction", "Provider",
           "Task", "TokenEstimate", "PredictionCache"]


def __getattr__(name):
    # Importing the core or fake provider does not require scikit-learn.
    if name == "XJevBoostClassifier":
        from .classifier import XJevBoostClassifier
        return XJevBoostClassifier
    raise AttributeError(name)
