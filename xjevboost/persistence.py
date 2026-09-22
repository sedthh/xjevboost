"""Versioned inference artifacts; providers, credentials and caches are never serialized."""
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .core import AdaptivePolicy, Node
from .providers import Task
from .views import ViewBuilder, ViewRecipe

SETTINGS = ("max_depth", "gamma", "min_samples_leaf", "max_nodes", "screening_samples",
            "leaf_l2", "max_training_calls", "max_total_tokens", "full_view_budget",
            "token_margin", "strict_tokens", "optimization")

def save_model(clf, path):
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(clf, "policy_")
    p = clf.policy_
    def encode(n):
        return {"path": list(n.path), "weights": n.weights.tolist(), "kind": n.kind,
                "view": n.view, "feature": n.feature,
                "lower": None if math.isinf(n.lower) else n.lower,
                "upper": None if math.isinf(n.upper) else n.upper,
                **{k: encode(getattr(n, k)) for k in ("child", "inside", "outside") if getattr(n, k) is not None}}
    data = {"format": "xjevboost", "version": 1,
            "provider_fingerprint": hashlib.sha256(p.provider.namespace.encode()).hexdigest(),
            "settings": {k: getattr(p, k) for k in SETTINGS},
            "task": p.task.payload(), "columns": list(p.builder.columns),
            "named_input": hasattr(clf, "feature_names_in_"),
            "query_columns": clf.query_columns,
            "examples": {"rows": p.builder.rows, "labels": p.builder.labels.tolist(), "ids": p.builder.row_ids},
            "recipes": [asdict(r) for r in p.recipes], "tree": encode(p.tree),
            "prior": p.prior.tolist(), "single_view": p.single_view,
            "training_stats": clf.training_stats_, "inspection": clf.export_tree()}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def load_model(cls, path, provider, cache=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "xjevboost" or data.get("version") != 1:
        raise ValueError("Unsupported xjevboost model format/version")
    fingerprint = hashlib.sha256(provider.namespace.encode()).hexdigest()
    if fingerprint != data["provider_fingerprint"]:
        raise ValueError("Provider namespace does not match the saved model")
    settings = data["settings"]
    if set(settings) != set(SETTINGS):
        raise ValueError("Invalid model settings")
    task = Task(tuple(data["task"]["classes"]), data["task"]["instructions"], tuple(data["task"]["descriptions"]))
    examples = data["examples"]
    builder = ViewBuilder(tuple(tuple(r) for r in examples["rows"]), np.asarray(examples["labels"], dtype=int),
                          tuple(data["columns"]), examples["ids"])
    if len(set(builder.columns)) != len(builder.columns) or len(builder.rows) != len(builder.labels):
        raise ValueError("Invalid saved example schema")
    recipes = [ViewRecipe(r["id"], tuple(r["columns"]), tuple(r["column_indices"]), tuple(r["example_ids"]),
                          r["seed"], r["kind"], r["format_version"], r.get("query_mode", "view")) for r in data["recipes"]]
    for r in recipes:
        if (r.query_mode not in {"view", "all"} or not r.column_indices or len(set(r.column_indices)) != len(r.column_indices)
                or any(not 0 <= i < len(builder.columns) for i in r.column_indices)
                or r.columns != tuple(builder.columns[i] for i in r.column_indices)
                or any(i not in builder.row_ids for i in r.example_ids)):
            raise ValueError("Invalid view recipe")
    if any(len(r) != len(builder.columns) for r in builder.rows):
        raise ValueError("Invalid example row width")
    if any(not 0 <= label < len(task.classes) for label in builder.labels):
        raise ValueError("Invalid example class")
    clf = cls(provider=provider, cache=cache, n_views=len(recipes),
              query_columns=data.get("query_columns", "view"), task_instructions=task.instructions,
              class_descriptions=dict(zip(task.classes, task.descriptions)) if task.descriptions else None, **settings)
    clf._validate_settings()
    from .cache import PredictionCache
    actual_cache = cache if isinstance(cache, PredictionCache) else PredictionCache(cache)
    policy = AdaptivePolicy(provider, task, builder, recipes, cache=actual_cache, **settings)
    count = 0
    def decode(d, expected=(), depth=0):
        nonlocal count
        count += 1
        if depth > 200 or count > 10000 or tuple(d["path"]) != expected:
            raise ValueError("Invalid model tree path or size")
        weights = np.asarray(d["weights"], dtype=float)
        if (weights.shape != (len(expected),) or not np.isfinite(weights).all() or (weights < 0).any()
                or (len(expected) and not np.isclose(weights.sum(), 1))):
            raise ValueError("Invalid leaf weights")
        n = Node(expected, weights, d["kind"])
        if n.kind == "acquire":
            v = d["view"]
            if not isinstance(v, int) or not 0 <= v < len(recipes) or v in expected:
                raise ValueError("Invalid acquired view")
            n.view = v
            n.child = decode(d["child"], expected + (v,), depth + 1)
        elif n.kind == "route":
            n.feature = d["feature"]
            if not isinstance(n.feature, int) or not 0 <= n.feature < len(expected) * (len(task.classes) + 2):
                raise ValueError("Invalid split feature")
            n.lower = -math.inf if d["lower"] is None else float(d["lower"])
            n.upper = math.inf if d["upper"] is None else float(d["upper"])
            if not n.lower < n.upper:
                raise ValueError("Invalid split bounds")
            n.inside = decode(d["inside"], expected, depth + 1)
            n.outside = decode(d["outside"], expected, depth + 1)
        elif n.kind != "stop":
            raise ValueError("Invalid node kind")
        return n
    policy.tree = decode(data["tree"])
    policy.prior = np.asarray(data["prior"], dtype=float)
    if (policy.prior.shape != (len(task.classes),) or not np.isfinite(policy.prior).all()
            or (policy.prior < 0).any() or not np.isclose(policy.prior.sum(), 1)):
        raise ValueError("Invalid prior")
    policy.single_view = data["single_view"]
    if not isinstance(policy.single_view, int) or not 0 <= policy.single_view < len(recipes):
        raise ValueError("Invalid single-view baseline")
    clf.policy_, clf.views_, clf.tree_ = policy, recipes, policy.tree
    clf.classes_ = np.asarray(task.classes)
    clf.n_features_in_, clf.n_views_ = len(builder.columns), len(recipes)
    if data["named_input"]:
        clf.feature_names_in_ = np.asarray(builder.columns, dtype=object)
    clf.training_stats_ = data["training_stats"]
    clf._saved_inspection = data["inspection"]
    clf.training_leaf_stats_ = clf._training_leaf_stats()
    return clf
