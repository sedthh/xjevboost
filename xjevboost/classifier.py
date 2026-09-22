"""Thin scikit-learn wrapper; all acquisition logic lives in core.py."""
from numbers import Integral, Real
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_fscore_support
from .splitting import stratified_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from .cache import PredictionCache
from .core import AdaptivePolicy, Node, brier_rows
from .providers import Task
from .views import ViewBuilder, cell, table


class XJevBoostClassifier(ClassifierMixin, BaseEstimator):
    """Learn an adaptive call tree from an example pool and calibration queries.

    With calibration_set=(X_cal, y_cal), all X/y supply labeled view examples.
    Alternatively set calibration_fraction in the constructor to reserve that
    fraction of X/y for learning the policy (stratified, reproducible split).
    Calibration labels train routing and aggregation, not merely probability
    calibration. Scores on them are fitting diagnostics, NOT holdout scores.
    Optional pruning_set=(X_prune, y_prune), or pruning_fraction, selects which continuations survive.
    Both fractions refer to the original X/y and must sum to less than one.
    Automatic splits are stratified and disjoint; identical feature rows stay
    together. Fractions are approximate when duplicate groups are present.
    Pruning selects continuations
    without fitting their thresholds or weights. It is selection data, not test
    data, and its full feature rows must not overlap the other input sets.

    n_views includes the balanced and broader recipes; duplicate recipes are
    removed. max_depth counts acquisitions, not comparisons of cached outputs.
    """

    def __init__(self, *, provider=None, task_instructions="", class_descriptions=None,
                 n_views=16, subsample=0.5, colsample_bytree=0.5, max_depth=3,
                 gamma=0.001, min_samples_leaf=10, random_state=None,
                 max_nodes=31, screening_samples=16, leaf_l2=0.01,
                 max_training_calls=None, max_total_tokens=None,
                 full_view_budget=True, token_margin=1.2, strict_tokens=False,
                 optimization="calls", cache=None, calibration_fraction=None,
                 pruning_fraction=None, query_columns="view"):
        self.query_columns = query_columns
        self.provider = provider
        self.task_instructions = task_instructions
        self.class_descriptions = class_descriptions
        self.n_views = n_views
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_nodes = max_nodes
        self.screening_samples = screening_samples
        self.leaf_l2 = leaf_l2
        self.max_training_calls = max_training_calls
        self.max_total_tokens = max_total_tokens
        self.full_view_budget = full_view_budget
        self.token_margin = token_margin
        self.strict_tokens = strict_tokens
        self.optimization = optimization
        self.cache = cache
        self.calibration_fraction = calibration_fraction
        self.pruning_fraction = pruning_fraction

    def _validate_settings(self):
        if self.query_columns not in {"view", "all", "mixed"}:
            raise ValueError("query_columns must be view, all, or mixed")
        for name in ("n_views", "max_depth", "min_samples_leaf", "max_nodes", "screening_samples"):
            value = getattr(self, name)
            minimum = 2 if name == "max_nodes" else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("subsample", "colsample_bytree"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("gamma", "leaf_l2", "token_margin"):
            value = getattr(self, name)
            minimum = 1 if name == "token_margin" else 0
            if not isinstance(value, Real) or not np.isfinite(value) or value < minimum:
                raise ValueError(f"{name} must be finite and >= {minimum}")
        for name in ("max_training_calls", "max_total_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
                raise ValueError(f"{name} must be None or a positive integer")
        if self.optimization not in {"calls", "tokens"}:
            raise ValueError("optimization must be 'calls' or 'tokens'")
        for name in ("calibration_fraction", "pruning_fraction"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                      or not 0 < value < 1):
                raise ValueError(f"{name} must be None or a fraction in (0, 1)")
        if (self.calibration_fraction or 0) + (self.pruning_fraction or 0) >= 1:
            raise ValueError("calibration_fraction and pruning_fraction must sum to less than 1")
        if self.provider is None:
            raise ValueError("Specify provider=FakeProvider() or a prediction adapter")
        if not isinstance(self.provider.namespace, str) or not self.provider.namespace:
            raise ValueError("provider.namespace must identify the provider/model version")

    def fit(self, X, y, *, calibration_set=None, pruning_set=None, verbose=False):
        if hasattr(self, "_saved_inspection"):
            del self._saved_inspection
        self._validate_settings()
        if (calibration_set is None) == (self.calibration_fraction is None):
            raise ValueError("Provide either calibration_set=(X_cal, y_cal) or calibration_fraction, but not both")
        if pruning_set is not None and self.pruning_fraction is not None:
            raise ValueError("Provide pruning_set or pruning_fraction, but not both")
        if calibration_set is not None and len(calibration_set) != 2:
            raise ValueError("calibration_set must be a pair (X_cal, y_cal)")
        # A failed refit must not leave an old policy advertised as fitted.
        for name in ("policy_", "classes_", "feature_names_in_", "calibration_indices_", "example_pool_indices_", "pruning_indices_", "inference_stats_"):
            self.__dict__.pop(name, None)
        rows, columns, named = table(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(rows) or not len(rows):
            raise ValueError("y must be one-dimensional and match nonempty X")
        classes = unique_labels(labels)
        if len(classes) < 2:
            raise ValueError("At least two classes are required")
        mapping = {label: i for i, label in enumerate(classes)}
        encoded = np.asarray([mapping[v] for v in labels])
        pool_indices = np.arange(len(rows))
        calibration_indices = None
        pruning_indices = None
        original_rows, original_encoded = rows, encoded
        if self.calibration_fraction is not None:
            pool_indices, calibration_indices = stratified_split(
                rows, encoded, pool_indices, int(np.ceil(len(rows) * self.calibration_fraction)),
                random_state=self.random_state)
            # Keep source ordering stable; recipe seeds control subsequent sampling.
            pool_indices, calibration_indices = np.sort(pool_indices), np.sort(calibration_indices)
            queries = tuple(rows[i] for i in calibration_indices)
            target = encoded[calibration_indices]
            rows, encoded = tuple(rows[i] for i in pool_indices), encoded[pool_indices]
            if len(np.unique(encoded)) != len(classes) or len(np.unique(target)) != len(classes):
                raise ValueError("The calibration split must retain every class on both sides; use more data or calibration_set")
        else:
            queries, _, _ = table(calibration_set[0], columns)
            valid_labels = np.asarray(calibration_set[1])
            if valid_labels.ndim != 1 or len(valid_labels) != len(queries):
                raise ValueError("calibration_set must have matching one-dimensional labels")
            try:
                target = np.asarray([mapping[v] for v in valid_labels])
            except (KeyError, TypeError) as exc:
                raise ValueError("calibration_set contains a class absent from y") from exc
        if len(queries) < self.min_samples_leaf:
            raise ValueError("Calibration requires at least min_samples_leaf rows")
        if self.pruning_fraction is not None:
            count = int(np.ceil(len(original_rows) * self.pruning_fraction))
            pool_indices, pruning_indices = stratified_split(
                original_rows, original_encoded, pool_indices, count, random_state=self.random_state)
            pool_indices, pruning_indices = np.sort(pool_indices), np.sort(pruning_indices)
            rows = tuple(original_rows[i] for i in pool_indices)
            encoded = original_encoded[pool_indices]
            if len(np.unique(encoded)) != len(classes):
                raise ValueError("The example split must retain every class; use more data or explicit sets")
            pruning_set = (np.asarray([original_rows[i] for i in pruning_indices], dtype=object),
                           labels[pruning_indices])
        from .views import canonical
        example_keys = {canonical(row) for row in rows}
        if any(canonical(row) in example_keys for row in queries):
            raise ValueError("calibration_set features must not overlap examples; split duplicate groups together using explicit sets")
        pruning_rows = 0
        if pruning_set is not None:
            if len(pruning_set) != 2:
                raise ValueError("pruning_set must be (X_prune, y_prune)")
            prune_queries, _, _ = table(pruning_set[0], columns)
            prune_labels = np.asarray(pruning_set[1])
            if prune_labels.ndim != 1 or len(prune_labels) != len(prune_queries) or len(prune_queries) < self.min_samples_leaf:
                raise ValueError("pruning_set must have matching labels and at least min_samples_leaf rows")
            try:
                prune_target = np.asarray([mapping[v] for v in prune_labels])
            except (KeyError, TypeError) as exc:
                raise ValueError("pruning_set contains an unknown class") from exc
            from .views import canonical
            previous = {canonical(row) for row in rows + queries}
            if any(canonical(row) in previous for row in prune_queries):
                raise ValueError("pruning_set features must not overlap examples or calibration queries; split duplicate groups together")
            pruning_rows = len(prune_queries)
            queries = queries + prune_queries
            target = np.concatenate([target, prune_target])
        descriptions = ()
        if self.class_descriptions is not None:
            try:
                descriptions = tuple(self.class_descriptions[v] for v in classes)
            except (KeyError, TypeError) as exc:
                raise ValueError("class_descriptions must map every original class label to a description") from exc
        task = Task(tuple(cell(v) for v in classes), self.task_instructions, descriptions)
        if hasattr(self.provider, "validate_task"):
            self.provider.validate_task(task)
        builder = ViewBuilder(rows, encoded, columns, row_ids=pool_indices)
        recipes = builder.recipes(self.n_views, self.subsample, self.colsample_bytree, self.random_state, self.query_columns)
        cache = self.cache if isinstance(self.cache, PredictionCache) else PredictionCache(self.cache)
        settings = {name: getattr(self, name) for name in (
            "max_depth", "gamma", "min_samples_leaf", "max_nodes", "screening_samples",
            "random_state", "leaf_l2", "max_training_calls", "max_total_tokens",
            "full_view_budget", "token_margin", "strict_tokens", "optimization")}
        policy = AdaptivePolicy(self.provider, task, builder, recipes, cache=cache, verbose=verbose, **settings)
        policy.fit(queries, target, pruning_rows=pruning_rows)
        self.classes_ = classes
        self.n_features_in_ = len(columns)
        if named:
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.policy_, self.views_, self.tree_ = policy, recipes, policy.tree
        self.training_stats_ = dict(policy.stats)
        self.training_stats_["classes"] = list(task.classes)
        self.pruning_history_ = list(policy.pruning_history)
        self.n_views_ = len(recipes)
        self.example_pool_indices_ = pool_indices
        self.calibration_indices_ = calibration_indices
        self.pruning_indices_ = pruning_indices
        self.calibration_results_ = {"policy_fit_brier": policy.stats["policy_fit_brier"],
                                     "single_view_brier": policy.stats["single_view_brier"]}
        self.training_leaf_stats_ = self._training_leaf_stats()
        return self

    def _training_leaf_stats(self):
        def leaves(node):
            if node["kind"] == "stop":
                yield {"stop_node": node["location"], **node["training_path_stats"]}
            for key in ("child", "inside", "outside"):
                if key in node:
                    yield from leaves(node[key])
        return list(leaves(self.export_tree()["root"]))

    def predict_with_trace(self, X):
        check_is_fitted(self, "policy_")
        rows, _, _ = table(X, self.policy_.builder.columns)
        started = time.perf_counter()
        traces = [self.policy_.predict_one(row) for row in rows]
        for trace in traces:
            trace["prediction"] = cell(self.classes_[np.argmax(trace["probabilities"])])
        self.inference_stats_ = self.summarize_traces(traces)
        self.inference_stats_["elapsed_seconds"] = time.perf_counter() - started
        return traces

    def save_model(self, path):
        """Save versioned JSON, including labeled examples but no provider or API key."""
        from .persistence import save_model
        save_model(self, path)

    @classmethod
    def load_model(cls, path, *, provider, cache=None):
        """Load for inference with an explicitly supplied matching provider."""
        from .persistence import load_model
        return load_model(cls, path, provider, cache)

    def summarize_traces(self, traces):
        """Summarize existing traces without calls; does not change last-run stats."""
        check_is_fitted(self, "policy_")
        from .stats import summarize_traces
        return summarize_traces(traces, max_depth=self.policy_.max_depth, n_views=len(self.views_))

    def predict_proba(self, X):
        return np.asarray([t["probabilities"] for t in self.predict_with_trace(X)]).reshape(-1, len(self.classes_))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def plot_tree(self, *, ax=None, fontsize=9):
        """Draw the fitted policy without provider calls; returns Matplotlib Axes."""
        from .plotting import plot_tree
        return plot_tree(self, ax=ax, fontsize=fontsize)

    def export_tree(self):
        """Export JSON-safe nodes and complete used-view examples, without calls."""
        from .plotting import export_tree
        return export_tree(self)

    def evaluate(self, X, y, *, policy="adaptive", view_ids=None):
        """Test metrics for adaptive, single_view, or fixed_sequence policies.

        fixed_sequence averages the explicitly selected views uniformly and
        respects the same depth/context/token budgets. No test labels select
        views or weights. single_view uses the screened root chosen at fit.
        """
        check_is_fitted(self, "policy_")
        rows, _, _ = table(X, self.policy_.builder.columns)
        mapping = {v: i for i, v in enumerate(self.classes_)}
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(rows) or not len(rows):
            raise ValueError("y must match nonempty X")
        try:
            target = np.eye(len(mapping))[[mapping[v] for v in labels]]
        except (KeyError, TypeError) as exc:
            raise ValueError("Unknown test class") from exc
        tree = None
        if policy in {"single_view", "fixed_sequence"}:
            if policy == "single_view":
                sequence = [self.policy_.single_view]
            else:
                lookup = {r.id: i for i, r in enumerate(self.views_)}
                if not view_ids or len(set(view_ids)) != len(view_ids) or len(view_ids) > self.max_depth:
                    raise ValueError("Provide unique view_ids with length <= max_depth")
                try:
                    sequence = [lookup[v] for v in view_ids]
                except KeyError as exc:
                    raise ValueError("Unknown view ID") from exc
            tree = Node((), np.array([]))
            node = tree
            for j, v in enumerate(sequence):
                node.kind, node.view = "acquire", v
                node.child = Node(tuple(sequence[:j + 1]), np.full(j + 1, 1 / (j + 1)))
                node = node.child
        elif policy != "adaptive":
            raise ValueError("policy must be adaptive, single_view, or fixed_sequence")
        started = time.perf_counter()
        traces = [self.policy_.predict_one(row, tree=tree) for row in rows]
        stats = self.summarize_traces(traces)
        stats["elapsed_seconds"] = time.perf_counter() - started
        self.inference_stats_ = stats
        probabilities = np.asarray([t["probabilities"] for t in traces])
        precision, recall, f1, support = precision_recall_fscore_support(
            target.argmax(axis=1), probabilities.argmax(axis=1),
            labels=np.arange(len(mapping)), zero_division=0)
        return {**stats, "precision_per_class": precision.tolist(),
                "recall_per_class": recall.tolist(), "f1_per_class": f1.tolist(),
                "support_per_class": support.tolist(), "brier": float(brier_rows(probabilities, target).mean()),
                "accuracy": float((probabilities.argmax(axis=1) == target.argmax(axis=1)).mean()),
                "log_loss": float(-np.log(np.clip(probabilities[np.arange(len(rows)), target.argmax(axis=1)], 1e-15, 1)).mean()),
                "training_calls": self.training_stats_["provider_calls"]}
