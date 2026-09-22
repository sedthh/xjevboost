"""Provider-independent adaptive policy. Only NumPy is needed by this module."""
from dataclasses import dataclass
import math
import sys
import time
import warnings

import numpy as np

from .cache import PredictionCache
from .providers import normalized


class BudgetError(ValueError):
    pass


class TrainingLimit(RuntimeError):
    pass


def brier_rows(p, target):
    return np.mean((p - target) ** 2, axis=1)


def fit_weights(outputs, target, l2=0.01):
    """Projected gradient on the simplex; L2 shrinks toward uniform weights."""
    n, m, k = outputs.shape
    weights = np.full(m, 1 / m)
    if m == 1:
        return weights
    flat = outputs.transpose(0, 2, 1).reshape(n * k, m)
    gram = flat.T @ flat / (n * k) + l2 * np.eye(m)
    linear = flat.T @ target.reshape(-1) / (n * k) + l2 / m
    step = 1 / max(2 * np.linalg.eigvalsh(gram)[-1], 1e-12)
    for _ in range(150):
        proposal = weights - step * 2 * (gram @ weights - linear)
        ordered = np.sort(proposal)[::-1]
        shifts = (np.cumsum(ordered) - 1) / np.arange(1, m + 1)
        rho = np.flatnonzero(ordered > shifts)[-1]
        updated = np.maximum(proposal - shifts[rho], 0)
        if np.max(np.abs(updated - weights)) < 1e-8:
            return updated / updated.sum()
        weights = updated
    return weights / weights.sum()


def output_features(outputs):
    """Every class probability, predicted class, and top-two gap for each view."""
    ordered = np.sort(outputs, axis=2)
    return np.concatenate([outputs.reshape(len(outputs), -1),
                           outputs.argmax(axis=2), ordered[:, :, -1] - ordered[:, :, -2]], axis=1)


@dataclass
class Node:
    path: tuple[int, ...]
    weights: np.ndarray
    kind: str = "stop"
    view: int | None = None
    child: "Node | None" = None
    feature: int | None = None
    lower: float = -math.inf
    upper: float = math.inf
    inside: "Node | None" = None
    outside: "Node | None" = None


class AdaptivePolicy:
    def __init__(self, provider, task, builder, recipes, *, max_depth=3,
                 gamma=0.001, min_samples_leaf=10, max_nodes=31,
                 screening_samples=16, random_state=None, leaf_l2=0.01,
                 max_training_calls=None, max_total_tokens=None,
                 full_view_budget=True, token_margin=1.2, strict_tokens=False,
                 optimization="calls", cache=None, verbose=False):
        self.provider, self.task, self.builder, self.recipes = provider, task, builder, recipes
        self.max_depth, self.gamma, self.min_samples_leaf = max_depth, gamma, min_samples_leaf
        self.max_nodes, self.screening_samples, self.leaf_l2 = max_nodes, screening_samples, leaf_l2
        self.max_training_calls, self.max_total_tokens = max_training_calls, max_total_tokens
        self.full_view_budget, self.token_margin, self.strict_tokens = full_view_budget, token_margin, strict_tokens
        self.optimization, self.verbose = optimization, verbose
        self.cache = cache if cache is not None else PredictionCache()
        self.rng = np.random.default_rng(random_state)
        self.stats = {"provider_calls": 0, "cache_hits": 0, "actual_tokens": 0,
                      "calls_without_usage": 0, "estimated_tokens": 0,
                      "budget_exhausted": False, "screening_stages": 0}
        self._matrix = {}
        self._nodes = 0
        self._reserved_nodes = 0
        self._warned_estimates = False

    def reserve(self, estimate):
        if self.strict_tokens and not estimate.is_upper_bound:
            raise BudgetError("strict_tokens requires a reliable pre-call upper bound; "
                              "this provider supplies only an estimate. Supply a bounded estimator. "
                              "Approximate budgets require explicit strict_tokens=False.")
        if not estimate.is_upper_bound and not self._warned_estimates:
            warnings.warn(
                "Provider token costs are approximate. The estimated cumulative ceiling "
                "is enforced before each call, but actual usage can exceed it. "
                "The hypothetical full view is estimated locally and is never sent, "
                "even if it exceeds the context window. Use strict_tokens=True "
                "to require reliable pre-call bounds.", RuntimeWarning, stacklevel=2)
            self._warned_estimates = True
        return estimate.total if estimate.is_upper_bound else math.ceil(estimate.total * self.token_margin)

    def ceiling(self, query):
        limit = math.inf if self.max_total_tokens is None else self.max_total_tokens
        if self.full_view_budget:
            limit = min(limit, self.reserve(self.provider.estimate(self.builder.full(query), self.task)))
        return limit

    def estimate(self, query, view):
        return self.provider.estimate(self.builder.build(self.recipes[view], query), self.task)

    def _charge(self, result, estimate):
        # Retain a reservation for each unreported component, rather than
        # silently interpreting unknown usage as zero.
        margin = 1 if estimate.is_upper_bound else self.token_margin
        return ((result.input_tokens if result.input_tokens is not None else math.ceil(estimate.input_tokens * margin))
                + (result.output_tokens if result.output_tokens is not None else math.ceil(estimate.output_tokens * margin)))

    def _check_bound(self, result, estimate):
        if estimate.is_upper_bound and (
            (result.input_tokens is not None and result.input_tokens > estimate.input_tokens)
            or (result.output_tokens is not None and result.output_tokens > estimate.output_tokens)
        ):
            raise BudgetError("Provider violated its declared token upper bound")

    def _request(self, query, view, training=False):
        built = self.builder.build(self.recipes[view], query)
        key = self.cache.key(self.provider, self.task, built)
        cached = self.cache.get(key)
        if cached is not None:
            if training:
                self.stats["cache_hits"] += 1
            result = normalized(cached, len(self.task.classes))
            self._check_bound(result, self.provider.estimate(built, self.task))
            return result, True
        if training and self.max_training_calls is not None and self.stats["provider_calls"] >= self.max_training_calls:
            self.stats["budget_exhausted"] = True
            raise TrainingLimit("max_training_calls reached")
        estimate = self.provider.estimate(built, self.task)
        reservation = self.reserve(estimate)
        if self.provider.context_limit is not None and reservation > self.provider.context_limit:
            raise BudgetError("View exceeds provider per-call context limit")
        # Count attempts before calling. No automatic retries: a failed request
        # could already have incurred provider usage.
        if training:
            self.stats["provider_calls"] += 1
            self.stats["estimated_tokens"] += reservation
            self._progress()
        result = normalized(self.provider.predict(built, self.task), len(self.task.classes))
        self._check_bound(result, estimate)
        self.cache.put(key, result)
        if training:
            if result.actual_tokens is None:
                self.stats["calls_without_usage"] += 1
            else:
                self.stats["actual_tokens"] += result.actual_tokens
        return result, False

    def _progress(self, message=None):
        if not self.verbose:
            return
        now = time.monotonic()
        if message is not None or now - self._last_progress >= 1:
            s = self.stats
            calls = str(s['provider_calls'])
            batch = getattr(self, '_batch_progress', None)
            rows = ""
            if batch is not None:
                view, completed, total = batch
                rows = f"view={view} rows={completed}/{total} ({100 * completed / total:.1f}% of batch) | "
            print(f"[xjevboost] {self._stage} | {message or 'calling provider'} | {rows}calls={calls} "
                  f"cache_hits={s['cache_hits']} tokens={s['actual_tokens']} "
                  f"elapsed={now - self._started:.1f}s", file=sys.stderr, flush=True)
            self._last_progress = now

    def _get(self, indices, view):
        result = []
        pending = any((int(i), view) not in self._matrix for i in indices)
        try:
            for position, i in enumerate(indices):
                if pending:
                    self._batch_progress = (self.recipes[view].id, position, len(indices))
                key = (int(i), view)
                if key not in self._matrix:
                    prediction, _ = self._request(self.queries[i], view, training=True)
                    self._matrix[key] = prediction
                result.append(self._matrix[key].probabilities)
                if pending:
                    self._batch_progress = (self.recipes[view].id, position + 1, len(indices))
                    self._progress("view batch complete" if position + 1 == len(indices) else None)
        finally:
            self._batch_progress = None
        return np.asarray(result)

    def _outputs(self, indices, path):
        return np.stack([self._get(indices, v) for v in path], axis=1)

    def _fits(self, indices, path, view):
        for i in indices:
            query = self.queries[i]
            estimate = self.estimate(query, view)
            cost = self.reserve(estimate)
            if self.provider.context_limit is not None and cost > self.provider.context_limit:
                return False
            previous = sum(self._charge(self._matrix[(int(i), v)], self.estimate(query, v)) for v in path)
            if previous + cost > self._ceilings[i]:
                return False
        return True

    def _screen(self, indices, path, stop_weights):
        """Local staged screening. Eliminated views return at every new node."""
        candidates = [v for v in range(len(self.recipes)) if v not in path]
        order = self.rng.permutation(indices)
        size = min(len(order), max(self.screening_samples, 2 * self.min_samples_leaf))
        records = {}
        while candidates:
            self.stats["screening_stages"] += 1
            sample = order[:size]
            self._progress(f"depth={len(path)} screening={len(candidates)} queries={size}/{len(indices)}")
            old = self._outputs(sample, path) if path else None
            stop = (np.einsum("nvc,v->nc", old, stop_weights)
                    if path else np.tile(self.prior, (size, 1)))
            stop_errors = brier_rows(stop, self.target[sample])
            records = {}
            for v in candidates:
                if not self._fits(sample, path, v):
                    continue
                try:
                    p = self._get(sample, v)
                except TrainingLimit:
                    continue
                combined = np.concatenate([old, p[:, None, :]], axis=1) if path else p[:, None, :]
                weights = fit_weights(combined, self.target[sample], self.leaf_l2)
                errors = brier_rows(np.einsum("nvc,v->nc", combined, weights), self.target[sample])
                delta = stop_errors - errors
                cost = (1.0 if self.optimization == "calls" else
                        max(1, np.mean([self.reserve(self.estimate(self.queries[i], v)) for i in sample])))
                uncertainty = float(np.std(delta, ddof=1) / np.sqrt(size)) if size > 1 else 0.0
                records[v] = {"errors": errors, "weights": weights,
                              "loss": float(errors.mean()),
                              "rank": (float(delta.mean()) + uncertainty) / cost}
            if not records or size == len(order) or len(records) <= 2:
                return sample, records
            # Upper-confidence ranking gives still-uncertain views a chance.
            candidates = sorted(records, key=lambda v: records[v]["rank"], reverse=True)[:max(2, math.ceil(len(records) / 2))]
            size = min(len(order), size * 2)
        return order[:size], records

    def fit(self, queries, labels, *, pruning_rows=0):
        self.queries = queries
        self.target = np.eye(len(self.task.classes))[labels]
        counts = np.bincount(self.builder.labels, minlength=len(self.task.classes))
        self.prior = counts / counts.sum()
        self._started = self._last_progress = time.monotonic()
        self._stage = "Growth"
        self._ceilings = [self.ceiling(q) for q in queries]
        indices = np.arange(len(queries) - pruning_rows)
        self.growth_indices = indices
        pruning_indices = np.arange(len(indices), len(queries))
        self.pruning_history = []
        self.stats.update(growth_rows=len(indices), pruning_rows=pruning_rows,
                          growth_class_counts=np.bincount(labels[indices], minlength=len(self.task.classes)).tolist(),
                          pruning_class_counts=np.bincount(labels[pruning_indices], minlength=len(self.task.classes)).tolist())
        sample, candidates = self._screen(indices, (), np.array([]))
        if not candidates:
            raise BudgetError("No root view could be evaluated; reduce view size or increase training/token budgets")
        # Quality selects the final root. Ratios only allocate screening calls.
        ranked = sorted(candidates, key=lambda v: candidates[v]["loss"])
        root = None
        for v in ranked:
            if not self._fits(indices, (), v):
                continue
            try:
                self._get(indices, v)
            except TrainingLimit:
                continue
            root = v
            break
        if root is None:
            raise BudgetError("No root fits all policy rows within the training/context/token limits")
        self.single_view = root
        self._progress(f"selected root={self.recipes[root].id}")
        self._nodes = 1
        self.tree = Node((), np.array([]), "acquire", view=root)
        self.tree.child = self._grow(indices, (root,))
        self.stats["nodes_before_pruning"] = self._node_count(self.tree)
        self.stats["growth_brier_before_pruning"] = float(brier_rows(self._training_predict(self.tree, indices), self.target[indices]).mean())
        self.stats["pruning_brier"] = None
        self.stats["pruning_single_view_brier"] = None
        calls_before = self.stats["provider_calls"]
        self.stats["growth_provider_calls"] = calls_before
        self.stats["growth_elapsed_seconds"] = time.monotonic() - self._started
        if pruning_rows:
            self._stage = "Pruning"
            self._progress(f"pruning on {pruning_rows} separate rows")
            try:
                if not self._fits(pruning_indices, (), root):
                    raise BudgetError("Root view does not fit pruning rows")
                root_p = self._get(pruning_indices, root)
                self.tree.child = self._prune(self.tree.child, pruning_indices, "root")
                prediction = self._training_predict(self.tree, pruning_indices)
                self.stats["pruning_brier"] = float(brier_rows(prediction, self.target[pruning_indices]).mean())
                self.stats["pruning_single_view_brier"] = float(brier_rows(root_p, self.target[pruning_indices]).mean())
            except (TrainingLimit, BudgetError):
                # Never keep an unvalidated continuation when pruning cannot run.
                self.tree.child = Node((root,), np.ones(1))
                self.pruning_history.append({"location": "root", "decision": "prune", "reason": "validation_budget_unavailable", "rows": pruning_rows})
        self.stats["pruning_provider_calls"] = self.stats["provider_calls"] - calls_before
        self.stats["pruning_elapsed_seconds"] = time.monotonic() - self._started - self.stats["growth_elapsed_seconds"]
        self.stats["nodes_after_pruning"] = self._node_count(self.tree)
        self.stats["nodes_removed"] = self.stats["nodes_before_pruning"] - self.stats["nodes_after_pruning"]
        self.stats["pruning_decisions"] = len(self.pruning_history)
        p = self._training_predict(self.tree, indices)
        self.stats["policy_fit_brier"] = float(brier_rows(p, self.target[indices]).mean())
        self.stats["growth_accuracy"] = float((p.argmax(axis=1) == labels[indices]).mean())
        self.stats["growth_prior_brier"] = float(brier_rows(np.tile(self.prior, (len(indices), 1)), self.target[indices]).mean())
        self.stats["single_view_brier"] = float(brier_rows(self._get(indices, root), self.target[indices]).mean())
        self.stats["unique_query_view_pairs"] = len(self._matrix)
        self.stats["query_view_upper_bound"] = len(queries) * len(self.recipes)
        dense = self.stats["query_view_upper_bound"]
        evaluated = self.stats["unique_query_view_pairs"]
        self.stats["query_view_pairs_skipped"] = dense - evaluated
        self.stats["calls_saved_by_cache"] = evaluated - self.stats["provider_calls"]
        self.stats["screening_reduction_pct"] = 100 * (1 - evaluated / dense)
        self.stats["provider_call_reduction_pct"] = 100 * (1 - self.stats["provider_calls"] / dense)
        self.stats["elapsed_seconds"] = time.monotonic() - self._started
        n_queries = len(queries)
        calls = self.stats["provider_calls"]
        actual = self.stats["actual_tokens"] if not self.stats["calls_without_usage"] else None
        self.stats["average_provider_calls_per_training_query"] = calls / n_queries
        self.stats["average_new_actual_tokens_per_training_query"] = actual / n_queries if actual is not None else None
        self.stats["average_new_actual_tokens_per_provider_call"] = actual / calls if actual is not None and calls else None
        self.stats["average_evaluated_views_per_training_query"] = evaluated / n_queries
        self._stage = "Done"
        self._progress(f"done policy_fit_brier={self.stats['policy_fit_brier']:.6f}")
        return self

    @staticmethod
    def _node_count(node):
        if node.kind == "acquire":
            return 1 + AdaptivePolicy._node_count(node.child)
        if node.kind == "route":
            return 1 + AdaptivePolicy._node_count(node.inside) + AdaptivePolicy._node_count(node.outside)
        return 1

    def _prune(self, node, indices, location):
        """Bottom-up selection only: never fit weights/thresholds on pruning labels."""
        if node.kind == "stop":
            return node
        stop = Node(node.path, node.weights.copy())
        record = {"location": location, "kind": node.kind, "rows": len(indices),
                  "path": [self.recipes[v].id for v in node.path],
                  "class_counts": self.target[indices].sum(axis=0).astype(int).tolist()}
        if len(indices) < self.min_samples_leaf:
            record.update(decision="prune", reason="insufficient_pruning_rows")
            self.pruning_history.append(record)
            return stop
        before_p = self._training_predict(stop, indices)
        try:
            if node.kind == "acquire":
                if not self._fits(indices, node.path, node.view):
                    raise BudgetError("Continuation does not fit pruning path budgets")
                self._get(indices, node.view)
                node.child = self._prune(node.child, indices, location + ".next")
            else:
                values = output_features(self._outputs(indices, node.path))[:, node.feature]
                mask = (values > node.lower) & (values <= node.upper)
                node.inside = self._prune(node.inside, indices[mask], location + ".inside")
                node.outside = self._prune(node.outside, indices[~mask], location + ".outside")
            after_p = self._training_predict(node, indices)
        except (TrainingLimit, BudgetError):
            record.update(decision="prune", reason="validation_budget_unavailable")
            self.pruning_history.append(record)
            return stop
        before = float(brier_rows(before_p, self.target[indices]).mean())
        after = float(brier_rows(after_p, self.target[indices]).mean())
        keep = before - after > self.gamma
        record.update(stop_brier=before, subtree_brier=after, improvement=before - after,
                      decision="keep" if keep else "prune", reason="held_out_improvement" if keep else "insufficient_improvement")
        self.pruning_history.append(record)
        self._progress(f"pruning {location}: {record['decision']} gain={before - after:.6f} rows={len(indices)}")
        return node if keep else stop

    def _grow(self, indices, path):
        self._nodes += 1
        old = self._outputs(indices, path)
        weights = fit_weights(old, self.target[indices], self.leaf_l2)
        node = Node(path, weights)
        if (len(path) >= self.max_depth or self._nodes + 1 + self._reserved_nodes > self.max_nodes
                or len(indices) < self.min_samples_leaf
                or len(path) == len(self.recipes) or self.stats["budget_exhausted"]):
            return node
        sample, records = self._screen(indices, path, weights)
        if not records:
            return node
        sample_outputs = self._outputs(sample, path)
        stop_errors = brier_rows(np.einsum("nvc,v->nc", sample_outputs, weights), self.target[sample])
        options = [None] + list(records)
        losses = np.column_stack([stop_errors] + [records[v]["errors"] for v in records])

        def choose(mask):
            means = losses[mask].mean(axis=0)
            best = int(np.argmin(means))
            if best and means[0] - means[best] <= self.gamma:
                best = 0
            return float(means[best]), options[best]

        best_loss, direct = choose(np.ones(len(sample), dtype=bool))
        split = None
        features = output_features(sample_outputs)
        if len(sample) >= 2 * self.min_samples_leaf and self._nodes + 4 + self._reserved_nodes <= self.max_nodes:
            for feature in range(features.shape[1]):
                values = np.unique(features[:, feature])
                if len(values) < 2:
                    continue
                mids = values[:-1] + (values[1:] - values[:-1]) / 2
                if len(mids) > 8:
                    mids = mids[np.unique(np.linspace(0, len(mids) - 1, 8).astype(int))]
                edges = np.r_[-np.inf, mids, np.inf]
                for lo in range(len(edges) - 1):
                    for hi in range(lo + 1, len(edges)):
                        mask = (features[:, feature] > edges[lo]) & (features[:, feature] <= edges[hi])
                        if min(mask.sum(), (~mask).sum()) < self.min_samples_leaf:
                            continue
                        inside_loss, a = choose(mask)
                        outside_loss, b = choose(~mask)
                        if a == b:
                            continue
                        loss = (inside_loss * mask.sum() + outside_loss * (~mask).sum()) / len(sample)
                        if loss < best_loss - max(self.gamma, 1e-10):
                            best_loss = loss
                            split = (feature, float(edges[lo]), float(edges[hi]))
        if split is not None:
            feature, lower, upper = split
            values = output_features(old)[:, feature]
            mask = (values > lower) & (values <= upper)
            if min(mask.sum(), (~mask).sum()) >= self.min_samples_leaf:
                node.kind, node.feature, node.lower, node.upper = "route", feature, lower, upper
                self._reserved_nodes += 1
                node.inside = self._grow(indices[mask], path)
                self._reserved_nodes -= 1
                node.outside = self._grow(indices[~mask], path)
        elif direct is not None and self._fits(indices, path, direct):
            try:
                self._get(indices, direct)
            except TrainingLimit:
                return node
            node.kind, node.view = "acquire", direct
            node.child = self._grow(indices, path + (direct,))
        if node.kind != "stop":
            before = brier_rows(np.einsum("nvc,v->nc", old, weights), self.target[indices]).mean()
            after = brier_rows(self._training_predict(node, indices), self.target[indices]).mean()
            if before - after <= self.gamma:
                return Node(path, weights)
        return node

    def _training_predict(self, node, indices):
        if node.kind == "acquire":
            return self._training_predict(node.child, indices)
        outputs = self._outputs(indices, node.path)
        if node.kind == "stop":
            return np.einsum("nvc,v->nc", outputs, node.weights)
        values = output_features(outputs)[:, node.feature]
        mask = (values > node.lower) & (values <= node.upper)
        p = np.empty((len(indices), len(self.task.classes)))
        if mask.any():
            p[mask] = self._training_predict(node.inside, indices[mask])
        if (~mask).any():
            p[~mask] = self._training_predict(node.outside, indices[~mask])
        return p

    def predict_one(self, query, tree=None, max_depth=None):
        node = self.tree if tree is None else tree
        location = "root"
        acquired, steps, decisions = {}, [], []
        ceiling, spent = self.ceiling(query), 0
        full_estimate = (self.provider.estimate(self.builder.full(query), self.task)
                         if self.full_view_budget else None)
        strict = (not self.full_view_budget or
                  self.provider.estimate(self.builder.full(query), self.task).is_upper_bound)
        reason = "leaf"
        depth = self.max_depth if max_depth is None else min(max_depth, self.max_depth)
        while True:
            stop = (np.asarray([acquired[v] for v in node.path]).T @ node.weights
                    if node.path else self.prior)
            if node.kind == "stop":
                break
            if node.kind == "route":
                value = float(output_features(np.asarray([[acquired[v] for v in node.path]]))[0, node.feature])
                inside = node.lower < value <= node.upper
                names = ([f"{self.recipes[v].id}.p[{label!r}]" for v in node.path for label in self.task.classes]
                         + [f"{self.recipes[v].id}.predicted_class_index" for v in node.path]
                         + [f"{self.recipes[v].id}.top_two_gap" for v in node.path])
                decisions.append({"feature": node.feature, "feature_name": names[node.feature],
                                  "path": [self.recipes[v].id for v in node.path],
                                  "lower": None if math.isinf(node.lower) else node.lower,
                                  "upper": None if math.isinf(node.upper) else node.upper,
                                  "value": value, "branch": "inside" if inside else "outside"})
                node = node.inside if inside else node.outside
                location += ".inside" if inside else ".outside"
                continue
            if len(acquired) >= depth:
                reason = "max_depth"
                break
            estimate = self.estimate(query, node.view)
            reservation = self.reserve(estimate)
            strict = strict and estimate.is_upper_bound
            context_exceeded = self.provider.context_limit is not None and reservation > self.provider.context_limit
            if context_exceeded or spent + reservation > ceiling:
                reason = "context_limit" if context_exceeded else "token_limit"
                if not acquired:
                    raise BudgetError("First view cannot fit this query; reduce view size or increase its budget")
                break
            result, hit = self._request(query, node.view)
            acquired[node.view] = result.probabilities
            charged = self._charge(result, estimate)
            spent += charged
            recipe = self.recipes[node.view]
            steps.append({"view_id": recipe.id, "columns": list(recipe.columns),
                          "query_mode": recipe.query_mode,
                          "query_columns": list(self.builder.columns if recipe.query_mode == "all" else recipe.columns),
                          "example_ids": list(recipe.example_ids), "seed": recipe.seed,
                          "format_version": recipe.format_version,
                          "estimated_input_tokens": estimate.input_tokens,
                          "estimated_output_tokens": estimate.output_tokens,
                          "reserved_tokens": reservation, "is_upper_bound": estimate.is_upper_bound,
                          "actual_input_tokens": result.input_tokens, "actual_output_tokens": result.output_tokens,
                          "cache_hit": hit, "probabilities": list(result.probabilities)})
            node = node.child
            location += ".child"
        known = all(s["actual_input_tokens"] is not None and s["actual_output_tokens"] is not None for s in steps)
        new_steps = [s for s in steps if not s["cache_hit"]]
        new_known = all(s["actual_input_tokens"] is not None and s["actual_output_tokens"] is not None for s in new_steps)
        return {"probabilities": stop.tolist(), "steps": steps, "branches": decisions,
                "stop_reason": reason, "stop_node": location, "calls": len(steps),
                "provider_calls": sum(not s["cache_hit"] for s in steps),
                "token_ceiling": None if math.isinf(ceiling) else ceiling,
                "full_view_estimated_tokens": full_estimate.total if full_estimate is not None else None,
                "accounted_tokens": spent, "budget_kind": "bounded" if strict else "estimated",
                "budget_exceeded": spent > ceiling,
                "new_actual_tokens": sum(s["actual_input_tokens"] + s["actual_output_tokens"] for s in new_steps) if new_known else None,
                "actual_tokens": sum(s["actual_input_tokens"] + s["actual_output_tokens"] for s in steps) if known else None}
