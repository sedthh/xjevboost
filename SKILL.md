---
name: xjevboost
description: Use xjevboost to fit and evaluate adaptive Jev or provider-neutral classifiers, or integrate a prediction adapter. Applies to xjevboost usage, not ordinary XGBoost training.
---

# Use xjevboost

An xjevboost model learns which fixed row-and-column view to acquire next and
when to stop. It does not fine-tune Jev or implement gradient boosting.

## Fit with the correct data roles

Use one of these two patterns:

```python
from xjevboost import FakeProvider, XJevBoostClassifier

# Reserve 20% for growth, 20% for pruning, and 60% for view examples.
clf = XJevBoostClassifier(
    provider=FakeProvider(), calibration_fraction=0.2, pruning_fraction=0.2,
    n_views=8, max_depth=2, random_state=42,
)
clf.fit(X_train, y_train, verbose=True)

# Or supply the routing data explicitly; do not also set calibration_fraction.
clf = XJevBoostClassifier(provider=FakeProvider(), random_state=42)
clf.fit(X_train, y_train, calibration_set=(X_cal, y_cal),
        pruning_set=(X_prune, y_prune), verbose=True)
```

Each role accepts either a constructor fraction or an explicit `(X, y)` pair
in `fit`, never both for the same role. Calibration is required; pruning is
optional. Explicit and automatic roles can be mixed. Fractions refer to the
original training set and must sum to less than 1. Remaining rows supply view
examples. Source positions are recorded in `example_pool_indices_`,
`calibration_indices_`, and `pruning_indices_`; explicit query sets have `None`
for their indices.

Pruning selects/prunes a frozen grown tree;
it does not fit thresholds or weights. These rows must not overlap the full
feature rows in examples or calibration. Automatic splits keep identical
feature rows together and approximate class proportions with seeded group
sampling, so actual sizes may differ from the requested fractions. Explicit
sets must also keep duplicate feature groups separate across roles. Keep final test rows separate from all three roles. Pruning can need
uncached outputs, shares the training call limit, and removes branches when
the budget or row support is insufficient. Inspect `pruning_history_` and the
growth/pruning losses, class counts, calls, and node counts in `training_stats_`.
Without either `pruning_set` or `pruning_fraction`, there is no held-out pruning check.

Calibration labels train the routing decisions, stopping rules, and probability
weights. They are not an independent holdout or just probability calibration.
Keep final test data separate. Use explicit sets for time-ordered splits or custom groups such as customer IDs;
automatic grouping recognizes identical feature rows, not entity identity.
The API has no `eval_set`, `example_set`, or `n_estimators` argument.

Every generated view includes every class, increasing its row count if necessary.
Views retain their sampled examples at inference. `subsample` and
`colsample_bytree` control provider context. `max_depth` counts acquisitions,
not probability comparisons. Use `min_samples_leaf` and `gamma` to limit
overfitting to calibration data. `classes_` defines probability-column order.

## Query column coverage

`query_columns="view"` (default) matches the query columns to the labeled
examples. `"all"` sends every query column while examples retain the
`colsample_bytree` subset. `"mixed"` independently samples either mode with
50% probability per recipe, seeded by `random_state`; a small pool need not
contain both modes. This does not increase `n_views`. Never randomize the mode
per request or change it only at final evaluation: the same fixed recipe applies
to growth, pruning and inference. This option does not remove labeled examples.

Adapters must read `view.effective_query_columns` for `view.query` and
`view.columns` for example values; their widths may differ. Cache keys and token
estimates include the actual query contents and schema. Traces and exported
views expose `query_mode` and `query_columns`. JSON saves preserve the choice.
Use descriptive column names and categorical values, especially for query
features absent from the examples. Extra columns may help pretrained reasoning,
but do not teach their dataset-specific effect without examples.

## Use Jev

```python
import os
from xjevboost.jev import JevProvider

provider = JevProvider(
    model=model_version,
    context_limit=model_token_allowance,
    api_key=os.environ["TYPESAFE_API_KEY"],
)
clf = XJevBoostClassifier(
    provider=provider,
    task_instructions=task_instructions,
    class_descriptions=descriptions_by_original_label,
    calibration_fraction=0.2,
    pruning_fraction=0.2,
    max_training_calls=200,
    cache="predictions.sqlite",
    random_state=42,
)
```

`api_key` may be omitted to read `TYPESAFE_API_KEY` automatically. Never embed
credentials in committed examples or notebook outputs. Use a real pinned model
version and its context allowance; the adapter rejects mutable `latest` aliases.
Require substantive instructions and a description for every original class
label. Infer synthetic class membership from labeled examples rather than
inventing semantic class definitions. Offline experiments use `FakeProvider`.
Choose live API execution only when it is within the user's requested scope.

The default token ceiling is a locally estimated hypothetical full request; no
full-view API call is made. `max_total_tokens` can lower that ceiling. Jev's
built-in estimator is approximate. `strict_tokens=True` needs an adapter
estimator with a reliable input/output upper bound. `max_training_calls` limits
fit calls only; subsequent test predictions may incur further calls.

## Establish zero-shot and full-view baselines

Before tuning views, it is useful to measure Jev zero-shot: send the task,
class definitions, column names, and one query row, but no labeled example rows.
Evaluate the same metrics intended for xjevboost. Use a representative tuning
split rather than final test labels when the result will guide hyperparameters.

Treat this as evidence about how much the task depends on examples. If zero-shot
already performs well, favor views with more columns and fewer example rows by
trying a higher `colsample_bytree` and lower `subsample`. If it performs poorly,
retain more labeled examples or compare other row/column balances. This is a
starting hypothesis, not a rule; confirm it on data kept out of fitting, and
count zero-shot provider calls and tokens when reporting benchmark cost.

When it is reasonably likely to fit and the cost is acceptable, also consider
a full-view baseline: send all columns and the complete labeled training/example
pool with each held-out query row. Estimate the request size before calling Jev
and start with a small representative query sample. A full view can be very
expensive, and Jev may reject it when the request exceeds its context or input
limits; skip it when the estimate is near or over the model allowance, and do
not retry an oversized request unchanged. Report failures and cost alongside
quality. Because xjevboost sees only partial views of this same information at
evaluation time, full-view quality is likely an upper-bound reference for what
the method can achieve. It is not a strict mathematical bound: partial views
can occasionally do better by excluding distracting examples or columns. Treat
the full-view benchmark as optional, not a required xjevboost step.

## Reuse and inspect results

The cache hashes provider/model namespace, task, complete view contents, and
projected query values, excluding query IDs and true labels. Two rows identical
on all columns sent in that request share an API response but retain separate training labels.
Use a shared `PredictionCache` or a SQLite path to reuse outputs across fits;
keep seeds, task, and contents stable. Do not coordinate concurrent fitting
processes through this serial cache as though it guarantees exactly-once calls.

```python
probabilities = clf.predict_proba(X_test)
traces = clf.predict_with_trace(X_test)
metrics = clf.evaluate(X_test, y_test)
single = clf.evaluate(X_test, y_test, policy="single_view")
```

`training_stats_` reports fit-call usage and fitting loss. Traces include view
provenance, branches, estimates, actual usage, and budget stops. `calls` counts
logical acquisitions; `provider_calls` counts uncached requests. `actual_tokens`
replays reported usage for comparable policy accounting; `new_actual_tokens`
counts only new requests. Unknown usage is `None`, not zero.

`evaluate` returns `precision_per_class`, `recall_per_class`, `f1_per_class`,
and `support_per_class` in `classes_` order. Undefined precision/recall/F1 are
reported as zero. These use argmax predictions; do not assume a positive class
without checking its label and index.

Brier loss averages squared probability error across classes. The single-view
baseline uses the selected root. `policy="fixed_sequence"` requires explicit
`view_ids` and uses uniform averaging. Choose settings and baselines without
looking at final test labels. Compare policies on the same test rows. The goal
is comparable full-context quality with fewer tokens; distinguish savings versus
all candidate views from savings versus one full-context request. Report fitting
cost separately. A full-context benchmark is optional and is not required to
estimate the inference ceiling. Include a class-prior baseline and inspect
per-class recall when accuracy is dominated by class imbalance. The fake provider validates mechanics, not Jev
quality or large-dataset speedups.

## Repository entry points

When working in the source checkout:

- `xjevboost/providers.py`: `Provider`, `Task`, `TokenEstimate`, and `Prediction`.
- `xjevboost/core.py`: provider-neutral adaptive policy, independent of sklearn.
- `xjevboost/jev.py`: HTTP adapter; no hosted SDK dependency.
- `xjevboost/splitting.py`: seeded stratified splitting with duplicate groups.
- `tests/`: offline provider, policy, budgeting, plotting, and split checks.

Install with `pip install -e .`, add `.[notebooks]` for Jupyter or `.[test]`
for tests, and use ordinary terminal commands in shared documentation.
`sqlite3` belongs to Python's standard library. Verbose output uses stderr,
not tqdm. A replacement provider implements `namespace`, `context_limit`,
`estimate(view, task)`, and `predict(view, task)`; namespace must identify the
model, formatting, and prediction configuration that affect cached results.

## Persistence and path costs

Use `clf.save_model(path)` and
`XJevBoostClassifier.load_model(path, provider=provider, cache=None)` for versioned
JSON inference artifacts. Supply the same provider namespace (model/configuration);
only its fingerprint is saved. Provider objects, credentials, response caches and
calibration/pruning query records are excluded. Retained labeled examples, task
instructions, view recipes and budget settings are included. Do not put secrets
inside task instructions or example cells. Loading makes no provider requests.
This is not a resumable training checkpoint; refitting requires supplying training
configuration and data again.

`training_stats_` averages describe search expenditure: new calls/tokens per training
query and per new provider call, and evaluated views per training query. They include
screening work for discarded candidates. `training_leaf_stats_` instead reports the
finished policy's cumulative path cost on growth rows, read from the training matrix
without new calls. `export_tree()` exposes `training_path_stats` at every node.

After inference, `inference_stats_` (or `evaluate`'s return value) includes
`average_actual_tokens_per_query`, `average_actual_tokens_per_view`, their accounted
counterparts, and `leaf_stats`. Group by `stop_node` plus `stop_reason`: a budget stop
at an acquisition node is not a normal leaf. Costs include every call up to that stop.
Unknown actual usage is `None`, never an estimate disguised as actual. Cached usage
counts toward policy cost; new usage counts only uncached requests. Node locations
match the exported tree, including after JSON loading.
