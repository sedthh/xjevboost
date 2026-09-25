# xjevboost

Classify tabular data with Jev, Laya, or compatible OpenJev servers without putting your entire dataset of labeled examples into their context.

xjevboost learns to use a series of smaller slices of your table, asking for more only when it helps and stopping when it has enough.

It should work particularly well when the table contains categorical data or unstructured text, or when its contents need cleaning.

The goal: get close to full-context prediction quality while spending fewer tokens. Sometimes one small view is all you need.

## How it works

We don't retrain the model. We use the Jev, Laya, or compatible model you provide and learn which context to send it.

A **view** contains selected labeled examples and columns, plus the row you want to classify. A greedy tree search learns which views help and when to stop asking for more.

At prediction time, each answer guides the next step. Different rows can take different paths, within your call and token budgets.

Despite the name, there's no gradient boosting going on here yet, but it's likely coming in a future version.

## Try it

From a local checkout:

```sh
pip install -e .
export TYPESAFE_API_KEY="your-api-key"
```

Using your training and test tables, with labels `0` (stayed) and `1` (cancelled):

```python
from xjevboost import XJevBoostClassifier
from xjevboost.jev import JevProvider

clf = XJevBoostClassifier(
    provider=JevProvider(model="jev-1.13.0", context_limit=32_000),
    task_instructions=(
        "Predict whether a customer will cancel next month from their account "
        "and usage history. Use the labeled examples as reference."
    ),
    class_descriptions={0: "Stayed subscribed next month.", 1: "Cancelled next month."},
    n_views=4,
    max_depth=2,
    calibration_fraction=0.3,
    pruning_fraction=0.2,
    max_training_calls=200,
    random_state=42,
)
clf.fit(X_train, y_train, verbose=True)
print(clf.evaluate(X_test, y_test))
```

Use a model version and context allowance available to your account. You can also pass `api_key="..."` directly to `JevProvider`.

### Other decision models

The same classifier works with Jev-compatible HTTP servers, including
[Laya](https://github.com/NandhaKishorM/laya) and
[OpenJev](https://github.com/razorback16/openjev). Swap the provider:

```python
from xjevboost import LayaProvider, OpenJevProvider, SystemOneProvider

# Start Laya separately with: pip install "laya[serve]" && laya-serve
provider = LayaProvider(
    model="english",
    context_limit=512,  # Set this to your deployed checkpoint's allowance.
    revision="my-laya-deployment-v1",
)

# An existing OpenJev server, with an explicit model and endpoint.
provider = OpenJevProvider(
    endpoint="http://127.0.0.1:8080/v1/systemone",
    model="openjev-0.1",
    context_limit=32_000,  # Check your server's configured allowance.
    revision="my-openjev-deployment-v1",
)
```

Pass your chosen `provider` to `XJevBoostClassifier` as above. No extra dependencies
are needed in xjevboost; install and run the model server separately. Laya reads
`LAYA_API_KEY` and OpenJev reads `OPENJEV_API_KEY` if set, or accepts `api_key=`.
Neither borrows your TypeSafe key. Unauthenticated local servers work too.

`revision` identifies the server version, weights and inference configuration for
caching; it does not pin weights on the server. Change it when the deployment
changes, and refit the policy for a different model. Keep views small for encoder
models with short contexts; a server may truncate oversized inputs. Token
estimates remain approximate unless you supply a tokenizer-based bound.

For other `/v1/systemone` Choice servers, use `SystemOneProvider` with the same
arguments and optional `api_key_env`, `response_model` (the canonical response ID),
and `max_classes`. For example, OpenJev's Verdict backend needs its own model ID,
context allowance and 24-class limit. This supports the HTTP contract, not every
project named OpenJev or arbitrary chat/completions endpoints. Laya specifically
targets `laya-serve` and validates its selected router checkpoint.

Here, 30% of training rows grow the tree, 20% check which branches to keep, and the rest provide examples for the views. Splits preserve class balance as closely as possible and keep identical feature rows together. The test set stays separate.

Already have your own splits? Pass `calibration_set=(X_cal, y_cal)` and `pruning_set=(X_prune, y_prune)` to `fit` instead of the corresponding fractions. Pruning is optional; fractions must sum to less than 1.

## Hyperparameters

Start with these:

| Parameter | Default | What it controls |
|---|---|---|
| `n_views` | `16` | Number of candidate views, including balanced and broader baselines |
| `subsample` | `0.5` | Fraction of example rows per view; every class is represented |
| `colsample_bytree` | `0.5` | Fraction of example columns per view |
| `query_columns` | `"view"` | Query columns: match examples (`"view"`), keep all (`"all"`), or sample either per recipe (`"mixed"`) |
| `max_depth` | `3` | Maximum calls per prediction |
| `calibration_fraction` | `None` | Fraction of training rows reserved for growing the tree |
| `pruning_fraction` | `None` | Fraction reserved for checking and pruning branches |
| `max_training_calls` | `None` | Cap on new provider calls during fitting |
| `max_total_tokens` | `None` | Optional tighter total-token ceiling per prediction |
| `random_state` | `None` | Seed for repeatable splits, views, and search |

Tree search and budgets:

| Parameter | Default | What it controls |
|---|---|---|
| `gamma` | `0.001` | Minimum Brier-loss improvement needed to keep a continuation |
| `min_samples_leaf` | `10` | Minimum training-query support for a leaf |
| `max_nodes` | `31` | Maximum tree size, including routing and stop nodes |
| `screening_samples` | `16` | Initial query sample size for comparing candidate views |
| `leaf_l2` | `0.01` | Regularization of the probability weights at stopping leaves |
| `optimization` | `"calls"` | Rank screening candidates by improvement per call or per token (`"tokens"`) |
| `full_view_budget` | `True` | Use a hypothetical full-context request as the inference token ceiling |
| `token_margin` | `1.2` | Safety multiplier for approximate token estimates |
| `strict_tokens` | `False` | Require reliable pre-call token bounds instead of approximate estimates |

Provider and other constructor settings:

| Parameter | Default | What it controls |
|---|---|---|
| `provider` | `None` | Configured `JevProvider`, `LayaProvider`, `OpenJevProvider`, or custom adapter |
| `task_instructions` | `""` | What to predict; meaningful instructions are required for Jev |
| `class_descriptions` | `None` | Mapping from each class label to its meaning; required for Jev |
| `cache` | `None` | In-memory cache by default; pass a SQLite path or `PredictionCache` to reuse requests |

Sampling fractions are targets; the broader baseline view uses more context. Jev's token estimates are approximate, so the default ceiling isn't a hard guarantee on actual usage. `strict_tokens=True` requires an adapter estimator that supplies reliable bounds.

## See what happened

- `training_stats_`: training calls, tokens, cache savings, and pruning results.
- `evaluate(X_test, y_test)`: accuracy, probability losses, per-class precision/recall/F1, and inference costs.
- `predict_with_trace(X)`: which views were called and why the tree stopped.
- `plot_tree()`: draw the learned tree (install `.[plot]`).

Requests can be cached in SQLite with `cache="predictions.sqlite"`. Matching requests are reused across runs. Other model providers can plug in through an adapter.

## Save and inspect

```python
clf.save_model("model.json")
loaded = XJevBoostClassifier.load_model("model.json", provider=provider)
print(clf.training_leaf_stats_)
print(clf.inference_stats_)  # after prediction or evaluation
```

JSON stores the tree, retained labeled examples, task and budgets. It excludes
providers, API keys and response caches. Supply the same provider/model when loading;
this is an inference artifact, not a training checkpoint.

Inference summaries include average tokens per query/view and `leaf_stats` with
cumulative path costs. `training_leaf_stats_` describes the finished policy on
growth rows; `training_stats_` separately reports the cost of searching for it.
Unknown actual token usage stays `None`; accounted tokens include estimates.
