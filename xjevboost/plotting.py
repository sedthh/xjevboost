"""Inspect a fitted policy without evaluating any provider requests."""
import math
import textwrap

import numpy as np

from .core import output_features


def export_tree(classifier):
    """Return JSON-safe tree structure and complete contents of used views.

    sample_count refers to calibration observations, not held-out performance.
    This export contains labeled example data; share it accordingly.
    """
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(classifier, "policy_")
    if hasattr(classifier, "_saved_inspection"):
        import copy
        return copy.deepcopy(classifier._saved_inspection)
    policy = classifier.policy_
    used = set()
    next_id = 0

    def visit(node, indices, location="root"):
        nonlocal next_id
        node_id = f"n{next_id}"
        next_id += 1
        path = [policy.recipes[v].id for v in node.path]
        actual, accounted = [], []
        for i in indices:
            predictions = [policy._matrix[(int(i), v)] for v in node.path]
            actual.append(sum(p.actual_tokens for p in predictions)
                          if all(p.actual_tokens is not None for p in predictions) else None)
            accounted.append(sum(policy._charge(p, policy.estimate(policy.queries[i], v))
                                 for v, p in zip(node.path, predictions)))
        known = bool(len(indices)) and all(v is not None for v in actual)
        cost = {"source": "growth_rows_final_policy", "n_queries": len(indices),
                "average_calls": len(node.path),
                "average_actual_tokens_per_query": float(np.mean(actual)) if known else None,
                "average_actual_tokens_per_view": float(np.mean(actual)) / len(node.path) if known and node.path else None,
                "average_accounted_tokens_per_query": float(np.mean(accounted)) if len(indices) else None}
        result = {"id": node_id, "location": location, "training_path_stats": cost,
                  "kind": node.kind, "sample_count": len(indices),
                  "acquisitions": len(node.path), "path": path,
                  "weights": dict(zip(path, node.weights.tolist()))}
        if node.kind == "acquire":
            used.add(node.view)
            result["view_id"] = policy.recipes[node.view].id
            result["child"] = visit(node.child, indices, location + ".child")
        elif node.kind == "route":
            names = ([f"{policy.recipes[v].id}.p[{label!r}]" for v in node.path for label in policy.task.classes]
                     + [f"{policy.recipes[v].id}.predicted_class_index" for v in node.path]
                     + [f"{policy.recipes[v].id}.top_two_gap" for v in node.path])
            result.update(feature=node.feature, feature_name=names[node.feature],
                          lower=None if math.isinf(node.lower) else node.lower,
                          upper=None if math.isinf(node.upper) else node.upper)
            # Read only the existing training matrix. Inspection never fills it.
            if len(indices):
                outputs = np.asarray([[policy._matrix[(int(i), v)].probabilities for v in node.path]
                                      for i in indices])
                values = output_features(outputs)[:, node.feature]
                inside = (values > node.lower) & (values <= node.upper)
            else:
                inside = np.zeros(0, dtype=bool)
            result["inside"] = visit(node.inside, indices[inside], location + ".inside")
            result["outside"] = visit(node.outside, indices[~inside], location + ".outside")
        return result

    root = visit(policy.tree, policy.growth_indices)
    views = {}
    for v in sorted(used):
        recipe = policy.recipes[v]
        view = policy.builder.build(recipe, (None,) * len(policy.builder.columns))
        views[recipe.id] = {
            "columns": list(recipe.columns), "example_ids": list(recipe.example_ids),
            "query_mode": recipe.query_mode, "query_columns": list(view.effective_query_columns),
            "seed": recipe.seed, "kind": recipe.kind, "format_version": recipe.format_version,
            "examples": [{"id": row_id, "values": list(values), "class": policy.task.classes[label]}
                         for row_id, (values, label) in zip(recipe.example_ids, view.examples)],
        }
    return {"classes": list(policy.task.classes), "root": root, "views": views}


def plot_tree(classifier, *, ax=None, fontsize=9):
    """Plot acquisitions (blue), routing (amber), and stopping (green).

    Returns a Matplotlib Axes. Save with ax.figure.savefig(...).
    Full example values are available separately through export_tree().
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError('Tree plotting requires matplotlib: pip install "xjevboost[plot]"') from exc
    data = export_tree(classifier)
    positions, edges, nodes = {}, [], []
    leaves = 0

    def layout(node, depth=0):
        nonlocal leaves
        children = ([("result", node["child"])] if node["kind"] == "acquire" else
                    [("inside", node["inside"]), ("outside", node["outside"])] if node["kind"] == "route" else [])
        children_x = []
        for label, child in children:
            children_x.append(layout(child, depth + 1))
            edges.append((node["id"], child["id"], label))
        if children_x:
            x = sum(children_x) / len(children_x)
        else:
            x = leaves
            leaves += 1
        positions[node["id"]] = (x, -depth)
        nodes.append(node)
        return x

    layout(data["root"])
    depth = max(-p[1] for p in positions.values())
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, leaves * 4.5), max(4, (depth + 1) * 2.4)))
    colors = {"acquire": "#dbeafe", "route": "#fef3c7", "stop": "#dcfce7"}
    for parent, child, label in edges:
        start, end = positions[parent], positions[child]
        ax.annotate("", xy=(end[0], end[1] + .25), xytext=(start[0], start[1] - .25),
                    arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.3}, zorder=1)
        if label != "result":
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, label,
                    ha="center", va="center", fontsize=fontsize,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=2)
    for node in nodes:
        mix = ", ".join(f"{v}: {w:.3f}" for v, w in node["weights"].items())
        if node["kind"] == "acquire":
            view = data["views"][node["view_id"]]
            columns = ", ".join(view["columns"])
            columns = textwrap.shorten(columns, width=45, placeholder=" ...")
            ids = ", ".join(map(str, view["example_ids"][:5]))
            if len(view["example_ids"]) > 5:
                ids += ", ..."
            lines = [f"{node['id']} · Acquire {node['view_id']} · call {node['acquisitions'] + 1}",
                     f"Example columns: {columns}", f"Query columns: {len(view.get('query_columns', view['columns']))}", f"{len(view['example_ids'])} examples: {ids}"]
            if mix:
                lines.append("Budget stop: " + textwrap.shorten(mix, width=45, placeholder=" ..."))
        elif node["kind"] == "route":
            feature = node["feature_name"]
            lo, hi = node["lower"], node["upper"]
            condition = (f"{feature} <= {hi:.4g}" if lo is None else
                         f"{feature} > {lo:.4g}" if hi is None else
                         f"{lo:.4g} < {feature} <= {hi:.4g}")
            lines = [f"{node['id']} · Route · no call", *textwrap.wrap(condition, width=45)]
        else:
            lines = [f"{node['id']} · Stop · {node['acquisitions']} acquired",
                     "Probability mixture:", *textwrap.wrap(mix, width=45)]
        lines.append(f"Calibration rows: {node['sample_count']}")
        cost = node["training_path_stats"]
        mean_tokens = cost["average_actual_tokens_per_query"]
        if mean_tokens is not None:
            lines.append(f"Mean path tokens (growth): {mean_tokens:.0f}")
        ax.text(*positions[node["id"]], "\n".join(lines), ha="center", va="center",
                fontsize=fontsize, bbox={"boxstyle": "round,pad=0.6", "facecolor": colors[node["kind"]],
                                          "edgecolor": "#94a3b8"}, zorder=3)
    ax.set_xlim(-.6, max(.6, leaves - .4))
    ax.set_ylim(-depth - .6, .7)
    ax.set_axis_off()
    ax.set_title("Adaptive view tree", pad=30)
    ax.legend(handles=[Patch(facecolor=color, label=kind.title()) for kind, color in colors.items()],
              loc="upper center", bbox_to_anchor=(.5, 1.03), ncol=3, frameon=False)
    return ax
