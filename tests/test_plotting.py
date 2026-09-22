import json

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from xjevboost import XJevBoostClassifier
from test_xjevboost import branch_data, scripted_policy


def fitted_classifier():
    rows, labels = branch_data()
    policy = scripted_policy(rows, labels)
    clf = XJevBoostClassifier(provider=policy.provider)
    clf.policy_ = policy
    return clf


def test_export_contains_exact_views_and_routing_counts_without_calls():
    clf = fitted_classifier()
    before = len(clf.provider.calls)
    data = clf.export_tree()
    json.dumps(data, allow_nan=False)
    assert len(clf.provider.calls) == before
    assert set(data["views"]) == {"v0", "v1", "v2"}
    assert data["root"]["sample_count"] == 100
    assert data["views"]["v1"]["columns"] == ["left"]
    assert data["views"]["v1"]["examples"] == [
        {"id": 0, "values": [0.], "class": 0},
        {"id": 1, "values": [1.], "class": 1},
    ]

    def check(node):
        if node["kind"] == "route":
            assert node["sample_count"] == node["inside"]["sample_count"] + node["outside"]["sample_count"]
            assert node["feature_name"].startswith("v0.")
            check(node["inside"])
            check(node["outside"])
        elif node["kind"] == "acquire":
            assert node["sample_count"] == node["child"]["sample_count"]
            check(node["child"])
        else:
            assert np.isclose(sum(node["weights"].values()), 1)
    check(data["root"])


def test_plot_renders_all_node_types_and_saves(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clf = fitted_classifier()
    before = len(clf.provider.calls)
    ax = clf.plot_tree()
    text = "\n".join(t.get_text() for t in ax.texts)
    for phrase in ("Acquire v0", "Acquire v1", "Acquire v2", "Route", "Stop", "Example columns:", "Query columns:", "Probability mixture:"):
        assert phrase in text
    output = tmp_path / "tree.svg"
    ax.figure.savefig(output, bbox_inches="tight")
    assert output.stat().st_size > 0
    assert len(clf.provider.calls) == before
    plt.close(ax.figure)


def test_export_requires_fitted_classifier():
    with pytest.raises(NotFittedError):
        XJevBoostClassifier().export_tree()
