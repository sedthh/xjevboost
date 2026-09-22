from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from xjevboost import FakeProvider, XJevBoostClassifier, PredictionCache, Task
from xjevboost.views import ViewBuilder
from xjevboost.jev import JevProvider


def builder():
    return ViewBuilder(((1, 10, 'mud'), (2, 20, 'gravel')), np.array([0, 1]), ('a', 'b', 'surface'))


def test_asymmetric_payload_cache_and_fake_alignment():
    b = builder()
    r = b.recipes(1, 1, .34, 42)[0]
    r = replace(r, columns=('b',), column_indices=(1,))
    v = b.build(r, (1, 10, 'mud'))
    full = b.build(replace(r, query_mode='all'), (1, 10, 'mud'))
    changed = b.build(replace(r, query_mode='all'), (1, 10, 'snow'))
    assert len(full.examples[0][0]) == 1 and len(full.query) == 3
    assert full.payload()['query']['columns'] == b.columns
    assert 'columns' not in v.payload()['query']
    fake = FakeProvider()
    task = Task((0, 1), 'Predict outcome.', ('Failed.', 'Finished.'))
    assert fake.predict(v, task).probabilities == fake.predict(full, task).probabilities
    cache = PredictionCache()
    assert len({cache.key(fake, task, x) for x in (v, full, changed)}) == 3
    assert fake.estimate(full, task).total > fake.estimate(v, task).total
    jev = JevProvider(model='jev-1.13.0', context_limit=32000)
    payload = jev.payload(full, task)['state']
    assert payload['query']['features'] == dict(zip(b.columns, full.query))
    assert payload['examples'][0]['features'] == {'b': 10}


def test_mixed_is_seeded_and_preserves_example_sampling():
    b = builder()
    first = b.recipes(30, .5, .34, 42, 'mixed')
    assert first == b.recipes(30, .5, .34, 42, 'mixed')
    assert {r.query_mode for r in first} == {'view', 'all'}
    for mode in ['view', 'all']:
        r = b.recipes(1, .5, .34, 42, mode)[0]
        assert r.example_ids == first[0].example_ids
        assert r.columns == first[0].columns
    with pytest.raises(ValueError, match='query_columns'):
        b.recipes(1, .5, .5, 42, 'invalid')


@pytest.mark.parametrize('mode', ['view', 'all', 'mixed'])
def test_fit_trace_persistence_and_budget(mode, tmp_path):
    X = pd.DataFrame({'a': np.arange(60), 'b': np.arange(60)*2, 'surface': ['mud', 'gravel']*30})
    y = np.arange(60) % 2
    clf = XJevBoostClassifier(provider=FakeProvider(), query_columns=mode, n_views=6,
        colsample_bytree=.34, calibration_fraction=.4, pruning_fraction=.2,
        min_samples_leaf=2, random_state=42, max_depth=2)
    assert clone(clf).query_columns == mode
    clf.fit(X, y)
    traces = clf.predict_with_trace(X.iloc[:5])
    for t in traces:
        assert t['accounted_tokens'] <= t['token_ceiling']
        for step in t['steps']:
            expected = list(X.columns) if step['query_mode'] == 'all' else step['columns']
            assert step['query_columns'] == expected
    path = tmp_path/'model.json'
    clf.save_model(path)
    loaded = XJevBoostClassifier.load_model(path, provider=FakeProvider())
    assert loaded.query_columns == mode
    assert loaded.views_ == clf.views_
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[:5]), clf.predict_proba(X.iloc[:5]))
    data = json.loads(path.read_text())
    data['recipes'][0]['query_mode'] = 'invalid'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='recipe'):
        XJevBoostClassifier.load_model(path, provider=FakeProvider())
    if mode == 'view':
        data.pop('query_columns')
        for r in data['recipes']:
            r.pop('query_mode')
        path.write_text(json.dumps(data))
        legacy = XJevBoostClassifier.load_model(path, provider=FakeProvider())
        assert legacy.query_columns == 'view'
        np.testing.assert_allclose(legacy.predict_proba(X.iloc[:5]), clf.predict_proba(X.iloc[:5]))
