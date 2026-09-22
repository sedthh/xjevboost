"""Generate a small, balanced robot obstacle-course classification dataset.

Run from the repository root:
    python examples/robot_course_data.py --output .jupyter-local/robot-course

Returns descriptive pandas columns, ordered categories and real booleans.
The simulator uses correlated physical measurements and fixed interactions,
not independent random columns. Labels depend only on the displayed features
plus small outcome noise. This is fictional data, not a robotics model.
Do not include this simulator's scoring rule in Jev's task instructions.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _candidates(n, rng, noise):
    chassis = rng.choice(['wheeled', 'tracked', 'legged'], n)
    surface = rng.choice(['concrete', 'gravel', 'mud'], n)
    condition = rng.choice(['poor', 'fair', 'good'], n, p=[.25, .45, .30])
    difficulty = rng.choice(['easy', 'moderate', 'hard'], n)
    health = pd.Series(condition).map({'poor': -1, 'fair': 0, 'good': 1}).to_numpy()
    power = rng.normal(size=n)
    battery = np.round(np.clip(65 + 12 * power + 9 * health + rng.normal(0, 9, n), 10, 100), 1)
    torque = np.round(np.clip(24 + 6 * power + 8 * (chassis == 'tracked') + rng.normal(0, 3, n), 5, 55), 1)
    payload = np.round(np.clip(8 + 2 * power + 4 * (chassis == 'tracked') + rng.normal(0, 3, n), 1, 22), 1)
    error = np.round(np.clip(3 - .7 * health + rng.normal(0, .8, n), .2, 6), 1)
    traction = rng.random(n) < np.where(chassis == 'tracked', .8, .45)
    waterproof = rng.random(n) < .5
    X = pd.DataFrame({
        'chassis_type': pd.Categorical(chassis, categories=['wheeled', 'tracked', 'legged']),
        'course_surface': pd.Categorical(surface, categories=['concrete', 'gravel', 'mud']),
        'maintenance_condition': pd.Categorical(condition, categories=['poor', 'fair', 'good'], ordered=True),
        'obstacle_difficulty': pd.Categorical(difficulty, categories=['easy', 'moderate', 'hard'], ordered=True),
        'battery_charge_percent': battery,
        'motor_torque_nm': torque,
        'payload_weight_kg': payload,
        'sensor_error_cm': error,
        'traction_control_enabled': traction,
        'waterproof_casing': waterproof,
    })
    # Fixed across seeds and splits; all terms use the rounded, visible values.
    difficulty_cost = pd.Series(difficulty).map({'easy': -.9, 'moderate': 0, 'hard': .9}).to_numpy()
    score = ((battery - 65) / 25 + (torque - 1.8 * payload - 8) / 12
             - (error - 2.5) / 2 - difficulty_cost + .25 * health)
    score += np.where(surface == 'mud',
                      -.9 + .9 * waterproof + .8 * (chassis == 'tracked'), 0)
    score += np.where(surface == 'gravel', -.6 + .9 * traction + .4 * (chassis == 'legged'), 0)
    score += .4 * ((surface == 'concrete') & (chassis == 'wheeled'))
    score += rng.normal(0, noise, n)
    y = pd.Series(np.where(score >= 0, 'finished', 'failed'), name='course_outcome')
    return X, y


def make_robot_course(n_samples=300, random_state=42, noise=.2):
    """Return (X, y) with exactly 50/50 outcomes and no duplicate feature rows.

    n_samples must be positive and even. noise is the standard deviation of
    additive outcome noise. Rejection sampling balances the fixed-rule outcomes;
    it does not change the rule or choose a threshold using test rows.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 2 or n_samples % 2:
        raise ValueError('n_samples must be a positive even integer of at least 2')
    if not np.isfinite(noise) or noise < 0:
        raise ValueError('noise must be finite and nonnegative')
    rng = np.random.default_rng(random_state)
    pieces = []
    counts = {'finished': 0, 'failed': 0}
    seen = set()
    while min(counts.values()) < n_samples // 2:
        X, y = _candidates(max(n_samples, 128), rng, noise)
        keep = []
        for i, row in enumerate(X.itertuples(index=False, name=None)):
            label = y.iloc[i]
            if row not in seen and counts[label] < n_samples // 2:
                seen.add(row)
                counts[label] += 1
                keep.append(i)
        pieces.append(X.iloc[keep].assign(course_outcome=y.iloc[keep]))
    data = pd.concat(pieces, ignore_index=True)
    data = data.iloc[rng.permutation(len(data))].reset_index(drop=True)
    return data.drop(columns='course_outcome'), data.course_outcome


def split_robot_course(X, y, random_state=42):
    """Default 300 rows -> 200 train, 20 sanity, 80 untouched test rows.

    Stratified at both steps. Other sizes retain approximately these proportions.
    In XJevBoostClassifier use calibration_fraction=.5, pruning_fraction=.3:
    the default train split then supplies 40 example, 100 growth and 60 pruning rows.
    """
    train, rest = train_test_split(np.arange(len(y)), test_size=1/3, stratify=y, random_state=random_state)
    sanity, test = train_test_split(rest, test_size=.8, stratify=y.iloc[rest], random_state=random_state)
    return {name: (X.iloc[idx].copy(), y.iloc[idx].copy())
            for name, idx in [('train', train), ('sanity', sanity), ('test', test)]}


def check_baselines(X_train, y_train):
    """Training-only five-fold screening; no Jev calls and no final test access."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    categorical = X_train.select_dtypes(include='category').columns.tolist()
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {}
    for name, model in [
        ('majority', DummyClassifier(strategy='prior')),
        ('random_forest', RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ('extra_trees', ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ]:
        pipeline = make_pipeline(ColumnTransformer([
            ('categories', OneHotEncoder(handle_unknown='ignore'), categorical)
        ], remainder='passthrough'), model)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                                scoring=['accuracy', 'balanced_accuracy'])
        results[name] = {metric: float(scores['test_' + metric].mean())
                         for metric in ['accuracy', 'balanced_accuracy']}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise', type=float, default=.2)
    parser.add_argument('--output', type=Path, default=Path('.jupyter-local/robot-course'))
    args = parser.parse_args()
    X, y = make_robot_course(args.n_samples, args.seed, args.noise)
    splits = split_robot_course(X, y, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (features, labels) in splits.items():
        features.assign(course_outcome=labels).to_csv(args.output / (name + '.csv'), index=False)
    report = {
        'seed': args.seed, 'noise': args.noise,
        'split_counts': {name: labels.value_counts().to_dict() for name, (_, labels) in splits.items()},
        'ordinal_levels': {col: X[col].cat.categories.tolist() for col in X.select_dtypes('category') if X[col].cat.ordered},
        'training_only_cv': check_baselines(*splits['train']),
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('CSV files saved to', args.output.resolve())
    print('No Jev calls made. Estimate serialized requests before using the 40-example pool.')
