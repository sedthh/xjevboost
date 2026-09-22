"""Seeded stratified splits that keep identical feature rows together."""
import numpy as np
from sklearn.model_selection import train_test_split
from .views import canonical


def stratified_split(rows, labels, indices, count, random_state=None):
    """Return remaining/selected source indices; count is a target with groups.

    For duplicate groups, minimize class-count deviation over seeded candidate
    partitions. Exact fractions may be impossible without splitting a group.
    """
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(labels)
    count = int(count)
    if not 0 < count < len(indices):
        raise ValueError("Split sizes must leave rows for both partitions")
    groups = {}
    for i in indices:
        groups.setdefault(canonical(rows[i]), []).append(int(i))
    if len(groups) == len(indices):
        a, b = train_test_split(indices, test_size=count, stratify=labels[indices],
                                random_state=random_state)
        return np.sort(a), np.sort(b)
    classes, encoded = np.unique(labels, return_inverse=True)
    members = list(groups.values())
    counts = np.array([np.bincount(encoded[g], minlength=len(classes)) for g in members])
    total = counts.sum(axis=0)
    target = total * count / len(indices)
    rng = np.random.default_rng(random_state)
    best = None
    # Search uses labels only to balance partitions, never prediction quality.
    for _ in range(256):
        order = rng.permutation(len(members))
        cumulative = np.cumsum(counts[order], axis=0)[:-1]
        valid = np.all((cumulative > 0) & (cumulative < total), axis=1)
        scores = np.sum(((cumulative - target) / np.maximum(total, 1)) ** 2, axis=1)
        scores[~valid] = np.inf
        if not len(scores) or not np.isfinite(scores).any():
            continue
        cut = int(np.argmin(scores)) + 1
        score = float(scores[cut - 1])
        if best is None or score < best[0]:
            best = score, order[:cut]
    if best is None:
        raise ValueError("Cannot stratify duplicate groups with every class on both sides; use more independent groups")
    selected = np.sort(np.concatenate([members[g] for g in best[1]]))
    return np.setdiff1d(indices, selected), selected
