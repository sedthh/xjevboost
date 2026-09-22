"""Stable, provider-neutral values and fixed view recipes."""
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math

import numpy as np


def cell(value):
    """JSON-safe scalar representation; missing values always become null."""
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported cell type: {type(value).__name__}; use scalar values")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def table(X, expected_columns=None):
    named = hasattr(X, "columns") and hasattr(X, "to_numpy")
    if named:
        columns = tuple(X.columns)
        if not all(isinstance(c, str) for c in columns):
            raise ValueError("DataFrame column names must be strings")
        values = X.to_numpy(dtype=object)
    else:
        values = np.asarray(X, dtype=object)
        columns = tuple(f"x{i}" for i in range(values.shape[1])) if values.ndim == 2 else ()
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("X must be a two-dimensional table with at least one column")
    if len(set(columns)) != len(columns):
        raise ValueError("Column names must be unique")
    if expected_columns is not None:
        if values.shape[1] != len(expected_columns) or (named and columns != expected_columns):
            raise ValueError("Columns must match training columns in the same order")
        columns = expected_columns
    rows = tuple(tuple(cell(v) for v in row) for row in values)
    return rows, columns, named


@dataclass(frozen=True)
class ViewRecipe:
    id: str
    columns: tuple[str, ...]
    column_indices: tuple[int, ...]
    example_ids: tuple[int, ...]
    seed: int
    kind: str = "random"
    format_version: str = "records-v1"
    query_mode: str = "view"


@dataclass(frozen=True)
class View:
    columns: tuple[str, ...]
    examples: tuple  # (projected scalar values, class index), preserving multiplicity
    query: tuple
    format_version: str = "records-v1"

    query_columns: tuple[str, ...] | None = None

    @property
    def effective_query_columns(self):
        return self.columns if self.query_columns is None else self.query_columns

    def payload(self):
        # IDs and seeds are provenance, not model input. The true query label
        # cannot be included because this object has no such field.
        result = {"format": self.format_version, "columns": self.columns,
                "examples": [{"values": row, "class": int(label)}
                             for row, label in self.examples],
                "query": {"values": self.query}}
        if self.effective_query_columns != self.columns:
            result["query"]["columns"] = self.effective_query_columns
        return result


class ViewBuilder:
    def __init__(self, rows, labels, columns, row_ids=None):
        self.rows, self.labels, self.columns = rows, labels, columns
        self.row_ids = tuple(range(len(rows))) if row_ids is None else tuple(int(i) for i in row_ids)
        self._row_lookup = {source: position for position, source in enumerate(self.row_ids)}

    def build(self, recipe, query):
        if recipe.query_mode not in {"view", "all"}:
            raise ValueError("Invalid recipe query_mode")
        indices = recipe.column_indices
        return View(recipe.columns,
                    tuple((tuple(self.rows[self._row_lookup[r]][c] for c in indices),
                           int(self.labels[self._row_lookup[r]])) for r in recipe.example_ids),
                    tuple(query) if recipe.query_mode == "all" else tuple(query[c] for c in indices),
                    recipe.format_version, self.columns if recipe.query_mode == "all" else None)

    def full(self, query):
        return View(self.columns, tuple((r, int(y)) for r, y in zip(self.rows, self.labels)), query)

    def recipes(self, n_views, subsample, colsample, random_state, query_columns="view"):
        if query_columns not in {"view", "all", "mixed"}:
            raise ValueError("query_columns must be view, all, or mixed")
        rng = np.random.default_rng(random_state)
        nr = max(1, math.ceil(len(self.rows) * subsample))
        nc = max(1, math.ceil(len(self.columns) * colsample))
        class_rows = [np.flatnonzero(self.labels == k) for k in np.unique(self.labels)]
        result, seen = [], set()
        for i in range(n_views):
            seed = int(rng.integers(0, 2**32))
            local = np.random.default_rng(seed)
            kind = "balanced" if i == 0 else "broader" if i == 1 else "random"
            if kind == "balanced":
                groups = [list(local.permutation(rows)) for rows in class_rows]
                balanced_count = max(nr, len(groups))
                selected = []
                while len(selected) < balanced_count:
                    for group in groups:
                        if group and len(selected) < balanced_count:
                            selected.append(group.pop())
                rids = tuple(sorted(int(r) for r in selected))
            else:
                count = min(len(self.rows), math.ceil(nr * 1.5)) if kind == "broader" else nr
                count = max(count, len(class_rows))
                # Guarantee coverage, then fill randomly without replacement.
                selected = [int(local.choice(rows)) for rows in class_rows]
                remaining = np.setdiff1d(np.arange(len(self.rows)), selected)
                selected.extend(local.choice(remaining, count - len(selected), replace=False))
                rids = tuple(sorted(int(r) for r in selected))
            count = min(len(self.columns), math.ceil(nc * 1.5)) if kind == "broader" else nc
            cids = tuple(sorted(int(c) for c in local.choice(len(self.columns), count, replace=False)))
            recipe = ViewRecipe(f"v{i}", tuple(self.columns[c] for c in cids), cids,
                                tuple(self.row_ids[r] for r in rids), seed, kind,
                                query_mode=(str(local.choice(["view", "all"])) if query_columns == "mixed" else query_columns))
            # Equal contents from different source IDs also collapse. Examples
            # inside a view remain intact: deleting repetitions changes context.
            key = digest(self.build(recipe, (None,) * len(self.columns)).payload())
            if key not in seen:
                result.append(recipe)
                seen.add(key)
        return result
