"""Content-addressed response cache, optionally persisted in SQLite."""
import json
import sqlite3

from .providers import Prediction
from .views import canonical, digest


class PredictionCache:
    def __init__(self, path=None):
        self.path = str(path) if path is not None else None
        self._memory = {}
        if self.path:
            with sqlite3.connect(self.path) as db:
                db.execute("CREATE TABLE IF NOT EXISTS predictions (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @staticmethod
    def key(provider, task, view):
        return digest({"provider": provider.namespace, "task": task.payload(), "view": view.payload()})

    def get(self, key):
        if key in self._memory:
            return self._memory[key]
        if self.path:
            with sqlite3.connect(self.path) as db:
                row = db.execute("SELECT value FROM predictions WHERE key = ?", (key,)).fetchone()
            if row:
                value = json.loads(row[0])
                result = Prediction(tuple(value["probabilities"]), value["input_tokens"], value["output_tokens"])
                self._memory[key] = result
                return result
        return None

    def put(self, key, result):
        if self.path:
            value = canonical({"probabilities": result.probabilities,
                               "input_tokens": result.input_tokens, "output_tokens": result.output_tokens})
            with sqlite3.connect(self.path) as db:
                db.execute("INSERT OR IGNORE INTO predictions VALUES (?, ?)", (key, value))
        self._memory[key] = result

