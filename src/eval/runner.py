"""Tiny harness: run a list of (task, model) pairs and collect results."""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import time

from ..tasks.base import Task, TaskOutput
from ..models.base import BaselineModel


def run_pairs(adata, pairs: Iterable[Tuple[Task, BaselineModel]]) -> List[TaskOutput]:
    out = []
    for task, model in pairs:
        t0 = time.time()
        res = task.run(adata, model)
        res.extras["wall_seconds"] = time.time() - t0
        res.extras["model"] = model.name
        out.append(res)
    return out


def to_long_table(results: List[TaskOutput]):
    import pandas as pd
    rows = []
    for r in results:
        for metric, val in r.metrics.items():
            rows.append({
                "task": r.name, "model": r.extras.get("model", "?"),
                "metric": metric, "value": val,
                "n_train": r.n_train, "n_eval": r.n_eval,
                "wall_s": r.extras.get("wall_seconds", 0),
            })
    return pd.DataFrame(rows)
