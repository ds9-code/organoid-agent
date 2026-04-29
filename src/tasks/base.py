"""Abstract Task interface. Every benchmark implements this."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
import anndata as ad
import numpy as np


@dataclass
class TaskOutput:
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_eval: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


class Task:
    """Base class. Subclasses implement `prepare`, `score`, and declare `name`."""
    name: str = "task"

    def prepare(self, adata: ad.AnnData) -> Tuple[Any, Any, Any, Any]:
        """Return (X_train, y_train, X_eval, y_eval). Shapes are task-defined."""
        raise NotImplementedError

    def score(self, y_pred: Any, y_eval: Any) -> Dict[str, float]:
        """Return a dict of metric_name -> float. Higher-is-better unless prefixed `mse`."""
        raise NotImplementedError

    def run(self, adata: ad.AnnData, model) -> TaskOutput:
        X_tr, y_tr, X_ev, y_ev = self.prepare(adata)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_ev)
        metrics = self.score(y_pred, y_ev)
        return TaskOutput(
            name=self.name,
            metrics=metrics,
            n_train=_safe_len(X_tr),
            n_eval=_safe_len(X_ev),
        )


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return getattr(x, "shape", [0])[0] if hasattr(x, "shape") else 0
