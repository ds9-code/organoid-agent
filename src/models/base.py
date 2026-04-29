"""Common interface for both classical baselines and FM wrappers."""
from __future__ import annotations
from typing import Any


class BaselineModel:
    name: str = "baseline"

    def fit(self, X, y) -> "BaselineModel":
        return self

    def predict(self, X) -> Any:
        raise NotImplementedError
