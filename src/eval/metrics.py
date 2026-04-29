"""Metric library — small, dependency-light, reused across tasks."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def pearson_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between two (N, G) arrays."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-12
    return num / den


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def sign_accuracy_lineage(pred: np.ndarray, true: np.ndarray, ref: np.ndarray | None = None) -> float:
    """For each (sample, gene), did we get the sign of the *change* right?

    If `ref` is None we treat the per-sample mean as the reference baseline,
    which is a fair stand-in when the upstream task supplies (x_t, x_{t+1})
    pairs and only the prediction is fed in here.
    """
    if ref is None:
        ref = pred.mean(axis=0, keepdims=True)
    sp = np.sign(pred - ref)
    st = np.sign(true - ref)
    return float((sp == st).mean())
