"""End-to-end smoke test on tiny synthetic data."""
import numpy as np
import pytest

from src.data import make_synthetic_hnoca
from src.tasks import NextTimepointTask, CellTypeOOPTask, PseudotimeTask
from src.models import (
    IdentityNextTimepoint, PopulationMeanNextTimepoint, LinearNextTimepoint,
    LogRegCellType, KNNCellType, PC1Pseudotime,
)


@pytest.fixture(scope="module")
def adata():
    return make_synthetic_hnoca(n_cells=1500, n_genes=200, n_organoids=12, seed=0)


def test_t1_runs(adata):
    task = NextTimepointTask(holdout_organoid_frac=0.25, hvg_top=100)
    res = task.run(adata, IdentityNextTimepoint())
    assert "pearson" in res.metrics
    assert -1 <= res.metrics["pearson"] <= 1
    assert res.metrics["mse"] >= 0


def test_t1_linear_beats_pop_mean_in_pearson(adata):
    task = NextTimepointTask(holdout_organoid_frac=0.25, hvg_top=100)
    res_pm = task.run(adata, PopulationMeanNextTimepoint())
    task2 = NextTimepointTask(holdout_organoid_frac=0.25, hvg_top=100)
    res_lin = task2.run(adata, LinearNextTimepoint(alpha=1.0))
    # Linear should be at least as good as pop-mean on Pearson
    assert res_lin.metrics["pearson"] >= res_pm.metrics["pearson"] - 0.05


def test_t2_runs(adata):
    task = CellTypeOOPTask(holdout_frac=0.25)
    res = task.run(adata, LogRegCellType(max_iter=100))
    assert 0.0 <= res.metrics["accuracy"] <= 1.0


def test_t5_pc1(adata):
    task = PseudotimeTask(hvg_top=80)
    res = task.run(adata, PC1Pseudotime())
    # spearman should beat random by a comfortable margin on synthetic
    assert res.metrics["spearman"] > 0.2
