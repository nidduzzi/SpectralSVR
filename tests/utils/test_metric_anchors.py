"""Hand-computed anchors for the get_metrics formula suite.

Fixed tiny arrays with every expected value derived by hand in the comments
(oracle: hand math, not this codebase), including the near-constant-target
regime that exposed torchmetrics' per-column rrse blow-up in the operator
benchmark (rrse ~1.2e5 alongside r2 0.987).
"""

import math

import pytest
import torch

from SpectralSVR.utils import get_metrics

pytestmark = [pytest.mark.no_fuzz, pytest.mark.oracle_analytic]


def test_get_metrics_hand_computed_values():
    # preds/targets chosen so every sum is exact in float64:
    # errors e = p - t = [1, -1, 2, 0]
    p = torch.tensor([[2.0, 1.0], [5.0, 4.0]], dtype=torch.float64)
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    m = get_metrics(p, t)

    # mse  = (1 + 1 + 4 + 0) / 4 = 1.5 ; rmse = sqrt(1.5)
    assert math.isclose(m["mse"], 1.5, rel_tol=1e-6)
    assert math.isclose(m["rmse"], math.sqrt(1.5), rel_tol=1e-6)
    # mae  = (1 + 1 + 2 + 0) / 4 = 1.0
    assert math.isclose(m["mae"], 1.0, rel_tol=1e-6)
    # global target mean = 2.5, SST = 1.5^2+0.5^2+0.5^2+1.5^2 = 5 ; SSE = 6
    # rse  = 6/5 = 1.2 ; rrse = sqrt(1.2)
    assert math.isclose(m["rse"], 1.2, rel_tol=1e-6)
    assert math.isclose(m["rrse"], math.sqrt(1.2), rel_tol=1e-6)
    # smape = mean(2|e| / (|p| + |t|))
    #       = (2*1/3 + 2*1/3 + 2*2/8 + 0) / 4 = (2/3 + 2/3 + 1/2) / 4
    assert math.isclose(m["smape"], (2 / 3 + 2 / 3 + 0.5) / 4, rel_tol=1e-6)
    # torchmetrics r2 on (N, d) input averages per-column scores:
    # col0: SSE=1+4=5, mean=2, SST=1+1=2 -> r2 = 1 - 5/2 = -1.5
    # col1: SSE=1+0=1, mean=3, SST=1+1=2 -> r2 = 1 - 1/2 =  0.5
    # uniform average = -0.5
    assert math.isclose(m["r2"], -0.5, rel_tol=1e-6)
    # per-sample relative L2: row 0 |[1,-1]|/|[1,2]| = sqrt(2/5); row 1 |[2,0]|/|[3,4]| = 2/5
    assert math.isclose(
        m["mean_per_sample_relative_l2"], (math.sqrt(2 / 5) + 2 / 5) / 2, rel_tol=1e-12
    )
    assert m["pred_nan_sum"] == 0


def test_get_metrics_rrse_near_constant_column():
    # Regression for the benchmark anomaly: one target column nearly constant.
    # Global-definition rrse must stay small when predictions are accurate --
    # the per-column torchmetrics formula exploded here (column SST ~ 0).
    t = torch.tensor(
        [[1.0, 0.0], [1.0 + 1e-9, 2.0], [1.0 - 1e-9, 4.0]], dtype=torch.float64
    )
    p = t + 0.01
    m = get_metrics(p, t)
    # SSE = 6 * 1e-4 ; global mean = 8/6 -> SST dominated by column 1 (=8+2/3+...)
    sse = 6 * 1e-4
    sst = ((t - t.mean()) ** 2).sum().item()
    assert math.isclose(m["rse"], sse / sst, rel_tol=1e-6)
    assert math.isclose(m["rrse"], math.sqrt(sse / sst), rel_tol=1e-6)
    assert m["rrse"] < 0.02  # sanity: no blow-up


def test_get_metrics_rse_consistent_with_definition_rrse_squared():
    p = torch.tensor([[0.5, 1.5, -0.25]], dtype=torch.float64).T
    t = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float64).T
    m = get_metrics(p, t)
    assert math.isclose(m["rrse"] ** 2, m["rse"], rel_tol=1e-9)


def test_get_metrics_nan_count():
    p = torch.tensor([[float("nan"), 1.0], [2.0, float("nan")]])
    t = torch.zeros(2, 2)
    assert get_metrics(p, t)["pred_nan_sum"] == 2
