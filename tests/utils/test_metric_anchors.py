"""Hand-computed anchors for the get_metrics formula suite.

Fixed tiny arrays with every expected value derived by hand in the comments
(oracle: hand math, not this codebase), including the near-constant-target
regime that exposed torchmetrics' per-column rrse blow-up in the operator
benchmark (rrse ~1.2e5 alongside r2 0.987).
"""

import math

import numpy
import pytest
import torch
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from SpectralSVR.utils import get_metrics

SETTINGS = settings(
    deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow]
)
INDEPENDENT_RTOL = 1e-10
NEAR_CONSTANT_RTOL = 1e-6
DEGENERATE_SPREAD = 1e-3


@st.composite
def predictions_and_targets(draw):
    shape = (draw(st.integers(2, 20)), draw(st.integers(1, 5)))
    elements = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)
    return (
        draw(hnp.arrays(numpy.float64, shape, elements=elements)),
        draw(hnp.arrays(numpy.float64, shape, elements=elements)),
    )


def metrics_by_numpy(p, t):
    error = p - t
    sse = (error**2).sum()
    sst = ((t - t.mean()) ** 2).sum()
    per_column = 1 - (error**2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
    error_rows = error.reshape(len(error), -1)
    target_rows = t.reshape(len(t), -1)
    return {
        "mse": (error**2).mean(),
        "rmse": math.sqrt((error**2).mean()),
        "mae": numpy.abs(error).mean(),
        "rse": sse / sst,
        "rrse": math.sqrt(sse / sst),
        "smape": (2 * numpy.abs(error) / (numpy.abs(p) + numpy.abs(t))).mean(),
        "r2": per_column.mean(),
        "mean_per_sample_relative_l2": (
            numpy.linalg.norm(error_rows, axis=1)
            / numpy.linalg.norm(target_rows, axis=1)
        ).mean(),
    }


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
@pytest.mark.scenario("Neither replaces the other")
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


@pytest.mark.oracle_independent
@pytest.mark.scenario("Neither replaces the other")
@SETTINGS
@given(pair=predictions_and_targets())
@example(
    pair=(
        numpy.array([[2.0, 1.0], [5.0, 4.0]]),
        numpy.array([[1.0, 2.0], [3.0, 4.0]]),
    )
)
def test_get_metrics_match_independent_numpy_formulas(pair):
    p, t = pair
    assume(((t - t.mean(0)) ** 2).sum(0).min() > DEGENERATE_SPREAD)
    assume(((p - t) ** 2).sum(0).min() > DEGENERATE_SPREAD)
    assume((numpy.abs(p) + numpy.abs(t)).min() > DEGENERATE_SPREAD)
    assume(numpy.linalg.norm(t, axis=1).min() > DEGENERATE_SPREAD)
    measured = get_metrics(torch.from_numpy(p), torch.from_numpy(t))
    for name, expected in metrics_by_numpy(p, t).items():
        assert math.isclose(measured[name], expected, rel_tol=INDEPENDENT_RTOL), name
    assert measured["pred_nan_sum"] == 0


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    spread=st.floats(0.0, 1e-6),
    offset=st.floats(1e-3, 1e-1),
)
@example(spread=1e-9, offset=0.01)
def test_get_metrics_rrse_near_constant_column(spread, offset):
    # Regression for the benchmark anomaly: one target column nearly constant.
    # Global-definition rrse must stay small when predictions are accurate --
    # the per-column torchmetrics formula exploded here (column SST ~ 0).
    t = torch.tensor(
        [[1.0, 0.0], [1.0 + spread, 2.0], [1.0 - spread, 4.0]], dtype=torch.float64
    )
    m = get_metrics(t + offset, t)
    sse = 6 * offset**2
    sst = ((t - t.mean()) ** 2).sum().item()
    assert math.isclose(m["rse"], sse / sst, rel_tol=NEAR_CONSTANT_RTOL)
    assert math.isclose(m["rrse"], math.sqrt(sse / sst), rel_tol=NEAR_CONSTANT_RTOL)
    assert m["rrse"] < offset


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    mask=hnp.arrays(numpy.bool_, st.tuples(st.integers(1, 6), st.integers(1, 4))),
    seed=st.integers(0, 10_000),
)
@example(mask=numpy.array([[True, False], [False, True]]), seed=0)
def test_get_metrics_nan_count(mask, seed):
    values = numpy.random.default_rng(seed).standard_normal(mask.shape)
    predictions = torch.from_numpy(numpy.where(mask, numpy.nan, values))
    assert get_metrics(predictions, torch.zeros_like(predictions))[
        "pred_nan_sum"
    ] == int(mask.sum())
