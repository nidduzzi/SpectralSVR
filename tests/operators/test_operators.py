"""Operator baselines: dataset, SpectralSVR adapter, benchmark, DeepONet smoke."""

import importlib.util
import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from SpectralSVR import Antiderivative, FourierBasis
from SpectralSVR.model import LSSVR
from SpectralSVR.operators import (
    DeepONetOperator,
    Operator,
    OperatorDataset,
    SpectralSVROperator,
    benchmark,
    sensor_grid,
)
from tests._anchors import half_open_grid

_HAS_DEEPXDE = importlib.util.find_spec("deepxde") is not None

SENSORS = 16

QUERY = torch.linspace(0, 1, 40, dtype=torch.float64).unsqueeze(-1)
PROBE_MODES = (1, 2, 3, 4, 5)
# worst relative L2 measured over seeds 0-9 was 0.539; the bound sits above it
# with headroom rather than being tightened until the test passed
CLOSED_FORM_TOLERANCE = 0.70


def _antiderivative_split(seed, n_train=120, n_test=30, modes=SENSORS):
    # operator maps the derivative ut -> the antiderivative u (integration)
    g = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(
        FourierBasis, n_train + n_test, modes, generator=g, u0=0
    )
    train = OperatorDataset.from_fields(ut[:n_train], u[:n_train], n_sensors=modes)
    test = OperatorDataset.from_fields(ut[n_train:], u[n_train:], n_sensors=modes)
    return train, test


@pytest.mark.oracle_smoke
@given(n=st.integers(2, 8), d=st.integers(1, 2))
def test_sensor_grid_shape(n, d):
    domain = tuple((0.0, 1.0) for _ in range(d))
    grid = sensor_grid(domain, n)
    assert grid.shape == (n**d, d)
    # half-open: endpoint excluded on every axis
    assert grid.max() < 1.0


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_operator_dataset_shapes():
    train, _ = _antiderivative_split(0)
    assert train.sensors.shape == (16, 1)
    assert train.f.shape == (120, 16)
    assert train.n_functions == 120
    x = torch.linspace(0, 1, 25).unsqueeze(-1)
    assert train.targets(x).shape == (120, 25)


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_spectralsvr_operator_learns_antiderivative():
    # A*cos(2*pi*m*x) integrates to A/(2*pi*m)*sin(2*pi*m*x), which is zero-mean
    # and so is the antiderivative this operator is trained to produce. The probe
    # amplitude follows the training set's own scale: an RBF kernel extrapolates
    # badly, and a unit-amplitude cosine sits ~20x below the generated inputs.
    x_sensors = half_open_grid(SENSORS)
    x_query = QUERY.squeeze(-1)
    unit = torch.cos(2 * math.pi * x_sensors).norm()
    for seed in range(5):
        train, _ = _antiderivative_split(seed)
        op = SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0)).fit(train)
        amp = float(train.f.norm(dim=1).mean() / unit)

        probes = torch.stack(
            [amp * torch.cos(2 * math.pi * m * x_sensors) for m in PROBE_MODES]
        )
        exact = torch.stack(
            [
                amp / (2 * math.pi * m) * torch.sin(2 * math.pi * m * x_query)
                for m in PROBE_MODES
            ]
        )
        pred = op.predict(probes, QUERY).cpu().to(torch.float64)
        assert pred.shape == exact.shape

        rel = (pred - exact).norm(dim=1) / exact.norm(dim=1)
        assert float(rel.max()) < CLOSED_FORM_TOLERANCE, (
            f"seed={seed}: {[f'{v:.3f}' for v in rel.tolist()]}"
        )

        # discriminating: against the same closed form with its 1/(2*pi*m)
        # amplitude dropped, every probe misses the bound by a wide margin
        perturbed = torch.stack(
            [amp * torch.sin(2 * math.pi * m * x_query) for m in PROBE_MODES]
        )
        off = (pred - perturbed).norm(dim=1) / perturbed.norm(dim=1)
        assert float(off.min()) > CLOSED_FORM_TOLERANCE


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_predict_before_fit_raises():
    op = SpectralSVROperator(FourierBasis(), LSSVR())
    with pytest.raises(AssertionError, match="fit before predict"):
        op.predict(torch.randn(2, 16), torch.linspace(0, 1, 10).unsqueeze(-1))


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_benchmark_returns_metric_table():
    train, test = _antiderivative_split(1)
    query = torch.linspace(0, 1, 40).unsqueeze(-1)
    df = benchmark(
        [SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0))],
        train,
        test,
        query,
    )
    assert list(df.index) == ["SpectralSVR"]
    for col in ("fit_seconds", "rmse", "r2", "rrse"):
        assert col in df.columns
    assert df.loc["SpectralSVR", "r2"] > 0.5


class _Exploding(Operator):
    name = "Exploding"

    def fit(self, dataset):
        raise RuntimeError("fit blew up")

    def predict(self, f, x):
        return torch.zeros(f.shape[0], x.shape[0], dtype=torch.float64)


class _Constant(Operator):
    """Predicts one number everywhere, chosen from the data it was fitted on."""

    name = "Constant"

    def fit(self, dataset):
        self.value = float(dataset.targets(QUERY).mean())
        return self

    def predict(self, f, x):
        return torch.full((f.shape[0], x.shape[0]), self.value, dtype=torch.float64)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_benchmark_propagates_a_failing_operator():
    train, test = _antiderivative_split(0, n_train=20, n_test=5)
    with pytest.raises(RuntimeError, match="fit blew up"):
        benchmark([_Exploding()], train, test, QUERY)


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_benchmark_scores_two_operators_against_the_same_targets():
    train, test = _antiderivative_split(1, n_train=40, n_test=10)
    constant = _Constant()
    df = benchmark(
        [SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0)), constant],
        train,
        test,
        QUERY,
    )
    assert list(df.index) == ["SpectralSVR", "Constant"]

    # the constant row's error is computable by hand, so it pins which targets
    # the benchmark scored against: test.targets(QUERY), not anything per
    # operator. Self-referential by construction -- the check is which library
    # output was used, so the library supplies both sides
    target = test.targets(QUERY)
    expected_mse = float(((target - constant.value) ** 2).mean())
    assert df.loc["Constant", "mse"] == pytest.approx(expected_mse, rel=1e-9)


@pytest.mark.skipif(not _HAS_DEEPXDE, reason="needs the 'baselines' extra (deepxde)")
@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_deeponet_operator_smoke():
    train, test = _antiderivative_split(0, n_train=40, n_test=10)
    query = torch.linspace(0, 1, 20).unsqueeze(-1)
    op = DeepONetOperator(trunk=query, iterations=20, seed=0).fit(train)
    pred = op.predict(test.f, query)
    assert pred.shape == (test.n_functions, query.shape[0])
    assert not torch.isnan(pred).any()
