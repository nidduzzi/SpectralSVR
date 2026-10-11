"""Fuzzed tests for LSSVR."""

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from SpectralSVR import LSSVR, accelerator

from _model_data import SETTINGS, linear_dataset, smooth_dataset

LSSVR_KERNELS = ("linear", "poly", "rbf", "tri", "frob", "max")


@pytest.mark.oracle_consistency
@SETTINGS
@given(
    n=st.integers(30, 60),
    d=st.integers(2, 5),
    o=st.integers(1, 3),
    seed=st.integers(0, 10_000),
)
def test_lssvr_linear_recovers_linear_map(n, d, o, seed):
    # Manufacture y = X W + b; a linear-kernel LSSVR recovers it. The Gram
    # system is rank-deficient/ill-conditioned; C=1e4 regularizes it and float64
    # keeps the lstsq stable on adversarial random draws (float32 drifts).
    X, y = linear_dataset(n, d, o, seed)
    model = LSSVR(kernel="linear", C=1e4, dtype=torch.float64)
    model.fit(X.double(), y.double())
    assert torch.allclose(model.predict(X.double()), y.double(), atol=1e-4)


@pytest.mark.oracle_consistency
@SETTINGS
@given(n=st.integers(20, 50), seed=st.integers(0, 10_000))
def test_lssvr_rbf_interpolates_training_points(n, seed):
    # Manufactured smooth target; rbf LSSVR with large C interpolates closely.
    X, y = smooth_dataset(n, seed)
    model = LSSVR(kernel="rbf", C=1e3)
    model.fit(X, y)
    assert torch.allclose(model.predict(X), y, atol=3e-1)


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(10, 40), d=st.integers(1, 4), o=st.integers(1, 3), seed=st.integers(0, 10_000))
def test_lssvr_predict_numpy_roundtrips(n, d, o, seed):
    X, y = linear_dataset(n, d, o, seed)
    model = LSSVR(kernel="linear")
    model.fit(X, y)
    out = model.predict(X.numpy())
    assert isinstance(out, np.ndarray)
    assert out.shape == (n, o)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    kernel=st.sampled_from(["linear", "poly", "rbf", "tri", "frob", "max"]),
    n=st.integers(10, 30),
    seed=st.integers(0, 10_000),
)
def test_lssvr_all_kernels_fit_predict(kernel, n, seed):
    # every kernel builds, fits and predicts a finite output of the right shape
    X, y = linear_dataset(n, 3, 2, seed)
    model = LSSVR(kernel=kernel, sigma=1.5, d=2.0)
    model.fit(X, y)
    out = model.predict(X)
    assert out.shape == y.shape
    assert not torch.isnan(out).any()


@pytest.mark.oracle_smoke
@SETTINGS
@given(kernel=st.text().filter(lambda name: name not in LSSVR_KERNELS))
@example(kernel="bogus")
def test_lssvr_invalid_kernel_raises(kernel):
    with pytest.raises(KeyError):
        LSSVR(kernel=kernel).fit(torch.randn(6, 2), torch.randn(6, 1))


from SpectralSVR.model import torch_json_encoder  # noqa: E402


@pytest.mark.oracle_consistency
@SETTINGS
@given(n=st.integers(6, 40), d=st.integers(1, 4), seed=st.integers(0, 10_000))
def test_lssvr_batched_kernel_matches_unbatched(n, d, seed):
    # tiny batch_size forces the chunked kernel loop; result must match the
    # single-shot kernel exactly.
    X, y = linear_dataset(n, d, 2, seed)
    unbatched = LSSVR(kernel="rbf", C=10.0, dtype=torch.float64)
    unbatched.fit(X.double(), y.double())
    batched = LSSVR(
        kernel="rbf", C=10.0, dtype=torch.float64, batch_size_func=lambda dims: 1
    )
    batched.fit(X.double(), y.double())
    assert torch.allclose(batched.predict(X.double()), unbatched.predict(X.double()), atol=1e-8)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    unencodable=st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(),
        st.text(),
        st.binary(),
        st.frozensets(st.integers()),
        st.sets(st.integers()),
        st.lists(st.integers()),
    )
)
@example(unencodable={1, 2, 3})
@example(unencodable=object())
def test_torch_json_encoder_rejects_unknown_type(unencodable):
    with pytest.raises(TypeError):
        torch_json_encoder(unencodable)


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(6, 30), d=st.integers(1, 4), seed=st.integers(0, 10_000))
def test_lssvr_fit_accepts_numpy_and_1d_targets(n, d, seed):
    # numpy inputs go through from_numpy; 1D y is viewed as a column
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)  # 1D targets
    model = LSSVR(kernel="linear")
    model.fit(X, y)
    out = model.predict(X)
    assert out.shape[0] == n


@pytest.mark.oracle_independent
@pytest.mark.scenario("A process-wide setting is followed")
@pytest.mark.scenario("The host is not overridden")
@SETTINGS
@given(
    name=st.sampled_from(
        ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    ),
    stated=st.sampled_from((None, "cpu")),
)
@example(name=accelerator().type, stated="cpu")
def test_lssvr_follows_the_process_default(name, stated):
    torch.set_default_device(name)
    expected = torch.zeros(1).device if stated is None else torch.device(stated)
    assert LSSVR(device=stated).device == expected


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "tri", "frob", "max"])
def test_lssvr_each_kernel_covered(kernel):
    # deterministic: every kernel function is exercised on every run
    X, y = linear_dataset(15, 3, 2, seed=0)
    model = LSSVR(kernel=kernel, sigma=1.5, d=2.0)
    model.fit(X, y)
    out = model.predict(X)
    assert out.shape == y.shape and not torch.isnan(out).any()


@pytest.mark.oracle_independent
@SETTINGS
@given(
    n=st.integers(5, 20),
    d=st.integers(1, 4),
    o=st.integers(1, 3),
    c=st.floats(0.5, 100.0),
    seed=st.integers(0, 10_000),
)
def test_lssvr_matches_independent_kkt_solve(n, d, o, c, seed):
    # Independent oracle: the LS-SVM dual (Suykens & Vandewalle 1999) is the
    # linear system  [[0, 1^T], [1, K + I/C]] [b; alpha] = [0; y]  with
    # prediction f(x) = K(x, X) alpha + b. Solve that system inline (different
    # code path: torch.linalg.solve on a hand-assembled matrix with an explicit
    # linear kernel) and require LSSVR to reproduce its predictions.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, o, generator=g, dtype=torch.float64)
    xq = torch.randn(7, d, generator=g, dtype=torch.float64)

    k = x @ x.T  # explicit linear kernel
    a = torch.zeros(n + 1, n + 1, dtype=torch.float64)
    a[0, 1:] = 1
    a[1:, 0] = 1
    a[1:, 1:] = k + torch.eye(n, dtype=torch.float64) / c
    rhs = torch.cat([torch.zeros(1, o, dtype=torch.float64), y])
    sol = torch.linalg.lstsq(a, rhs).solution
    b, alpha = sol[0], sol[1:]
    expected = (xq @ x.T) @ alpha + b

    model = LSSVR(kernel="linear", C=c, dtype=torch.float64).fit(x, y)
    got = model.predict(xq)
    scale = expected.abs().max().clamp(min=1.0)
    assert torch.allclose(got, expected, atol=1e-5 * float(scale), rtol=1e-4)


SIGMA_FROM_DATA = ("rbf", "poly")


class LSSVRLifecycle(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.build("rbf", 1.0, None)

    @rule(
        kernel=st.sampled_from(LSSVR_KERNELS),
        C=st.sampled_from((1.0, 10.0)),
        sigma=st.sampled_from((None, 1.5)),
    )
    def build(self, kernel, C, sigma):
        stated = {"d": 2.0}
        self.stated_sigma = 1.5 if kernel == "tri" else sigma
        if self.stated_sigma is not None:
            stated["sigma"] = self.stated_sigma
        self.kernel = kernel
        self.model = LSSVR(kernel=kernel, C=C, **stated)
        self.fitted_on = None

    @rule(
        n=st.integers(10, 30),
        d=st.integers(1, 4),
        o=st.integers(1, 3),
        seed=st.integers(0, 10_000),
    )
    def fit(self, n, d, o, seed):
        X, y = linear_dataset(n, d, o, seed)
        self.model.fit(X, y)
        self.fitted_on = (X, y)
        assert self.model.trained

    @rule()
    def predict(self):
        if self.fitted_on is None:
            with pytest.raises(AssertionError, match="fitted"):
                self.model.predict(torch.randn(4, 2))
            return
        X, y = self.fitted_on
        out = self.model.predict(X)
        assert out.shape == y.shape
        assert not torch.isnan(out).any()

    @rule()
    def read_the_correlation_image(self):
        if self.fitted_on is None:
            with pytest.raises(AssertionError, match="trained first"):
                self.model.get_correlation_image()
            return
        X, _ = self.fitted_on
        assert self.model.get_correlation_image().shape == X.shape

    @rule()
    def read_the_p_matrix(self):
        if self.fitted_on is None:
            with pytest.raises(AssertionError, match="trained first"):
                self.model.get_p_matrix()
            return
        X, y = self.fitted_on
        assert self.model.get_p_matrix().shape == (X.shape[1], y.shape[1])

    @precondition(lambda self: self.fitted_on is None)
    @rule()
    def read_the_kernel_before_fitting(self):
        assert callable(self.model.K)
        expected = self.stated_sigma
        if expected is None and self.kernel in SIGMA_FROM_DATA:
            expected = 1.0
        assert self.model.kernel_params.get("sigma") == expected

    @invariant()
    def trained_exactly_when_fitted(self):
        assert self.model.trained == (self.fitted_on is not None)

    @initialize()
    def reproduce_every_pinned_case(self):
        self.build("rbf", 1.0, None)
        self.predict()
        self.read_the_correlation_image()
        self.read_the_p_matrix()
        self.read_the_kernel_before_fitting()
        for kernel in LSSVR_KERNELS:
            for sigma in (None, 1.5):
                self.build(kernel, 10.0, sigma)
                self.read_the_kernel_before_fitting()
                self.fit(20, 3, 2, 0)
                self.predict()
                self.read_the_correlation_image()
                self.read_the_p_matrix()
        self.build("linear", 10.0, None)


LSSVRLifecycle.TestCase.settings = settings(
    deadline=None,
    max_examples=25,
    stateful_step_count=8,
    suppress_health_check=[HealthCheck.too_slow],
)
TestLSSVRLifecycle = pytest.mark.oracle_smoke(LSSVRLifecycle.TestCase)
