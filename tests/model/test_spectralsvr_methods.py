"""Coverage for SpectralSVR.test / inverse / inverse_coeff and error paths.

Behavioral tests are fuzzed over sizes/seed; error paths are structural and
opt out of fuzzing. None manufacture a PDE solution.
"""

import logging

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from SpectralSVR import (
    Antiderivative,
    FourierBasis,
    LSSVR,
    SpectralSVR,
    accelerator,
    to_complex_coeff,
    to_real_coeff,
)

SETTINGS = settings(
    deadline=None, max_examples=15, suppress_health_check=[HealthCheck.too_slow]
)


def _trained_model(n=80, modes=8, seed=42, kernel="rbf", device=None):
    g = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(FourierBasis, n, modes, generator=g, u0=0)
    f = to_real_coeff(ut.coeff)  # real input features
    model = SpectralSVR(
        FourierBasis(domain=(0.0, 1.0)), LSSVR(kernel=kernel, C=10.0, device=device)
    )
    model.train(f, u.coeff)
    return model, f, u


# --------------------------------------------------------------------------- #
# behavioral (fuzzed)
# --------------------------------------------------------------------------- #
@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_test_returns_metric_groups(n, modes, seed):
    model, f, u = _trained_model(n, modes, seed)
    metrics = model.test(f, u.coeff)
    assert set(metrics) == {"spectral", "function value"}
    for group in metrics.values():
        assert "mse" in group and group["mse"] >= 0.0


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_inverse_coeff_shape(n, modes, seed):
    model, _, u = _trained_model(n, modes, seed)
    out = model.inverse_coeff(u.coeff, epochs=3)
    assert out.shape == (u.coeff.shape[0], modes)
    assert out.is_complex()


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_inverse_returns_field_values(n, modes, seed):
    model, _, u = _trained_model(n, modes, seed)
    points = FourierBasis.grid(slice(0, 1, 20))
    f_pred = model.inverse(u.coeff, points, epochs=3)
    assert f_pred.shape[0] == u.coeff.shape[0]
    assert not torch.isnan(f_pred).any()


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_forward_1d_x_and_complex_input(n, modes, seed):
    model, f, u = _trained_model(n, modes, seed)
    x = torch.linspace(0, 1, 20)  # 1D coordinate vector -> unsqueezed internally
    assert model.forward(f, x).shape[0] == n
    # complex input features are converted to interleaved real internally
    cf = torch.complex(f[:, ::2], f[:, 1::2])
    assert model.forward(cf, x).shape[0] == n


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_test_with_complex_input(n, modes, seed):
    model, f, u = _trained_model(n, modes, seed)
    cf = torch.complex(f[:, ::2], f[:, 1::2])  # complex features
    assert "spectral" in model.test(cf, u.coeff)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    n=st.integers(20, 60),
    nt=st.integers(2, 8),
    modes=st.integers(4, 10),
    seed=st.integers(0, 10_000),
)
def test_test_time_dependent_and_complex_funcs(n, nt, modes, seed):
    # time-dependent targets + complex-valued functions exercise those branches
    g = torch.Generator().manual_seed(seed)
    u_coeff = torch.randn(n, nt, modes, dtype=torch.complex64, generator=g)
    f = torch.randn(n, 12, generator=g)
    model = SpectralSVR(
        FourierBasis(domain=((0.0, 1.0), (0.0, 1.0)), complex_funcs=True),
        LSSVR(kernel="rbf"),
    )
    model.train(f, u_coeff, u_time_dependent=True)
    metrics = model.test(f, u_coeff)
    assert set(metrics) == {"spectral", "function value"}


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_train_with_complex_input_features(n, modes, seed):
    # complex input features are converted to interleaved real inside train()
    g = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(FourierBasis, n, modes, generator=g, u0=0)
    model = SpectralSVR(FourierBasis(domain=(0.0, 1.0)), LSSVR(kernel="rbf", C=10.0))
    model.train(ut.coeff, u.coeff)  # complex f
    assert model.features == 2 * modes


COMPLEX_PRECISIONS = (torch.complex64, torch.complex128)
REGRESSOR_PRECISIONS = (torch.float32, torch.float64)
INTERLEAVED_PRECISION = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}
BOUNDARY_FUNCTIONS = st.integers(4, 8)
BOUNDARY_SEEDS = st.integers(0, 1000)


class WarningsHeard(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class SpectralSVRBoundary(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.warnings = WarningsHeard()
        self.logger = logging.getLogger(f"boundary.{id(self)}")
        self.logger.propagate = False
        self.logger.addHandler(self.warnings)
        self.model = None
        self.basis_dtype = None
        self.regressor_dtype = None
        self.modes = None
        self.trained = False

    def regressor_discards(self, dtype):
        return (
            INTERLEAVED_PRECISION[dtype].itemsize > self.model.regressor.dtype.itemsize
        )

    def regressor_is_wider(self, dtype):
        return (
            INTERLEAVED_PRECISION[dtype].itemsize < self.model.regressor.dtype.itemsize
        )

    def coefficients(self, dtype, functions, seed):
        return FourierBasis.generate_coeff(
            functions,
            self.modes,
            generator=torch.Generator().manual_seed(seed),
            coeff_dtype=dtype,
        )

    def cross(self, dtype, reaches_the_regressor, roles):
        regressor_was_fitted = self.model.regressor.trained
        self.warnings.messages.clear()
        if self.regressor_discards(dtype):
            with pytest.raises(ValueError, match="discard precision"):
                reaches_the_regressor()
            assert self.model.regressor.trained == regressor_was_fitted
            return None
        result = reaches_the_regressor()
        if self.regressor_is_wider(dtype):
            assert len(self.warnings.messages) == roles
            assert all("Nothing is discarded" in m for m in self.warnings.messages)
        else:
            assert not self.warnings.messages
        return result

    @rule(
        basis_dtype=st.sampled_from(COMPLEX_PRECISIONS),
        regressor_dtype=st.sampled_from(REGRESSOR_PRECISIONS),
        kernel=st.sampled_from(("linear", "rbf")),
        modes=st.integers(2, 5),
    )
    def build(self, basis_dtype, regressor_dtype, kernel, modes):
        self.model = SpectralSVR(
            FourierBasis(coeff_dtype=basis_dtype),
            LSSVR(kernel=kernel, C=1.0, dtype=regressor_dtype),
            logger=self.logger,
        )
        self.basis_dtype = basis_dtype
        self.regressor_dtype = regressor_dtype
        self.modes = modes
        self.trained = False

    @precondition(lambda self: self.model is not None)
    @rule()
    def build_with_a_regressor_fitted_directly(self):
        regressor = LSSVR(kernel="linear", C=10.0, dtype=self.model.regressor.dtype)
        regressor.fit(torch.randn(12, 16), torch.randn(12, 16))
        self.model = SpectralSVR(
            FourierBasis(coeff_dtype=self.basis_dtype), regressor, logger=self.logger
        )
        self.trained = False

    @precondition(lambda self: self.model is not None)
    @rule(functions=BOUNDARY_FUNCTIONS, seed=BOUNDARY_SEEDS)
    def train(self, functions, seed):
        f = self.coefficients(self.basis_dtype, functions, seed)
        u = self.coefficients(self.basis_dtype, functions, seed + 1)
        if self.cross(self.basis_dtype, lambda: self.model.train(f, u), 2) is None:
            return
        self.trained = True
        assert self.model.regressor.trained
        assert self.model.features == 2 * self.modes
        assert self.model.output_shape == (self.modes,)
        interleaved = to_real_coeff(u)
        assert interleaved.dtype == INTERLEAVED_PRECISION[self.basis_dtype]
        assert torch.equal(to_complex_coeff(interleaved), u)

    @precondition(lambda self: self.model is not None)
    @rule()
    def train_on_real_targets(self):
        regressor_was_fitted = self.model.regressor.trained
        with pytest.raises(ValueError, match="must match the basis coeff_dtype"):
            self.model.train(torch.randn(10, 16), torch.randn(10, 8))
        assert self.model.regressor.trained == regressor_was_fitted

    @precondition(lambda self: self.trained)
    @rule(
        dtype=st.sampled_from(COMPLEX_PRECISIONS),
        functions=BOUNDARY_FUNCTIONS,
        seed=BOUNDARY_SEEDS,
    )
    def test(self, dtype, functions, seed):
        f = self.coefficients(dtype, functions, seed)
        u = self.coefficients(dtype, functions, seed + 1)
        metrics = self.cross(dtype, lambda: self.model.test(f, u), 1)
        if metrics is not None:
            assert set(metrics) == {"spectral", "function value"}

    @precondition(lambda self: self.trained)
    @rule(functions=BOUNDARY_FUNCTIONS, seed=BOUNDARY_SEEDS)
    def test_on_real_targets(self, functions, seed):
        f = self.coefficients(self.basis_dtype, functions, seed)
        with pytest.raises(ValueError, match="must match the basis coeff_dtype"):
            self.model.test(
                f, self.coefficients(self.basis_dtype, functions, seed).real
            )

    @precondition(lambda self: self.trained)
    @rule(
        dtype=st.sampled_from(COMPLEX_PRECISIONS),
        functions=BOUNDARY_FUNCTIONS,
        seed=BOUNDARY_SEEDS,
    )
    def forward(self, dtype, functions, seed):
        f = self.coefficients(dtype, functions, seed)
        x = torch.linspace(0, 1, 5)
        values = self.cross(dtype, lambda: self.model.forward(f, x), 1)
        if values is not None:
            assert values.shape == (functions, 5)

    @precondition(lambda self: self.trained)
    @rule(functions=BOUNDARY_FUNCTIONS, seed=BOUNDARY_SEEDS)
    def forward_over_points_of_the_wrong_dimension(self, functions, seed):
        f = self.coefficients(self.basis_dtype, functions, seed)
        with pytest.raises(ValueError, match="num_points"):
            self.model.forward(f, torch.randn(20, 2))

    @precondition(lambda self: self.trained)
    @rule(
        dtype=st.sampled_from(COMPLEX_PRECISIONS),
        functions=BOUNDARY_FUNCTIONS,
        seed=BOUNDARY_SEEDS,
    )
    def inverse(self, dtype, functions, seed):
        u = self.coefficients(dtype, functions, seed)
        recovered = self.cross(dtype, lambda: self.model.inverse_coeff(u, epochs=1), 1)
        if recovered is not None:
            assert recovered.shape == (functions, self.modes)
            assert recovered.is_complex()

    @precondition(lambda self: self.model is not None and not self.trained)
    @rule()
    def invert_before_training(self):
        u = self.coefficients(self.basis_dtype, 4, 0)
        if self.model.regressor.trained:
            message = "features is None"
        else:
            message = "has not been trained"
        with pytest.raises(RuntimeError, match=message):
            self.model.inverse_coeff(u, epochs=1)

    @invariant()
    def the_regressor_keeps_the_precision_it_was_built_with(self):
        if self.model is not None:
            assert self.model.regressor.dtype == self.regressor_dtype

    @initialize()
    def reproduce_every_pinned_case(self):
        for basis_dtype in COMPLEX_PRECISIONS:
            for regressor_dtype in REGRESSOR_PRECISIONS:
                self.build(basis_dtype, regressor_dtype, "linear", 4)
                self.invert_before_training()
                self.train_on_real_targets()
                self.build_with_a_regressor_fitted_directly()
                self.invert_before_training()
                self.build(basis_dtype, regressor_dtype, "linear", 4)
                self.train(functions=6, seed=0)
                self.train_on_real_targets()
                if not self.trained:
                    continue
                for dtype in COMPLEX_PRECISIONS:
                    self.test(dtype, 6, 9)
                    self.forward(dtype, 6, 9)
                    self.inverse(dtype, 6, 9)
                self.test_on_real_targets(6, 9)
                self.forward_over_points_of_the_wrong_dimension(6, 9)
        self.build(torch.complex64, torch.float32, "rbf", 8)
        self.train(functions=80, seed=42)
        self.test_on_real_targets(80, 42)
        self.forward_over_points_of_the_wrong_dimension(80, 42)


SpectralSVRBoundary.TestCase.settings = settings(
    deadline=None,
    max_examples=25,
    stateful_step_count=8,
    suppress_health_check=[HealthCheck.too_slow],
)
TestSpectralSVRBoundary = pytest.mark.oracle_invariant(SpectralSVRBoundary.TestCase)


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_bridge_round_trips_exactly_at_both_precisions():
    generator = torch.Generator().manual_seed(6)
    for complex_dtype in (torch.complex64, torch.complex128):
        for shape in ((5, 9), (2, 8, 8)):
            c = torch.randn(*shape, dtype=complex_dtype, generator=generator)
            assert torch.equal(to_complex_coeff(to_real_coeff(c)), c), (
                f"{complex_dtype} {shape}"
            )


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("A source the library builds for itself agrees by construction")
def test_the_inverse_builds_its_own_source_where_the_work_is():
    if not torch.accelerator.is_available():
        pytest.skip("an accelerator must be present for the work to be placed off host")
    model, _, u = _trained_model(device=accelerator())
    assert model.regressor.device.type == accelerator().type
    out = model.inverse_coeff(u.coeff, epochs=3)
    assert out.shape == u.coeff.shape
    assert out.device == u.coeff.device


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("A mismatched random source names both devices")
def test_the_inverse_refuses_a_source_placed_elsewhere():
    if not torch.accelerator.is_available():
        pytest.skip("two device types must be present for them to disagree")
    model, _, u = _trained_model(device=accelerator())
    with pytest.raises(RuntimeError, match="move the source"):
        model.inverse_coeff(u.coeff, epochs=3, generator=torch.Generator())
