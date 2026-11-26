"""Anchored values of coefficient manipulation and time-axis conversion and calculus."""

import math

import numpy

import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from tests._anchors import (
    half_open_grid,
    hermitian_error,
    hermitian_spectrum,
    resized_by_frequency,
    single_mode_spectrum,
)
from SpectralSVR import FourierBasis
from SpectralSVR.basis.sampling import ClosedUniform, PeriodicUniform
from SpectralSVR.basis.strategy import EvaluationStrategy
from SpectralSVR.utils import (
    resize_modes,
)

SETTINGS = settings(deadline=None, max_examples=20)


# --------------------------------------------------------------------------- #
# resize_modes / zero_coeff: literal hand-derived spectra.
#
# fft bin layout for N samples: [0, 1, ..., ceil(N/2)-1, -floor(N/2), ..., -1].
# Resizing keeps the lowest |frequency| bins; zeroing wipes the middle (high
# |frequency|) block. Expected arrays below are written from that layout by
# hand, not computed with the functions under test.
# --------------------------------------------------------------------------- #
@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_resize_modes_truncate_values():
    c0, c1, c2, c3 = 1.0 + 1j, 2.0 - 1j, -3.0 + 0j, 0.5 + 2j
    x = torch.tensor([[c0, c1, c2, c3]])  # freqs [0, 1, +-2, -1]
    # target 2 modes -> freqs {0, -1}. On a 2-point grid freqs +1 and -1 are the
    # same sampled wave, (-1)^n, so the target's -1 bin carries c3 + c1.
    expected = torch.tensor([[c0, c3 + c1]])
    got = resize_modes(x, 2, rescale=False)
    assert torch.equal(got, expected)
    # rescale multiplies by target/current = 2/4
    got_rescaled = resize_modes(x, 2, rescale=True)
    assert torch.allclose(got_rescaled, expected * 0.5)


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_resize_modes_pad_values():
    c0, c1, c2, c3 = 1.0 + 1j, 2.0 - 1j, -3.0 + 0j, 0.5 + 2j
    x = torch.tensor([[c0, c1, c2, c3]])
    # target 6 modes: low block freqs {0, 1} = bins {0, 1}; high (negative)
    # block freqs {-2, -1} = old bins {2, 3} land at new bins {4, 5}; new bin 3
    # (freq -3) is zero-padding. c2 sat at freq -2, which the 4-mode grid held
    # without a +2 partner; the 6-mode grid has one, so c2 halves across both.
    expected = torch.tensor([[c0, c1, c2 / 2, 0, c2 / 2, c3]])
    got = resize_modes(x, 6, rescale=False)
    assert torch.equal(got, expected)
    got_rescaled = resize_modes(x, 6, rescale=True)
    assert torch.allclose(got_rescaled, expected * 6 / 4)


# --------------------------------------------------------------------------- #
# resize_modes boundary rule: invariants rather than literal spectra.
#
# On an even mode count the top magnitude has a negative bin and no positive
# one, so it is its own conjugate partner. Padding splits it across both of the
# target's bins; truncating folds both back into the one bin the target has.
# The two are inverses, so reverting either alone breaks the round-trip.
# --------------------------------------------------------------------------- #
RESIZE_SOURCE_MODES = (4, 5, 8, 9, 12)
RESIZE_TOLERANCE = 1e-14


@pytest.mark.oracle_independent
@SETTINGS
@given(
    rows=st.integers(1, 3),
    modes=st.integers(1, 24),
    target=st.integers(1, 48),
    seed=st.integers(0, 10_000),
)
@example(rows=1, modes=4, target=2, seed=0)
@example(rows=1, modes=4, target=6, seed=0)
def test_resizing_matches_an_independent_frequency_map(rows, modes, target, seed):
    generator = numpy.random.default_rng(seed)
    x = generator.standard_normal((rows, modes)) + 1j * generator.standard_normal(
        (rows, modes)
    )
    got = resize_modes(torch.from_numpy(x), target, rescale=False)
    assert torch.equal(got, torch.from_numpy(resized_by_frequency(x, target)))


class ResizeRoundTrips(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.start(4, 4)

    @rule(modes=st.integers(2, 16), seed=st.integers(0, 10_000))
    def start(self, modes, seed):
        self.source = hermitian_spectrum(modes, seed=seed)
        self.modes = modes
        self.current = self.source
        assert hermitian_error(self.source) == 0.0

    @rule(growth=st.integers(0, 40))
    def pad(self, growth):
        target = self.current.shape[1] + growth
        self.current = resize_modes(self.current, target, rescale=False)

    @rule()
    def truncate_back_to_the_source_count(self):
        returned = resize_modes(self.current, self.modes, rescale=False)
        assert float((returned - self.source).abs().max()) < RESIZE_TOLERANCE
        self.current = returned

    @invariant()
    def the_padded_spectrum_describes_a_real_field(self):
        assert hermitian_error(self.current) < RESIZE_TOLERANCE

    @invariant()
    def truncating_returns_the_source(self):
        returned = resize_modes(self.current, self.modes, rescale=False)
        assert float((returned - self.source).abs().max()) < RESIZE_TOLERANCE

    @invariant()
    def padding_preserves_the_function_at_the_source_points(self):
        count = self.current.shape[1]
        if count % self.modes:
            return
        coarse = torch.fft.ifft(self.source) * self.modes
        fine = torch.fft.ifft(self.current) * count
        measured = float((fine[:, :: count // self.modes] - coarse).abs().max())
        assert measured < RESIZE_TOLERANCE

    @invariant()
    def an_odd_source_count_is_copied_through(self):
        if self.modes % 2 == 0:
            return
        count = self.current.shape[1]
        for j in range(self.modes):
            frequency = j if j <= (self.modes - 1) // 2 else j - self.modes
            assert self.current[0, frequency % count] == self.source[0, j]

    @initialize()
    def reproduce_every_pinned_case(self):
        for modes in RESIZE_SOURCE_MODES:
            self.start(modes, modes)
            self.pad(modes)
            self.truncate_back_to_the_source_count()
            self.pad(3 * modes)
            self.truncate_back_to_the_source_count()
            self.pad(modes)
            self.pad(2 * modes)
        self.start(4, 4)


ResizeRoundTrips.TestCase.settings = settings(
    deadline=None, max_examples=40, stateful_step_count=8
)
TestResizeRoundTrips = pytest.mark.oracle_invariant(ResizeRoundTrips.TestCase)


# --------------------------------------------------------------------------- #
# Time-axis calculus on time-dependent fields (finite differences over the
# time samples). Field u(t, x) = g(t) sin(2 pi x): the coefficients at each
# time slice are g(t_j) * S where S is the fixed spatial spectrum, so the
# time-derivative coefficients must be g'(t_j) * S. With quadratic g the
# interior central difference is EXACT (error term carries g''' = 0).
# --------------------------------------------------------------------------- #
@pytest.mark.oracle_analytic
@SETTINGS
@given(
    a=st.floats(-2, 2),
    b=st.floats(-2, 2),
    c=st.floats(-2, 2),
    m=st.integers(1, 5),
)
def test_time_dependent_grad_time_axis_matches_analytic(a, b, c, m):
    nt, ns, T = 17, 16, 2.0
    t = torch.linspace(0, T, nt, dtype=torch.float64)  # closed evolution axis
    x = half_open_grid(ns)
    g = a + b * t + c * t * t
    dg = b + 2 * c * t
    values = g.view(nt, 1) * torch.sin(2 * math.pi * m * x)
    coeff = FourierBasis.transform(values + 0j).unsqueeze(0)  # (1, nt, ns)
    u = FourierBasis(coeff, time_dependent=True, domain=((0.0, T), (0.0, 1.0)))

    ut = u.grad(dim=0)
    got = FourierBasis.inv_transform(ut.coeff[0]).real  # (nt, ns) values
    expected = dg.view(nt, 1) * torch.sin(2 * math.pi * m * x)
    # interior points: central difference exact for quadratic g
    assert torch.allclose(got[1:-1], expected[1:-1].to(got.dtype), atol=1e-3)


@pytest.mark.oracle_analytic
@SETTINGS
@given(a=st.floats(-2, 2), b=st.floats(-2, 2), m=st.integers(1, 5))
def test_time_dependent_integral_time_axis_matches_analytic(a, b, m):
    # integral is a left/right Riemann sum (cumsum * dt): first-order accurate,
    # so anchor against the exact antiderivative with an O(dt) tolerance that
    # scales with the drawn g -- NOT a tuned constant.
    nt, ns, T = 201, 16, 1.0
    t = torch.linspace(0, T, nt, dtype=torch.float64)
    dt = T / (nt - 1)
    x = half_open_grid(ns)
    g = a + b * t
    G = a * t + b * t * t / 2  # exact running integral from t=0
    values = g.view(nt, 1) * torch.sin(2 * math.pi * m * x)
    coeff = FourierBasis.transform(values + 0j).unsqueeze(0)
    u = FourierBasis(coeff, time_dependent=True, domain=((0.0, T), (0.0, 1.0)))

    ui = u.integral(dim=0)
    got = FourierBasis.inv_transform(ui.coeff[0]).real
    expected = G.view(nt, 1) * torch.sin(2 * math.pi * m * x)
    tol = dt * (abs(a) + abs(b) * T) * (nt * dt) + dt * (abs(a) + abs(b) * T) + 1e-3
    assert torch.allclose(got, expected.to(got.dtype), atol=tol)


DOUBLE_PRECISION_ROUNDING = 1e-12
ROUNDING = {torch.complex64: 1e-5, torch.complex128: DOUBLE_PRECISION_ROUNDING}
PRECISIONS = st.sampled_from(tuple(ROUNDING))
SAMPLINGS = st.sampled_from(
    (
        None,
        PeriodicUniform(supports_fft=False),
        (ClosedUniform(), PeriodicUniform()),
        (ClosedUniform(), PeriodicUniform(supports_fft=False)),
    )
)
SPATIAL_SAMPLINGS = st.sampled_from((None, PeriodicUniform(supports_fft=False)))
STRATEGIES = st.sampled_from(
    (None, EvaluationStrategy(memory_budget_mb=1e-4, allow_approximate=False))
)
INTEGER_TYPES = st.sampled_from((int, numpy.int64, numpy.array, torch.tensor))
AMPLITUDES = st.lists(
    st.complex_numbers(min_magnitude=0.5, max_magnitude=2.0, allow_subnormal=False),
    min_size=1,
    max_size=3,
)


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    m_t=st.integers(1, 3),
    m_x=st.integers(1, 5),
    amp=st.floats(0.5, 2.0),
    phase=st.floats(0, 2 * math.pi),
)
def test_to_time_dependent_slice_spectra_match_analytic(m_t, m_x, amp, phase):
    nt = ns = 16
    t = half_open_grid(nt)
    x = half_open_grid(ns)
    values = torch.cos(2 * math.pi * m_t * t).view(nt, 1) * (
        amp * torch.cos(2 * math.pi * m_x * x + phase)
    )
    coeff = FourierBasis.transform(values.unsqueeze(0) + 0j)
    u = FourierBasis(coeff, domain=((0.0, 1.0), (0.0, 1.0)))

    td = u.to_time_dependent()
    assert td.time_dependent and td.coeff.shape == (1, nt, ns)
    g = torch.cos(2 * math.pi * m_t * torch.linspace(0, 1, nt, dtype=torch.float64))
    s = single_mode_spectrum(ns, m_x, amp, phase)
    for j in range(nt):
        assert torch.allclose(
            td.coeff[0, j], g[j] * s, atol=DOUBLE_PRECISION_ROUNDING * ns * amp
        ), f"slice {j}"


def time_periodic_field(
    modes, start, span, m_t, m_x, phase, amplitudes, precision, sampling, strategy
):
    t = half_open_grid(modes[0], start, start + span)
    x = half_open_grid(modes[1])
    profile = torch.cos(2 * math.pi * m_t * (t - start) / span + phase).view(
        -1, 1
    ) * torch.cos(2 * math.pi * m_x * x)
    values = torch.tensor(amplitudes, dtype=torch.complex128).view(-1, 1, 1) * profile
    return FourierBasis(
        FourierBasis.transform(values).to(precision),
        complex_funcs=True,
        domain=((start, start + span), (0.0, 1.0)),
        sampling=sampling,
        strategy=strategy,
        coeff_dtype=precision,
    )


def time_periodic_values(points, start, span, m_t, m_x, phase, amplitudes):
    t, x = points[:, 0], points[:, 1]
    profile = torch.cos(2 * math.pi * m_t * (t - start) / span + phase) * torch.cos(
        2 * math.pi * m_x * x
    )
    return torch.tensor(amplitudes, dtype=torch.complex128).view(-1, 1) * profile


@pytest.mark.oracle_analytic
@pytest.mark.scenario("A converted field reads as its source at every time")
@pytest.mark.scenario("Any slice count is accepted")
@pytest.mark.scenario("A round trip stays within interpolation error")
@SETTINGS
@given(
    modes=st.tuples(st.integers(2, 8), st.integers(2, 6)),
    slices=st.one_of(st.none(), st.integers(2, 12)),
    integer_type=INTEGER_TYPES,
    start=st.floats(-2.0, 2.0),
    span=st.floats(0.5, 4.0),
    wavenumbers=st.tuples(st.integers(0, 3), st.integers(0, 2)),
    phase=st.floats(0, 2 * math.pi),
    amplitudes=AMPLITUDES,
    precision=PRECISIONS,
    sampling=SAMPLINGS,
    strategy=STRATEGIES,
    probes=st.lists(st.floats(0, 1), min_size=1, max_size=8),
)
@example(
    modes=(3, 2),
    slices=2,
    integer_type=int,
    start=0.0,
    span=1.0,
    wavenumbers=(1, 0),
    phase=0.0,
    amplitudes=[1.0],
    precision=torch.complex128,
    sampling=None,
    strategy=None,
    probes=[0.5],
)
@example(
    modes=(8, 4),
    slices=5,
    integer_type=numpy.int64,
    start=1.0,
    span=2.0,
    wavenumbers=(1, 1),
    phase=0.3,
    amplitudes=[1.0, 0.5j, -1.5 + 1j],
    precision=torch.complex64,
    sampling=(ClosedUniform(), PeriodicUniform()),
    strategy=None,
    probes=[0.1, 0.6],
)
@example(
    modes=(6, 4),
    slices=6,
    integer_type=numpy.array,
    start=1.0,
    span=2.0,
    wavenumbers=(2, 1),
    phase=1.0,
    amplitudes=[1.0],
    precision=torch.complex128,
    sampling=PeriodicUniform(supports_fft=False),
    strategy=None,
    probes=[0.25, 0.75],
)
@example(
    modes=(4, 3),
    slices=11,
    integer_type=torch.tensor,
    start=-1.5,
    span=0.5,
    wavenumbers=(1, 1),
    phase=2.0,
    amplitudes=[2.0],
    precision=torch.complex128,
    sampling=None,
    strategy=EvaluationStrategy(memory_budget_mb=1e-4, allow_approximate=False),
    probes=[0.05, 0.95],
)
@example(
    modes=(8, 8),
    slices=None,
    integer_type=int,
    start=0.0,
    span=1.0,
    wavenumbers=(1, 1),
    phase=0.0,
    amplitudes=[1.0],
    precision=torch.complex64,
    sampling=None,
    strategy=None,
    probes=[0.5, 0.75],
)
def test_to_time_dependent_and_back_read_as_the_source(
    modes,
    slices,
    integer_type,
    start,
    span,
    wavenumbers,
    phase,
    amplitudes,
    precision,
    sampling,
    strategy,
    probes,
):
    m_t = wavenumbers[0] % ((modes[0] + 1) // 2)
    m_x = wavenumbers[1] % ((modes[1] + 1) // 2)
    field = (start, span, m_t, m_x, phase, amplitudes)
    source = time_periodic_field(modes, *field, precision, sampling, strategy)
    converted = source.to_time_dependent(
        nt=None if slices is None else integer_type(slices)
    )
    nt = modes[0] if slices is None else slices
    assert converted.time_dependent and converted.time_size == nt
    assert converted.coeff.dtype == converted.config.coeff_dtype == precision
    assert len(converted.axis_sampling) == converted.ndim == 1

    rounding = ROUNDING[precision] * max(abs(a) for a in amplitudes)
    interpolation_bound = (
        (span / (nt - 1)) ** 2
        / 8
        * (2 * math.pi * m_t / span) ** 2
        * max(abs(a) for a in amplitudes)
    )
    x = half_open_grid(modes[1])
    slice_times = torch.linspace(start, start + span, nt, dtype=torch.float64)
    probe_times = start + torch.tensor(probes, dtype=torch.float64) * span
    for times, tolerance in (
        (slice_times, rounding),
        (probe_times, interpolation_bound + rounding),
    ):
        points = torch.cartesian_prod(times, x)
        got = FourierBasis.evaluate_time_dependent_at(
            converted.coeff, points.to(precision.to_real()), domain=converted.domain
        )
        expected = time_periodic_values(points, *field)
        assert (got.to(torch.complex128) - expected).abs().max() <= tolerance

    round_trip = converted.to_time_independent()
    assert round_trip.coeff.dtype == round_trip.config.coeff_dtype == precision
    assert len(round_trip.axis_sampling) == round_trip.ndim == 2
    assert isinstance(round_trip.axis_sampling[0], PeriodicUniform)
    node_points = torch.cartesian_prod(half_open_grid(nt, start, start + span), x)
    got = FourierBasis.inv_transform(round_trip.coeff).flatten(1)
    expected = time_periodic_values(node_points, *field)
    assert (got.to(torch.complex128) - expected).abs().max() <= (
        interpolation_bound + rounding
    )


@pytest.mark.oracle_independent
@pytest.mark.scenario(
    "A time-independent field reads as its time-dependent source at its nodes"
)
@SETTINGS
@given(
    slices=st.integers(2, 12),
    spatial_modes=st.integers(2, 6),
    functions=st.integers(1, 3),
    start=st.floats(-2.0, 2.0),
    span=st.floats(0.5, 4.0),
    precision=PRECISIONS,
    sampling=SPATIAL_SAMPLINGS,
    seed=st.integers(0, 10_000),
)
@example(
    slices=2,
    spatial_modes=2,
    functions=1,
    start=0.0,
    span=1.0,
    precision=torch.complex128,
    sampling=None,
    seed=0,
)
@example(
    slices=8,
    spatial_modes=4,
    functions=3,
    start=1.0,
    span=2.0,
    precision=torch.complex64,
    sampling=PeriodicUniform(supports_fft=False),
    seed=1,
)
@example(
    slices=12,
    spatial_modes=6,
    functions=1,
    start=-2.0,
    span=3.0,
    precision=torch.complex128,
    sampling=None,
    seed=2,
)
def test_to_time_independent_reads_as_its_time_dependent_source_at_its_nodes(
    slices, spatial_modes, functions, start, span, precision, sampling, seed
):
    generator = torch.Generator().manual_seed(seed)
    slice_values = torch.randn(
        functions, slices, spatial_modes, generator=generator, dtype=torch.complex128
    )
    coeff = FourierBasis.transform(slice_values.flatten(0, 1)).unflatten(
        0, (functions, slices)
    )
    source = FourierBasis(
        coeff.to(precision),
        complex_funcs=True,
        time_dependent=True,
        domain=((start, start + span), (0.0, 1.0)),
        sampling=sampling,
        coeff_dtype=precision,
    )

    converted = source.to_time_independent()
    assert not converted.time_dependent
    assert converted.coeff.shape == (functions, slices, spatial_modes)
    assert converted.coeff.dtype == converted.config.coeff_dtype == precision
    got = FourierBasis.inv_transform(converted.coeff).to(torch.complex128).numpy()
    node_times = half_open_grid(slices, start, start + span).numpy()
    slice_times = torch.linspace(start, start + span, slices, dtype=torch.float64)
    expected = numpy.stack(
        [
            numpy.stack(
                [
                    numpy.interp(node_times, slice_times.numpy(), column.real)
                    + 1j * numpy.interp(node_times, slice_times.numpy(), column.imag)
                    for column in function.T.numpy()
                ],
                axis=1,
            )
            for function in slice_values
        ]
    )
    rounding = ROUNDING[precision] * float(slice_values.abs().max())
    assert numpy.abs(got - expected).max() <= rounding


APPROXIMATE = EvaluationStrategy(memory_budget_mb=1e-6, allow_approximate=True)


@pytest.mark.oracle_consistency
@pytest.mark.scenario("A conversion honours its evaluation strategy")
@SETTINGS
@given(
    modes=st.tuples(st.integers(2, 12), st.integers(2, 8)),
    slices=st.integers(2, 12),
    strategy=st.sampled_from((APPROXIMATE, EvaluationStrategy())),
    seed=st.integers(0, 10_000),
)
@example(modes=(4, 4), slices=3, strategy=APPROXIMATE, seed=0)
@example(modes=(2, 4), slices=3, strategy=APPROXIMATE, seed=0)
@example(modes=(12, 8), slices=12, strategy=APPROXIMATE, seed=1)
@example(modes=(4, 4), slices=3, strategy=EvaluationStrategy(), seed=0)
def test_a_conversion_honours_its_evaluation_strategy(modes, slices, strategy, seed):
    generator = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.transform(
        torch.randn(2, *modes, generator=generator, dtype=torch.float64) + 0j
    )
    source = FourierBasis(coeff, strategy=strategy)

    converted = source.to_time_dependent(nt=slices)
    values_at_slice_times = FourierBasis.inv_transform(
        coeff,
        res=(slice(0.0, 1.0, slices), slice(0.0, 1.0, modes[1])),
        sampling=(ClosedUniform(), PeriodicUniform()),
        strategy=strategy,
        domain=source.domain,
    )
    by_hand = FourierBasis.transform(
        values_at_slice_times.flatten(0, 1), strategy=strategy
    ).unflatten(0, (2, slices))
    assert torch.allclose(
        converted.coeff, by_hand, atol=DOUBLE_PRECISION_ROUNDING, rtol=0
    )


@pytest.mark.oracle_independent
@SETTINGS
@given(m_t=st.integers(1, 3), m_x=st.integers(1, 5), amp=st.floats(0.5, 2.0))
def test_to_time_independent_values_match_analytic(m_t, m_x, amp):
    nt, ns = 16, 16
    slice_times = torch.linspace(0, 1, nt, dtype=torch.float64)
    x = half_open_grid(ns)
    spatial = amp * torch.sin(2 * math.pi * m_x * x)
    g = torch.cos(2 * math.pi * m_t * slice_times)
    coeff = FourierBasis.transform(g.view(nt, 1) * spatial + 0j).unsqueeze(0)
    u = FourierBasis(coeff, time_dependent=True, domain=((0.0, 1.0), (0.0, 1.0)))

    ti = u.to_time_independent()
    assert not ti.time_dependent
    got = FourierBasis.inv_transform(ti.coeff)[0].real.double().cpu()
    g_at_nodes = numpy.interp(
        half_open_grid(nt).numpy(), slice_times.numpy(), g.numpy()
    )
    expected = torch.from_numpy(g_at_nodes).view(nt, 1) * spatial
    assert torch.allclose(got, expected, atol=DOUBLE_PRECISION_ROUNDING * amp)
