"""Fuzzed + MMS tests for the ETDRK4 exponential integrator and the
monolithic-to-semilinear solver conversion."""

import math

import pytest
import torch
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from SpectralSVR.utils import (
    _etdrk4_weights,
    as_semilinear_solver,
    etdrk4_solver,
    euler_solver,
)

SETTINGS = settings(
    deadline=None, max_examples=20, suppress_health_check=[HealthCheck.too_slow]
)


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    N=st.integers(8, 64),
    nu=st.floats(0.01, 0.5),
    seed=st.integers(0, 10_000),
)
def test_etdrk4_linear_heat_matches_exact_1d(N, nu, seed):
    # With N(v)=0 the ODE dv/dt = L v has the exact solution v(t)=v0 exp(L t);
    # ETDRK4 integrates the linear part exactly, so it must match to fp error.
    k = torch.fft.fftfreq(N) * N
    linear = -nu * k**2
    v0 = torch.randn(
        3, N, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed)
    )
    t = torch.linspace(0, 1.0, 50, dtype=torch.float64)

    def zero_nl(ti, v):
        return torch.zeros_like(v)

    sol = etdrk4_solver(linear, zero_nl, v0, t)
    exact = v0 * torch.exp(linear * float(t[-1]))
    assert torch.allclose(sol[-1], exact, atol=1e-8)


@pytest.mark.oracle_analytic
@SETTINGS
@given(N=st.integers(4, 16), nu=st.floats(0.01, 0.3), seed=st.integers(0, 10_000))
def test_etdrk4_linear_heat_matches_exact_2d(N, nu, seed):
    # Same exact-integration check on a 2D mode grid (dimension-agnostic solver).
    k = torch.fft.fftfreq(N) * N
    kx = k.view(-1, 1)
    ky = k.view(1, -1)
    linear = -nu * (kx**2 + ky**2)
    v0 = torch.randn(
        2, N, N, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed)
    )
    t = torch.linspace(0, 0.5, 30, dtype=torch.float64)

    def zero_nl(ti, v):
        return torch.zeros_like(v)

    sol = etdrk4_solver(linear, zero_nl, v0, t)
    exact = v0 * torch.exp(linear * float(t[-1]))
    assert torch.allclose(sol[-1], exact, atol=1e-8)


@pytest.mark.oracle_invariant
@SETTINGS
@given(N=st.sampled_from([64, 128]), seed=st.integers(0, 10_000))
def test_etdrk4_burgers_is_stable(N, seed):
    # Nonlinear viscous Burgers should integrate without blowing up and should
    # not increase energy (viscous dissipation).
    length = 2 * math.pi
    k = torch.fft.fftfreq(N, d=length / N) * 2 * math.pi
    nu = 0.05
    linear = -nu * k**2
    dealias = (torch.fft.fftfreq(N) * N).abs() <= N // 3

    def burgers_nl(ti, v):
        u = torch.fft.ifft(v, dim=-1).real.to(v.dtype)
        return -0.5j * k * torch.fft.fft(u * u, dim=-1) * dealias

    x = torch.linspace(0, length, N + 1, dtype=torch.float64)[:-1]
    u0 = torch.sin(x).unsqueeze(0) + 0.1 * torch.randn(
        2, N, generator=torch.Generator().manual_seed(seed)
    )
    v0 = torch.fft.fft(u0 + 0j, dim=-1)
    t = torch.linspace(0, 1.0, 100, dtype=torch.float64)

    sol = etdrk4_solver(linear, burgers_nl, v0, t)
    u = torch.fft.ifft(sol, dim=-1).real
    assert not torch.isnan(u).any()
    e0 = (u[0] ** 2).mean()
    ef = (u[-1] ** 2).mean()
    assert ef <= e0 + 1e-6


FORCED_MMS_ERROR_ALL_STAGES_AT_T_N = 1.42e-3
FORCED_MMS_ERROR_COX_MATTHEWS_STAGE_TIMES = 1.68e-7
FORCED_MMS_TOLERANCE = 2e-6


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    mode=st.integers(1, 4),
    nu=st.floats(0.02, 0.2),
    seed=st.integers(0, 10_000),
)
def test_etdrk4_reproduces_forced_manufactured_solution(mode, nu, seed):
    # MMS fuzzing the nonlinear solver: manufacture u = A sin(kx+phi) exp(-nu k^2 t),
    # which solves the heat equation. Forcing f = u u_x makes it an exact Burgers
    # solution, so integrating u_t = -u u_x + nu u_xx + f must reproduce u.
    N = 128
    length = 2 * math.pi
    g = torch.Generator().manual_seed(seed)
    k = (2 * math.pi * torch.fft.fftfreq(N, d=length / N)).view(1, N)
    kx = 2 * math.pi * mode / length
    A = (0.5 + torch.rand(2, 1, generator=g)).double()
    phi = (2 * math.pi * torch.rand(2, 1, generator=g)).double()
    x = torch.linspace(0, length, N + 1, dtype=torch.float64)[:-1].view(1, N)
    dealias = (torch.fft.fftfreq(N) * N).abs() <= N // 3

    def u_exact(tt):
        return A * torch.sin(kx * x + phi) * math.exp(-nu * kx**2 * tt)

    def ux_exact(tt):
        return A * kx * torch.cos(kx * x + phi) * math.exp(-nu * kx**2 * tt)

    def nonlinear(ti, v):
        u = torch.fft.ifft(v, dim=-1).real.to(v.dtype)
        advection = -0.5j * k * torch.fft.fft(u * u, dim=-1) * dealias
        forcing = torch.fft.fft(
            (u_exact(float(ti)) * ux_exact(float(ti))).to(v.dtype), dim=-1
        )
        return advection + forcing

    linear = -nu * k.flatten() ** 2
    v0 = torch.fft.fft(u_exact(0.0) + 0j, dim=-1)
    t = torch.linspace(0, 1.0, 300, dtype=torch.float64)
    sol = etdrk4_solver(linear, nonlinear, v0, t)
    u_final = torch.fft.ifft(sol[-1], dim=-1).real
    exact = u_exact(1.0)
    relative_error = (u_final - exact).abs().max() / exact.abs().max()
    assert relative_error < FORCED_MMS_TOLERANCE


SCALAR_DECAY = -200.0
STAGE_TIME_ORDERS_ALL_AT_T_N = (1.22, 1.20, 1.12, 1.07, 1.04)
STAGE_TIME_ORDERS_COX_MATTHEWS = (3.54, 3.85, 3.96, 3.99, 4.00)


def _scalar_forced_error(steps: int) -> float:
    def exact(tt):
        return math.sin(3 * tt) + 2

    def forcing(tt):
        return 3 * math.cos(3 * tt) - SCALAR_DECAY * exact(tt)

    t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    linear = torch.tensor([SCALAR_DECAY], dtype=torch.float64)
    y0 = torch.tensor([[exact(0.0)]], dtype=torch.complex128)
    sol = etdrk4_solver(
        linear, lambda ti, v: torch.full_like(v, forcing(float(ti))), y0, t
    )
    return abs(sol[-1, 0, 0].real.item() - exact(1.0))


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_etdrk4_order_with_time_dependent_forcing():
    steps = (40, 80, 160, 320, 640, 1280)
    errors = [_scalar_forced_error(n) for n in steps]
    orders = [math.log2(a / b) for a, b in zip(errors, errors[1:], strict=False)]
    assert all(order > 3.9 for order in orders[-3:]), orders


def _closed_form_weights(z: torch.Tensor, h: float):
    ez = torch.exp(z)
    return (
        h * (torch.exp(z / 2) - 1) / z,
        h * (-4 - z + ez * (4 - 3 * z + z**2)) / z**3,
        h * (2 + z + ez * (-2 + z)) / z**3,
        h * (-4 - 3 * z - z**2 + ez * (4 - z)) / z**3,
    )


CLOSED_FORM_ATOL = 1e-10


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    h=st.floats(0.01, 0.5),
    frequencies=st.lists(
        st.floats(1.0, 200.0).flatmap(lambda size: st.sampled_from((size, -size))),
        min_size=1,
        max_size=6,
    ),
)
@example(h=0.1, frequencies=[-2.0, -20.0, -200.0, 5.0])
def test_etdrk4_complex_weights_match_closed_form(h, frequencies):
    linear = 1j * torch.tensor(frequencies, dtype=torch.float64).to(torch.complex128)
    contour = _etdrk4_weights(linear, h, contour_points=32)
    exact = _closed_form_weights(h * linear, h)
    for weight, reference in zip(contour, exact, strict=True):
        assert weight.is_complex()
        assert (weight - reference).abs().max() < CLOSED_FORM_ATOL


@pytest.mark.oracle_consistency
@SETTINGS
@given(
    h=st.floats(0.01, 0.5),
    nu=st.floats(0.001, 0.5),
    size=st.integers(4, 64),
    contour_points=st.integers(8, 64),
)
@example(h=0.1, nu=0.01, size=16, contour_points=32)
def test_etdrk4_real_weights_unchanged(h, nu, size, contour_points):
    k = torch.fft.fftfreq(size) * size
    linear = (-nu * k**2).to(torch.complex64).to(torch.complex128)
    assert not linear.imag.any()

    j = torch.arange(1, contour_points + 1, dtype=torch.float64)
    r = torch.exp(1j * torch.pi * (j - 0.5) / contour_points)
    lr = h * linear.reshape(-1).unsqueeze(-1) + r.unsqueeze(0)

    def half_circle_real(expr):
        return (h * expr.mean(dim=-1).real).reshape(linear.shape)

    expected = (
        half_circle_real((torch.exp(lr / 2) - 1) / lr),
        half_circle_real((-4 - lr + torch.exp(lr) * (4 - 3 * lr + lr**2)) / lr**3),
        half_circle_real((2 + lr + torch.exp(lr) * (-2 + lr)) / lr**3),
        half_circle_real((-4 - 3 * lr - lr**2 + torch.exp(lr) * (4 - lr)) / lr**3),
    )
    for weight, reference in zip(
        _etdrk4_weights(linear, h, contour_points), expected, strict=True
    ):
        assert torch.equal(weight, reference)


def an_uneven_grid(first, second):
    return torch.cat(
        [torch.linspace(0, 0.5, first), torch.linspace(0.6, 1.0, second)]
    ).double()


SOLVER_REFUSALS = (
    (
        "uniformly spaced",
        lambda first, second: (first, second) != (6, 5) and second >= 2,
        lambda first, second: etdrk4_solver(
            torch.tensor([-1.0], dtype=torch.float64),
            lambda ti, v: torch.zeros_like(v),
            torch.ones(1, 1, dtype=torch.complex128),
            an_uneven_grid(first, second),
        ),
    ),
    (
        "must match y0 mode grid",
        lambda first, second: first != second,
        lambda first, second: etdrk4_solver(
            torch.zeros(first),
            lambda ti, v: v,
            torch.zeros(2, second, dtype=torch.complex128),
            torch.linspace(0, 1, 5),
        ),
    ),
    (
        "more than one evaluation points",
        lambda first, second: True,
        lambda first, second: euler_solver(
            lambda ti, y: y, torch.zeros(first), torch.tensor([float(second)])
        ),
    ),
)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    refusal=st.sampled_from(SOLVER_REFUSALS),
    first=st.integers(2, 12),
    second=st.integers(0, 12),
)
@example(refusal=SOLVER_REFUSALS[0], first=6, second=3)
@example(refusal=SOLVER_REFUSALS[1], first=8, second=4)
@example(refusal=SOLVER_REFUSALS[2], first=2, second=0)
def test_a_solver_refuses_what_it_cannot_integrate(refusal, first, second):
    message, admits, call = refusal
    assume(admits(first, second))
    with pytest.raises(AssertionError, match=message):
        call(first, second)


# --------------------------------------------------------------------------- #
# as_semilinear_solver
# --------------------------------------------------------------------------- #
@pytest.mark.oracle_consistency
@SETTINGS
@given(
    N=st.integers(4, 32),
    decay=st.floats(0.1, 2.0),
    seed=st.integers(0, 10_000),
    complex_state=st.booleans(),
)
@example(N=8, decay=0.01, seed=0, complex_state=True)
def test_conversion_equals_the_combined_right_hand_side(N, decay, seed, complex_state):
    generator = torch.Generator().manual_seed(seed)
    linear = -decay * torch.arange(N, dtype=torch.float64)
    state_dtype = torch.complex128 if complex_state else torch.float64
    y0 = torch.randn(3, N, dtype=state_dtype, generator=generator)
    t = torch.linspace(0, 1.0, 11, dtype=torch.float64)

    def nonlinear(ti, v):
        return 0.1 * v / (1 + v.abs())

    converted = as_semilinear_solver(euler_solver)(linear, nonlinear, y0, t)
    assert converted.dtype == y0.dtype
    direct = euler_solver(lambda ti, v: linear * v + nonlinear(ti, v), y0, t)
    assert torch.equal(converted, direct)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    points=st.integers(2, 40),
    stop=st.floats(0.1, 10.0),
    width=st.integers(1, 6),
    rows=st.integers(1, 4),
)
@example(points=7, stop=2.0, width=3, rows=2)
def test_conversion_forwards_the_time_grid_unchanged(points, stop, width, rows):
    seen = {}

    def recording_solver(rhs, y0, t):
        seen["t"] = t
        seen["y0"] = y0
        return y0.unsqueeze(0).expand(len(t), *y0.shape)

    t = torch.linspace(0, stop, points, dtype=torch.float64)
    y0 = torch.ones(rows, width, dtype=torch.float64)
    as_semilinear_solver(recording_solver)(
        torch.zeros(width, dtype=torch.float64), lambda ti, v: v, y0, t
    )
    assert seen["t"] is t
    assert seen["y0"] is y0
