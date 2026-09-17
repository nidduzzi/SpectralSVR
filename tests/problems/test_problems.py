"""Fuzzed + MMS tests for the problem generators, including exact known
solutions for every path.

* Antiderivative: exact spectral derivative of known band-limited fields.
* Burgers MMS path: manufactured (u, f) with f = L(u); spectral residual == 0.
* Burgers numerical path: Cole-Hopf exact unforced solution reproduced by the
  ETDRK4 solver.
"""

import math
from unittest.mock import patch

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from SpectralSVR import Antiderivative, FourierBasis
from SpectralSVR.problems import Burgers
from SpectralSVR.basis import __base as base_module
from SpectralSVR.utils import (
    WavenumberCutoff,
    as_semilinear_solver,
    etdrk4_solver,
    implicit_adams_solver,
)

from _exact import (
    SETTINGS,
    burgers_periodic_manufactured,
    cole_hopf_params,
    cole_hopf_u,
    heat_mode_params,
    heat_mode_u_and_ux,
    space_grid,
)


def _time_dependent_field(vals, t_length, length):
    """Build a time-dependent FourierBasis from (n, nt, ns) sampled values."""
    n, nt, ns = vals.shape
    coeff = FourierBasis.transform((vals + 0j).reshape(n * nt, ns)).reshape(n, nt, ns)
    return FourierBasis(
        coeff, time_dependent=True, domain=((0.0, t_length), (0.0, length))
    )


def _residual_interior(problem, u, f, nu, res, ns, b):
    """residual() values with the time boundary trimmed (torch.gradient is one-
    sided at the evolution-axis ends), reshaped (n, nt, ns)."""
    r = problem.residual(u, f, nu, res=res)
    rv = r.inv_transform(r.coeff.flatten(0, 1)).real.reshape(u.coeff.shape[0], -1, ns)
    return rv[:, b:-b, :]


# =========================================================================== #
# Problem.axis_diff -- the finite-difference primitive, validated on its own
# against analytic derivatives (independent of any transform/residual).
# =========================================================================== #
@pytest.mark.oracle_analytic
@SETTINGS
@given(
    k=st.integers(1, 4),
    order=st.sampled_from([1, 2]),
    periodic=st.booleans(),
)
def test_axis_diff_matches_analytic(k, order, periodic):
    N, length = 128, 1.0  # fine enough that the O((k*dx)^2) stencil error < tol
    kx = 2 * math.pi * k
    if periodic:  # half-open grid, spacing L/N
        x = torch.linspace(0, length, N + 1, dtype=torch.float64)[:-1]
        spacing = length / N
    else:  # closed grid, spacing L/(N-1)
        x = torch.linspace(0, length, N, dtype=torch.float64)
        spacing = length / (N - 1)
    u = torch.sin(kx * x).view(1, -1)
    got = Burgers.axis_diff(u, spacing, dim=1, order=order, periodic=periodic)
    analytic = kx * torch.cos(kx * x) if order == 1 else -(kx**2) * torch.sin(kx * x)
    tol = 0.02 * float(analytic.abs().max().clamp(min=1.0))
    if periodic:  # wrap-around stencil -> exact at the ends too
        assert torch.allclose(got.view(-1), analytic, atol=tol)
    else:  # torch.gradient is one-sided at the ends -> compare the interior
        assert torch.allclose(got.view(-1)[2:-2], analytic[2:-2], atol=tol)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_axis_diff_rejects_bad_order():
    with pytest.raises(ValueError, match="order 1 or 2"):
        Burgers.axis_diff(torch.zeros(1, 4), 1.0, dim=1, order=3, periodic=True)


# =========================================================================== #
# Antiderivative
# =========================================================================== #
@pytest.mark.oracle_analytic
@SETTINGS
@given(
    n=st.integers(1, 4),
    m=st.integers(1, 6),
    modes=st.integers(16, 32),
    seed=st.integers(0, 10_000),
)
def test_antiderivative_exact_derivative_of_known_field(n, m, modes, seed):
    # Exact known field u = A sin(2*pi*m*x/L); its derivative is analytic.
    length = 1.0
    g = torch.Generator().manual_seed(seed)
    amp = (0.5 + torch.rand(n, generator=g)).view(n, 1)
    x = space_grid(modes, length)
    kx = 2 * math.pi * m / length
    u_val = amp * torch.sin(kx * x).view(1, -1)
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=(0.0, length))

    du_val = u.grad().inv_transform(u.grad().coeff).real
    analytic = amp * kx * torch.cos(kx * x).view(1, -1)
    assert torch.allclose(du_val, analytic, atol=1e-3)


@pytest.mark.oracle_consistency
@SETTINGS
@given(n=st.integers(1, 5), modes=st.integers(3, 16), seed=st.integers(0, 10_000))
def test_antiderivative_spectral_residual_is_zero(n, modes, seed):
    problem = Antiderivative()
    u, ut = problem.generate(
        FourierBasis, n, modes, generator=torch.Generator().manual_seed(seed), u0=0
    )
    residual = problem.spectral_residual(u, ut)
    assert residual.coeff.abs().max() < 1e-3


# --------------------------------------------------------------------------- #
# Residual-operator validation against INDEPENDENT exact fields.
#
# The MMS/generate tests above are self-referential for the operator: they feed
# a pair the operator itself built, so residual == 0 only proves determinism.
# These feed fields + forcings derived from ANALYTIC derivatives (never from the
# operator under test), so they pin down the operator's actual coefficients and
# signs -- and also assert the non-trivial output (residual of a non-solution
# equals the known analytic residual, so an always-zero operator would fail).
# --------------------------------------------------------------------------- #
@pytest.mark.oracle_analytic
@SETTINGS
@given(m=st.integers(1, 5), seed=st.integers(0, 10_000))
def test_antiderivative_spectral_residual_matches_analytic(m, seed):
    problem = Antiderivative()
    length, modes = 1.0, 48
    amp = float(0.5 + torch.rand(1, generator=torch.Generator().manual_seed(seed)))
    x = space_grid(modes, length)
    k = 2 * math.pi * m / length
    u_val = (amp * torch.sin(k * x)).view(1, -1)
    ut_val = (amp * k * torch.cos(k * x)).view(1, -1)  # analytic derivative
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=(0.0, length))
    ut = FourierBasis(FourierBasis.transform(ut_val + 0j), domain=(0.0, length))
    zero = FourierBasis(torch.zeros_like(u.coeff), domain=(0.0, length))

    # true pair (ut from analytic cos, not u.grad) -> residual vanishes
    r0 = problem.spectral_residual(u, ut)
    r0v = r0.inv_transform(r0.coeff).real
    assert torch.allclose(r0v, torch.zeros_like(r0v), atol=1e-4)
    # non-solution (ut = 0) -> residual equals the analytic derivative
    r = problem.spectral_residual(u, zero)
    assert torch.allclose(r.inv_transform(r.coeff).real, ut_val, atol=1e-4)


@pytest.mark.oracle_analytic
@SETTINGS
@given(m=st.integers(1, 3), seed=st.integers(0, 10_000))
def test_antiderivative_finite_diff_residual_matches_analytic(m, seed):
    # residual() differences a periodic axis with a wrap-around stencil (values
    # sit on the basis's half-open grid), so it is accurate over the FULL field.
    problem = Antiderivative()
    length, modes = 1.0, 96  # fine grid
    amp = float(0.5 + torch.rand(1, generator=torch.Generator().manual_seed(seed)))
    x = space_grid(modes, length)
    k = 2 * math.pi * m / length
    u_val = (amp * torch.sin(k * x)).view(1, -1)
    ut_val = (amp * k * torch.cos(k * x)).view(1, -1)
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=(0.0, length))
    ut = FourierBasis(FourierBasis.transform(ut_val + 0j), domain=(0.0, length))
    zero = FourierBasis(torch.zeros_like(u.coeff), domain=(0.0, length))

    tol = 0.05 * float(ut_val.abs().max())  # finite-difference accuracy
    # true pair (ut analytic) -> residual vanishes; a wrong derivative would not
    r0 = problem.residual(u, ut)
    r0v = r0.inv_transform(r0.coeff).real
    assert torch.allclose(r0v, torch.zeros_like(r0v), atol=tol)
    # non-solution -> residual equals the analytic derivative (non-trivial output)
    rn = problem.residual(u, zero)
    assert torch.allclose(rn.inv_transform(rn.coeff).real, ut_val, atol=tol)


@pytest.mark.oracle_analytic
@SETTINGS
@given(nu=st.floats(0.01, 0.2), seed=st.integers(0, 10_000))
def test_burgers_spectral_residual_matches_analytic(nu, seed):
    # space-time periodic manufactured field; forcing from analytic derivatives
    problem = Burgers()
    nt = ns = 16
    # both axes are Fourier -> half-open periodic grids
    x, t = space_grid(ns, 1.0), space_grid(nt, 1.0)
    u_val, f_val = burgers_periodic_manufactured(x, t, nu)
    domain = ((0.0, 1.0), (0.0, 1.0))
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=domain)
    f = FourierBasis(FourierBasis.transform(f_val + 0j), domain=domain)
    zero = FourierBasis(torch.zeros_like(u.coeff), domain=domain)

    # exact forcing -> residual vanishes to spectral accuracy
    r0 = problem.spectral_residual(u, f, nu)
    r0v = r0.inv_transform(r0.coeff).real
    assert torch.allclose(r0v, torch.zeros_like(r0v), atol=1e-4)
    # zero forcing -> residual equals the analytic Burgers operator applied to u
    r = problem.spectral_residual(u, zero, nu)
    assert torch.allclose(r.inv_transform(r.coeff).real, f_val, atol=1e-4)


@pytest.mark.oracle_analytic
@SETTINGS
@given(nu=st.floats(0.01, 0.2), seed=st.integers(0, 10_000))
def test_burgers_finite_diff_residual_matches_analytic(nu, seed):
    # periodic-aware finite-diff residual (both axes periodic here) -> accurate
    # over the FULL field, not just the interior.
    problem = Burgers()
    nt = ns = 16
    res = 96  # fine grid
    x, t = space_grid(ns, 1.0), space_grid(nt, 1.0)
    u_val, f_val = burgers_periodic_manufactured(x, t, nu)
    domain = ((0.0, 1.0), (0.0, 1.0))
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=domain)
    f = FourierBasis(FourierBasis.transform(f_val + 0j), domain=domain)
    zero = FourierBasis(torch.zeros_like(u.coeff), domain=domain)

    f_on_grid = f.get_values(res=res).real
    tol = 0.05 * float(f_on_grid.abs().max())  # finite-difference accuracy
    # exact forcing -> residual vanishes
    r0 = problem.residual(u, f, nu, res=res)
    r0v = r0.inv_transform(r0.coeff).real
    assert torch.allclose(r0v, torch.zeros_like(r0v), atol=tol)
    # zero forcing -> residual equals the analytic Burgers operator applied to u
    rn = problem.residual(u, zero, nu, res=res)
    assert torch.allclose(rn.inv_transform(rn.coeff).real, f_on_grid, atol=tol)


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    n=st.integers(1, 3),
    n_modes=st.integers(1, 3),
    nu=st.floats(0.05, 0.2),
    seed=st.integers(0, 10_000),
)
def test_burgers_finite_diff_residual_cole_hopf(n, n_modes, nu, seed):
    # Cole-Hopf: exact *unforced* Burgers solution (f = 0), independent of the
    # operator. Time-dependent field -> finite-diff residual, interior in time.
    problem = Burgers()
    length, ns, nt, tlen, b = 2 * math.pi, 48, 48, 0.5, 6
    m, a, bb = cole_hopf_params(n, n_modes, seed, max_m=3)
    x = space_grid(ns, length)
    t = torch.linspace(0, tlen, nt, dtype=torch.float64)
    u_tx = cole_hopf_u(x, t, m, a, bb, nu, length)  # (n, nt, ns)
    u = _time_dependent_field(u_tx, tlen, length)
    f = _time_dependent_field(torch.zeros_like(u_tx), tlen, length)
    r_int = _residual_interior(problem, u, f, nu, ns, ns, b)
    tol = 0.05 * float(u_tx.abs().max())
    assert torch.allclose(r_int, torch.zeros_like(r_int), atol=tol)


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    n=st.integers(1, 3),
    n_modes=st.integers(1, 3),
    nu=st.floats(0.05, 0.2),
    seed=st.integers(0, 10_000),
)
def test_burgers_finite_diff_residual_heat_mode(n, n_modes, nu, seed):
    # Heat-mode: u solves the heat equation exactly, so the Burgers forcing that
    # makes it exact is f = u u_x (independent, analytic). ns is fine because the
    # nonlinear term (0.5 u^2)_x has double the top wavenumber and is differenced
    # by a central stencil.
    problem = Burgers()
    length, ns, nt, tlen, b = 2 * math.pi, 128, 96, 0.5, 8
    m, amp, phi = heat_mode_params(n, n_modes, seed, max_m=3)
    x = space_grid(ns, length)
    t = torch.linspace(0, tlen, nt, dtype=torch.float64)
    u_tx, ux_tx = heat_mode_u_and_ux(x, t, m, amp, phi, nu, length)  # (n, nt, ns)
    u = _time_dependent_field(u_tx, tlen, length)
    f = _time_dependent_field(u_tx * ux_tx, tlen, length)  # Burgers forcing f = u u_x
    r_int = _residual_interior(problem, u, f, nu, ns, ns, b)
    tol = 0.1 * float(u_tx.abs().max())
    assert torch.allclose(r_int, torch.zeros_like(r_int), atol=tol)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_antiderivative_rejects_bad_sizes():
    problem = Antiderivative()
    with pytest.raises(AssertionError):
        problem.generate(FourierBasis, 0, 8)
    with pytest.raises(AssertionError):
        problem.generate(FourierBasis, 4, 0)


# =========================================================================== #
# Burgers -- MMS path (forcing chosen so u solves the equation exactly)
# =========================================================================== #
@pytest.mark.oracle_consistency
@SETTINGS
@given(
    n=st.integers(1, 3),
    space_modes=st.integers(8, 16),
    nu=st.floats(0.01, 0.2),
    t_start=st.floats(-3.0, 3.0),
    t_len=st.floats(0.5, 3.0),
    x_start=st.floats(-3.0, 3.0),
    x_len=st.floats(0.5, 3.0),
    seed=st.integers(0, 10_000),
)
def test_burgers_mms_solution_satisfies_equation(
    n, space_modes, nu, t_start, t_len, x_start, x_len, seed
):
    # Fuzzing the per-axis (start, length) -- starts span 0 -- exercises MMS
    # exactness on both the canonical [0, L) and shifted/negative-start domains.
    # grad works off axis lengths, so a shift keeps the residual at machine zero.
    problem = Burgers()
    modes = (space_modes, space_modes)
    domain = ((t_start, t_start + t_len), (x_start, x_start + x_len))
    u, f = problem.mms_solution(
        FourierBasis,
        n,
        modes,
        nu=nu,
        domain=domain,
        generator=torch.Generator().manual_seed(seed),
    )
    residual = problem.spectral_residual(u, f, nu)
    assert residual.coeff.abs().max() < 1e-4
    assert f.coeff.abs().max() > 0  # non-trivial forcing
    assert u.domain == domain


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    x_start=st.floats(-3.0, 3.0),
    x_len=st.floats(0.5, 3.0),
    t_start=st.floats(-3.0, 3.0),
    t_len=st.floats(0.5, 3.0),
    seed=st.integers(0, 10_000),
)
def test_burgers_generate_honors_domain_start(x_start, x_len, t_start, t_len, seed):
    # generate() must thread space_domain.start / time_domain.start into the
    # output field's domain, not just the lengths.
    problem = Burgers()
    u_gen, _ = problem.generate(
        FourierBasis,
        2,
        (8, 8),
        space_domain=slice(x_start, x_start + x_len, 64),
        time_domain=slice(t_start, t_start + t_len, 16),
        generator=torch.Generator().manual_seed(seed),
    )
    assert u_gen.domain[0] == (t_start, t_start + t_len)  # time axis start honored
    assert u_gen.domain[1] == (x_start, x_start + x_len)  # space axis start honored


# =========================================================================== #
# Burgers -- numerical path (ETDRK4) reproduces an exact Cole-Hopf solution
# =========================================================================== #
@pytest.mark.oracle_analytic
@SETTINGS
@given(
    n_modes=st.integers(1, 3),
    seed=st.integers(0, 10_000),
)
def test_burgers_numerical_matches_cole_hopf_exact(n_modes, seed):
    # Cole-Hopf gives an exact solution of the *unforced* viscous Burgers eqn.
    # Seed the solver with u(.,0) and compare the final state to the exact one.
    nu = 0.08
    length = 2 * math.pi
    ns = 128
    T = 1.0
    nt = 300
    n = 2
    m, a, b = cole_hopf_params(n, n_modes, seed, max_m=3)
    x = space_grid(ns, length)
    t0 = torch.zeros(1, dtype=torch.float64)
    u0 = cole_hopf_u(x, t0, m, a, b, nu, length)[:, 0]  # (n, ns)

    problem = Burgers()
    u_gen, _ = problem.generate(
        FourierBasis,
        n,
        (ns, ns),
        nu=nu,
        u0=u0,
        f=0,
        space_domain=slice(0, length, ns),
        time_domain=slice(0, T, nt),
    )
    final = u_gen.inv_transform(u_gen.coeff[:, -1]).real.cpu()
    exact = cole_hopf_u(
        x, torch.full((1,), T, dtype=torch.float64), m, a, b, nu, length
    )[:, 0]
    rel = (final - exact).abs().max() / exact.abs().max()
    assert rel < 5e-2


@pytest.mark.oracle_invariant
@pytest.mark.no_fuzz
def test_burgers_numerical_solution_dissipates_energy():
    # Viscous Burgers with zero forcing must not gain energy over time.
    problem = Burgers()
    u, _ = problem.generate(
        FourierBasis,
        2,
        (64, 64),
        generator=torch.Generator().manual_seed(0),
        nu=0.05,
        space_domain=slice(0, 1, 64),
        time_domain=slice(0, 1, 100),
    )
    vals = u.get_values(res=64).real
    energy = (vals**2).mean(dim=(0, 2))
    assert energy[-1] <= energy[0] + 1e-6


# --------------------------------------------------------------------------- #
# Additional coverage
# --------------------------------------------------------------------------- #
@pytest.mark.oracle_smoke
@SETTINGS
@given(
    n=st.integers(1, 3),
    space_modes=st.integers(8, 16),
    seed=st.integers(0, 10_000),
)
def test_burgers_manufactured_branch_shapes(n, space_modes, seed):
    # u0="random", f="random" -> spectral manufactured-solution branch
    u, f = Burgers().generate(
        FourierBasis,
        n,
        (space_modes, space_modes),
        u0="random",
        f="random",
        nu=0.05,
        generator=torch.Generator().manual_seed(seed),
    )
    assert u.time_dependent and f.time_dependent
    assert u.coeff.shape[0] == n and not torch.isnan(u.coeff).any()


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_manufactured_branch_rejects_a_solver():
    with pytest.raises(ValueError, match="integrates nothing"):
        Burgers().generate(
            FourierBasis,
            1,
            (8, 8),
            u0="random",
            f="random",
            nu=0.05,
            generator=torch.Generator().manual_seed(0),
            solver=etdrk4_solver,
        )


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_numerical_branch_uses_the_supplied_solver():
    def build(**kwargs):
        u, _ = Burgers().generate(
            FourierBasis,
            2,
            (16, 16),
            u0="random",
            f=0,
            nu=0.01,
            time_domain=slice(0, 1, 40),
            generator=torch.Generator().manual_seed(0),
            **kwargs,
        )
        return u.coeff

    assert torch.equal(build(), build(solver=etdrk4_solver))

    boom = RuntimeError("solver was called")

    def failing_solver(linear, nonlinear, y0, t):
        raise boom

    with pytest.raises(RuntimeError, match="solver was called"):
        build(solver=failing_solver)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    u0=st.floats(-1.0, 1.0),
    fval=st.floats(0.05, 0.5),
    space_modes=st.integers(8, 24),
    seed=st.integers(0, 10_000),
)
def test_burgers_numerical_constant_ic_and_forcing(u0, fval, space_modes, seed):
    # constant initial condition + constant forcing exercise the Number branches
    u, f = Burgers().generate(
        FourierBasis,
        2,
        (space_modes, space_modes),
        u0=u0,
        f=fval,
        nu=0.05,
        space_domain=slice(0, 1, space_modes),
        time_domain=slice(0, 1, 40),
        generator=torch.Generator().manual_seed(seed),
    )
    assert u.time_dependent
    assert f.coeff.abs().max() > 0  # non-zero constant forcing
    assert not torch.isnan(u.coeff).any()


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    n=st.integers(1, 3),
    space_modes=st.integers(8, 24),
    nu=st.floats(0.01, 0.2),
    seed=st.integers(0, 10_000),
)
def test_burgers_numeric_residual_runs(n, space_modes, nu, seed):
    problem = Burgers()
    u, f = problem.generate(
        FourierBasis,
        n,
        (space_modes, space_modes),
        nu=nu,
        u0="random",
        f="random",
        generator=torch.Generator().manual_seed(seed),
    )
    res = problem.residual(u, f, nu, res=64)
    assert res.time_dependent
    assert not torch.isnan(res.coeff).any()


@pytest.mark.oracle_analytic
@SETTINGS
@given(n=st.integers(1, 3), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_antiderivative_complex_integration_constant(n, modes, seed):
    # complex u0 exercises the complex integration-constant branch
    u, ut = Antiderivative().generate(
        FourierBasis,
        n,
        modes,
        generator=torch.Generator().manual_seed(seed),
        u0=1 + 2j,
    )
    assert u.coeff.shape[0] == n
    assert torch.allclose(u.coeff[:, 0], torch.tensor(1 + 2j, dtype=u.coeff.dtype))


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(1, 4), modes=st.integers(4, 16), seed=st.integers(0, 10_000))
def test_antiderivative_numeric_residual_runs(n, modes, seed):
    problem = Antiderivative()
    u, ut = problem.generate(
        FourierBasis, n, modes, generator=torch.Generator().manual_seed(seed), u0=0
    )
    res = problem.residual(u, ut)
    assert not torch.isnan(res.coeff).any()


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    u0=st.floats(-3.0, 3.0),
    n=st.integers(1, 3),
    modes=st.integers(4, 12),
    seed=st.integers(0, 10_000),
)
def test_antiderivative_float_integration_constant(u0, n, modes, seed):
    u, ut = Antiderivative().generate(
        FourierBasis, n, modes, generator=torch.Generator().manual_seed(seed), u0=u0
    )
    assert torch.allclose(
        u.coeff[:, 0], torch.tensor(u0 + 0j, dtype=u.coeff.dtype), atol=1e-5
    )


@pytest.mark.oracle_smoke
@SETTINGS
@given(m=st.integers(8, 16), seed=st.integers(0, 10_000))
def test_burgers_int_modes_and_1d_ic(m, seed):
    # int modes -> (m, m); a 1D u0 tensor is broadcast across samples
    ic = torch.rand(m, dtype=torch.float64)
    u, _ = Burgers().generate(
        FourierBasis,
        2,
        m,
        u0=ic,
        f=0,
        nu=0.05,
        space_domain=slice(0, 1, m),
        time_domain=slice(0, 1, 40),
        generator=torch.Generator().manual_seed(seed),
    )
    assert u.coeff.shape[0] == 2 and u.coeff.shape[2] == m


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    n=st.integers(1, 3),
    m=st.integers(6, 12),
    nu=st.floats(0.01, 0.2),
    seed=st.integers(0, 10_000),
)
def test_burgers_residual_non_time_dependent(n, m, nu, seed):
    # residual() on a non-time-dependent 2D field exercises the else branch
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.transform(torch.randn(n, m, m, generator=g) + 0j)
    u = FourierBasis(coeff, domain=((0.0, 1.0), (0.0, 1.0)))
    f = FourierBasis(torch.zeros_like(coeff), domain=((0.0, 1.0), (0.0, 1.0)))
    res = Burgers().residual(u, f, nu, res=m)
    assert not res.time_dependent
    assert not torch.isnan(res.coeff).any()


@pytest.mark.oracle_analytic
@SETTINGS
@given(n=st.integers(1, 3), modes=st.integers(4, 10), seed=st.integers(0, 10_000))
def test_antiderivative_tensor_integration_constant(n, modes, seed):
    # a tensor u0 hits the final else branch that assigns it directly
    g = torch.Generator().manual_seed(seed)
    u0 = torch.randn(n, dtype=torch.complex64, generator=g)
    u, _ = Antiderivative().generate(FourierBasis, n, modes, generator=g, u0=u0)
    assert torch.allclose(u.coeff[:, 0], u0.to(u.coeff.dtype), atol=1e-5)


@pytest.mark.no_fuzz
@pytest.mark.oracle_independent
def test_numerical_burgers_agrees_with_an_independent_integrator():
    """ETDRK4 against torchdiffeq's implicit Adams, in a regime both resolve.

    The two schemes share no code: ETDRK4 is implemented in this repo, implicit
    Adams comes from torchdiffeq. Agreement is only expected while the step size
    resolves the stiff eigenvalue, i.e. while ``h * nu * k_max**2`` is of order
    one or below. At the literal parameters below that quantity is
    ``(1 / 400) * 0.01 * (2 * pi * 4) ** 2``, about 0.016. Both schemes are
    fourth order in time, so their difference is O(h^4); the bound is set from
    that, not from the observed value, and no stiffer regime is claimed.
    """
    nu = 0.01
    space_modes = 8
    n_steps = 401

    def run(solver):
        u, _ = Burgers().generate(
            FourierBasis,
            2,
            (space_modes, space_modes),
            u0="random",
            f=0,
            nu=nu,
            time_domain=slice(0, 1, n_steps),
            generator=torch.Generator().manual_seed(7),
            solver=solver,
        )
        return u.coeff.cpu()

    exponential = run(etdrk4_solver)
    adams = run(as_semilinear_solver(implicit_adams_solver))

    assert not torch.equal(exponential, adams), (
        "identical output means the solver argument was ignored, not that the "
        "two integrators agree"
    )
    relative = (exponential - adams).abs().max() / exponential.abs().max()
    assert relative < 1e-4


# --------------------------------------------------------------------------- #
# The manufactured pair as a continuous solution
# --------------------------------------------------------------------------- #
MANUFACTURED_CONTINUOUS_RESIDUAL = {8: 2.565e-07, 16: 1.999e-07, 32: 2.954e-07}
MANUFACTURED_RESIDUAL_TOLERANCE = 5e-6
MANUFACTURED_RESIDUAL_CORRUPTED = 1.0e-2
MANUFACTURED_IMAGINARY_FLOOR = 1e-6
ANALYTIC_CONTROL_RESIDUAL = {12: 4.164e-08, 16: 3.887e-08, 32: 3.887e-08}

DOMAIN_2D = ((0.0, 1.0), (0.0, 1.0))


def _offset_grid(nq):
    """Query points strictly between the sample points of any grid dividing nq."""
    step = 1.0 / nq
    axis = torch.arange(nq, dtype=torch.float64) * step + 0.5 * step
    t, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([t.flatten(), x.flatten()], dim=-1), nq


def analytic_periodic_pair(n, nu):
    """Doubly-periodic pair whose forcing comes from analytic derivatives.

    Correct by construction and independent of the generator under test, so a
    clean reading here shows the measurement is not simply always-red.
    """
    x, t = space_grid(n, 1.0), space_grid(n, 1.0)
    u_val, f_val = burgers_periodic_manufactured(x, t, nu)
    return (
        FourierBasis(FourierBasis.transform(u_val + 0j), domain=DOMAIN_2D),
        FourierBasis(FourierBasis.transform(f_val + 0j), domain=DOMAIN_2D),
    )


def continuous_burgers_residual(u, f, nu, nq=96):
    """Relative residual of u_t + u u_x - nu u_xx - f, evaluated off the grid.

    Every term comes from the field's own exact spectral derivatives, so the
    oracle is the equation itself rather than any code that produced the pair.
    """
    query, n = _offset_grid(nq)

    def at(field):
        return field(query).real.to(torch.float64).reshape(-1, n, n)

    lhs = (
        at(u.grad(dim=0, ord=1))
        + at(u) * at(u.grad(dim=1, ord=1))
        - nu * at(u.grad(dim=1, ord=2))
    )
    return float((lhs - at(f)).abs().max() / lhs.abs().max())


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_manufactured_pair_satisfies_the_continuous_equation():
    nu = 0.05
    for modes, recorded in ANALYTIC_CONTROL_RESIDUAL.items():
        u, f = analytic_periodic_pair(modes, nu)
        measured = continuous_burgers_residual(u, f, nu)
        assert measured < MANUFACTURED_RESIDUAL_TOLERANCE, (
            f"positive control modes={modes}: {measured:.3e}, recorded {recorded:.3e}"
        )

    for modes, recorded in MANUFACTURED_CONTINUOUS_RESIDUAL.items():
        u, f = Burgers().mms_solution(
            FourierBasis,
            1,
            (modes, modes),
            nu,
            DOMAIN_2D,
            generator=torch.Generator().manual_seed(3),
        )
        measured = continuous_burgers_residual(u, f, nu)
        assert measured < MANUFACTURED_RESIDUAL_TOLERANCE, (
            f"modes={modes}: {measured:.3e}, recorded {recorded:.3e}"
        )


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_continuous_residual_detects_a_corrupted_forcing():
    nu = 0.05
    u, f = Burgers().mms_solution(
        FourierBasis,
        1,
        (16, 16),
        nu,
        DOMAIN_2D,
        generator=torch.Generator().manual_seed(3),
    )
    clean = continuous_burgers_residual(u, f, nu)
    corrupted = continuous_burgers_residual(
        u, FourierBasis(coeff=f.coeff * 1.01, domain=DOMAIN_2D), nu
    )
    assert clean < MANUFACTURED_RESIDUAL_TOLERANCE
    assert corrupted > 1e-3, (
        f"a one percent error must be visible: clean {clean:.3e}, "
        f"corrupted {corrupted:.3e}"
    )
    assert corrupted / clean > 1e3


# ---------------------------------------------------------------------------
# Burgers -- the manufactured band (manufactured-band-derivation)


def _pad_without_boundary_repair(coeff, target_modes, rescale=True):
    """``resize_modes`` with the even-source boundary repair cut out.

    Stands in for ``resize_modes`` everywhere ``mms_solution`` reaches it,
    including the no-op resizes ``Basis``'s arithmetic performs at
    ``rescale=True``, so it takes the same arguments and rescales the same way.
    Only the boundary repair differs.

    That repair is what makes a padded even-sized band conjugate-symmetric
    downstream. Removing it is what distinguishes a field generated symmetric
    from one corrected afterwards.
    """
    padded = coeff
    for dim, target in enumerate(target_modes, 1):
        current = padded.shape[dim]
        if target <= current:
            continue
        low = torch.arange((current - 1) // 2 + 1)
        high = (
            torch.arange(current - current // 2, current)
            if current > 1
            else torch.arange(0)
        )
        shape = list(padded.shape)
        shape[dim] = target - current
        padded = torch.concat(
            (
                padded.index_select(dim, low),
                torch.zeros(shape).to(padded),
                padded.index_select(dim, high),
            ),
            dim,
        )
    if rescale:
        source = torch.tensor(list(coeff.shape[1:])).prod()
        padded = padded * torch.tensor(target_modes).prod() / source
    return padded


def _hermitian_error(coeff):
    """``max |c[k] - conj(c[-k])|`` over every mode axis."""
    axes = tuple(range(1, coeff.ndim))
    mirrored = torch.flip(
        torch.roll(coeff, shifts=(-1,) * len(axes), dims=axes), dims=axes
    )
    return float((coeff - mirrored.conj()).abs().max())


def _worst_outside_band(coeff, cutoff):
    """Largest coefficient at ``|k| > cutoff`` on any mode axis."""
    axes = torch.meshgrid(
        *(FourierBasis.wave_number(m).flatten().abs() for m in coeff.shape[1:]),
        indexing="ij",
    )
    keep = torch.stack([k <= cutoff for k in axes]).all(dim=0)
    outside = coeff[:, ~keep]
    return float(outside.abs().max()) if outside.numel() else 0.0


def _manufactured_without_repair(modes, nu, seed, band=None):
    """``mms_solution``'s own output with the resize boundary repair disabled.

    Patched as a context manager rather than through the ``monkeypatch``
    fixture, which Hypothesis refuses to reset between generated inputs.
    """
    with patch.object(base_module, "resize_modes", _pad_without_boundary_repair):
        return Burgers().mms_solution(
            FourierBasis,
            2,
            (modes, modes),
            nu,
            DOMAIN_2D,
            generator=torch.Generator().manual_seed(seed),
            band=band,
        )


@pytest.mark.oracle_invariant
@pytest.mark.scenario("The generated field is conjugate-symmetric at every mode count")
@pytest.mark.scenario("The band cannot be stated in a form that breaks symmetry")
@given(modes=st.integers(min_value=6, max_value=40), seed=st.integers(0, 2**32 - 1))
@SETTINGS
def test_manufactured_field_is_conjugate_symmetric_as_generated(modes, seed):
    kept = Burgers.widest_manufactured_band(modes).mode_count(modes)
    assert kept % 2 == 1, f"kept set {kept} is even, so a mode has no partner"

    u, _ = _manufactured_without_repair(modes, 0.05, seed)
    generated = _hermitian_error(u.coeff)
    assert generated == 0.0, (
        f"modes={modes}: mms_solution's field is symmetric only after repair, "
        f"error {generated:.3e}"
    )
    values = FourierBasis.inv_transform(u.coeff)
    imaginary = float(values.imag.abs().max() / values.real.abs().max())
    assert imaginary < MANUFACTURED_IMAGINARY_FLOOR, (
        f"modes={modes}: the field describes a complex function, imaginary part "
        f"{imaginary:.3e} of the real part"
    )

    if kept + 1 <= modes:
        unpaired = FourierBasis.generate_coeff(
            2, (kept + 1, kept + 1), generator=torch.Generator().manual_seed(seed)
        )
        control = _hermitian_error(
            _pad_without_boundary_repair(unpaired, (modes, modes))
        )
        assert control > 0.0, (
            f"modes={modes}: an even band read zero too, so the measurement is "
            "blind rather than the field being symmetric"
        )


@pytest.mark.oracle_analytic
@pytest.mark.scenario("The bandwidth the caller can state is visible")
@given(
    modes=st.integers(min_value=6, max_value=40),
    seed=st.integers(0, 2**32 - 1),
    narrower=st.integers(min_value=0, max_value=8),
)
@SETTINGS
def test_a_stated_band_is_the_field_support(modes, seed, narrower):
    widest = Burgers.widest_manufactured_band(modes)
    cutoff = WavenumberCutoff(max(0, widest.cutoff - narrower))

    def support(band):
        u, _ = Burgers().mms_solution(
            FourierBasis,
            2,
            (modes, modes),
            0.05,
            DOMAIN_2D,
            generator=torch.Generator().manual_seed(seed),
            band=band,
        )
        return u.coeff

    stated = support(cutoff)
    outside = _worst_outside_band(stated, cutoff.cutoff)
    assert outside == 0.0, (
        f"modes={modes}, |k|<={cutoff.cutoff}: {outside:.3e} outside the stated band"
    )
    assert int(stated[0].abs().sum(dim=0).ne(0).sum()) == cutoff.mode_count(modes)

    if cutoff != widest:
        assert not torch.equal(stated, support(None)), (
            f"modes={modes}: a stated band narrower than the default changed nothing"
        )


@pytest.mark.oracle_analytic
@pytest.mark.scenario("Self-consistency is not accepted in place of the guarantee")
@pytest.mark.scenario("Bandwidth that overruns the grid is rejected or resolved")
@given(
    modes=st.integers(min_value=12, max_value=40),
    seed=st.integers(0, 2**32 - 1),
    over_by=st.integers(min_value=1, max_value=6),
)
@SETTINGS
def test_a_band_past_the_threshold_is_refused_though_the_pair_is_self_consistent(
    modes, seed, over_by
):
    nu = 0.05
    widest = Burgers.widest_manufactured_band(modes)
    assert 2 * widest.cutoff <= modes / 3
    assert 2 * (widest.cutoff + 1) > modes / 3

    def manufacture(band):
        return Burgers().mms_solution(
            FourierBasis,
            1,
            (modes, modes),
            nu,
            DOMAIN_2D,
            generator=torch.Generator().manual_seed(seed),
            band=band,
        )

    u, f = manufacture(widest)
    admitted = continuous_burgers_residual(u, f, nu)
    assert admitted < MANUFACTURED_RESIDUAL_TOLERANCE, (
        f"modes={modes}: the widest admitted band leaves {admitted:.3e}, so the "
        "threshold is too wide rather than exact"
    )

    over = WavenumberCutoff(widest.cutoff + over_by)
    with pytest.raises(ValueError, match=r"past the|more than"):
        manufacture(over)

    if 2 * over.cutoff + 1 > modes:
        return
    coeff = FourierBasis.generate_coeff(
        1, (over.mode_count(modes),) * 2, generator=torch.Generator().manual_seed(seed)
    )
    u_over = FourierBasis(
        FourierBasis(coeff, domain=DOMAIN_2D)
        .resize_modes((modes, modes), rescale=False)
        .coeff,
        domain=DOMAIN_2D,
    )
    f_over = Burgers().spectral_residual(
        u_over,
        FourierBasis(FourierBasis.generate_empty(1, (modes, modes)), domain=DOMAIN_2D),
        nu,
    )
    refused_discrete = float(
        Burgers().spectral_residual(u_over, f_over, nu).coeff.abs().max()
    )
    refused_continuous = continuous_burgers_residual(u_over, f_over, nu)
    assert refused_discrete < 1e-10, (
        f"modes={modes}: the refused pair fails its own de-aliased equation at "
        f"{refused_discrete:.3e}, so this is not the self-consistency case"
    )
    assert refused_continuous > admitted * 100, (
        f"modes={modes}: the refused pair leaves {refused_continuous:.3e} against "
        f"{admitted:.3e} admitted, so the refusal has little to prevent"
    )
