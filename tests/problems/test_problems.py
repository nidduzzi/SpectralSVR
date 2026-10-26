"""Fuzzed + MMS tests for the problem generators, including exact known
solutions for every path.

* Antiderivative: exact spectral derivative of known band-limited fields.
* Burgers MMS path: manufactured (u, f) with f = L(u); spectral residual == 0.
* Burgers numerical path: Cole-Hopf exact unforced solution reproduced by the
  ETDRK4 solver.
"""

import inspect
import math
import warnings
from typing import NamedTuple
from unittest.mock import patch

import pytest
import torch
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from SpectralSVR import Antiderivative, FourierBasis, accelerator
from SpectralSVR.problems import BandLimited, Burgers, SampledForcing, SpectralForcing
from SpectralSVR.basis import __base as base_module
from SpectralSVR.utils import (
    WavenumberCutoff,
    as_semilinear_solver,
    etdrk4_solver,
    implicit_adams_solver,
)

from tests._devices import HOST
from tests._time_varying import grid_points, slice_times

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
    order=st.integers(-3, 5),
    periodic=st.booleans(),
)
@example(k=1, order=1, periodic=True)
@example(k=1, order=1, periodic=False)
@example(k=1, order=2, periodic=True)
@example(k=1, order=2, periodic=False)
@example(k=1, order=3, periodic=True)
def test_axis_diff_matches_analytic_and_refuses_other_orders(k, order, periodic):
    N, length = 128, 1.0  # fine enough that the O((k*dx)^2) stencil error < tol
    kx = 2 * math.pi * k
    if periodic:  # half-open grid, spacing L/N
        x = torch.linspace(0, length, N + 1, dtype=torch.float64)[:-1]
        spacing = length / N
    else:  # closed grid, spacing L/(N-1)
        x = torch.linspace(0, length, N, dtype=torch.float64)
        spacing = length / (N - 1)
    u = torch.sin(kx * x).view(1, -1)
    if order not in (1, 2):
        with pytest.raises(ValueError, match="order 1 or 2"):
            Burgers.axis_diff(u, spacing, dim=1, order=order, periodic=periodic)
        return
    got = Burgers.axis_diff(u, spacing, dim=1, order=order, periodic=periodic)
    analytic = kx * torch.cos(kx * x) if order == 1 else -(kx**2) * torch.sin(kx * x)
    tol = 0.02 * float(analytic.abs().max().clamp(min=1.0))
    if periodic:  # wrap-around stencil -> exact at the ends too
        assert torch.allclose(got.view(-1), analytic, atol=tol)
    else:  # torch.gradient is one-sided at the ends -> compare the interior
        assert torch.allclose(got.view(-1)[2:-2], analytic[2:-2], atol=tol)


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


ANTIDERIVATIVE_IMAGINARY_FLOOR = 1e-6


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
    values = u.get_values()
    imaginary = float(values.imag.abs().max() / values.real.abs().max())
    assert imaginary < ANTIDERIVATIVE_IMAGINARY_FLOOR


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


@pytest.mark.oracle_smoke
@SETTINGS
@given(refused=st.sampled_from(("samples", "modes")), size=st.integers(-8, 0))
@example(refused="samples", size=0)
@example(refused="modes", size=0)
def test_antiderivative_rejects_bad_sizes(refused, size):
    samples, modes = (size, 8) if refused == "samples" else (4, size)
    message = {
        "samples": "number of samples n must be more than 0",
        "modes": "number of modes m must be more than 0 at dim 0",
    }[refused]
    with pytest.raises(AssertionError, match=message):
        Antiderivative().generate(FourierBasis, samples, modes)


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
@SETTINGS
@given(
    nu=st.floats(0.01, 0.2),
    modes=st.integers(16, 64),
    seed=st.integers(0, 10_000),
)
@example(nu=0.05, modes=64, seed=0)
def test_burgers_numerical_solution_dissipates_energy(nu, modes, seed):
    # Viscous Burgers with zero forcing must not gain energy over time.
    problem = Burgers()
    u, _ = problem.generate(
        FourierBasis,
        2,
        (modes, modes),
        generator=torch.Generator().manual_seed(seed),
        nu=nu,
        space_domain=slice(0, 1, 64),
        time_domain=slice(0, 1, 100),
    )
    vals = u.get_values(res=64).real
    energy = (vals**2).mean(dim=(0, 2))
    assert energy[-1] <= energy[0] + 1e-6


BRANCH_ARGUMENTS = {
    "manufactured": dict(n=1, modes=(8, 8), u0="random", f="random", nu=0.05),
    "numerical": dict(
        n=2,
        modes=(16, 16),
        u0="random",
        f=0,
        nu=0.01,
        time_domain=slice(0, 1, 40),
    ),
}


GENERATORS = {
    "Burgers": (
        Burgers.generate,
        lambda **stated: Burgers().generate(
            FourierBasis, 1, (8, 8), u0="random", f="random", **stated
        ),
    ),
    "antiderivative": (
        Antiderivative.generate,
        lambda **stated: Antiderivative().generate(FourierBasis, 1, 8, **stated),
    ),
    "basis": (
        FourierBasis.generate,
        lambda **stated: FourierBasis.generate(1, 8, **stated),
    ),
}


@pytest.mark.oracle_smoke
@pytest.mark.scenario("A stated precision a generator does not take is refused")
@pytest.mark.scenario("Any argument a generator does not name is refused")
@pytest.mark.parametrize("generator", tuple(GENERATORS))
@SETTINGS
@given(
    keyword=st.one_of(
        st.sampled_from(("dtype", "coeff_dtype", "periods")),
        st.from_regex(r"[a-z_]{2,16}", fullmatch=True),
    ),
    value=st.sampled_from((torch.complex128, torch.float64, (1.0,), 0)),
)
@example(keyword="dtype", value=torch.complex128)
@example(keyword="coeff_dtype", value=torch.complex128)
@example(keyword="periods", value=(1.0,))
def test_a_generator_refuses_an_argument_it_does_not_take(generator, keyword, value):
    signature, call = GENERATORS[generator]
    assume(keyword not in inspect.signature(signature).parameters)
    with pytest.raises(TypeError, match=f"'{keyword}'"):
        call(**{keyword: value})


def failing_solver(linear, nonlinear, y0, t):
    raise RuntimeError("solver was called")


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.parametrize("solver", (None, etdrk4_solver, failing_solver))
@pytest.mark.parametrize("branch", tuple(BRANCH_ARGUMENTS))
def test_only_the_numerical_branch_takes_a_solver(branch, solver):
    def build(**supplied):
        u, _ = Burgers().generate(
            FourierBasis,
            generator=torch.Generator().manual_seed(0),
            **BRANCH_ARGUMENTS[branch],
            **supplied,
        )
        return u.coeff

    supplied = {} if solver is None else {"solver": solver}
    if branch == "manufactured" and solver is not None:
        with pytest.raises(ValueError, match="integrates nothing"):
            build(**supplied)
    elif solver is failing_solver:
        with pytest.raises(RuntimeError, match="solver was called"):
            build(**supplied)
    elif branch == "numerical":
        assert torch.equal(build(**supplied), build(solver=etdrk4_solver))
    else:
        assert not torch.isnan(build(**supplied)).any()


CONSTANT_FORCING_CLOSED_FORM_TOLERANCE = 3e-5


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    u0=st.floats(-1.0, 1.0),
    fval=st.floats(0.05, 0.5),
    space_modes=st.integers(8, 24),
    seed=st.integers(0, 10_000),
)
def test_burgers_numerical_constant_ic_and_forcing(u0, fval, space_modes, seed):
    steps = 40
    u, f = Burgers().generate(
        FourierBasis,
        2,
        (space_modes, space_modes),
        u0=u0,
        f=fval,
        nu=0.05,
        space_domain=slice(0, 1, space_modes),
        time_domain=slice(0, 1, steps),
        generator=torch.Generator().manual_seed(seed),
    )
    assert u.time_dependent
    assert f.coeff.abs().max() > 0
    assert not torch.isnan(u.coeff).any()

    t = FourierBasis.grid(slice(0, 1, steps)).flatten().to(u.coeff.device)
    values = FourierBasis.inv_transform(u.coeff.flatten(0, 1)).real.reshape(
        2, steps, space_modes
    )
    closed_form = (u0 + fval * t)[None, :, None].expand(2, steps, space_modes)
    assert (values - closed_form).abs().max() < CONSTANT_FORCING_CLOSED_FORM_TOLERANCE


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


STIFFNESS_CEILING = 0.03
UNCONVERGED_ADAMS = "Functional iteration did not converge"


@st.composite
def non_stiff_regimes(draw):
    nu = draw(st.floats(0.005, 0.02))
    space_modes = draw(st.integers(6, 12))
    fewest_steps = math.ceil(nu * (math.pi * space_modes) ** 2 / STIFFNESS_CEILING) + 1
    return (
        nu,
        space_modes,
        draw(st.integers(max(401, fewest_steps), max(801, fewest_steps))),
    )


@pytest.mark.oracle_independent
@SETTINGS
@given(regime=non_stiff_regimes(), seed=st.integers(0, 10_000))
@example(regime=(0.01, 8, 401), seed=7)
def test_numerical_burgers_agrees_with_an_independent_integrator(regime, seed):
    """ETDRK4 against torchdiffeq's implicit Adams, in a regime both resolve.

    The two schemes share no code: ETDRK4 is implemented in this repo, implicit
    Adams comes from torchdiffeq. Agreement is only expected while the step size
    resolves the stiff eigenvalue, i.e. while ``h * nu * k_max**2`` is of order
    one or below. At the literal parameters below that quantity is
    ``(1 / 400) * 0.01 * (2 * pi * 4) ** 2``, about 0.016. Both schemes are
    fourth order in time, so their difference is O(h^4); the bound is set from
    that, not from the observed value, and no stiffer regime is claimed.
    """
    nu, space_modes, n_steps = regime

    def run(solver):
        u, _ = Burgers().generate(
            FourierBasis,
            2,
            (space_modes, space_modes),
            u0="random",
            f=0,
            nu=nu,
            time_domain=slice(0, 1, n_steps),
            generator=torch.Generator().manual_seed(seed),
            solver=solver,
        )
        return u.coeff.cpu()

    exponential = run(etdrk4_solver)
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        adams = run(as_semilinear_solver(implicit_adams_solver))
    messages = [str(warning.message) for warning in raised]
    assume(not any(UNCONVERGED_ADAMS in message for message in messages))
    assert not messages

    assert not torch.equal(exponential, adams), (
        "identical output means the solver argument was ignored, not that the "
        "two integrators agree"
    )
    relative = (exponential - adams).abs().max() / exponential.abs().max()
    assert relative < 1e-4


# --------------------------------------------------------------------------- #
# The manufactured pair as a continuous solution
# --------------------------------------------------------------------------- #
MANUFACTURED_RESIDUAL_TOLERANCE = 5e-6
MANUFACTURED_RESIDUAL_CORRUPTED = 1.0e-2
MANUFACTURED_IMAGINARY_FLOOR = 1e-6

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


@pytest.mark.oracle_analytic
@settings(SETTINGS, max_examples=8)
@given(modes=st.integers(10, 40), nu=st.floats(0.01, 0.2))
@example(modes=12, nu=0.05)
@example(modes=16, nu=0.05)
@example(modes=32, nu=0.05)
def test_an_analytic_pair_satisfies_the_continuous_equation(modes, nu):
    u, f = analytic_periodic_pair(modes, nu)
    measured = continuous_burgers_residual(u, f, nu)
    assert measured < MANUFACTURED_RESIDUAL_TOLERANCE, (
        f"positive control modes={modes}: {measured:.3e}"
    )


@pytest.mark.oracle_analytic
@settings(SETTINGS, max_examples=8)
@given(
    modes=st.integers(6, 40),
    nu=st.floats(0.01, 0.2),
    seed=st.integers(0, 10_000),
)
@example(modes=8, nu=0.05, seed=3)
@example(modes=16, nu=0.05, seed=3)
@example(modes=32, nu=0.05, seed=3)
def test_manufactured_pair_satisfies_the_continuous_equation(modes, nu, seed):
    u, f = Burgers().mms_solution(
        FourierBasis,
        1,
        (modes, modes),
        nu,
        DOMAIN_2D,
        generator=torch.Generator().manual_seed(seed),
    )
    measured = continuous_burgers_residual(u, f, nu)
    assert measured < MANUFACTURED_RESIDUAL_TOLERANCE, f"modes={modes}: {measured:.3e}"


@pytest.mark.oracle_analytic
@settings(SETTINGS, max_examples=8)
@given(
    modes=st.integers(6, 40),
    nu=st.floats(0.01, 0.2),
    seed=st.integers(0, 10_000),
)
@example(modes=16, nu=0.05, seed=3)
def test_continuous_residual_detects_a_corrupted_forcing(modes, nu, seed):
    u, f = Burgers().mms_solution(
        FourierBasis,
        1,
        (modes, modes),
        nu,
        DOMAIN_2D,
        generator=torch.Generator().manual_seed(seed),
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


SLICE_TIME_TOLERANCE = 1e-4
UNIT_START_AND_LENGTH = dict(
    time_start=0.0, time_length=1.0, space_start=0.0, space_length=1.0
)


def fourier_series_at(coeff, domain, times, space):
    (time_start, time_stop), (space_start, space_stop) = domain
    time_modes, space_modes = coeff.shape[-2:]
    time_wavenumbers = torch.fft.fftfreq(
        time_modes, 1 / time_modes, dtype=torch.float64
    )
    space_wavenumbers = torch.fft.fftfreq(
        space_modes, 1 / space_modes, dtype=torch.float64
    )
    time_phase = torch.outer(times - time_start, time_wavenumbers) / (
        time_stop - time_start
    )
    space_phase = torch.outer(space - space_start, space_wavenumbers) / (
        space_stop - space_start
    )
    series = torch.einsum(
        "nkl,tk,xl->ntx",
        coeff.to(torch.complex128),
        torch.exp(2j * torch.pi * time_phase),
        torch.exp(2j * torch.pi * space_phase),
    )
    return series / (time_modes * space_modes)


@pytest.mark.oracle_analytic
@pytest.mark.scenario("Each slice reads as the manufactured field at its time")
@SETTINGS
@given(
    n=st.integers(1, 3),
    slices=st.integers(2, 40),
    time_modes=st.integers(6, 16),
    space_modes=st.integers(6, 16),
    time_start=st.floats(-3.0, 3.0),
    time_length=st.floats(0.5, 3.0),
    space_start=st.floats(-3.0, 3.0),
    space_length=st.floats(0.5, 3.0),
    seed=st.integers(0, 10_000),
)
@example(n=1, slices=2, time_modes=8, space_modes=8, seed=0, **UNIT_START_AND_LENGTH)
@example(n=2, slices=8, time_modes=8, space_modes=8, seed=0, **UNIT_START_AND_LENGTH)
@example(n=2, slices=7, time_modes=7, space_modes=9, seed=0, **UNIT_START_AND_LENGTH)
@example(n=2, slices=5, time_modes=8, space_modes=8, seed=0, **UNIT_START_AND_LENGTH)
@example(n=2, slices=4, time_modes=9, space_modes=7, seed=0, **UNIT_START_AND_LENGTH)
@example(n=2, slices=11, time_modes=8, space_modes=8, seed=0, **UNIT_START_AND_LENGTH)
@example(n=3, slices=12, time_modes=9, space_modes=6, seed=0, **UNIT_START_AND_LENGTH)
@example(
    n=2,
    slices=9,
    time_modes=6,
    space_modes=13,
    seed=0,
    time_start=0.5,
    time_length=2.7,
    space_start=-1.0,
    space_length=3.0,
)
def test_each_manufactured_slice_holds_the_field_at_its_time(
    n,
    slices,
    time_modes,
    space_modes,
    time_start,
    time_length,
    space_start,
    space_length,
    seed,
):
    domain = (
        (time_start, time_start + time_length),
        (space_start, space_start + space_length),
    )

    def manufacture(time_dependent_coeff):
        return Burgers().generate(
            FourierBasis,
            n,
            (time_modes, space_modes),
            u0="random",
            f="random",
            nu=0.05,
            time_domain=slice(*domain[0], slices),
            space_domain=slice(*domain[1], space_modes),
            time_dependent_coeff=time_dependent_coeff,
            generator=torch.Generator().manual_seed(seed),
        )

    times = slice_times(slices, HOST, domain[0])
    space = grid_points(space_modes, HOST, domain[1])
    for stored, manufactured in zip(manufacture(True), manufacture(False), strict=True):
        assert stored.time_dependent
        assert stored.domain == domain
        assert stored.coeff.shape == (n, slices, space_modes)
        expected = fourier_series_at(
            manufactured.coeff, manufactured.domain, times, space
        )
        displacement = (stored.get_values() - expected).abs().max()
        relative = float(displacement / expected.abs().max())
        assert relative < SLICE_TIME_TOLERANCE, (
            f"slices={slices}, modes=({time_modes}, {space_modes}), "
            f"domain={domain}: {relative:.3e}"
        )


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


def _integrate_with_forcing(f, space_modes, nt=20, n=2, space_domain=slice(0, 1, 200)):
    return Burgers().generate(
        FourierBasis,
        n,
        (nt, space_modes),
        u0=0.0,
        f=f,
        nu=0.01,
        space_domain=space_domain,
        time_domain=slice(0, 1, nt),
        generator=torch.Generator().manual_seed(0),
    )


def _spectral_forcing(space_modes, time=(0.0, 1.0), space=(0.0, 1.0), seed=0):
    """A (time, space) field spectral in both axes, on the stated intervals."""
    return FourierBasis.generate(
        2,
        (space_modes, space_modes),
        domain=(time, space),
        generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.oracle_smoke
@pytest.mark.scenario("Sampled-time field raises")
@SETTINGS
@given(
    space_modes=st.integers(8, 24),
    nt=st.integers(6, 20),
    seed=st.integers(0, 10_000),
)
def test_numerical_forcing_rejects_a_per_time_slice_field(space_modes, nt, seed):
    values = torch.rand(
        2, nt, space_modes, generator=torch.Generator().manual_seed(seed)
    )
    sampled = _time_dependent_field(values, 1.0, 1.0)
    with pytest.raises(ValueError, match="per-time-slice"):
        _integrate_with_forcing(sampled, space_modes, nt)


@pytest.mark.oracle_smoke
@pytest.mark.scenario("Mismatched time domain raises")
@SETTINGS
@given(
    space_modes=st.integers(8, 24),
    offset=st.floats(0.1, 3.0),
    seed=st.integers(0, 10_000),
)
def test_numerical_forcing_rejects_a_mismatched_time_domain(space_modes, offset, seed):
    shifted = _spectral_forcing(space_modes, time=(offset, offset + 1.0), seed=seed)
    with pytest.raises(ValueError, match="forcing time domain"):
        _integrate_with_forcing(shifted, space_modes)


@pytest.mark.oracle_smoke
@pytest.mark.scenario("Mismatched space domain raises")
@SETTINGS
@given(
    space_modes=st.integers(8, 24),
    stretch=st.floats(0.1, 3.0),
    seed=st.integers(0, 10_000),
)
def test_numerical_forcing_rejects_a_mismatched_space_domain(
    space_modes, stretch, seed
):
    stretched = _spectral_forcing(space_modes, space=(0.0, 1.0 + stretch), seed=seed)
    with pytest.raises(ValueError, match="forcing space domain"):
        _integrate_with_forcing(stretched, space_modes)

    matching = _spectral_forcing(space_modes, seed=seed)
    u, _ = _integrate_with_forcing(matching, space_modes)
    assert not torch.isnan(u.coeff).any()


def _forcing_field(space_modes, nt, field_modes, n=2):
    time_nodes = FourierBasis.default_sampling().nodes(nt, 0.0, 1.0)
    space_nodes = FourierBasis.default_sampling().nodes(field_modes, 0.0, 1.0)
    values = _travelling_forcing(time_nodes[:, None], space_nodes[None, :]) + 0j
    coeff = FourierBasis.transform(values.reshape(1, nt, field_modes))
    return FourierBasis(
        coeff=coeff.expand(n, nt, field_modes).clone(),
        domain=((0.0, 1.0), (0.0, 1.0)),
    )


@pytest.mark.oracle_smoke
@SETTINGS
@given(space_modes=st.integers(8, 24), nt=st.integers(12, 24))
def test_numerical_forcing_rejects_a_bare_callable(space_modes, nt):
    nodes = FourierBasis.default_sampling().nodes(space_modes, 0.0, 1.0)

    def undeclared(t):
        return FourierBasis.transform(
            _travelling_forcing(torch.as_tensor(float(t)), nodes).expand(2, space_modes)
            + 0j
        )

    with pytest.raises(ValueError, match="SpectralForcing or SampledForcing"):
        _integrate_with_forcing(undeclared, space_modes, nt)


@pytest.mark.oracle_smoke
@SETTINGS
@given(space_modes=st.integers(8, 24), nt=st.integers(12, 24), flag=st.booleans())
def test_numerical_forcing_rejects_a_bool(space_modes, nt, flag):
    with pytest.raises(ValueError, match="bool is not a forcing"):
        _integrate_with_forcing(flag, space_modes, nt)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    step=st.integers(-10, 1),
    time_modes=st.integers(1, 24),
    space_modes=st.integers(8, 24),
)
@example(step=1, time_modes=1, space_modes=16)
def test_generate_rejects_a_time_domain_without_a_step(step, time_modes, space_modes):
    with pytest.raises(AssertionError, match="at least two grid points"):
        Burgers().generate(
            FourierBasis,
            2,
            (time_modes, space_modes),
            u0=0.0,
            f=0.0,
            nu=0.01,
            space_domain=slice(0, 1, 200),
            time_domain=slice(0, 1, step),
            generator=torch.Generator().manual_seed(0),
        )


@pytest.mark.oracle_invariant
@pytest.mark.scenario("A forcing the grid cannot represent is refused")
@SETTINGS
@given(
    space_modes=st.integers(8, 24),
    nt=st.integers(12, 24),
    wider=st.integers(1, 12),
)
def test_a_forcing_the_grid_cannot_represent_is_refused(space_modes, nt, wider):
    field_modes = space_modes + wider
    field = _forcing_field(space_modes, nt, field_modes)

    with pytest.raises(ValueError, match="fold its highest wavenumbers"):
        _integrate_with_forcing(field, space_modes, nt)

    _, by_hatch = _integrate_with_forcing(BandLimited(field), space_modes, nt)
    _, by_hand = _integrate_with_forcing(
        field.resize_modes((nt, space_modes)), space_modes, nt
    )
    assert torch.equal(by_hatch.coeff, by_hand.coeff)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    space_modes=st.integers(8, 24),
    nt=st.integers(12, 24),
    wrong=st.integers(1, 8),
)
def test_numerical_forcing_rejects_a_wrong_shaped_callable(space_modes, nt, wrong):
    def misshaped(t):
        return FourierBasis.transform(torch.zeros(2, space_modes + wrong) + 0j)

    with pytest.raises(ValueError, match="forcing callable must return"):
        _integrate_with_forcing(SpectralForcing(misshaped), space_modes, nt)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    space_modes=st.integers(8, 24),
    nt=st.integers(12, 24),
    seed=st.integers(0, 10_000),
)
def test_numerical_forcing_rejects_a_field_without_a_time_axis(space_modes, nt, seed):
    values = torch.randn(2, space_modes, generator=torch.Generator().manual_seed(seed))
    one_axis = FourierBasis(FourierBasis.transform(values + 0j), domain=(0.0, 1.0))
    with pytest.raises(ValueError, match="must span .time, space."):
        _integrate_with_forcing(one_axis, space_modes, nt)


@pytest.mark.oracle_smoke
@SETTINGS
@given(space_modes=st.integers(8, 24), seed=st.integers(0, 10_000))
def test_numerical_forcing_rejects_a_bare_tensor(space_modes, seed):
    values = torch.rand(2, space_modes, generator=torch.Generator().manual_seed(seed))
    with pytest.raises(NotImplementedError, match="bare tensor"):
        _integrate_with_forcing(values, space_modes)


@pytest.mark.oracle_smoke
@SETTINGS
@given(space_modes=st.integers(8, 24), u0=st.floats(-1.0, 1.0))
def test_numerical_branch_rejects_a_manufactured_forcing(space_modes, u0):
    with pytest.raises(ValueError, match="manufactures a forcing"):
        Burgers().generate(
            FourierBasis,
            2,
            (20, space_modes),
            u0=u0,
            f="random",
            nu=0.01,
            time_domain=slice(0, 1, 20),
            generator=torch.Generator().manual_seed(0),
        )


FORCING_ROUND_TRIP_TOLERANCE = 5e-6


def _travelling_forcing(t, x):
    return torch.sin(2 * math.pi * x) * torch.cos(2 * math.pi * t)


def _forcing_values_on_grid(f_gen, n, nt, ns):
    flat = FourierBasis.inv_transform(f_gen.coeff.flatten(0, 1))
    return flat.real.reshape(n, nt, ns)


def _analytic_on_grid(nt, ns, n):
    t = FourierBasis.grid(slice(0.0, 1.0, nt)).flatten()
    x = FourierBasis.default_sampling().nodes(ns, 0.0, 1.0).to(t.dtype)
    return _travelling_forcing(t[None, :, None], x[None, None, :]).expand(n, nt, ns)


@pytest.mark.oracle_analytic
@pytest.mark.scenario("Callable forcing is integrated")
@SETTINGS
@given(
    space_modes=st.integers(8, 32),
    nt=st.integers(12, 40),
    n=st.integers(1, 3),
    seed=st.integers(0, 10_000),
)
def test_callable_forcing_is_integrated_and_returned(space_modes, nt, n, seed):
    nodes = FourierBasis.default_sampling().nodes(space_modes, 0.0, 1.0)

    def coefficients_at(t):
        values = _travelling_forcing(torch.as_tensor(float(t)), nodes)
        return FourierBasis.transform(values.expand(n, space_modes) + 0j)

    def values_at(t):
        values = _travelling_forcing(torch.as_tensor(float(t)), nodes)
        return values.expand(n, space_modes)

    u, f = _integrate_with_forcing(
        SpectralForcing(coefficients_at), space_modes, nt, n=n
    )
    _, same = _integrate_with_forcing(SampledForcing(values_at), space_modes, nt, n=n)
    assert torch.allclose(f.coeff, same.coeff, atol=1e-6)

    assert u.time_dependent and f.time_dependent
    assert f.coeff.shape == (n, nt, space_modes)
    on_grid = _forcing_values_on_grid(f, n, nt, space_modes)
    want = _analytic_on_grid(nt, space_modes, n).to(on_grid.device)
    assert (on_grid - want).abs().max() < FORCING_ROUND_TRIP_TOLERANCE


@pytest.mark.oracle_analytic
@pytest.mark.scenario("Fully spectral forcing field is integrated")
@SETTINGS
@given(
    space_modes=st.integers(12, 32),
    nt=st.integers(12, 40),
    narrower=st.integers(0, 4),
    n=st.integers(1, 3),
)
def test_spectral_forcing_field_is_integrated_and_upsampled(
    space_modes, nt, narrower, n
):
    field_modes = max(8, space_modes - narrower)
    t_nodes = FourierBasis.default_sampling().nodes(nt, 0.0, 1.0)
    x_nodes = FourierBasis.default_sampling().nodes(field_modes, 0.0, 1.0)
    values = _travelling_forcing(t_nodes[:, None], x_nodes[None, :]) + 0j
    coeff = FourierBasis.transform(values.reshape(1, nt, field_modes))
    field = FourierBasis(
        coeff=coeff.expand(n, nt, field_modes).clone(),
        domain=((0.0, 1.0), (0.0, 1.0)),
    )

    _, f = _integrate_with_forcing(field, space_modes, nt, n=n)

    assert f.coeff.shape == (n, nt, space_modes)
    on_grid = _forcing_values_on_grid(f, n, nt, space_modes)
    want = _analytic_on_grid(nt, space_modes, n).to(on_grid.device)
    assert (on_grid - want).abs().max() < FORCING_ROUND_TRIP_TOLERANCE


@pytest.mark.oracle_analytic
@pytest.mark.scenario("Constant forcing byte-identity")
@SETTINGS
@given(
    space_modes=st.integers(8, 32),
    nt=st.integers(12, 40),
    n=st.integers(1, 3),
    value=st.floats(-2.0, 2.0),
)
def test_constant_forcing_is_the_exact_constant_spectrum(space_modes, nt, n, value):
    _, f = _integrate_with_forcing(value, space_modes, nt, n=n)

    expected = FourierBasis.transform(
        torch.full((n, space_modes), value, device=f.coeff.device) + 0j
    ).to(f.coeff.dtype)
    for step in range(nt):
        assert torch.equal(f.coeff[:, step, :], expected)


MANUFACTURED_ROUND_TRIP_MODES = 24
MANUFACTURED_ROUND_TRIP_STEPS = (13, 25, 49, 97)
MANUFACTURED_ROUND_TRIP_WORST_ERROR = 3.422348e-06
MANUFACTURED_ROUND_TRIP_TOLERANCE = 3e-5
MANUFACTURED_ROUND_TRIP_ORDERS = (3.867, 4.017, 4.150)
MANUFACTURED_ROUND_TRIP_ORDER_BOUND = 3.8
MANUFACTURED_ROUND_TRIP_TRAVEL = 1.1738
MANUFACTURED_ROUND_TRIP_TRAVEL_FLOOR = 1.0
COMPLEX64_ROUND_OFF_FLOOR = 1.5e-6
STANDING_STILL_WORST_ERROR = 1.173838e00
FROZEN_FORCING_ORDERS = (1.165, 1.012, 0.942)
FROZEN_FORCING_ORDER_CEILING = 2.0


class RoundTrip(NamedTuple):
    endpoint_error: float
    worst_error: float
    travel_from_initial: float


def _stand_still_solver(linear, nonlinear, y0, t):
    return y0.unsqueeze(0).expand(t.shape[0], *y0.shape).clone()


def _manufactured_round_trip(nt, freeze=False, solver=None, nu=0.01, n=2, seed=1234):
    """Integrate a manufactured pair over one period and measure the error.

    ``endpoint_error`` compares the final slice, ``worst_error`` every output
    time, and ``travel_from_initial`` is how far the manufactured solution moves
    from its own initial slice. All three are relative to the solution's scale.
    """
    modes = MANUFACTURED_ROUND_TRIP_MODES
    problem = Burgers()
    domain = ((0.0, 1.0), (0.0, 1.0))
    exact_field, forcing_field = problem.mms_solution(
        FourierBasis,
        n,
        (modes, modes),
        nu,
        domain,
        generator=torch.Generator().manual_seed(seed),
    )
    ns = 2 * modes
    nodes = FourierBasis.default_sampling().nodes(ns, 0.0, 1.0)

    def sample(field, time):
        query = torch.stack([torch.full_like(nodes, float(time)), nodes], dim=-1)
        return field(query)

    step = 1.0 / (nt - 1)

    def frozen_at_step_start(t):
        start = math.floor(float(t) / step + 1e-9) * step
        return FourierBasis.transform(sample(forcing_field, start))

    frozen_forcing = SpectralForcing(frozen_at_step_start)

    solution, _ = problem.generate(
        FourierBasis,
        n,
        (nt, ns),
        u0=sample(exact_field, 0.0).real,
        f=frozen_forcing if freeze else forcing_field,
        nu=nu,
        space_domain=slice(0.0, 1.0, 200),
        time_domain=slice(0.0, 1.0, nt),
        generator=torch.Generator().manual_seed(seed),
        **({} if solver is None else {"solver": solver}),
    )
    grid = FourierBasis.grid(slice(0.0, 1.0, nt)).flatten()
    numeric = FourierBasis.inv_transform(solution.coeff.flatten(0, 1)).reshape(
        n, nt, ns
    )
    exact = torch.stack([sample(exact_field, float(time)) for time in grid], dim=1).to(
        numeric.device
    )
    scale = exact.abs().max()
    initial = exact[:, 0, :].unsqueeze(1)
    return RoundTrip(
        endpoint_error=float((numeric[:, -1, :] - exact[:, -1, :]).abs().max() / scale),
        worst_error=float((numeric - exact).abs().max() / scale),
        travel_from_initial=float((exact - initial).abs().max() / scale),
    )


def _observed_orders(errors, steps):
    """Convergence order between consecutive runs, from the step sizes used.

    ``nt`` counts grid points, so the step is ``1 / (nt - 1)`` and consecutive
    counts need not halve it exactly; the ratio is taken from the steps rather
    than assumed to be two.
    """
    h = [1.0 / (nt - 1) for nt in steps]
    return [
        math.log(a / b) / math.log(ha / hb)
        for a, b, ha, hb in zip(errors, errors[1:], h, h[1:], strict=False)
    ]


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
@pytest.mark.scenario("Round-trip matches the manufactured solution")
@pytest.mark.scenario("Exact solution reached within time-stepping error")
def test_numerical_burgers_reproduces_manufactured_solution():
    result = _manufactured_round_trip(MANUFACTURED_ROUND_TRIP_STEPS[-1])
    assert result.travel_from_initial == pytest.approx(
        MANUFACTURED_ROUND_TRIP_TRAVEL, rel=0.05
    )
    assert result.travel_from_initial > MANUFACTURED_ROUND_TRIP_TRAVEL_FLOOR
    assert result.worst_error == pytest.approx(
        MANUFACTURED_ROUND_TRIP_WORST_ERROR, rel=0.2
    )
    assert result.worst_error < MANUFACTURED_ROUND_TRIP_TOLERANCE
    assert result.worst_error > COMPLEX64_ROUND_OFF_FLOOR


MANUFACTURED_TIME_PERIODICITY = 1e-5


@pytest.mark.oracle_invariant
@pytest.mark.scenario("The manufactured field returns to its initial slice")
@SETTINGS
@given(modes=st.integers(12, 40), seed=st.integers(0, 10_000))
def test_manufactured_field_returns_to_its_initial_slice(modes, seed):
    u, _ = Burgers().mms_solution(
        FourierBasis,
        2,
        (modes, modes),
        0.05,
        DOMAIN_2D,
        generator=torch.Generator().manual_seed(seed),
    )
    nodes = FourierBasis.default_sampling().nodes(modes, 0.0, 1.0)

    def slice_at(time):
        query = torch.stack([torch.full_like(nodes, time), nodes], dim=-1)
        return u(query)

    start, end = slice_at(0.0), slice_at(1.0)
    assert (end - start).abs().max() / start.abs().max() < MANUFACTURED_TIME_PERIODICITY


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
@pytest.mark.scenario("The round-trip comparison is not satisfied by standing still")
@pytest.mark.scenario("A full-period comparison states what the solver had to do")
def test_round_trip_is_not_satisfied_by_a_solver_that_stands_still():
    standing = _manufactured_round_trip(
        MANUFACTURED_ROUND_TRIP_STEPS[-1], solver=_stand_still_solver
    )
    integrating = _manufactured_round_trip(MANUFACTURED_ROUND_TRIP_STEPS[-1])

    assert standing.endpoint_error < MANUFACTURED_ROUND_TRIP_TOLERANCE
    assert standing.endpoint_error < integrating.endpoint_error

    assert standing.worst_error == pytest.approx(STANDING_STILL_WORST_ERROR, rel=0.2)
    assert standing.worst_error > MANUFACTURED_ROUND_TRIP_TOLERANCE
    assert integrating.worst_error < standing.worst_error / 1e4


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
@pytest.mark.scenario("Round-trip converges at the integrator's order")
@pytest.mark.scenario("Rate, not just size, is asserted")
def test_numerical_burgers_round_trip_converges_at_the_integrator_order():
    errors = [
        _manufactured_round_trip(nt).endpoint_error
        for nt in MANUFACTURED_ROUND_TRIP_STEPS
    ]
    assert min(errors) > COMPLEX64_ROUND_OFF_FLOOR, (
        f"finest error {min(errors):.6e} is at or beneath the complex64 floor "
        f"{COMPLEX64_ROUND_OFF_FLOOR:.6e}; the sweep is measuring round-off, not "
        "the method, so the orders below are saturation rather than a rate"
    )
    orders = _observed_orders(errors, MANUFACTURED_ROUND_TRIP_STEPS)
    assert orders == pytest.approx(list(MANUFACTURED_ROUND_TRIP_ORDERS), abs=0.15)
    assert min(orders) > MANUFACTURED_ROUND_TRIP_ORDER_BOUND, (
        f"measured orders {orders} against a bound of "
        f"{MANUFACTURED_ROUND_TRIP_ORDER_BOUND}; an order materially under 4 is "
        "a finding to chase, not a bound to lower"
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
@pytest.mark.scenario("Forcing is evaluated at stage times, not step starts")
def test_numerical_burgers_forcing_frozen_at_step_start_loses_the_order():
    frozen = [
        _manufactured_round_trip(nt, freeze=True).endpoint_error
        for nt in MANUFACTURED_ROUND_TRIP_STEPS
    ]
    orders = _observed_orders(frozen, MANUFACTURED_ROUND_TRIP_STEPS)
    assert orders == pytest.approx(list(FROZEN_FORCING_ORDERS), abs=0.15)
    assert (
        max(orders) < FROZEN_FORCING_ORDER_CEILING < min(MANUFACTURED_ROUND_TRIP_ORDERS)
    )
    assert frozen[-1] > MANUFACTURED_ROUND_TRIP_TOLERANCE


@pytest.mark.oracle_smoke
@pytest.mark.scenario("Generating a problem can be directed")
@pytest.mark.scenario("Parts of a workload are directed separately")
@SETTINGS
@given(
    space_modes=st.integers(8, 16),
    nt=st.integers(12, 20),
    seed=st.integers(0, 10_000),
)
def test_generating_a_problem_is_placed_where_it_is_told(space_modes, nt, seed):
    if not torch.accelerator.is_available():
        pytest.skip("directing placement elsewhere needs a second device to direct to")
    elsewhere = accelerator()

    manufactured, forcing = Burgers().generate(
        FourierBasis,
        2,
        (space_modes, space_modes),
        u0="random",
        f="random",
        nu=0.05,
        generator=torch.Generator(device=elsewhere).manual_seed(seed),
        device=elsewhere,
    )
    assert manufactured.coeff.device.type == elsewhere.type
    assert forcing.coeff.device.type == elsewhere.type

    integrated, _ = Burgers().generate(
        FourierBasis,
        2,
        (nt, space_modes),
        u0=0.0,
        f=0.0,
        nu=0.01,
        time_domain=slice(0, 1, nt),
        generator=torch.Generator(device=elsewhere).manual_seed(seed),
        device=elsewhere,
    )
    assert integrated.coeff.device.type == elsewhere.type

    stayed, _ = Burgers().generate(
        FourierBasis,
        2,
        (nt, space_modes),
        u0=0.0,
        f=0.0,
        nu=0.01,
        time_domain=slice(0, 1, nt),
        generator=torch.Generator().manual_seed(seed),
    )
    assert stayed.coeff.device == torch.get_default_device()


PLACEMENTS = {"cpu": lambda: torch.device("cpu"), "accelerator": accelerator}


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("A workload placed against the process default still runs")
@pytest.mark.parametrize("stated", PLACEMENTS)
@pytest.mark.parametrize("process_default", PLACEMENTS)
def test_generation_survives_a_process_default_pointing_elsewhere(
    process_default, stated
):
    if (
        "accelerator" in (process_default, stated)
        and not torch.accelerator.is_available()
    ):
        pytest.skip("an accelerator must be present to be named as a device")
    torch.set_default_device(PLACEMENTS[process_default]())
    placement = PLACEMENTS[stated]()
    generator = torch.Generator(device=placement).manual_seed(0)
    u, ut = Antiderivative().generate(
        FourierBasis, 8, 8, generator=generator, u0=0, device=placement
    )
    assert u.coeff.device.type == placement.type
    assert ut.coeff.device.type == placement.type
    assert u[:4].coeff.shape[0] == 4
    values = u.get_values(res=8, device=placement)[0]
    assert values.device.type == placement.type
