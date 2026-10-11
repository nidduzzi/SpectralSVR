"""Tests for FourierBasis and the Basis base-class machinery it implements.

Testing strategy (enforced by ``conftest.py``):

* Every test is either fuzzed with Hypothesis (``@given``) or explicitly
  opts out with ``@pytest.mark.no_fuzz``.
* Every test carries exactly one oracle marker
  (``@pytest.mark.oracle_{independent,analytic,invariant,consistency,smoke}``)
  classifying the kind of oracle it asserts against.

Correctness is additionally cross-checked against PyTorch's own FFT.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from SpectralSVR import (
    FourierBasis,
    accelerator,
    to_complex_coeff,
    to_real_coeff,
)
from SpectralSVR.basis import Basis
from SpectralSVR.basis.sampling import PeriodicUniform
from SpectralSVR.basis.strategy import EvaluationStrategy
from tests._anchors import fourier_series_at

CPU = torch.device("cpu")

# Hypothesis is slow with torch; drop the per-example deadline and allow the
# fixed-shape function-scoped helpers.
SETTINGS = settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# Hypothesis strategies
# --------------------------------------------------------------------------- #
_elements = st.floats(
    min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False
)


@st.composite
def spectra_1d(draw, min_n=2, max_n=32, max_rows=3):
    """A manufactured complex spectrum of shape (rows, n)."""
    n = draw(st.integers(min_n, max_n))
    rows = draw(st.integers(1, max_rows))
    re = draw(hnp.arrays(np.float64, (rows, n), elements=_elements))
    im = draw(hnp.arrays(np.float64, (rows, n), elements=_elements))
    return torch.from_numpy(re + 1j * im).to(torch.complex128)


@st.composite
def real_signals_1d(draw, min_n=2, max_n=32, max_rows=3):
    n = draw(st.integers(min_n, max_n))
    rows = draw(st.integers(1, max_rows))
    a = draw(hnp.arrays(np.float64, (rows, n), elements=_elements))
    return torch.from_numpy(a).to(torch.complex128)


@st.composite
def spectra_2d(draw, min_n=2, max_n=10, max_rows=2):
    r = draw(st.integers(min_n, max_n))
    c = draw(st.integers(min_n, max_n))
    rows = draw(st.integers(1, max_rows))
    re = draw(hnp.arrays(np.float64, (rows, r, c), elements=_elements))
    im = draw(hnp.arrays(np.float64, (rows, r, c), elements=_elements))
    return torch.from_numpy(re + 1j * im).to(torch.complex128)


# =========================================================================== #
# MMS tests: manufacture a solution from a known spectrum, recover the answer
# =========================================================================== #
@pytest.mark.oracle_independent
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_transform_recovers_spectrum(coeff):
    # Manufacture u from the known spectrum, then the forward transform must
    # return that exact spectrum.
    u = FourierBasis.inv_transform(coeff)
    recovered = FourierBasis.transform(u)
    assert torch.allclose(recovered, coeff, atol=1e-6)
    # torch.fft agrees on both legs of the manufacture/recover round trip.
    assert torch.allclose(u, torch.fft.ifft(coeff, dim=1), atol=1e-6)
    assert torch.allclose(recovered, torch.fft.fft(u, dim=1), atol=1e-6)


@pytest.mark.oracle_independent
@SETTINGS
@given(coeff=spectra_2d())
def test_mms_transform_recovers_spectrum_2d(coeff):
    u = FourierBasis.inv_transform(coeff)
    recovered = FourierBasis.transform(u)
    assert torch.allclose(recovered, coeff, atol=1e-5)
    # torch.fft 2D agrees on the manufacture and recovery.
    assert torch.allclose(u, torch.fft.ifft2(coeff, dim=(1, 2)), atol=1e-5)
    assert torch.allclose(recovered, torch.fft.fft2(u, dim=(1, 2)), atol=1e-5)


@pytest.mark.oracle_independent
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_grad_is_spectral_derivative(coeff):
    # Known spectrum => known derivative spectrum: c_k * 2*pi*i*k (period 1),
    # with the DC (k=0) mode dropped.
    n = coeff.shape[1]
    basis = FourierBasis(coeff)
    k = FourierBasis.wave_number(n).T.to(coeff)
    expected = coeff * 2j * torch.pi * k
    expected[:, 0] = 0
    grad_coeff = basis.grad().coeff
    assert torch.allclose(grad_coeff, expected, atol=1e-6)
    # torch.fft-based spectral derivative in value space agrees.
    k_torch = (torch.fft.fftfreq(n) * n).to(coeff)
    u = FourierBasis.inv_transform(coeff)
    dval_torch = torch.fft.ifft(
        torch.fft.fft(u, dim=1) * 2j * torch.pi * k_torch, dim=1
    )
    dval_ours = basis.grad().inv_transform(grad_coeff)
    assert torch.allclose(dval_ours, dval_torch, atol=1e-4, rtol=1e-6)


@pytest.mark.oracle_consistency
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_integral_inverts_grad(coeff):
    # grad then integral recovers the manufactured spectrum, minus its DC mode.
    coeff = coeff.clone()
    coeff[:, 0] = 0  # zero-mean manufactured solution
    basis = FourierBasis(coeff)
    recovered = basis.grad().integral().coeff
    assert torch.allclose(recovered, coeff, atol=1e-6)


@pytest.mark.oracle_analytic
@SETTINGS
@given(coeff=spectra_1d(max_n=12, max_rows=2))
def test_mms_evaluate_closed_form(coeff):
    # Point evaluation must equal the closed form (1/N) sum_k c_k exp(2pi i k x).
    n = coeff.shape[1]
    x = torch.tensor([[0.1], [0.37], [0.83]], dtype=torch.float64)
    out = FourierBasis.evaluate(coeff, x)
    k = FourierBasis.wave_number(n).flatten().to(coeff)
    closed = (coeff.unsqueeze(1) * torch.exp(2j * torch.pi * x * k)).sum(-1) / n
    assert torch.allclose(out, closed, atol=1e-6)


@pytest.mark.oracle_independent
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_evaluate_matches_inv_transform_on_grid(coeff):
    # Evaluating the manufactured function on the sample grid reproduces the
    # inverse transform used to manufacture it.
    n = coeff.shape[1]
    basis = FourierBasis(coeff)
    grid = torch.arange(0, 1, 1.0 / n, dtype=torch.float64)
    ev = basis(grid, device=CPU)
    iv = FourierBasis.inv_transform(coeff)
    assert torch.allclose(ev, iv, atol=1e-6)
    # both match torch.fft.ifft of the manufactured spectrum on the grid.
    assert torch.allclose(ev, torch.fft.ifft(coeff, dim=1), atol=1e-6)


@pytest.mark.oracle_analytic
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_parseval(coeff):
    # Parseval for this DFT convention: sum|u|^2 == (1/N) sum|c|^2.
    n = coeff.shape[1]
    u = FourierBasis.inv_transform(coeff)
    lhs = (u.abs() ** 2).sum()
    rhs = (coeff.abs() ** 2).sum() / n
    assert torch.isclose(lhs, rhs, atol=1e-5, rtol=1e-5)


@SETTINGS
@given(
    amp=st.floats(0.5, 5.0),
    freq=st.integers(1, 5),
    length=st.floats(0.5, 4.0),
    start=st.floats(-3.0, 3.0),
)
@pytest.mark.oracle_analytic
def test_mms_derivative_of_sine_with_period(amp, freq, length, start):
    # Manufactured u = A sin(2*pi*m*(x-a)/L) on the domain [a, a+L); its
    # derivative is A(2*pi*m/L) cos(2*pi*m*(x-a)/L). Fuzzing ``start`` (which
    # spans 0) exercises both the canonical [0, L) and shifted domains.
    n = 64
    stop = start + length
    domain = (start, stop)
    x = torch.arange(start, stop, length / n, dtype=torch.float64)[:n]
    u = (amp * torch.sin(2 * torch.pi * freq * (x - start) / length) + 0j).unsqueeze(0)
    coeff = FourierBasis.transform(u, domain=domain)
    basis = FourierBasis(coeff, domain=domain)
    g = basis.grad()
    dval = g.inv_transform(g.coeff, domain=domain).real.flatten()
    analytic = (
        amp
        * (2 * torch.pi * freq / length)
        * torch.cos(2 * torch.pi * freq * (x - start) / length)
    )
    assert torch.allclose(dval, analytic, atol=1e-2)


@pytest.mark.oracle_analytic
@SETTINGS
@given(amp=st.floats(0.5, 5.0), freq=st.integers(1, 6))
def test_mms_integral_of_cosine(amp, freq):
    # Manufactured u = A cos(2*pi*m*x); integral = A sin(2*pi*m*x)/(2*pi*m).
    n = 128
    x = torch.arange(0, 1, 1.0 / n, dtype=torch.float64)
    u = (amp * torch.cos(2 * torch.pi * freq * x) + 0j).unsqueeze(0)
    basis = FourierBasis(FourierBasis.transform(u))
    ig = basis.integral()
    val = ig.inv_transform(ig.coeff).real.flatten()
    analytic = amp * torch.sin(2 * torch.pi * freq * x) / (2 * torch.pi * freq)
    assert torch.allclose(val, analytic, atol=1e-2)


@pytest.mark.oracle_analytic
@SETTINGS
@given(amp=st.floats(-5.0, 5.0))
def test_mms_constant_is_dc_only(amp):
    # Manufactured constant A -> spectrum is A*N at DC, zero elsewhere.
    n = 16
    f = torch.full((1, n), amp, dtype=torch.float64) + 0j
    coeff = FourierBasis.transform(f)
    assert torch.isclose(
        coeff[0, 0].real, torch.tensor(amp * n, dtype=torch.float64), atol=1e-4
    )
    assert coeff[0, 1:].abs().max() < 1e-4


@pytest.mark.oracle_analytic
@SETTINGS
@given(freq=st.integers(1, 7), amp=st.floats(0.5, 4.0))
def test_mms_cosine_spectrum(freq, amp):
    # A cos(2*pi*m*x) -> real peaks A*N/2 at +m and -m (== N-m), zero elsewhere.
    n = 16
    assert 0 < freq < n // 2
    x = torch.arange(0, 1, 1.0 / n, dtype=torch.float64)
    coeff = FourierBasis.transform(
        (amp * torch.cos(2 * torch.pi * freq * x) + 0j).unsqueeze(0)
    )
    peak = torch.tensor(amp * n / 2, dtype=torch.float64)
    assert torch.isclose(coeff[0, freq].real, peak, atol=1e-3)
    assert torch.isclose(coeff[0, n - freq].real, peak, atol=1e-3)


# =========================================================================== #
# Fuzzed cross-checks against torch.fft and structural invariants (no MMS)
# =========================================================================== #
@pytest.mark.oracle_independent
@SETTINGS
@given(f=real_signals_1d())
def test_fuzz_transform_matches_torch_fft(f):
    assert torch.allclose(FourierBasis.transform(f), torch.fft.fft(f, dim=1), atol=1e-6)


@pytest.mark.oracle_independent
@SETTINGS
@given(coeff=spectra_1d())
def test_fuzz_inv_transform_matches_torch_ifft(coeff):
    assert torch.allclose(
        FourierBasis.inv_transform(coeff),
        torch.fft.ifft(coeff, dim=1),
        atol=1e-6,
    )


@pytest.mark.oracle_independent
@SETTINGS
@given(f=spectra_2d())
def test_fuzz_transform_matches_torch_fft2(f):
    assert torch.allclose(
        FourierBasis.transform(f),
        torch.fft.fft2(f, dim=(1, 2)),
        atol=1e-5,
    )


@pytest.mark.oracle_consistency
@SETTINGS
@given(f=real_signals_1d())
def test_fuzz_roundtrip_real(f):
    coeff = FourierBasis.transform(f)
    back = FourierBasis.inv_transform(coeff)
    assert torch.allclose(back, f, atol=1e-6)


@pytest.mark.oracle_consistency
@SETTINGS
@given(f=real_signals_1d())
def test_fuzz_transform_non_fft_path_matches_fft_path(f):
    fft_path = FourierBasis.transform(f)
    mm_path = FourierBasis.transform(f, sampling=PeriodicUniform(supports_fft=False))
    # The explicit matmul path accumulates more float error than the FFT.
    assert torch.allclose(fft_path, mm_path, atol=1e-3, rtol=1e-4)


@pytest.mark.oracle_invariant
@SETTINGS
@given(a=st.floats(-3, 3), b=st.floats(-3, 3), f=spectra_1d(), g=spectra_1d())
def test_fuzz_transform_linearity(a, b, f, g):
    n = min(f.shape[1], g.shape[1])
    rows = min(f.shape[0], g.shape[0])
    f, g = f[:rows, :n], g[:rows, :n]
    lhs = FourierBasis.transform(a * f + b * g)
    rhs = a * FourierBasis.transform(f) + b * FourierBasis.transform(g)
    assert torch.allclose(lhs, rhs, atol=1e-5)


@pytest.mark.oracle_independent
@SETTINGS
@given(modes=st.integers(1, 64))
def test_fuzz_wave_number_matches_fftfreq(modes):
    k = FourierBasis.wave_number(modes).flatten()
    assert torch.allclose(k, torch.fft.fftfreq(modes) * modes)


@pytest.mark.oracle_consistency
@SETTINGS
@given(coeff=spectra_1d())
def test_fuzz_real_complex_coeff_roundtrip(coeff):
    c64 = coeff.to(torch.complex64)
    assert torch.equal(to_complex_coeff(to_real_coeff(c64)), c64)


@SETTINGS
@given(
    points=st.integers(2, 20),
    modes=st.integers(1, 16),
)
@pytest.mark.oracle_analytic
def test_fuzz_fn_columns_are_complex_exponentials(points, modes):
    x = torch.linspace(0, 1, points, dtype=torch.float64).view(-1, 1)
    basis = FourierBasis.fn(x, modes=modes)
    k = FourierBasis.wave_number(modes).flatten().to(x)
    expected = torch.exp(2j * torch.pi * x * k)
    assert basis.shape == (points, modes)
    assert torch.allclose(basis, expected, atol=1e-6)


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(1, 5), modes=st.integers(2, 16), seed=st.integers(0, 10_000))
def test_fuzz_generate_coeff_reproducible(n, modes, seed):
    a = FourierBasis.generate_coeff(
        n, modes, generator=torch.Generator().manual_seed(seed)
    )
    b = FourierBasis.generate_coeff(
        n, modes, generator=torch.Generator().manual_seed(seed)
    )
    assert torch.equal(a, b)
    assert a.shape == (n, modes)


@pytest.mark.oracle_invariant
@SETTINGS
@given(n=st.integers(1, 4), modes=st.integers(4, 32), seed=st.integers(0, 10_000))
def test_fuzz_generate_real_signal_has_no_imaginary_part(n, modes, seed):
    coeff = FourierBasis.generate_coeff(
        n, modes, complex_funcs=False, generator=torch.Generator().manual_seed(seed)
    )
    vals = FourierBasis.inv_transform(coeff)
    assert vals.imag.abs().max() < 1e-4


@pytest.mark.oracle_smoke
@SETTINGS
@given(target=st.integers(2, 64))
def test_fuzz_resize_modes_sets_target(target):
    basis = FourierBasis(FourierBasis.generate_empty(2, 16))
    resized = basis.resize_modes(target)
    assert resized.modes[0] == target


@pytest.mark.oracle_smoke
@SETTINGS
@given(seed=st.integers(0, 10_000), std=st.floats(0.01, 0.5))
def test_fuzz_perturb_keeps_shape_and_changes_values(seed, std):
    gen = torch.Generator().manual_seed(seed)
    basis = FourierBasis(FourierBasis.generate_coeff(1, 32, generator=gen))
    perturbed = basis.perturb(std_ratio=std, generator=gen)
    assert perturbed.coeff.shape == basis.coeff.shape
    assert not torch.allclose(perturbed.coeff, basis.coeff)


@pytest.mark.oracle_invariant
@SETTINGS
@given(coeff=spectra_1d())
def test_fuzz_add_doubles_and_sub_zeroes(coeff):
    basis = FourierBasis(coeff)
    assert torch.allclose((basis + basis).coeff, coeff * 2, atol=1e-6)
    assert torch.allclose((basis - basis).coeff, torch.zeros_like(coeff), atol=1e-6)


@pytest.mark.oracle_smoke
@SETTINGS
@given(rows=st.integers(2, 6), stop=st.integers(1, 2))
def test_fuzz_getitem_selects_rows(rows, stop):
    stop = min(stop, rows)
    basis = FourierBasis(FourierBasis.generate_empty(rows, 8))
    assert len(basis[:stop]) == stop


# =========================================================================== #
# Structural / error-path tests: explicitly not fuzzed, no MMS
# =========================================================================== #
REFUSALS = {
    "transform of a one-dimensional tensor": (
        AssertionError,
        "needs to have at least two dimensions",
        lambda size: FourierBasis.transform(torch.randn(size) + 0j),
    ),
    "inverse transform of a one-dimensional tensor": (
        AssertionError,
        "needs to have at least two dimensions",
        lambda size: FourierBasis.inv_transform(torch.randn(size) + 0j),
    ),
    "basis functions without modes": (
        ValueError,
        "modes should not be None",
        lambda size: FourierBasis.fn(torch.zeros(size, 1)),
    ),
    "basis functions with modes of another dimension": (
        AssertionError,
        r"both need to have the same dimensions \(modes",
        lambda size: FourierBasis.fn(torch.zeros(size, 1), modes=(4, 4)),
    ),
    "real coefficients": (
        AssertionError,
        "need to be complex",
        lambda size: FourierBasis(torch.randn(1, size)),
    ),
    "evaluation without coefficients": (
        AssertionError,
        "basis has no coefficients",
        lambda size: FourierBasis()(torch.linspace(0, 1, size)),
    ),
    "time-dependent evaluation without times": (
        AssertionError,
        "t must not be none",
        lambda size: FourierBasis.evaluate(
            FourierBasis.transform(torch.randn(1, size, size) + 0j),
            torch.linspace(0, 1, size).view(-1, 1),
            time_dependent=True,
            domain=((0.0, 1.0), (0.0, 1.0)),
        ),
    ),
    "an unknown value type": (
        RuntimeError,
        "expects value_type random or zero",
        lambda size: FourierBasis.generate(2, size, value_type=f"bogus{size}"),  # type: ignore[arg-type]
    ),
    "adding a number": (
        TypeError,
        "unsupported operand",
        lambda size: FourierBasis(FourierBasis.generate_empty(1, size)) + size,  # type: ignore[operator]
    ),
    "subtracting a number": (
        TypeError,
        "unsupported operand",
        lambda size: FourierBasis(FourierBasis.generate_empty(1, size)) - size,  # type: ignore[operator]
    ),
    "plotting past the last function": (
        AssertionError,
        "out of bounds",
        lambda size: FourierBasis(FourierBasis.generate_empty(1, size)).plot(
            i=0, n=2, device=CPU
        ),
    ),
    "plotting a time-dependent spectrum": (
        NotImplementedError,
        "not implemented for time-dependent bases",
        lambda size: (
            FourierBasis(FourierBasis.transform(torch.randn(1, size, size) + 0j))
            .to_time_dependent(nt=size)
            .plot_coefficients()
        ),
    ),
}


@pytest.mark.oracle_smoke
@pytest.mark.parametrize("kind", tuple(REFUSALS))
@SETTINGS
@given(size=st.integers(2, 16))
@example(size=8)
def test_a_refused_call_raises_its_error(kind, size):
    error, message, call = REFUSALS[kind]
    with pytest.raises(error, match=message):
        call(size)
    plt.close("all")


@pytest.mark.oracle_smoke
@SETTINGS
@given(functions=st.integers(1, 3), size=st.integers(2, 16))
@example(functions=1, size=8)
def test_the_transforms_take_real_values_and_a_resolution_as_count_or_slice(
    functions, size
):
    values = torch.randn(functions, size)
    assert FourierBasis.transform(values).is_complex()
    assert FourierBasis.inv_transform(values).is_complex()
    for res in (size, slice(0, 1, size)):
        assert FourierBasis.transform(values + 0j, res=res).shape == (functions, size)


@pytest.mark.oracle_smoke
@SETTINGS
@given(points=st.integers(1, 12), modes=st.integers(1, 12))
@example(points=4, modes=4)
def test_basis_functions_take_integer_points_in_either_orientation(points, modes):
    at_integers = FourierBasis.fn(torch.arange(points).view(-1, 1), modes=modes)
    assert at_integers.shape == (points, modes) and at_integers.is_complex()
    x = torch.linspace(0, 1, points).view(-1, 1)
    assert (
        FourierBasis.fn(x, modes, transpose=True).shape
        == FourierBasis.fn(x, modes).shape
    )


@pytest.mark.oracle_smoke
@SETTINGS
@given(functions=st.integers(1, 4), modes=st.integers(1, 16))
@example(functions=2, modes=8)
def test_a_zero_value_type_generates_no_nonzero_coefficient(functions, modes):
    generated = FourierBasis.generate(functions, modes, value_type="zero")
    assert generated.coeff.shape == (functions, modes)
    assert torch.count_nonzero(generated.coeff) == 0


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    starts=st.tuples(st.floats(-2.0, 2.0), st.floats(-2.0, 2.0)),
    spans=st.tuples(st.floats(0.5, 4.0), st.floats(0.5, 4.0)),
    points=st.tuples(st.integers(2, 12), st.integers(2, 12)),
)
@example(starts=(0.0, 0.0), spans=(1.0, 2.0), points=(4, 6))
def test_a_grid_spans_each_stated_interval_end_to_end(starts, spans, points):
    stated = tuple(
        slice(start, start + span, count)
        for start, span, count in zip(starts, spans, points, strict=True)
    )
    line = FourierBasis.grid(stated[0])
    assert line.shape == (points[0], 1)
    assert torch.isclose(line[0, 0], torch.tensor(starts[0]))
    assert torch.isclose(line[-1, 0], torch.tensor(starts[0] + spans[0]))
    plane = FourierBasis.grid(stated)
    assert plane.shape == (*points, 2)
    assert torch.isclose(plane[-1, 0, 0], torch.tensor(starts[0] + spans[0]))
    assert torch.isclose(plane[0, -1, 1], torch.tensor(starts[1] + spans[1]))
    unit = FourierBasis.grid(points[0])
    assert unit.shape == (points[0], 1)
    assert unit[0, 0] == 0 and unit[-1, 0] == 1


@pytest.mark.oracle_consistency
@SETTINGS
@given(
    functions=st.integers(1, 3),
    modes=st.integers(2, 32),
    points=st.integers(1, 40),
    seed=st.integers(0, 10_000),
)
@example(functions=1, modes=16, points=20, seed=0)
def test_evaluation_reads_real_and_complex_coefficients_alike(
    functions, modes, points, seed
):
    generator = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.transform(
        torch.randn(functions, modes, generator=generator) + 0j
    )
    x = torch.rand(points, generator=generator)
    assert torch.equal(
        FourierBasis.evaluate(coeff, x),
        FourierBasis.evaluate(to_real_coeff(coeff), x),
    )


@pytest.mark.oracle_analytic
@SETTINGS
@given(modes=st.integers(1, 64))
@example(modes=12)
def test_the_dealias_mask_keeps_the_lowest_two_thirds_of_wavenumbers(modes):
    wavenumbers = torch.from_numpy(np.fft.fftfreq(modes, 1 / modes))
    mask = FourierBasis.dealias_mask(modes)
    assert bool(mask[0])
    assert torch.equal(mask, wavenumbers.abs() <= (2.0 / 3.0) * (modes / 2))


PLOT_FIELDS = {
    ("1d", "real"): lambda: _real_1d(),
    ("1d", "complex"): lambda: _complex_1d(),
    ("2d", "real"): lambda: FourierBasis(
        FourierBasis.transform(torch.randn(1, 8, 8) + 0j)
    ),
    ("2d", "complex"): lambda: _complex_2d(),
}


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.parametrize("field", tuple(PLOT_FIELDS))
@pytest.mark.parametrize("component", [None, "real", "imag"])
@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("on_axes", [False, True])
def test_every_plot_of_a_field_draws(field, component, scatter, on_axes, caplog):
    figure, axes = plt.subplots()
    with caplog.at_level("WARNING", logger="SpectralSVR.basis._plot"):
        PLOT_FIELDS[field]().plot(
            plot_component=component,
            complex_scatter=scatter,
            plt=axes if on_axes else None,
            device=CPU,
        )
    plt.close("all")
    dims, values = field
    warns = dims == "2d" and values == "complex" and component is None and not scatter
    assert ("plotting only real component" in caplog.text) == warns


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.parametrize("field", tuple(PLOT_FIELDS))
@pytest.mark.parametrize("component", ["magnitude", "real", "imag"])
@pytest.mark.parametrize("on_axes", [False, True])
def test_every_spectrum_plot_of_a_field_draws(field, component, on_axes):
    figure, axes = plt.subplots()
    PLOT_FIELDS[field]().plot_coefficients(
        component=component, plt=axes if on_axes else None
    )
    plt.close("all")


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_prefered_evaluation_mode():
    assert FourierBasis.prefered_evaluation_mode() == "inverse transform"


DOUBLE_PRECISION_ROUNDING = 1e-12


@pytest.mark.oracle_independent
@SETTINGS
@given(
    m=st.integers(2, 8),
    seed=st.integers(0, 10_000),
    starts=st.tuples(st.floats(-2.0, 2.0), st.floats(-2.0, 2.0)),
    spans=st.tuples(st.floats(0.5, 4.0), st.floats(0.5, 4.0)),
)
@example(m=8, seed=0, starts=(0.0, 0.0), spans=(1.0, 1.0))
@example(m=5, seed=1, starts=(1.0, -2.0), spans=(2.0, 3.5))
def test_to_time_dependent_preserves_values(m, seed, starts, spans):
    g = torch.Generator().manual_seed(seed)
    u = FourierBasis(
        FourierBasis.transform(
            torch.randn(2, m, m, generator=g, dtype=torch.float64) + 0j
        ),
        domain=tuple(
            (start, start + span) for start, span in zip(starts, spans, strict=True)
        ),
    )
    td = u.to_time_dependent()
    tdv, td_grid = td.get_values_and_grid(res=m, device=CPU)
    source_at_slices = fourier_series_at(
        u.coeff, td_grid.flatten(0, -2), u.domain
    ).reshape(tdv.shape)
    assert np.abs(tdv.numpy() - source_at_slices).max() <= DOUBLE_PRECISION_ROUNDING

    tiv, ti_grid = td.to_time_independent().get_values_and_grid(res=m, device=CPU)
    time_wavenumbers = np.abs(np.fft.fftfreq(m, 1 / m)).reshape(1, m, 1)
    largest_second_time_derivative = (
        (
            u.coeff.abs().numpy()
            / m**2
            * (2 * math.pi * time_wavenumbers / spans[0]) ** 2
        )
        .sum(axis=(1, 2))
        .max()
    )
    interpolation_bound = (spans[0] / (m - 1)) ** 2 / 8 * largest_second_time_derivative
    source_at_nodes = fourier_series_at(
        u.coeff, ti_grid.flatten(0, -2), u.domain
    ).reshape(tiv.shape)
    assert np.abs(tiv.numpy() - source_at_nodes).max() <= (
        interpolation_bound + DOUBLE_PRECISION_ROUNDING
    )


def _real_1d():
    x = torch.arange(0, 1, 1.0 / 32)
    sig = (torch.sin(2 * torch.pi * x) + 0j).unsqueeze(0)
    return FourierBasis(FourierBasis.transform(sig))


def _complex_1d():
    return FourierBasis(
        FourierBasis.generate_coeff(1, 32, complex_funcs=True), complex_funcs=True
    )


def _complex_2d():
    return FourierBasis(
        FourierBasis.generate_coeff(1, (8, 8), complex_funcs=True), complex_funcs=True
    )


_EXACT = EvaluationStrategy(allow_approximate=False)


@SETTINGS
@given(
    freq=st.integers(1, 14),
    amp=st.floats(0.3, 4.0),
    length=st.floats(0.5, 3.0),
    start=st.floats(-3.0, 3.0),
    npts=st.integers(1, 60),
    seed=st.integers(0, 10_000),
)
@pytest.mark.oracle_analytic
def test_separable_evaluate_single_mode_1d(freq, amp, length, start, npts, seed):
    # MMS: coeff of A*exp(2*pi*i*m*(x-a)/L) is A*N at wavenumber m on domain
    # [a, a+L). The separable (exact) evaluate must reproduce that closed form at
    # arbitrary points. Fuzzing ``start`` (spanning 0) covers both the canonical
    # and shifted domains.
    n = 32
    assert 0 < freq < n // 2
    coeff = torch.zeros(1, n, dtype=torch.complex128)
    coeff[0, freq] = amp * n  # fft index == wavenumber for freq < n/2
    x = start + (
        torch.rand(
            npts, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
        )
        * length
    )
    got = FourierBasis(coeff, domain=(start, start + length), strategy=_EXACT)(
        x, device=CPU
    )
    exact = amp * torch.exp(2j * torch.pi * freq * (x - start) / length)
    assert torch.allclose(got[0], exact, atol=1e-9)


@SETTINGS
@given(
    mx=st.integers(1, 3),
    my=st.integers(1, 3),
    ax=st.floats(-2.0, 2.0),
    ay=st.floats(-2.0, 2.0),
    lx=st.floats(0.5, 2.0),
    ly=st.floats(0.5, 2.0),
    npts=st.integers(1, 40),
    seed=st.integers(0, 10_000),
)
@pytest.mark.oracle_analytic
def test_separable_evaluate_single_mode_2d(mx, my, ax, ay, lx, ly, npts, seed):
    # MMS in 2D on a per-axis domain [ax, ax+Lx) x [ay, ay+Ly): the separable
    # kernel must reproduce exp(2*pi*i*(mx*(x-ax)/Lx + my*(y-ay)/Ly)). Fuzzing the
    # per-axis starts (spanning 0) covers canonical and shifted domains.
    n = 8
    coeff = torch.zeros(1, n, n, dtype=torch.complex128)
    coeff[0, mx, my] = float(n * n)
    g = torch.Generator().manual_seed(seed)
    starts = torch.tensor([ax, ay])
    x = starts + torch.rand(npts, 2, generator=g, dtype=torch.float64) * torch.tensor(
        [lx, ly]
    )
    got = FourierBasis(coeff, domain=((ax, ax + lx), (ay, ay + ly)), strategy=_EXACT)(
        x, device=CPU
    )
    exact = torch.exp(
        2j * torch.pi * (mx * (x[:, 0] - ax) / lx + my * (x[:, 1] - ay) / ly)
    )
    assert torch.allclose(got[0], exact, atol=1e-9)


@pytest.mark.oracle_independent
@SETTINGS
@given(n=st.integers(2, 32), rows=st.integers(1, 3), seed=st.integers(0, 10_000))
def test_separable_evaluate_on_grid_matches_torch_ifft(n, rows, seed):
    # On the uniform grid, the separable evaluate equals torch.fft.ifft.
    g = torch.Generator().manual_seed(seed)
    coeff = torch.randn(rows, n, dtype=torch.complex128, generator=g)
    grid = torch.arange(0, 1, 1.0 / n, dtype=torch.float64)
    got = FourierBasis(coeff, domain=(0.0, 1.0), strategy=_EXACT)(grid, device=CPU)
    assert torch.allclose(got, torch.fft.ifft(coeff, dim=1), atol=1e-9)


@pytest.mark.oracle_independent
@SETTINGS
@given(m=st.integers(2, 10), seed=st.integers(0, 10_000))
def test_separable_evaluate_on_grid_matches_torch_ifft2(m, seed):
    g = torch.Generator().manual_seed(seed)
    coeff = torch.randn(1, m, m, dtype=torch.complex128, generator=g)
    ax = torch.arange(0, 1, 1.0 / m, dtype=torch.float64)
    xg, yg = torch.meshgrid(ax, ax, indexing="ij")
    pts = torch.stack([xg.flatten(), yg.flatten()], dim=1)
    got = FourierBasis(coeff, domain=((0.0, 1.0), (0.0, 1.0)), strategy=_EXACT)(
        pts, device=CPU
    )
    ref = torch.fft.ifft2(coeff, dim=(1, 2)).reshape(1, m * m)
    assert torch.allclose(got, ref, atol=1e-9)


@SETTINGS
@given(
    n=st.integers(2, 24),
    npts=st.integers(1, 100),
    seed=st.integers(0, 10_000),
)
@pytest.mark.oracle_consistency
def test_separable_evaluate_chunking_is_exact(n, npts, seed):
    # Chunking the separable path over points must not change the result.
    g = torch.Generator().manual_seed(seed)
    coeff = torch.randn(2, n, dtype=torch.complex128, generator=g)
    x = torch.rand(npts, generator=g, dtype=torch.float64)
    whole = FourierBasis(
        coeff,
        domain=(0.0, 1.0),
        strategy=EvaluationStrategy(
            allow_approximate=False, memory_budget_mb=float("inf")
        ),
    )(x, device=CPU)
    chunked = FourierBasis(
        coeff,
        domain=(0.0, 1.0),
        strategy=EvaluationStrategy(allow_approximate=False, memory_budget_mb=1e-4),
    )(x, device=CPU)
    assert torch.allclose(whole, chunked, atol=1e-12)


@SETTINGS
@given(
    budget_mb=st.floats(1e-4, 64.0),
    per_point=st.integers(1, 4096),
    itemsize=st.sampled_from([8, 16]),
)
@pytest.mark.oracle_invariant
def test_strategy_chunk_and_approx_from_budget(budget_mb, per_point, itemsize):
    s = EvaluationStrategy(memory_budget_mb=budget_mb, allow_approximate=True)
    budget = s.max_elements(itemsize)
    chunk = s.chunk_points(per_point, itemsize)
    assert chunk >= 1
    # a block stays within (or at) the element budget
    assert chunk * per_point <= max(per_point, budget)
    # approximate kicks in exactly when the dense work exceeds the budget
    assert s.use_approximate(budget + 1, itemsize) is True
    assert s.use_approximate(budget - 1 if budget > 1 else 0, itemsize) is False


@pytest.mark.oracle_invariant
@SETTINGS
@given(budget_mb=st.floats(1e-3, 128.0), per_point=st.integers(1, 4096))
def test_strategy_budget_scales_with_dtype(budget_mb, per_point):
    # The element budget must halve when the dtype doubles in size: a fixed byte
    # budget fits half as many complex128 (16B) entries as complex64 (8B).
    s = EvaluationStrategy(memory_budget_mb=budget_mb)
    assert math.isclose(s.max_elements(8), 2 * s.max_elements(16))
    # so complex128 chunks no more points per block than complex64
    assert s.chunk_points(per_point, 16) <= s.chunk_points(per_point, 8)


@SETTINGS
@given(
    per_point=st.integers(1, 8192),
    itemsize=st.sampled_from([8, 16]),
    dense=st.floats(1.0, 1e18),
)
@pytest.mark.oracle_invariant
def test_strategy_infinite_budget_never_approximates(per_point, itemsize, dense):
    s = EvaluationStrategy(memory_budget_mb=float("inf"))
    assert s.use_approximate(dense, itemsize) is False
    assert s.chunk_points(per_point, itemsize) >= 10**6  # effectively one block


@pytest.mark.oracle_invariant
@SETTINGS
@given(dense=st.floats(1.0, 1e18), itemsize=st.sampled_from([8, 16]))
def test_strategy_disallow_approximate(dense, itemsize):
    s = EvaluationStrategy(memory_budget_mb=0.0, allow_approximate=False)
    assert s.use_approximate(dense, itemsize) is False  # never approximate


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_basis_diagonal_hooks_default_to_none():
    # Base defaults signal "not diagonally differentiable / no de-aliasing rule"
    # for bases that don't override them (Chebyshev, wavelet).
    assert Basis.derivative_eigenvalues(8, 1.0) is None
    assert Basis.dealias_mask(8) is None


@pytest.mark.oracle_smoke
@pytest.mark.scenario("Generating coefficients can be directed")
@settings(deadline=None, max_examples=15)
@given(rows=st.integers(1, 3), modes=st.integers(4, 16))
def test_generating_coefficients_is_placed_where_it_is_told(rows, modes):
    if not torch.accelerator.is_available():
        pytest.skip("directing placement elsewhere needs a second device to direct to")
    elsewhere = accelerator()

    drawn = FourierBasis.generate_coeff(
        rows,
        modes,
        generator=torch.Generator(device=elsewhere).manual_seed(0),
        device=elsewhere,
    )
    empty = FourierBasis.generate_empty(rows, modes, device=elsewhere)
    field = FourierBasis.generate(
        rows,
        modes,
        generator=torch.Generator(device=elsewhere).manual_seed(0),
        device=elsewhere,
    )
    assert drawn.device.type == elsewhere.type
    assert empty.device.type == elsewhere.type
    assert field.coeff.device.type == elsewhere.type

    assert FourierBasis.generate_empty(rows, modes).device == torch.get_default_device()


@pytest.mark.oracle_invariant
@pytest.mark.scenario("Stating a device does not change what is drawn")
@settings(deadline=None, max_examples=15)
@given(rows=st.integers(1, 3), modes=st.integers(4, 16), seed=st.integers(0, 10_000))
def test_how_a_device_is_stated_does_not_change_the_draw(rows, modes, seed):
    if not torch.accelerator.is_available():
        pytest.skip("the two ways of stating a device differ only when a second exists")

    def drawn_on_the_host(state_it: bool) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if state_it:
            return FourierBasis.generate_coeff(
                rows, modes, generator=generator, device="cpu"
            )
        return FourierBasis.generate_coeff(rows, modes, generator=generator)

    torch.set_default_device(accelerator())
    stated_at_the_call = drawn_on_the_host(True)
    torch.set_default_device("cpu")
    followed_the_process = drawn_on_the_host(False)

    assert torch.equal(stated_at_the_call, followed_the_process)


LIFECYCLE_MODES = st.integers(2, 8)
LIFECYCLE_FUNCTIONS = st.integers(1, 4)
NOT_AN_INTEGER = "not an integer"
ACCEPTED_INTEGER_TYPES = (int, np.int64, np.array, torch.tensor)
SLICE_COUNTS = st.one_of(
    st.just((None, None)),
    st.integers(-2, 12).flatmap(
        lambda count: st.sampled_from(ACCEPTED_INTEGER_TYPES).map(
            lambda integer_type: (integer_type(count), count)
        )
    ),
    st.sampled_from(
        (
            2.0,
            True,
            False,
            np.True_,
            torch.tensor(True),
            torch.tensor(2.0),
            torch.tensor([3]),
            np.array([3]),
        )
    ).map(lambda value: (value, NOT_AN_INTEGER)),
)
REFUSED_SLICE_COUNTS = (
    (1, 1),
    (0, 0),
    (2.0, NOT_AN_INTEGER),
    (True, NOT_AN_INTEGER),
    (False, NOT_AN_INTEGER),
    (4.0, NOT_AN_INTEGER),
    (np.True_, NOT_AN_INTEGER),
    (torch.tensor(True), NOT_AN_INTEGER),
    (torch.tensor([3]), NOT_AN_INTEGER),
    (np.array([3]), NOT_AN_INTEGER),
)


class BasisLifecycle(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.basis = FourierBasis()
        self.stated_domain = None

    def sampling_fits(self):
        sampling = self.basis.config.sampling
        return not isinstance(sampling, tuple) or len(sampling) == self.basis.ndim

    def placeable(self):
        basis = self.basis
        if basis.coeff.numel() == 0 or not self.sampling_fits():
            return False
        if basis.time_dependent:
            return basis.ndim >= 1 and basis.time_size >= 2
        return True

    @initialize()
    def reproduce_every_pinned_case(self):
        self.start_empty(functions=2, modes=(8, 4))
        self.set_domain(spans=(2.0, 3.0))
        self.start_empty(functions=7, modes=(8,))
        self.generate(functions=1, modes=(16,), complex_funcs=True, seed=1)
        self.read_values_at_a_resolution(points=20)
        self.read_a_selection_of_functions(start=0, count=1, points=20)
        self.read_values_with_their_grid(points=50)
        self.generate(functions=4, modes=(16,), complex_funcs=False, seed=0)
        self.read_a_selection_of_functions(start=1, count=2, points=20)
        self.generate(functions=2, modes=(8, 8), complex_funcs=False, seed=0)
        self.read_values_at_a_resolution(points=8)
        self.convert_to_time_dependent(slices=(None, None))
        self.convert_again_and_mutate_the_result()
        self.read_native_values()
        self.read_values_at_a_resolution(points=8)
        self.differentiate(axis=0)
        self.differentiate(axis=1)
        self.integrate(axis=0)
        self.integrate(axis=1)
        self.evaluate_through_time(times=5, points=10)
        self.resize_through_time(modes=4, slices=4)
        self.convert_to_time_independent()
        self.convert_again_and_mutate_the_result()
        for time_dependent in (False, True):
            self.start_from_a_two_axis_field(time_dependent=time_dependent)
            for slices in REFUSED_SLICE_COUNTS:
                self.convert_to_time_dependent(slices=slices)
        self.start_from_a_two_axis_field(time_dependent=True)
        for value, requested in (
            (None, None),
            (4, 4),
            (np.int64(4), 4),
            (np.array(4), 4),
            (torch.tensor(4), 4),
            (5, 5),
            (torch.tensor(1), 1),
            (np.array(6), 6),
        ):
            self.convert_to_time_dependent(slices=(value, requested))
        self.start_from_a_two_axis_field(time_dependent=False)
        self.convert_to_time_independent()
        for time_dependent, slices in ((False, (None, None)), (True, (3, 3))):
            self.start_without_coefficients(time_dependent=time_dependent)
            self.convert_to_time_dependent(slices=slices)
        self.start_without_coefficients(time_dependent=True)
        self.convert_to_time_independent()
        self.start_empty(functions=1, modes=(8,))
        self.convert_to_time_dependent(slices=(None, None))
        for spatial_modes, source_slices, slices in (
            ((4,), 1, (None, None)),
            ((4,), 1, (1, 1)),
            ((4,), 1, (3, 3)),
            ((4,), 2, (None, None)),
            ((4,), 2, (2, 2)),
            ((), 4, (None, None)),
        ):
            self.start_time_dependent(
                functions=1,
                slices=source_slices,
                spatial_modes=spatial_modes,
                seed=0,
            )
            self.convert_to_time_dependent(slices=slices)
            self.convert_to_time_independent()
        self.start_empty(functions=1, modes=(8, 4))
        for length in (1, 3):
            self.restate_the_sampling(length=length)
            self.convert_to_time_dependent(slices=(None, None))
        self.basis = FourierBasis()
        self.stated_domain = None

    @rule(
        functions=LIFECYCLE_FUNCTIONS,
        modes=st.lists(LIFECYCLE_MODES, min_size=1, max_size=2).map(tuple),
    )
    def start_empty(self, functions, modes):
        coeff = FourierBasis.generate_empty(functions, modes)
        assert coeff.is_complex() and torch.count_nonzero(coeff) == 0
        self.basis = FourierBasis(coeff)
        self.stated_domain = None

    @rule(
        functions=LIFECYCLE_FUNCTIONS,
        modes=st.lists(LIFECYCLE_MODES, min_size=1, max_size=2).map(tuple),
        complex_funcs=st.booleans(),
        seed=st.integers(0, 1000),
    )
    def generate(self, functions, modes, complex_funcs, seed):
        self.basis = FourierBasis.generate(
            functions,
            modes,
            complex_funcs=complex_funcs,
            generator=torch.Generator().manual_seed(seed),
        )
        self.stated_domain = None
        if complex_funcs:
            assert FourierBasis.inv_transform(self.basis.coeff).imag.abs().sum() > 0

    @rule(time_dependent=st.booleans())
    def start_without_coefficients(self, time_dependent):
        self.basis = FourierBasis(time_dependent=time_dependent)
        self.stated_domain = None

    @rule(time_dependent=st.booleans())
    def start_from_a_two_axis_field(self, time_dependent):
        self.basis = FourierBasis(
            FourierBasis.transform(torch.randn(1, 8, 4, dtype=torch.float64) + 0j),
            coeff_dtype=torch.complex128,
        )
        if time_dependent:
            self.basis = self.basis.to_time_dependent(nt=4)
        self.stated_domain = None

    @rule(
        functions=st.integers(1, 3),
        slices=st.integers(1, 4),
        spatial_modes=st.lists(LIFECYCLE_MODES, max_size=2).map(tuple),
        seed=st.integers(0, 1000),
    )
    def start_time_dependent(self, functions, slices, spatial_modes, seed):
        generator = torch.Generator().manual_seed(seed)
        coeff = torch.randn(
            functions,
            slices,
            *spatial_modes,
            generator=generator,
            dtype=torch.complex128,
        )
        self.basis = FourierBasis(
            coeff, time_dependent=True, coeff_dtype=torch.complex128
        )
        self.stated_domain = None

    @precondition(lambda self: self.basis.coeff.numel() > 0)
    @rule(length=st.integers(0, 3))
    def restate_the_sampling(self, length):
        self.basis = FourierBasis(
            self.basis.coeff,
            time_dependent=self.basis.time_dependent,
            domain=self.basis.config.domain,
            sampling=(PeriodicUniform(),) * length,
        )

    @precondition(lambda self: self.basis.coeff.numel() > 0)
    @rule(spans=st.lists(st.floats(0.5, 4.0), min_size=1, max_size=3).map(tuple))
    def set_domain(self, spans):
        axes = self.basis.coeff.ndim - 1
        stated = tuple((0.0, span) for span in (spans * axes)[:axes])
        self.basis.domain = stated
        self.stated_domain = stated

    @precondition(lambda self: self.basis.coeff.numel() > 0)
    @rule()
    def copy_and_mutate_the_copy(self):
        before = self.basis.coeff.clone()
        duplicate = self.basis.copy()
        duplicate.coeff = duplicate.coeff * 2 + 1
        assert torch.equal(self.basis.coeff, before)

    @rule(slices=SLICE_COUNTS)
    def convert_to_time_dependent(self, slices):
        value, requested = slices
        source = self.basis
        if requested == NOT_AN_INTEGER:
            with pytest.raises(
                TypeError, match="a slice count is a zero-dimensional integer"
            ):
                source.to_time_dependent(nt=value)
            return
        if source.coeff.numel() == 0:
            with pytest.raises(ValueError, match="holding no coefficients"):
                source.to_time_dependent(nt=value)
            return
        if source.time_dependent:
            if requested is not None and requested != source.time_size:
                refusal = (
                    f"of {source.time_size} slices converts only to its own slice "
                    f"count; got {requested}$"
                )
            elif source.ndim < 1:
                refusal = "a spatial axis beside time; got 0 spatial axes$"
            elif source.time_size < 2:
                refusal = f"at least two slices; got {source.time_size}$"
            else:
                again = source.to_time_dependent(nt=value)
                assert again is not source and again.config == source.config
                assert again.coeff.data_ptr() == source.coeff.data_ptr()
                return
            with pytest.raises(ValueError, match=refusal):
                source.to_time_dependent(nt=value)
            return
        slices_expected = source.modes[0] if requested is None else requested
        if source.ndim < 2:
            refusal = f"a spatial axis beside time; got {source.ndim - 1} spatial axes$"
        elif slices_expected < 2:
            refusal = f"at least two slices; got {slices_expected}$"
        elif not self.sampling_fits():
            refusal = f"schemes for {source.ndim} axes$"
        else:
            self.basis = source.to_time_dependent(nt=value)
            assert self.basis.time_dependent
            assert self.basis.time_size == slices_expected
            assert len(self.basis) == len(source)
            assert self.basis.modes == source.modes[1:]
            return
        with pytest.raises(ValueError, match=refusal):
            source.to_time_dependent(nt=value)

    @rule()
    def convert_to_time_independent(self):
        source = self.basis
        if not source.time_dependent:
            unchanged = source.to_time_independent()
            assert unchanged is not source and unchanged.config == source.config
            assert unchanged.coeff.data_ptr() == source.coeff.data_ptr()
            return
        if source.coeff.numel() == 0:
            refusal = "holding no coefficients"
        elif source.ndim < 1:
            refusal = "a spatial axis beside time; got 0 spatial axes$"
        elif source.time_size < 2:
            refusal = f"at least two slices; got {source.time_size}$"
        elif not self.sampling_fits():
            refusal = f"schemes for {source.ndim} axes$"
        else:
            self.basis = source.to_time_independent()
            assert not self.basis.time_dependent
            assert self.basis.coeff.shape == source.coeff.shape
            return
        with pytest.raises(ValueError, match=refusal):
            source.to_time_independent()

    @precondition(lambda self: self.placeable())
    @rule()
    def convert_again_and_mutate_the_result(self):
        before = self.basis.coeff.clone()
        unchanged = (
            self.basis.to_time_dependent()
            if self.basis.time_dependent
            else self.basis.to_time_independent()
        )
        assert unchanged.time_dependent == self.basis.time_dependent
        unchanged.coeff = unchanged.coeff * 2 + 1
        assert torch.equal(self.basis.coeff, before)

    @precondition(lambda self: self.placeable())
    @rule(axis=st.integers(0, 2))
    def differentiate(self, axis):
        axis = axis % (self.basis.coeff.ndim - 1)
        assert self.basis.grad(dim=axis).coeff.shape == self.basis.coeff.shape

    @precondition(lambda self: self.placeable())
    @rule(axis=st.integers(0, 2))
    def integrate(self, axis):
        axis = axis % (self.basis.coeff.ndim - 1)
        assert self.basis.integral(dim=axis).coeff.shape == self.basis.coeff.shape

    @precondition(lambda self: self.placeable())
    @rule()
    def read_native_values(self):
        assert self.basis.get_values(device=CPU).shape == self.basis.coeff.shape

    @precondition(lambda self: self.placeable())
    @rule(points=st.integers(2, 20))
    def read_values_at_a_resolution(self, points):
        axes = self.basis.coeff.ndim - 1
        by_slices = tuple(
            slice(start, stop, points) for start, stop in self.basis.domain
        )
        expected = (len(self.basis), *(points,) * axes)
        for mode in ("auto", "basis"):
            assert (
                self.basis.get_values(
                    res=by_slices, evaluation_mode=mode, device=CPU
                ).shape
                == expected
            )
        if axes == 1 and not self.basis.time_dependent:
            assert self.basis.get_values(res=points, device=CPU).shape == expected
            assert self.basis.get_values(res=by_slices[0], device=CPU).shape == expected

    @precondition(lambda self: self.placeable() and not self.basis.time_dependent)
    @rule(start=st.integers(0, 3), count=st.integers(1, 4), points=st.integers(2, 20))
    def read_a_selection_of_functions(self, start, count, points):
        start = start % len(self.basis)
        count = min(count, len(self.basis) - start)
        values = self.basis.get_values(i=start, n=count, res=points, device=CPU)
        assert values.shape[0] == count
        if self.basis.ndim == 1:
            x = torch.linspace(0, 1, points)
            selected = FourierBasis.evaluate(self.basis.coeff, x, i=start, n=count)
            assert torch.equal(
                selected,
                FourierBasis.evaluate(self.basis.coeff[start : start + count], x),
            )

    @precondition(lambda self: self.placeable() and not self.basis.time_dependent)
    @rule(points=st.integers(2, 50))
    def read_values_with_their_grid(self, points):
        values, grid = self.basis.get_values_and_grid(res=points, device=CPU)
        assert values.shape[0] == len(self.basis)
        assert grid.shape[-1] == self.basis.ndim

    @precondition(
        lambda self: (
            self.placeable() and self.basis.time_dependent and self.basis.ndim == 1
        )
    )
    @rule(times=st.integers(1, 6), points=st.integers(1, 12))
    def evaluate_through_time(self, times, points):
        out = FourierBasis.evaluate(
            self.basis.coeff,
            torch.linspace(0, 1, points).view(-1, 1),
            t=torch.linspace(0, 1, times),
            time_dependent=True,
            domain=self.basis.domain,
        )
        assert out.shape == (len(self.basis), times, points)

    @precondition(
        lambda self: (
            self.placeable() and self.basis.time_dependent and self.basis.ndim == 1
        )
    )
    @rule(modes=st.integers(2, 6), slices=st.integers(2, 6))
    def resize_through_time(self, modes, slices):
        target = FourierBasis(
            FourierBasis.transform(torch.randn(1, modes, modes) + 0j)
        ).to_time_dependent(nt=slices)
        resized = self.basis.resize_modes(target)
        assert resized.modes == (modes,)
        assert resized.time_size == slices

    @invariant()
    def the_shape_properties_follow_the_coefficients(self):
        coeff = self.basis.coeff
        assert len(self.basis) == (coeff.shape[0] if coeff.ndim else 0)
        if coeff.numel() == 0 and coeff.ndim < 2:
            assert self.basis.modes == (0,)
            if not self.basis.time_dependent:
                assert self.basis.ndim == 0
            return
        if self.basis.time_dependent and coeff.ndim < 3:
            return
        spatial = coeff.shape[2:] if self.basis.time_dependent else coeff.shape[1:]
        assert self.basis.modes == tuple(spatial)
        assert self.basis.ndim == len(spatial)
        assert self.basis.time_size == (
            coeff.shape[1] if self.basis.time_dependent else 0
        )

    @invariant()
    def a_stated_domain_is_kept(self):
        if (
            self.stated_domain is None
            or len(self.stated_domain) != self.basis.coeff.ndim - 1
        ):
            return
        assert self.basis.domain == self.stated_domain
        assert self.basis.lengths == tuple(
            stop - start for start, stop in self.stated_domain
        )


BasisLifecycle.TestCase.settings = settings(
    deadline=None,
    max_examples=40,
    stateful_step_count=10,
    suppress_health_check=[HealthCheck.too_slow],
)
TestBasisLifecycle = BasisLifecycle.TestCase
for mark in (
    pytest.mark.oracle_invariant,
    pytest.mark.scenario("A conversion that cannot be placed in time is refused"),
    pytest.mark.scenario(
        "A time-independent conversion that cannot be placed is refused"
    ),
    pytest.mark.scenario("A time-dependent field converts only to its own slice count"),
):
    TestBasisLifecycle = mark(TestBasisLifecycle)
