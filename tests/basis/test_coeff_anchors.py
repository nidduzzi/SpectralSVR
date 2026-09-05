"""Closed-form anchors for coefficient manipulation and time-axis calculus.

These pin the VALUES (not just shapes) of resize_modes / zero_coeff /
to_time_dependent and the time-axis grad/integral of time-dependent fields,
against hand-derived expectations independent of the code under test.
"""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._anchors import half_open_grid, single_mode_spectrum
from SpectralSVR import FourierBasis
from SpectralSVR.utils import resize_modes, zero_coeff

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


def hermitian_error(coeff):
    """How far a coefficient grid is from describing a real field."""
    mirrored = coeff
    for dim in range(1, coeff.ndim):
        index = (-torch.arange(coeff.shape[dim])) % coeff.shape[dim]
        mirrored = mirrored.index_select(dim, index)
    return float((coeff - mirrored.conj()).abs().max())


def hermitian_spectrum(n, seed):
    raw = torch.randn(
        1, n, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed)
    )
    return 0.5 * (raw + raw.index_select(1, (-torch.arange(n)) % n).conj())


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_padding_preserves_conjugate_symmetry():
    for n in RESIZE_SOURCE_MODES:
        source = hermitian_spectrum(n, seed=n)
        assert hermitian_error(source) == 0.0, f"n={n}: source is not Hermitian"
        for target in (2 * n, 4 * n):
            measured = hermitian_error(resize_modes(source, target, rescale=False))
            assert measured < RESIZE_TOLERANCE, f"n={n} -> {target}: {measured:.3e}"


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_padding_then_truncating_returns_the_original():
    for n in RESIZE_SOURCE_MODES:
        source = hermitian_spectrum(n, seed=n)
        for target in (2 * n, 4 * n):
            padded = resize_modes(source, target, rescale=False)
            measured = float(
                (resize_modes(padded, n, rescale=False) - source).abs().max()
            )
            assert measured < RESIZE_TOLERANCE, (
                f"n={n} -> {target} -> {n}: {measured:.3e}"
            )


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_odd_source_counts_have_nothing_to_redistribute():
    """Isolation control: odd counts read clean under either convention."""
    for n in (5, 9):
        source = hermitian_spectrum(n, seed=n)
        for target in (2 * n, 4 * n):
            padded = resize_modes(source, target, rescale=False)
            for j in range(n):
                k = j if j <= (n - 1) // 2 else j - n
                assert padded[0, k % target] == source[0, j], (
                    f"n={n} -> {target}: coefficient at frequency {k} was altered"
                )


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_padding_preserves_the_function_at_the_original_sample_points():
    for n in RESIZE_SOURCE_MODES:
        source = hermitian_spectrum(n, seed=n)
        coarse = torch.fft.ifft(source) * n
        for target in (2 * n, 4 * n):
            fine = torch.fft.ifft(resize_modes(source, target, rescale=False)) * target
            measured = float((fine[:, :: target // n] - coarse).abs().max())
            assert measured < RESIZE_TOLERANCE, f"n={n} -> {target}: {measured:.3e}"


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_zero_coeff_values():
    x = (
        torch.arange(1, 9, dtype=torch.float32).to(torch.complex64).unsqueeze(0)
    )  # bins 0..7
    # keep 4 modes -> keep freqs {0, 1, -2, -1} = bins {0, 1, 6, 7}; zero the
    # high-|frequency| middle block bins {2, 3, 4, 5}
    expected = torch.tensor([[1, 2, 0, 0, 0, 0, 7, 8]], dtype=torch.complex64)
    assert torch.equal(zero_coeff(x, 4), expected)


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


# --------------------------------------------------------------------------- #
# to_time_dependent: per-time-slice spectra must equal g(t_j) * S with S the
# hand-computed single-mode spectrum -- values anchored, not just shapes.
# --------------------------------------------------------------------------- #
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
    t = half_open_grid(nt)  # 2D field: BOTH axes are periodic Fourier axes
    x = half_open_grid(ns)
    g = torch.cos(2 * math.pi * m_t * t)
    values = g.view(nt, 1) * (amp * torch.cos(2 * math.pi * m_x * x + phase))
    # batched 2D transform: (1, nt, ns) input -> true (time, space) spectrum
    coeff = FourierBasis.transform(values.unsqueeze(0) + 0j)
    u = FourierBasis(coeff, domain=((0.0, 1.0), (0.0, 1.0)))

    td = u.to_time_dependent()
    assert td.time_dependent and td.coeff.shape == (1, nt, ns)
    s = single_mode_spectrum(ns, m_x, amp, phase)  # hand-computed spatial DFT
    for j in (0, 3, nt // 2, nt - 1):
        expected = (g[j] * s).to(torch.complex64)
        assert torch.allclose(
            td.coeff[0, j].to(torch.complex64), expected, atol=2e-3 * ns * amp
        ), f"slice {j}"


@pytest.mark.oracle_analytic
@SETTINGS
@given(m_t=st.integers(1, 3), m_x=st.integers(1, 5), amp=st.floats(0.5, 2.0))
def test_to_time_independent_values_match_analytic(m_t, m_x, amp):
    # time-dependent field u(t, x) = cos(2 pi m_t t) * amp sin(2 pi m_x x) whose
    # slices sit on the closed uniform time nodes; to_time_independent resamples
    # that axis and transforms both axes. The resulting 2D field's values must
    # reproduce the closed form AT ITS OWN NODES exactly -- a shifted or
    # misinterpolated time axis (the to_time_dependent bug class) fails this.
    nt, ns = 16, 16
    t = torch.linspace(0, 1, nt, dtype=torch.float64)  # closed evolution nodes
    x = half_open_grid(ns)
    g = torch.cos(2 * math.pi * m_t * t)
    values = g.view(nt, 1) * (amp * torch.sin(2 * math.pi * m_x * x))
    coeff = FourierBasis.transform(values + 0j).unsqueeze(0)  # (1, nt, ns)
    u = FourierBasis(coeff, time_dependent=True, domain=((0.0, 1.0), (0.0, 1.0)))

    ti = u.to_time_independent()
    assert not ti.time_dependent
    got = FourierBasis.inv_transform(ti.coeff)[0].real.double().cpu()
    assert torch.allclose(got, values, atol=1e-3 * amp)
