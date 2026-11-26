"""Shared closed-form anchors for oracle tests.

Everything here computes expected values by hand-written math, independent of
the library's transform/evaluate paths, so tests comparing against these are
anchored outside the code under test. Exact solutions reusable by the library
itself live in ``SpectralSVR.problems.exact``; this module is test-only.
"""

import math

import numpy
import torch


def half_open_grid(n: int, start: float = 0.0, stop: float = 1.0) -> torch.Tensor:
    """The periodic (half-open) uniform grid Fourier values sample on."""
    return torch.linspace(start, stop, n + 1, dtype=torch.float64)[:-1]


def fourier_series_at(
    coeff: torch.Tensor, points: torch.Tensor, domain: tuple[tuple[float, float], ...]
) -> numpy.ndarray:
    modes = coeff.shape[1:]
    starts = numpy.array([start for start, _ in domain])
    lengths = numpy.array([stop - start for start, stop in domain])
    wavenumber_axes = numpy.meshgrid(
        *(numpy.fft.fftfreq(m, 1 / m) for m in modes), indexing="ij"
    )
    wavenumbers = numpy.stack([axis.ravel() for axis in wavenumber_axes], axis=1)
    cell_points = (points.numpy() - starts) / lengths
    phases = numpy.exp(2j * numpy.pi * cell_points @ wavenumbers.T)
    return coeff.reshape(len(coeff), -1).numpy() @ phases.T / len(wavenumbers)


def single_mode_values(
    n_points: int, m: int, amp: float, phase: float, length: float = 1.0
) -> torch.Tensor:
    """Values of ``amp * cos(2 pi m x / L + phase)`` on the half-open grid."""
    x = half_open_grid(n_points, 0.0, length)
    return amp * torch.cos(2 * math.pi * m * x / length + phase)


def single_mode_spectrum(
    n_points: int, m: int, amp: float, phase: float
) -> torch.Tensor:
    """Hand-computed unnormalized DFT spectrum of :func:`single_mode_values`.

    ``amp*cos(k x + phase) = amp/2 (e^{i phase} e^{ikx} + e^{-i phase} e^{-ikx})``
    so the DFT (sum convention, matching ``torch.fft.fft``) has
    ``N * amp/2 * e^{+/- i phase}`` at bins ``m`` and ``N - m`` and zero
    elsewhere. Valid for ``0 < m < N/2``.
    """
    assert 0 < m < n_points / 2
    coeff = torch.zeros(n_points, dtype=torch.complex128)
    coeff[m] = (
        n_points * amp / 2 * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128))
    )
    coeff[n_points - m] = (
        n_points
        * amp
        / 2
        * torch.exp(torch.tensor(-1j * phase, dtype=torch.complex128))
    )
    return coeff


def hermitian_error(coeff: torch.Tensor) -> float:
    """How far a coefficient grid is from describing a real field.

    Mirrors every mode axis about bin 0 and compares against the conjugate;
    zero exactly when ``c[-k] == conj(c[k])`` on every axis.
    """
    mirrored = coeff
    for dim in range(1, coeff.ndim):
        index = (-torch.arange(coeff.shape[dim])) % coeff.shape[dim]
        mirrored = mirrored.index_select(dim, index)
    return float((coeff - mirrored.conj()).abs().max())


def hermitian_spectrum(n: int, seed: int) -> torch.Tensor:
    """A random spectrum symmetrised into one describing a real field."""
    raw = torch.randn(
        1, n, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed)
    )
    return 0.5 * (raw + raw.index_select(1, (-torch.arange(n)) % n).conj())


def separable_time_field(
    g_coeffs: torch.Tensor,
    t: torch.Tensor,
    x: torch.Tensor,
    m: int,
    length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``u(t,x) = g(t) sin(2 pi m x / L)`` with polynomial ``g`` and its du/dt.

    ``g(t) = sum_j g_coeffs[j] * t**j`` -- the time derivative is closed-form,
    independent of any library differentiation.

    Returns ``(u, u_t)`` of shape ``(nt, nx)``.
    """
    powers = torch.arange(len(g_coeffs), dtype=torch.float64)
    g = (g_coeffs * t.unsqueeze(-1).pow(powers)).sum(-1)  # (nt,)
    dg = ((g_coeffs[1:] * powers[1:]) * t.unsqueeze(-1).pow(powers[:-1])).sum(
        -1
    )  # (nt,)
    s = torch.sin(2 * math.pi * m * x / length)  # (nx,)
    return g.unsqueeze(-1) * s, dg.unsqueeze(-1) * s


def antiderivative_pair(
    n_points: int, m: int, amp: float, length: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form (derivative, antiderivative) pair on the half-open grid.

    ``ut = amp * cos(2 pi m x / L)`` integrates (zero-mean antiderivative) to
    ``u = amp * L / (2 pi m) * sin(2 pi m x / L)``.
    """
    x = half_open_grid(n_points, 0.0, length)
    k = 2 * math.pi * m / length
    ut = amp * torch.cos(k * x)
    u = amp / k * torch.sin(k * x)
    return ut, u
