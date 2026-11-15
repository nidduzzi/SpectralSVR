"""Shared closed-form anchors for oracle tests.

Everything here computes expected values by hand-written math, independent of
the library's transform/evaluate paths, so tests comparing against these are
anchored outside the code under test. Exact solutions reusable by the library
itself live in ``SpectralSVR.problems.exact``; this module is test-only.
"""

import math

import torch


def half_open_grid(n: int, start: float = 0.0, stop: float = 1.0) -> torch.Tensor:
    """The periodic (half-open) uniform grid Fourier values sample on."""
    return torch.linspace(start, stop, n + 1, dtype=torch.float64)[:-1]


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
    coeff[m] = n_points * amp / 2 * torch.exp(torch.tensor(1j * phase))
    coeff[n_points - m] = n_points * amp / 2 * torch.exp(torch.tensor(-1j * phase))
    return coeff


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
    dg = (
        (g_coeffs[1:] * powers[1:]) * t.unsqueeze(-1).pow(powers[:-1])
    ).sum(-1)  # (nt,)
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
