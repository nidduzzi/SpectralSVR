"""Analytic anchors for the SpectralSVR pipeline (forward and inverse).

Training data is built from closed forms (superpositions of single modes whose
antiderivative is known exactly) and the target spectra come from torch.fft
directly -- independent of the library's transform/generate paths -- so these
tests anchor the model's numerical output, not just its plumbing. The forward
tolerance is fixed from the operator's conditioning, not tuned to pass.
"""

import math

import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st

from SpectralSVR import LSSVR, FourierBasis, SpectralSVR, to_real_coeff
from tests._anchors import half_open_grid

SETTINGS = settings(deadline=None, max_examples=10)

N_SENSORS = 16
N_MODES = 16
MAX_M = 4


def _fields(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor):
    """Closed-form derivative/antiderivative pair for coefficients (n, MAX_M).

    f(x) = sum_m a_m cos(k_m x) + b_m sin(k_m x),  k_m = 2 pi m
    u(x) = sum_m (a_m sin(k_m x) - b_m cos(k_m x)) / k_m   (zero-mean, u' = f)
    """
    m = torch.arange(1, MAX_M + 1, dtype=torch.float64)
    k = 2 * math.pi * m
    cos = torch.cos(k * x.unsqueeze(-1))  # (Q, MAX_M)
    sin = torch.sin(k * x.unsqueeze(-1))
    f = a @ cos.T + b @ sin.T
    u = (a / k) @ sin.T - (b / k) @ cos.T
    return f, u


def _train_model(gen: torch.Generator, n_train: int = 60):
    x_s = half_open_grid(N_SENSORS)
    a = torch.randn(n_train, MAX_M, generator=gen, dtype=torch.float64)
    b = torch.randn(n_train, MAX_M, generator=gen, dtype=torch.float64)
    f_train, u_train = _fields(a, b, x_s)
    # target spectra straight from torch.fft (independent of library transform)
    u_coeff = torch.fft.fft(u_train.to(torch.complex128)).to(torch.complex64)
    model = SpectralSVR(
        FourierBasis(domain=(0.0, 1.0)),
        LSSVR(kernel="linear", C=1e6, dtype=torch.float64),
    )
    model.train(f_train, u_coeff)
    return model, x_s


@pytest.mark.oracle_analytic
@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_forward_matches_closed_form_antiderivative(seed):
    gen = torch.Generator().manual_seed(seed)
    model, x_s = _train_model(gen)

    a = torch.randn(3, MAX_M, generator=gen, dtype=torch.float64)
    b = torch.randn(3, MAX_M, generator=gen, dtype=torch.float64)
    xq = torch.rand(40, generator=gen, dtype=torch.float64)
    f_test, _ = _fields(a, b, x_s)
    _, u_expected = _fields(a, b, xq)

    pred = model.forward(f_test, xq.unsqueeze(-1)).real.double().cpu()
    # linear operator learned by a linear-kernel LSSVR: accuracy limited only
    # by regularization (1/C) and complex64 spectra; 1e-2 absolute on O(1)
    # fields is a fixed bound, not tuned
    assert torch.allclose(pred, u_expected, atol=1e-2), (
        f"max err {(pred - u_expected).abs().max()}"
    )


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    train_seed=st.integers(0, 10_000),
    inverse_seed=st.integers(0, 10_000),
    n_train=st.integers(20, 100),
)
@example(train_seed=0, inverse_seed=1, n_train=60)
def test_inverse_recovers_known_input(train_seed, inverse_seed, n_train):
    # inverse_coeff recovers the model's input features; with a complex-output
    # basis it interprets them as interleaved real/imag coefficient pairs, so
    # train this model on f SPECTRA (torch.fft of the closed-form values) and
    # anchor the recovered spectrum against the analytic one.
    gen = torch.Generator().manual_seed(train_seed)
    x_s = half_open_grid(N_SENSORS)
    a = torch.randn(n_train, MAX_M, generator=gen, dtype=torch.float64)
    b = torch.randn(n_train, MAX_M, generator=gen, dtype=torch.float64)
    f_train, u_train = _fields(a, b, x_s)
    f_feat = to_real_coeff(
        torch.fft.fft(f_train.to(torch.complex128)).to(torch.complex64)
    )
    u_coeff = torch.fft.fft(u_train.to(torch.complex128)).to(torch.complex64)
    model = SpectralSVR(
        FourierBasis(domain=(0.0, 1.0)),
        LSSVR(kernel="linear", C=1e6, dtype=torch.float64),
    )
    model.train(f_feat, u_coeff)

    # known input: a single analytic field, NOT produced by the model
    a0 = torch.zeros(1, MAX_M, dtype=torch.float64)
    b0 = torch.zeros(1, MAX_M, dtype=torch.float64)
    a0[0, 0], b0[0, 1] = 1.0, 0.5
    f_true, u_true = _fields(a0, b0, x_s)
    f_spec_true = torch.fft.fft(f_true.to(torch.complex128))
    u_coeff0 = torch.fft.fft(u_true.to(torch.complex128)).to(torch.complex64)

    f_rec = model.inverse_coeff(
        u_coeff0,
        epochs=800,
        generator=torch.Generator().manual_seed(inverse_seed),
        lr=0.5,
    )
    # error asserted against the analytic input spectrum, not a loss value;
    # scale-relative bound (spectra are O(N_SENSORS))
    err = (f_rec.cpu().to(torch.complex128) - f_spec_true).abs().max()
    assert err < 0.05 * f_spec_true.abs().max(), f"inverse recovery error {err}"
