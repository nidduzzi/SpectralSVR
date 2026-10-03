"""Per-axis sampling: broadcast helper + mixed-scheme transform dispatch."""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from SpectralSVR import FourierBasis
from SpectralSVR.basis import SpectralConfig
from SpectralSVR.basis.sampling import (
    ClosedUniform,
    PeriodicUniform,
    samplings_to_tuple,
)

CPU = torch.device("cpu")
SETTINGS = settings(
    deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow]
)

# a scheme per drawn flag: True -> periodic (half-open), False -> closed
_schemes = st.lists(st.booleans(), min_size=1, max_size=4).map(
    lambda flags: [PeriodicUniform() if p else ClosedUniform() for p in flags]
)


@pytest.mark.oracle_smoke
@SETTINGS
@given(ndim=st.integers(1, 4), fft=st.booleans())
def test_samplings_to_tuple_broadcast_and_default(ndim, fft):
    default = PeriodicUniform()
    one = PeriodicUniform(supports_fft=fft)
    # None -> default on every axis
    assert samplings_to_tuple(None, ndim, default) == tuple(default for _ in range(ndim))
    # single scheme -> broadcast to every axis (same object)
    assert samplings_to_tuple(one, ndim, default) == tuple(one for _ in range(ndim))


@pytest.mark.oracle_smoke
@SETTINGS
@given(schemes=_schemes, as_tuple=st.booleans())
def test_samplings_to_tuple_per_axis_sequence(schemes, as_tuple):
    seq = tuple(schemes) if as_tuple else schemes
    assert samplings_to_tuple(seq, len(schemes), PeriodicUniform()) == tuple(schemes)


@pytest.mark.oracle_smoke
@SETTINGS
@given(schemes=_schemes)
def test_config_resolve_sampling(schemes):
    cfg = SpectralConfig(sampling=schemes, domain=None)
    assert cfg.resolve_sampling(len(schemes), PeriodicUniform()) == tuple(schemes)


@pytest.mark.oracle_consistency
@SETTINGS
@given(
    n=st.integers(2, 10),
    fft0=st.booleans(),
    fft1=st.booleans(),
    seed=st.integers(0, 10_000),
)
def test_per_axis_sampling_transform_matches_uniform(n, fft0, fft1, seed):
    # Two periodic axes, each independently on the FFT or the matmul path via
    # per-axis schemes -> result matches the all-FFT single-scheme case.
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.transform(
        torch.randn(1, n, n, dtype=torch.complex128, generator=g)
    )
    mixed = FourierBasis.inv_transform(
        coeff,
        sampling=[
            PeriodicUniform(supports_fft=fft0),
            PeriodicUniform(supports_fft=fft1),
        ],
    )
    uniform = FourierBasis.inv_transform(coeff, sampling=PeriodicUniform())
    # the explicit matmul path accumulates more float error than the FFT
    assert torch.allclose(mixed, uniform, atol=1e-6, rtol=1e-6)


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    m=st.integers(1, 6),
    amp=st.floats(0.5, 2.0),
    phase=st.floats(0, 6.283),
)
def test_matmul_transform_leg_matches_closed_form(m, amp, phase):
    # Anchor the non-FFT (explicit matmul) transform leg DIRECTLY against the
    # hand-computed DFT of a single-mode sinusoid -- previously this leg was
    # only ever compared to the FFT leg, so a shared convention error was
    # invisible.
    import math

    from tests._anchors import single_mode_spectrum

    n = 16
    x = torch.linspace(0, 1, n + 1, dtype=torch.float64)[:-1]
    values = amp * torch.cos(2 * math.pi * m * x + phase)
    got = FourierBasis.transform(
        (values + 0j).unsqueeze(0), sampling=PeriodicUniform(supports_fft=False)
    )[0]
    expected = single_mode_spectrum(n, m, amp, phase).to(got.dtype)
    assert torch.allclose(got, expected, atol=2e-3 * n * amp)


@pytest.mark.oracle_smoke
@SETTINGS
@given(ndim=st.integers(1, 4), length=st.integers(0, 6))
def test_a_per_axis_sampling_names_every_axis(ndim, length):
    schemes = tuple(PeriodicUniform() for _ in range(length))
    if length == ndim:
        assert samplings_to_tuple(schemes, ndim, ClosedUniform()) == schemes
        return
    with pytest.raises(ValueError, match=f"got {length} schemes for {ndim} axes$"):
        samplings_to_tuple(schemes, ndim, ClosedUniform())
