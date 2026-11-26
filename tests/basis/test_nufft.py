"""Tests for the NUFFT-backed non-uniform evaluation path.

The NUFFT is validated against the exact dense (matmul) evaluation, which is
the ground truth here (not a manufactured PDE solution).
Kaiser-Bessel gridding is approximate, hence the loose tolerance.
"""

import importlib
from unittest.mock import patch

import pytest
import torch
import torchkbnufft
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from memory_profiler import memory_usage

from SpectralSVR import FourierBasis
from SpectralSVR.basis._nufft import nufft_available, nufft_evaluate
from SpectralSVR.basis.strategy import EvaluationStrategy

_EXACT = EvaluationStrategy(allow_approximate=False)
_FORCE_NUFFT = EvaluationStrategy(memory_budget_mb=0.0, allow_approximate=True)

CPU = torch.device("cpu")
BASIS_MODULE = importlib.import_module("SpectralSVR.basis.FourierBasis")
BUDGETS_MB = (0.0, 2**-10, 2**-8, 2**-6, 2**-4)
COMPLEX128_BYTES = 16
SETTINGS = settings(
    deadline=None, max_examples=20, suppress_health_check=[HealthCheck.too_slow]
)


def _exact_evaluate(coeff, x, domain):
    # exact dense path: disable the NUFFT for this call only
    return FourierBasis.evaluate(coeff, x, domain=domain, strategy=_EXACT)


@pytest.mark.oracle_independent
@SETTINGS
@given(
    modes=st.integers(1, 48),
    rows=st.integers(1, 3),
    npts=st.integers(20, 200),
    length=st.floats(0.5, 3.0),
    start=st.floats(-3.0, 3.0),
    seed=st.integers(0, 10_000),
)
@example(modes=1, rows=2, npts=20, length=1.0, start=0.0, seed=0)
@example(modes=2, rows=2, npts=20, length=1.0, start=-0.5, seed=0)
def test_nufft_matches_dense_1d(modes, rows, npts, length, start, seed):
    # ``start`` spans 0, so the NUFFT's (x - start) / L normalisation is checked
    # against the exact path on both canonical and shifted domains.
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(rows, modes, generator=g).to(torch.complex128)
    x = start + (torch.rand(npts, 1, generator=g, dtype=torch.float64) * length)
    domain = ((start, start + length),)
    approx = nufft_evaluate(coeff, x, domain)
    exact = _exact_evaluate(coeff, x, domain)
    assert (approx - exact).abs().max() < 5e-2 * exact.abs().max()


@pytest.mark.oracle_independent
@SETTINGS
@given(
    m=st.integers(1, 24),
    npts=st.integers(20, 120),
    seed=st.integers(0, 10_000),
)
@example(m=1, npts=20, seed=0)
@example(m=2, npts=20, seed=0)
def test_nufft_matches_dense_2d(m, npts, seed):
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(1, (m, m), generator=g).to(torch.complex128)
    x = torch.rand(npts, 2, generator=g, dtype=torch.float64) * torch.tensor([1.3, 0.7])
    domain = ((0.0, 1.3), (0.0, 0.7))
    approx = nufft_evaluate(coeff, x, domain)
    exact = _exact_evaluate(coeff, x, domain)
    assert (approx - exact).abs().max() < 5e-2 * exact.abs().max()


MODE_TUPLES = st.lists(st.integers(1, 12), min_size=1, max_size=3).map(tuple)
SMALL_AXIS_CASES = ((1,), (2,), (2, 5), (3, 1, 2))
APPROXIMATION_BOUND = 5e-2


def _field_and_points(modes, rows, npts, seed):
    generator = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(rows, modes, generator=generator).to(
        torch.complex128
    )
    points = torch.rand(npts, len(modes), generator=generator, dtype=torch.float64)
    return coeff, points, tuple((0.0, 1.0) for _ in modes)


@pytest.mark.oracle_smoke
@pytest.mark.scenario("A field with a one- or two-mode axis evaluates approximately")
@SETTINGS
@given(
    modes=MODE_TUPLES,
    rows=st.integers(1, 3),
    npts=st.integers(1, 50),
    seed=st.integers(0, 10_000),
)
@example(modes=SMALL_AXIS_CASES[0], rows=2, npts=5, seed=0)
@example(modes=SMALL_AXIS_CASES[1], rows=2, npts=5, seed=0)
@example(modes=SMALL_AXIS_CASES[2], rows=2, npts=5, seed=0)
@example(modes=SMALL_AXIS_CASES[3], rows=2, npts=5, seed=0)
def test_every_mode_count_evaluates_approximately(modes, rows, npts, seed):
    coeff, points, domain = _field_and_points(modes, rows, npts, seed)
    values = FourierBasis.evaluate(coeff, points, domain=domain, strategy=_FORCE_NUFFT)
    assert values.shape == (rows, npts)


@pytest.mark.oracle_independent
@pytest.mark.scenario("Approximate and exact values agree at every mode count")
@SETTINGS
@given(
    modes=MODE_TUPLES,
    rows=st.integers(1, 3),
    npts=st.integers(1, 50),
    seed=st.integers(0, 10_000),
)
@example(modes=SMALL_AXIS_CASES[0], rows=2, npts=5, seed=0)
@example(modes=SMALL_AXIS_CASES[1], rows=2, npts=5, seed=0)
@example(modes=SMALL_AXIS_CASES[2], rows=2, npts=5, seed=0)
@example(modes=SMALL_AXIS_CASES[3], rows=2, npts=5, seed=0)
def test_approximate_and_exact_agree_at_every_mode_count(modes, rows, npts, seed):
    coeff, points, domain = _field_and_points(modes, rows, npts, seed)
    approx = FourierBasis.evaluate(coeff, points, domain=domain, strategy=_FORCE_NUFFT)
    exact = _exact_evaluate(coeff, points, domain)
    assert (approx - exact).abs().max() <= APPROXIMATION_BOUND * exact.abs().max()


@pytest.mark.oracle_smoke
@SETTINGS
@given(modes=MODE_TUPLES, seed=st.integers(0, 10_000))
@example(modes=(3,), seed=0)
@example(modes=(2, 7), seed=0)
def test_only_axes_below_the_kernel_width_get_a_wider_grid(modes, seed):
    coeff, points, domain = _field_and_points(modes, 1, 5, seed)
    with patch.object(torchkbnufft, "KbNufft", wraps=torchkbnufft.KbNufft) as operator:
        nufft_evaluate(coeff, points, domain)
    kernel_width = 6
    assert operator.call_args.kwargs["numpoints"] == kernel_width
    assert operator.call_args.kwargs["grid_size"] == tuple(
        2 * m if 2 * m >= kernel_width else kernel_width for m in modes
    )


@pytest.mark.oracle_independent
@SETTINGS
@given(
    modes=st.integers(1, 48),
    budget_mb=st.sampled_from(BUDGETS_MB),
    around_threshold=st.integers(-20, 20),
    seed=st.integers(0, 10_000),
)
@example(modes=32, budget_mb=0.0, around_threshold=2000, seed=0)
@example(modes=2, budget_mb=0.0, around_threshold=2000, seed=0)
def test_evaluate_switches_to_nufft_above_threshold(
    modes, budget_mb, around_threshold, seed
):
    threshold_points = int(budget_mb * 2**20) // (modes * COMPLEX128_BYTES)
    npts = max(1, threshold_points + around_threshold)
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(1, modes, generator=g).to(torch.complex128)
    x = torch.rand(npts, dtype=torch.float64, generator=g) * 1.3

    exact = FourierBasis(coeff, domain=(0.0, 1.3), strategy=_EXACT)(x, device=CPU)
    with patch.object(
        BASIS_MODULE, "nufft_evaluate", wraps=BASIS_MODULE.nufft_evaluate
    ) as nufft:
        evaluated = FourierBasis(
            coeff,
            domain=(0.0, 1.3),
            strategy=EvaluationStrategy(memory_budget_mb=budget_mb),
        )(x, device=CPU)

    assert (nufft.call_count > 0) == (
        npts * modes * COMPLEX128_BYTES > budget_mb * 2**20
    )
    assert (evaluated - exact).abs().max() < 5e-2 * exact.abs().max()


@pytest.mark.oracle_consistency
@SETTINGS
@given(
    modes=st.integers(4, 32),
    rows=st.integers(1, 3),
    npts=st.integers(1, 200),
    seed=st.integers(0, 10_000),
)
def test_dense_evaluate_chunking_is_exact(modes, rows, npts, seed):
    # Chunking the exact path must not change the result vs a single block.
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(rows, modes, generator=g).to(torch.complex128)
    x = torch.rand(npts, generator=g, dtype=torch.float64)

    # exact matmul (inf threshold); one block vs many small chunks -- per instance
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


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_dense_evaluate_chunk_bounds_memory():
    # The exact (matmul) path, chunked, must also stay well under the full
    # dense (npts x modes) basis matrix.
    torch.set_default_dtype(torch.float64)
    try:
        modes = 128
        npts = 400_000
        dense_bytes = npts * modes * 16
        coeff = FourierBasis.generate_coeff(1, modes).to(torch.complex128).cpu()
        x = (torch.rand(npts, dtype=torch.float64)).cpu()
        # exact path (inf threshold), ~1M-entry chunks -- per instance
        basis = FourierBasis(
            coeff,
            domain=(0.0, 1.0),
            strategy=EvaluationStrategy(allow_approximate=False, memory_budget_mb=16.0),
        )
        holder: dict = {}

        def run():
            holder["out"] = basis(x, device=CPU)

        baseline = memory_usage(-1, max_usage=True)
        peak = memory_usage((run, (), {}), max_usage=True, interval=0.02)
        delta_bytes = max(0.0, peak - baseline) * 1024 * 1024
        assert holder["out"].shape == (1, npts)
        assert delta_bytes < 0.5 * dense_bytes
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.oracle_independent
@SETTINGS
@given(mode=st.integers(1, 40), big=st.integers(400, 2000), seed=st.integers(0, 10_000))
@example(mode=2, big=400, seed=0)
def test_inv_transform_large_res_matches_dense(mode, big, seed):
    # inv_transform at a resolution far above the mode count hits the dense
    # (res x mode) path; forcing the NUFFT there must match the exact matmul.
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(1, mode, generator=g).to(torch.complex128)
    res = slice(0, 1, big)

    exact = FourierBasis.inv_transform(coeff, res=res, strategy=_EXACT)
    approx = FourierBasis.inv_transform(coeff, res=res, strategy=_FORCE_NUFFT)
    assert (approx - exact).abs().max() < 5e-2 * exact.abs().max()


@pytest.mark.oracle_smoke
@settings(deadline=None, max_examples=4, suppress_health_check=[HealthCheck.too_slow])
@given(
    mode=st.sampled_from([64, 128]),
    res=st.sampled_from([200_000, 400_000]),
    seed=st.integers(0, 10_000),
)
def test_inv_transform_nufft_memory_bounded(mode, res, seed):
    # The NUFFT inverse path must not materialize the dense (res x mode) basis
    # matrix. Force the NUFFT and check the peak RSS stays well under it.
    torch.set_default_dtype(torch.float64)
    try:
        g = torch.Generator().manual_seed(seed)
        coeff = FourierBasis.generate_coeff(1, mode, generator=g).to(torch.complex128)
        dense_bytes = res * mode * 16
        holder: dict = {}

        def run():
            holder["out"] = FourierBasis.inv_transform(
                coeff, res=slice(0, 1, res), strategy=_FORCE_NUFFT
            )

        baseline = memory_usage(-1, max_usage=True)
        peak = memory_usage((run, (), {}), max_usage=True, interval=0.02)
        delta_bytes = max(0.0, peak - baseline) * 1024 * 1024
        assert holder["out"].shape == (1, res)
        assert not torch.isnan(holder["out"]).any()
        assert delta_bytes < 0.5 * dense_bytes
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_nufft_available():
    assert nufft_available() is True


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    rows=st.integers(1, 4),
    modes=st.integers(4, 32),
    npts=st.integers(1, 120),
    dtype=st.sampled_from((torch.complex64, torch.complex128)),
)
@example(rows=3, modes=16, npts=50, dtype=torch.complex64)
def test_nufft_evaluate_shape(rows, modes, npts, dtype):
    coeff = FourierBasis.generate_coeff(rows, modes).to(dtype)
    x = torch.rand(npts, 1, dtype=coeff.real.dtype)
    out = nufft_evaluate(coeff, x, ((0.0, 1.0),))
    assert out.shape == (rows, npts)
    assert out.is_complex()


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_nufft_evaluate_does_not_materialize_dense_matrix():
    # The bottleneck the NUFFT fixes: the dense path builds a (npts x modes)
    # complex basis matrix. Here that would be ~0.8 GB; the NUFFT must stay far
    # below that. Run on CPU and measure the peak RSS increase.
    modes = 128
    npts = 400_000
    dense_bytes = npts * modes * 16  # complex128 basis matrix (~0.8 GiB)

    coeff = FourierBasis.generate_coeff(1, modes).to(torch.complex128).cpu()
    x = (torch.rand(npts, 1, dtype=torch.float64) * 1.0).cpu()

    holder: dict = {}

    def run():
        holder["out"] = nufft_evaluate(coeff, x, ((0.0, 1.0),))

    baseline = memory_usage(-1, max_usage=True)
    peak = memory_usage((run, (), {}), max_usage=True, interval=0.02)
    delta_bytes = max(0.0, (peak - baseline)) * 1024 * 1024  # MiB -> bytes

    assert holder["out"].shape == (1, npts)
    assert not torch.isnan(holder["out"]).any()
    # comfortably under the dense matrix that the matmul path would allocate
    assert delta_bytes < 0.5 * dense_bytes, (
        f"NUFFT peak delta {delta_bytes / 1e6:.0f} MB exceeded half the dense "
        f"matrix {dense_bytes / 1e6:.0f} MB"
    )
