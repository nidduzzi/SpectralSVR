"""The band type's contract: which modes each form names, and which kept sets
cannot be named at all.

The union is closed, so enumerating its members is an exhaustive check rather
than a sample: no fourth form can promise a real result while keeping one
member of a +-k pair without the other.
"""

import typing

import pytest
import torch

import SpectralSVR
from SpectralSVR.utils import (
    Band,
    ExactModeCount,
    RealBand,
    WavenumberCutoff,
    WholeGrid,
    zero_coeff,
)

GRID_SIZES = (7, 8, 9, 12, 16)


def kept_frequencies(band, n):
    """Which frequencies survive band-limiting an all-ones grid of n bins.

    fft order runs [0, 1, ... , -floor(n/2), ... , -1], so on an even grid the
    bin at n/2 carries the negative boundary frequency, not a positive one.
    """
    out = zero_coeff(torch.ones(1, n, dtype=torch.complex64), band)[0]
    return {b if b < (n + 1) // 2 else b - n for b in range(n) if out[b] != 0}


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_the_forms_that_promise_a_real_result_are_exactly_two():
    assert set(typing.get_args(RealBand)) == {WavenumberCutoff, WholeGrid}
    assert set(typing.get_args(Band)) == {WavenumberCutoff, WholeGrid, ExactModeCount}


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_cutoff_keeps_two_k_plus_one_modes_symmetric_about_zero():
    for cutoff in range(8):
        assert WavenumberCutoff(cutoff).mode_count(64) == 2 * cutoff + 1
        kept = kept_frequencies(WavenumberCutoff(cutoff), 32)
        assert kept == set(range(-cutoff, cutoff + 1))
        assert kept == {-k for k in kept}


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_the_whole_grid_is_the_real_kept_set_no_cutoff_expresses():
    for n in GRID_SIZES:
        assert WholeGrid().mode_count(n) == n
    # on 16 bins the cutoffs give the odd sizes and reach 16 only by clamping,
    # where the boundary bin is its own conjugate partner
    reachable = sorted({WavenumberCutoff(k).mode_count(16) for k in range(16)})
    assert reachable == [1, 3, 5, 7, 9, 11, 13, 15, 16]


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_no_real_band_names_an_even_kept_set_short_of_the_grid():
    for n in GRID_SIZES:
        real_bands = [WavenumberCutoff(k) for k in range(n + 4)] + [WholeGrid()]
        for band in real_bands:
            kept = band.mode_count(n)
            assert kept % 2 == 1 or kept == n, f"n={n}, {band} keeps {kept}"


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_exact_count_keeps_the_lowest_frequencies_from_each_end():
    # fft order is [0, 1, ... , -floor(N/2), ... , -1], so keeping m modes takes
    # the ceil(m/2) lowest non-negative frequencies and the floor(m/2) highest
    # negative ones
    for n in GRID_SIZES:
        for m in range(n + 1):
            expected = {k for k in range((m + 1) // 2)} | {
                -k for k in range(1, m // 2 + 1)
            }
            assert kept_frequencies(ExactModeCount(m), n) == expected, f"n={n}, m={m}"


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_the_raw_form_states_the_guarantee_it_does_not_make():
    doc = ExactModeCount.__doc__ or ""
    assert "not guaranteed to be conjugate-symmetric" in doc
    assert "RealBand" in doc
    assert not any(
        promise in doc.lower() for promise in ("will be", "future", "for now", "soon")
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_a_band_cannot_be_stated_with_a_negative_width():
    with pytest.raises(ValueError):
        WavenumberCutoff(-1)
    with pytest.raises(ValueError):
        ExactModeCount(-1)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_exported_from_the_package_root():
    for name in ("Band", "RealBand", "WavenumberCutoff", "WholeGrid", "ExactModeCount"):
        assert hasattr(SpectralSVR, name), name
