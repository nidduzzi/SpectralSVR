"""The band type's contract: which modes each form names, which kept sets
cannot be named at all, and what survives a sequence of band-limitings.

The union is closed, so drawing from ``ANY_BAND`` covers every form there is:
no fourth form can promise a real result while keeping one member of a +-k pair
without the other. The ``@example`` cases pin the measurements the change was
argued from, so a shrinking search can never quietly drop them.
"""

import typing

import pytest
import torch
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

import SpectralSVR
from SpectralSVR.utils import (
    Band,
    ExactModeCount,
    RealBand,
    WavenumberCutoff,
    WholeGrid,
    zero_coeff,
)
from tests._anchors import hermitian_error, hermitian_spectrum

SETTINGS = settings(deadline=None, max_examples=50)

GRID = st.integers(4, 20)
SYMMETRIC_GRID = st.sampled_from((4, 5, 8, 9, 12, 16))
SEED = st.integers(0, 10_000)
REAL_BAND = st.one_of(
    st.builds(WavenumberCutoff, st.integers(0, 12)), st.just(WholeGrid())
)
ANY_BAND = st.one_of(REAL_BAND, st.builds(ExactModeCount, st.integers(0, 24)))

REAL_FORMS = typing.get_args(RealBand)


def kept_bins(band, n):
    """Which bins of an all-ones grid of n bins survive band-limiting."""
    out = zero_coeff(torch.ones(1, n, dtype=torch.complex64), band)[0]
    return {b for b in range(n) if out[b] != 0}


def kept_frequencies(band, n):
    """The same set as signed frequencies.

    fft order runs [0, 1, ... , -floor(n/2), ... , -1], so on an even grid the
    bin at n/2 carries the negative boundary frequency, not a positive one.
    """
    return {b if b < (n + 1) // 2 else b - n for b in kept_bins(band, n)}


def named_bins(band, n):
    """The bins each form names, derived by hand from the fft layout.

    Independent of ``zero_coeff`` and of ``mode_count``: a band of ``m`` modes
    takes the ``ceil(m/2)`` bins from the start of fft order and the
    ``floor(m/2)`` from its end, and each form says what ``m`` is.
    """
    match band:
        case WholeGrid():
            modes = n
        case WavenumberCutoff(cutoff=cutoff):
            modes = min(2 * cutoff + 1, n)
        case ExactModeCount(modes=exact):
            modes = min(exact, n)
    return set(range((modes + 1) // 2)) | {n - 1 - k for k in range(modes // 2)}


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_the_forms_that_promise_a_real_result_are_exactly_two():
    assert set(REAL_FORMS) == {WavenumberCutoff, WholeGrid}
    assert set(typing.get_args(Band)) == {WavenumberCutoff, WholeGrid, ExactModeCount}


@pytest.mark.oracle_analytic
@SETTINGS
@given(n=GRID, band=ANY_BAND)
@example(n=16, band=WavenumberCutoff(2))
@example(n=16, band=WholeGrid())
@example(n=8, band=ExactModeCount(4))
def test_every_band_keeps_the_bins_its_form_names(n, band):
    named = named_bins(band, n)
    assert kept_bins(band, n) == named
    assert band.mode_count(n) == len(named)
    assert 0 <= band.mode_count(n) <= n


@pytest.mark.oracle_analytic
@SETTINGS
@given(n=GRID, cutoff=st.integers(0, 12))
@example(n=8, cutoff=2)  # the restated zero_coeff anchor's band
def test_a_cutoff_keeps_two_k_plus_one_modes_until_the_grid_runs_out(n, cutoff):
    band = WavenumberCutoff(cutoff)
    assert band.mode_count(2 * cutoff + 1) == 2 * cutoff + 1
    if 2 * cutoff + 1 <= n:
        assert kept_frequencies(band, n) == set(range(-cutoff, cutoff + 1))
    else:
        assert kept_bins(band, n) == set(range(n))


@pytest.mark.oracle_analytic
@SETTINGS
@given(n=GRID, modes=st.integers(0, 24))
@example(n=8, modes=4)  # the count the pre-change anchor pinned
@example(n=8, modes=2)  # same numeral as the cutoff above, different band
def test_exact_count_keeps_the_lowest_frequencies_from_each_end(n, modes):
    kept = min(modes, n)
    expected = {k for k in range((kept + 1) // 2)} | {
        -k for k in range(1, kept // 2 + 1)
    }
    assert kept_frequencies(ExactModeCount(modes), n) == expected


@pytest.mark.oracle_invariant
@SETTINGS
@given(n=GRID, band=REAL_BAND)
@example(n=16, band=WavenumberCutoff(2))
def test_a_real_band_keeps_a_set_closed_under_negation(n, band):
    kept = kept_bins(band, n)
    assert {(-b) % n for b in kept} == kept
    size = band.mode_count(n)
    assert size % 2 == 1 or size == n


@pytest.mark.oracle_analytic
@SETTINGS
@given(n=st.integers(4, 32).filter(lambda n: n % 2 == 0))
@example(n=16)
def test_the_whole_grid_is_the_real_kept_set_no_cutoff_expresses(n):
    # on an even grid the boundary bin is its own conjugate partner, so keeping
    # everything is real -- and 2K+1 is odd, so only clamping reaches it
    assert WholeGrid().mode_count(n) == n
    unclamped = {2 * k + 1 for k in range(n)}
    assert n not in unclamped
    assert {WavenumberCutoff(k).mode_count(n) for k in range(n)} == {
        size for size in unclamped if size < n
    } | {n}


@pytest.mark.oracle_invariant
@SETTINGS
@given(n=SYMMETRIC_GRID, band=REAL_BAND, seed=SEED)
@example(n=16, band=WavenumberCutoff(2), seed=16)
def test_a_real_band_preserves_conjugate_symmetry(n, band, seed):
    source = hermitian_spectrum(n, seed)
    assert hermitian_error(source) == 0.0
    assert hermitian_error(zero_coeff(source, band)) == 0.0


@pytest.mark.oracle_invariant
@SETTINGS
@given(n=SYMMETRIC_GRID, half=st.integers(1, 12), seed=SEED)
@example(n=16, half=2, seed=16)  # the proposal's keep-4 measurement
@example(n=16, half=3, seed=16)  # keep-6
@example(n=16, half=4, seed=16)  # keep-8
def test_an_even_exact_count_cannot_describe_a_real_field(n, half, seed):
    # this is what makes the symmetry invariant above discriminating rather
    # than a restatement: the same widths stated raw are not conjugate-symmetric
    assume(0 < 2 * half < n)
    source = hermitian_spectrum(n, seed)
    assert hermitian_error(zero_coeff(source, ExactModeCount(2 * half))) > 0.0


@pytest.mark.oracle_invariant
@SETTINGS
@given(n=GRID, band=ANY_BAND, seed=SEED)
def test_band_limiting_is_idempotent(n, band, seed):
    x = torch.randn(
        2, n, dtype=torch.complex64, generator=torch.Generator().manual_seed(seed)
    )
    once = zero_coeff(x, band)
    assert torch.equal(zero_coeff(once, band), once)


@pytest.mark.oracle_invariant
@SETTINGS
@given(n=GRID, a=st.integers(0, 12), b=st.integers(0, 12))
def test_a_wider_cutoff_keeps_a_superset(n, a, b):
    narrow, wide = min(a, b), max(a, b)
    assert kept_bins(WavenumberCutoff(narrow), n) <= kept_bins(WavenumberCutoff(wide), n)


@pytest.mark.oracle_invariant
@SETTINGS
@given(n=GRID, over=st.integers(0, 8), seed=SEED)
@example(n=16, over=0, seed=0)  # zero_coeff(x, 16) raised IndexError before
@example(n=7, over=1, seed=0)
def test_keeping_everything_is_a_no_op(n, over, seed):
    x = torch.randn(
        1, n, dtype=torch.complex64, generator=torch.Generator().manual_seed(seed)
    )
    for band in (WholeGrid(), WavenumberCutoff(n + over), ExactModeCount(n + over)):
        assert torch.equal(zero_coeff(x, band), x), band


@pytest.mark.oracle_smoke
@SETTINGS
@given(width=st.integers(max_value=-1))
def test_a_band_cannot_be_stated_with_a_negative_width(width):
    with pytest.raises(ValueError):
        WavenumberCutoff(width)
    with pytest.raises(ValueError):
        ExactModeCount(width)


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
def test_exported_from_the_package_root():
    for name in ("Band", "RealBand", "WavenumberCutoff", "WholeGrid", "ExactModeCount"):
        assert hasattr(SpectralSVR, name), name


class BandLimiting(RuleBasedStateMachine):
    """Band-limitings compose to the intersection of their kept sets.

    Applying bands one after another must land on the same grid as applying
    their intersection once, in any order -- and while every band applied so far
    guarantees a real result, the grid stays conjugate-symmetric. A single
    ``ExactModeCount`` in the sequence can break that and never repairs it.
    """

    MODES = 16

    def __init__(self):
        super().__init__()
        self.source = hermitian_spectrum(self.MODES, seed=16)
        assert self.source.count_nonzero() == self.MODES
        self.grid = self.source.clone()
        self.expected = set(range(self.MODES))
        self.every_band_guaranteed_real = True

    @rule(band=ANY_BAND)
    def band_limit(self, band):
        self.grid = zero_coeff(self.grid, band)
        self.expected &= named_bins(band, self.MODES)
        self.every_band_guaranteed_real &= isinstance(band, REAL_FORMS)

    @invariant()
    def the_surviving_bins_are_the_intersection_so_far(self):
        surviving = {b for b in range(self.MODES) if self.grid[0, b] != 0}
        assert surviving == self.expected

    @invariant()
    def surviving_coefficients_keep_their_values(self):
        directly = self.source.clone()
        for bin_index in range(self.MODES):
            if bin_index not in self.expected:
                directly[0, bin_index] = 0
        assert torch.equal(self.grid, directly)

    @invariant()
    def real_bands_alone_never_break_conjugate_symmetry(self):
        if self.every_band_guaranteed_real:
            assert hermitian_error(self.grid) == 0.0


BandLimiting.TestCase.settings = settings(
    deadline=None, max_examples=25, stateful_step_count=8
)
TestBandLimiting = pytest.mark.oracle_invariant(BandLimiting.TestCase)
