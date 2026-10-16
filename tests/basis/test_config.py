"""Tests for SpectralConfig and the domain helpers it wraps."""

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from SpectralSVR.basis import SpectralConfig
from SpectralSVR.basis.domain import domainInputType_to_tuple
from SpectralSVR.basis.sampling import PeriodicUniform
from SpectralSVR.basis.strategy import DEFAULT_EVALUATION_STRATEGY

SETTINGS = settings(deadline=None, max_examples=60)


@st.composite
def intervals(draw):
    start = draw(st.floats(-100.0, 100.0))
    return (start, start + draw(st.floats(0.01, 100.0)))


MODE_COUNTS = st.lists(st.integers(1, 64), min_size=1, max_size=4).map(tuple)


@pytest.mark.oracle_smoke
@SETTINGS
@given(pair=intervals(), modes=MODE_COUNTS)
@example(pair=(-1.0, 1.0), modes=(8, 4))
def test_config_resolve_domain_broadcasts_single_pair(pair, modes):
    cfg = SpectralConfig(sampling=PeriodicUniform(), domain=pair)
    assert cfg.resolve_domain(modes) == (pair,) * len(modes)


@pytest.mark.oracle_smoke
@SETTINGS
@given(axes=st.lists(intervals(), min_size=1, max_size=4))
@example(axes=[(0.0, 2.0), (1.0, 4.0)])
def test_config_lengths(axes):
    cfg = SpectralConfig(sampling=PeriodicUniform(), domain=tuple(axes))
    assert cfg.lengths((8,) * len(axes)) == tuple(stop - start for start, stop in axes)


@pytest.mark.oracle_smoke
@SETTINGS
@given(first=intervals(), second=intervals())
@example(first=(0.0, 1.0), second=(-1.0, 1.0))
def test_config_with_domain_is_immutable_copy(first, second):
    cfg = SpectralConfig(sampling=PeriodicUniform(), domain=first)
    other = cfg.with_domain(second)
    assert cfg.domain == first
    assert other.domain == second
    assert other.sampling is cfg.sampling
    assert other.strategy is DEFAULT_EVALUATION_STRATEGY


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    bounds=st.lists(st.floats(-100.0, 100.0), min_size=1, max_size=6).filter(
        lambda bounds: len(bounds) != 2
    )
)
@example(bounds=[1.0, 2.0, 3.0])
def test_domain_malformed_pair_raises(bounds):
    with pytest.raises(ValueError, match="unpack"):
        domainInputType_to_tuple(tuple(bounds), (8,))


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.parametrize("bad", [(1.0, 1.0), (2.0, 1.0)])
def test_domain_degenerate_or_inverted_raises(bad):
    # zero-length or inverted intervals would divide by zero / flip frequency sign
    with pytest.raises(ValueError, match="positively-oriented interval"):
        domainInputType_to_tuple(bad, (8,))
