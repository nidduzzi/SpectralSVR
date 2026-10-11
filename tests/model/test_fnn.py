"""Fuzzed + serialization tests for FNN."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from SpectralSVR.model.FNN import FNN

from _model_data import SETTINGS, constant_dataset, linear_dataset


@pytest.mark.oracle_consistency
@SETTINGS
@given(const=st.floats(-3.0, 3.0), seed=st.integers(0, 10_000))
def test_fnn_overfits_constant_target(const, seed):
    # Manufactured constant target; a small FNN trained long enough recovers it.
    X, y = constant_dataset(32, 3, 2, const, seed)
    model = FNN(MAX_EPOCH=400, w_hidden=16, n_hidden=2, batch_size=32, lr=0.01)
    model.fit(X, y)
    assert torch.allclose(model.predict(X), y, atol=1e-1)


@pytest.mark.oracle_smoke
@SETTINGS
@given(
    n=st.integers(8, 24),
    d=st.integers(1, 4),
    o=st.integers(1, 3),
    seed=st.integers(0, 10_000),
)
def test_fnn_predict_shape(n, d, o, seed):
    X, y = linear_dataset(n, d, o, seed)
    model = FNN(MAX_EPOCH=2, w_hidden=8, n_hidden=1, batch_size=8)
    model.fit(X, y)
    assert model.predict(X).shape == (n, o)

