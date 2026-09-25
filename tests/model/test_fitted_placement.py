from functools import partial

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from SpectralSVR import LSSVR, accelerator
from SpectralSVR.model.FNN import FNN

from _model_data import SETTINGS, linear_dataset

EVERY_KIND = [
    partial(LSSVR, kernel="linear"),
    partial(FNN, MAX_EPOCH=1, w_hidden=4, n_hidden=1, batch_size=8),
]
X, Y = linear_dataset(16, 3, 2, seed=0)
HOST = torch.device("cpu")
DEVICES_PRESENT = [HOST] + ([accelerator()] if torch.accelerator.is_available() else [])

needs_a_second_device = pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="naming both devices needs a device other than the host to move to",
)


@needs_a_second_device
@pytest.mark.oracle_smoke
@pytest.mark.scenario("A fitted model refuses to be moved")
@SETTINGS
@given(kind=st.sampled_from(EVERY_KIND))
def test_a_fitted_model_refuses_to_be_moved(kind):
    model = kind(device=HOST).fit(X, Y)
    elsewhere = accelerator()

    with pytest.raises(RuntimeError) as refused:
        model.device = elsewhere

    message = str(refused.value)
    assert f"fitted on {HOST}" in message
    assert f"cannot move to {elsewhere}" in message
    assert f"{type(model).__name__}(device='{elsewhere}')" in message
    assert model.device == HOST


@needs_a_second_device
@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("Every kind of model refuses alike")
def test_every_kind_of_model_refuses_alike():
    refusals = set()
    for kind in EVERY_KIND:
        model = kind(device=HOST).fit(X, Y)
        with pytest.raises(RuntimeError) as refused:
            model.device = accelerator()
        refusals.add(str(refused.value).replace(type(model).__name__, "<kind>"))

    assert len(refusals) == 1


@pytest.mark.oracle_smoke
@pytest.mark.scenario("An unfitted model is directed freely")
@SETTINGS
@given(
    kind=st.sampled_from(EVERY_KIND),
    target=st.sampled_from(DEVICES_PRESENT),
    stated_as_text=st.booleans(),
)
def test_an_unfitted_model_is_directed_freely(kind, target, stated_as_text):
    model = kind(device=HOST)

    model.device = str(target) if stated_as_text else target

    assert model.device == target


class DirectingAndFitting(RuleBasedStateMachine):
    @initialize(
        kind=st.sampled_from(EVERY_KIND), device=st.sampled_from(DEVICES_PRESENT)
    )
    def build(self, kind, device):
        self.model = kind(device=device)

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def direct(self, device):
        if self.model.trained:
            with pytest.raises(RuntimeError):
                self.model.device = device
        else:
            self.model.device = device

    @rule()
    def fit(self):
        self.model.fit(X, Y)

    @invariant()
    def a_fitted_model_predicts_on_the_device_it_reports(self):
        if self.model.trained:
            assert self.model.predict(X).shape == Y.shape


DirectingAndFitting.TestCase.settings = settings(
    deadline=None, max_examples=50, stateful_step_count=8
)
TestDirectingAndFitting = pytest.mark.oracle_invariant(DirectingAndFitting.TestCase)
