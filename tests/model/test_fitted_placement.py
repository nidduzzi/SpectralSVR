import contextlib
import tempfile
from functools import partial
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from SpectralSVR import LSSVR, accelerator
from SpectralSVR.model.FNN import FNN

from tests._devices import ACCELERATOR, DEVICES_PRESENT, HOST, needs_an_accelerator
from _model_data import SETTINGS, linear_dataset

EVERY_KIND = [
    partial(LSSVR, kernel="linear"),
    partial(LSSVR, kernel="rbf"),
    partial(FNN, MAX_EPOCH=1, w_hidden=4, n_hidden=1, batch_size=8),
]
X, Y = linear_dataset(16, 3, 2, seed=0)
SPELLINGS_OF_ONE_MEMORY = [(HOST, HOST)] + (
    [(accelerator(), ACCELERATOR), (ACCELERATOR, accelerator())]
    if ACCELERATOR != HOST
    else []
)
LOADED_ACROSS_DEVICES_ATOL = 1e-6
HOST_SOLVE_NOISE_ATOL = 1e-5
INPUTS_NO_MODEL_CAN_USE = torch.randn(
    16, 3, 2, generator=torch.Generator().manual_seed(0)
)


@needs_an_accelerator
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
    assert f"{type(model).__name__}.load(..., device='{elsewhere}')" in message
    assert model.device == HOST


@needs_an_accelerator
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


@pytest.mark.oracle_invariant
@pytest.mark.scenario("An unfitted model is directed freely")
@SETTINGS
@given(
    kind=st.sampled_from(EVERY_KIND),
    target=st.sampled_from(DEVICES_PRESENT),
    stated_as_text=st.booleans(),
)
def test_an_unfitted_model_is_directed_freely(kind, target, stated_as_text):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        directed = kind(device=HOST)
        directed.device = str(target) if stated_as_text else target
        predicted_where_directed = directed.fit(X, Y).predict(X)
        torch.manual_seed(0)
        predicted_where_built = kind(device=target).fit(X, Y).predict(X)

    assert directed.device == target
    assert torch.allclose(
        predicted_where_directed,
        predicted_where_built,
        rtol=0,
        atol=HOST_SOLVE_NOISE_ATOL,
    )


@pytest.mark.oracle_invariant
@pytest.mark.scenario("A move to where the model already is is accepted")
@SETTINGS
@given(
    kind=st.sampled_from(EVERY_KIND), spellings=st.sampled_from(SPELLINGS_OF_ONE_MEMORY)
)
def test_a_move_to_where_the_model_already_is_is_accepted(kind, spellings):
    fitted_on, directed_to = spellings
    model = kind(device=fitted_on).fit(X, Y)
    predicted = model.predict(X)

    model.device = directed_to

    assert torch.equal(model.predict(X), predicted)


@pytest.mark.filterwarnings(
    "ignore:The use of `x.T` on tensors of dimension other than 2:UserWarning"
)
@pytest.mark.oracle_invariant
@pytest.mark.scenario("A fit that fails leaves the model as it was")
@SETTINGS
@given(kind=st.sampled_from(EVERY_KIND), fitted_before=st.booleans())
def test_a_fit_that_fails_leaves_the_model_as_it_was(kind, fitted_before):
    model = kind(device=HOST)
    if fitted_before:
        model.fit(X, Y)
    predicted_before = model.predict(X) if fitted_before else None

    with pytest.raises(RuntimeError):
        model.fit(INPUTS_NO_MODEL_CAN_USE, Y)

    assert model.trained == fitted_before
    if fitted_before:
        assert torch.equal(model.predict(X), predicted_before)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            after_the_failure = model.fit(X, Y).predict(X)
            torch.manual_seed(0)
            never_failed = kind(device=HOST).fit(X, Y).predict(X)
        assert torch.allclose(
            after_the_failure, never_failed, rtol=0, atol=HOST_SOLVE_NOISE_ATOL
        )


@pytest.mark.oracle_consistency
@pytest.mark.scenario("A loaded model is placed where the caller states")
@SETTINGS
@given(
    kind=st.sampled_from(EVERY_KIND),
    dumped_on=st.sampled_from(DEVICES_PRESENT),
    loaded_to=st.sampled_from(DEVICES_PRESENT),
)
def test_a_loaded_model_is_placed_where_the_caller_states(kind, dumped_on, loaded_to):
    model = kind(device=dumped_on).fit(X, Y)
    with tempfile.TemporaryDirectory() as directory:
        saved = str(Path(directory) / "model")
        model.dump(saved)
        loaded = type(model).load(saved, device=loaded_to)

    assert loaded.device == loaded_to
    assert torch.allclose(
        loaded.predict(X), model.predict(X), atol=LOADED_ACROSS_DEVICES_ATOL
    )


class DirectingAndFitting(RuleBasedStateMachine):
    @initialize(
        kind=st.sampled_from(EVERY_KIND),
        device=st.sampled_from(DEVICES_PRESENT),
        fitted=st.booleans(),
    )
    def build(self, kind, device, fitted):
        self.model = kind(device=device)
        if fitted:
            self.fit()

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def direct(self, device):
        if self.model.trained:
            with contextlib.suppress(RuntimeError):
                self.model.device = device
        else:
            self.model.device = device
            assert self.model.device == device

    @rule()
    def fit(self):
        self.model.fit(X, Y)
        self.fitted_on = self.model.device
        self.predicted = self.model.predict(X)

    @invariant()
    def a_fitted_model_stays_where_it_was_fitted(self):
        if self.model.trained:
            assert self.model.device == self.fitted_on

    @invariant()
    def directing_never_changes_what_a_fitted_model_predicts(self):
        if self.model.trained:
            assert torch.equal(self.model.predict(X), self.predicted)


DirectingAndFitting.TestCase.settings = settings(
    deadline=None, max_examples=50, stateful_step_count=8
)
TestDirectingAndFitting = pytest.mark.oracle_invariant(DirectingAndFitting.TestCase)
