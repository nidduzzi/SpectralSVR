import importlib.util
import itertools

import pytest
import torch
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from SpectralSVR import Antiderivative, FourierBasis
from SpectralSVR.operators import DeepONetOperator, Split
from tests._devices import DEVICES_PRESENT

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("deepxde") is None,
    reason="needs the 'baselines' extra (deepxde)",
)

MODES = 8
QUERY = torch.linspace(0, 1, 12, dtype=torch.float64).unsqueeze(-1)
ITERATIONS = 20


def antiderivative_split(seed: int) -> Split:
    generator = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(
        FourierBasis, 16, MODES, generator=generator, u0=0
    )
    return Split.partition(ut, u, 12, n_sensors=MODES)


SPLITS = {seed: antiderivative_split(seed) for seed in (0, 1)}


class DeepONetFits(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.first_predictions = {}

    @initialize()
    def fit_at_every_pinned_placement(self):
        for default, query_device in itertools.product(DEVICES_PRESENT, repeat=2):
            self.point_the_process_default_elsewhere(default)
            self.fit_and_predict(seed=0, data=0, query_device=query_device)
            self.draw_from_the_global_generators(draws=1)
            self.fit_and_predict(seed=0, data=0, query_device=query_device)

    @rule(draws=st.integers(1, 64))
    def draw_from_the_global_generators(self, draws):
        torch.rand(draws)

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def point_the_process_default_elsewhere(self, device):
        torch.set_default_device(device)

    @rule(
        seed=st.integers(0, 1),
        data=st.sampled_from(sorted(SPLITS)),
        query_device=st.sampled_from(DEVICES_PRESENT),
    )
    def fit_and_predict(self, seed, data, query_device):
        split = SPLITS[data]
        operator = DeepONetOperator(trunk=QUERY, iterations=ITERATIONS, seed=seed)
        fitted = operator.fit(split.train)

        predictions = fitted.predict(split.test.f, QUERY.to(query_device))

        assert predictions.device == query_device
        assert predictions.shape == (split.test.n_functions, len(QUERY))
        assert not predictions.isnan().any()
        training_device = torch.get_default_device()
        first = self.first_predictions.setdefault(
            (seed, data, training_device), predictions.cpu()
        )
        assert torch.equal(predictions.cpu(), first)


DeepONetFits.TestCase.settings = settings(
    deadline=None, max_examples=10, stateful_step_count=8
)
TestDeepONetFits = pytest.mark.scenario(
    "A fit does not depend on its place in the run"
)(
    pytest.mark.scenario("A baseline predicts where its query is")(
        pytest.mark.oracle_invariant(DeepONetFits.TestCase)
    )
)
