import importlib.util
import itertools
import math
import statistics

import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from SpectralSVR import Antiderivative, FourierBasis
from SpectralSVR.operators import (
    DeepONetOperator,
    Lu2021DeepxdeExample,
    Lu2022Burgers,
    Split,
)
from SpectralSVR.operators.deeponet import PointwiseOutputScale
from tests._devices import DEVICES_PRESENT, HOST

ACCELERATORS = [device for device in DEVICES_PRESENT if device != HOST]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("deepxde") is None,
    reason="needs the 'baselines' extra (deepxde)",
)

MODES = 8
N_TRAIN = 12
QUERY = torch.linspace(0, 1, 12, dtype=torch.float64).unsqueeze(-1)
ITERATIONS = 20
RECIPES = {
    "Lu 2021": Lu2021DeepxdeExample(),
    "Lu 2022": Lu2022Burgers(periods=(1.0,)),
}
PERIOD_CHOICES = [None, 1.0, 0.5, 0.0, -1.0]
RECIPE_BUILD_EDGES = [
    ((1.0,), 5, 1),
    ((1.0,), 4, 1),
    ((1.0,), 5, 2),
    ((None, 1.0), 5, 2),
    ((None,), 5, 1),
    ((), 5, 0),
    ((0.0,), 5, 1),
    ((-1.0, 1.0), 5, 2),
    ((None, 1.0), 5, 1),
]
NAMED_ACCELERATOR = torch.device("cuda")


def antiderivative_fields(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return Antiderivative().generate(
        FourierBasis, N_TRAIN + 4, MODES, generator=generator, u0=0
    )


def spliced(first: FourierBasis, second: FourierBasis) -> FourierBasis:
    return FourierBasis(
        coeff=torch.cat([first.coeff[:N_TRAIN], second.coeff[N_TRAIN:]]),
        domain=first.domain,
    )


FIELDS = {seed: antiderivative_fields(seed) for seed in (0, 1)}
SPLITS = {
    seed: Split.partition(ut, u, N_TRAIN, n_sensors=MODES)
    for seed, (u, ut) in FIELDS.items()
}
SPLIT_SHARING_THE_FIRST_TRAINING_HALF = Split.partition(
    spliced(FIELDS[0][1], FIELDS[1][1]),
    spliced(FIELDS[0][0], FIELDS[1][0]),
    N_TRAIN,
    n_sensors=MODES,
)
CONSTANT_OUTPUT_SPLIT = Split.partition(
    FIELDS[0][1],
    FourierBasis(
        coeff=torch.zeros_like(FIELDS[0][0].coeff), domain=FIELDS[0][0].domain
    ),
    N_TRAIN,
    n_sensors=MODES,
)


def fitted(recipe_name: str, split: Split, seed: int = 0, device=None):
    operator = DeepONetOperator(
        trunk=QUERY,
        recipe=RECIPES[recipe_name],
        device=ACCELERATORS[0] if device is None else device,
        iterations=ITERATIONS,
        seed=seed,
    )
    default_before = torch.get_default_device()
    model = operator.fit(split.train)
    assert torch.get_default_device() == default_before
    return model


class DeepONetFits(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.first_predictions = {}

    @initialize()
    def check_every_pinned_edge(self):
        for (periods, iterations, trunk_columns), device in itertools.product(
            RECIPE_BUILD_EDGES, DEVICES_PRESENT
        ):
            self.build_a_recipe(periods, iterations, trunk_columns, device)
        for recipe in RECIPES:
            self.query_elsewhere(recipe)
            self.fit_where_the_targets_are_constant(recipe)
            self.fit_beside_another_test_half(recipe)
            for default, training, query in itertools.product(
                DEVICES_PRESENT, ACCELERATORS, DEVICES_PRESENT
            ):
                self.point_the_process_default_elsewhere(default)
                self.fit_and_predict(recipe, 0, 0, training, query)
                self.draw_from_the_global_generators(draws=1)
                self.fit_and_predict(recipe, 0, 0, training, query)

    @rule(draws=st.integers(1, 64))
    def draw_from_the_global_generators(self, draws):
        torch.rand(draws)

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def point_the_process_default_elsewhere(self, device):
        torch.set_default_device(device)

    @rule(
        recipe=st.sampled_from(sorted(RECIPES)),
        seed=st.integers(0, 1),
        data=st.sampled_from(sorted(SPLITS)),
        training_device=st.sampled_from(ACCELERATORS),
        query_device=st.sampled_from(DEVICES_PRESENT),
    )
    def fit_and_predict(self, recipe, seed, data, training_device, query_device):
        split = SPLITS[data]

        model = fitted(recipe, split, seed, training_device)
        predictions = model.predict(split.test.f, QUERY.to(query_device))

        assert model.device == training_device
        assert predictions.device == query_device
        assert predictions.shape == (split.test.n_functions, len(QUERY))
        assert not predictions.isnan().any()
        first = self.first_predictions.setdefault(
            (recipe, seed, data, training_device), predictions.cpu()
        )
        assert torch.equal(predictions.cpu(), first)

    @rule(recipe=st.sampled_from(sorted(RECIPES)))
    def query_elsewhere(self, recipe):
        model = fitted(recipe, SPLITS[0])
        elsewhere = QUERY[:-1] + 0.01
        match RECIPES[recipe]:
            case Lu2022Burgers():
                with pytest.raises(ValueError, match="only at the points"):
                    model.predict(SPLITS[0].test.f, elsewhere)
            case Lu2021DeepxdeExample():
                answer = model.predict(SPLITS[0].test.f, elsewhere)
                assert answer.shape == (SPLITS[0].test.n_functions, len(elsewhere))

    @rule(recipe=st.sampled_from(sorted(RECIPES)))
    def fit_where_the_targets_are_constant(self, recipe):
        split = CONSTANT_OUTPUT_SPLIT

        predictions = fitted(recipe, split).predict(split.test.f, QUERY)

        match RECIPES[recipe]:
            case Lu2022Burgers():
                assert torch.equal(predictions, torch.zeros_like(predictions))
            case Lu2021DeepxdeExample():
                assert not predictions.isnan().any()

    @rule(recipe=st.sampled_from(sorted(RECIPES)))
    def fit_beside_another_test_half(self, recipe):
        probe = SPLITS[0].test.f
        original = fitted(recipe, SPLITS[0]).predict(probe, QUERY)
        sharing = fitted(recipe, SPLIT_SHARING_THE_FIRST_TRAINING_HALF).predict(
            probe, QUERY
        )
        assert not torch.equal(
            SPLITS[0].test.f, SPLIT_SHARING_THE_FIRST_TRAINING_HALF.test.f
        )
        assert torch.equal(original, sharing)

    @rule(
        periods=st.lists(st.sampled_from(PERIOD_CHOICES), max_size=3).map(tuple),
        iterations=st.integers(-1, 8),
        trunk_columns=st.integers(1, 3),
        device=st.sampled_from(DEVICES_PRESENT),
    )
    def build_a_recipe(self, periods, iterations, trunk_columns, device):
        periodic = [period for period in periods if period is not None]
        recipe_is_valid = bool(periodic) and all(period > 0 for period in periodic)
        trunk = torch.zeros(4, trunk_columns, dtype=torch.float64)
        if not recipe_is_valid:
            with pytest.raises(ValueError, match="periodic coordinate"):
                Lu2022Burgers(periods=periods)
            return
        recipe = Lu2022Burgers(periods=periods)
        arguments = dict(
            trunk=trunk, recipe=recipe, device=device, iterations=iterations
        )
        if device == HOST:
            with pytest.raises(ValueError, match="only on an accelerator"):
                DeepONetOperator(**arguments)
        elif iterations < recipe.minimum_iterations:
            with pytest.raises(ValueError, match="at least 5 iterations"):
                DeepONetOperator(**arguments)
        elif len(periods) != trunk_columns:
            with pytest.raises(
                ValueError,
                match=f"states {len(periods)} trunk coordinates but the trunk "
                f"points have {trunk_columns}",
            ):
                DeepONetOperator(**arguments)
        else:
            operator = DeepONetOperator(**arguments)
            assert (operator.recipe, operator.device) == (recipe, device)


DeepONetFits.TestCase.settings = settings(
    deadline=None, max_examples=10, stateful_step_count=8
)
TestDeepONetFits = pytest.mark.skipif(
    not ACCELERATORS, reason="DeepONet trains only on an accelerator"
)(
    pytest.mark.scenario("A fit does not depend on its place in the run")(
        pytest.mark.scenario("A baseline predicts where its query is")(
            pytest.mark.scenario("Every stage runs where the run says")(
                pytest.mark.oracle_invariant(DeepONetFits.TestCase)
            )
        )
    )
)


@pytest.mark.oracle_invariant
@settings(deadline=None, max_examples=50)
@given(
    x=st.floats(0.0, 1.0),
    t=st.floats(0.0, 1.0),
    period=st.floats(0.25, 4.0),
    shift=st.integers(-3, 3),
)
@example(x=0.0, t=0.0, period=1.0, shift=1)
@example(x=1.0, t=1.0, period=0.25, shift=-3)
@example(x=0.5, t=0.5, period=4.0, shift=3)
def test_trunk_features_repeat_with_the_period_and_pass_time_through(
    x, t, period, shift
):
    recipe = Lu2022Burgers(periods=(None, period))
    point = torch.tensor([[t, x]], dtype=torch.float64)
    shifted = torch.tensor([[t, x + shift * period]], dtype=torch.float64)

    features = recipe.trunk_features(point)

    reference = [
        t,
        math.cos(2 * math.pi * x / period),
        math.sin(2 * math.pi * x / period),
        math.cos(4 * math.pi * x / period),
        math.sin(4 * math.pi * x / period),
    ]

    assert features.shape == (1, recipe.trunk_feature_width)
    assert torch.allclose(features, recipe.trunk_features(shifted), atol=1e-9)
    for value, expected in zip(features[0].tolist(), reference, strict=True):
        assert math.isclose(value, expected, abs_tol=1e-12)


@pytest.mark.oracle_independent
@pytest.mark.scenario("The reference protocol is followed and recorded")
@settings(deadline=None, max_examples=50)
@given(iterations=st.integers(-2, 600_000))
@example(iterations=0)
@example(iterations=1)
@example(iterations=4)
@example(iterations=5)
@example(iterations=7)
@example(iterations=10_000)
@example(iterations=50_000)
def test_recipes_transcribe_their_reference_code(iterations):
    lu2021, lu2022 = Lu2021DeepxdeExample(), Lu2022Burgers(periods=(1.0,))

    assert (lu2021.branch_widths, lu2021.trunk_widths) == ((40, 40), (40, 40))
    assert (lu2021.activation, lu2021.learning_rate) == ("relu", 1e-3)
    assert lu2021.source_iterations == 50_000
    assert lu2022.branch_widths == (128, 128, 128, 128)
    assert lu2022.trunk_widths == (128, 128, 128)
    assert (lu2022.activation, lu2022.learning_rate) == ("tanh", 1e-3)
    assert lu2022.harmonics == (1, 2)
    assert lu2022.source_iterations == 500_000
    if iterations >= 5:
        assert lu2022.decay(iterations) == ("inverse time", iterations // 5, 0.5)
    for recipe, minimum in ((lu2021, 1), (lu2022, 5)):
        build = dict(trunk=QUERY, recipe=recipe, device=NAMED_ACCELERATOR)
        if iterations < minimum:
            with pytest.raises(ValueError, match=f"at least {minimum} iterations"):
                DeepONetOperator(**build, iterations=iterations)
        else:
            assert (
                DeepONetOperator(**build, iterations=iterations).iterations
                == iterations
            )


@pytest.mark.oracle_independent
@settings(deadline=None, max_examples=40)
@given(
    functions=st.integers(1, 6),
    points=st.integers(1, 5),
    seed=st.integers(0, 10_000),
)
@example(functions=1, points=1, seed=0)
@example(functions=6, points=5, seed=0)
def test_the_output_scale_is_the_per_point_mean_and_deviation(functions, points, seed):
    generator = torch.Generator().manual_seed(seed)
    targets = torch.randn(functions, points, generator=generator).numpy()
    outputs = torch.randn(3, points, generator=generator)
    columns = [targets[:, column].tolist() for column in range(points)]

    scale = PointwiseOutputScale.of(targets)
    scaled = scale(None, outputs)

    for column, values in enumerate(columns):
        mean, deviation = statistics.fmean(values), statistics.pstdev(values)
        assert math.isclose(scale.mean[column].item(), mean, rel_tol=1e-5, abs_tol=1e-6)
        assert math.isclose(
            scale.std[column].item(), deviation, rel_tol=1e-5, abs_tol=1e-6
        )
        for row in range(3):
            expected = outputs[row, column].item() * deviation + mean
            assert math.isclose(
                scaled[row, column].item(), expected, rel_tol=1e-5, abs_tol=1e-5
            )
