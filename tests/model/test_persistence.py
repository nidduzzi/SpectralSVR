"""A dumped regressor loads back as the regressor that was dumped."""

import json
import tempfile
from dataclasses import dataclass

import pytest
import torch
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    precondition,
    rule,
)

from SpectralSVR import LSSVR
from SpectralSVR.model.FNN import FNN

from _model_data import linear_dataset

KINDS = (FNN, LSSVR)
OTHER_KIND = {FNN: LSSVR, LSSVR: FNN}
PREDICTION_ATOL = {FNN: 1e-6, LSSVR: 1e-4}
HYPERPARAMETERS = {
    FNN: (
        "MAX_EPOCH",
        "lr",
        "batch_size",
        "activation",
        "n_hidden",
        "w_hidden",
        "dtype",
    ),
    LSSVR: ("C", "kernel_params"),
}


@dataclass(frozen=True)
class Dump:
    path: str
    kind: type
    carries_parameters: bool
    hyperparameters: dict
    inputs: torch.Tensor
    predictions: torch.Tensor | None


class RegressorPersistence(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.directory = tempfile.TemporaryDirectory()
        self.dumps: list[Dump] = []
        self.build(FNN, 0, torch.float32)

    def teardown(self):
        self.directory.cleanup()

    def next_path(self, stem):
        return f"{self.directory.name}/{stem}{len(self.dumps)}"

    @rule(
        kind=st.sampled_from(KINDS),
        seed=st.integers(0, 1000),
        precision=st.sampled_from((torch.float32, torch.float64)),
    )
    def build(self, kind, seed, precision):
        if kind is FNN:
            self.regressor = FNN(
                MAX_EPOCH=2, w_hidden=8, n_hidden=1, batch_size=8, dtype=precision
            )
        else:
            self.regressor = LSSVR(
                kernel="linear" if seed % 2 else "rbf", C=1.0 + seed % 7
            )
        self.kind = kind
        x, self.y = linear_dataset(20, 3, 2, seed)
        self.x = x.to(self.regressor.dtype)
        self.y = self.y.to(self.regressor.dtype)

    @rule()
    def fit(self):
        self.regressor.fit(self.x, self.y)
        assert self.regressor.trained

    @rule(only_hyperparams=st.booleans())
    def dump(self, only_hyperparams):
        path = self.next_path("regressor")
        self.regressor.dump(path, only_hyperparams=only_hyperparams)
        fitted = self.regressor.trained
        self.dumps.append(
            Dump(
                path=path,
                kind=self.kind,
                carries_parameters=fitted and not only_hyperparams,
                hyperparameters={
                    name: getattr(self.regressor, name)
                    for name in HYPERPARAMETERS[self.kind]
                },
                inputs=self.x,
                predictions=self.regressor.predict(self.x) if fitted else None,
            )
        )

    @precondition(lambda self: bool(self.dumps))
    @rule(only_hyperparams=st.booleans())
    def load(self, only_hyperparams):
        dumped = self.dumps[-1]
        loaded = dumped.kind.load(dumped.path, only_hyperparams=only_hyperparams)
        assert isinstance(loaded, dumped.kind)
        for name, value in dumped.hyperparameters.items():
            assert getattr(loaded, name) == value, name
        if dumped.carries_parameters and not only_hyperparams:
            assert loaded.trained
            assert torch.allclose(
                loaded.predict(dumped.inputs.to(loaded.dtype)).to(
                    dumped.predictions.dtype
                ),
                dumped.predictions,
                atol=PREDICTION_ATOL[dumped.kind],
            )
        else:
            assert not loaded.trained

    @rule(loading=st.sampled_from(KINDS))
    def load_a_dump_of_the_other_kind(self, loading):
        stated = OTHER_KIND[loading].__name__
        if loading is FNN:
            path = self.next_path("fnn_payload") + ".pt"
            torch.save({"type": stated, "hyperparameters": {}}, path)
        else:
            path = self.next_path("lssvr_payload") + ".json"
            with open(path, "w") as payload:
                json.dump({"type": stated, "hyperparameters": {}}, payload)
        with pytest.raises(Exception, match=f"doesn't match '{loading.__name__}'"):
            loading.load(path)

    @initialize()
    def reproduce_every_pinned_case(self):
        for kind in KINDS:
            for fitted in (False, True):
                self.build(kind, 0, torch.float32)
                if fitted:
                    self.fit()
                for only_hyperparams in (False, True):
                    self.dump(only_hyperparams)
                    for load_only_hyperparams in (False, True):
                        self.load(load_only_hyperparams)
            self.load_a_dump_of_the_other_kind(kind)
        self.build(FNN, 1, torch.float64)
        self.fit()
        self.dump(False)
        self.load(False)
        self.build(LSSVR, 2, torch.float32)


RegressorPersistence.TestCase.settings = settings(
    deadline=None,
    max_examples=25,
    stateful_step_count=8,
    suppress_health_check=[HealthCheck.too_slow],
)
TestRegressorPersistence = pytest.mark.oracle_invariant(RegressorPersistence.TestCase)
