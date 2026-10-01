import importlib.util
import math
import pathlib
import shutil
import statistics
import subprocess
import tempfile

import pandas as pd
import pytest
import torch
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from SpectralSVR.operators import Lu2021DeepxdeExample, Lu2022Burgers, summarise
from SpectralSVR.operators.benchmark import RUN_KEY, SUMMARISED_COLUMNS
from tests._devices import DEVICES_PRESENT, HOST

SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark.py"
)
CHEAP_DATASETS = ["antiderivative", "antiderivative-grf"]
PROVENANCE_COLUMNS = [
    "data_device",
    "fit_device",
    "fit_covers",
    "protocol",
    "quick",
    "seed_set",
    "revision",
    "tree_clean",
    "torch_version",
    "deepxde_version",
    "coeff_dtype",
]


MEASURED_DEEPONET_PLANS = {
    "antiderivative": (Lu2021DeepxdeExample(), 10_000),
    "burgers-mms": (Lu2022Burgers(periods=(None, 1.0)), 50_000),
    "burgers-numerical": (Lu2022Burgers(periods=(1.0,)), 10_000),
    "antiderivative-grf": (Lu2021DeepxdeExample(), 50_000),
    "burgers-forced": (Lu2022Burgers(periods=(None, 1.0)), 10_000),
}


def load_script():
    spec = importlib.util.spec_from_file_location("_run_benchmark_record", SCRIPT_PATH)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


SCRIPT = load_script()
HAS_DEEPXDE = importlib.util.find_spec("deepxde") is not None
ACCELERATORS = [device for device in DEVICES_PRESENT if device != HOST]


def keys_of(table: pd.DataFrame) -> set[tuple[str, str, int]]:
    return set(
        zip(table["dataset"], table["operator"], table["seed"].astype(int), strict=True)
    )


class BenchmarkRecord(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.directory = pathlib.Path(tempfile.mkdtemp())
        self.out = self.directory / "runs.csv"
        self.expected_keys: set[tuple[str, str, int]] = set()

    def teardown(self):
        shutil.rmtree(self.directory)

    def run_script(
        self, out, seeds, datasets, device, append, operators=("spectralsvr",)
    ):
        argv = [
            "--quick",
            "--out",
            str(out),
            "--device",
            str(device),
            "--operators",
            *operators,
            "--seeds",
            *map(str, seeds),
            "--datasets",
            *datasets,
        ]
        SCRIPT.main([*argv, "--append"] if append else argv)
        return pd.read_csv(out)

    @initialize()
    def check_every_pinned_edge(self):
        with pytest.raises(ValueError, match="once"):
            summarise(pd.concat([self.record_one_run(), self.record_one_run()]))
        untouched = self.directory / "untouched.csv"
        untouched.write_text("a record a refused run must not overwrite\n")
        with pytest.raises(SystemExit):
            self.run_script(
                untouched,
                [0],
                ["antiderivative"],
                HOST,
                append=False,
                operators=("spectralsvr", "deeponet"),
            )
        assert untouched.read_text() == "a record a refused run must not overwrite\n"
        if HAS_DEEPXDE and ACCELERATORS:
            table = self.run_script(
                self.directory / "with_deeponet.csv",
                [0],
                ["antiderivative"],
                ACCELERATORS[0],
                append=False,
                operators=("spectralsvr", "deeponet"),
            )
            assert (table["fit_device"] == str(ACCELERATORS[0])).all()
            deeponet = table[table["operator"] == "DeepONet"].iloc[0]
            assert deeponet["fit_covers"] == (
                "200 Adam iterations of the Lu 2021 / deepxde example recipe"
            )
            assert deeponet["protocol"] == (
                "Lu 2021 / deepxde example recipe from deepxde "
                "examples/operator/antiderivative_aligned.py; lululxvi/deeponet "
                "src/deeponet_pde.py; 200 iterations against the source's 50000"
            )
        assert {
            key: (spec.deeponet.recipe, spec.deeponet.iterations)
            for key, spec in SCRIPT.DATASETS.items()
        } == MEASURED_DEEPONET_PLANS
        self.run(seeds=[0], datasets=["antiderivative"], device=HOST, append=False)
        self.run(seeds=[0, 1], datasets=CHEAP_DATASETS, device=HOST, append=True)

    def record_one_run(self):
        return self.run_script(
            self.directory / "one.csv", [0], ["antiderivative"], HOST, append=False
        )

    @rule(
        seeds=st.lists(st.integers(0, 2), min_size=1, max_size=2, unique=True),
        datasets=st.lists(st.sampled_from(CHEAP_DATASETS), min_size=1, unique=True),
        device=st.sampled_from(DEVICES_PRESENT),
        append=st.booleans(),
    )
    def run(self, seeds, datasets, device, append):
        before = pd.read_csv(self.out) if self.out.exists() else None
        labels = [SCRIPT.DATASETS[key].label for key in datasets]
        requested = {(label, "SpectralSVR", seed) for label in labels for seed in seeds}

        table = self.run_script(self.out, seeds, datasets, device, append)

        if append and before is not None:
            kept = table.iloc[: len(before)]
            pd.testing.assert_frame_equal(
                kept.reset_index(drop=True), before, check_dtype=False
            )
            self.expected_keys |= requested
        else:
            self.expected_keys = requested

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def point_the_process_default_elsewhere(self, device):
        torch.set_default_device(device)

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def compare_with_another_device(self, device):
        other = self.run_script(
            self.directory / "other.csv", [0], ["antiderivative"], device, append=False
        )
        base = self.record_one_run()

        both = pd.concat([base, other], ignore_index=True)

        assert list(both.columns) == list(base.columns)
        assert not both[PROVENANCE_COLUMNS].isna().any().any()
        assert set(both["data_device"]) == {str(HOST), str(device)}

    @invariant()
    def the_file_holds_exactly_the_expected_runs(self):
        if not self.out.exists():
            return
        table = pd.read_csv(self.out)
        assert keys_of(table) == self.expected_keys
        assert not table.duplicated(subset=RUN_KEY).any()

    @invariant()
    def every_row_was_fitted_where_its_data_was_built(self):
        if not self.out.exists():
            return
        table = pd.read_csv(self.out)
        assert (table["fit_device"] == table["data_device"]).all()

    @invariant()
    def every_row_carries_its_provenance(self):
        if not self.out.exists():
            return
        table = pd.read_csv(self.out)
        assert not table[PROVENANCE_COLUMNS].isna().any().any()
        assert table["quick"].all()
        assert (table["revision"].str.len() == 40).all()

    @invariant()
    def the_summary_is_the_mean_and_sample_deviation_of_the_rows(self):
        if not self.out.exists():
            return
        table = pd.read_csv(self.out)
        summary = summarise(table)
        for (dataset, operator), group in table.groupby(["dataset", "operator"]):
            row = summary.loc[(dataset, operator)]
            assert row["n_seeds"] == len(group)
            assert row["seeds"] == str(sorted(int(s) for s in group["seed"]))
            for column in SUMMARISED_COLUMNS:
                values = group[column].tolist()
                assert math.isclose(
                    row[f"{column}_mean"],
                    statistics.fmean(values),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                if len(values) > 1:
                    assert math.isclose(
                        row[f"{column}_std"],
                        statistics.stdev(values),
                        rel_tol=1e-9,
                        abs_tol=1e-15,
                    )
                else:
                    assert math.isnan(row[f"{column}_std"])


BenchmarkRecord.TestCase.settings = settings(
    deadline=None, max_examples=8, stateful_step_count=6
)
TestBenchmarkRecord = pytest.mark.scenario("Every stage runs where the run says")(
    pytest.mark.scenario("Provenance travels with the numbers")(
        pytest.mark.scenario("A changed condition is visible")(
            pytest.mark.scenario("Spread accompanies every metric")(
                pytest.mark.oracle_independent(BenchmarkRecord.TestCase)
            )
        )
    )
)


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_provenance_reports_a_dirty_tree_and_a_clean_one(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q")
    git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "base",
    )
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    clean = SCRIPT.RunProvenance.of_repository(tmp_path)
    (tmp_path / "uncommitted.txt").write_text("a number from uncommitted code")
    dirty = SCRIPT.RunProvenance.of_repository(tmp_path)

    assert clean.tree_clean and not dirty.tree_clean
    assert clean.revision == dirty.revision == head
    assert dirty._replace(tree_clean=True) == clean
