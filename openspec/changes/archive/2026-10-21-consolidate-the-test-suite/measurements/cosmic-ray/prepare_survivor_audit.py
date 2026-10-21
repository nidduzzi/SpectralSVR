import collections
import pathlib
import sqlite3
import subprocess
import sys

from cosmic_ray.work_db import WorkDB, use_db

from compare_survivors import ROOT, SURVIVED

CONFIGS = pathlib.Path(__file__).parent
PHASE = "phase5audit"
TREES = ["p4a-after", "p4b-after", "p4c-after", "p4d-after", "p4e-after", "p4a-before"]
WHOLE_SUITE_COMMAND = (
    f"{CONFIGS}/timed_pytest.sh -x -q -p no:cacheprovider --hypothesis-seed=0 tests"
)
FINAL_SESSIONS = {
    "after": ["phase1-_plot", "phase1-sampling"],
    "p2-after2": [
        "phase2-__base",
        "phase2-FNN",
        "phase2-LSSVR",
        "phase2-SpectralSVR",
        "phase2-scaler",
    ],
    "p3-after": ["phase3-devices_and_resize"],
    "p4a-after": ["phase4a-utils"],
    "p4b-after": [
        "phase4b-after-problems___base",
        "phase4b-after-problems_antiderivative",
        "phase4b-after2-problems_burgers",
        "phase4b-after-problems_exact",
    ],
    "p4c-after": [
        f"phase4c-after-basis_{name}"
        for name in ("__base", "config", "domain", "FourierBasis", "_nufft")
    ],
    "p4d-after": [
        f"phase4d-after-model_{name}"
        for name in ("__base", "FNN", "LSSVR", "SpectralSVR")
    ],
    "p4e-after": [
        f"phase4e-after-operators_{name}"
        for name in ("__base", "benchmark", "dataset", "deeponet", "spectralsvr")
    ],
}


def survivor_keys() -> dict[str, dict[tuple, tuple]]:
    chosen: dict[str, dict[tuple, tuple]] = collections.defaultdict(dict)
    for tree, sessions in FINAL_SESSIONS.items():
        for session in sessions:
            path = ROOT / f"{tree}-{session}.sqlite"
            sources: dict[pathlib.Path, list[str]] = {}
            with use_db(str(path), WorkDB.Mode.open) as db:
                for item, result in db.completed_work_items:
                    if str(result.test_outcome) != SURVIVED:
                        continue
                    mutation = item.mutations[0]
                    source = ROOT / tree / mutation.module_path
                    lines = sources.setdefault(source, source.read_text().splitlines())
                    line = lines[mutation.start_pos[0] - 1].strip()
                    pair = (mutation.operator_name.split("/")[-1], line)
                    position = (
                        mutation.operator_name,
                        mutation.occurrence,
                        mutation.start_pos,
                        mutation.end_pos,
                    )
                    chosen[str(mutation.module_path)].setdefault(pair, position)
    return chosen


def prepare() -> None:
    chosen = survivor_keys()
    load = {tree: 0 for tree in TREES}
    for module_path, pairs in sorted(chosen.items(), key=lambda kv: -len(kv[1])):
        tree = min(load, key=load.__getitem__)
        load[tree] += len(pairs)
        name = f"{PHASE}-{tree}-{module_path.replace('/', '_').removesuffix('.py')}"
        config = CONFIGS / f"{name}.toml"
        config.write_text(
            "[cosmic-ray]\n"
            f'module-path = "{module_path}"\n'
            "timeout = 600.0\n"
            "excluded-modules = []\n"
            f'test-command = "{WHOLE_SUITE_COMMAND}"\n\n'
            "[cosmic-ray.distributor]\n"
            'name = "local"\n'
        )
        session = ROOT / f"{tree}-{name}.sqlite"
        subprocess.run(
            ["uv", "run", "--extra", "mutation", "cosmic-ray", "init", str(config), str(session)],
            cwd=ROOT / tree,
            check=True,
            capture_output=True,
        )
        wanted = set(pairs.values())
        connection = sqlite3.connect(session)
        keep = [
            job_id
            for job_id, operator, occurrence, row, column, end_row, end_column in connection.execute(
                "select job_id, operator_name, occurrence, start_pos_row, start_pos_col, "
                "end_pos_row, end_pos_col from mutation_specs"
            )
            if (operator, occurrence, (row, column), (end_row, end_column)) in wanted
        ]
        marks = ",".join("?" * len(keep))
        for table in ("mutation_specs", "work_items"):
            connection.execute(
                f"delete from {table} where job_id not in ({marks})", keep
            )
        connection.commit()
        print(f"{name}: {len(keep)} of {len(pairs)} wanted mutants kept on {tree}")
        if len(keep) != len(pairs):
            print("  MISMATCH", file=sys.stderr)


if __name__ == "__main__":
    prepare()
