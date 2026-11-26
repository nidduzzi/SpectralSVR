import pathlib
import random
import shutil
import sqlite3

from compare_survivors import ROOT

CONFIGS = pathlib.Path(__file__).parent
TREES = ["p4c-after", "p4d-after"]
PER_OUTCOME = 6
SELECTED_COMMAND = f"{CONFIGS}/selected_pytest.sh"


def prepare() -> None:
    sessions = sorted(ROOT.glob("p4*-phase5audit-*.sqlite"))
    for index, source in enumerate(sessions):
        tree = TREES[index % len(TREES)]
        module = source.name.split("-SpectralSVR_")[1].removesuffix(".sqlite")
        name = f"phase5check-{tree}-{module}"
        session = ROOT / f"{tree}-{name}.sqlite"
        shutil.copy(source, session)
        connection = sqlite3.connect(session)
        chosen = []
        for outcome in ("SURVIVED", "KILLED"):
            jobs = [
                job_id
                for (job_id,) in connection.execute(
                    "select job_id from work_results where test_outcome = ?", (outcome,)
                )
            ]
            chosen += random.Random(0).sample(jobs, min(PER_OUTCOME, len(jobs)))
        marks = ",".join("?" * len(chosen))
        connection.execute(
            f"create table expected as select job_id, test_outcome from work_results where job_id in ({marks})",
            chosen,
        )
        for table in ("work_results", "mutation_specs", "work_items"):
            connection.execute(
                f"delete from {table} where job_id not in ({marks})", chosen
            )
        connection.execute("delete from work_results")
        connection.commit()
        audit_config = next(CONFIGS.glob(f"phase5audit-*-SpectralSVR_{module}.toml"))
        lines = [
            f'test-command = "{SELECTED_COMMAND}"'
            if line.startswith("test-command")
            else line
            for line in audit_config.read_text().splitlines()
        ]
        (CONFIGS / f"{name}.toml").write_text("\n".join(lines) + "\n")
        print(f"{name}: {len(chosen)} mutants on {tree}")


if __name__ == "__main__":
    prepare()
