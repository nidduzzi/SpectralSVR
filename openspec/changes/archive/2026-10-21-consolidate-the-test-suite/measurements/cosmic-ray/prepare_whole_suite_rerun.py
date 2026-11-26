import argparse
import pathlib
import shutil
import sqlite3

from compare_survivors import ROOT, survivors

CONFIGS = pathlib.Path(__file__).parent
WHOLE_SUITE_COMMAND = (
    f"{CONFIGS}/timed_pytest.sh -x -q -p no:cacheprovider --hypothesis-seed=0 tests"
)


def introduced_job_ids(
    before_tree: str, before_session: pathlib.Path, after_tree: str, after_session: pathlib.Path
) -> list[str]:
    introduced = survivors(after_tree, after_session) - survivors(before_tree, before_session)
    sources: dict[pathlib.Path, list[str]] = {}
    connection = sqlite3.connect(after_session)
    chosen = []
    for job_id, operator, module_path, row in connection.execute(
        "select s.job_id, s.operator_name, s.module_path, s.start_pos_row "
        "from mutation_specs s join work_results r using (job_id) "
        "where r.test_outcome = 'SURVIVED'"
    ):
        path = ROOT / after_tree / module_path
        lines = sources.setdefault(path, path.read_text().splitlines())
        if (operator.split("/")[-1], lines[row - 1].strip()) in introduced:
            chosen.append(job_id)
    return chosen


def prepare(phase: str, before_tree: str, after_tree: str, module: str) -> None:
    before_session = ROOT / f"{before_tree}-{phase}-before-{module}.sqlite"
    after_session = ROOT / f"{after_tree}-{phase}-after-{module}.sqlite"
    whole = f"{phase}-after-whole-{module}"
    whole_session = ROOT / f"{after_tree}-{whole}.sqlite"
    shutil.copy(after_session, whole_session)
    chosen = introduced_job_ids(before_tree, before_session, after_tree, whole_session)
    connection = sqlite3.connect(whole_session)
    marks = ",".join("?" * len(chosen))
    for table in ("work_results", "mutation_specs", "work_items"):
        connection.execute(f"delete from {table} where job_id not in ({marks})", chosen)
    connection.execute(f"delete from work_results where job_id in ({marks})", chosen)
    connection.commit()
    original = (CONFIGS / f"{phase}-after-{module}.toml").read_text().splitlines()
    rewritten = [
        f'test-command = "{WHOLE_SUITE_COMMAND}"' if line.startswith("test-command") else line
        for line in original
    ]
    (CONFIGS / f"{whole}.toml").write_text("\n".join(rewritten) + "\n")
    print(f"{whole}: {len(chosen)} mutants to rerun against the whole suite")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase")
    parser.add_argument("before_tree")
    parser.add_argument("after_tree")
    parser.add_argument("modules", nargs="+")
    arguments = parser.parse_args()
    for module in arguments.modules:
        prepare(arguments.phase, arguments.before_tree, arguments.after_tree, module)
