import argparse
import collections
import pathlib

from cosmic_ray.work_db import WorkDB, use_db

ROOT = pathlib.Path.home() / ".cache" / "spectralsvr-mutation"
SURVIVED = "TestOutcome.SURVIVED"


def survivors(tree: str, session: pathlib.Path) -> collections.Counter:
    sources: dict[pathlib.Path, list[str]] = {}
    counted: collections.Counter = collections.Counter()
    with use_db(str(session), WorkDB.Mode.open) as db:
        for item, result in db.completed_work_items:
            if str(result.test_outcome) != SURVIVED:
                continue
            mutation = item.mutations[0]
            path = ROOT / tree / mutation.module_path
            lines = sources.setdefault(path, path.read_text().splitlines())
            line = lines[mutation.start_pos[0] - 1].strip()
            counted[(mutation.operator_name.split("/")[-1], line)] += 1
    return counted


def main(phase: str, before_tree: str, after_tree: str, modules: list[str]) -> None:
    for module in modules:
        before = survivors(
            before_tree, ROOT / f"{before_tree}-{phase}-{module}.sqlite"
        )
        after = survivors(after_tree, ROOT / f"{after_tree}-{phase}-{module}.sqlite")
        introduced = after - before
        print(
            f"{module}: before {sum(before.values())}, after {sum(after.values())}, "
            f"survivors after but not before {sum(introduced.values())}, "
            f"before but not after {sum((before - after).values())}"
        )
        for (operator, line), count in sorted(introduced.items()):
            print(f"  only after x{count}: {operator}: {line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="phase1")
    parser.add_argument("--before", default="base")
    parser.add_argument("--after", default="after")
    parser.add_argument(
        "--modules", nargs="+", default=["__base", "FourierBasis", "_plot", "sampling"]
    )
    arguments = parser.parse_args()
    main(arguments.phase, arguments.before, arguments.after, arguments.modules)
