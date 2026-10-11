import collections
import pathlib
import sys

from cosmic_ray.work_db import WorkDB, use_db

ROOT = pathlib.Path.home() / ".cache" / "spectralsvr-mutation"
MODULES = ("__base", "FourierBasis", "_plot", "sampling")
SURVIVED = "TestOutcome.SURVIVED"


def survivors(tree: str, module: str, session: pathlib.Path) -> collections.Counter:
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


def main(after_sessions: dict[str, pathlib.Path]) -> None:
    for module in MODULES:
        before = survivors("base", module, ROOT / f"base-phase1-{module}.sqlite")
        after = survivors("after", module, after_sessions[module])
        introduced = after - before
        print(
            f"{module}: before {sum(before.values())}, after {sum(after.values())}, "
            f"survivors after but not before {sum(introduced.values())}, "
            f"before but not after {sum((before - after).values())}"
        )
        for (operator, line), count in sorted(introduced.items()):
            print(f"  only after x{count}: {operator}: {line}")


if __name__ == "__main__":
    default = {m: ROOT / f"after-phase1-{m}.sqlite" for m in MODULES}
    overrides = dict(arg.split("=", 1) for arg in sys.argv[1:])
    main({m: pathlib.Path(overrides.get(m, default[m])) for m in MODULES})
