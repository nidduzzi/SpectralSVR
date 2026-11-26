import collections
import glob
import pathlib
import sqlite3
import sys

from cosmic_ray.work_db import WorkDB, use_db

from compare_survivors import ROOT, SURVIVED


def finished_sessions() -> list[pathlib.Path]:
    done = []
    for path in sorted(ROOT.glob("p4*-phase5audit-*.sqlite")):
        connection = sqlite3.connect(path)
        total = connection.execute("select count(*) from work_items").fetchone()[0]
        complete = connection.execute("select count(*) from work_results").fetchone()[0]
        if total == complete:
            done.append(path)
    return done


def survivors_by_line(path: pathlib.Path) -> dict[int, list[tuple[str, str]]]:
    tree = path.name.split("-phase5audit-")[0]
    by_line = collections.defaultdict(list)
    with use_db(str(path), WorkDB.Mode.open) as db:
        for item, result in db.completed_work_items:
            if str(result.test_outcome) != SURVIVED:
                continue
            mutation = item.mutations[0]
            diff = [
                line
                for line in result.diff.splitlines()
                if line[:1] in "+-" and line[:3] not in ("+++", "---")
            ]
            by_line[mutation.start_pos[0]].append(
                (mutation.operator_name.split("/")[-1], " | ".join(diff))
            )
    return by_line


if __name__ == "__main__":
    wanted = sys.argv[1:]
    for path in finished_sessions():
        if wanted and not any(name in path.name for name in wanted):
            continue
        by_line = survivors_by_line(path)
        print(f"## {path.name} : {sum(map(len, by_line.values()))} survivors on {len(by_line)} lines")
        for line, mutants in sorted(by_line.items()):
            for operator, diff in mutants:
                print(f"{line}\t{operator}\t{diff[:230]}")
