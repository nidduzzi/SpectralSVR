import json
import pathlib
import re
import subprocess
import sys

WHOLE_SUITE = "tests"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))?")


def mutated_lines() -> dict[str, set[int]]:
    diff = subprocess.run(
        ["git", "diff", "-U0"], capture_output=True, text=True, check=True
    ).stdout
    lines: dict[str, set[int]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("--- a/"):
            path = line.removeprefix("--- a/")
        hunk = HUNK.match(line)
        if hunk and path:
            start, count = int(hunk.group(1)), int(hunk.group(2) or 1)
            lines.setdefault(path, set()).update(range(start, start + max(count, 1)))
    return lines


def covering_tests(test_map: dict, lines: dict[str, set[int]]) -> list[str] | None:
    chosen: set[str] = set()
    for path, numbers in lines.items():
        by_line = test_map["lines"].get(path, {})
        for number in numbers:
            tests = by_line.get(str(number))
            if not tests:
                return None
            chosen.update(tests)
    durations = test_map["durations"]
    return sorted(chosen, key=lambda nodeid: durations.get(nodeid, 0.0))


if __name__ == "__main__":
    test_map = json.loads(pathlib.Path(sys.argv[1]).read_text())
    tests = covering_tests(test_map, mutated_lines())
    if tests is None:
        tests = [WHOLE_SUITE]
        seconds = sum(test_map["durations"].values())
    else:
        seconds = sum(test_map["durations"].get(test, 0.0) for test in tests)
    print(seconds, *tests, sep="\n")
