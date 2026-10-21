import json
import os
import pathlib
import subprocess
import sys

import coverage

CONFIGS = pathlib.Path(__file__).parent


def measure(tree: pathlib.Path, output: pathlib.Path) -> None:
    data_file = output.with_suffix(".coverage")
    durations_file = output.with_suffix(".durations.json")
    subprocess.run(
        [
            "uv",
            "run",
            "--extra",
            "test",
            "--extra",
            "baselines",
            "coverage",
            "run",
            f"--data-file={data_file}",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--hypothesis-seed=0",
            "-p",
            "coverage_contexts",
            "tests",
        ],
        cwd=tree,
        env={
            "TEST_MAP_DURATIONS": str(durations_file),
            **os.environ,
            "PYTHONPATH": str(CONFIGS),
            "COVERAGE_CORE": os.environ.get("MAP_COVERAGE_CORE", "ctrace"),
        },
        check=True,
    )
    data = coverage.CoverageData(basename=str(data_file))
    data.read()
    lines: dict[str, dict[int, list[str]]] = {}
    for measured in data.measured_files():
        relative = (
            pathlib.Path(measured).resolve().relative_to(tree.resolve()).as_posix()
        )
        if not relative.startswith("SpectralSVR/"):
            continue
        lines[relative] = {
            line: sorted(context for context in contexts if context)
            for line, contexts in data.contexts_by_lineno(measured).items()
        }
    durations = json.loads(durations_file.read_text())
    output.write_text(json.dumps({"lines": lines, "durations": durations}))


if __name__ == "__main__":
    measure(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
