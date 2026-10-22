import ast
import json
import sys
from pathlib import Path

DRAWN_OR_TABLED = {"given", "parametrize"}


def decorator_names(node):
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        while isinstance(target, ast.Attribute):
            yield target.attr
            target = target.value
        if isinstance(target, ast.Name):
            yield target.id


def scenario_names(node):
    for decorator in node.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "scenario"
        ):
            yield from (
                arg.value for arg in decorator.args if isinstance(arg, ast.Constant)
            )


def single_tests(root):
    for module in sorted((root / "tests").rglob("test_*.py")):
        tree = ast.parse(module.read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith(
                "test_"
            ):
                continue
            if DRAWN_OR_TABLED & set(decorator_names(node)):
                continue
            yield (
                f"{module.relative_to(root).as_posix()}::{node.name}",
                list(scenario_names(node)),
            )


root, output = Path(sys.argv[1]), Path(sys.argv[2])
output.write_text(json.dumps(dict(single_tests(root)), indent=1) + "\n")
