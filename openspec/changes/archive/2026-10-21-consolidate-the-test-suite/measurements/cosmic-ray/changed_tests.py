import ast
import subprocess
import sys


def source_at(revision: str, path: str) -> str | None:
    shown = subprocess.run(
        ["git", "show", f"{revision}:{path}"], capture_output=True, text=True
    )
    return shown.stdout if shown.returncode == 0 else None


def test_nodes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    definitions = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    nodes = {
        name: segment
        for name, segment in definitions.items()
        if name.startswith("test_")
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for reference in ast.walk(node.value):
                if (
                    isinstance(reference, ast.Attribute)
                    and reference.attr == "TestCase"
                    and isinstance(reference.value, ast.Name)
                ):
                    nodes[node.targets[0].id] = definitions[reference.value.id]
    return nodes


def main(old: str, new: str, paths: list[str]) -> None:
    before_ids, after_ids = [], []
    for path in paths:
        old_source, new_source = source_at(old, path), source_at(new, path)
        old_nodes = test_nodes(old_source) if old_source else {}
        new_nodes = test_nodes(new_source) if new_source else {}
        before_ids += [
            f"{path}::{name}"
            for name, segment in old_nodes.items()
            if new_nodes.get(name) != segment
        ]
        after_ids += [
            f"{path}::{name}"
            for name, segment in new_nodes.items()
            if old_nodes.get(name) != segment
        ]
    print("before", *before_ids)
    print("after", *after_ids)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
