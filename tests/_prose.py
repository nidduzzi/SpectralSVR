"""How much prose a source file carries: comment lines, and docstring lines by visibility."""

import ast
import io
import re
import tokenize
import tomllib
from dataclasses import dataclass
from pathlib import Path

MACHINE_DIRECTIVE = re.compile(
    r"^#!|#\s*(noqa|type:|pragma:|ruff:|mypy:|pyright:|fmt:|isort:)"
)
LIBRARY = "SpectralSVR"
COUNTED_AREAS = ("SpectralSVR", "tests", "scripts")


@dataclass(frozen=True)
class Prose:
    """Counted lines in one file; ``public_docstrings`` is measured but never budgeted."""

    comments: int
    internal_docstrings: int
    public_docstrings: int


def comment_lines(source: str) -> int:
    return sum(
        1
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and not MACHINE_DIRECTIVE.search(token.string)
    )


def is_internal_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def docstring_span(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    docstring = ast.get_docstring(node, clean=False)
    return 0 if docstring is None else len(docstring.splitlines())


def docstring_lines(source: str, exported: frozenset[str] | None) -> tuple[int, int]:
    """``(public, internal)`` docstring lines; ``exported=None`` means the leading underscore alone decides."""
    tree = ast.parse(source)
    module_public = exported is None or bool(exported)
    public = docstring_span(tree) if module_public else 0
    internal = 0 if module_public else docstring_span(tree)
    definitions = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if not isinstance(node, definitions):
            continue
        node_public = not is_internal_name(node.name) and (
            exported is None or node.name in exported
        )
        for inner in ast.walk(node):
            if not isinstance(inner, definitions):
                continue
            if inner is node:
                inner_public = node_public
            else:
                inner_public = (
                    node_public
                    and isinstance(node, ast.ClassDef)
                    and inner in node.body
                    and not is_internal_name(inner.name)
                )
            span = docstring_span(inner)
            public, internal = (
                (public + span, internal) if inner_public else (public, internal + span)
            )
    return public, internal


def module_file(package_directory: Path, level: int, module: str | None) -> Path:
    base = package_directory
    for _ in range(level - 1):
        base = base.parent
    target = base.joinpath(*module.split(".")) if module else base
    return target / "__init__.py" if target.is_dir() else target.with_suffix(".py")


def library_exports(root: Path) -> dict[Path, frozenset[str]]:
    """Every library file mapped to the names its packages export from it."""
    imports: dict[Path, dict[str, tuple[Path, str]]] = {}
    defined: dict[Path, set[str]] = {}
    for init in (root / LIBRARY).rglob("__init__.py"):
        tree = ast.parse(init.read_text())
        imports[init] = {
            (alias.asname or alias.name): (
                module_file(init.parent, node.level, node.module),
                alias.name,
            )
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.level >= 1
            for alias in node.names
        }
        defined[init] = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def source_of(
        file: Path, name: str, seen: frozenset[tuple[Path, str]]
    ) -> tuple[Path, str]:
        if (file, name) in seen or name not in imports.get(file, {}):
            return file, name
        target, original = imports[file][name]
        return source_of(target, original, seen | {(file, name)})

    exported: dict[Path, set[str]] = {}
    for init, names in imports.items():
        for name in names:
            file, original = source_of(init, name, frozenset())
            exported.setdefault(file, set()).add(original)
    for init, names in defined.items():
        exported.setdefault(init, set()).update(names)
    return {file: frozenset(names) for file, names in exported.items()}


def measure(root: Path) -> dict[str, Prose]:
    """Prose per file under every counted area, keyed by path relative to ``root``."""
    exports = library_exports(root)
    measured = {}
    for area in COUNTED_AREAS:
        for file in sorted((root / area).rglob("*.py")):
            source = file.read_text()
            exported = exports.get(file, frozenset()) if area == LIBRARY else None
            public, internal = docstring_lines(source, exported)
            measured[file.relative_to(root).as_posix()] = Prose(
                comments=comment_lines(source),
                internal_docstrings=internal,
                public_docstrings=public,
            )
    return measured


@dataclass(frozen=True)
class Budget:
    """The prose one file may carry; an unlisted file's budget is zero in both columns."""

    comments: int = 0
    internal_docstrings: int = 0


BUDGETED_COLUMNS = ("comments", "internal_docstrings")


def read_budget(path: Path) -> dict[str, Budget]:
    with path.open("rb") as handle:
        return {
            file: Budget(**columns) for file, columns in tomllib.load(handle).items()
        }


def over_budget(measured: dict[str, Prose], budget: dict[str, Budget]) -> list[str]:
    return [
        f"{file}: {column} {getattr(prose, column)} over its budget of "
        f"{getattr(budget.get(file, Budget()), column)}"
        for file, prose in sorted(measured.items())
        for column in BUDGETED_COLUMNS
        if getattr(prose, column) > getattr(budget.get(file, Budget()), column)
    ]


def slack(measured: dict[str, Prose], budget: dict[str, Budget]) -> list[str]:
    absent = Prose(comments=0, internal_docstrings=0, public_docstrings=0)
    return [
        f"{file}: {column} budget {getattr(allowance, column)} is above the "
        f"{getattr(measured.get(file, absent), column)} the file carries; lower it"
        for file, allowance in sorted(budget.items())
        for column in BUDGETED_COLUMNS
        if getattr(allowance, column) > getattr(measured.get(file, absent), column)
    ]
