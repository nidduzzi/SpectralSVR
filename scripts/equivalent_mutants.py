import sys
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cosmic_ray.commands.init import init
from cosmic_ray.mutating import mutate_code
from cosmic_ray.plugins import get_operator
from cosmic_ray.work_db import WorkDB, use_db
from cosmic_ray.work_item import MutationSpec, WorkerOutcome, WorkResult

REPOSITORY = Path(__file__).resolve().parent.parent
REGISTRY = REPOSITORY / "mutation" / "equivalent.toml"


@dataclass(frozen=True)
class Equivalent:
    module: str
    definition: str
    line: str
    mutant: str
    reason: str


@dataclass(frozen=True)
class SourceSite:
    module: Path
    definition: str
    line: str


@dataclass(frozen=True)
class MutantSite:
    source: SourceSite
    mutant: str


def declared(registry: Path = REGISTRY) -> tuple[Equivalent, ...]:
    return tuple(
        Equivalent(**entry)
        for entry in tomllib.loads(registry.read_text())["equivalent"]
    )


def stripped_line(source: str, row: int) -> str:
    return source.splitlines()[row - 1].strip()


def source_site_of_mutation(
    mutation: MutationSpec, sources: dict[Path, str]
) -> SourceSite:
    module = Path(mutation.module_path).resolve()
    if module not in sources:
        sources[module] = module.read_text()
    return SourceSite(
        module,
        mutation.definition_name or "",
        stripped_line(sources[module], mutation.start_pos[0]),
    )


def mutant_of(mutation: MutationSpec, source: str) -> str:
    operator = get_operator(mutation.operator_name)(**mutation.operator_args)
    mutated = mutate_code(source, operator, mutation.occurrence)
    if mutated is None:
        raise ValueError(f"{mutation} mutates nothing in its module")
    return stripped_line(mutated, mutation.start_pos[0])


def source_site_of(entry: Equivalent, root: Path) -> SourceSite:
    return SourceSite((root / entry.module).resolve(), entry.definition, entry.line)


def mutant_site_of(entry: Equivalent, root: Path) -> MutantSite:
    return MutantSite(source_site_of(entry, root), entry.mutant)


def mutations_in(module: Path) -> tuple[MutationSpec, ...]:
    with (
        tempfile.TemporaryDirectory() as scratch,
        use_db(Path(scratch) / "session.sqlite") as session,
    ):
        init([module], session, {})
        return tuple(
            mutation for item in session.work_items for mutation in item.mutations
        )


def refusals(entries: Iterable[Equivalent], root: Path = REPOSITORY) -> list[str]:
    entries_by_module: dict[str, list[Equivalent]] = defaultdict(list)
    for entry in entries:
        entries_by_module[entry.module].append(entry)
    refused = []
    for module, module_entries in entries_by_module.items():
        sources: dict[Path, str] = {}
        source = (root / module).read_text()
        declared_sources = {source_site_of(entry, root) for entry in module_entries}
        mutated_sources: set[SourceSite] = set()
        mutant_counts: dict[MutantSite, int] = defaultdict(int)
        for mutation in mutations_in(root / module):
            site = source_site_of_mutation(mutation, sources)
            if site in declared_sources:
                mutated_sources.add(site)
                mutant_counts[MutantSite(site, mutant_of(mutation, source))] += 1
        for entry in module_entries:
            count = mutant_counts[mutant_site_of(entry, root)]
            if not entry.reason.strip():
                refused.append(f"{entry}: states no reason")
            if source_site_of(entry, root) not in mutated_sources:
                refused.append(
                    f"{entry}: no mutated line of {module} reads as its line"
                )
            elif count != 1:
                refused.append(f"{entry}: {count} mutants of its line read as declared")
    return refused


def skip_declared(
    session_path: Path, entries: Iterable[Equivalent], root: Path = REPOSITORY
) -> int:
    entries = tuple(entries)
    declared_sources = {source_site_of(entry, root) for entry in entries}
    declared_mutants = {mutant_site_of(entry, root) for entry in entries}
    sources: dict[Path, str] = {}
    with use_db(session_path, WorkDB.Mode.open) as session:
        skipped = [
            item.job_id
            for item in session.pending_work_items
            for mutation in item.mutations
            if (site := source_site_of_mutation(mutation, sources)) in declared_sources
            and MutantSite(site, mutant_of(mutation, sources[site.module]))
            in declared_mutants
        ]
        session.set_multiple_results(
            skipped,
            WorkResult(
                worker_outcome=WorkerOutcome.SKIPPED, output="declared equivalent"
            ),
        )
    return len(skipped)


def main(session_path: str) -> None:
    entries = declared()
    refused = refusals(entries)
    if refused:
        raise SystemExit("\n".join(refused))
    print(
        f"skipped {skip_declared(Path(session_path), entries)} declared equivalent mutants"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
