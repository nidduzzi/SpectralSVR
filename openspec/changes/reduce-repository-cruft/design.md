## Context

See proposal.md for the motivation. The facts that shape the approach:

- `just check` runs `lint` and `test`. `test` wraps pytest in `systemd-run --user --scope` with a 16 GiB cap and uses six xdist workers of two threads. A GitHub runner has no user systemd session, four vCPUs and 16 GB.
- The lock resolves torch from the `pytorch-cu128` index. Those wheels run on a CPU-only machine and weigh about 3 GB.
- Accelerator tests carry `needs_an_accelerator`, which skips them without CUDA.
- Cosmic-ray 8.7 ships filters that mark jobs skipped in the session database. `cr-filter-pragma` skips every operator on a line that carries `# pragma: no mutate`. `cr-filter-operators` skips an operator everywhere. Neither skips one operator on one line.
- Each root `cosmic-ray*.toml` repeats the same memory-cap prefix inside its `test-command`.
- `scripts/run_benchmark.py` writes the record to `benchmarks/operator_comparison.csv` and any other run to the gitignored `results/`. AGENTS.md names `results/` as the record.
- The prose ratchet already delegates docstring shape to ruff's `D` findings (`test_every_ruff_docstring_finding_lands_in_a_counted_docstring`). Its own work is counting comment lines and internal docstrings per file and refusing growth.

## Goals / Non-Goals

**Goals:**
- The gate that runs locally is the gate CI runs, with the same lock.
- An equivalence judgement is stated once, beside nothing but its reason, and dies with the line it judged.
- After this change, no tracked notebook can drift from the library unnoticed.

**Non-Goals:**
- Running mutation testing on every push. A full target takes tens of minutes.
- GPU CI.
- Migrating the evidence of in-flight or archived changes. `fix-real-derivative-at-nyquist` keeps its `survivors.md`, which is its record at its commit.

## Decisions

**CI runs `just check` with the cap and worker count overridden.** `host_memory_cap` and `test_workers` become overridable just variables. CI calls `just --set host_memory_cap "" --set test_workers "-n 4" check`, with `timeout-minutes: 30` and per-branch concurrency that cancels superseded runs. The runner's VM is the memory boundary that the scope provides locally. A separate CI script was the alternative. It would let local and CI gates drift apart, which defeats the point.

**CI installs from the lock as it stands.** `astral-sh/setup-uv` with its cache holds the 3 GB environment between runs. Resolving torch from the CPU index in CI was the alternative. It would test a different torch build from the one the lock and the benchmark use. A lock-cache miss costs a few minutes; testing a different build costs trust in the gate.

**Equivalent mutants live in `mutation/equivalent.toml`, matched by module, operator and source line text.** A filter script in `scripts/` runs after `cosmic-ray init` and marks matching jobs skipped through cosmic-ray's `WorkDB`, the same way the shipped filters do. A test enumerates the mutations of each registry module in-process with cosmic-ray's own operators, without running any, and fails on an entry that matches none, or that has an empty reason. That needs cosmic-ray in the `test` extra. It is pure Python and small.
- Alternative: `# pragma: no mutate`. Rejected. It skips every operator on the line, so it hides killable mutants beside the equivalent one, and its reason would be a prose comment in source.
- Alternative: line numbers. Rejected. Any edit above the line moves them, which is what makes `survivors.md` go stale.
- Two identical lines in one module are matched together. The test refuses an entry whose line text occurs more than once in its module, so that case is never silently ambiguous.

**One config per target under `mutation/`, wrapped once.** `mutation/operators.toml`, `basis.toml`, `model.toml` and `scripts.toml` carry plain pytest test-commands. `just mutate <target> [base]` wraps the whole cosmic-ray session in the memory cap, writes the session database and the report under `${XDG_CACHE_HOME:-~/.cache}/spectralsvr-mutation/`, so a run leaves `git status` unchanged, applies the git filter when `base` is given, and applies the registry filter always. `mutate-changed` merges into it. `.github/workflows/mutate.yml` runs a target on `workflow_dispatch` and uploads the report as an artifact.

**Examples live in `examples/` and run inside the suite.** A test parametrized over `examples/*.ipynb` executes each with `nbclient`, in a kernel whose environment sets `CUDA_VISIBLE_DEVICES=""`, and names the failing cell. The same module fails on a tracked `.ipynb` outside `examples/`, or one carrying an output or execution count. `just strip` clears outputs with `jupyter nbconvert --clear-output --inplace`.
- Alternative: `pytest --nbmake examples` as a separate recipe. Rejected. A run outside the suite cannot carry the scenario and oracle markers the guards require, and it is one more command for the gate to remember.
- Alternative: a pre-commit hook for stripping. Rejected. The repository has no pre-commit setup, and a test catches what a skipped hook misses.
- Network access is not blocked. An example reading remote data is refused at curation (3.2), not by the test.

**Which notebooks become examples is decided one by one with the user.** A notebook qualifies when it runs offline on CPU in under two minutes against the current API, and shows a public path the README does not. The six named scratch notebooks are deleted outright. A notebook that produced a figure in the paper is never deleted without the user saying so. Git history keeps every deleted notebook.

**The prose ratchet stays.** No ruff rule counts comment lines per file or refuses an increase. `ERA001` finds commented-out code only, and the `D` rules are already joined. Replacing the ratchet with ruff would drop the part of the rule that ruff cannot express. Recorded as considered and declined.

**Citation.** `CITATION.cff` at the root, version 1.2.0 of the format, with the paper as `preferred-citation`. The paper commit gets an annotated tag `paper/<venue>-<year>`, pointing at the commit that wrote the benchmark record the paper reports. The tag is created locally. Pushing it is a separate decision for the user.

## Risks / Trade-offs

- [The CI environment differs from the shared host in thread count and CPU model] → reductions can round differently. The suite's tolerances were measured on the host. A CI-only failure is investigated as a measurement, never fixed by loosening.
- [Executing examples adds to the gate's 45 s] → the two-minute qualifying limit per example, and few examples, bound it. The gate's wall time is recorded before and after.
- [Mutation reports as CI artifacts expire after 90 days] → the durable record is the registry plus the commit, from which any run can be repeated. A report a change relies on is summarised with its figures in the commit message, as now.
- [Deleting a notebook loses its committed outputs from HEAD] → git history keeps them, and paper figures are never deleted without sign-off.
- [Removing the local `backup/pre-replay*` tags makes their commits unreachable once git collects garbage] → they are listed with their remote state, and none is deleted without the user's say-so.

## Open Questions

- The paper's title, authors, venue and DOI for `CITATION.cff`, and which commit's record the paper reports. The user supplies these at that task.
