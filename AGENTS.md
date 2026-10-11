# AGENTS.md

Context for coding agents working on SpectralSVR. Follows the
[AGENTS.md](https://agents.md/) convention: a predictable place for the build
steps, tests and conventions that would clutter a README.

This is a guide, not a log. It states the conventions the repository holds to as
it stands: when a change introduces a tool or a rule that outlives it, add the
rule to the section it belongs in, on the branch that introduces it. Reasoning
about a particular change belongs in `openspec/changes/<name>/` instead.

## What this project is

SpectralSVR is an operator-learning library: support-vector regression over
spectral (Fourier) coefficient representations, compared against neural-operator
baselines. It is research software, and **its code and reported numbers appear in
a peer-reviewed paper**.

That single fact drives most of what follows. A number in `results/` is a claim
someone will cite. A test that passes for the wrong reason is worse than no test,
because it licenses a claim nobody checked.

## Setup and commands

Python 3.13 exactly (`>=3.13,<3.14`), managed with [uv](https://docs.astral.sh/uv/).

```bash
just sync          # create/refresh the dev environment
just check         # THE GATE: lint (ruff + mypy + ty) then the full test suite
just test -k name  # a subset; extra args pass through to pytest
just cov           # suite with a missing-lines report
just fmt           # ruff format + ruff check --fix
just mutate        # cosmic-ray over SpectralSVR/operators (slow)
```

`just check` is the gate. Run it before claiming work is done, and report its
output rather than a summary of it. Every pytest run ends with its fifteen
slowest phases above one second, so a slow suite names its own cost; read that
report before guessing where the time goes.

Every test and measurement run is hard-capped, so a runaway process is killed
rather than taking the machine down. Host memory: `just test` and `just cov`
run inside `host_memory_cap` (a systemd scope, 16 GiB, no swap), and the
cosmic-ray test commands use the same limit; wrap an ad hoc measurement script
the same way. Accelerator memory: `tests/conftest.py` limits the CUDA
allocator to a quarter of the device; a script outside pytest calls
`torch.cuda.set_per_process_memory_fraction` itself. Optional extras: `test`, `lint`, `baselines`
(DeepONet via deepxde), `notebook`.

## Tests carry an oracle class

Every test declares where its expected values came from. `tests/conftest.py`
enforces this: a test must carry **exactly one** `oracle_*` marker, and must
either be fuzzed with Hypothesis or be marked `no_fuzz` deliberately.

| marker | expected values come from |
| --- | --- |
| `oracle_independent` | an independent implementation |
| `oracle_analytic` | a closed form or manufactured solution |
| `oracle_invariant` | a mathematical invariant (linearity, conservation, a rate) |
| `oracle_consistency` | this codebase's own output — self-referential |
| `oracle_smoke` | shape, dtype, no-crash, raises |

The first three are **anchors**. `tests/test_oracle_coverage.py` requires every
test module to carry at least one, because consistency and smoke tests cannot
catch a bug that moves both sides of a comparison. Its `EXEMPT` set is short and
each entry states why.

Marking honestly matters more than marking favourably. A test comparing a value
against the library's own output is `oracle_consistency` even when the arithmetic
around it was hand-derived.

### Every stated scenario is checkable

A scenario in a spec is a claim, so `tests/test_scenario_coverage.py` requires one
of two things for each: a test naming it with `@pytest.mark.scenario("...")`, or an
entry in `EXEMPT` giving the reason no test can. It reads the main specs *and* the
deltas of changes still in flight, so a contract is watched from the moment it is
written, not from the moment it is archived.

`EXEMPT` is debt, not furniture. Entries are keyed per scenario rather than per
capability, so a scenario added later to an exempt area is still caught, and a
second test refuses any entry that has quietly gained a test. Shrink the list; do
not grow it.

Fuzz a scenario that ranges over inputs rather than pinning it to one literal.
Where a requirement constrains what stays true *across* a sequence rather than
within one call, model it with a `RuleBasedStateMachine` — single-shot cases
cannot reach the states a sequence produces.

### A test that cannot fail is worth nothing

`just mutate` runs cosmic-ray over the operator harness and reports which mutants
survive. Hand-picked mutations test what the author expected to matter; this
enumerates them instead.

Read the survivors rather than counting them. Some are equivalent mutants no test
can kill, and identifying one is a result to record in the change's artifacts.
The rest are real gaps, and they are often asymmetric: a suite that only ever
moves a quantity in one direction leaves the opposite comparison alive until both
directions are covered.

## Numerical discipline

**Measure before you bound.** A tolerance, a convergence order, an error
threshold — measure it first, record the measurement in the change's artifacts,
then set the assertion beneath what was measured. A bound loosened until the test
passes asserts nothing; a bound predicted from theory and never checked asserts
less than it appears to.

**Say what a number cannot establish.** The benchmark harness is not run-to-run
deterministic; multi-threaded reductions in the solve move metrics between runs of
identical code. Before calling a difference real, know the noise floor and compare
against it. Regenerate rather than hand-edit any recorded result.

**Reported numbers move only deliberately.** When a change should not move a
benchmark row or a seeded field, prove it rather than assert it. A diff argument
works when nothing executable changed; otherwise build a base run from a detached
worktree **with its own environment** and compare. A `PYTHONPATH` override loses
to an editable install, which silently benchmarks the wrong library — print the
signature of what loaded before trusting the numbers.

**Precision is configuration, not a constant.** `SpectralConfig.coeff_dtype`
carries the complex dtype of a coefficient grid, `real_dtype` derives the matching
real precision, and `with_precision` returns a new config rather than mutating
one. A basis and the model consuming it move together, and `SpectralSVR` checks
that at the regressor boundary on train, test, forward and inverse. The check is
asymmetric by design: converting to a **lower** precision than the regressor
expects raises, because it discards information the caller asked to keep;
converting to a **higher** one warns, because it only wastes memory. Double
precision costs twice the memory for the coefficient grid — choose it for work
whose accuracy demands it, not by default.

## Planning lives in openspec/

`openspec/` is the audit trail, not documentation-after-the-fact.

- `openspec/specs/<capability>/spec.md` — the behavioural contract the code is
  held to. Requirements with `SHALL`, each carrying `WHEN`/`THEN` scenarios.
- `openspec/changes/<name>/` — a change in flight: `proposal.md` (why),
  `specs/` (delta), `design.md` (how), `tasks.md` (steps, each with a stated
  verification). Archived to `changes/archive/` once its specs are synced.

Implementation should trace to a task. Tick a task only when its stated
verification has actually been run. When a task turns out to be wrong, amend the
task and say so — an artifact that quietly disagrees with the code is worse than
one that records the correction.

## Branches and commits

Work lands in stacked branches, each one change, rebased onto its parent rather
than merged. A branch carries only what its own change introduces; when a shared
file needs an entry for a change, add it on the branch where that change becomes
active, not on one further up.

Commits are prose, not bullet lists: what changed, what was measured, what the
measurement showed, and what it does not establish. Include the figures. A reader
reconstructing a decision from `git log` should not have to guess.

Do not commit or push without being asked. Each commit is an attributable claim in
the paper's provenance.

## Code style

Ruff for lint and format, mypy and ty for types, all wired into `just lint`.

Express intent through names and structure. Contracts belong in docstrings;
explanatory prose comments in source do not. Where a rule can be carried by a
type rather than a runtime check, prefer the type: structural `Protocol`s over
inheritance, frozen value types over mutable ones, and a constructor that refuses
an invalid argument over a check that notices it later.

Where a set of alternatives is genuinely closed, model it as a closed union of
frozen dataclasses rather than an abstract base. `Band = RealBand |
ExactModeCount` in `utils` is the worked form: enumerating the union is
exhaustive rather than a sample, so no further member can be written that claims
a guarantee it does not deliver. An operation that needs the guarantee annotates
the narrower alias, and the type checker refuses the rest at the call site.

Two limits apply before reaching for a `Protocol`. A member written as
`name: str` demands a *settable* attribute, which a frozen dataclass cannot
satisfy — declare it as a read-only property instead. And `@runtime_checkable`
checks member presence only, never signatures, so a one-method protocol is a weak
runtime discriminator where that method name is common; the static checkers carry
the guarantee, `isinstance` only spells it.

## Do not

- Loosen an assertion, tolerance or oracle marker to make a test pass. That is a
  scientific decision, not a fix.
- Hand-edit anything under `results/` or any recorded measurement. Regenerate it.
- Edit anything under `openspec/changes/archive/`. Archived changes are the
  immutable record of what was decided and why.
- Rewrite published history.
