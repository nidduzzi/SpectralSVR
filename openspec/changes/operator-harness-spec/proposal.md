# Specify the operator comparison harness

## Why

The `operators/` package is what the paper is about, and it is the only significant module in the repository with no stated requirements. It was built directly in `0d94805` with no proposal and no capability. The eight existing main specs cover coefficients, solvers, bases and test policy:

| capability | requirements |
| --- | --- |
| coefficient-encodings | 7 |
| coefficient-precision | 4 |
| manufactured-solutions | 3 |
| numerical-anchor-tests | 12 |
| pde-solver-protocol | 5 |
| semilinear-time-integration | 4 |
| spectral-band-selection | 4 |
| test-oracle-convention | 4 |

None of them mentions `operators/`, `benchmark` or DeepONet. The word `operator`
does occur in `pde-solver-protocol` and `semilinear-time-integration`, but every
occurrence there is the mathematical linear operator of a semilinear system, not
this package.

The gap compounds. `tests/operators/test_operators.py` is listed in the anchor-policy `EXEMPT` set of `tests/test_oracle_coverage.py`, justified as *"thin adapters over anchored model/basis layers"*. That was fair when the package was three adapters. It now carries `benchmark()`, `OperatorDataset`, `sensor_grid`, the DeepONet wrapper and four datasets, and the table it produces is the paper's headline result.

So everything supporting the claim is rigorously specified and independently anchored, and the claim itself is neither. A reader of the specs cannot tell what the comparison guarantees, and a reader of the table cannot tell what it licenses them to conclude.

## What Changes

Specification, documentation, tests — and the constructive modelling that turns
three of the specified guarantees from things the code asks callers to honour
into things the types will not let them break. **BREAKING**: the operator
interface splits, the benchmark takes a split rather than two loose datasets,
and an absent optional dependency stops being a silent row drop.

The computation is untouched, so no reported number moves; what changes is which
object carries a method and which states can be constructed at all.

- State what the `Operator` interface guarantees, and make the order structural: `fit` returns a fitted value that alone carries `predict`, so predicting before fitting is a type error and both adapters' `assert self._model is not None` is deleted rather than documented.
- State what `OperatorDataset` guarantees, which is the whole point of the harness: every operator draws from one object — the same sensor-sampled inputs and the same evaluable output field — each taking the view its method needs, so a difference between two rows is a difference of method rather than of data.
- State what `benchmark()` guarantees: every operator scored against the same targets at the same query points through the same metric suite, one row per operator, with `fit_seconds` measured around `fit` alone.
- Make the two defects below unconstructible rather than documented. Train and test come from partitioning one source, so an overlapping split cannot be stated; a dataset whose sensor count, sampled width and output field disagree fails at construction; and resolving a requested operator whose optional dependency is absent fails unless the caller opts into a short table explicitly.
- State what the resulting table is and is not evidence for. The hyperparameters on both sides are fixed defaults, chosen once and not tuned per dataset, and every dataset runs at a single seed, so it is a demonstration that the harness works, not a controlled comparison of the methods.
- Withdraw the anchor-policy exemption, now that the package carries real numerics rather than thin adapters, and supply the anchor it was standing in for: the operator learns `ut -> u`, and `tests/_anchors.py:antiderivative_pair` already gives that map in closed form. See design.md — Decisions for why the anchor lands here rather than in a follow-up.

**Explicitly deferred** to a separate benchmark-reproducibility change: multi-seed statistics, device and library-version provenance columns, per-sample relative L2 alongside the global rrse, a tuning protocol, and any new dataset. This change writes down what exists; it does not improve it.

Two defects found while writing the capability are fixed here rather than
recorded, because the constructive modelling above removes them: `benchmark()`
never verified that the train and test splits it was handed were disjoint, and
`build_operators` in `scripts/run_benchmark.py` dropped an operator whose
optional dependency was missing, so the table could carry fewer rows than were
asked for with only a warning to say so.

## Capabilities

### New Capabilities
- `operator-comparison`: what the operator interface, the shared dataset and the benchmark table guarantee, and what a reader may and may not conclude from the reported numbers.

### Modified Capabilities
- `test-oracle-convention`: states that an exemption must be re-examined once its module outgrows the justification, and that a withdrawal is a recorded decision — either the anchor lands with it or the anchor it owes is written down. The operators exemption is withdrawn here, with its anchor supplied.

## Impact

- `openspec/specs/operator-comparison/spec.md`: new capability, created on archive.
- `SpectralSVR/operators/`: the `Operator` / `FittedOperator` split, validated `OperatorDataset` construction, a `Split` type that can only be built by partitioning, and the docstrings that currently understate what each component guarantees.
- `scripts/run_benchmark.py`: the module docstring carrying what the table does and does not establish, `disjoint_split` moving into the library as the `Split` constructor, and operator resolution refusing to drop a requested row without an explicit opt-in.
- `tests/operators/test_operators.py`: `test_spectralsvr_operator_learns_antiderivative` is rewritten to score against a closed form instead of the library's own generated field, becoming the module's anchor. Two tests are added for scenarios that had none: a failing operator propagating out of `benchmark()`, and two operators scored against one shared target.
- `tests/test_oracle_coverage.py`: the `EXEMPT` entry is removed, which the anchor above makes possible without leaving the guard failing. What the anchor covers is the SpectralSVR adapter's accuracy; `benchmark()`'s metric plumbing leans on `get_metrics`, anchored in `tests/utils/test_metric_anchors.py`, and `sensor_grid` and `OperatorDataset` are structural. The residual gap is the DeepONet wrapper, whose numerics stay smoke-only — anchoring those would mean anchoring `deepxde`.
- No number moves, but this is no longer provable from the diff alone: library source changes, so the four benchmark rows and the seeded field are compared against a base run in a worktree, read against this stack's measured noise floor.
