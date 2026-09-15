## 1. Confirm the spec matches the code before writing it down

- [x] 1.1 Verify `benchmark()` scores every operator against one shared `target` tensor computed once from `test.targets(query)`, and that `fit_seconds` brackets only the `fit` call; both are visible at `SpectralSVR/operators/benchmark.py` around lines 47-55
- [x] 1.2 Verify `OperatorDataset.from_fields` samples inputs at a sensor grid and keeps outputs as an evaluable field rather than pre-sampled values, so `targets(x)` is evaluated at whatever query points are asked for
- [x] 1.3 Verify the `Operator` interface requires `fit` before `predict`, and record *how* each adapter enforces it rather than only whether it does: both currently use `assert self._model is not None`, which is a development-time check, so the capability must say that instead of promising an exception that survives optimised execution

## 2. Write the capability

- [x] 2.1 Confirm every requirement in `specs/operator-comparison/spec.md` describes behaviour the code already has; verify each scenario names something checkable against the current implementation. Three overstatements were found during review and are already amended — check the amendments hold: train/test disjointness is a caller precondition that nothing verifies, the operators take differing views of one shared field rather than identical inputs, and the failing-operator guarantee binds `benchmark()` but not the script above it
- [x] 2.2 Verify the two failing-operator scenarios against the code: `benchmark()` (`SpectralSVR/operators/benchmark.py`) has no exception handling, so a failure propagates; `build_operators` in `scripts/run_benchmark.py` drops an operator whose optional dependency is missing and warns, so the table can carry fewer rows than were requested. Record the second as a defect for a follow-up rather than specifying the behaviour we wish it had
- [x] 2.3 Cover the scenarios that had no test: a failing operator must propagate out of `benchmark()` rather than be swallowed, and two operators must be scored against one shared target. Verify each by mutation — swallowing the exception, and scoring against `train.targets` instead of `test.targets` — rather than by the assertion passing

## 3. Docstrings that understate their guarantees

- [x] 3.1 Bring the `benchmark()`, `OperatorDataset` and `Operator` docstrings into line with the capability, adding only what the code already does; verify no signature and no behaviour changes
- [x] 3.2 Put the scope-of-claim statement in `scripts/run_benchmark.py`'s module docstring rather than `benchmark()`'s, per design.md — Decisions: the harness fixes data and metrics but not tuning, and every figure comes from one seed, so the table demonstrates the harness rather than comparing the methods. Verify the wording does not promise the multi-seed and provenance work deferred to the reproducibility change
- [x] 3.3 Name in the same place what differs between the operators' `fit_seconds` beyond the method: `DeepONetOperator.fit` carries `_import_deepxde()` on first call plus a numpy conversion, and `SpectralSVROperator.fit` carries neither. Verify the statement is measured or plainly attributed, not an estimate

## 4. The anchor-policy exemption

The decision is recorded in design.md — Decisions: the exemption is withdrawn and
its anchor lands here. These tasks carry it out in the order that keeps the guard
passing throughout.

- [x] 4.1 Verify the fact that makes the `EXEMPT` entry load-bearing: `tests/operators/test_operators.py` carries only `oracle_smoke` and `oracle_consistency` markers and no anchor marker, so deleting the entry before an anchor exists fails `test_every_module_has_an_anchor` immediately
- [x] 4.2 Measure, before writing any assertion, how far `SpectralSVROperator`'s prediction sits from the closed-form antiderivative across a range of seeds, and record the worst observed error in this change. A tolerance arrived at by tightening until the test passes asserts nothing
- [x] 4.3 Rewrite `test_spectralsvr_operator_learns_antiderivative` to score against `tests/_anchors.py:antiderivative_pair` rather than `test.targets(query)`, so the expected values come from a closed form the library had no hand in; set the bound above the error measured in 4.2. Verify the marker becomes `oracle_analytic`, and verify it discriminates by checking the test fails when the closed form it compares against is perturbed
- [x] 4.4 Delete the `operators/test_operators.py` entry from `EXEMPT` in `tests/test_oracle_coverage.py`; verify both guards pass — `test_every_module_has_an_anchor` because the module now has one, and `test_exempt_list_is_current` because the stale entry is gone

### What the anchor's bound was measured against

`SpectralSVROperator` is scored against `A*cos(2*pi*m*x) -> A/(2*pi*m)*sin(2*pi*m*x)`,
the closed form `antiderivative_pair` states, at probe modes 1-5.

| probe amplitude | worst relative L2, seeds 0-9 |
| --- | --- |
| unit (`A = 1`) | 2.386 |
| tied to the training set's scale | 0.539 |

The first row is not the model's error, it is a measurement mistake worth
recording: a unit-amplitude cosine has L2 `2.83` at the sensors while the
generated training inputs average `57.2`, so the probes sat 20x below the
training distribution and an RBF kernel was extrapolating. Tying the probe
amplitude to `train.f.norm(dim=1).mean()` puts them back in distribution.

`CLOSED_FORM_TOLERANCE = 0.70` sits above the measured 0.539 with headroom, and
the perturbation guard boxes it from the other side: `off.min` runs 0.887-0.898,
so loosening the bound past ~0.887 breaks the test. The admissible window is
`(0.539, 0.887)`.

Mutation testing says what that bound does and does not catch. Zeros
(`rel.max` 1.000), a sign flip (2.000), the `1/(2*pi*m)` factor dropped (30.416)
and a mean-of-training-outputs predictor (1.068) all fail it. A *uniform scale
error does not*: `exact * 1.5` and `exact * 0.5` both read 0.500 and pass. The
anchor catches gross breakage, not a systematic gain error, which is the ceiling
of any bound set by model accuracy rather than by rounding. For
scale, the same fitted operator scored against held-out *generated* functions
reaches a worst relative L2 of `0.809` over the same seeds, so the closed-form
anchor is a tighter bar than the model's general error rather than a looser one.
The test loops seeds 0-4; the bound was measured over 0-9.

## 6. Invalid states the types should not allow

Innermost first, so each step leaves the suite green before the next begins. See
design.md — Decisions for why each rule moves into a constructor.

- [ ] 6.1 Validate `OperatorDataset` at construction: sensor rows against the sampled width, input function count against the output field. Verify each mismatch raises where the dataset is built, and that `from_fields` cannot produce one
- [ ] 6.2 Add a `Split` holding the training and scoring halves, constructible only by partitioning one source, and move `disjoint_split` out of `scripts/run_benchmark.py` into the library as that constructor. Change `benchmark` to take the `Split`. Verify a `Split` cannot be assembled from two independently built datasets, and that the partition covers its source exactly once
- [ ] 6.3 Recast the interface as typestate with no inheritance: make `Operator` and `FittedOperator` `typing.Protocol`s — `name` plus `fit` on the first, `predict` on the second — and have each adapter satisfy them structurally, returning its own fitted value from `fit`. Delete both adapters' `assert self._model is not None`, and the `abc` base they no longer need. Verify no assertion remains; verify the runtime consequence with `assert not hasattr(op, "predict")` on an unfitted operator, since a type-checker rejection is not assertable at run time; and verify `just lint` stays clean, which is where `mypy` and `ty` carry the compile-time half. `test_predict_before_fit_raises` is replaced, not deleted — the scenario it covered still exists, it has just moved to the type level
- [ ] 6.4 Make a silently short table unreachable: resolving a requested operator whose optional dependency is absent raises, and a missing row is produced only under an explicit opt-in flag. Verify the default path cannot drop a requested row, and that the opt-in path still warns
- [ ] 6.5 Verify the four benchmark rows still carry the same numbers after 6.1-6.4, by the base-run comparison described in 8.2 rather than by inspection — the computation is unchanged, so any movement beyond the noise floor is a defect

## 7. Every scenario verifiable, and verified

- [ ] 7.1 Register a `scenario` marker in `tests/conftest.py` beside the oracle markers, so a test declares the scenario it covers as `@pytest.mark.scenario("Predict after fit")`. Verify an unregistered marker name is still a `pytest` warning, so typos surface
- [ ] 7.2 Add `tests/test_scenario_coverage.py`, guarding **both** spec sources: `openspec/specs/**/spec.md`, which is the permanent contract, and the delta specs of *active* changes under `openspec/changes/*/specs/**/spec.md`, which are not yet synced. Exclude `openspec/changes/archive/**`. Verify the guard still sees this change's scenarios after they are archived and synced — a guard reading only the active delta would find nothing once the change moves, and pass by having nothing to check
- [ ] 7.3 Seed the exemption list mechanically from the current state — 92 scenarios in the main specs at the time of writing, plus whatever in this change's deltas sections 6 and 7 do not close. Entries are **per scenario**, keyed by `(capability, scenario name)`, so a scenario newly added to an old capability is still caught rather than inheriting a blanket exemption. Verify that adding a scenario to an already-exempt capability fails the guard
- [ ] 7.9 Record the seeded exemptions as tracked debt rather than a permanent fixture: state in the change that the list is to be worked down capability by capability in follow-up changes, and verify the guard's failure message tells a reader how to remove an entry rather than how to add one
- [ ] 7.4 Fuzz the scenarios that range over inputs — grid sizes, sensor counts, query points, operator lists, seeds — with Hypothesis strategies, replacing the hand-picked literals the current tests use. Verify each property fails under a mutation of the behaviour it describes, not merely that it passes
- [ ] 7.5 Model the harness with a `RuleBasedStateMachine` over dataset construction and repeated `benchmark` runs, asserting after every step that the scored target does not vary with the operator list, that each run yields one row per operator, and that `fit_seconds` never absorbs a prediction. Verify the machine catches a target recomputed inside the operator loop
- [ ] 7.6 Assert the documentation contracts the capability states, the way `tests/utils/test_band.py` does for `ExactModeCount`: `scripts/run_benchmark.py.__doc__` carries the untuned-and-single-seed caveat and the `fit_seconds` asymmetry, and `benchmark.__doc__` states what it guarantees. Verify each assertion fails when its sentence is deleted
- [ ] 7.7 Mark every scenario this change closes with the `scenario` marker, and give a reason to each remaining exemption from 7.3 — including the scenarios no test can reach, such as those describing decisions taken during the change itself. Verify the guard reports exactly the exempt set and nothing else, so the list cannot drift out of date silently
- [ ] 7.8 Verify `tests/test_scenario_coverage.py` satisfies `tests/test_oracle_coverage.py` itself: it is a new test module, so it needs an anchor marker or its own entry in that guard's `EXEMPT`, the way `test_oracle_coverage.py` exempts itself

## 8. Verification

- [ ] 8.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes. The collected test count is no longer expected to hold still — sections 6 and 7 add tests and the operator split rewrites call sites — so verify instead that every test that disappears is one section 6 or 7 deliberately replaced, naming each
- [ ] 8.2 Verify no number moved. The diff argument no longer applies: library source changes, so this needs a base run. Build a detached worktree at the branch point, give it its own environment (a `PYTHONPATH` override loses to the editable install — `explicit-band-type` lost a base run that way and the mistake was caught only by printing a signature first), run the benchmark on both, and compare. Read the result against this stack's measured noise floor of `2.512e-05`, which is what two runs of *identical* code differ by, rather than expecting byte-identity. Also compare the seeded numerical Burgers field, which is deterministic and must hash identically
- [ ] 8.3 Verify no source file gained an explanatory prose comment; the additions here belong in docstrings and in the capability, not inline
