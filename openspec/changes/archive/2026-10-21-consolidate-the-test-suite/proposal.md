## Why

The suite holds 187 single-case tests beside 188 fuzzed ones and 10 state machines. A single case checks one literal. Where the behaviour it guards ranges over inputs or depends on a sequence of operations, the literal stands in for a property nobody asserted, and the states a sequence produces go unvisited. The test-oracle convention already requires the opposite of every scenario check -- "Scenarios over an input range are fuzzed" and "A sequence of operations is modelled as one" -- and the changes on this stack wrote their own tests that way, but the older suite predates the rule.

A read-only inventory of all 187, recorded in `inventory.md`, classifies each:

| outcome | tests | what happens |
| --- | ---: | --- |
| stays single | 35 | checks a repository or record fact with no input range: guards over the real repository, docstring text, registries, recorded convergence constants, memory and wall-clock measurements |
| folds into a state machine | 66 | 54 into seven new machines, 12 into existing ones |
| becomes fuzzed | 73 | ranges over its inputs, with today's literal pinned as an example |
| merges or goes | 13 | four polar duplicates, one vacuous import check, two legacy tests in exempt modules, six that become examples elsewhere |

## What Changes

- **Seven new state machines**, each replacing the single tests that share its state:
  - `BasisLifecycle`, ~~23~~ 21 tests: generate, empty, copy, resize, time-dependent conversion, grad, integral, evaluation and the domain setter. *Amended after the rebase onto `fix-time-dependent-resampling`:* that change deleted two of the members, `test_to_time_dependent_and_back` and `test_to_time_dependent_default_nt`, which its fuzzed tests subsume. The machine also absorbs four of its refusal and own-slice-count tests as rules.
  - `SpectralSVRBoundary`, 10: precision narrowing, widening and matching on every path, inverse before training, dtype refusals.
  - `RegressorPersistence`, 6: dump and load for both regressors, full or hyperparameters only, and refusal on a kind mismatch.
  - `DeviceStatements`, 5: stated host or index, unstated index, process default, generator placement.
  - `ResizeRoundTrips`, 4: pad, truncate, conjugate symmetry, preservation of the function, odd counts.
  - `LSSVRLifecycle`, 4: access before and after fitting.
  - `ScalerLifecycle`, 3: fit, subset, save and load, misuse.
- **Existing machines absorb twelve more:** `BandLimiting`, `DirectingAndFitting`, `BenchmarkRuns`, `BenchmarkCasesStayPut` and `SensorsAndDatasets`.
- **73 tests become fuzzed** over the inputs they range over, with today's literal kept as a pinned example and each closed set enumerated.
- **Thirteen duplicates and legacy tests merge or go**, each named in the inventory.
- **The three hand-computed analytic anchors keep their literal** (`test_resize_modes_truncate_values`, `test_resize_modes_pad_values`, `test_get_metrics_hand_computed_values`), decided by the user, and each gains a separate fuzzed test against an independent implementation.

No library code changes, no recorded number moves, and no test loses its oracle class or the scenarios it names.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. The rule this change applies is already stated in `test-oracle-convention` by "Scenarios over an input range are fuzzed" and "A sequence of operations is modelled as one"; this change brings older tests into line with it rather than adding a requirement, so the change declares `skip_specs`.

## Impact

- Most of `tests/`: `basis/test_fourier_basis.py` (55 single tests), `utils/test_utils.py` (23), `problems/test_problems.py` (14), `model/test_spectralsvr_methods.py` (13), `operators/test_operators.py` (11), `model/test_lssvr.py` (10) and the smaller modules listed in `inventory.md`.
- `tests/test_oracle_coverage.py`'s exempt list, where removing a legacy test empties a module it exempts.
- `tests/prose_budget.toml`, whose entries for the touched modules fall as their prose goes.
- No library, script or recorded result.
