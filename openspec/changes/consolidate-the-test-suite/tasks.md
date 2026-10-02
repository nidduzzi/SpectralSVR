## 0. Baseline

- [ ] 0.1 Record the baseline: `just check` green, the suite's test count and wall time, and the scenario names carried by every test in `inventory.md`; verify each single test's row in the inventory still matches the code at this branch's base
- [ ] 0.2 For each phase below, write its hand mutants of the code its tests cover and record which the current tests kill; verify the list exists before the phase's first edit

## 1. BasisLifecycle

- [ ] 1.1 Build `BasisLifecycle` with rules for generate, empty, copy, resize, time-dependent conversion, grad, integral, evaluation and the domain setter, and invariants for shape, length, domain spans and independence of copies; pin every literal from its 23 member tests in its initialisation; verify each member's scenario names move to it and the scenario guard passes
- [ ] 1.2 Fuzz the other single tests of `basis/test_fourier_basis.py` the inventory marks FUZZ, keeping each literal as an example and each closed set enumerated; merge its duplicate; verify every phase-1 mutant the old tests killed is killed, oracle markers are unchanged or stronger, and the suite is green

## 2. Precision and persistence

- [ ] 2.1 Build `SpectralSVRBoundary` over drawn basis and regressor precisions and every training, test, forward and inverse path in any order, absorbing its 10 members; verify as in 1.2
- [ ] 2.2 Build `RegressorPersistence` over both regressor kinds, full and hyperparameter-only dumps, and loads as the same or another kind, absorbing its 6 members; verify as in 1.2
- [ ] 2.3 Build `LSSVRLifecycle` and `ScalerLifecycle`, absorbing their 4 and 3 members; fuzz the rest of `model/test_lssvr.py`; verify as in 1.2

## 3. Devices and resizing

- [ ] 3.1 Build `DeviceStatements` over stated host, stated and unstated indices, a repointed process default and generator placement, absorbing its 5 members; verify as in 1.2
- [ ] 3.2 Build `ResizeRoundTrips` over drawn pad and truncate counts, with the loops' enumerated counts pinned in its initialisation, absorbing its 4 members; keep `test_resize_modes_truncate_values` and `test_resize_modes_pad_values` as pinned analytic cases and add a fuzzed test of resizing against an independent implementation; verify as in 1.2

## 4. Existing machines and the rest of the suite

- [ ] 4.1 Fold the 12 tests the inventory assigns to `BandLimiting`, `DirectingAndFitting`, `BenchmarkRuns`, `BenchmarkCasesStayPut` and `SensorsAndDatasets` as rules, invariants or initialisation pins; verify as in 1.2
- [ ] 4.2 Fuzz the remaining FUZZ tests module by module (`utils/test_utils.py`, `problems/test_problems.py`, `basis/test_config.py`, `basis/test_nufft.py`, `basis/test_coeff_anchors.py`, `utils/test_solvers.py`, `operators/`, `model/`), keeping every measured bound verbatim and drawing only from its measured range, or measuring a wider range first and recording it here; verify as in 1.2
- [ ] 4.3 Keep `test_get_metrics_hand_computed_values` as its pinned analytic case and add a fuzzed test of the metric suite against an independent implementation; verify a mutant of each metric fails it
- [ ] 4.4 Merge or remove the 13 MERGE-EXISTING tests as the inventory names; where a legacy test empties an exempt module, remove the module and its oracle-guard exemption together; verify the oracle guard and its staleness check pass

## 5. Verification

- [ ] 5.1 Verify the inventory is discharged: every single test outside the 35 STAY-SINGLE rows is gone, each STAY-SINGLE row is still single with its reason, and no scenario name was lost
- [ ] 5.2 Run `just check`; verify lint is clean and the suite passes, and record the test count and wall time against the baseline
- [ ] 5.3 Verify no source file gained an explanatory prose comment, every touched docstring is in shape, and the prose budget fell only where prose went
- [ ] 5.4 Verify no recorded number moves, by diff argument: no library, script or record file changed
