## 0. Baseline

- [x] 0.1 Record the baseline: `just check` green, the suite's test count and wall time, and the scenario names carried by every test in `inventory.md`; verify each single test's row in the inventory still matches the code at this branch's base. *Amended after the rebase onto `fix-time-dependent-resampling`:* record on the new base (`4238a5e`) and regenerate `inventory.md` and `baseline_single_tests.json` there; the first recording (424 passed, 90 s, 187 single tests) predates the fix
  - Recorded on a detached worktree at `86bbb1c` (the fix's archive plus this change's amended plan, without the phase-1 work in progress), with its own environment; `SpectralSVR.__file__` resolved to the worktree. `just lint` is clean, and `just test` gives 441 passed in 92.2 s (105 s wall, lint included). `measurements/scan_single_tests.py` reproduces the first recording exactly when run at `734d7f9` (187 tests, scenario lists equal). At the new base it finds 185 single tests, 27 of them carrying scenarios. The two the fix deleted are gone; none are new, and no scenario name changed. `measurements/build_inventory.py` regenerates `inventory.md` from that scan, the base durations (`measurements/base_test_durations.txt`) and the unchanged classifications. Every row's oracle, classification, target and flag matches the first inventory; only line numbers and durations moved. Counts: FUZZ 73, FOLD-MACHINE 64 (was 66), STAY-SINGLE 35, MERGE-EXISTING 13.
- [ ] 0.2 For each phase below, write its hand mutants of the code its tests cover and record which the current tests kill; verify the list exists before the phase's first edit. *Amended:* rerun phase 1's mutants on the new base, adding the fix's conversion mutants
  - Phase 1, recorded on the base worktree: `measurements/phase1_mutants.json` lists 45 mutants of `__base.py`, `FourierBasis.py`, `sampling.py` and `tests/_anchors.py`, the first recording's 11 plus the fix's conversion mutants (its recorded equivalents excluded). `measurements/phase1_mutants_on_base.json` records all 45 killed by `test_fourier_basis.py`, `test_config.py`, `test_coeff_anchors.py` and `test_sampling.py`. The first recording's three survivors (the two no-op copies and `time_size` reading the wrong axis) are now killed by the fix's tests. Phases 2 to 4 are recorded before their first edits.

## 1. BasisLifecycle

- [ ] 1.1 Build `BasisLifecycle` with rules for generate, empty, copy, resize, time-dependent conversion, grad, integral, evaluation and the domain setter, and invariants for shape, length, domain spans and independence of copies; pin every literal from its ~~23~~ 21 member tests in its initialisation; verify each member's scenario names move to it and the scenario guard passes. *Amended:* it also absorbs the fix's slice-count refusal, both field-refusal tests and the own-slice-count test as rules, drawing `nt` freely across accepted and refused values, with their `@example`s as initialisation pins (each accepted integer type, booleans, arrays of more than zero dimensions, empty bases, one-slice fields both ways, a field without a spatial axis, a wrong-length sampling, shared storage on a no-op); verify their scenario names move and the fix's conversion mutants stay killed under several Hypothesis seeds
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
