## 1. Pin the defects

- [ ] 1.1 Register the change's two scenarios as in flight in `tests/test_scenario_coverage.py`; verify the scenario guard passes with them
- [ ] 1.2 Write the fuzzed conversion test first: slice counts drawn below, at and above the time axis's mode count, with modes `(3, 2)` at `nt=2`, each of those three relations and the default pinned; at the slice times the converted field must match the source to rounding, between them within `Δt² / 8 · max |∂²u/∂t²|`; verify it fails on the current code, on the crash and on the placement
- [ ] 1.3 Measure the agreement at the slice times over the drawn range before setting the rounding tolerance, and record the measurement here

## 2. Fix

- [ ] 2.1 Make `to_time_dependent` evaluate the source at the inclusive slice times on each spatial axis's periodic grid and reshape to `(len, nt, *modes[1:])`, refusing fewer than two slices with a named error; verify the new test passes and the refusal raises naming the count
- [ ] 2.2 Rewrite `test_to_time_dependent_slice_spectra_match_analytic` to expect slice `j` at `j / (nt - 1)` with its tolerance unchanged; verify it fails on the old placement and passes on the new
- [ ] 2.3 Name the two scenarios on the tests that check them and remove their in-flight entries; verify the scenario guards pass

## 3. Verification

- [ ] 3.1 Run `just check`; verify the suite passes in full, naming any test whose expectation moved, and record the wall time against the parent
- [ ] 3.2 Verify no benchmark row moves: the benchmark's time-dependent fields come from the Burgers generator, untouched here, so a diff argument suffices
- [ ] 3.3 Hand-mutate the fix (half-open placement restored, the old reshape restored, the two-slice refusal removed) and verify each fails a test
- [ ] 3.4 Verify no explanatory prose comment was added and every touched docstring is in shape, lowering the prose budget where touched prose went
