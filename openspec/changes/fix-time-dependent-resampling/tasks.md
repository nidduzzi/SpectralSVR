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
  - First run: 1 failed, 425 passed in 98 s. `test_to_time_dependent_preserves_values` failed because `to_time_independent` carries the mirror defect (see the proposal), which led to section 4. Rerun in 4.8.
- [ ] 3.2 Verify no benchmark row moves: the benchmark's time-dependent fields come from the Burgers generator, untouched here, so a diff argument suffices
- [ ] 3.3 Hand-mutate the fix (half-open placement restored, the old reshape restored, the two-slice refusal removed) and verify each fails a test
- [ ] 3.4 Verify no explanatory prose comment was added and every touched docstring is in shape, lowering the prose budget where touched prose went

## 4. The inverse conversion

*Added after the gate's first run; the user decided to fix it here by interpolation.*

- [ ] 4.1 Register the two new scenarios as in flight; verify the scenario guard passes
- [ ] 4.2 Write the fuzzed tests first: a converted time-dependent field reads, at each node of its time-independent result, as the time-dependent field at that node's time; a field periodic in time round-trips within `Δt² / 8 · max |∂²u/∂t²|`, with slice counts below, at and above the time axis's mode count pinned; verify both fail on the current code
- [ ] 4.3 Measure the node agreement and the round-trip error against the bound over the drawn ranges, record them here, and set the rounding tolerance above what was measured
- [ ] 4.4 Make `to_time_independent` read the slices at the periodic times by linear interpolation and transform; verify the new tests pass
- [ ] 4.5 Rewrite `test_to_time_independent_values_match_analytic` to expect each half-open node to hold the slices' linear interpolation at its time, computed with `numpy.interp`, and state the oracle class honestly; verify it fails on the closed-time read and passes on the fix
- [ ] 4.6 Rewrite `test_to_time_dependent_preserves_values` to compare each representation with the source at its own grid's coordinates, and the round trip within the interpolation bound, stating the oracle class honestly; verify it fails with either method's misplacement restored
- [ ] 4.7 Name the two scenarios on the tests that check them and remove their in-flight entries; verify the scenario guards pass
- [ ] 4.8 Rerun 3.1, 3.3 with the closed-time read in `to_time_independent` restored as a further mutant, and 3.4
