## 1. Failing first

- [ ] 1.1 Record the defect as a measurement script in `measurements/`: generate a manufactured pair time-dependent and not, from one seed, and report the largest difference between each stored slice and the non-time-dependent pair at that slice's inclusive time, relative to the largest value, at 2, 5, 11 and 50 slices for field and forcing; verify it reproduces the 0.86, 0.49 and 0.11 of the proposal on the unchanged library
- [ ] 1.2 Measure the conversion's rounding over the test's ranges (slices 2 to 40, time and space modes 4 to 16, single and double precision) with the library's own `to_time_dependent`, record it, and set the test's tolerance a stated factor above the recorded worst case; verify the worst case sits under the tolerance
- [ ] 1.3 Add a fuzzed `oracle_consistency` test (its expected values are the library's own non-time-dependent pair, which the continuous-residual anchors already hold to the equation) naming the scenario "Each slice reads as the manufactured field at its time", pinning 2 slices, slices equal to the time modes, fewer and more slices than time modes, and odd and even counts as `@example`s; verify it fails on the unchanged library on every pinned case

## 2. Fix

- [ ] 2.1 Convert the manufactured pair through `to_time_dependent` in `Burgers.generate` and remove the local helper; verify the test of 1.3 passes, the script of 1.1 reports single-precision rounding, and `tests/problems/test_problems.py` passes

## 3. Verification

- [ ] 3.1 Run cosmic-ray over the manufactured branch of `SpectralSVR/problems/burgers.py`, against the tests that cover each line, and record each survivor with a verdict; verify every mutant of the changed lines is killed or recorded as equivalent or dead
- [ ] 3.2 Run `just check` through the resource wrapper; verify lint is clean and the suite passes, and record the test count and wall time
- [ ] 3.3 Verify no recorded number moves: grep `scripts/` and `notebooks/` for a Burgers generator call that takes the time-dependent manufactured path (expected: none), confirm the only library change is in that branch, and that no file under `results/` or `benchmarks/` changes
