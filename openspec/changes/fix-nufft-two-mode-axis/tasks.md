## 1. Failing first

- [x] 1.1 Record the crash map as a measurement script in `measurements/` (every mode tuple of one to three axes, 1 to 6 modes per axis) and verify it reproduces the 174 failures, exactly the tuples with an axis of 1 or 2 modes, on the unchanged library
- [x] 1.2 Add to `tests/basis/test_nufft.py` a fuzzed test of the scenario "A field with a one- or two-mode axis evaluates approximately" (one to three axes, modes drawn from 1 upward, a 1-mode and a 2-mode axis pinned as `@example`s) and a fuzzed `oracle_independent` test of "Approximate and exact values agree at every mode count" against exact dense evaluation, each carrying its scenario mark; verify both fail with the torchkbnufft `IndexError` on the unchanged library

## 2. Fix

- [x] 2.1 Make `nufft_evaluate` state a grid size per axis of the larger of twice the mode count and the operator's neighbourhood size, read from the operator's own default; verify the tests of 1.2 pass and the crash map of 1.1 now reports no failure
  - `nufft_evaluate` passes `grid_size` per axis as `max(2 * m, numpoints)`, with `numpoints` read from `KbNufft`'s own signature (6 in torchkbnufft 1.5.2). The two tests of 1.2 pass (12 of 12 in `test_nufft.py`) and the crash map reports 0 failing mode tuples. For axes of 3 or more modes the grid is the operator's default `2 * m`, and a direct comparison of the two operators over 30 mode tuples of one and two axes from 3 to 13 modes gives bitwise-equal values in all 30.
- [ ] 2.2 Measure the worst relative error of approximate against exact evaluation over modes 1 to 48 on one axis and 1 to 12 on two and three axes, record it in `measurements/`, and verify the existing 5e-2 bound sits above it before the widened tests assert it

## 3. Widen the guarded tests

- [ ] 3.1 Draw time modes from 2 again in the strategy-conversion test of `tests/basis/test_coeff_anchors.py`, pinning the 2-mode case as an `@example`, and draw mode counts in the existing NUFFT agreement tests from 1 upward where they ran from 8; verify they pass and that no test still restricts a mode count to avoid the crash (grep the suite for the restriction)
- [ ] 3.2 Verify by cosmic-ray over `SpectralSVR/basis/_nufft.py`, against the tests that cover each line, that the grid-size mutants (the comparison, the factor two, the neighbourhood read) are killed, and record the survivors with a verdict each

## 4. Verification

- [ ] 4.1 Run `just check`; verify lint is clean and the suite passes, and record the test count and wall time
- [ ] 4.2 Verify no recorded number moves, by diff argument: the only behavioural change is the grid for axes of 1 or 2 modes, which raised before, and no file under `results/` or `benchmarks/` changes
