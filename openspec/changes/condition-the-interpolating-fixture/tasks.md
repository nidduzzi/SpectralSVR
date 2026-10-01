## 1. Record the scenarios and pin the failure

- [ ] 1.1 Add in-flight entries in `tests/test_scenario_coverage.py` for the two new scenarios, "An oracle's accuracy does not depend on the luck of the draw" and "A draw found to exceed the tolerance is pinned"; verify the scenario guard passes with them and fails with either removed
- [ ] 1.2 Pin seed 2211 (four functions, one slice, three modes) on both failing tests: an `@example` on the predictions test and an `@initialize` rule on `FittingAndPredicting` that fits that field and checks the invariant; verify both fail on the current fixture with the error reported against `PREDICTION_RTOL`, before anything else changes

## 2. Condition the fixture

- [ ] 2.1 Make `distinct_inputs` return orthonormal rows: the same seeded Gaussian draw, orthogonalised through QR with R's diagonal made positive, so a seed always gives the same rows; verify over a fuzzed range of seeds, functions 1 to 4 and four sensors that the Gram matrix equals the identity to float64 rounding and that two calls with one seed are identical, and that both pinned seed-2211 cases now pass
- [ ] 2.2 Run the whole module before and after the change; verify every test that passed before still passes, since the refusal, shape, reported-modes and bit-identity checks depend on the inputs' shape and distinctness rather than their values

## 3. Measure, then bound

- [ ] 3.1 Measure the worst relative prediction error with the constructed inputs over several thousand seeds spanning every functions, slices and modes the strategies allow, with the one-slice, low-mode corner sampled densely and seed 2211 included; record the worst value, its location and the distribution's upper percentiles in `measurements/`, beside the script that produced them
- [ ] 3.2 Set `PREDICTION_RTOL` beneath the measured worst, recording the headroom in this task's note rather than in a comment; verify the two tests pass at several Hypothesis seeds, and that setting the bound just under the measured worst makes the pinned or sampled worst case fail, so the bound is shown to be live
- [ ] 3.3 Name the new scenarios on the tests that check them (the predictions test and the state machine for the first, the pinned examples for the second), and remove their in-flight entries; verify the scenario guard and the stale-exemption guard pass

## 4. Verification

- [ ] 4.1 Run `just check`; verify lint is clean and the suite passes in full, the two failures this change exists for included
- [ ] 4.2 Verify no source or test file gained an explanatory prose comment; the fixture's name and the recorded measurement carry the reasoning
