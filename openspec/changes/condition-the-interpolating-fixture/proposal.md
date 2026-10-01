## Why

`just check` fails on every branch above `66a605c`. Two tests in `tests/operators/test_time_varying_scoring.py` (`test_predictions_of_a_time_varying_output_are_read_at_the_same_points` and `TestFittingAndPredicting`) exceed their prediction bound, `PREDICTION_RTOL = 5e-4` of the field's largest value. The failing field is one Hypothesis drew: seed 2211, four functions, one slice, three modes. Its error is 5.6e-04 at the head of `benchmark-reproducibility` and 5.7e-04 at `66a605c`, measured from a worktree with its own environment. The failure therefore predates the change that found it.

The bound is not wrong, and neither is the library. The fixture is. `interpolating_regressor` is a linear-kernel LSSVR with `C = 1e6`. That is ridge regression, not interpolation, and its bias grows as the Gram matrix of the random inputs nears singularity. The inputs are a 4x4 `torch.randn` draw, so nothing bounds that conditioning. These figures were measured on seed 2211:

- **Precision is not the cause.** In float64 the error is 5.1e-04 at the same `C`, against 5.4e-04 in float32.
- **The error scales as 1/C.** It is 4.6e-02 at `C = 1e4`; at `C = 1e8` it is 1.1e-04 in float32 and 5.1e-06 in float64.
- **The Gram matrix is near-singular.** Its smallest eigenvalue is 7.3e-04.

Over 400 seeds at the same shape the worst error was 2.25e-04, and log error correlated with log smallest eigenvalue at -0.48. The error has a heavy tail with no ceiling, so no fixed relative bound holds over every draw. The 5e-4 bound was measured on a sample that had not reached the tail.

What these tests check is that predictions are read at the right (t, x) points, in the targets' layout. The regressor's accuracy is only the means to that, so its accuracy must not depend on the luck of the draw.

## What Changes

- The fixture's inputs become well-conditioned by construction: orthonormal rows, so the Gram matrix is the identity and the ridge bias is bounded near 1/C whatever the draw. The bound is not loosened, and no library code changes.
- The worst error is measured over a large sample that includes the one-slice, low-mode corner where the failure was found. It is recorded with the change, and `PREDICTION_RTOL` is set beneath the measurement. The bound may tighten.
- Seed 2211 is pinned as an example on both failing tests.
- The rule the fixture broke is stated as a testing convention: a fuzzed anchor's tolerance holds over every input its fuzzing can draw. Where the oracle's accuracy depends on a property of the drawn inputs, the fixture guarantees that property by construction rather than leaving it to the draw.

**Not in scope:** the library's LSSVR, its regularisation, and its precision checks; and the other tolerances in the module, which do not depend on input conditioning (the targets test compares exact evaluations to rounding).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `test-oracle-convention`: gains a requirement that a fuzzed anchor's tolerance holds over the whole range its fuzzing draws from, with the fixture constraining any input property the oracle's accuracy depends on.

## Impact

- `tests/operators/test_time_varying_scoring.py`: the fixture's inputs, the measured `PREDICTION_RTOL`, and the pinned example.
- `tests/test_scenario_coverage.py`: in-flight entries for the new scenarios until their tests name them.
- No library, script or recorded result changes. `benchmarks/operator_comparison.csv` is untouched: the benchmark does not use this fixture.
- Depends on nothing beneath it except the module as it stands; it sits at the top of the stack, above `ratchet-code-prose`.
