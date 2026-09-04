# Specify the operator comparison harness

## Why

The `operators/` package is what the paper is about, and it is the only significant module in the repository with no stated requirements. It was built directly in `0d94805` with no proposal and no capability. The four existing main specs cover solvers, bases and test policy:

| capability | requirements |
| --- | --- |
| numerical-anchor-tests | 10 |
| pde-solver-protocol | 5 |
| semilinear-time-integration | 4 |
| test-oracle-convention | 4 |

None of them mentions `operators/`, `benchmark` or DeepONet.

The gap compounds. `tests/operators/test_operators.py` is listed in the anchor-policy `EXEMPT` set of `tests/test_oracle_coverage.py`, justified as *"thin adapters over anchored model/basis layers"*. That was fair when the package was three adapters. It now carries `benchmark()`, `OperatorDataset`, `sensor_grid`, the DeepONet wrapper and four datasets, and the table it produces is the paper's headline result.

So everything supporting the claim is rigorously specified and independently anchored, and the claim itself is neither. A reader of the specs cannot tell what the comparison guarantees, and a reader of the table cannot tell what it licenses them to conclude.

## What Changes

Documentation and specification only. **No code changes and no behaviour changes.**

- State what the `Operator` interface guarantees: fit then predict, function to function, evaluable at arbitrary query points.
- State what `OperatorDataset` guarantees, which is the whole point of the harness: every operator sees identical data — the same sensor-sampled inputs and the same evaluable output field.
- State what `benchmark()` guarantees: every operator scored against the same targets at the same query points through the same metric suite, one row per operator, with `fit_seconds` measured around `fit` alone.
- State what the resulting table is and is not evidence for. Hyperparameters are currently untuned on both sides and every dataset runs at a single seed, so it is a demonstration that the harness works, not a controlled comparison of the methods.
- Re-justify or remove the anchor-policy exemption, now that the package carries real numerics rather than thin adapters.

**Explicitly deferred** to a separate benchmark-reproducibility change: multi-seed statistics, device and library-version provenance columns, per-sample relative L2 alongside the global rrse, a tuning protocol, and any new dataset. This change writes down what exists; it does not improve it.

## Capabilities

### New Capabilities
- `operator-comparison`: what the operator interface, the shared dataset and the benchmark table guarantee, and what a reader may and may not conclude from the reported numbers.

### Modified Capabilities
- `test-oracle-convention`: revisits the exemption granted to the operators tests, so the exemption is either re-justified against what the package now contains or withdrawn.

## Impact

- `openspec/specs/operator-comparison/spec.md`: new capability, created on archive.
- `SpectralSVR/operators/`: docstrings only, where they currently understate what a component guarantees. No signatures, no behaviour.
- `tests/test_oracle_coverage.py`: the `EXEMPT` entry's justification, or its removal — which, if removed, obliges a follow-up to add an anchor and is therefore recorded as a decision rather than made silently.
- No number moves. No seeded field changes. No benchmark row changes.
