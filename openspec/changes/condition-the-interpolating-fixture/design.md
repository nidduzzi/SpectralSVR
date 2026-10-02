## Context

See proposal.md - Why for the failure and its measured cause. The module's tests build their inputs with `distinct_inputs(functions, sensors, seed)`, a `torch.randn` draw of shape `(functions, 4)`, and fit `interpolating_regressor()`, a linear-kernel LSSVR at `C = 1e6` in float32. Neither `SpectralSVR.train` nor the regressor rescales its inputs, so the linear kernel sees the drawn matrix as it is: the Gram matrix is `F Fᵀ`. Every strategy in the module draws `functions` from 1 to 4 against 4 sensors, so `functions <= sensors` always holds.

Five tests and the `FittingAndPredicting` state machine use the fixture. Only the two that compare predictions with the manufactured values against `PREDICTION_RTOL` depend on its accuracy. The others check refusals, shapes, reported modes, and bit-identity of time-invariant scoring, none of which depend on conditioning.

## Goals / Non-Goals

**Goals:**

- Every input the fixture can produce gives a Gram matrix with a known, bounded conditioning, so the ridge bias is bounded independently of the seed.
- `PREDICTION_RTOL` is set beneath a measurement over that input range, recorded with the change.
- The seed that exposed the gap stays checked on every run.

**Non-Goals:**

- Changing the regressor, its `C`, or its precision. Raising `C` shrinks the bias, but leaves its dependence on conditioning in place: the tail moves, but it does not end.
- Retuning tolerances in the module that do not depend on input conditioning.

## Decisions

### The inputs are orthonormal rows, drawn and then orthogonalised

`distinct_inputs` keeps its signature. It draws the same Gaussian matrix from the same seed, then returns the transpose of the Q factor of its transposed QR decomposition, giving `functions` orthonormal rows in `sensors` dimensions. The Gram matrix is then the identity for every seed, and the linear LSSVR's ridge bias is bounded by roughly the targets' size over `C`, a 1e-6 relative term, plus float32 rounding. The rows stay distinct and seed-dependent, which is what the refusal and layout tests need.

Q is not unique up to sign. The construction fixes the signs, by making R's diagonal positive, so a seed always gives the same inputs.

*Alternative rejected:* reject draws whose smallest Gram eigenvalue falls under a threshold (Hypothesis `assume`). It filters rather than constructs, so the guarantee depends on the threshold chosen, and it leaves the error's dependence on conditioning in place just above it.

*Alternative rejected:* raise `C` to 1e8 and move to float64. On seed 2211 that gives 5.1e-06, but the error is still governed by the draw's conditioning, so a worse draw exists. It also departs from the precision the time-varying path runs at.

### The bound is measured over the constructed range, then set beneath it

The worst relative error is measured with the constructed inputs over a sample far larger than the 300 fields behind the current bound. The sample covers several thousand seeds across every `functions`, `slices` and `modes` the strategies allow, with the one-slice, low-mode corner that failed sampled densely, including seed 2211 itself. The worst value and its location are recorded under the change's `measurements/`. `PREDICTION_RTOL` is set beneath that worst value with stated headroom, and may come out tighter than 5e-4.

### Seed 2211 is pinned on both failing tests

The predictions test gains an `@example` for seed 2211, four functions, one slice, three modes. The state machine takes no `@example`, so it gains an `@initialize` rule that fits that field and checks the invariant, as the other machines in the suite pin their edges.

## Risks / Trade-offs

- **The new inputs change every example these tests draw.** Refusal, shape and bit-identity checks do not depend on the inputs' values, only on their shape and distinctness. → The whole module runs before and after, and every test that passed still passes.
- **Orthonormal rows have norm 1 where `randn` rows had norm near 2.** The targets do not depend on the inputs, so only the fitted weights' scale changes. → None needed; the bias bound is relative to the targets.
- **A tighter bound could become fragile on another machine.** → The headroom is stated and measured; the bound sits beneath the worst observed value, not at it.

## Decided after the review

**The QR sign correction stays, and an independent Gram-Schmidt pins it.** Decided
by the user. QR's sign convention follows the platform's LAPACK, so the correction
cannot be observed on one machine; classical Gram-Schmidt, written out in the
test, always yields a positive R diagonal, and comparing the fixture with it makes
the convention observable everywhere. The same test asserts that different seeds
give different rows, which the review showed nothing checked.

**Tightness is anchored to the measurement.** Decided by the user. A test reads the
recorded worst from `measurements/` and requires `PREDICTION_RTOL` to sit within
ten times it, so loosening the bound fails and moving it means regenerating the
record.

**Liveness is shown at the run-to-run floor, and both tests' metrics are measured.**
The worst fields' errors move between identical runs (seed 253 alternates between
1.37e-06 and 1.72e-06), so a single run's value is not a liveness threshold. The
state machine's check, scaled by ``max(|targets|, 1)`` over modes 1 to 7 and a
time-invariant branch, is measured alongside the predictions test's.

