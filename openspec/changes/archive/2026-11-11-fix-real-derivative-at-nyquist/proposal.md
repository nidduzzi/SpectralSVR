## Why

On an even number of modes, the boundary mode at `N/2` and the one at `-N/2` are the same sampled wave: both alternate `+1, -1, +1, ...` across the nodes. `FourierBasis` stores that mode as a single wave at `-N/2`. That is a fair reading for a complex field, but for a real field the honest reading is the cosine `c cos(pi N (x - a) / L)`, which splits the coefficient evenly between `+N/2` and `-N/2`. Reading it as one wave makes a real field non-real wherever the two readings differ:

- **Odd-order derivative.** `grad` multiplies the real boundary coefficient by `i (-pi N / L)`, so its values are imaginary at every node. The cosine's derivative is zero at every node. Measured on `Antiderivative` at 16 modes, `ut` has an imaginary part up to 7.9 times its real part.
- **Integration.** `integral` divides by the same eigenvalue, so it inherits the error.
- **Off-grid evaluation.** Between the nodes, a single `-N/2` wave is complex. A default real field at 8 modes evaluates 8.3% imaginary off the grid, against 2.6e-8 at 9 modes. At its own nodes it reads 0, because there the two readings agree.

`coefficient-encodings` already requires that an operation which can return a real field does so, rather than document the limitation. Here the repair exists, so the current behaviour violates that requirement.

An investigation on 2026-10-26 prototyped the derivative repair alone. It moved only the project antiderivative benchmark rows. SpectralSVR moved by about 1e-6, but DeepONet moved by up to 0.057 rrse per seed (mean 0.655 to 0.663), although unrepaired reruns are bit-identical. DeepONet's training is chaotic under rounding-level input changes. That bears on how far its recorded rows can be trusted.

## What Changes

- **Cosine reading of the boundary mode.** A real field (`complex_funcs=False`) on an even mode count reads its boundary mode as the cosine everywhere: in off-grid evaluation (dense, approximate and time-dependent), in the inverse transform at a resolution other than the mode count, and in differentiation and integration. Odd-order derivatives and integrals drop the mode; even-order derivatives keep it.
- **Complex fields keep the single wave.** A field with `complex_funcs=True` keeps the single `-N/2` wave, bit-for-bit as now.
- **Static entry points.** `evaluate`, `inv_transform`, `evaluate_time_dependent_at` and `derivative_eigenvalues` take `complex_funcs`, defaulting to `False` like the field.
- **Native grid unchanged.** Values on the field's own grid are unchanged bit-for-bit, because there the two readings agree.
- **Tests.** The two tests that feed arbitrary complex spectra to `grad` and `integral` state `complex_funcs=True`, and the real case gets its own assertions.
- **Benchmark.** Moved rows are regenerated deliberately. DeepONet's sensitivity to rounding-level input is recorded as a noise floor.

## Capabilities

### New Capabilities

None.

### Modified Capabilities
- `coefficient-encodings`: adds the requirement that a real field reads its boundary mode as a cosine, in evaluation, differentiation and integration.

## Impact

- **Code:** `SpectralSVR/basis/FourierBasis.py` (evaluation, inverse transform off the native grid, derivative eigenvalues, `grad`, `integral`), `SpectralSVR/basis/_nufft.py`, and `SpectralSVR/basis/__base.py` (instance reads passing `complex_funcs`). `Burgers` uses `derivative_eigenvalues` under a de-aliasing mask that already excludes the boundary mode, so its solver is expected not to move. The design gives the check.
- **Tests:** `tests/basis/test_fourier_basis.py` (the two pinned tests and the lifecycle machine), plus whichever tests read real fields off the grid at even counts.
- **Recorded numbers:** the benchmark's targets and predictions are read at query points off the grid, so rows whose fields carry boundary content will move. Every dataset is measured before and after. Moved rows are regenerated with `scripts/run_benchmark.py` from a clean commit, never hand-edited.
- **Callers:** a static call passing complex coefficients without `complex_funcs=True` now has its boundary mode read as a cosine. The library's own callers are checked, and the two tests are amended.
