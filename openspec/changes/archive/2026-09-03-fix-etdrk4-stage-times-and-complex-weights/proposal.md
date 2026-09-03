# Fix ETDRK4 stage times and complex-eigenvalue weights

## Why

`etdrk4_solver` (`SpectralSVR/utils/__init__.py`) has two defects that the test suite cannot see because every test shares the solver's hidden assumptions. Both were reported in a user-supplied review of this module (§9 of the ETDRK4 course PDF) and reproduced in this session against the shipped torch code, not a re-implementation.

**Bug 1.** The Cox–Matthews scheme evaluates the nonlinear term at four stage times, `t_n, t_n+h/2, t_n+h/2, t_n+h`. The shipped loop passes `ti = t[i-1]` to all four calls. On `v' = -200v + N(t)` with exact solution `sin(3t)+2`, the measured order as `h` halves is `1.22, 1.20, 1.12, 1.07, 1.04`; with the correct stage times it is `3.54, 3.85, 3.96, 3.99, 4.00`. A fourth-order method ships as first order the moment `N` depends on `t`.

**Bug 2.** `_coef` averages the weight functions over the upper half of the unit circle and takes `.real`. That shortcut relies on the two half-circles being conjugate mirror images, which holds only when `L` is real. For `L ∈ {-2i, -20i, -200i, 5i}` at `h = 0.1` the four weights are off by `2.3e-2, 2.1e-2, 1.4e-2, 5.2e-3`; a full-circle average with the complex mean kept is exact to `1e-15`. Any wave-like operator (KdV, Schrödinger, a purely advective linear part) gets wrong weights with no error raised, and the `_solve_spectral` docstring's claim to work for "any basis whose differentiation is diagonal" is false as written.

Neither bug touches the data the library has actually generated: every Burgers run uses constant forcing (so `N` has no `t`) and a real `L = -νk²`. That is also why 249 tests pass — the suite's one time-dependent-forcing test asserts `< 1e-2` at 300 steps, which a first-order method clears. The tests share the assumptions the code gets wrong; the `numerical-anchor-tests` capability exists precisely to prevent that.

## What Changes

- Evaluate the four ETDRK4 stages at `t_n, t_n+h/2, t_n+h/2, t_n+h`. Output is unchanged for any `N` without explicit time dependence.
- Compute the contour weights by a branch on the linear operator's imaginary content: keep today's half-circle average with `.real` when `L` has no imaginary part (identical arithmetic, identical results); use the full unit circle and keep the complex mean when it does. Detection is on the tensor's imaginary values, not its dtype — the Fourier second-derivative eigenvalues are stored as `complex64` with zero imaginary part, so a dtype check would route Burgers onto the new path and break byte-identity.
- Assert the time grid is uniform. `h` is read from `t[1] - t[0]`; a non-uniform grid is currently integrated silently with the wrong step everywhere after the first.
- Correct the docstrings in `etdrk4_solver` and `Burgers._solve_spectral` so the diagonal-basis claim is accurate once Bug 2 is fixed, and states the uniform-grid requirement.
- Add tests that would have caught both bugs: a convergence-order test with time-dependent `N`, a complex-`L` weight check against the closed form where it is well-conditioned, a non-uniform-grid raise, and a tightened tolerance on the existing time-dependent-forcing test so an order-1 method cannot pass it.
- **Constraint, not a change:** seeded numerical Burgers generation must remain byte-identical to the parent commit. Both fixes are designed to leave the constant-forcing, real-`L` path untouched, and the change is verified against that.

## Capabilities

### New Capabilities
- `semilinear-time-integration`: what the ETDRK4 integrator guarantees — its order with time-dependent nonlinear terms, the class of linear operators it handles correctly, the time grids it accepts, and how it fails when it cannot honour those.

### Modified Capabilities
- `numerical-anchor-tests`: adds anchors for the two assumptions the current suite shares with the solver — convergence *rate* under time-dependent forcing as an invariant, and weight correctness for complex eigenvalues against the closed form.

## Impact

- `SpectralSVR/utils/__init__.py`: `etdrk4_solver` loop (stage times), `_coef` (branch), a uniform-grid assertion, docstring.
- `SpectralSVR/problems/burgers.py`: `_solve_spectral` docstring only. No behaviour change.
- `tests/utils/test_solvers.py`: four new or tightened tests.
- No API change. No dependency change. No change to generated data, verified by comparison against a worktree at the parent commit.
- The user-supplied PDF, §9 and Appendix B, contains a numpy harness that reproduces both bugs; it is the independent reference the new tests are checked against.
