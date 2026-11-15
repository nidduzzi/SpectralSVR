# Unify the ODE solver protocol

## Why

`Burgers.generate` advertises a `solver` parameter typed `SolverSignatureType` and defaulting to `implicit_adams_solver`, but the numerical branch never reads it: `_solve_spectral` calls `etdrk4_solver` directly (`SpectralSVR/problems/burgers.py:183`). The knob is silently inert, and it cannot be honoured as declared -- ETDRK4 is an exponential integrator that needs the diagonal linear operator `L` as a tensor to build `exp(hL)` and the Kassam-Trefethen contour weights, which cannot be recovered from a monolithic `rhs(t, y)` closure. Passing the one solver that actually runs, `solver=etdrk4_solver`, is currently a type error.

Two incompatible protocols coexist with no bridge between them, so the thesis/paper solvers (`euler_solver`, `implicit_adams_solver`, `lsoda_solver`) cannot be pointed at a spectral PDE problem at all, and ETDRK4's Burgers trajectories have no independent integrator to be cross-checked against.

## What Changes

- Adopt the semilinear `(linear, nonlinear, y0, t)` protocol (`ExpIntSolverSignatureType`) as the solver interface that problem generators accept, since it is the wider of the two: `(L, N)` collapses to `rhs` but not the reverse.
- Add an adapter in `SpectralSVR.utils` turning any `SolverSignatureType` solver into a semilinear one by forming `rhs(t, v) = linear * v + nonlinear(t, v)`. Every existing monolithic solver becomes usable on spectral PDE problems through it.
- **BREAKING** `Burgers.generate`'s `solver` parameter is retyped to the semilinear protocol and its default becomes `etdrk4_solver` (the integrator already in use). Callers that passed `implicit_adams_solver` were being ignored; they must now wrap it in the adapter to get the behaviour the parameter always claimed.
- Thread `solver` from `Burgers.generate` into `_solve_spectral`, replacing the hardcoded call.
- Keep the existing monolithic protocol, its type aliases, and all three solvers as public API; they are used by the linked thesis and paper.
- Cross-solver agreement becomes available as an independent oracle for the numerical Burgers path in a mildly stiff regime.

## Capabilities

### New Capabilities
- `pde-solver-protocol`: which solver interfaces the library exposes, how a problem generator accepts a solver, and how the two protocols convert into one another.

### Modified Capabilities
- `numerical-anchor-tests`: adds cross-solver agreement as an independent oracle for the ETDRK4 Burgers trajectories, which are currently anchored only by MMS and Cole-Hopf.

## Impact

- `SpectralSVR/utils/__init__.py`: new adapter; solver type aliases unchanged and still exported.
- `SpectralSVR/problems/burgers.py`: `generate` signature (breaking), `_solve_spectral` signature, removal of the hardcoded solver call.
- `SpectralSVR/__init__.py`: export the adapter alongside `SolverSignatureType`.
- `tests/utils/test_solvers.py`, `tests/problems/test_problems.py`: adapter round-trip tests and the cross-solver anchor.
- Non-stiff steppers on a stiff spectral operator are inaccurate; the adapter's docstring and the spec must state the regime in which a cross-solver comparison is meaningful, so the knob does not become a silent accuracy footgun.
