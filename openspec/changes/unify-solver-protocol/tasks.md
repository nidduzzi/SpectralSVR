## 1. Solver interface in `utils`

- [x] 1.1 Rename `ExpIntSolverSignatureType` to `SemilinearSolverSignatureType` in `SpectralSVR/utils/__init__.py`, keeping `ExpIntSolverSignatureType` as an assignment alias; verify `uv run --extra lint ty check SpectralSVR` passes and `from SpectralSVR.utils import ExpIntSolverSignatureType` still resolves
- [x] 1.2 Add `as_semilinear_solver(solver: SolverSignatureType) -> SemilinearSolverSignatureType` forming `rhs(t, v) = linear * v + nonlinear(t, v)` and forwarding `y0` and `t` unchanged; verify a converted `euler_solver` returns a trajectory of shape `(steps, *y0.shape)` whose first entry is `y0`
- [x] 1.3 Document the stiffness regime `h * nu * k_max^2` in the conversion's docstring, per the `pde-solver-protocol` requirement "Accuracy regime of converted solvers is stated"; verify the docstring names the limitation and the regime

## 2. Burgers generator

- [x] 2.1 Add a `solver: SemilinearSolverSignatureType` parameter to `Burgers._solve_spectral` and call it in place of the hardcoded `etdrk4_solver` at `burgers.py:183`; verify `tests/problems/test_problems.py` still passes
- [x] 2.2 Retype `Burgers.generate`'s `solver` parameter to `SemilinearSolverSignatureType` with default `etdrk4_solver` and pass it into `_solve_spectral`; verify `ty check` reports no error for `solver=etdrk4_solver` at a call site
- [x] 2.3 Raise when a solver is supplied on the MMS branch, which integrates nothing; verify a test asserts the raise and that the numerical branch is unaffected
- [x] 2.4 Confirm the default path is byte-identical to before: generate a seeded numerical Burgers field on this branch and on `main`, and verify the coefficients match exactly

## 3. Exports

- [x] 3.1 Export `as_semilinear_solver` and `SemilinearSolverSignatureType` from `SpectralSVR/__init__.py` alongside `SolverSignatureType`; verify `from SpectralSVR import as_semilinear_solver` works
- [x] 3.2 Drop the now-valid `solver=etdrk4_solver` type error in `scripts/run_benchmark.py` by keeping the argument and verifying `ty check` is clean and `python scripts/run_benchmark.py --quick --operators spectralsvr` still writes `results/bench_results.csv`

## 4. Tests

- [x] 4.1 In `tests/utils/test_solvers.py`, assert the conversion is result-preserving: a converted solver's trajectory equals elementwise what the same solver returns from the combined right-hand side
- [x] 4.2 Assert a converted solver integrates complex spectral state without raising and returns the same complex dtype
- [x] 4.3 Assert the converted solver receives the caller's time grid unchanged (no substituted step size or tolerance)
- [x] 4.4 Add the cross-solver Burgers anchor in `tests/problems/test_problems.py`: generate once with `etdrk4_solver` and once with `as_semilinear_solver(implicit_adams_solver)` at a fixed seed and literal `nu=0.01, modes=8, nt=401`, and verify the trajectories agree within a tolerance fixed from the integrators' orders rather than tuned to pass
- [x] 4.5 Verify the anchor emits no warning under the repo's `filterwarnings = ["error"]`: if `Functional iteration did not converge` appears, treat it as the regime criterion being violated and tighten the regime rather than allowlisting the warning

## 5. Verification

- [x] 5.1 Run `just check` (ruff, mypy, ty, full pytest) and verify it passes clean
- [x] 5.2 Verify no source file gained an explanatory prose comment; the conversion and the raise must read from their names and structure, with external context confined to docstrings
