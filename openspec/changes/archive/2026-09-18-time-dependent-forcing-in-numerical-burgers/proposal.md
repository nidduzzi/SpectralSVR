# Time-dependent forcing in the numerical Burgers path

## Why

`Burgers._solve_spectral` (`SpectralSVR/problems/burgers.py`) accepts only a constant forcing. `generate` advertises `f: ParamInput | BasisSubType`, but the numerical branch raises `NotImplementedError("numerical Burgers supports a constant forcing f")` for anything that is not a `Number`, builds `f_hat` once, and its `nonlinear` closure drops the time argument the solver passes it. The only time-dependent forcing in the library lives in the manufactured-solution (MMS) branch, which integrates nothing.

Two consequences, one scientific and one already paid for:

- **The "f -> u" benchmark dataset is MMS-only.** MMS picks a random band-limited `u` and *defines* `f` as its Burgers residual. The forcing distribution the operator is trained on is therefore "whatever a random `u` leaves behind", not a forcing prior anyone would choose or a reader would recognise. A forced-Burgers dataset solved from a sampled forcing does not exist in the library and cannot be built with the current numerical path.
- **The restriction hid a real bug.** Because `N` never depended on `t`, the ETDRK4 stage-time defect (all four stages evaluated at `t_n`) had zero observable effect and survived every test. The blind spot and the restriction are the same thing.

## What Changes

- `Burgers.generate`'s numerical branch accepts a forcing evaluable at arbitrary time: a callable `f(t) -> (n, ns)` spatial spectral coefficients, or a Basis field that is fully spectral in time (the `time_dependent_coeff=False` form the MMS branch already produces). Per-time-slice fields (`time_dependent_coeff=True`) are rejected as forcing, because the integrator evaluates at stage midpoints that fall between grid points.
- `nonlinear` evaluates the forcing spectrum at the stage time the solver passes: `-½ D1 F(u²) · dealias + f_hat(t)`.
- A constant `Number` forcing keeps today's path and output byte-for-byte.
- A new anchor: the MMS round-trip. Manufacture `u`, derive `f(t, x)`, integrate numerically from `u(0)` with that `f`, and compare to the manufactured `u(T)`. Also its convergence rate as `h` halves, which is **measured before any bound is written** — no rate has ever been observed on this path, so the expected ~4 is a prediction until the measurement exists.
- The forced-Burgers benchmark dataset this capability makes possible is **not** built here. It adds a reported row, and every other change in this stack has been able to claim that no reported number moves; it lands as its own change stacked directly above this one, so that claim survives here and the new row arrives where reporting is the subject.
- **Hard dependency:** `align-spectral-residual-with-solver` must land first. `spectral_residual` and the solver form the nonlinear term differently -- one takes the real part and de-aliases, the other does not -- so the manufactured pair is not a solution of the equation the solver integrates. Until that is fixed the round-trip error is flat at 2.59e-3 to 6.89e-3 regardless of step count, and no rate can be measured. (The earlier ETDRK4 stage-time dependency is satisfied: that change is archived and in `main`.)
- `etdrk4_solver` is not modified here.

## Capabilities

### New Capabilities
- `burgers-time-dependent-forcing`: what forcings the numerical Burgers generator accepts, how they are evaluated during integration, and what is rejected and why.

### Modified Capabilities
- `numerical-anchor-tests`: adds the MMS round-trip as an anchor for the numerical path — the first test in the suite where the integrator is checked against an exact solution *with* time-dependent forcing, rather than against Cole–Hopf (unforced) or the spectral residual (no integration).

## Impact

- `SpectralSVR/problems/burgers.py`: `_solve_spectral` forcing handling and the `nonlinear` closure; `generate`'s type and docstring for `f`.
- `tests/problems/test_problems.py`: round-trip anchor, convergence rate, rejection of per-slice forcing, constant-forcing byte-identity.
- No change to `SpectralSVR/utils`. No API removal. Existing callers passing a `Number` are unaffected; callers passing a Basis forcing today get `NotImplementedError` and will instead get a result.
- Blocked on `align-spectral-residual-with-solver`: the round-trip anchor cannot pass until the manufactured pair is a true continuous solution. A flat, step-count-independent error is the signal that dependency is missing; a first-order reading would mean something else.
