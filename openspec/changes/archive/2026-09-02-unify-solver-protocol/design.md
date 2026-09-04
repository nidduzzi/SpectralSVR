## Context

See proposal.md - Why for the motivation. The design-relevant state:

```
Burgers.generate(solver=...)          declared burgers.py:47, never read
        |
        +-- MMS branch (u0="random", f="random")   analytic, no integration
        |
        +-- _solve_spectral (numerical)            calls etdrk4_solver directly, burgers.py:183
```

`_solve_spectral` is the library's only integration site. It builds both halves of the semilinear system locally: `linear = nu * d2` from the basis eigenvalues, and a `nonlinear` closure carrying the de-aliasing mask and the constant forcing spectrum. The state it integrates is complex (`v0` is a spectral coefficient tensor).

Two interfaces exist in `utils`:

```
SolverSignatureType        (rhs(t, y),  y0, t) -> (steps, *y0.shape)
ExpIntSolverSignatureType  (L, N(t, v), y0, t) -> (steps, *y0.shape)
```

Measured facts that shape the decisions below:

| Solver | Complex spectral state | Notes |
| --- | --- | --- |
| `euler_solver` | works | `solution.to(y0)` already casts |
| `implicit_adams_solver` | works | 4.3e-9 vs the exact linear decay at `modes=8, nt=401, nu=0.01`, no warning raised |
| `lsoda_solver` | **fails** | `ValueError: y0 is complex, but the chosen solver does not support integration in a complex domain` |
| `etdrk4_solver` | works | the current hardcoded integrator |

## Goals / Non-Goals

**Goals:**

- One interface that a problem generator accepts, with every shipped solver reachable through it.
- Conversion that changes nothing about the wrapped solver's numerics - same time grid, same tolerances, same result as calling it with the combined right-hand side.
- A second integrator available for the numerical Burgers path so ETDRK4's time axis gets an independent check.

**Non-Goals:**

- A real-embedding conversion so `lsoda_solver` can run on complex spectral state. `to_real_coeff` / `to_complex_coeff` already exist and make this a natural follow-up, but it is a separate capability with its own accuracy questions.
- Changing ETDRK4 itself, its contour-integral weights, or the de-aliasing scheme.
- Making the MMS branch solver-driven; it is analytic by construction.

## Decisions

### Problem generators accept the semilinear interface

`(L, N)` collapses to `rhs = L*v + N(t, v)`; `rhs` cannot be split back into `L` and `N`. ETDRK4 needs `L` as a tensor to form `exp(hL)` and the Kassam-Trefethen weights, so the semilinear interface is strictly the wider one and is what the generator must take.

*Alternative rejected:* keep `SolverSignatureType` on the generator and have the caller bind `L` via `partial`. `L` is derived inside `_solve_spectral` from the basis eigenvalues and `nu`; the caller does not have it. Binding it would force the caller to recompute basis internals.

*Alternative rejected:* accept either interface and dispatch on arity. Arity dispatch on callables is fragile under `partial` and gives a confusing failure when a solver is passed with the wrong meaning for its first argument.

### The conversion is `as_semilinear_solver`, and it lives in `utils`

Signature `as_semilinear_solver(solver: SolverSignatureType) -> SemilinearSolverSignatureType`, forming `rhs(t, v) = linear * v + nonlinear(t, v)` and forwarding `y0` and `t` untouched. It sits beside the solvers it wraps and beside the type aliases it converts between.

The name says what it produces, so the conversion needs no comment at its call sites: `solver=as_semilinear_solver(implicit_adams_solver)` reads as its own explanation.

*Alternative rejected:* a `Solver` protocol class with two implementations. The solvers are plain functions and `partial` objects today; a class hierarchy would force the thesis's existing call sites to change for no behavioural gain.

### `ExpIntSolverSignatureType` is renamed to `SemilinearSolverSignatureType`

`ExpInt` is an abbreviation that does not survive a reader who has not read the ETDRK4 references; "semilinear" is the standard term for the `dv/dt = Lv + N(t, v)` form the interface encodes, and it matches the conversion's name. The old name stays as an assignment alias so any thesis import keeps resolving.

### Passing a solver to the MMS branch raises

The MMS branch manufactures a solution analytically and integrates nothing. Accepting a `solver` there and ignoring it would reproduce exactly the bug this change fixes, one branch over. `Burgers.generate` raises when a solver is supplied on a code path that will not integrate.

*Alternative rejected:* silently ignore it. That is the current defect.

*Alternative rejected:* split into two public generators. The `u0`/`f` dispatch is existing API surface the thesis depends on; splitting it is a larger break than this change warrants.

### `_solve_spectral` takes the solver as a parameter

The hardcoded call at `burgers.py:183` becomes the passed-in solver, with `etdrk4_solver` as `generate`'s default so seeded output is unchanged. `_solve_spectral` stays private; its signature already carries eleven positional arguments, and the solver joins them rather than being reached through instance state.

### The cross-solver anchor picks its regime from a stated criterion

Agreement between an exponential integrator and a non-stiff stepper only holds while the stiff eigenvalue is resolved by the step size, i.e. `h * nu * k_max^2` of order one or below. The test fixes `nu`, mode count, and step count so that criterion holds and writes them literally rather than inheriting generator defaults. `implicit_adams_solver` at `modes=8, nt=401, nu=0.01` already satisfies it - 4.3e-9 against the exact linear decay, and no convergence warning.

*Alternative rejected:* `lsoda_solver` as the independent integrator. It rejects complex state outright.

*Alternative rejected:* `euler_solver`. First-order accuracy would force a tolerance so loose the comparison stops being an anchor.

## Risks / Trade-offs

- **A converted non-stiff solver is silently inaccurate on a stiff operator.** The knob accepts any solver, and nothing stops a caller from using one outside its stability regime. → The conversion's docstring states the `h * nu * k_max^2` criterion, and the spec requires the regime to be documented rather than papered over by substituting a step size.

- **`pytest` runs with `filterwarnings = ["error", ...]`, and torchdiffeq emits `UserWarning: Functional iteration did not converge` when `implicit_adams` is pushed past its regime.** A cross-solver test drifting into stiffness fails as a warning-turned-error rather than as an accuracy assertion, which reads as an unrelated failure. → Keep the anchor inside the measured regime, where no warning is emitted; if the warning ever appears, treat it as the regime criterion being violated rather than allowlisting it.

- **The `solver` parameter's default value changes from `implicit_adams_solver` to `etdrk4_solver`.** Anyone reading the old signature would have believed Adams was in use. → No output changes, because Adams was never actually called; the release note has to say the old default was inert.

- **Raising on a solver passed to the MMS branch turns a previously silent call into an error.** → It could only ever have been a caller expecting an effect that never happened.

- **Renaming the type alias touches a name the thesis may import.** → The old name is kept as an alias, so imports keep working.

## Migration Plan

1. Add `SemilinearSolverSignatureType` plus the `ExpIntSolverSignatureType` alias, and `as_semilinear_solver`, in `utils`.
2. Thread `solver` through `_solve_spectral`; default `Burgers.generate`'s `solver` to `etdrk4_solver` and retype it.
3. Add the MMS-branch guard.
4. Export `as_semilinear_solver` from the package root next to `SolverSignatureType`.
5. Add the conversion round-trip tests and the cross-solver Burgers anchor.
6. Regenerate any results that depend on Burgers generation only if a seeded field actually changes - step 2 is intended to leave them identical, and that is the check.

Rollback is a revert: nothing persists state and no data format changes.

## Open Questions

- Whether a real-embedding conversion (via the existing `to_real_coeff` / `to_complex_coeff`) is worth adding so `lsoda_solver` becomes usable on spectral state. Deferrable: it adds a capability without changing this change's specs, approach, or tasks.
