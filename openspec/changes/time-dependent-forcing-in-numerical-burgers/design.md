## Context

See proposal.md - Why. The design-relevant state of `Burgers._solve_spectral`:

```
if isinstance(f, Number):  f_hat = transform(full((n, ns), f))       once
else:                       raise NotImplementedError

nonlinear(ti, v) = -0.5 * d1 * transform(inv_transform(v).real ** 2) * dealias + f_hat
                                                                          ^ constant, ti unused
```

What the rest of the library already provides:

- `mms_solution` returns `u` and `f` as 2D Fourier fields over `(time, space)`, fully spectral in both axes (`time_dependent=False`). Such a field is evaluable at any `(t, x)` through the Basis interface, and is *periodic in time* on the integration interval — `u(0) = u(T)` by construction.
- `u` from MMS is band-limited to half the mode budget per axis so `u²` is alias-free at that budget. The numerical path applies the Orszag 2/3 mask at its own resolution `ns`.
- The solver calls `nonlinear` at `t_n`, `(t_n + t_{n+1})/2`, `(t_n + t_{n+1})/2`, `t_{n+1}` (the stage-time fix is archived and in `main`). The midpoints are not on the output grid.

## Goals / Non-Goals

**Goals:**

- One internal representation of forcing — `forcing_at(t) -> (n, ns)` spectral coefficients — that the constant, callable, and spectral-field inputs all normalise to, with the constant case producing today's bits.
- The MMS round-trip as an integration-level anchor with a fourth-order rate assertion.

**Non-Goals:**

- Interpolating per-time-slice forcing fields. The choice of interpolant would silently set the accuracy of the whole integration; the caller should convert to a spectral or callable form and own that decision.
- Non-constant forcing for the *implicit_adams / euler* solvers through `as_semilinear_solver`. They receive the same `nonlinear(t, v)`, so it works, but its accuracy regime is the previous change's concern.
- The forced-Burgers benchmark dataset itself. The capability here makes it buildable; constructing it adds a reported row and belongs to its own change directly above this one. The design work for it — sampling from rest, the GRF length scales, the 2D `targets` question — moves with it.
- Any change to `etdrk4_solver`.

## Decisions

### Forcing normalises to a `forcing_at(t)` callable inside `_solve_spectral`

Three input forms, one closure:

| Input | `forcing_at(t)` |
| --- | --- |
| `Number` | returns the precomputed constant `f_hat` — the existing tensor, computed exactly as today |
| `Callable[[Tensor], Tensor]` | calls it, asserts the result is `(n, ns)` |
| Basis with `time_dependent=False` | evaluates the field on `(t, spatial grid of ns points)`, transforms spatially |

`nonlinear(ti, v)` becomes `-0.5 * d1 * transform(u * u) * dealias + forcing_at(ti)`. For the `Number` case `forcing_at` returns the same object every call, so the arithmetic — and the bits — are unchanged.

*Alternative rejected:* precompute `f_hat` on the output grid and index by step. The solver evaluates at midpoints that are not on the grid; indexing would silently use the wrong time for two of the four stages, reintroducing the bug class this change exists to close.

### Spectral forcing fields are evaluated through values, not hand-rolled mode sums

The spatial coefficients of a `(time, space)` Fourier field at time `t` are a partial inverse transform along the time axis. Rather than write that sum, evaluate the field on `(t, x_grid)` with the Basis `__call__`/`get_values` path and apply `basis.transform` spatially. This reuses the tested evaluation code, is exact for a band-limited field, and automatically yields `ns` spatial modes when the grid has `ns` points — so a forcing field at a different spatial resolution is resampled without a separate resize step.

Cost: one field evaluation and one FFT per stage. Four per step. Acceptable for data generation; it is not the solver's hot loop.

*Alternative rejected:* `resize_modes` the field to `(mt, ns)` and sum time modes by hand. Faster, but duplicates the basis's normalisation conventions in a second place.

### Time domain must match exactly; per-slice fields are rejected

A spectral forcing field's `domain[0]` must equal `(time_domain.start, time_domain.stop)`. A mismatch means the periodic field is being read outside the interval it was built on; the generator raises rather than wrap silently. A field with `time_dependent=True` raises with a message naming the two accepted forms.

### The round-trip anchor runs at higher spatial resolution than the manufactured field

`mms_solution` band-limits `u` to `|k| ≤ m/4`, so `u²` reaches `|k| ≤ m/2`. The solver's 2/3 mask at `ns` keeps `|k| ≤ ns/3`. For the numerical `N` to equal the `N` that produced `f`, `u²` must sit inside the kept band: `m/2 ≤ ns/3`, i.e. `ns ≥ 1.5 m`. The anchor solves at `ns = 2m` — the manufactured `u(0)` and `f` are resampled to `2m` spatial modes on the way in — so the only error left is time-stepping. This is what makes a fourth-order rate observable rather than an aliasing floor.

*Alternative rejected:* manufacture `u` at a third of the budget. Also works, but changes `mms_solution`'s contract, which the MMS tests and the existing "f -> u" dataset rely on.

*Alternative rejected:* disable de-aliasing for the anchor. Tests a code path the benchmark never uses.

### Time periodicity of MMS is a feature of the anchor, not a flaw

The manufactured `u` is periodic in time, so `u(T) = u(0)`. The round-trip therefore checks that integrating for one full period with the manufactured forcing returns the initial condition. The forcing is genuinely time-varying throughout, so the stage-time behaviour is fully exercised; the periodicity only makes the target easy to state.

### Rate assertion over the resolved range, with a recorded tolerance

Step counts `nt ∈ {50, 100, 200, 400}` on `[0, 1]`. The order is **measured first and the bound set beneath it**, rather than predicted from the scheme's nominal order and checked afterwards.

The reasoning that produced the original `> 3.5` still stands as an expectation — the nonlinear problem is rougher than the scalar test in the ETDRK4 change, and the reference is itself band-limited, so 3.9 would be optimistic. But it remains an expectation: no rate has ever been observed on this path, because until `align-spectral-residual-with-solver` landed the error was flat at 2.59e-3 to 6.89e-3 at every step count. A measured order materially under 4 is a finding to chase — de-aliasing resolution, stage times — not a number to assert beneath. The absolute tolerance at `nt = 400` is likewise fixed one order of magnitude above the measured value, and every measured order is recorded in both `tasks.md` and the test.

## Risks / Trade-offs

- **This change is blocked on `align-spectral-residual-with-solver`.** `spectral_residual` builds the nonlinear term from complex values with no de-aliasing while the solver takes the real part and de-aliases, so the manufactured pair does not solve the equation being integrated and the round-trip error is flat at 2.59e-3 to 6.89e-3 at every step count. → The anchor is written to expect ~4; a flat, step-count-independent error is the signal that the dependency is missing. The Nyquist representation is *not* implicated -- a lone Nyquist coefficient evaluates to the correct cosine (1.3e-7) with derivative ratio 1.0000.

- **Evaluating a Basis field four times per step is slower than the constant path.** → Data generation only. Recorded as a known cost; if it matters later, the mode-sum alternative is the optimisation.

- **`generate`'s `f` parameter widens to accept a callable.** Its type annotation changes; `ty` will check every call site. → No in-repo caller passes a callable today; the annotation change is additive.

## Migration Plan

1. Rebase this branch onto `align-spectral-residual-with-solver`. Nothing below is safe to assert before that. Note that rebasing changes this branch's parent, so the seeded byte-identity checks in tasks 1.1 and 7.3 must be re-measured against the new parent rather than assumed to still hold -- that change does not touch `generate_coeff`, so the expectation is unchanged output, but the check must actually be re-run.
2. `_solve_spectral`: introduce `forcing_at`, route the three input forms, keep the `Number` path's tensor unchanged. Run the suite; constant-forcing tests must still pass and the seeded byte-identity check must hold.
3. Rejection paths and their tests.
4. MMS round-trip anchor with rate assertion; record measured values.
5. Forced dataset in `scripts/run_benchmark.py`; `--quick` run to confirm it builds and scores.
6. Docstring for `generate`'s `f` parameter naming the three accepted forms and the rejected one.

Rollback is a revert on this branch alone; the bugfix beneath is untouched.

## Open Questions

- Whether `mms_solution` should expose its band-limit fraction so the round-trip anchor can request `|k| ≤ m/6` directly instead of solving at `2m`. Deferrable: the `2m` route needs no API change.
