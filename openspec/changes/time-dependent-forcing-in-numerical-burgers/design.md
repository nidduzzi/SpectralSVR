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
- `u` from MMS is band-limited by a stated cutoff `K = m // 6` per axis (`widest_manufactured_band`, from `manufactured-band-derivation`), so `u²` reaches `|k| ≤ 2K ≤ m/3` and is alias-free at that budget. The numerical path applies the Orszag 2/3 mask at its own resolution `ns`.
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

`manufactured-band-derivation`, now archived beneath this change, states the manufactured band as a cutoff rather than a count: `widest_manufactured_band(m)` returns `WavenumberCutoff(m // 6)`. So `u` reaches `|k| ≤ K` with `K = m // 6`, and `u²` reaches `|k| ≤ 2K`. The solver's 2/3 mask at `ns` keeps `|k| ≤ ns/3`. For the numerical `N` to equal the `N` that produced `f`, `u²` must sit inside the kept band: `2K ≤ ns/3`, i.e. **`ns ≥ 6K`**. Since `6 (m // 6) ≤ m`, the choice `ns = m` always satisfies it — which is where this section's earlier `ns ≥ m` reading came from, and it remains true, but it is a consequence of the derivation rather than the derivation itself.

The anchor nevertheless solves at `ns = 2m`, resampling the manufactured `u(0)` and `f` on the way in. That was originally reasoned as insurance: the condition looked tight at `ns = m`, and the measured order must not depend on where the band sits. The insurance has since been cashed in and found unnecessary. Measured across `m ∈ {12, 18, 24, 36}` and `ns ∈ {m, 2m, 3m}` over the sweep `nt ∈ {25, 50, 100}` — shorter than the table below, run only to compare resolutions against one another — the orders agree to within 0.1 and the errors to about 5%. At `m = 24, ns = m` that sweep reads errors `8.998e-04`, `5.096e-05`, `3.456e-06`, so orders 4.14 and 3.88: already fourth order with no margin at all. At `m = 24, ns = 2m` the same sweep reads 4.16 and 4.02. The margin stays because it costs nothing, not because the rate needs it. What makes the anchor sound is the derivation above, not the factor of two.

*Alternative superseded:* manufacture `u` at a third of the budget. Recorded here as rejected when the band was `m/4`; `align-spectral-residual-with-solver` made it the actual band, and `manufactured-band-derivation` then replaced the count with the cutoff the derivation above uses. It is no longer an alternative to anything.

*Alternative rejected:* disable de-aliasing for the anchor. Tests a code path the benchmark never uses.

### Time periodicity of MMS is a feature of the anchor, not a flaw

The manufactured `u` is periodic in time, so `u(T) = u(0)`. The round-trip therefore checks that integrating for one full period with the manufactured forcing returns the initial condition. The forcing is genuinely time-varying throughout, so the stage-time behaviour is fully exercised; the periodicity only makes the target easy to state.

### Rate assertion over the resolved range, with a recorded tolerance

The order is **measured first and the bound set beneath it**, rather than predicted from the scheme's nominal order and checked afterwards. Measured at `m = 24`, `ns = 2m`, `nu = 0.01`, `n = 2`, seed 1234, on `[0, 1]`:

| `nt` | `h = 1/(nt-1)` | relative error | order |
| --- | --- | --- | --- |
| 13 | 1/12 | 1.401947e-02 | — |
| 25 | 1/24 | 9.606420e-04 | 3.867 |
| 49 | 1/48 | 5.935034e-05 | 4.017 |
| 97 | 1/96 | 3.342465e-06 | 4.150 |
| 193 | 1/192 | 1.594647e-06 | 1.068 |
| 385 | 1/384 | 1.416505e-06 | 0.171 |
| 769 | 1/768 | 2.284473e-06 | -0.690 |

The step counts halve `h = 1/(nt - 1)` exactly. This matters because the order is only `log2` of the error ratio when the step ratio is two: an earlier sweep here used `nt ∈ {13, 25, 50, 100}`, whose step ratios are 2.000, 2.042 and 2.020, and reading it as `log2` inflated two orders to 4.160 and 4.022 against ratio-corrected values of 4.040 and 3.964. The conclusion was unchanged — fourth order either way — but the figures were wrong as stated, and the scenario asks for halving step sizes rather than approximately halving ones.

Fourth order, so the expectation behind the original `> 3.5` was sound — the nonlinear problem is rougher than the scalar test in the ETDRK4 change and the reference is itself band-limited, and 4.0 to 4.2 is what that reasoning would have allowed. The step counts that expectation arrived with were not sound: this section first named `nt ∈ {50, 100, 200, 400}`, whose last two rows read the floor — leaving two of its three orders as floor artefacts rather than rates.

**The resolved range is bounded above as well as below.** Coefficients are stored at `SpectralConfig.coeff_dtype`, complex64 by default, and the round-trip reaches a round-off floor near `1.5e-6` relative by `nt = 200`, rising again to `2.7e-6` by `nt = 800`. The rise is what identifies it: a fixed spatial-truncation floor sits flat, accumulating round-off does not. `torch.set_default_dtype(torch.float64)` does not move it, so the floor is the stored coefficient precision and not the default dtype. Raising the anchor to double would mean widening `generate`, which takes the basis as a *class* and constructs from it, leaving no seam to pass a precision through; that is out of scope here, and `numerical-anchor-tests` already carries the scenario distinguishing saturation from order loss.

The asserted range is therefore `nt ∈ {13, 25, 49, 97}`, and the bound sits beneath the **minimum** order measured across it (3.867) rather than beneath the last pair — keyed to the last pair, any sweep reaching the floor asserts against saturation. A measured order materially under 4 is a finding to chase — de-aliasing resolution, stage times — not a number to assert beneath. The absolute tolerance at `nt = 97` is likewise fixed one order of magnitude above the measured `3.342465e-06`, and every measured order is recorded in both `tasks.md` and the test.

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

None. The band-limit question this section carried — whether `mms_solution` should expose its band so the anchor can request a cutoff rather than oversolve — is answered by `manufactured-band-derivation`, which states the band as a cutoff satisfying `2K ≤ m/3`. The `2m` margin here is kept deliberately, for the reason the resolution decision records, and is not a workaround for a missing accessor.
