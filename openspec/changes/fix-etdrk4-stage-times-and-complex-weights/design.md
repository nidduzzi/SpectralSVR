## Context

See proposal.md - Why for the two defects and their reproduction. The design-relevant state of `etdrk4_solver` in `SpectralSVR/utils/__init__.py`:

```
h  = t[1] - t[0]                          only ndim==1 and len>1 are asserted
r  = exp(i pi (j - 1/2) / m),  j = 1..m   upper half of the unit circle
lr = h * L + r
_coef(expr) = (h * mean_over_r(expr)).real
Q, f1, f2, f3 = _coef(...)                real tensors, cast to y0's real dtype

loop:  ti = t[i-1]
       nv = N(ti, v);  a = E2 v + Q nv;  na = N(ti, a)
       b  = E2 v + Q na;                 nb = N(ti, b)
       c  = E2 a + Q (2 nb - nv);        nc = N(ti, c)
       v  = E v + nv f1 + 2 (na + nb) f2 + nc f3
```

Facts measured in this session that constrain the design:

| Fact | Consequence |
| --- | --- |
| `FourierBasis.derivative_eigenvalues(ord=2)` returns `complex64` with imaginary part exactly zero | The Bug 2 branch must test imaginary *content*, not dtype, or Burgers changes path |
| Stage-time fix changes nothing when `N` ignores `t` | All shipped Burgers data (constant forcing only) stays byte-identical |
| Half-circle + `.real` and full-circle + complex agree to 1e-18 on real `L` | Either path is numerically right for Burgers; only the branch keeps it *bit*-identical |
| Existing forcing test: 300 steps, asserts rel error `< 1e-2` | Passes at order 1; tolerance must drop below the buggy error and stay above the fixed error |
| `burgers.py:70` asserts only `t[1]-t[0] == dt` | The caller does not guarantee uniformity either; the solver must |

## Goals / Non-Goals

**Goals:**

- Fourth order restored for time-dependent `N`, with a test that measures the *rate*.
- Correct weights for complex `L`, with the real-`L` path producing bit-identical weights to today.
- Loud failure on a non-uniform grid.
- Zero change to any seeded Burgers field, proven against the parent commit rather than argued.

**Non-Goals:**

- Adaptive stepping or non-uniform grids. The weights are precomputed once from `h`; supporting variable `h` is a different integrator.
- Switching the real-`L` path to the full circle for uniformity. Correct, but it moves published numbers at rounding level for no accuracy gain.
- A KdV or Schrödinger problem generator. The complex-`L` path is tested at the solver level only; a problem that uses it is a separate change.
- Changing `contour_points` defaults or the precision policy (complex128 weights cast to `y0`'s dtype). The PDF's precision note is real but affects only convergence studies run in complex64, which the new tests do not do.

## Decisions

### Stage times come from the grid, not from accumulated `h`

The four evaluations use `t[i-1]`, `(t[i-1] + t[i]) / 2`, `(t[i-1] + t[i]) / 2`, `t[i]`. Reading the endpoints off the grid and averaging them for the midpoint keeps every stage time exact on the caller's grid and needs no `h` arithmetic in the loop.

*Alternative rejected:* `ti + h/2`, `ti + h` with `h` a Python float. Equivalent in exact arithmetic; drifts by accumulated rounding over long runs and mixes a float into tensor time arguments.

### Bug 2 is a branch on imaginary content

```
if linear has any nonzero imaginary entry:
    r = exp(2 i pi (j - 1/2) / m'), full circle;  weights = h * mean(expr)      (complex)
else:
    r = exp(  i pi (j - 1/2) / m ), half circle;  weights = h * mean(expr).real (today)
```

The real branch is today's code, untouched, so the Burgers path produces the same bits. The complex branch is the general Cauchy-integral average with nothing discarded.

Detection is `torch.is_complex(linear) and bool(linear.imag.any())`. A dtype test would send the Fourier `-νk²` eigenvalues — stored complex with zero imaginary part — down the new path and break byte-identity. An exact `any()` rather than a tolerance means a linear operator with rounding-noise imaginary parts takes the complex path; that path is still correct, just not bit-identical to the real one, and the byte-identity task will show whether any existing caller trips it (none is expected to: the Fourier eigenvalues are exactly zero-imaginary).

*Alternative rejected:* always use the full circle. One code path, correct everywhere, but real-`L` weights pick up imaginary parts of order 1e-20 and every Burgers trajectory shifts at the last bit. For a repository whose numbers are going into a paper, a fix that moves nothing is worth one `if`.

*Alternative rejected:* branch on `linear.dtype`. Fails for the reason in the facts table.

### Full circle uses twice the points

The half circle places `contour_points` (default 32) nodes over an arc of length π. The full circle uses `2 * contour_points` over 2π so node density, and therefore trapezoid-rule accuracy on the analytic integrand, matches the half-circle path. Kassam–Trefethen report 32 or 64 as ample; 64 on the full circle is the upper end of their range.

### Uniform grid is asserted, with tolerance

`torch.allclose(t.diff(), t.diff()[0])` at default tolerances, raised as an `AssertionError` alongside the existing `t.ndim == 1 and len(t) > 1` assertion. `linspace` grids pass; a caller who concatenates two grids or hand-writes uneven points fails before the first step.

*Alternative rejected:* accept non-uniform grids by recomputing weights per step. That is a different integrator with different cost; out of scope.

### The tightened tolerance is set from both measurements

The existing MMS forcing test runs 300 steps. The task measures the final relative error with all stages at `t_n` (the shipped behaviour) and with the corrected stage times, then fixes the tolerance one order of magnitude above the corrected error. Both numbers are recorded in the test so the choice is auditable: the test must fail on the old code and pass on the new one, and the recorded values make that checkable without re-running the bug.

### Complex-weight anchor lives where the formula is well-conditioned

The closed-form weight expressions suffer catastrophic cancellation near `z = 0` — that is the whole reason for the contour trick — so they cannot be the oracle there. At `|z| = |hL| >= 1` they are accurate to near machine precision and can be. The anchor evaluates them directly at `L ∈ {-2i, -20i, -200i, 5i}` with `h = 0.1` and compares to the solver's weights at a tolerance of `1e-10`, well below the `~1e-2` the half-circle shortcut produces and well above the `~1e-15` the full circle produces. The weights are reached through the same code path the loop uses; if the solver's weight computation is not separately callable, the test calls the solver on a one-step problem whose result exposes the weights.

## Risks / Trade-offs

- **Two contour paths mean two things to keep correct.** → The real path is unchanged code, and the "real eigenvalues are unchanged" scenario pins it bit-for-bit. Only the complex path is new, and it is anchored to the closed form.

- **A caller with rounding-noise imaginary parts silently takes the slower, non-identical path.** → Results remain correct. The byte-identity check in this change covers the only in-repo caller. Documented in the docstring so a future caller knows the branch exists.

- **The convergence-order test depends on a stiff scalar problem staying in the resolved regime.** With `λ = -200` and `h` from `1/40` down to `1/1280`, `|hλ|` runs from 5 to 0.16 — the coarse end is stiff, which is what the exponential integrator exists for, and the measured orders reported in the proposal were taken over exactly this range. → The test asserts the order over the last three halvings only, where it is `3.96, 3.99, 4.00`, and records the coarser values for reference without asserting on them.

- **Tightening the forcing test could flake if the fixed error sits close to the new bound.** → One order of magnitude of headroom above the measured fixed error, and the measured numbers are written into the test.

- **`pytest` runs with `filterwarnings = ["error"]`.** No torchdiffeq call is involved here, so the warning risk from the previous change does not apply.

## Migration Plan

1. Stage times: edit the four `nonlinear(...)` calls. Run the existing suite — everything must still pass, since nothing in it depends on `t`.
2. Uniform-grid assertion and the tightened forcing tolerance, with both error measurements recorded.
3. Contour branch: extract the weight computation so the real branch is byte-for-byte the current code, add the complex branch beside it.
4. Docstrings in `utils` and `burgers.py`.
5. New tests: convergence order, complex-weight closed form, non-uniform grid, real-path bit-identity.
6. Byte-identity: generate a seeded numerical Burgers field on this branch and at the parent commit via a worktree; coefficients must be `torch.equal`.

Rollback is a revert. No persisted state, no data format, no API signature changes.

## Open Questions

- Whether `contour_points` should be exposed on `Burgers.generate` now that a complex-`L` path exists. Deferrable: no in-repo problem has a complex linear operator yet, and the default is adequate for both branches.
