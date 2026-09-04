## Context

See proposal.md - Why for the two defects and their isolation. The design-relevant code, both in `SpectralSVR/problems/burgers.py`:

```
_solve_spectral, line 209          spectral_residual, lines 292-295
  u = inv_transform(v).real          u_val   = inv_transform(u.coeff)
  ...                                u_x_val = inv_transform(u_x.coeff)
  -0.5 * d1 * transform(u*u)         uu_x.coeff = transform(u_val * u_x_val)
         * dealias                    (no .real, no dealias)
```

Two differences, and both matter only when the field reaches the top of the grid — which is why every test passed. Facts established during diagnosis that constrain the design:

| fact | consequence |
| --- | --- |
| A lone Nyquist coefficient evaluates to the correct cosine (1.3e-7) and its derivative is correct (ratio 1.0000) | the representation is right; do not touch `enforce_hermitian` |
| That derivative is purely imaginary *at the sample points* | `.real` before multiplying is the fix, not a coefficient change |
| Odd mode counts have no unpaired top mode and read clean under both variants | a free control, no construction needed |
| `spectral_residual` is used by `mms_solution` and by `Antiderivative.spectral_residual`'s sibling path | the change is confined to the Burgers method |

## Goals / Non-Goals

**Goals:**

- One convention for the nonlinear term, used by both the solver and the residual.
- An anchor that fails for this defect class, where the current on-grid check cannot.
- Only the Burgers MMS benchmark row moves; everything seeded stays byte-identical.

**Non-Goals:**

- Any change to `enforce_hermitian`, `generate_coeff` or `resize_modes`. All three are correct; the earlier suspicion against them was a measurement artifact and is recorded here so it is not re-raised.
- Changing what `mms_solution` band-limits to. Once both sides de-alias, the existing band-limit is consistent; narrowing it is a separate question about dataset difficulty, not correctness.
- The MMS round-trip through `Burgers.generate`. That needs a Basis forcing, which belongs to `time-dependent-forcing-in-numerical-burgers`.

## Decisions

### Extract the nonlinear term into one function used by both sides

Rather than patching `spectral_residual` to mirror the solver, both call a single helper that takes the field's spectral coefficients and the de-aliasing mask and returns the spectral nonlinear term. Two call sites cannot drift apart if there is only one implementation.

*Alternative rejected:* copy `.real` and the mask into `spectral_residual`. Fixes today's divergence and leaves the same trap for the next person who edits one and not the other — which is exactly how this arose.

### The de-aliasing mask comes from the basis, at the field's own mode count

`_solve_spectral` obtains it as `basis.dealias_mask(ns)`. `spectral_residual` works at the field's own resolution, so it asks for the mask at that resolution. A basis returning `None` (no de-aliasing defined) keeps today's behaviour on both sides, so no basis is newly excluded.

### The anchor measures the continuous residual, with two controls

The oracle is the equation itself, evaluated where the pair was not constructed: a grid finer than and offset from the sample points, with every term obtained from the field's own exact spectral derivatives. Two controls make the reading meaningful rather than merely small.

- **Negative control.** Perturb a known-good forcing by one percent; the residual must rise far above its clean value. Without this, a clean number could mean a blind instrument.
- **Natural control.** An odd mode count has no unpaired top mode, so it reads clean under both the old and new conventions. It costs nothing to include and it shows the measurement is not simply always-clean.

Tolerances are fixed from the float32 coefficient dtype rather than tuned: clean readings sit near `1e-7`, defective ones at `1e-2`, five orders apart, so the bound does not need to be delicate.

### The regeneration is part of this change, not a follow-up

`f` changes for the manufactured dataset, so the Burgers MMS row of the benchmark moves. Regenerating it here keeps the table and the code in one commit rather than leaving a window where the recorded numbers describe superseded behaviour. The other three rows must be shown unchanged, which doubles as evidence that `generate_coeff` really was untouched.

## Risks / Trade-offs

- **The MMS dataset becomes a different learning problem.** A correctly de-aliased forcing has different content from an aliased one, so the `f -> u` scores are not comparable across this change. → The regeneration task records both the old and new row so the discontinuity is visible in the history rather than silent.

- **`mms_solution`'s band-limit may now be more conservative than needed.** With both sides de-aliasing, some of the bandwidth headroom that was compensating for the defect is no longer required. → Out of scope by decision; noted so it is not mistaken for a remaining bug.

- **Two consumers of `spectral_residual` exist** — the Burgers MMS path and the residual test helpers. → The extraction keeps the public signature unchanged, so only the internal computation moves; existing callers see a corrected result, not a different API.

- **The odd-mode-count control passes today.** A control that already passes proves the measurement is not always-red, but it cannot prove the fix works. → The negative control carries that burden, and defect 1's isolation at four slots is the positive evidence.

## Migration Plan

1. Extract the nonlinear term into a shared helper; have `_solve_spectral` call it and confirm every existing test still passes and seeded output is unchanged.
2. Route `spectral_residual` through the same helper. Existing MMS tests that assert a zero discrete residual should still pass, since both sides move together.
3. Add the continuous-residual anchor with both controls; confirm it fails against the pre-change convention and passes after.
4. Regenerate the benchmark, recording the old and new MMS row and showing the other three unchanged.

Rollback is a revert; no persisted state, no API change.

## Open Questions

- Whether `Antiderivative.spectral_residual` has the same convention mismatch. Its nonlinear term is absent, so the `.real` and de-aliasing questions may not arise — worth a look during implementation, but it does not change this change's specs or tasks.
