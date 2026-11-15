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
| ~~Odd mode counts have no unpaired top mode and read clean under both variants~~ — the first clause holds, the second does not | *disproved during this change:* nine slots is odd, has no unpaired top mode (imaginary/real `4.85e-08`) and still reads `4.34e-02` under the complex product, because its product reaches mode 8 on a grid topping out at 7. Parity does not select a control; use the analytic pair instead |
| `spectral_residual` is used by `mms_solution` and by `Antiderivative.spectral_residual`'s sibling path | the change is confined to the Burgers method |
| `Burgers.rhs` is a fourth implementation, wrong by 2/3, with one smoke-test caller and none in 24 notebooks | delete rather than unify |
| the finite-difference `residual` at :328 backs five tests as a physical-space cross-check | leave it alone — see below |

## Goals / Non-Goals

**Goals:**

- One convention for the nonlinear term, used by both the solver and the residual.
- An anchor that fails for this defect class, where the current on-grid check cannot.
- Only the Burgers MMS benchmark row moves; everything seeded stays byte-identical.

**Non-Goals:**

- Any change to `enforce_hermitian`, `generate_coeff` or `resize_modes`. All three are correct; the earlier suspicion against them was a measurement artifact and is recorded here so it is not re-raised.
- Tuning the band-limit for dataset difficulty. Narrowing it to `m//3` is in scope because it is required for correctness (below); choosing a *different* value to make the learning problem harder or easier is not.
- The MMS round-trip through `Burgers.generate`. That needs a Basis forcing, which belongs to `time-dependent-forcing-in-numerical-burgers`.

## Decisions

### Extract the nonlinear term into one function used by both sides

Rather than patching `spectral_residual` to mirror the solver, both call a single helper that takes the field's spectral coefficients and the de-aliasing mask and returns the spectral nonlinear term. Two call sites cannot drift apart if there is only one implementation.

*Alternative rejected:* copy `.real` and the mask into `spectral_residual`. Fixes today's divergence and leaves the same trap for the next person who edits one and not the other — which is exactly how this arose.

### `Burgers.rhs` is deleted, not folded into the helper

Unifying it would work, but it is unused outside its own smoke test, its de-aliasing strategy is a second way of doing what the mask already does, and its stated purpose was superseded when `unify-solver-protocol` made pluggable solvers real. Fixing an unused duplicate leaves the duplicate; removing it leaves one nonlinear term and one de-aliasing strategy in the file, which is the outcome this change is for.

*Alternative rejected:* fix its scaling and route it through the helper. Correct but keeps a fourth entry point to the same computation, which is exactly the condition that produced this bug.

### The physical-space residual is deliberately NOT unified

`residual` at `burgers.py:328` computes the same equation by finite differences on sampled values, and five tests use it as a cross-check on the spectral path. It is arguably the only genuinely independent oracle among the implementations here, because it shares no code with the thing it checks. Collapsing it into the shared helper would destroy that independence and leave the spectral path verifying itself — the failure this whole change exists to correct.

### The band-limit narrows to `m//3`, and that is a correctness requirement

*Discovered during implementation; this reverses a Non-Goal in the original design.* Aligning the two sides is necessary but not sufficient. The mask discards real content above `m/3`, so once the residual de-aliases, a pair whose product overruns the band solves the de-aliased equation but not the continuous one. Measured at `m = 16` after the alignment: `m//2` leaves `1.50e-1`, `m//2 - 1` leaves `7.56e-2`, and `m//3` reaches `3.64e-7`.

The threshold is exact, not a margin: the product reaches `2*kmax`, `kmax` is about `half/2`, and the band keeps `m/3`, so `half <= m/3`. `m//2 - 1` is nowhere near it despite looking like a small step down.

This also settles what parity was doing. Before the `.real` fix, odd slot counts read clean and even ones did not, which looked like the Nyquist bin being at fault. With `.real` in place, parity stops predicting anything — even counts that fit are clean, odd counts that overrun are not. The `.real` fix removed the Nyquist-derivative pollution; the band-limit removes the aliasing. Two defects, two fixes, and the earlier parity signal was the first defect showing through while the second was still present.

### The de-aliasing mask comes from the basis, at the field's own mode count

`_solve_spectral` obtains it as `basis.dealias_mask(ns)`. `spectral_residual` works at the field's own resolution, so it asks for the mask at that resolution. A basis returning `None` (no de-aliasing defined) keeps today's behaviour on both sides, so no basis is newly excluded.

### The anchor measures the continuous residual, with two controls

The oracle is the equation itself, evaluated where the pair was not constructed: a grid finer than and offset from the sample points, with every term obtained from the field's own exact spectral derivatives. Two controls make the reading meaningful rather than merely small.

- **Negative control.** Perturb a known-good forcing by one percent; the residual must rise far above its clean value. Without this, a clean number could mean a blind instrument.
- **Positive control.** The doubly-periodic analytic pair, which satisfies the continuous equation by construction and is alias-free when kept inside `ns > 4*mx`. It reads `3.41e-8` and shows the measurement is not simply always-red on a pair independently known to be correct. An odd mode count was planned here instead, on the belief that having no unpaired top mode made it clean; that belief was wrong, because an odd count still overruns the band if it is wide enough, so it would have controlled for one defect while silently carrying the other.

Tolerances are fixed from the float32 coefficient dtype rather than tuned: clean readings sit near `1e-7`, defective ones at `1e-2`, five orders apart, so the bound does not need to be delicate.

### The regeneration is part of this change, not a follow-up

`f` changes for the manufactured dataset, so the Burgers MMS row of the benchmark moves. Regenerating it here keeps the table and the code in one commit rather than leaving a window where the recorded numbers describe superseded behaviour. The other three rows must be shown unchanged, which doubles as evidence that `generate_coeff` really was untouched.

## Measured outcome

Continuous residual of the manufactured pair, before and after, with the analytic doubly-periodic pair as a positive control at `3.41e-8`:

| mode count | before | after |
| --- | --- | --- |
| 8 | 2.53e-2 | 2.15e-7 |
| 16 | 2.62e-2 | 2.10e-7 |
| 32 | 2.72e-3 | 2.62e-7 |

Benchmark rows, same command at the parent commit and after:

| dataset | change |
| --- | --- |
| Burgers MMS (`f -> u`) | r2 `0.533047` -> `0.464886`, a 12.8% relative move in the metrics |
| numerical Burgers, DeepONet GRF | bit-identical |
| antiderivative | unchanged by this change (see below) |

The antiderivative row showed a `9.06e-7` difference between the two runs, which is *not* attributable to this change: running the same pre-change code in two different worktrees differs by `4.41e-6`, while running the new code against the old in comparable environments differs by exactly `0`. The pipeline is deterministic within one environment and not across them, at the `1e-6` level that float32 permits. That is a reproducibility property of the benchmark rather than a defect here, and it is the kind of thing `benchmark-reproducibility` exists to record.

## Risks / Trade-offs

- **The MMS dataset becomes a different learning problem.** A correctly de-aliased forcing has different content from an aliased one, so the `f -> u` scores are not comparable across this change. → The regeneration task records both the old and new row so the discontinuity is visible in the history rather than silent.

- **`mms_solution`'s band-limit may now be more conservative than needed.** With both sides de-aliasing, some of the bandwidth headroom that was compensating for the defect is no longer required. → Out of scope by decision; noted so it is not mistaken for a remaining bug.

- **Two consumers of `spectral_residual` exist** — the Burgers MMS path and the residual test helpers. → The extraction keeps the public signature unchanged, so only the internal computation moves; existing callers see a corrected result, not a different API.

- **The positive control passes today.** A control that already passes cannot prove the fix works; it proves the instrument reads clean on a pair that is correct for reasons the code under test did not supply. → The negative control carries the discriminating burden, and defect 1's isolation at four slots is the positive evidence. Unlike the odd-mode-count control it replaced, this one stays informative after `utils-encoding-cleanup` repairs `resize_modes`: that change makes even counts conjugate-symmetric too, which would have left a parity-based control passing for every input and testing nothing.

## Migration Plan

1. Extract the nonlinear term into a shared helper; have `_solve_spectral` call it and confirm every existing test still passes and seeded output is unchanged.
2. Route `spectral_residual` through the same helper. Existing MMS tests that assert a zero discrete residual should still pass, since both sides move together.
3. Add the continuous-residual anchor with both controls; confirm it fails against the pre-change convention and passes after.
4. Regenerate the benchmark, recording the old and new MMS row and showing the other three unchanged.

Rollback is a revert; no persisted state, no API change.

## Open Questions

*Closed during implementation.* `Antiderivative.spectral_residual` is `u.grad() - ut` with the DC mode zeroed — purely linear, no product, so neither the real-part nor the de-aliasing question arises. Nothing to align there.
