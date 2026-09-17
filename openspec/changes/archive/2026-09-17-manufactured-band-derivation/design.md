## Context

See proposal.md — Why for the measurements. The design-relevant code:

```
Burgers.generate            burgers.py:37    dispatches three branches;
                                             manufactured branch at :90
  raises when solver is passed to the manufactured branch  <- the precedent
mms_solution                burgers.py:309
  band_limited = max(1, m // 3)     :337     a band expressed as a grid size
  basis.generate(n, band_limited)   :339     small array
  .resize_modes(modes)              :340     then padded
Problem.generate            __base.py:34     the shared contract
Antiderivative.generate     antiderivative.py:20   no band: linear, no product
```

| fact | consequence |
| --- | --- |
| `mms_solution` has one non-test caller, `Burgers.generate:90` | a band stated only on `mms_solution` reaches nobody; it has to be threaded |
| `Antiderivative` never band-limits — it is linear, so there is no doubled frequency | the band belongs on `Burgers.generate`, not the `Problem.generate` contract |
| the manufactured branch already raises when handed `solver` | rejecting a branch-inapplicable argument is the established pattern here |
| `burgers.py` has 11 `raise` statements; the library has one `logger.warning`, in the plotter | refusing beats warning by house convention |
| `resize_modes`' boundary rule already makes the padded field symmetric | this change removes the *need* for that repair in one caller, not the repair |

## Goals / Non-Goals

**Goals:**

- The manufactured field is conjugate-symmetric *as generated*, provable with the
  downstream repair disabled.
- The de-aliasing threshold is expressed exactly rather than approximately.
- The manufactured problem's difficulty is visible at the call site.

**Non-Goals:**

- `resize_modes`. Its boundary rule stays: grids arrive from user data, saved
  coefficients and predictions, and an even-sized source is permanently
  legitimate.
- Retuning the `2/3` fraction. Only the rounding within it moves.
- Deliberately aliased datasets. Refusing an overrunning band keeps the archived
  spec's continuous-versus-discrete conflict unreachable rather than inheriting
  it; a change that wants those datasets must settle that first.
- Regenerating the benchmark table.

## Decisions

### The threshold is exact once the band is a cutoff

The count-based band `B` spans `-(B//2) … +((B-1)//2)`, so its highest frequency
is `B//2` and the relationship to the band's width is approximate — the archived
design wrote it as "`kmax` is about `half/2`". A cutoff `K` keeps `-K … +K`, so
the highest frequency *is* `K`, the product reaches exactly `2K`, and the mask
keeps `m/3`. The condition becomes `2K <= m/3` with no rounding slack hidden in
it.

This also means the new band is expected to be **wider**, not narrower. At 8
modes the current band is 2 wide with a highest frequency of 1; the cutoff with
that same highest frequency keeps 3 modes. The asymmetry was costing a slot. The
tasks verify this rather than assume it, because a narrower band would silently
make every manufactured problem easier and move the benchmark row for a reason
nobody intended.

### The band is a `Burgers.generate` argument, not a `Problem.generate` one

`Antiderivative` has no band and cannot have one: it is linear, so nothing
doubles a frequency and no de-aliasing headroom is needed. Putting a band on the
shared contract would give every future problem an argument only one of them can
use.

`Burgers.generate` already carries `u0`, `f`, `nu`, `solver` and
`time_dependent_coeff` as its own named parameters beyond the contract's four, so
one more is the established pattern rather than an exception.

*Note on `**kwargs`:* the contract declares it and `Antiderivative` forwards it to
`basis.generate`. `Burgers.generate` never reads it, which is why `ty` reports it
unused. The band must therefore be an explicit parameter — routing it through
`**kwargs` there would be silently discarded.

### An overrunning band raises rather than warns

A band wider than the threshold produces a pair that is internally consistent —
forcing and solver de-alias alike — and does not satisfy the continuous equation.
Measured during `align-spectral-residual-with-solver` at `m = 16`: `m//2` leaves
`1.50e-1`, `m//2 - 1` leaves `7.56e-2`, `m//3` reaches `3.64e-7`.

A pair that passes its own check while failing the property it claims is the
exact defect class this chain of changes exists to remove. A warning would let
one into a dataset and then into a paper.

*Alternative rejected — warn and proceed.* Enables deliberately aliased datasets,
which is a real use, at the cost of a silently non-oracular pair and a warning
that scrolls past in a notebook. That use deserves its own change and its own
reconciliation of the archived overrun scenarios.

*Alternative rejected — clamp silently to the widest valid band.* Never fails, and
gives a caller a different problem than they asked for without saying so.

### The numerical branch rejects the band

`Burgers.generate` dispatches three branches and the band is meaningful only to
the manufactured one. The manufactured branch already raises when handed
`solver`, which "would have no effect" there. The band is the mirror case and
gets the mirror treatment, so an argument that cannot act is never silently
dropped.

### Symmetry is proved with the downstream repair disabled

Asserting the generated field is conjugate-symmetric proves nothing on its own —
`resize_modes`' boundary rule already guarantees that. The anchor has to disable
that rule and still read zero, which is what distinguishes *generated correctly*
from *repaired afterwards*. That is the whole point of this change over the one
beneath it.

## Risks / Trade-offs

- **The Burgers MMS benchmark row moves.** → Regenerated by
  `benchmark-reproducibility`, which sits above this change and above
  `utils-encoding-cleanup`, the two that move a row beneath it. Not duplicated
  here.

- **The manufactured dataset becomes a different learning problem.** The band's
  width changes, so `f -> u` scores are not comparable across this change. → The
  same discontinuity `align-spectral-residual-with-solver` recorded; the tasks
  record the old and new band at each mode count so it is visible in history.

- **A new argument on a public generator.** → Optional and defaulting to the
  derived band, so every existing call site is unaffected, and validated so a
  wrong value fails loudly at generation rather than quietly in a result.

- **`MANUFACTURED_CONTINUOUS_RESIDUAL`'s recorded values move.** → They are
  documentary, appearing in assertion messages rather than assertions. Re-measure
  and record rather than adjusting a tolerance to fit.

## Migration Plan

Derive the cutoff and prove the width relationship first, then thread the
argument through `Burgers.generate` with its validation and the numerical
branch's rejection, then the anchors. Each step leaves the suite green: the
derived default is the only path until the argument exists.

Rollback is a revert. No persisted state; the only signature change is an
optional argument.

## Open Questions

- Whether the archived `manufactured-solutions` scenario "Bandwidth that overruns
  the grid is rejected or resolved" should keep offering both, now that the code
  chooses rejection. This change settles its content on rejection while keeping
  the name; whether the alternative is worth preserving as a documented option is
  a question for whatever change wants aliased datasets, and does not affect this
  one's approach or tasks.
