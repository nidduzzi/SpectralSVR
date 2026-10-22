## Context

See proposal.md — Why for the two `resize_modes` defects, the `to_mag_angle`
defects, and the measurements. The design-relevant facts, all in
`SpectralSVR/utils/__init__.py`:

```
resize_modes, :334-348 (down)        resize_modes, :349-374 (up)
  keep -M/2 .. +(M-1)/2                copy source -N/2 into an ordinary slot
  drop the source's +M/2               leave +N/2 at zero
```

| fact | consequence |
| --- | --- |
| `Re` of the naively padded field equals the split field to `4.441e-16`, and `8.882e-16` after differentiation | every consumer calling `.real` already sees correct values; the defect is in the representation, not the physics |
| `mms_solution`'s `u.coeff` is a regression target and reaches the model through `to_real_coeff`, where nothing calls `.real` | this is the consumer the defect actually reaches |
| `Basis.resize_modes` delegates to the free `resize_modes` for both the time-dependent and plain paths | one implementation to fix, not two |
| `zero_coeff` carries the same boundary asymmetry (Hermitian error `1.124` at keep 4, `0.557` at 6, `0.633` at 8; odd counts exactly `0`) | same defect family, deliberately out of scope — see Decisions |
| `dealias_mask` expresses its band as `k.abs() <= fraction * (modes / 2)` | the codebase already has the correct concept in one place; the count-based sites are the ones that drifted |

## Goals / Non-Goals

**Goals:**

- One boundary rule for `resize_modes`, applied in both directions, so a resized
  array is well-formed on its own rather than depending on every consumer to
  call `.real`.
- The polar encoding symmetric with the interleaved-real one, and working at any
  axis count.
- The `zero_coeff` instance of the same defect recorded in the code where a
  caller will meet it, without changing what it computes.

**Non-Goals:**

- A band type, and any API change that follows from one. The constructive fix —
  making an invalid band unrepresentable rather than repairing its consequences
  — is a separate change stacked on this one. It reaches `zero_coeff`'s
  signature and `mms_solution`'s band derivation in `burgers.py`, neither of
  which belongs in a utils repair.
- Any behavioural change to `zero_coeff`.
- `enforce_hermitian` and `generate_coeff`. `align-spectral-residual-with-solver`
  established both are correct; this change does not reopen that.
- Regenerating the benchmark. `benchmark-reproducibility` owns it.

## Decisions

### Split when padding, fold when truncating, and land them together

On an even mode count the top magnitude has a negative bin and no positive one,
so it is its own conjugate partner and is forced real. It is exempt from the
pairing rule only because it is the highest mode the grid holds. Padding moves
it somewhere it is no longer the highest, where the exemption does not apply.

Padding therefore halves it across both of the target's `±N/2` slots, and
truncating sums the source's `±M/2` content into the target's single `-M/2` bin.
The two are inverses, and each is wrong without the other: splitting alone
halves the coefficient on the way back down, folding alone leaves the symmetry
broken. Verified over source counts `{4, 5, 8, 9, 12}` and targets `{2N, 4N}` —
padded Hermitian error `0.00e+00`, round-trip error `0.00e+00`, represented
function unchanged at the original sample points to `8.88e-16`.

*Alternative rejected — duplicate the coefficient into both slots.* Restores
symmetry and doubles the amplitude, breaking agreement at the original sample
points by `5.357e-01`.

*Alternative rejected — zero it before padding.* Also restores symmetry, and
loses the mode entirely: same `5.357e-01` disagreement.

Halving is right because the lone coefficient is real, so one coefficient
`a·e^{-iθ}` and the split pair `(a/2)(e^{-iθ} + e^{+iθ}) = a·cos θ` describe the
same wave.

### Repair the array rather than lean on `.real`

Every consumer that samples through `.real` already gets the correct answer, so
this could be left alone. It is not, because the defect reaches consumers that
have no `.real` to save them: coefficients are the regression target, and the
`dtype-seam` change stores them as interleaved real pairs where the imaginary
half is data rather than something to discard.

*Alternative rejected — document the asymmetry and require callers to sample
through `.real`.* That is the arrangement that produced this bug: a correctness
obligation held by every call site instead of by the function.

### Fix the free function, not the `Basis` method

`Basis.resize_modes` delegates to the free `resize_modes` on both its
time-dependent and plain paths, so the free function is the single place the
rule can live. Fixing the method would leave direct callers of the function —
including `Burgers.mms_solution` — on the old behaviour.

### The truncation fold before the padding split

They are inverses, and only this order leaves the suite green at every step.
Folding first keeps the round-trip intact, because naive padding writes `a` at
`-N/2` and the fold reads back `a + 0`. Splitting first without the fold halves
the coefficient coming back down, so the round-trip is broken until the fold
lands. Each section carries its own anchor rewrite for the same reason.

### Both value anchors are rewritten by hand

`test_resize_modes_pad_values` and `test_resize_modes_truncate_values` are
`oracle_analytic`: their expected arrays are derived from the bin layout by
hand, and regenerating them from the function under test would convert them into
`oracle_consistency` tests that assert the code agrees with itself. That is the
failure `align-spectral-residual-with-solver` existed to correct, and repeating
it here to save effort would be the same mistake in a new place.

### `zero_coeff` is documented, not repaired

It carries the same defect, and it cannot be repaired the same way. A resize
changes the array's length, so the boundary content has somewhere symmetric to
go; `zero_coeff` filters in place, so folding the `+m/2` content into the kept
`-m/2` bin leaves that bin's partner still zeroed — measured Hermitian error
`1.111`, `0.987`, `0.201`, no better than doing nothing.

The reason is structural. A kept set that describes a real field is some `±k`
pairs plus the lone mode `0`, so its size is always odd. **An even keep-count has
no valid answer**, and every repair is therefore a choice of which invalid
request to substitute — round up to `m+1`, round down to `m-1`, or refuse.

That choice disappears under the constructive model: a band expressed as a
cutoff `K` yields `2K + 1` modes, so an even band cannot be constructed and
there is nothing to round. Committing to a rounding convention here would
invent an interim semantic that the band type deletes, and would change existing
notebook results silently. So this change adds only a docstring stating that an
even keep-count yields a field that is not real, and the constructive fix lands
in the stacked change.

*Alternative rejected — reject even counts here.* Forward-compatible, since
anything passing such a check is expressible as a band, and it fails loudly
rather than silently. Rejected because it breaks working calls in four notebooks
ahead of the change that would give them the replacement spelling.

### `to_mag_angle` is rebuilt on `to_real_coeff`'s pattern

It is rewritten to stack the magnitude/phase pair and flatten over the trailing
axes, the way `to_real_coeff` already does, rather than having its hardcoded
`(shape[0], shape[1] * 2)` allocation widened. The stacking form generalises to
any axis count and inherits device and precision from its input, which fixes all
three of its defects at once instead of one at a time.

`from_mag_angle` is added rather than left as a documented one-liner, so both
encodings offer the same shape of API and neither is one-way.

## Risks / Trade-offs

- **The manufactured dataset becomes a different learning problem.** `u.coeff`
  changes for every even band-limit, so the Burgers MMS row moves. → The row is
  regenerated by `benchmark-reproducibility`, which sits above this change in
  the stack. Recording it here as well would couple the two changes.

- **The band type will move that row a second time.** Rounding `m // 3` down to
  odd narrows the band slightly. → It stays inside the `band <= m/3` threshold
  that `align-spectral-residual-with-solver` established, so it remains correct;
  the ordering requirement is only that the benchmark is regenerated above both.

- **Truncation now retains content it used to discard.** A caller who wanted a
  strict low-pass at the boundary gets the folded value instead. → The two mode
  numbers are the same sampled wave on the target grid, so the content was
  representable and discarding it was the defect, not the feature. The only
  in-library callers are `Basis`'s arithmetic paths, which resize operands to a
  common grid and are strictly better off.

- **The boundary rule is applied per axis inside the existing loop.** A grid
  even on both axes has it applied twice. → The anchor covers a two-dimensional
  case; the `rescale` multiply happens afterwards and is uniform, so it does not
  interact with the order.

- **`zero_coeff` ships with a live defect and only a docstring.** → Accepted:
  the constructive fix supersedes any repair made here, and a rounding
  convention adopted now would change notebook results silently and then be
  deleted. The exposure is that the band change may be deferred, in which case a
  documented defect remains in public API.

## Migration Plan

The task sections are a chain — documentation, truncation fold, padding split,
invariant anchors, evidence — with the polar work independent of all of them.
Each of the two `resize_modes` sections carries its own anchor rewrite, so
`just check` passes at the end of every section rather than only at the end.

Rollback is a revert. No persisted state; the only API addition is
`from_mag_angle`, and no existing signature changes.

## Open Questions

- Whether the band type, when it lands, should also express `dealias_mask`'s
  `fraction` argument. `dealias_mask` already computes a cutoff on `|k|` rather
  than a count, so it is the existing precedent rather than another defect site.
  Safely deferrable: it changes neither this change's approach nor its tasks.
