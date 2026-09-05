# Repair the coefficient encodings and the mode resize

## Why

Two functions in `utils` convert or reshape spectral coefficients, and both are
defective at the edges of their input space. `to_mag_angle` is public API and
breaks on any multi-dimensional problem. `resize_modes` is used throughout the
library and silently produces coefficient arrays that no longer describe a real
function. Neither defect is caught today.

### `to_mag_angle` is two-dimensional, device-losing and one-way

It converts complex coefficients to a magnitude/phase pair, mirroring what
`to_real_coeff` does for real/imaginary pairs. It is exported from the package
root, so it is public API.

**It only works in two dimensions.** The allocation is
`torch.empty((coeff.shape[0], coeff.shape[1] * 2))` and the writes are
`[:, ::2]` and `[:, 1::2]`. Any coefficient grid with more than one mode axis
raises:

```
expand(torch.DoubleTensor{[2, 8, 8]}, size=[2, 8])
```

That is every two-dimensional problem, Burgers included. Its sibling
`to_real_coeff` handles any number of axes.

**It loses the device.** The same allocation omits `device=`, so a CUDA tensor
produces a CPU result. Not an error — a silently wrong device.

**It has no inverse.** The reconstruction `mag * exp(1j * angle)` is asserted in
`tests/utils/test_utils.py:205`, so it is known and tested, but it exists
nowhere in the library. The polar pair is one-way in the API while the
interleaved-real pair is symmetric.

### `resize_modes` breaks conjugate symmetry when it pads

For a coefficient array to describe a real function, mode `-k` must hold the
conjugate of mode `+k`. On an even mode count that rule has one exception: the
top magnitude `N/2` has a negative bin and no positive one, so it is its own
conjugate partner and is therefore forced real. It is allowed to stand alone
only because it is the highest mode the grid can hold.

The upward path (`utils/__init__.py:349-374`) copies that lone coefficient into
the target's ordinary `-N/2` slot and leaves `+N/2` at zero. On the larger grid
the mode is no longer the top one, so it is no longer exempt, and the array is
no longer conjugate-symmetric. Hermitian error of the padded array, source built
Hermitian, padded to 16 modes:

| source modes | 4 | 5 | 8 | 9 |
| --- | --- | --- | --- | --- |
| Hermitian error after padding | 7.67e-01 | 0 | 3.70e-01 | 0 |

Split exactly by parity, because only even counts have the lone coefficient.

### `resize_modes` discards content when it truncates to an even target

The downward path (`:334-348`) keeps source modes `-M/2 .. +(M-1)/2` and drops
the source's `+M/2` content outright. On an `M`-point grid the modes `+M/2` and
`-M/2` are the same sampled cosine — `e^{i 2π (M/2) n / M} = (-1)^n =
e^{-i 2π (M/2) n / M}` — so they alias onto each other exactly and must combine
rather than one being thrown away.

### Why this survived the `spectral_residual` diagnosis

`Re` of the naively padded field equals the correctly split field exactly:
`4.441e-16`, and `8.882e-16` after differentiation. One real coefficient at
`-N/2` is `a·e^{-iθ}` and the split pair is `(a/2)(e^{-iθ} + e^{+iθ}) = a·cos θ`,
and those have the same real part. So every consumer that calls `.real` already
sees the right numbers. That is why
`align-spectral-residual-with-solver`'s `.real` fix worked, and why this defect
left no trace in the residual.

The defect is in the stored representation, not in the physics. It bites where
the coefficients are consumed *without* `.real` — and the manufactured MMS
solution is exactly that case, because its coefficients are the regression
target, reaching the model through `to_real_coeff`. Hermitian error of the
shipped `mms_solution` `u.coeff`, whose band-limit is `m // 3`:

| mode count | 8 | 12 | 16 | 32 |
| --- | --- | --- | --- | --- |
| band-limit | 2 (even) | 4 (even) | 5 (odd) | 10 (even) |
| Hermitian error | 7.859e-01 | 2.957e+00 | 0.000e+00 | 7.392e+00 |

The `16` column is the isolation control: its band-limit is the only odd one, it
has no lone coefficient, and it is clean to the bit.

### `zero_coeff` carries the same defect, and cannot be repaired the same way

Band-limiting in place keeps the array's length, so the kept `-m/2` bin's
partner at `+m/2` is the one being zeroed. Hermitian error of the filtered
field: `1.124` at keep 4, `0.557` at 6, `0.633` at 8, and exactly `0` for every
odd count.

Folding does not help here, unlike in `resize_modes` — measured `1.111`, `0.987`
and `0.201`, no better than leaving it alone. The reason is structural: a kept
set that describes a real field is some `±k` pairs plus the lone mode `0`, so its
size is always odd, and **an even keep-count has no valid answer at all**. Every
repair is a choice of which invalid request to substitute for the one asked.

That choice disappears once a band is expressed as a cutoff rather than a count,
which is a separate stacked change. This change documents the limitation where a
caller will meet it and leaves the behaviour alone. See design.md — Decisions.

### The `rescale` flag is undocumented

`Basis.resize_modes` (`basis/__base.py:847`) documents `target_modes` and omits
`rescale`, even though that flag decides whether the represented *function* or
the stored *coefficients* are preserved: `rescale=True` multiplies by
`prod(target_modes)/prod(current_modes)` to compensate the transform
normalisation; `rescale=False` leaves coefficients alone and so scales the
function by the mode-count ratio raised to the dimension count — measured
exactly `0.2500` for a doubling in two dimensions and `0.0625` for a
quadrupling. Both behaviours are correct and deliberate. Only the documentation
is missing, and that absence produced a false-positive bug report during the
`spectral_residual` diagnosis. That is the concrete cost.

## What Changes

- `to_mag_angle` works for any number of mode axes, matching `to_real_coeff`.
- It preserves the device of its input.
- A named inverse is added, so the polar encoding is symmetric with the
  interleaved-real encoding.
- `resize_modes` splits the source's lone top coefficient across both of the
  target's `±N/2` slots when it pads, so a padded array still describes a real
  function.
- `resize_modes` folds the source's `±M/2` content into the target's single
  `-M/2` bin when it truncates to an even target, so nothing is silently
  discarded.
- `resize_modes`'s `rescale` parameter is documented on both the `Basis` method
  and the underlying function, stating which of the function and the
  coefficients each setting preserves.
- `zero_coeff`'s docstring states that an even keep-count returns a field that
  is not real, and why no count-based repair exists. Its behaviour is unchanged.

The two `resize_modes` fixes are inverses of one another and must land together.
Splitting alone breaks the pad-then-truncate round-trip; folding alone leaves
the symmetry broken. Verified over source counts `{4, 5, 8, 9, 12}` and targets
`{2N, 4N}`: Hermitian error of the padded array `0.00e+00`, round-trip error
`0.00e+00`, and the represented function unchanged at the original sample points
to `8.88e-16`.

## Capabilities

### New Capabilities
- `coefficient-encodings`: what the library offers for representing a
  coefficient grid — the conversions between complex coefficients and real
  representations, and the resizing of a grid to a different mode count — what
  each guarantees about shape, device and invertibility, and which are
  symmetric.

## Impact

- `SpectralSVR/utils/__init__.py`: `to_mag_angle` and its new inverse; both
  branches of `resize_modes`; the `rescale` documentation; `zero_coeff`'s
  docstring only.
- `SpectralSVR/basis/__base.py`: the `rescale` documentation.
- `SpectralSVR/__init__.py`: export the inverse alongside the existing encoding.
- `tests/utils/test_utils.py`: extend the existing polar tests to more axes and
  to device preservation.
- `tests/basis/test_coeff_anchors.py`: `test_resize_modes_pad_values` and
  `test_resize_modes_truncate_values` pin the old convention by hand-written
  literal and must be rewritten by hand from the new rule. They are the only two
  failures the fix produces: measured 252 passed, 2 failed.

**The polar half changes nothing observable** — `to_mag_angle` has zero callers
inside the package and two in the tests, so no number can move through it.

**The resize half does change observable output, deliberately.**
`mms_solution`'s `u.coeff` becomes conjugate-symmetric, which changes the
regression target, so the Burgers MMS benchmark row moves. The represented
function does not change: the continuous Burgers residual moves only in its last
digits (`1.876e-07 → 1.750e-07` at 8 modes, `2.329e-07 → 2.176e-07` at 12,
unchanged at 16), so `align-spectral-residual-with-solver`'s anchors stay green.
Regenerating the table is *not* a task here — `benchmark-reproducibility` owns
the regeneration and sits above this change in the branch stack once the pending
rebases land. Duplicating it would couple the two changes for no gain.

**`zero_coeff`'s behaviour is deliberately unchanged.** Repairing it means
picking a rounding convention that a cutoff-based band type would delete, and
that would change existing notebook results silently in the meantime. The
constructive fix — making an invalid band unrepresentable rather than repairing
its consequences — lands in a change stacked on this one, together with
`mms_solution`'s band derivation in `burgers.py`. Neither belongs in a utils
repair.

`align-spectral-residual-with-solver` made `resize_modes` a non-goal on the
grounds that "the representation is right; do not touch `enforce_hermitian`".
That conclusion was correct about `enforce_hermitian` and `generate_coeff`,
which remain untouched here. `resize_modes` is a different function and a
different defect; this change does not reopen that one.
