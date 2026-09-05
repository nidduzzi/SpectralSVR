## Execution order

Sections 1-5 are a chain: each depends only on the sections before it, none on a
section after it. Section 6 touches a different function and depends on nothing
here, so it may run at any point. Section 7 closes.

```
1 documentation ─┐
                 ├─ 2 truncate fold ── 3 padding split ── 4 invariant anchor ── 5 evidence ── 7 verify
6 to_mag_angle ──┘   (independent of 1 and 6; 3 depends on 2)
```

**Section 2 must precede section 3.** They are inverses, and only that order
keeps the tree green at every step. Folding first leaves the round-trip intact
(naive padding writes `a` at `-N/2` and the fold reads back `a + 0`). Splitting
first without the fold halves the coefficient on the way back down, so the
round-trip breaks until section 2 lands. Each section carries its own anchor
rewrite so `just check` passes at the end of every one.

## 1. Documentation

- [x] 1.1 Document `rescale` on `Basis.resize_modes` (`basis/__base.py:847`) and on the underlying `resize_modes` (`utils/__init__.py:321`): `True` preserves the represented function by compensating the transform normalisation, `False` preserves the coefficient values and therefore scales the function by the mode-count ratio raised to the dimension count; verify both docstrings state which quantity each setting preserves
- [x] 1.2 Verify the documented factor matches measurement: a two-dimensional doubling with `rescale=False` scales the function by `0.2500` and a quadrupling by `0.0625`
- [x] 1.3 Document on `zero_coeff` (`utils/__init__.py:385`) that an even keep-count returns a field that is not real, and why: a kept set describing a real field is some `±k` pairs plus the lone mode `0`, so its size is always odd, and no substitute result answers the request that was made. Verify the behaviour is unchanged — no anchor moves and `test_zero_coeff_values` passes untouched
- [x] 1.4 Verify the docstring does not promise a future fix by name; it states the limitation, not a roadmap

## 2. Truncation folds the boundary mode

- [x] 2.1 In `resize_modes`'s downward branch (`utils/__init__.py:334-348`), add the source's `+M/2` content to the target's `-M/2` bin when the target mode count is even, instead of discarding it; verify the two mode numbers are the ones the target grid cannot distinguish
- [x] 2.2 Rewrite `test_resize_modes_truncate_values` (`tests/basis/test_coeff_anchors.py:32`) by hand from the new rule, not from the function's output — it is `oracle_analytic` and that is its purpose. For `x = [c0, c1, c2, c3]` at frequencies `[0, 1, ±2, -1]` resized to 2 modes, the target holds frequencies `{0, -1}` and its `-1` bin now carries `c3 + c1`, so the expected array is `[1+1j, 2.5+1j]` rather than `[1+1j, 0.5+2j]`
- [x] 2.3 Run `just check`; verify clean, with the padding anchor still passing untouched

## 3. Padding splits the lone top coefficient

- [x] 3.1 In `resize_modes`'s upward branch (`utils/__init__.py:349-374`), when the source mode count is even, halve the source's lone `-N/2` coefficient and write that half to both the target's `-N/2` and `+N/2` slots; verify an odd source count is left alone, having no lone coefficient
- [x] 3.2 Verify splitting rather than duplicating or zeroing: duplicating doubles the amplitude and breaks agreement at the original sample points by `5.357e-01`, zeroing breaks it by the same amount, and only halving leaves both the symmetry and the sampled values correct
- [x] 3.3 Rewrite `test_resize_modes_pad_values` (`tests/basis/test_coeff_anchors.py:46`) by hand from the new rule. For the same `x` resized to 6 modes, `c2 = -3+0j` at frequency `±2` is the lone coefficient and splits, so the expected array is `[c0, c1, -1.5, 0, -1.5, c3]` rather than `[c0, c1, 0, 0, c2, c3]`
- [x] 3.4 Run `just check`; verify clean

## 4. Anchor the invariants the two fixes establish

- [x] 4.1 Add an anchor asserting a conjugate-symmetric input stays conjugate-symmetric after padding, over source counts `{4, 5, 8, 9, 12}` and targets `{2N, 4N}`; verify it reads `0.00e+00` and that it fails against the pre-fix padding
- [x] 4.2 Add an anchor asserting pad-then-truncate returns the original grid to rounding, over the same counts; verify by ablation which anchor catches which half. *Measured, correcting this task as first written:* the round-trip anchor does not fail under either single revert. Removing the fold breaks it; removing the split does not, because naive padding writes `a` at `-N/2` and the fold reads back `a + 0`. It is the **pair** of anchors that is inseparable — the symmetry anchor catches a missing split, the round-trip anchor catches a missing fold, and neither catches the other's half
- [x] 4.3 Include the odd source counts as the isolation control: they have no lone coefficient, so they must read clean both before and after the fix, showing the anchor is measuring the defect and not merely always-green
- [x] 4.4 Assert the represented function is unchanged at the original sample points, to `8.88e-16`, so the repair is shown to alter the representation and not the field

## 5. Record the effect on the manufactured solution

- [ ] 5.1 Record the Hermitian error of `mms_solution`'s `u.coeff` before and after, at mode counts 8, 12, 16 and 32 — `7.859e-01`, `2.957e+00`, `0.000e+00`, `7.392e+00` before, and zero after — noting that 16 is clean before the fix because its band-limit `16 // 3 = 5` is the only odd one
- [ ] 5.2 Verify the continuous Burgers residual moves only in its last digits (`1.876e-07 → 1.750e-07` at 8 modes, `2.329e-07 → 2.176e-07` at 12, unchanged at 16), confirming `align-spectral-residual-with-solver`'s anchors stay green and that the field itself did not move
- [ ] 5.3 Verify `enforce_hermitian` and `generate_coeff` are untouched, so the earlier conclusion that they are correct is not reopened

## 6. Polar encoding (independent of sections 1-5)

- [ ] 6.1 Rewrite `to_mag_angle` (`utils/__init__.py:51`) to build its result the way `to_real_coeff` does — from a stacked magnitude/phase pair flattened over the trailing axes — rather than from a hardcoded `(shape[0], shape[1] * 2)` allocation with `[:, ::2]` writes; verify a `(2, 8, 8)` coefficient tensor converts instead of raising `expand(... size=[2, 8])`
- [ ] 6.2 Verify the single-mode-axis result is bit-identical to the previous implementation, so `test_to_mag_angle_reconstructs` and `test_to_mag_angle_idempotent_on_real` pass untouched
- [ ] 6.3 Verify the interleaving convention is unchanged: magnitude at even trailing positions, phase at odd, matching what the existing test reads with `[:, ::2]` and `[:, 1::2]`
- [ ] 6.4 Ensure the result is allocated on the input's device; verify a CUDA input returns a CUDA tensor, and skip cleanly when no accelerator is present rather than failing the suite
- [ ] 6.5 Verify the real precision corresponds to the input's complex precision at both single and double, so a complex128 input yields float64
- [ ] 6.6 Add `from_mag_angle` implementing `mag * exp(1j * angle)` for any number of mode axes, mirroring `to_complex_coeff`'s role for the interleaved-real pair; verify it round-trips `to_mag_angle` to rounding on 2-D and 3-D inputs
- [ ] 6.7 Export it from the package root next to `to_mag_angle`; verify `from SpectralSVR import from_mag_angle` resolves
- [ ] 6.8 Verify the inverse is idempotent on a real input in the same way `to_mag_angle` is, so the pair behaves consistently on the no-op path
- [ ] 6.9 Extend `tests/utils/test_utils.py` to cover multi-axis conversion, device preservation and the inverse round-trip

## 7. Verification

- [ ] 7.1 Run `just check`; verify clean, and that the only tests whose expected values changed are the two rewritten anchors
- [ ] 7.2 Verify `tests/test_oracle_coverage.py` passes with the markers chosen for every anchor added here
- [ ] 7.3 Verify the polar half moved nothing observable: `to_mag_angle` has no callers inside `SpectralSVR/`, so the seeded numerical Burgers field and the three non-MMS benchmark rows are unchanged
- [ ] 7.4 Record that the Burgers MMS row moves and is regenerated by `benchmark-reproducibility`, which sits above this change in the branch stack; do not regenerate the table here
- [ ] 7.5 Verify no source file gained an explanatory prose comment; the conversions and the resize must read from their names, with the rationale confined to the docstrings
