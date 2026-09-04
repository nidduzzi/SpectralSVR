## 1. Polar encoding: any number of mode axes

- [ ] 1.1 Rewrite `to_mag_angle` (`utils/__init__.py:51`) to build its result the way `to_real_coeff` does — from a stacked magnitude/phase pair flattened over the trailing axes — rather than from a hardcoded `(shape[0], shape[1] * 2)` allocation with `[:, ::2]` writes; verify a `(2, 8, 8)` coefficient tensor converts instead of raising `expand(... size=[2, 8])`
- [ ] 1.2 Verify the single-mode-axis result is bit-identical to the previous implementation, so `test_to_mag_angle_reconstructs` and `test_to_mag_angle_idempotent_on_real` pass untouched
- [ ] 1.3 Verify the interleaving convention is unchanged: magnitude at even trailing positions, phase at odd, matching what the existing test reads with `[:, ::2]` and `[:, 1::2]`

## 2. Device and precision

- [ ] 2.1 Ensure the result is allocated on the input's device; verify a CUDA input returns a CUDA tensor, and skip cleanly when no accelerator is present rather than failing the suite
- [ ] 2.2 Verify the real precision corresponds to the input's complex precision at both single and double, so a complex128 input yields float64

## 3. The named inverse

- [ ] 3.1 Add `from_mag_angle` implementing `mag * exp(1j * angle)` for any number of mode axes, mirroring `to_complex_coeff`'s role for the interleaved-real pair; verify it round-trips `to_mag_angle` to rounding on 2-D and 3-D inputs
- [ ] 3.2 Export it from the package root next to `to_mag_angle`; verify `from SpectralSVR import from_mag_angle` resolves
- [ ] 3.3 Verify the inverse is idempotent on a real input in the same way `to_mag_angle` is, so the pair behaves consistently on the no-op path

## 4. Documentation of the resize flag

- [ ] 4.1 Document `rescale` on `Basis.resize_modes` (`basis/__base.py:847`) and on the underlying `resize_modes` (`utils/__init__.py:321`): `True` preserves the represented function by compensating the transform normalisation, `False` preserves the coefficient values and therefore scales the function by the mode-count ratio raised to the dimension count; verify both docstrings state which quantity each setting preserves
- [ ] 4.2 Verify the documented factor matches measurement: a two-dimensional doubling with `rescale=False` scales the function by `0.2500` and a quadrupling by `0.0625`

## 5. Verification

- [ ] 5.1 Extend `tests/utils/test_utils.py` to cover multi-axis conversion, device preservation and the inverse round-trip; verify `tests/test_oracle_coverage.py` still passes with the markers chosen
- [ ] 5.2 Run `just check`; verify clean
- [ ] 5.3 Verify nothing observable moved: `to_mag_angle` has no callers inside `SpectralSVR/`, so regenerate the benchmark and confirm all four rows and the seeded numerical Burgers field are unchanged
- [ ] 5.4 Verify no source file gained an explanatory prose comment; the conversions must read from their names, with the normalisation rationale confined to the docstrings
