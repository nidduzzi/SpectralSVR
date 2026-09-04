# Repair the polar coefficient encoding and document the resize flag

## Why

`to_mag_angle` converts complex coefficients to a magnitude/phase pair, mirroring what `to_real_coeff` does for real/imaginary pairs. It is exported from the package root, so it is public API. It carries two defects and one asymmetry, and a related flag elsewhere is undocumented in a way that has already cost real time.

**It only works in two dimensions.** The allocation is `torch.empty((coeff.shape[0], coeff.shape[1] * 2))` and the writes are `[:, ::2]` and `[:, 1::2]`. Any coefficient grid with more than one mode axis raises:

```
expand(torch.DoubleTensor{[2, 8, 8]}, size=[2, 8])
```

That is every two-dimensional problem, Burgers included. Its sibling `to_real_coeff` handles any number of axes.

**It loses the device.** The same allocation omits `device=`, so a CUDA tensor produces a CPU result. Not an error — a silently wrong device.

**It has no inverse.** The reconstruction `mag * exp(1j * angle)` is asserted in `tests/utils/test_utils.py:205`, so it is known and tested, but it exists nowhere in the library. The polar pair is one-way in the API while the interleaved-real pair is symmetric.

**`resize_modes`'s `rescale` flag is undocumented.** `Basis.resize_modes` (`basis/__base.py:847`) documents `target_modes` and omits `rescale`, even though that flag decides whether the represented *function* or the stored *coefficients* are preserved: `rescale=True` multiplies by `prod(target_modes)/prod(current_modes)` to compensate the transform normalisation; `rescale=False` leaves coefficients alone and so scales the function by the mode-count ratio raised to the dimension count — measured exactly `0.2500` for a doubling in two dimensions and `0.0625` for a quadrupling. Both behaviours are correct and deliberate. Only the documentation is missing, and that absence produced a false-positive bug report during the `spectral_residual` diagnosis. That is the concrete cost.

## What Changes

- `to_mag_angle` works for any number of mode axes, matching `to_real_coeff`.
- It preserves the device of its input.
- A named inverse is added, so the polar encoding is symmetric with the interleaved-real encoding.
- `resize_modes`'s `rescale` parameter is documented on both the `Basis` method and the underlying function, stating which of the function and the coefficients each setting preserves.

## Capabilities

### New Capabilities
- `coefficient-encodings`: what conversions the library offers between complex coefficients and real representations, what each guarantees about shape, device and invertibility, and which are symmetric.

## Impact

- `SpectralSVR/utils/__init__.py`: `to_mag_angle` and its new inverse.
- `SpectralSVR/basis/__base.py` and `SpectralSVR/utils/__init__.py`: the `rescale` documentation.
- `SpectralSVR/__init__.py`: export the inverse alongside the existing encoding.
- `tests/utils/test_utils.py`: extend the existing polar tests to more axes and to device preservation.
- **Nothing observable changes.** `to_mag_angle` has zero callers inside the package and only two in the tests, so no number, no seeded field and no benchmark row can move.
