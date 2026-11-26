## Why

The Burgers generator's manufactured branch, asked for a time-dependent pair (the default), stores slices that are not the manufactured field at the times they are read at. It samples the 2-D (time, space) field on that field's own periodic grid, `start + i (stop - start) / nt`, while a time-dependent field reads its slices at the inclusive times `linspace(start, stop, nt)`, the contract `coefficient-encodings` has stated since `fix-time-dependent-resampling`. Measured on two functions of 8 x 8 modes, the stored slices differ from the manufactured field at their stated slice times by 86% of the largest value at 5 slices, 49% at 11 and 11% at 50. Converting through the library's own time-dependent conversion gives 1.5e-7 to 4.5e-7 against the field's closed-form Fourier series, single-precision rounding. A manufactured pair is an oracle, so an oracle displaced in time is worse than none.

Measuring the fix on drawn domains exposed a second defect on the same path. Reading a time-dependent field on its own grid computes each time's slice index in single precision without bounding it. On 1611 of 8000 drawn intervals, the last time's index rounds past the last slice and the read raises `RuntimeError: INDICES element is out of DATA bounds`. Three other read paths bound the index, one of them repeating the interpolation by hand, and resizing in time builds its own index.

The adversarial review found a third defect on the same entry point. `Burgers.generate` takes `**kwargs` and reads none of them, so `dtype=torch.complex128` or `coeff_dtype=torch.complex128` returns single-precision coefficients without a word. The abstract `Problem.generate`, `Basis.generate` and `FourierBasis.generate` swallow the same way, and `Antiderivative.generate` forwards its extras into the last. The committed `burgers.ipynb` passes `periods=` to `FourierBasis.generate`, which has never read it.

## What Changes

- The manufactured branch converts its pair through the library's time-dependent conversion instead of sampling the field by hand, so each slice holds the manufactured field at its stated time.
- A requirement in `manufactured-solutions` states that a time-dependent manufactured pair holds the field's values at its slice times.
- A fuzzed test checks each slice against the manufactured field's closed-form Fourier series at the slice times, over slice counts, mode counts and domains, with the edge cases pinned. It absorbs the two smoke tests of the same branch.
- One value type places times between neighbouring slices, computed in double precision. It is the only way to address slices, so an index past the last slice cannot be formed. All five read paths use it.
- A requirement in `coefficient-encodings` states that a read addresses only the slices the field holds.
- Every problem and basis `generate` names its parameters and takes no `**kwargs`, so an argument it does not take is refused with `TypeError` at the call. `Antiderivative.generate` passes on only named parameters. A requirement in `coefficient-precision` states it.

## Capabilities

### New Capabilities

None.

### Modified Capabilities
- `manufactured-solutions`: adds the requirement that a time-dependent manufactured pair holds the field at its slice times.
- `coefficient-encodings`: adds the requirement that a read of a time-dependent field addresses only the slices it holds.
- `coefficient-precision`: adds the requirement that a generator refuses an argument it does not take.

## Impact

- Code: `SpectralSVR/problems/burgers.py` (`Burgers.generate`, manufactured branch only); `SpectralSVR/utils` (`interpolate_tensor` takes slice positions); `SpectralSVR/basis/__base.py` and `FourierBasis.py` (every time-dependent read, and `generate`); `SpectralSVR/problems/__base.py` and `antiderivative.py` (`generate`). A caller passing an argument a generator never read now gets `TypeError`; the committed notebook call with `periods=` is one, and it changed nothing when it ran.
- Tests: `tests/problems/test_problems.py` gains the new test and loses the two smoke tests it absorbs; `tests/utils/test_utils.py` and `tests/basis/` test the slice positions and the reads. The four existing call sites of the default path check shapes, devices and that a residual runs, not slice values, so none pins the defect.
- Recorded numbers: none move. The benchmark's manufactured case asks for a non-time-dependent pair, and its forced and numerical cases use the numerical branch. Every Burgers generator call in the committed notebooks passes `time_dependent_coeff=False`. The design gives the argument and the check.
