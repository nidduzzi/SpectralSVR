## Why

The Burgers generator's manufactured branch, asked for a time-dependent pair (the default), stores slices that are not the manufactured field at the times they are read at. It samples the 2-D (time, space) field on that field's own periodic grid, `start + i (stop - start) / nt`, while a time-dependent field reads its slices at the inclusive times `linspace(start, stop, nt)`, the contract `coefficient-encodings` has stated since `fix-time-dependent-resampling`. Measured on two functions of 8 x 8 modes, the stored slices differ from the manufactured field at their stated slice times by 86% of the largest value at 5 slices, 49% at 11 and 11% at 50. Converting through the library's own time-dependent conversion gives 1.7e-7 to 1.9e-7, single-precision rounding. A manufactured pair is an oracle, so an oracle displaced in time is worse than none.

## What Changes

- The manufactured branch converts its pair through the library's time-dependent conversion instead of sampling the field by hand, so each slice holds the manufactured field at its stated time.
- A requirement in `manufactured-solutions` states that a time-dependent manufactured pair holds the field's values at its slice times.
- A fuzzed test checks it against the non-time-dependent pair evaluated at the slice times, over slice counts and mode counts with the edge counts pinned.

## Capabilities

### New Capabilities

None.

### Modified Capabilities
- `manufactured-solutions`: adds the requirement that a time-dependent manufactured pair holds the field at its slice times.

## Impact

- Code: `SpectralSVR/problems/burgers.py` (`Burgers.generate`, manufactured branch only).
- Tests: `tests/problems/test_problems.py` gains the new test. The four existing call sites of the default path check shapes, devices and that a residual runs, not slice values, so none pins the defect.
- Recorded numbers: none move. The benchmark's manufactured case asks for a non-time-dependent pair, and its forced and numerical cases use the numerical branch. Every Burgers generator call in the committed notebooks passes `time_dependent_coeff=False`. The design gives the argument and the check.
