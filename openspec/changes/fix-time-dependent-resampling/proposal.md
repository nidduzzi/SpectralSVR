## Why

`Basis.to_time_dependent(nt)` turns a field whose first axis is time into a time-dependent representation, one spatial spectrum per time slice. The `BasisLifecycle` state machine of `consolidate-the-test-suite` found it broken in two ways.

**It crashes for any slice count but one.** It reshapes the resampled spectra to `(len, *modes)`, but the resampled grid has shape `(len, nt, *modes[1:])`. Whenever `nt` differs from the time axis's mode count it raises: modes `(3, 2)` converted at `nt=2` gives `RuntimeError: shape '[1, 3, 2]' is invalid for input of size 4`. Every test used `nt` equal to that count or the default, which hid it.

**It puts its slices at the wrong times.** It samples the time axis on the periodic, half-open grid `t_i = i T / nt`, but a time-dependent field is read with its slices spanning the interval inclusively, `t_i = i T / (nt - 1)`, the convention `score-time-dependent-outputs` established and `evaluate_time_dependent_at` implements. A converted field therefore reports values at the wrong times. For a field periodic in time with eight modes on each axis, evaluated at `t = 0.5`, the source gives 0.0 and the converted field 0.354; at `t = 0.75`, -1.0 against -0.78. They agree only at `t = 0`.

An analytic anchor, `test_to_time_dependent_slice_spectra_match_analytic`, pins the half-open placement: it expects slice `j` to hold the field at `j / nt`. It encodes the defect.

## What Changes

- `to_time_dependent(nt)` reshapes to `(len, nt, *modes[1:])`, so any slice count is accepted.
- It samples the time axis at the inclusive slice times `t_i = start + i (stop - start) / (nt - 1)`, so a converted field holds its source's values at its slice times, the interval's end included, which for a field periodic in time equals its start, and interpolates between them.
- The anchor pinning half-open placement is rewritten to expect slice `j` at `j / (nt - 1)`; its oracle class stays analytic.
- A fuzzed test draws slice counts below, at and above the time axis's mode count, with those edges and the default pinned, and checks the converted field against the source at the slice times to rounding and between them against linear interpolation's analytic bound.

**Not in this change:** the Burgers manufactured-solution generator in `SpectralSVR/problems/burgers.py` carries its own copy of the half-open sampling and feeds the benchmark's MMS row. Fixing it moves a recorded number, so it is the next change, decided by the user, with the row's move measured and the record regenerated there.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coefficient-encodings`: gains a requirement that converting to a time-dependent representation samples the field at the slice times it will be read at, for any slice count.

## Impact

- `SpectralSVR/basis/__base.py`: `to_time_dependent`.
- `tests/basis/test_coeff_anchors.py`: the anchor's expected slice times.
- `tests/basis/`: a fuzzed conversion test.
- No benchmark row moves here: the benchmark's time-dependent fields come from the Burgers generator, which this change does not touch. `notebooks/cases/burgers/burgers.ipynb` reaches `to_time_dependent` only through that generator.
- `consolidate-the-test-suite` is paused above this change and rebases onto it.
