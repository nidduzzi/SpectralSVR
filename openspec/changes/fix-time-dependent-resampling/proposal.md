## Why

`Basis.to_time_dependent(nt)` turns a field whose first axis is time into a time-dependent representation, one spatial spectrum per time slice. The `BasisLifecycle` state machine of `consolidate-the-test-suite` found it broken in two ways.

**It crashes for any slice count but one.** It reshapes the resampled spectra to `(len, *modes)`, but the resampled grid has shape `(len, nt, *modes[1:])`. Whenever `nt` differs from the time axis's mode count it raises: modes `(3, 2)` converted at `nt=2` gives `RuntimeError: shape '[1, 3, 2]' is invalid for input of size 4`. Every test used `nt` equal to that count or the default, which hid it.

**It puts its slices at the wrong times.** It samples the time axis on the periodic, half-open grid `t_i = i T / nt`, but a time-dependent field is read with its slices spanning the interval inclusively, `t_i = i T / (nt - 1)`, the convention `score-time-dependent-outputs` established and `evaluate_time_dependent_at` implements. A converted field therefore reports values at the wrong times. For a field periodic in time with eight modes on each axis, evaluated at `t = 0.5`, the source gives 0.0 and the converted field 0.354; at `t = 0.75`, -1.0 against -0.78. They agree only at `t = 0`.

An analytic anchor, `test_to_time_dependent_slice_spectra_match_analytic`, pins the half-open placement: it expects slice `j` to hold the field at `j / nt`. It encodes the defect.

**Its inverse is misplaced the same way.** *Added after the gate's first run.* `to_time_independent` reads slice `j` at the inclusive time `j T / (nt - 1)` but transforms it as though it were the periodic sample at `j T / N`. The two misplacements cancelled in the round trip, so `test_to_time_dependent_preserves_values` passed: it compares the two representations' values on their own grids, which differ along time. With `to_time_dependent` fixed alone, the round trip of `sin(2πt) cos(2πx)` is off by up to 1.0 at four modes, 0.71 at eight and 0.38 at sixteen. The anchor `test_to_time_independent_values_match_analytic` expects the result's half-open node `j` to hold the field at the inclusive time `j / (nt - 1)`, so it encodes this defect too. Nothing in the library or the benchmark calls `to_time_independent`; only tests do.

## What Changes

- `to_time_dependent(nt)` reshapes to `(len, nt, *modes[1:])`, so any slice count is accepted.
- It samples the time axis at the inclusive slice times `t_i = start + i (stop - start) / (nt - 1)`, so a converted field holds its source's values at its slice times, the interval's end included, which for a field periodic in time equals its start, and interpolates between them.
- The anchor pinning half-open placement is rewritten to expect slice `j` at `j / (nt - 1)`; its oracle class stays analytic.
- `to_time_independent` reads a time-dependent field of `N` slices at the periodic times `t_i = start + i (stop - start) / N` through the representation's linear interpolation between slices, then transforms. The time axis keeps `N` modes. A round trip of a field periodic in time therefore holds within linear interpolation's bound, not exactly.
- `test_to_time_independent_values_match_analytic` and `test_to_time_dependent_preserves_values` are rewritten to compare values at matching coordinates.
- After review: both conversions construct their result with a sampling of the right arity; a time-dependent field converts only to its own slice count; invalid slice counts, empty bases and fields without a spatial axis are refused with named errors.
- A fuzzed test draws slice counts below, at and above the time axis's mode count, with those edges and the default pinned, and checks the converted field against the source at the slice times to rounding and between them against linear interpolation's analytic bound.

**Also not in this change:** a `FourierBasis` under `ClosedUniform` sampling is inconsistent with itself, and `copy()` drops a configured precision; both predate this change, surfaced in its review, and are their own changes.

**Not in this change:** the Burgers manufactured-solution generator in `SpectralSVR/problems/burgers.py` carries its own copy of the half-open sampling and feeds the benchmark's MMS row. Fixing it moves a recorded number, so it is the next change, decided by the user, with the row's move measured and the record regenerated there.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coefficient-encodings`: gains a requirement that converting to a time-dependent representation samples the field at the slice times it will be read at, for any slice count, and a requirement that converting back samples it at the periodic times the result's nodes are read at.

## Impact

- `SpectralSVR/basis/__base.py`: `to_time_dependent` and `to_time_independent`.
- `tests/basis/test_coeff_anchors.py`: both anchors' expected times.
- `tests/basis/test_fourier_basis.py`: `test_to_time_dependent_preserves_values`.
- `tests/basis/`: a fuzzed conversion test.
- No benchmark row moves here: the benchmark's time-dependent fields come from the Burgers generator, which this change does not touch. `notebooks/cases/burgers/burgers.ipynb` reaches `to_time_dependent` only through that generator.
- `consolidate-the-test-suite` is paused above this change and rebases onto it.
