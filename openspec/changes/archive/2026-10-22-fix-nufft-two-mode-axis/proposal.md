## Why

The approximate (NUFFT) evaluation path raises `IndexError: index out of bounds` inside torchkbnufft whenever any evaluated axis has one or two modes, so a field the exact path evaluates fine crashes as soon as the strategy picks the fast path. A map over every mode tuple of up to three axes with 1 to 6 modes per axis found 174 failures, exactly the tuples with an axis of 1 or 2 modes. The cause is torchkbnufft's defaults: the oversampled grid is twice the mode count, and the Kaiser-Bessel table needs at least as many grid points as its 6 interpolation neighbours, so a grid of 2 or 4 points cannot be built. The consolidation found it and kept its strategy test away from it by drawing time modes from 3 upward; this change removes the defect so that restriction can go.

## What Changes

- `nufft_evaluate` states a grid size per axis of at least the interpolation neighbourhood, so a 1- or 2-mode axis builds a valid table. Axes of 3 or more modes keep the grid they have today (twice the mode count), so no input that ran before evaluates differently.
- A new capability states that approximate evaluation agrees with exact evaluation for every mode count, including 1 and 2, within the approximation bound the suite already holds.
- The strategy-conversion test in `tests/basis/test_coeff_anchors.py` draws time modes from 2 again, and the NUFFT tests draw mode counts that include 1 and 2, with those edges pinned.

## Capabilities

### New Capabilities
- `basis-evaluation`: how a Fourier basis evaluates a field at arbitrary points under its evaluation strategy, and how close the approximate path is to the exact one.

### Modified Capabilities

None. `coefficient-encodings` already says a conversion honours its evaluation strategy and adds the approximation's own error; that stays true, and only the crash below three modes goes.

## Impact

- Code: `SpectralSVR/basis/_nufft.py` (`nufft_evaluate`). `FourierBasis.evaluate` and `FourierBasis._raw_transform` call it and need no change.
- Tests: `tests/basis/test_nufft.py`, `tests/basis/test_coeff_anchors.py`.
- Recorded numbers: none can move. The changed code only alters the grid for axes of 1 or 2 modes, which raised before, so no recorded run can have taken that path; the design gives the diff argument.
