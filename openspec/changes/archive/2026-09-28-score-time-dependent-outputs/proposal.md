## Why

A time-dependent output field can be fitted but not scored. Measured while
building the forced-Burgers row:

- `OperatorDataset.targets(x)` calls `self.u(x)` without `t`, and a
  time-dependent field raises
  `AssertionError: t must not be none for time dependent evaluations`.
- `SpectralSVR.forward(f, x)` refuses a `(t, x)` query. A time-dependent
  output's `modes` are spatial only, so `x.shape[1] != len(self.modes)` raises
  `ValueError`. It also never passes `t` to `evaluate`, and it reshapes the
  predicted coefficients to `(n, *self.modes)`, dropping the time axis.
  `train(..., u_time_dependent=True)` is accepted, so the model can be fitted
  on something it cannot then evaluate.

`burgers-forced-benchmark` therefore landed as a pilot. It scores `u` as a field
periodic in time over `[0, T)`, only at the times where that is exact, and
leaves `u(T)` unscored. Its label says so. This change makes the pilot
unnecessary.

## What Changes

- `OperatorDataset.targets` evaluates a time-dependent output at `(t, x)`
  query rows. The first column is time, the rest are space, which is the
  layout the forced row's queries already use.
- `SpectralSVR.forward` evaluates a time-dependent output at the same `(t, x)`
  rows. It keeps the time axis of the predicted coefficients and passes each
  row's time.
- Each query row gives one value. `FourierBasis.evaluate` with
  `time_dependent=True` currently returns a grid, `out[:, i, j] = u(t_i, x_j)`
  for `Q` times by `Q` points. Measured: it is exact at every slice time,
  including `t = t1`, and linear between slices. Scoring paired rows needs
  paired evaluation. Taking the diagonal of the grid would cost `O(Q²)`.
- The forced row drops its pilot. `u` stays time-dependent, the query covers
  `[0, T]` including `u(T)`, `periodic_in_time` and its test are removed, and
  the label loses `pilot: t periodic`.
- **A reported number moves deliberately.** The forced row now scores the
  whole interval, so its numbers change. The other rows must be shown
  unchanged.

## Capabilities

### New Capabilities

None. What the comparison scores, and where, is `operator-comparison`'s
contract.

### Modified Capabilities

- `operator-comparison`: gains a requirement that a time-dependent output is
  scored at `(t, x)` query rows, by the targets and by every operator's
  prediction alike, over the whole time interval. The existing requirements'
  text is unchanged. "Targets are evaluated, not pre-sampled" already requires
  evaluation from the stored field at the query points, and this extends that
  to fields that vary in time.

## Impact

- `SpectralSVR/operators/dataset.py`: `targets` for time-dependent outputs.
- `SpectralSVR/model/SpectralSVR.py`: `forward` for time-dependent outputs.
- `SpectralSVR/basis/`: paired evaluation of a time-dependent field at query
  rows, wherever the design places it.
- `scripts/run_benchmark.py`: the pilot removed from `burgers_forced_case`,
  and `periodic_in_time` deleted.
- `tests/`: the new scenarios, and the removal of the `periodic_in_time` test.
- `results/`: the forced row moves. Every other row must be shown unchanged
  against a base worktree with its own environment, on the same `--device`.
  That comparison has to allow for the first DeepONet fit of a run not being
  reproducible, as recorded by `burgers-forced-benchmark`.
- **Not touched:** time-independent evaluation. Every existing row takes that
  path, and it must be shown bit-identical.
- **Not touched:** sensor placement, since `sensor_grid` follows the process
  default. That is the next change.
