## Why

A benchmark case's numbers depend on the process's default device as well as
on `--device`. Measured while building the forced-Burgers row:

- `sensor_grid(domain, n_per_axis)` builds its grid with `torch.linspace` and
  no device, so the grid lands on the process default.
  `OperatorDataset.from_fields` then evaluates the input field on it, and
  `Basis.__call__` moves the field's coefficients to the grid. So sensor
  *values* are computed wherever the process default points.
- Under `--device cuda`, every case's sensor grid and sensor values sat on the
  host, while its output coefficients were on `cuda:0`.
- The same case built under a cuda default and under a cpu default gave sensor
  values differing by up to 7.6e-6 (`antiderivative`), 4.8e-7
  (`antiderivative-grf`), 2.4e-7 (`burgers-numerical`), 1.2e-7 (`burgers-mms`)
  and 4.5e-8 (`burgers-forced`).
- `deepxde` sets the process default to cuda when it imports, and it imports
  inside the first DeepONet fit. So under `--device cpu`, every dataset built
  after that fit has its sensor values evaluated on cuda. A reported cpu row
  therefore depends on whether DeepONet ran, and on its place in the order.
- `spatial_field_at` in `scripts/run_benchmark.py` calls `.cpu()` on the
  slice's coefficients, so `burgers-numerical`'s output is pinned to the host
  whatever `--device` says.

`device-selection` already requires work derived for placed data to follow that
data. A sensor grid is exactly that, and so is the slice a script takes of a
placed field.

## What Changes

- `sensor_grid` takes the device to place its grid on. It is public, so
  leaving the device unstated still follows the process default, as any
  factory of a caller's data does. That keeps existing callers working.
- `OperatorDataset.from_fields` states the input field's device to
  `sensor_grid`, so sampling happens where the field is.
- `spatial_field_at` keeps the slice on its field's device.
- The `BenchmarkCasesStayPut` state machine regains the rule that repoints the
  process default between builds. It was dropped in `burgers-forced-benchmark`
  because it exposed this defect. Its comparison checks devices as well as
  values.
- **Reported numbers move deliberately.** A `--device cpu` row built after the
  first DeepONet fit had its sensor values evaluated on cuda, and now evaluates
  them on the host. Rows built before that fit, and every SpectralSVR-only run,
  must be shown unchanged.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `device-selection`: the requirement "Work derived for placed data follows
  that data" names sampling points among the values it covers, and gains three
  scenarios:
  - sensors are sampled where their field is;
  - a case does not depend on the process default;
  - a slice keeps its field's device.

## Impact

- `SpectralSVR/operators/dataset.py`: `sensor_grid` and
  `OperatorDataset.from_fields`.
- `scripts/run_benchmark.py`: `spatial_field_at`.
- `tests/`: the scenarios, the restored state-machine rule, and the callers of
  `sensor_grid` in `tests/operators/test_operators.py`.
- `results/`: `--device cpu` rows built after the first DeepONet fit move. They
  are compared against a base worktree with its own environment. The first
  DeepONet fit of a run is compared against its measured spread, since it is not
  reproducible even between runs of identical code.
- **Not touched:** DeepONet's own placement; the device asymmetry between
  targets and predictions recorded by `score-time-dependent-outputs`; and
  `inverse`'s time-awareness.
