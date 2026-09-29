## Context

See proposal.md - Why. The state that shapes the approach:

| fact | consequence |
| --- | --- |
| `sensor_grid` is exported from `SpectralSVR.operators` | changing its signature is a public API change, so the device has to be additive |
| `from_fields` is the only library caller of `sensor_grid` | stating the device there fixes every dataset the harness builds |
| `Basis.__call__` moves the coefficients to the points' device | sampling happens wherever the grid is, so placing the grid is enough |
| DeepONet moves its inputs to the host itself (`.detach().cpu().numpy()`) | sensor values on an accelerator are safe for the baseline |
| `spatial_field_at` calls `.cpu()` explicitly | nothing else pins `burgers-numerical`'s output to the host |

## Goals / Non-Goals

**Goals:**

- A case's values and their devices depend on the seed and the stated device
  alone, never on the process default.
- Existing callers of `sensor_grid` keep working unchanged.

**Non-Goals:**

- DeepONet's own placement.
- The device asymmetry between targets and predictions, recorded by
  `score-time-dependent-outputs`.
- `inverse`'s time-awareness.

## Decisions

### `sensor_grid` states a device; `from_fields` states the field's

`sensor_grid(domain, n_per_axis, device: DeviceInput = None)` builds its grid on
`resolve_device(device)`. Leaving the device unstated keeps today's behaviour,
which `device-selection` requires of a factory that materialises a caller's
data. `from_fields` passes `input_field.coeff.device`, because the grid it builds
is derived for that field. That is the case the requirement covers, and the
process default has no say in it.

*Alternative rejected:* making `device` required. It would give the same
placement inside the library, but it breaks every external caller of a public
function, for no gain the keyword does not already give.

### The slice stays where its field is

`spatial_field_at` drops `.cpu()`. The slice is data derived from a placed
field, so it keeps that field's device.

### The state machine regains the process-default rule

`BenchmarkCasesStayPut` regains `point_the_process_default_elsewhere(device)`
and compares devices as well as values. Two builds of one case count as the
same only when every tensor matches bit for bit and sits on the same device.
It was dropped in `burgers-forced-benchmark` because it failed on exactly this
defect, and this change is what lets it pass.

### A dataset is placed on one device, or refused

`from_fields` refuses an input field and an output field on different devices,
naming both. `OperatorDataset` refuses sensors and sampled values on different
devices. Measured by review, a mixed dataset built without complaint: targets
came back on the host and predictions on the accelerator, and the benchmark
compared them without surfacing the split. A refusal at construction stops the
split at the point where it is made.

### `sensor_grid` takes its device by keyword, and refuses an empty grid

`device` is keyword-only, so a positional argument cannot set it by accident.
The change is still additive, because the parameter is new. A count below one,
or a domain with no axes, is refused with a `ValueError`. It used to return an
empty grid, or fail deep inside torch.

### The state machine repoints and rebuilds in one rule

A rule that repoints the process default and rebuilds a case already built
reaches the property on every use. Measured by review: with separate rules,
about 1 example in 11 reached build, repoint, rebuild, and the old defect was
caught in 8 of 10 seeds.

## Risks / Trade-offs

- **Reported cpu rows move.** A `--device cpu` row built after the first
  DeepONet fit had its sensor values evaluated on cuda. → That is the defect,
  and the move is deliberate. The table is compared against a base worktree
  with its own environment on the same `--device`. Every row whose sensor values
  were already on the host must match within its measured run-to-run spread,
  and every moved row is recorded with its size. *Corrected:* this used to say
  "bit-identical", but `antiderivative` SpectralSVR differs by 1.36e-5 relative
  between runs of identical code. The first DeepONet fit of a run is compared against
  its measured spread.
- **Sensor evaluation now runs on the accelerator under `--device cuda`.** Its
  values change by rounding. → No cuda table is reported yet, and the change is
  measured and recorded.
- **An external caller that relied on the grid following the process default**
  keeps that behaviour by stating nothing.

## Migration Plan

1. `sensor_grid` gains `device`; `from_fields` states the field's device.
2. `spatial_field_at` keeps the slice's device.
3. The state machine regains its rule.

Rollback is a revert of all three.
