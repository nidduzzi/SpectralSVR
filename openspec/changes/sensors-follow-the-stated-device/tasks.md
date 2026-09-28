## Execution order

```
1 scenarios as debt ── 2 sensors ── 3 the slice ── 4 cases do not depend on the default ── 5 verification
```

A chain. Every property is written as a test that fails against the code
before the code that satisfies it is written. Fuzz rather than pin: edges are
pinned as `@example`s. Closed sets are enumerated with `parametrize`, not
sampled. Where state carries across calls, model it with a
`RuleBasedStateMachine`. Measure every tolerance before asserting it, and record
the measurement here.

## 1. Scenarios as debt

- [x] 1.1 Add `IN_FLIGHT` entries for the three scenarios the delta adds. Verify `test_every_stated_scenario_is_covered_or_exempt` fails without them and passes with them
  - Without the entries, the guard failed naming the 3 scenarios; with them, 3 passed.

## 2. Sensors

- [ ] 2.1 Cover `Sensors are sampled where their field is`. For a field placed on each device under a process default pointing at each device, `OperatorDataset.from_fields` puts the grid and the sampled values on the field's device. Enumerate the field's device and the process default over every device present, and fuzz the dimension, the sensor count and the field. Skip the accelerator pairs, with a reason, where none exists. Verify it fails first
- [ ] 2.2 Extend `test_sensor_grid_shape`: a stated device places the grid there, an unstated one follows the process default, and the grid's values are the same on every device. Fuzz as it does today, with the stated device enumerated. Verify the stated-device part fails first
- [ ] 2.3 `sensor_grid(domain, n_per_axis, device: DeviceInput = None)` builds on `resolve_device(device)`, and `from_fields` passes `input_field.coeff.device`. Verify 2.1 and 2.2 pass and every existing operator test passes unchanged

## 3. The slice

- [ ] 3.1 Cover `A slice keeps its field's device`: `spatial_field_at` on a time-dependent field placed on each device gives a slice on that device, with the values of that time slice. Fuzz over slice index, mode count and device, pinning the first and last slice. Verify it fails first on the accelerator
- [ ] 3.2 Drop `.cpu()` from `spatial_field_at`. Verify 3.1 passes

## 4. Cases do not depend on the default

- [ ] 4.1 Cover `A case does not depend on the process default`. Restore `point_the_process_default_elsewhere(device)` in `BenchmarkCasesStayPut`, and make `the_same_case` require every tensor to match bit for bit and sit on the same device. Verify it fails against the code before sections 2 and 3, and passes after them
- [ ] 4.2 Measure what sections 2 and 3 change in each registered case under a cuda default: for every case, the largest gap in sensor values between the old placement and the new, at seed 0 in quick and full mode. Record the figures

## 5. Verification

- [ ] 5.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [ ] 5.2 Build a base from a detached worktree of this branch's parent with its own environment, and print the signature of the library each side loads. Run the full table on both with `--device cpu` and the same seed, twice on the new side. Every row whose sensor values were already on the host must be bit-identical. Record every row that moves, with its size, and compare the first DeepONet fit of a run against its measured spread
- [ ] 5.3 Run `just mutate-changed` with the operators and script configs against this branch's parent. Read every survivor, close each real gap with a test, and confirm each equivalent survivor by re-running it
- [ ] 5.4 Verify no source file gained an explanatory prose comment, and that prose was removed where touched. Make a second structural pass: no throwaway helpers or private methods
