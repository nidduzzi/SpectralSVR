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

- [x] 2.1 Cover `Sensors are sampled where their field is`. For a field placed on each device under a process default pointing at each device, `OperatorDataset.from_fields` puts the grid and the sampled values on the field's device. Enumerate the field's device and the process default over every device present, and fuzz the dimension, the sensor count and the field. Skip the accelerator pairs, with a reason, where none exists. Verify it fails first
  - `test_sensors_are_sampled_where_their_field_is` in `tests/operators/test_operators.py`, `0e0f483`. It enumerates the field's device against the process default over all four pairs, the accelerator pairs skipping with a reason where none exists, and fuzzes dimension, sensor count and field. It failed first on the two mismatched pairs (`assert 'cpu' == 'cuda'` and `'cuda' == 'cpu'`).
- [x] 2.2 Extend `test_sensor_grid_shape`: a stated device places the grid there, an unstated one follows the process default, and the grid's values are the same on every device. Fuzz as it does today, with the stated device enumerated. Verify the stated-device part fails first
  - `test_sensor_grid_shape` enumerates the stated device (unstated, host, accelerator) and requires the grid on it, following the process default when unstated, with the same values on every device. `linspace` was measured identical on cpu and cuda in 252 of 252 grids, so that is asserted with `torch.equal`. It failed first with `unexpected keyword argument 'device'`, and its prose comment was removed.
- [x] 2.3 `sensor_grid(domain, n_per_axis, device: DeviceInput = None)` builds on `resolve_device(device)`, and `from_fields` passes `input_field.coeff.device`. Verify 2.1 and 2.2 pass and every existing operator test passes unchanged
  - `sensor_grid(domain, n_per_axis, device: DeviceInput = None)` builds on `resolve_device(device)`, and `from_fields` states `input_field.coeff.device`. All 42 operator tests pass, the existing ones unchanged. The GPU driver went missing between writing and committing this section. `just check` was rerun once it was restored, with every accelerator test running: 388 passed.

## 3. The slice

- [x] 3.1 Cover `A slice keeps its field's device`: `spatial_field_at` on a time-dependent field placed on each device gives a slice on that device, with the values of that time slice. Fuzz over slice index, mode count and device, pinning the first and last slice. Verify it fails first on the accelerator
  - `test_a_slice_keeps_its_field_s_device`, `bb70379`. The field's device and which slice (first, middle, last) are both closed sets, so both are parametrized. Slice and mode counts are fuzzed, pinning the smallest and largest. With `.cpu()` restored, the three accelerator cases fail and the three host cases pass.
- [x] 3.2 Drop `.cpu()` from `spatial_field_at`. Verify 3.1 passes
  - `.cpu()` dropped from `spatial_field_at`. All six cases pass.

## 4. Cases do not depend on the default

- [x] 4.1 Cover `A case does not depend on the process default`. Restore `point_the_process_default_elsewhere(device)` in `BenchmarkCasesStayPut`, and make `the_same_case` require every tensor to match bit for bit and sit on the same device. Verify it fails against the code before sections 2 and 3, and passes after them
  - `d5a3ede`. `point_the_process_default_elsewhere` is restored, and `the_same_case` requires each tensor pair to share a device and match bit for bit. The machine also carries the scenario mark. Against the dataset and script from `116b157` it fails with the sequence build (`antiderivative`, cpu, seed 0), repoint the default, rebuild; against the current code it passes.
- [x] 4.2 Measure what sections 2 and 3 change in each registered case under a cuda default: for every case, the largest gap in sensor values between the old placement and the new, at seed 0 in quick and full mode. Record the figures
  - Measured against a worktree of `9d7cc97` with its own environment. Every case was built with `cpu` stated under a cuda process default, at seed 0. The base put every case's sensor values on `cuda:0`; the new code puts them on `cpu`. Worst gap, quick and full: `antiderivative` 7.63e-6 and 7.63e-6; `burgers-mms` 1.19e-7 and 2.38e-7; `burgers-numerical` 2.38e-7 and 3.58e-7; `antiderivative-grf` 4.77e-7 and 4.77e-7; `burgers-forced` 4.47e-8 and 2.98e-8.

## 5. Verification

- [x] 5.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
  - `just check`: ruff, mypy and ty are clean, and 394 passed with the accelerator present.
- [x] 5.2 Build a base from a detached worktree of this branch's parent with its own environment, and print the signature of the library each side loads. Run the full table on both with `--device cpu` and the same seed, twice on the new side. Every row whose sensor values were already on the host must be bit-identical. Record every row that moves, with its size, and compare the first DeepONet fit of a run against its measured spread
  - The base came from a worktree of `9d7cc97` with its own `just sync`. The base's `sensor_grid` has no `device`, and the new one has. The full table ran with both operators, `--device cpu` and seed 0, once on the base and twice on the new side. The new side is bit-reproducible across its two runs on every row except the first DeepONet fit (0.0187 relative), which is known not to reproduce. `antiderivative` is built before DeepONet imports `deepxde`, so its sensor values were already on the host. SpectralSVR differs from the base by 1.36e-5 relative, exactly the run-to-run spread measured by `score-time-dependent-outputs`. DeepONet is the first fit of the run: r2 0.322 on the base, 0.471 and 0.480 on the new side, within the measured spread of 0.159 to 0.797. Every later dataset moved, as predicted, because the base evaluated its sensor values on cuda after `deepxde` switched the default. SpectralSVR, relative: MMS 1.35e-6, numerical 9.23e-7, GRF 6.48e-6, forced 8.08e-7, with r2 unchanged to four places. DeepONet, r2: MMS -4.087 to -6.120, numerical -1.519 to -1.667, GRF 0.9991 to 0.9991, forced 0.742 to 0.801. **Finding for `benchmark-reproducibility`:** DeepONet's reported numbers are extremely sensitive to rounding-level changes in its inputs. Sensor values shifted by 1e-7 to 1e-6 moved the MMS r2 by about 2, so a DeepONet row needs a spread over seeds, not a single run.
- [x] 5.3 Run `just mutate-changed` with the operators and script configs against this branch's parent. Read every survivor, close each real gap with a test, and confirm each equivalent survivor by re-running it
  - `just mutate-changed` against `9d7cc97`: the operators config covered 19 mutants in scope, and all 19 were killed. The script config had 0 in scope: its only change deletes `.cpu()`, which leaves nothing to mutate.
- [x] 5.4 Verify no source file gained an explanatory prose comment, and that prose was removed where touched. Make a second structural pass: no throwaway helpers or private methods
  - No comment was added, and the `half-open` prose comment in `test_sensor_grid_shape` was removed. The only addition to a docstring is the `device` entry in `sensor_grid`'s parameter list, a contract in the format of its siblings. Second pass: the library change is one parameter and one call that states it, the script change is a deletion, the test's comparison is one `all` over paired tensors, and there are no helpers or private methods.
