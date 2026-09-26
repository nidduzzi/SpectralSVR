## Execution order

```
1 the case ── 2 the row ── 3 scenario coverage ── 4 verification
```

Group 2 cannot run before group 1 builds, and group 4's base comparison is only
meaningful once the row exists. Nothing depends on a later group.

## 1. The forced case

- [ ] 1.1 Verify the dependency this rests on: `Burgers.generate` accepts a Basis forcing that is spectral in time, and rejects one that is per-time-slice. `Burgers._forcing_at` in `SpectralSVR/problems/burgers.py` is the accepting path — confirm it is present on this branch before building anything against it, since without it the dataset cannot exist
- [ ] 1.2 Add `burgers_forced_case(seed, quick, device)` to `scripts/run_benchmark.py` beside `burgers_mms_case`, matching the `CaseBuilder` signature: `u0 = 0`, forcing from `FourierBasis.generate(n, (mt, ms), random_func=grf_coeff(...), domain=((0, 1), (0, 1)), device=device)` with the length scales written as named constants, a generator built with `torch.Generator(device=device)`, and the solution from `Burgers().generate(..., device=device)` kept `time_dependent=True`. Verify it builds in `--quick` mode without raising, on the host and on the accelerator
- [ ] 1.3 Build the split with `Split.partition(f, u, n_train, n_sensors=...)` and return `BenchmarkCase(split, query)`, matching `burgers_mms_case`. Verify against the harness rather than against the inherited design note, which named `OperatorDataset.from_fields` — an API two changes below replaced
- [ ] 1.4 Verify `Split.partition` accepts a 2D-domain input field: the sensor grid is a tensor product over `(time, space)`, so `train.f` must be `(n, n_sensors ** 2)` and `train.sensors` must be `(n_sensors ** 2, 2)`. If it does not, stop and report rather than reshaping around it

## 2. The row

- [ ] 2.1 Register `DatasetSpec("project Burgers forced (f->u, GRF forcing, from rest)", burgers_forced_case)` under key `burgers-forced`; verify `--help` lists the key and the label reaches the CSV's `dataset` column
- [ ] 2.2 Measure how far `OperatorDataset.targets(query)` on the forced dataset is from direct evaluation of the stored coefficients at output-grid `(t, x)` points, over several seeds, and record the worst gap here. Set the assertion's bound from that measurement, beneath nothing it has not seen. This row is the first to evaluate a time-dependent output field at 2D query points; the existing `u0 -> u(T)` case sliced to spatial fields and never exercised it. If the measurement shows time-interpolation error at off-grid times, restrict the query grid to output-grid times and record why in the `DatasetSpec` docstring rather than setting a bound loose enough to absorb it
- [ ] 2.3 Run `--datasets burgers-forced --operators spectralsvr --quick` and verify it writes one row with finite metrics and `pred_nan_sum == 0`. Record the row in this file

## 3. Scenario coverage

- [ ] 3.1 Cover `Forced dataset is registered` with a test naming it via `@pytest.mark.scenario`, asserting the key appears in the script's dataset choices and the label distinguishes it from the MMS row. Verify the test fails when the registration is removed
- [ ] 3.2 Cover `Forced dataset builds and scores` the same way, building the case in quick mode and asserting finite metrics. Verify it fails when the builder raises
- [ ] 3.3 Cover `The derived and drawn forcings are distinguishable` by asserting both labels are present and differ in the way a reader needs. Verify it fails if the forced row is relabelled to collide with the MMS one
- [ ] 3.4 Cover `Adding a row leaves the others alone` with an `oracle_invariant` test: build every existing case, build `burgers-forced`, build every existing case again, and require the splits and queries bit-identical. The base-run comparison in 4.2 is the other half of the scenario and stays a verification, not a test. Verify the test fails when a builder is mutated to draw from shared state, for example the process-default generator. Verify `tests/test_scenario_coverage.py` reports no uncovered scenario for this change and no stale exemption
- [ ] 3.5 Fuzz what ranges over inputs: the case builder over sample counts, mode counts and seeds, asserting shapes and finiteness rather than one hand-picked configuration. Verify each property fails under a mutation of the behaviour it describes

## 4. Verification

- [ ] 4.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [ ] 4.2 Verify the four existing rows are unchanged. Build a detached worktree at this branch's parent with its own environment — a `PYTHONPATH` override loses to the editable install, and `explicit-band-type` lost a base run that way — print the signature of the library each side loads, run the benchmark on both with the same stated `--device`, and record it. Byte-identity is the expectation since no library source changes here. Read any difference against this stack's measured noise floor of `2.512e-05`, which was measured on the host, before calling it real
- [ ] 4.3 Verify the fifth row is reported as new rather than as a moved number: record it in this file with its seed, the GRF length scales and the `--device` it was built on, so the next regeneration can be compared against something
- [ ] 4.4 Verify no source file gained an explanatory prose comment; the dataset's provenance belongs in the `DatasetSpec` label and the builder's docstring
