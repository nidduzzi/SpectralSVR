## Execution order

```
1 the case ── 2 the row ── 3 tests ── 4 verification
```

Group 2 cannot run before group 1 builds, and group 4's base comparison is only
meaningful once the row exists. Nothing depends on a later group.

## 1. The forced case

- [ ] 1.1 Verify the dependency this rests on: `Burgers.generate` accepts a Basis forcing that is spectral in time, and rejects one that is per-time-slice. `Burgers._forcing_at` in `SpectralSVR/problems/burgers.py` is the accepting path — confirm it is present on this branch before building anything against it, since without it the dataset cannot exist
- [ ] 1.2 Add `burgers_forced_case(seed, quick, device, *, n_train=None, n_test=None, modes=None)` to `scripts/run_benchmark.py` beside `burgers_mms_case`. It still satisfies `CaseBuilder`, and each size left unstated takes the value `quick` selects, so the registry is unaffected while tests can state any size: `u0 = 0`, forcing from `FourierBasis.generate(n, (mt, ms), random_func=grf_coeff(...), domain=((0, 1), (0, 1)), device=device)` with the length scales written as named constants, a generator built with `torch.Generator(device=device)`, and the solution from `Burgers().generate(..., device=device)` kept `time_dependent=True`. Verify it builds in `--quick` mode without raising, on the host and on the accelerator
- [ ] 1.3 Build the split with `Split.partition(f, u, n_train, n_sensors=...)` and return `BenchmarkCase(split, query)`, matching `burgers_mms_case`. Verify against the harness rather than against the inherited design note, which named `OperatorDataset.from_fields` — an API two changes below replaced
- [ ] 1.4 Verify `Split.partition` accepts a 2D-domain input field: the sensor grid is a tensor product over `(time, space)`, so `train.f` must be `(n, n_sensors ** 2)` and `train.sensors` must be `(n_sensors ** 2, 2)`. If it does not, stop and report rather than reshaping around it

## 2. The row

- [ ] 2.1 Register `DatasetSpec("project Burgers forced (f->u, GRF forcing, from rest)", burgers_forced_case)` under key `burgers-forced`; verify `--help` lists the key and the label reaches the CSV's `dataset` column
- [ ] 2.2 Measure how far `OperatorDataset.targets(query)` on the forced dataset is from direct evaluation of the stored coefficients at output-grid `(t, x)` points, over several seeds, and record the worst gap here. Set the assertion's bound from that measurement, beneath nothing it has not seen. This row is the first to evaluate a time-dependent output field at 2D query points; the existing `u0 -> u(T)` case sliced to spatial fields and never exercised it. If the measurement shows time-interpolation error at off-grid times, restrict the query grid to output-grid times and record why in the `DatasetSpec` docstring rather than setting a bound loose enough to absorb it
- [ ] 2.3 Run `--datasets burgers-forced --operators spectralsvr --quick` and verify it writes one row with finite metrics and `pred_nan_sum == 0`. Record the row in this file

## 3. Tests

The rule for this section is to fuzz rather than pin, and to model state rather
than check a single case. Edges are pinned as `@example`s on the fuzzed tests
rather than written as separate single-case tests. A closed set, the dataset
registry, is enumerated exhaustively, because enumerating it covers every case
and sampling it covers only some. Every property is shown to fail under a
mutation of what it describes, and each mutation is recorded. Each forced build
runs a numerical solve, so time the tests before settling `max_examples` and
`stateful_step_count`, and record the timings.

- [ ] 3.1 `BenchmarkCasesStayPut`, a `RuleBasedStateMachine` marked `oracle_invariant`, covers `Adding a row leaves the others alone`. It has three rules. `build(key, seed, device)` builds any registered case and compares it with the first build recorded for that `(key, seed, device)`. `point_the_process_default_elsewhere(device)` stands in for `deepxde` resetting the default mid-run. `build_the_forced_case(seed, device)` builds the new case in between. The invariant is that every rebuild is bit-identical to its first build, in splits and in query. Verify it fails when one builder draws from the process-default generator instead of its own
- [ ] 3.2 Analytic anchor, `oracle_analytic`: from rest, the forced case's output at `t = 0` is identically zero for any drawn forcing. Fuzz over seed, sizes and device. Verify it fails when `u0` is made nonzero
- [ ] 3.3 Analytic anchor, `oracle_analytic`: the solve the builder uses gives `u ≡ 0` at every output time when the forcing is zero and the start is rest. Fuzz over sizes and device, and pin the full-mode sizes as an `@example`. Verify it fails when a constant forcing replaces the zero one
- [ ] 3.4 Fuzz the forced case over `seed × (n_train, n_test, modes) × device`. Check the shapes: `train.f` is `(n_train, s²)` and `train.sensors` is `(s², 2)`. Check that the values are finite, that train and test are disjoint, and that the drawn forcing is real-valued, meaning conjugate symmetry holds. Pin as `@example`s: `seed=0`, `seed=2**32 - 1`, the smallest sizes the harness accepts, and the `quick` and full defaults. Verify the shape assertion fails under a transposed sensor grid
- [ ] 3.5 Fuzz reproducibility over seed and device: the same seed gives a bit-identical case, and two different seeds give different forcings. Verify it fails when the generator is built without the seed
- [ ] 3.6 Fuzz the device boundary. On the host and on the accelerator, every stage of the case is on the stated device. A generator placed on the other device is refused, and the refusal names both devices. Skip the accelerator case with a reason where none exists. Verify it fails when `device=` is dropped from one `generate` call
- [ ] 3.7 Turn 2.2's measurement into a fuzzed `targets(query)` check over seeds and output-grid points, asserting the bound 2.2 recorded. Pin `t = 0`, `t = T`, `x = 0` and `x = L` as `@example`s, the last two for the periodic wrap
- [ ] 3.8 Fuzz scoring over seeds, which covers `Forced dataset builds and scores`: a quick SpectralSVR fit on the case gives finite metrics and `pred_nan_sum == 0`. Verify it fails when the builder raises
- [ ] 3.9 Enumerate the registry exhaustively, deliberately `no_fuzz`, which covers `Forced dataset is registered`: every key and label is unique, and `burgers-forced` is present. Verify it fails when the entry is removed
- [ ] 3.10 Enumerate every forcing-to-solution entry, deliberately `no_fuzz`, which covers `The derived and drawn forcings are distinguishable`: each label names how its forcing was obtained, and no two collide. Verify it fails when the forced row is relabelled to collide with the MMS one
- [ ] 3.11 Verify every scenario of this change is named by a test and its `IN_FLIGHT` entry is removed, with `tests/test_scenario_coverage.py` forcing each removal, and that `tests/test_oracle_coverage.py` passes for the new module

## 4. Verification

- [ ] 4.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [ ] 4.2 Verify the four existing rows are unchanged. Build a detached worktree at this branch's parent with its own environment — a `PYTHONPATH` override loses to the editable install, and `explicit-band-type` lost a base run that way — print the signature of the library each side loads, run the benchmark on both with the same stated `--device`, and record it. Byte-identity is the expectation since no library source changes here. Read any difference against this stack's measured noise floor of `2.512e-05`, which was measured on the host, before calling it real
- [ ] 4.3 Verify the fifth row is reported as new rather than as a moved number: record it in this file with its seed, the GRF length scales and the `--device` it was built on, so the next regeneration can be compared against something
- [ ] 4.4 Verify no source file gained an explanatory prose comment; the dataset's provenance belongs in the `DatasetSpec` label and the builder's docstring
