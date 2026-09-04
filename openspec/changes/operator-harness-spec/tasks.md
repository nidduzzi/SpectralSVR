## 1. Confirm the spec matches the code before writing it down

- [ ] 1.1 Verify `benchmark()` scores every operator against one shared `target` tensor computed once from `test.targets(query)`, and that `fit_seconds` brackets only the `fit` call; both are visible at `SpectralSVR/operators/benchmark.py` around lines 47-55
- [ ] 1.2 Verify `OperatorDataset.from_fields` samples inputs at a sensor grid and keeps outputs as an evaluable field rather than pre-sampled values, so `targets(x)` is evaluated at whatever query points are asked for
- [ ] 1.3 Verify the `Operator` interface requires `fit` before `predict` and that at least one adapter raises when that order is violated; if no adapter enforces it, record the gap rather than specifying a guarantee the code does not make

## 2. Write the capability

- [ ] 2.1 Confirm every requirement in `specs/operator-comparison/spec.md` describes behaviour the code already has, amending any that overstates it; verify each scenario names something checkable against the current implementation
- [ ] 2.2 Verify the requirement about failing operators matches reality: if `benchmark()` currently lets an exception propagate, say so; if it would silently omit a row, record that as a defect for a follow-up rather than specifying the behaviour we wish it had

## 3. Docstrings that understate their guarantees

- [ ] 3.1 Bring the `benchmark()`, `OperatorDataset` and `Operator` docstrings into line with the capability, adding only what the code already does; verify no signature and no behaviour changes
- [ ] 3.2 State in the benchmark's documentation that the harness fixes the data and the metrics but not the tuning, and that figures come from a single seed, so the table is a demonstration rather than a controlled comparison; verify the wording does not promise the multi-seed and provenance work deferred to the reproducibility change

## 4. The anchor-policy exemption

- [ ] 4.1 Decide, and record in the change, whether the `operators/test_operators.py` exemption in `tests/test_oracle_coverage.py` is re-justified against the package's current contents or withdrawn; verify the decision names what is actually exempt if kept
- [ ] 4.2 If withdrawn, record the required anchor as outstanding work in a follow-up rather than leaving `test_oracle_coverage.py` failing; verify the suite is green either way at the end of this change

## 5. Verification

- [ ] 5.1 Run `just check`; verify clean and that the test count is unchanged, since this change adds no tests and alters no behaviour
- [ ] 5.2 Verify no number moved: regenerate the benchmark and confirm all four rows and the seeded numerical Burgers field are identical to the committed values
- [ ] 5.3 Verify no source file gained an explanatory prose comment; the additions here belong in docstrings and in the capability, not inline
