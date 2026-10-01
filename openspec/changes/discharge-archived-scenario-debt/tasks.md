## 1. Close the hole first

- [ ] 1.1 Add the check that every `IN_FLIGHT` entry names a scenario stated by a change in flight; verify it fails now, naming exactly the three entries "One command regenerates", "The notebook reads the record" and "The reference protocol is followed and recorded", and passes once they are removed from `EXEMPT` without being covered -- which the coverage check then refuses, so the two checks hand the gap to each other rather than both passing

## 2. The three tests

- [ ] 2.1 Name "The reference protocol is followed and recorded" on the recipe-transcription test and on `BenchmarkRecord`, whose pinned DeepONet run asserts a row's protocol string; verify by removing the recipe's source name from the protocol string and confirming a named test fails
- [ ] 2.2 Test the `bench` recipe through `just --dry-run bench`: the command runs the benchmark script into `benchmarks/operator_comparison.csv` with `--device cuda` stated and needs no positional argument; skip with a stated reason where `just` is absent. Name "One command regenerates". Verify the test fails against a recipe edited to drop the output path
- [ ] 2.3 Test the notebook by executing its code cells in order from a temporary directory holding a fixture record in the record's schema: the summary has one row per dataset and operator in the fixture, and no operator name or `benchmark` is bound in the cells' namespace. Name "The notebook reads the record". Verify the test fails against a notebook with a cell that imports `DeepONetOperator`
- [ ] 2.4 Remove the three entries from `EXEMPT`; verify the coverage, staleness and new in-flight checks all pass

## 3. Verification

- [ ] 3.1 Run `just check`; verify lint is clean and no failure beyond the two pre-existing ones in `test_time_varying_scoring.py`, which `condition-the-interpolating-fixture` fixes
- [ ] 3.2 Verify no source file gained an explanatory prose comment, and that no library, script, notebook or record file changed
