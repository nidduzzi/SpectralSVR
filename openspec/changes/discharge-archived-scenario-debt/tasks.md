## 1. Close the hole first

- [x] 1.1 Add the check that every `IN_FLIGHT` entry names a scenario stated by a change in flight; verify it fails now, naming exactly the three entries "One command regenerates", "The notebook reads the record" and "The reference protocol is followed and recorded", and passes once they are removed from `EXEMPT` without being covered -- which the coverage check then refuses, so the two checks hand the gap to each other rather than both passing
  - `da3d024`. The check failed first naming exactly the three entries. With them removed and no test added, it passed while `test_every_stated_scenario_is_covered_or_exempt` failed, so the two checks hand the gap to each other.

## 2. The three tests

- [x] 2.1 Name "The reference protocol is followed and recorded" on the recipe-transcription test and on `BenchmarkRecord`, whose pinned DeepONet run asserts a row's protocol string; verify by removing the recipe's source name from the protocol string and confirming a named test fails
  - `da3d024`. `test_recipes_transcribe_their_reference_code` and `BenchmarkRecord` carry the scenario. With the source dropped from `DeepONetOperator.protocol`, `BenchmarkRecord` failed; restored.
- [x] 2.2 Test the `bench` recipe through `just --dry-run bench`: the command runs the benchmark script into `benchmarks/operator_comparison.csv` with `--device cuda` stated and needs no positional argument; skip with a stated reason where `just` is absent. Name "One command regenerates". Verify the test fails against a recipe edited to drop the output path
  - `da3d024`. `test_one_recipe_regenerates_the_record_with_every_default_stated` reads `just --dry-run bench`: the script into `benchmarks/operator_comparison.csv`, `--device cuda`, no `--seeds` or `--datasets`, and `SEED_SET` is 0 to 4. With the output path dropped from the recipe it failed; restored.
- [x] 2.3 Test the notebook by executing its code cells in order from a temporary directory holding a fixture record in the record's schema: the summary has one row per dataset and operator in the fixture, and no operator name or `benchmark` is bound in the cells' namespace. Name "The notebook reads the record". Verify the test fails against a notebook with a cell that imports `DeepONetOperator`
  - `da3d024`. `test_the_notebook_presents_the_record_and_fits_nothing` writes a record with the script (SpectralSVR, host, seeds 0 and 1, antiderivative) into a temporary directory and executes the notebook's code cells there: one dataset and operator, two seeds, no fitting name bound. A cell importing `DeepONetOperator` failed it; restored.
- [x] 2.4 Remove the three entries from `EXEMPT`; verify the coverage, staleness and new in-flight checks all pass
  - `da3d024`. The three entries are gone; the coverage, staleness and in-flight checks pass, four of four in the module.

## 3. Verification

- [x] 3.1 Run `just check`; verify lint is clean and no failure beyond the two pre-existing ones in `test_time_varying_scoring.py`, which `condition-the-interpolating-fixture` fixes
  - `just check`: ruff, mypy and ty clean; 391 passed, 2 failed, the pre-existing failures in `test_time_varying_scoring.py`.
- [x] 3.2 Verify no source file gained an explanatory prose comment, and that no library, script, notebook or record file changed
  - `git diff 81c8441 --stat -- SpectralSVR scripts notebooks benchmarks Justfile` is empty. The test changes add no explanatory comment; the touched helpers' docstrings were removed in favour of their names.
