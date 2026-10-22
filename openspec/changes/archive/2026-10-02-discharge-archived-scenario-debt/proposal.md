## Why

`benchmark-reproducibility`, archived at `81c8441`, left three of its scenarios with no test naming them: "One command regenerates", "The notebook reads the record" and "The reference protocol is followed and recorded". They still sit in `tests/test_scenario_coverage.py`'s `EXEMPT` with the reason `IN_FLIGHT`, "stated by a change still being implemented". That reason has been false since the archive. The behaviour each scenario describes was built and verified by hand during that change (tasks 6.1, 6.2 and 2.8), but nothing checks it on every run, so the archive recorded as done a contract the suite does not watch.

The guard could not see this. `test_no_exemption_is_stale` refuses an entry whose scenario no spec states any more, or one a test now names. An `IN_FLIGHT` entry whose change was archived without its test is neither, so it passes. The same hole will open again at the next archive that misses a scenario.

## What Changes

- **Three tests, and the three entries removed:**
  - "One command regenerates": the `bench` recipe regenerates the record with every default stated, so a reader needs no remembered argument.
  - "The notebook reads the record": the comparison notebook reads the committed record and fits nothing. Its code is executed against a fixture record, and it imports no operator.
  - "The reference protocol is followed and recorded": the existing tests that pin each recipe to its reference code, and the record machine's assertion of a row's protocol, are named for the scenario they already check.
- **An `IN_FLIGHT` exemption must name a scenario stated by a change still in flight.** A scenario stated only by the main specs or by an archived change cannot carry the reason, so the guard refuses the entry at the first run after the archive that strands it. The new check fails today on the three entries, so it lands with their tests.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `test-oracle-convention`: "Every scenario a change states is verifiable" gains a scenario that an in-flight exemption ends with its change.

## Impact

- `tests/test_scenario_coverage.py`: the new check; three entries removed.
- `tests/operators/`: a test for the notebook, a test for the `bench` recipe, and scenario names on existing recipe and record tests.
- No library, script, notebook or recorded result changes.
- Sits directly on `benchmark-reproducibility`'s archive, beneath `ratchet-code-prose`, which is rebased onto it.
