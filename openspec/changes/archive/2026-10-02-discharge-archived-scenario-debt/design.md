## Context

See proposal.md - Why. `tests/test_scenario_coverage.py` already separates where a scenario is stated, the main specs plus the deltas of changes outside `openspec/changes/archive/`, from where it is covered, a `scenario` mark in `tests/`. The new check needs one more distinction: which of those sources is a change in flight.

## Goals / Non-Goals

**Goals:**

- Each of the three stranded scenarios is checked on every run.
- An `IN_FLIGHT` reason is true by construction: the guard refuses it wherever no change in flight states the scenario.

**Non-Goals:**

- Revisiting the `PREDATES`, `DECISION` and `POLICY` entries. They do not depend on a change being in flight.
- Changing the notebook, the script or the `bench` recipe. The tests check them as they are.

## Decisions

### Stated-by-active-change is read from the same files the guard already reads

The guard's spec files split into main specs and active deltas. The check collects the scenarios the active deltas state, and refuses every `EXEMPT` entry whose reason is `IN_FLIGHT` and whose scenario is outside that set. Keying on the reason constant, not on its text, keeps the rule from depending on wording.

### The notebook is executed, not grepped

A test that only searched the notebook for the record's path would pass a notebook that also fitted an operator. The test executes the code cells in order, from a temporary directory holding a fixture record, with plotting on a non-interactive backend. It then requires that no operator name or `benchmark` is bound in the cells' namespace, and that the summary has one row per dataset and operator in the fixture. The fixture is a few rows in the record's schema; it is not the committed record, so the test never depends on a regeneration.

### The recipe is read from the Justfile

`just --dry-run bench` prints the command the recipe runs without running it. The test requires that the command runs the benchmark script into the committed record's path, with the accelerator stated and no positional argument the caller must supply. That is checkable without a GPU or a regeneration.

## Risks / Trade-offs

- **Executing the notebook couples a test to its cells.** → That is the scenario's claim: what the notebook shows is what was recorded. A notebook edit that breaks the test has broken the claim.
- **`just` must be installed for the recipe test.** → The suite already runs under `just check`, so the test requires what the gate requires. It skips with a stated reason where `just` is absent, rather than passing.
