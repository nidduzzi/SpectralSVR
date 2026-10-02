## Context

See proposal.md - Why. `inventory.md` holds one row per single test: its oracle marker, the scenarios it names, what it checks, and its classification. The test-oracle convention, the scenario guard, the oracle-coverage guard and the prose budget all hold during the work, and the suite is green at the start.

## Goals / Non-Goals

**Goals:**

- Every behaviour check that ranges over inputs is fuzzed, every one that concerns a sequence is a state machine, and edge cases and happy paths are pinned in both.
- Nothing the suite checks today is lost: no oracle marker weakens, no scenario name disappears, no kill rate regresses.

**Non-Goals:**

- Changing library code or any tolerance's value.
- Converting the 35 tests that check a fixed fact of the repository or a record.
- Widening a range whose bound has not been measured over the wider range.

## Decisions

### One phase per machine or module, BasisLifecycle first

`BasisLifecycle` absorbs the most tests and the module with the most single cases, so it goes first and sets the pattern for the rest. Each later phase is one machine or one module, committed green on its own, so a regression is found where it was introduced.

### What a phase must prove before it lands

- **No oracle marker weakens.** A test's class moves with it; a folded test's anchor becomes an anchored rule or invariant, never a consistency check.
- **No scenario name is lost.** Every scenario a removed test named is named by the test that replaces it, in the same commit, and the scenario guard passes.
- **No recorded number moves.** Library code is untouched, and a diff argument suffices.
- **Kill rate does not regress.** Before a phase, a set of hand mutants of the code its tests cover is run against the old tests; after it, every mutant the old tests killed is killed by the new ones.

### Literals become pinned examples, never discarded

A literal from a single test becomes an `@example` on the fuzzed test, or an `@initialize` rule that calls the machine's rules, since state machines take no examples. A closed set is enumerated with `parametrize` or inside the initialisation. A regression literal stays verbatim.

### Analytic anchors keep their literal and gain an independent fuzz

Decided by the user. The three hand-computed anchors hold only at their literal, so fuzzing them would change their oracle class. Each stays as a pinned `oracle_analytic` case, and a separate fuzzed test checks the same function against an independent implementation written in the test.

### Measured bounds stay verbatim; a wider range is measured first

29 tests carry a measured bound or regression constant (`RESIZE_TOLERANCE`, `CLOSED_FORM_TOLERANCE`, `MANUFACTURED_RESIDUAL_TOLERANCE`, the round-trip and frozen-forcing constants, the ETDRK4 order, the dense-bytes ratio, the smape tolerances among them). A fuzzed version keeps the bound and draws only from the range it was measured over; where a phase wants a wider range, it measures the bound over that range first and records the measurement in this change.

### Legacy tests in exempt modules leave with their exemption

Two legacy tests sit in modules the oracle-coverage guard exempts (`basis/test_basis.py`, `model/test_SpectralSVR.py`). Where removing them empties a module, the module and its exemption go together, so the guard's staleness check passes.

## Risks / Trade-offs

- **A machine with too many rules explores poorly.** → A machine is split where its rules stop sharing state, and each records its example and step counts.
- **Folding changes failure messages.** A single test named what failed; a machine names a rule sequence. → Each rule keeps a descriptive name, and the pinned initialisation reproduces the old literal case first.
- **Suite time grows with examples.** No single test runs over 0.67 s today. → Each phase records the suite's time before and after, and examples are sized to keep the gate near its current 90 s.
