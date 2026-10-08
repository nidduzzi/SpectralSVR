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

## After the rebase onto fix-time-dependent-resampling

*This change was paused while `fix-time-dependent-resampling` landed beneath it, then rebased onto its archive, `4238a5e`.*

**The baseline is re-recorded on the new base.** The first recording (424 passed in 90 s, 187 single tests) predates the fix, which grew the suite to 441, deleted two single tests and added fuzzed and parametrized ones. The inventory is regenerated against the new base rather than edited by hand, and the per-phase mutants are rerun there.

**`BasisLifecycle` absorbs the fix's refusal tests.** The fix left four tests that check conversions refusing what they cannot place: the slice-count refusal, the two field refusals, and the own-slice-count test. A conversion refusing is a transition of the same lifecycle, so they become the machine's rules and carry their scenario names. Their `@example`s become initialisation pins: integer slice counts of every accepted type, booleans and arrays of more than zero dimensions, an empty basis in both directions, a one-slice time-dependent field in both directions, one without a spatial axis, a wrong-length per-axis sampling, and shared storage on a no-op conversion. The fix's analytic and independent value tests stay fuzzed single tests: their oracles hold per draw, not across a sequence.

## Mutation testing by cosmic-ray

*Decided by the user during phase 1.* Hand-picked mutants test what their author expected to matter, so a phase is now verified by cosmic-ray, which enumerates every mutant of the modules the phase's tests cover. Each phase runs cosmic-ray twice: on a detached worktree of the base with the old tests, and on a worktree of the phase's result. Each worktree has its own environment, and the main tree stays free while the runs proceed. The phase passes when every mutant killed before is killed after; survivors after must be a subset of survivors before. The user chose the serial local distributor over parallel http workers. Phase 1 covers `__base.py` (1132 mutants), `FourierBasis.py` (1652), `sampling.py` (96) and `_plot.py` (432) per tree, against `tests/basis` with Hypothesis seed 0. The hand mutants recorded earlier stay in `measurements/` as a record, superseded.

The runs live in `~/.cache/spectralsvr-mutation/`, as the user chose after a reboot cleared the first attempt under `/tmp`: a worktree per tree with its own environment, one `.sqlite` session per tree and module, and `measurements/cosmic-ray/run_phase.sh` running them as a `systemd-run --user` unit capped at 16 GiB without swap. The unit outlives the Claude session, and the runner resumes after a reboot, because it initialises a session only once and `cosmic-ray exec` skips mutants already recorded. The test command carries no memory scope of its own, since the unit's cap covers it.

