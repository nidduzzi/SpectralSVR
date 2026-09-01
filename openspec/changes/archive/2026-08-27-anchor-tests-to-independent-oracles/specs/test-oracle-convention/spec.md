## ADDED Requirements

### Requirement: Every test declares its oracle class
Every test in the suite SHALL carry exactly one oracle-class marker from the set `oracle_independent`, `oracle_analytic`, `oracle_invariant`, `oracle_consistency`, `oracle_smoke`, registered in `tests/conftest.py`.

#### Scenario: Unmarked test fails at setup
- **WHEN** a test with no oracle-class marker is run
- **THEN** the test errors at setup with a message naming the test and listing the valid oracle markers

#### Scenario: Multiply-marked test fails at setup
- **WHEN** a test carries more than one oracle-class marker
- **THEN** the test errors at setup identifying the conflicting markers

#### Scenario: Correctly marked test runs
- **WHEN** a test carries exactly one oracle-class marker (alongside the existing fuzz markers)
- **THEN** the test runs normally

### Requirement: MMS markers retired in favor of oracle classes
The `mms`/`no_mms` markers and their setup-time enforcement SHALL be removed; `oracle_analytic` subsumes the MMS declaration.

#### Scenario: Former MMS test
- **WHEN** a test previously marked `mms` is migrated
- **THEN** it carries `oracle_analytic` (or the stronger class the audit assigned) and no `mms`/`no_mms` marker remains in the suite

#### Scenario: MMS enforcement removed
- **WHEN** the suite runs after migration
- **THEN** no setup check requires `mms`/`no_mms`, and the fuzz check (`@given` or `no_fuzz`) still applies unchanged

### Requirement: Per-module anchor coverage guard
The suite SHALL contain a meta-test asserting that every non-exempt test module includes at least one test marked `oracle_independent`, `oracle_analytic`, or `oracle_invariant`.

#### Scenario: Module with only consistency and smoke tests
- **WHEN** a non-exempt test module contains only `oracle_consistency`/`oracle_smoke` tests
- **THEN** the meta-test fails naming that module

#### Scenario: Exempt module
- **WHEN** a module is on the explicit exempt list (pure plumbing, e.g. config tests)
- **THEN** the meta-test passes regardless of that module's oracle classes

### Requirement: Existing tests are classified
All existing tests SHALL be marked with the oracle class from the 2026-07 audit, marking borderline cases conservatively (as `oracle_consistency` rather than a stronger class).

#### Scenario: Round-trip test classification
- **WHEN** a test compares output of the code under test against other output of the same codebase (e.g. transform/inv_transform round-trip)
- **THEN** it is marked `oracle_consistency`

#### Scenario: External-reference test classification
- **WHEN** a test compares against an independent implementation (e.g. `torch.fft`) or a closed form computed in the test
- **THEN** it is marked `oracle_independent` or `oracle_analytic` respectively
