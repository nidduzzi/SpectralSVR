## MODIFIED Requirements

### Requirement: Every scenario a change states is verifiable
A scenario SHALL be checkable against the implementation, and a test covering one SHALL declare which scenario it covers so the link is machine-readable rather than inferred. A scenario with no check SHALL carry a recorded reason, per scenario rather than per capability, so that a scenario newly added to an area that was exempt is still caught. Where a scenario ranges over inputs, the check SHALL vary them rather than pin one literal; where it states a documented guarantee, the check SHALL assert that the documentation carries it.

#### Scenario: A scenario without a check is caught
- **WHEN** a change states a scenario that no test exercises and no exemption names
- **THEN** a guard reports it, so the gap is visible before archive rather than discovered by a later reader of the spec

#### Scenario: The guard survives archiving
- **WHEN** a change is archived and its requirements are synced into the main specs
- **THEN** its scenarios remain guarded, because the guard reads the main specs as well as the deltas of changes still in flight — a guard that watched only active changes would stop watching a contract at the moment it became permanent

#### Scenario: An exemption does not cover a later scenario
- **WHEN** a scenario is added to a capability whose existing scenarios are exempt
- **THEN** the new one is reported, because exemptions name individual scenarios rather than whole capabilities

#### Scenario: Scenarios over an input range are fuzzed
- **WHEN** a scenario describes behaviour that holds across a range of sizes, counts, points or seeds
- **THEN** the check draws from that range rather than asserting one hand-picked case, so a property that holds only at the chosen literal is not mistaken for the requirement

#### Scenario: A sequence of operations is modelled as one
- **WHEN** a requirement constrains what stays true across a sequence of operations rather than within a single call
- **THEN** the check drives that sequence and asserts the invariant after each step, because single-shot cases cannot reach the states a sequence produces

#### Scenario: A scenario that cannot be code-tested is named
- **WHEN** a scenario describes something no test can reach, such as a decision recorded during the change itself
- **THEN** it is listed with the reason, so the exemption is auditable rather than an unexplained gap in the guard

#### Scenario: An in-flight exemption ends with its change
- **WHEN** an exemption gives as its reason that the scenario's change is still being implemented, and no change still in flight states that scenario
- **THEN** the guard reports the entry, so a change archived without a test for one of its scenarios is caught at the next run rather than left exempt under a reason that is no longer true
