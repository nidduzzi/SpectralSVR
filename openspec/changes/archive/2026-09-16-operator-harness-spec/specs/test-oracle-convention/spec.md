## ADDED Requirements

### Requirement: An anchor exemption is re-examined when its module grows
An exemption from the independent-oracle requirement SHALL rest on a justification that still describes the module. When a module acquires numerical behaviour beyond what its exemption describes, the exemption SHALL be re-justified against the module's current contents or withdrawn.

#### Scenario: The operators exemption is re-examined
- **WHEN** the exemption justified as covering thin adapters is checked against a module that now carries a benchmark, a dataset builder, a sensor grid and a third-party wrapper
- **THEN** the justification is either rewritten to describe what is actually exempt, or the exemption is removed and the anchor it owed is settled

#### Scenario: Withdrawal never leaves a failing guard
- **WHEN** an exemption is withdrawn
- **THEN** the anchor it was standing in for is supplied in the same change, or, where that is not possible, recorded as outstanding work — in neither case is the guard left failing, because a failing guard gets muted rather than fixed

#### Scenario: The anchor is preferred to the record
- **WHEN** the anchor a withdrawal requires can be written without new mathematics, because a closed form for the module's behaviour already exists
- **THEN** it is written with the withdrawal rather than deferred, so the module is covered at the moment its exemption ends

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

