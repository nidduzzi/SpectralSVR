## ADDED Requirements

### Requirement: An anchor exemption is re-examined when its module grows
An exemption from the independent-oracle requirement SHALL rest on a justification that still describes the module. When a module acquires numerical behaviour beyond what its exemption describes, the exemption SHALL be re-justified against the module's current contents or withdrawn.

#### Scenario: The operators exemption is re-examined
- **WHEN** the exemption justified as covering thin adapters is checked against a module that now carries a benchmark, a dataset builder, a sensor grid and a third-party wrapper
- **THEN** the justification is either rewritten to describe what is actually exempt, or the exemption is removed and the obligation to add an anchor recorded

#### Scenario: Withdrawal is a recorded decision
- **WHEN** an exemption is withdrawn
- **THEN** the anchor it now requires is recorded as outstanding work rather than left as a silently failing guard
