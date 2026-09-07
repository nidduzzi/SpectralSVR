## MODIFIED Requirements

### Requirement: An operation that cannot return a real field says so
Where an argument makes it impossible for an operation to return a coefficient grid describing a real field, the operation's documentation SHALL state that, and SHALL state why no choice of result satisfies the request. Silently returning a grid that does not describe a real field is not acceptable, whether or not the operation's behaviour is changed. Where the request can instead be made unstatable, making it unstatable SHALL be preferred to documenting it.

#### Scenario: The limitation is stated where the caller meets it
- **WHEN** a caller reads the documentation for a form of request whose result is not guaranteed to describe a real field
- **THEN** it states that the result may not be real, states why, and names the form to use instead when a real result is required

#### Scenario: Documenting is not a substitute for a fix that exists
- **WHEN** an operation with the same limitation can be given a result that does describe a real field without changing what the caller asked for
- **THEN** it is repaired rather than documented, and the documentation records the rule it now follows

#### Scenario: Unstatable beats documented
- **WHEN** the request that cannot be satisfied can be removed from what a caller is able to express
- **THEN** it is removed rather than documented, so the defect cannot be reached by a caller who did not read the documentation

#### Scenario: Band-limiting is no longer an instance of this
- **WHEN** a caller band-limits a coefficient grid through a band that guarantees a real result
- **THEN** the result is conjugate-symmetric for every band that can be stated, so the operation no longer relies on documentation to warn about an even count
