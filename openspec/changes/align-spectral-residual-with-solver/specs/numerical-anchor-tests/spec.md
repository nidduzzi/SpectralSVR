## ADDED Requirements

### Requirement: Manufactured pairs anchored off the sample grid
The manufactured-solution generator SHALL be anchored by evaluating the governing equation at points that are not its sample points, because every existing check evaluates on-grid, where a pair carrying a shared defect reads exactly zero. The anchor SHALL include a positive control that is clean because it satisfies the governing equation, and a negative control that is dirty because it was deliberately corrupted, so a clean reading is shown to be informative rather than blind. A construction that is merely correlated with cleanliness SHALL NOT serve as the positive control, since the correlation can hold for reasons unrelated to the defect and can cease to hold when unrelated code changes.

#### Scenario: Continuous residual is the oracle
- **WHEN** a manufactured pair is generated and its residual is measured on a fine grid offset from the sample points
- **THEN** the test asserts that residual, not the pair's own discrete self-check

#### Scenario: A pair known to be correct reads clean
- **WHEN** the same measurement is applied to a pair that satisfies the governing equation analytically and is constructed to stay within the resolved band
- **THEN** it reads at rounding level, establishing that the measurement distinguishes correct pairs from incorrect ones rather than reporting a defect on every input

#### Scenario: Deliberate corruption is detected
- **WHEN** a known-good manufactured forcing is perturbed by a small relative amount
- **THEN** the measured residual rises far above its clean value, so the anchor is shown to have power rather than merely reporting small numbers
