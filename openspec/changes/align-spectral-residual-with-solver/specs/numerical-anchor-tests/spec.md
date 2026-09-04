## ADDED Requirements

### Requirement: Manufactured pairs anchored off the sample grid
The manufactured-solution generator SHALL be anchored by evaluating the governing equation at points that are not its sample points, because every existing check evaluates on-grid, where a pair carrying a shared defect reads exactly zero. The anchor SHALL include a control whose construction cannot exhibit the defect, so a clean reading is shown to be informative rather than blind.

#### Scenario: Continuous residual is the oracle
- **WHEN** a manufactured pair is generated and its residual is measured on a fine grid offset from the sample points
- **THEN** the test asserts that residual, not the pair's own discrete self-check

#### Scenario: A construction that cannot carry the defect reads clean
- **WHEN** the same measurement is applied to a field built with an odd mode count, which has no top-mode-without-a-partner
- **THEN** it reads at rounding level, establishing that the measurement distinguishes correct pairs from incorrect ones

#### Scenario: Deliberate corruption is detected
- **WHEN** a known-good manufactured forcing is perturbed by a small relative amount
- **THEN** the measured residual rises far above its clean value, so the anchor is shown to have power rather than merely reporting small numbers
