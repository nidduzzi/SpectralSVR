## ADDED Requirements

### Requirement: The manufactured field is periodic in time
A manufactured `(u, f)` pair SHALL be periodic along its time axis over the interval it is generated on, so that the field at the final time equals the field at the initial time to rounding level. The time axis is a Fourier axis and periodic by construction, so this is a property of how the pair is built rather than a choice a caller makes.

Because the pair returns to its own initial state over one period, a consumer integrating it for a full period and comparing only the final state SHALL NOT treat agreement as evidence that integration occurred: a solver that leaves its initial condition untouched reproduces that comparison exactly. A consumer relying on the pair as an oracle over a full period SHALL compare across the trajectory, or state the distance the field travels from its initial slice, or both.

#### Scenario: The manufactured field returns to its initial slice
- **WHEN** a manufactured pair is generated and its field is evaluated at the start and the end of its time interval
- **THEN** the two agree at rounding level, at every supported mode count

#### Scenario: A full-period comparison states what the solver had to do
- **WHEN** a manufactured pair is used as an oracle for an integrator run over one full period
- **THEN** the check does not rest on the final slice alone, and the distance the manufactured field travels from its initial slice is asserted alongside the error, so a solver that stands still fails rather than passes
