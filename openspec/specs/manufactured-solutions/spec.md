# manufactured-solutions Specification

## Purpose
Defines what a manufactured solution pair guarantees: that the chosen field together with its derived forcing satisfies the continuous equation everywhere, not merely at the sample points, and that the forcing is built with the same nonlinear-term convention the numerical solver uses, so a pair can serve as an oracle for that solver.

## Requirements

### Requirement: The manufactured pair satisfies the continuous equation
A manufactured `(u, f)` pair SHALL satisfy the governing equation as a continuous statement, so that evaluating every term at points that are not sample points leaves a residual at the level of floating-point rounding.

#### Scenario: Residual vanishes off the sample grid
- **WHEN** a manufactured pair is generated and every term of the equation is evaluated on a grid finer than and offset from the field's own sample points
- **THEN** the residual relative to the magnitude of the equation's left-hand side is at rounding level, not at a level that would be mistaken for a physical effect

#### Scenario: Bandwidth that overruns the grid is rejected or resolved
- **WHEN** the chosen field's bandwidth is such that its nonlinear term cannot be represented on the mode grid
- **THEN** the manufactured forcing is still the residual of the equation the solver integrates, because both apply the same de-aliasing

### Requirement: The forcing uses the solver's nonlinear-term convention
The nonlinear term used to derive a manufactured forcing SHALL be formed the same way the numerical solver forms it: from the real part of the sampled field, and through the same de-aliasing mask. A manufactured pair derived under a different convention is not an oracle for that solver.

#### Scenario: Real part taken before multiplying
- **WHEN** the field carries content in the top representable mode, whose spectral derivative is purely imaginary at the sample points
- **THEN** that component is discarded before the product is formed, as the solver discards it, so the two agree

#### Scenario: Same de-aliasing on both sides
- **WHEN** the manufactured forcing and the solver's nonlinear term are computed for the same field
- **THEN** they agree to rounding level, for every mode count the generator supports

### Requirement: Self-consistency is not evidence of correctness
The check that a pair's own discrete residual is zero SHALL NOT be presented as verification that the pair is a manufactured solution, because a pair carrying a defect shared by both sides of that check still reads zero.

#### Scenario: A wrong pair is caught
- **WHEN** the nonlinear term is deliberately formed by a convention the solver does not use
- **THEN** the discrete self-check still reports zero, while the continuous off-grid check reports a residual far above rounding
