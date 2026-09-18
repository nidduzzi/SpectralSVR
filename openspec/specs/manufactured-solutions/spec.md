# manufactured-solutions Specification

## Purpose
Defines what a manufactured solution pair guarantees: that the chosen field together with its derived forcing satisfies the continuous equation everywhere, not merely at the sample points, and that the forcing is built with the same nonlinear-term convention the numerical solver uses, so a pair can serve as an oracle for that solver.

## Requirements

### Requirement: The manufactured pair satisfies the continuous equation
A manufactured `(u, f)` pair SHALL satisfy the governing equation as a continuous statement, so that evaluating every term at points that are not sample points leaves a residual at the level of floating-point rounding. Where a choice of bandwidth would break that, the generator SHALL refuse the choice rather than produce a pair that is only self-consistent.

#### Scenario: Residual vanishes off the sample grid
- **WHEN** a manufactured pair is generated and every term of the equation is evaluated on a grid finer than and offset from the field's own sample points
- **THEN** the residual relative to the magnitude of the equation's left-hand side is at rounding level, not at a level that would be mistaken for a physical effect

#### Scenario: Bandwidth that overruns the grid is rejected or resolved
- **WHEN** a caller states a bandwidth whose nonlinear term cannot be represented within the band the solver keeps
- **THEN** it is rejected: the generator raises, naming the stated bandwidth and the widest one the mode count admits. Resolving it by letting both sides de-alias alike would leave a pair satisfying the de-aliased equation but not the continuous one, which this requirement forbids

#### Scenario: Self-consistency is not accepted in place of the guarantee
- **WHEN** a pair whose product overruns the kept band would still be internally consistent, since the forcing and the solver apply the same de-aliasing
- **THEN** that is not sufficient: the requirement is satisfaction of the continuous equation, and a pair meeting only the discrete condition is refused at generation rather than returned

### Requirement: The manufactured field is real by construction
The field a manufactured pair is built from SHALL describe a real function as generated, not as repaired. Its band SHALL be stated in a form that cannot select one member of a conjugate pair without the other, so the field is conjugate-symmetric before any resizing is applied to it.

#### Scenario: The generated field is conjugate-symmetric at every mode count
- **WHEN** a manufactured pair is generated at any supported mode count
- **THEN** the field's coefficients are conjugate-symmetric, and remain so with any downstream repair of resizing disabled, showing the field was generated correctly rather than corrected afterwards

#### Scenario: The band cannot be stated in a form that breaks symmetry
- **WHEN** the generator derives or accepts a band
- **THEN** it is a band that keeps a magnitude and its negative together with the zero mode, so no derivation or caller choice can produce an unpaired highest coefficient

#### Scenario: The bandwidth the caller can state is visible
- **WHEN** a caller generates a manufactured problem and wants a narrower field than the default
- **THEN** the bandwidth is a stated argument rather than an internal constant, so the difficulty of the manufactured problem is visible at the call site

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

### Requirement: The manufactured field is periodic in time
A manufactured `(u, f)` pair SHALL be periodic along its time axis over the interval it is generated on, so that the field at the final time equals the field at the initial time to rounding level. The time axis is a Fourier axis and periodic by construction, so this is a property of how the pair is built rather than a choice a caller makes.

Because the pair returns to its own initial state over one period, a consumer integrating it for a full period and comparing only the final state SHALL NOT treat agreement as evidence that integration occurred: a solver that leaves its initial condition untouched reproduces that comparison exactly. A consumer relying on the pair as an oracle over a full period SHALL compare across the trajectory, or state the distance the field travels from its initial slice, or both.

#### Scenario: The manufactured field returns to its initial slice
- **WHEN** a manufactured pair is generated and its field is evaluated at the start and the end of its time interval
- **THEN** the two agree at rounding level, at every supported mode count

#### Scenario: A full-period comparison states what the solver had to do
- **WHEN** a manufactured pair is used as an oracle for an integrator run over one full period
- **THEN** the check does not rest on the final slice alone, and the distance the manufactured field travels from its initial slice is asserted alongside the error, so a solver that stands still fails rather than passes
