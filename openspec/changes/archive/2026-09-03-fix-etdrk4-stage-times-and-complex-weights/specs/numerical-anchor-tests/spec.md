## ADDED Requirements

### Requirement: Exponential integrator anchored on the assumptions the Burgers path does not exercise
The ETDRK4 integrator SHALL be tested on the two assumptions every Burgers-driven test leaves implicit — a nonlinear term that depends on time, and a linear operator with complex eigenvalues — using oracles independent of the integrator's own output: a closed-form solution for the convergence rate, and the closed-form weight expressions for the coefficients.

#### Scenario: Convergence rate is asserted as an invariant
- **WHEN** the integrator is run on a scalar semilinear problem with a stiff real eigenvalue and time-dependent forcing whose exact solution is known, at a sequence of halving step sizes
- **THEN** the test asserts the measured order over the resolved range, not merely that the final error is small, so a drop from fourth to first order fails the test

#### Scenario: Complex-eigenvalue weights are checked against the closed form
- **WHEN** the four weights are computed for purely imaginary eigenvalues at `|hL| >= 1`
- **THEN** the test compares each to the closed-form expression evaluated directly, at a tolerance that the half-circle real-part shortcut cannot meet

#### Scenario: The existing forcing test cannot be passed by a first-order method
- **WHEN** the pre-existing time-dependent-forcing MMS test is run against an integrator that evaluates all stages at `t_n`
- **THEN** it fails, because its tolerance is set below the error such an integrator produces at the test's step count while remaining above the error the corrected integrator produces
