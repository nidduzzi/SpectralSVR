## Purpose

States what the library's exponential time integrator guarantees when it integrates `dv/dt = L v + N(t, v)`: the accuracy order it delivers when the nonlinear term depends on time, the class of diagonal linear operators it handles correctly, the time grids it accepts, and how it fails when a caller steps outside those.

## ADDED Requirements

### Requirement: Fourth-order accuracy with a time-dependent nonlinear term
The exponential integrator SHALL evaluate the nonlinear term at the four Cox–Matthews stage times `t_n`, `t_n + h/2`, `t_n + h/2`, `t_n + h`, so that its global error is fourth order in the step size whether or not `N` depends explicitly on `t`.

#### Scenario: Convergence rate under time-dependent forcing
- **WHEN** a semilinear scalar problem with a stiff real eigenvalue and a time-dependent nonlinear term whose exact solution is known in closed form is integrated at a sequence of halving step sizes
- **THEN** the measured order `log2(e_h / e_{h/2})` approaches 4, and never sits near 1, over the resolved part of the sequence

#### Scenario: Time-independent nonlinear term is unaffected
- **WHEN** the nonlinear term ignores its time argument
- **THEN** the trajectory is identical to what the integrator produced before this change at the same inputs

### Requirement: Correct weights for any diagonal linear operator
The integrator SHALL compute its four contour-integral weights correctly for a diagonal linear operator with real, imaginary, or general complex eigenvalues. When the eigenvalues have no imaginary part it SHALL use the arithmetic it uses today, so results on real operators do not move; when any eigenvalue has an imaginary part it SHALL average over the full unit circle and keep the complex mean.

#### Scenario: Complex eigenvalues match the closed form
- **WHEN** the weights are computed for purely imaginary eigenvalues at a step size where `|hL| >= 1`, so the closed-form weight expressions are well-conditioned
- **THEN** each weight matches its closed form to within floating-point rounding, not to within one percent

#### Scenario: Real eigenvalues are unchanged
- **WHEN** the weights are computed for eigenvalues stored in a complex dtype whose imaginary parts are all exactly zero
- **THEN** the weights are bit-for-bit what the half-circle real-part average produced before this change

#### Scenario: Branch is decided by imaginary content, not dtype
- **WHEN** the linear operator is a complex-dtype tensor with zero imaginary part
- **THEN** the integrator takes the real-operator path

### Requirement: Uniform time grid is required and checked
The integrator SHALL read its step size from the supplied time grid and SHALL reject a grid whose spacing is not uniform, rather than silently integrating every step after the first with the wrong step size.

#### Scenario: Non-uniform grid is rejected
- **WHEN** the integrator is given a strictly increasing time grid whose consecutive differences are not all equal within floating-point tolerance
- **THEN** it raises before taking any step

#### Scenario: Uniform grid from linspace is accepted
- **WHEN** the grid is produced by evenly spacing a start and end point
- **THEN** integration proceeds

### Requirement: Documentation states the operator class accurately
The integrator's documentation, and that of any problem generator that relies on it, SHALL state that the linear operator must be diagonal in the spectral basis, that its eigenvalues may be complex, and that the time grid must be uniform. It SHALL NOT claim support for a class of operators the implementation gets wrong.

#### Scenario: Docstrings match behaviour
- **WHEN** a caller reads the integrator's docstring and the Burgers numerical-path docstring
- **THEN** every stated capability is one the tests in this change exercise, and the uniform-grid requirement is stated
