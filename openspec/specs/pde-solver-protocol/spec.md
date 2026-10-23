# pde-solver-protocol Specification

## Purpose
Defines the solver interfaces the library exposes for time integration, how a problem generator accepts a caller-supplied solver, and how a solver written against one interface is converted to the other so any available integrator can drive a spectral PDE problem.

## Requirements
### Requirement: Two solver interfaces are public
The library SHALL expose two time-integration interfaces and keep both usable by external callers: a monolithic one taking a single right-hand side `rhs(t, y)` together with `y0` and `t`, and a semilinear one taking a diagonal linear operator, a nonlinear term `N(t, v)`, `y0`, and `t`. Every solver the library ships SHALL declare which interface it satisfies.

#### Scenario: Monolithic solvers remain callable
- **WHEN** a caller invokes a shipped monolithic solver with `rhs(t, y)`, an initial state, and a time grid
- **THEN** it returns a solution of shape `(steps, *y0.shape)` whose first entry equals the initial state

#### Scenario: Semilinear solver requires the split form
- **WHEN** a caller invokes a shipped semilinear solver
- **THEN** it accepts the linear operator and nonlinear term as separate arguments and rejects a mode grid that does not match the initial state's trailing shape

### Requirement: Monolithic solvers convert to the semilinear interface
The library SHALL provide a conversion from any monolithic solver to the semilinear interface, formed by evaluating `linear * v + N(t, v)` as the right-hand side. The conversion SHALL preserve the wrapped solver's own result exactly: converting a solver and integrating a semilinear system MUST return what that solver returns when handed the equivalent combined right-hand side directly.

#### Scenario: Conversion is result-preserving
- **WHEN** a monolithic solver is converted and applied to a linear operator and nonlinear term
- **THEN** the returned trajectory equals, elementwise, the trajectory the same solver produces from the combined right-hand side

#### Scenario: Complex spectral state is supported
- **WHEN** a converted solver integrates a complex-valued state on a mode grid
- **THEN** it returns a complex trajectory of the same dtype without raising

#### Scenario: No conversion in the other direction
- **WHEN** a caller holds only a monolithic right-hand side and wants a semilinear solver
- **THEN** no conversion is offered, because the linear operator cannot be recovered from a combined right-hand side

### Requirement: Problem generators accept a solver on the semilinear interface
A problem generator that integrates a PDE numerically SHALL accept a solver on the semilinear interface and use it for that integration. The parameter SHALL NOT be accepted and then ignored.

#### Scenario: Supplied solver is the one that runs
- **WHEN** a caller passes a solver to a numerically integrated problem generator
- **THEN** the returned trajectories are produced by that solver, and a solver that raises causes generation to raise

#### Scenario: Monolithic solver via conversion
- **WHEN** a caller converts a monolithic solver and passes the result to a problem generator
- **THEN** generation completes and returns fields of the same shape and type as with the default solver

#### Scenario: Wrong interface is rejected
- **WHEN** a caller passes a monolithic solver directly, without converting it
- **THEN** the mismatch is a declared type error rather than a silently ignored argument

### Requirement: The default solver is the exponential integrator
The default solver for numerically integrated PDE generation SHALL be the exponential time-differencing integrator that handles the stiff diagonal linear part exactly, so that generation with no `solver` argument keeps producing the trajectories it produces today.

#### Scenario: Default generation is unchanged
- **WHEN** a problem is generated with no solver argument, at a fixed seed
- **THEN** the resulting fields match those produced before this change at the same seed and settings

### Requirement: Accuracy regime of converted solvers is stated
Documentation for the conversion SHALL state that a solver designed for non-stiff systems loses accuracy on a stiff spectral operator, and SHALL identify the regime (viscosity, mode count, step size) in which a converted solver is expected to agree with the exponential integrator. The conversion SHALL NOT silently substitute a different step size or tolerance to hide that loss.

#### Scenario: Regime is documented
- **WHEN** a caller reads the conversion's documentation
- **THEN** it states the stiffness limitation and the regime in which cross-solver agreement is expected

#### Scenario: Time grid is passed through unchanged
- **WHEN** a converted solver is given a time grid
- **THEN** the wrapped solver receives that same grid, with no substituted step size or tolerance
