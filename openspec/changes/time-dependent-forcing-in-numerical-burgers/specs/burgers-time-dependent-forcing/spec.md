## Purpose

Defines what forcings the numerically integrated Burgers generator accepts, how a time-dependent forcing enters the integration, and which forcing forms are rejected and why. Accepting a forcing that varies in time is what makes a dataset solved from a sampled forcing prior possible, rather than one whose forcing is whatever a random field leaves behind; building such a dataset belongs to the capability that reports it.

## ADDED Requirements

### Requirement: Numerical Burgers accepts a time-dependent forcing
The numerical Burgers generator SHALL accept a forcing that can be evaluated at any time in the integration interval, in either of two forms: a callable of time, or a spectral field over time and space that is fully spectral along its time axis. A callable SHALL declare whether it returns spectral coefficients or sampled values, because the two are indistinguishable by shape and produce different physics; a callable that does not declare its form SHALL be refused rather than assumed. The forcing SHALL be evaluated at the time the integrator passes for each stage, not frozen at the start of the step.

#### Scenario: Callable forcing is integrated
- **WHEN** the generator is given a callable `f(t)` returning `(n, ns)` spectral coefficients and a random initial condition
- **THEN** it returns a time-dependent solution field and a forcing field whose values at the output time grid equal `f` evaluated there

#### Scenario: Fully spectral forcing field is integrated
- **WHEN** the generator is given a Basis field over `(time, space)` with `time_dependent=False`, whose time domain equals the requested integration interval and whose spatial content the integration grid can represent
- **THEN** it integrates using that field's value at each stage time, and the returned forcing field is the input sampled onto the output time grid

#### Scenario: A forcing the grid cannot represent is refused
- **WHEN** a forcing field carries spatial modes the integration grid cannot represent, so sampling it there would fold that content onto lower wavenumbers
- **THEN** the generator raises rather than aliasing silently, names both mode counts, and names the explicit way to proceed: band-limit the field deliberately, either by resizing it or by wrapping it so the generator does it on the caller's instruction

#### Scenario: Forcing is evaluated at stage times, not step starts
- **WHEN** the same problem is integrated once with a forcing that varies within a step and once with that forcing frozen at each step's start
- **THEN** the two solutions differ, and only the former converges at fourth order as the step size halves

### Requirement: Per-time-slice forcing is rejected
The generator SHALL reject a forcing field whose time axis is sampled rather than spectral (`time_dependent=True`), because the integrator evaluates the forcing between grid points and such a field cannot be evaluated there without an interpolation the caller did not choose. It SHALL likewise reject a field whose time or space domain does not equal the interval being integrated: the field is sampled on the integration grid, so a field carrying either axis on another interval describes a different function at the points read from it.

#### Scenario: Sampled-time field raises
- **WHEN** a Basis field with `time_dependent=True` is passed as forcing to the numerical branch
- **THEN** the generator raises before integrating, and the error names the accepted forms

#### Scenario: Mismatched time domain raises
- **WHEN** a fully spectral forcing field's time domain does not equal the requested `time_domain` interval
- **THEN** the generator raises before integrating

#### Scenario: Mismatched space domain raises
- **WHEN** a fully spectral forcing field's space domain does not equal the requested `space_domain` interval
- **THEN** the generator raises before integrating, naming both intervals, rather than sampling the field at points that mean something else on its own period

### Requirement: Constant forcing is unchanged
A forcing given as a plain number SHALL take the path it takes today and produce the same coefficients bit for bit.

#### Scenario: Constant forcing byte-identity
- **WHEN** the numerical generator is run with a `Number` forcing
- **THEN** the returned forcing is exactly the constant spectrum at every output time, `torch.equal` with no tolerance, so normalising the forcing to a function of time changed no arithmetic on this path. Whether a seeded run is `torch.equal` to the same run before that normalisation is a comparison between two versions of the library, which a test cannot make; it belongs to the record of the change that makes such a claim

### Requirement: Manufactured solutions round-trip through the integrator
Given a manufactured `(u, f)` pair from the MMS branch, integrating numerically from `u(0)` with forcing `f` SHALL reproduce `u` at the final time to within time-stepping error, and that error SHALL shrink at the integrator's design order as the step size halves, provided the spatial resolution resolves the nonlinear term without aliasing and the error remains above the floor set by the stored coefficient precision. Beneath that floor the reading is saturation rather than a rate, which `numerical-anchor-tests` already requires be distinguished from order loss; the range the order is asserted over SHALL therefore be bounded above as well as below. The order SHALL be measured before any bound is asserted, and the bound SHALL sit beneath the measurement rather than being predicted from the scheme's nominal order.

#### Scenario: Round-trip matches the manufactured solution
- **WHEN** a manufactured pair is generated, and the numerical branch is run from its initial slice with its forcing at a spatial resolution that keeps `u²` inside the de-aliased band
- **THEN** the numerical solution matches the manufactured one at **every** output time within a tolerance fixed from the integrator's order at the chosen step count

#### Scenario: The round-trip comparison is not satisfied by standing still
- **WHEN** the manufactured solution is periodic over the integration interval, so its final slice equals its initial one
- **THEN** the comparison SHALL NOT be made against the final slice alone, and the distance the manufactured solution travels from its initial slice SHALL be asserted alongside the error, so a solver returning its initial condition unchanged fails rather than passes

#### Scenario: Round-trip converges at the integrator's order
- **WHEN** the round-trip is repeated at halving step sizes
- **THEN** the measured order over the resolved range is recorded, and the asserted bound sits beneath what was measured, so a rate that degrades is a failure rather than a number the bound was lowered to accommodate
