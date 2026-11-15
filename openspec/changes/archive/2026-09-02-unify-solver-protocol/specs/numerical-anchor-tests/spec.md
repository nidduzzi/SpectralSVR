## ADDED Requirements

### Requirement: Numerical Burgers trajectories anchored by an independent integrator
The numerically integrated Burgers path SHALL be tested against a second, independently implemented integrator reached through the monolithic-to-semilinear conversion, in a regime chosen so that the non-stiff integrator is itself accurate (mild viscosity, coarse mode grid, fine time grid). This complements the existing MMS and Cole-Hopf anchors, which check the spatial operator and exact solutions but share the exponential integrator on the time axis.

#### Scenario: Two integrators agree in a non-stiff regime
- **WHEN** the same Burgers initial-value problem is generated once with the default exponential integrator and once with a converted non-stiff integrator, at a fixed seed and in a regime where both are accurate
- **THEN** the two trajectories agree within a tolerance fixed in advance from the integrators' orders, not tuned to make the test pass

#### Scenario: Regime boundary is asserted, not assumed
- **WHEN** the viscosity or mode count is increased so the problem becomes stiff for the non-stiff integrator
- **THEN** the test does not claim agreement in that regime, and the chosen regime's parameters appear explicitly in the test rather than as defaults
