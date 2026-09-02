## ADDED Requirements

### Requirement: Numerical Burgers anchored by a manufactured solution with time-dependent forcing
The numerically integrated Burgers path SHALL be tested against a manufactured solution whose forcing depends on time: the MMS branch supplies an exact `(u, f)` pair, the numerical branch integrates from `u(0)` with that `f`, and the result is compared to the exact `u(T)`. This is the first anchor for the numerical path that exercises time dependence in the nonlinear term; the existing Cole–Hopf anchor is unforced and the spectral-residual anchor performs no integration.

#### Scenario: Exact solution reached within time-stepping error
- **WHEN** the round-trip is run at a fixed seed with spatial modes chosen so the manufactured `u²` lies inside the de-aliased band
- **THEN** the numerical `u(T)` matches the manufactured `u(T)` within a tolerance fixed in advance from the integrator's order, not tuned to pass

#### Scenario: Rate, not just size, is asserted
- **WHEN** the round-trip error is measured at a sequence of halving step sizes
- **THEN** the test asserts the measured order over the resolved range, so an integrator that evaluates the forcing at the wrong stage time fails the test even if its final error at one step count is small
