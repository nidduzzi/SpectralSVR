# numerical-anchor-tests Specification

## Purpose
The library's numerical APIs are tested against oracles independent of the code under test -- closed forms, hand-computed values, and independently coded solves -- so a bug affecting both sides of a round-trip comparison cannot survive unnoticed.

## Requirements

### Requirement: Metric formulas anchored to hand-computed values
`utils.get_metrics` SHALL be tested against hand-computed expected values for every metric it returns (`mse`, `rmse`, `mae`, `r2`, `smape`, `rse`, `rrse`) on small fixed arrays, including a near-constant-target case that reproduces the observed rrse blow-up pattern.

#### Scenario: Fixed-array metric check
- **WHEN** `get_metrics` is called on a small fixed prediction/target pair with hand-computed expected metrics
- **THEN** every returned metric matches its hand-computed value

#### Scenario: rrse on near-constant target
- **WHEN** the target contains a near-constant output column
- **THEN** rrse matches the standard definition sqrt(SSE/SST); if the current implementation disagrees, it is fixed and the benchmark results regenerated

### Requirement: Time-axis derivative and integral anchored analytically
Time-dependent `grad`/`integral` (time axis) SHALL be tested against closed-form time derivatives of a separable field u(t,x) = g(t)·sin(2πx), fuzzed over the closed form's free parameters.

#### Scenario: Time derivative of known field
- **WHEN** a time-dependent field with analytically known time derivative is differentiated along the time axis
- **THEN** interior-point values match the closed form within finite-difference accuracy, for Hypothesis-drawn parameters of g

### Requirement: Coefficient-manipulation values anchored
`resize_modes`, `zero_coeff`, and `to_time_dependent`/`to_time_independent` SHALL be tested for actual coefficient values (not just shapes) against hand-written expected spectra.

#### Scenario: Truncation and padding values
- **WHEN** a hand-written spectrum is resized down and up
- **THEN** surviving coefficients equal the expected literal values and padded positions are zero

#### Scenario: Mode zeroing values
- **WHEN** `zero_coeff` masks a hand-written spectrum
- **THEN** exactly the expected positions are zeroed and the rest are unchanged

### Requirement: Non-FFT transform leg anchored independently
The matmul (non-FFT) transform path and per-axis sampling dispatch SHALL be tested against closed-form spectra directly, not solely against the FFT path.

#### Scenario: Matmul path on single-mode input
- **WHEN** the non-FFT path transforms a Hypothesis-drawn single-mode sinusoid
- **THEN** the spectrum matches the closed form computed from the drawn mode, amplitude, and phase

### Requirement: LSSVR anchored to an independent solve
`LSSVR.fit`/`predict` SHALL be tested against an independently coded solution of the LSSVR KKT linear system (written from the paper definition inside the test) on a small problem.

#### Scenario: Predictions match the KKT solve
- **WHEN** LSSVR is fit on a small dataset and the same KKT system is solved inline with `torch.linalg.solve`
- **THEN** LSSVR predictions match the independent solve's predictions

### Requirement: SpectralSVR forward accuracy anchored analytically
`SpectralSVR.train`/`forward` SHALL be tested on the antiderivative operator with single-mode sine inputs whose exact output is known in closed form, fuzzed over mode index and amplitude.

#### Scenario: Forward matches closed-form antiderivative
- **WHEN** the model is trained on single-mode derivative/antiderivative pairs and queried on a held-out drawn mode/amplitude
- **THEN** predicted values match −cos(2πmx)/(2πm)-form closed-form targets within a fixed (not tuned-to-pass) tolerance

### Requirement: Inverse problem anchored to known inputs
`SpectralSVR.inverse`/`inverse_coeff` SHALL be tested by recovering an analytically known input function from its known output, asserting error against the analytic target rather than loss decrease.

#### Scenario: Inverse recovers known input
- **WHEN** `inverse` is run (fixed seed) on the output of an analytically constructed input
- **THEN** the recovered input matches the analytic input within a documented tolerance

### Requirement: New anchor tests follow the fuzz policy
New anchor tests SHALL use Hypothesis fuzzing over the oracle's free parameters wherever the closed form stays exact under parametrization; `no_fuzz` is reserved for literal fixed-point anchors (hand-written spectra, hand-computed metric tables, the tiny KKT system).

#### Scenario: Fuzzed analytic anchor
- **WHEN** an anchor's expected value is computable from drawn parameters (mode, amplitude, phase, sizes, seed)
- **THEN** the test draws those parameters with `@given` and computes the expectation from them inside the test

#### Scenario: Fixed-point anchor opts out explicitly
- **WHEN** an anchor's point is a literal hand-computed value
- **THEN** the test is marked `no_fuzz` per the existing conftest policy

### Requirement: Numerical Burgers trajectories anchored by an independent integrator
The numerically integrated Burgers path SHALL be tested against a second, independently implemented integrator reached through the monolithic-to-semilinear conversion, in a regime chosen so that the non-stiff integrator is itself accurate (mild viscosity, coarse mode grid, fine time grid). This complements the existing MMS and Cole-Hopf anchors, which check the spatial operator and exact solutions but share the exponential integrator on the time axis.

#### Scenario: Two integrators agree in a non-stiff regime
- **WHEN** the same Burgers initial-value problem is generated once with the default exponential integrator and once with a converted non-stiff integrator, at a fixed seed and in a regime where both are accurate
- **THEN** the two trajectories agree within a tolerance fixed in advance from the integrators' orders, not tuned to make the test pass

#### Scenario: Regime boundary is asserted, not assumed
- **WHEN** the viscosity or mode count is increased so the problem becomes stiff for the non-stiff integrator
- **THEN** the test does not claim agreement in that regime, and the chosen regime's parameters appear explicitly in the test rather than as defaults

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

### Requirement: Manufactured pairs anchored off the sample grid
The manufactured-solution generator SHALL be anchored by evaluating the governing equation at points that are not its sample points, because every existing check evaluates on-grid, where a pair carrying a shared defect reads exactly zero. The anchor SHALL include a positive control that is clean because it satisfies the governing equation, and a negative control that is dirty because it was deliberately corrupted, so a clean reading is shown to be informative rather than blind. A construction that is merely correlated with cleanliness SHALL NOT serve as the positive control, since the correlation can hold for reasons unrelated to the defect and can cease to hold when unrelated code changes.

#### Scenario: Continuous residual is the oracle
- **WHEN** a manufactured pair is generated and its residual is measured on a fine grid offset from the sample points
- **THEN** the test asserts that residual, not the pair's own discrete self-check

#### Scenario: A pair known to be correct reads clean
- **WHEN** the same measurement is applied to a pair that satisfies the governing equation analytically and is constructed to stay within the resolved band
- **THEN** it reads at rounding level, establishing that the measurement distinguishes correct pairs from incorrect ones rather than reporting a defect on every input

#### Scenario: Deliberate corruption is detected
- **WHEN** a known-good manufactured forcing is perturbed by a small relative amount
- **THEN** the measured residual rises far above its clean value, so the anchor is shown to have power rather than merely reporting small numbers

### Requirement: An anchor may request the precision it needs
An anchor whose oracle requires more precision than the library's default SHALL be able to request it through the precision setting, rather than being relocated to a lower layer or abandoned. Where an anchor's measurement would otherwise saturate at the default precision, the test SHALL state which precision it runs at and why.

#### Scenario: A convergence anchor runs at double precision
- **WHEN** an anchor measures a convergence rate whose error range extends below single precision's resolution
- **THEN** it selects double precision explicitly, and records that the default would floor the measurement

#### Scenario: Saturation is distinguished from order loss
- **WHEN** a convergence measurement flattens
- **THEN** the test makes clear whether the flattening is the method's behaviour or the precision floor, rather than reporting the measured order alone
