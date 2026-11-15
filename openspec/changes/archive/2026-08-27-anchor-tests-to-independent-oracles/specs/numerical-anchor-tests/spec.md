## ADDED Requirements

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
