## Purpose

Defines how a caller chooses the numeric precision that spectral coefficients are stored and computed in, what the library guarantees at each precision, and which parts of the pipeline must move together so that precision is never silently lost between a basis and the model that consumes it.

## ADDED Requirements

### Requirement: Precision is configuration, not a class constant
The precision a basis stores its coefficients in SHALL be part of a basis's resolvable configuration, selectable per instance. It SHALL NOT be fixed as a constant on the basis class.

#### Scenario: Two precisions coexist
- **WHEN** two bases of the same class are constructed, one at the default precision and one at double
- **THEN** each stores and returns coefficients at its own precision, and neither affects the other

#### Scenario: Default is unchanged
- **WHEN** a basis is constructed without naming a precision
- **THEN** it uses single precision, and every seeded field it produces is bit-for-bit what it produced before this change

### Requirement: Basis precision and model precision move together
Selecting a precision SHALL determine both the complex precision a basis stores and the real precision the regressor consumes, because the model boundary converts complex coefficients to interleaved real pairs. A configuration where the two disagree SHALL be rejected or reconciled, never silently truncated.

#### Scenario: Double precision survives the model boundary
- **WHEN** a double-precision basis supplies coefficients to a regressor
- **THEN** the interleaved real values reaching the regressor are double precision, and no cast reduces them

#### Scenario: A mismatch is not silent
- **WHEN** a caller pairs a double-precision basis with a regressor fixed at single precision
- **THEN** the mismatch is refused rather than resolved by discarding precision

#### Scenario: A widening mismatch proceeds but is reported
- **WHEN** a caller pairs a single-precision basis with a regressor fixed at double precision
- **THEN** the work proceeds, because widening discards nothing, and the mismatch is reported so the cost of the wider precision is not paid unknowingly

#### Scenario: Inputs a caller supplies are not governed by the basis's precision
- **WHEN** a caller supplies real-valued inputs at a precision of their own choosing, rather than coefficients carrying the basis's precision through the interleaved-real bridge
- **THEN** they are converted as before without either report, because the basis's configuration makes no claim about them

### Requirement: The interleaved-real bridge is precision-transparent and robust
Conversion between complex coefficients and interleaved real pairs SHALL preserve values exactly in both directions at every supported precision and any number of coefficient axes, and SHALL return its result on the device its input was on. It SHALL accept an input whose memory layout is not contiguous, and SHALL allow the target precision to be stated rather than only inferred.

#### Scenario: Exact round-trip at both precisions
- **WHEN** coefficients at single or double precision are converted to interleaved real pairs and back
- **THEN** the result equals the original exactly, for two- and three-axis coefficient grids alike

#### Scenario: The device is carried through
- **WHEN** coefficients on an accelerator are converted in either direction
- **THEN** the result is on that same accelerator, so a caller does not have to move it back

#### Scenario: Non-contiguous input is accepted
- **WHEN** the real input is a view whose last axis is not contiguous in memory
- **THEN** the conversion succeeds rather than raising

#### Scenario: Target precision can be stated
- **WHEN** real values that passed through a lower-precision stage are converted back
- **THEN** the caller can name the intended complex precision, instead of the result being inferred from the reduced input

### Requirement: The precision cost is documented where it is chosen
Documentation for the precision setting SHALL state that double precision is substantially slower for matrix-bound work on consumer hardware and doubles memory per coefficient, so that a caller selects it deliberately for accuracy-critical work rather than by habit.

#### Scenario: Cost is stated at the point of choice
- **WHEN** a caller reads the documentation for the precision setting
- **THEN** it names the performance and memory consequences and the kind of work that justifies them
