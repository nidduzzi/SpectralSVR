## Purpose

Defines the conversions the library offers between complex spectral coefficients and real-valued representations of them, and what each guarantees: the shapes it accepts, the device and precision it preserves, and whether it can be inverted.

## ADDED Requirements

### Requirement: Encodings accept any number of mode axes
Every conversion between complex coefficients and a real representation SHALL accept a coefficient tensor with any number of mode axes, not only one. A coefficient grid for a multi-dimensional problem is a valid input to each of them.

#### Scenario: Two mode axes convert
- **WHEN** a coefficient tensor with a sample axis and two mode axes is passed to any coefficient encoding
- **THEN** it converts, producing a result whose trailing axis is the doubled representation and whose leading axes are unchanged

#### Scenario: One mode axis is unchanged
- **WHEN** a coefficient tensor with a single mode axis is converted
- **THEN** the result is what the library produced before, so existing callers are unaffected

### Requirement: Encodings preserve device and precision
A conversion SHALL return a tensor on the same device as its input and at the real precision corresponding to the input's complex precision. It SHALL NOT move data between devices as a side effect.

#### Scenario: Device is carried through
- **WHEN** a coefficient tensor on an accelerator is converted
- **THEN** the result is on that same accelerator, not on the host

#### Scenario: Precision corresponds
- **WHEN** coefficients at a given complex precision are converted
- **THEN** the real result is at the matching real precision

### Requirement: The polar encoding is invertible through a named function
The magnitude/phase encoding SHALL have a named inverse in the library, so it is symmetric with the real/imaginary encoding. Reconstructing from magnitude and phase SHALL return the original coefficients to rounding.

#### Scenario: Round-trip through polar form
- **WHEN** coefficients are converted to magnitude and phase and then back through the named inverse
- **THEN** the result matches the original to floating-point rounding, for any number of mode axes

#### Scenario: Both encodings offer the same shape of API
- **WHEN** a caller looks for the inverse of either encoding
- **THEN** both pairs are present and named consistently, rather than one being one-way

### Requirement: Resizing states which quantity it preserves
Documentation for resizing a coefficient grid SHALL state, for each setting of its rescaling option, whether the represented function or the stored coefficients are preserved, since the two cannot both be.

#### Scenario: The flag's meaning is documented
- **WHEN** a caller reads the documentation for resizing a basis to a different mode count
- **THEN** it states that one setting preserves the function it represents and the other preserves the coefficient values, and why they differ
