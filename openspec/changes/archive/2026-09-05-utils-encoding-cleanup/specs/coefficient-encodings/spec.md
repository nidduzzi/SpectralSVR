## Purpose

Defines how the library represents a grid of complex spectral coefficients: the conversions between those coefficients and real-valued representations of them, the resizing of a grid to a different mode count, and the band-limiting of a grid in place. States what each guarantees — the shapes it accepts, the device and precision it preserves, whether it can be inverted, and whether the array it produces still describes a real function.

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

### Requirement: Resizing preserves the symmetry that makes a field real
Resizing a coefficient grid to a larger mode count SHALL produce an array that is conjugate-symmetric whenever its input was. A coefficient that was exempt from the symmetry rule only because it was the highest mode the source grid could hold SHALL NOT be carried unchanged onto a grid where it is no longer the highest, since it is no longer exempt there.

#### Scenario: An even-count field stays real after padding
- **WHEN** a conjugate-symmetric coefficient grid with an even mode count is resized to a larger mode count
- **THEN** the result is conjugate-symmetric, and the values it represents have no imaginary part beyond rounding

#### Scenario: An odd-count field is unaffected
- **WHEN** a conjugate-symmetric coefficient grid with an odd mode count is resized to a larger mode count
- **THEN** the result is conjugate-symmetric, as it already was, and no coefficient is redistributed — an odd count has no exempt coefficient to redistribute

#### Scenario: The represented function is unchanged
- **WHEN** a coefficient grid is resized to a larger mode count
- **THEN** the function it represents agrees with the original at the original sample points, so restoring the symmetry does not alter the field

### Requirement: Resizing conserves content at the boundary mode
Resizing a coefficient grid to a smaller mode count SHALL combine, rather than discard, the source content at the two mode numbers that the target grid cannot distinguish. On the target grid those two mode numbers describe the same sampled wave, so keeping one and dropping the other loses content that the target can represent.

#### Scenario: Truncating to an even count keeps both halves
- **WHEN** a coefficient grid is resized down to an even mode count
- **THEN** the target's single boundary coefficient carries the combined source content at both of the mode numbers that map onto it

#### Scenario: Enlarging then reducing returns the original
- **WHEN** a coefficient grid is resized to a larger mode count and then back to its original mode count
- **THEN** the result equals the original grid to rounding, for both even and odd mode counts

### Requirement: An operation that cannot return a real field says so
Where an argument makes it impossible for an operation to return a coefficient grid describing a real field, the operation's documentation SHALL state that, and SHALL state why no choice of result satisfies the request. Silently returning a grid that does not describe a real field is not acceptable, whether or not the operation's behaviour is changed.

#### Scenario: The limitation is stated where the caller meets it
- **WHEN** a caller reads the documentation for band-limiting a coefficient grid in place
- **THEN** it states that an even count of kept modes yields a field that is not real, because a kept set describing a real field is some pairs around a lone zero mode and is therefore always odd

#### Scenario: Documenting is not a substitute for a fix that exists
- **WHEN** an operation with the same limitation can be given a result that does describe a real field without changing what the caller asked for
- **THEN** it is repaired rather than documented, and the documentation records the rule it now follows
