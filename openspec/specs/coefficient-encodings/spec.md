# coefficient-encodings Specification

## Purpose
Defines how the library represents a grid of complex spectral coefficients: the conversions between those coefficients and real-valued representations of them, the resizing of a grid to a different mode count, and the band-limiting of a grid in place. States what each guarantees — the shapes it accepts, the device and precision it preserves, whether it can be inverted, and whether the array it produces still describes a real function.

## Requirements

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
Where an argument makes it impossible for an operation to return a coefficient grid describing a real field, the operation's documentation SHALL state that, and SHALL state why no choice of result satisfies the request. Silently returning a grid that does not describe a real field is not acceptable, whether or not the operation's behaviour is changed. Where the request can instead be made unstatable, making it unstatable SHALL be preferred to documenting it.

#### Scenario: The limitation is stated where the caller meets it
- **WHEN** a caller reads the documentation for a form of request whose result is not guaranteed to describe a real field
- **THEN** it states that the result may not be real, states why, and names the form to use instead when a real result is required

#### Scenario: Documenting is not a substitute for a fix that exists
- **WHEN** an operation with the same limitation can be given a result that does describe a real field without changing what the caller asked for
- **THEN** it is repaired rather than documented, and the documentation records the rule it now follows

#### Scenario: Unstatable beats documented
- **WHEN** the request that cannot be satisfied can be removed from what a caller is able to express
- **THEN** it is removed rather than documented, so the defect cannot be reached by a caller who did not read the documentation

#### Scenario: Band-limiting is no longer an instance of this
- **WHEN** a caller band-limits a coefficient grid through a band that guarantees a real result
- **THEN** the result is conjugate-symmetric for every band that can be stated, so the operation no longer relies on documentation to warn about an even count

### Requirement: A time-dependent conversion samples the field at the times its slices are read at
Converting a field whose first axis is time into a time-dependent representation of `nt` slices SHALL sample that field at the slice times the representation is read at, spread over the time interval inclusively from its start to its end, so the converted field holds the source's values at its slice times and interpolates between them. A slice count is a zero-dimensional, non-boolean integer: a Python integer, a numpy integer, or a zero-dimensional integer array or tensor. Any such count of at least two SHALL be accepted, whatever the time axis's mode count; a boolean of any kind and a value of more than zero dimensions SHALL be refused. A field that is already time-dependent SHALL be converted only to its own slice count, and only when it has at least two slices. A slice count that cannot span the interval, a basis without coefficients, and a field without a spatial axis SHALL be refused with a named error. The conversion SHALL honour the basis's evaluation strategy; the accuracy it guarantees below holds under exact evaluation, and an approximate strategy adds that approximation's own error.

#### Scenario: A converted field reads as its source at every time
- **WHEN** a field periodic in time is converted under exact evaluation at any slice count and evaluated at a time in its interval
- **THEN** at the slice times, the interval's end included, the converted field's value matches the source's to rounding, and between slices it differs by no more than linear interpolation's bound, the squared slice spacing over eight times the field's largest second time derivative

#### Scenario: Any slice count is accepted
- **WHEN** a field that is not yet time-dependent is converted at a slice count below, equal to or above its time axis's mode count
- **THEN** the conversion succeeds with that many slices, rather than only at the count equal to the mode count

#### Scenario: A time-dependent field converts only to its own slice count
- **WHEN** a field that is already time-dependent is converted with no slice count or with its own
- **THEN** the result is a new field over the same coefficients, as `copy()` gives, and with any other slice count the conversion raises naming both counts; a time-dependent field of fewer than two slices is refused rather than passed through

#### Scenario: A conversion that cannot be placed in time is refused
- **WHEN** the slice count is not a zero-dimensional integer, is a boolean of any kind, or is below two, the basis holds no coefficients, whether time-dependent or not, or the field has no spatial axis
- **THEN** the conversion raises a named error saying which, rather than failing inside the transform

#### Scenario: A conversion honours its evaluation strategy
- **WHEN** a field is converted to a time-dependent representation under an evaluation strategy, approximate or exact
- **THEN** the result matches the same sampling and transform carried out under that strategy, to rounding

### Requirement: A time-independent conversion samples the field at the times its nodes are read at
Converting a time-dependent field of `nt` slices into a time-independent representation SHALL sample that field at the periodic, half-open times `start + i (stop - start) / nt` of its time interval, whatever the basis's default sampling, reading between slices by the representation's linear interpolation, so the result holds the time-dependent field's values at its own nodes. The time axis SHALL keep `nt` modes. A time-dependent field of fewer than two slices, without a spatial axis, or without coefficients SHALL be refused with a named error. The accuracy guaranteed below holds under exact evaluation.

#### Scenario: A time-independent field reads as its time-dependent source at its nodes
- **WHEN** a time-dependent field is converted to a time-independent one under exact evaluation
- **THEN** at each of the result's nodes its value matches the time-dependent field's value at that node's time to rounding

#### Scenario: A round trip stays within interpolation error
- **WHEN** a field periodic in time is converted to a time-dependent representation and back under exact evaluation
- **THEN** at every node the result differs from the source by no more than the squared slice spacing over eight times the field's largest second time derivative

#### Scenario: A time-independent conversion that cannot be placed is refused
- **WHEN** a time-dependent field has fewer than two slices, no spatial axis, or no coefficients
- **THEN** the conversion to a time-independent field raises a named error saying which, rather than returning a field constant in time or failing inside the transform

### Requirement: A per-axis sampling names every axis of the field it samples
Resolving a per-axis sampling, for a transform, an inverse transform, an evaluation on a grid or a conversion, SHALL refuse one whose length differs from the number of axes it is resolved for, naming both counts, rather than using some of its schemes. A basis without coefficients has no axes, so resolving its per-axis sampling SHALL be refused with a named error.

#### Scenario: A sampling of the wrong length is refused wherever it is resolved
- **WHEN** a per-axis sampling shorter or longer than the axis count is passed to a transform or inverse transform, or held by a basis whose values or conversions are requested
- **THEN** the call raises naming the sampling's length and the axis count, and a basis without coefficients raises saying so
