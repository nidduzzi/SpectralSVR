## ADDED Requirements

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
