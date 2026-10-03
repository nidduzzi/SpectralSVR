## ADDED Requirements

### Requirement: A time-dependent conversion samples the field at the times its slices are read at
Converting a field whose first axis is time into a time-dependent representation of `nt` slices SHALL sample that field at the slice times the representation is read at, spread over the time interval inclusively from its start to its end, so the converted field holds the source's values at its slice times and interpolates between them. Any slice count of at least two SHALL be accepted, whatever the time axis's mode count.

#### Scenario: A converted field reads as its source at every time
- **WHEN** a field periodic in time is converted at any slice count and evaluated at a time in its interval
- **THEN** at the slice times, the interval's end included, the converted field's value matches the source's to rounding, and between slices it differs by no more than linear interpolation's bound, the squared slice spacing over eight times the field's largest second time derivative

#### Scenario: Any slice count is accepted
- **WHEN** a field is converted at a slice count below, equal to or above its time axis's mode count
- **THEN** the conversion succeeds with that many slices, rather than only at the count equal to the mode count

### Requirement: A time-independent conversion samples the field at the times its nodes are read at
Converting a time-dependent field of `nt` slices into a time-independent representation SHALL sample that field at the periodic times `start + i (stop - start) / nt` of its time interval, reading between slices by the representation's linear interpolation, so the result holds the time-dependent field's values at its own nodes. The time axis SHALL keep `nt` modes.

#### Scenario: A time-independent field reads as its time-dependent source at its nodes
- **WHEN** a time-dependent field is converted to a time-independent one
- **THEN** at each of the result's nodes its value matches the time-dependent field's value at that node's time to rounding

#### Scenario: A round trip stays within interpolation error
- **WHEN** a field periodic in time is converted to a time-dependent representation and back
- **THEN** at every node the result differs from the source by no more than the squared slice spacing over eight times the field's largest second time derivative
