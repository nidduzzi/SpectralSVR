## ADDED Requirements

### Requirement: A read of a time-dependent field addresses only the slices it holds
Reading a time-dependent field at a time SHALL place that time between two neighbouring slices of the field, computed in double precision from the field's time interval, so that a time at the interval's start or end places all its weight on the first or last slice, and no read addresses a slice the field does not hold, whatever the floating-point bounds of the interval. A read that accepts a time outside the interval SHALL read it at the nearer end; a read that refuses such a time keeps refusing it. A non-finite time, an interval of zero or negative length, and a field of no slices SHALL be refused with a named error.

#### Scenario: A read at either end of any interval returns the end slice
- **WHEN** a time-dependent field of at least two slices, on a time interval with any finite bounds, is read on its own grid
- **THEN** the read succeeds and its values match the slices at their inclusive times, the end slices included, to rounding

#### Scenario: A slice position cannot address a missing slice
- **WHEN** positions are taken for any finite times, inside, at the ends of, or outside a time interval of any finite bounds, among any positive slice count
- **THEN** each lies between two neighbouring slices the field holds, with a weight from zero to one, a time at or beyond either end reads that end's slice exactly, and a non-finite time, an empty or non-finite interval, or no slices is refused
