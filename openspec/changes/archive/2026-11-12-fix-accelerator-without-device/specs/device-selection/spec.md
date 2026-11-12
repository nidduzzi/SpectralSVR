## MODIFIED Requirements

### Requirement: The accelerator is named, not detected ad hoc
The library SHALL expose one way to name the platform's accelerator, resolving to the host where none is present. Callers SHALL NOT be expected to write their own availability check, and the library SHALL NOT contain more than one.

#### Scenario: Naming the accelerator is a statement, not a search
- **WHEN** a caller wants the accelerator without knowing which kind the platform has
- **THEN** naming it yields whichever accelerator is present, and the host where none is, without the caller testing for a particular vendor

#### Scenario: A build that supports an accelerator the machine lacks names the host
- **WHEN** the installed torch supports an accelerator but the process sees no such device
- **THEN** naming the accelerator yields the process default device, never the unavailable accelerator
