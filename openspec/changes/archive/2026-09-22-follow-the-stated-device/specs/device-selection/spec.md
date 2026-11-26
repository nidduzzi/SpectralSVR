## Purpose
Defines who decides where the library places work: the order in which an explicit argument, a process-wide setting and the library's own opinion are consulted, and what happens when a caller's random source disagrees with the result. States that the library carries no device preference of its own, so a caller who says nothing gets what the surrounding process says rather than what the library would prefer.

## ADDED Requirements

### Requirement: The stated device wins, and the library states none
Resolving a device SHALL consult exactly two sources, in order: a device given at the call site, then the process-wide default. The library SHALL NOT introduce a third answer of its own, and SHALL NOT prefer an accelerator because one happens to be present.

A caller who names no device is still making a choice — that the surrounding process should decide — and that choice SHALL be honoured. Where the process has expressed no preference, the answer is whatever the platform's own default is, not a device the library selects on the caller's behalf.

This is a deliberate refusal to be clever. A per-component preference cannot be honoured together with a process-wide one, because a process default naming the host is indistinguishable from no preference at all; any rule that accelerated "by default" would silently override a caller who asked for the host. The library therefore offers no defaults to be overridden.

#### Scenario: A stated device is used
- **WHEN** a device is given at a call site
- **THEN** work is placed there, whatever the process-wide setting says

#### Scenario: A process-wide setting is followed
- **WHEN** no device is given at the call site and the process has set one
- **THEN** work is placed on the process's device, for every device the platform supports rather than only the one the library was written against

#### Scenario: Stating a device does not change what is drawn
- **WHEN** the same seed and the same device are used, once by naming the device at the call site and once by setting it as the process default
- **THEN** the values drawn are identical, so a caller may move between the two ways of saying where work goes without changing a result

#### Scenario: The host is not overridden
- **WHEN** the process states the host as its device and an accelerator is present
- **THEN** work is placed on the host, because a stated preference is not a gap to be filled

### Requirement: Placement is statable everywhere work is placed
Every entry point that places work SHALL accept a device, so that a caller can direct one part of a workload without directing all of it. An entry point that resolves a device but offers no way to state one SHALL be treated as incomplete, because it forces the caller to move the whole process to move one computation.

#### Scenario: Parts of a workload are directed separately
- **WHEN** a caller wants one stage of a workload on an accelerator and another on the host
- **THEN** each is stated at its own call site, with no process-wide setting required and no effect on the other

#### Scenario: Generating a problem can be directed
- **WHEN** a caller generates a problem and states a device
- **THEN** the generation is placed there, so a workload large enough to benefit from an accelerator can reach one

#### Scenario: Generating coefficients can be directed
- **WHEN** a caller generates coefficients directly, without going through a problem
- **THEN** the generation is placed where they state, so the lower entry point is directable on the same terms as the higher one rather than reachable only by moving the whole process

### Requirement: Work derived for placed data follows that data
Where the library builds a value for data that already exists -- an index, a scale factor, a comparison constant -- it SHALL take the device from that data rather than from the process default. Asking the process again is only correct while everything follows the process; once placement is statable, the answer can name a device the data is not on.

This is distinct from a factory that materialises a caller's data. Such a factory correctly follows the process default when the caller states nothing, and SHALL keep doing so.

#### Scenario: A workload placed against the process default still runs
- **WHEN** a caller states one device for their work while the process default names another
- **THEN** generation, indexing, evaluation and resizing all complete, because each derived value is built where the data it meets already is

### Requirement: The accelerator is named, not detected ad hoc
The library SHALL expose one way to name the platform's accelerator, resolving to the host where none is present. Callers SHALL NOT be expected to write their own availability check, and the library SHALL NOT contain more than one.

#### Scenario: Naming the accelerator is a statement, not a search
- **WHEN** a caller wants the accelerator without knowing which kind the platform has
- **THEN** naming it yields whichever accelerator is present, and the host where none is, without the caller testing for a particular vendor

### Requirement: A random source that disagrees is refused clearly
Where a caller supplies a random source, it SHALL be placed on the device the work is placed on. A disagreement SHALL be refused, and the refusal SHALL name both devices and which of the two the caller is expected to move.

Refusing is correct rather than unhelpful: the two cannot be reconciled without either relocating the caller's work or drawing different numbers from the same seed, and both are decisions the caller must make knowingly.

Where the library builds a random source for itself, because the caller supplied none, it SHALL build it on the device the work is placed on. A library that refused its own source would be reporting its own defect to a caller who chose nothing.

#### Scenario: A source the library builds for itself agrees by construction
- **WHEN** a caller invokes an operation that needs a random source, supplies none, and states a device or has one stated for the process
- **THEN** the source the library builds is on that device, and the refusal above cannot fire against a choice the caller did not make

#### Scenario: A mismatched random source names both devices
- **WHEN** a caller supplies a random source on one device while work is placed on another
- **THEN** the generation is refused before any work is done, and the message names the device the source is on, the device the work is on, and that the source is what moves

#### Scenario: A stated index is part of the statement
- **WHEN** the source and the work both name an index and the two differ
- **THEN** the generation is refused, because they are separate memories and a draw on one cannot be read from the other; where either leaves its index unstated, the two agree, since an unstated index means whichever the process is using

#### Scenario: An agreeing random source is unremarkable
- **WHEN** the random source and the placement agree
- **THEN** generation proceeds with no diagnostic, whether they agree because the caller stated both or because both followed the process
