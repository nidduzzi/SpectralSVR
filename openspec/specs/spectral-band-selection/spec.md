# spectral-band-selection Specification

## Purpose
Defines how a caller states which modes an operation keeps when it band-limits a coefficient grid, what each form of that statement guarantees about whether the result describes a real field, and which statements can be made at all.

## Requirements

### Requirement: A band is stated, not counted
An operation that keeps some modes of a coefficient grid and discards the rest SHALL take a band, not a bare count. The band SHALL carry which modes it selects, so that a caller reading the call site can tell what is kept without knowing the operation's internal convention for turning a number into a set of modes.

#### Scenario: The kept set is evident at the call site
- **WHEN** a caller states a band and applies it to a coefficient grid
- **THEN** the modes kept follow from the band alone, not from an unstated convention about how a count is distributed across positive and negative modes

#### Scenario: A bare count is not accepted
- **WHEN** a caller passes a plain number where a band is expected
- **THEN** the call is rejected, because a number alone does not say which modes it means

### Requirement: A band that promises a real result cannot be stated wrongly
A band that guarantees a real-valued result SHALL be constructible only in forms that actually deliver one. It SHALL NOT be possible to state such a band that selects one member of a conjugate pair without the other, because no result satisfies that request: modes pair around a lone zero mode, so a kept set with a real inverse transform is either odd-sized or the whole grid.

#### Scenario: A symmetric band about the zero mode
- **WHEN** a caller states a band by the highest magnitude it keeps
- **THEN** the kept set is that magnitude's positive and negative modes together with the zero mode, and applying it to a conjugate-symmetric grid yields a grid that is still conjugate-symmetric

#### Scenario: The whole grid is a statable band
- **WHEN** a caller states the band that keeps everything
- **THEN** it is accepted and the operation is a no-op, including on an even-sized grid where the highest magnitude has no positive counterpart and is its own conjugate partner

#### Scenario: An unsatisfiable band cannot be stated
- **WHEN** a caller tries to state a real-guaranteeing band that would keep an even number of modes short of the whole grid
- **THEN** no such band can be constructed, rather than one being constructed and silently returning a grid that is not conjugate-symmetric

### Requirement: An operation can require a band that guarantees a real result
An operation that must return a coefficient grid describing a real field SHALL be able to require a band that guarantees one, and a band making no such guarantee SHALL be refused where the call is written rather than discovered in the result. The guarantee SHALL be carried by the band itself, so an operation states the requirement once and needs no check of its own.

#### Scenario: A guaranteeing band is accepted
- **WHEN** an operation that must return a real field is passed a band stated by the highest magnitude it keeps, or the band that keeps everything
- **THEN** it is accepted, because both keep a set closed under negation

#### Scenario: A band without the guarantee is refused where it is passed
- **WHEN** that same operation is passed a band stated as an exact number of modes
- **THEN** it is refused at the call site, rather than after the operation has produced a grid that is not conjugate-symmetric

#### Scenario: Band-limiting itself does not require the guarantee
- **WHEN** a caller band-limits a coefficient grid that was never conjugate-symmetric
- **THEN** the operation still accepts a band stated as an exact number of modes, because truncating such a grid is a legitimate request rather than a defect

### Requirement: Truncation without a realness promise remains available
A band form that keeps an exact number of modes SHALL remain available, and SHALL state that its result is not guaranteed to describe a real field. Coefficient grids that are not conjugate-symmetric are legitimate — a predicted grid is not symmetrised — so an operation that truncates one is not defective merely because its result is complex.

#### Scenario: An exact count is honoured
- **WHEN** a caller states a band by an exact number of modes and applies it
- **THEN** exactly that many modes are kept, whatever the parity of the number

#### Scenario: The absent guarantee is stated
- **WHEN** a caller reads what that form promises
- **THEN** it says the result is not guaranteed to be conjugate-symmetric, and says which other form to state when a real result is required

#### Scenario: Choosing it is deliberate
- **WHEN** a caller wants the behaviour that does not guarantee a real result
- **THEN** they reach it by naming that form explicitly, so it cannot be selected by accident from an ambiguous argument
