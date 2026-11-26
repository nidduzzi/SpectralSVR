## ADDED Requirements

### Requirement: A fuzzed anchor's tolerance holds over everything it can draw
A fuzzed anchor's tolerance SHALL hold for every input its strategies can produce, not only for the ones a measurement happened to draw. Where an oracle's accuracy depends on a property of the drawn inputs, such as the conditioning of a matrix built from them, the fixture SHALL guarantee that property by construction, and the tolerance SHALL be set beneath a measurement taken over the inputs the fixture can now produce. A tolerance that a rare draw can exceed fails the suite for a reason unrelated to the behaviour under test, and a fuzzer will find that draw eventually.

#### Scenario: An oracle's accuracy does not depend on the luck of the draw
- **WHEN** a fuzzed anchor's oracle is accurate only for well-conditioned inputs
- **THEN** its fixture produces only well-conditioned inputs, so every draw is held to the same tolerance and a failure means the behaviour under test is wrong

#### Scenario: A draw found to exceed the tolerance is pinned
- **WHEN** a fuzzer has found an input on which an anchor exceeded its tolerance, and the fixture is corrected
- **THEN** that input is pinned as an example of the anchor, so the case that exposed the gap is checked on every run rather than only when the fuzzer happens to reach it again
