## Context

See proposal.md — Why for the motivation, and `specs/operator-comparison/spec.md`
for the requirements this design has to be able to satisfy.

The design-relevant state:

```
benchmark()                     SpectralSVR/operators/benchmark.py:16
  target computed once at :47, fit bracketed at :50-52, both .cpu() before metrics
OperatorDataset                 SpectralSVR/operators/dataset.py:44
  from_fields samples the input at sensors, keeps the output as a Basis
Operator                        SpectralSVR/operators/__base.py:18
  fit / predict, both abstract; order enforced per-adapter
build_operators                 scripts/run_benchmark.py:227
  drops an operator whose optional dependency is missing, with a warning
EXEMPT                          tests/test_oracle_coverage.py:27
  "operators/test_operators.py"  # thin adapters over anchored model/basis layers
```

| fact | consequence |
| --- | --- |
| `tests/operators/test_operators.py` carries 5 `oracle_smoke` + 1 `oracle_consistency` and no anchor marker | the `EXEMPT` entry is load-bearing; deleting it fails `test_every_module_has_an_anchor` on the spot |
| `tests/_anchors.py:antiderivative_pair` already returns a closed-form `(ut, u)` | an analytic anchor for the operator path costs a test edit, not new mathematics |
| `SpectralSVROperator` learns exactly `ut -> u` | that closed form is the operator's own oracle, not a neighbouring one |
| neither adapter's `fit`/`predict` order check survives `python -O` | the guarantee is development-time, and `test_predict_before_fit_raises` pins the `AssertionError` itself |

## Goals / Non-Goals

**Goals:**

- Land the capability against the harness as built, not as one might wish it were.
- Leave the anchor-policy guard telling the truth about `operators/`, on this
  branch, with the suite green.
- Keep the added prose to the part that cannot be carried by a name.

**Non-Goals:**

- Changing what the harness computes. Every number the benchmark reports comes
  out of the same arithmetic; what moves is which object carries a method and
  which states can be constructed.
- Tuning, seeds, provenance — everything on the deferred list in proposal.md.
- Modelling away the remaining documented weakness. The table still rests on
  untuned hyperparameters at a single seed, and no type can fix that; it is
  stated in prose because prose is the only place it can live.

## Decisions

### The exemption is withdrawn and its anchor lands here

Task 4.1 offered re-justify or withdraw, and task 4.2 asked for withdrawal, a
follow-up, and a green suite at once. Those three cannot hold together: the
`EXEMPT` entry is the only reason the guard passes for that module, so removing
it without an anchor turns the branch red.

Withdrawing and anchoring in the same change is the resolution. The anchor is
cheap because the closed form already exists — `antiderivative_pair` gives an
exact `(derivative, antiderivative)` pair, which is precisely the map
`SpectralSVROperator` is trained on — so this is a test upgrade rather than new
mathematics.

*Alternative rejected — re-justify and keep.* Consistent with "documentation
only", and it leaves the headline result unanchored while the change whose whole
premise is that gap declares itself finished. Rewriting the justification would
make the exemption accurate without making it less of a hole.

*Alternative rejected — withdraw, anchor in a follow-up.* Leaves
`test_oracle_coverage.py` failing until that change lands. A red guard is worse
than the exemption it replaced, because a failing suite gets muted.

This widens the change past "no code changes": one test is rewritten, one
`EXEMPT` entry is deleted. No library source and no behaviour changes, so no
number can move. proposal.md's framing and section 4's tasks are amended to
match. The collected test count no longer holds still either: sections 6 and 7
add tests and the operator split rewrites call sites, so task 8.1 checks that
every test which disappears is one those sections deliberately replaced.

### The anchor scores against the closed form, not against the generator

`test_spectralsvr_operator_learns_antiderivative` is `oracle_consistency` today
because it scores predictions against `test.targets(query)` — the library's own
generated field. Both sides of that comparison come from the code under test.

The anchor keeps training on generated data and replaces only the scoring set:
held-out inputs and expected outputs both taken from `antiderivative_pair`, so
the expected values are a closed form the library had no hand in.

Its tolerance is not a rounding bound. The operator is a fitted model, so the
residual is learning error, and the assertion has to reflect that honestly: the
margin is **measured first and recorded**, then asserted with headroom, rather
than tightened until the test happens to pass. A tolerance discovered by tuning
until green asserts nothing.

### The scope-of-claim caveat goes where the table is produced

The capability requires documentation stating that the harness fixes data and
metrics but not tuning, and that figures come from one seed. Two candidate homes:
`benchmark()`'s docstring, and `scripts/run_benchmark.py`'s module docstring.

It goes in the script. `benchmark()` is a library function that another caller
could drive over many seeds with tuned operators, so the single-seed, untuned
caveat is a property of how this script drives it, not of the function.
`benchmark()`'s docstring gets the measurement protocol — one target, one metric
suite, `fit_seconds` around `fit` alone — which is true wherever it is called.

### Prose is added only where a name cannot carry the meaning

The project's standing preference is minimal prose in comments and docstrings,
with meaning expressed through names and structure. This change adds prose by
design, which makes the boundary worth stating rather than assuming.

What a name can carry stays a name. What cannot — that hyperparameters are
untuned on both sides, that a figure is one seed, that a fit time includes one
adapter's lazy import and not the other's — is irreducible: no signature
expresses it, and a reader who misses it misreads the table. That much goes in
the docstring, at minimum length. Everything else lives in the capability, which
is where a reader goes for what the harness guarantees.

## Risks / Trade-offs

- **The anchor could be flaky.** A fitted model's error varies with the draw, and
  an anchor that fails intermittently gets deleted rather than investigated. →
  Measure across a seed range before choosing the bound, record the worst
  observed error in the change, and set the assertion with headroom above it.

- **The change became BREAKING after the capability was written.** → Writing the
  guarantees down made three of them read as apologies: disjointness the caller
  must remember, an order enforced by an assertion, a row that can vanish with a
  warning. Modelling them away is cheaper than specifying them as defects and
  owing a follow-up. The cost is that "no number moves" stops being provable
  from the diff and needs a base-run comparison instead.

- **Specifying weaknesses reads like endorsing them.** The capability now states
  that disjointness is unverified and that a row can go missing. → Both are
  written as what the harness does, with the gap visible, which is what makes
  them findable later. Recorded as Non-Goals here so the follow-up is owed.

- **`test_exempt_list_is_current` guards the EXEMPT list in both directions.** It
  asserts each entry exists and genuinely lacks anchors. → Removing the entry
  satisfies it; the risk is only that the two guard tests must be re-run after
  the edit, which task 4.4 requires explicitly.

### Invalid states are removed where a type can carry the rule

Three of the capability's guarantees were, as written, requests to the caller.
Each becomes a constructor's job instead.

*Fit order, as typestate rather than inheritance.* `Operator` and
`FittedOperator` become `typing.Protocol`s, not base classes: `Operator` requires
`name` and `fit`, `FittedOperator` requires `predict`. An adapter satisfies them
structurally and inherits from nothing, so `SpectralSVROperator.fit` returns a
`FittedSpectralSVR` that is a separate value carrying `predict` and the trained
model. The unfitted value has no prediction method at all, so predicting before
fitting is rejected where the call is written, and both
`assert self._model is not None` lines are deleted rather than documented.

The two protocols are the two states, and `fit` is the only transition between
them — the shape a typestate encoding takes in a language without affine types.
What Python cannot enforce is *linearity*: nothing stops a caller fitting twice
and holding two fitted values, or keeping the unfitted operator alive alongside
its fitted result. Neither is harmful here — `fit` has no side effect on the
operator — so the encoding buys the ordering guarantee and stops there.

*Alternative rejected — an abstract `FittedOperator` base.* It delivers the same
ordering guarantee, but forces every adapter through an inheritance chain for a
single method, and makes third-party adapters depend on our class rather than on
the shape of the interface. `explicit-band-type` made the same call when it
replaced an abstract `Band` with a closed union of dataclasses.

*Dataset consistency.* `OperatorDataset` validates at construction — sensor rows
against the sampled width, function count against the output field. Python has
no way to make the mismatch unrepresentable in the type, so this is the smart
constructor pattern, the same boundary `WavenumberCutoff.__post_init__` settles
for a negative width.

*Split disjointness.* `benchmark` stops taking two loose datasets and takes a
`Split`, which only a partitioning function produces. `disjoint_split` moves out
of `scripts/` into the library as that constructor. Two overlapping datasets can
still be built individually; what cannot be built is a `Split` from them.

*A short table.* Resolving a requested operator whose dependency is absent
raises. A table missing a requested row is reachable only through an explicit
opt-in, so the default path cannot produce one quietly.

*Alternative rejected — validate inside `benchmark`.* Checking disjointness and
dataset consistency on entry catches the same mistakes, one call later, and
leaves every other caller of these types unprotected. The type is the earlier
boundary.

### Scenario coverage is enforced, not audited by hand

The capability states seventeen scenarios and, before this section, seven had a
test. Rather than close the gap once and let it reopen, a guard parses the delta
specs and reports any scenario with no test naming it, with deliberate
exemptions listed and reasoned. That rule belongs to `test-oracle-convention`
because it binds every future change, not only this one.

Scenarios that range over inputs are fuzzed rather than pinned to one literal,
and the harness gets a state machine, because several requirements constrain what
stays true *across* a sequence — a target that does not vary with the operator
list, one row per operator however many are run — which single-shot cases cannot
reach.

## Migration Plan

Measure the anchor's error before writing its assertion. Then the anchor, then
the `EXEMPT` deletion — in that order, so the guard never sees a module without
one. Docstrings after, since they are inert.

The constructive modelling follows, innermost first: the dataset's own
constructor, then `Split`, then the operator split, then operator resolution —
each one leaving the suite green before the next begins, since every step has
callers in `scripts/` and in the tests. The scenario guard lands last, because it
reports against the capability as finally written.

Rollback is a revert; nothing persists. The API breaks, so anything outside this
repository holding an `Operator` would need the two-line change from
`op.fit(d); op.predict(...)` to `fitted = op.fit(d); fitted.predict(...)`.

## Open Questions

- Whether the four dataset definitions in `scripts/run_benchmark.py` deserve
  their own capability, separate from the harness that consumes them. They are
  the subject of `benchmark-reproducibility`, which will have to describe them to
  do its job; deciding now would either duplicate that or constrain it. Safely
  deferrable: it changes neither this change's requirements nor its tasks.
