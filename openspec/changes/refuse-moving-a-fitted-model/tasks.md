## Execution order

```
1 state the rule ── 2 the refusal ── 3 coverage ── 4 verification
```

A chain. Section 1 writes tests that fail against today's setter; nothing in
section 2 is written until they do, so the refusal is pinned by a test that was
red before it existed rather than written around code that already passes.

## 1. State the rule before the code obeys it

- [ ] 1.1 Assert a fitted `LSSVR` refuses a device change, and that the message names the device it was fitted on, the device it was asked to move to, and the constructor as where to state it. Verify the test **fails first**, and record what today's code does instead: the assignment is accepted and `predict` later raises `RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu, ...`. Assert the message's content, not merely that something raised -- a test checking only for an exception would pass against the present behaviour once `predict` is reached
- [ ] 1.2 Assert a fitted `FNN` refuses identically. It holds an `nn.Module` rather than loose tensors and today fails with a different message, `... mat1 is on cuda:0, differ`, so a single-subclass test would leave the second kind of model uncovered while appearing to cover the rule. Verify this one fails first too
- [ ] 1.3 Assert an **unfitted** model still accepts a device change. `tests/model/test_lssvr.py:218` is the one `.device = ` assignment in the repository and is on an unfitted model; verify it passes unchanged rather than assuming it does
- [ ] 1.4 Where no second device exists the refusal cannot be provoked with two real devices; verify each test that needs one skips with a reason naming what it needs rather than passing vacuously

## 2. The refusal

- [ ] 2.1 The `device` setter on `MultiRegression` refuses when `self.trained`, and accepts otherwise. Use the `trained` property the base already declares -- both subclasses implement it, so no new abstract method is added and no subclass enumerates its own fitted state. Verify the tests from section 1 now pass and the whole suite is green
- [ ] 2.2 Verify the setter still resolves what it is given, so a string and a `torch.device` are both accepted on an unfitted model and `.device` reads back as a resolved `torch.device`
- [ ] 2.3 Verify by mutation that removing the refusal restores the old behaviour: the assignment is accepted and the failure returns to `predict`. A refusal no test can kill is worth nothing

## 3. Coverage

- [ ] 3.1 Mark each of the three `device-selection` scenarios this change adds with `@pytest.mark.scenario` and remove its `IN_FLIGHT` entry from `tests/test_scenario_coverage.py` in the same commit. All three enter `EXEMPT` when the artifacts land, because a stated scenario with neither a test nor an entry leaves the suite red; verify the stale-exemption test forces each removal rather than tolerating it
- [ ] 3.2 Verify no scenario of this capability sits in `EXEMPT` at the end, and that the entries belonging to the parent change are untouched
- [ ] 3.3 Give each new test exactly one `oracle_*` marker and either fuzz it or mark it `no_fuzz` deliberately. The rule ranges over the model kind rather than over numeric input, so fuzzing over the two kinds is the form that fits; verify `tests/test_oracle_coverage.py` still passes for every module touched

## 4. Verification

- [ ] 4.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [ ] 4.2 Verify no source file gained an explanatory prose comment, and that prose was removed from what this change touched rather than merely not added. The refusal must read from the condition and the message, not from a comment explaining them
- [ ] 4.3 Verify the scope held: `LSSVR` and `FNN` are unchanged, no device index validation was added, and construction behaves exactly as before
