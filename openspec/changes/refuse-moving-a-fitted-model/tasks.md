## Execution order

```
1 state the rule ── 2 the refusal ── 3 coverage ── 4 verification
```

A chain. Section 1 writes tests that fail against today's setter; nothing in
section 2 is written until they do, so the refusal is pinned by a test that was
red before it existed rather than written around code that already passes.

## 1. State the rule before the code obeys it

- [x] 1.1 Assert a fitted `LSSVR` refuses a device change, and that the message names the device it was fitted on, the device it was asked to move to, and the constructor as where to state it. Verify the test **fails first**, and record what today's code does instead: the assignment is accepted and `predict` later raises `RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu, ...`. Assert the message's content, not merely that something raised -- a test checking only for an exception would pass against the present behaviour once `predict` is reached
  - Done in `tests/model/test_fitted_placement.py::test_a_fitted_model_refuses_to_be_moved`, fuzzed over both kinds. Failed first with `DID NOT RAISE RuntimeError`. Today's behaviour recorded on this machine: the assignment is accepted, `device` reads `cuda`, and `predict` raises `Expected all tensors to be on the same device, but got mat2 is on cpu, different from other tensors on cuda:0`
- [x] 1.2 Assert a fitted `FNN` refuses identically. It holds an `nn.Module` rather than loose tensors and today fails with a different message, `... mat1 is on cuda:0, differ`, so a single-subclass test would leave the second kind of model uncovered while appearing to cover the rule. Verify this one fails first too
  - Covered by the same fuzzed test, whose strategy draws `FNN` as well as `LSSVR`, and by `test_every_kind_of_model_refuses_alike`. Both failed first. Today's `FNN` behaviour recorded: accepted, then `predict` raises `... mat1 is on cuda:0, different from other tensors on cpu`
- [x] 1.3 Assert an **unfitted** model still accepts a device change. `tests/model/test_lssvr.py:218` is the one `.device = ` assignment in the repository and is on an unfitted model; verify it passes unchanged rather than assuming it does
  - `test_lssvr_device_setter` passes unchanged. `test_an_unfitted_model_is_directed_freely` adds both kinds and both devices
- [x] 1.4 Where no second device exists the refusal cannot be provoked with two real devices; verify each test that needs one skips with a reason naming what it needs rather than passing vacuously
  - With `CUDA_VISIBLE_DEVICES=""`: 2 passed, 2 skipped, reason `naming both devices needs a device other than the host to move to`

## 2. The refusal

- [x] 2.1 The `device` setter on `MultiRegression` refuses when `self.trained`, and accepts otherwise. Use the `trained` property the base already declares -- both subclasses implement it, so no new abstract method is added and no subclass enumerates its own fitted state. Verify the tests from section 1 now pass and the whole suite is green
  - Four lines in the setter, no helper, no new abstract member. Section 1 tests pass; `just check` 344 passed
- [x] 2.2 Verify the setter still resolves what it is given, so a string and a `torch.device` are both accepted on an unfitted model and `.device` reads back as a resolved `torch.device`
  - **Amended.** The premise was wrong: the old setter never resolved anything, so `m.device = "cpu"` stored the string `'cpu'`, measured before the change. The setter now resolves with `resolve_device`, as construction does, which the refusal message also needs. `test_an_unfitted_model_is_directed_freely` failed first on `assert 'cpu' == device(type='cpu')` and now passes. The getter gained its `-> torch.device` annotation
- [x] 2.3 Verify by mutation that removing the refusal restores the old behaviour: the assignment is accepted and the failure returns to `predict`. A refusal no test can kill is worth nothing
  - With the refusal replaced by `if False:`, the refusal test, the likeness test and the state machine fail; the unfitted test survives, as it should. The failure returns to `predict` with both original torch messages. To test the state machine's invariant apart from its own rule's `pytest.raises`, the rule was stubbed out as well. With `build` as a rule, the invariant killed the mutant in 7 of 10 seeds. Moving `build` to `@initialize` raised that to 18 of 20 at 25 examples, and 50 examples gave 19 of 20 (0.57 s). The invariant alone is therefore not a certain kill. The two refusal tests kill it deterministically, so the suite as a whole always does

## 3. Coverage

- [x] 3.1 Mark each of the three `device-selection` scenarios this change adds with `@pytest.mark.scenario` and remove its `IN_FLIGHT` entry from `tests/test_scenario_coverage.py` in the same commit. All three enter `EXEMPT` when the artifacts land, because a stated scenario with neither a test nor an entry leaves the suite red; verify the stale-exemption test forces each removal rather than tolerating it
  - Before the entries came off, `test_no_exemption_is_stale` failed and named all three
- [x] 3.2 Verify no scenario of this capability sits in `EXEMPT` at the end, and that the entries belonging to the parent change are untouched
  - The parent's `device-selection` scenarios had no `EXEMPT` entries and none were touched. No scenario of this capability remains
- [x] 3.3 Give each new test exactly one `oracle_*` marker and either fuzz it or mark it `no_fuzz` deliberately. The rule ranges over the model kind rather than over numeric input, so fuzzing over the two kinds is the form that fits; verify `tests/test_oracle_coverage.py` still passes for every module touched
  - One refusal and one unfitted test are fuzzed over the kinds, marked smoke. The likeness test enumerates every kind, since the claim is a comparison across them, so it is `no_fuzz`. `DirectingAndFitting` is the module's anchor, `oracle_invariant`. `test_oracle_coverage.py` passes

## 4. Verification

- [x] 4.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
  - ruff, mypy (`no issues found in 29 source files`) and ty are clean; 344 passed
- [x] 4.2 Verify no source file gained an explanatory prose comment, and that prose was removed from what this change touched rather than merely not added. The refusal must read from the condition and the message, not from a comment explaining them
  - No comments added. The only `#` lines left in `model/__base.py` are the tool pragmas on lines the change does not touch. The new test module has no docstrings
- [x] 4.3 Verify the scope held: `LSSVR` and `FNN` are unchanged, no device index validation was added, and construction behaves exactly as before
  - `git diff` against `LSSVR.py` and `FNN.py` is empty. No index validation was added, and `__init__` is unchanged
