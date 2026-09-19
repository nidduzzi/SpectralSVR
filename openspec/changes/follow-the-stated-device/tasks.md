## 1. State the rule before changing what obeys it

- [x] 1.1 Replace `tests/utils/test_utils.py::test_resolve_device_default_matches_availability`, which asserts `expected = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` and so pins the behaviour being removed. The replacement asserts the precedence: a stated device is used, and with none stated the process default is. Verify the new test **fails** against the current resolver before anything else changes, and record that failure
- [x] 1.2 Assert a stated host is honoured while an accelerator is present -- the case the current resolver gets wrong and the one a user hit. Verify it fails first. Do **not** add per-test restoration: `tests/conftest.py` already carries an autouse fixture that snapshots and restores the process default around every test, for this exact reason. Verify its reach instead, by making the new assertion fail deliberately and confirming the rest of the suite still passes
- [x] 1.3 Where no accelerator exists the disagreement in 1.2 cannot be constructed; verify the test skips with a reason naming what it needs rather than passing vacuously

## 2. The seam

- [x] 2.1 Resolution consults a stated device, then the process default, and stops. Verify the whole suite is **324 passed, 1 failed** beforehand with only the 1.1 test failing, and green afterwards -- that count is measured, so a second failure means something beyond this change moved
- [x] 2.2 Add the accelerator helper over torch's own accelerator API, falling back to the process default where none is present. Verify it names whichever accelerator the platform has rather than testing for one vendor, and that it does not raise where none exists. Today's hardcoded `cuda:0` silently means no acceleration on a Mac, so the fallback is the behaviour under test, not an edge case
- [x] 2.3 Verify the helper's result is usable where a device is expected even though it carries no index: it is not `==` `torch.device("cuda:0")`, and placing a tensor on it lands on `cuda:0`. Confirm no device equality exists in the library -- every site is `.to(device=...)` -- so the missing index changes nothing internally

## 3. Statable placement

- [ ] 3.1 The problem generators accept a device, on the base and both subclasses; the Burgers path stops resolving without one. Verify a caller can place generation on an accelerator, which is impossible today at any price, and that stating nothing leaves behaviour unchanged
- [ ] 3.2 The basis generation entry points accept a device too -- on the abstract base and on `FourierBasis`. They place work and accept none today, so without this the requirement that placement is statable wherever work is placed would be stated and unmet. Verify a caller reaching the basis directly can direct placement without moving the process, and that every existing caller and test of those entry points still passes unchanged when nothing is stated
- [ ] 3.3 Verify the reachable gain is real rather than assumed, by timing the numerical Burgers path on both devices at a size where it matters and at a size where it does not. Expect the accelerator to win by roughly an order of magnitude at `(500, 64, 800)` and the host to win by roughly half again at `(50, 16, 100)`. Assert the **direction and rough ratio**, never a figure: host timings move by a factor of two to three between runs -- `(500, 64, 800)` measured 4880 ms and then 2278 ms on the same idle machine -- while accelerator timings repeat to a decimal. A reversed winner is a finding; a different number is not
- [ ] 3.4 Forward `None` into a factory rather than resolving it where no concrete device is needed -- a factory resolves `None` itself, while `.to(None)` is a no-op rather than a move. Verify each remaining resolution call materialises a device for a `.to(...)` or for storage

## 4. The hardcoded host

- [ ] 4.1 The model base stops hardcoding the host and routes through the seam; the stale comment acknowledging it goes. Verify the existing device-setter test still passes -- it pins the host explicitly, so it is unaffected -- and that with nothing stated the model now follows the process
- [ ] 4.2 Verify nothing else in the library carries its own device answer, by searching for hardcoded device construction outside the seam and the helper

## 5. The disagreeing random source

- [ ] 5.1 Refuse a random source whose device differs from the resolved device, naming both and which one moves. Verify a test asserts the message names both devices, not merely that something raised: torch's own `Expected a 'cuda' device type for generator but found 'cpu'` already raises, so a test that only checks for an exception would pass without this change
- [ ] 5.2 Record in the test why this is a run-time refusal rather than a type one: both generators are the same type and the device is a runtime attribute, so no checker can see it. Verify by asserting `type(torch.Generator()) is type(torch.Generator(device=...))`, which keeps the reason checkable rather than asserted in prose
- [ ] 5.3 Verify an agreeing random source produces no diagnostic, whether it agrees because both were stated or because both followed the process. Assert the **absence of a warning**, with `pytest.warns(None)` or by capturing the logger, not merely the absence of an exception: a test that only checks nothing raised would pass against code that warns on every call, which is the vacuous form this scenario is most at risk of
- [ ] 5.4 The model builds its own random source where the work is, rather than on the host regardless. `inverse` and `inverse_coeff` construct `torch.Generator().manual_seed(42)` with no device today, so under a stated accelerator the library trips 5.1's refusal against a source the caller never supplied. Both gain a device argument like the other entry points, and build the source on the resolved device. Verify the refusal cannot fire when the caller supplied nothing, and that the existing behaviour is unchanged when nothing is stated

## 6. The application states its device

- [ ] 6.1 The benchmark script states the device it runs on rather than inheriting whatever the last import left. Verify it runs with all four datasets and both operators -- the configuration that fails today with `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'`, because an operator import sets a process default the script never asked for
- [ ] 6.2 Verify the metrics are unchanged within what was measured: identical to about six significant figures, `r2` moving in its last digit only (0.081452 against 0.081453). A larger move is a finding to chase, not a tolerance to widen

## 7. Scenario coverage

- [ ] 7.1 Mark each scenario of the `device-selection` delta with `@pytest.mark.scenario` as its task completes and remove its `IN_FLIGHT` entry in the same commit. All eight enter `EXEMPT` when the change is proposed, because artifacts land before implementation and a stated scenario with neither a test nor an entry leaves the suite red; verify the stale-exemption test forces each removal rather than tolerating it
- [ ] 7.2 Fuzz what ranges over inputs rather than pinning one case: the precedence over stated devices and process settings, and the refusal over the device pairs that should be refused. Verify each property fails under a mutation of the behaviour it describes
- [ ] 7.3 Verify no scenario of this capability sits in `EXEMPT` at the end

## 8. Verification

- [ ] 8.1 Run `just check`; verify clean
- [ ] 8.2 Verify no source file gained an explanatory prose comment, and that prose was removed from what this change touched rather than merely not added
- [ ] 8.3 Verify the superseded change is withdrawn rather than left live: `align-device-resolution`'s branch and artifacts are removed, and everything carried from it -- the device-versus-precision asymmetry, the measured noise floor, the stale-comment cleanup -- is present here
