### Tests: the harness#361 XPASS marker cites the refusal the driver actually returns

The strict-xfail cell in
`tests/drivers/test_g1_send_action_succeeds_on_a_healthy_wired_driver.py`
documented its deferral by quoting the shipped refusal.  When #2865 tightened
that refusal (dropping the parenthetical `(harness#361 PR-C)` because the PR
had landed and the issue reference alone is the resolvable pointer), the test
file - which merged two hours later as #2861 - kept the older wording.  Pytest
treats an `xfail` `reason` as descriptive prose and never asserts it, so the
drift was silent: a reader following the marker to the shipped refusal would
find a substring that no longer appears.

Two changes hold the citation in place:

1. The docstring header and the `pytest.mark.xfail(reason=...)` argument are
   rewritten to the current refusal text
   (`"FSM id unknown - motion-switcher source has not been wired; see issue
   #2765 for the wire-side decision"`).  The XPASS trigger is unchanged - it
   still fires when a motion-switcher decoder gives `_fsm_id` a producer -
   only the human-readable citation is corrected.
2. A new cell,
   `test_the_xfail_reason_quotes_the_shipped_refusal_verbatim`, reads the
   marker's `reason` off the test object and requires the driver's actual
   refusal text to appear verbatim inside it.  A future wording change on
   either side now breaks a cell rather than remaining silent, matching the
   pattern #2872 established for driver-docstring deferral citations.

On the wired-FSM day the refusal disappears; the XPASS cell fails
(`strict=True`), `test_the_current_refusal_still_names_the_fsm_and_the_motion_switcher`
fails on the missing-refusal side, and the new grading cell also fails
because `refusal["status"]` is no longer `"error"`.  All three point the
author of the wiring commit at the marker in the same run.
