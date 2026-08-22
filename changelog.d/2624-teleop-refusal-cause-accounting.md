### Fixed

- **A refused teleop frame is now counted under the guard that refused it, and each guard's
  rate-limited log line is spent on its own budget.**
  `InputReceiver` refuses a frame for six reasons; the apply-rate ceiling and the per-joint slew bound
  each reported themselves in `rate_dropped` and `slew_rejected`, while an E-stop lockout, a stale or
  missing frame timestamp, and a frame `validate_input_frame` refused all shared `rejected`. Beyond
  leaving a report unable to say which guard refused a stream, the warning each guard emits was
  rate-limited against that shared counter, so a follower that had already refused a few frames for
  one reason -- an ordinary clock-skewed start -- refused the next reason in complete silence, and the
  log line is the only place a refusal names the value it refused and the bound it exceeded. `stats`
  now reports `rejected_lockout`, `rejected_freshness` and `rejected_invalid` beside the unchanged
  `rejected` total, which always equals their sum, and the budget is spent per cause. No refusal
  condition changes: the same frames are refused, and the two causes that already had their own
  counter keep it. The `stats` docstring also enumerated an ACL check this path never makes while
  omitting the frame-validation refusal, and now names the causes it reports.
