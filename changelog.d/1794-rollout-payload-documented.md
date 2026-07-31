### Docs: `run_policy` / `eval_policy` document the json payload they return

The rollout facades return their facts in the result's `{"json": {...}}` content
block, but their `Returns:` sections had drifted far behind it: 18 of
`run_policy`'s 30 payload fields were undocumented, and all 22 of
`eval_policy`'s (which had no `Returns:` section at all). The omissions included
every field that detects a crippled rollout - `action_resolution_rate`,
`partial_action_failure_rate` and the async-RTC telemetry - so a caller had no
way to discover them and gated on the outer `status` instead. That reads clean
on exactly the rollouts worth catching: a policy driving 1 of a Panda's 8
actuators returns `status="success"` with `action_errors=0` (the one field that
*was* documented) and `partial_action_failure_rate=0.875`.

Both sections now enumerate every field, state that `status` reports whether the
call ran rather than whether the rollout achieved anything, and show the
index-free way to read the block - `next(b["json"] for b in result["content"] if
"json" in b)`. A hardcoded `content[1]["json"]` raises `IndexError` on an early
caller-error return, which carries a `text` block only. A new guard pins each
documented field list to the payload a real rollout produces, so the two cannot
drift apart again.
