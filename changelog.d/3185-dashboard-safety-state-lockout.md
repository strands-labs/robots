### Added: a `Lockout` value type states what a dashboard may say about an e-stop

`strands_robots.dashboard.safety_state` is a pure value module for what the
operator dashboard is entitled to claim about a fleet-wide e-stop lockout. A
frozen `Lockout` carries the `state` (`locked` / `clear` / `unknown`), `since`,
`by` and a human `reason`, and renders to wire fields through `as_fields()`.
The transitions that fold mesh events into it (`apply_event`,
`note_command_accepted`) are pure functions with no mesh or transport
dependency, so the vocabulary the UI badge draws from is testable on its own and
the same `Lockout` can back both the REST snapshot and the websocket snapshot
without the two disagreeing.
