### Fixed: `targets_from_action` refuses a `control_period` that would disable its own gate

The speed gate sizes a step as `ceiling * control_period`. A `nan` period made
`allowed` `nan`, every `step > allowed` comparison false, and the gate vanished
without a word - inside a success envelope, which is the one failure a caller
could not detect from the result. The parameter now goes through the shared
`positive_finite_number_error` domain, and explicit `None` remains the way to
ask for no gate.
