### Fixed: an IoT rule whose action fails no longer drops the message silently

`mesh.iot.rules.rules_for` gave the e-stop, events, fault and telemetry rules no `errorAction`, so when an action
failed (an SNS publish refused, a deleted bucket, a throttled table) AWS IoT dropped the message and nothing was
recorded where an operator would look. That is how an earlier SNS misconfiguration stayed invisible: 0 messages
reached SNS in 90 s after a real e-stop, with no error anywhere. Every generated rule now carries an `errorAction`:
`rules_for(..., error_action=)` when given, otherwise into S3 under `errors/<tenant>/<kind>/<ms>.json` when a
`s3_bucket` is set, else republished to the tenant's `strands/<tenant>/@error` topic. `apply_rules` now also
replaces a rule whose `errorAction` changed.
