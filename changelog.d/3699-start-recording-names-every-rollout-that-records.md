### Changed: `start_recording` names every rollout that feeds the recorder

Every rollout that records is named, because `policy_running` - the flag the
recording guards read - is raised for each of them: `_announce_rollout` for
`run_policy` and `start_policy`, whose rollouts feed the recorder through the
same per-step hook, and `run_multi_policy`, whose synchronized loop calls
`add_frame` itself and which `describe()` advertises as the path for bimanual
data collection. Naming `run_policy` alone sent a caller who used either of the
other two looking for a defect that was not there.

`start_recording`'s success text states the rule where the recording begins
instead of the softer "Run policies to capture frames".
