### Fixed: the Device Connect `stop` RPC reports which rollouts it halted

`SimulationDeviceDriver.stop` lowered `policy_running` itself and answered a
fixed `"All policies stopped"`, so four different facts arrived as one sentence:
a halted rollout, an idle simulation, a world already torn down, and - because
the loop had no guard - a scene teardown racing it, which escaped as a
`RuntimeError` past the RPC instead of an envelope. None of the four named a
robot, so the answer could not be checked against `list_policies_running`.

It now routes every robot through the simulation's own `stop_policy` and reads
the verdict, naming the rollouts that really were in flight under `stopped` and
any that refused under `not_stopped`, graded through the same
`_reports_failure_to_stop` the mesh fleet stop reads. A simulation with nothing
to halt answers affirmatively-empty rather than erroring, for the reason that
branch states: a peer reported as "did not stop" when it had nothing to stop is
the false alarm that teaches an operator to ignore the warning.

`MuJoCoSimEngine.stop_policy` now carries its verdict as data (a `was_running`
`json` block) as well as in its sentence, matching `stop_task` and
`stop_teleoperate`, so an aggregating caller does not re-derive "was a rollout
in flight" from its own reading of the flag.
