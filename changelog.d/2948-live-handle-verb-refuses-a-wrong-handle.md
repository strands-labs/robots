### Fixed: a live-handle G1 verb refuses a wrong driver handle instead of raising

The three verbs that read a cached DDS snapshot took the caller's handle on
trust. `g1_imu`, `g1_lidar_state` and `g1_lidar_summary` each reached
straight for `driver._snapshot(...)`, so an agent that passed the robot
*name* rather than the driver object got
`AttributeError: 'str' object has no attribute '_snapshot'` out of a `@tool`
whose module docstring promises that "every verb returns the structured
envelope and never raises". An omitted handle raised the same on
`NoneType`, and a handle carrying `_snapshot` as data rather than as a
method - a dataclass or a namespace built from a cache dump - reached the
call and raised `TypeError: 'NoneType' object is not callable`.

Their four siblings were already guarded: `g1_list_joints` and
`g1_list_motion_gates` take no handle at all, and `g1_fsm_admits` opens by
validating its `fsm_id` through `fsm_id_refusal`. The family was split
three ways on whether the first argument is checked, and the three
unguarded verbs are exactly the three that dereference it.

`snapshot_handle_refusal` joins the six refusal builders already in
`_g1_common.py` and answers in two branches, because the two cases need
different advice. An omitted handle is not something an agent can
synthesise from a robot name, so the refusal says so and points at the
driver object the mesh already holds; a handle of the wrong *type* names
the type it received, so a caller who passed a name sees which value
arrived. Both quote the verb and the parameter, matching `fsm_id_refusal`
beside them.

Nothing that was accepted before is refused now. The guard reports only for
a handle that would have raised on the very next line, and a driver whose
subscriber has not received a message yet still reports `present=False`
with every field `None` rather than a refusal - the verb does not conflate
"you gave me the wrong object" with "the robot has not sent one yet".

`tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py` derives
the graded population from the package rather than listing it: every public
`@tool` under `strands_robots.tools.g1` whose body dereferences
`_snapshot` must consult the shared refusal. A fourth verb planted
unguarded fires that sweep by name; the same plant with the population
hardcoded to today's three names is silent, which is why the population is
derived.

Refs `strands-labs/robots#358`: hardening for the neon-the-g1 verb bundle
port, after `g1_joint_reference` (`strands-labs/robots#2932`),
`g1_list_motion_gates` / `g1_fsm_admits` (`strands-labs/robots#2933`),
`g1_imu` (`strands-labs/robots#2939`), `g1_lidar_state`
(`strands-labs/robots#2941`) and `g1_lidar_summary`
(`strands-labs/robots#2943`).
