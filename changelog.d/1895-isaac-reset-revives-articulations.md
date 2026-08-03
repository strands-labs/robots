### Fixed: Isaac reset() revives the articulation handles world.reset() kills on 6.0.x

On the pip Isaac Sim 6.0.x wheels, `world.reset()` tears down and rebuilds the
physics-tensor simulation view, and every registered robot's
`SingleArticulation` handle is left holding the torn-down view - so after ANY
reset, `get_joint_positions()` returned `None`, `get_observation()` degraded to
its documented silent-empty mode, and every consumer of post-reset joint state
broke (`examples/so101_curobo --backend isaac` died on a `KeyError` reading
`home_q`; any `reset_between=True` recording flow read empty joint state after
episode 1). Same defect family as #1798 - invalidate-on-stop; wrapper handles
need explicit re-init on 6.0.x - one layer up: #1798 fixed the scene-object
path, this fixes the articulation path. `IsaacSimulation.reset()` now probes
every registered robot's handle after `world.reset()` completes and
re-initializes the dead ones against the fresh view; a probe-alive handle is
left untouched, a robot without a handle (Phase-1 stub) is skipped, and a
failed re-init is logged loudly naming the robot.
