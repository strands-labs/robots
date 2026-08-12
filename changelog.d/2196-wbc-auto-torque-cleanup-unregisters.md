### Fixed: unregister the auto-installed WBC torque shim so a second rollout gets one

`run_policy` installs a torque shim when a WBC policy meets a position-servo
scene, and returns a cleanup that ran only
`WBCTorqueController.uninstall` - which restores the actuator gains and nothing
else. `install_wbc_torque_control` does two things: it flips the driven
actuators to torque *and* registers the controller for `_apply_sim_action` to
dispatch to. Undoing one half left the controller registered on a scene whose
actuators were back to position servos, so it kept converting the policy's
position targets into PD torques that a position servo reads as targets, and
the hook's "a manually-installed controller wins" check then declined to
install on the next `run_policy`.

Measured on a Unitree G1 with the real SONIC balance checkpoint, two
consecutive 3 s rollouts on one sim: the pelvis excursion band grows from
0.0339 m on the first rollout to 0.1298 m on the second, and 15% of the
rendered pixels differ from the same rollout after this change. The cleanup now
removes the registry entry it wrote - and only that entry, so a manual
installation registered in the meantime survives - before restoring the gains,
which keeps the second rollout as steady as the first (0.0259 m).

The release now lives in `WBCTorqueController.uninstall` itself rather than in a
cleanup the hook wraps around it, so the *documented manual* pair -
`controller = install_wbc_torque_control(...)` then `controller.uninstall()` -
hands the world back just as completely. That path leaked the registration too,
and the docstring of `install_wbc_torque_control` names `uninstall` as its
counterpart, so one implementation now covers both callers.

The hook's five no-op conditions are now all driven. Three had never executed -
no `[wbc]` extra, no compiled world, and `wbc_uses_position_servo` reporting no
position-servo actuator - and its docstring, which listed four of the five and
described the predicate check as "the actuators are already torque mode", now
names all five in check order and both reasons the predicate can decline.
