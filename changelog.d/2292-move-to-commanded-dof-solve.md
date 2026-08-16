### Fixed

`move_to` now solves inverse kinematics over the joints it commands instead of
the whole model, so `ik_residual_m` is the error the servo descent is actually
left with. `mink` optimizes every degree of freedom in the model it is handed,
so on a mobile manipulator or a humanoid the solve satisfied the Cartesian task
by sliding the floating base, left the commanded arm joints untouched, and
reported a near-zero residual for a target well outside the arm's reach; the
primitive then accepted it, servoed its whole `max_steps` budget, and blamed the
servo for a point it could never reach. Where the discovered tool frame sits on
the jaw, the same borrowing opened the gripper the primitive documents as held.
`MinkIKBridge` gained `commanded_dofs` for this, and the unreachable refusal now
reports `unrestricted_ik_residual_m` and `uncommanded_joints_moved` so a target
outside the robot's workspace is distinguishable from one that needs base
motion. Behaviour on a fully actuated fixed-base arm is unchanged.
