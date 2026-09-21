### Fixed: `move_to` on a robot with a slide/hinge base no longer spends the solve on the base

`MinkIKBridge(commanded_dofs=...)` zeroed uncommanded degrees of freedom
*after* the QP had solved, so on a kinematic planar base the QP moved the
chassis to reach the target, the mask threw that motion away, and the arm
alone never converged - a target the arm reaches by forward kinematics
solved to a 0.31 m residual with every arm joint at a limit, and `move_to`
reported it as needing base motion. The uncommanded joints now carry a zero
`mink.VelocityLimit` inside the QP and the commanded joints a
`mink.ConfigurationLimit`, so the solve is over the joints the caller can
realise, inside their ranges; the same target solves to 0.001 m. A floating
base (free joint) keeps the post-solve mask, as before; a `mink` without the
limit classes keeps the historical path.
