### Fixed: the Isaac backend applies the shared action-value domain

`IsaacSimulation.send_action` hand-rolled its own action conversion instead of
routing through `SimEngine._coerce_action`, the shared coercion every backend
applies. On the Isaac backend alone a boolean reached the articulation as a
`1.0`/`0.0` PD position target, `nan`/`inf` reached it as a target no solver can
honor, a vector whose width did not match `robot_action_keys` was handed to the
articulation anyway, and a multi-element value, a non-numeric value, `None` or
the single-element row a 1-DoF key carries under the
`Policy.get_actions -> list[dict]` contract raised `TypeError`/`ValueError`
straight past `send_action`'s structured envelope -- so a policy emitting the
documented shape could not drive this backend at all. `send_action` now applies
the shared domain, and the structural guard that asserts no backend can ship
without it covers all three backends rather than two.
