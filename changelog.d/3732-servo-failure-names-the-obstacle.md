### `move_to` and `rotate_wrist` name what stopped the servo

A `move_to` that IK could solve but the servo could not reach used to end in
"the pose fights joint limits/contacts"; a `rotate_wrist` that did not turn
reported only the residual it fell short by. Both now read the engine at the
final tick and say which: `The servo was stopped: the robot is in contact:
'so100/Base/geom_3' <-> 'so100/Fixed_Jaw/geom_19' (d=-0.0456 m) and 12 more`
(active contacts touching the robot's own bodies, nearest first, at most
three, total reported) and/or `commanded joint(s) at a limit:
'arm/Wrist_Roll' at its upper limit (2.7905 vs 2.7900)`. When neither is true
it says so and points at `max_steps`/`tol`. The same facts travel as
`json.obstruction` (`contacts`, `contacts_total`, `joints_at_limit`), scoped
to the joints each primitive commands to a new value. On Isaac the joint half
is reported for `move_to` and `contacts_total` is `null` (contacts are not
read there). Found by the v0.5.2 devx replay: three `move_to` failures in a
row whose real cause, a jaw-against-base self-collision, was only visible in
a separate `get_contacts` call - measured again on the wrist, which did not
turn at all (0.003 of 2.500 rad) while 15 contact pairs held it.
