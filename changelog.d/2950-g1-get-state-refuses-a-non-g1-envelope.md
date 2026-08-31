### Fixed: `g1_get_state` refuses a handle whose status envelope is not a G1's

`g1_get_state` takes a live driver handle typed `Any`, so the annotation carries
no type into the generated tool schema, and `get_status` is a member of the
shared `DRIVER_SURFACE` every native driver implements -- so a duck-type check
on it cannot tell a G1 from any other driver in a mixed fleet. The verb's whole
answer, `admits_arm` and `admits_loco`, is decided from the `fsm_id` that
handle's envelope reports, and a sibling driver's envelope carries the shared
`tool_name` / `connected` / `battery_pct` triple and no FSM field. Driving the
shipped `MicroduckDriver` through the verb therefore returned `status="success"`
with `fsm_id=None` and both booleans `False`: a decided answer about a gate that
robot does not have, byte-identical to the answer a genuinely disconnected G1
earns, with nothing downstream able to separate the two. Three unusable handles
also raised past the `@tool` boundary the module's docstring says never raises --
`None`, a handle with no `get_status`, and one whose `get_status` returns
anything other than the envelope shape.

The verb now refuses all four before it reads a gate answer, naming the
parameter, the type it was handed and the remedy. The discriminator is the five
FSM fields `G1Driver.get_status` reports and no other shipped driver does, held
in one constant so a reader widening it finds a single place. The documented
answers are unchanged: a G1 envelope reporting `fsm_id=None` still earns
`success` with both booleans `False`, and a wired envelope still round-trips
every field.
