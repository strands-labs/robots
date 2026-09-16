### Added: `Robot(tool_name=...)` - two robots of one type can share an Agent

`Robot()` hard-coded the name the agent sees the robot under: `"<name>_sim"` in
simulation, the canonical robot name on hardware. So
`Agent(tools=[Robot("so101"), Robot("so101")])` - a bimanual pair, or a real
arm beside its sim twin - died in the Strands tool registry with
`Tool name 'so101_sim' already exists` (naming an object repr), and the
obvious escape, `Robot("so101", tool_name="left_arm")`, raised a `TypeError`
from the factory's own internal forward.

`tool_name` is now a keyword-only factory parameter on every path (simulation,
lerobot hardware, native driver); the default is unchanged. A name no model
provider would accept is refused at the call site, with the remedy, before any
backend is built: the character set, the 64-character limit, and a trailing
newline, each naming its own reason. The local tool registry validates none of
them - such a name registers fine and only fails at the first model call, as a
validation error about a slot in the request body that never mentions the
robot. The refusal renders the offending value through `refusal_repr`, so a
value whose own `repr` raises is answered rather than re-raised.
