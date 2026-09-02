### Fixed: a command envelope's routing fields are validated as key-expression segments

`Mesh` answers a command on
`strands/{sender_id}/response/{responder}/{turn_id}`, built from two fields read
straight off the wire. Zenoh accepts a wildcard on a `put` and routes it by
intersection, so an envelope carrying `sender_id="**"` produced a reply key that
every peer's `strands/{peer}/response/**` subscription matched -- the answering
robot's dispatch result (task status, battery, pose) reached the whole fleet
instead of the peer that asked. Measured with two live Zenoh sessions: both
bystander peers received a reply to a command neither of them sent.

`validate_command` already type-checked, length-bounded and charset-validated a
`turn_id` / `sender_id` carried inside the *command* dict, and said in its
docstring that those were the fields `Mesh` correlates a turn and keys its
command-replay cache on. They were not: routing reads the enclosing envelope's
own copies, which reached the key expression unchecked. The routing pair is now
held to `validate_mesh_identifier` -- the same `[A-Za-z0-9_.-]+`, at most 128
characters rule the teleop identifiers follow, because printable ASCII is not a
sufficient bound when `*` and `/` are printable and are what widen a key.

An envelope that breaks the rule is refused whole: nothing is dispatched, nothing
is published, and one audit record is written. The refusal cannot forge a log
record either -- the warning names the field and the rule but never the rejected
value, so a line break in it cannot split the record, while the structured audit
record carries a bounded `repr` for forensics. An **absent** `sender_id` is
unchanged and still means fire-and-forget: the command runs and no reply is
published.
