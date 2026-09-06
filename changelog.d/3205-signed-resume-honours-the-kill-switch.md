### Fixed: the resume half of the signed safety rail reports a switched-off rail as a fault

`MeshBridge.signed_estop` distinguishes "you switched this off" from "this is
broken", because an operator acts on the two differently: one is the switch they
set, the other is a fault to chase. `MeshBridge.signed_resume` is the same rail
and the same operator and did not - with `STRANDS_MESH` set to any kill spelling
it answered `safety mesh unavailable`, which is the wording the rail's own suite
pins as meaning a fault. The troubleshooting sheet documents exactly two causes
for a refused resume, both about `override_code`, so an operator who set the
switch was sent to a code that is fine rather than to the switch they set.

Both verbs now answer through one owner, so the two wordings have a single
spelling and a second copy cannot disagree with the first. The rail's guard is
widened with it: keying on `Mesh(...)` construction grades "did we open a
session" and is structurally blind to a verb that constructs nothing, which is
how the resume half drifted. Every answer that reports the rail unavailable is
now graded for asking the kill switch, whether or not it opens a session, and
`signed_resume` gains the behavioural cells it had none of.
