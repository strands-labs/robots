### Docs: the `driver='strands'` refusal example names a robot that has no native driver

`docs/getting-started/robot-factory.md` shows the "no native driver is
registered" refusal as a verbatim `>>>` transcript. It refused `so101`, which
`FeetechDriver` has since come to serve, so the block asserted `No native driver
is registered for 'so101'` while `Robot("so101", mode="real", driver="strands")`
returns a driver and raises nothing.

The list inside that transcript rotted the same way, naming two robots when
fourteen are registered. An incomplete list reads as authoritative: a reader
checking whether their robot is natively driven found `trossen_wxai`, `vx300s`,
`wx250s` and `dynamixel_2r` absent and concluded no driver existed for them.

The example now refuses `ur5e`, which has none, and shows the refusal the code
raises. The section opener no longer implies the two driver paths are exclusive
-- `so100`, `so101`, `koch`, `aloha`, `lekiwi` and `hope_jr` each have a native
driver *and* a lerobot type -- and names `list_native_drivers()` as the live
answer rather than the captured list.

`TestTheNativeDriverRefusalExampleIsStillTrue` grades the names rather than the
wording: the refused robot must have no driver, every robot the transcript calls
natively driven must have one, and the refusal is checked by raising it.
Correctness is graded and completeness deliberately is not, so a fifteenth
driver leaves the capture older rather than wrong.
