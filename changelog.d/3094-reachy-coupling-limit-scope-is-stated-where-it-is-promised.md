### Fixed: two Reachy verbs no longer promise a head-body twist limit they do not reach

`envelope_error` carries two kinds of limit, and only one of them is a property
of a single value. Per-axis travel is checked whenever the axis is present; the
head-body yaw *coupling* limit bounds `head_yaw - body_yaw`, so it is checked
only when one action carries both members. While `send_action` was the only
surface taking both, that was a clean split, and
`reachy_mini_driver._reject_unusable` documented it as one: "the pairwise limit
stays with `send_action`, which takes both".

The Reachy tool surface (#3082) put two verbs on `send_action` that send one
member. `reachy_body_turn` sends `body_yaw` alone, and `reachy_look` omits
`body_yaw` whenever the caller leaves it at its `None` default. So the
delegation above no longer names a surface that covers the pair, and the two
descriptions an agent reads promised a check that does not run on the path they
take:

* `reachy_look` said the driver refuses "a head-body yaw twist beyond 65 deg",
  unconditionally, on a verb whose default omits the body value;
* `reachy_body_turn` recommended itself for turning "while the head stays put",
  which is precisely the case whose twist is `head_current - body_new` and
  which nothing compares.

A tool description is the only thing the model driving the verb reads, so a
promised refusal is one it plans against - and here it moves a real head on a
real neck. Measured against a counterpart at 0, `reachy_body_turn(yaw=160)` asks
for 2.46x the 65 deg limit and `reachy_look(yaw=180)` for 2.77x, and both report
`success`.

Each of the three surfaces now states the scope it actually enforces:
`envelope_error`'s docstring says the check is a property of one action rather
than of the robot, `_reject_unusable` says its exclusion is a scope and not a
delegation, and the two verb descriptions state the condition (`reachy_look`
names passing `body_yaw` as the way to get the limit applied, which the suite
now grades as a real remedy).

**No behaviour changed, and the gap is not closed.** Which surface should own the
missing half is #3094, and it turns on a fact not in this tree: whether the
daemon reads `head_pose` in the world frame or relative to the platform's base.
On the first reading the gate is bypassed by two verbs; on the second the
`+/-180 deg` head-yaw bound is itself 2.77x the mechanical limit and the fix has
a different shape. Picking one here would have been a guess about hardware.

What the change does buy, beyond the honest descriptions, is that the gap is
graded rather than assumed. `TestTheCouplingLimitIsNotReachableHere` previously
asserted that no Device Connect RPC maps both members and that `send_action`
refuses when given both - nothing anywhere exercised one member, which is the
shape the new verbs send, and is why the suite was green over this. Two rows now
pin the single-member behaviour at the envelope and through the native driver, so
whichever answer #3094 takes arrives as an edited expectation. A fifth fact
family on the consolidated surface suite derives, per verb and by calling it,
which members of the pair reach one action, and refuses a description that quotes
the coupling limit without saying when it applies.
