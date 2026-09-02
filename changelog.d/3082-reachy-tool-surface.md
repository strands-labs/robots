### Added: the Reachy Mini's agent tool surface, 14 verbs on the native driver

`ReachyDriver` shipped in #2762 and could reach the hardware, but nothing on the
agent surface could reach the driver. An agent asked to make the Mini look at
something had `send_action` and a joint dict, which is the wire shape rather
than the verb shape, and none of the daemon's expressive surface - recorded
emotions, wake and sleep, the antennas, the speaker - was addressable at all.

The surface is 12 execution verbs plus 2 read verbs, in the consolidated
table-driven shape the g1 family locked (refs #3037, #3070): a `_VERBS` table
names each verb's driver accessor, one shared handle judgement is worded from
that table, and each `@tool` makes exactly one driver call and returns that
envelope verbatim. So the tools add no second opinion about what the robot did -
a refusal an agent reads is the driver's own refusal, including the motion
envelope's travel limits and its head-body yaw *coupling* limit, which no
per-axis check can see.

Six `ReachyDriver` accessors landed with it, because a verb with no accessor is
a verb that cannot be written: `play_move`, `list_moves`, `wake_up`,
`goto_sleep`, `set_motors` and `state_snapshot`. `set_motors` admits
`enabled` and `disabled` and refuses the SDK's third mode
(`gravity_compensation`) *by name*, because that mode has no daemon-link
command - mapping it silently onto one of the two that do would report success
for a torque state the robot is not in. `state_snapshot` copies each cache
under the lock it is written behind, so a caller mutating what it reads cannot
reach the sensor state the mesh publishes.

Move names are matched against a URL-safe alphabet before they reach the
daemon's path template, which is the same guard the proven Device Connect
driver ships: a name is interpolated into
`/api/move/play/recorded-move-dataset/{dataset}/{move}`, so `../` in a name an
agent chose would address a different endpoint than the one the verb documents.

Four media verbs refuse rather than pretend. `ReachyDriver` has no accessor for
them yet, and a verb that returned a plausible-looking envelope for a camera
frame nothing captured would be worse than one that names the accessor still to
be added.

One wire shape is not the one the other endpoints use.
`GET /api/move/recorded-move-datasets/list/{dataset}` is declared `-> list[str]`
by the daemon and the transport returns the decoded body unreshaped, so a
*successful* catalogue read is a JSON array while every failure is still the
`{"error": ...}` dict. Reading an error key off that array raised
`AttributeError` out through `reachy_list_emotions`, on the happy path of a
shipped verb. The read now goes through an accessor whose return type says what
the endpoint can answer, so the shape must be narrowed before it is indexed and
this class of mistake is a type error at check time rather than an exception on
a real daemon.

`reachy_wake` selects one of two physical motions, so its `sleep` flag is
checked against the shared `boolean_flag_error` domain rather than read by
truthiness. Every non-empty string is truthy, so `'false'`, `'no'`, `'off'` and
`'0'` - the spellings a caller reaches for to opt *out* - each commanded
go-to-sleep and reported success. The check precedes the accessor because the
accessor is derived from the flag: the same misread also decided which accessor
the handle judgement required to be callable, so a handle that could wake but
not sleep was refused for a request to wake. It is the domain
`build_lerobot_command` already applies to its own flags, so a posture flag is
refused identically wherever it is supplied rather than merely equivalently.
