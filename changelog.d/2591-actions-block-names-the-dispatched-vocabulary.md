### Fixed

`pose_tool`'s `Actions:` block no longer advertises `"calibrate_motor"`, a verb
the tool dispatches nowhere. The entry sat among thirteen real ones, in the same
quoted-and-colon format and under the same `Motor Control:` sub-heading, from the
tool's original commit onwards; `action == "calibrate_motor"` has never existed
in any commit. This is not only a docstring: `@tool` publishes the docstring as
`tool_spec["description"]`, so the entry was in the text a model reads to choose
a verb. Selecting it produced `Unknown action: calibrate_motor` - byte-identical
to the refusal for a misspelling - followed by an `Available actions:` list that
omitted it, so the tool described itself twice and the two halves disagreed, with
the half read first being the wrong one.

An "interactive" action was never implementable on this surface: `pose_tool` has
no `input()` or `stdin` anywhere, and an agent tool has no channel to be
interactive on. Calibration already has two homes, so the block now points at
them the way `lerobot_teleoperate` already points at one: `lerobot_calibrate`
manages stored calibrations, and `lerobot_teleoperate`'s
`auto_accept_calibration` answers the prompt LeRobot shows when a device has
none. The thirteen dispatched verbs, their descriptions and the refusal text are
unchanged.

Every `@tool` module's `Actions:` block is now compared against the vocabulary it
dispatches, in both directions, by
`tests/tools/test_tool_actions_block_names_the_dispatched_vocabulary.py`. Nine of
the ten graded surfaces were already exact, so the guard pins a convention the
package keeps rather than introducing one. It also grades the other half of
`pose_tool`'s self-description: the refusal's `Available actions:` list was
already exact while the block was not, which is how the two came to disagree.
The existing guards on this contract sit at the published-schema boundary of the
MuJoCo `Simulation` tool and grade an enum, so a prose block was graded by
nothing.
