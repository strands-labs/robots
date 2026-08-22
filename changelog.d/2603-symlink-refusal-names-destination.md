### Fixed

The symlink refusal in `validate_output_path` now names where the link points,
and hands over that path when this same call would accept it. Refusing to follow
a link planted at an output target is right -- it is an arbitrary-write vector --
but the refusal named no way forward, and on macOS the single most reachable
scratch path IS a symlink: `/tmp` points at `/private/tmp`. A caller that asked
for the most ordinary directory on the machine got a dead end that read like an
attack had been detected. The reachable surface is the two `label="output_dir"`
sinks, where the target is a directory rather than a file inside one: measured
through `start_cameras_recording(output_dir=<a directory reached through a
symlink>)`, the call refused and wrote nothing, and following the corrected
message writes the recording (1 MP4, 16 frames).

The link itself is still never followed; `resolve()` is a read.

Two properties the hint holds. It does not raise: naming the destination means
resolving it while a refusal is being built, and `Path.resolve(strict=False)` is
not total -- CPython 3.12 raises `RuntimeError("Symlink loop from ...")` for a
cycle while 3.13 returns the link unresolved, and `requires-python = ">=3.12"`
admits both. A handler naming only `OSError` would catch neither, letting 3.12's
`RuntimeError` escape a function documented to raise `ValueError`, past the
`except ValueError` every sink maps to a structured tool error. And it does not
advertise a path this same call would refuse: under active confinement a link
pointing outside the sandbox has its destination named as a diagnosis rather than
offered, so a caller who follows the message cannot land in a second refusal.

A dangling link still gets a hint. `resolve(strict=False)` returns its target
without raising, and a not-yet-existing path is exactly what an output sink is
about to create.
