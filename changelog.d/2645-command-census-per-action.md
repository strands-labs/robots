### Fixed: the command census names every action it validates a field for

`validate_command` documents its per-action work under a `Performed checks:` census. Six
action branches validate at least one field out of `cmd`; the census named four of them, and
for one of those four it described the check inaccurately.

`resume` and `teleop_stop` were absent entirely, so nothing in the census said that
`resume.override_code` -- the operator's second factor for clearing an e-stop lockout -- is
bounded at 256 printable-ASCII characters, or that `teleop_stop.device_name` is checked at
all. `teleop_receive` was named but only for one of its two fields.

The inaccuracy is the sharpest of the four. The census said `source_peer_id` must be "a
non-empty str", where the body hands it to `validate_mesh_identifier`. Measured, that
validator refuses the Zenoh wildcards `*` and `**`, `/`, `?`, `#`, `$`, control characters and
anything over `MAX_PEER_ID_LEN` -- every one of which the census's own words admit. A reader
sizing an allowlist entry or a fixture from the census therefore picks a value the validator
rejects, and the rejection names a charset the census never mentioned.

The census now names all six branches, every field each one validates, and the shared
validator the two teleop identifiers are decided by, so the admitted charset is discoverable
from the census rather than by reading the body.

`validate_command`'s docstring is the whole package diff: with that docstring removed from
both versions the two files are byte-identical, so no verdict, message or accepted input
moves. The existing census guard gains the per-action completeness rule, with the action set
derived from the body's own `if action == "x"` chain and the field set from every way the body
reads `cmd` -- `.get`, `.pop`, `cmd["x"]` and `"x" in cmd` -- so a seventh branch, or a new
field on an existing one, is graded on arrival.
