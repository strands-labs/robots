### Fixed: `build_lerobot_command` emits `--play_sounds` for the modes that accept it

`play_sounds` was declared on `build_lerobot_command` and on the `lerobot_teleoperate`
tool, documented as "Enable audio feedback", forwarded from the tool to the builder -
and then emitted by nothing. Every mode returned a byte-identical argv for `True` and
`False`, so `play_sounds=False` - the request an unattended session on a shared machine
makes - was silently the default, on a detached subprocess's command line the supplying
call cannot read a failure back from.

Three of lerobot's four entry points declare the field: `RecordConfig`, `ReplayConfig`
and `RolloutConfig`. Those modes now emit `--play_sounds true|false` explicitly - like
`--dataset.video`, so this signature's own default is what the session runs with rather
than whatever the lerobot config happens to default to - and the flag is checked against
the shared `boolean_flag_error` domain, so a truthy spelling of off such as `"false"` is
refused instead of selecting the opposite posture. Plain teleoperation still emits
nothing for it: `TeleoperateConfig` does not declare the field and that CLI exits with
`unrecognized arguments: --play_sounds`, so refusing a value there would be a false
rejection.

`play_sounds` also gains an `Args:` entry on `build_lerobot_command`, which had declared
and consumed it without documenting it.
