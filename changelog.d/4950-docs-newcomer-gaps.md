### Documentation: the steps a first-time user missed

`tools.md` passes the tool `robot.tool(policies=...)` returns to the agent (the call returns a new tool). `mesh.md`
shows that `watchdog_ms` is an argument of `Mesh.mount`, that a move over the mesh to a real robot needs `confirm=`,
and that every `cert` runs on the admin machine and is copied over (`cert` replaces the identity in its `--out`).
`lerobot.md` says where to get `zenohd` and imports `server_config`. `recording.md` says what `my_policy` stands for and
that every Record route takes a JSON body. `hardware.md` says calibration is Feetech-only, and `dashboard.md` that a
token is good on every dashboard sharing an auth file.
