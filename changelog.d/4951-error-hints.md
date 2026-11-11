### Changed: refusals say what to change

An agent's refusal of an unlisted Hub checkpoint names the exact setting (`STRANDS_POLICY_REPOS=owner/repo`, or
`owner/*`). A world change from the dashboard without the operator's yes says it needs `X-Confirm: yes` on a
signed-in request, instead of asking a signed-in caller to sign in. A real robot's refusal for a missing `confirm=`
names the mesh as a source too. An option a driver does not take says where it belongs: `watchdog_ms` and the other
`Mesh.mount` arguments, or `calibration=` (a Feetech bus's). `python -m strands_robots.mesh cert` says when it replaces
the identity in its `--out` folder, and `python -m strands_robots.dashboard --help` lists the environment it reads.
Nothing is allowed or refused differently.
