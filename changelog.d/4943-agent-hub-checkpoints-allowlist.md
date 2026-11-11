### Changed: an agent loads a Hugging Face Hub checkpoint only if the operator lists it in `STRANDS_POLICY_REPOS`

- **Why:** loading a LeRobot checkpoint rebuilds the processor steps named in its own JSON, and some of them load
  further repos with remote code enabled, so a Hub repo is code its publisher chose. An agent could name any repo
  on the simulation rail, where nothing asks the operator.
- **Now:** under `from_agent()` (robot and world tools, mesh, dashboard runs), a `<owner>/<repo>` spec loads only if
  `STRANDS_POLICY_REPOS` lists it exactly or lists `<owner>/*`; anything else is refused before anything is
  downloaded. An operator yes for a run does not lift this. A malformed entry is an error, not a wildcard. The model
  is told only the allowed repos.
- **Unchanged:** your own `Policy("owner/repo")` and `robot.run("owner/repo")`, handles you register with
  `tool(policies=...)`, presets, and local checkpoint and dataset directories.
- **Migration:** if your agent picks Hub checkpoints itself, set `STRANDS_POLICY_REPOS=lerobot/smolvla_base,myorg/*`
  or register them as handles.
