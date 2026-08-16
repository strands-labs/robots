### Fixed

- **Simulation / IK**: `discover_ee_frame` now resolves a tool-point site over the
  end-effector body that shares its name. Its site rung only searched TCP-specific
  spellings (`attachment_site` / `grasp` / `tcp` / ...) while the body rung matched a
  wider end-effector vocabulary (`gripper` / `hand` / `tool` / ...), so a model
  publishing its tool point as a site named for the end effector was answered with a
  link origin instead - `so101` (98.4 mm behind its own fingertips), `aloha` (130.0 mm)
  and `toddlerbot_2xc`/`toddlerbot_2xm` (60.8 mm). Because `move_to` measures the
  residual it reports at the discovered frame, the offset was invisible: on `so101` it
  reported 18.1 mm convergence while the fingertips were 100.5 mm from the commanded
  target. The site rung now searches its own spellings first and then the body rung's,
  so a site wins whenever one names the end effector. 58 of the 62 loadable registry
  robots resolve unchanged, and a site-less model still resolves to its end-effector
  body.
