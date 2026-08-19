### Added: cosmos3 OpenArm embodiment (post-training path)

The cosmos3 provider's embodiment table gains an `openarm` entry
(`Cosmos3Embodiment`: domain `openarm_lerobot`, 10D unified action = 9D EE pose
+ 1D grasp, `midtrain` action space, front + wrist camera keys), closing the
record -> post-train -> deploy loop for the Enactic OpenArm with cosmos3 the
way `droid`/`umi`/`av`/`bridge` already close it. The entry is data-driven
through the same service-mode, in-proc diffusers and `sim_ik` decode paths as
the existing four; aliases `openarm_lerobot`, `openarm_follower` and
`enactic_openarm` resolve to it. It documents the post-training expectation
explicitly: no released Cosmos 3 checkpoint ships this domain, so it maps a
checkpoint post-trained on OpenArm episodes via the existing `cosmos3` trainer
(SFT recipe TOML -> `cosmos_framework` launch) rather than promising zero-shot
behavior, and no action stats are bundled - the decode refuses the unbundled
domain by name and accepts the post-trained domain's own quantiles via
`stats=` + `stats_domain="openarm_lerobot"`. Unknown embodiment names keep
failing loudly. `bi_openarm` (bimanual) is a deliberate follow-up once the
single-arm mapping is validated.
