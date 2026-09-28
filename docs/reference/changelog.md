# Changelog

Release notes live on GitHub; this page is the map to them. After this page you know where the full notes are, what the last three releases changed in one paragraph each, and how a change is logged between releases.

- Full notes: [github.com/strands-labs/robots/releases](https://github.com/strands-labs/robots/releases)
- Assembled file: [`CHANGELOG.md`](https://github.com/strands-labs/robots/blob/main/CHANGELOG.md) in the repository root
- Between releases: every pull request adds one fragment under `changelog.d/`; the file is assembled at tag time with `python scripts/assemble_changelog.py --apply`

The site documents `main` at the commit it was built from, which is ahead of the newest tag. A feature on these pages that is not in the release you installed is in the next one.

## v0.5.2

Released 2026-09-17, 1,526 commits over v0.5.1. The first release that drives real robots without lerobot in the middle: a native driver layer (`Robot(name, mode="real", driver="strands")`) for Unitree G1 and Go2, Booster T1, Franka, UR, Robotiq, Crazyflie, EarthRover, Reachy Mini, Pollen Microduck and the Feetech and Dynamixel buses. The Microduck joins the registry as a sim and real robot with its own policy provider. The operator dashboard serves again (`strands-robots dashboard`) with passkey auth and a human-in-the-loop gate in front of every call that can move hardware. A security pass closed eight findings in eleven pull requests. Removed: `vera`, `motionbricks`, the vendored LIBERO suite and `lerobot_calibrate`. Floors raised to `strands-agents` 1.13.0 and `eclipse-zenoh` 1.6.1 (the current floor is in `pyproject.toml`). [Notes](https://github.com/strands-labs/robots/releases/tag/v0.5.2).

## v0.5.1

Released 2026-08-06, 51 commits over v0.5.0. A correctness-only patch: no new features or API additions. It raised the lerobot floor to `>=0.6.1` so `stream_dataset(repo_type="bucket")` resolves to a `StreamingLeRobotDataset` that accepts `repo_type`, and continued the numeric-input hardening pass across the training, tools and simulation surfaces. [Notes](https://github.com/strands-labs/robots/releases/tag/v0.5.1).

## v0.5.0

Released 2026-08-04, 806 commits over v0.4.1. The largest release to that point: the NVIDIA Isaac Sim backend, agent-facing analytic motion primitives (`move_to`, `set_gripper`, `rotate_wrist`), a remote-inference client and server split, terrain locomotion curricula, and a hardening pass across every numeric input the agent and mesh surfaces accept. Changelog assembly moved to per-PR fragments in `changelog.d/` in this cycle. [Notes](https://github.com/strands-labs/robots/releases/tag/v0.5.0).

## Versioning

The version comes from the nearest release tag (`git describe --match 'v[0-9]*'`); a checkout with no reachable tag builds as `0.1.dev...`, and `git fetch upstream --tags` restores the real number. `strands-robots --version` prints what is installed. The road from 0.5.x to 1.0 is on the [roadmap](../project/roadmap.md).
