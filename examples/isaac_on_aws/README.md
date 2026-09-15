# Isaac Sim on AWS - reference deployment

Runs the strands-robots **Isaac backend** on an AWS GPU instance, end to end:
provision, smoke-test, tear down. AWS is **optional** for this backend -
`create_simulation("isaac")` discovers the Isaac Sim runtime in the local
Python environment and never talks to a cloud - this example exists for the
common case where the machine in front of you has no NVIDIA RT-core GPU (a
Mac, most laptops).

```bash
cd examples/isaac_on_aws
./provision.sh     # ~15-25 min: instance + driver + docker + the 32 GB container
./run_smoke.sh     # ships YOUR local tree, runs smoke.py in the container
./teardown.sh      # terminate + clean up. Do not skip this.
```

## What the smoke test proves

`smoke.py` runs inside the pinned container against the tree you have checked
out, and fails loudly on the first thing that is not real: a world that
creates, a robot articulated from a bare URDF, a cube that actually **falls**
(measured by position, not by a success envelope), joint position + velocity
observations, a contact report for the resting cube, a ray that names what it
hit, a latched external force, domain randomization, and an RTX frame whose
pixels have variance. Exit code 0 only when every check passes.

## Choices this example makes, and why

- **`g5.2xlarge` (NVIDIA A10G) by default.** Isaac Sim requires an RT-core
  GPU: `g5` (A10G) and `g6` (L4) work; `p3`/`p4`/`p5` (V100/A100/H100) do
  not - datacenter compute GPUs have no RT cores and Isaac Sim refuses them.
  A10G is the GPU this backend's whole verification history ran on. Override
  with `INSTANCE_TYPE=g6.xlarge ./provision.sh` for the cheaper L4.
  Indicative on-demand price for g5.2xlarge in us-west-2 at the time of
  writing: about $1.2/hour - check the EC2 pricing page for yours, and tear
  the instance down when done.
- **The NGC container, pinned to `6.0.1`.** Two measured reasons. The pip
  wheels run physics but produce **no RTX pixels** (verified on this exact
  instance type: the same script renders under the container and returns
  empty buffers under the wheels). And NVIDIA publishes only full
  `major.minor.patch` tags for this image - `isaac-sim:6.0` and `:latest`
  both fail with `no such manifest`.
- **SSM only, no SSH, no ingress.** The security group has no inbound rules;
  every command travels through AWS Systems Manager, and the code travels as
  a presigned S3 URL the instance fetches - it needs no S3 permissions of its
  own. The instance profile carries exactly one policy:
  `AmazonSSMManagedInstanceCore`.
- **Your working tree, not a release.** `run_smoke.sh` tars the repository
  you are standing in, so what runs in the cloud is what you edited - the
  same loop this backend's own fixes were verified with.

## Costs and cleanup

The instance bills while it exists, including while idle. `teardown.sh`
terminates it, waits, and deletes the security group; the IAM role is left
(inert and free) and documented there. If a run is interrupted, the instance
id and region are in `.instance.json` next to these scripts.

## Troubleshooting

- **Provision stalls at "waiting for SSM"**: the instance type may be out of
  capacity in your region (g5 frequently is) - terminate and retry in another
  region: `REGION=us-east-1 ./provision.sh`.
- **`nvidia-smi failed`**: the driver install did not land; check
  `/var/log/cloud-init-output.log` on the instance via SSM Session Manager.
- **Exit code 134 after a successful summary**: Isaac Sim's known atexit
  segfault, harmless - `smoke.py` already reports its verdict and exits
  through `os._exit` before it can matter.
- **`add_robot` says a registered robot's "model file is not on disk"**: the
  asset was never fetched, and the two reasons are both properties of the NGC
  image rather than of your setup. It ships **no git**, and the registry
  resolver for the 57 entries naming a `robot_descriptions` module works by
  importing it, which clones on first use - so in-container the fetch dies as
  `FileNotFoundError: [Errno 2] No such file or directory: 'git'`. And it runs
  as **uid 1234 (`isaac-sim`), not root**, so `apt-get` cannot supply git
  either (`E: Unable to acquire the dpkg frontend lock ... are you root?`).
  Both surface two errors downstream of the cause, as a merely missing file.
  `run_smoke.sh` therefore clones on the host, copies with `cp -rL` (the
  resolver's own symlink is absolute and would dangle inside the container),
  and mounts the result via `STRANDS_ASSETS_DIR`.

  If you add a robot to the smoke, fetch it the same way. Do not reach for a
  `pip install` of the asset inside the container - there is nothing to
  install; the assets are a git clone.
