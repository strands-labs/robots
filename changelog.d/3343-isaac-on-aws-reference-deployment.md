### Added: `examples/isaac_on_aws` - a reference AWS deployment for the Isaac backend

Isaac Sim needs an RT-core GPU, and the docs offered no way to get one. This adds
three scripts and a smoke test that stand a working Isaac host up from nothing and
tear it down again:

```bash
cd examples/isaac_on_aws
./provision.sh      # g5.2xlarge (A10G), driver, docker, NGC image
./run_smoke.sh      # packs the local tree, runs 14 checks on the GPU
./teardown.sh       # terminates the instance and removes the security group
```

The deployment is **optional**. The Isaac backend resolves and runs against any
local Isaac Sim install exactly as before; nothing here is on the import path, and
`create_simulation("isaac")` neither knows nor cares whether the runtime came from
this example. It exists because "you need an RT-core GPU" was previously the end of
the sentence.

`run_smoke.sh` is what makes this a reference rather than a snippet: it exercises
the backend surface end to end on real hardware - `create_world`, a bare-URDF
`add_robot`, `add_object`, `reset`, `step`, a cube that actually falls, joint
position *and* velocity observations, `get_contacts`, `raycast`, `apply_force`,
`randomize`, and both RTX colour and depth carrying real pixels. All 14 pass on a
freshly provisioned instance.

Two fixes the first end-to-end run found, neither of which any unit test could
have:

* **`ubuntu-drivers install --gpgpu` returns 0 while installing nothing** on this
  AMI. No nvidia package, no kernel module, no `nvidia-smi` - and because the exit
  code was a success, the `|| apt-get install nvidia-driver-535-server` fallback
  never fired. `provision.sh` now names the driver explicitly. It loads without a
  reboot; the smoke run saw the A10G immediately after `modprobe`.
* **`teardown.sh` printed the security-group API's raw JSON** to the operator's
  terminal on the happy path, because the `2>/dev/null` guard was on stderr only.

Verified twice: once resuming the half-provisioned instance the first run left, and
once as a clean `provision` -> `run_smoke` -> `teardown` cycle with the corrected
script, which completed in about ten minutes and left nothing running.
