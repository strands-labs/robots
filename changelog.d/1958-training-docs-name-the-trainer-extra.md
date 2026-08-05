### Docs: the training-overview install rows name a stack that can actually train

`docs/training/overview.md` said the base `strands-robots[lerobot]` extra was
"enough for ACT / diffusion from scratch" and that the `lerobot_local` install
"works out of the box". Neither held for the operation the page is about.
LeRobot's `train()` calls `require_package("accelerate", extra="training")`
before it branches on device, and nothing on the `lerobot_local` path declares
`accelerate` -- the `[lerobot]` extra is exactly `lerobot[feetech,dataset]`, and
the only `strands-robots` extra that declares it is `cosmos3-diffusers`, a
different provider. So the documented install refused on the first `train()`, on
CPU and on GPU alike, and because `train()` reports that in its `TrainResult`
rather than raising, an unchecked call passed a `checkpoint_dir` of `None` on
instead of stopping.

Every `lerobot_local` row now names `lerobot[training]` in its install line and
quotes the call that refuses, so the "CPU needs it too" claim is checkable
rather than asserted. The `HW floor` column read `1 consumer GPU`, reinforcing
the same misconception; notebook 3 trains ACT on CPU, so it now reads `CPU for a
toy run; 1 consumer GPU in practice`. `docs/troubleshooting.md` gains the
symptom verbatim, since that string is what a reader searches for.
