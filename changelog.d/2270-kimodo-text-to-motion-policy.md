### Added: `kimodo` policy provider - text-to-motion diffusion for the Unitree G1

`KimodoPolicy` wraps NVIDIA's Kimodo (`nvidia/Kimodo-G1-RP-v1`) text-conditioned
motion diffusion model as a first-class provider, taking the seat next to
`motionbricks` in the kinematic-motion-generator family: a natural-language
prompt in, per-frame full-body G1 `qpos` out, adapted to the per-tick action-dict
contract and SLERP-upsampled from the sampler's native 30Hz to the 50Hz a
tracker consumes. Both emit the canonical 29-joint WBC action dict, so either
can drive a shared WBC/PD tracker through `CompositePolicy`.

The provider carries a `KimodoMotionAgent` injection seam, so the frame ->
action-dict mapping is unit-testable without torch, diffusers, CUDA, or weights.
A missing `[kimodo]` extra raises with an install hint rather than falling back
silently, and the custom sampler class is gated behind
`STRANDS_TRUST_REMOTE_CODE`.

Every field the registry advertises in the provider's `config_keys` is an
explicit keyword parameter of the constructor, merged onto an optional `config=`
with per-field overrides winning and the merged result re-validated by
`KimodoConfig`. The provider declares no `**kwargs`, so a misspelled knob is
refused at construction instead of being swallowed by a parameter nothing reads.
