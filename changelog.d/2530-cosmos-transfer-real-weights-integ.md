### Added: a real-weights Cosmos-Transfer integration test through the pipeline seam

`tests_integ/transforms/test_cosmos_transfer_real_weights.py` binds the
diffusers-hosted `nvidia/Cosmos-Transfer2.5-2B` checkpoints (general variant +
edge controlnet, real Cosmos Guardrail enabled) to
`CosmosTransferTransform`'s vendor-neutral pipeline seam and runs the full
dataset round trip on real inference: record one LeRobot episode, transform,
reopen, assert schema parity, action/state byte-equality, provenance rows
naming the real pipeline version, and pixels changed. A second
identically-seeded run pins the determinism claim the provenance `seed` field
makes - measured byte-identical on the pinned stack (L4, torch 2.11/CUDA 13,
diffusers 0.40, bf16). The generative path introduced by #2480 was previously
verified only up to the injection seam; per AGENTS.md convention 8 it now has
a real-inference `tests_integ/` item like every policy backend.
