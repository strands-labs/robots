### Removed: the Isaac Replicator synth-data example that demonstrated a method that does not exist

`examples/isaac/isaac_replicator_synthdata.py` delegated 100% of its
capability to `IsaacSimulation.generate_synth_dataset`, a method with zero
hits under `strands_robots/`. On a fully provisioned Isaac Sim 6.0 + RTX
host the script boots the real SimulationApp, loads the real Franka USD,
then skips and writes nothing (`replicator_status=skipped frames_written=0`),
and its gating references point at issues in the retired robots-sim repo
that can never land. Per the "No dead code" convention the file, the
now-empty `examples/isaac/` directory, and the `examples/README.md` index
row are removed. #1891 stays open as the durable tracker for the
`generate_synth_dataset` capability; the example returns alongside the
feature in the same PR, and the deleted file remains recoverable at
`d0aaac9` as a spec.
