### Fixed: the doctor reports a torch build that carries no code for this GPU

`torch.cuda.is_available()` answers about the driver, not about the build, so it is `True` on a host
whose torch build carries no CUDA code for the GPU that driver reports. torch's own compatibility check
calls such a build "not compatible with the current PyTorch installation" and refuses every kernel on
it - but it says so through a `warnings.warn` on the first CUDA call, which goes to stderr, is shown
once per process, and is consumed by anything that touched CUDA earlier in the same process.

`check_cuda` therefore passed on such an install, naming the device and the version, and
`strands-robots doctor` printed `All checks passed` and exited 0. Measured with an arch list of
`sm_80 sm_90` against an `sm_110` device: the table read `PASS  CUDA available: NVIDIA Thor (torch
2.11.0+cu130)`, the run exited 0, and 964 bytes of torch's own "not compatible" went to stderr - the
same shape as the EGL tracebacks, in the same command a new reader runs to find out whether their
machine is set up.

`check_torch_arch` asks. It names the architectures the build carries, the one the driver reports, and
the CUDA releases whose torch build covers this device, taken from the table torch ships for its own
remedy. The comparison is torch's own predicate rather than a rule derived here, so the doctor and
torch cannot reach opposite verdicts about one install. That predicate reads a table of intervals: code
carrying `sm_80` supports `>=8.0,<9.0 except {8.7}`, so it does not cover a Jetson Orin, while the
coarser backward-compatible-within-a-major-version rule that torch's cubin check documents admits that
pair. The interval table is preferred for that reason and the coarse rule is only the fallback for an
install too old to expose it, since the supported floor is `torch>=2.0.0`. torch ships in the extras
that run a policy, and an install without it - or without a CUDA device, or with a CPU-only build that
reports no architectures at all - has no disagreement to report, so the check declines rather than
failing.

The Warp check's scope paragraph excluded torch on the grounds that a build without the device's arch
compiles from PTX instead. That is refuted three ways: the torch build shipped for an `sm_110` device
carries no `compute_*` PTX entry at all, torch's own words for the case are "not compatible", and its
rule is the interval table rather than a PTX fallback. `check_cuda` is unchanged - it answers about the
driver, which is the half it was always reporting.
