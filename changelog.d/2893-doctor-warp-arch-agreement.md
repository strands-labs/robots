### Fixed: the doctor reports a Warp build that cannot target this GPU's architecture

Warp chooses a device's architecture out of the table its binary was built with, so a build older than
the GPU compiles for a different architecture rather than refusing. On an NVIDIA Thor, whose driver
reports `sm_110`, Warp's CUDA-12 build offers a table with no `110` in it and settles on `sm_101` - and
nothing says so. A simple kernel returns the right answer under that substitution, so the mismatch
surfaces only once a kernel needs an instruction `sm_101` cannot express, by which point the symptom is
a wrong result rather than a setup problem. `strands-robots doctor` reported `All checks passed` on such
a node: `check_cuda` named the GPU and never asked whether the accelerator libraries agreed with the
driver about its architecture.

`check_warp_arch` asks. It names the architecture the driver reports, the one Warp settled on, the CUDA
toolkit the build was made against, and the `+cuNN` wheel tag matching this driver. The verdict reads
the arch table Warp reports rather than the wheel the environment was installed from, because a wheel
tag cannot answer it: PyPI's `warp_lang` filenames carry no local version segment, so one release's
CUDA-12 and CUDA-13 builds are indistinguishable by name, while the table is the value Warp itself
reads. Warp ships in the `sim-newton` extra, and an install without it - or without a CUDA device - has
no disagreement to report, so the check declines rather than failing.

Only Warp is asked here. torch's arch table is asked by `check_torch_arch`, because the two builds
fail differently: Warp substitutes a nearby architecture and keeps computing, while torch's own
compatibility check reports the device as not compatible with the build.
