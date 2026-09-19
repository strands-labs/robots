### Fixed: `IsaacSimulation.is_available()` no longer requires torch

Isaac Sim's runtime is PhysX + Warp and does not import torch, but
`is_available()` treated an absent torch as proof the backend was unusable and
answered `(False, "PyTorch not installed. Isaac Sim requires torch with CUDA
support.")`. `[sim-isaac]` does not declare torch either, so the requirement
could not be satisfied by installing the extra that enables this backend.

Measured on `nvcr.io/nvidia/isaac-sim:6.0.1` -- the image
`_install.ISAAC_SIM_DOCKER_IMAGE` names, which ships no torch: `import isaacsim`
and `from isaacsim import SimulationApp` both succeed, `create_world` steps
physics, `send_action` drives a joint to its commanded target and `render`
returns a 640x480 RTX frame, while `is_available()` reported the backend
unavailable.

An absent torch is now no verdict at all. Where torch *is* importable it remains
a real signal: a torch reporting no CUDA device still yields `False`. Separately,
the `torch.cuda.is_available()` call moved out of the `try` that classifies the
import, so an `ImportError` from CUDA initialisation is no longer misreported as
"PyTorch not installed" -- the wrong diagnosis for a torch that is installed and
cannot reach the GPU.
