### Docs: the pip install route runs physics but produces no RTX pixels

Measured on an AWS `g5.2xlarge` (A10G, an RT-core GPU): a pip-wheel install boots
`SimulationApp`, steps physics and reports success, and every RTX camera read comes
back empty - while the same script under `nvcr.io/nvidia/isaac-sim:6.0.1` on the
same instance returns real frames. It reproduces in *pure Isaac Sim* with no
`strands-robots` code in the process, which isolates it to the install route rather
than to the GPU, the driver, or this backend.

Nothing raises, which is what makes it expensive: `render()` degrades to a blank
frame by contract, so a rollout recording video writes an all-black MP4 and reports
success. The page now says so where the pip route is offered, and the install block
lists Docker first.
