### Fixed: the GPU test that should have caught this graded nothing

`tests_integ/simulation/test_isaac_gpu.py::test_replicate_fleet_creates_parallel_envs`
carried the docstring "`replicate()` must create the requested parallel
environments" and asserted only `status == "success"` and `"16" in text` - both of
which a complete no-op produces, and did, for as long as the stub shipped. It now
reads the stage: the environment prims must exist, one env root per requested
environment beyond the source, `prims_created` must equal the measured stage delta,
and `build_time_ms` must be above zero.
