### Fixed: the Isaac Sim install hint named a docker image that does not exist

`ISAAC_SIM_DOCKER_IMAGE` was `nvcr.io/nvidia/isaac-sim:6.0`, and NVIDIA publishes
no `major.minor` tag for that image. Measured against the registry with
`docker manifest inspect`:

```
isaac-sim:6.0     -> no such manifest
isaac-sim:latest  -> no such manifest
isaac-sim:6.0.0   -> exists
isaac-sim:6.0.1   -> exists
isaac-sim:5.0.0   -> exists
isaac-sim:4.5.0   -> exists
```

What made this worse than a stale line in a document is *where* that constant is
read. It is the single source the **recovery instructions** are composed from:
`IsaacSimulation.is_available()` returns it in the hint it gives when the runtime
is absent, and `create_world()` names it in the structured error it returns for the
same reason. So the one message a user sees when they have no Isaac Sim installed
told them to pull an image that cannot be pulled.

Now pinned to `6.0.1`, the tag this backend is verified against on an A10G, and
pinned by a test that grades the property a registry lookup would confirm - a
resolvable tag carries all three version components - so it needs no network call
and fails on the exact value that shipped. The two pre-existing tests over this
constant assert only that it appears *in* those messages, which a wrong tag
satisfies perfectly.
