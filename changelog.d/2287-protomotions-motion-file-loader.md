### Security

- `policies/protomotions`: a raw `.pt` reference motion is now read with
  `torch.load(..., weights_only=True)`. A motion file is an artifact that
  travels between machines, and the unrestricted unpickler executes whatever
  `__reduce__` the file names while reading it, so loading a third-party clip
  was enough to run code on the host. The documented payload is tensors plus two
  scalars, which the restricted unpickler reads, so no supported motion stops
  loading; one it refuses is reported with the `save_cache_npz` conversion route
  instead of a bare `UnpicklingError`.
