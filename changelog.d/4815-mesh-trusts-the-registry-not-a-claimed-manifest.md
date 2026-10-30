### Changed: a mesh trusts a claimed name only in-process, and a connected Robot trusts the registry over a manifest

A claimed name is not a credential. A resume by name is now accepted without operator keys only when the
transport is genuinely in-process (`mem://`, the fake IoT broker); a zenoh endpoint, even one that only listens
on loopback, accepts connections from any other local process (a container, another user, an `ssh -L` tunnel),
so resuming over it needs `Mesh(resume_authority=Authority(public_keys=...))` or an explicit `trust_names=True`.
`Mesh.connect` already took a device's limits and action_keys from the local registry by the announced type, not
from the peer's manifest; this is now pinned, so a peer cannot widen a device's limits or reorder its keys by
lying in the manifest it announces.
