### Docs: `Gr00tPolicy.reset` no longer claims a nonexistent auto-mounted determinism wrapper

The docstring said deployments could "use the ``Robot()`` factory which
auto-mounts the wrapper" and pointed at
``examples/gr00t_server_deterministic_wrapper.py`` "in robots-sim". No
``Robot()`` auto-mount exists anywhere in this library, and neither path was
correct. A reader following the old text would forward per-episode seeds via
``reset(options={seed})`` and believe they were getting server-side
determinism while the server's ``reset`` remained the upstream no-op - the
#187 success-rate-gap failure mode this docstring exists to prevent. The
docstring now spells out that the forwarded seed does nothing unless the
server is patched, and points at the actual mechanism that ships today: the
packaged wrapper (``strands_robots.policies.groot.server_wrapper``), which
the ``gr00t_inference`` container-lifecycle tool mounts when called with
``deterministic=True`` (#1790).
