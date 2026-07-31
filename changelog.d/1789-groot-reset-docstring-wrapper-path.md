### Docs: `Gr00tPolicy.reset` no longer claims a nonexistent auto-mounted determinism wrapper

The docstring said deployments could "use the ``Robot()`` factory which
auto-mounts the wrapper" and pointed at
``examples/gr00t_server_deterministic_wrapper.py`` "in robots-sim". No
auto-mount exists anywhere in this library, and the wrapper lives in this repo
at ``examples/libero/gr00t_server_deterministic_wrapper.py`` (moved in #1282).
A reader following the old text would forward per-episode seeds via
``reset(options={seed})`` and believe they were getting server-side
determinism while the server's ``reset`` remained the upstream no-op - the
#187 success-rate-gap failure mode this docstring exists to prevent. The
docstring now names the correct in-repo path, states that the wrapper must be
mounted manually into the container, and cross-references #1790 (curated
``deterministic=`` lifecycle mount) as the future no-manual-docker path.
