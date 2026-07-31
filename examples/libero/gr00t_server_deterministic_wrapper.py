"""Back-compat shim - the determinism wrapper moved into the package.

The canonical, wheel-shipped wrapper now lives at
``strands_robots/policies/groot/server_wrapper.py`` so installed users have
it on disk without a repo checkout, and so the ``gr00t_inference`` tool can
mount it via the ``deterministic=True`` lifecycle flag::

    gr00t_inference(action="lifecycle", lifecycle="full", deterministic=True,
                    protocol="n1.7", ...)

If you were mounting THIS file into the container by hand, mount the
packaged file instead (the escape hatch for custom container setups)::

    WRAP=$(python -c "import strands_robots.policies.groot.server_wrapper as m; print(m.__file__)")
    docker run ... -v "$WRAP":/srv_wrap.py:ro \\
      gr00t:latest python /srv_wrap.py --model-path ... --use-sim-policy-wrapper --port 8000

This shim re-exports :func:`main` for host-side importers. It cannot run
mounted alone inside the container anymore (``strands-robots`` is not
installed there); the error below says where to point the mount instead.
"""

from __future__ import annotations

try:
    from strands_robots.policies.groot.server_wrapper import main
except ImportError as e:  # pragma: no cover - only reachable inside a bare container
    raise ImportError(
        "gr00t_server_deterministic_wrapper.py is now a thin shim; the real "
        "wrapper moved to strands_robots/policies/groot/server_wrapper.py. "
        "Mount that file into the container instead (or pass deterministic=True "
        "to the gr00t_inference lifecycle tool, which mounts it for you)."
    ) from e

__all__ = ["main"]

if __name__ == "__main__":
    main()
