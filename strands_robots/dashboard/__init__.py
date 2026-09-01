"""Strands Robots Dashboard - fleet cockpit for the robot mesh.

``[tool.hatch.build.targets.wheel]`` packages ``strands_robots`` entire, so every
module in this package ships in every install of ``strands-robots`` -- including
installs that never asked for a web dashboard and so carry none of its server
dependencies. Importing any of them then failed with a bare
``ModuleNotFoundError: No module named 'fastapi'``, which reads as a broken
environment rather than as a missing extra, and sent the reader looking for the
fault in their own venv.

The gate below is the one chokepoint that closes that: Python executes a
package's ``__init__`` before any of its submodules, so a single call here covers
``auth`` and everything else added to this package later, and the refusal names
``strands-robots[dashboard]`` -- the extra that actually supplies the missing
module. ``require_optionals`` (plural) is deliberate: the absent extra means
several modules are missing at once, and reporting them one ImportError at a time
turns one install into four round trips.

Every distribution the gate names must also be in the ``dashboard`` extra, and
the reverse holds for anything this package imports. ``jwt`` (PyJWT) is listed
because ``auth`` signs the operator session token with it; leaving it out let the
one module it backs fail with exactly the bare ``ModuleNotFoundError`` this gate
exists to prevent, since the only copy in a developer environment arrives
transitively through an unrelated package.

``python-multipart`` is in the extra but not in this list. It is a runtime
dependency of FastAPI's form parsing rather than something this package imports,
so it has no module name worth naming in a refusal here.
"""

from strands_robots.utils import require_optionals

__all__: list[str] = []

require_optionals(
    ["fastapi", "uvicorn", "webauthn", "jwt"],
    extra="dashboard",
    purpose="the operator web dashboard (strands_robots.dashboard)",
    pip_install={"jwt": "PyJWT"},
)
