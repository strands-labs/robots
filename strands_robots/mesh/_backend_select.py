"""Sole owner of the ``STRANDS_MESH_BACKEND`` vocabulary.

Two modules resolve that variable and both must agree.
:func:`strands_robots.mesh.session._backend_choice` runs first, on every
session and publish path, and its verdict is what
``session._is_transport_backend()`` gates on;
:func:`strands_robots.mesh.transport.factory.get_transport` runs only once that
verdict is ``iot`` or ``bridge``. Each used to read the variable itself, and
they disagreed about the one case that matters: the factory reported an unknown
value and the gate in front of it did not, so a typo resolved to ``zenoh``, the
factory was never consulted, and the report could not fire for the input class
it named. ``STRANDS_MESH_BACKEND=iott`` produced a plain Zenoh session
indistinguishable from an explicit ``zenoh``.

This module holds the accepted values, the fallback and the report, so the two
readers cannot drift again. It imports nothing from the mesh package: the
factory reaches ``session`` through ``transport.zenoh_transport``, so a resolver
either reader had to import from the other would close that cycle, and a
call-time import inside ``session`` would raise in the documented "no zenoh
installed" case, where ``get_session`` must return ``None`` rather than
propagate an ``ImportError``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: The env var this module owns.
BACKEND_ENV_VAR = "STRANDS_MESH_BACKEND"

#: Transports :func:`strands_robots.mesh.transport.factory.get_transport` can
#: construct. Anything else is a typo.
BACKENDS = ("zenoh", "iot", "bridge")

#: Where an unset or unrecognized value lands.
DEFAULT_BACKEND = "zenoh"

#: Unknown values already reported. Keyed by the offending value, so a second
#: distinct typo is still news. A set mutated via ``.add`` (never reassigned)
#: avoids a ``global`` rebind, matching
#: ``strands_robots.simulation.predicates._RESOLUTION_WARNED``.
_UNKNOWN_WARNED: set[str] = set()


def select_backend() -> str:
    """Return the configured mesh transport, one of :data:`BACKENDS`.

    Case and surrounding whitespace are normalised, so ``IOT`` and ``" iot "``
    both select ``iot``. An unset variable selects :data:`DEFAULT_BACKEND`.

    An unrecognized value also falls back to :data:`DEFAULT_BACKEND` - the
    policy is to keep the mesh running rather than crash the host on a typo -
    and is reported once per distinct offending value. Once per value rather
    than once per call because the gate that consults this runs per published
    message: reporting every call would put one line per telemetry sample in
    the operator's log.

    Returns:
        The selected backend name.
    """
    raw = os.getenv(BACKEND_ENV_VAR, DEFAULT_BACKEND).strip().lower()
    if raw in BACKENDS:
        return raw
    if raw not in _UNKNOWN_WARNED:
        _UNKNOWN_WARNED.add(raw)
        logger.warning(
            "Unknown %s=%r - falling back to %r. Valid values: %s.",
            BACKEND_ENV_VAR,
            raw,
            DEFAULT_BACKEND,
            ", ".join(BACKENDS),
        )
    return DEFAULT_BACKEND
