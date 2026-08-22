"""Sole owner of the ``STRANDS_MESH`` vocabulary.

Two modules resolve that variable and neither can report a typo on its own.
:func:`strands_robots.robot._mesh_env_opt_in` reads the affirmative spellings to
decide whether a bare ``Robot()`` opts in, while
:func:`strands_robots.mesh.core.mesh_disabled_by_env`
reads the negative ones to decide whether an explicit ``mesh=True`` is
overridden, and answers the same question for the robot-less gateway peer in
:mod:`strands_robots.tools.robot_mesh`.

Each half correctly treats the other's spellings as none of its business: to the
opt-in reader ``false`` is simply not an opt-in, and to the kill switch ``true``
is simply not a kill. So neither is in a position to call a *third* value a
mistake -- doing that requires knowing both halves, to tell a typo apart from a
legitimate spelling the reader does not own. The silence was structural rather
than an omission at either site, which is why the report lives in one owner here
instead of being added twice.

The direction that falls through is not the safe one. ``STRANDS_MESH=off`` reads
as neither half, so an operator who asked for no mesh and passed an explicit
``mesh=True`` still got a real Zenoh session, a ``gateway-*`` peer advertised to
the fleet and the threads ``Mesh.start`` spawns -- the outcome #2515 removed for
the gateway call site, arriving through the vocabulary instead. ``off`` is not a
hypothetical spelling here: ``STRANDS_ROBOT_MESH_DC`` accepts ``("off", "0",
"false", "no")`` and ``REACHY_DAEMON_TLS`` accepts ``("1", "true", "yes",
"on")``, so the package itself teaches both words it ignores on this variable.

This module owns the accepted values and the report, and nothing else: whether
``off`` should *mean* ``false`` is a behaviour change on a safety switch and is
deliberately not decided here. The warning names the spellings that work, which
is what the "warn on unrecognized values" rule in AGENTS.md asks of it.

It imports nothing from ``strands_robots``. :mod:`strands_robots.robot` reaches
the mesh package only lazily, inside the function that starts a mesh, because
``mesh/__init__`` eagerly imports the Zenoh-backed session and core; a resolver
living under ``mesh`` would make that import eager at ``Robot`` import time and
break the documented "no zenoh installed" path.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: The env var this module owns.
MESH_ENV_VAR = "STRANDS_MESH"

#: Spellings that opt a bare ``Robot()`` into the mesh.
AFFIRMATIVE = ("true", "1", "yes")

#: Spellings that trip the hard kill switch. The kill-switch regression test
#: parametrizes over this tuple directly; ``mesh.core`` deliberately keeps no
#: copy of it, and the vocabulary guard asserts that it does not.
NEGATIVE = ("false", "0", "no")

#: Unrecognized values already reported. Keyed by the offending value, so a
#: second distinct typo is still news. A set mutated via ``.add`` (never
#: reassigned) avoids a ``global`` rebind, matching ``_UNKNOWN_WARNED`` in
#: :mod:`strands_robots.mesh._backend_select`.
_UNKNOWN_WARNED: set[str] = set()


def mesh_env_request() -> bool | None:
    """Report what ``STRANDS_MESH`` asks for, as a tristate.

    Case and surrounding whitespace are normalised, so ``FALSE`` and
    ``"  false  "`` both read as the kill switch.

    The three outcomes are not two: an unset variable is not a request for a
    mesh *or* against one, and collapsing it into either would make one of the
    two readers wrong. Returning ``None`` for "said nothing" is what lets both
    ask this one resolver -- ``is True`` for the opt-in, ``is False`` for the
    kill switch -- and it keeps the property their docstrings already state,
    that an unset variable answers False to both.

    An unrecognized non-empty value also says nothing, and is reported once per
    distinct offending value. Once per value rather than once per call because
    every construction site of a ``Mesh`` consults the kill switch: reporting
    per call would put one line per bring-up in the operator's log. It reports
    rather than raises, so a typo cannot crash a robot host.

    Returns:
        True when the environment opts in, False when it forces the mesh off,
        and None when it is unset, empty or unrecognized.
    """
    raw = os.getenv(MESH_ENV_VAR, "").strip().lower()
    if raw in AFFIRMATIVE:
        return True
    if raw in NEGATIVE:
        return False
    if raw and raw not in _UNKNOWN_WARNED:
        _UNKNOWN_WARNED.add(raw)
        logger.warning(
            "Unrecognized %s=%r - ignored, so the mesh is neither opted into "
            "nor disabled by it. To turn the mesh on use one of: %s. To force "
            "it off (overriding an explicit mesh=True) use one of: %s.",
            MESH_ENV_VAR,
            raw,
            ", ".join(AFFIRMATIVE),
            ", ".join(NEGATIVE),
        )
    return None
