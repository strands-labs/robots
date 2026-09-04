"""Single owner of the operator-response audit row every HITL gate owes.

Four gates stop and ask a human before an agent-issued command reaches a robot or
a training run. Three are tool bodies: :mod:`~strands_robots.tools.robot_mesh`
gates the physical-actuation mesh actions, :mod:`~strands_robots.tools.use_ros`
gates a ``publish`` / ``service_call`` / ``action_send_goal`` aimed at a
safety-critical graph surface, and :mod:`~strands_robots.tools.lerobot_train`
gates the ``extra_flags`` that control output paths, telemetry and code loading.
The fourth is not a tool at all: the dashboard's motion hook gates any agent tool
call that would put real hardware in motion, from outside the tool being called.

That fourth caller is why this list is worth keeping current. The audit log is
read by grepping one phrasing, and a reader who believes only tools ask a human
has no reason to look for a row from a hook -- so a gate this docstring does not
name is a gate an incident reader does not know to search for. The graded set is
derived from the ``interrupt()`` call sites under ``strands_robots`` rather than
from this paragraph, so a fifth gate fails that grading on arrival instead of
inheriting the silence -- but the derivation cannot correct the prose, which is
why the count above is worth reading as a claim rather than as decoration.

Each gate owes the operator's reply two things that pull in opposite directions.
The reply must NOT reach the model - echoing it turns the human into a
prompt-injection content side-channel, because an agent that authors the approval
reason can make the operator's typed answer carry data back into the context - so
every gate returns a flat, fixed sentinel. But the reply is also the only record
of *why* a commanded action did or did not happen: every gate accepts a canonical
affirmative only, so a reply that carries a reason is always a decline, and the
local audit row is the one place that reason survives.

This module owns the second half. Which fields the row carries, and the wording a
forensic reader greps for, must not differ between two gates writing to the same
audit log - one gate spelling the row ``operator refused`` would be invisible to a
search for another's ``operator declined``, and the reader has no way to know a
third gate wrote nothing at all.

The audit module is imported inside the call rather than at module scope because
``strands_robots.mesh`` pulls in the transport stack, and a tool that gates a ROS
graph or a training run must not pay for that on import.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Bound on the recorded reply, matching the audit detail bound the mesh tool
#: already applies: an operator can paste an arbitrarily long reason, and one
#: audit row must not become the reason the log is unusable.
_MAX_DETAIL = 500


def log_operator_response(
    source: str,
    action: str,
    target: str,
    *,
    approved: bool,
    response: object,
) -> None:
    """Record an operator's HITL interrupt reply in the local audit log.

    Call this on BOTH outcomes, from ONE site the gate reaches as soon as the
    verdict is known and before any later refusal can return. A decline is the row
    that carries the operator's reason; an approval is the record that a human
    authorised an agent to reach a physical surface, which is the question an audit
    of an incident asks first - and a gate may still refuse an APPROVED action for
    its own reasons (the mesh tool re-checks its rate limit under the lock, and a
    concurrent invocation can take the last slot while the operator is deciding).
    Recording per-branch after such a check leaves that path with no operator row
    at all, so the log says only why the action was refused and nothing about who
    authorised it.

    Args:
        source: Which tool asked, e.g. ``"use_ros_tool"``. Recorded as the event
            source so a reader can tell which gate a row came from.
        action: The gated verb - the mesh action, the ``use_ros`` command verb, or
            ``"train"``.
        target: What the verb was aimed at: a peer, a graph surface, or the
            blocked flag names.
        approved: The verdict the gate reached from *response*.
        response: The operator's literal reply, recorded verbatim (bounded, and
            ``repr``-quoted so a reply containing a newline cannot forge a second
            record). It MUST NOT be returned to the model by the caller.

    Raises:
        Nothing. An audit write that fails is logged at DEBUG and swallowed: this
        runs inside a safety gate, and an unwritable audit log must not turn a
        decline into an exception the gate never accounted for. The catch is
        deliberately wide for that reason - see AGENTS.md > "Exception Clauses
        Must Be Narrow" - because the write reaches the filesystem and a
        JSON encoder, and neither failure mode is worth enumerating here.
    """
    verdict = "approved" if approved else "declined"
    detail = f"operator {verdict}: {response!r}"
    try:
        from strands_robots.mesh.audit import log_safety_event

        log_safety_event(
            "llm_tool_action",
            source,
            {
                "action": action,
                "target": target,
                "success": approved,
                "detail": detail[:_MAX_DETAIL],
            },
        )
    except Exception as audit_exc:  # noqa: BLE001 - see Raises
        logger.debug("[%s] operator-response audit unavailable: %s", source, audit_exc)
