"""Turn a safety refusal into something a human can answer.

A refusal is recognised by its ``code`` and described by its ``subject`` -- the
contract :mod:`strands_robots.refusal_codes` states and ``docs/reference/security.md``
documents. The message travels with the request for the operator to read; it
decides nothing here, so the SDK may reword any refusal without this module
noticing. The environment variable a grant writes is read from
:data:`~strands_robots.refusal_codes.REFUSAL_GRANTS`, never spelled here.

One refusal is the dashboard's own and carries no SDK code: the verdict
:func:`~strands_robots.dashboard.agent_motion.agent_motion_allowed` returns
when the agent asks to start motion on real hardware. It is accepted as the
dict it already is, with the peer name passed alongside, not read back out of
its sentence.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from strands_robots import refusal_codes
from strands_robots.dashboard.agent_motion import MOTION_ENV as _AGENT_MOTION_ENV

#: Same charset the mesh allowlist validator accepts for one ENTRY (``<org>`` or
#: ``<org>/<repo>``). Validates what would be written to the allowlist, not a sentence.
_HF_ENTRY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?\Z")
#: A provider, policy type or peer name as it may be shown and granted.
_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
#: The SDK's own charset for an ENTRY in the host allowlist (security._POLICY_HOST_ENTRY_RE),
#: copied rather than imported so this module stays free of the mesh at import time.
_POLICY_HOST_ENTRY_RE = re.compile(r"^[A-Za-z0-9.:/_\-]{1,253}\Z")

#: The teleop slew bound has no refusal code yet, so it is never classified; it is written
#: alongside the value bound because the degrees preset is one envelope with two edges.
_TELEOP_SLEW_ENV = "STRANDS_MESH_INPUT_SLEW_ABS"
#: Degrees plus a percent gripper: 400 covers a multi-turn wrist with headroom and still
#: refuses a runaway three orders of magnitude out.
_TELEOP_DEGREE_VALUE = "400"
_TELEOP_DEGREE_SLEW = "800"

#: The refusal text can be a whole traceback; keep the evidence bounded.
_MAX_MESSAGE = 2000

#: The only things an operator can be asked to approve.
KINDS: tuple[str, ...] = (
    "trust_remote_code",
    "hf_repo_allow",
    "teleop_degree_units",
    "agent_physical_motion",
    "policy_type_allow",
    "policy_host_allow",
)

#: Which card answers which SDK refusal. Closed on both sides: every code in
#: :data:`~strands_robots.refusal_codes.REFUSAL_CODES` has a kind here, and the
#: environment variable for each comes from :data:`~strands_robots.refusal_codes.REFUSAL_GRANTS`.
_KIND_BY_CODE: dict[str, str] = {
    refusal_codes.TRUST_REMOTE_CODE_REQUIRED: "trust_remote_code",
    refusal_codes.HF_REPO_NOT_ALLOWED: "hf_repo_allow",
    refusal_codes.POLICY_TYPE_NOT_ALLOWED: "policy_type_allow",
    refusal_codes.POLICY_HOST_NOT_ALLOWED: "policy_host_allow",
    refusal_codes.TELEOP_VALUE_OUT_OF_RANGE: "teleop_degree_units",
}
_CODE_BY_KIND: dict[str, str] = {kind: code for code, kind in _KIND_BY_CODE.items()}


#: Every variable a consent grant may write - derived from the contract, never spelled twice.
#: This is the consent routes' own allowlist for the ``.env`` file: these keys are gate-bearing
#: by design (a grant IS a gate opening), so they are refused on the general ``/api/config``
#: page surface and admitted only here, one kind + one subject at a time.
GRANT_ENV_KEYS: frozenset[str] = frozenset(
    {_AGENT_MOTION_ENV, _TELEOP_SLEW_ENV, *refusal_codes.REFUSAL_GRANTS.values()}
)


def _env_var(kind: str) -> str:
    """The variable a grant of ``kind`` writes, read from the contract."""
    if kind == "agent_physical_motion":
        return _AGENT_MOTION_ENV
    return refusal_codes.REFUSAL_GRANTS[_CODE_BY_KIND[kind]]


def _entries(env: Mapping[str, str], var: str) -> list[str]:
    return [e.strip() for e in str(env.get(var, "")).split(",") if e.strip()]


def _truthy(env: Mapping[str, str], var: str) -> bool:
    return str(env.get(var, "")).strip().lower() in ("1", "true", "yes", "on")


def _host_entry(raw: object, *, strip_url: bool = False) -> str | None:
    """The allowlist ENTRY that grants ``raw``, or None if nothing safe can be derived.

    The host refusal's subject is either a ``policy_host`` (already an entry) or the whole
    ``server_address`` the host was taken from; ``strip_url`` reduces the latter to its host.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not strip_url:
        # Already an entry (a policy_host, or a subject the browser sent back): validate, never
        # rewrite - the approval endpoint rebuilds the request from this string.
        return s if s and _POLICY_HOST_ENTRY_RE.match(s) else None
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0]  # path (a CIDR entry never arrives via a refusal, so / cannot be kept)
    if s.startswith("["):  # bracketed IPv6, with or without a port
        if "]" not in s:
            return None
        s = s[1 : s.index("]")]
    elif s.count(":") == 1:  # host:port - an IPv6 literal has more, and keeps them
        s = s.split(":", 1)[0]
    s = s.strip()
    if not s or not _POLICY_HOST_ENTRY_RE.match(s):
        return None
    return s


@dataclass(frozen=True)
class ConsentRequest:
    """One thing the operator can approve, with the risk in the SDK's own words."""

    kind: str
    scope: str  # stable id to persist an approval against
    title: str
    risk: str
    env_var: str
    subject: str | None = None
    message: str = ""
    grants: tuple[str, ...] = field(default_factory=tuple)

    @property
    def grantable(self) -> bool:
        """Would approving this actually change the environment?"""
        return bool(env_patch(self, {}))

    def as_dict(self) -> dict:
        """The request as the ``needs_consent`` payload the dashboard renders.

        Every field the browser needs to draw the approval button and to send the
        request back to the approval endpoint, plus the computed ``grantable``.
        """
        return {
            "kind": self.kind,
            "scope": self.scope,
            "title": self.title,
            "risk": self.risk,
            "env_var": self.env_var,
            "subject": self.subject,
            "message": self.message,
            "grants": list(self.grants),
            # Computed against an EMPTY env deliberately: the question is "is there a grant here at all",
            # not "is it already in place on this machine" - which env_patch answers for the live env at
            # approval time, and which must not disable the button (a refusal from a process started
            # before the last approval still needs its explanation).
            "grantable": self.grantable,
        }


def build_request(kind: str, subject: object = None, message: str = "") -> ConsentRequest | None:
    """Construct a consent request from ``kind`` + ``subject``, validating both."""
    if kind not in KINDS:
        return None
    text = message.strip()[:_MAX_MESSAGE] if isinstance(message, str) else ""
    name = subject.strip() if isinstance(subject, str) and subject.strip() else None
    env_var = _env_var(kind)

    if kind == "trust_remote_code":
        if name is not None and not _PROVIDER_NAME_RE.match(name):
            name = None
        shown = name or "this policy provider"
        return ConsentRequest(
            kind=kind,
            scope="trust_remote_code",
            title=f"Run model code from HuggingFace ({shown})?",
            risk=(
                f"{shown} loads the model with trust_remote_code=True: code stored in the "
                "model repository executes on this machine, with your files and your robots. "
                "Approve only for organisations you trust."
            ),
            env_var=env_var,
            subject=name,
            message=text,
            grants=("run repository code for every policy load from now on",),
        )

    if kind == "agent_physical_motion":
        # Subject is the peer that was refused, for the wording only: the grant is MACHINE-wide,
        # because the gate reads one env var.
        if name is not None and not _PROVIDER_NAME_RE.match(name):
            name = None
        shown = name or "a real robot"
        return ConsentRequest(
            kind=kind,
            scope="agent_physical_motion",
            title="Let the agent start motion on real robots?",
            risk=(
                f"The fleet agent asked to run a task on {shown}, which is real hardware. Approving "
                "lets it start physical motion on ANY real robot on this mesh from now on - by itself, "
                "from a chat sentence or a voice command, with no confirmation step and without the "
                "check that the policy fits the robot that the play button performs. It cannot see your "
                "room. Stopping is never gated either way, so 'everyone stop' works regardless."
            ),
            env_var=env_var,
            subject=name,
            message=text,
            grants=("start tasks on real robots unattended, until you revoke it",),
        )

    if kind == "teleop_degree_units":
        # Subject is the joint key the frame was refused for, for the dialog's wording only: the
        # envelope is a MACHINE-wide setting, so the scope deliberately is not per-joint.
        shown = f"joint {name}" if name else "this arm"
        return ConsentRequest(
            kind=kind,
            scope="teleop_degree_units",
            title="Set the teleop envelope to degrees?",
            risk=(
                f"The mesh refuses the teleop frame for {shown} because its safety envelope "
                f"assumes RADIANS (4*pi, about 12.57) and the arm reports DEGREES (a wrist at 170). "
                f"Approving widens the envelope to {_TELEOP_DEGREE_VALUE} units and the per-joint "
                f"speed bound to {_TELEOP_DEGREE_SLEW} units/s for every teleop stream on this "
                "machine - a wider envelope means a single frame may command a longer reach, so a "
                "faulty leader can ask for a bigger move before the bound stops it. It stays an "
                "envelope: a runaway three orders of magnitude out is still refused."
            ),
            env_var=env_var,
            subject=name,
            message=text,
            grants=(
                f"{env_var}={_TELEOP_DEGREE_VALUE} (how far one frame may reach)",
                f"{_TELEOP_SLEW_ENV}={_TELEOP_DEGREE_SLEW} (how fast one joint may be driven)",
            ),
        )

    if kind == "policy_host_allow":
        # Subject is normalised to an ENTRY, not kept as the operator typed it: see _host_entry.
        host = _host_entry(name)  # validate only: classify_refusal already derived the entry
        shown = host or "that address"
        return ConsentRequest(
            kind=kind,
            scope=f"policy_host_allow:{host}" if host else "policy_host_allow",
            title=f"Let policies run on {shown}?",
            risk=(
                f"The mesh only talks to policy servers on loopback by default, and {shown} is not "
                "in the allowlist. Approving sends your robots' camera frames and joint states to "
                "that host, and lets the actions it returns drive real hardware - so it is trusted "
                "with what the arms SEE and what they DO. Hostnames are matched literally with no "
                "DNS resolution, so this trusts whatever that name resolves to at the time; an IP "
                "literal keeps the boundary under your control."
            ),
            env_var=env_var,
            subject=host,
            message=text,
            grants=(f"reach the policy server at {host}" if host else "nothing yet - the host could not be read",),
        )

    if kind == "policy_type_allow":
        # One variable, two things the SDK refuses with it (provider and policy_type).
        if name is not None and not _PROVIDER_NAME_RE.match(name):
            name = None  # unparseable/hostile: ask, but grant nothing automatically
        shown = name or "the requested policy"
        return ConsentRequest(
            kind=kind,
            scope=f"policy_type_allow:{name}" if name else "policy_type_allow",
            title=f"Allow the policy {shown}?",
            risk=(
                f"{shown} is not in this machine's policy allowlist, so the mesh refused to build "
                "it. A policy decides what the arms DO, and approving lets this one be constructed "
                "and run on real hardware from now on. Exactly this name is added - no wildcard, "
                "and no other provider."
            ),
            env_var=env_var,
            subject=name,
            message=text,
            grants=(f"build and run the policy {name}" if name else "nothing yet - the policy name could not be read",),
        )

    if name is not None and not _HF_ENTRY_RE.match(name):
        name = None  # unparseable/hostile: ask, but grant nothing automatically
    shown = name or "the requested model"
    return ConsentRequest(
        kind=kind,
        scope=f"hf_repo_allow:{name}" if name else "hf_repo_allow",
        title=f"Allow the model {shown}?",
        risk=(
            f"{shown} is not in this machine's HuggingFace allowlist, so the mesh refused "
            "to load it. Approving adds exactly this repository - no other org, no wildcard."
        ),
        env_var=env_var,
        subject=name,
        message=text,
        grants=(f"load {name}" if name else "nothing yet - the repository name could not be read",),
    )


def _agent_motion_request(verdict: Mapping[str, object], subject: object) -> ConsentRequest | None:
    """The dashboard's own refusal: a gated, physical, not-granted verdict from agent_motion."""
    if verdict.get("allowed") is not False or not verdict.get("gated") or not verdict.get("physical"):
        return None
    reason = verdict.get("reason")
    return build_request("agent_physical_motion", subject, reason if isinstance(reason, str) else "")


def classify_refusal(refusal: object, *, subject: object = None) -> ConsentRequest | None:
    """Recognise a *continuable* refusal by its code, else ``None``.

    ``refusal`` is an exception carrying ``.code`` and ``.subject``
    (``SecurityError``, ``UntrustedRemoteCodeError``), or a mapping with ``code`` /
    ``subject`` / ``message`` keys (the wire shape), or the verdict dict
    :func:`~strands_robots.dashboard.agent_motion.agent_motion_allowed` returns, in
    which case ``subject`` names the peer. A refusal with ``code`` ``None`` is not
    continuable; a code outside :data:`~strands_robots.refusal_codes.REFUSAL_CODES` is
    left for the message to explain. Prose is never inspected.
    """
    if isinstance(refusal, Mapping):
        if "code" not in refusal and "allowed" in refusal:
            return _agent_motion_request(refusal, subject)
        code = refusal.get("code")
        about = refusal.get("subject")
        text = refusal.get("message", "")
    elif isinstance(refusal, BaseException):
        code = getattr(refusal, "code", None)
        about = getattr(refusal, "subject", None)
        text = str(refusal)
    else:
        return None  # a bare string has no code, so it has nothing to offer

    if not isinstance(code, str):
        return None
    kind = _KIND_BY_CODE.get(code)
    if kind is None:
        return None
    if kind == "policy_host_allow":
        # The subject is the host, or the whole server_address the host was taken from.
        about = _host_entry(about, strip_url=True)
    return build_request(kind, about, text if isinstance(text, str) else "")


def env_patch(request: ConsentRequest, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The smallest env change that grants ``request``, given the current ``env``."""
    env = env or {}
    var = request.env_var

    if request.kind in ("trust_remote_code", "agent_physical_motion"):
        return {} if _truthy(env, var) else {var: "1"}

    if request.kind == "teleop_degree_units":
        patch = {}
        if str(env.get(var, "")).strip() != _TELEOP_DEGREE_VALUE:
            patch[var] = _TELEOP_DEGREE_VALUE
        if str(env.get(_TELEOP_SLEW_ENV, "")).strip() != _TELEOP_DEGREE_SLEW:
            patch[_TELEOP_SLEW_ENV] = _TELEOP_DEGREE_SLEW
        return patch

    if request.kind == "policy_host_allow":
        host = _host_entry(request.subject)
        if not host:
            return {}
        current = _entries(env, var)
        return {} if host in current else {var: ",".join(current + [host])}

    if request.kind == "policy_type_allow":
        name = request.subject
        if not name or not _PROVIDER_NAME_RE.match(name):
            return {}
        current = _entries(env, var)
        # No org shortcut here: a policy name has no hierarchy, so nothing broader can cover it.
        return {} if name in current else {var: ",".join(current + [name])}

    if request.kind == "hf_repo_allow":
        repo = request.subject
        if not repo or not _HF_ENTRY_RE.match(repo):
            return {}
        current = _entries(env, var)
        org = repo.split("/", 1)[0]
        # An existing broader entry (the org, or the repo itself) already covers it.
        if repo in current or org in current:
            return {}
        return {var: ",".join(current + [repo])}

    return {}


def granted_state(env: Mapping[str, str] | None = None) -> dict:
    """What this machine currently grants - every kind, in one place."""
    env = os.environ if env is None else env
    value_abs = str(env.get(_env_var("teleop_degree_units"), "")).strip()
    slew_abs = str(env.get(_TELEOP_SLEW_ENV, "")).strip()
    return {
        "kinds": list(KINDS),
        "trust_remote_code": _truthy(env, _env_var("trust_remote_code")),
        # Reported as what the environment ACTUALLY holds: a hand-set "on" is in force and must be
        # visible, or the screen would deny a permission the agent is currently using.
        "agent_physical_motion": _truthy(env, _env_var("agent_physical_motion")),
        "hf_repo_allow": _entries(env, _env_var("hf_repo_allow")),
        # Shown for the same reason the teleop envelope had to be: a grant with no surface cannot
        # be revoked, while the dialog promises it can.
        "policy_type_allow": _entries(env, _env_var("policy_type_allow")),
        # Loopback is allowed by the SDK's own default and is NOT listed: this key answers "what has
        # this machine been opened up to", and printing localhost as a grant would bury the one entry
        # that matters among defaults nobody approved.
        "policy_host_allow": _entries(env, _env_var("policy_host_allow")),
        "teleop_degree_units": {
            "granted": bool(value_abs or slew_abs),
            "value_abs": value_abs or None,
            "slew_abs": slew_abs or None,
            # True only when it is exactly the pair this module grants; a hand-tuned wider bound
            # must not be described to the operator as "the degrees preset".
            "is_degree_preset": value_abs == _TELEOP_DEGREE_VALUE and slew_abs == _TELEOP_DEGREE_SLEW,
        },
    }


def revoke_patch(request: ConsentRequest, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The env change that takes a grant BACK, given the current ``env``."""
    env = env or {}
    var = request.env_var

    if request.kind == "trust_remote_code":
        return {var: ""} if _truthy(env, var) else {}

    if request.kind == "agent_physical_motion":
        # Cleared rather than set to "0": an absent line lets a stale 1 from a shell profile win the
        # next restart, and a revocation that does not hold across a restart is the worst kind.
        return {var: ""} if str(env.get(var, "")).strip() else {}

    if request.kind == "teleop_degree_units":
        # Back to the SDK defaults by CLEARING both, not by writing 12.566...:
        # a number frozen here would silently override a future SDK default.
        patch = {}
        if str(env.get(var, "")).strip():
            patch[var] = ""
        if str(env.get(_TELEOP_SLEW_ENV, "")).strip():
            patch[_TELEOP_SLEW_ENV] = ""
        return patch

    if request.kind in ("policy_host_allow", "policy_type_allow", "hf_repo_allow"):
        entry = _host_entry(request.subject) if request.kind == "policy_host_allow" else request.subject
        if not entry:
            return {}
        current = _entries(env, var)
        if entry not in current:
            # A broader entry the operator added by hand (an org, a CIDR) may still cover it; say
            # nothing changed rather than narrowing something this module did not write.
            return {}
        return {var: ",".join(e for e in current if e != entry)}

    return {}


def attach_consent(payload: dict, *sources: object, subject: object = None) -> dict:
    """Add ``needs_consent`` to an error ``payload`` if any source is continuable."""
    for source in sources:
        request = classify_refusal(source, subject=subject)
        if request is not None:
            payload["needs_consent"] = request.as_dict()
            break
    return payload
