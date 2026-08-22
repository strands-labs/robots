"""Stable machine-readable identifiers for the refusals an operator can answer.

Some refusals are *continuable*: the request was well formed, and an operator
who accepts the risk can grant something that makes the identical request
succeed. A consumer that wants to offer that choice -- a UI consent card, an
approval endpoint, a supervising agent -- has to recognise which refusal it is
looking at and what it is about.

Without a code the only thing on offer is the message text, so recognition
becomes prose matching: an env-var name appearing somewhere in the sentence,
the subject pulled back out with an anchored regex. That makes every wording
improvement a silent breaking change for every consumer, and nothing in this
package can detect it -- a reworded refusal keeps the whole suite green.

The codes here are that missing contract, in the shape
:data:`~strands_robots.episode_labels.FAILURE_MODES` already uses for the same
reason: a fixed vocabulary so consumers match on identity rather than parsing
prose, with the free text left free. A refusal carries its code and its
subject structurally (``exc.code`` / ``exc.subject``); the message stays a
message, and may be improved without breaking anyone.

Only continuable refusals get a code. ``instruction exceeds 4096 chars`` is
not continuable -- there is no grant that makes it succeed, so a consumer has
nothing to offer and nothing to recognise. :data:`REFUSAL_GRANTS` records, per
code, the operator grant that lifts it, which is also what lets the guard test
derive the raise sites it must find instead of keeping a hand-written list.
"""

from __future__ import annotations

#: A HuggingFace-backed policy provider would execute code from a model
#: repository. Subject: the provider name.
TRUST_REMOTE_CODE_REQUIRED = "TRUST_REMOTE_CODE_REQUIRED"

#: A model repo is outside the mesh allowlist. Subject: the repo id.
HF_REPO_NOT_ALLOWED = "HF_REPO_NOT_ALLOWED"

#: A policy type or provider is outside the mesh allowlist. Subject: the type
#: or provider name (both share one allowlist, so both carry this code).
POLICY_TYPE_NOT_ALLOWED = "POLICY_TYPE_NOT_ALLOWED"

#: A policy host is outside the mesh allowlist. Subject: the host, or the
#: whole ``server_address`` the host was taken from.
POLICY_HOST_NOT_ALLOWED = "POLICY_HOST_NOT_ALLOWED"

#: A teleop input frame commands a joint past the value envelope. Subject: the
#: joint key. This refusal names no env var at all, which is why a consumer
#: today has to recognise it by its own words.
TELEOP_VALUE_OUT_OF_RANGE = "TELEOP_VALUE_OUT_OF_RANGE"

#: Every code this package raises. Closed: a consumer may switch on it.
REFUSAL_CODES: tuple[str, ...] = (
    TRUST_REMOTE_CODE_REQUIRED,
    HF_REPO_NOT_ALLOWED,
    POLICY_TYPE_NOT_ALLOWED,
    POLICY_HOST_NOT_ALLOWED,
    TELEOP_VALUE_OUT_OF_RANGE,
)

#: The operator grant that lifts each refusal. This is what makes a refusal
#: continuable, and it is the environment variable a consumer offering the
#: choice has to set -- reading it from here rather than hard-coding it keeps
#: the consumer and this package from drifting apart.
REFUSAL_GRANTS: dict[str, str] = {
    TRUST_REMOTE_CODE_REQUIRED: "STRANDS_TRUST_REMOTE_CODE",
    HF_REPO_NOT_ALLOWED: "STRANDS_MESH_HF_REPO_ALLOW",
    POLICY_TYPE_NOT_ALLOWED: "STRANDS_MESH_POLICY_TYPE_ALLOW",
    POLICY_HOST_NOT_ALLOWED: "STRANDS_MESH_POLICY_HOST_ALLOW",
    TELEOP_VALUE_OUT_OF_RANGE: "STRANDS_MESH_INPUT_VALUE_ABS",
}

__all__ = [
    "HF_REPO_NOT_ALLOWED",
    "POLICY_HOST_NOT_ALLOWED",
    "POLICY_TYPE_NOT_ALLOWED",
    "REFUSAL_CODES",
    "REFUSAL_GRANTS",
    "TELEOP_VALUE_OUT_OF_RANGE",
    "TRUST_REMOTE_CODE_REQUIRED",
]
