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
code, the operator grant that lifts it, so a consumer reads the variable to
offer from here rather than hard-coding it.

That table is deliberately *not* what scopes the guard test. Deriving the raise
sites it must find from the grants already listed here would only ever find a
refusal naming a grant this module already knows, so the first refusal to offer
a *new* grant -- the one case the guard exists for -- would be invisible. The
scope comes from the exception types that can carry a code at all instead.

Scope has a second dimension: how a raise site *spells* the code it passes. A
code is checked against this vocabulary by a static scan, because nothing
validates it at runtime -- it is stored as given, and raising from an exception
constructor would replace a security refusal with a constructor error. So the
scan reads the value in every spelling that reaches a declared code: an
attribute of this module, a name imported from it (every code is exported under
its own name and listed in ``__all__``, so that is a first-class way to reach
one), or the literal string. A spelling it cannot read is reported rather than
skipped, because a code nothing can check is a code that may not be in this
vocabulary at all.
"""

from __future__ import annotations

#: A HuggingFace-backed policy provider would execute code from a model
#: repository. Subject: the provider name. Grant: set the variable to
#: ``1``; the subject is not the value.
TRUST_REMOTE_CODE_REQUIRED = "TRUST_REMOTE_CODE_REQUIRED"

#: A model repo is outside the mesh allowlist. Subject: the repo id.
#: Grant: add the subject to the allowlist.
HF_REPO_NOT_ALLOWED = "HF_REPO_NOT_ALLOWED"

#: A policy type or provider is outside the mesh allowlist. Subject: the type
#: or provider name (both share one allowlist, so both carry this code).
#: Grant: add the subject to the allowlist.
POLICY_TYPE_NOT_ALLOWED = "POLICY_TYPE_NOT_ALLOWED"

#: A policy host is outside the mesh allowlist. Subject: the host, or the
#: whole ``server_address`` the host was taken from. Grant: add the
#: subject to the allowlist.
POLICY_HOST_NOT_ALLOWED = "POLICY_HOST_NOT_ALLOWED"

#: A teleop input frame commands a joint past the value envelope. Subject: the
#: joint key. Grant: raise the bound above the refused magnitude. The
#: subject is not the value here, and the magnitude appears only in the
#: message -- so a consumer offering this grant still has to read it out
#: of the prose.
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
#:
#: The variable is half the answer: a consumer also has to know what to set it
#: to, and that differs by code. Three of these are allowlists the refusal's
#: own ``subject`` is appended to; the other two are not, and applying the
#: subject to them is a silent no-op that returns the identical message. Each
#: code states its own operation above. A grant applied to the wrong code is
#: equally silent, which is why the pairing is driven by a test rather than
#: taken on trust.
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
