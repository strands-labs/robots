"""Judge whether a dataset upload to the Hub can succeed before the operator clicks it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["upload_preflight", "destination"]


def destination(dataset: str, user: str | None) -> str | None:
    """The repo id a push would really create, or None when it cannot be known yet."""
    name = (dataset or "").strip().strip("/")
    if not name:
        return None
    if "/" in name:
        return name
    if not user:
        return None
    return f"{user}/{name}"


def upload_preflight(
    *,
    dataset: str | None,
    auth: Mapping[str, Any] | None,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge an upload before it is armed. Returns ``{ok, state, detail, destination, user}``."""
    name = (dataset or "").strip().strip("/")
    a = dict(auth or {})
    user = a.get("user") if isinstance(a.get("user"), str) else None
    authed = a.get("authenticated") is True

    if not name:
        return {
            "ok": False,
            "state": "no_dataset",
            "needs_force": False,
            "user": user,
            "destination": None,
            "detail": (
                "this session has no dataset name yet, so there is nowhere to publish it. "
                "Name the dataset when you open the session"
            ),
        }

    if not authed:
        detail = a.get("detail")
        rejected = isinstance(detail, str) and "rejected" in detail
        return {
            "ok": False,
            "state": "credential_rejected" if rejected else "no_credential",
            "needs_force": False,
            "user": None,
            "destination": destination(name, None),
            "detail": (
                # A revoked token and no token behave differently and are fixed differently.
                "the Hugging Face token on this machine is present but REJECTED, so the push would "
                "fail after the session is finished. Log in again (huggingface-cli login) before "
                "you finish; the episodes stay on this machine either way"
                if rejected
                else "no Hugging Face credential on this machine, so the push would fail after the "
                "session is finished - and closing the session destroys the recorder, so there is "
                "no retry from here. Run huggingface-cli login (or set HF_TOKEN) first; the "
                "episodes stay on this machine either way"
            ),
        }

    if "/" in name:
        namespace = name.split("/", 1)[0]
        if user and namespace != user:
            return {
                "ok": False,
                "state": "foreign_namespace",
                # The one honest unknown: an org membership cannot be established from here.
                "needs_force": True,
                "user": user,
                "destination": name,
                "detail": (
                    f"this dataset is named {name}, so the push targets the '{namespace}' namespace "
                    f"while you are logged in as '{user}'. If '{namespace}' is an organisation you can "
                    f"write to this works; if not, the Hub refuses it AFTER the session is finished"
                ),
            }

    dest = destination(name, user)

    ex = dict(existing or {})
    if ex.get("exists") is True:
        n = ex.get("episodes") if isinstance(ex.get("episodes"), int) else None
        count = f" with {n} episode(s)" if n else ""
        return {
            "ok": False,
            "state": "destination_exists",
            # The operator's call: replacing their own earlier take deliberately is legitimate, and
            # only they know whether that is what this is.
            "needs_force": True,
            "user": user,
            "destination": dest,
            "detail": (
                f"{dest} already exists on the Hub{count}. Publishing this session uploads INTO that "
                "repo rather than creating a new one - the episodes recorded here are a fresh, "
                "shorter dataset, so its meta would claim fewer episodes than the files that stay "
                "behind. Rename this dataset, or tick below if replacing that published take is "
                "what you intend"
            ),
        }

    return {
        "ok": True,
        "state": "ready",
        "needs_force": False,
        "user": user,
        "destination": dest,
        # Says where it goes, not just what it is called: the namespace is otherwise invisible.
        "detail": f"publishes as {dest} (you are logged in as {user})",
    }
