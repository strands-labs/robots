"""The dashboard's passkeys and its session key: one JSON file only this user can read.

``$STRANDS_ROBOTS_DASHBOARD_AUTH``, else ``$STRANDS_ROBOTS_CACHE/dashboard-auth.json`` (default
``~/.cache/strands_robots/dashboard-auth.json``), next to (not inside) the browser engines' cache, so clearing
that cache keeps the passkeys. It holds public keys and sign counts only: a private key never leaves the authenticator.
Deleting the file signs everyone out and puts the dashboard back into first-time setup.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...description.assets import cache_dir


def default_path() -> Path:
    env = os.environ.get("STRANDS_ROBOTS_DASHBOARD_AUTH")
    return Path(env).expanduser() if env else cache_dir() / "dashboard-auth.json"


class Store:
    """Read-modify-write of the file under one lock; every write replaces the file atomically, mode 0600."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_path()
        self._lock = threading.Lock()

    def read(self) -> dict[str, Any]:
        with self._lock:
            return self._read()

    def update(self, change: Callable[[dict[str, Any]], Any]) -> Any:
        """``change(doc)`` edits the document in place; it is written back unless ``change`` raises."""
        with self._lock:
            doc = self._read()
            out = change(doc)
            self._write(doc)
            return out

    @property
    def secret(self) -> bytes:
        return base64.urlsafe_b64decode(self.read()["secret"])

    def credentials(self) -> list[dict[str, Any]]:
        return list(self.read()["credentials"])

    def _read(self) -> dict[str, Any]:
        try:
            doc = json.loads(self.path.read_text())
        except FileNotFoundError:
            doc = None
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{self.path}: unreadable passkey store ({exc}); fix or delete it") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("credentials"), list) or "secret" not in doc:
            if doc is not None:
                raise RuntimeError(f"{self.path}: not a passkey store; fix or delete it")
            doc = {
                "secret": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
                "user_id": base64.urlsafe_b64encode(secrets.token_bytes(16)).decode().rstrip("="),
                "credentials": [],
            }
            self._write(doc)
        return doc

    def _write(self, doc: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{secrets.token_hex(4)}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)
