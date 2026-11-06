"""Passkey sign-in for the dashboard: :class:`Auth`, and :func:`configure` for ``Dashboard(auth=...)``."""

from __future__ import annotations

from .core import DEFAULT_TTL, Auth, configure

__all__ = ["DEFAULT_TTL", "Auth", "configure"]
