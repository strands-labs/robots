"""Shared fakes for the Isaac backend suite.

The ``cloner`` fixture lives here rather than in a test module because two
modules need it and a fixture cannot be imported into a second one without
shadowing: pytest resolves a fixture by name, so a test taking ``cloner`` as a
parameter collides with the module-level import of the same name (``F811``). A
conftest is the one home that gives it a single owner and no import at all.

What it fakes and why it has to: ``replicate`` builds a real fleet through
Isaac's ``isaacsim.core.cloner`` extension, which resolves only inside a running
Isaac Sim application. Every count that IS honored therefore needs the extension
present, and this suite runs on hosts without Isaac Sim. The live half is
exercised on GPU.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _Prim:
    def __init__(self, path: str) -> None:
        self._path = path

    def GetPath(self) -> Any:  # noqa: N802 - USD API spelling
        return types.SimpleNamespace(pathString=self._path)


class _Stage:
    """A stage whose prim population the fake cloner can grow."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = list(paths)
        self.defined: list[tuple[str, str]] = []

    def Traverse(self) -> list[_Prim]:  # noqa: N802 - USD API spelling
        return [_Prim(p) for p in self.paths]

    def DefinePrim(self, path: str, type_name: str = "") -> _Prim:  # noqa: N802 - USD API spelling
        self.defined.append((path, type_name))
        if path not in self.paths:
            self.paths.append(path)
        return _Prim(path)


class _FakeCloner:
    """Records what it was asked to clone, and writes what a real cloner writes.

    A clone lands *at* each requested ``prim_paths`` entry, plus some descendants.
    That fidelity is what makes the anti-fabrication guard gradable: the guard
    reads the expected clone paths, so a fake that invented its own naming would
    let the guard pass on paths nothing asked for.
    """

    #: Prims a single ``clone`` call adds per target path, counting the clone root
    #: itself. ``0`` reproduces the shape measured on the real cloner when a
    #: target's parent scope is absent: no error, and no clone.
    per_clone = 3
    fail_clone: BaseException | None = None
    fail_filter: BaseException | None = None
    stage: _Stage | None = None

    instances: list[_FakeCloner] = []

    def __init__(self, spacing: float | None = None) -> None:
        self.spacing = spacing
        self.base_envs: list[str] = []
        self.clones: list[dict[str, Any]] = []
        self.filters: list[dict[str, Any]] = []
        type(self).instances.append(self)

    def define_base_env(self, base_env_path: str) -> None:
        self.base_envs.append(base_env_path)

    def clone(self, **kwargs: Any) -> None:
        self.clones.append(kwargs)
        # Bound to a local before the check: ``type(self).X is not None`` does not
        # narrow the following ``raise type(self).X``, so the raise reads as
        # possibly-None.
        failure = type(self).fail_clone
        if failure is not None:
            raise failure
        stage = type(self).stage
        if stage is None or type(self).per_clone <= 0:
            return
        for target in kwargs.get("prim_paths", []):
            # The clone root itself, which is the path the caller asked for and
            # the path the guard checks.
            stage.paths.append(target)
            for i in range(type(self).per_clone - 1):
                stage.paths.append(f"{target}/link{i}")

    def filter_collisions(self, **kwargs: Any) -> None:
        self.filters.append(kwargs)
        failure = type(self).fail_filter
        if failure is not None:
            raise failure


@pytest.fixture
def cloner(monkeypatch) -> type[_FakeCloner]:
    """Install a fake ``isaacsim.core.cloner`` and a fake ``omni.usd`` stage."""
    _FakeCloner.per_clone = 3
    _FakeCloner.fail_clone = None
    _FakeCloner.fail_filter = None
    _FakeCloner.instances = []
    stage = _Stage(["/World", "/World/Robots", "/World/Robots/arm"])
    _FakeCloner.stage = stage

    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.cloner"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["isaacsim"].core = sys.modules["isaacsim.core"]  # type: ignore[attr-defined]
    sys.modules["isaacsim.core"].cloner = sys.modules["isaacsim.core.cloner"]  # type: ignore[attr-defined]
    sys.modules["isaacsim.core.cloner"].GridCloner = _FakeCloner  # type: ignore[attr-defined]

    omni = types.ModuleType("omni")
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)  # type: ignore[attr-defined]
    omni.usd = omni_usd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)
    return _FakeCloner
