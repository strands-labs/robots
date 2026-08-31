"""The UR controller double, installed as the SDK the driver imports."""

from __future__ import annotations

import sys
import types

import pytest

from tests.mocks.ur_rtde import FakeRTDE


@pytest.fixture
def fake_rtde(monkeypatch: pytest.MonkeyPatch) -> FakeRTDE:
    """Install the doubles as ``rtde_control`` / ``rtde_receive`` importables.

    The driver resolves its SDK with ``importlib.import_module``, which reads
    ``sys.modules`` first - so this drives the driver's own resolution path
    rather than patching over it.
    """
    fake = FakeRTDE()
    control_module = types.ModuleType("rtde_control")
    receive_module = types.ModuleType("rtde_receive")
    control_module.RTDEControlInterface = fake.make_control  # type: ignore[attr-defined]
    receive_module.RTDEReceiveInterface = fake.make_receive  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rtde_control", control_module)
    monkeypatch.setitem(sys.modules, "rtde_receive", receive_module)
    return fake
