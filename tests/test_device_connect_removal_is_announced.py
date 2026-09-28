"""Every door into Device Connect announces its 0.7 removal and names the mesh.

The doors are the ``.run()`` that ``Robot()`` binds and a public name resolved
through :mod:`strands_robots.device_connect`, directly or through the top-level
re-export. Each warning must be attributed to the caller's line, or the default
filter hides it. The package is imported inside the test, never at collection:
sibling suites swap its modules in and out at collection time.
"""

from __future__ import annotations

import importlib
import types

import pytest

_PUBLIC = (
    "init_device_connect",
    "init_device_connect_sync",
    "resolve_allow_insecure",
    "RobotDeviceDriver",
    "SimulationDeviceDriver",
    "ReachyMiniDriver",
)
_DOORS = [("run", "strands_robots.device_connect", "Robot(...).run()")]
_DOORS += [(n, "strands_robots.device_connect", f"strands_robots.device_connect.{n}") for n in _PUBLIC]
_DOORS += [("init_device_connect", "strands_robots", "strands_robots.device_connect.init_device_connect")]


@pytest.mark.parametrize(("name", "via", "spelling"), _DOORS, ids=[f"{d[1]}.{d[0]}" for d in _DOORS])
def test_each_door_warns_at_the_callers_line(
    monkeypatch: pytest.MonkeyPatch, name: str, via: str, spelling: str
) -> None:
    dc = importlib.import_module("strands_robots.device_connect")
    assert tuple(dc.__all__) == _PUBLIC
    with pytest.warns(DeprecationWarning, match=r"removed in 0\.7") as caught:
        if name == "run":
            robot_module = importlib.import_module("strands_robots.robot")
            monkeypatch.setattr(robot_module, "_run_device_connect_foreground", lambda instance: None)
            instance = types.SimpleNamespace()
            robot_module._attach_device_connect(instance, "so100", "sim", "so100-a")
            instance.run()
        else:
            package = importlib.import_module(via)
            for owner in {package, dc}:
                monkeypatch.delitem(vars(owner), name, raising=False)
            getattr(package, name)
    (warning,) = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert str(warning.message).startswith(spelling)
    assert "Robot(..., mesh=True)" in str(warning.message)
    assert warning.filename == __file__
