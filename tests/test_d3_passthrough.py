"""D3: every parameter of a driver the public Robot(...) builds is reachable from the public call (or on a
reviewed INTERNAL list), proven by CALLING through Robot and watching the driver constructor receive it.
A new driver parameter without a public path fails here. Camera specs carry width/height/fps and are checked.
"""

from __future__ import annotations

import importlib

import pytest

from strands_robots import Description, Robot
from strands_robots.robot.drivers.driver import DRIVERS, driver_parameters

SENTINEL = object()
# Parameters Robot(...) names itself (forwarded under the same name), and test doubles that are internal on purpose.
ROBOT_NAMED = {"port", "cameras", "calibration"}
INTERNAL = {"bus", "subscriber", "client"}  # injected transports for tests; not a user knob


def _driver_class(kind: str) -> type:
    mod, cls = DRIVERS[kind].split(":")
    return getattr(importlib.import_module(mod), cls)


@pytest.mark.parametrize("kind", sorted(DRIVERS))
def test_d3_every_driver_parameter_is_reachable_from_robot(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _driver_class(kind)
    params = set(driver_parameters(cls)) - INTERNAL
    seen: dict = {}

    class Spy(cls):  # type: ignore[misc, valid-type]
        def __init__(self, description, **kw):  # noqa: ANN001, ANN003
            seen.update(kw)
            raise RuntimeError("stop here: the driver received its arguments")

    mod, name = DRIVERS[kind].split(":")
    monkeypatch.setattr(importlib.import_module(mod), name, Spy)
    options = {p: SENTINEL for p in params - ROBOT_NAMED}
    named = {p: SENTINEL for p in params & ROBOT_NAMED}
    model = "so101" if kind == "feetech" else "unitree_g1" if kind == "unitree" else "microduck"
    with pytest.raises(RuntimeError, match="received"):
        Robot(model, driver=kind, twin=False, **{**named, **options})
    for p in params:
        assert seen.get(p) is SENTINEL, f"{cls.__name__}({p}=) is not reachable from Robot(...)"


def test_d3_a_misspelt_driver_option_is_refused_naming_the_options() -> None:
    with pytest.raises(TypeError, match="robot_id"):
        Robot("so101", port="/dev/null-x", baudrate=1_000_000)  # the option is baud=


def test_d3_world_robots_take_no_driver_options() -> None:
    with pytest.raises(TypeError, match="driver options"):
        Robot("so101", baud=1_000_000)


def test_d3_hardware_camera_specs_carry_width_height_fps() -> None:
    from strands_robots.robot.drivers.cameras import Cameras, CameraSpec

    c = Cameras({"front": 0, "wrist": {"index": 1, "width": 640, "height": 480, "fps": 30}})
    assert c.specs["wrist"] == CameraSpec(1, 640, 480, 30) and c.specs["front"] == CameraSpec(0)
    with pytest.raises(TypeError, match="use index"):
        Cameras({"x": {"index": 0, "w": 640}})
    assert Description.load("so101").name == "so101"
