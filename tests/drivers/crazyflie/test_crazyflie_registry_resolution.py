"""``Robot("crazyflie", mode="real")`` reaches the native driver, by declaration.

The registry entry carries ``hardware.driver = "strands"``, so a caller who never
mentions ``driver=`` still gets the CRTP driver rather than a lerobot robot -
which matters because lerobot has no robot type for a Crazyflie at all. Before
this driver that call raised ``ValueError: Unsupported robot type: 'crazyflie'``
while ``mode="sim"`` happily served the MuJoCo asset.

The alias cell is the one that would break quietly: ``cf2`` and
``bitcraze_crazyflie`` are how the registry lets a caller spell it, and a
registration keyed on the wrong string resolves for one spelling and not the
others.
"""

from __future__ import annotations

import pytest

from strands_robots.drivers import get_native_driver_class, list_native_drivers
from strands_robots.drivers.base import missing_driver_members
from strands_robots.drivers.crazyflie import SUPPORTED_ROBOTS, CrazyflieDriver
from strands_robots.registry import get_driver, get_robot, resolve_name


class TestTheRegistryRoutesToTheNativeDriver:
    """A declaration on the entry, so no ``driver=`` keyword is needed."""

    def test_the_entry_declares_the_strands_driver(self) -> None:
        assert get_driver("crazyflie") == "strands", (
            "without this declaration the default driver is lerobot, which has no Crazyflie type"
        )

    def test_the_driver_is_registered_for_the_advertised_robot(self) -> None:
        assert list_native_drivers()["crazyflie"] == "CrazyflieDriver"

    @pytest.mark.parametrize("spelling", ["crazyflie", "cf2", "bitcraze_crazyflie"])
    def test_every_registry_spelling_resolves_to_the_same_driver(self, spelling: str) -> None:
        assert resolve_name(spelling) == "crazyflie"
        assert get_native_driver_class(spelling) is CrazyflieDriver

    def test_the_advertised_robot_exists_in_the_registry(self) -> None:
        """Advertising a robot the factory cannot build is a promise not kept."""
        for name in SUPPORTED_ROBOTS:
            assert get_robot(name) is not None, f"{name} is advertised but has no registry entry"

    def test_the_sibling_drone_is_not_claimed(self) -> None:
        """``skydio_x2`` is a different airframe behind a different SDK."""
        assert "skydio_x2" not in SUPPORTED_ROBOTS
        assert get_native_driver_class("skydio_x2") is None


class TestTheDriverSatisfiesTheSeamContract:
    """The check ``register_native_driver`` runs, run here against the class."""

    def test_the_whole_driver_surface_is_present(self) -> None:
        assert missing_driver_members(CrazyflieDriver) == ()

    def test_the_factory_constructor_keywords_are_accepted(self) -> None:
        """The three the factory always passes, plus the extras it forwards."""
        driver = CrazyflieDriver(
            tool_name="crazyflie",
            cameras={"wrist": {"type": "opencv", "index_or_path": 0}},
            data_config="unused",
            port="radio://0/100/2M/E7E7E7E7E8",
            some_future_keyword=1,
        )
        assert driver.tool_name == "crazyflie"
        assert driver.tool_type == "robot"

    def test_the_tool_spec_names_the_robot_it_drives(self) -> None:
        spec = CrazyflieDriver(tool_name="rooftop_cf").tool_spec
        assert spec["name"] == "rooftop_cf"
        assert spec["inputSchema"]["json"]["required"] == ["action"]
