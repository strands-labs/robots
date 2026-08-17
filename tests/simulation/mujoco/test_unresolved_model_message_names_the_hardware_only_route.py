"""A registered robot with no simulation asset must be told so, not spell-checked.

``strands_robots/registry/robots.json`` holds two kinds of entry. Most declare an
``asset`` block and can be spawned in simulation; nine declare only a
``hardware`` block, because they are robots strands drives over LeRobot that
mujoco_menagerie has no model for. ``list_robots(mode="sim")`` already reports
that split, so the distinction is registry fact rather than a guess.

``MuJoCoSimEngine._unknown_model_msg`` did not consult it. A hardware-only name
took the unknown-name branch, so a request whose name was already correct was
answered with a spelling suggestion and a pointer to the robot listing - and the
suggestion pool was the *whole* registry, so the correction offered could itself
be hardware-only. ``earthrover`` was the extreme: its sole offered remedy was
``hope_jr``, which has no simulation asset either, so following the one remedy on
offer reproduced the identical refusal.

These tests pin both halves of that: the refusal for a hardware-only robot names
the hardware entry point instead of a spelling, and every name any refusal from
this engine offers as a remedy is one the engine can actually spawn.
"""

import pytest

from strands_robots.registry import get_hardware_type, get_robot, has_hardware, has_sim, list_robots
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

# A hardware-only entry and one of its aliases. Both postures are asserted as a
# premise below rather than assumed, so editing robots.json fails here with a
# readable reason instead of quietly making a case vacuous.
HARDWARE_ONLY_ROBOT = "reachy2"
HARDWARE_ONLY_ALIAS = "earth_rover"  # -> earthrover
HARDWARE_ONLY_ALIAS_CANONICAL = "earthrover"
DISTINCTLY_TYPED_ROBOT = "omx"  # lerobot_type "omx_follower" != the registry name


def _offered(message: str) -> list[str]:
    """The names a refusal offers as corrections, in the order it offers them."""
    if "Did you mean:" not in message:
        return []
    listed = message.split("Did you mean:")[1].split("?")[0]
    return [part.strip() for part in listed.split(",") if part.strip()]


def _hardware_only_names() -> list[str]:
    """Every registry robot that declares hardware support and no sim asset."""
    return sorted(r["name"] for r in list_robots() if not has_sim(r["name"]) and has_hardware(r["name"]))


def test_the_registry_split_these_tests_depend_on_still_exists() -> None:
    """Non-vacuity: the registry must still hold hardware-only entries."""
    hardware_only = _hardware_only_names()
    assert hardware_only, (
        "no registry entry declares hardware support without a simulation asset, "
        "so every case below would pass without exercising anything"
    )
    sim_listed = {r["name"] for r in list_robots(mode="sim")}
    assert not sim_listed & set(hardware_only), (
        f"list_robots(mode='sim') offers hardware-only robots {sorted(sim_listed & set(hardware_only))}, "
        "so it is no longer the authority this message reads"
    )
    for subject in (HARDWARE_ONLY_ROBOT, HARDWARE_ONLY_ALIAS_CANONICAL):
        assert subject in hardware_only, f"{subject!r} is no longer a hardware-only registry entry"


class TestAHardwareOnlyRobotIsToldItIsHardwareOnly:
    """The name is already correct, so the remedy is the route, not the spelling."""

    def test_the_refusal_does_not_offer_a_spelling_fix(self) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg(HARDWARE_ONLY_ROBOT)
        assert "Did you mean:" not in msg, (
            f"{HARDWARE_ONLY_ROBOT!r} is spelled exactly as the registry holds it, yet the "
            f"refusal offered a correction: {msg}"
        )
        assert "No model found" not in msg, f"a robot the registry knows was reported as an unknown name: {msg}"

    def test_the_refusal_says_why_there_is_no_model(self) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg(HARDWARE_ONLY_ROBOT)
        assert "hardware" in msg, msg
        assert "no simulation asset" in msg, msg

    def test_the_refusal_names_the_hardware_entry_point(self) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg(HARDWARE_ONLY_ROBOT)
        assert f"Robot('{HARDWARE_ONLY_ROBOT}', mode='real')" in msg, (
            f"the refusal does not name the route that can drive this robot: {msg}"
        )

    def test_the_refusal_names_the_lerobot_type_the_route_is_keyed_on(self) -> None:
        # Graded on a subject whose LeRobot type differs from its registry name,
        # so the echoed request name cannot satisfy the assertion on its own.
        expected = get_hardware_type(DISTINCTLY_TYPED_ROBOT)
        assert expected and expected != DISTINCTLY_TYPED_ROBOT, (
            f"premise: {DISTINCTLY_TYPED_ROBOT!r} must declare a lerobot_type that is not "
            f"its own name, or this case passes on the echoed request name alone (got {expected!r})"
        )
        assert expected in MuJoCoSimEngine._unknown_model_msg(DISTINCTLY_TYPED_ROBOT)

    def test_the_named_route_exists_for_every_hardware_only_robot(self) -> None:
        """The advertised remedy is checked against the registry, not just quoted."""
        for name in _hardware_only_names():
            msg = MuJoCoSimEngine._unknown_model_msg(name)
            assert "mode='real'" in msg, f"{name}: {msg}"
            assert has_hardware(name), f"{name} was pointed at the hardware route but declares no hardware backend"
            assert get_hardware_type(name), f"{name} was pointed at the hardware route with no lerobot_type to use"

    def test_an_alias_reports_the_canonical_route(self) -> None:
        assert get_robot(HARDWARE_ONLY_ALIAS), f"premise: {HARDWARE_ONLY_ALIAS!r} no longer resolves"
        msg = MuJoCoSimEngine._unknown_model_msg(HARDWARE_ONLY_ALIAS)
        assert HARDWARE_ONLY_ALIAS in msg, msg
        assert f"Robot('{HARDWARE_ONLY_ALIAS_CANONICAL}', mode='real')" in msg, msg

    def test_the_refusal_points_at_what_this_backend_can_spawn(self) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg(HARDWARE_ONLY_ROBOT)
        assert "list_robots(mode='sim')" in msg, msg

    def test_the_message_stays_ascii(self) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg(HARDWARE_ONLY_ROBOT)
        assert msg.isascii(), [ch for ch in msg if not ch.isascii()]


class TestEveryOfferedRemedyIsOneThisBackendCanSpawn:
    """A correction the engine would refuse in turn is not a correction."""

    def test_no_hardware_only_name_is_offered_to_a_hardware_only_request(self) -> None:
        for name in _hardware_only_names():
            offered = _offered(MuJoCoSimEngine._unknown_model_msg(name))
            unloadable = [candidate for candidate in offered if not has_sim(candidate)]
            assert not unloadable, (
                f"the refusal for {name!r} offered {unloadable} as corrections, and those have no "
                "simulation asset either, so the caller lands back on this same message"
            )

    @pytest.mark.parametrize("typo", ["hope", "rebot", "reachy2x", "lekiwi_clint", "omxx", "earthrove"])
    def test_no_hardware_only_name_is_offered_to_a_typo(self, typo: str) -> None:
        offered = _offered(MuJoCoSimEngine._unknown_model_msg(typo))
        unloadable = [candidate for candidate in offered if not has_sim(candidate)]
        assert not unloadable, (
            f"the refusal for {typo!r} offered {unloadable}, which this backend cannot spawn: "
            f"{MuJoCoSimEngine._unknown_model_msg(typo)}"
        )

    def test_a_typo_of_a_hardware_only_name_still_gets_help(self) -> None:
        """Narrowing the pool must not empty it: a near miss still gets a name."""
        msg = MuJoCoSimEngine._unknown_model_msg("lekiwi_clint")
        assert "lekiwi" in _offered(msg), msg


class TestTheOtherTwoConditionsAreUnchanged:
    """Over-reach controls: the new branch sits between two existing ones."""

    def test_a_registered_robot_with_a_missing_asset_still_names_the_asset(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
        asset = (get_robot("panda") or {})["asset"]
        msg = MuJoCoSimEngine._unknown_model_msg("panda")
        assert f"{asset['dir']}/{asset['model_xml']}" in msg, msg
        assert "download_assets" in msg, msg
        assert "hardware" not in msg, f"a robot with an asset was reported as hardware-only: {msg}"

    def test_an_unknown_name_still_gets_the_generic_form(self) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg("so10x")
        assert "No model found for 'so10x'." in msg, msg
        assert "so101" in msg or "so100" in msg, msg
        assert "list_urdfs" in msg, msg

    def test_a_name_with_no_neighbours_omits_the_suggestion(self) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg("zzqqxx0000nope")
        assert "Did you mean:" not in msg, msg
        assert "list_urdfs" in msg, msg
