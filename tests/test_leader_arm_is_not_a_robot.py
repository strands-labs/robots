"""A leader arm is a teleoperator, not a robot.

``so101_leader`` was an alias of the ``so101`` registry entry, whose
``hardware.lerobot_type`` is ``so101_follower``. So
``Robot("so101_leader", mode="real", port=<leader port>)`` built an
SO101Follower driver on the leader's motor bus - torque-enabling the arm a
human is holding and turning it into a rigid position servo. These tests pin
the refusal and the registry invariant behind it.

That refusal is reachable only while the leader name is *unregistered*: it sits
behind ``get_robot(canonical) is None and not has_hardware(canonical)``. An
operator who registers their leader arm -- as itself, with
``hardware.lerobot_type="so101_leader"``, which is the honest thing to register
and satisfies the invariant above -- makes it unreachable, and the request fell
through to lerobot's ``RobotConfig`` lookup instead. That answered
``Unsupported robot type: 'so101_leader'. Known lerobot robot types: [...]``,
whose list is every follower type: the retry this refusal exists to remove.
lerobot knows ``so101_leader`` perfectly well -- as a *teleoperator* -- so the
deeper site names the kind it really is, in both directions.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import strands_robots.hardware_robot as hardware_robot
import strands_robots.robot as robot_module
from strands_robots import Robot
from strands_robots.registry import get_hardware_type, get_robot, resolve_name

REGISTRY_PATH = Path(__file__).parent.parent / "strands_robots" / "registry" / "robots.json"


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    """Load the shipped robot registry once per module."""
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    entries: dict[str, Any] = data.get("robots", data)
    return entries


def test_no_leader_name_resolves_to_a_follower_entry(registry: dict[str, Any]) -> None:
    """Registry invariant: a ``*_leader`` name never names a follower robot.

    Generalises the ``so101_leader`` regression to the whole file - a leader
    name may only appear as a key or alias of an entry that is itself that
    leader device (``hardware.lerobot_type`` matching), never of the follower
    it drives.
    """
    offenders = []
    for name, info in registry.items():
        lerobot_type = (info.get("hardware") or {}).get("lerobot_type")
        for candidate in (name, *info.get("aliases", [])):
            if candidate.endswith("_leader") and candidate != lerobot_type:
                offenders.append(f"{candidate!r} -> entry {name!r} (lerobot_type={lerobot_type!r})")
    assert not offenders, "leader names resolving to a non-leader entry: " + "; ".join(offenders)


def test_so101_leader_does_not_resolve_to_the_follower() -> None:
    """The leader name must not carry the follower's hardware type."""
    assert resolve_name("so101_leader") == "so101_leader"
    assert get_robot("so101_leader") is None
    assert get_hardware_type("so101_leader") is None


def test_real_mode_refuses_a_leader_before_touching_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal lands before any driver is constructed on the leader's port."""
    constructed: list[dict[str, Any]] = []

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

    monkeypatch.setattr(hardware_robot, "Robot", _Recorder)

    with pytest.raises(ValueError, match="teleoperator"):
        Robot("so101_leader", mode="real", port="/dev/ttyACM1")

    assert constructed == []


def test_the_refusal_names_the_teleoperator_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message points at ``Teleoperator``/``attach_teleop``, not the follower.

    Answering with the follower name would invite the caller to retry
    ``Robot("so101", mode="real", port=<leader port>)`` - the same hazard by
    another spelling.
    """
    monkeypatch.setattr(hardware_robot, "Robot", lambda **kwargs: pytest.fail("built a driver for a leader"))

    with pytest.raises(ValueError) as excinfo:
        Robot("so101_leader", mode="real", port="/dev/ttyACM1")

    message = str(excinfo.value)
    assert "Teleoperator('so101_leader', port=...)" in message
    assert "attach_teleop('so101_leader', port=...)" in message
    assert "so101_follower" not in message


@pytest.mark.parametrize("mode", ["sim", "real", "auto"])
@pytest.mark.parametrize("name", ["so101_leader", "SO101-Leader", "so100_leader", "koch_leader"])
def test_a_leader_is_refused_in_every_mode(mode: str, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """No mode accepts a leader - sim would silently hand back the follower."""
    monkeypatch.setattr(hardware_robot, "Robot", lambda **kwargs: pytest.fail("built a driver for a leader"))

    with pytest.raises(ValueError, match="is a teleoperator .leader. device, not a robot"):
        Robot(name, mode=mode, port="/dev/ttyACM1")


def test_follower_names_still_resolve() -> None:
    """The follower aliases this fix did not touch keep resolving."""
    assert resolve_name("so101_follower") == "so101"
    assert resolve_name("so101_dualcam") == "so101"
    assert get_hardware_type("so101") == "so101_follower"


def test_an_unknown_non_leader_name_keeps_the_registry_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The leader branch must not hijack the generic unknown-robot error."""
    monkeypatch.setattr(robot_module, "is_discoverable", lambda name: False)

    with pytest.raises(ValueError, match="Unknown robot 'so101_leaderr'"):
        Robot("so101_leaderr", mode="sim")


# --- The deeper site: a name lerobot knows, as the other kind of device -------
#
# ``Robot()``'s registry guard above only sees an UNREGISTERED ``*_leader``
# name. These pin the refusal a registered leader actually receives, and the
# mirror-image refusal on the teleoperator side.


LEROBOT_LEADERS = [
    "so100_leader",
    "so101_leader",
    "koch_leader",
    "bi_so_leader",
]

LEROBOT_FOLLOWERS = [
    "so100_follower",
    "so101_follower",
    "koch_follower",
]


@pytest.fixture
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the user registry + asset paths at temp dirs.

    Mirrors ``tests/registry/conftest.py``'s autouse fixture so registering a
    probe robot here cannot touch the developer's ``~/.strands_robots``.
    """
    from strands_robots.registry.user_registry import _invalidate_cache

    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets))
    _invalidate_cache()
    yield assets
    _invalidate_cache()


def _register_leader(assets: Path, name: str, lerobot_type: str) -> None:
    """Register ``name`` as the leader device it is, the way an operator would."""
    from strands_robots.registry import register_robot

    model = assets / name / "so101"
    model.mkdir(parents=True, exist_ok=True)
    (model / "so101.xml").write_text(
        '<mujoco model="stub"><worldbody><body name="link">'
        '<joint name="j1" type="hinge" axis="0 0 1"/>'
        '<geom type="capsule" size="0.02 0.1"/></body></worldbody></mujoco>'
    )
    register_robot(
        name=name,
        model_xml="so101/so101.xml",
        description="SO-101 leader arm on this rig",
        category="arm",
        joints=6,
        hardware={"lerobot_type": lerobot_type, "port": "/dev/ttyACM0"},
        overwrite=True,
    )


def test_a_registered_leader_is_named_as_a_teleoperator(isolated_registry: Path) -> None:
    """The reachability half: registering the leader must not lose the refusal.

    ``Robot()``'s registry guard cannot fire once the name is registered, so
    this is the path a real rig takes. It has to arrive at the same verdict.
    """
    _register_leader(isolated_registry, "probe_rig_leader", "so101_leader")

    with pytest.raises(ValueError) as exc:
        Robot("probe_rig_leader", mode="real")

    msg = str(exc.value)
    assert "teleoperator" in msg, msg
    assert "Teleoperator('so101_leader', port=...)" in msg, msg
    assert "drive the arm a human is holding" in msg, msg


def test_the_registered_leader_refusal_never_offers_the_follower_types(
    isolated_registry: Path,
) -> None:
    """The safety half, and the whole reason this refusal is special.

    A list of robot types is a list of followers, and retrying with one of them
    on the leader's port is the mistake being prevented. A caller whose name IS
    a known teleoperator has not made a typo, so there is nothing a listing
    could usefully offer.
    """
    _register_leader(isolated_registry, "probe_rig_leader", "so101_leader")

    with pytest.raises(ValueError) as exc:
        Robot("probe_rig_leader", mode="real")

    msg = str(exc.value)
    assert "Known lerobot robot types" not in msg, msg
    assert "so101_follower" not in msg, msg


@pytest.mark.parametrize("leader", LEROBOT_LEADERS)
def test_every_lerobot_leader_type_is_refused_as_a_teleoperator(leader: str) -> None:
    """Not just ``so101_leader``: every leader lerobot ships behaves the same.

    Driven at the config site so the assertion is about the refusal rather than
    about any one registry entry.
    """
    pytest.importorskip("lerobot")
    inst = hardware_robot.Robot.__new__(hardware_robot.Robot)

    with pytest.raises(ValueError) as exc:
        inst._create_minimal_config(leader, None)

    msg = str(exc.value)
    assert "teleoperator" in msg, msg
    assert f"Teleoperator({leader!r}, port=...)" in msg, msg
    assert "Known lerobot robot types" not in msg, msg


@pytest.mark.parametrize("follower", LEROBOT_FOLLOWERS)
def test_a_follower_handed_to_teleoperator_is_named_as_a_robot(follower: str) -> None:
    """The mirror image: the other entry point is equally blind on its own.

    Not a safety case -- driving a follower is the normal thing -- but the same
    wrong-entry-point answer, so it gets the same treatment.
    """
    pytest.importorskip("lerobot")
    from strands_robots.teleoperator import _build_teleop_config

    with pytest.raises(ValueError) as exc:
        _build_teleop_config(follower, port="/dev/null")

    msg = str(exc.value)
    assert "robot" in msg, msg
    assert f"Robot({follower!r}, mode='real', port=...)" in msg, msg
    assert "Known lerobot teleoperator types" not in msg, msg


def test_an_unknown_robot_type_keeps_its_listing() -> None:
    """Control: a genuinely unknown name is a typo, and a listing helps.

    Fails if the cross-kind branch widens into every refusal.
    """
    pytest.importorskip("lerobot")
    inst = hardware_robot.Robot.__new__(hardware_robot.Robot)

    with pytest.raises(ValueError) as exc:
        inst._create_minimal_config("not_a_real_robot", None)

    msg = str(exc.value)
    assert "Unsupported robot type: 'not_a_real_robot'" in msg, msg
    assert "Known lerobot robot types" in msg, msg
    assert "teleoperator" not in msg, msg


def test_an_unknown_teleoperator_type_keeps_its_listing() -> None:
    """Control: the same, on the teleoperator side."""
    pytest.importorskip("lerobot")
    from strands_robots.teleoperator import _build_teleop_config

    with pytest.raises(ValueError) as exc:
        _build_teleop_config("not_a_real_teleoperator", port="/dev/null")

    msg = str(exc.value)
    assert "Unsupported teleoperator type: 'not_a_real_teleoperator'" in msg, msg
    assert "Known lerobot teleoperator types" in msg, msg


def test_a_follower_type_still_builds_a_config() -> None:
    """Control: the accepted path is untouched -- only refusals changed."""
    pytest.importorskip("lerobot")
    inst = hardware_robot.Robot.__new__(hardware_robot.Robot)
    # The accepted path reads ``tool_name_str`` to namespace lerobot's
    # calibration files; the refusal path above never gets that far.
    inst.tool_name_str = "probe"

    config = inst._create_minimal_config("so101_follower", None, port="/dev/null")

    assert config.port == "/dev/null"
    assert config.id == "probe"


def test_the_rule_refuses_a_kind_it_does_not_know() -> None:
    """A ``wanted`` that names neither kind is a caller error, not a silent None.

    The helper answers a question about two registries; asked about a third it
    has no honest answer, so it says so rather than reporting "not the other
    kind".
    """
    from strands_robots.teleoperator import _other_lerobot_kind_refusal

    with pytest.raises(ValueError, match="wanted must be 'robot' or 'teleoperator'"):
        _other_lerobot_kind_refusal("so101_leader", wanted="gripper")
